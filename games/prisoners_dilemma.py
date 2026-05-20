"""
Prisoner's Dilemma — ReAct agent with memory, self-consistency, and trace judging.

Game modes:
  Agent vs NPC:    python games/prisoners_dilemma.py
  Agent vs Agent:  python games/prisoners_dilemma.py --mode ava

Options (vs NPC):
  --strategy  tit_for_tat | always_defect | always_cooperate | random | pavlov | generous_tft | grim_trigger
  --rounds    number of rounds (default 5)
  --memory    enable Chain-of-Hindsight reflection between rounds
  --k         self-consistency samples (default 1 = off)
  --judge     auto-run Tracer judge after game
"""

import sys
import random
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client
from agent.tracing import TraceLogger, parse_react_response, snapshot_game_state, judge_trace, write_judge_result

GAME_NAME = "prisoners_dilemma"
GAME_NAME_AVA = "prisoners_dilemma_agent_vs_agent"


# =============================================================================
# GAME ENVIRONMENT
# =============================================================================

class PrisonersDilemma:
    """
    Prisoner's Dilemma — agent vs a scripted NPC opponent.

    PAYOFF MATRIX:
                       Opponent Cooperates    Opponent Defects
      You Cooperate:       You +3, Opp +3       You +0, Opp +5
      You Defect:          You +5, Opp +0       You +1, Opp +1

    Opponent strategies:
      tit_for_tat     — cooperates round 1, then mirrors agent's last move
      grim_trigger    — cooperates until agent defects once, then defects forever
      pavlov          — win-stay lose-shift: repeat if scored ≥3, else switch
      generous_tft    — like tit_for_tat but forgives defection ~10% of the time
      always_defect   — defects every round
      always_cooperate — cooperates every round
      random          — 50/50 each round
    """

    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"):    (0, 5),
        ("defect",    "cooperate"): (5, 0),
        ("defect",    "defect"):    (1, 1),
    }

    OPPONENT_STRATEGIES = [
        "tit_for_tat", "grim_trigger", "pavlov", "generous_tft",
        "always_defect", "always_cooperate", "random",
    ]

    def __init__(self, rounds: int = 5, opponent_strategy: str = "tit_for_tat"):
        if opponent_strategy not in self.OPPONENT_STRATEGIES:
            raise ValueError(f"Unknown strategy '{opponent_strategy}'. Choose from {self.OPPONENT_STRATEGIES}")
        self.rounds = rounds
        self.opponent_strategy = opponent_strategy
        self.current_round = 1
        self.agent_score = 0
        self.opponent_score = 0
        self.history: list[tuple[str, str]] = []

    def get_game_state(self) -> str:
        last = (
            f"Opponent's last move: {self.history[-1][1]}."
            if self.history else "No moves yet."
        )
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last} "
            f"Score: You {self.agent_score}, Opponent {self.opponent_score}."
        )

    def get_legal_moves(self) -> list[str]:
        return ["cooperate", "defect"]

    def opponent_move(self) -> str:
        s = self.opponent_strategy
        if s == "always_defect":
            return "defect"
        if s == "always_cooperate":
            return "cooperate"
        if s == "random":
            return random.choice(["cooperate", "defect"])
        if s == "tit_for_tat":
            return self.history[-1][0] if self.history else "cooperate"
        if s == "grim_trigger":
            if any(a == "defect" for a, _ in self.history):
                return "defect"
            return "cooperate"
        if s == "pavlov":
            if not self.history:
                return "cooperate"
            last_agent, last_opp = self.history[-1]
            _, opp_pts = self.PAYOFFS[(last_agent, last_opp)]
            return last_opp if opp_pts >= 3 else ("defect" if last_opp == "cooperate" else "cooperate")
        if s == "generous_tft":
            if not self.history:
                return "cooperate"
            if self.history[-1][0] == "defect":
                return "cooperate" if random.random() < 0.1 else "defect"
            return "cooperate"
        return "cooperate"

    def make_move(self, agent_move: str) -> str:
        agent_move = agent_move.strip().lower()
        if agent_move not in self.get_legal_moves():
            return f"Invalid move: '{agent_move}'. Legal moves are: {self.get_legal_moves()}"
        opp = self.opponent_move()
        agent_pts, opp_pts = self.PAYOFFS[(agent_move, opp)]
        self.agent_score += agent_pts
        self.opponent_score += opp_pts
        self.history.append((agent_move, opp))
        self.current_round += 1
        return (
            f"Move accepted. You played {agent_move}, opponent played {opp}. "
            f"Points this round: You +{agent_pts}, Opponent +{opp_pts}."
        )

    def is_over(self) -> bool:
        return self.current_round > self.rounds


class SymmetricPD:
    """
    Prisoner's Dilemma where both players are AI agents.
    Moves are decided independently and revealed simultaneously.
    Each agent sees state from their own perspective via get_state_for(side).
    """

    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"):    (0, 5),
        ("defect",    "cooperate"): (5, 0),
        ("defect",    "defect"):    (1, 1),
    }

    def __init__(self, rounds: int = 5):
        self.rounds = rounds
        self.current_round = 1
        self.score_a = 0
        self.score_b = 0
        self.history: list[tuple[str, str]] = []

    def get_state_for(self, side: str) -> str:
        my_score = self.score_a if side == "A" else self.score_b
        opp_score = self.score_b if side == "A" else self.score_a
        if self.history:
            last_a, last_b = self.history[-1]
            opp_last = last_b if side == "A" else last_a
            last_str = f"Opponent's last move: {opp_last}."
        else:
            last_str = "No moves yet."
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last_str} Score: You {my_score}, Opponent {opp_score}."
        )

    def get_legal_moves(self) -> list[str]:
        return ["cooperate", "defect"]

    def apply_moves(self, move_a: str, move_b: str) -> tuple[str, str]:
        move_a = move_a.strip().lower()
        move_b = move_b.strip().lower()
        if move_a not in ("cooperate", "defect"):
            move_a = "cooperate"
        if move_b not in ("cooperate", "defect"):
            move_b = "cooperate"
        pts_a, pts_b = self.PAYOFFS[(move_a, move_b)]
        self.score_a += pts_a
        self.score_b += pts_b
        self.history.append((move_a, move_b))
        self.current_round += 1
        obs_a = (
            f"Move accepted. You played {move_a}, opponent played {move_b}. "
            f"Points this round: You +{pts_a}, Opponent +{pts_b}."
        )
        obs_b = (
            f"Move accepted. You played {move_b}, opponent played {move_a}. "
            f"Points this round: You +{pts_b}, Opponent +{pts_a}."
        )
        return obs_a, obs_b

    def is_over(self) -> bool:
        return self.current_round > self.rounds


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

GAME_SYSTEM_PROMPT = """
Your available actions:

get_game_state:
e.g. get_game_state: now
Returns the current round, scores, and opponent's last move.

get_legal_moves:
e.g. get_legal_moves: now
Returns the list of legal moves.

make_move:
e.g. make_move: cooperate
Submits your move. Use EXACTLY one of the legal moves: cooperate or defect.

Example session:

Thought: First, let's reason out loud. It's round 3, opponent cooperated the last two rounds. Mutual cooperation gives +3/round vs +1 from mutual defection. Defecting now gains +5 this round but likely triggers retaliation. Expected value favors cooperation. Let me verify the state.
Action: get_game_state: now
PAUSE

Observation: Round 3/5. Opponent's last move: cooperate. Score: You 6, Opponent 6.

Thought: Confirmed — opponent cooperating consistently. The rational move is to cooperate for sustained mutual gain.
Action: make_move: cooperate
PAUSE

Observation: Move accepted. You played cooperate, opponent played cooperate. Points this round: You +3, Opponent +3.

Decision: cooperate
""".strip()

AVA_SYSTEM_PROMPT = """
Your available actions:

get_game_state:
e.g. get_game_state: now
Returns current round, scores, and opponent's last move.

get_legal_moves:
e.g. get_legal_moves: now
Returns available moves.

make_move:
e.g. make_move: cooperate
Submit your move. Must be exactly 'cooperate' or 'defect'.

Example session:

Thought: Which action maximizes my probability of winning? My opponent is an AI running the same analysis. If they expect me to cooperate, defecting gives +5. But mutual defection yields only +1. I should check the game state first.
Action: get_game_state: now
PAUSE

Observation: Round 2/5. Opponent's last move: cooperate. Score: You 3, Opponent 3.

Thought: Opponent cooperated last round. Sustaining cooperation yields more over remaining rounds.
Action: make_move: cooperate
PAUSE

Observation: Move accepted. You played cooperate, opponent played cooperate. Points this round: You +3, Opponent +3.

Decision: cooperate
""".strip()


def build_game_context(game: PrisonersDilemma, memory: str = "") -> str:
    """Build the agent's round prompt with optional Chain-of-Hindsight memory."""
    memory_block = (
        f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n"
        if memory.strip() else ""
    )
    return f"""
You are playing Prisoner's Dilemma. Your goal is to maximize your total score over {game.rounds} rounds.

RULES:
- Both cooperate  → both get +3 points.
- You cooperate, they defect  → you +0, they +5.
- You defect, they cooperate  → you +5, they +0.
- Both defect  → both get +1 point.
{memory_block}
CURRENT STATE:
{game.get_game_state()}

LEGAL MOVES:
{game.get_legal_moves()}

First, reason out loud about which action maximizes your probability of winning.
Note: your opponent sometimes makes mistakes — a single defection may be an error, not a strategy shift.
Use the Thought/Action/PAUSE loop to decide.
""".strip()


def build_ava_context(game: SymmetricPD, side: str, memory: str = "") -> str:
    """Build the round prompt for one agent in Agent vs Agent mode."""
    memory_block = (
        f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n"
        if memory.strip() else ""
    )
    return f"""
You are playing Prisoner's Dilemma against another AI agent. Maximize your total score over {game.rounds} rounds.
Both of you decide simultaneously — your opponent cannot see your choice before they commit.

RULES:
- Both cooperate  → both get +3 points.
- You cooperate, they defect  → you +0, they +5.
- You defect, they cooperate  → you +5, they +0.
- Both defect  → both get +1 point.
{memory_block}
CURRENT STATE:
{game.get_state_for(side)}

LEGAL MOVES:
{game.get_legal_moves()}

First, reason out loud about which action maximizes your probability of winning.
Your opponent is a reasoning AI — think about what they will likely do and best-respond.
Use the Thought/Action/PAUSE loop to decide.
""".strip()


def generate_reflection(history: list[tuple[str, str]], client) -> str:
    """
    Chain-of-Hindsight: ask the LLM to reflect on the rounds so far and
    produce a short strategic insight to inject into the next round's context.
    """
    history_str = "\n".join(
        f"  Round {i + 1}: You={a}, Opponent={o}"
        for i, (a, o) in enumerate(history)
    )
    messages = [{
        "role": "user",
        "content": (
            f"You just played these rounds of Prisoner's Dilemma:\n{history_str}\n\n"
            "In 2-3 sentences: What patterns did you observe in the opponent's behavior? "
            "What should you adjust going forward? Be concise and actionable."
        ),
    }]
    return client.complete(
        system="You are a strategic game analyst. Provide concise, actionable insights.",
        messages=messages,
    )


# =============================================================================
# REACT LOOP HELPERS
# =============================================================================

def _run_react_round(
    agent: Agent,
    game,
    context: str,
    logger: TraceLogger,
    round_num: int,
    tools: list[str],
    get_state_fn,
    max_iter: int = 10,
    dry_run: bool = False,
) -> str:
    """
    Run one round of the ReAct loop for a PD agent (vs NPC or vs Agent dry-run).
    Returns the move string chosen, or 'cooperate' as fallback.
    """
    next_prompt = context
    move_made = None

    for step in range(1, max_iter + 1):
        state_before = snapshot_game_state(game)
        result = agent(next_prompt)
        parsed = parse_react_response(result)
        print(result)

        if parsed["has_pause"] and parsed["action"]:
            chosen_tool = parsed["action"].strip()
            arg = (parsed["argument"] or "").strip()

            if chosen_tool == "get_game_state":
                obs = get_state_fn()
            elif chosen_tool == "get_legal_moves":
                obs = str(game.get_legal_moves())
            elif chosen_tool == "make_move":
                move = arg.lower()
                if dry_run:
                    obs = f"[dry-run] Intended move: {move}"
                    move_made = move if move in ("cooperate", "defect") else None
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       observation=obs, state_before=state_before,
                                       state_after=snapshot_game_state(game))
                    print(f"Observation: {obs}")
                    break
                else:
                    obs = game.make_move(arg)
                    move_made = arg.lower() if arg.lower() in ("cooperate", "defect") else None
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       observation=obs, state_before=state_before,
                                       state_after=snapshot_game_state(game))
                    logger.record_round_result(round_num, {
                        "observation": obs,
                        "history": game.history[-1] if game.history else None,
                        "agent_score": game.agent_score,
                        "opponent_score": game.opponent_score,
                    })
                    print(f"Observation: {obs}")
                    break
            else:
                obs = f"Unknown tool '{chosen_tool}'. Available: {tools}"

            logger.record_step(round_num, step, next_prompt, result, parsed,
                               observation=obs, state_before=state_before,
                               state_after=snapshot_game_state(game))
            next_prompt = f"Observation: {obs}"
            print(next_prompt)
            continue

        if parsed["decision"]:
            logger.record_step(round_num, step, next_prompt, result, parsed,
                               state_before=state_before, state_after=snapshot_game_state(game))
            print("Agent reached Decision without calling make_move.")
            break

        obs = "No valid Action or Decision found. Use Action: tool_name: argument."
        logger.record_step(round_num, step, next_prompt, result, parsed,
                           observation=obs, state_before=state_before,
                           state_after=snapshot_game_state(game))
        next_prompt = f"Observation: {obs}"

    return move_made or "cooperate"


# =============================================================================
# GAME RUNNERS
# =============================================================================

def run_game(
    game: PrisonersDilemma,
    max_iter: int = 10,
    use_memory: bool = False,
    self_consistency_k: int = 1,
    auto_judge: bool = False,
):
    """Run Agent vs NPC Prisoner's Dilemma with optional memory and self-consistency."""
    client = create_client()
    logger = TraceLogger(GAME_NAME)
    tools = ["get_game_state", "get_legal_moves", "make_move"]
    memory = ""

    strategy_label = game.opponent_strategy.replace("_", " ").title()
    print("=" * 60)
    print(f"PRISONER'S DILEMMA — Agent vs {strategy_label}")
    print(f"Provider : {client.provider.upper()} / {client.default_model}")
    print(f"Memory   : {'on' if use_memory else 'off'} | Self-consistency k={self_consistency_k}")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        print(f"\n── Round {round_num} ──")
        logger.start_round(round_num, snapshot_game_state(game))
        context = build_game_context(game, memory=memory)
        system = build_system_prompt(GAME_SYSTEM_PROMPT, game_name=GAME_NAME)

        if self_consistency_k > 1:
            print(f"  [Self-consistency] Sampling {self_consistency_k} agents...")
            votes = []
            for k in range(self_consistency_k):
                a = Agent(client=client, system=system, name=f"Agent_k{k}")
                m = _run_react_round(
                    a, game, context, logger, round_num, tools,
                    game.get_game_state, max_iter, dry_run=True,
                )
                votes.append(m)
            move = Counter(votes).most_common(1)[0][0]
            print(f"  [Self-consistency] Votes: {Counter(votes)} → applying: {move}")
            obs = game.make_move(move)
            logger.record_round_result(round_num, {
                "observation": obs,
                "history": game.history[-1] if game.history else None,
                "agent_score": game.agent_score,
                "opponent_score": game.opponent_score,
            })
            print(f"Observation: {obs}")
        else:
            agent = Agent(client=client, system=system, name="Agent")
            _run_react_round(
                agent, game, context, logger, round_num, tools,
                game.get_game_state, max_iter,
            )

        if use_memory and game.history and not game.is_over():
            print("\n  [Memory] Generating Chain-of-Hindsight reflection...")
            memory = generate_reflection(game.history, client)
            print(f"  [Memory] {memory}\n")

    print("\n" + "=" * 60)
    print(f"GAME OVER — You: {game.agent_score} | Opponent: {game.opponent_score}")
    if game.agent_score > game.opponent_score:
        print("Agent wins!")
    elif game.opponent_score > game.agent_score:
        print("Opponent wins.")
    else:
        print("Draw.")

    print("\nMove history:")
    for i, (a, o) in enumerate(game.history, 1):
        print(f"  Round {i}: You={a:<10} Opponent={o}")

    logger.finish({
        "agent_score": game.agent_score,
        "opponent_score": game.opponent_score,
        "history": game.history,
        "opponent_strategy": game.opponent_strategy,
    })
    trace_path = logger.save()
    print(f"\nTrace saved to: {trace_path}")

    if auto_judge:
        print("\n" + "─" * 60)
        print("  TRACER JUDGE — localizing reasoning failures...")
        print("─" * 60)
        judge_result = judge_trace(trace_path, client=client, append_to_reflections=True)
        write_judge_result(trace_path, judge_result)
        for f in judge_result.get("failures", []):
            print(f"  Round {f.get('round','?')} Step {f.get('step','?')}: [{f.get('category','?')}]")
            print(f"    Reason : {f.get('explanation','')}")
            print(f"    Fix    : {f.get('suggested_fix','')}")
        reflection = judge_result.get("reflection", "")
        if reflection:
            print(f"\n  Reflection saved → agent/reflections/{GAME_NAME}.md")
            print(f"  Lesson: {reflection}")


def run_agent_vs_agent(rounds: int = 5, max_iter: int = 10, use_memory: bool = False):
    """Run Agent A vs Agent B Prisoner's Dilemma — moves revealed simultaneously."""
    client = create_client()
    game = SymmetricPD(rounds=rounds)
    logger = TraceLogger(GAME_NAME_AVA)
    tools = ["get_game_state", "get_legal_moves", "make_move"]
    system = build_system_prompt(AVA_SYSTEM_PROMPT, game_name=GAME_NAME_AVA)
    memory_a = memory_b = ""

    print("=" * 60)
    print("  PRISONER'S DILEMMA — Agent A vs Agent B")
    print(f"  Provider : {client.provider.upper()} / {client.default_model}")
    print(f"  Rounds: {rounds}  |  Moves revealed simultaneously")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        print(f"\n{'─'*60}")
        print(f"  ROUND {round_num}  |  Score: Agent_A {game.score_a}  vs  Agent_B {game.score_b}")
        print(f"{'─'*60}")

        logger.start_round(round_num, {
            "score_a": game.score_a,
            "score_b": game.score_b,
            "history": game.history,
        })

        context_a = build_ava_context(game, "A", memory_a)
        context_b = build_ava_context(game, "B", memory_b)

        print("\n  [Agent A reasoning...]")
        agent_a = Agent(client=client, system=system, name="Agent_A")
        move_a = _run_react_round(
            agent_a, game, context_a, logger, round_num, tools,
            lambda: game.get_state_for("A"), max_iter, dry_run=True,
        )

        print("\n  [Agent B reasoning...]")
        agent_b = Agent(client=client, system=system, name="Agent_B")
        move_b = _run_react_round(
            agent_b, game, context_b, logger, round_num, tools,
            lambda: game.get_state_for("B"), max_iter, dry_run=True,
        )

        obs_a, obs_b = game.apply_moves(move_a, move_b)
        pts_a, pts_b = SymmetricPD.PAYOFFS[(move_a, move_b)]

        print(f"\n  MOVES REVEALED:")
        print(f"     Agent_A → {move_a.upper()}")
        print(f"     Agent_B → {move_b.upper()}")
        print(f"     Points: Agent_A +{pts_a}  |  Agent_B +{pts_b}")

        logger.record_round_result(round_num, {
            "move_a": move_a, "move_b": move_b,
            "pts_a": pts_a, "pts_b": pts_b,
            "score_a": game.score_a, "score_b": game.score_b,
        })

        if use_memory and game.history and not game.is_over():
            print("\n  [Memory] Generating reflections...")
            memory_a = generate_reflection([(a, b) for a, b in game.history], client)
            memory_b = generate_reflection([(b, a) for a, b in game.history], client)

    print(f"\n{'='*60}")
    print("  GAME OVER")
    print(f"  Final: Agent_A {game.score_a}  |  Agent_B {game.score_b}")
    if game.score_a > game.score_b:
        print("  Agent_A wins!")
    elif game.score_b > game.score_a:
        print("  Agent_B wins!")
    else:
        print("  Draw.")

    print("\n  Move history:")
    for i, (ma, mb) in enumerate(game.history, 1):
        pts = SymmetricPD.PAYOFFS[(ma, mb)]
        print(f"    Round {i}: Agent_A={ma:<10} Agent_B={mb:<10} pts: +{pts[0]} / +{pts[1]}")

    logger.finish({
        "score_a": game.score_a, "score_b": game.score_b,
        "history": game.history,
        "winner": ("agent_a" if game.score_a > game.score_b
                   else "agent_b" if game.score_b > game.score_a else "draw"),
    })
    trace_path = logger.save()
    print(f"\n  Trace saved to: {trace_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma")
    parser.add_argument("--mode", choices=["npc", "ava"], default="npc",
                        help="npc = agent vs scripted opponent, ava = agent vs agent")
    parser.add_argument("--strategy", default="tit_for_tat",
                        choices=PrisonersDilemma.OPPONENT_STRATEGIES,
                        help="NPC opponent strategy (npc mode only)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--memory", action="store_true",
                        help="Enable Chain-of-Hindsight memory between rounds")
    parser.add_argument("--k", type=int, default=1,
                        help="Self-consistency samples (1 = off)")
    parser.add_argument("--judge", action="store_true",
                        help="Auto-run Tracer judge after game (npc mode only)")
    args = parser.parse_args()

    if args.mode == "ava":
        run_agent_vs_agent(rounds=args.rounds, use_memory=args.memory)
    else:
        game = PrisonersDilemma(rounds=args.rounds, opponent_strategy=args.strategy)
        run_game(game, use_memory=args.memory, self_consistency_k=args.k, auto_judge=args.judge)
