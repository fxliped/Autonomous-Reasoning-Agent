"""
Prisoner's Dilemma — local practice runners (NPC and agent-vs-agent).

For the real tournament, use tournament_agent.py and altruagent_runner.py.

Modes:
  python games/prisoners_dilemma.py                           # agent vs tit-for-tat
  python games/prisoners_dilemma.py --strategy always_defect
  python games/prisoners_dilemma.py --mode ava                # agent vs agent
  python games/prisoners_dilemma.py --memory                  # chain-of-hindsight on
  python games/prisoners_dilemma.py --k 3                     # self-consistency voting
  python games/prisoners_dilemma.py --judge                   # auto-judge trace after
"""

import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.tracing import (  # noqa: E402
    TraceLogger, parse_react_response, snapshot_game_state,
    judge_trace, write_judge_result,
)
from games.pd_game import PrisonersDilemma, SymmetricPD  # noqa: E402

GAME_NAME = "prisoners_dilemma"
GAME_NAME_AVA = "prisoners_dilemma_agent_vs_agent"

# =============================================================================
# SYSTEM PROMPTS
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
Submits your move. Use EXACTLY one of: cooperate or defect.

Example session:

Thought: Round 3, opponent cooperated twice. Mutual cooperation gives +2/round.
Defecting now gets +5 but triggers retaliation. Expected value favors cooperation.
Action: get_game_state: now
PAUSE

Observation: Round 3/10. Opponent's last move: cooperate. Score: You 4, Opponent 4.

Thought: Confirmed cooperator. Rational move is to sustain mutual cooperation.
Action: make_move: cooperate
PAUSE

Observation: Move accepted. You cooperate, opponent cooperate. You +2, Opponent +2.

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

Thought: My opponent is an AI. If they cooperate, defecting gets +5; if they defect,
cooperating gets -1. Check state first.
Action: get_game_state: now
PAUSE

Observation: Round 2/10. Opponent's last move: cooperate. Score: You 2, Opponent 2.

Thought: Opponent cooperated. Sustaining cooperation yields more over remaining rounds.
Action: make_move: cooperate
PAUSE

Observation: Move accepted. You cooperate, opponent cooperate. You +2, Opponent +2.

Decision: cooperate
""".strip()


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def build_game_context(game: PrisonersDilemma, memory: str = "") -> str:
    memory_block = f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n" if memory.strip() else ""
    return f"""
You are playing Prisoner's Dilemma. Maximize your total score over {game.rounds} rounds.

RULES:
- Both cooperate:            you +2, them +2
- You cooperate, they defect: you -1, them +5
- You defect, they cooperate: you +5, them -1
- Both defect:               both get 0
{memory_block}
CURRENT STATE:
{game.get_game_state()}

LEGAL MOVES:
{game.get_legal_moves()}

Reason out loud, then use the Thought/Action/PAUSE loop to decide.
""".strip()


def build_ava_context(game: SymmetricPD, side: str, memory: str = "") -> str:
    memory_block = f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n" if memory.strip() else ""
    return f"""
You are playing Prisoner's Dilemma against another AI agent. Maximize your score over {game.rounds} rounds.
Both decide simultaneously — your opponent cannot see your choice before committing.

RULES:
- Both cooperate:            you +2, them +2
- You cooperate, they defect: you -1, them +5
- You defect, they cooperate: you +5, them -1
- Both defect:               both get 0
{memory_block}
CURRENT STATE:
{game.get_state_for(side)}

LEGAL MOVES:
{game.get_legal_moves()}

Your opponent is a reasoning AI — think about what they will likely do and best-respond.
Use the Thought/Action/PAUSE loop to decide.
""".strip()


def generate_reflection(history: list[tuple[str, str]], client) -> str:
    """Chain-of-Hindsight: LLM reflects on match so far, returns a 2-3 sentence insight."""
    history_str = "\n".join(
        f"  Round {i+1}: You={a}, Opponent={o}" for i, (a, o) in enumerate(history)
    )
    return client.complete(
        system="You are a strategic game analyst. Provide concise, actionable insights.",
        messages=[{"role": "user", "content": (
            f"You just played these rounds of Prisoner's Dilemma:\n{history_str}\n\n"
            "In 2-3 sentences: What patterns did you observe? What should you adjust? Be concise."
        )}],
    )


# =============================================================================
# REACT LOOP
# =============================================================================

_PD_TOOLS = ["get_game_state", "get_legal_moves", "make_move"]


def _run_react_round(
    agent: Agent,
    game,
    context: str,
    logger: TraceLogger,
    round_num: int,
    get_state_fn,
    max_iter: int = 10,
    dry_run: bool = False,
) -> str:
    """
    Run one round of the ReAct loop.
    dry_run=True captures the move without mutating game state (for self-consistency voting).
    Returns the chosen move string, or 'cooperate' as fallback.
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
                obs = f"[dry-run] Intended move: {move}" if dry_run else game.make_move(arg)
                move_made = move if move in ("cooperate", "defect") else None
                logger.record_step(round_num, step, next_prompt, result, parsed,
                                   observation=obs, state_before=state_before,
                                   state_after=snapshot_game_state(game))
                if not dry_run:
                    logger.record_round_result(round_num, {
                        "observation": obs,
                        "history": game.history[-1] if game.history else None,
                        "agent_score": game.agent_score,
                        "opponent_score": game.opponent_score,
                    })
                print(f"Observation: {obs}")
                break
            else:
                obs = f"Unknown tool '{chosen_tool}'. Available: {_PD_TOOLS}"

            logger.record_step(round_num, step, next_prompt, result, parsed,
                               observation=obs, state_before=state_before,
                               state_after=state_before)
            next_prompt = f"Observation: {obs}"
            print(next_prompt)
            continue

        if parsed["decision"]:
            logger.record_step(round_num, step, next_prompt, result, parsed,
                               state_before=state_before, state_after=state_before)
            break

        obs = "No valid Action or Decision found. Use Action: tool_name: argument."
        logger.record_step(round_num, step, next_prompt, result, parsed,
                           observation=obs, state_before=state_before,
                           state_after=state_before)
        next_prompt = f"Observation: {obs}"

    return move_made or "cooperate"


# =============================================================================
# RUNNERS
# =============================================================================

def run_game(
    game: PrisonersDilemma,
    max_iter: int = 10,
    use_memory: bool = False,
    self_consistency_k: int = 1,
    auto_judge: bool = False,
) -> None:
    """Agent vs scripted NPC with optional memory and self-consistency."""
    client = create_client()
    logger = TraceLogger(GAME_NAME)
    memory = ""

    print("=" * 60)
    print(f"PRISONER'S DILEMMA — Agent vs {game.opponent_strategy.replace('_',' ').title()}")
    print(f"Provider : {client.provider.upper()} / {client.default_model}")
    print(f"Memory   : {'on' if use_memory else 'off'} | k={self_consistency_k}")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        print(f"\n── Round {round_num} ──")
        logger.start_round(round_num, snapshot_game_state(game))
        context = build_game_context(game, memory=memory)
        system = build_system_prompt(GAME_SYSTEM_PROMPT, game_name=GAME_NAME)

        if self_consistency_k > 1:
            print(f"  [Self-consistency] Sampling {self_consistency_k} agents...")
            votes = [
                _run_react_round(
                    Agent(client=client, system=system, name=f"Agent_k{k}"),
                    game, context, logger, round_num,
                    game.get_game_state, max_iter, dry_run=True,
                )
                for k in range(self_consistency_k)
            ]
            move = Counter(votes).most_common(1)[0][0]
            print(f"  [Self-consistency] Votes: {Counter(votes)} → {move}")
            obs = game.make_move(move)
            logger.record_round_result(round_num, {
                "observation": obs,
                "history": game.history[-1] if game.history else None,
                "agent_score": game.agent_score,
                "opponent_score": game.opponent_score,
            })
            print(f"Observation: {obs}")
        else:
            _run_react_round(
                Agent(client=client, system=system, name="Agent"),
                game, context, logger, round_num, game.get_game_state, max_iter,
            )

        if use_memory and game.history and not game.is_over():
            print("\n  [Memory] Generating reflection...")
            memory = generate_reflection(game.history, client)
            print(f"  {memory}\n")

    print("\n" + "=" * 60)
    print(f"GAME OVER — You: {game.agent_score} | Opponent: {game.opponent_score}")
    print("Agent wins!" if game.agent_score > game.opponent_score else
          "Opponent wins." if game.opponent_score > game.agent_score else "Draw.")

    print("\nMove history:")
    for i, (a, o) in enumerate(game.history, 1):
        print(f"  Round {i}: You={a:<10} Opponent={o}")

    logger.finish({
        "agent_score": game.agent_score, "opponent_score": game.opponent_score,
        "history": game.history, "opponent_strategy": game.opponent_strategy,
    })
    trace_path = logger.save()
    print(f"\nTrace saved: {trace_path}")

    if auto_judge:
        print("\n" + "─" * 60)
        print("  JUDGE — localizing reasoning failures...")
        judge_result = judge_trace(trace_path, client=client, append_to_reflections=True)
        write_judge_result(trace_path, judge_result)
        for f in judge_result.get("failures", []):
            print(f"  R{f.get('round','?')} S{f.get('step','?')}: [{f.get('category','?')}] {f.get('explanation','')}")
        if judge_result.get("reflection"):
            print(f"\n  Lesson: {judge_result['reflection']}")


def run_agent_vs_agent(rounds: int = 10, max_iter: int = 10, use_memory: bool = False) -> None:
    """Agent A vs Agent B — simultaneous moves."""
    client = create_client()
    game = SymmetricPD(rounds=rounds)
    logger = TraceLogger(GAME_NAME_AVA)
    system = build_system_prompt(AVA_SYSTEM_PROMPT, game_name=GAME_NAME_AVA)
    memory_a = memory_b = ""

    print("=" * 60)
    print(f"  PRISONER'S DILEMMA — Agent A vs Agent B | {rounds} rounds")
    print(f"  Provider: {client.provider.upper()} / {client.default_model}")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        print(f"\n── Round {round_num} | A:{game.score_a} B:{game.score_b} ──")
        logger.start_round(round_num, {"score_a": game.score_a, "score_b": game.score_b})

        print("  [Agent A...]")
        move_a = _run_react_round(
            Agent(client=client, system=system, name="Agent_A"),
            game, build_ava_context(game, "A", memory_a), logger, round_num,
            lambda: game.get_state_for("A"), max_iter, dry_run=True,
        )
        print("  [Agent B...]")
        move_b = _run_react_round(
            Agent(client=client, system=system, name="Agent_B"),
            game, build_ava_context(game, "B", memory_b), logger, round_num,
            lambda: game.get_state_for("B"), max_iter, dry_run=True,
        )

        game.apply_moves(move_a, move_b)
        pts_a, pts_b = SymmetricPD.PAYOFFS[(move_a, move_b)]
        print(f"\n  REVEALED: A={move_a.upper()} B={move_b.upper()} | +{pts_a} / +{pts_b}")

        logger.record_round_result(round_num, {
            "move_a": move_a, "move_b": move_b,
            "pts_a": pts_a, "pts_b": pts_b,
            "score_a": game.score_a, "score_b": game.score_b,
        })

        if use_memory and game.history and not game.is_over():
            memory_a = generate_reflection([(a, b) for a, b in game.history], client)
            memory_b = generate_reflection([(b, a) for a, b in game.history], client)

    print(f"\n{'='*60}")
    print(f"  FINAL: A={game.score_a}  B={game.score_b}")
    print("  " + ("A wins!" if game.score_a > game.score_b else
                  "B wins!" if game.score_b > game.score_a else "Draw."))

    logger.finish({
        "score_a": game.score_a, "score_b": game.score_b,
        "history": game.history,
        "winner": ("agent_a" if game.score_a > game.score_b
                   else "agent_b" if game.score_b > game.score_a else "draw"),
    })
    print(f"\n  Trace: {logger.save()}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma — local practice")
    parser.add_argument("--mode", choices=["npc", "ava"], default="npc")
    parser.add_argument("--strategy", default="tit_for_tat",
                        choices=PrisonersDilemma.OPPONENT_STRATEGIES)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--k", type=int, default=1, help="Self-consistency samples")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    if args.mode == "ava":
        run_agent_vs_agent(rounds=args.rounds, use_memory=args.memory)
    else:
        run_game(
            PrisonersDilemma(rounds=args.rounds, opponent_strategy=args.strategy),
            use_memory=args.memory, self_consistency_k=args.k, auto_judge=args.judge,
        )
