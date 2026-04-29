"""
Prisoner's Dilemma: Agent A vs Agent B — two LLM agents compete.

Both agents run the same ReAct + SCoT reasoning loop. Moves are decided
independently and revealed simultaneously (true PD semantics).
Chain-of-Hindsight reflection is generated per-agent after each round.

Also exposes smarter rule-based opponent strategies from prisoners_dilemma.py:
  pavlov, grim_trigger, generous_tft   (add to run_game() in run_pd.py)

Run from the project root:
    python v2/run_pd_vs_agent.py
"""

import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core import LLMClient, Agent, GameTrace, TraceStep
from games.prisoners_dilemma import SymmetricPD, build_sym_context, generate_reflection

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")

# ── System Prompt ─────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop output a Decision.

IMPORTANT: Always start with "Thought:" and reason step by step before acting.
First, reason out loud: "Which action maximizes my probability of winning this round?"
Your opponent is a reasoning AI agent running the same type of analysis — factor that into your choice.

Your available actions:

get_game_state:
e.g. get_game_state: now
Returns the current round, your score, opponent's score, and their last move.

get_legal_moves:
e.g. get_legal_moves: now
Returns the list of legal moves.

make_move:
e.g. make_move: cooperate
Submits your move. Must be exactly one of: cooperate or defect.

Example session:

Thought: First, let's reason out loud. It's round 2. Opponent cooperated last round — they may be running tit-for-tat or a cooperative strategy. If I defect now I gain +5 this round but likely lose future cooperation. Expected value over remaining rounds favors mutual cooperation.
Action: get_game_state: now
PAUSE

Observation: Round 2/5. Opponent's last move: cooperate. Score: You 3, Opponent 3.

Thought: Confirmed. Continuing mutual cooperation gives +3/round. Defecting now gives +5 but likely triggers retaliation (+1/round). Net over 3 more rounds: cooperate = 9, defect now then mutual defect = 5+3 = 8. Cooperation wins.
Action: make_move: cooperate
PAUSE

Observation: Move recorded. Waiting for opponent's decision...

Decision: cooperate
""".strip()

TOOLS = ["get_game_state", "get_legal_moves", "make_move"]


# ── ReAct loop for symmetric PD ───────────────────────────────────────────────

def run_react_round_sym(
    agent: Agent,
    side: str,
    game: SymmetricPD,
    game_context: str,
    trace: GameTrace,
    round_num: int,
    max_iter: int = 10,
) -> str:
    """
    Run one round of the ReAct loop for one agent in symmetric PD.
    Always dry-runs the make_move — the move is captured but not applied until
    both agents have decided (applied in run_agent_vs_agent via apply_moves).

    side: 'A' or 'B' — controls which game state perspective is shown.
    Returns the chosen move (defaults to 'cooperate' if agent never calls make_move).
    """
    next_prompt = game_context
    step = 0
    move = None

    while step < max_iter:
        step += 1
        result = agent(next_prompt)

        thought = ""
        m = re.search(r"Thought:(.*?)(?=Action:|Decision:|$)", result, re.DOTALL)
        if m:
            thought = m.group(1).strip()

        if "PAUSE" in result and "Action" in result:
            action_match = re.findall(
                r"Action:\s*([a-z_]+):\s*(.+)", result, re.IGNORECASE
            )
            if not action_match:
                obs = "Could not parse Action. Format must be: Action: tool_name: argument"
                next_prompt = f"Observation: {obs}"
                print(f"  Observation: {obs}")
                continue

            tool = action_match[0][0].strip().lower()
            arg = action_match[0][1].strip()

            if tool == "get_game_state":
                obs = game.get_state_for(side)
            elif tool == "get_legal_moves":
                obs = str(game.get_legal_moves())
            elif tool == "make_move":
                move = arg.strip().lower()
                obs = f"Move '{move}' recorded. Waiting for opponent's decision..."
                trace.add(TraceStep(
                    round=round_num, step=step, agent_name=agent.name,
                    thought=thought, action="make_move", tool=tool, arg=arg,
                    observation=obs, raw_response=result,
                ))
                print(f"  Observation: {obs}")
                break
            else:
                obs = f"Unknown tool '{tool}'. Available: {TOOLS}"

            if tool != "make_move":
                trace.add(TraceStep(
                    round=round_num, step=step, agent_name=agent.name,
                    thought=thought, action=f"{tool}: {arg}", tool=tool, arg=arg,
                    observation=obs, raw_response=result,
                ))
                next_prompt = f"Observation: {obs}"
                print(f"  Observation: {obs}")
            continue

        if "Decision" in result:
            dec = re.search(r"Decision:\s*(\w+)", result)
            if dec:
                move = dec.group(1).strip().lower()
            break

    return move if move in game.get_legal_moves() else "cooperate"


# ── Main game runner ───────────────────────────────────────────────────────────

def run_agent_vs_agent(
    game: SymmetricPD,
    client: LLMClient,
    max_iter: int = 10,
    use_memory: bool = True,
    save_trace: bool = True,
) -> GameTrace:
    """
    Run a full Agent vs Agent Prisoner's Dilemma game.

    Both agents share the same LLMClient (same API key / provider) but are
    instantiated independently with fresh conversation histories.

    use_memory:   inject Chain-of-Hindsight reflection each round per agent.
    save_trace:   write trace to v2/traces/pd_vs_agent_trace.json.
    """
    trace = GameTrace(
        game_name="prisoners_dilemma_agent_vs_agent",
        provider=client.provider,
        model=client.model,
        metadata={"rounds": game.rounds, "use_memory": use_memory},
    )

    memory_a = ""
    memory_b = ""

    print("=" * 60)
    print("PRISONER'S DILEMMA — Agent A vs Agent B")
    print(f"Provider : {client.provider.upper()} / {client.model}")
    print(f"Rounds   : {game.rounds} | Memory: {'on' if use_memory else 'off'}")
    print("=" * 60)

    while not game.is_over():
        rnd = game.current_round
        print(f"\n{'─' * 60}")
        print(f"  ROUND {rnd}/{game.rounds}")
        print(f"  Score: Agent_A {game.score_a}  |  Agent_B {game.score_b}")
        print(f"{'─' * 60}")

        # ── Agent A decides ────────────────────────────────────────────────────
        print(f"\n  [Agent_A reasoning...]")
        agent_a = Agent(client=client, system=REACT_SYSTEM_PROMPT, name="Agent_A")
        ctx_a = build_sym_context(game, side="A", memory=memory_a)
        move_a = run_react_round_sym(agent_a, "A", game, ctx_a, trace, rnd, max_iter)

        # ── Agent B decides ────────────────────────────────────────────────────
        print(f"\n  [Agent_B reasoning...]")
        agent_b = Agent(client=client, system=REACT_SYSTEM_PROMPT, name="Agent_B")
        ctx_b = build_sym_context(game, side="B", memory=memory_b)
        move_b = run_react_round_sym(agent_b, "B", game, ctx_b, trace, rnd, max_iter)

        # ── Reveal moves simultaneously ────────────────────────────────────────
        obs_a, obs_b = game.apply_moves(move_a, move_b)
        print(f"\n  MOVES REVEALED:")
        print(f"    Agent_A played: {move_a}")
        print(f"    Agent_B played: {move_b}")
        print(f"  {obs_a}")

        # ── Chain-of-Hindsight per agent ───────────────────────────────────────
        if use_memory and not game.is_over():
            print("\n  [Memory] Generating reflections...")
            # Each agent reflects from its own perspective
            history_a = game.history            # (move_a, move_b) — A's moves are [0]
            history_b = [(b, a) for a, b in game.history]  # flip for B's perspective
            memory_a = generate_reflection(history_a, client)
            memory_b = generate_reflection(history_b, client)
            print(f"  [Agent_A memory] {memory_a}")
            print(f"  [Agent_B memory] {memory_b}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    score_line = f"Agent_A {game.score_a}  |  Agent_B {game.score_b}"
    if game.score_a > game.score_b:
        result_str = "Agent_A wins!"
    elif game.score_b > game.score_a:
        result_str = "Agent_B wins!"
    else:
        result_str = "Draw."
    print(f"FINAL SCORE: {score_line}")
    print(result_str)

    print("\nMove history:")
    for i, (a, b) in enumerate(game.history, 1):
        print(f"  Round {i}: Agent_A={a:<10} Agent_B={b}")

    trace.outcome = f"{result_str} | {score_line}"

    if save_trace:
        os.makedirs(TRACES_DIR, exist_ok=True)
        trace.save(os.path.join(TRACES_DIR, "pd_vs_agent_trace.json"))

    return trace


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = LLMClient()
    game = SymmetricPD(rounds=5)
    run_agent_vs_agent(game, client, max_iter=10, use_memory=True)
