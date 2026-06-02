"""
Prisoner's Dilemma: Agent A vs Agent B — two LLM agents compete.

Both agents run the same ReAct + SCoT reasoning loop. Moves are decided
independently and revealed simultaneously (true PD semantics).

Run:
    python games/pd_agent_vs_agent.py
    python games/pd_agent_vs_agent.py --rounds 7
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.tracing import TraceLogger, parse_react_response  # noqa: E402
from games.pd_game import SymmetricPD  # noqa: E402
from games.prisoners_dilemma import build_ava_context  # noqa: E402

GAME_NAME = "prisoners_dilemma_agent_vs_agent"

SCOT_GAME_PROMPT = """
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

Thought: Which action maximizes my probability of winning? My opponent is an AI running the same analysis. If they expect me to cooperate, defecting gives me +5. But if we both defect, we each get +1. I should check the game state first.
Action: get_game_state: now
PAUSE

Observation: Round 2/5. Opponent's last move: cooperate. Score: You 3, Opponent 3.

Thought: Opponent cooperated last round. They may cooperate again expecting reciprocity — but I must weigh whether to exploit now or sustain mutual cooperation for higher total points. Given we have 3 rounds left, sustained cooperation yields more.
Action: make_move: cooperate
PAUSE

Observation: Move recorded. Waiting for opponent's decision...

Decision: cooperate
""".strip()


def run_round_for_agent(
    agent: Agent,
    side: str,
    game: SymmetricPD,
    max_iter: int = 10,
) -> tuple[str, str]:
    """
    Run the ReAct loop for one agent. Returns (move, reasoning_summary).
    The move is captured but NOT applied until both agents have decided.
    """
    tools = ["get_game_state", "get_legal_moves", "make_move"]
    next_prompt = build_ava_context(game, side)

    captured_move = None
    reasoning_parts = []

    for _ in range(max_iter):
        result = agent(next_prompt)
        if not result:
            break

        parsed = parse_react_response(result)
        if parsed.get("thought"):
            reasoning_parts.append(parsed["thought"])

        if parsed.get("has_pause") and parsed.get("action"):
            chosen_tool = parsed["action"].strip()
            arg = parsed.get("argument", "").strip()

            if chosen_tool == "get_game_state":
                obs = game.get_state_for(side)
            elif chosen_tool == "get_legal_moves":
                obs = str(game.get_legal_moves())
            elif chosen_tool == "make_move":
                move = arg.strip().lower()
                if move in ("cooperate", "defect"):
                    captured_move = move
                    obs = f"Move '{move}' recorded. Waiting for opponent's decision..."
                    next_prompt = f"Observation: {obs}"
                    agent(next_prompt)
                    break
                else:
                    obs = f"Invalid move '{move}'. Must be 'cooperate' or 'defect'."
            else:
                obs = f"Unknown tool '{chosen_tool}'. Available: {tools}"

            next_prompt = f"Observation: {obs}"

        elif parsed.get("decision"):
            decision = parsed["decision"].strip().lower()
            if decision in ("cooperate", "defect"):
                captured_move = decision
            break

    return (captured_move or "cooperate"), " ".join(reasoning_parts)


def run_agent_vs_agent(rounds: int = 5, max_iter: int = 10):
    """Run a full Agent A vs Agent B Prisoner's Dilemma game."""
    client = create_client()
    game = SymmetricPD(rounds=rounds)
    logger = TraceLogger(GAME_NAME)

    system = build_system_prompt(SCOT_GAME_PROMPT, game_name=GAME_NAME)

    print("=" * 60)
    print("  PRISONER'S DILEMMA — Agent A vs Agent B")
    print(f"  Provider : {client.provider.upper()} / {client.default_model}")
    print(f"  Rounds: {rounds}  |  Both agents: independent ReAct + SCoT")
    print("  Moves revealed simultaneously after both decide.")
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

        agent_a = Agent(client=client, system=system, name="Agent_A")
        agent_b = Agent(client=client, system=system, name="Agent_B")

        print("\n  [Agent A reasoning...]")
        move_a, reasoning_a = run_round_for_agent(agent_a, "A", game, max_iter)

        print("\n  [Agent B reasoning...]")
        move_b, reasoning_b = run_round_for_agent(agent_b, "B", game, max_iter)

        obs_a, obs_b = game.apply_moves(move_a, move_b)
        pts_a, pts_b = SymmetricPD.PAYOFFS[(move_a, move_b)]

        print(f"\n  MOVES REVEALED:")
        print(f"     Agent_A → {move_a.upper()}")
        print(f"     Agent_B → {move_b.upper()}")
        print(f"     Points this round: Agent_A +{pts_a}  |  Agent_B +{pts_b}")

        print(f"\n  Agent A thought: {reasoning_a[:250].strip()}{'...' if len(reasoning_a) > 250 else ''}")
        print(f"  Agent B thought: {reasoning_b[:250].strip()}{'...' if len(reasoning_b) > 250 else ''}")

        logger.record_round_result(round_num, {
            "move_a": move_a, "move_b": move_b,
            "pts_a": pts_a, "pts_b": pts_b,
            "score_a": game.score_a, "score_b": game.score_b,
        })

    print(f"\n{'='*60}")
    print("  GAME OVER")
    print(f"  Final Score: Agent_A {game.score_a}  |  Agent_B {game.score_b}")
    if game.score_a > game.score_b:
        print("  Agent_A wins!")
    elif game.score_b > game.score_a:
        print("  Agent_B wins!")
    else:
        print("  Draw.")

    print("\n  Move history:")
    for i, (ma, mb) in enumerate(game.history, 1):
        outcome = SymmetricPD.PAYOFFS[(ma, mb)]
        print(f"    Round {i}: Agent_A={ma:<10}  Agent_B={mb:<10}  pts: +{outcome[0]} / +{outcome[1]}")

    logger.finish({
        "score_a": game.score_a,
        "score_b": game.score_b,
        "history": game.history,
        "winner": ("agent_a" if game.score_a > game.score_b
                   else "agent_b" if game.score_b > game.score_a else "draw"),
    })
    trace_path = logger.save()
    print(f"\n  Trace saved to: {trace_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma: Agent A vs Agent B")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    run_agent_vs_agent(rounds=args.rounds)
