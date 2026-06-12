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
from agent.tracing import TraceLogger, parse_react_response, snapshot_game_state  # noqa: E402
from games.prisoners_dilemma import SymmetricPD  # noqa: E402

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

IMPORTANT: Each round has two phases:
1. MESSAGE PHASE — you send a short message to your opponent (max 50 words).
   Messages can be honest, strategic, or deceptive. Your opponent's message will
   be shown to you before you choose your action.
2. ACTION PHASE — you choose cooperate or defect using the Thought/Action/PAUSE loop.

Example session:

Thought: Which action maximizes my probability of winning? My opponent is an AI running the same analysis. If they expect me to cooperate, defecting gives me +5. But if we both defect, we each get +0. I should check the game state first.
Action: get_game_state: now
PAUSE

Observation: Round 2/5. Opponent's last move: cooperate. Score: You 2, Opponent 2.

Thought: Opponent cooperated last round. They may cooperate again expecting reciprocity — but I must weigh whether to exploit now or sustain mutual cooperation for higher total points. Given we have 3 rounds left, sustained cooperation yields more.
Action: make_move: cooperate
PAUSE

Observation: Move recorded. Waiting for opponent's decision...

Decision: cooperate
""".strip()


def build_message_context(game: SymmetricPD, side: str) -> str:
    """Build context for the message phase — game state and prior message history."""
    state = game.get_state_for(side)
    msg_history = getattr(game, "message_history", [])
    if msg_history:
        lines = []
        for i, (ma, mb) in enumerate(msg_history, 1):
            my_msg = ma if side == "A" else mb
            opp_msg = mb if side == "A" else ma
            lines.append(f"  Round {i}: You said: \"{my_msg}\" | Opponent said: \"{opp_msg}\"")
        msg_block = "\nMESSAGE HISTORY:\n" + "\n".join(lines)
    else:
        msg_block = "\nNo messages exchanged yet."

    return (
        f"You are playing Prisoner's Dilemma against another AI agent. "
        f"Maximize your total score over {game.rounds} rounds.\n\n"
        f"RULES:\n"
        f"- Both cooperate  → both get +2 points.\n"
        f"- You cooperate, they defect  → you -1, they +5.\n"
        f"- You defect, they cooperate  → you +5, they -1.\n"
        f"- Both defect  → both get +0 points.\n\n"
        f"CURRENT STATE:\n{state}\n"
        f"{msg_block}"
    )


def run_round_for_agent(
    agent: Agent,
    side: str,
    game: SymmetricPD,
    logger: TraceLogger,
    round_num: int,
    opponent_message: str = "",
    max_iter: int = 10,
) -> tuple[str, str]:
    """
    Run the ReAct loop for one agent. Returns (move, reasoning_summary).
    The move is captured but NOT applied until both agents have decided.
    The agent already has the message phase in its conversation history.
    """
    tools = ["get_game_state", "get_legal_moves", "make_move"]

    opp_msg_block = ""
    if opponent_message:
        opp_msg_block = (
            f"\nOPPONENT'S MESSAGE THIS ROUND:\n\"{opponent_message}\"\n"
            "Remember: messages can be honest or deceptive. Weigh this against their actions.\n"
        )

    next_prompt = f"""
Now it's the ACTION PHASE. Choose cooperate or defect.
{opp_msg_block}
CURRENT STATE:
{game.get_state_for(side)}

LEGAL MOVES:
{game.get_legal_moves()}

Consider what you just said to your opponent and what they said to you.
Your opponent cannot see your action before they commit.
Use the Thought/Action/PAUSE loop to decide.
""".strip()

    captured_move = None
    reasoning_parts = []
    label = agent.name

    for step in range(1, max_iter + 1):
        state_before = snapshot_game_state(game)
        result = agent(next_prompt)
        if not result:
            break

        parsed = parse_react_response(result)

        if parsed.get("thought"):
            reasoning_parts.append(parsed["thought"])
            print(f"     [{label}] Step {step} Thought: {parsed['thought']}")

        if parsed.get("has_pause") and parsed.get("action"):
            chosen_tool = parsed["action"].strip()
            arg = parsed.get("argument", "").strip()
            print(f"     [{label}] Step {step} Action: {chosen_tool}: {arg}")

            if chosen_tool == "get_game_state":
                obs = game.get_state_for(side)
            elif chosen_tool == "get_legal_moves":
                obs = str(game.get_legal_moves())
            elif chosen_tool == "make_move":
                move = arg.strip().lower()
                if move in ("cooperate", "defect"):
                    captured_move = move
                    obs = f"Move '{move}' recorded. Waiting for opponent's decision..."
                    print(f"     [{label}] Step {step} Observation: {obs}")
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       observation=obs, state_before=state_before,
                                       state_after=snapshot_game_state(game))
                    next_prompt = f"Observation: {obs}"
                    agent(next_prompt)
                    break
                else:
                    obs = f"Invalid move '{move}'. Must be 'cooperate' or 'defect'."
            else:
                obs = f"Unknown tool '{chosen_tool}'. Available: {tools}"

            print(f"     [{label}] Step {step} Observation: {obs}")
            logger.record_step(round_num, step, next_prompt, result, parsed,
                               observation=obs, state_before=state_before,
                               state_after=snapshot_game_state(game))
            next_prompt = f"Observation: {obs}"

        elif parsed.get("decision"):
            decision = parsed["decision"].strip().lower()
            print(f"     [{label}] Step {step} Decision: {decision}")
            logger.record_step(round_num, step, next_prompt, result, parsed,
                               state_before=state_before,
                               state_after=snapshot_game_state(game))
            if decision in ("cooperate", "defect"):
                captured_move = decision
            break

        else:
            obs = "No valid Action or Decision found. Use Action: tool_name: argument."
            logger.record_step(round_num, step, next_prompt, result, parsed,
                               observation=obs, state_before=state_before,
                               state_after=snapshot_game_state(game))
            next_prompt = f"Observation: {obs}"

    return (captured_move or "cooperate"), " ".join(reasoning_parts)


def run_agent_vs_agent(rounds: int = 5, max_iter: int = 10, model_a: str | None = None, model_b: str | None = None):
    """Run a full Agent A vs Agent B Prisoner's Dilemma game."""
    client = create_client()
    game = SymmetricPD(rounds=rounds)
    logger = TraceLogger(GAME_NAME)

    system = build_system_prompt(SCOT_GAME_PROMPT, game_name=GAME_NAME)

    label_a = model_a or client.default_model
    label_b = model_b or client.default_model
    print("=" * 60)
    print("  PRISONER'S DILEMMA — Agent A vs Agent B")
    print(f"  Provider : {client.provider.upper()}")
    print(f"  Agent_A  : {label_a}")
    print(f"  Agent_B  : {label_b}")
    print(f"  Rounds: {rounds}  |  Both agents: independent ReAct + SCoT")
    print("  Moves revealed simultaneously after both decide.")
    print("=" * 60)

    if not hasattr(game, "message_history"):
        game.message_history = []

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

        agent_a = Agent(client=client, system=system, model=model_a, name="Agent_A")
        agent_b = Agent(client=client, system=system, model=model_b, name="Agent_B")

        # ── MESSAGE PHASE ──
        msg_context_a = build_message_context(game, "A")
        msg_context_b = build_message_context(game, "B")

        print("\n  [Message Phase]")
        msg_a = agent_a.generate_message(msg_context_a)
        print(f"     Agent_A says: \"{msg_a}\"")

        msg_b = agent_b.generate_message(msg_context_b)
        print(f"     Agent_B says: \"{msg_b}\"")

        game.message_history.append((msg_a, msg_b))

        logger.record_step(round_num, 0, "message_phase", "", {
            "thought": "", "action": "send_message", "argument": "",
            "has_pause": False, "decision": None, "parse_error": None,
        }, observation=f"Agent_A: \"{msg_a}\" | Agent_B: \"{msg_b}\"",
           state_before=snapshot_game_state(game),
           state_after=snapshot_game_state(game))

        # ── ACTION PHASE ──
        print("\n  [Agent A action...]")
        move_a, reasoning_a = run_round_for_agent(agent_a, "A", game, logger, round_num, opponent_message=msg_b, max_iter=max_iter)

        print("\n  [Agent B action...]")
        move_b, reasoning_b = run_round_for_agent(agent_b, "B", game, logger, round_num, opponent_message=msg_a, max_iter=max_iter)

        obs_a, obs_b = game.apply_moves(move_a, move_b)
        pts_a, pts_b = SymmetricPD.PAYOFFS[(move_a, move_b)]

        print(f"\n  MOVES REVEALED:")
        print(f"     Agent_A → {move_a.upper()}")
        print(f"     Agent_B → {move_b.upper()}")
        print(f"     Points this round: Agent_A +{pts_a}  |  Agent_B +{pts_b}")


        logger.record_round_result(round_num, {
            "move_a": move_a, "move_b": move_b,
            "msg_a": msg_a, "msg_b": msg_b,
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
    parser.add_argument("--model-a", default=None, help="Model for Agent A (e.g. gemini-2.5-flash)")
    parser.add_argument("--model-b", default=None, help="Model for Agent B (e.g. gemini-2.0-flash)")
    args = parser.parse_args()
    run_agent_vs_agent(rounds=args.rounds, model_a=args.model_a, model_b=args.model_b)
