"""
Are You the Traitor? V1 — Multi-agent social deduction with ReAct reasoning.

4 AI agents are assigned hidden roles: Hero, Traitor, Healer, Healer.
Each round:
  1. Discussion: every active player makes a public statement (ReAct loop).
  2. Voting: every player votes to eliminate one other player (ReAct loop).
  3. Elimination: most-voted player is removed; role is publicly revealed.
  4. Win check — repeat until someone wins or rounds expire.

This is the first social deduction game in the system.
The Traitor must deceive. Hero and Healers must deduce. Watch the reasoning!

Run from the project root:
    python v2/run_traitor.py
"""

import re
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core import LLMClient, Agent, GameTrace, TraceStep
from games.are_you_traitor import (
    AreYouTheTraitor,
    build_role_system_prompt,
    build_statement_prompt,
    build_vote_prompt,
)

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
TOOLS = ["get_game_state", "review_history", "make_statement", "cast_vote"]


# ── ReAct loop (one phase: statement or vote) ─────────────────────────────────

def run_react_phase(
    player: str,
    agent: Agent,
    game: AreYouTheTraitor,
    initial_prompt: str,
    terminal_tool: str,
    trace: GameTrace,
    max_iter: int = 8,
) -> str:
    """
    Run a ReAct loop for one player in one phase.

    terminal_tool: "make_statement" → returns the statement text.
                   "cast_vote"      → returns the vote target player name.

    The agent can call get_game_state and review_history any number of times
    before submitting its terminal action.
    """
    next_prompt = initial_prompt
    step = 0
    result_value = None

    while step < max_iter:
        step += 1
        result = agent(next_prompt)

        # Print the agent's reasoning for the demo
        print(f"\n  [{player}]")
        print(result)

        thought = ""
        thought_match = re.search(
            r"Thought:(.*?)(?=Action:|Decision:|$)", result, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        if "PAUSE" in result and "Action" in result:
            # For statement phase: make_statement | get_game_state | review_history
            # For vote phase: cast_vote | get_game_state | review_history
            action_match = re.findall(
                r"Action:\s*([\w_]+):\s*(.+)", result, re.IGNORECASE | re.DOTALL
            )
            if not action_match:
                obs = "Could not parse Action. Format: Action: tool_name: argument"
                next_prompt = f"Observation: {obs}"
                continue

            tool = action_match[0][0].strip().lower()
            arg = action_match[0][1].strip().strip('"').rstrip()

            if tool == "get_game_state":
                obs = game.get_public_state()

            elif tool == "review_history":
                obs = game.get_full_history()

            elif tool == terminal_tool:
                if terminal_tool == "cast_vote":
                    others = [p for p in game.active_players if p != player]
                    vote_target = arg.split()[0]  # take first word in case of trailing text
                    if vote_target not in others:
                        obs = (
                            f"Invalid vote '{vote_target}'. "
                            f"Must be one of: {', '.join(others)}"
                        )
                        next_prompt = f"Observation: {obs}"
                        trace.add(TraceStep(
                            round=game.current_round, step=step, agent_name=player,
                            thought=thought, action=f"{tool}: {arg}", tool=tool, arg=arg,
                            observation=obs, raw_response=result,
                        ))
                        continue
                    result_value = vote_target
                else:
                    result_value = arg  # the statement text

                obs = (
                    "Statement recorded." if terminal_tool == "make_statement"
                    else f"Vote for {result_value} recorded."
                )
                trace.add(TraceStep(
                    round=game.current_round, step=step, agent_name=player,
                    thought=thought, action=f"{tool}: {arg}", tool=tool, arg=arg,
                    observation=obs, raw_response=result,
                ))
                print(f"  Observation: {obs}")
                break

            else:
                obs = (
                    f"In this phase, use '{terminal_tool}'. "
                    f"Also available: get_game_state, review_history."
                )

            trace.add(TraceStep(
                round=game.current_round, step=step, agent_name=player,
                thought=thought, action=f"{tool}: {arg}", tool=tool, arg=arg,
                observation=obs, raw_response=result,
            ))
            next_prompt = f"Observation: {obs}"
            print(f"  Observation: {obs[:120]}")
            continue

        # Agent output a Decision without a tool call — extract from Decision line
        if "Decision" in result:
            dec_match = re.search(r"Decision:\s*(.+)", result)
            if dec_match:
                result_value = dec_match.group(1).strip().strip('"')
            break

    # Fallbacks if agent never submitted a valid terminal action
    if result_value is None:
        if terminal_tool == "cast_vote":
            others = [p for p in game.active_players if p != player]
            result_value = random.choice(others) if others else player
            print(f"  [Fallback] {player} random vote → {result_value}")
        else:
            result_value = "I am observing carefully and have no strong suspicions yet."
            print(f"  [Fallback] {player} default statement")

    return result_value


# ── Main game runner ───────────────────────────────────────────────────────────

def run_traitor_game(
    game: AreYouTheTraitor,
    client: LLMClient,
    max_iter: int = 8,
    save_trace: bool = True,
) -> GameTrace:
    """Run a full Are You the Traitor? game between N AI agents."""
    trace = GameTrace(
        game_name="are_you_the_traitor",
        provider=client.provider,
        model=client.model,
        metadata={"num_players": game.num_players, "max_rounds": game.max_rounds},
    )

    # One Agent instance per player, with role-specific system prompts
    agents: dict[str, Agent] = {
        player: Agent(
            client=client,
            system=build_role_system_prompt(player, game.roles[player]),
            name=player,
        )
        for player in game.players
    }

    print("\n" + "=" * 60)
    print("  ARE YOU THE TRAITOR? — Multi-Agent Social Deduction")
    print("=" * 60)
    print(f"  Provider : {client.provider.upper()} / {client.model}")
    print(f"  Players  : {game.num_players} | Max rounds: {game.max_rounds}")
    print(f"\n  SECRET ROLE ASSIGNMENTS (hidden from agents — for your eyes only):")
    for p, r in game.roles.items():
        marker = " ← TRAITOR" if r == "traitor" else (" ← HERO" if r == "hero" else "")
        print(f"    {p}: {r.upper()}{marker}")
    print("=" * 60)

    while not game.is_over():
        print(f"\n\n{'─' * 60}")
        print(f"  ROUND {game.current_round} of {game.max_rounds}")
        print(f"  Active players: {', '.join(game.active_players)}")
        print(f"{'─' * 60}")

        # ── Discussion phase ───────────────────────────────────────────────────
        print(f"\n  📢  DISCUSSION PHASE — each player makes a statement")
        statements: dict[str, str] = {}

        for player in list(game.active_players):
            agents[player].reset()  # fresh history per phase
            prompt = build_statement_prompt(game, player)
            statement = run_react_phase(
                player, agents[player], game, prompt,
                terminal_tool="make_statement", trace=trace, max_iter=max_iter,
            )
            game.record_statement(player, statement)
            statements[player] = statement

        print(f"\n  📋  Statements this round:")
        for p, s in statements.items():
            role_label = f"({game.roles[p].upper()})"
            print(f"    {p} {role_label}: \"{s}\"")

        # ── Voting phase ───────────────────────────────────────────────────────
        print(f"\n  🗳️   VOTING PHASE — each player votes to eliminate one other")
        votes: dict[str, str] = {}

        for player in list(game.active_players):
            agents[player].reset()  # fresh history for vote phase
            prompt = build_vote_prompt(game, player)
            vote = run_react_phase(
                player, agents[player], game, prompt,
                terminal_tool="cast_vote", trace=trace, max_iter=max_iter,
            )
            # Safety: can't vote for self
            others = [p for p in game.active_players if p != player]
            if vote not in others:
                vote = random.choice(others) if others else player
            game.record_vote(player, vote)
            votes[player] = vote

        print(f"\n  📊  Votes cast:")
        for voter, target in votes.items():
            print(f"    {voter} ({game.roles[voter].upper()}) → {target}")

        # ── Tally and eliminate ────────────────────────────────────────────────
        eliminated, vote_counts = game.tally_votes()
        print(f"\n  📊  Vote tally: {vote_counts}")
        role_revealed, game_over = game.eliminate(eliminated)

        print(f"\n  🚨  ELIMINATED: {eliminated}")
        print(f"      Role revealed: {role_revealed.upper()}")

        if role_revealed == "traitor":
            print("      ⚠️  The Traitor has been found!")
        elif role_revealed == "hero":
            print("      ⚠️  The Hero has been eliminated!")

        if game_over:
            break

    # ── Final result ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GAME OVER")
    print("=" * 60)

    if game.winner == "village":
        print("  🏘️  VILLAGE WINS — The Traitor was found!")
    else:
        print("  🗡️  TRAITOR WINS — The deception succeeded!")

    print(f"\n  Full role reveal:")
    for p, r in game.roles.items():
        status = "(eliminated)" if p not in game.active_players else "(survived)"
        print(f"    {p}: {r.upper()} {status}")

    trace.outcome = f"Winner: {game.winner}"
    trace.metadata["eliminated_sequence"] = [
        {"player": p, "role": r} for p, r in game.eliminated
    ]

    if save_trace:
        os.makedirs(TRACES_DIR, exist_ok=True)
        trace.save(os.path.join(TRACES_DIR, "traitor_trace.json"))

    return trace


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = LLMClient()  # auto-detects OpenAI → Gemini
    game = AreYouTheTraitor(num_players=4, max_rounds=4)
    run_traitor_game(game, client, max_iter=8)
