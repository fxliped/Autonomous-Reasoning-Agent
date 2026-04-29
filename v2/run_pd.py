"""
Prisoner's Dilemma V2 — ReAct agent with SCoT, round memory, and trace logging.

New vs. original React_agent_gameV1.py:
  ✓ Dual provider: OpenAI (gpt-5.4-mini) → Gemini fallback
  ✓ SCoT: agent reasons out loud before every action
  ✓ Round memory: Chain-of-Hindsight reflection injected each round
  ✓ Self-consistency: optional majority vote over k samples
  ✓ Full trace logging to v2/traces/pd_trace.json
  ✓ Configurable opponent strategy

Run from the project root:
    python v2/run_pd.py
"""

import re
import os
import sys
from collections import Counter

# Allow imports from the v2/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core import LLMClient, Agent, GameTrace, TraceStep
from games.prisoners_dilemma import PrisonersDilemma, build_game_context, generate_reflection

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")

# ── System Prompt (SCoT version) ───────────────────────────────────────────────
# Key upgrades over V1:
#   - Explicit "reason out loud" instruction before acting (SCoT)
#   - Forgiveness heuristic: single defections may be mistakes, not strategy shifts

REACT_SYSTEM_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output a Decision.

IMPORTANT: Always start with "Thought:" and reason step by step before acting.
First, reason out loud: "Which action maximizes my probability of winning this round?"
Note: your opponent sometimes makes mistakes — a single defection may be an error, not a permanent strategy shift.

Your available actions:

get_game_state:
e.g. get_game_state: now
Returns the current game state, scores, and opponent's last move.

get_legal_moves:
e.g. get_legal_moves: now
Returns the list of legal moves.

make_move:
e.g. make_move: cooperate
Submits your move. Use EXACTLY one of the legal moves: cooperate or defect.

Example session:

Thought: First, let's reason out loud. It's round 3, opponent cooperated the last two rounds. Mutual cooperation gives us both +3/round vs +1 from mutual defection. Defecting now would gain me +5 this round but likely trigger retaliation. Expected value favors cooperation. Let me verify the state before deciding.
Action: get_game_state: now
PAUSE

Observation: Round 3/5. Opponent's last move: cooperate. Score: You 6, Opponent 6.

Thought: Confirmed — opponent is cooperating consistently. The rational move is to cooperate for sustained mutual gain.
Action: make_move: cooperate
PAUSE

Observation: Move accepted. You played cooperate, opponent played cooperate. Points this round: You +3, Opponent +3.

Decision: cooperate
""".strip()

TOOLS = ["get_game_state", "get_legal_moves", "make_move"]


# ── ReAct loop (one round) ────────────────────────────────────────────────────

def run_react_round(
    agent: Agent,
    game: PrisonersDilemma,
    game_context: str,
    trace: GameTrace,
    round_num: int,
    max_iter: int = 10,
    dry_run: bool = False,
) -> str:
    """
    Run one round of the Thought/Action/PAUSE/Observation loop.

    dry_run: if True, intercepts make_move without applying it to game state.
             Used for self-consistency sampling.
    Returns the move chosen (or "cooperate" as fallback).
    """
    next_prompt = game_context
    step = 0
    move_made = None

    while step < max_iter:
        step += 1
        result = agent(next_prompt)
        print(result)

        # Extract Thought block for trace
        thought = ""
        thought_match = re.search(r"Thought:(.*?)(?=Action:|Decision:|$)", result, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        if "PAUSE" in result and "Action" in result:
            action_match = re.findall(
                r"Action:\s*([a-z_]+):\s*(.+)", result, re.IGNORECASE
            )
            if not action_match:
                obs = "Could not parse your Action. Format must be: Action: tool_name: argument"
                next_prompt = f"Observation: {obs}"
                print(next_prompt)
                continue

            tool = action_match[0][0].strip().lower()
            arg = action_match[0][1].strip()

            if tool == "get_game_state":
                obs = game.get_game_state()

            elif tool == "get_legal_moves":
                obs = str(game.get_legal_moves())

            elif tool == "make_move":
                if dry_run:
                    move_made = arg.strip().lower()
                    obs = f"[dry-run] Intended move: {move_made}"
                    trace.add(TraceStep(
                        round=round_num, step=step, agent_name=agent.name,
                        thought=thought, action="make_move", tool=tool, arg=arg,
                        observation=obs, raw_response=result,
                    ))
                    print(f"Observation: {obs}")
                    break
                else:
                    obs = game.make_move(arg)
                    move_made = arg.strip().lower()
                    trace.add(TraceStep(
                        round=round_num, step=step, agent_name=agent.name,
                        thought=thought, action="make_move", tool=tool, arg=arg,
                        observation=obs, raw_response=result,
                    ))
                    print(f"Observation: {obs}")
                    break

            else:
                obs = f"Unknown tool '{tool}'. Available: {TOOLS}"

            trace.add(TraceStep(
                round=round_num, step=step, agent_name=agent.name,
                thought=thought, action=f"{tool}: {arg}", tool=tool, arg=arg,
                observation=obs, raw_response=result,
            ))
            next_prompt = f"Observation: {obs}"
            print(next_prompt)
            continue

        if "Decision" in result:
            break

    return move_made or "cooperate"  # fallback if agent never called make_move


# ── Main game runner ───────────────────────────────────────────────────────────

def run_game(
    game: PrisonersDilemma,
    client: LLMClient,
    max_iter: int = 10,
    use_memory: bool = True,
    self_consistency_k: int = 1,
    save_trace: bool = True,
) -> GameTrace:
    """
    Run a full Prisoner's Dilemma game.

    use_memory:          inject Chain-of-Hindsight reflection each round.
    self_consistency_k:  run decision k times and take majority vote (1 = disabled).
    save_trace:          write trace JSON to v2/traces/pd_trace.json.
    """
    trace = GameTrace(
        game_name="prisoners_dilemma",
        provider=client.provider,
        model=client.model,
        metadata={
            "rounds": game.rounds,
            "opponent_strategy": game.opponent_strategy,
            "use_memory": use_memory,
            "self_consistency_k": self_consistency_k,
        },
    )
    memory = ""

    print("=" * 60)
    print("PRISONER'S DILEMMA V2 — ReAct + SCoT + Memory")
    print(f"Provider : {client.provider.upper()} / {client.model}")
    print(f"Opponent : {game.opponent_strategy} | Rounds: {game.rounds}")
    print(f"Memory   : {'on' if use_memory else 'off'} | "
          f"Self-consistency k={self_consistency_k}")
    print("=" * 60)

    while not game.is_over():
        print(f"\n── Round {game.current_round} ──")
        game_context = build_game_context(game, memory=memory)

        if self_consistency_k > 1:
            # Run k independent agents on the same frozen state, take majority vote
            print(f"  [Self-consistency] Sampling {self_consistency_k} agents...")
            votes = []
            for k in range(self_consistency_k):
                a = Agent(client=client, system=REACT_SYSTEM_PROMPT, name=f"Agent_k{k}")
                m = run_react_round(
                    a, game, game_context, trace,
                    game.current_round, max_iter, dry_run=True
                )
                votes.append(m)
            move = Counter(votes).most_common(1)[0][0]
            print(f"  [Self-consistency] Votes: {Counter(votes)} → applying: {move}")
            obs = game.make_move(move)
            print(f"Observation: {obs}")
        else:
            agent = Agent(client=client, system=REACT_SYSTEM_PROMPT, name="Agent")
            run_react_round(agent, game, game_context, trace, game.current_round, max_iter)

        # Chain-of-Hindsight: reflect after each completed round
        if use_memory and game.history and not game.is_over():
            print("\n  [Memory] Generating Chain-of-Hindsight reflection...")
            memory = generate_reflection(game.history, client)
            print(f"  [Memory] {memory}\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    score_line = f"You {game.agent_score} | Opponent {game.opponent_score}"
    if game.agent_score > game.opponent_score:
        result_str = "Agent wins!"
    elif game.opponent_score > game.agent_score:
        result_str = "Opponent wins."
    else:
        result_str = "Draw."
    print(f"GAME OVER — {score_line}")
    print(result_str)
    print("\nMove history:")
    for i, (a, o) in enumerate(game.history, 1):
        print(f"  Round {i}: Agent={a:<10} Opponent={o}")

    trace.outcome = f"{result_str} | {score_line}"

    if save_trace:
        os.makedirs(TRACES_DIR, exist_ok=True)
        trace.save(os.path.join(TRACES_DIR, "pd_trace.json"))

    return trace


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = LLMClient()  # auto-detects OpenAI → Gemini

    # Opponent strategies (ordered easiest → hardest):
    #   "always_cooperate"  "random"  "tit_for_tat"
    #   "generous_tft"  "pavlov"  "grim_trigger"  "always_defect"
    game = PrisonersDilemma(rounds=5, opponent_strategy="tit_for_tat")
    run_game(game, client, max_iter=10, use_memory=True, self_consistency_k=1)
