"""
=============================================================================
ReAct Game Agent — Prisoner's Dilemma
=============================================================================

WHAT IS THIS?
-------------
This file implements a general-purpose AI game-playing agent using the
ReAct (Reasoning + Acting) architecture. ReAct is a framework where the
agent alternates between:

    Thought  → the agent reasons about the current situation
    Action   → the agent calls a tool (e.g. make_move)
    PAUSE    → the agent waits for the tool result
    Observation → the result comes back, and the loop repeats

The agent is "general" because the core reasoning loop never changes.
Only the game context (rules, state, legal moves) is swapped out per game.
This means when the professor releases the tournament game, we only need
to write a new game class — the agent brain stays the same.

ARCHITECTURE OVERVIEW:
----------------------

    Game Context (rules + state)
            ↓
    ┌───────────────────────────┐
    │     ReAct Agent Core      │  ← never changes between games
    │  Thought → Action → Obs  │
    └───────────────┬───────────┘
                    ↓
            Tool Dispatcher
           /       |        \
    get_state  make_move  get_legal_moves
                    ↓
            Game Environment  ← swap this for new games

HOW TO ADAPT THIS FOR A NEW GAME:
----------------------------------
1. Write a new game class with these three methods:
       - get_game_state()   → returns a string describing the current state
       - get_legal_moves()  → returns a list of valid moves
       - make_move(move)    → applies the move, returns result string
       - is_over()          → returns True when the game ends

2. Update build_game_context() to describe the new game's rules.

3. Everything else (Agent class, agent_loop, tool dispatcher) stays the same.

"""

import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client
from agent.tracing import TraceLogger, parse_react_response, snapshot_game_state, judge_trace, write_judge_result


client = create_client()
GAME_NAME = "prisoners_dilemma"


# =============================================================================
# SECTION 1: SYSTEM PROMPT
# =============================================================================
# This is the "brain" of the agent — it never changes regardless of which
# game is being played. It defines the Thought/Action/PAUSE/Observation loop
# that makes this a ReAct agent.
#
#by telling the LLM to always PAUSE after an Action,
# we force it to wait for real tool results instead of hallucinating them.
# =============================================================================

GAME_SYSTEM_PROMPT = """
Your available actions are:

get_game_state:
e.g. get_game_state: now
Returns the current game state and your score.

get_legal_moves:
e.g. get_legal_moves: now
Returns a list of moves you are allowed to make.

make_move:
e.g. make_move: cooperate
Submits your chosen move. Use EXACTLY one of the legal moves.

Example session:

Thought: I need to see the current state before deciding.
Action: get_game_state: now
PAUSE

Observation: Round 3. Opponent cooperated last 2 rounds. Score: You 6, Opponent 6.

Thought: Opponent is cooperating. I should cooperate to keep mutual benefit.
Action: get_legal_moves: now
PAUSE

Observation: ['cooperate', 'defect']

Thought: I'll cooperate since opponent has been cooperative.
Action: make_move: cooperate
PAUSE

Observation: Move accepted.

Decision: cooperate
""".strip()

# =============================================================================
# SECTION 3: GAME ENVIRONMENT — PRISONER'S DILEMMA
# =============================================================================
# This class defines the game the agent plays. It handles all game logic:
# tracking scores, determining the opponent's moves, and applying moves.
#
# PAYOFF MATRIX (standard Prisoner's Dilemma):
#
#                  Opponent Cooperates    Opponent Defects
#   You Cooperate:     You +3, Opp +3       You +0, Opp +5
#   You Defect:        You +5, Opp +0       You +1, Opp +1
#
# KEY INSIGHT: Defecting always gives YOU more points in any single round.
# But if both players always defect, you each get 1/round instead of 3/round.
# This tension between individual and collective rationality is the whole point.
#
# TO SWAP IN A NEW GAME: Replace this class with one that implements the
# same four methods: get_game_state(), get_legal_moves(), make_move(), is_over()
# =============================================================================

class PrisonersDilemma:
    """
    Prisoner's Dilemma game environment.

    Manages game state, scoring, opponent behavior, and move validation.
    The opponent strategy is defined in opponent_move() and can be easily
    swapped to test how the agent adapts to different opponents.

    Attributes:
        rounds (int): Total number of rounds to play.
        current_round (int): Which round we're currently on (starts at 1).
        agent_score (int): Cumulative points for the AI agent.
        opponent_score (int): Cumulative points for the opponent.
        history (list): List of (agent_move, opponent_move) tuples per round.
        pending_move (str|None): Unused — reserved for future async support.
    """

    # Payoff matrix: keys are (agent_move, opponent_move), values are (agent_pts, opp_pts)
    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),  # mutual cooperation — best collective outcome
        ("cooperate", "defect"):    (0, 5),  # agent gets exploited
        ("defect",    "cooperate"): (5, 0),  # agent exploits opponent
        ("defect",    "defect"):    (1, 1),  # mutual defection — worst collective outcome
    }

    OPPONENT_STRATEGIES = [
        "grim_trigger",    # cooperates until agent defects once, then defects forever
        "tit_for_tat",     # copies agent's last move; cooperates first
        "pavlov",          # win-stay lose-shift: repeat if it paid off, else switch
        "generous_tft",    # like tit_for_tat but forgives defection ~10% of the time
        "always_defect",
        "always_cooperate",
    ]

    def __init__(self, rounds=5, opponent_strategy="grim_trigger"):
        """
        Set up a new game.

        Args:
            rounds: How many rounds to play. Default is 5.
            opponent_strategy: One of OPPONENT_STRATEGIES. Default is 'grim_trigger'.
        """
        if opponent_strategy not in self.OPPONENT_STRATEGIES:
            raise ValueError(f"Unknown strategy '{opponent_strategy}'. Choose from {self.OPPONENT_STRATEGIES}")
        self.rounds = rounds
        self.opponent_strategy = opponent_strategy
        self.current_round = 1
        self.agent_score = 0
        self.opponent_score = 0
        self.history = []       # grows by one tuple per completed round
        self.pending_move = None

    def get_game_state(self):
        """
        Return a plain-English description of the current game state.

        This is what the agent reads when it calls get_game_state: now.
        Keeping this informative but concise helps the agent reason well.

        Returns:
            String describing round number, opponent's last move, and scores.
        """
        last = (
            f"Opponent's last move: {self.history[-1][1]}."
            if self.history else "No moves yet."
        )
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last} "
            f"Score: You {self.agent_score}, Opponent {self.opponent_score}."
        )

    def get_legal_moves(self):
        """
        Return the list of valid moves the agent can make.

        Returns:
            List of valid move strings. In Prisoner's Dilemma: ['cooperate', 'defect']
        """
        return ['cooperate', 'defect']

    def opponent_move(self):
        """
        Determine what the opponent plays this round using the configured strategy.

        Strategies:
          grim_trigger  — cooperates until agent defects once, then defects forever
          tit_for_tat   — cooperates round 1, then mirrors agent's last move
          pavlov        — win-stay lose-shift (repeat if points >= 3, else switch)
          generous_tft  — tit_for_tat but forgives defection ~10% of the time
          always_defect — always defects
          always_cooperate — always cooperates
        """
        import random as _random

        strategy = self.opponent_strategy
        if strategy == "always_defect":
            return "defect"

        if strategy == "always_cooperate":
            return "cooperate"

        if strategy == "tit_for_tat":
            if not self.history:
                return "cooperate"
            return self.history[-1][0]

        if strategy == "grim_trigger":
            if not self.history:
                return "cooperate"
            if any(agent_mv == "defect" for agent_mv, _ in self.history):
                return "defect"
            return "cooperate"

        if strategy == "pavlov":
            if not self.history:
                return "cooperate"
            last_agent, last_opp = self.history[-1]
            opp_pts_last = self.PAYOFFS[(last_agent, last_opp)][1]
            if opp_pts_last >= 3:
                return last_opp          # stay
            return "defect" if last_opp == "cooperate" else "cooperate"  # shift

        if strategy == "generous_tft":
            if not self.history:
                return "cooperate"
            if self.history[-1][0] == "defect" and _random.random() > 0.10:
                return "defect"
            return "cooperate"

        return "cooperate"  # fallback
    
    
    
    
    def make_move(self, agent_move):
        """
        Apply the agent's move for this round and update scores.

        Validates the move, determines the opponent's response,
        looks up the payoff, updates scores, and records history.

        Args:
            agent_move: The move string the agent chose (e.g. 'cooperate').

        Returns:
            String describing what happened this round (shown to agent as Observation).
        """
        agent_move = agent_move.strip().lower()

        # Validate the move — the agent occasionally hallucinates invalid moves
        if agent_move not in self.get_legal_moves():
            return f"Invalid move: '{agent_move}'. Legal moves are: {self.get_legal_moves()}"

        opp = self.opponent_move()
        agent_pts, opp_pts = self.PAYOFFS[(agent_move, opp)]

        # Update running scores
        self.agent_score += agent_pts
        self.opponent_score += opp_pts

        # Record this round in history (used by opponent_move() next round)
        self.history.append((agent_move, opp))
        self.current_round += 1

        return (
            f"Move accepted. You played {agent_move}, opponent played {opp}. "
            f"Points this round: You +{agent_pts}, Opponent +{opp_pts}."
        )

    def is_over(self):
        """
        Check whether the game has ended.

        Returns:
            True if all rounds have been played, False otherwise.
        """
        return self.current_round > self.rounds


# =============================================================================
# SECTION 4: GAME CONTEXT INJECTION
# =============================================================================
# This function is the "adapter" between the game environment and the agent.
# It converts the current game state into a prompt the agent can act on.
#
# This is the ONLY thing you change when switching to a new game.
# The agent never sees the Python game object directly — it only sees text.
# =============================================================================

def build_game_context(game: PrisonersDilemma) -> str:
    """
    Build the prompt that tells the agent what game it's playing and what to do.

    This is called at the START of every round with the latest game state.
    It combines static rules (never change) with dynamic state (changes each round).

    Think of this as the "briefing" the agent gets before each round.

    Args:
        game: The current game object with up-to-date state.

    Returns:
        A formatted string prompt ready to send to the agent.
    """
    return f"""
You are playing Prisoner's Dilemma. Your goal is to win the game! 

RULES:
- Each round, both players simultaneously choose to cooperate or defect.
- Both cooperate  → both get 3 points.
- You cooperate, they defect  → you get 0, they get 5.
- You defect, they cooperate  → you get 5, they get 0.
- Both defect  → both get 1 point.
- There are {game.rounds} rounds total. Maximize your total score.

CURRENT STATE:
{game.get_game_state()}

LEGAL MOVES:
{game.get_legal_moves()}

Now make your decision using the Thought/Action/PAUSE loop.
""".strip()


# =============================================================================
# SECTION 5: MAIN AGENT LOOP
# =============================================================================
# For each round:
#   1. Build the game context prompt
#   2. Feed it to the agent
#   3. Parse the agent's response for Action: tool: arg
#   4. Call the right tool and get the Observation
#   5. Feed the Observation back to the agent
#   6. Repeat until the agent makes a move (hits make_move) or runs out of steps
#
# The inner while loop handles one ROUND of thinking.
# The outer while loop handles one GAME (all rounds).
# =============================================================================

def run_game(game: PrisonersDilemma, max_iterations: int = 10, auto_judge: bool = False):
    """
    Run a full game between the ReAct agent and the game environment.

    The agent gets a fresh conversation history each round (new Agent instance),
    but the game object persists across rounds, maintaining scores and history.

    Args:
        game: An initialized PrisonersDilemma game object.
        max_iterations: Safety cap on how many LLM calls per round before
                        giving up. Prevents infinite loops if agent gets stuck.
                        Default 10 is plenty for a 2-3 tool call round.
    """

    # Define which tool names the agent is allowed to call.
    # If the agent hallucinates a tool name, it gets an error Observation.
    tools = ['get_game_state', 'get_legal_moves', 'make_move']
    logger = TraceLogger(GAME_NAME)

    strategy_label = game.opponent_strategy.replace("_", " ").title()
    print("=" * 50)
    print(f"PRISONER'S DILEMMA — Agent vs {strategy_label}")
    print("=" * 50)

    # ── OUTER LOOP: one iteration per game round ──────────────────────────────
    while not game.is_over():
        print(f"\n── Round {game.current_round} ──")
        logger.start_round(game.current_round, snapshot_game_state(game))

        # Fresh agent each round — clears conversation history.
        # This is intentional: we don't want the agent's round 1 reasoning
        # to fill up the context window by round 5.
        # (In Reflexion, we'll pass a memory summary instead.)
        system_prompt = build_system_prompt(GAME_SYSTEM_PROMPT, game_name=GAME_NAME)
        agent = Agent(client=client, system=system_prompt)

        # Start the round with the current game state as context
        next_prompt = build_game_context(game)
        i = 0

        # ── INNER LOOP: one iteration per Thought/Action/Observation step ────
        while i < max_iterations:
            i += 1

            # Send the prompt to the agent, get its Thought + Action
            prompt = next_prompt
            state_before = snapshot_game_state(game)
            result = agent(prompt)
            parsed = parse_react_response(result)
            print(result)

            # ── PARSE: did the agent request a tool call? ─────────────────────
            # We look for the pattern "Action: tool_name: argument"
            # re.IGNORECASE handles "action:" vs "Action:" variations
            if parsed["has_pause"] and ("Action" in result or parsed["action"]):
                if parsed["parse_error"] or not parsed["action"]:
                    # Agent used the wrong format — give it a helpful error
                    obs = "Could not parse your Action. Format must be: Action: tool_name: argument"
                    next_prompt = f"Observation: {obs}"
                    logger.record_step(
                        game.current_round,
                        i,
                        prompt,
                        result,
                        parsed,
                        observation=obs,
                        state_before=state_before,
                        state_after=snapshot_game_state(game),
                    )
                    continue

                chosen_tool = parsed["action"].strip()
                arg = parsed["argument"].strip()

                # ── DISPATCH: call the right tool ─────────────────────────────
                if chosen_tool == 'get_game_state':
                    obs = game.get_game_state()

                elif chosen_tool == 'get_legal_moves':
                    obs = str(game.get_legal_moves())

                elif chosen_tool == 'make_move':
                    # make_move ends the round — break out of inner loop
                    round_number = game.current_round
                    obs = game.make_move(arg)
                    next_prompt = f"Observation: {obs}"
                    print(next_prompt)
                    logger.record_step(
                        round_number,
                        i,
                        prompt,
                        result,
                        parsed,
                        observation=obs,
                        state_before=state_before,
                        state_after=snapshot_game_state(game),
                    )
                    logger.record_round_result(round_number, {
                        "observation": obs,
                        "history": game.history[-1] if game.history else None,
                        "agent_score": game.agent_score,
                        "opponent_score": game.opponent_score,
                    })
                    break  # round is done, move to next round in outer loop

                else:
                    # Agent hallucinated a tool name
                    obs = f"Unknown tool '{chosen_tool}'. Available tools: {tools}"

                next_prompt = f"Observation: {obs}"
                print(next_prompt)
                logger.record_step(
                    game.current_round,
                    i,
                    prompt,
                    result,
                    parsed,
                    observation=obs,
                    state_before=state_before,
                    state_after=snapshot_game_state(game),
                )
                continue

            # ── TERMINAL: agent output a Decision without calling make_move ───
            # This shouldn't normally happen — the agent should always call
            # make_move — but we handle it gracefully just in case.
            if parsed["decision"]:
                logger.record_step(
                    game.current_round,
                    i,
                    prompt,
                    result,
                    parsed,
                    state_before=state_before,
                    state_after=snapshot_game_state(game),
                )
                print("Agent reached Decision without calling make_move. Ending round.")
                break

            obs = "No valid Action or Decision found. Use Action: tool_name: argument."
            next_prompt = f"Observation: {obs}"
            logger.record_step(
                game.current_round,
                i,
                prompt,
                result,
                parsed,
                observation=obs,
                state_before=state_before,
                state_after=snapshot_game_state(game),
            )

    # ── GAME SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"GAME OVER — You: {game.agent_score} | Opponent: {game.opponent_score}")

    if game.agent_score > game.opponent_score:
        print("You win!")
    elif game.opponent_score > game.agent_score:
        print("Opponent wins.")
    else:
        print("Draw.")

    # Print move history so we can analyze the agent's decisions
    print("\nMove history (your move, opponent move):")
    for i, (agent_mv, opp_mv) in enumerate(game.history, 1):
        print(f"  Round {i}: You={agent_mv:<10} Opponent={opp_mv}")

    logger.finish({
        "agent_score": game.agent_score,
        "opponent_score": game.opponent_score,
        "history": game.history,
        "opponent_strategy": game.opponent_strategy,
    })
    trace_path = logger.save()
    print(f"\nTrace saved to: {trace_path}")

    if auto_judge:
        print("\n" + "─" * 50)
        print("  TRACER JUDGE — localizing reasoning failures...")
        print("─" * 50)
        judge_result = judge_trace(trace_path, client=client, append_to_reflections=True)
        write_judge_result(trace_path, judge_result)
        failures = judge_result.get("failures", [])
        if failures:
            for f in failures:
                print(f"  Round {f.get('round','?')} Step {f.get('step','?')}: [{f.get('category','?')}]")
                print(f"    Reason : {f.get('explanation','')}")
                print(f"    Fix    : {f.get('suggested_fix','')}")
        else:
            print("  No failures found.")
        reflection = judge_result.get("reflection", "")
        if reflection:
            print(f"\n  Reflection saved → agent/reflections/{GAME_NAME}.md")
            print(f"  Lesson: {reflection}")


# =============================================================================
# ENTRY POINT
# =============================================================================
# Standard Python pattern: only run the game if this file is executed directly,
# not if it's imported as a module by another file.
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma: Agent vs scripted opponent")
    parser.add_argument("--strategy", default="grim_trigger", choices=PrisonersDilemma.OPPONENT_STRATEGIES)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--judge", action="store_true", help="Auto-run Tracer judge after game ends")
    args = parser.parse_args()
    game = PrisonersDilemma(rounds=args.rounds, opponent_strategy=args.strategy)
    run_game(game, max_iterations=10, auto_judge=args.judge)
