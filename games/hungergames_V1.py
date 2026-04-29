"""
=============================================================================
Hunger Games Agent — Human vs ReAct AI
=============================================================================

WHAT IS THIS GAME?
------------------
Both players start with 10 HP and get 10 energy every round.
Each round, SECRETLY allocate your 10 energy across:

    ⚔️  ATTACK  → deals damage to opponent equal to energy spent
    🛡️  DEFEND  → blocks incoming damage equal to energy spent
    💊  HEAL    → restores your own HP equal to energy spent

Resolution (simultaneous reveal):
    Damage you take   = max(opponent_attack - your_defend, 0)
    Damage agent takes = max(your_attack - agent_defend, 0)
    Both heal their respective heal amounts

Win condition:
    - First to reach 0 HP loses immediately
    - After all rounds: highest HP wins
    - Equal HP: draw

WHAT MAKES THIS INTERESTING FOR THE DEMO:
------------------------------------------
The agent must reason about what the HUMAN is likely to do —
this is opponent modeling, a key skill in multi-agent AI.

Example agent reasoning:
    "Human has only 3 HP left. They are desperate and will
     likely go all-in on attack to try to win in one shot.
     I should heavily defend this round, then counterattack
     next round when they have spent all energy on attack..."

The human plays LIVE during the presentation — the audience
can shout suggestions, making it interactive and fun.

HOW TO RUN:
-----------
    python hunger_games_agent.py

    You will be prompted each round to enter your allocation.
    Format: attack defend heal (space separated, must sum to 10)
    Example: 5 3 2

AUTHORS: [Your team names here]
CLASS:   Statistics C163/C263 — Generative Data Science, UCLA
=============================================================================
"""

import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_gemini_client
from agent.tracing import TraceLogger, parse_react_response, snapshot_game_state


client = create_gemini_client()

# Model to use — swap to gemini-2.5-flash-lite if hitting rate limits
GAME_MODEL = "gemini-2.5-flash"
GAME_NAME = "hunger_games"


# =============================================================================
# SECTION 1: SYSTEM PROMPT
# =============================================================================
# Same ReAct structure as the Prisoner's Dilemma agent.
# The game changes, the brain doesn't.
# =============================================================================

GAME_SYSTEM_PROMPT = """
When reasoning, consider what the human is likely to do based on HP, round number, and recent history.

Your available actions are:

get_game_state:
e.g. get_game_state: now
Returns current HP for both players, round number, and history.

get_legal_moves:
e.g. get_legal_moves: now
Returns the rules for allocating energy this round.

make_move:
e.g. make_move: attack=4 defend=4 heal=2
Submits your energy allocation. Values must sum to exactly 10.
All three values (attack, defend, heal) must be specified.

Example session:

Thought: I need to check the current game state before deciding.
Action: get_game_state: now
PAUSE

Observation: Round 2/5. My HP: 8, Human HP: 6. Last round: I attacked 3, defended 5, healed 2. Human attacked 6, defended 2, healed 2.

Thought: Human attacked heavily last round. They might do it again or switch to defense. My HP lead means I can afford to play defensively and chip away. I'll prioritize defense this round.
Action: make_move: attack=3 defend=5 heal=2
PAUSE

Observation: Move submitted.

Decision: attack=3 defend=5 heal=2
""".strip()


REACT_SYSTEM_PROMPT = build_system_prompt(GAME_SYSTEM_PROMPT, game_name=GAME_NAME)



# =============================================================================
# SECTION 3: GAME ENVIRONMENT — HUNGER GAMES
# =============================================================================
# This is the ONLY thing that changes from the PD agent.
# The Agent class and ReAct loop above are identical.
#
# ENERGY ALLOCATION RULES:
#   - 10 energy per round, must allocate all of it
#   - attack + defend + heal must equal exactly 10
#   - All values must be non-negative integers
#
# DAMAGE RESOLUTION (simultaneous):
#   - damage_to_human = max(agent_attack - human_defend, 0)
#   - damage_to_agent = max(human_attack - agent_defend, 0)
#   - healing is applied AFTER damage
#   - HP cannot exceed starting HP (10) via healing
# =============================================================================

class HungerGames:
    """
    Hunger Games environment: Human vs AI agent.

    Each round both players secretly allocate 10 energy across
    attack, defend, and heal. Resolution is simultaneous.

    Attributes:
        rounds (int):         Total rounds to play.
        starting_hp (int):    HP both players start with.
        energy (int):         Energy available each round.
        agent_hp (int):       Current agent HP.
        human_hp (int):       Current human HP.
        current_round (int):  Which round we're on.
        history (list):       List of round result dicts.
    """

    def __init__(self, rounds=5, starting_hp=10, energy=10):
        self.rounds = rounds
        self.starting_hp = starting_hp
        self.energy = energy
        self.agent_hp = starting_hp
        self.human_hp = starting_hp
        self.current_round = 1
        self.history = []  # each entry: {round, agent_alloc, human_alloc, damage_to_agent, damage_to_human}

    def get_game_state(self) -> str:
        """
        Return plain-English description of current game state.
        Called by agent via get_game_state: now tool.
        """
        if not self.history:
            last = "No rounds played yet."
        else:
            last = self.history[-1]
            last = (
                f"Last round: Agent played attack={last['agent_attack']} "
                f"defend={last['agent_defend']} heal={last['agent_heal']}. "
                f"Human played attack={last['human_attack']} "
                f"defend={last['human_defend']} heal={last['human_heal']}. "
                f"Agent took {last['damage_to_agent']} damage, "
                f"Human took {last['damage_to_human']} damage."
            )

        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"Agent HP: {self.agent_hp}, Human HP: {self.human_hp}. "
            f"{last}"
        )

    def get_legal_moves(self) -> str:
        """
        Return the rules for energy allocation this round.
        Called by agent via get_legal_moves: now tool.
        """
        return (
            f"Allocate exactly {self.energy} energy across attack, defend, heal. "
            f"All values must be non-negative integers summing to {self.energy}. "
            f"Format: make_move: attack=X defend=Y heal=Z"
        )

    def parse_agent_move(self, move_str: str) -> tuple[int, int, int]:
        """
        Parse the agent's move string into (attack, defend, heal).

        Handles formats like:
            "attack=4 defend=4 heal=2"
            "attack=4, defend=4, heal=2"

        Returns:
            Tuple of (attack, defend, heal) integers.

        Raises:
            ValueError if format is wrong or values don't sum to energy.
        """
        try:
            attack = int(re.search(r'attack=(\d+)', move_str).group(1))
            defend = int(re.search(r'defend=(\d+)', move_str).group(1))
            heal   = int(re.search(r'heal=(\d+)',   move_str).group(1))
        except (AttributeError, ValueError):
            raise ValueError(
                f"Could not parse move: '{move_str}'. "
                f"Expected format: attack=X defend=Y heal=Z"
            )

        if attack + defend + heal != self.energy:
            raise ValueError(
                f"Values must sum to {self.energy}. "
                f"Got attack={attack} + defend={defend} + heal={heal} = {attack+defend+heal}"
            )
        if any(v < 0 for v in [attack, defend, heal]):
            raise ValueError("All values must be non-negative.")

        return attack, defend, heal

    def get_human_input(self) -> tuple[int, int, int]:
        """
        Prompt the human player for their energy allocation.

        Keeps asking until valid input is received.
        This is the interactive part — called during the live demo.

        Returns:
            Tuple of (attack, defend, heal) integers.
        """
        print(f"\n  Your turn! You have {self.energy} energy to allocate.")
        print(f"  Your HP: {self.human_hp}  |  Agent HP: {self.agent_hp}")
        print(f"  ⚔️  ATTACK = damage to agent (blocked by agent's defend)")
        print(f"  🛡️  DEFEND = blocks incoming agent attack")
        print(f"  💊  HEAL   = restores your own HP (max {self.starting_hp})")
        print(f"  Enter three numbers separated by spaces (attack defend heal).")
        print(f"  They must sum to {self.energy}. Example: 5 3 2")

        while True:
            try:
                raw = input("\n  Your allocation (attack defend heal): ").strip()
                parts = raw.split()
                if len(parts) != 3:
                    print(f"  ❌ Enter exactly 3 numbers. Got {len(parts)}.")
                    continue
                attack, defend, heal = int(parts[0]), int(parts[1]), int(parts[2])
                if attack + defend + heal != self.energy:
                    print(f"  ❌ Must sum to {self.energy}. Got {attack+defend+heal}.")
                    continue
                if any(v < 0 for v in [attack, defend, heal]):
                    print("  ❌ No negative numbers.")
                    continue
                return attack, defend, heal
            except ValueError:
                print("  ❌ Please enter integers only.")

    def resolve_round(
        self,
        agent_attack: int, agent_defend: int, agent_heal: int,
        human_attack: int, human_defend: int, human_heal: int
    ) -> dict:
        """
        Resolve one round: apply damage and healing to both players.

        Damage is calculated simultaneously — both players use their
        defend value against the other's attack. Healing is applied
        after damage but cannot exceed starting HP.

        Args:
            agent_attack/defend/heal: Agent's allocation this round.
            human_attack/defend/heal: Human's allocation this round.

        Returns:
            Dict with full round results for display and history.
        """
        # Calculate net damage (cannot go below 0)
        damage_to_agent = max(human_attack - agent_defend, 0)
        damage_to_human = max(agent_attack - human_defend, 0)

        # Apply damage first, then healing (cap at starting HP)
        self.agent_hp = min(
            self.starting_hp,
            max(0, self.agent_hp - damage_to_agent + agent_heal)
        )
        self.human_hp = min(
            self.starting_hp,
            max(0, self.human_hp - damage_to_human + human_heal)
        )

        result = {
            'round':           self.current_round,
            'agent_attack':    agent_attack,
            'agent_defend':    agent_defend,
            'agent_heal':      agent_heal,
            'human_attack':    human_attack,
            'human_defend':    human_defend,
            'human_heal':      human_heal,
            'damage_to_agent': damage_to_agent,
            'damage_to_human': damage_to_human,
            'agent_hp_after':  self.agent_hp,
            'human_hp_after':  self.human_hp,
        }
        self.history.append(result)
        self.current_round += 1
        return result

    def is_over(self) -> bool:
        """Game ends if either player hits 0 HP or all rounds are played."""
        return self.agent_hp <= 0 or self.human_hp <= 0 or self.current_round > self.rounds


# =============================================================================
# SECTION 4: GAME CONTEXT INJECTION
# =============================================================================
# Feeds the current game state + rules into the agent's prompt.
# Same pattern as PD — only the game description changes.
# =============================================================================

def build_game_context(game: HungerGames) -> str:
    """
    Build the prompt the agent receives at the start of each round.

    Includes game rules, current state, and strategic framing.
    The human's move is NOT included here — the agent decides first,
    then the human inputs their move, then round resolves.

    Args:
        game: Current HungerGames object with up-to-date state.

    Returns:
        Formatted prompt string ready to send to agent.
    """
    # Build history summary for opponent modeling
    if game.history:
        history_lines = []
        for h in game.history[-3:]:  # show last 3 rounds max to keep context short
            history_lines.append(
                f"  Round {h['round']}: "
                f"Human → attack={h['human_attack']} defend={h['human_defend']} heal={h['human_heal']} "
                f"| Agent → attack={h['agent_attack']} defend={h['agent_defend']} heal={h['agent_heal']} "
                f"| Damage: Human took {h['damage_to_human']}, Agent took {h['damage_to_agent']}"
            )
        history_text = "RECENT HISTORY:\n" + "\n".join(history_lines)
    else:
        history_text = "RECENT HISTORY: No rounds played yet."

    return f"""
You are playing Hunger Games against a human opponent. Win at all costs. but also dont lose the game 
in the spirit of winning - stay rational. The goal is to win but also not to lose. 

RULES:
- Each round, both players secretly allocate exactly {game.energy} energy across attack, defend, heal.
- Damage to opponent = max(your_attack - opponent_defend, 0)
- Damage to you      = max(opponent_attack - your_defend, 0)
- Healing restores your own HP (cannot exceed {game.starting_hp})
- First to reach 0 HP loses. After {game.rounds} rounds, highest HP wins.

STRATEGY TIPS:
- If opponent has low HP, they may go all-in on attack — defend heavily
- If you have low HP, consider healing + defend to survive
- Watch the human's patterns across rounds — humans are often predictable
- A balance of attack and defend is safer than going all-in on one

{history_text}

CURRENT STATE:
{game.get_game_state()}

LEGAL MOVES:
{game.get_legal_moves()}

Reason carefully about what the human is likely to do this round, then make your move.
""".strip()


# =============================================================================
# SECTION 5: DISPLAY HELPERS
# =============================================================================

def print_round_result(result: dict, agent_hp: int, human_hp: int):
    """Print a clear, formatted round result for the live demo."""
    print("\n" + "━" * 50)
    print(f"  ROUND {result['round']} RESULT")
    print("━" * 50)
    print(f"  {'':10}  {'⚔️ ATTACK':>10}  {'🛡️ DEFEND':>10}  {'💊 HEAL':>10}")
    print(f"  {'Human':10}  {result['human_attack']:>10}  {result['human_defend']:>10}  {result['human_heal']:>10}")
    print(f"  {'Agent':10}  {result['agent_attack']:>10}  {result['agent_defend']:>10}  {result['agent_heal']:>10}")
    print()
    print(f"  💥 Human took {result['damage_to_human']} damage | Agent took {result['damage_to_agent']} damage")
    print()
    print(f"  ❤️  Human HP: {result['human_hp_after']:>3}/10  {'█' * result['human_hp_after']}{'░' * (10 - result['human_hp_after'])}")
    print(f"  🤖 Agent HP: {result['agent_hp_after']:>3}/10  {'█' * result['agent_hp_after']}{'░' * (10 - result['agent_hp_after'])}")
    print("━" * 50)


def print_agent_reasoning(reasoning: str):
    """Print the agent's reasoning in a clearly marked box."""
    print("\n   AGENT'S REASONING:")
    print("  " + "─" * 46)
    # indent each line for readability
    for line in reasoning.strip().split('\n'):
        print(f"  {line}")
    print("  " + "─" * 46)


# =============================================================================
# SECTION 6: MAIN GAME LOOP
# =============================================================================

def run_hunger_games(game: HungerGames, max_iterations: int = 10):
    """
    Run a full Hunger Games match between the ReAct agent and a human.

    Flow each round:
        1. Agent reasons and decides its allocation (hidden from human)
        2. Human inputs their allocation
        3. Round resolves simultaneously
        4. Results displayed with agent's reasoning revealed

    The agent's reasoning is shown AFTER the human commits —
    this creates a dramatic reveal moment during the demo.

    Args:
        game:           Initialized HungerGames object.
        max_iterations: Safety cap on agent reasoning steps per round.
    """
    tools = ['get_game_state', 'get_legal_moves', 'make_move']
    logger = TraceLogger(GAME_NAME)

    print("\n" + "=" * 50)
    print("  ⚔️  HUNGER GAMES — Human vs AI Agent  ⚔️")
    print("=" * 50)
    print(f"  Rounds: {game.rounds}  |  Starting HP: {game.starting_hp}  |  Energy/round: {game.energy}")
    print("  Highest HP after all rounds wins.")
    print("  First to reach 0 HP loses immediately.")
    print("=" * 50)

    while not game.is_over():
        print(f"\n\n{'='*50}")
        print(f"  ROUND {game.current_round} of {game.rounds}")
        print(f"{'='*50}")
        logger.start_round(game.current_round, snapshot_game_state(game))

        # ── Step 1: Agent decides (hidden from human) ─────────────────────
        print("\n  🤖 Agent is thinking...")
        system_prompt = build_system_prompt(GAME_SYSTEM_PROMPT, game_name=GAME_NAME)
        agent = Agent(client=client, system=system_prompt, model=GAME_MODEL)
        next_prompt = build_game_context(game)

        agent_attack = agent_defend = agent_heal = None
        agent_reasoning = ""
        i = 0

        while i < max_iterations:
            i += 1
            prompt = next_prompt
            state_before = snapshot_game_state(game)
            result = agent(prompt)
            parsed = parse_react_response(result)

            # Capture reasoning for the reveal later
            if parsed["thought"]:
                agent_reasoning += parsed["thought"] + " "

            if parsed["has_pause"] and ("Action" in result or parsed["action"]):
                if parsed["parse_error"] or not parsed["action"]:
                    obs = "Could not parse action. Format: Action: tool_name: argument"
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

                if chosen_tool == 'get_game_state':
                    obs = game.get_game_state()

                elif chosen_tool == 'get_legal_moves':
                    obs = game.get_legal_moves()

                elif chosen_tool == 'make_move':
                    try:
                        agent_attack, agent_defend, agent_heal = game.parse_agent_move(arg)
                        obs = f"Move submitted: attack={agent_attack} defend={agent_defend} heal={agent_heal}"
                        next_prompt = f"Observation: {obs}"
                        print(f"  ✅ Agent has locked in its move (hidden until reveal)")
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
                        break
                    except ValueError as e:
                        obs = f"Invalid move: {e}. Try again."

                else:
                    obs = f"Unknown tool '{chosen_tool}'. Available: {tools}"

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

        # Fallback if agent failed to make a valid move
        if agent_attack is None:
            print("  ⚠️  Agent failed to decide — defaulting to balanced allocation")
            agent_attack, agent_defend, agent_heal = 4, 4, 2
            logger.record_step(
                game.current_round,
                i + 1,
                "fallback",
                "Agent failed to make a valid move.",
                {
                    "thought": "",
                    "action": "fallback_move",
                    "argument": "attack=4 defend=4 heal=2",
                    "has_pause": False,
                    "decision": None,
                    "parse_error": "Agent did not submit a valid make_move action.",
                },
                observation="Defaulted to attack=4 defend=4 heal=2",
                state_before=snapshot_game_state(game),
                state_after=snapshot_game_state(game),
            )

        # ── Step 2: Human inputs their move ───────────────────────────────
        human_attack, human_defend, human_heal = game.get_human_input()

        # ── Step 3: Resolve the round ──────────────────────────────────────
        round_result = game.resolve_round(
            agent_attack, agent_defend, agent_heal,
            human_attack, human_defend, human_heal
        )
        logger.record_round_result(round_result["round"], round_result)

        # ── Step 4: Reveal results + agent reasoning ───────────────────────
        print_round_result(round_result, game.agent_hp, game.human_hp)
        print_agent_reasoning(agent_reasoning)

        # Check for early game over
        if game.agent_hp <= 0 or game.human_hp <= 0:
            break

        time.sleep(1)  # brief pause between rounds for readability

    # ── Final result ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ⚔️  GAME OVER")
    print("=" * 50)
    print(f"  Final HP — Human: {game.human_hp}  |  Agent: {game.agent_hp}")
    print()

    if game.human_hp <= 0 and game.agent_hp <= 0:
        print("  🤝 Draw — both reached 0 HP!")
    elif game.human_hp <= 0:
        print("  🤖 Agent wins! Human reached 0 HP.")
    elif game.agent_hp <= 0:
        print("  🎉 Human wins! Agent reached 0 HP.")
    elif game.human_hp > game.agent_hp:
        print("  🎉 Human wins with more HP!")
    elif game.agent_hp > game.human_hp:
        print("  🤖 Agent wins with more HP!")
    else:
        print("  🤝 Draw — equal HP!")

    print()
    print("  Round-by-round summary:")
    print(f"  {'Round':>6}  {'Human ATK':>10} {'Human DEF':>10} {'Human HLR':>10}  "
          f"{'Agent ATK':>10} {'Agent DEF':>10} {'Agent HLR':>10}  "
          f"{'Human HP':>9} {'Agent HP':>9}")
    for h in game.history:
        print(f"  {h['round']:>6}  {h['human_attack']:>10} {h['human_defend']:>10} {h['human_heal']:>10}  "
              f"{h['agent_attack']:>10} {h['agent_defend']:>10} {h['agent_heal']:>10}  "
              f"{h['human_hp_after']:>9} {h['agent_hp_after']:>9}")

    logger.finish({
        "human_hp": game.human_hp,
        "agent_hp": game.agent_hp,
        "history": game.history,
    })
    trace_path = logger.save()
    print(f"\n  Trace saved to: {trace_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    game = HungerGames(rounds=5, starting_hp=10, energy=10)
    run_hunger_games(game, max_iterations=10)
