"""
=============================================================================
Hunger Games Agent V2 — with Error Localization Pipeline
=============================================================================

Identical gameplay to V1. The difference is that every ReAct step is now
traced, checked, and attributed so that after each round (and at game end)
you get a ranked list of which steps were most to blame for bad outcomes.

Pipeline stages wired into run_hunger_games:
  1. TRACE   — tracer.span() wraps each llm_call / tool_dispatch / tool exec
  2. CHECK   — per-step critics run inside each span (format, tool validity,
               move constraints, strategy, thought-action consistency)
  3. ATTRIBUTE — after round resolves, RoundOutcome back-propagates blame
  4. CLUSTER  — across all rounds, cluster_failures groups failure patterns
  5. REPORT   — format_round_report prints after each round;
               format_cluster_report prints at game end
=============================================================================
"""

import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

from error_localization import (
    Tracer,
    CheckResult,
    RoundOutcome,
    attribute_round,
    check_react_format,
    check_action_parses,
    check_tool_known,
    check_move_constraints,
    check_strategy,
    check_thought_action_consistency,
    format_round_report,
    format_cluster_report,
    export_trace_jsonl,
)

load_dotenv()
client = genai.Client(api_key=os.getenv("gemeni_api_key"))

GAME_MODEL = "gemini-2.5-flash"


# =============================================================================
# SECTION 1: SYSTEM PROMPT  (unchanged from V1)
# =============================================================================

REACT_SYSTEM_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output a Decision.

IMPORTANT: You MUST always start your response with 'Thought:' before any Action.

Use Thought to reason about the current game state and what the human is likely to do.
Use Action to call one of your available tools, then return PAUSE.
Observation will be the result of that tool.

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


# =============================================================================
# SECTION 2: AGENT CLASS  (unchanged from V1)
# =============================================================================

class Agent:
    def __init__(self, client, system, model=GAME_MODEL):
        self.client = client
        self.system = system
        self.model = model
        self.messages = []

    def __call__(self, message):
        if message:
            self.messages.append({"role": "user", "parts": [{"text": message}]})
            result = self.execute()
            self.messages.append({"role": "model", "parts": [{"text": result}]})
            return result

    def execute(self):
        config = types.GenerateContentConfig(system_instruction=self.system)
        for attempt in range(3):
            try:
                completion = self.client.models.generate_content(
                    model=self.model,
                    contents=self.messages,
                    config=config
                )
                return completion.text
            except Exception as e:
                if attempt < 2:
                    print(f"  [API busy, retrying in 5s... ({attempt+1}/3)]")
                    time.sleep(5)
                else:
                    raise e


# =============================================================================
# SECTION 3: GAME ENVIRONMENT  (unchanged from V1)
# =============================================================================

class HungerGames:
    def __init__(self, rounds=5, starting_hp=10, energy=10):
        self.rounds = rounds
        self.starting_hp = starting_hp
        self.energy = energy
        self.agent_hp = starting_hp
        self.human_hp = starting_hp
        self.current_round = 1
        self.history = []

    def get_game_state(self) -> str:
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
        return (
            f"Allocate exactly {self.energy} energy across attack, defend, heal. "
            f"All values must be non-negative integers summing to {self.energy}. "
            f"Format: make_move: attack=X defend=Y heal=Z"
        )

    def parse_agent_move(self, move_str: str) -> tuple[int, int, int]:
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
        damage_to_agent = max(human_attack - agent_defend, 0)
        damage_to_human = max(agent_attack - human_defend, 0)
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
        return self.agent_hp <= 0 or self.human_hp <= 0 or self.current_round > self.rounds


# =============================================================================
# SECTION 4: GAME CONTEXT INJECTION  (unchanged from V1)
# =============================================================================

def build_game_context(game: HungerGames) -> str:
    if game.history:
        history_lines = []
        for h in game.history[-3:]:
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
# SECTION 5: DISPLAY HELPERS  (unchanged from V1)
# =============================================================================

def print_round_result(result: dict, agent_hp: int, human_hp: int):
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
    print("\n   AGENT'S REASONING:")
    print("  " + "─" * 46)
    for line in reasoning.strip().split('\n'):
        print(f"  {line}")
    print("  " + "─" * 46)


# =============================================================================
# SECTION 6: MAIN GAME LOOP — instrumented with error localization pipeline
# =============================================================================

def run_hunger_games(game: HungerGames, max_iterations: int = 10,
                     export_jsonl: str = ""):
    """
    Run a full Hunger Games match with error localization instrumented
    throughout the ReAct loop.

    After each round:   format_round_report shows per-step blame scores.
    After the game:     format_cluster_report summarizes failure patterns.

    Args:
        game:           Initialized HungerGames object.
        max_iterations: Safety cap on agent reasoning steps per round.
        export_jsonl:   If non-empty, dump all spans to this path at game end.
    """
    tools = ['get_game_state', 'get_legal_moves', 'make_move']

    # ── Error localization: one tracer for the whole match ─────────────────
    tracer = Tracer()

    print("\n" + "=" * 50)
    print("  ⚔️  HUNGER GAMES — Human vs AI Agent  ⚔️")
    print("=" * 50)
    print(f"  Rounds: {game.rounds}  |  Starting HP: {game.starting_hp}  |  Energy/round: {game.energy}")
    print("  Highest HP after all rounds wins.")
    print("  First to reach 0 HP loses immediately.")
    print("=" * 50)

    while not game.is_over():
        round_num = game.current_round          # capture before resolve increments it
        tracer.set_round(round_num)             # tag all spans from here with this round

        print(f"\n\n{'='*50}")
        print(f"  ROUND {round_num} of {game.rounds}")
        print(f"{'='*50}")

        # ── Step 1: Agent decides (hidden from human) ─────────────────────
        print("\n  🤖 Agent is thinking...")
        agent = Agent(client=client, system=REACT_SYSTEM_PROMPT, model=GAME_MODEL)
        next_prompt = build_game_context(game)

        agent_attack = agent_defend = agent_heal = None
        agent_reasoning = ""
        used_fallback = False
        hard_error = False
        i = 0

        while i < max_iterations:
            i += 1
            tracer.tick_iter()

            # ── TRACE: LLM call ────────────────────────────────────────────
            with tracer.span("llm_call", next_prompt) as llm_span:
                result = agent(next_prompt)
                llm_span.output = result
                llm_span.checks.append(check_react_format(result))
                llm_span.checks.append(check_action_parses(result))

            # Accumulate reasoning text for the post-round reveal
            if "Thought:" in result:
                thought_match = re.search(r'Thought:(.*?)(?=Action:|$)', result, re.DOTALL)
                if thought_match:
                    agent_reasoning += thought_match.group(1).strip() + " "

            if "PAUSE" in result and "Action" in result:
                action_match = re.findall(r'Action: ([a-z_]+): (.+)', result, re.IGNORECASE)
                if not action_match:
                    next_prompt = "Observation: Could not parse action. Format: Action: tool_name: argument"
                    continue

                chosen_tool = action_match[0][0].strip()
                arg = action_match[0][1].strip()

                # ── TRACE: tool dispatch ───────────────────────────────────
                with tracer.span("tool_dispatch", chosen_tool) as td_span:
                    td_span.checks.append(check_tool_known(chosen_tool, tools))
                    td_span.output = chosen_tool

                if chosen_tool == 'get_game_state':
                    with tracer.span("tool:get_game_state", arg) as ts:
                        obs = game.get_game_state()
                        ts.output = obs

                elif chosen_tool == 'get_legal_moves':
                    with tracer.span("tool:get_legal_moves", arg) as ts:
                        obs = game.get_legal_moves()
                        ts.output = obs

                elif chosen_tool == 'make_move':
                    # ── TRACE: make_move — most checks live here ───────────
                    with tracer.span("tool:make_move", arg) as ts:
                        try:
                            agent_attack, agent_defend, agent_heal = game.parse_agent_move(arg)
                            ts.output = f"attack={agent_attack} defend={agent_defend} heal={agent_heal}"

                            ts.checks.append(check_move_constraints(
                                agent_attack, agent_defend, agent_heal, game.energy))
                            ts.checks.append(check_strategy(
                                agent_attack, agent_defend, agent_heal,
                                game.agent_hp, game.human_hp, game.history))
                            # pass full accumulated reasoning so consistency check
                            # can compare the Thought text against the final move
                            ts.checks.append(check_thought_action_consistency(
                                agent_reasoning,
                                agent_attack, agent_defend, agent_heal))

                            obs = f"Move submitted: attack={agent_attack} defend={agent_defend} heal={agent_heal}"
                            next_prompt = f"Observation: {obs}"
                            print(f"  ✅ Agent has locked in its move (hidden until reveal)")
                        except ValueError as e:
                            obs = f"Invalid move: {e}. Try again."
                            ts.output = f"INVALID: {e}"
                            ts.checks.append(CheckResult(
                                "parse", False, 0.8, str(e)))
                            next_prompt = f"Observation: {obs}"
                            continue        # keep loop going, don't break

                    if agent_attack is not None:
                        break               # successful make_move — exit ReAct loop

                else:
                    obs = f"Unknown tool '{chosen_tool}'. Available: {tools}"
                    next_prompt = f"Observation: {obs}"

                continue

            if "Decision" in result:
                break

        # ── Fallback if agent never produced a valid move ──────────────────
        if agent_attack is None:
            used_fallback = True
            print("  ⚠️  Agent failed to decide — defaulting to balanced allocation")
            with tracer.span("fallback", "default_4_4_2") as fs:
                agent_attack, agent_defend, agent_heal = 4, 4, 2
                fs.output = "attack=4 defend=4 heal=2"
                fs.checks.append(CheckResult(
                    "fallback_used", False, 0.85,
                    "agent exhausted iterations without calling make_move"))

        # ── Step 2: Human inputs their move ───────────────────────────────
        human_attack, human_defend, human_heal = game.get_human_input()

        # ── Step 3: Resolve the round ──────────────────────────────────────
        round_result = game.resolve_round(
            agent_attack, agent_defend, agent_heal,
            human_attack, human_defend, human_heal
        )

        # ── Step 4: Attribute blame across spans for this round ────────────
        outcome = RoundOutcome(
            round_num=round_num,
            damage_to_agent=round_result['damage_to_agent'],
            damage_to_human=round_result['damage_to_human'],
            agent_hp_after=round_result['agent_hp_after'],
            used_fallback=used_fallback,
            hit_max_iterations=(i >= max_iterations and not used_fallback),
            hard_error=hard_error,
        )
        attribute_round(tracer.round_spans(round_num), outcome)

        # ── Step 5: Display results + reasoning + error localization report ─
        print_round_result(round_result, game.agent_hp, game.human_hp)
        print_agent_reasoning(agent_reasoning)
        print(format_round_report(tracer, round_num))

        if game.agent_hp <= 0 or game.human_hp <= 0:
            break

        time.sleep(1)

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

    # ── Error localization: end-of-game cluster summary ────────────────────
    print(format_cluster_report(tracer))

    if export_jsonl:
        export_trace_jsonl(tracer, export_jsonl)
        print(f"\n  📄 Full trace exported to {export_jsonl}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    game = HungerGames(rounds=5, starting_hp=10, energy=10)
    run_hunger_games(game, max_iterations=10, export_jsonl="trace.jsonl")
