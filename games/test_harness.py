"""
Test harness for A/B testing Prisoner's Dilemma agents.

Supports any combination of:
  - ReAct agent (existing)
  - SingleCall agent (new)
  - Hardcoded strategies (tit_for_tat, always_cooperate, always_defect, random, pavlov, grim_trigger)

Usage:
  python games/test_harness.py --player-a single_call --player-b react --rounds 5
  python games/test_harness.py --player-a single_call --player-b tit_for_tat --rounds 10
  python games/test_harness.py --player-a react --player-b always_defect --rounds 5

Options:
  --model-a / --model-b    LLM model override per player
  --rounds                 Rounds per match (default 5)
"""

import sys
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent import Agent, LLMClient, build_system_prompt, create_client
from agent.single_call_agent import SingleCallAgent
from agent.tracing import TraceLogger, parse_react_response, snapshot_game_state
from games.prisoners_dilemma import SymmetricPD

GAME_NAME = "pd_test_harness"

REACT_SYSTEM_PROMPT = """
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


# =============================================================================
# GAME CONTEXT BUILDER (shared by all player types)
# =============================================================================

def build_game_context(game: SymmetricPD, side: str) -> str:
    """Build full context string for any agent — includes state, action history, and message history."""
    state = game.get_state_for(side)

    if game.history:
        action_lines = []
        for i, (a, b) in enumerate(game.history, 1):
            my_move = a if side == "A" else b
            opp_move = b if side == "A" else a
            pts_mine, pts_opp = SymmetricPD.PAYOFFS[(my_move, opp_move)]
            action_lines.append(
                f"  Round {i}: You={my_move}, Opponent={opp_move} "
                f"(You {'+' if pts_mine >= 0 else ''}{pts_mine}, Opp {'+' if pts_opp >= 0 else ''}{pts_opp})"
            )
        action_block = "\nACTION HISTORY:\n" + "\n".join(action_lines)
    else:
        action_block = "\nNo actions yet."

    if game.message_history:
        msg_lines = []
        for i, (ma, mb) in enumerate(game.message_history, 1):
            my_msg = ma if side == "A" else mb
            opp_msg = mb if side == "A" else ma
            msg_lines.append(f"  Round {i}: You said: \"{my_msg}\" | Opponent said: \"{opp_msg}\"")
        msg_block = "\nMESSAGE HISTORY:\n" + "\n".join(msg_lines)
    else:
        msg_block = "\nNo messages exchanged yet."

    return (
        f"You are playing Prisoner's Dilemma against another agent. "
        f"Maximize your total score over {game.rounds} rounds.\n\n"
        f"RULES:\n"
        f"- Both cooperate  → both get +2 points.\n"
        f"- You cooperate, they defect  → you -1, they +5.\n"
        f"- You defect, they cooperate  → you +5, they -1.\n"
        f"- Both defect  → both get +0 points.\n\n"
        f"CURRENT STATE:\n{state}\n"
        f"{action_block}\n"
        f"{msg_block}"
    )


# =============================================================================
# PLAYER ADAPTERS
# =============================================================================

class PlayerAdapter:
    """Common interface so the harness can treat all player types the same."""
    name: str

    def get_message(self, game: SymmetricPD, side: str) -> str:
        raise NotImplementedError

    def get_action(self, game: SymmetricPD, side: str, opponent_message: str) -> str:
        raise NotImplementedError


class SingleCallPlayer(PlayerAdapter):
    """Wraps SingleCallAgent. Stores raw LLM responses for logging."""

    def __init__(self, client: LLMClient, model: str | None = None, name: str = "SingleCall"):
        self.agent = SingleCallAgent(client=client, model=model, name=name)
        self.name = name
        self._last_msg_raw: str = ""
        self._last_msg_prompt: str = ""
        self._last_action_raw: str = ""
        self._last_action_prompt: str = ""

    def get_message(self, game: SymmetricPD, side: str) -> str:
        ctx = build_game_context(game, side)
        msg, raw, prompt = self.agent.generate_message(ctx)
        self._last_msg_raw = raw
        self._last_msg_prompt = prompt
        return msg

    def get_action(self, game: SymmetricPD, side: str, opponent_message: str) -> str:
        ctx = build_game_context(game, side)
        action, raw, prompt = self.agent.choose_action(ctx, opponent_message)
        self._last_action_raw = raw
        self._last_action_prompt = prompt
        return action


class ReactPlayer(PlayerAdapter):
    """Wraps the existing ReAct Agent for message + action phases."""

    def __init__(self, client: LLMClient, model: str | None = None, name: str = "ReAct"):
        self.client = client
        self.model = model
        self.name = name
        self.system = build_system_prompt(REACT_SYSTEM_PROMPT, game_name=GAME_NAME)
        self._round_agent: Agent | None = None

    def _new_agent(self) -> Agent:
        agent = Agent(client=self.client, system=self.system, model=self.model, name=self.name)
        self._round_agent = agent
        return agent

    def get_message(self, game: SymmetricPD, side: str) -> str:
        agent = self._new_agent()
        ctx = build_game_context(game, side)
        return agent.generate_message(ctx)

    def get_action(self, game: SymmetricPD, side: str, opponent_message: str, logger: TraceLogger | None = None, round_num: int = 0, max_iter: int = 10) -> str:
        agent = self._round_agent
        if agent is None:
            agent = self._new_agent()

        opp_block = ""
        if opponent_message:
            opp_block = (
                f"\nOPPONENT'S MESSAGE THIS ROUND:\n\"{opponent_message}\"\n"
                "Remember: messages can be honest or deceptive. Weigh this against their actions.\n"
            )

        next_prompt = (
            f"Now it's the ACTION PHASE. Choose cooperate or defect.\n"
            f"{opp_block}\n"
            f"CURRENT STATE:\n{game.get_state_for(side)}\n\n"
            f"LEGAL MOVES:\n{game.get_legal_moves()}\n\n"
            "Consider what you just said to your opponent and what they said to you.\n"
            "Your opponent cannot see your action before they commit.\n"
            "Use the Thought/Action/PAUSE loop to decide."
        )

        tools = ["get_game_state", "get_legal_moves", "make_move"]
        captured_move = None

        for step in range(1, max_iter + 1):
            state_before = snapshot_game_state(game)
            result = agent(next_prompt)
            if not result:
                break

            parsed = parse_react_response(result)

            if parsed.get("thought"):
                print(f"     [{self.name}] Step {step} Thought: {parsed['thought']}")

            if parsed.get("has_pause") and parsed.get("action"):
                chosen_tool = parsed["action"].strip()
                arg = parsed.get("argument", "").strip()
                print(f"     [{self.name}] Step {step} Action: {chosen_tool}: {arg}")

                if chosen_tool == "get_game_state":
                    obs = game.get_state_for(side)
                elif chosen_tool == "get_legal_moves":
                    obs = str(game.get_legal_moves())
                elif chosen_tool == "make_move":
                    move = arg.strip().lower()
                    if move in ("cooperate", "defect"):
                        captured_move = move
                        obs = f"Move '{move}' recorded. Waiting for opponent's decision..."
                        print(f"     [{self.name}] Step {step} Observation: {obs}")
                        if logger:
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

                print(f"     [{self.name}] Step {step} Observation: {obs}")
                if logger:
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       observation=obs, state_before=state_before,
                                       state_after=snapshot_game_state(game))
                next_prompt = f"Observation: {obs}"

            elif parsed.get("decision"):
                decision = parsed["decision"].strip().lower()
                print(f"     [{self.name}] Step {step} Decision: {decision}")
                if logger:
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       state_before=state_before,
                                       state_after=snapshot_game_state(game))
                if decision in ("cooperate", "defect"):
                    captured_move = decision
                break

            else:
                obs = "No valid Action or Decision found. Use Action: tool_name: argument."
                if logger:
                    logger.record_step(round_num, step, next_prompt, result, parsed,
                                       observation=obs, state_before=state_before,
                                       state_after=snapshot_game_state(game))
                next_prompt = f"Observation: {obs}"

        return captured_move or "cooperate"


class ScriptedPlayer(PlayerAdapter):
    """Hardcoded strategies with simple messages."""

    STRATEGIES = ["tit_for_tat", "always_cooperate", "always_defect", "random", "pavlov", "grim_trigger"]

    def __init__(self, strategy: str, name: str | None = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")
        self.strategy = strategy
        self.name = name or strategy.replace("_", " ").title()

    def get_message(self, game: SymmetricPD, side: str) -> str:
        messages = {
            "tit_for_tat": "I mirror your last move. Cooperate and I cooperate. Defect and I defect.",
            "always_cooperate": "I will cooperate every round no matter what.",
            "always_defect": "I play to win. Good luck.",
            "random": "Let's see how this round goes.",
            "pavlov": "If we both did well last round, let's keep it up.",
            "grim_trigger": "I cooperate as long as you do. One defection and I never cooperate again.",
        }
        return messages.get(self.strategy, "")

    def get_action(self, game: SymmetricPD, side: str, opponent_message: str = "") -> str:
        my_history = [(a, b) if side == "A" else (b, a) for a, b in game.history]

        if self.strategy == "always_cooperate":
            return "cooperate"
        if self.strategy == "always_defect":
            return "defect"
        if self.strategy == "random":
            return random.choice(["cooperate", "defect"])
        if self.strategy == "tit_for_tat":
            return my_history[-1][1] if my_history else "cooperate"
        if self.strategy == "grim_trigger":
            if any(opp == "defect" for _, opp in my_history):
                return "defect"
            return "cooperate"
        if self.strategy == "pavlov":
            if not my_history:
                return "cooperate"
            my_last, opp_last = my_history[-1]
            my_pts, _ = SymmetricPD.PAYOFFS[(my_last, opp_last)]
            return my_last if my_pts >= 2 else ("defect" if my_last == "cooperate" else "cooperate")
        return "cooperate"


# =============================================================================
# HELPERS
# =============================================================================

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    import re
    return re.sub(r"```(?:json)?\s*", "", text).replace("```", "")


def _extract_reasoning(raw: str) -> str:
    """Extract reasoning text from a SingleCall agent response, stripping the JSON part."""
    cleaned = _strip_code_fences(raw)
    brace_depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                reasoning = cleaned[:start].strip()
                return reasoning if reasoning else ""
    return cleaned.strip()


def _extract_metadata(raw: str) -> dict:
    """Extract rule_applied and classification_signal from the JSON in an agent response."""
    import json
    cleaned = _strip_code_fences(raw)
    brace_depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return {}
    return {}


# =============================================================================
# MATCH RUNNER
# =============================================================================

def run_match(player_a: PlayerAdapter, player_b: PlayerAdapter, rounds: int = 5):
    """Run a single match between two players."""
    game = SymmetricPD(rounds=rounds)
    logger = TraceLogger(GAME_NAME)

    print("=" * 65)
    print(f"  MATCH: {player_a.name}  vs  {player_b.name}")
    print(f"  Rounds: {rounds}")
    print("=" * 65)

    while not game.is_over():
        round_num = game.current_round
        print(f"\n{'─'*65}")
        print(f"  ROUND {round_num}  |  {player_a.name}: {game.score_a}  vs  {player_b.name}: {game.score_b}")
        print(f"{'─'*65}")

        logger.start_round(round_num, snapshot_game_state(game))

        # ── MESSAGE PHASE ──
        print(f"\n  [Message Phase]")
        msg_a = player_a.get_message(game, "A")
        if isinstance(player_a, SingleCallPlayer):
            print(f"     [{player_a.name}] Raw LLM response:")
            print(f"     {player_a._last_msg_raw}")
        print(f"     {player_a.name}: \"{msg_a}\"")

        msg_b = player_b.get_message(game, "B")
        if isinstance(player_b, SingleCallPlayer):
            print(f"     [{player_b.name}] Raw LLM response:")
            print(f"     {player_b._last_msg_raw}")
        print(f"     {player_b.name}: \"{msg_b}\"")

        game.message_history.append((msg_a, msg_b))

        state_snap = snapshot_game_state(game)

        # Log message phase for each player
        for player, side, msg in [(player_a, "A", msg_a), (player_b, "B", msg_b)]:
            if isinstance(player, SingleCallPlayer):
                msg_reasoning = _extract_reasoning(player._last_msg_raw)
                msg_metadata = _extract_metadata(player._last_msg_raw)
                logger.record_step(round_num, 0, player._last_msg_prompt, player._last_msg_raw, {
                    "thought": msg_reasoning, "action": "send_message", "argument": msg,
                    "has_pause": False, "decision": None, "parse_error": None,
                    "agent": player.name, "phase": "message",
                    "rule_applied": msg_metadata.get("rule_applied", ""),
                    "classification_signal": msg_metadata.get("classification_signal", ""),
                }, observation=f"Message sent: \"{msg}\"",
                   state_before=state_snap, state_after=state_snap)
            else:
                logger.record_step(round_num, 0, "message_phase", "", {
                    "thought": "", "action": "send_message", "argument": msg,
                    "has_pause": False, "decision": None, "parse_error": None,
                    "agent": player.name, "phase": "message",
                }, observation=f"Message sent: \"{msg}\"",
                   state_before=state_snap, state_after=state_snap)

        # ── ACTION PHASE ──
        print(f"\n  [{player_a.name} action...]")
        if isinstance(player_a, ReactPlayer):
            move_a = player_a.get_action(game, "A", msg_b, logger=logger, round_num=round_num)
        else:
            move_a = player_a.get_action(game, "A", msg_b)
            if isinstance(player_a, SingleCallPlayer):
                print(f"     [{player_a.name}] Raw LLM response:")
                print(f"     {player_a._last_action_raw}")
                action_reasoning = _extract_reasoning(player_a._last_action_raw)
                action_metadata = _extract_metadata(player_a._last_action_raw)
                logger.record_step(round_num, 1, player_a._last_action_prompt, player_a._last_action_raw, {
                    "thought": action_reasoning, "action": "choose_action", "argument": move_a,
                    "has_pause": False, "decision": move_a, "parse_error": None,
                    "agent": player_a.name, "phase": "action",
                    "rule_applied": action_metadata.get("rule_applied", ""),
                    "classification_signal": action_metadata.get("classification_signal", ""),
                }, observation=f"Action chosen: {move_a}",
                   state_before=state_snap, state_after=snapshot_game_state(game))

        print(f"\n  [{player_b.name} action...]")
        if isinstance(player_b, ReactPlayer):
            move_b = player_b.get_action(game, "B", msg_a, logger=logger, round_num=round_num)
        else:
            move_b = player_b.get_action(game, "B", msg_a)
            if isinstance(player_b, SingleCallPlayer):
                print(f"     [{player_b.name}] Raw LLM response:")
                print(f"     {player_b._last_action_raw}")
                action_reasoning = _extract_reasoning(player_b._last_action_raw)
                action_metadata = _extract_metadata(player_b._last_action_raw)
                logger.record_step(round_num, 1, player_b._last_action_prompt, player_b._last_action_raw, {
                    "thought": action_reasoning, "action": "choose_action", "argument": move_b,
                    "has_pause": False, "decision": move_b, "parse_error": None,
                    "agent": player_b.name, "phase": "action",
                    "rule_applied": action_metadata.get("rule_applied", ""),
                    "classification_signal": action_metadata.get("classification_signal", ""),
                }, observation=f"Action chosen: {move_b}",
                   state_before=state_snap, state_after=snapshot_game_state(game))

        obs_a, obs_b = game.apply_moves(move_a, move_b)
        pts_a, pts_b = SymmetricPD.PAYOFFS[(move_a, move_b)]

        print(f"\n  MOVES REVEALED:")
        print(f"     {player_a.name} → {move_a.upper()}")
        print(f"     {player_b.name} → {move_b.upper()}")
        print(f"     Points: {player_a.name} {'+' if pts_a >= 0 else ''}{pts_a}  |  {player_b.name} {'+' if pts_b >= 0 else ''}{pts_b}")

        logger.record_round_result(round_num, {
            "move_a": move_a, "move_b": move_b,
            "msg_a": msg_a, "msg_b": msg_b,
            "pts_a": pts_a, "pts_b": pts_b,
            "score_a": game.score_a, "score_b": game.score_b,
        })

    # ── SUMMARY ──
    avg_a = game.score_a / game.rounds
    avg_b = game.score_b / game.rounds

    print(f"\n{'='*65}")
    print(f"  MATCH OVER")
    print(f"  Final Score: {player_a.name} {game.score_a}  |  {player_b.name} {game.score_b}")
    print(f"  Avg Payoff:  {player_a.name} {avg_a:.2f}  |  {player_b.name} {avg_b:.2f}")
    if game.score_a > game.score_b:
        print(f"  Winner: {player_a.name}")
    elif game.score_b > game.score_a:
        print(f"  Winner: {player_b.name}")
    else:
        print(f"  Draw.")

    print(f"\n  Round-by-round:")
    for i, (ma, mb) in enumerate(game.history, 1):
        pts = SymmetricPD.PAYOFFS[(ma, mb)]
        msg_a_i, msg_b_i = game.message_history[i - 1]
        print(f"    R{i}: {player_a.name}={ma:<10} {player_b.name}={mb:<10} pts: {'+' if pts[0] >= 0 else ''}{pts[0]} / {'+' if pts[1] >= 0 else ''}{pts[1]}")
        print(f"         msgs: \"{msg_a_i[:60]}\" | \"{msg_b_i[:60]}\"")

    logger.finish({
        "score_a": game.score_a, "score_b": game.score_b,
        "avg_a": avg_a, "avg_b": avg_b,
        "history": game.history,
        "message_history": game.message_history,
        "player_a": player_a.name, "player_b": player_b.name,
        "winner": (player_a.name if game.score_a > game.score_b
                   else player_b.name if game.score_b > game.score_a else "draw"),
    })
    trace_path = logger.save()
    print(f"\n  Trace saved to: {trace_path}")

    return {"score_a": game.score_a, "score_b": game.score_b, "avg_a": avg_a, "avg_b": avg_b}


# =============================================================================
# PLAYER FACTORY
# =============================================================================

PLAYER_TYPES = ["react", "single_call"] + ScriptedPlayer.STRATEGIES


def make_player(player_type: str, client: LLMClient | None, model: str | None = None, name: str | None = None) -> PlayerAdapter:
    if player_type == "react":
        return ReactPlayer(client=client, model=model, name=name or "ReAct")
    elif player_type == "single_call":
        return SingleCallPlayer(client=client, model=model, name=name or "SingleCall")
    elif player_type in ScriptedPlayer.STRATEGIES:
        return ScriptedPlayer(strategy=player_type, name=name)
    else:
        raise ValueError(f"Unknown player type '{player_type}'. Choose from: {PLAYER_TYPES}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="A/B test harness for PD agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available player types: {', '.join(PLAYER_TYPES)}",
    )
    parser.add_argument("--player-a", default="single_call", choices=PLAYER_TYPES)
    parser.add_argument("--player-b", default="react", choices=PLAYER_TYPES)
    parser.add_argument("--model-a", default=None, help="LLM model for player A")
    parser.add_argument("--model-b", default=None, help="LLM model for player B")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    needs_client = args.player_a in ("react", "single_call") or args.player_b in ("react", "single_call")
    client = create_client() if needs_client else None

    if client:
        print(f"  LLM Provider: {client.provider.upper()} / {client.default_model}")

    pa = make_player(args.player_a, client, model=args.model_a)
    pb = make_player(args.player_b, client, model=args.model_b)

    run_match(pa, pb, rounds=args.rounds)
