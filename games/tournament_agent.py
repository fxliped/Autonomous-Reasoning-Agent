"""
TournamentAgent — AltruAgent platform submission interface.

The two-phase round structure mirrors the platform exactly:

  MESSAGING PHASE (blind — you write before seeing opponent's message):
    message = agent.compose_message(round_num, total_rounds, history, my_score, opp_score)

  MOVING PHASE (opponent's message now revealed):
    action_int = agent.choose_action(round_num, total_rounds, opp_msg, my_msg, history, ...)
    # 0 = Cooperate, 1 = Defect

  After round result is known:
    agent.record_round_result(round_num, opp_action_label, my_pts, opp_pts)

  After game ends:
    agent.end_match(my_avg_score, opp_avg_score)  # per-round averages

Test locally:
    python games/tournament_agent.py --strategy tit_for_tat
"""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client
from agent.memory import (
    load_opponent_profile,
    save_opponent_profile,
    log_tournament_result,
    format_opponent_context,
    update_profile_after_match,
)
from games.pd_game import PrisonersDilemma

GAME_NAME = "prisoners_dilemma"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

TOURNAMENT_SYSTEM_PROMPT = """
You are a strategic agent competing in a Prisoner's Dilemma tournament.

PAYOFF MATRIX (platform-confirmed values):
  Both cooperate:             you +2, them +2  (action 0)
  You cooperate, they defect: you -1, them +5
  You defect, they cooperate: you +5, them -1  (action 1)
  Both defect:                you  0, them  0

KEY IMPLICATIONS:
- Being suckered is actively punishing (-1), not just neutral
- Exploitation gap: +5 vs -1 = 6-point swing per round
- Mutual defection yields nothing — pointless arms races cost you
- Final score is per-round average, so every round counts equally

ROUND STRUCTURE (BLIND MESSAGING):
1. MESSAGING PHASE: Both players write one message simultaneously (up to 50 words).
   You CANNOT see the opponent's message yet. Both messages reveal together.
2. MOVING PHASE: Both messages are revealed. You now see what they said.
   Both players choose cooperate or defect simultaneously.
3. Payoffs revealed. Next round begins.

STRATEGIC PRINCIPLES:
- Your message is a strategic signal, not a commitment
- Opponent messages are signals, not commitments — check their history, not their words
- Calibrated deception: cooperative reputation with selective betrayal at high-value moments
- Against a confirmed Always-Defector: defect every round immediately
- Against unknowns: cooperate round 1, observe, classify, adapt
- Late-round defection (final 1-2 rounds) has minimal retaliation cost
""".strip()


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def _history_block(match_history: list[dict]) -> str:
    if not match_history:
        return "PREVIOUS ROUNDS: None — this is round 1."
    lines = [
        f"  R{r['round']}: I played {r.get('my_action','?')} | "
        f"they played {r.get('opp_action','?')} | "
        f"pts me{'+' if (r.get('my_pts') or 0)>=0 else ''}{r.get('my_pts','?')} "
        f"them{'+' if (r.get('opp_pts') or 0)>=0 else ''}{r.get('opp_pts','?')} | "
        f"they said: \"{r.get('opp_msg','')}\""
        for r in match_history
    ]
    return "PREVIOUS ROUNDS:\n" + "\n".join(lines)


def _compose_message_context(
    round_num: int,
    total_rounds: int,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
) -> str:
    """Context for MESSAGING PHASE — opponent's current message not yet visible."""
    rounds_left = total_rounds - round_num + 1
    return f"""
MESSAGING PHASE — Round {round_num}/{total_rounds} | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}

{format_opponent_context(opponent_profile)}

{_history_block(match_history)}

IMPORTANT: You are writing your message BEFORE seeing the opponent's message this round.
Both messages reveal simultaneously. Your message does NOT commit you to any action.

Complete every field:

[SITUATION]
Opponent type hypothesis: <Always Defect | Naive Cooperator | Tit-for-Tat | Grim Trigger | Pavlov | Strategic/Adaptive | Unknown>
Confidence: <%>
Evidence: <cite specific rounds or patterns>

[MESSAGE STRATEGY]
Goal: <what I want them to believe or do>
Approach: <cooperative signal | credible threat | deception | sympathy | neutral>

FINAL MESSAGE: "<exact message text, max 50 words>"
""".strip()


def _choose_action_context(
    round_num: int,
    total_rounds: int,
    opponent_message: str,
    my_message: str,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
) -> str:
    """Context for MOVING PHASE — opponent's current message is now visible."""
    rounds_left = total_rounds - round_num + 1
    return f"""
MOVING PHASE — Round {round_num}/{total_rounds} | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}

{format_opponent_context(opponent_profile)}

{_history_block(match_history)}

THIS ROUND'S MESSAGES (just revealed — both sent blind):
  My message:         "{my_message}"
  Opponent's message: "{opponent_message}"

Complete every field:

[OPPONENT ANALYSIS]
Type hypothesis: <Always Defect | Naive Cooperator | Tit-for-Tat | Grim Trigger | Pavlov | Strategic/Adaptive | Unknown>
Confidence: <%>
Evidence: <cite specific rounds>

[OPPONENT MESSAGE ANALYSIS]
What it signals: <what they want me to believe or do>
Is it credible? <yes | no | partial — compare to their action history>

[ACTION DECISION]
EV(cooperate) = <if opponent cooperates P% of the time: P*2 + (1-P)*(-1)>
EV(defect)    = <if opponent cooperates P% of the time: P*5 + (1-P)*0>
Decision: <cooperate | defect>
Reason: <one sentence>
Deception (my message differs from my action)? <yes | no>

FINAL ACTION: <cooperate | defect>
""".strip()


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

def _parse_message(response: str) -> str:
    """Extract final message text from compose_message response."""
    if not response:
        return "Let's cooperate for mutual benefit."
    hit = re.search(r'FINAL MESSAGE:\s*["\']?(.*?)["\']?\s*$', response,
                    re.IGNORECASE | re.MULTILINE)
    if hit:
        msg = hit.group(1).strip().strip("\"'")
        if msg:
            return msg
    quotes = re.findall(r'"([^"]{5,})"', response)
    if quotes:
        return quotes[-1].strip()
    return "Let's cooperate for mutual benefit."


def _parse_action(response: str) -> int:
    """Extract action int (0=cooperate, 1=defect) from choose_action response."""
    if not response:
        return 0
    hit = re.search(r'FINAL ACTION:\s*(cooperate|defect)', response, re.IGNORECASE)
    if hit:
        return 0 if hit.group(1).lower() == "cooperate" else 1
    hits = re.findall(r'\bDecision:\s*(cooperate|defect)\b', response, re.IGNORECASE)
    if hits:
        return 0 if hits[-1].lower() == "cooperate" else 1
    words = re.findall(r'\b(cooperate|defect)\b', response, re.IGNORECASE)
    if words:
        return 0 if words[-1].lower() == "cooperate" else 1
    return 0


# =============================================================================
# TOURNAMENT AGENT
# =============================================================================

class TournamentAgent:
    """
    Tournament-ready Prisoner's Dilemma agent for the AltruAgent platform.

    Matches the platform's two-phase round structure:
      MESSAGING PHASE (blind — both write simultaneously):
        message = agent.compose_message(round_num, total_rounds, history, my_score, opp_score)
        # POST /message {"content": message, "type": "chat"} then POST /message {"type": "terminate"}

      MOVING PHASE (both messages now revealed):
        action_int = agent.choose_action(round_num, total_rounds, opp_msg, my_msg, history, ...)
        # POST /step {"action": action_int}  (0=Cooperate, 1=Defect)

      After round completes (last_round visible in state):
        agent.record_round_result(round_num, opp_action_label, my_pts, opp_pts)

      After game terminal:
        agent.end_match(my_avg_score, opp_avg_score)  # per-round averages from returns
    """

    ACTION_COOPERATE = 0
    ACTION_DEFECT = 1

    PAYOFFS = PrisonersDilemma.PAYOFFS

    def __init__(self, opponent_id: str = "unknown", total_rounds: int = 10):
        self.opponent_id = opponent_id
        self.total_rounds = total_rounds
        self.client = create_client()
        self.profile = load_opponent_profile(opponent_id)
        self._match_rounds: list[dict] = []
        self._system = build_system_prompt(TOURNAMENT_SYSTEM_PROMPT, game_name=GAME_NAME)

    def compose_message(
        self,
        round_num: int,
        total_rounds: int,
        match_history: list[dict],
        my_score: float = 0.0,
        opp_score: float = 0.0,
    ) -> str:
        """
        MESSAGING PHASE: compose our message before seeing the opponent's.
        Returns message string capped at 50 words.
        match_history: completed rounds (keys: round, my_action, opp_action, my_pts, opp_pts, opp_msg).
        """
        context = _compose_message_context(
            round_num=round_num,
            total_rounds=total_rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
            opponent_profile=self.profile,
        )
        agent = Agent(client=self.client, system=self._system)
        response = agent(context)
        print(f"\n[TournamentAgent R{round_num} MESSAGE]\n{response}\n")

        message = _parse_message(response or "")
        words = message.split()
        if len(words) > 50:
            message = " ".join(words[:50])
        return message

    def choose_action(
        self,
        round_num: int,
        total_rounds: int,
        opponent_message: str,
        my_message: str,
        match_history: list[dict],
        my_score: float = 0.0,
        opp_score: float = 0.0,
    ) -> int:
        """
        MOVING PHASE: choose action after seeing opponent's message.
        Returns 0 (Cooperate) or 1 (Defect).
        """
        context = _choose_action_context(
            round_num=round_num,
            total_rounds=total_rounds,
            opponent_message=opponent_message,
            my_message=my_message,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
            opponent_profile=self.profile,
        )
        agent = Agent(client=self.client, system=self._system)
        response = agent(context)
        print(f"\n[TournamentAgent R{round_num} ACTION]\n{response}\n")

        action_int = _parse_action(response or "")
        action_label = "cooperate" if action_int == 0 else "defect"

        self._match_rounds.append({
            "round": round_num,
            "opp_msg": opponent_message,
            "my_msg": my_message,
            "my_action": action_label,
            "opp_action": None,
            "my_pts": None,
            "opp_pts": None,
        })
        return action_int

    def record_round_result(
        self, round_num: int, opp_action: str, my_pts: float, opp_pts: float
    ) -> None:
        """Call after both moves are revealed to record the completed round."""
        for r in self._match_rounds:
            if r["round"] == round_num:
                r["opp_action"] = opp_action
                r["my_pts"] = my_pts
                r["opp_pts"] = opp_pts
                break

    def end_match(self, my_avg_score: float, opp_avg_score: float) -> None:
        """
        Summarize match via LLM, update opponent profile, persist to disk.
        my_avg_score / opp_avg_score: per-round averages from the platform's returns field.
        """
        self.profile = update_profile_after_match(
            self.profile,
            self._match_rounds,
            my_avg_score,
            opp_avg_score,
            self.client,
        )
        save_opponent_profile(self.profile)
        log_tournament_result({
            "opponent_id": self.opponent_id,
            "my_avg_score": my_avg_score,
            "opp_avg_score": opp_avg_score,
            "rounds": self._match_rounds,
        })
        self._match_rounds = []
        print(f"\n[TournamentAgent] Profile saved for '{self.opponent_id}'.")


# =============================================================================
# LOCAL TEST HARNESS
# =============================================================================

def run_tournament_match(
    opponent_strategy: str = "tit_for_tat",
    rounds: int = 10,
    opponent_id: str | None = None,
) -> None:
    """Test TournamentAgent locally against a scripted NPC (no platform required)."""
    opp_id = opponent_id or f"npc_{opponent_strategy}"
    agent = TournamentAgent(opponent_id=opp_id, total_rounds=rounds)
    game = PrisonersDilemma(rounds=rounds, opponent_strategy=opponent_strategy)
    match_history: list[dict] = []

    _NPC_MESSAGES = {
        "always_cooperate": "I always cooperate — let's both benefit.",
        "always_defect": "Do what you want. I play my own game.",
        "tit_for_tat": "I mirror what you do. Cooperate and so will I.",
        "grim_trigger": "Betray me once and I defect forever.",
        "pavlov": "I adjust based on what worked last round.",
        "generous_tft": "I cooperate by default and forgive mistakes.",
        "random": "Who knows? Let's see what happens.",
    }
    npc_msg = _NPC_MESSAGES.get(opponent_strategy, "Let's play.")

    print("=" * 60)
    print(f"TOURNAMENT MATCH — TournamentAgent vs {opponent_strategy}")
    print(f"Rounds: {rounds}  |  Opponent ID: {opp_id}")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        my_score = float(game.agent_score)
        opp_score = float(game.opponent_score)
        print(f"\n── Round {round_num} ── Score: me {my_score:.0f} | them {opp_score:.0f}")

        # MESSAGING PHASE
        message = agent.compose_message(
            round_num=round_num,
            total_rounds=rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
        )
        print(f"  Agent message : {message}")

        # MOVING PHASE (NPC message revealed)
        action_int = agent.choose_action(
            round_num=round_num,
            total_rounds=rounds,
            opponent_message=npc_msg,
            my_message=message,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
        )
        action = "cooperate" if action_int == 0 else "defect"
        print(f"  Agent action  : {action}")

        game.make_move(action)
        last_a, last_o = game.history[-1]
        my_pts, opp_pts = PrisonersDilemma.PAYOFFS[(last_a, last_o)]

        print(f"  NPC action    : {last_o}")
        print(f"  Points        : me {my_pts:+} | them {opp_pts:+}")

        agent.record_round_result(round_num, last_o, float(my_pts), float(opp_pts))
        match_history.append({
            "round": round_num,
            "opp_msg": npc_msg,
            "my_action": action,
            "opp_action": last_o,
            "my_pts": my_pts,
            "opp_pts": opp_pts,
        })

    print("\n" + "=" * 60)
    print(f"MATCH OVER — me {game.agent_score} | them {game.opponent_score}")
    if game.agent_score > game.opponent_score:
        print("Agent wins!")
    elif game.opponent_score > game.agent_score:
        print("Opponent wins.")
    else:
        print("Draw.")

    agent.end_match(game.agent_score / rounds, game.opponent_score / rounds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TournamentAgent local test")
    parser.add_argument("--strategy", default="tit_for_tat",
                        choices=PrisonersDilemma.OPPONENT_STRATEGIES)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--opponent", default=None)
    args = parser.parse_args()
    run_tournament_match(
        opponent_strategy=args.strategy,
        rounds=args.rounds,
        opponent_id=args.opponent,
    )
