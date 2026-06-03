"""
Context string builders for TournamentAgent's two-phase round structure.

Each function assembles a fully-formatted context block that gets passed to the LLM.
All context is explicit and numeric — no "fill in the blank" for facts the agent can compute.

  build_message_context   — MESSAGING PHASE (opponent message not yet visible)
  build_action_context    — MOVING PHASE (opponent message revealed)

Supporting helpers:
  arc_phase        — returns (phase_name, directive) for the current match round
  score_gap_block  — score differential + concrete threshold recommendation
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.memory import format_opponent_context  # noqa: E402
from .classifier import BehavioralProfile  # noqa: E402


# =============================================================================
# MATCH ARC
# =============================================================================

def arc_phase(round_num: int, total_rounds: int) -> tuple[str, str]:
    """
    Returns (phase_name, directive) for the current round.

    Phases: PROBE → CLASSIFY → EXECUTE → BUILD → PRE-HARVEST → FINAL.
    Each phase provides a concrete directive that grounds the LLM's reasoning.
    """
    rounds_left = total_rounds - round_num + 1

    if round_num == 1:
        return (
            "PROBE",
            "Round 1 — Probe: Cooperate and observe. Your goal is a data point, not +5. "
            "Send an intelligence signal (reference their likely strategy, project memory). "
            "Their action this round reveals more than their message.",
        )
    if round_num == 2 and total_rounds >= 4:
        return (
            "CLASSIFY",
            "Round 2 — Classify: You now have one data point. "
            "Defect round 1 from them → likely Always Defect or punishing — switch to defect now. "
            "Cooperate round 1 from them → continue cooperating to confirm — one round is not enough.",
        )
    if round_num >= total_rounds:
        return (
            "FINAL",
            "FINAL ROUND — No retaliation possible after this. "
            "Defect unless you are already locked in mutual defection (both scoring 0). "
            "Cooperating here while opponent defects costs 6 points with zero future recovery. "
            "This is the one round where defection has no downside against any non-zero-defecting opponent.",
        )
    if round_num == total_rounds - 1:
        return (
            "PRE-HARVEST",
            f"Round {round_num} — Pre-Harvest ({rounds_left} rounds left including this): "
            "Low-forgiveness opponent (Grim-like): defect NOW — 2 rounds of retaliation is cheap. "
            "High-forgiveness / Naive Cooperator: cooperate this round, defect final. "
            "Your message should reinforce cooperation to prime their final-round action.",
        )
    if total_rounds - round_num <= max(1, round(total_rounds * 0.3)):
        return (
            "BUILD",
            f"Round {round_num} — Build ({rounds_left} rounds left): "
            "Entering the harvest window. Reinforce cooperative narrative in your message. "
            "Stay cooperative to prime the late defection — but mirror defect immediately if they defect.",
        )
    return (
        "EXECUTE",
        f"Round {round_num} — Execute ({rounds_left} rounds left): "
        "Maintain your classified strategy. Mirror defection immediately — "
        "don't let a single unreciprocated cooperation become a pattern.",
    )


# =============================================================================
# SCORE GAP
# =============================================================================

def score_gap_block(
    my_score: float,
    opp_score: float,
    round_num: int,
    total_rounds: int,
    p_opp_c: float,
    ev_c: float,
    ev_d: float,
) -> str:
    """Score differential + concrete threshold recommendation for the action context."""
    rounds_left = total_rounds - round_num + 1
    gap = my_score - opp_score
    ev_diff = ev_d - ev_c
    max_coop_pts = rounds_left * 2

    header = (
        f"SCORE GAP: you {my_score:+.1f} | them {opp_score:+.1f} | "
        f"{'leading' if gap > 0 else 'trailing' if gap < 0 else 'tied'} by {abs(gap):.1f}  "
        f"({rounds_left} round(s) left)"
    )

    if gap > 0 and gap > rounds_left * 3:
        rec = (
            "SAFE LEAD: Opponent cannot catch you with all-defect play. "
            "Cooperate freely — +2/round locks in the win."
        )
    elif gap < 0 and abs(gap) > rounds_left * 3:
        rec = (
            f"CRITICAL DEFICIT: Trail by {abs(gap):.1f} with {rounds_left} round(s) left. "
            f"Full cooperation nets at most +{max_coop_pts:.0f}. "
            "Must take defection risks — accept -1 exposure to attempt +5 extractions."
        )
    elif gap < -0.5:
        breakeven = "ABOVE" if p_opp_c > 0.33 else "BELOW"
        rec = (
            f"BEHIND: P(opp_C)={p_opp_c:.0%} is {breakeven} the 33% defection breakeven. "
            f"EV(defect)={ev_d:.2f} vs EV(cooperate)={ev_c:.2f} → "
            f"defection has {ev_diff:+.2f} EV advantage."
        )
    elif gap > 0.5:
        rec = (
            f"LEADING: EV(cooperate)={ev_c:.2f} may protect the lead. "
            f"EV(defect)={ev_d:.2f} has {ev_diff:+.2f} EV advantage — "
            f"{'justified if P(opp_C) is reliable.' if ev_diff > 1.5 else 'marginal — cooperation safer to hold lead.'}"
        )
    else:
        rec = (
            f"EVEN: EV(cooperate)={ev_c:.2f} vs EV(defect)={ev_d:.2f} "
            f"({ev_diff:+.2f} defection advantage per round)."
        )

    return f"{header}\n{rec}"


# =============================================================================
# HISTORY BLOCK
# =============================================================================

def _history_block(match_history: list[dict]) -> str:
    if not match_history:
        return "PREVIOUS ROUNDS: None — this is round 1."
    lines = []
    for r in match_history:
        my_pts = r.get('my_pts') or 0
        opp_pts = r.get('opp_pts') or 0
        my_said = f' | I said: "{r["my_msg"]}"' if r.get("my_msg") else ""
        lines.append(
            f"  R{r['round']}: I played {r.get('my_action','?')} | "
            f"they played {r.get('opp_action','?')} | "
            f"pts me{'+' if my_pts >= 0 else ''}{my_pts} "
            f"them{'+' if opp_pts >= 0 else ''}{opp_pts} | "
            f"they said: \"{r.get('opp_msg','')}\"" + my_said
        )
    return "PREVIOUS ROUNDS:\n" + "\n".join(lines)


def _grudge_warning(profile: dict) -> str:
    """Warning injected when we've previously defected against this opponent."""
    if profile.get("matches_played", 0) == 0:
        return ""
    history = profile.get("match_history", [])
    prior_defections = sum(
        1 for m in history
        for r in m.get("rounds", [])
        if r.get("my_action") == "defect"
    )
    if prior_defections == 0:
        return ""
    last_match = history[-1] if history else {}
    if last_match.get("opp_score", 0) > last_match.get("my_score", 0):
        return (
            f"\nGRUDGE ALERT: You defected {prior_defections} time(s) in prior matches. "
            "Opponent likely knows this and may defect preemptively. "
            "Consider Apology-Recovery strategy, or accept arms race and defect immediately.\n"
        )
    return (
        f"\nHISTORY NOTE: You defected {prior_defections} time(s) against this opponent before "
        "and came out ahead. They may be wary — open with cooperative intelligence signaling.\n"
    )


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def build_message_context(
    round_num: int,
    total_rounds: int,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
    deception_count: int = 0,
    leaderboard_block: str = "",
    behavioral: BehavioralProfile | None = None,
) -> str:
    """Context for MESSAGING PHASE — opponent's current message not yet visible."""
    rounds_left = total_rounds - round_num + 1
    prior_rounds = round_num - 1
    deception_note = (
        f" | Deception used: {deception_count}/{prior_rounds} prior round(s)"
        if prior_rounds > 0 else ""
    )
    grudge = _grudge_warning(opponent_profile)
    lb_line = f"\n{leaderboard_block.strip()}" if leaderboard_block.strip() else ""
    arc_name, arc_directive = arc_phase(round_num, total_rounds)
    behavioral_line = (
        f"\nBEHAVIORAL SIGNALS: {behavioral.summary()}"
        if behavioral and behavioral.rounds_seen > 0 else ""
    )
    return f"""
MESSAGING PHASE — Round {round_num}/{total_rounds} [{arc_name}] | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}{deception_note}{grudge}{lb_line}

ARC DIRECTIVE: {arc_directive}{behavioral_line}

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


def build_action_context(
    round_num: int,
    total_rounds: int,
    opponent_message: str,
    my_message: str,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
    hypothesis: str = "Unknown",
    hypothesis_violations: int = 0,
    leaderboard_block: str = "",
    behavioral: BehavioralProfile | None = None,
    p_opp_c: float = 0.5,
) -> str:
    """Context for MOVING PHASE — opponent's current message is now visible."""
    rounds_left = total_rounds - round_num + 1
    reclassify_banner = (
        f"\n⚠ RE-EXAMINE CLASSIFICATION: Opponent acted against your '{hypothesis}' "
        f"hypothesis {hypothesis_violations} time(s). Ignore prior label — reason from raw move history.\n"
    ) if hypothesis_violations >= 2 else ""
    lb_line = f"\n{leaderboard_block.strip()}" if leaderboard_block.strip() else ""
    arc_name, arc_directive = arc_phase(round_num, total_rounds)

    ev_c = p_opp_c * 2 + (1 - p_opp_c) * (-1)
    ev_d = p_opp_c * 5
    ev_diff = ev_d - ev_c

    behavioral_line = (
        f"Behavioral P(opp cooperates) = {p_opp_c:.0%} | {behavioral.summary()}"
        if behavioral and behavioral.rounds_seen > 0
        else f"P(opp cooperates) = {p_opp_c:.0%} (no behavioral data yet — using prior)"
    )
    gap_block = score_gap_block(my_score, opp_score, round_num, total_rounds, p_opp_c, ev_c, ev_d)

    return f"""
MOVING PHASE — Round {round_num}/{total_rounds} [{arc_name}] | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}{reclassify_banner}{lb_line}

ARC DIRECTIVE: {arc_directive}

{gap_block}

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
What they want me to do: <cooperate | defect | uncertain>
What I suspect they will actually do (ignore their message, use action history): <cooperate | defect>

[THEORY OF MIND]
Level 1 — What will opponent do THIS round? (cite their action pattern, not their message)
Level 2 — What does opponent think I will do? (based on my prior actions + my message this round)
Level 3 — Given Level 2, what is their optimal play against that expectation?
Level 4 — Given Level 3, what is MY optimal counter-move?
Conclusion: <cooperate | defect and why>

[ACTION DECISION]
{behavioral_line}
EV(cooperate) = {p_opp_c:.2f}×2 + {1-p_opp_c:.2f}×(-1) = {ev_c:.2f}
EV(defect)    = {p_opp_c:.2f}×5 + {1-p_opp_c:.2f}×0    = {ev_d:.2f}
EV advantage of defection: {ev_diff:+.2f} pts/round
Override P(opp cooperates) if you have strong evidence (cite rounds): <updated_p or "no override">
Decision: <cooperate | defect>
Reason: <one sentence>
Deception (my message differs from my action)? <yes | no>

FINAL ACTION: <cooperate | defect>
""".strip()
