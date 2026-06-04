"""
Opponent memory helpers — format and LLM-based post-match analysis.

  format_opponent_context(profile)   → str  (context block for system prompts)
  update_profile_after_match(...)    → dict (LLM classification + lesson extraction)

Storage is handled by analytics.db. Callers that need load/save/log should import
directly from analytics.db: load_opponent, save_opponent, log_match.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def format_opponent_context(profile: dict) -> str:
    """Format opponent profile as a text block for system prompt injection."""
    if profile.get("matches_played", 0) == 0:
        return "OPPONENT HISTORY: First time facing this opponent — no prior data."

    lines = [f"OPPONENT HISTORY: {profile['matches_played']} previous match(es)."]

    if profile.get("classified_type"):
        conf = f"{profile['type_confidence']:.0%}" if profile.get("type_confidence") else "?"
        lines.append(
            f"Classified strategy: {profile['classified_type']} (confidence: {conf})"
        )

    my_total = profile.get("total_my_score", 0) or 0
    opp_total = profile.get("total_opp_score", 0) or 0
    adv = my_total - opp_total
    sign = "+" if adv >= 0 else ""
    lines.append(
        f"Cumulative score: you {my_total} vs them {opp_total} ({sign}{adv} net)"
    )

    if profile.get("match_history"):
        last = profile["match_history"][-1]
        lines.append(
            f"Last match: you {last.get('my_score','?')} | them {last.get('opp_score','?')} — "
            f"strategy used: {last.get('strategy_used', 'unknown')}"
        )
        if last.get("notes"):
            lines.append(f"Key takeaway: {last['notes']}")

    lie_rate = profile.get("msg_lie_rate")
    if lie_rate is not None:
        lie_pct = f"{lie_rate:.0%}"
        if lie_rate >= 0.5:
            credibility = f"LOW ({lie_pct} of cooperative messages were followed by defection — treat their messages as noise)"
        elif lie_rate >= 0.25:
            credibility = f"MIXED ({lie_pct} cooperative messages were lies — verify against actions)"
        else:
            credibility = f"HIGH ({lie_pct} lie rate — their cooperative messages are generally reliable)"
        lines.append(f"Their message credibility: {credibility}")

    my_lie_rate = profile.get("my_lie_rate_to_opp")
    if my_lie_rate is not None:
        if my_lie_rate >= 0.4:
            lines.append(
                f"YOUR credibility to them: LOW ({my_lie_rate:.0%} of your cooperative messages were lies). "
                "They likely discount your cooperative framing — Apology-Recovery before any extraction."
            )
        elif my_lie_rate >= 0.15:
            lines.append(
                f"YOUR credibility to them: MODERATE ({my_lie_rate:.0%} lie rate). "
                "Selective deception still possible but varied messaging is critical."
            )
        else:
            lines.append(
                f"YOUR credibility to them: HIGH ({my_lie_rate:.0%} lie rate). "
                "Your cooperative messages carry weight — use this window strategically."
            )

    if profile.get("effective_messages"):
        lines.append(
            "Messaging that worked: " + "; ".join(profile["effective_messages"][-3:])
        )
    if profile.get("failed_messages"):
        lines.append(
            "Messaging that backfired: " + "; ".join(profile["failed_messages"][-3:])
        )

    return "\n".join(lines)


def update_profile_after_match(
    profile: dict,
    match_rounds: list[dict],
    my_final_score: float,
    opp_final_score: float,
    client,
) -> dict:
    """
    LLM call: summarize the match, classify opponent type, extract lessons.
    match_rounds: list of dicts with keys round, opp_msg, my_msg, my_action, opp_action, my_pts, opp_pts
    Updates profile in-place and returns it.
    """
    rounds_str = "\n".join(
        f"  R{r['round']}: opp_msg='{r.get('opp_msg', '')}' | "
        f"my={r.get('my_action','?')} opp={r.get('opp_action','?')} | "
        f"pts me+{r.get('my_pts','?')} them+{r.get('opp_pts','?')}"
        for r in match_rounds
    )
    prompt = (
        f"Prisoner's Dilemma match against '{profile['opponent_id']}':\n{rounds_str}\n"
        f"Final: me {my_final_score}, them {opp_final_score}\n\n"
        "Respond with JSON only — no markdown fences:\n"
        "{\n"
        '  "classified_type": "Always Defect|Naive Cooperator|Tit-for-Tat|Grim Trigger|Pavlov|Strategic/Adaptive|Unknown",\n'
        '  "type_confidence": 0.0-1.0,\n'
        '  "strategy_used": "short label for what I did",\n'
        '  "effective_messages": ["phrase1"],\n'
        '  "failed_messages": ["phrase1"],\n'
        '  "notes": "one sentence — what to do differently next match"\n'
        "}"
    )

    analysis: dict = {}
    try:
        raw = client.complete(
            system="You are a strategic analyst for iterated Prisoner's Dilemma. Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = raw.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        analysis = json.loads(text.strip())
    except Exception:
        analysis = {}

    profile["matches_played"] = profile.get("matches_played", 0) + 1
    profile["total_my_score"] = (profile.get("total_my_score") or 0) + my_final_score
    profile["total_opp_score"] = (profile.get("total_opp_score") or 0) + opp_final_score

    if analysis.get("classified_type"):
        profile["classified_type"] = analysis["classified_type"]
    if analysis.get("type_confidence") is not None:
        profile["type_confidence"] = float(analysis["type_confidence"])

    for msg in analysis.get("effective_messages", []):
        if msg and msg not in profile.get("effective_messages", []):
            profile.setdefault("effective_messages", []).append(msg)
    for msg in analysis.get("failed_messages", []):
        if msg and msg not in profile.get("failed_messages", []):
            profile.setdefault("failed_messages", []).append(msg)

    profile.setdefault("match_history", []).append({
        "my_score": my_final_score,
        "opp_score": opp_final_score,
        "strategy_used": analysis.get("strategy_used", "unknown"),
        "notes": analysis.get("notes", ""),
        "rounds": match_rounds,
    })

    return profile
