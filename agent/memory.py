"""Persistent opponent memory for cross-tournament learning."""

import json
import re
from datetime import date
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parent / "memory"
OPPONENTS_DIR = MEMORY_DIR / "opponents"
TOURNAMENTS_FILE = MEMORY_DIR / "tournaments" / "history.json"

_EMPTY_PROFILE: dict = {
    "opponent_id": None,
    "matches_played": 0,
    "classified_type": None,
    "type_confidence": 0.0,
    "total_my_score": 0,
    "total_opp_score": 0,
    "match_history": [],
    "effective_messages": [],
    "failed_messages": [],
    "notes": "",
    "last_updated": None,
}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def load_opponent_profile(opponent_id: str) -> dict:
    """Load opponent profile, or return a blank one if not yet seen."""
    OPPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OPPONENTS_DIR / f"{_safe(opponent_id)}.json"
    if not path.exists():
        profile = dict(_EMPTY_PROFILE)
        profile["opponent_id"] = opponent_id
        return profile
    return json.loads(path.read_text(encoding="utf-8"))


def save_opponent_profile(profile: dict) -> None:
    OPPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OPPONENTS_DIR / f"{_safe(profile['opponent_id'])}.json"
    profile["last_updated"] = str(date.today())
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def log_tournament_result(result: dict) -> None:
    """Append a match result to the global tournament history log."""
    TOURNAMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = (
        json.loads(TOURNAMENTS_FILE.read_text(encoding="utf-8"))
        if TOURNAMENTS_FILE.exists()
        else []
    )
    result["date"] = str(date.today())
    history.append(result)
    TOURNAMENTS_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def format_opponent_context(profile: dict) -> str:
    """Format opponent profile as a text block for system prompt injection."""
    if profile["matches_played"] == 0:
        return "OPPONENT HISTORY: First time facing this opponent — no prior data."

    lines = [f"OPPONENT HISTORY: {profile['matches_played']} previous match(es)."]

    if profile.get("classified_type"):
        conf = f"{profile['type_confidence']:.0%}" if profile.get("type_confidence") else "?"
        lines.append(
            f"Classified strategy: {profile['classified_type']} (confidence: {conf})"
        )

    adv = profile["total_my_score"] - profile["total_opp_score"]
    sign = "+" if adv >= 0 else ""
    lines.append(
        f"Cumulative score: you {profile['total_my_score']} vs them {profile['total_opp_score']} "
        f"({sign}{adv} net)"
    )

    if profile.get("match_history"):
        last = profile["match_history"][-1]
        lines.append(
            f"Last match: you {last.get('my_score','?')} | them {last.get('opp_score','?')} — "
            f"strategy used: {last.get('strategy_used', 'unknown')}"
        )
        if last.get("notes"):
            lines.append(f"Key takeaway: {last['notes']}")

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
        # strip markdown fences if LLM adds them anyway
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        analysis = json.loads(text.strip())
    except Exception:
        analysis = {}

    profile["matches_played"] += 1
    profile["total_my_score"] += my_final_score
    profile["total_opp_score"] += opp_final_score

    if analysis.get("classified_type"):
        profile["classified_type"] = analysis["classified_type"]
    if analysis.get("type_confidence") is not None:
        profile["type_confidence"] = float(analysis["type_confidence"])

    for msg in analysis.get("effective_messages", []):
        if msg and msg not in profile["effective_messages"]:
            profile["effective_messages"].append(msg)
    for msg in analysis.get("failed_messages", []):
        if msg and msg not in profile["failed_messages"]:
            profile["failed_messages"].append(msg)

    profile["match_history"].append({
        "my_score": my_final_score,
        "opp_score": opp_final_score,
        "strategy_used": analysis.get("strategy_used", "unknown"),
        "notes": analysis.get("notes", ""),
        "rounds": match_rounds,
    })

    return profile
