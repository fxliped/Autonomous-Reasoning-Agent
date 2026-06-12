"""Cross-tournament opponent memory for the AltruAgent runner.

The platform reuses the same opponents across tournament instances, and the
"Honest Reciprocator with Memory" strategy (Rules 1 & 5) depends on remembering
who defected/exploited so it can open with defection against them next time.

This stores one record per opponent (keyed by opponent name) in a JSON file,
updated at the end of each match and read back into the agent's context at the
start of the next one. Classification is computed by a deterministic heuristic
over the opponent's observed moves, kept stable so the agent's behavior is
predictable across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = ROOT_DIR / "altruagent_opponent_memory.json"

# Classification labels mirror the strategy's vocabulary.
COOPERATOR = "COOPERATOR"
RECIPROCATOR = "RECIPROCATOR"
DEFECTOR = "DEFECTOR"
EXPLOITER = "EXPLOITER"
UNKNOWN = "UNKNOWN"


def cold_defection_stats(opp_moves: list[str], my_moves: list[str]) -> tuple[int, int, float]:
    """Unprovoked-defection stats, the single source of truth for 'hostile'.

    An OPPORTUNITY is a round where we gave them a fair chance to cooperate:
    Round 1 (free choice), or any round where WE cooperated the prior round. A
    COLD defection is a defection on an opportunity — a true defector defects
    cold; a reciprocator only defects AFTER we defect (provoked, not counted).

    Returns (opportunities, cold_defections, cold_rate). Used by BOTH classify()
    (over a completed match) and the runner's in-match threat signal (over the
    partial match so far), so the two can never drift apart.
    """
    opportunities = 0
    cold = 0
    for i, opp in enumerate(opp_moves):
        my_prev = my_moves[i - 1] if i > 0 and (i - 1) < len(my_moves) else None
        if i == 0 or my_prev == "cooperate":
            opportunities += 1
            if opp == "defect":
                cold += 1
    cold_rate = (cold / opportunities) if opportunities else 0.0
    return opportunities, cold, cold_rate


def classify(opp_moves: list[str], my_moves: list[str]) -> str:
    """Deterministic classification from one match's move sequences.

    moves are "cooperate"/"defect" strings, round-aligned (index 0 = round 1).

    The DEFECTOR label is the dangerous one: it is sticky and Rule 1 makes us
    open with defection against it forever, so a FALSE hostile label costs us
    cooperation (-2/round) permanently. To avoid mislabeling a reciprocator we
    provoked, we classify on *cold* (unprovoked) defections, not a flat rate:

    - An OPPORTUNITY is a round where we gave them a fair chance to cooperate:
      Round 1 (free choice), or any round where WE cooperated the prior round.
    - A COLD defection is a defection on an opportunity. A true defector defects
      cold; a reciprocator only defects AFTER we defect (provoked — not counted).

    Labels:
    - No defections at all                       -> COOPERATOR
    - Cold defection after 3+ mutual-coop streak -> EXPLOITER (calculated betrayal)
    - Cold on >50% of opportunities, with >=2
      opportunities seen (confidence gate)       -> DEFECTOR
    - Otherwise (responsive / provoked only)     -> RECIPROCATOR
    """
    if not opp_moves:
        return UNKNOWN

    defections = sum(1 for m in opp_moves if m == "defect")
    if defections == 0:
        return COOPERATOR

    # Detect a betrayal: opponent defected while we cooperated, after a stretch
    # of mutual cooperation. Highly diagnostic of a planned exploiter.
    coop_streak = 0
    for i, opp in enumerate(opp_moves):
        mine = my_moves[i] if i < len(my_moves) else "cooperate"
        if opp == "defect" and mine == "cooperate" and coop_streak >= 3:
            return EXPLOITER
        if opp == "cooperate" and mine == "cooperate":
            coop_streak += 1
        else:
            coop_streak = 0

    # Cold-defection analysis (shared with the runner's in-match threat signal).
    opportunities, _cold, cold_rate = cold_defection_stats(opp_moves, my_moves)
    # Confidence gate: never brand a hostile DEFECTOR off a single data point.
    if cold_rate > 0.5 and opportunities >= 2:
        return DEFECTOR
    return RECIPROCATOR


# Labels that mean "open with defection next time" (strategy Rule 1).
HOSTILE = {DEFECTOR, EXPLOITER}


class OpponentMemory:
    """JSON-backed per-opponent record store."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_MEMORY_PATH
        self.data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                self.data = {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def _key(self, opponent_name: str) -> str:
        return (opponent_name or "unknown").strip().lower()

    def get(self, opponent_name: str) -> Optional[dict]:
        return self.data.get(self._key(opponent_name))

    def classification(self, opponent_name: str) -> str:
        rec = self.get(opponent_name)
        return rec["classification"] if rec else UNKNOWN

    def matches_played(self, opponent_name: str) -> int:
        """How many COMPLETED matches we've had with this opponent.

        The runner uses this to derive the tournament number: since each
        opponent is faced once per tournament, the match about to start is
        tournament (matches_played + 1).
        """
        rec = self.get(opponent_name)
        return int(rec.get("matches_played", 0)) if rec else 0

    def forgiveness_used(self, opponent_name: str) -> bool:
        """Whether we've ever spent our one forgiveness on this opponent.

        One-time resource per opponent for the ENTIRE competition (all
        tournaments, all matches). Defaults False for a never-seen opponent.
        """
        rec = self.get(opponent_name)
        return bool(rec.get("forgiveness_used", False)) if rec else False

    def mark_forgiveness_used(self, opponent_name: str) -> None:
        """Permanently set the one-time forgiveness flag and persist it.

        Creates a minimal record if the opponent has never been recorded yet
        (forgiveness can be granted during the very first match, before any
        record_match call). This flag is NEVER reset anywhere — once True it
        stays True across every future match and tournament.
        """
        key = self._key(opponent_name)
        rec = self.data.get(key)
        if not rec:
            rec = {
                "name": opponent_name,
                "matches_played": 0,
                "last_match_opp_moves": [],
                "last_match_my_moves": [],
                "history_summary": "",
                "forgiveness_used": False,
            }
            self.data[key] = rec
        rec["forgiveness_used"] = True
        self.save()

    def is_hostile(self, opponent_name: str) -> bool:
        return self.classification(opponent_name) in HOSTILE

    def record_match(
        self,
        opponent_name: str,
        opp_moves: list[str],
        my_moves: list[str],
        my_avg: Optional[float] = None,
        opp_avg: Optional[float] = None,
        agent_note: str = "",
    ) -> str:
        """Update (or create) the opponent record after a finished match.

        Returns the (possibly escalated) classification. Once an opponent has
        been seen as DEFECTOR/EXPLOITER, that label sticks — we don't downgrade
        a confirmed hostile just because they behaved in one later match.
        """
        key = self._key(opponent_name)
        new_label = classify(opp_moves, my_moves)
        prior = self.data.get(key)

        if prior and prior.get("classification") in HOSTILE:
            label = prior["classification"]  # sticky: stay hostile
        else:
            label = new_label

        record = prior or {
            "name": opponent_name,
            "matches_played": 0,
            "last_match_opp_moves": [],
            "last_match_my_moves": [],
            "history_summary": "",
            "forgiveness_used": False,
        }
        record["name"] = opponent_name
        record["matches_played"] = record.get("matches_played", 0) + 1
        record["classification"] = label
        record["last_match_opp_moves"] = opp_moves
        record["last_match_my_moves"] = my_moves
        if my_avg is not None:
            record["last_my_avg"] = round(my_avg, 3)
        if opp_avg is not None:
            record["last_opp_avg"] = round(opp_avg, 3)
        if agent_note:
            record["last_agent_note"] = agent_note[:300]
        self.data[key] = record
        return label

    def context_block(self, opponent_name: str, prior_forgiveness: Optional[bool] = None) -> str:
        """One-paragraph PRIOR-TOURNAMENT memory summary for the agent's context.

        This reports only state from COMPLETED prior matches. The forgiveness
        line is driven by `prior_forgiveness` — a snapshot of the flag taken at
        THIS match's start — so it always reflects prior-tournament usage and can
        never show the paradox "0 prior matches but forgiveness used" (which used
        to make the model hallucinate prior tournaments). Forgiveness spent during
        the current match is reported separately by the runner in a THIS-MATCH
        line; if `prior_forgiveness` is omitted we fall back to the stored flag.
        """
        rec = self.get(opponent_name)
        if prior_forgiveness is None:
            prior_forgiveness = bool(rec.get("forgiveness_used", False)) if rec else False
        prior_f = "yes" if prior_forgiveness else "no"
        if not rec:
            return (
                "No prior history with this opponent. Treat as UNKNOWN. "
                f"Forgiveness used in a prior tournament: {prior_f}."
            )
        moves = rec.get("last_match_opp_moves", [])
        move_str = ", ".join(moves) if moves else "n/a"
        parts = [
            f"Prior matches with this opponent: {rec.get('matches_played', 0)}.",
            f"Stored classification: {rec.get('classification', UNKNOWN)}.",
            f"Their moves in your last match (round order): [{move_str}].",
        ]
        if "last_opp_avg" in rec:
            parts.append(
                f"Last match avg/round — you: {rec.get('last_my_avg')}, them: {rec.get('last_opp_avg')}."
            )
        if rec.get("last_agent_note"):
            parts.append(f"Prior note: {rec['last_agent_note']}")
        parts.append(f"Forgiveness used in a prior tournament: {prior_f}.")
        return " ".join(parts)
