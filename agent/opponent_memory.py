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


def classify(opp_moves: list[str], my_moves: list[str]) -> str:
    """Deterministic classification from one match's move sequences.

    moves are "cooperate"/"defect" strings, round-aligned (index 0 = round 1).
    - No defections at all                      -> COOPERATOR
    - Defected while we were cooperating, after
      3+ rounds of mutual cooperation (mid/late
      betrayal)                                  -> EXPLOITER
    - Defected on >50% of rounds                 -> DEFECTOR
    - Otherwise (responsive defections)          -> RECIPROCATOR
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

    if defections / len(opp_moves) > 0.5:
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

    def context_block(self, opponent_name: str) -> str:
        """One-paragraph memory summary to inject into the agent's context."""
        rec = self.get(opponent_name)
        if not rec:
            return "No prior history with this opponent. Treat as UNKNOWN."
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
        return " ".join(parts)
