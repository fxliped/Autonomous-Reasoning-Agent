"""Parse AltruAgent repeated-PD trace JSONs into two analysis CSVs.

Scope: ONLY matches played as the tournament agent "Secret Agent" (the agent
used in the final tournament). Practice matches played as sarthak_test_agent are
ignored entirely, and so is our OWN other test agent "Secret Agents #1" (it was
one of our registrations, not an opposing team). That leaves the 6 real
opposing teams (18 matches, 3 each) — the 7-team field minus us.

Two outputs (written to ./analysis/):
  - match_summary.csv : one row per match (the headline table)
  - round_detail.csv  : one row per round (move-by-move, with rule + message)

The authoritative moves are final_result.my_moves / opp_moves (indexed by
round). The per-round message step gives the planned action + rule + message;
comparing planned vs. the move that actually landed flags move-window misses
(planned a move, but no action step fired -> platform auto-cooperated).

Usage:
  python3 parse_traces.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACES_DIR = ROOT / "traces"
OUT_DIR = ROOT / "analysis"

SELF_AGENT = "Secret Agent"  # the tournament agent; everything else is ignored

# Our OWN other registration — an internal test agent, NOT an opposing team.
# Excluded from the CSVs and all stats so the field is exactly the 6 real teams.
EXCLUDE_OPPONENTS = {"Secret Agents #1"}

# Repeated-PD payoff matrix. Key: (my_move, opp_move) -> (my_payoff, opp_payoff).
# 0=Cooperate, 1=Defect. CC=(2,2) CD=(-1,5) DC=(5,-1) DD=(0,0).
PAYOFF = {
    ("cooperate", "cooperate"): (2.0, 2.0),
    ("cooperate", "defect"): (-1.0, 5.0),
    ("defect", "cooperate"): (5.0, -1.0),
    ("defect", "defect"): (0.0, 0.0),
}

OUTCOME = {
    ("cooperate", "cooperate"): "CC",
    ("cooperate", "defect"): "CD",   # I got suckered
    ("defect", "cooperate"): "DC",   # I suckered them
    ("defect", "defect"): "DD",
}


def parse_run_id(run_id: str) -> tuple[str, str, str]:
    """run_id like '61b14a75_20260604_172344' -> (short_id, date, time)."""
    m = re.match(r"([0-9a-f]+)_([0-9]{8})_([0-9]{6})", run_id or "")
    if not m:
        return (run_id or "", "", "")
    sid, d, t = m.groups()
    date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    tm = f"{t[:2]}:{t[2:4]}:{t[4:]}"
    return (sid, date, tm)


def step_index(steps: list[dict]) -> dict[int, dict]:
    """Index a match's steps by round -> {'msg': step, 'action': step}."""
    by_round: dict[int, dict] = {}
    for s in steps:
        rnd = s.get("round")
        if rnd is None:
            continue
        phase = (s.get("parsed") or {}).get("phase")
        slot = by_round.setdefault(rnd, {})
        if phase == "message":
            slot["msg"] = s
        elif phase == "action":
            slot["action"] = s
    return by_round


def build_rows():
    match_rows = []
    round_rows = []

    files = sorted(glob.glob(str(TRACES_DIR / "altruagent_repeated_pd_*.json")))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d.get("self_name") != SELF_AGENT:
            continue
        if d.get("opponent", "") in EXCLUDE_OPPONENTS:
            continue  # our own test agent, not an opposing team

        run_id = d.get("run_id", "")
        short_id, date, time_ = parse_run_id(run_id)
        opp = d.get("opponent", "")
        fr = d.get("final_result") or {}
        my_moves = fr.get("my_moves") or []
        opp_moves = fr.get("opp_moves") or []
        n = min(len(my_moves), len(opp_moves))
        my_avg = fr.get("my_avg")
        opp_avg = fr.get("opp_avg")

        by_round = step_index(d.get("steps") or [])

        # --- per-round accumulation ---
        my_def = opp_def = mutual_coop = suckered = exploited = misses = 0
        for i in range(n):
            rnd = i + 1
            mm, om = my_moves[i], opp_moves[i]
            mp, op = PAYOFF.get((mm, om), (None, None))
            outcome = OUTCOME.get((mm, om), "??")
            if mm == "defect":
                my_def += 1
            if om == "defect":
                opp_def += 1
            if outcome == "CC":
                mutual_coop += 1
            elif outcome == "CD":
                suckered += 1
            elif outcome == "DC":
                exploited += 1

            slot = by_round.get(rnd, {})
            msg_step = slot.get("msg") or {}
            act_step = slot.get("action") or {}
            msg_parsed = msg_step.get("parsed") or {}
            planned = msg_parsed.get("planned_action")
            rule = msg_parsed.get("rule_applied") or ""
            message = msg_parsed.get("message") or ""
            has_action_step = bool(act_step)
            move_source = (act_step.get("parsed") or {}).get("move_source", "")
            # Miss = we planned a move but no action step fired and the move that
            # landed differs from the plan (platform auto-cooperated us).
            missed = bool(planned) and not has_action_step and planned != mm
            if missed:
                misses += 1

            round_rows.append({
                "run_id": short_id,
                "opponent": opp,
                "round": rnd,
                "my_move": mm,
                "opp_move": om,
                "my_payoff": mp,
                "opp_payoff": op,
                "outcome": outcome,
                "rule_applied": rule,
                "planned_action": planned or "",
                "move_landed": (planned == mm) if planned else "",
                "move_window_miss": missed,
                "move_source": move_source,
                "my_message": message,
            })

        margin = (my_avg - opp_avg) if (my_avg is not None and opp_avg is not None) else None
        if margin is None:
            result = ""
        elif margin > 1e-9:
            result = "W"
        elif margin < -1e-9:
            result = "L"
        else:
            result = "T"

        match_rows.append({
            "run_id": short_id,
            "date": date,
            "time": time_,
            "opponent": opp,
            "my_avg": my_avg,
            "opp_avg": opp_avg,
            "margin": round(margin, 4) if margin is not None else "",
            "result": result,
            "classification": fr.get("classification", ""),
            "n_rounds": n,
            "my_defect_count": my_def,
            "opp_defect_count": opp_def,
            "mutual_coop_rounds": mutual_coop,
            "i_got_suckered": suckered,
            "i_exploited_them": exploited,
            "move_window_misses": misses,
            "my_first_move": my_moves[0] if my_moves else "",
            "opp_first_move": opp_moves[0] if opp_moves else "",
            "my_moves": "|".join(my_moves),
            "opp_moves": "|".join(opp_moves),
        })

    return match_rows, round_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):3d} rows -> {path.relative_to(ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    match_rows, round_rows = build_rows()
    # Sort matches chronologically for readability.
    match_rows.sort(key=lambda r: (r["date"], r["time"]))
    write_csv(OUT_DIR / "match_summary.csv", match_rows)
    write_csv(OUT_DIR / "round_detail.csv", round_rows)

    teams = sorted({r["opponent"] for r in match_rows})
    print(f"\n  {len(match_rows)} matches vs {len(teams)} real teams: "
          f"{', '.join(teams)}")


if __name__ == "__main__":
    main()
