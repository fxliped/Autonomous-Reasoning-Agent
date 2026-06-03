"""
Analytics CLI — query and display data from agent.db.

Usage:
  python analytics/report.py                      # summary overview
  python analytics/report.py --ingest             # ingest all trace JSONs first
  python analytics/report.py --deception          # deception breakdown by actor
  python analytics/report.py --trends             # cooperation/deception trend over last 20 runs
  python analytics/report.py --opponent <id>      # per-opponent deep-dive
  python analytics/report.py --phases             # arc-phase cooperation rates
  python analytics/report.py --ingest --deception # ingest then show deception
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analytics.db import ingest_all, get_conn, DB_PATH  # noqa: E402
from analytics.queries import (  # noqa: E402
    deception_rate,
    decision_trends,
    opponent_summary,
    phase_adherence,
    top_opponents,
)


# ─── PRINTERS ─────────────────────────────────────────────────────────────────

def _print_summary() -> None:
    print(f"\n{'='*60}")
    print("  ANALYTICS SUMMARY")
    print(f"  DB: {DB_PATH}")
    print(f"{'='*60}")

    with get_conn() as conn:
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_steps = conn.execute("SELECT COUNT(*) FROM react_steps").fetchone()[0]
        n_rounds = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
        n_opp = conn.execute("SELECT COUNT(*) FROM opponents").fetchone()[0]
        n_dec = conn.execute("SELECT SUM(is_deception) FROM deception_events").fetchone()[0] or 0

    print(f"\n  Runs ingested:      {n_runs}")
    print(f"  ReAct steps:        {n_steps}")
    print(f"  Rounds recorded:    {n_rounds}")
    print(f"  Opponents tracked:  {n_opp}")
    print(f"  Deception events:   {n_dec}")

    if n_opp > 0:
        print(f"\n  Top opponents by matches played:")
        for opp in top_opponents(5):
            diff = opp["score_diff"]
            sign = "+" if diff >= 0 else ""
            print(f"    {opp['opponent_id']:<25}  {opp['matches_played']} matches  "
                  f"net {sign}{diff:.1f}  type: {opp['classified_type'] or '?'}")


def _print_deception() -> None:
    print(f"\n{'='*60}")
    print("  DECEPTION ANALYSIS")
    print(f"{'='*60}")
    for actor, label in [("agent", "Our agent"), ("opponent", "Opponents")]:
        d = deception_rate(actor=actor)
        if d["total_rounds"] == 0:
            print(f"\n  {label}: no data yet.")
            continue
        print(f"\n  {label}:")
        print(f"    Coop-message + defect-action (lie) rounds: {d['lie_rounds']}")
        print(f"    Rounds with cooperative messages:          {d['coop_msg_rounds']}")
        print(f"    Deception rate:                            {d['deception_rate']:.1%}")
        print(f"    Total rounds in DB:                        {d['total_rounds']}")


def _print_trends(game_name: str = "prisoners_dilemma") -> None:
    print(f"\n{'='*60}")
    print(f"  DECISION TRENDS — {game_name} (last 20 runs)")
    print(f"{'='*60}")
    rows = decision_trends(game_name=game_name, last_n_runs=20)
    if not rows:
        print("  No data yet.")
        return
    print(f"\n  {'Run ID':<32}  {'Outcome':<7}  {'Coop%':>6}  {'Lie%':>6}  {'Rounds':>7}")
    print(f"  {'-'*32}  {'-'*7}  {'------':>6}  {'------':>6}  {'-------':>7}")
    for r in rows:
        print(f"  {r['run_id']:<32}  {r['outcome']:<7}  "
              f"{r['coop_rate']:>5.0%}  {r['agent_deception_rate']:>5.0%}  {r['n_rounds']:>7}")


def _print_opponent(opponent_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"  OPPONENT: {opponent_id}")
    print(f"{'='*60}")
    d = opponent_summary(opponent_id)
    if "error" in d:
        print(f"  Not found in DB.")
        return
    print(f"\n  Matches played:  {d.get('matches_played', 0)}")
    print(f"  Classified type: {d.get('classified_type') or 'Unknown'} "
          f"({d.get('type_confidence', 0):.0%})")
    my = d.get('total_my_score', 0)
    opp = d.get('total_opp_score', 0)
    print(f"  Score: us {my:.1f} | them {opp:.1f}  (net {my - opp:+.1f})")
    print(f"  Lie rate (their coop msg → defect): {d.get('opp_deception_rate', 0):.0%}")
    if d.get("effective_messages"):
        print(f"  Messages that worked: {'; '.join(d['effective_messages'][-3:])}")
    if d.get("failed_messages"):
        print(f"  Messages that failed: {'; '.join(d['failed_messages'][-3:])}")
    if d.get("match_history_summary"):
        print(f"\n  Match history:")
        for m in d["match_history_summary"][-5:]:
            diff = (m.get("my_score") or 0) - (m.get("opp_score") or 0)
            print(f"    us {m.get('my_score','?'):.2f} | them {m.get('opp_score','?'):.2f}  "
                  f"({diff:+.2f})  {m.get('strategy_used','')}")


def _print_phases(game_name: str = "prisoners_dilemma") -> None:
    print(f"\n{'='*60}")
    print(f"  ARC-PHASE ADHERENCE — {game_name}")
    print(f"{'='*60}")
    phases = phase_adherence(game_name=game_name)
    if not phases:
        print("  No data yet.")
        return
    order = ["PROBE", "CLASSIFY", "EXECUTE", "BUILD", "PRE-HARVEST", "FINAL"]
    print(f"\n  {'Phase':<14}  {'C':>5}  {'D':>5}  {'Coop%':>7}")
    print(f"  {'-'*14}  {'-----':>5}  {'-----':>5}  {'-------':>7}")
    for phase in order:
        if phase not in phases:
            continue
        b = phases[phase]
        print(f"  {phase:<14}  {b['cooperate']:>5}  {b['defect']:>5}  {b['coop_rate']:>6.0%}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent analytics report")
    parser.add_argument("--ingest",    action="store_true", help="Ingest trace JSONs before reporting")
    parser.add_argument("--deception", action="store_true", help="Show deception breakdown")
    parser.add_argument("--trends",    action="store_true", help="Show decision trends over last 20 runs")
    parser.add_argument("--opponent",  metavar="ID",        help="Per-opponent deep-dive")
    parser.add_argument("--phases",    action="store_true", help="Arc-phase cooperation breakdown")
    parser.add_argument("--game",      default="prisoners_dilemma", help="Game name filter (default: prisoners_dilemma)")
    args = parser.parse_args()

    if args.ingest:
        n = ingest_all()
        print(f"  Ingested {n} new trace file(s).")

    show_all = not any([args.deception, args.trends, args.opponent, args.phases])
    if show_all:
        _print_summary()
        _print_deception()
    else:
        if args.deception:
            _print_deception()
        if args.trends:
            _print_trends(args.game)
        if args.opponent:
            _print_opponent(args.opponent)
        if args.phases:
            _print_phases(args.game)

    print()


if __name__ == "__main__":
    main()
