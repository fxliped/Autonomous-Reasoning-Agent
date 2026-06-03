"""
Strategy review — reads accumulated match data + reflections and proposes targeted
updates to the tournament agent's strategic framework.

Run manually between tournaments, not during. Output is written to
agent/reflections/strategy_updates.md and automatically injected into future
match system prompts alongside per-match reflections.

Usage:
    python analytics/strategy_review.py             # review and write update
    python analytics/strategy_review.py --dry-run   # print proposal, don't write
    python analytics/strategy_review.py --min 5     # require at least 5 matches (default: 3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import create_client, load_reflections, REFLECTIONS_DIR  # noqa: E402
from analytics.queries import (  # noqa: E402
    deception_rate,
    decision_trends,
    highest_regret_rounds,
    phase_adherence,
    top_opponents,
)
from tournament.core.prompts import TOURNAMENT_SYSTEM_PROMPT  # noqa: E402

GAME_NAME = "prisoners_dilemma"
UPDATE_FILE = REFLECTIONS_DIR / "strategy_updates.md"


def _build_review_prompt(
    reflections: str,
    trends: list[dict],
    phases: dict,
    deception: dict,
    opponents: list[dict],
    regret_rounds: list[dict] | None = None,
) -> str:
    n_matches = len(trends)
    wins = sum(1 for t in trends if t.get("outcome") == "WIN")
    losses = sum(1 for t in trends if t.get("outcome") == "LOSS")
    avg_coop = sum(t.get("coop_rate", 0) for t in trends) / max(n_matches, 1)
    avg_dec = sum(t.get("agent_deception_rate", 0) for t in trends) / max(n_matches, 1)

    phase_lines = "\n".join(
        f"  {phase}: coop {data['coop_rate']:.0%} "
        f"({data['cooperate']}C / {data['defect']}D)"
        for phase, data in sorted(phases.items())
    ) or "  No phase data yet."

    opp_lines = "\n".join(
        f"  {o['opponent_id']}: {o['matches_played']} match(es), "
        f"score diff {o.get('score_diff', 0):+.2f}, type={o.get('classified_type','?')}"
        for o in opponents
    ) or "  No opponent data yet."

    regret_lines = ""
    if regret_rounds:
        lines = "\n".join(
            f"  R{r['round_num']} run={r['run_id'][-8:]}: played {r['my_action']} vs "
            f"{r['opp_action']} → {r['my_pts']:+.0f} pts | "
            f"other action would have scored {r['counterfactual_pts']:+.0f} "
            f"(regret {r['regret']:+.1f} pts)"
            for r in regret_rounds
        )
        regret_lines = f"\nHIGHEST-REGRET ROUNDS (costliest individual decisions):\n{lines}"

    return f"""You are a strategic coach for a Prisoner's Dilemma tournament agent.
Review the agent's performance data and reflections, then propose specific improvements
to its strategic framework. Be concrete — cite specific data, not vague advice.

CURRENT STRATEGIC FRAMEWORK (do not repeat this — propose changes to it):
---
{TOURNAMENT_SYSTEM_PROMPT}
---

PERFORMANCE SUMMARY ({n_matches} matches played):
  Win/Loss/Draw: {wins}W / {losses}L / {n_matches - wins - losses}D
  Avg cooperation rate: {avg_coop:.0%}
  Avg deception rate (coop msg + defect): {avg_dec:.0%}
  Agent deception: {deception.get('lie_rounds', 0)} lie rounds / {deception.get('coop_msg_rounds', 0)} coop-msg rounds

PHASE ADHERENCE (actual cooperation rate per arc phase):
{phase_lines}

TOP OPPONENTS FACED:
{opp_lines}{regret_lines}

RECENT PER-MATCH REFLECTIONS (most recent first):
{reflections or "No reflections yet."}

---

Identify 1-4 specific strategic improvements. For each, provide:
1. What the data shows (cite numbers)
2. Which part of the framework to change (be exact — quote the relevant line)
3. The proposed replacement or addition

Focus on:
- Arc phase timing (is PROBE/BUILD/HARVEST/FINAL happening at the right rounds?)
- Deception calibration (too much, too little, or predictable pattern?)
- Opponent type handling gaps (are there types the agent consistently loses to?)
- Message strategy (what framing is working or failing?)

End with a STRATEGY UPDATE SUMMARY: a single paragraph (≤120 words) of the most important
changes, written as a directive the agent can act on immediately in its next match.
This summary will be injected directly into the agent's system prompt.

Return your full analysis, ending with:
STRATEGY UPDATE SUMMARY:
<the paragraph>""".strip()


def run_strategy_review(
    min_matches: int = 3,
    dry_run: bool = False,
    verbose: bool = True,
) -> str | None:
    """
    Run a strategy review and optionally write the update to disk.
    Returns the proposed strategy update summary, or None if insufficient data.
    """
    trends = decision_trends(GAME_NAME, last_n_runs=20)
    if len(trends) < min_matches:
        print(
            f"[StrategyReview] Only {len(trends)} match(es) in DB — "
            f"need at least {min_matches}. Run more tournament matches first."
        )
        return None

    reflections = load_reflections(GAME_NAME, max_reflections=10)
    phases = phase_adherence(GAME_NAME)
    dec = deception_rate("agent", GAME_NAME)
    opponents = top_opponents(n=8)
    regret_rounds = highest_regret_rounds(GAME_NAME, last_n_runs=20, top_n=5)

    prompt = _build_review_prompt(reflections, trends, phases, dec, opponents, regret_rounds)

    if verbose:
        print(f"[StrategyReview] Reviewing {len(trends)} matches...")

    client = create_client()
    system = (
        "You are a game theory strategy coach. Be specific, data-driven, and actionable. "
        "Do not pad with generic advice."
    )
    response = client.complete(system, [{"role": "user", "content": prompt}])

    if verbose:
        print("\n" + "=" * 60)
        print(response)
        print("=" * 60 + "\n")

    # Extract the summary paragraph
    summary = ""
    if "STRATEGY UPDATE SUMMARY:" in response:
        summary = response.split("STRATEGY UPDATE SUMMARY:")[-1].strip()

    if not summary:
        print("[StrategyReview] Could not extract STRATEGY UPDATE SUMMARY from response.")
        return None

    if dry_run:
        print("[StrategyReview] Dry run — not writing to disk.")
        return summary

    # ── Pre-deployment validation gate ──────────────────────────────────────────
    # Run Tier 1 fast benchmarks before applying any strategy update.
    # A failing Tier 1 means the update would be applied to a broken baseline.
    print("\n[StrategyReview] Running Tier 1 benchmarks before applying update...")
    from tournament.eval.benchmark import run_fast_benchmarks  # noqa: PLC0415
    tier1_failures = run_fast_benchmarks()
    if tier1_failures > 0:
        print(
            f"\n[StrategyReview] BLOCKED: {tier1_failures} Tier 1 benchmark(s) failed. "
            "Fix the failures before applying a strategy update. "
            "Run: python tournament/eval/benchmark.py"
        )
        return None

    print("[StrategyReview] Tier 1 passed — writing strategy update.")

    # Write to strategy_updates.md
    REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime  # noqa: PLC0415
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    existing = UPDATE_FILE.read_text(encoding="utf-8") if UPDATE_FILE.exists() else ""
    separator = "\n\n" if existing.strip() else ""
    with UPDATE_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{separator}## Strategy Update — {timestamp} ({len(trends)} matches)\n\n{summary}\n")

    print(f"[StrategyReview] Strategy update written to {UPDATE_FILE.relative_to(ROOT_DIR)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament strategy self-review")
    parser.add_argument("--dry-run", action="store_true", help="Print proposal without writing")
    parser.add_argument("--min", type=int, default=3, dest="min_matches",
                        help="Minimum matches required (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="Suppress full LLM output")
    args = parser.parse_args()

    run_strategy_review(
        min_matches=args.min_matches,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
