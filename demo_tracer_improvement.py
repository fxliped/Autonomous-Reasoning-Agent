"""
Tracer Improvement Demo — Before / After

Shows exactly how Tracer-style error localization improves agent behavior:

  Step 1 (BEFORE): Run agent vs Grim Trigger with NO prior reflections.
                   Agent may cooperate sub-optimally or defect too early.
  Step 2 (JUDGE):  Point Tracer at the saved trace.
                   Localizes the earliest reasoning failure to a specific
                   round + step, with category and suggested fix.
  Step 3 (AFTER):  Reflection is written to agent/reflections/prisoners_dilemma.md.
                   Run agent again — it reads the lesson and plays better.

Run:
    python demo_tracer_improvement.py
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import create_client, reflection_path
from agent.tracing import judge_trace, write_judge_result
from games.React_agent_gameV1 import PrisonersDilemma, run_game

DEMO_GAME_NAME = "prisoners_dilemma"  # shared with main PD game so reflections feed back in
OPPONENT = "grim_trigger"
ROUNDS = 5

SEPARATOR = "=" * 62


def clear_reflection():
    """Remove any prior reflection for this game so the BEFORE run is clean."""
    path = reflection_path(DEMO_GAME_NAME)
    if path.exists():
        path.write_text(
            f"# Prisoners Dilemma Reflections\n\nNo judged reflections yet.\n",
            encoding="utf-8",
        )
        print(f"  [Cleared prior reflection at {path}]")


def print_header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main():
    client = create_client()
    print(f"\n{'*'*62}")
    print(f"  TRACER IMPROVEMENT DEMO")
    print(f"  Provider: {client.provider}  |  Game: Prisoner's Dilemma vs {OPPONENT}")
    print(f"{'*'*62}")

    # ── STEP 1: BEFORE — run without any reflection ───────────────────────────
    print_header("STEP 1 — BEFORE TRACER: Agent plays with no prior lessons")
    print(
        "  Without Tracer, we can only observe the final score.\n"
        "  We don't know WHY the agent made suboptimal choices.\n"
    )
    clear_reflection()
    time.sleep(1)

    game_before = PrisonersDilemma(rounds=ROUNDS, opponent_strategy=OPPONENT)
    run_game(game_before, max_iterations=10, auto_judge=False)

    # Find the trace that was just saved
    traces_dir = ROOT_DIR / "traces"
    all_traces = sorted(
        traces_dir.glob(f"{DEMO_GAME_NAME}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not all_traces:
        print("ERROR: No trace file found. Run the game first.")
        sys.exit(1)
    trace_path = all_traces[-1]

    score_before = game_before.agent_score
    opp_score_before = game_before.opponent_score
    print(f"\n  WITHOUT TRACER — we only know the final score:")
    print(f"    Agent: {score_before}  |  Opponent: {opp_score_before}")
    print(f"    We cannot tell WHERE the reasoning went wrong.\n")

    # ── STEP 2: JUDGE — run Tracer on the trace ───────────────────────────────
    print_header("STEP 2 — TRACER JUDGE: Localizing reasoning failures")
    print(f"  Pointing Tracer at: {trace_path.name}\n")
    time.sleep(1)

    judge_result = judge_trace(
        trace_path,
        client=client,
        append_to_reflections=True,
    )
    write_judge_result(trace_path, judge_result)

    failures = judge_result.get("failures", [])
    reflection = judge_result.get("reflection", "")

    print("  WITH TRACER — step-level blame report:")
    if failures:
        for f in failures:
            print(f"\n    Round {f.get('round','?')}  Step {f.get('step','?')}")
            print(f"    Category : {f.get('category','?')}")
            print(f"    Evidence : {f.get('explanation','')}")
            print(f"    Fix      : {f.get('suggested_fix','')}")
    else:
        print("    No failures localized (agent played well).")

    if reflection:
        refl_path = reflection_path(DEMO_GAME_NAME)
        print(f"\n  Lesson written to: {refl_path}")
        print(f"  \"{reflection}\"")

    # ── STEP 3: AFTER — run again with reflection in system prompt ────────────
    print_header("STEP 3 — AFTER TRACER: Agent plays with reflection in memory")
    print(
        "  The agent now reads its past lesson at the start of every round.\n"
        "  Watch if the reasoning and score improve.\n"
    )
    time.sleep(1)

    game_after = PrisonersDilemma(rounds=ROUNDS, opponent_strategy=OPPONENT)
    run_game(game_after, max_iterations=10, auto_judge=False)

    score_after = game_after.agent_score
    opp_score_after = game_after.opponent_score

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print_header("DEMO SUMMARY — Before vs After Tracer")
    print(f"  {'':20}  {'Agent':>8}  {'Opponent':>10}")
    print(f"  {'BEFORE (no Tracer)':20}  {score_before:>8}  {opp_score_before:>10}")
    print(f"  {'AFTER  (w/ Tracer)':20}  {score_after:>8}  {opp_score_after:>10}")
    delta = score_after - score_before
    sign = "+" if delta >= 0 else ""
    print(f"\n  Agent score change: {sign}{delta} points")
    if delta > 0:
        print("  ✅ Tracer improved the agent's performance.")
    elif delta == 0:
        print("  ➖ Performance unchanged — agent already played near-optimally.")
    else:
        print("  ⚠️  Score lower — the specific run may have varied. Check reasoning quality.")
    print(f"\n  Trace files saved to: {traces_dir}/")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
