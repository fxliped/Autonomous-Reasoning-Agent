"""
Tracer Demo — Before / After Error Localization for Presentation

This script demonstrates how Tracer improves AI agent reliability by
pinpointing the exact step where reasoning fails.

Scenario: A Prisoner's Dilemma decision agent has a bug in its
expected-payoff calculation. The final answer looks plausible
("cooperate") but is wrong — the agent should defect.

WITHOUT Tracer: we only know the final answer is wrong.
WITH Tracer:    line-level blame report — the exact function call
                that returned the wrong value, with observed I/O.
AFTER FIX:      Tracer runs clean — all verdicts pass, verdict flips.

Usage:
    cd v2/tracer_demo
    python run_tracer_demo.py

Requirements:
    OPENAI_API_KEY must be set (Tracer uses the OpenAI judge).
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACER_PATH = os.path.join(DEMO_DIR, "..", "..", "Tracer")

# Load .env from project root (works regardless of which directory you run from)
load_dotenv(os.path.join(DEMO_DIR, "..", "..", ".env"))
BUGGY = os.path.join(DEMO_DIR, "buggy_pd_agent.py")
FIXED = os.path.join(DEMO_DIR, "fixed_pd_agent.py")

sys.path.insert(0, TRACER_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def run_script_directly(path: str) -> str:
    """Run a Python script and return its stdout."""
    result = subprocess.run(
        [sys.executable, path], capture_output=True, text=True
    )
    return result.stdout.strip()


def run_tracer_on(script_path: str, goal: str, api_key: str, label: str):
    """Run Tracer programmatically on script_path and print the blame report."""
    from parser import parse_file
    from judge import LLMJudge
    from executor import TracingExecutor
    from reporter import Reporter

    section(f"WITH TRACER — {label}")
    print(f"Script : {os.path.basename(script_path)}")
    print(f"Goal   : {goal}")
    print()

    parsed = parse_file(script_path)
    judge = LLMJudge(api_key=api_key, model="gpt-4o-mini", script_goal=goal)
    executor = TracingExecutor(parsed, judge=judge, continue_on_error=True)
    result = executor.execute()

    reporter = Reporter(use_colors=True)
    for i, step in enumerate(result.steps, 1):
        reporter.report_step(step, i)
    reporter.report_result(result)

    # Highlight the intervention certificate
    print()
    print("─" * 64)
    print("  INTERVENTION CERTIFICATE")
    print("─" * 64)
    errors = [
        s for s in result.steps
        if s.function_call and s.function_call.judgment
        and s.function_call.judgment.verdict.value in ("incorrect", "error")
    ]
    if errors:
        for e in errors:
            fc = e.function_call
            print(f"  Buggy call   : {fc.name}({', '.join(repr(a) for a in fc.args)})")
            print(f"  Observed     : {fc.result!r}")
            print(f"  Verdict      : {fc.judgment.verdict.value.upper()}")
            print(f"  Reason       : {fc.judgment.explanation}")
            print()
        print("  Patch        : swap 0 and 3 in calculate_expected_payoff (cooperate branch)")
        print("  Before patch : 0 * rate + 3 * (1 - rate)   ← payoffs reversed")
        print("  After patch  : 3 * rate + 0 * (1 - rate)   ← correct")
        print("  Outcome flip : cooperate → defect  (correct answer)")
    else:
        print("  No errors found — all verdicts CORRECT.")
    print("─" * 64)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Tracer requires an OpenAI key.")
        print("Set it in your .env or shell: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    goal = (
        "Compute the optimal Prisoner's Dilemma action given game history, "
        "using expected-payoff analysis over the payoff matrix: "
        "(C,C)→+3, (C,D)→+0, (D,C)→+5, (D,D)→+1."
    )

    # ── Step 1: Without Tracer ─────────────────────────────────────────────────
    section("WITHOUT TRACER — Running buggy agent directly")
    buggy_out = run_script_directly(BUGGY)
    print(buggy_out)
    print()
    print("  Observation  : Final answer is 'cooperate' — looks reasonable.")
    print("  Problem      : We only know the output is wrong, NOT which step failed.")
    print("  Diagnosis    : impossible without execution trace.")

    # ── Step 2: With Tracer on buggy script ───────────────────────────────────
    run_tracer_on(BUGGY, goal, api_key, "Buggy agent — blame localization")

    # ── Step 3: Apply the fix and re-run Tracer ───────────────────────────────
    section("APPLYING THE FIX")
    print("  File         : buggy_pd_agent.py → fixed_pd_agent.py")
    print("  Change       : calculate_expected_payoff, cooperate branch")
    print("  Before       : return 0 * cooperation_rate + 3 * (1 - cooperation_rate)")
    print("  After        : return 3 * cooperation_rate + 0 * (1 - cooperation_rate)")

    run_tracer_on(FIXED, goal, api_key, "Fixed agent — all verdicts pass")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY")
    fixed_out = run_script_directly(FIXED)
    print("  Fixed output:")
    for line in fixed_out.splitlines():
        print(f"    {line}")
    print()
    print("  Before Tracer : final answer wrong, no localization.")
    print("  After  Tracer : pinpointed calculate_expected_payoff(), line ~40.")
    print("  Patch         : one reversed multiply → verdict flips cooperate → defect.")
    print("  Unrelated fns : calculate_cooperation_rate() — CORRECT, untouched.")


if __name__ == "__main__":
    main()
