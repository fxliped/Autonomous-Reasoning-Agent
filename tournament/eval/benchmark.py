"""
Agent benchmark suite — two tiers.

TIER 1 — Fast (no LLM calls, runs in seconds):
  Tests classifier convergence, parser reliability, arc phase logic.
  Run after any code change to catch regressions.

TIER 2 — Eval (LLM calls required, ~3-5 min per full suite):
  Oracle efficiency: agent actual score / theoretical max score per NPC.
  Behavioral scenarios: does the agent make the right call in clear-cut situations?
  Run before a tournament to measure readiness.

Usage:
  python tournament/eval/benchmark.py                  # tier 1 only (fast)
  python tournament/eval/benchmark.py --eval           # tier 1 + tier 2 (LLM calls)
  python tournament/eval/benchmark.py --eval --n 3     # 3 games per NPC strategy (default: 2)
  python tournament/eval/benchmark.py --eval --scenarios  # also run behavioral assertions
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tournament.core.classifier import (  # noqa: E402
    BehavioralProfile, bayes_update, top_classification, _UNIFORM_PRIOR,
)
from tournament.core.prompts import parse_action, parse_message  # noqa: E402
from tournament.core.context import arc_phase, build_action_context  # noqa: E402
from games.pd_game import PrisonersDilemma  # noqa: E402

# =============================================================================
# ORACLE SCORES
# Oracle = best average score any agent can achieve against each NPC (10 rounds).
# Used to compute efficiency: actual_avg / oracle_avg.
# =============================================================================

ORACLE_SCORES: dict[str, float] = {
    "always_cooperate": 5.00,   # defect every round
    "random":           3.00,   # defect every round (random opp cooperates ~50%)
    "pavlov":           2.50,   # defect every round (Pavlov oscillates C/D)
    "tit_for_tat":      2.30,   # cooperate 9, defect final
    "grim_trigger":     2.30,   # cooperate 9, defect final
    "generous_tft":     2.30,   # cooperate 9, defect final
    "always_defect":    0.00,   # defect (best = 0; cooperating gives -1)
}

_SKIP_EFFICIENCY = {"always_defect", "random"}


# =============================================================================
# TIER 1 — FAST BENCHMARKS (no LLM)
# =============================================================================

def _check(name: str, passed: bool, detail: str = "") -> bool:
    status = "  PASS" if passed else "  FAIL"
    suffix = f"  — {detail}" if detail and not passed else ""
    print(f"{status}  {name}{suffix}")
    return passed


def run_fast_benchmarks() -> int:
    """Run all fast (no-LLM) benchmarks. Returns number of failures."""
    failures = 0
    print("\n── TIER 1: Fast benchmarks (no LLM) ──────────────────────────────")

    # --- Parser reliability ---
    cases = [
        ("FINAL ACTION: cooperate",                               0),
        ("FINAL ACTION: defect",                                  1),
        ("Decision: defect\nFINAL ACTION: cooperate",             0),
        ("",                                                       0),  # default
        ("cooperate cooperate defect",                            1),  # last word
    ]
    for raw, expected in cases:
        ok = parse_action(raw) == expected
        if not ok:
            failures += 1
        _check(f"parse_action({raw!r:.40}) == {expected}", ok)

    ok = parse_message('FINAL MESSAGE: "hello world"') == "hello world"
    failures += 0 if ok else 1
    _check("parse_message extracts quoted text", ok)

    ok = parse_message("") == "Let's cooperate for mutual benefit."
    failures += 0 if ok else 1
    _check("parse_message fallback default", ok)

    # --- Bayesian convergence ---
    prior = dict(_UNIFORM_PRIOR)
    post = prior
    history = []
    for rnd in range(1, 4):
        history.append({"round": rnd, "my_action": "cooperate", "opp_action": "defect",
                        "my_pts": -1, "opp_pts": 5, "opp_msg": ""})
        post = bayes_update(post, "defect", rnd, history)
    t, conf = top_classification(post)
    ok = t == "Always Defect" and conf > 0.7
    failures += 0 if ok else 1
    _check(f"Bayes 3×defect → Always Defect ({conf:.0%})", ok, f"got {t} ({conf:.0%})")

    prior2 = dict(_UNIFORM_PRIOR)
    h2 = [{"round": 1, "my_action": "cooperate", "opp_action": "cooperate",
            "my_pts": 2, "opp_pts": 2, "opp_msg": ""}]
    post2 = bayes_update(prior2, "cooperate", 2, h2)
    t2, _ = top_classification(post2)
    ok2 = t2 != "Always Defect"
    failures += 0 if ok2 else 1
    _check(f"Bayes after (C,C) → not Always Defect (got {t2})", ok2)

    # --- BehavioralProfile ---
    bp = BehavioralProfile()
    bp.update([
        {"round": 1, "my_action": "cooperate", "opp_action": "defect",
         "my_pts": -1, "opp_pts": 5, "opp_msg": "cooperate with me"},
        {"round": 2, "my_action": "cooperate", "opp_action": "defect",
         "my_pts": -1, "opp_pts": 5, "opp_msg": "trust me"},
        {"round": 3, "my_action": "defect",    "opp_action": "defect",
         "my_pts": 0,  "opp_pts": 0, "opp_msg": ""},
    ])
    ok = bp.coop_rate == 0.0 and bp.rounds_seen == 3
    failures += 0 if ok else 1
    _check(f"BehavioralProfile: coop_rate=0 after 3 opp defects", ok, f"coop_rate={bp.coop_rate}")

    ok = bp.msg_cred == 0.0
    failures += 0 if ok else 1
    _check("BehavioralProfile: msg_cred=0 (all coop msgs lied)", ok, f"msg_cred={bp.msg_cred:.0%}")

    # --- Arc phase sequence ---
    expected_phases = {
        (1, 8): "PROBE", (2, 8): "CLASSIFY", (7, 8): "PRE-HARVEST", (8, 8): "FINAL",
        (1, 3): "PROBE", (3, 3): "FINAL",
    }
    for (rnd, total), expected in expected_phases.items():
        got = arc_phase(rnd, total)[0]
        ok = got == expected
        failures += 0 if ok else 1
        _check(f"arc_phase({rnd}/{total}) == {expected}", ok, f"got {got}")

    mid_phases = {arc_phase(r, 8)[0] for r in range(3, 7)}
    ok = mid_phases <= {"EXECUTE", "BUILD"}
    failures += 0 if ok else 1
    _check(f"arc_phase mid-match in {{EXECUTE,BUILD}}", ok, str(mid_phases))

    return failures


# =============================================================================
# TIER 2A — ORACLE EFFICIENCY (LLM calls)
# =============================================================================

def run_oracle_efficiency(n_games: int = 2, rounds: int = 10) -> dict[str, dict]:
    """
    Run n_games per NPC strategy, compute efficiency vs. oracle.
    Returns {strategy: {avg_score, oracle, efficiency_pct, games}}.
    """
    from tournament.train import run_self_play_match

    print(f"\n── TIER 2A: Oracle efficiency ({n_games} game(s) × {len(ORACLE_SCORES)} strategies, "
          f"{rounds} rounds each) ──")
    print("  This requires LLM calls — please wait...\n")

    results: dict[str, dict] = {}

    for strategy, oracle in ORACLE_SCORES.items():
        scores = []
        for game_i in range(1, n_games + 1):
            t0 = time.time()
            _, my_avg, opp_avg = run_self_play_match(
                strategy=strategy, rounds=rounds, verbose=False,
            )
            elapsed = time.time() - t0
            scores.append(my_avg)
            print(f"  vs {strategy:<20} game {game_i}/{n_games}  "
                  f"agent={my_avg:.2f}  npc={opp_avg:.2f}  ({elapsed:.0f}s)")

        avg = sum(scores) / len(scores)
        efficiency = (avg / oracle * 100) if oracle > 0 else None
        results[strategy] = {
            "avg_score": round(avg, 3),
            "oracle": oracle,
            "efficiency_pct": round(efficiency, 1) if efficiency is not None else None,
            "games": n_games,
        }

    return results


def print_efficiency_table(results: dict[str, dict]) -> int:
    """Print oracle efficiency table. Returns number of failing strategies."""
    THRESHOLD = 70.0

    print(f"\n  {'Strategy':<20}  {'Actual':>7}  {'Oracle':>7}  {'Efficiency':>10}  {'Status'}")
    print(f"  {'-'*20}  {'-------':>7}  {'-------':>7}  {'----------':>10}  {'------'}")

    failures = 0
    for strategy, r in sorted(results.items(), key=lambda kv: -(kv[1]['efficiency_pct'] or 0)):
        eff = r["efficiency_pct"]
        if strategy in _SKIP_EFFICIENCY:
            status = "skip"
        elif eff is None:
            status = "n/a"
        elif eff >= THRESHOLD:
            status = "OK"
        else:
            status = f"LOW (<{THRESHOLD:.0f}%)"
            failures += 1

        eff_str = f"{eff:.1f}%" if eff is not None else "n/a"
        print(f"  {strategy:<20}  {r['avg_score']:>7.2f}  {r['oracle']:>7.2f}  {eff_str:>10}  {status}")

    print(f"\n  Efficiency = actual avg score / oracle avg score × 100%.")
    print(f"  Threshold: ≥{THRESHOLD:.0f}% per strategy (except always_defect, random).")
    return failures


# =============================================================================
# TIER 2B — BEHAVIORAL SCENARIOS (LLM calls)
# =============================================================================

_SCENARIOS = [
    {
        "name": "Final round vs confirmed cooperator",
        "description": "Round 8/8, opponent cooperated every round — should defect (free +5)",
        "round_num": 8, "total_rounds": 8,
        "opp_msg": "One last round together — let's both cooperate.",
        "my_msg": "I'll cooperate as always.",
        "match_history": [
            {"round": r, "my_action": "cooperate", "opp_action": "cooperate",
             "my_pts": 2, "opp_pts": 2, "opp_msg": "cooperating"}
            for r in range(1, 8)
        ],
        "my_score": 14.0, "opp_score": 14.0,
        "expected_action": 1,
        "rationale": "Final round with no retaliation possible against cooperative opponent.",
    },
    {
        "name": "Always Defector confirmed — defect back",
        "description": "Opponent defected every round — Bayesian posterior is Always Defect >90%",
        "round_num": 4, "total_rounds": 8,
        "opp_msg": "Do what you want.",
        "my_msg": "I cooperate.",
        "match_history": [
            {"round": r, "my_action": "cooperate", "opp_action": "defect",
             "my_pts": -1, "opp_pts": 5, "opp_msg": "whatever"}
            for r in range(1, 4)
        ],
        "my_score": -3.0, "opp_score": 15.0,
        "expected_action": 1,
        "rationale": "Always Defector confirmed — cooperating only gifts them +5.",
    },
    {
        "name": "Round 1 — cooperate to probe",
        "description": "No history, should cooperate to gather data",
        "round_num": 1, "total_rounds": 8,
        "opp_msg": "Let us see how this goes.",
        "my_msg": "Looking forward to playing.",
        "match_history": [],
        "my_score": 0.0, "opp_score": 0.0,
        "expected_action": 0,
        "rationale": "First round — cooperate to establish baseline and classify opponent.",
    },
]

MIN_PASS_RATE = 0.67


def run_behavioral_scenarios(n_trials: int = 3) -> int:
    """
    Run each behavioral scenario n_trials times, assert expected action.
    Returns number of failing scenarios.
    """
    from agent.agent import Agent, create_client, build_system_prompt
    from tournament.core.prompts import TOURNAMENT_SYSTEM_PROMPT
    from tournament.core.classifier import BehavioralProfile

    GAME_NAME = "prisoners_dilemma"
    client = create_client()
    system = build_system_prompt(TOURNAMENT_SYSTEM_PROMPT, game_name=GAME_NAME, use_react=False)

    print(f"\n── TIER 2B: Behavioral scenarios ({n_trials} trial(s) each, "
          f"pass if ≥{MIN_PASS_RATE:.0%}) ──")

    failures = 0
    for sc in _SCENARIOS:
        print(f"\n  Scenario: {sc['name']}")
        print(f"  Expected: {'cooperate' if sc['expected_action'] == 0 else 'defect'}  "
              f"— {sc['rationale']}")

        posterior = dict(_UNIFORM_PRIOR)
        history = sc["match_history"]
        for r in history:
            posterior = bayes_update(posterior, r["opp_action"], r["round"], history)
        bayes_type, bayes_conf = top_classification(posterior)

        bp = BehavioralProfile()
        bp.update(history)

        ctx = build_action_context(
            round_num=sc["round_num"],
            total_rounds=sc["total_rounds"],
            opponent_message=sc["opp_msg"],
            my_message=sc["my_msg"],
            match_history=history,
            my_score=sc["my_score"],
            opp_score=sc["opp_score"],
            opponent_profile={"matches_played": 0},
            hypothesis=bayes_type,
            behavioral=bp,
            p_opp_c=bp.p_opp_cooperates(
                history[-1]["my_action"] if history else None,
                history[-1]["opp_action"] if history else None,
            ),
        )

        pass_count = 0
        for trial in range(1, n_trials + 1):
            agent_llm = Agent(client=client, system=system)
            response = agent_llm(ctx) or ""
            action = parse_action(response)
            correct = action == sc["expected_action"]
            if correct:
                pass_count += 1
            label = "cooperate" if action == 0 else "defect"
            mark = "✓" if correct else "✗"
            print(f"    Trial {trial}: {label} {mark}")

        passed = pass_count >= (n_trials * MIN_PASS_RATE)
        rate = pass_count / n_trials
        status = "PASS" if passed else "FAIL"
        print(f"  Result: {pass_count}/{n_trials} correct ({rate:.0%}) → {status}")
        if not passed:
            failures += 1

    return failures


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TournamentAgent benchmark suite")
    parser.add_argument("--eval",       action="store_true",
                        help="Run Tier 2 eval benchmarks (requires LLM calls)")
    parser.add_argument("--scenarios",  action="store_true",
                        help="Also run behavioral scenarios (Tier 2B, adds more LLM calls)")
    parser.add_argument("--n",          type=int, default=2,
                        help="Games per NPC strategy for oracle efficiency (default: 2)")
    parser.add_argument("--rounds",     type=int, default=10,
                        help="Rounds per game (default: 10)")
    args = parser.parse_args()

    total_failures = 0
    t_start = time.time()

    t1_failures = run_fast_benchmarks()
    total_failures += t1_failures
    t1_label = f"{t1_failures} failure(s)" if t1_failures else "all passed"
    print(f"\n  Tier 1 result: {t1_label}")

    if args.eval:
        efficiency_results = run_oracle_efficiency(n_games=args.n, rounds=args.rounds)
        t2a_failures = print_efficiency_table(efficiency_results)
        total_failures += t2a_failures
        t2a_label = f"{t2a_failures} strategy(ies) below threshold" if t2a_failures else "all above threshold"
        print(f"\n  Tier 2A result: {t2a_label}")

    if args.scenarios:
        t2b_failures = run_behavioral_scenarios()
        total_failures += t2b_failures
        t2b_label = f"{t2b_failures} scenario(s) failed" if t2b_failures else "all passed"
        print(f"\n  Tier 2B result: {t2b_label}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    if total_failures == 0:
        print("  All benchmarks passed.")
    else:
        print(f"  {total_failures} failure(s) total.")
    print()


if __name__ == "__main__":
    main()
