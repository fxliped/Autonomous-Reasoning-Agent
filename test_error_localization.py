"""
Quick demo of the error localization pipeline firing on a bad round.

Simulates a round where:
  - LLM output is missing 'Thought:' (format failure)
  - Agent picks an unknown tool (tool_dispatch failure)
  - Agent is at 2 HP and goes full yolo: attack=10 defend=0 heal=0 (strategy failure)
  - Thought says "defend heavily" but defend=0 (consistency failure)
  - Agent takes 8 damage (bad outcome — amplifies blame scores)
"""

import time
from error_localization import (
    Tracer, Span, CheckResult, RoundOutcome,
    check_react_format,
    check_action_parses,
    check_tool_known,
    check_move_constraints,
    check_strategy,
    check_thought_action_consistency,
    attribute_round,
    format_round_report,
    format_cluster_report,
)


def fake_span(tracer: Tracer, kind: str, input=None, output=None,
              checks: list[CheckResult] = None) -> Span:
    """Helper: open and immediately close a span with preset checks."""
    with tracer.span(kind, input) as s:
        s.output = output
        for c in (checks or []):
            s.checks.append(c)
    return s


def run_bad_round(tracer: Tracer, round_num: int):
    tracer.set_round(round_num)
    tracer.tick_iter()

    # ── 1. LLM output missing "Thought:" ──────────────────────────────────
    bad_llm_output = "Action: nuke_opponent: now\nPAUSE"
    fake_span(tracer, "llm_call",
        input="[game context]",
        output=bad_llm_output,
        checks=[
            check_react_format(bad_llm_output),
            check_action_parses(bad_llm_output),
        ])

    # ── 2. Tool dispatch picks a hallucinated tool ─────────────────────────
    fake_span(tracer, "tool_dispatch",
        input="nuke_opponent",
        output="nuke_opponent",
        checks=[
            check_tool_known("nuke_opponent", ["get_game_state", "get_legal_moves", "make_move"]),
        ])

    tracer.tick_iter()

    # ── 3. Second LLM call — format is fine but move is yolo ──────────────
    good_llm_output = "Thought: I should defend heavily this round.\nAction: make_move: attack=10 defend=0 heal=0\nPAUSE"
    fake_span(tracer, "llm_call",
        input="[retry prompt]",
        output=good_llm_output,
        checks=[
            check_react_format(good_llm_output),
            check_action_parses(good_llm_output),
        ])

    # ── 4. make_move — strategy + consistency both fail ────────────────────
    attack, defend, heal = 10, 0, 0
    agent_hp, human_hp = 2, 8
    history = [
        {"agent_attack": 0, "human_attack": 8},
        {"agent_attack": 0, "human_attack": 7},
        {"agent_attack": 0, "human_attack": 9},
    ]
    fake_span(tracer, "tool:make_move",
        input="attack=10 defend=0 heal=0",
        output=f"attack={attack} defend={defend} heal={heal}",
        checks=[
            check_move_constraints(attack, defend, heal, energy=10),
            check_strategy(attack, defend, heal, agent_hp, human_hp, history),
            check_thought_action_consistency(good_llm_output, attack, defend, heal),
        ])


def run_good_round(tracer: Tracer, round_num: int):
    """A clean round for contrast in the cluster report."""
    tracer.set_round(round_num)
    tracer.tick_iter()
    llm_out = "Thought: I'll balance defense and attack.\nAction: make_move: attack=4 defend=4 heal=2\nPAUSE"
    fake_span(tracer, "llm_call", input="[game context]", output=llm_out,
        checks=[check_react_format(llm_out), check_action_parses(llm_out)])
    fake_span(tracer, "tool_dispatch", input="make_move", output="make_move",
        checks=[check_tool_known("make_move", ["get_game_state", "get_legal_moves", "make_move"])])
    fake_span(tracer, "tool:make_move", input="attack=4 defend=4 heal=2",
        checks=[
            check_move_constraints(4, 4, 2, 10),
            check_strategy(4, 4, 2, agent_hp=8, human_hp=6, recent_history=[]),
            check_thought_action_consistency(llm_out, 4, 4, 2),
        ])


if __name__ == "__main__":
    tracer = Tracer()

    # Round 1: good round — agent took 2 damage, no failures
    run_good_round(tracer, round_num=1)
    attribute_round(tracer.round_spans(1), RoundOutcome(
        round_num=1, damage_to_agent=2, damage_to_human=4,
        agent_hp_after=8, used_fallback=False,
        hit_max_iterations=False, hard_error=False,
    ))

    # Round 2: bad round — agent took 8 damage, multiple check failures
    run_bad_round(tracer, round_num=2)
    attribute_round(tracer.round_spans(2), RoundOutcome(
        round_num=2, damage_to_agent=8, damage_to_human=0,
        agent_hp_after=2, used_fallback=False,
        hit_max_iterations=False, hard_error=False,
    ))

    # Round 3: another bad round for cluster aggregation
    run_bad_round(tracer, round_num=3)
    attribute_round(tracer.round_spans(3), RoundOutcome(
        round_num=3, damage_to_agent=9, damage_to_human=0,
        agent_hp_after=0, used_fallback=False,
        hit_max_iterations=False, hard_error=True,
    ))

    print("\n" + "=" * 60)
    print("  PER-ROUND ERROR LOCALIZATION")
    print("=" * 60)
    for r in [1, 2, 3]:
        print(format_round_report(tracer, r))

    print(format_cluster_report(tracer))
