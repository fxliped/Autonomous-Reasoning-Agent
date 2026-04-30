"""
=============================================================================
Error Localization for the Hunger Games ReAct Agent
=============================================================================

WHAT THIS DOES
--------------
The agent in hungergames_V2.py runs a ReAct loop each round:
    Thought -> Action -> Observation -> ... -> make_move

Each of those steps can fail in a different way. When the round goes badly
(agent took heavy damage, hit max_iterations, made an illegal move, etc.)
this module tells you WHICH step is to blame instead of just "the round
went badly."

HOW IT WORKS
------------
1. TRACE       — Tracer records every ReAct step as a Span with kind, input,
                 output, and timing.
2. SCORE       — Per-step critics ("checks") flag local issues:
                   - format_check    : did the LLM output Thought:/Action:?
                   - tool_check      : was the chosen tool valid?
                   - parse_check     : did make_move parse cleanly?
                   - constraint_check: does it sum to energy, non-negative?
                   - strategy_check  : is the move reasonable given HP/history?
                   - consistency_chk : does the move match the Thought?
3. ATTRIBUTE   — When the round resolves, RoundOutcome (damage taken, win/loss,
                 fallback used) is back-propagated. Each span gets a blame_score
                 combining its own check failures with proximity to bad outcome.
4. CLUSTER     — FailureClusters groups spans by (kind, top failing check) so
                 across many rounds you see "12 spans failed strategy_check on
                 round-3+" rather than 12 unrelated complaints.
5. REPORT      — Pretty-prints a per-round localized failure report and a
                 game-level cluster summary.

WHY NOT JUST LOG
----------------
A log gives you a transcript. This gives you a ranked list of suspect steps.
That's the difference between "the agent is bad" and "the agent's strategy
step is making suicidal all-in attacks when it has <=3 HP, 4 times out of 5."
=============================================================================
"""

from __future__ import annotations
import time
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional
from collections import defaultdict


# =============================================================================
# SECTION 1: SPAN — one step in the ReAct loop
# =============================================================================
# A Span is the atomic unit of trace. Every iteration of the agent loop
# produces one span (or a small sub-tree), and every span can carry check
# results that later get aggregated into a blame score.
# =============================================================================

@dataclass
class CheckResult:
    """One per-step check outcome. severity in [0.0, 1.0]."""
    name: str
    passed: bool
    severity: float          # 0 = harmless, 1 = catastrophic
    detail: str = ""

    def __repr__(self):
        mark = "✓" if self.passed else "✗"
        return f"{mark} {self.name} (sev={self.severity:.2f}): {self.detail}"


@dataclass
class Span:
    """
    A single traced step. `kind` is what the step was trying to do —
    that's the dimension we cluster on.

    Kinds used in this agent:
        llm_call      : raw Gemini call returning Thought/Action text
        tool_dispatch : choosing which tool to invoke based on parsed Action
        tool:get_game_state, tool:get_legal_moves, tool:make_move : tool execs
        fallback      : the safety-net default allocation when agent failed
    """
    span_id: int
    round_num: int
    iteration: int                       # which iteration of the ReAct loop
    kind: str
    input: Any = None
    output: Any = None
    started_at: float = 0.0
    ended_at: float = 0.0
    checks: list[CheckResult] = field(default_factory=list)
    blame_score: float = 0.0             # filled in during attribution

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def max_severity(self) -> float:
        return max((c.severity for c in self.failed_checks), default=0.0)


# =============================================================================
# SECTION 2: TRACER — context-managed span recorder
# =============================================================================
# Used as a context manager so spans are guaranteed to close even if the
# code inside raises. This is the standard OpenTelemetry-style pattern.
# =============================================================================

class Tracer:
    def __init__(self):
        self.spans: list[Span] = []
        self._next_id = 0
        self._current_round = 1
        self._current_iter = 0

    def set_round(self, round_num: int):
        self._current_round = round_num
        self._current_iter = 0

    def tick_iter(self):
        self._current_iter += 1

    def span(self, kind: str, input: Any = None) -> "_SpanCtx":
        return _SpanCtx(self, kind, input)

    def round_spans(self, round_num: int) -> list[Span]:
        return [s for s in self.spans if s.round_num == round_num]


class _SpanCtx:
    def __init__(self, tracer: Tracer, kind: str, input: Any):
        self.tracer = tracer
        self.span = Span(
            span_id=tracer._next_id,
            round_num=tracer._current_round,
            iteration=tracer._current_iter,
            kind=kind,
            input=input,
        )
        tracer._next_id += 1

    def __enter__(self) -> Span:
        self.span.started_at = time.time()
        self.tracer.spans.append(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.span.ended_at = time.time()
        if exc is not None:
            # An exception inside a span is itself a strong signal.
            self.span.checks.append(CheckResult(
                name="no_exception",
                passed=False,
                severity=1.0,
                detail=f"{exc_type.__name__}: {exc}",
            ))
        return False  # don't swallow exceptions


# =============================================================================
# SECTION 3: CHECKS — per-step critics
# =============================================================================
# Each check is a pure function: (span, context) -> CheckResult.
# Keep them small and orthogonal. New failure mode? Add a new check.
# =============================================================================

def check_react_format(llm_text: str) -> CheckResult:
    """The system prompt requires 'Thought:' before any Action."""
    has_thought = "Thought:" in llm_text
    has_action_or_decision = "Action:" in llm_text or "Decision" in llm_text
    if has_thought and has_action_or_decision:
        return CheckResult("format", True, 0.0, "Thought + Action/Decision present")
    if not has_thought:
        return CheckResult("format", False, 0.6, "Missing 'Thought:' prefix")
    return CheckResult("format", False, 0.4, "Missing Action and Decision")


def check_action_parses(llm_text: str) -> CheckResult:
    """Action: <tool>: <arg> must regex-match if PAUSE is present."""
    if "PAUSE" not in llm_text:
        return CheckResult("action_parse", True, 0.0, "no Action emitted yet")
    m = re.findall(r'Action: ([a-z_]+): (.+)', llm_text, re.IGNORECASE)
    if not m:
        return CheckResult("action_parse", False, 0.7, "PAUSE present but Action regex failed")
    return CheckResult("action_parse", True, 0.0, f"parsed tool={m[0][0]}")


def check_tool_known(tool_name: str, allowed: list[str]) -> CheckResult:
    """Did the LLM hallucinate a tool name?"""
    if tool_name in allowed:
        return CheckResult("tool_known", True, 0.0, tool_name)
    return CheckResult("tool_known", False, 0.7,
                      f"unknown tool '{tool_name}', allowed={allowed}")


def check_move_constraints(attack: int, defend: int, heal: int, energy: int) -> CheckResult:
    """Sums to energy budget? All non-negative?"""
    s = attack + defend + heal
    if s != energy:
        return CheckResult("constraint", False, 0.9,
                           f"sum={s} expected {energy}")
    if any(v < 0 for v in (attack, defend, heal)):
        return CheckResult("constraint", False, 1.0, "negative value")
    return CheckResult("constraint", True, 0.0, f"{attack}/{defend}/{heal}")


def check_strategy(attack: int, defend: int, heal: int,
                   agent_hp: int, human_hp: int,
                   recent_history: list[dict]) -> CheckResult:
    """
    Domain-specific sanity rules for Hunger Games.

    These aren't "is this the optimal move" — that's too hard to judge
    statically. They flag moves that are obviously suspect:

      - 0 defend when low HP and human has been attacking heavily
      - all-in attack while at 1 HP (the 'desperate yolo' antipattern)
      - heavy heal when already at full HP (wasted energy)
      - 0 attack for 3+ consecutive rounds (passive drift)
    """
    issues = []
    severity = 0.0

    # Wasted heal at full HP — only flag substantial waste, not small hedges
    if heal >= 4 and agent_hp >= 10:
        issues.append(f"heal={heal} at full HP wastes a lot of energy")
        severity = max(severity, 0.35 + 0.05 * (heal - 4))

    # Suicidal yolo
    if agent_hp <= 2 and defend == 0 and heal == 0:
        issues.append(f"HP={agent_hp} but defend=0 heal=0 (yolo)")
        severity = max(severity, 0.85)

    # Underdefend against an aggressive human
    if recent_history:
        avg_human_atk = sum(h['human_attack'] for h in recent_history[-3:]) / min(3, len(recent_history))
        if avg_human_atk >= 5 and defend < avg_human_atk - 1 and agent_hp <= 5:
            issues.append(
                f"defend={defend} but human's avg attack is {avg_human_atk:.1f} and HP={agent_hp}")
            severity = max(severity, 0.6)

    # Passive drift
    if recent_history and len(recent_history) >= 3:
        last3_atk = [h['agent_attack'] for h in recent_history[-3:]]
        if all(a == 0 for a in last3_atk) and attack == 0:
            issues.append("4 consecutive rounds of attack=0 (passive drift)")
            severity = max(severity, 0.5)

    if issues:
        return CheckResult("strategy", False, severity, "; ".join(issues))
    return CheckResult("strategy", True, 0.0, "no obvious antipattern")


def check_thought_action_consistency(llm_text: str,
                                     attack: int, defend: int, heal: int) -> CheckResult:
    """
    Cheap heuristic: scan the Thought for keywords like 'defend heavily',
    'go all in attack', 'heal up', and check the move actually reflects them.
    Imperfect, but catches the most blatant Thought/Action contradictions.
    """
    thought_match = re.search(r'Thought:(.*?)(?=Action:|Decision|$)', llm_text, re.DOTALL)
    if not thought_match:
        return CheckResult("consistency", True, 0.0, "no thought to compare against")
    thought = thought_match.group(1).lower()

    contradictions = []
    if ("defend heavily" in thought or "prioritize defense" in thought or "heavy defense" in thought) and defend < 4:
        contradictions.append(f"thought says defend heavily but defend={defend}")
    if ("all in" in thought or "all-in" in thought) and "attack" in thought and attack < 7:
        contradictions.append(f"thought mentions all-in attack but attack={attack}")
    if ("heal up" in thought or "prioritize heal" in thought) and heal < 3:
        contradictions.append(f"thought mentions heal priority but heal={heal}")

    if contradictions:
        return CheckResult("consistency", False, 0.55, "; ".join(contradictions))
    return CheckResult("consistency", True, 0.0, "no contradiction detected")


# =============================================================================
# SECTION 4: ATTRIBUTION — back-propagating round outcome onto spans
# =============================================================================
# After a round resolves we know:
#   - damage_to_agent       (continuous outcome signal)
#   - whether agent hit max_iterations and used the fallback
#   - whether a hard error occurred (illegal move, exception)
#
# We turn that into per-span blame_score = local_severity + outcome_factor.
# The outcome_factor is shared across spans in the round but weighted by
# kind: make_move and strategy spans absorb most of it; pure observation
# tool calls absorb very little.
# =============================================================================

@dataclass
class RoundOutcome:
    round_num: int
    damage_to_agent: int
    damage_to_human: int
    agent_hp_after: int
    used_fallback: bool
    hit_max_iterations: bool
    hard_error: bool


# How much of the round's outcome blame each span kind absorbs.
# Tuned so causal steps (move selection, strategy) get more than
# read-only observations (game_state lookups).
KIND_OUTCOME_WEIGHT = {
    "tool:make_move":     1.0,
    "llm_call":           0.6,
    "tool_dispatch":      0.4,
    "fallback":           1.0,
    "tool:get_game_state": 0.1,
    "tool:get_legal_moves": 0.1,
}


def attribute_round(spans: list[Span], outcome: RoundOutcome) -> None:
    """Mutates spans in place to set blame_score."""
    # Outcome severity in [0, 1]
    # Heavy damage (>=5) or fallback or hard error all push toward 1.0.
    outcome_factor = 0.0
    if outcome.hard_error:
        outcome_factor = max(outcome_factor, 1.0)
    if outcome.used_fallback:
        outcome_factor = max(outcome_factor, 0.85)
    if outcome.hit_max_iterations:
        outcome_factor = max(outcome_factor, 0.7)
    # Damage scales: 0 dmg -> 0, 10 dmg -> 1.0
    outcome_factor = max(outcome_factor, min(1.0, outcome.damage_to_agent / 10.0))

    for s in spans:
        if s.round_num != outcome.round_num:
            continue
        local = s.max_severity
        weight = KIND_OUTCOME_WEIGHT.get(s.kind, 0.3)
        s.blame_score = round(local * 0.6 + outcome_factor * weight * 0.4, 3)


# =============================================================================
# SECTION 5: CLUSTERING — across-round failure aggregation
# =============================================================================

@dataclass
class ClusterEntry:
    signature: str           # e.g. "tool:make_move / strategy"
    count: int = 0
    total_blame: float = 0.0
    examples: list[Span] = field(default_factory=list)


def cluster_failures(all_spans: list[Span], top_k_examples: int = 2) -> list[ClusterEntry]:
    """
    Group spans by (kind, top failing check name). Each group becomes a
    cluster. The output is sorted by total_blame descending — the first
    cluster is your single biggest source of agent failure.
    """
    by_sig: dict[str, ClusterEntry] = {}
    for s in all_spans:
        if not s.failed_checks and s.blame_score < 0.2:
            continue
        top_check = max(s.failed_checks, key=lambda c: c.severity, default=None)
        sig = f"{s.kind} / {top_check.name if top_check else 'outcome_only'}"
        entry = by_sig.setdefault(sig, ClusterEntry(signature=sig))
        entry.count += 1
        entry.total_blame += s.blame_score
        if len(entry.examples) < top_k_examples:
            entry.examples.append(s)
    return sorted(by_sig.values(), key=lambda e: e.total_blame, reverse=True)


# =============================================================================
# SECTION 6: REPORTING
# =============================================================================

def format_round_report(tracer: Tracer, round_num: int) -> str:
    """Per-round localized report: top suspects in this round."""
    spans = sorted(
        tracer.round_spans(round_num),
        key=lambda s: s.blame_score,
        reverse=True,
    )
    if not spans:
        return f"  [round {round_num}] no spans"

    lines = [f"  ┌─ ERROR LOCALIZATION — Round {round_num} " + "─" * 18]
    for s in spans[:5]:
        marker = "🔴" if s.blame_score >= 0.5 else ("🟡" if s.blame_score >= 0.2 else "🟢")
        lines.append(f"  │ {marker} blame={s.blame_score:>5.2f}  iter={s.iteration}  kind={s.kind}")
        for c in s.failed_checks:
            lines.append(f"  │     ✗ {c.name}: {c.detail}")
    lines.append("  └" + "─" * 50)
    return "\n".join(lines)


def format_cluster_report(tracer: Tracer) -> str:
    """End-of-game summary: failure clusters across all rounds."""
    clusters = cluster_failures(tracer.spans)
    if not clusters:
        return "\n  ✅ No localized failures across the match.\n"

    lines = ["", "  ╔═ FAILURE CLUSTERS (across the full match) " + "═" * 8 + "╗"]
    for c in clusters:
        lines.append(f"  ║ [{c.count:>2}x] total_blame={c.total_blame:>5.2f}  {c.signature}")
        for ex in c.examples:
            top = max(ex.failed_checks, key=lambda x: x.severity, default=None)
            detail = top.detail if top else "(outcome-only blame)"
            lines.append(f"  ║       e.g. round {ex.round_num} iter {ex.iteration}: {detail}")
    lines.append("  ╚" + "═" * 52 + "╝")
    return "\n".join(lines)


def export_trace_jsonl(tracer: Tracer, path: str) -> None:
    """Dump every span as one JSON line for offline analysis / replay."""
    with open(path, "w") as f:
        for s in tracer.spans:
            d = asdict(s)
            d["duration_ms"] = s.duration_ms
            f.write(json.dumps(d, default=str) + "\n")
