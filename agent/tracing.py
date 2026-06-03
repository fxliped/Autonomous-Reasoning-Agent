"""Tracer-style logging and judging for ReAct game agents."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.agent import ROOT_DIR, append_reflection


TRACES_DIR = ROOT_DIR / "traces"


def make_json_safe(value: Any) -> Any:
    """Convert common Python values into JSON-safe data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


def snapshot_game_state(game: Any) -> dict[str, Any]:
    """Capture simple public game attributes for trace context."""
    state = {}
    for key, value in vars(game).items():
        if key.startswith("_"):
            continue
        state[key] = make_json_safe(value)
    return state


def parse_react_response(response: str | None) -> dict[str, Any]:
    """Extract Thought/Action/Decision fields from a ReAct response."""
    text = response or ""
    thought = ""
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\n\s*Action:|\n\s*Decision:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    action = None
    argument = None
    action_match = re.search(
        r"Action:\s*([a-zA-Z_][\w]*)\s*:\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if action_match:
        action = action_match.group(1).strip()
        argument = action_match.group(2).strip()

    decision = None
    decision_match = re.search(r"Decision:\s*(.+)", text, re.IGNORECASE)
    if decision_match:
        decision = decision_match.group(1).strip()

    parse_error = None
    if "Action" in text and not action_match:
        parse_error = "Could not parse Action. Expected: Action: tool_name: argument"

    return {
        "thought": thought,
        "action": action,
        "argument": argument,
        "has_pause": "PAUSE" in text,
        "decision": decision,
        "parse_error": parse_error,
    }


class TraceLogger:
    """Collect and persist one game run as a Tracer-style JSON trace."""

    def __init__(
        self,
        game_name: str,
        run_id: str | None = None,
        trace_dir: str | Path = TRACES_DIR,
    ):
        self.game_name = game_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_dir = Path(trace_dir)
        self.path = None
        self.data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "game_name": game_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "steps": [],
            "round_results": [],
            "final_result": None,
            "judge": None,
        }

    def start_round(self, round_number: int, state_before: dict[str, Any]) -> None:
        self.data["steps"].append({
            "event": "start_round",
            "round": round_number,
            "state_before": make_json_safe(state_before),
        })

    def record_step(
        self,
        round_number: int,
        step_number: int,
        prompt: str,
        response: str,
        parsed: dict[str, Any],
        observation: str | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
    ) -> None:
        self.data["steps"].append({
            "event": "react_step",
            "round": round_number,
            "step": step_number,
            "prompt": prompt,
            "response": response,
            "parsed": make_json_safe(parsed),
            "observation": observation,
            "state_before": make_json_safe(state_before or {}),
            "state_after": make_json_safe(state_after or {}),
        })

    def record_round_result(self, round_number: int, result: Any) -> None:
        self.data["round_results"].append({
            "round": round_number,
            "result": make_json_safe(result),
        })

    def finish(self, final_result: dict[str, Any]) -> None:
        self.data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.data["final_result"] = make_json_safe(final_result)

    def attach_judge(self, judge_result: dict[str, Any]) -> None:
        self.data["judge"] = make_json_safe(judge_result)

    def save(self) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self.path is None:
            self.path = self.trace_dir / f"{self.game_name}_{self.run_id}_{uuid.uuid4().hex[:8]}.json"
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return self.path


def judge_trace(
    trace_path: str | Path,
    client,
    model: str | None = None,
    append_to_reflections: bool = True,
) -> dict[str, Any]:
    """Ask the LLM client to localize reasoning failures in a trace."""
    path = Path(trace_path)
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace_text = json.dumps(trace, indent=2)

    prompt = f"""
You are a Tracer-style reasoning error localizer for a game-playing ReAct agent.
Analyze the JSON trace and identify the earliest important reasoning failures.

Return ONLY valid JSON with this shape:
{{
  "failures": [
    {{
      "round": 1,
      "step": 1,
      "category": "format_error|illegal_move|state_error|opponent_model_error|strategy_error|goal_error|tool_use_error|none",
      "explanation": "short explanation",
      "suggested_fix": "short actionable fix"
    }}
  ],
  "reflection": "one concise lesson to help this game next time"
}}

Trace:
{trace_text}
""".strip()

    system = "Return strict JSON. Do not include markdown."
    raw = client.complete(system, [{"role": "user", "content": prompt}], model).strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "failures": [],
            "reflection": "",
            "raw_response": raw,
            "parse_error": "Judge did not return valid JSON.",
        }

    reflection = result.get("reflection", "")
    if append_to_reflections and reflection:
        append_reflection(trace["game_name"], reflection, source_file=path.name)

    return result


def write_judge_result(trace_path: str | Path, judge_result: dict[str, Any]) -> Path:
    """Write a judge result into an existing trace JSON file."""
    path = Path(trace_path)
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace["judge"] = make_json_safe(judge_result)
    trace["judged_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path


def judge_match_rounds(
    match_rounds: list[dict],
    my_avg_score: float,
    opp_avg_score: float,
    opponent_id: str,
    classified_type: str,
    type_confidence: float,
    deception_count: int,
    client,
    run_id: str | None = None,
    append_to_reflections: bool = True,
) -> dict[str, Any]:
    """
    Judge a completed live tournament match and optionally append a reflection.

    Unlike judge_trace() which analyzes ReAct step reasoning, this analyzes
    tournament round outcomes: opponent classification accuracy, deception
    effectiveness, arc phase adherence, and missed scoring opportunities.
    Reflections written here feed every future tournament match's system prompt.
    """
    rounds_str = "\n".join(
        f"  R{r['round']}: I={r.get('my_action','?')} opp={r.get('opp_action','?') or '?'} | "
        f"pts me{'+' if (r.get('my_pts') or 0) >= 0 else ''}{r.get('my_pts','?')} "
        f"them{'+' if (r.get('opp_pts') or 0) >= 0 else ''}{r.get('opp_pts','?')} | "
        f"they said: \"{r.get('opp_msg','')}\" | I said: \"{r.get('my_msg','')}\""
        for r in match_rounds
    )

    outcome = (
        "WIN" if my_avg_score > opp_avg_score
        else "LOSS" if my_avg_score < opp_avg_score
        else "DRAW"
    )
    total = len(match_rounds)

    prompt = f"""You are a tournament strategy coach for a Prisoner's Dilemma agent competing against LLM-based opponents on a live leaderboard.

MATCH SUMMARY
Opponent: {opponent_id}
Classified as: {classified_type} (confidence: {type_confidence:.0%})
Outcome: {outcome} | Avg score: me {my_avg_score:.3f} | them {opp_avg_score:.3f}
Deception used: {deception_count}/{total} rounds (cooperative message sent but defected)

PAYOFFS: C/C=+2/+2 | C/D=-1/+5 | D/C=+5/-1 | D/D=0/0

ROUND-BY-ROUND
{rounds_str}

Analyze from a competitive tournament perspective. Be specific — cite round numbers.

1. OPPONENT CLASSIFICATION: Was "{classified_type}" correct? When did the signal become unambiguous?
2. DECEPTION: Did sending cooperative messages while defecting help or backfire? Which rounds?
3. ARC PHASES: Did we probe R1, classify by R2-3, build cooperative narrative mid-game, harvest/defect late?
4. MISSED POINTS: Which specific rounds left points on the table and why?
5. SELF-PLAY PATTERN: What does our own action sequence reveal about exploitable habits in our style?

Return ONLY valid JSON — no markdown fences:
{{
  "classification_accurate": true/false,
  "classification_note": "one sentence — when the type was confirmed or misread",
  "deception_verdict": "effective|backfired|unused|neutral",
  "deception_note": "one sentence — which round and why it helped or hurt",
  "key_mistake": "the single most costly decision this match",
  "self_pattern": "one sentence — what our action sequence reveals about our own exploitable habits",
  "reflection": "2-3 sentences of sharp, actionable lessons for future tournament matches. Cover: what to do against this opponent type, when deception is worth the risk, and what arc phase adjustment to make. Be specific enough that the agent can act on it immediately."
}}""".strip()

    system = "You are a competitive game theory coach. Return strict JSON only."
    raw = client.complete(system, [{"role": "user", "content": prompt}]).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        hit = re.search(r"\{.*\}", raw, re.DOTALL)
        if hit:
            try:
                result = json.loads(hit.group())
            except json.JSONDecodeError:
                result = {"reflection": "", "raw_response": raw, "parse_error": "invalid JSON"}
        else:
            result = {"reflection": "", "raw_response": raw, "parse_error": "invalid JSON"}

    reflection = result.get("reflection", "")
    if append_to_reflections and reflection:
        rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        source = f"tournament_{opponent_id}_{rid}"
        append_reflection("prisoners_dilemma", reflection, source_file=source)
        print(f"[Judge] Reflection appended for match vs '{opponent_id}' ({outcome}).")

    return result
