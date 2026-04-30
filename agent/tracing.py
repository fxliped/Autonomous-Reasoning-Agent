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
