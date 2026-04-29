"""Offline CLI for judging a specific saved trace file."""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import DEFAULT_MODEL, create_gemini_client
from agent.tracing import judge_trace, write_judge_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM judge on one Tracer-style game trace JSON file."
    )
    parser.add_argument(
        "trace_path",
        help="Path to a trace JSON file, e.g. traces/hunger_games_20260429_130541_b2f946f9.json",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use for judging. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Do not append the judge reflection to agent/reflections/<game>.md.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write the judge result back into the trace JSON.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    client = create_gemini_client()
    result = judge_trace(
        trace_path,
        client=client,
        model=args.model,
        append_to_reflections=not args.no_reflection,
    )

    if not args.no_write:
        write_judge_result(trace_path, result)

    print(json.dumps(result, indent=2))
    if not args.no_write:
        print(f"\nJudge result written to: {trace_path}")
    if not args.no_reflection and result.get("reflection"):
        print("Reflection memory updated.")


if __name__ == "__main__":
    main()
