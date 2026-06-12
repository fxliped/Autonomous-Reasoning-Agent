"""LIVE variant of the fast runner — tighter move-window guard (config only).

Why this exists: the fast runner (games/altruagent_runner_fast.py) reuses the
message-phase action as the move, which is great, but the inherited play loop
polls game state every 5s. On a real match the ~30s move window can open and
close *inside one 5s sleep* (especially right after a ~15-20s message-phase LLM
call), so we never see phase=="moving", never submit, and the platform
auto-cooperates us into a -1 sucker. That is exactly what cost trace 572810e6
two rounds (R2 and R8: planned=defect, but no move landed -> auto-cooperate).

This variant changes ONLY THREE CLIENT CONFIG NUMBERS — no logic change:
  - poll_interval 5s -> 1.5s  : catch phase=="moving" within ~1.5s of it opening
  - retry_backoff 0.75 -> 0.5 : cap the 429 backoff so a retry burst can never
  - max_retries_429 4 -> 3      sleep through a move window (worst case ~3.5s, not ~11s)

It is a thin SUBCLASS of FastRunner: reuse-the-move, the 429 resilience,
_safe_terminate, the threat signal and opponent memory are all INHERITED
unchanged. The fast runner and the standard runner are left completely
untouched — if this misbehaves, just run altruagent_runner_fast.py (or the
standard altruagent_runner.py); nothing needs to be reverted.

Usage (identical to the fast runner):
  python games/altruagent_runner_fast_live.py --tournament <id> --verbose
  python games/altruagent_runner_fast_live.py --queue <queue_id> --verbose
  python games/altruagent_runner_fast_live.py --list
  add --memory <path> to isolate opponent memory while testing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from games.altruagent_runner import _list_all
from games.altruagent_runner_fast import FastRunner, FAST_MODEL

# Tighter move-window guard. These are the ONLY differences from the fast runner.
LIVE_POLL_INTERVAL = 1.5   # was 5.0 — close the blind gap between polls
LIVE_RETRY_BACKOFF = 0.5   # was 0.75 — cap 429 backoff so it can't eat a window
LIVE_MAX_RETRIES_429 = 3   # was 4    — worst-case ~0.5+1.0+2.0s, not ~11s


class LiveFastRunner(FastRunner):
    """FastRunner with a tighter poll cadence so move windows are never missed."""

    def __init__(self, verbose=False, memory_path=None, judge=False, model=FAST_MODEL):
        super().__init__(verbose=verbose, memory_path=memory_path, judge=judge, model=model)
        # The whole point of this file: poll fast enough to always catch the
        # move window, and keep the 429 backoff small enough that a retry burst
        # can never sleep through one. Inherited logic is otherwise identical.
        self.client.poll_interval = LIVE_POLL_INTERVAL
        self.client.retry_backoff = LIVE_RETRY_BACKOFF
        self.client.max_retries_429 = LIVE_MAX_RETRIES_429
        print(f"  [live fast runner] tight move-window guard: "
              f"poll={self.client.poll_interval}s, "
              f"429 backoff={self.client.retry_backoff}s x{self.client.max_retries_429}")


def main():
    ap = argparse.ArgumentParser(
        description="Run SingleCallAgent on AltruAgent (LIVE fast variant: tight move-window guard)."
    )
    ap.add_argument("--list", action="store_true", help="List queues/tournaments/competitions and exit")
    ap.add_argument("--queue", help="Queue id to join, or 'auto' for the first active queue")
    ap.add_argument("--tournament", help="Tournament id to join and play")
    ap.add_argument("--session", help="Single competition session id to join and play")
    ap.add_argument("--verbose", action="store_true", help="Dump raw state/messages each poll")
    ap.add_argument("--memory", help="Path to opponent memory JSON (use a separate path to isolate test runs)")
    ap.add_argument("--judge", action="store_true", help="Run the LLM trace-judge after each match")
    ap.add_argument("--model", default=FAST_MODEL, help=f"Gemini model to use (default {FAST_MODEL})")
    args = ap.parse_args()

    if args.list:
        _list_all()
        return

    if not (args.queue or args.tournament or args.session):
        print("Nothing to do. Use --list, --queue auto, --tournament <id>, or --session <id>.")
        return

    runner = LiveFastRunner(verbose=args.verbose, memory_path=args.memory, judge=args.judge, model=args.model)
    if args.queue:
        runner.run_queue(args.queue)
    elif args.tournament:
        runner.run_tournament(args.tournament)
    elif args.session:
        runner.run_competition(args.session)


if __name__ == "__main__":
    main()
