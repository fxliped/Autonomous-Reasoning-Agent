#!/usr/bin/env bash
#
# Launch your agent + all 4 test bots into one queue (or tournament) at once,
# instead of opening 5 terminals. Each process logs to its own file; Ctrl+C
# stops everything.
#
# Usage:
#   ./run_test_tournament.sh <QUEUE_ID>                 # queue mode (default)
#   ./run_test_tournament.sh --queue <QUEUE_ID>
#   ./run_test_tournament.sh --tournament <TOURNAMENT_ID>
#
set -uo pipefail
cd "$(dirname "$0")"

# Parse: a leading --flag means explicit mode; otherwise assume --queue.
if [[ "${1:-}" == --* ]]; then
  MODE="$1"; ID="${2:-}"
else
  MODE="--queue"; ID="${1:-}"
fi

if [[ -z "$ID" ]]; then
  echo "Usage: $0 [--queue|--tournament] <id>"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/testrun/$TS"
mkdir -p "$LOGDIR"

pids=()
launch() {
  local name="$1"; shift
  echo "  starting $name"
  "$@" > "$LOGDIR/$name.log" 2>&1 &
  pids+=($!)
}

cleanup() {
  echo
  echo "Stopping all ${#pids[@]} processes..."
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

echo "Launching agent + 4 bots on $MODE $ID"
echo "Logs -> $LOGDIR/"
echo
launch agent      python3 games/altruagent_runner.py "$MODE" "$ID" --verbose
launch defect_bot python3 games/scripted_bot.py --name sarthak_defect_bot --strategy always_defect "$MODE" "$ID" --verbose
launch tft_bot    python3 games/scripted_bot.py --name sarthak_tft_bot    --strategy tit_for_tat   "$MODE" "$ID" --verbose
launch grim_bot   python3 games/scripted_bot.py --name sarthak_grim_bot   --strategy grim          "$MODE" "$ID" --verbose
launch random_bot python3 games/scripted_bot.py --name sarthak_random_bot --strategy random        "$MODE" "$ID" --verbose

echo
echo "All 5 launched. Useful commands:"
echo "  Watch your agent:  tail -f $LOGDIR/agent.log"
echo "  Watch everything:  tail -f $LOGDIR/*.log"
echo "  Stop all:          press Ctrl+C here"
echo
echo "Waiting for all processes to finish (Ctrl+C to stop early)..."
wait
echo "All processes finished. Logs saved in $LOGDIR/"
