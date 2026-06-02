"""Scripted (no-LLM) bot for the AltruAgent platform — a test opponent.

Used to validate that our real SingleCallAgent plays correctly on the live
platform. Runs a fixed strategy (tit_for_tat / always_defect / always_cooperate
/ grim), joins the same queue as the real agent, and plays every child match.

Usage:
  python games/scripted_bot.py --name sarthak_tft_bot --strategy tit_for_tat --queue <queue_id>
  python games/scripted_bot.py --api-key sk_agent_... --strategy always_defect --queue auto

Credentials: looked up by --name in .bot_credentials.json, or pass --api-key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.altruagent_client import AltruAgentClient, AltruAgentError, normalize_game_server_url
from games.altruagent_runner import action_label, opponent_key, decision_to_index

BOT_CREDENTIALS = ROOT_DIR / ".bot_credentials.json"


# --- strategies: (my_moves, opp_moves) -> "cooperate"/"defect" --------------
def tit_for_tat(my_moves, opp_moves):
    return opp_moves[-1] if opp_moves else "cooperate"

def always_defect(my_moves, opp_moves):
    return "defect"

def always_cooperate(my_moves, opp_moves):
    return "cooperate"

def grim(my_moves, opp_moves):
    return "defect" if "defect" in opp_moves else "cooperate"

STRATEGIES = {
    "tit_for_tat": tit_for_tat,
    "always_defect": always_defect,
    "always_cooperate": always_cooperate,
    "grim": grim,
}

MESSAGES = {
    "tit_for_tat": "I cooperate and mirror your last move. Let's both do well.",
    "always_defect": "I play hard. Good luck.",
    "always_cooperate": "I always cooperate. Let's both win.",
    "grim": "I cooperate until crossed. Then never again.",
}


def resolve_api_key(name, api_key):
    if api_key:
        return api_key
    if name and BOT_CREDENTIALS.exists():
        creds = json.loads(BOT_CREDENTIALS.read_text())
        if name in creds:
            return creds[name]["api_key"]
    raise SystemExit("No api_key. Pass --api-key or --name present in .bot_credentials.json")


class ScriptedBot:
    def __init__(self, api_key, strategy, verbose=False):
        self.client = AltruAgentClient(api_key=api_key)
        self.client.login()
        me = self.client.me()
        if me.get("status") != "claimed":
            raise SystemExit(f"Bot not claimed (status={me.get('status')}). Claim it in the web UI first.")
        self.self_name = me["name"]
        self.self_id = me["id"]
        self.strategy_name = strategy
        self.strategy = STRATEGIES[strategy]
        self.message = MESSAGES.get(strategy, "Good game.")
        self.verbose = verbose
        print(f"[bot {self.self_name}] ready, strategy={strategy}")

    def _sequences(self, state):
        """Return (my_moves, opp_moves) from round_history, round-ordered."""
        my_moves, opp_moves = [], []
        for r in state.get("round_history") or []:
            actions = r.get("actions") or {}
            ok = opponent_key(actions, self.self_name)
            my_moves.append(action_label(actions.get(self.self_name, actions.get(self.self_name.lower(), 0))))
            opp_moves.append(action_label(actions.get(ok)) if ok else "cooperate")
        return my_moves, opp_moves

    def play_session(self, gsu, sid):
        gsu = normalize_game_server_url(gsu)
        print(f"  [bot {self.self_name}] playing {sid[:8]}")
        chatted, moved = set(), set()
        stalls = 0
        while True:
            try:
                state = self.client.game_state(gsu, sid)
            except AltruAgentError as exc:
                if exc.code in ("user_not_in_game", "session_not_found"):
                    return
                time.sleep(self.client.poll_interval); continue

            if state.get("is_terminal"):
                ret = state.get("returns") or {}
                print(f"  [bot {self.self_name}] done {sid[:8]} returns={ret}")
                return

            cur = state.get("current_round") or 1
            phase = state.get("phase")
            acted = False

            if phase == "messaging" and cur not in chatted:
                try:
                    self.client.send_message(gsu, sid, self.message)
                except AltruAgentError:
                    pass
                self.client.terminate_messaging(gsu, sid)
                chatted.add(cur)
                acted = True
            elif phase == "moving" and state.get("legal_actions") and cur not in moved:
                my_moves, opp_moves = self._sequences(state)
                decision = self.strategy(my_moves, opp_moves)
                idx = decision_to_index(decision, state.get("legal_actions") or [], state.get("legal_actions_str") or [])
                try:
                    self.client.step(gsu, sid, idx)
                    moved.add(cur)
                    if self.verbose:
                        print(f"    [bot {self.self_name}] R{cur} {decision}")
                    acted = True
                except AltruAgentError:
                    pass

            if acted:
                stalls = 0
                continue
            stalls += 1
            if stalls > 120:
                try: self.client.resign(gsu, sid)
                except AltruAgentError: pass
                return
            time.sleep(self.client.poll_interval)

    def run_queue(self, queue_id):
        if queue_id == "auto":
            qs = [q for q in self.client.list_queues() if q.get("is_active")]
            if not qs:
                raise SystemExit("No active queues.")
            queue_id = qs[0]["queue_id"]
        print(f"  [bot {self.self_name}] joining queue {queue_id[:8]}")
        join = self.client.join_queue(queue_id)
        tid = join.get("tournament_id")
        while True:
            q = self.client.get_queue(queue_id)
            if not q.get("current_waiting_tournament"):
                break
            time.sleep(self.client.poll_interval)
        self.run_tournament(tid)

    def run_tournament(self, tid):
        try:
            self.client.join_tournament(tid)
        except AltruAgentError as exc:
            if exc.code != "already_joined":
                pass
        played = set()
        while True:
            t = self.client.get_tournament(tid)
            status = t.get("tournament", {}).get("status")
            gsu = t.get("tournament", {}).get("game_server_url")
            viewer = t.get("viewer", {}) or {}
            if status == "completed":
                print(f"  [bot {self.self_name}] tournament complete")
                return
            for sid in (viewer.get("active_child_session_ids") or []):
                if sid not in played:
                    self.play_session(gsu, sid)
                    played.add(sid)
            time.sleep(self.client.poll_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Bot agent name (looks up api_key in .bot_credentials.json)")
    ap.add_argument("--api-key", help="Bot api_key (overrides --name lookup)")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--queue", help="Queue id or 'auto'")
    ap.add_argument("--tournament", help="Tournament id (instead of --queue)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    api_key = resolve_api_key(args.name, args.api_key)
    bot = ScriptedBot(api_key, args.strategy, verbose=args.verbose)
    if args.queue:
        bot.run_queue(args.queue)
    elif args.tournament:
        bot.run_tournament(args.tournament)
    else:
        raise SystemExit("Pass --queue or --tournament.")


if __name__ == "__main__":
    main()
