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
import random
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

def random_5050(my_moves, opp_moves):
    """Unconditional coin flip each round — 50% cooperate, 50% defect."""
    return "defect" if random.random() < 0.5 else "cooperate"

STRATEGIES = {
    "tit_for_tat": tit_for_tat,
    "always_defect": always_defect,
    "always_cooperate": always_cooperate,
    "grim": grim,
    "random": random_5050,
}

MESSAGES = {
    "tit_for_tat": "I cooperate and mirror your last move. Let's both do well.",
    "always_defect": "I play hard. Good luck.",
    "always_cooperate": "I always cooperate. Let's both win.",
    "grim": "I cooperate until crossed. Then never again.",
    "random": "I keep things unpredictable. Good luck reading me.",
}

# Poll fast while inside a match so we never miss the ~30s move window (the
# platform auto-COOPERATES on a timed-out move, which would silently corrupt a
# defect bot's strategy). The slower client.poll_interval is fine for the
# queue/tournament waiting loops.
FAST_POLL = 1.5
STEP_RETRIES = 4  # quick retries if a step call transiently fails


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

    def _submit_move(self, gsu, sid, idx, decision, cur):
        """Submit a move, retrying fast on transient errors. Returns True on success.

        Replaces the old silent `except: pass` — a swallowed step error used to
        let the round time out and auto-cooperate without anyone noticing. Now we
        retry quickly and shout if the move never lands.
        """
        for attempt in range(STEP_RETRIES):
            try:
                self.client.step(gsu, sid, idx)
                if self.verbose:
                    print(f"    [bot {self.self_name}] R{cur} {decision} (action={idx})")
                return True
            except AltruAgentError as exc:
                # Round already resolved / not our turn — retrying won't help.
                if exc.code in ("game_already_finished", "already_moved", "not_your_turn"):
                    return False
                print(f"    [bot {self.self_name}] R{cur} step failed ({exc.code}) "
                      f"— retry {attempt + 1}/{STEP_RETRIES}")
                time.sleep(0.5)
        print(f"  [bot {self.self_name}] !! R{cur} FAILED to submit {decision!r} after "
              f"{STEP_RETRIES} tries — platform may auto-cooperate this round")
        return False

    def play_session(self, gsu, sid):
        gsu = normalize_game_server_url(gsu)
        print(f"  [bot {self.self_name}] playing {sid[:8]}")
        chatted, moved = set(), set()
        last_decision: dict[int, str] = {}
        last_history_len = 0
        stalls = 0
        while True:
            try:
                state = self.client.game_state(gsu, sid)
            except AltruAgentError as exc:
                if exc.code in ("user_not_in_game", "session_not_found"):
                    return
                time.sleep(FAST_POLL); continue

            # Audit completed rounds: flag any round the platform resolved with a
            # move we did NOT submit (a missed window → silent auto-default), or
            # where the recorded action differs from what our strategy chose.
            hist = state.get("round_history") or []
            if len(hist) > last_history_len:
                for r in hist[last_history_len:]:
                    rn = r.get("round")
                    acts = r.get("actions") or {}
                    mine = action_label(acts.get(self.self_name, acts.get(self.self_name.lower(), 0)))
                    if rn not in moved:
                        print(f"  [bot {self.self_name}] !! MISSED MOVE R{rn}: platform recorded "
                              f"{mine!r} for me (auto-default, NOT my strategy)")
                    elif rn in last_decision and mine != last_decision[rn]:
                        print(f"  [bot {self.self_name}] !! R{rn} MISMATCH: recorded {mine!r} but "
                              f"I submitted {last_decision[rn]!r}")
                last_history_len = len(hist)

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
                if self._submit_move(gsu, sid, idx, decision, cur):
                    moved.add(cur)
                    last_decision[cur] = decision
                    acted = True

            if acted:
                stalls = 0
                continue
            stalls += 1
            if stalls > 400:  # ~10 min at FAST_POLL
                try: self.client.resign(gsu, sid)
                except AltruAgentError: pass
                return
            time.sleep(FAST_POLL)

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
