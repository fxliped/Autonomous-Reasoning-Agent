"""Run the SingleCallAgent on the AltruAgent competition platform.

Bridges three things:
  - agent.altruagent_client.AltruAgentClient  (HTTP: control + data plane)
  - agent.single_call_agent.SingleCallAgent   (LLM decision: message + action)
  - agent.opponent_memory.OpponentMemory      (cross-tournament classification)

Entry modes:
  python games/altruagent_runner.py --list
  python games/altruagent_runner.py --queue auto        # join first active queue
  python games/altruagent_runner.py --queue <queue_id>
  python games/altruagent_runner.py --tournament <id>
  python games/altruagent_runner.py --session <id>      # single competition
  add --verbose to dump raw game state / messages (use for the first live game)

repeated_pd facts the loop encodes (confirmed live, 2026-06):
  action 0 = Cooperate, 1 = Defect. 8 rounds. blind messaging, per_all_moves.
  1 chat per messaging phase, <=50 words. 30s inactivity timeout (auto-cooperate).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime

from agent.agent import create_client
from agent.single_call_agent import SingleCallAgent
from agent.altruagent_client import AltruAgentClient, AltruAgentError, normalize_game_server_url
from agent.opponent_memory import OpponentMemory
from agent.tracing import TraceLogger, make_json_safe

# repeated_pd payoff matrix (you, opponent), keyed by (your_move, opp_move).
PAYOFFS = {
    ("cooperate", "cooperate"): (2, 2),
    ("cooperate", "defect"): (-1, 5),
    ("defect", "cooperate"): (5, -1),
    ("defect", "defect"): (0, 0),
}


# ---------------------------------------------------------------------------
# State interpretation helpers
# ---------------------------------------------------------------------------
def action_label(value) -> str:
    """Normalize a platform action (0/1 or 'Cooperate'/'Defect') to our string."""
    if isinstance(value, (int, float)):
        return "cooperate" if int(value) == 0 else "defect"
    s = str(value).strip().lower()
    if s in ("cooperate", "defect"):
        return s
    return "cooperate" if s in ("0", "c") else "defect"


def opponent_key(d: dict, self_name: str):
    """Given a name-keyed dict (actions/scores), return the opponent's key."""
    if not d:
        return None
    sl = self_name.strip().lower()
    for k in d:
        if str(k).strip().lower() != sl:
            return k
    return None


def decision_to_index(decision: str, legal_actions: list, legal_actions_str: list) -> int:
    """Map 'cooperate'/'defect' to the right int from legal_actions."""
    target = decision.strip().lower()
    if legal_actions and legal_actions_str:
        for idx, label in zip(legal_actions, legal_actions_str):
            if str(label).strip().lower() == target:
                return int(idx)
    # Fallback to the canonical mapping; clamp to a legal value if possible.
    canonical = 0 if target == "cooperate" else 1
    if legal_actions and canonical not in legal_actions:
        return int(legal_actions[0])
    return canonical


# ---------------------------------------------------------------------------
# Match player — drives one repeated_pd child session start to finish
# ---------------------------------------------------------------------------
class MatchPlayer:
    def __init__(self, client, agent, memory, self_name, self_id, game_server_url, session_id, verbose=False):
        self.client = client
        self.agent = agent
        self.memory = memory
        self.self_name = self_name
        self.self_id = self_id
        self.gsu = normalize_game_server_url(game_server_url)
        self.sid = session_id
        self.verbose = verbose

        self.opp_name: str | None = None
        self.total_rounds: int | None = None
        # Per-round bookkeeping for context + post-match memory.
        self.my_msgs: dict[int, str] = {}
        self.opp_msgs: dict[int, str] = {}
        self.chatted_rounds: set[int] = set()
        self.moved_rounds: set[int] = set()
        # Filled from round_history at the end for memory.
        self.opp_moves: list[str] = []
        self.my_moves: list[str] = []

        # Structured trace: one JSON file per match (full reasoning + state).
        self.logger = TraceLogger(
            game_name="altruagent_repeated_pd",
            run_id=f"{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        self.logger.data["session_id"] = session_id
        self.logger.data["self_name"] = self_name
        self.logger.data["opponent"] = None

    # -- trace helpers -----------------------------------------------------
    def _trace_snapshot(self, state: dict) -> dict:
        keys = ("current_round", "total_rounds", "phase", "round_history",
                "cumulative_scores", "legal_actions", "legal_actions_str",
                "new_messages", "is_terminal")
        return make_json_safe({k: state.get(k) for k in keys})

    @staticmethod
    def _extract_meta(raw: str):
        """Pull rule_applied + classification_signal out of the agent's JSON."""
        from agent.single_call_agent import SingleCallAgent
        import json as _json
        try:
            d = _json.loads(SingleCallAgent._extract_json(raw or ""))
            return d.get("rule_applied", ""), d.get("classification_signal", "")
        except Exception:
            return "", ""

    # -- identity ----------------------------------------------------------
    def _is_me(self, sender) -> bool:
        s = str(sender).strip().lower()
        return s in (self.self_name.strip().lower(), str(self.self_id).strip().lower())

    def _learn_opponent(self, state: dict) -> None:
        if self.opp_name:
            return
        for field in ("cumulative_scores", "returns"):
            k = opponent_key(state.get(field) or {}, self.self_name)
            if k:
                self.opp_name = k
                break
        if not self.opp_name:
            lr = state.get("last_round") or {}
            k = opponent_key(lr.get("actions") or {}, self.self_name)
            if k:
                self.opp_name = k
        if self.opp_name:
            self.logger.data["opponent"] = self.opp_name

    # -- message tracking --------------------------------------------------
    def _ingest_messages(self, state: dict) -> None:
        """Record any newly visible messages, attributing sender + round."""
        cur = state.get("current_round") or 1
        for m in state.get("new_messages") or []:
            if not isinstance(m, dict):
                continue
            sender = (
                m.get("sender") or m.get("sender_name") or m.get("agent_name")
                or m.get("from") or m.get("player")
            )
            content = m.get("content") or m.get("message") or m.get("text") or ""
            rnd = m.get("round") or m.get("round_number") or cur
            if not content:
                continue
            if sender is not None and self._is_me(sender):
                self.my_msgs.setdefault(int(rnd), content)
            else:
                # Unknown/opponent sender -> treat as opponent's.
                self.opp_msgs[int(rnd)] = content
            if self.verbose:
                print(f"        [msg] round={rnd} sender={sender!r} content={content!r}")

    # -- context -----------------------------------------------------------
    def _build_context(self, state: dict, phase: str) -> str:
        cur = state.get("current_round") or 1
        total = state.get("total_rounds") or self.total_rounds or 8
        self.total_rounds = total
        opp = self.opp_name or "the opponent"

        # Action history from round_history (authoritative).
        lines = []
        my_seq, opp_seq = [], []
        for r in state.get("round_history") or []:
            actions = r.get("actions") or {}
            ok = opponent_key(actions, self.self_name)
            mine = action_label(actions.get(self.self_name, actions.get(self.self_name.lower(), 0)))
            theirs = action_label(actions.get(ok)) if ok else "?"
            my_seq.append(mine)
            opp_seq.append(theirs)
            pm, po = PAYOFFS.get((mine, theirs), (0, 0))
            lines.append(
                f"  Round {r.get('round')}: You={mine}, Opponent={theirs} "
                f"(You {pm:+d}, Opp {po:+d})"
            )
        action_block = "\nACTION HISTORY (this match):\n" + ("\n".join(lines) if lines else "  No actions yet.")

        # Message history. Current round's opponent msg is blind during MESSAGING.
        msg_lines = []
        for r in range(1, cur + 1):
            mine = self.my_msgs.get(r, "")
            theirs = self.opp_msgs.get(r, "")
            if not mine and not theirs:
                continue
            theirs_disp = theirs if theirs else ("(hidden until you move)" if r == cur and phase == "messaging" else "(none)")
            msg_lines.append(f"  Round {r}: You said: \"{mine}\" | Opponent said: \"{theirs_disp}\"")
        msg_block = "\nMESSAGE HISTORY:\n" + ("\n".join(msg_lines) if msg_lines else "  No messages yet.")

        scores = state.get("cumulative_scores") or {}
        ok = opponent_key(scores, self.self_name)
        score_line = ""
        if scores:
            mine_s = scores.get(self.self_name, scores.get(self.self_name.lower()))
            opp_s = scores.get(ok) if ok else None
            score_line = f"\nCumulative raw score so far — You: {mine_s}, Opponent: {opp_s}."

        return (
            f"You are playing Repeated Prisoner's Dilemma against an opponent named \"{opp}\".\n"
            f"This match is {total} rounds total. You are about to play round {cur}.\n"
            f"Final score is your AVERAGE payoff per round, so widespread mutual cooperation wins.\n\n"
            f"RULES (per round):\n"
            f"- Both cooperate  -> you +2, opp +2.\n"
            f"- You cooperate, they defect  -> you -1, they +5.\n"
            f"- You defect, they cooperate  -> you +5, they -1.\n"
            f"- Both defect  -> you 0, opp 0.\n\n"
            f"OPPONENT MEMORY (from prior tournaments):\n  {self.memory.context_block(opp)}\n"
            f"{action_block}\n"
            f"{msg_block}"
            f"{score_line}"
        )

    # -- phase actions -----------------------------------------------------
    def _do_messaging(self, state: dict) -> None:
        cur = state.get("current_round") or 1
        if cur in self.chatted_rounds:
            return  # already sent my one chat + terminate this round; just wait
        ctx = self._build_context(state, phase="messaging")
        snap = self._trace_snapshot(state)
        try:
            message, raw, prompt = self.agent.generate_message(ctx)
        except Exception as exc:
            print(f"      [LLM error in message phase: {exc}] -> terminating without chat")
            self.client.terminate_messaging(self.gsu, self.sid)
            self.chatted_rounds.add(cur)
            return
        rule, signal = self._extract_meta(raw)
        # one chat (<=50 words; agent already truncates), then terminate
        try:
            self.client.send_message(self.gsu, self.sid, message)
            self.my_msgs[cur] = message
            print(f"      [R{cur} msg] {message!r}")
            if self.verbose:
                print(f"      [R{cur} msg-reasoning] {raw.strip()[:500]}")
        except AltruAgentError as exc:
            print(f"      [send_message failed: {exc.code}] continuing to terminate")
        self.client.terminate_messaging(self.gsu, self.sid)
        self.chatted_rounds.add(cur)
        self.logger.record_step(
            round_number=cur, step_number=1,
            prompt=prompt, response=raw,
            parsed={"agent": self.self_name, "phase": "message",
                    "message": message, "rule_applied": rule,
                    "classification_signal": signal},
            observation="message phase (blind: opponent's message hidden until move)",
            state_before=snap,
        )

    def _do_move(self, state: dict) -> None:
        cur = state.get("current_round") or 1
        if cur in self.moved_rounds:
            return
        opp_msg = self.opp_msgs.get(cur, "")
        ctx = self._build_context(state, phase="moving")
        snap = self._trace_snapshot(state)
        prompt = ""
        try:
            decision, raw, prompt = self.agent.choose_action(ctx, opp_msg)
        except Exception as exc:
            print(f"      [LLM error in action phase: {exc}] -> defaulting to cooperate")
            decision = "cooperate"
            raw = ""
        rule, signal = self._extract_meta(raw)
        idx = decision_to_index(decision, state.get("legal_actions") or [], state.get("legal_actions_str") or [])
        try:
            self.client.step(self.gsu, self.sid, idx)
            self.moved_rounds.add(cur)
            print(f"      [R{cur} move] {decision} (action={idx})")
            if self.verbose and raw:
                print(f"      [R{cur} move-reasoning] {raw.strip()[:500]}")
        except AltruAgentError as exc:
            if exc.code in ("game_already_finished",):
                return
            print(f"      [step failed: {exc.code}] {exc.detail}")
        self.logger.record_step(
            round_number=cur, step_number=2,
            prompt=prompt, response=raw,
            parsed={"agent": self.self_name, "phase": "action",
                    "decision": decision, "action_index": idx,
                    "rule_applied": rule, "classification_signal": signal},
            observation=(f"Opponent message this round: {opp_msg!r}" if opp_msg
                         else "No opponent message visible this round"),
            state_before=snap,
        )

    # -- main loop ---------------------------------------------------------
    def play(self) -> dict | None:
        print(f"  >> Playing match {self.sid[:8]} on {self.gsu}")
        empty_polls = 0
        while True:
            try:
                state = self.client.game_state(self.gsu, self.sid)
            except AltruAgentError as exc:
                print(f"     [state error: {exc.code}] {exc.detail}")
                if exc.code in ("user_not_in_game", "session_not_found"):
                    return None
                time.sleep(self.client.poll_interval)
                continue

            if self.verbose:
                print(f"     [state] round={state.get('current_round')}/{state.get('total_rounds')} "
                      f"phase={state.get('phase')} legal={state.get('legal_actions')} "
                      f"next={[a.get('action') for a in state.get('next_actions', [])]}")

            self._learn_opponent(state)
            self._ingest_messages(state)

            if state.get("is_terminal"):
                self._finalize(state)
                return state.get("returns")

            phase = state.get("phase")
            next_action = (state.get("next_actions") or [{}])[0].get("action")

            acted = False
            if phase == "messaging":
                cur = state.get("current_round") or 1
                if cur not in self.chatted_rounds:
                    self._do_messaging(state)
                    acted = True
            elif phase == "moving":
                cur = state.get("current_round") or 1
                if state.get("legal_actions") and cur not in self.moved_rounds:
                    self._do_move(state)
                    acted = True
            elif next_action == "game_over":
                self._finalize(state)
                return state.get("returns")

            if acted:
                empty_polls = 0
                continue  # re-poll immediately after acting

            # Nothing to do: waiting on opponent. Guard against an infinite stall.
            empty_polls += 1
            if empty_polls > 120:  # ~10 min at 5s
                print("     [stalled too long — resigning]")
                try:
                    self.client.resign(self.gsu, self.sid)
                except AltruAgentError:
                    pass
                return None
            time.sleep(self.client.poll_interval)

    def _finalize(self, state: dict) -> None:
        # Pull final per-round sequences from round_history for memory.
        my_seq, opp_seq = [], []
        for r in state.get("round_history") or []:
            actions = r.get("actions") or {}
            ok = opponent_key(actions, self.self_name)
            my_seq.append(action_label(actions.get(self.self_name, actions.get(self.self_name.lower(), 0))))
            opp_seq.append(action_label(actions.get(ok)) if ok else "cooperate")
        returns = state.get("returns") or {}
        ok = opponent_key(returns, self.self_name)
        my_avg = returns.get(self.self_name, returns.get(self.self_name.lower()))
        opp_avg = returns.get(ok) if ok else None
        opp = self.opp_name or "unknown"
        label = self.memory.record_match(opp, opp_seq, my_seq, my_avg, opp_avg)
        self.memory.save()
        print(f"  << Match done vs {opp}: you={my_avg}, them={opp_avg}, "
              f"classified={label} (memory saved)")
        # Persist the full structured trace for this match.
        self.logger.data["opponent"] = opp
        self.logger.finish({
            "opponent": opp, "my_avg": my_avg, "opp_avg": opp_avg,
            "classification": label, "my_moves": my_seq, "opp_moves": opp_seq,
            "returns": make_json_safe(returns),
        })
        try:
            path = self.logger.save()
            print(f"     trace saved -> {path}")
        except Exception as exc:
            print(f"     [trace save failed: {exc}]")


# ---------------------------------------------------------------------------
# Orchestration: queues / tournaments / single competitions
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, verbose=False, memory_path=None, judge=False):
        self.client = AltruAgentClient()
        self.client.login()
        me = self.client.me()
        if me.get("status") != "claimed":
            raise SystemExit(f"Agent not claimed (status={me.get('status')}). Claim it first.")
        self.self_name = me["name"]
        self.self_id = me["id"]
        self.agent = SingleCallAgent(create_client(), name=self.self_name)
        self.memory = OpponentMemory(memory_path)
        self.verbose = verbose
        self.judge = judge
        print(f"Runner ready as '{self.self_name}' ({self.self_id[:8]}).")

    def _play_session(self, game_server_url, session_id):
        player = MatchPlayer(
            self.client, self.agent, self.memory, self.self_name, self.self_id,
            game_server_url, session_id, verbose=self.verbose,
        )
        returns = player.play()
        if self.judge and player.logger.path:
            try:
                from agent.tracing import judge_trace, write_judge_result
                result = judge_trace(player.logger.path, self.agent.client, append_to_reflections=False)
                write_judge_result(player.logger.path, result)
                print(f"     judge: {len(result.get('failures', []))} issue(s) noted")
            except Exception as exc:
                print(f"     [judge failed: {exc}]")
        return returns

    # -- single competition -----------------------------------------------
    def run_competition(self, session_id):
        print(f"Joining competition {session_id[:8]}...")
        try:
            self.client.join_competition(session_id)
        except AltruAgentError as exc:
            if exc.code == "join_failed" and "already" in (exc.detail or "").lower():
                print("  (already joined)")
            elif exc.code == "tournament_child_join_forbidden":
                raise SystemExit("That session is a tournament child — use --tournament.")
            else:
                raise
        comp = self.client.get_competition(session_id)
        gsu = comp.get("game_server_url")
        # Wait for the data plane to be ready (up to 5 min).
        for _ in range(60):
            try:
                state = self.client.game_state(gsu, session_id)
                if "is_terminal" in state:
                    break
            except AltruAgentError:
                pass
            time.sleep(self.client.poll_interval)
        self._play_session(gsu, session_id)

    # -- tournament --------------------------------------------------------
    def run_tournament(self, tournament_id, skip_join=False):
        # Via a queue you are AUTO-ENROLLED, so a second join on an already
        # in-progress instance returns 400 join_failed. That's expected and
        # non-fatal — membership is by stable agent_id, so just poll and play.
        if not skip_join:
            print(f"Joining tournament {tournament_id[:8]}...")
            try:
                self.client.join_tournament(tournament_id)
            except AltruAgentError as exc:
                if exc.code == "already_joined":
                    print("  (already joined)")
                else:
                    print(f"  (join returned {exc.code}: {exc.detail} — proceeding to poll)")
        played = set()
        while True:
            t = self.client.get_tournament(tournament_id)
            status = t.get("tournament", {}).get("status")
            gsu = t.get("tournament", {}).get("game_server_url")
            viewer = t.get("viewer", {}) or {}
            if status == "completed":
                print("Tournament complete.")
                self._print_leaderboard(t)
                return
            children = [c for c in (viewer.get("active_child_session_ids") or []) if c not in played]
            if children:
                for sid in children:
                    print(f"-- Active child session {sid[:8]}")
                    self._play_session(gsu, sid)
                    played.add(sid)
            else:
                if self.verbose:
                    na = (viewer.get("next_actions") or [{}])[0].get("action")
                    print(f"  waiting (status={status}, next={na})")
                time.sleep(self.client.poll_interval)

    # -- queue -------------------------------------------------------------
    def run_queue(self, queue_id):
        if queue_id == "auto":
            queues = [q for q in self.client.list_queues() if q.get("is_active")]
            if not queues:
                raise SystemExit("No active queues.")
            queue_id = queues[0]["queue_id"]
            print(f"Auto-selected queue {queues[0].get('name')} ({queue_id[:8]}).")
        print(f"Joining queue {queue_id[:8]}...")
        join = self.client.join_queue(queue_id)
        tournament_id = join.get("tournament_id")
        print(f"  -> tournament instance {str(tournament_id)[:8]}. Waiting for it to start "
              f"(needs >=3 agents + 120s timer)...")
        # Poll the queue until our instance leaves the waiting state.
        while True:
            q = self.client.get_queue(queue_id)
            waiting = q.get("current_waiting_tournament")
            if not waiting:
                print("  Tournament started (or no longer waiting).")
                break
            secs = waiting.get("seconds_until_start")
            parts = waiting.get("current_participants")
            print(f"  waiting... participants={parts} seconds_until_start={secs}")
            time.sleep(self.client.poll_interval)
        # Queue already enrolled us in this tournament — don't re-join.
        self.run_tournament(tournament_id, skip_join=True)

    def _print_leaderboard(self, t):
        lb = t.get("tournament", {}).get("leaderboard") or t.get("leaderboard")
        if not lb:
            return
        print("\n  Final leaderboard:")
        for row in lb:
            mark = " <- you" if row.get("agent_id") == self.self_id else ""
            print(f"   #{row.get('rank')} avg={row.get('average_payoff')} "
                  f"W/D/L={row.get('wins')}/{row.get('draws')}/{row.get('losses')}{mark}")


def _list_all():
    c = AltruAgentClient(); c.login()
    print("== ACTIVE QUEUES ==")
    for q in c.list_queues():
        if q.get("is_active"):
            print(f"  {q['queue_id']}  {q.get('name')}  ({q.get('game_type')}, "
                  f"{q.get('current_participants')}/{q.get('max_participants')})")
    print("== TOURNAMENTS ==")
    for t in c.list_tournaments().get("tournaments", []):
        print(f"  {t['tournament_id']}  {t.get('status')}  "
              f"{t.get('current_participants')}/{t.get('max_participants')}")
    print("== COMPETITIONS (waiting) ==")
    for comp in c.list_competitions():
        if comp.get("status") == "waiting":
            print(f"  {comp['session_id']}  {comp.get('game_type')}")


def main():
    ap = argparse.ArgumentParser(description="Run SingleCallAgent on AltruAgent.")
    ap.add_argument("--list", action="store_true", help="List queues/tournaments/competitions and exit")
    ap.add_argument("--queue", help="Queue id to join, or 'auto' for the first active queue")
    ap.add_argument("--tournament", help="Tournament id to join and play")
    ap.add_argument("--session", help="Single competition session id to join and play")
    ap.add_argument("--verbose", action="store_true", help="Dump raw state/messages each poll")
    ap.add_argument("--memory", help="Path to opponent memory JSON")
    ap.add_argument("--judge", action="store_true",
                    help="Run the LLM trace-judge on each match trace after it finishes")
    args = ap.parse_args()

    if args.list:
        _list_all()
        return

    if not (args.queue or args.tournament or args.session):
        print("Nothing to do. Use --list, --queue auto, --tournament <id>, or --session <id>.")
        return

    runner = Runner(verbose=args.verbose, memory_path=args.memory, judge=args.judge)
    if args.queue:
        runner.run_queue(args.queue)
    elif args.tournament:
        runner.run_tournament(args.tournament)
    elif args.session:
        runner.run_competition(args.session)


if __name__ == "__main__":
    main()
