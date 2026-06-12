"""FAST variant of the AltruAgent runner — "reuse-the-move" architecture.

Why this exists: the standard runner makes TWO LLM calls per round (message
phase, then action phase). On the stronger but slower gemini-2.5-flash the
second call races the ~30s move timeout and the platform auto-cooperates.

This variant makes ONE LLM call (the message phase already DECIDES the action)
and then submits that same action the instant the move window opens — no second
call on the critical path. That lets us run the stronger flash model (better
reasoning, far fewer hallucinations) without ever missing a move window.

It is a thin SUBCLASS of the standard runner: everything (queue handling,
trace-save recovery, opponent memory, the threat signal, _build_context,
_finalize, the whole play loop) is INHERITED unchanged. Only _do_messaging and
_do_move are overridden, plus the model is set to flash explicitly. The standard
runner (games/altruagent_runner.py) is left completely untouched — if this
variant misbehaves, just run the standard one; nothing needs to be reverted.

Usage (identical to the standard runner):
  python games/altruagent_runner_fast.py --queue <queue_id> --verbose
  python games/altruagent_runner_fast.py --tournament <id> --verbose
  python games/altruagent_runner_fast.py --list
  add --memory <path> to isolate opponent memory while testing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import create_client
from agent.single_call_agent import SingleCallAgent
from agent.altruagent_client import AltruAgentError
from agent.opponent_memory import cold_defection_stats

from games.altruagent_runner import (
    MatchPlayer,
    Runner,
    action_label,
    opponent_key,
    decision_to_index,
    _list_all,
)

# The stronger reasoner. Only THIS runner uses it; the standard runner and the
# agent.py default are untouched.
FAST_MODEL = "gemini-2.5-flash"


class FastMatchPlayer(MatchPlayer):
    """MatchPlayer that reuses the message-phase action as the move."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The action (and the rule that produced it) decided in each round's
        # message phase, to be submitted unchanged when the move window opens.
        self.planned_actions: dict[int, str] = {}
        self.planned_rule: dict[int, str] = {}

    # -- best-effort end of the chat phase (never fatal) ---------------------
    def _safe_terminate(self) -> None:
        """Terminate the messaging phase, tolerating a rate-limit/transient error.

        The client already retries 429s with backoff; if one still survives, we
        swallow it here — terminating the chat is best-effort, and the phase will
        time out on its own. A failed terminate must NOT crash the whole match.
        """
        try:
            self.client.terminate_messaging(self.gsu, self.sid)
        except AltruAgentError as exc:
            print(f"      [terminate_messaging failed: {exc.code} — non-fatal, phase will time out]")

    # -- message phase: decide + send message, and STORE the action ----------
    def _do_messaging(self, state: dict) -> None:
        cur = state.get("current_round") or 1
        if cur in self.chatted_rounds:
            return
        ctx = self._build_context(state, phase="messaging")
        snap = self._trace_snapshot(state)
        try:
            message, raw, prompt = self.agent.generate_message(ctx)
        except Exception as exc:
            # No planned action this round -> _do_move uses the deterministic
            # fallback (NOT a blind cooperate).
            print(f"      [LLM error in message phase: {exc}] -> terminating without chat")
            self._safe_terminate()
            self.chatted_rounds.add(cur)
            return
        rule, signal = self._extract_meta(raw)
        # The action decided here IS the move we will submit (reuse-the-move).
        decision = self.agent._parse_action(raw)
        self.planned_actions[cur] = decision
        self.planned_rule[cur] = rule
        try:
            self.client.send_message(self.gsu, self.sid, message)
            self.my_msgs[cur] = message
            print(f"      [R{cur} msg] {message!r}  (planned move: {decision})")
            if self.verbose:
                print(f"      [R{cur} msg-reasoning] {raw.strip()[:500]}")
        except AltruAgentError as exc:
            print(f"      [send_message failed: {exc.code}] continuing to terminate")
        self._safe_terminate()
        self.chatted_rounds.add(cur)
        self.logger.record_step(
            round_number=cur, step_number=1,
            prompt=prompt, response=raw,
            parsed={"agent": self.self_name, "phase": "message",
                    "message": message, "rule_applied": rule,
                    "classification_signal": signal, "planned_action": decision},
            observation="message phase decides the binding action (reused at move time)",
            state_before=snap,
        )

    # -- move phase: submit the already-decided action, NO LLM call ----------
    def _do_move(self, state: dict) -> None:
        cur = state.get("current_round") or 1
        if cur in self.moved_rounds:
            return
        snap = self._trace_snapshot(state)
        if cur in self.planned_actions:
            decision = self.planned_actions[cur]
            rule = self.planned_rule.get(cur, "")
            source = "reused-from-message-phase"
        else:
            decision = self._fallback_action(state)
            rule = "deterministic fallback (message phase produced no action)"
            source = "DETERMINISTIC-FALLBACK"
            print(f"      [R{cur} no planned action -> deterministic fallback: {decision}]")
        idx = decision_to_index(decision, state.get("legal_actions") or [], state.get("legal_actions_str") or [])
        try:
            resp = self.client.step(self.gsu, self.sid, idx)
            self.moved_rounds.add(cur)
            # Burn the one-time forgiveness if this move was the Rule 9 grant
            # (the message-phase rule is the authoritative decision here).
            if (
                self.opp_name
                and self._looks_like_forgiveness(rule, decision)
                and not self.memory.forgiveness_used(self.opp_name)
            ):
                self.memory.mark_forgiveness_used(self.opp_name)
                self.forgave_round = cur
                print(f"      [R{cur} forgiveness granted -> forgiveness_used=True "
                      f"for {self.opp_name} (permanent)]")
            if isinstance(resp, dict) and (
                "round_history" in resp or "returns" in resp or resp.get("is_terminal")
            ):
                self.last_state = resp
            print(f"      [R{cur} move] {decision} (action={idx}) [{source}]")
        except AltruAgentError as exc:
            if exc.code in ("game_already_finished",):
                return
            print(f"      [step failed: {exc.code}] {exc.detail}")
        self.logger.record_step(
            round_number=cur, step_number=2,
            prompt="(reuse-the-move: action decided in the message phase; no LLM call)",
            response="",
            parsed={"agent": self.self_name, "phase": "action",
                    "decision": decision, "action_index": idx,
                    "rule_applied": rule, "classification_signal": "",
                    "move_source": source},
            observation=f"Move {source}: {decision}",
            state_before=snap,
        )

    # -- deterministic fallback if the LLM produced no action ----------------
    def _fallback_action(self, state: dict) -> str:
        """A sensible move when the message-phase call failed entirely.

        Uses the SAME signals the prompt would have (in-match cold-defection
        threat + stored hostile classification + post-forgiveness state), so a
        total LLM failure defends instead of blindly cooperating.
        """
        my_seq, opp_seq = [], []
        for r in state.get("round_history") or []:
            actions = r.get("actions") or {}
            ok = opponent_key(actions, self.self_name)
            my_seq.append(action_label(actions.get(self.self_name, actions.get(self.self_name.lower(), 0))))
            opp_seq.append(action_label(actions.get(ok)) if ok else "cooperate")
        opps, _cold, cold_rate = cold_defection_stats(opp_seq, my_seq)
        threat_active = opps >= 2 and cold_rate > 0.5
        hostile = bool(self.opp_name) and self.memory.is_hostile(self.opp_name)
        # Defected again after we spent forgiveness this match -> lockout.
        post_forgive_defect = (self.forgave_round is not None and opp_seq[-1:] == ["defect"])
        if threat_active or hostile or post_forgive_defect:
            return "defect"
        return "cooperate"


class FastRunner(Runner):
    """Runner that plays with FastMatchPlayer on the flash model."""

    def __init__(self, verbose=False, memory_path=None, judge=False, model=FAST_MODEL):
        super().__init__(verbose=verbose, memory_path=memory_path, judge=judge)
        # Override the agent to use the stronger model EXPLICITLY. This overrides
        # only this runner's agent; agent.py's default and the standard runner
        # are untouched.
        self.agent = SingleCallAgent(create_client(), model=model, name=self.self_name)
        print(f"  [fast runner] reuse-the-move enabled; model={model}")

    def _play_session(self, game_server_url, session_id):
        player = FastMatchPlayer(
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


def main():
    ap = argparse.ArgumentParser(description="Run SingleCallAgent on AltruAgent (FAST reuse-the-move variant).")
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

    runner = FastRunner(verbose=args.verbose, memory_path=args.memory, judge=args.judge, model=args.model)
    if args.queue:
        runner.run_queue(args.queue)
    elif args.tournament:
        runner.run_tournament(args.tournament)
    elif args.session:
        runner.run_competition(args.session)


if __name__ == "__main__":
    main()
