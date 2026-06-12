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
from agent.opponent_memory import OpponentMemory, cold_defection_stats
from agent.tracing import TraceLogger, make_json_safe

# Total tournaments in the competition. The agent faces every opponent once
# per tournament, so per-opponent matches_played + 1 = current tournament
# number. The endgame exploit fires only in the FINAL tournament.
TOTAL_TOURNAMENTS = 3

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
        # Our own integer seat index (0/1), learned from our echoed message.
        self.my_index: int | None = None
        # Highest message index ingested from the /messages cursor endpoint.
        self.msg_cursor: int = -1
        # Last good game state (used for a best-effort finalize if the session
        # is torn down before we can read the terminal state) + finalize guard.
        self.last_state: dict | None = None
        self._finalized: bool = False
        # Forgiveness state, split so the prompt never conflates prior-tournament
        # usage with this-match usage. `forgiveness_at_start` is snapshotted when
        # the opponent is first learned (before any move); `forgave_round` is the
        # round we spent forgiveness during THIS match (None if not yet).
        self.forgiveness_at_start: bool = False
        self.forgave_round: int | None = None
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

    @staticmethod
    def _looks_like_forgiveness(rule: str, decision: str) -> bool:
        """True only for a Rule 9 forgiveness cooperation (not normal coops).

        Gated on the move being a COOPERATE whose rule_applied names Rule 9 —
        the single early-defection forgiveness grant. Spacing-insensitive so
        "Rule 9" / "rule9" both match; no other rule number collides.
        """
        if decision != "cooperate":
            return False
        return "rule9" in (rule or "").lower().replace(" ", "")

    # -- identity ----------------------------------------------------------
    @staticmethod
    def _get_sender(m: dict):
        """Pull the sender out of a message dict.

        Checks each candidate key for `is not None` (NOT truthiness) so a
        valid integer seat index of 0 is never dropped.
        """
        for key in ("sender", "sender_index", "sender_id", "sender_name",
                    "agent_name", "from", "player"):
            if key in m and m[key] is not None:
                return m[key]
        return None

    def _sender_is_me(self, sender, content: str, rnd: int) -> bool:
        """True if `sender` is us. Handles both name/id strings and int seats."""
        # Name/id string senders (non-numeric).
        if isinstance(sender, str) and not sender.strip().lstrip("-").isdigit():
            s = sender.strip().lower()
            return s in (self.self_name.strip().lower(), str(self.self_id).strip().lower())
        # Integer seat index (sender may be an int or a digit string like "0").
        try:
            idx = int(sender)
        except (TypeError, ValueError):
            return False
        if self.my_index is not None:
            return idx == self.my_index
        # Learn our own seat once: our sent message echoes back to us.
        if content and self.my_msgs.get(rnd) == content:
            self.my_index = idx
            return True
        return False

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
            # Snapshot prior-tournament forgiveness BEFORE any move this match, so
            # the memory block reports only prior usage (never this-match usage).
            self.forgiveness_at_start = self.memory.forgiveness_used(self.opp_name)

    # -- message tracking --------------------------------------------------
    def _collect_raw_messages(self, state: dict) -> list[dict]:
        """Gather message dicts from the cursor endpoint plus new_messages.

        The cursor endpoint is the reliable source under blind/per_all_moves
        messaging (new_messages is empty for the opponent's current message
        until the phase flips, then cleared again after we move). We advance
        self.msg_cursor by the highest index seen so each batch is fetched once.
        """
        raw: list[dict] = []
        try:
            resp = self.client.messages(self.gsu, self.sid, since=self.msg_cursor)
            batch = resp.get("messages") if isinstance(resp, dict) else resp
            for m in batch or []:
                if not isinstance(m, dict):
                    continue
                raw.append(m)
                for ik in ("index", "id", "seq", "message_index"):
                    if isinstance(m.get(ik), int):
                        self.msg_cursor = max(self.msg_cursor, m[ik])
                        break
        except AltruAgentError as exc:
            if self.verbose:
                print(f"        [messages endpoint: {exc.code}]")
        # Supplement with new_messages (duplicates are idempotent on ingest).
        for m in state.get("new_messages") or []:
            if isinstance(m, dict):
                raw.append(m)
        return raw

    def _ingest_messages(self, state: dict) -> None:
        """Record newly visible messages, attributing each to us or the opponent.

        Learns our own integer seat index from our echoed message (first pass)
        so an opponent message seen first can't be misfiled, then files the rest.
        """
        cur = state.get("current_round") or 1
        parsed = []
        for m in self._collect_raw_messages(state):
            sender = self._get_sender(m)
            content = m.get("content") or m.get("message") or m.get("text") or ""
            rnd = int(m.get("round") or m.get("round_number") or cur)
            if content:
                parsed.append((sender, content, rnd))
        if not parsed:
            return
        # First pass: pin our own seat index from our echoed message.
        if self.my_index is None:
            for sender, content, rnd in parsed:
                if self.my_msgs.get(rnd) == content:
                    try:
                        self.my_index = int(sender)
                    except (TypeError, ValueError):
                        pass
                    break
        # Second pass: file each message under us or the opponent.
        for sender, content, rnd in parsed:
            if sender is not None and self._sender_is_me(sender, content, rnd):
                self.my_msgs.setdefault(rnd, content)
            elif self.opp_msgs.get(rnd) != content:
                self.opp_msgs[rnd] = content
                if self.verbose:
                    print(f"        [opp msg] round={rnd} sender={sender!r} content={content!r}")

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

        # Authoritative tournament number, derived from how many times we have
        # already finished a match against this opponent. The brain must NOT
        # guess this — the endgame exploit hinges on it being correct.
        prior = self.memory.matches_played(self.opp_name) if self.opp_name else 0
        tnum = prior + 1
        is_final = tnum >= TOTAL_TOURNAMENTS
        if is_final:
            tnote = (
                f"This is the FINAL tournament (Tournament {tnum} of {TOTAL_TOURNAMENTS}). "
                f"After Round {total} of this match there are NO future interactions with "
                f"this opponent ever again — reputation no longer matters, so the endgame "
                f"exploit (Rules 2, 3) IS live."
            )
        else:
            remaining = TOTAL_TOURNAMENTS - tnum
            tnote = (
                f"This is Tournament {tnum} of {TOTAL_TOURNAMENTS} — NOT the final tournament. "
                f"You WILL face this opponent again in {remaining} more tournament(s), so "
                f"reputation still matters. The endgame exploit (Rules 2, 3) does NOT apply: "
                f"defecting in the last round here only poisons a relationship you still need."
            )
        tournament_block = (
            "\nTOURNAMENT CONTEXT (authoritative — derived from memory, do NOT guess):\n"
            f"  {tnote}\n"
        )

        # THIS-MATCH forgiveness status, kept separate from prior-tournament
        # memory so the model never sees a "0 prior matches but forgiveness used"
        # paradox (which used to make it fabricate prior tournaments).
        if self.forgave_round is not None:
            this_match_f = (
                f"  Forgiveness already spent THIS match at Round {self.forgave_round} — "
                f"do not forgive again; if they defect, apply the post-forgiveness lockout."
            )
        elif self.forgiveness_at_start:
            this_match_f = (
                "  Your one-time forgiveness was already used in a prior tournament — "
                "none available this match."
            )
        else:
            this_match_f = "  Forgiveness not yet used — available once this match."
        this_match_block = f"\nTHIS MATCH:\n{this_match_f}\n"

        # In-match high-defection THREAT signal, computed deterministically from
        # the SAME cold-defection definition as classify() (shared helper). Fires
        # only on frequent UNPROVOKED defectors (random / mostly-defect), never on
        # a reciprocator we provoked. Earliest it can fire is round 3 (needs >=2
        # opportunities), so it can never touch the round-1 opening.
        opps, cold, cold_rate = cold_defection_stats(opp_seq, my_seq)
        threat_active = opps >= 2 and cold_rate > 0.5
        if threat_active:
            threat_block = (
                "\nTHREAT THIS MATCH (computed, authoritative):\n"
                f"  Opponent has defected UNPROVOKED on {cold} of {opps} of your cooperative "
                f"overtures ({cold_rate:.0%}). They are exploiting your cooperation, not reciprocating.\n"
                "  DEFEND: defect this round. Do NOT forgive (Rule 9) and do NOT cooperate just "
                "because they cooperated last round (Rule 10) — one isolated cooperate from a "
                "frequent defector is noise, not a trust signal.\n"
            )
        elif opps >= 1:
            threat_block = (
                f"\nTHREAT THIS MATCH (computed): opponent unprovoked-defection rate {cold}/{opps} "
                f"({cold_rate:.0%}) — below the defense threshold; normal rules apply.\n"
            )
        else:
            threat_block = ""

        return (
            f"You are playing Repeated Prisoner's Dilemma against an opponent named \"{opp}\".\n"
            f"This match is {total} rounds total. You are about to play round {cur}.\n"
            f"Final score is your AVERAGE payoff per round, so widespread mutual cooperation wins.\n\n"
            f"RULES (per round):\n"
            f"- Both cooperate  -> you +2, opp +2.\n"
            f"- You cooperate, they defect  -> you -1, they +5.\n"
            f"- You defect, they cooperate  -> you +5, they -1.\n"
            f"- Both defect  -> you 0, opp 0.\n"
            f"{tournament_block}"
            f"\nOPPONENT MEMORY (from prior tournaments):\n  {self.memory.context_block(opp, prior_forgiveness=self.forgiveness_at_start)}\n"
            f"{this_match_block}"
            f"{threat_block}"
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
            resp = self.client.step(self.gsu, self.sid, idx)
            self.moved_rounds.add(cur)
            # The move is now played. If it was the Rule 9 forgiveness-cooperate,
            # burn the one-time forgiveness for this opponent — permanently and
            # immediately persisted (survives across matches/tournaments/crashes).
            if (
                self.opp_name
                and self._looks_like_forgiveness(rule, decision)
                and not self.memory.forgiveness_used(self.opp_name)
            ):
                self.memory.mark_forgiveness_used(self.opp_name)
                self.forgave_round = cur
                print(f"      [R{cur} forgiveness granted -> forgiveness_used=True "
                      f"for {self.opp_name} (permanent)]")
            # The step response usually carries the post-move state. Stash it so
            # that if the platform tears the session down right after the final
            # move (before our next poll), we can still finalize from full data.
            if isinstance(resp, dict) and (
                "round_history" in resp or "returns" in resp or resp.get("is_terminal")
            ):
                self.last_state = resp
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
                self.last_state = state
            except AltruAgentError as exc:
                print(f"     [state error: {exc.code}] {exc.detail}")
                if exc.code in ("user_not_in_game", "session_not_found"):
                    return self._handle_session_gone()
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

    def _finalize(self, state: dict, partial: bool = False) -> None:
        if self._finalized:
            return  # one-shot: never write a trace/memory entry twice
        self._finalized = True
        state = state or {}
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

        # On a best-effort (partial) finalize the session vanished before we
        # could read the terminal state. Only write to opponent MEMORY if the
        # match was effectively complete (we have ~all rounds) and we know the
        # opponent — otherwise a short fragment would mislabel them. The full
        # per-round trace is saved either way so nothing is lost.
        rounds_seen = len(my_seq)
        complete_enough = (not partial) or (
            bool(self.total_rounds) and rounds_seen >= self.total_rounds - 1
        )
        label = "(not recorded)"
        if complete_enough and opp != "unknown":
            label = self.memory.record_match(opp, opp_seq, my_seq, my_avg, opp_avg)
            self.memory.save()
            tag = " [recovered after session teardown]" if partial else ""
            print(f"  << Match done vs {opp}: you={my_avg}, them={opp_avg}, "
                  f"classified={label} (memory saved){tag}")
        else:
            print(f"  << Match vs {opp} ended early ({rounds_seen} rounds seen) — "
                  f"trace saved, memory NOT updated")
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

    def _handle_session_gone(self) -> dict | None:
        """Recover when the session vanishes mid-poll.

        If we've already made a move, a session_not_found is almost always
        end-of-match teardown that raced ahead of our terminal-state poll (the
        bug that silently dropped a whole match's trace). Retry briefly; if the
        session is truly gone, finalize from the last state we saw so the trace
        (and memory, when the match was effectively complete) survive.
        """
        if self._finalized or not self.moved_rounds:
            return None
        for _ in range(3):
            time.sleep(self.client.poll_interval)
            try:
                state = self.client.game_state(self.gsu, self.sid)
            except AltruAgentError:
                continue
            self.last_state = state
            if state.get("is_terminal"):
                self._finalize(state)
                return state.get("returns")
        print("     [session gone after final move — finalizing from last known state]")
        self._finalize(self.last_state or {}, partial=True)
        return (self.last_state or {}).get("returns")


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
    @staticmethod
    def _extract_tournament_id(obj) -> str | None:
        """Best-effort pull of a tournament id from a queue/join/waiting dict.

        The platform returns the id under different keys depending on shape
        (top-level on join, nested inside the waiting-instance object, etc.),
        so check the common spots and never return the literal string 'None'.
        """
        if not isinstance(obj, dict):
            return None
        for k in ("tournament_id", "id", "instance_id"):
            v = obj.get(k)
            if v and str(v) != "None":
                return str(v)
        for nest in ("tournament", "current_waiting_tournament", "waiting_tournament"):
            got = Runner._extract_tournament_id(obj.get(nest))
            if got:
                return got
        return None

    def _find_my_active_tournament(self) -> str | None:
        """Scan tournaments for the live one we're enrolled in.

        Field-name-independent fallback: once a queue instance starts, the
        queue stops exposing its id, but we are a participant in exactly one
        waiting/in-progress tournament. Pick the most recently created.
        """
        try:
            tours = self.client.list_tournaments().get("tournaments", [])
        except AltruAgentError:
            return None
        candidates = []
        for tt in tours:
            if tt.get("status") not in ("waiting", "in_progress"):
                continue
            tid = tt.get("tournament_id")
            try:
                full = self.client.get_tournament(tid)
            except AltruAgentError:
                continue
            if (full.get("viewer") or {}).get("is_tournament_participant"):
                candidates.append((tt.get("created_at", ""), tid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[-1][1]

    def run_queue(self, queue_id):
        if queue_id == "auto":
            queues = [q for q in self.client.list_queues() if q.get("is_active")]
            if not queues:
                raise SystemExit("No active queues.")
            queue_id = queues[0]["queue_id"]
            print(f"Auto-selected queue {queues[0].get('name')} ({queue_id[:8]}).")
        print(f"Joining queue {queue_id[:8]}...")
        join = self.client.join_queue(queue_id)
        tournament_id = self._extract_tournament_id(join)
        print(f"  -> tournament instance {str(tournament_id)[:8]}. Waiting for it to start "
              f"(needs >=3 agents + 120s timer)...")
        # Poll the queue until our instance leaves the waiting state.
        while True:
            q = self.client.get_queue(queue_id)
            waiting = q.get("current_waiting_tournament")
            if not waiting:
                print("  Tournament started (or no longer waiting).")
                break
            # Capture the id from the waiting instance if join didn't give us one.
            if not tournament_id:
                tournament_id = self._extract_tournament_id(waiting)
            secs = waiting.get("seconds_until_start")
            parts = waiting.get("current_participants")
            print(f"  waiting... participants={parts} seconds_until_start={secs}")
            time.sleep(self.client.poll_interval)
        # If we still don't have the id (queue no longer exposes it post-start),
        # resolve it by finding the live tournament we're enrolled in.
        if not tournament_id:
            print("  Resolving tournament id via participant scan...")
            tournament_id = self._find_my_active_tournament()
        if not tournament_id:
            raise SystemExit(
                "Could not determine the tournament instance id. Run "
                "`--list` to find it, then use `--tournament <id>`."
            )
        print(f"  Joining tournament instance {tournament_id[:8]}.")
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
