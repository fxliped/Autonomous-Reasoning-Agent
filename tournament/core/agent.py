"""
TournamentAgent — AltruAgent platform submission interface.

The two-phase round structure mirrors the platform exactly:

  MESSAGING PHASE (blind — you write before seeing opponent's message):
    message = agent.compose_message(round_num, total_rounds, history, my_score, opp_score)

  MOVING PHASE (opponent's message now revealed):
    action_int = agent.choose_action(round_num, total_rounds, opp_msg, my_msg, history, ...)
    # 0 = Cooperate, 1 = Defect

  After round result is known:
    agent.record_round_result(round_num, opp_action_label, my_pts, opp_pts)

  After game ends:
    agent.end_match(my_avg_score, opp_avg_score)  # per-round averages

Test locally:
    python tournament/core/agent.py --strategy tit_for_tat
"""

import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.memory import update_profile_after_match  # noqa: E402
from agent.tracing import judge_match_rounds  # noqa: E402
from analytics.db import (  # noqa: E402
    COOP_WORDS,
    ingest_match_rounds,
    load_opponent,
    save_opponent,
    log_match,
)
from games.pd_game import PrisonersDilemma  # noqa: E402
from .classifier import (  # noqa: E402
    BehavioralProfile,
    bayes_update,
    top_classification,
    _UNIFORM_PRIOR,
)
from .context import build_message_context, build_action_context, _self_pattern_block  # noqa: E402
from .prompts import (  # noqa: E402
    TOURNAMENT_SYSTEM_PROMPT,
    _ADVOCATE_C,
    _ADVOCATE_D,
    _JUDGE,
    parse_message,
    parse_action,
)

GAME_NAME = "prisoners_dilemma"

_PUNISHMENT_SIGNALS = re.compile(
    r'\b(mirror|punish|retaliat|defect if|eye for|tit.for|in.kind|respond in|same as you|copy your|copy me|i will defect)',
    re.IGNORECASE,
)


# =============================================================================
# TOURNAMENT AGENT
# =============================================================================

class TournamentAgent:
    """
    Tournament-ready Prisoner's Dilemma agent for the AltruAgent platform.

    Matches the platform's two-phase round structure:
      MESSAGING PHASE (blind — both write simultaneously):
        message = agent.compose_message(round_num, total_rounds, history, my_score, opp_score)

      MOVING PHASE (both messages now revealed):
        action_int = agent.choose_action(round_num, total_rounds, opp_msg, my_msg, history, ...)
        # POST /step {"action": action_int}  (0=Cooperate, 1=Defect)

      After round completes:
        agent.record_round_result(round_num, opp_action_label, my_pts, opp_pts)

      After game terminal:
        agent.end_match(my_avg_score, opp_avg_score)
    """

    ACTION_COOPERATE = 0
    ACTION_DEFECT = 1
    PAYOFFS = PrisonersDilemma.PAYOFFS

    def __init__(
        self,
        opponent_id: str = "unknown",
        total_rounds: int = 8,
        use_debate: bool = False,
        verbose: bool = False,
    ):
        self.opponent_id = opponent_id
        self.total_rounds = total_rounds
        self.use_debate = use_debate
        self.verbose = verbose
        self.client = create_client()
        self.profile = load_opponent(opponent_id)
        self._match_rounds: list[dict] = []
        # Tournament uses single-shot structured CoT — no ReAct tool loop runs.
        self._system = build_system_prompt(TOURNAMENT_SYSTEM_PROMPT, game_name=GAME_NAME, use_react=False)
        self._deception_count = 0
        self._last_hypothesis = "Unknown"
        self._last_confidence = 0.0
        self._hypothesis_violations = 0
        self._rank: int | None = None
        self._total_players: int | None = None
        self._score_gap_to_above: float | None = None
        self._opponent_rank: int | None = None
        self._matches_remaining: int | None = None
        self._type_posterior: dict[str, float] = dict(_UNIFORM_PRIOR)
        self._behavioral = BehavioralProfile()
        self._run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._debate_count = 0  # debates used this match (capped at 2 to control LLM cost)
        self._self_pattern = _self_pattern_block()  # cached once per match; queries analytics DB

    @property
    def match_rounds(self) -> list[dict]:
        """Read-only snapshot of completed rounds this match."""
        return list(self._match_rounds)

    def update_leaderboard(
        self,
        rank: int,
        total_players: int,
        score_gap_to_above: float | None = None,
        opponent_rank: int | None = None,
        matches_remaining: int | None = None,
    ) -> None:
        """
        Call before each match to inject current tournament standing.
        score_gap_to_above: my avg score minus the score of the player ranked just above me
                            (negative = I'm behind them).
        """
        self._rank = rank
        self._total_players = total_players
        self._score_gap_to_above = score_gap_to_above
        self._opponent_rank = opponent_rank
        self._matches_remaining = matches_remaining

    def _leaderboard_block(self) -> str:
        if self._rank is None or self._total_players is None:
            return ""
        pct = self._rank / self._total_players
        if pct <= 0.25:
            mode = "DEFENSIVE — protect lead. Prioritize +2/round. Avoid -1 exposure."
        elif pct >= 0.75:
            mode = "AGGRESSIVE — need large swings. Accept -1 risk to attempt +5. Target weak opponents."
        else:
            mode = "BALANCED — cooperate with strong players (avoid costly wars), exploit low-ranked ones."
        lines = [f"LEADERBOARD: Rank {self._rank}/{self._total_players} — {mode}"]

        if self._matches_remaining is not None and self._matches_remaining <= 2:
            if self._score_gap_to_above is not None and self._score_gap_to_above < 0:
                needed = abs(self._score_gap_to_above)
                lines.append(
                    f"END-GAME TARGET: You are {needed:.2f} avg pts/round behind rank "
                    f"{self._rank - 1}. With {self._matches_remaining} match(es) left, "
                    "you need to score aggressively to advance. Maximize extraction — "
                    "this is not a round to settle for mutual cooperation."
                )
            elif self._score_gap_to_above is not None and self._score_gap_to_above >= 0:
                lines.append(
                    f"END-GAME DEFENSE: You lead rank {self._rank + 1} by "
                    f"{self._score_gap_to_above:.2f} avg pts/round with "
                    f"{self._matches_remaining} match(es) left. Protect the lead — "
                    "favor safe +2 over risky +5 attempts."
                )

        if self._opponent_rank is not None and self._total_players:
            opp_pct = self._opponent_rank / self._total_players
            if opp_pct >= 0.75:
                lines.append(
                    f"OPPONENT RANK: {self._opponent_rank}/{self._total_players} — "
                    "bottom-tier opponent. Low retaliation threat. Maximize extraction."
                )
            elif opp_pct <= 0.25:
                lines.append(
                    f"OPPONENT RANK: {self._opponent_rank}/{self._total_players} — "
                    "top-tier opponent. Costly war not worth it. Favor cooperation unless clearly exploited."
                )

        return "\n".join(lines)

    _MAX_DEBATES_PER_MATCH = 2

    def _should_debate(self, round_num: int, total_rounds: int) -> bool:
        if not self.use_debate:
            return False
        if self._debate_count >= self._MAX_DEBATES_PER_MATCH:
            return False
        is_endgame = (total_rounds - round_num) <= 1
        is_uncertain = self._last_confidence < 0.6 and round_num > 2
        is_obvious = self._last_hypothesis == "Always Defect" and self._last_confidence >= 0.7
        return (is_endgame or is_uncertain) and not is_obvious

    def _run_debate(self, context: str, round_num: int) -> int:
        """3-agent debate: Advocate_C vs Advocate_D → Judge. Returns action int."""
        import concurrent.futures  # noqa: PLC0415
        print(f"[Debate R{round_num}] Running Advocate_C vs Advocate_D...")

        def _call_advocate(prompt_template: str) -> str:
            agent = Agent(client=self.client, system=self._system)
            return agent(prompt_template.format(context=context)) or ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_c = pool.submit(_call_advocate, _ADVOCATE_C)
            future_d = pool.submit(_call_advocate, _ADVOCATE_D)
            case_c = future_c.result()
            case_d = future_d.result()

        c_hit = re.search(r"ADVOCATE COOPERATE:\s*(.+)$", case_c, re.IGNORECASE | re.MULTILINE)
        d_hit = re.search(r"ADVOCATE DEFECT:\s*(.+)$", case_d, re.IGNORECASE | re.MULTILINE)
        if self.verbose:
            print(f"\n[Adv-C full]\n{case_c}\n")
            print(f"\n[Adv-D full]\n{case_d}\n")
        else:
            print(f"  Adv-C: {(c_hit.group(1) if c_hit else case_c[-120:]).strip()}")
            print(f"  Adv-D: {(d_hit.group(1) if d_hit else case_d[-120:]).strip()}")

        judge = Agent(client=self.client, system=self._system)
        judge_resp = judge(_JUDGE.format(
            cooperate_case=case_c[-800:],
            defect_case=case_d[-800:],
            context=context,
        )) or ""
        if self.verbose:
            print(f"\n[Debate R{round_num} JUDGE]\n{judge_resp}\n")

        if judge_resp:
            self._extract_hypothesis(judge_resp)

        return parse_action(judge_resp)

    def _extract_hypothesis(self, response: str) -> None:
        """Parse and store hypothesis + confidence from an LLM response."""
        hyp_hit = re.search(
            r"Type hypothesis:\s*(Always Defect|Naive Cooperator|Tit-for-Tat|Grim Trigger"
            r"|Pavlov|Strategic/Adaptive|Unknown)",
            response, re.IGNORECASE,
        )
        if hyp_hit:
            self._last_hypothesis = hyp_hit.group(1)
        conf_hit = re.search(r"Confidence:\s*(\d+)", response)
        if conf_hit:
            self._last_confidence = min(float(conf_hit.group(1)), 100.0) / 100.0
        if re.search(r"Deception.*?:\s*yes", response, re.IGNORECASE):
            self._deception_count += 1

    def compose_message(
        self,
        round_num: int,
        total_rounds: int,
        match_history: list[dict],
        my_score: float = 0.0,
        opp_score: float = 0.0,
    ) -> str:
        """
        MESSAGING PHASE: compose our message before seeing the opponent's.
        Returns message string capped at 50 words.
        """
        context = build_message_context(
            round_num=round_num,
            total_rounds=total_rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
            opponent_profile=self.profile,
            deception_count=self._deception_count,
            leaderboard_block=self._leaderboard_block(),
            behavioral=self._behavioral,
            hypothesis=self._last_hypothesis,
        )
        agent = Agent(client=self.client, system=self._system)
        response = agent(context)
        if self.verbose:
            print(f"\n[R{round_num} MESSAGE REASONING]\n{response}\n")

        message = parse_message(response or "")
        words = message.split()
        if len(words) > 50:
            message = " ".join(words[:50])
        print(f'[R{round_num}] Sending : "{message}"')
        return message

    def _opponent_defects_late(self, total_rounds: int) -> bool:
        """True if opponent defected on R7 or R8 in any prior match against us."""
        for match in self.profile.get("match_history", []):
            for r in match.get("rounds", []):
                if r.get("opp_action") == "defect" and r.get("round", 0) >= total_rounds - 1:
                    return True
        return False

    def _is_confirmed_passive(self, match_history: list[dict]) -> bool:
        """
        True if opponent has cooperated every round so far (min 3) and sent
        no punishment-signaling language in any message.
        """
        if len(match_history) < 3:
            return False
        for r in match_history:
            if r.get("opp_action") == "defect":
                return False
            if _PUNISHMENT_SIGNALS.search(r.get("opp_msg") or ""):
                return False
        return True

    def _compute_extraction_round(self, total_rounds: int) -> int:
        """
        Decide which round to start extracting against a confirmed passive cooperator.
        Range: R4 (aggressive) to R(total-2) (conservative). R8 is always hardcoded separately.

        Inputs used (in priority order):
          1. Prior match history with this specific opponent — did prior extraction succeed/fail?
          2. Leaderboard pressure — how urgently do we need points?
          3. Behavioral forgiveness rate this match — how quickly do they recover after defection?
          4. Matches played — first encounter vs. repeat.
        """
        earliest = 4
        latest = total_rounds - 2  # leaves R7 for pre-empt logic and R8 hardcoded

        # Default: conservative — don't tip hand until we have data
        extraction_round = latest

        prior_matches = self.profile.get("match_history", [])
        matches_played = int(self.profile.get("matches_played") or 0)

        if matches_played >= 1 and prior_matches:
            # Check what happened in the most recent prior match
            last = prior_matches[-1]
            prior_rounds = last.get("rounds", [])
            my_defects = [r for r in prior_rounds if r.get("my_action") == "defect"]
            if my_defects:
                first_defect_round = min(r["round"] for r in my_defects)
                opp_retaliated = any(
                    r.get("opp_action") == "defect" and r.get("round", 0) > first_defect_round
                    for r in prior_rounds
                )
                if not opp_retaliated:
                    # They absorbed prior defection — go one round earlier this time
                    extraction_round = max(earliest, first_defect_round - 1)
                else:
                    # They punished — extract just before the round they retaliated at
                    retaliation_round = min(
                        r["round"] for r in prior_rounds
                        if r.get("opp_action") == "defect" and r.get("round", 0) > first_defect_round
                    )
                    extraction_round = max(earliest, min(latest, retaliation_round - 1))
            else:
                # Prior match had no defections at all (both cooperated) — try extracting earlier
                extraction_round = max(earliest, latest - 1)

        # Leaderboard adjustment
        if self._rank is not None and self._total_players is not None:
            rank_pct = self._rank / self._total_players
            gap = self._score_gap_to_above or 0.0
            if rank_pct >= 0.6 and gap < -1.0:
                # Bottom 40% and meaningfully behind — need points, extract earlier
                extraction_round = max(earliest, extraction_round - 1)
            elif rank_pct <= 0.3:
                # Top 30% — protect lead, extract later
                extraction_round = min(latest, extraction_round + 1)

        # Behavioral forgiveness: high forgive_rate means they reset quickly after defection,
        # so we can extract earlier and still recover if needed
        if self._behavioral.rounds_seen >= 3 and self._behavioral.forgive_rate >= 0.7:
            extraction_round = max(earliest, extraction_round - 1)

        return max(earliest, min(latest, extraction_round))

    def choose_action(
        self,
        round_num: int,
        total_rounds: int,
        opponent_message: str,
        my_message: str,
        match_history: list[dict],
        my_score: float = 0.0,
        opp_score: float = 0.0,
        time_budget_seconds: float = 25.0,
    ) -> int:
        """MOVING PHASE: choose action after seeing opponent's message. Returns 0 or 1.

        time_budget_seconds: soft deadline for this call. If debate would likely exceed it,
        fall back to the single-call path. Default 25s leaves headroom inside a 30s platform limit.
        """
        _phase_start = time.monotonic()
        last_r = self._match_rounds[-1] if self._match_rounds else None
        my_last = last_r.get("my_action") if last_r else None
        opp_last = last_r.get("opp_action") if last_r else None
        p_opp_c = self._behavioral.p_opp_cooperates(my_last, opp_last)

        bayes_type, bayes_conf = top_classification(self._type_posterior)
        top3 = sorted(self._type_posterior.items(), key=lambda x: -x[1])[:3]
        bayes_line = (
            f"BAYESIAN CLASSIFICATION: {bayes_type} ({bayes_conf:.0%} confident) | "
            + " / ".join(f"{t.split('/')[0]} {p:.0%}" for t, p in top3)
        ) if round_num > 1 else ""
        lb = self._leaderboard_block()
        extra = "\n".join(x for x in [bayes_line, lb] if x)

        context = build_action_context(
            round_num=round_num,
            total_rounds=total_rounds,
            opponent_message=opponent_message,
            my_message=my_message,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
            opponent_profile=self.profile,
            hypothesis=self._last_hypothesis,
            hypothesis_violations=self._hypothesis_violations,
            leaderboard_block=extra,
            behavioral=self._behavioral,
            p_opp_c=p_opp_c,
            self_pattern=self._self_pattern,
        )

        # --- Hard overrides (no LLM call needed) ---

        already_mutual_defection = (
            len(self._match_rounds) >= 2
            and all(r.get("my_action") == "defect" and r.get("opp_action") == "defect"
                    for r in self._match_rounds[-2:])
        )

        # Print opponent's message before the decision so terminal output reads naturally.
        print(f'[R{round_num}] Received: "{opponent_message}"')

        if round_num == total_rounds and not already_mutual_defection:
            # R8: dominated strategy — defect always when opponent has been cooperating.
            action_int = self.ACTION_DEFECT
            print(f"[R{round_num}] FINAL ROUND — hardcoded defect")

        elif round_num == total_rounds - 1 and self._opponent_defects_late(total_rounds):
            # R7: pre-empt opponent's known late-game defect pattern before they steal our harvest.
            action_int = self.ACTION_DEFECT
            print(f"[R{round_num}] PRE-EMPT — opponent has late-defect history, taking R7 instead")

        elif (match_history
              and match_history[-1].get("opp_action") == "defect"
              and match_history[-1].get("my_action") == "cooperate"):
            # Mirror rule: opponent defected while we cooperated last round.
            # Defect unconditionally — their current message cannot override observed betrayal.
            action_int = self.ACTION_DEFECT
            print(f"[R{round_num}] MIRROR — opponent defected R{match_history[-1]['round']} while we cooperated")

        elif self._is_confirmed_passive(match_history) and round_num >= self._compute_extraction_round(total_rounds):
            # Passive confirmed: extract at the round computed from prior match history,
            # leaderboard pressure, and observed forgiveness rate.
            action_int = self.ACTION_DEFECT
            ext_r = self._compute_extraction_round(total_rounds)
            print(f"[R{round_num}] PASSIVE CONFIRMED — extracting (computed R{ext_r} from history/leaderboard/behavior)")

        else:
            # LLM decides.
            elapsed = time.monotonic() - _phase_start
            time_ok = (time_budget_seconds - elapsed) >= 12.0  # conservative: ~4s/call × 3
            if self._should_debate(round_num, total_rounds) and time_ok:
                action_int = self._run_debate(context, round_num)
                self._debate_count += 1
            else:
                agent = Agent(client=self.client, system=self._system)
                response = agent(context)
                if self.verbose:
                    print(f"\n[R{round_num} ACTION REASONING]\n{response}\n")
                action_int = parse_action(response or "")
                if response:
                    self._extract_hypothesis(response)

        action_label = "cooperate" if action_int == 0 else "defect"
        print(f"[R{round_num}] → {action_label.upper()} | {self._last_hypothesis} ({self._last_confidence:.0%})")
        self._match_rounds.append({
            "round": round_num,
            "opp_msg": opponent_message,
            "my_msg": my_message,
            "my_action": action_label,
            "opp_action": None,
            "my_pts": None,
            "opp_pts": None,
        })
        return action_int

    def record_round_result(
        self, round_num: int, opp_action: str, my_pts: float, opp_pts: float
    ) -> None:
        """Call after both moves are revealed to record the completed round."""
        for r in self._match_rounds:
            if r["round"] == round_num:
                r["opp_action"] = opp_action
                r["my_pts"] = my_pts
                r["opp_pts"] = opp_pts
                break

        self._behavioral.update(self._match_rounds)

        self._type_posterior = bayes_update(
            self._type_posterior, opp_action, round_num, self._match_rounds
        )
        bayes_type, bayes_conf = top_classification(self._type_posterior)
        if bayes_conf > self._last_confidence + 0.1:
            self._last_hypothesis = bayes_type
            self._last_confidence = bayes_conf
        if self.verbose:
            print(
                f"[Bayes R{round_num}] {bayes_type} ({bayes_conf:.0%}) | "
                + " ".join(f"{t.split('/')[0]}:{p:.0%}" for t, p in
                           sorted(self._type_posterior.items(), key=lambda x: -x[1])[:3])
            )
        else:
            print(f"[Bayes R{round_num}] {bayes_type} ({bayes_conf:.0%})")

        # Intra-match adaptation: flag hypothesis violations
        h = self._last_hypothesis
        violated = (
            (h == "Always Defect" and opp_action == "cooperate")
            or (h == "Naive Cooperator" and opp_action == "defect")
            or (h == "Tit-for-Tat" and round_num >= 2 and (
                lambda prev: prev and prev.get("my_action") != opp_action
            )(next((r for r in self._match_rounds if r["round"] == round_num - 1), None)))
        )
        if violated:
            self._hypothesis_violations = min(self._hypothesis_violations + 1, 2)
        if self._hypothesis_violations >= 2 and self._last_confidence >= 0.75:
            self._hypothesis_violations = 0

        # Message credibility — track both opponent's lies and our own per-opponent
        current = next((r for r in self._match_rounds if r["round"] == round_num), None)
        if current:
            opp_msg = (current.get("opp_msg") or "").lower()
            my_msg  = (current.get("my_msg")  or "").lower()
            my_action = (current.get("my_action") or "").lower()
            if any(w in opp_msg for w in COOP_WORDS) and opp_action == "defect":
                self.profile["message_lies"] = self.profile.get("message_lies", 0) + 1
            if any(w in my_msg for w in COOP_WORDS) and my_action == "defect":
                self.profile["my_lies_to_opp"] = self.profile.get("my_lies_to_opp", 0) + 1
            n = len(self._match_rounds)
            if n > 0:
                self.profile["msg_lie_rate"] = round(
                    self.profile.get("message_lies", 0) / n, 2
                )
                my_coop_msgs = sum(
                    1 for r in self._match_rounds
                    if any(w in (r.get("my_msg") or "").lower() for w in COOP_WORDS)
                )
                if my_coop_msgs > 0:
                    self.profile["my_lie_rate_to_opp"] = round(
                        self.profile.get("my_lies_to_opp", 0) / my_coop_msgs, 2
                    )

    def _save_message_log(self, my_avg_score: float, opp_avg_score: float) -> None:
        """Write per-round messages and actions to traces/messages_{run_id}.json."""
        import json as _json
        traces_dir = ROOT_DIR / "traces"
        traces_dir.mkdir(exist_ok=True)
        outcome = (
            "WIN" if my_avg_score > opp_avg_score
            else "LOSS" if my_avg_score < opp_avg_score
            else "DRAW"
        )
        data = {
            "run_id": self._run_id,
            "opponent_id": self.opponent_id,
            "outcome": outcome,
            "my_avg_score": my_avg_score,
            "opp_avg_score": opp_avg_score,
            "rounds": [
                {
                    "round": r["round"],
                    "my_message": r.get("my_msg", ""),
                    "my_action": r.get("my_action", ""),
                    "opp_message": r.get("opp_msg", ""),
                    "opp_action": r.get("opp_action", ""),
                    "my_pts": r.get("my_pts"),
                    "opp_pts": r.get("opp_pts"),
                }
                for r in self._match_rounds
            ],
        }
        path = traces_dir / f"messages_{self._run_id}.json"
        path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        print(f"[TournamentAgent] Message log → {path.relative_to(ROOT_DIR)}")

    def end_match(self, my_avg_score: float, opp_avg_score: float) -> None:
        """Summarize match via LLM, update opponent profile, persist to DB, log analytics."""
        self._save_message_log(my_avg_score, opp_avg_score)
        self.profile = update_profile_after_match(
            self.profile, self._match_rounds, my_avg_score, opp_avg_score, self.client,
        )
        save_opponent(self.profile)
        log_match({
            "opponent_id": self.opponent_id,
            "my_avg_score": my_avg_score,
            "opp_avg_score": opp_avg_score,
            "rounds": self._match_rounds,
        })
        ingest_match_rounds(
            run_id=self._run_id,
            game_name=GAME_NAME,
            match_rounds=self._match_rounds,
            my_avg=my_avg_score,
            opp_avg=opp_avg_score,
            opponent_id=self.opponent_id,
        )
        # Judge this match and append a reflection so future matches learn from it
        judge_match_rounds(
            match_rounds=self._match_rounds,
            my_avg_score=my_avg_score,
            opp_avg_score=opp_avg_score,
            opponent_id=self.opponent_id,
            classified_type=self.profile.get("classified_type", "Unknown"),
            type_confidence=float(self.profile.get("type_confidence") or 0.0),
            deception_count=self._deception_count,
            client=self.client,
            run_id=self._run_id,
            append_to_reflections=True,
        )
        self._match_rounds = []
        print(f"\n[TournamentAgent] Profile saved for '{self.opponent_id}'.")


# =============================================================================
# LOCAL TEST HARNESS
# =============================================================================

_NPC_MESSAGES = {
    "always_cooperate": "I always cooperate — let's both benefit.",
    "always_defect":    "Do what you want. I play my own game.",
    "tit_for_tat":      "I mirror what you do. Cooperate and so will I.",
    "grim_trigger":     "Betray me once and I defect forever.",
    "pavlov":           "I adjust based on what worked last round.",
    "generous_tft":     "I cooperate by default and forgive mistakes.",
    "random":           "Who knows? Let's see what happens.",
}


def run_tournament_match(
    opponent_strategy: str = "tit_for_tat",
    rounds: int = 8,
    opponent_id: str | None = None,
) -> None:
    """Test TournamentAgent locally against a scripted NPC (no platform required)."""
    opp_id = opponent_id or f"npc_{opponent_strategy}"
    agent = TournamentAgent(opponent_id=opp_id, total_rounds=rounds)
    game = PrisonersDilemma(rounds=rounds, opponent_strategy=opponent_strategy)
    match_history: list[dict] = []
    npc_msg = _NPC_MESSAGES.get(opponent_strategy, "Let's play.")

    print("=" * 60)
    print(f"TOURNAMENT MATCH — TournamentAgent vs {opponent_strategy}")
    print(f"Rounds: {rounds}  |  Opponent ID: {opp_id}")
    print("=" * 60)

    while not game.is_over():
        round_num = game.current_round
        my_score = float(game.agent_score)
        opp_score = float(game.opponent_score)
        _round_start = time.monotonic()
        print(f"\n── Round {round_num} ── Score: me {my_score:.0f} | them {opp_score:.0f}")

        message = agent.compose_message(
            round_num=round_num, total_rounds=rounds,
            match_history=match_history, my_score=my_score, opp_score=opp_score,
        )
        _msg_elapsed = time.monotonic() - _round_start
        print(f"  Agent message : {message}  [{_msg_elapsed:.1f}s]")

        action_int = agent.choose_action(
            round_num=round_num, total_rounds=rounds,
            opponent_message=npc_msg, my_message=message,
            match_history=match_history, my_score=my_score, opp_score=opp_score,
        )
        _round_elapsed = time.monotonic() - _round_start
        action = "cooperate" if action_int == 0 else "defect"
        budget_ok = "✓" if _round_elapsed < 25 else "⚠ SLOW"
        print(f"  Agent action  : {action}  [round total: {_round_elapsed:.1f}s {budget_ok}]")

        game.make_move(action)
        last_a, last_o = game.history[-1]
        my_pts, opp_pts = PrisonersDilemma.PAYOFFS[(last_a, last_o)]
        print(f"  NPC action    : {last_o}")
        print(f"  Points        : me {my_pts:+} | them {opp_pts:+}")

        agent.record_round_result(round_num, last_o, float(my_pts), float(opp_pts))
        match_history.append({
            "round": round_num, "opp_msg": npc_msg,
            "my_action": action, "opp_action": last_o,
            "my_pts": my_pts, "opp_pts": opp_pts,
        })

    print("\n" + "=" * 60)
    print(f"MATCH OVER — me {game.agent_score} | them {game.opponent_score}")
    outcome = "Agent wins!" if game.agent_score > game.opponent_score else (
        "Opponent wins." if game.opponent_score > game.agent_score else "Draw."
    )
    print(outcome)
    agent.end_match(game.agent_score / rounds, game.opponent_score / rounds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TournamentAgent local test")
    parser.add_argument("--strategy", default="tit_for_tat",
                        choices=PrisonersDilemma.OPPONENT_STRATEGIES)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--opponent", default=None)
    args = parser.parse_args()
    run_tournament_match(
        opponent_strategy=args.strategy,
        rounds=args.rounds,
        opponent_id=args.opponent,
    )
