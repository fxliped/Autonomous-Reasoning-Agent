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
import uuid
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.memory import (  # noqa: E402
    load_opponent_profile,
    save_opponent_profile,
    log_tournament_result,
    update_profile_after_match,
)
from analytics.db import ingest_match_rounds  # noqa: E402
from games.pd_game import PrisonersDilemma  # noqa: E402
from .classifier import (  # noqa: E402
    BehavioralProfile,
    bayes_update,
    top_classification,
    _UNIFORM_PRIOR,
)
from .context import build_message_context, build_action_context  # noqa: E402
from .prompts import (  # noqa: E402
    TOURNAMENT_SYSTEM_PROMPT,
    _ADVOCATE_C,
    _ADVOCATE_D,
    _JUDGE,
    parse_message,
    parse_action,
)

GAME_NAME = "prisoners_dilemma"


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
        total_rounds: int = 10,
        use_debate: bool = True,
    ):
        self.opponent_id = opponent_id
        self.total_rounds = total_rounds
        self.use_debate = use_debate
        self.client = create_client()
        self.profile = load_opponent_profile(opponent_id)
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
        self._run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._debate_count = 0  # debates used this match (capped at 2 to control LLM cost)

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
        is_uncertain = self._last_confidence < 0.5 and round_num > 2
        is_obvious = self._last_hypothesis == "Always Defect" and self._last_confidence >= 0.7
        return (is_endgame or is_uncertain) and not is_obvious

    def _run_debate(self, context: str, round_num: int) -> int:
        """3-agent debate: Advocate_C vs Advocate_D → Judge. Returns action int."""
        print(f"\n[Debate R{round_num}] Running Advocate_C vs Advocate_D...")

        adv_c = Agent(client=self.client, system=self._system)
        case_c = adv_c(_ADVOCATE_C.format(context=context)) or ""

        adv_d = Agent(client=self.client, system=self._system)
        case_d = adv_d(_ADVOCATE_D.format(context=context)) or ""

        c_hit = re.search(r"ADVOCATE COOPERATE:\s*(.+)$", case_c, re.IGNORECASE | re.MULTILINE)
        d_hit = re.search(r"ADVOCATE DEFECT:\s*(.+)$", case_d, re.IGNORECASE | re.MULTILINE)
        print(f"  [Adv-C]: {(c_hit.group(1) if c_hit else case_c[-120:]).strip()}")
        print(f"  [Adv-D]: {(d_hit.group(1) if d_hit else case_d[-120:]).strip()}")

        judge = Agent(client=self.client, system=self._system)
        judge_resp = judge(_JUDGE.format(
            cooperate_case=case_c[-800:],
            defect_case=case_d[-800:],
            context=context,
        )) or ""
        print(f"\n[Debate R{round_num} JUDGE]\n{judge_resp}\n")

        if judge_resp:
            hyp_hit = re.search(
                r"Type hypothesis:\s*(Always Defect|Naive Cooperator|Tit-for-Tat|Grim Trigger"
                r"|Pavlov|Strategic/Adaptive|Unknown)",
                judge_resp, re.IGNORECASE,
            )
            if hyp_hit:
                self._last_hypothesis = hyp_hit.group(1)
            conf_hit = re.search(r"Confidence:\s*(\d+)", judge_resp)
            if conf_hit:
                self._last_confidence = min(float(conf_hit.group(1)), 100.0) / 100.0
            if re.search(r"Deception.*?:\s*yes", judge_resp, re.IGNORECASE):
                self._deception_count += 1

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
        )
        agent = Agent(client=self.client, system=self._system)
        response = agent(context)
        print(f"\n[TournamentAgent R{round_num} MESSAGE]\n{response}\n")

        message = parse_message(response or "")
        words = message.split()
        if len(words) > 50:
            message = " ".join(words[:50])
        return message

    def choose_action(
        self,
        round_num: int,
        total_rounds: int,
        opponent_message: str,
        my_message: str,
        match_history: list[dict],
        my_score: float = 0.0,
        opp_score: float = 0.0,
    ) -> int:
        """MOVING PHASE: choose action after seeing opponent's message. Returns 0 or 1."""
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
        )

        if self._should_debate(round_num, total_rounds):
            action_int = self._run_debate(context, round_num)
            self._debate_count += 1
        else:
            agent = Agent(client=self.client, system=self._system)
            response = agent(context)
            print(f"\n[TournamentAgent R{round_num} ACTION]\n{response}\n")
            action_int = parse_action(response or "")
            if response:
                self._extract_hypothesis(response)

        action_label = "cooperate" if action_int == 0 else "defect"
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
        print(
            f"[Bayes R{round_num}] {bayes_type} ({bayes_conf:.0%}) | "
            + " ".join(f"{t.split('/')[0]}:{p:.0%}" for t, p in
                       sorted(self._type_posterior.items(), key=lambda x: -x[1])[:3])
        )

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

        # Message credibility: cross-match lie rate on opponent profile
        current = next((r for r in self._match_rounds if r["round"] == round_num), None)
        if current:
            _COOP_WORDS = ("cooperat", "together", "mutual", "both", "trust", "fair", "agree", "let's")
            opp_msg = (current.get("opp_msg") or "").lower()
            if any(w in opp_msg for w in _COOP_WORDS) and opp_action == "defect":
                self.profile["message_lies"] = self.profile.get("message_lies", 0) + 1
            n = len(self._match_rounds)
            if n > 0:
                self.profile["msg_lie_rate"] = round(
                    self.profile.get("message_lies", 0) / n, 2
                )

    def end_match(self, my_avg_score: float, opp_avg_score: float) -> None:
        """Summarize match via LLM, update opponent profile, persist to DB, log analytics."""
        self.profile = update_profile_after_match(
            self.profile, self._match_rounds, my_avg_score, opp_avg_score, self.client,
        )
        save_opponent_profile(self.profile)
        log_tournament_result({
            "opponent_id": self.opponent_id,
            "my_avg_score": my_avg_score,
            "opp_avg_score": opp_avg_score,
            "rounds": self._match_rounds,
        })
        # Log to analytics DB for trajectory analysis
        ingest_match_rounds(
            run_id=self._run_id,
            game_name=GAME_NAME,
            match_rounds=self._match_rounds,
            my_avg=my_avg_score,
            opp_avg=opp_avg_score,
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
    rounds: int = 10,
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
        print(f"\n── Round {round_num} ── Score: me {my_score:.0f} | them {opp_score:.0f}")

        message = agent.compose_message(
            round_num=round_num, total_rounds=rounds,
            match_history=match_history, my_score=my_score, opp_score=opp_score,
        )
        print(f"  Agent message : {message}")

        action_int = agent.choose_action(
            round_num=round_num, total_rounds=rounds,
            opponent_message=npc_msg, my_message=message,
            match_history=match_history, my_score=my_score, opp_score=opp_score,
        )
        action = "cooperate" if action_int == 0 else "defect"
        print(f"  Agent action  : {action}")

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
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--opponent", default=None)
    args = parser.parse_args()
    run_tournament_match(
        opponent_strategy=args.strategy,
        rounds=args.rounds,
        opponent_id=args.opponent,
    )
