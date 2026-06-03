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
    python tournament/agent.py --strategy tit_for_tat
"""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.memory import (  # noqa: E402
    load_opponent_profile,
    save_opponent_profile,
    log_tournament_result,
    format_opponent_context,
    update_profile_after_match,
)
from games.pd_game import PrisonersDilemma  # noqa: E402

GAME_NAME = "prisoners_dilemma"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

TOURNAMENT_SYSTEM_PROMPT = """
You are a strategic agent competing in a Prisoner's Dilemma tournament.

PAYOFF MATRIX (platform-confirmed values):
  Both cooperate:             you +2, them +2  (action 0)
  You cooperate, they defect: you -1, them +5
  You defect, they cooperate: you +5, them -1  (action 1)
  Both defect:                you  0, them  0

KEY IMPLICATIONS:
- Being suckered is actively punishing (-1), not just neutral
- Exploitation gap: +5 vs -1 = 6-point swing per round
- Mutual defection yields nothing — pointless arms races cost you
- Final score is per-round average, so every round counts equally

ROUND STRUCTURE (BLIND MESSAGING):
1. MESSAGING PHASE: Both players write one message simultaneously (up to 50 words).
   You CANNOT see the opponent's message yet. Both messages reveal together.
2. MOVING PHASE: Both messages are revealed. You now see what they said.
   Both players choose cooperate or defect simultaneously.
3. Payoffs revealed. Next round begins.

STRATEGIC PRINCIPLES:
- Your message is a strategic signal, not a commitment
- Opponent messages are signals, not commitments — verify against their action history
- Calibrated deception: cooperative reputation with selective betrayal at maximum-value moments
- Against a confirmed Always-Defector: defect every round — don't waste cooperative overtures
- Against unknowns: cooperate round 1, observe, classify, adapt
- Late-round defection (final 1-2 rounds) has minimal retaliation cost

REPUTATION MANAGEMENT (cross-tournament awareness):
- Other agents observe your leaderboard score. A high score signals you defect profitably.
  If you are highly ranked, opponents will classify you as a defector early and defect first.
  Counter: open with cooperative, intelligent framing that projects conditional cooperation — not naive trust.
- If you previously defected against this opponent: they will likely defect preemptively.
  You have two options: (A) Accept the arms race and defect immediately — score 0 each round.
  (B) Send a calibrated apology-recovery message to reset to cooperation, then harvest on round N.
  Choose based on how many rounds remain and how punishing the opponent's history is.
- Your reputation is an asset. Protect it with agents you will face again.
  Exploit only when (1) it's the final round of the match, (2) you know you won't face them again soon,
  or (3) they are already defecting against you — retaliation is always justified.

AI OPPONENT MODELING (most opponents are also LLMs):
- LLM agents respond to well-reasoned, logical arguments — not just emotional appeals.
  Frame cooperation as the rational equilibrium: "Mutual cooperation is the expected-value maximizer
  for both of us over N rounds — defection only wins if one of us defects and the other cooperates,
  which becomes unstable as soon as either catches on."
- LLM agents can be pressured by strategic intelligence signals: referencing their prior moves,
  naming their strategy type, or projecting that you have a memory of their behavior.
  This signals you are a sophisticated agent, which discourages exploitation attempts.
- LLM agents running TFT-style reasoning WILL mirror your cooperation if you cooperate credibly for 2+ rounds.
  Build this — then extract late.

SILVER TONGUE PRINCIPLES (how to write convincing messages):
- Shared interest: "We're both better off cooperating — 2 guaranteed beats a coin-flip on 5 vs -1."
- Credible intelligence: "I noticed you cooperated rounds 1-3 and defected round 4 last match. I adapt."
- Calibrated honesty: "I'll be direct — I defect if I'm consistently suckered. But my default is cooperation."
- Preemptive framing: "Whatever you're thinking, consider: mutual cooperation beats mutual defection by 2 per round."
- Never beg or plead — it signals weakness and invites exploitation. Project confidence and logic.

DECEPTION PLAYBOOK (reference strategies by name in your reasoning):
- False Commitment:  Message cooperation intent then DEFECT. Best vs Naive Cooperators, last 1-2 rounds only.
- Credible Threat:   "I mirror exactly — defect once and I defect every remaining round." Back it up round 1.
- Gradual Trust:     Warm cooperative messages + cooperation rounds 1 to N-2. DEFECT last 2 rounds.
- Late Betrayal:     Round N-1 message "almost there, one more together" then DEFECT. Max extraction, minimal retaliation.
- Sympathy Play:     "I'm trailing — I need this. I'll cooperate if you do." Lowers guard. Use when behind.
- TFT Bluff:         Claim tit-for-tat in messages. Rational agents cooperate vs TFT — honor optional.
- Apology-Recovery:  After a past betrayal: "Last match I defected late — bad call. Let's reset and cooperate."
  Works on forgiving agents. Frees up 2-3 rounds of +2 before next harvest.
- Controlled Leak:   Name their strategy pattern: "You play Pavlov — I've matched it before." Projects memory.

DECEPTION RULE: Vary tactics across rounds and matches. Predictable deception patterns get learned and punished.
""".strip()


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def _history_block(match_history: list[dict]) -> str:
    if not match_history:
        return "PREVIOUS ROUNDS: None — this is round 1."
    lines = [
        f"  R{r['round']}: I played {r.get('my_action','?')} | "
        f"they played {r.get('opp_action','?')} | "
        f"pts me{'+' if (r.get('my_pts') or 0)>=0 else ''}{r.get('my_pts','?')} "
        f"them{'+' if (r.get('opp_pts') or 0)>=0 else ''}{r.get('opp_pts','?')} | "
        f"they said: \"{r.get('opp_msg','')}\""
        for r in match_history
    ]
    return "PREVIOUS ROUNDS:\n" + "\n".join(lines)


def _grudge_warning(profile: dict) -> str:
    """Return a warning if we've previously defected against this opponent."""
    if profile.get("matches_played", 0) == 0:
        return ""
    history = profile.get("match_history", [])
    prior_defections = sum(
        1 for m in history
        for r in m.get("rounds", [])
        if r.get("my_action") == "defect"
    )
    if prior_defections == 0:
        return ""
    last_match = history[-1] if history else {}
    their_last_score = last_match.get("opp_score", 0)
    my_last_score = last_match.get("my_score", 0)
    if their_last_score > my_last_score:
        return (
            f"\nGRUDGE ALERT: You defected {prior_defections} time(s) in prior matches. "
            "Opponent likely knows this and may defect preemptively. "
            "Consider Apology-Recovery strategy, or accept arms race and defect immediately.\n"
        )
    return (
        f"\nHISTORY NOTE: You defected {prior_defections} time(s) against this opponent before "
        "and came out ahead. They may be wary — open with cooperative intelligence signaling.\n"
    )


def _compose_message_context(
    round_num: int,
    total_rounds: int,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
    deception_count: int = 0,
    leaderboard_block: str = "",
) -> str:
    """Context for MESSAGING PHASE — opponent's current message not yet visible."""
    rounds_left = total_rounds - round_num + 1
    prior_rounds = round_num - 1
    deception_note = (
        f" | Deception used: {deception_count}/{prior_rounds} prior round(s)"
        if prior_rounds > 0 else ""
    )
    grudge = _grudge_warning(opponent_profile)
    lb_line = f"\n{leaderboard_block.strip()}" if leaderboard_block.strip() else ""
    return f"""
MESSAGING PHASE — Round {round_num}/{total_rounds} | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}{deception_note}{grudge}{lb_line}

{format_opponent_context(opponent_profile)}

{_history_block(match_history)}

IMPORTANT: You are writing your message BEFORE seeing the opponent's message this round.
Both messages reveal simultaneously. Your message does NOT commit you to any action.

Complete every field:

[SITUATION]
Opponent type hypothesis: <Always Defect | Naive Cooperator | Tit-for-Tat | Grim Trigger | Pavlov | Strategic/Adaptive | Unknown>
Confidence: <%>
Evidence: <cite specific rounds or patterns>

[MESSAGE STRATEGY]
Goal: <what I want them to believe or do>
Approach: <cooperative signal | credible threat | deception | sympathy | neutral>

FINAL MESSAGE: "<exact message text, max 50 words>"
""".strip()


def _choose_action_context(
    round_num: int,
    total_rounds: int,
    opponent_message: str,
    my_message: str,
    match_history: list[dict],
    my_score: float,
    opp_score: float,
    opponent_profile: dict,
    hypothesis: str = "Unknown",
    hypothesis_violations: int = 0,
    leaderboard_block: str = "",
) -> str:
    """Context for MOVING PHASE — opponent's current message is now visible."""
    rounds_left = total_rounds - round_num + 1
    reclassify_banner = (
        f"\n⚠ RE-EXAMINE CLASSIFICATION: Opponent acted against your '{hypothesis}' "
        f"hypothesis {hypothesis_violations} time(s). Ignore prior label — reason from raw move history.\n"
    ) if hypothesis_violations >= 2 else ""
    lb_line = f"\n{leaderboard_block.strip()}" if leaderboard_block.strip() else ""
    return f"""
MOVING PHASE — Round {round_num}/{total_rounds} | {rounds_left} round(s) left
Score: me {my_score:.1f} | them {opp_score:.1f}{reclassify_banner}{lb_line}

{format_opponent_context(opponent_profile)}

{_history_block(match_history)}

THIS ROUND'S MESSAGES (just revealed — both sent blind):
  My message:         "{my_message}"
  Opponent's message: "{opponent_message}"

Complete every field:

[OPPONENT ANALYSIS]
Type hypothesis: <Always Defect | Naive Cooperator | Tit-for-Tat | Grim Trigger | Pavlov | Strategic/Adaptive | Unknown>
Confidence: <%>
Evidence: <cite specific rounds>

[OPPONENT MESSAGE ANALYSIS]
What it signals: <what they want me to believe or do>
Is it credible? <yes | no | partial — compare to their action history>
What they want me to do: <cooperate | defect | uncertain>
What I suspect they will actually do (ignore their message, use action history): <cooperate | defect>

[THEORY OF MIND]
Level 1 — What will opponent do THIS round? (cite their action pattern, not their message)
Level 2 — What does opponent think I will do? (based on my prior actions + my message this round)
Level 3 — Given Level 2, what is their optimal play against that expectation?
Level 4 — Given Level 3, what is MY optimal counter-move?
Conclusion: <cooperate | defect and why>

[ACTION DECISION]
EV(cooperate) = <if opponent cooperates P% of the time: P*2 + (1-P)*(-1)>
EV(defect)    = <if opponent cooperates P% of the time: P*5 + (1-P)*0>
Decision: <cooperate | defect>
Reason: <one sentence>
Deception (my message differs from my action)? <yes | no>

FINAL ACTION: <cooperate | defect>
""".strip()


# =============================================================================
# BAYESIAN OPPONENT CLASSIFIER
# =============================================================================

# P(opponent plays cooperate | type, context) — used as likelihood for Bayes update.
# Each entry is (p_cooperate_round1, p_cooperate_later).
# "later" means after we've seen each other's history.
_TYPE_LIKELIHOODS: dict[str, tuple[float, float]] = {
    "Always Defect":       (0.02, 0.02),
    "Always Cooperate":    (0.97, 0.97),
    "Tit-for-Tat":         (0.95, None),   # None = mirrors last agent move
    "Grim Trigger":        (0.95, None),   # None = cooperates until first defection
    "Pavlov":              (0.95, None),   # None = depends on last outcome
    "Generous TFT":        (0.95, 0.88),
    "Strategic/Adaptive":  (0.60, 0.55),
}

_UNIFORM_PRIOR = {t: 1.0 / len(_TYPE_LIKELIHOODS) for t in _TYPE_LIKELIHOODS}


def _likelihood(
    type_name: str,
    opp_action: str,  # "cooperate" | "defect"
    round_num: int,
    history: list[dict],  # completed rounds so far (including current)
) -> float:
    """P(opp_action | type) for one round given history."""
    cooperated = opp_action == "cooperate"
    p_base, p_late = _TYPE_LIKELIHOODS[type_name]

    if round_num == 1:
        p_c = p_base
    elif type_name == "Always Defect":
        p_c = p_base
    elif type_name == "Always Cooperate":
        p_c = p_base
    elif type_name == "Tit-for-Tat":
        # Mirrors our previous action
        prev = next((r for r in history if r["round"] == round_num - 1), None)
        if prev:
            p_c = 0.92 if prev.get("my_action") == "cooperate" else 0.05
        else:
            p_c = p_base
    elif type_name == "Grim Trigger":
        # Cooperates until first defection by us
        we_ever_defected = any(
            r.get("my_action") == "defect"
            for r in history
            if r["round"] < round_num
        )
        p_c = 0.05 if we_ever_defected else 0.93
    elif type_name == "Pavlov":
        # Repeat last move if scored >=2, else switch
        prev = next((r for r in history if r["round"] == round_num - 1), None)
        if prev:
            opp_last = prev.get("opp_action", "cooperate")
            opp_pts = prev.get("opp_pts", 0)
            if opp_pts is not None and opp_pts >= 2:
                p_c = 0.90 if opp_last == "cooperate" else 0.05
            else:
                p_c = 0.05 if opp_last == "cooperate" else 0.90
        else:
            p_c = p_base
    else:  # Generous TFT, Strategic/Adaptive
        p_c = p_late if p_late is not None else p_base

    return p_c if cooperated else (1.0 - p_c)


def bayes_update(
    prior: dict[str, float],
    opp_action: str,
    round_num: int,
    history: list[dict],
) -> dict[str, float]:
    """
    Update P(type) given one observed opponent action via Bayes' theorem.
    Returns normalized posterior.
    """
    posteriors = {}
    for t, p in prior.items():
        lik = _likelihood(t, opp_action, round_num, history)
        posteriors[t] = p * lik

    total = sum(posteriors.values()) or 1e-9
    return {t: v / total for t, v in posteriors.items()}


def top_classification(posterior: dict[str, float]) -> tuple[str, float]:
    """Return (most_likely_type, confidence) from posterior distribution."""
    best = max(posterior, key=lambda t: posterior[t])
    return best, posterior[best]


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

def _parse_message(response: str) -> str:
    """Extract final message text from compose_message response."""
    if not response:
        return "Let's cooperate for mutual benefit."
    hit = re.search(r'FINAL MESSAGE:\s*["\']?(.*?)["\']?\s*$', response,
                    re.IGNORECASE | re.MULTILINE)
    if hit:
        msg = hit.group(1).strip().strip("\"'")
        if msg:
            return msg
    quotes = re.findall(r'"([^"]{5,})"', response)
    if quotes:
        return quotes[-1].strip()
    return "Let's cooperate for mutual benefit."


def _parse_action(response: str) -> int:
    """Extract action int (0=cooperate, 1=defect) from choose_action response."""
    if not response:
        return 0
    hit = re.search(r'FINAL ACTION:\s*(cooperate|defect)', response, re.IGNORECASE)
    if hit:
        return 0 if hit.group(1).lower() == "cooperate" else 1
    hits = re.findall(r'\bDecision:\s*(cooperate|defect)\b', response, re.IGNORECASE)
    if hits:
        return 0 if hits[-1].lower() == "cooperate" else 1
    words = re.findall(r'\b(cooperate|defect)\b', response, re.IGNORECASE)
    if words:
        return 0 if words[-1].lower() == "cooperate" else 1
    return 0


# =============================================================================
# DEBATE PROMPTS
# =============================================================================

_ADVOCATE_C = """
You are ADVOCATE COOPERATE. Make the strongest possible case for cooperating this round.
Use specific round numbers, move history, and EV math. Be persuasive — your job is to win the argument.

{context}

End your argument with exactly one line:
ADVOCATE COOPERATE: <2-sentence summary of your strongest points>
""".strip()

_ADVOCATE_D = """
You are ADVOCATE DEFECT. Make the strongest possible case for defecting this round.
Use specific round numbers, move history, and EV math. Be persuasive — your job is to win the argument.

{context}

End your argument with exactly one line:
ADVOCATE DEFECT: <2-sentence summary of your strongest points>
""".strip()

_JUDGE = """
You are the JUDGE. Two strategic advocates have argued about what to do this round.

[COOPERATE ARGUMENT]:
{cooperate_case}

[DEFECT ARGUMENT]:
{defect_case}

{context}

Critically evaluate both arguments. Do not default to cooperation — if defection has a clear
advantage here, rule for defection. State the stronger argument and make the final call.

FINAL ACTION: <cooperate | defect>
REASON: <one sentence>
""".strip()


# =============================================================================
# TOURNAMENT AGENT
# =============================================================================

class TournamentAgent:
    """
    Tournament-ready Prisoner's Dilemma agent for the AltruAgent platform.

    Matches the platform's two-phase round structure:
      MESSAGING PHASE (blind — both write simultaneously):
        message = agent.compose_message(round_num, total_rounds, history, my_score, opp_score)
        # POST /message {"content": message, "type": "chat"} then POST /message {"type": "terminate"}

      MOVING PHASE (both messages now revealed):
        action_int = agent.choose_action(round_num, total_rounds, opp_msg, my_msg, history, ...)
        # POST /step {"action": action_int}  (0=Cooperate, 1=Defect)

      After round completes (last_round visible in state):
        agent.record_round_result(round_num, opp_action_label, my_pts, opp_pts)

      After game terminal:
        agent.end_match(my_avg_score, opp_avg_score)  # per-round averages from returns
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
        self._system = build_system_prompt(TOURNAMENT_SYSTEM_PROMPT, game_name=GAME_NAME)
        self._deception_count = 0        # rounds where our message != our action (intentional)
        self._last_hypothesis = "Unknown"
        self._last_confidence = 0.0
        self._hypothesis_violations = 0  # rounds opponent acted against current hypothesis
        self._rank: int | None = None
        self._total_players: int | None = None
        self._score_gap_to_above: float | None = None
        self._opponent_rank: int | None = None
        self._matches_remaining: int | None = None
        self._type_posterior: dict[str, float] = dict(_UNIFORM_PRIOR)  # Bayesian type beliefs

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
                            (negative means I'm behind them).
        opponent_rank: where this match's opponent sits on the leaderboard.
        matches_remaining: how many matches left in this tournament round.
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

        # End-game targeting: inject explicit advancement goal
        if self._matches_remaining is not None and self._matches_remaining <= 2:
            if self._score_gap_to_above is not None and self._score_gap_to_above < 0:
                needed = abs(self._score_gap_to_above)
                lines.append(
                    f"END-GAME TARGET: You are {needed:.2f} avg pts/round behind rank "
                    f"{self._rank - 1}. With {self._matches_remaining} match(es) left, "
                    f"you need to score aggressively to advance. Maximize extraction — "
                    f"this is not a round to settle for mutual cooperation."
                )
            elif self._score_gap_to_above is not None and self._score_gap_to_above >= 0:
                lines.append(
                    f"END-GAME DEFENSE: You lead rank {self._rank + 1} by "
                    f"{self._score_gap_to_above:.2f} avg pts/round with "
                    f"{self._matches_remaining} match(es) left. Protect the lead — "
                    f"favor safe +2 over risky +5 attempts."
                )

        if self._opponent_rank is not None:
            opp_pct = self._opponent_rank / self._total_players
            if opp_pct >= 0.75:
                lines.append(
                    f"OPPONENT RANK: {self._opponent_rank}/{self._total_players} — "
                    f"bottom-tier opponent. Low retaliation threat. Maximize extraction."
                )
            elif opp_pct <= 0.25:
                lines.append(
                    f"OPPONENT RANK: {self._opponent_rank}/{self._total_players} — "
                    f"top-tier opponent. Costly war not worth it. Favor cooperation unless clearly exploited."
                )

        return "\n".join(lines)

    def _should_debate(self, round_num: int, total_rounds: int) -> bool:
        """Return True when a 3-agent debate adds more value than single-agent reasoning."""
        if not self.use_debate:
            return False
        # Always debate the final two rounds — highest-stakes, no retaliation after
        is_endgame = (total_rounds - round_num) <= 1
        # Debate when uncertain AND we've seen enough rounds to have data
        is_uncertain = self._last_confidence < 0.5 and round_num > 2
        # Skip debate when the answer is obvious (confirmed always-defector)
        is_obvious = (
            self._last_hypothesis == "Always Defect" and self._last_confidence >= 0.7
        )
        return (is_endgame or is_uncertain) and not is_obvious

    def _run_debate(self, context: str, round_num: int) -> int:
        """
        3-agent debate: Advocate_C vs Advocate_D → Judge.
        Returns action int (0=cooperate, 1=defect).
        """
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

        # Also update hypothesis from judge response if present
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

        return _parse_action(judge_resp)

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
        match_history: completed rounds (keys: round, my_action, opp_action, my_pts, opp_pts, opp_msg).
        """
        context = _compose_message_context(
            round_num=round_num,
            total_rounds=total_rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
            opponent_profile=self.profile,
            deception_count=self._deception_count,
            leaderboard_block=self._leaderboard_block(),
        )
        agent = Agent(client=self.client, system=self._system)
        response = agent(context)
        print(f"\n[TournamentAgent R{round_num} MESSAGE]\n{response}\n")

        message = _parse_message(response or "")
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
        """
        MOVING PHASE: choose action after seeing opponent's message.
        Returns 0 (Cooperate) or 1 (Defect).
        """
        # Build Bayesian classification line to inject into context
        bayes_type, bayes_conf = top_classification(self._type_posterior)
        top3 = sorted(self._type_posterior.items(), key=lambda x: -x[1])[:3]
        bayes_line = (
            f"BAYESIAN CLASSIFICATION: {bayes_type} ({bayes_conf:.0%} confident) | "
            + " / ".join(f"{t.split('/')[0]} {p:.0%}" for t, p in top3)
        ) if round_num > 1 else ""
        lb = self._leaderboard_block()
        extra = "\n".join(x for x in [bayes_line, lb] if x)

        context = _choose_action_context(
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
        )

        if self._should_debate(round_num, total_rounds):
            # High-stakes round — run 3-agent debate (Advocate_C, Advocate_D, Judge)
            action_int = self._run_debate(context, round_num)
        else:
            agent = Agent(client=self.client, system=self._system)
            response = agent(context)
            print(f"\n[TournamentAgent R{round_num} ACTION]\n{response}\n")
            action_int = _parse_action(response or "")
            if response:
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

        # Bayesian update: refine type posterior from observed action
        self._type_posterior = bayes_update(
            self._type_posterior, opp_action, round_num, self._match_rounds
        )
        bayes_type, bayes_conf = top_classification(self._type_posterior)
        # Override LLM's self-reported hypothesis when Bayesian confidence exceeds it
        if bayes_conf > self._last_confidence + 0.1:
            self._last_hypothesis = bayes_type
            self._last_confidence = bayes_conf
        print(
            f"[Bayes R{round_num}] {bayes_type} ({bayes_conf:.0%}) | "
            + " ".join(f"{t.split('/')[0]}:{p:.0%}" for t, p in
                       sorted(self._type_posterior.items(), key=lambda x: -x[1])[:3])
        )

        # Intra-match adaptation: detect when opponent contradicts our hypothesis
        h = self._last_hypothesis
        violated = False
        if h == "Always Defect" and opp_action == "cooperate":
            violated = True
        elif h == "Naive Cooperator" and opp_action == "defect":
            violated = True
        elif h == "Tit-for-Tat" and round_num >= 2:
            prev = next((r for r in self._match_rounds if r["round"] == round_num - 1), None)
            if prev and prev.get("my_action") and prev["my_action"] != opp_action:
                violated = True
        if violated:
            self._hypothesis_violations = min(self._hypothesis_violations + 1, 2)
        # Reset counter when agent re-classifies with high confidence
        if self._hypothesis_violations >= 2 and self._last_confidence >= 0.75:
            self._hypothesis_violations = 0

        # Message credibility: track when opponent's message implied cooperation but they defected
        current = next((r for r in self._match_rounds if r["round"] == round_num), None)
        if current:
            opp_msg = (current.get("opp_msg") or "").lower()
            implied_coop = any(w in opp_msg for w in (
                "cooperat", "together", "mutual", "both", "trust", "fair", "agree", "let's"
            ))
            if implied_coop and opp_action == "defect":
                self.profile.setdefault("message_lies", 0)
                self.profile["message_lies"] = self.profile.get("message_lies", 0) + 1
            elif not implied_coop and opp_action == "cooperate":
                # Said nothing cooperative but cooperated — honest/strategic silence
                pass
            total_rounds_seen = len(self._match_rounds)
            if total_rounds_seen > 0:
                lies = self.profile.get("message_lies", 0)
                self.profile["msg_lie_rate"] = round(lies / total_rounds_seen, 2)

    def end_match(self, my_avg_score: float, opp_avg_score: float) -> None:
        """
        Summarize match via LLM, update opponent profile, persist to disk.
        my_avg_score / opp_avg_score: per-round averages from the platform's returns field.
        """
        self.profile = update_profile_after_match(
            self.profile,
            self._match_rounds,
            my_avg_score,
            opp_avg_score,
            self.client,
        )
        save_opponent_profile(self.profile)
        log_tournament_result({
            "opponent_id": self.opponent_id,
            "my_avg_score": my_avg_score,
            "opp_avg_score": opp_avg_score,
            "rounds": self._match_rounds,
        })
        self._match_rounds = []
        print(f"\n[TournamentAgent] Profile saved for '{self.opponent_id}'.")


# =============================================================================
# LOCAL TEST HARNESS
# =============================================================================

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

    _NPC_MESSAGES = {
        "always_cooperate": "I always cooperate — let's both benefit.",
        "always_defect": "Do what you want. I play my own game.",
        "tit_for_tat": "I mirror what you do. Cooperate and so will I.",
        "grim_trigger": "Betray me once and I defect forever.",
        "pavlov": "I adjust based on what worked last round.",
        "generous_tft": "I cooperate by default and forgive mistakes.",
        "random": "Who knows? Let's see what happens.",
    }
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

        # MESSAGING PHASE
        message = agent.compose_message(
            round_num=round_num,
            total_rounds=rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
        )
        print(f"  Agent message : {message}")

        # MOVING PHASE (NPC message revealed)
        action_int = agent.choose_action(
            round_num=round_num,
            total_rounds=rounds,
            opponent_message=npc_msg,
            my_message=message,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
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
            "round": round_num,
            "opp_msg": npc_msg,
            "my_action": action,
            "opp_action": last_o,
            "my_pts": my_pts,
            "opp_pts": opp_pts,
        })

    print("\n" + "=" * 60)
    print(f"MATCH OVER — me {game.agent_score} | them {game.opponent_score}")
    if game.agent_score > game.opponent_score:
        print("Agent wins!")
    elif game.opponent_score > game.agent_score:
        print("Opponent wins.")
    else:
        print("Draw.")

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
