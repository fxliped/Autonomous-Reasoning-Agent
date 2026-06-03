"""
Opponent classification for iterated Prisoner's Dilemma.

Two complementary models:

  BehavioralProfile — empirical rates derived directly from observed moves.
    No named-type assumptions. Works on LLM opponents that don't fit any archetype.
    Feeds directly into P(opp_C) for EV calculation.

  Bayesian classifier — named-type posterior over 7 canonical strategies.
    Useful as an early-round prior and for LLM-readable labels in context.
    Updated each round via bayes_update(); read via top_classification().

Both are updated in TournamentAgent.record_round_result() after each round.
"""


# =============================================================================
# BEHAVIORAL PROFILE (empirical, no named-type assumptions)
# =============================================================================

class BehavioralProfile:
    """
    Opponent behavioral features derived entirely from observed moves.

    Tracks four empirical rates that feed into P(opp_C) for EV math:
      coop_rate    = opp_cooperations / rounds_seen
      react_rate   = P(opp_D | my_prev_D)   — how retaliatory they are
      forgive_rate = P(opp_C | opp_prev_D)  — how quickly they return to cooperation
      msg_cred     = P(opp_C | opp_sent_cooperative_msg) — message reliability

    Degrades gracefully with few data points: all rates start at 0.5 (uninformative).
    """

    _COOP_WORDS = ("cooperat", "together", "mutual", "both", "trust", "fair", "agree", "let's")

    def __init__(self) -> None:
        self.coop_rate: float = 0.5
        self.react_rate: float = 0.5
        self.forgive_rate: float = 0.5
        self.msg_cred: float = 0.5
        self.rounds_seen: int = 0

    def update(self, history: list[dict]) -> None:
        """Recompute all features from completed round history."""
        completed = [r for r in history if r.get("opp_action")]
        self.rounds_seen = len(completed)
        if not completed:
            return

        coops = sum(1 for r in completed if r["opp_action"] == "cooperate")
        self.coop_rate = coops / len(completed)

        react_n = react_d = 0
        forgive_n = forgive_d = 0
        for i in range(1, len(completed)):
            prev, cur = completed[i - 1], completed[i]
            if prev.get("my_action") == "defect":
                react_d += 1
                if cur["opp_action"] == "defect":
                    react_n += 1
            if prev.get("opp_action") == "defect":
                forgive_d += 1
                if cur["opp_action"] == "cooperate":
                    forgive_n += 1
        self.react_rate = react_n / react_d if react_d else 0.5
        self.forgive_rate = forgive_n / forgive_d if forgive_d else 0.5

        cred_n = cred_d = 0
        for r in completed:
            msg = (r.get("opp_msg") or "").lower()
            if any(w in msg for w in self._COOP_WORDS):
                cred_d += 1
                if r["opp_action"] == "cooperate":
                    cred_n += 1
        self.msg_cred = cred_n / cred_d if cred_d else 0.5

    def p_opp_cooperates(self, my_last: str | None, opp_last: str | None) -> float:
        """
        Estimate P(opponent cooperates this round) given last round's actions.
        Uses empirical rates — no archetype assumptions.
        """
        if self.rounds_seen == 0:
            return 0.5
        p = self.coop_rate
        if my_last == "defect":
            # They may retaliate: blend toward P(opp_C | my_prev_D) = 1 - react_rate
            p = 0.4 * p + 0.6 * (1.0 - self.react_rate)
        if opp_last == "defect":
            # They just defected — will they return? Blend toward forgive_rate
            p = 0.5 * p + 0.5 * self.forgive_rate
        elif opp_last == "cooperate":
            # Cooperative momentum: slight upward push
            p = 0.7 * p + 0.3 * min(p + 0.15, 0.98)
        return max(0.02, min(0.98, p))

    def summary(self) -> str:
        if self.rounds_seen == 0:
            return "No behavioral data yet (round 1)."
        return (
            f"coop_rate={self.coop_rate:.0%}  "
            f"retaliation={self.react_rate:.0%}  "
            f"forgiveness={self.forgive_rate:.0%}  "
            f"msg_credibility={self.msg_cred:.0%}  "
            f"(n={self.rounds_seen})"
        )


# =============================================================================
# BAYESIAN NAMED-TYPE CLASSIFIER
# =============================================================================

# P(opponent plays cooperate | type, context)
# Each entry: (p_cooperate_round1, p_cooperate_later).
# None = context-dependent (computed in _likelihood).
_TYPE_LIKELIHOODS: dict[str, tuple[float, float | None]] = {
    "Always Defect":       (0.02, 0.02),
    "Always Cooperate":    (0.97, 0.97),
    "Tit-for-Tat":         (0.95, None),   # mirrors last agent move
    "Grim Trigger":        (0.95, None),   # cooperates until first defection by us
    "Pavlov":              (0.95, None),   # win-stay lose-shift
    "Generous TFT":        (0.95, 0.88),
    "Strategic/Adaptive":  (0.60, 0.55),
}

_UNIFORM_PRIOR: dict[str, float] = {t: 1.0 / len(_TYPE_LIKELIHOODS) for t in _TYPE_LIKELIHOODS}


def _likelihood(
    type_name: str,
    opp_action: str,
    round_num: int,
    history: list[dict],
) -> float:
    """P(opp_action | type) for one round given completed history."""
    cooperated = opp_action == "cooperate"
    p_base, p_late = _TYPE_LIKELIHOODS[type_name]

    if round_num == 1 or type_name in ("Always Defect", "Always Cooperate"):
        p_c = p_base
    elif type_name == "Tit-for-Tat":
        prev = next((r for r in history if r["round"] == round_num - 1), None)
        p_c = (0.92 if prev.get("my_action") == "cooperate" else 0.05) if prev else p_base
    elif type_name == "Grim Trigger":
        we_ever_defected = any(
            r.get("my_action") == "defect" for r in history if r["round"] < round_num
        )
        p_c = 0.05 if we_ever_defected else 0.93
    elif type_name == "Pavlov":
        prev = next((r for r in history if r["round"] == round_num - 1), None)
        if prev:
            opp_last = prev.get("opp_action", "cooperate")
            opp_pts = prev.get("opp_pts", 0)
            if (opp_pts or 0) >= 2:
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
    """Update P(type) given one observed opponent action. Returns normalized posterior."""
    posteriors = {t: p * _likelihood(t, opp_action, round_num, history) for t, p in prior.items()}
    total = sum(posteriors.values()) or 1e-9
    return {t: v / total for t, v in posteriors.items()}


def top_classification(posterior: dict[str, float]) -> tuple[str, float]:
    """Return (most_likely_type, confidence) from posterior distribution."""
    best = max(posterior, key=lambda t: posterior[t])
    return best, posterior[best]
