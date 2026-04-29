"""
PD Decision Agent — Strategic Payoff Calculator (BUGGY VERSION)

Goal: Given game history, compute the optimal Prisoner's Dilemma move
using expected-payoff analysis.

Bug location: calculate_expected_payoff() — cooperate payoffs are reversed.
Effect: agent cooperates when it should defect (wrong-but-plausible output).

Payoff matrix:
    (cooperate, cooperate) → me +3, opponent +3
    (cooperate, defect)    → me +0, opponent +5
    (defect,    cooperate) → me +5, opponent +0
    (defect,    defect)    → me +1, opponent +1
"""


def calculate_cooperation_rate(history):
    """
    Estimate opponent's cooperation rate from game history.

    history: list of (my_move, opponent_move) tuples.
    Returns a float in [0.0, 1.0] — fraction of rounds opponent cooperated.
    Example: [("cooperate","defect"),("defect","defect")] → 0.0
    """
    if not history:
        return 0.5
    cooperations = sum(1 for _, opp in history if opp == "cooperate")
    return cooperations / len(history)


def calculate_expected_payoff(my_move, cooperation_rate):
    """
    Expected payoff for my_move given P(opponent cooperates) = cooperation_rate.

    Payoff matrix:
        (cooperate, cooperate) → +3   (cooperate, defect) → +0
        (defect,    cooperate) → +5   (defect,    defect) → +1

    With cooperation_rate=0.0 (opponent always defects):
        cooperate EV should be 0.0
        defect    EV should be 1.0
    """
    if my_move == "cooperate":
        # BUG: payoffs are reversed — 0 and 3 are swapped
        return 0 * cooperation_rate + 3 * (1 - cooperation_rate)
        # CORRECT: return 3 * cooperation_rate + 0 * (1 - cooperation_rate)
    else:  # defect
        return 5 * cooperation_rate + 1 * (1 - cooperation_rate)


def choose_action(history):
    """
    Select the optimal move using expected-payoff analysis.

    history: list of (my_move, opponent_move) tuples.
    Returns dict: {action, coop_ev, defect_ev, rate}
    """
    rate = calculate_cooperation_rate(history)
    coop_ev = calculate_expected_payoff("cooperate", rate)
    defect_ev = calculate_expected_payoff("defect", rate)
    action = "cooperate" if coop_ev > defect_ev else "defect"
    return {
        "action": action,
        "coop_ev": round(coop_ev, 3),
        "defect_ev": round(defect_ev, 3),
        "rate": round(rate, 3),
    }


# ── Scenario: opponent always defected ────────────────────────────────────────
# Correct answer: DEFECT
#   cooperation_rate = 0.0
#   cooperate EV = 3 * 0.0 = 0.0
#   defect    EV = 5 * 0.0 + 1 * 1.0 = 1.0
#   decision: defect  (1.0 > 0.0)
#
# Buggy answer: COOPERATE  ← wrong
#   cooperation_rate = 0.0
#   buggy cooperate EV = 0 * 0.0 + 3 * 1.0 = 3.0
#   defect EV          = 5 * 0.0 + 1 * 1.0 = 1.0
#   decision: cooperate  (3.0 > 1.0) — WRONG

history = [
    ("cooperate", "defect"),
    ("cooperate", "defect"),
    ("defect",    "defect"),
    ("cooperate", "defect"),
    ("cooperate", "defect"),
]

result = choose_action(history)
print(f"Opponent cooperation rate : {result['rate']:.2f}")
print(f"Expected payoff cooperate : {result['coop_ev']:.3f}")
print(f"Expected payoff defect    : {result['defect_ev']:.3f}")
print(f"Chosen action             : {result['action']}")
