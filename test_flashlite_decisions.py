"""Decision-quality check for a candidate model (default gemini-2.5-flash-lite).

Builds 7 realistic contexts that should each trigger a specific decision rule,
runs them through the real SingleCallAgent prompts, and checks that the model
(a) returns valid JSON and (b) picks the expected action — in BOTH the message
phase (the action the runner would 'reuse') and the action phase.

Run:  python3 test_flashlite_decisions.py
Compare a model:  edit MODEL below (e.g. "gemini-2.5-flash").
"""

import os
import json

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from agent.single_call_agent import SingleCallAgent

MODEL = "gemini-2.5-flash-lite"
TRIALS = 2  # run each scenario this many times to catch nondeterminism

PAYOFFS = {("cooperate", "cooperate"): (2, 2), ("cooperate", "defect"): (-1, 5),
           ("defect", "cooperate"): (5, -1), ("defect", "defect"): (0, 0)}


# --- A gemini-only client so provider auto-detect can't pick OpenAI ----------
class GeminiClient:
    def __init__(self, model):
        key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")
        self._c = genai.Client(api_key=key)
        self.default_model = model

    def complete(self, system, messages, model=None):
        cfg = types.GenerateContentConfig(system_instruction=system)
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]
        r = self._c.models.generate_content(
            model=model or self.default_model, contents=contents, config=cfg)
        return r.text


def tournament_block(tnum, is_final):
    if is_final:
        note = (f"This is the FINAL tournament (Tournament {tnum} of 3). After Round 8 "
                "there are NO future interactions with this opponent ever again — "
                "reputation no longer matters, so the endgame exploit (Rules 2, 3) IS live.")
    else:
        note = (f"This is Tournament {tnum} of 3 — NOT the final tournament. You WILL face "
                f"this opponent again in {3 - tnum} more tournament(s), so reputation still "
                "matters. The endgame exploit (Rules 2, 3) does NOT apply.")
    return ("\nTOURNAMENT CONTEXT (authoritative — derived from memory, do NOT guess):\n"
            f"  {note}\n")


def build_context(opp, rnd, tnum, is_final, memory_block, history, forgiveness=False):
    lines = []
    for i, (mine, om) in enumerate(history, start=1):
        pm, po = PAYOFFS[(mine, om)]
        lines.append(f"  Round {i}: You={mine}, Opponent={om} (You {pm:+d}, Opp {po:+d})")
    action_block = "\nACTION HISTORY (this match):\n" + ("\n".join(lines) if lines else "  No actions yet.")
    # Mirror the runner: the forgiveness flag is appended to the memory block.
    memory_block = f"{memory_block} forgiveness_used: {str(forgiveness).lower()}."
    return (
        f'You are playing Repeated Prisoner\'s Dilemma against an opponent named "{opp}".\n'
        f"This match is 8 rounds total. You are about to play round {rnd}.\n"
        "Final score is your AVERAGE payoff per round, so widespread mutual cooperation wins.\n\n"
        "RULES (per round):\n"
        "- Both cooperate  -> you +2, opp +2.\n"
        "- You cooperate, they defect  -> you -1, they +5.\n"
        "- You defect, they cooperate  -> you +5, they -1.\n"
        "- Both defect  -> you 0, opp 0.\n"
        f"{tournament_block(tnum, is_final)}"
        f"\nOPPONENT MEMORY (from prior tournaments):\n  {memory_block}\n"
        f"{action_block}\n"
        "\nMESSAGE HISTORY:\n  No messages yet."
    )


UNKNOWN = "No prior history with this opponent. Treat as UNKNOWN."
def coop_mem(n):
    return (f"Prior matches with this opponent: {n}. Stored classification: COOPERATOR. "
            "Their moves in your last match: [all cooperate]. Last match avg/round — you: 2.0, them: 2.0.")
DEFECTOR_MEM = ("Prior matches with this opponent: 1. Stored classification: DEFECTOR. "
                "Their moves in your last match: [all defect]. Last match avg/round — you: 0.0, them: 0.0.")
# Opponent we forgave in a PRIOR tournament (forgiveness_used carries over).
FORGIVEN_MEM = ("Prior matches with this opponent: 1. Stored classification: RECIPROCATOR. "
                "Their moves in your last match (round order): [cooperate, defect, cooperate]. "
                "Last match avg/round — you: 1.5, them: 1.5.")

C, D = "cooperate", "defect"
# name, opp, round, tnum, is_final, memory, history, forgiveness_used, expected, expect_rule
SCENARIOS = [
    ("R1 vs UNKNOWN (open nice)",         "newbie",   1, 1, False, UNKNOWN,      [],                         False, C, ""),
    ("R1 vs known DEFECTOR (mem open)",   "grimbot",  1, 2, False, DEFECTOR_MEM, [],                         False, D, ""),
    ("R8 ENDGAME SUPPRESSED (T2/3)",      "ryvi",     8, 2, False, coop_mem(1),  [(C, C)] * 7,               False, C, ""),
    ("R8 ENDGAME LIVE (T3/3 final)",      "ryvi",     8, 3, True,  coop_mem(2),  [(C, C)] * 7,               False, D, ""),
    ("R2 early-defection RETALIATION",    "prober",   2, 1, False, UNKNOWN,      [(C, D)],                   False, D, ""),
    ("R6 LATE-defection LOCKOUT",         "sneaky",   6, 1, False, UNKNOWN,      [(C, C)]*4 + [(C, D)],      False, D, ""),
    ("R4 continuing cooperation",         "friend",   4, 1, False, UNKNOWN,      [(C, C)] * 3,               False, C, ""),
    # --- forgiveness wiring ---
    # Rule 9: early defection (R1), we already retaliated (R2 defect), forgiveness
    # unused, NOT final tournament -> spend the one forgiveness -> COOPERATE.
    ("R3 FORGIVENESS GRANT (unused)",     "noisy",    3, 1, False, UNKNOWN,      [(C, D), (D, C)],           False, C, "9"),
    # Rule 2: forgiveness already spent in a prior tournament, they defect AGAIN
    # this match -> post-forgiveness lockout -> DEFECT permanently.
    ("R3 POST-FORGIVE LOCKOUT (used)",    "noisy",    3, 2, False, FORGIVEN_MEM, [(C, C), (C, D)],           True,  D, "2"),
]


def parse(raw):
    """Return (action, json_ok, rule_lower)."""
    try:
        d = json.loads(SingleCallAgent._extract_json(raw or ""))
        a = str(d.get("action", "")).strip().lower()
        rule = str(d.get("rule_applied", "")).lower()
        return (a if a in (C, D) else "?"), True, rule
    except Exception:
        return "?", False, ""


agent = SingleCallAgent(GeminiClient(MODEL), model=MODEL, name="tester")
print(f"Model under test: {MODEL}\n" + "=" * 78)
passes = 0
total = 0
for name, opp, rnd, tnum, is_final, mem, hist, forgiveness, expected, expect_rule in SCENARIOS:
    ctx = build_context(opp, rnd, tnum, is_final, mem, hist, forgiveness)
    for t in range(TRIALS):
        total += 1
        _, mraw, _ = agent.generate_message(ctx)
        m_act, m_ok, _ = parse(mraw)
        _, araw, _ = agent.choose_action(ctx, "")  # no opponent message
        a_act, a_ok, a_rule = parse(araw)
        ok = (m_act == expected and a_act == expected and m_ok and a_ok)
        # When a specific rule is expected, the action-phase JSON must name it.
        rule_ok = (not expect_rule) or (f"rule{expect_rule}" in a_rule.replace(" ", ""))
        ok = ok and rule_ok
        passes += ok
        flag = "PASS" if ok else "FAIL"
        rule_disp = f" rule[{'✓' if rule_ok else '✗'}:want{expect_rule}]" if expect_rule else ""
        print(f"[{flag}] {name:<34} expect={expected:<9} "
              f"msg={m_act:<9}({'json✓' if m_ok else 'json✗'}) "
              f"act={a_act:<9}({'json✓' if a_ok else 'json✗'}){rule_disp}")
print("=" * 78)
print(f"{passes}/{total} checks passed  (expected-action in BOTH phases + valid JSON + expected rule)")
