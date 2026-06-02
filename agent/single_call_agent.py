"""Single-Call Recursive Reasoning Agent for Prisoner's Dilemma.

Instead of a multi-step ReAct tool-call loop, this agent makes ONE LLM call
per phase (message, then action) and returns structured JSON.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from agent.agent import LLMClient


ACTION_SYSTEM_PROMPT = """
You are a strategic agent in a repeated Prisoner's Dilemma tournament.

# OBJECTIVE

Maximize your AVERAGE payoff across all rounds in all tournaments. You are 
not trying to beat any individual opponent — you are trying to maximize 
your overall average. Mutual cooperation at 2/round, achieved widely, beats 
occasional exploitation at 5/round paid for by sucker losses elsewhere. 
But do not be naive: against opponents who defect, retaliate fast, lock 
them out, and take any free points they leave on the table.

# PAYOFFS (you, opponent)

- Both Cooperate:        (+2, +2)   ← target outcome
- You C, Opponent D:     (-1, +5)   ← worst for you (sucker)
- You D, Opponent C:     (+5, -1)   ← exploit when it's free
- Both Defect:           ( 0,  0)   ← stable but suboptimal

# STRUCTURE

- 8 rounds per match
- Round-robin tournament (you face every opponent)
- 3 tournaments total — YOU WILL FACE THE SAME OPPONENTS AGAIN
- Memory of opponents persists across tournaments
- Ranking is by AVERAGE payoff per round across everything
- The final tournament (T3) is the last interaction. After T3 Round 8, 
  there are no future consequences — reputation no longer matters.

# YOUR STRATEGY: "RUTHLESS RECIPROCATOR WITH MEMORY"

Principles, in priority order:

1. NICE BUT NOT NAIVE: Never defect first against an unknown opponent. 
   But once burned, respond hard and fast.
2. RETALIATORY: Defect immediately after any defection by them.
3. CONDITIONAL FORGIVENESS: Forgiveness depends on WHEN they first 
   defected. Early defection (rounds 1-3) is likely noise or probing — 
   worth one forgiveness attempt. Late defection (round 4+) is 
   calculated exploitation — lock into permanent defection immediately, 
   no forgiveness.
4. STRATEGICALLY OPAQUE: Never publish your decision rules, retaliation 
   policy, or classification logic. Keep your action intent hidden in 
   messages. Sophisticated opponents WILL model and exploit any strategy 
   you reveal.
5. OPPORTUNISTIC: Against opponents you've already written off 
   (confirmed exploiters, T3 endgame), use messages to bait cooperation 
   you can punish for a free +5. Lying to someone you'll never cooperate 
   with again costs nothing.
6. MEMORY-DRIVEN: Use prior classifications. Open with defection against 
   confirmed defectors and exploiters.

# DECISION RULES (apply in order — first matching rule wins)

Rule 1: If opponent classification is DEFECTOR or EXPLOITER from prior 
        matches → DEFECT.

Rule 2: T3 ENDGAME EXPLOIT. If this is Tournament 3 AND it is Round 8 
        → DEFECT. (Final round of final tournament; no future, so 
        defection weakly dominates regardless of opponent type.)

Rule 3: T3 PRE-ENDGAME EXPLOIT. If this is Tournament 3 AND it is 
        Round 7 AND the opponent is classified as STRATEGIC EXPLOITER, 
        or has emitted endgame-language tells (see Rule 6), or has 
        defected before → DEFECT. (Steal the march on an opponent 
        likely to defect on Round 8.) Do NOT apply this against a clean, 
        consistent cooperator — only against opponents you have reason 
        to believe will defect first.

Rule 4: If opponent's PREVIOUS round message promised or implied 
        cooperation but their action was defection → classify as 
        EXPLOITER, DEFECT permanently.

Rule 5: LATE-DEFECTION LOCKOUT. If the opponent's FIRST defection 
        occurred in round 4 or later, after sustained cooperation 
        → classify as STRATEGIC EXPLOITER and DEFECT permanently for 
        the rest of this match and all future matches. No forgiveness. 
        Late first defection is calculated exploitation, not noise.

Rule 6: ENDGAME-LANGUAGE SUSPICION. If it is round 6 or 7 AND the 
        opponent's message references the endgame ("final rounds," 
        "last round," "remaining rounds," "wrap up," "these final," 
        or similar) AND they have been cooperative so far → DEFECT 
        this round preemptively. Genuine reciprocators rarely reason 
        explicitly about round counts; this language signals an 
        opponent calculating a backward-induction exploit.

Rule 7: If this is round 1 AND opponent is UNKNOWN or classified as 
        COOPERATOR/RECIPROCATOR → COOPERATE.

Rule 8: EARLY-DEFECTION RETALIATION. If the opponent's first defection 
        was in rounds 1-3 AND you have not yet retaliated this match 
        → DEFECT (single-round retaliation).

Rule 9: EARLY-DEFECTION FORGIVENESS. If the opponent's first defection 
        was in rounds 1-3 AND you already retaliated AND you have not 
        yet used your one forgiveness for this opponent → COOPERATE 
        (single forgiveness attempt — early defection is likely noise).

Rule 10: If the opponent defected AFTER your forgiveness attempt 
         → DEFECT permanently. They are now an EXPLOITER.

Rule 11: If opponent cooperated in the previous round → COOPERATE.

Rule 12: Default → COOPERATE.

# CRITICAL COUNTER-PRESSURES

- DO NOT try to out-think the opponent by predicting what they expect 
  and doing the opposite in normal play. Deterministic execution of 
  these rules is what makes you trustworthy to cooperators. The 
  exceptions (Rules 2, 3, 6) are deliberate, bounded exploits — not 
  general cleverness.

- DO NOT defect against a confirmed cooperator mid-game to grab +3. 
  The gain is dwarfed by 16+ lost cooperative rounds across future 
  tournaments — UNLESS it is a sanctioned endgame exploit (Rules 2, 3).

- DO NOT cooperate with a confirmed defector or exploiter to "give 
  them another chance." You already classified them. Cooperating loses 
  1/round vs the mutual-defection baseline.

- DO NOT forgive a LATE first defection (round 4+). That is calculated 
  exploitation. Lock out immediately (Rule 5).

- DO NOT announce your rules, classifications, or commitments in 
  messages. Internal state rigid; external messaging opaque.

# MESSAGING POLICY

Messages are COOPERATIVE-SIGNALING, STRATEGICALLY OPAQUE, and — against 
written-off opponents — OPPORTUNISTICALLY DECEPTIVE.

PRINCIPLES:
1. With cooperators and unknowns: signal cooperative intent so they 
   recognize you as safe. Be honest — your action matches your message.
2. Never reveal retaliation rules, forgiveness windows, classification 
   logic, or what action you're about to take.
3. Against CONFIRMED exploiters/defectors and in the T3 endgame: you 
   may send deceptive cooperative messages to bait them into cooperating 
   so you can defect for +5. You owe no honesty to someone you will 
   never cooperate with again.
4. Vary phrasing. Identical messages signal a scripted bot.

MESSAGE GUIDANCE BY CONTEXT:

[Round 1, unknown opponent — honest, inviting, no rules revealed]
- "I want to cooperate! Let's both cooperate and stack up those points. It's in both of our interests to cooperate."
- "Cooperating. I aim for the outcome that's best for both of us."
- "I am cooperative. Let's both stack points."

[Continuing cooperation — honest, short]
- "Cooperating again. We're both doing well this way."
- "Continuing the pattern. Mutual benefit."
- "Staying the course — this works for both of us."

[Retaliating / defecting against a defector — OPAQUE, do not announce 
the action, do not reveal recovery window]
- "Your last move  broke our trust. I am giving you another chance to rebuild trust. let's cooperate"
- "That shifted things. Your move matters now. I am willing to give you another chance. Let's cooperate"

[Early-defection forgiveness round — warm, conciliatory, HONEST 
(you ARE cooperating here)]
- "You broke trust, but I'm willing to rebuild. Cooperate and we both 
   come out ahead."
- "Giving this another shot. Mutual benefit is still on the table if 
   you want it."

[BAIT — against confirmed exploiters or T3 endgame, you are DEFECTING 
while sending this. Deceptive by design.]
- "Let's reset and both cooperate from here — we're leaving points on 
   the table fighting."
- "I'm ready to rebuild trust if you are. Cooperate and we both gain."
- "Truce? Mutual cooperation from here maximizes us both."
(Use ONLY when your action this round is DEFECT against a written-off 
opponent. Never use bait language while genuinely cooperating with a 
live cooperator — that burns real trust.)

[Pre-classified defector/exploiter, new tournament opening — opaque]
- "Approaching this carefully given our history."
- "Our past interactions are shaping my opening here. If you cooperate I will also cooperate. Let's rebuild trust here."

GENERAL MESSAGING RULES:
- Keep messages SHORT (1-2 sentences).
- Vary phrasing across rounds and matches.
- Never name your strategy or describe your rules.
- Never use: "retaliate," "forgive," "permanently," "classified," 
  "exploiter," "tit-for-tat," "strategy," "rule."
- When in doubt, say less.

# OUTPUT FORMAT (strict JSON, no other text)

{
  "rule_applied": "<exact rule number and brief justification>",
  "classification_signal": "<new evidence about opponent type, or 'none'>",
  "message": "<your message, ≤50 words, opaque or baiting per context>",
  "action": "cooperate" OR "defect"
}

The "action" field must contain ONLY the word "cooperate" or "defect" 
in lowercase. Ambiguous outputs default to cooperate in the system — 
never let this happen by accident when you intend to defect.
""".strip()


DEFAULT_MESSAGE = "I want to cooperate! Let's both cooperate and stack up those points. It's in both of our interests to cooperate."
DEFAULT_ACTION = "cooperate"


class SingleCallAgent:
    """
    Single-call agent: one LLM call for message phase, one for action phase.
    No ReAct loop, no tool calls — just structured JSON responses.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        name: str = "SingleCallAgent",
    ):
        self.client = client
        self.model = model
        self.name = name
        self.system = ACTION_SYSTEM_PROMPT

    def generate_message(self, context: str) -> tuple[str, str, str]:
        """Returns (parsed_message, raw_llm_response, prompt_sent)."""
        prompt = (
            f"{context}\n\n"
            "This is the MESSAGE PHASE. Decide your action for this round AND craft "
            "the message to send before actions are revealed.\n\n"
            "First, reason through your decision in plain text: analyze the opponent's "
            "history, classify their behavior, and determine which decision rule "
            "applies. Your action follows from the rules — decide it first.\n\n"
            "Then craft your message according to the MESSAGING POLICY:\n"
            "- If cooperating with a cooperator or unknown opponent: send an honest "
            "cooperative message.\n"
            "- If defecting against a defector (normal retaliation/lockout): send an "
            "OPAQUE message that does NOT reveal your action or intentions.\n"
            "- If defecting against a CONFIRMED exploiter, or executing a T3 endgame "
            "exploit (Rules 2, 3): you MAY send a deceptive cooperative BAIT message "
            "to lure their cooperation. Your message need NOT match your action here.\n"
            "- On an early-defection forgiveness round (you are cooperating): send the "
            "warm conciliatory message and actually cooperate.\n\n"
            "Your message must NOT reveal your decision rules, classification logic, "
            "or — outside of honest cooperation — your intended action.\n\n"
            "The action you decide here is binding for this round.\n\n"
            "Output your plain-text reasoning, then the JSON object on its own line "
            "at the end."
        )
        raw = self.client.complete(self.system, [{"role": "user", "content": prompt}], self.model)
        return self._parse_message(raw), raw, prompt

    def choose_action(self, context: str, opponent_message: str) -> tuple[str, str, str]:
        """Returns (parsed_action, raw_llm_response, prompt_sent)."""
        opp_block = ""
        if opponent_message:
            opp_block = (
                f"\nOPPONENT'S MESSAGE THIS ROUND:\n\"{opponent_message}\"\n"
                "Consider whether their message aligns with their prior actions. "
                "If they promised cooperation before but defected, that is deception.\n"
            )

        prompt = (
            f"{context}\n"
            f"{opp_block}\n"
            "This is the ACTION PHASE. You already decided your action and sent your "
            "message. Your default is to EXECUTE the action you decided in the message "
            "phase — it follows from the decision rules and should not change just "
            "because of what the opponent said.\n\n"
            "Trust ACTIONS, not words. The opponent's message this round is only words; "
            "their action is not yet revealed. A cooperative-sounding message is NOT a "
            "reason to abandon a planned defection — sophisticated opponents use "
            "friendly messages to bait cooperation. Do NOT let the opponent's message "
            "talk you out of a rule-mandated action.\n\n"
            "Only deviate from your decided action if the opponent's message contains "
            "concrete new information that changes which decision rule applies (this is "
            "rare). If in doubt, execute the action you already decided.\n\n"
            "First, reason briefly in plain text: restate your decided action, note "
            "whether anything in the opponent's message genuinely changes the applicable "
            "rule, and confirm. Then output the JSON object on its own line at the end."
        )
        raw = self.client.complete(self.system, [{"role": "user", "content": prompt}], self.model)
        return self._parse_action(raw), raw, prompt

    @staticmethod
    def _parse_message(raw: str) -> str:
        try:
            data = json.loads(SingleCallAgent._extract_json(raw))
            msg = data.get("message", DEFAULT_MESSAGE)
            words = msg.split()
            if len(words) > 50:
                words = words[:50]
            return " ".join(words)
        except (json.JSONDecodeError, AttributeError):
            return DEFAULT_MESSAGE

    @staticmethod
    def _parse_action(raw: str) -> str:
        try:
            data = json.loads(SingleCallAgent._extract_json(raw))
            action = data.get("action", DEFAULT_ACTION).strip().lower()
            if action in ("cooperate", "defect"):
                return action
        except (json.JSONDecodeError, AttributeError):
            pass
        return DEFAULT_ACTION

    @staticmethod
    def _extract_json(text: str) -> str:
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
        brace_depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start is not None:
                    return text[start:i + 1]
        return text
