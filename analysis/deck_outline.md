# Secret Agent — STATS 263 Final Presentation Outline

> Paste this into Claude.ai and ask it to render a slide deck. Numbers and rules
> are pulled from the actual codebase and the 18 tournament matches vs the 6 real
> opposing teams (our own test agents excluded). The results chart is
> `analysis/per_opponent_results.png` (drop it on the Results slide).

---

## Slide 1 — Overview: What We Built
**Title:** Secret Agent: a game-theoretic LLM agent for Repeated Prisoner's Dilemma

A reasoning agent that plays 8-round Repeated PD on the AltruAgent platform, built on four pillars:
- **Game-theoretic core** — decisions follow an explicit, ordered rule ladder grounded in Axelrod's tournament findings, not free-form LLM vibes.
- **Two-layer memory** — *within-game* state (this match's move history, threat signal, forgiveness used) + *cross-game opponent memory* (persistent per-opponent classification that survives across matches and tournaments).
- **Conditional opponent modeling** — every opponent is deterministically classified (COOPERATOR / DEFECTOR / EXPLOITER) and that label conditions how we open and whether we retaliate.
- **Reasoning traces + LLM-as-judge** — every decision is logged with its rule and rationale; an LLM judge reviews traces to localize errors and improve between matches.
We already have a slide for this, so use your judgement in combining both or using whatever you think is best. 


---

## Slide 2 — Agent Architecture
**Title:** Architecture: one reasoning loop, two memories, deterministic guardrails

**Diagram (left→right flow):**
```
 Game state ─┐
 Opponent    ├─►  CONTEXT BUILDER ─►  LLM (Gemini 2.5 Flash) ─►  parsed decision ─►  MOVE
 memory ─────┤      (injects:           "reason over the          (action + rule)
 In-match    │       rules, memory,      rules; pick a move")           │
 threat ─────┘       threat signal)                                     ▼
                                                              TRACE LOGGER ─► LLM JUDGE
        ▲                                                                       │
        └───────────── opponent memory updated after each match ◄──────────────┘
```

**Components to label:**
- **Context builder** — assembles the prompt: payoff matrix, the 11-rule ladder, opponent memory block, and a *deterministically computed* in-match threat signal (unprovoked-defection rate).
- **Two-phase round** — blind **message phase** (send one ≤50-word message) then **move phase**. *Reuse-the-move* optimization: the message-phase reasoning already decides the binding action, so the move is submitted with no second LLM call — avoids racing the ~30s move timeout.
- **Two-layer memory:** within-game (history, threat, forgiveness-used) vs. cross-game (sticky opponent classification, persisted to disk).
- ** LLM as a judge** to improve agent after every game; 
We already have a slide for this, so use your judgement in combining both or using whatever you think is best. 
---

## Slide 3 — Axelrod's Principles → Our Decision Rules
**Title:** From Axelrod's four properties to an executable rule ladder

| Axelrod property | How we implement it |
|---|---|
| **Nice** (never defect first) | Rule 7: round-1 vs unknown → cooperate. Default (Rule 11) → cooperate. |
| **Retaliatory** (punish defection) | Rule 8: defect immediately after an early defection. Rule 1: open hostile vs known DEFECTOR/EXPLOITER. |
| **Forgiving** (don't hold grudges forever) | Rule 9: one-time forgiveness for an *early* (round 1–2) defection — likely noise. Rule 10: cooperate if they cooperated last round. |
| **Clear** (be legible/predictable) | Consistent, rule-driven play so cooperative opponents can learn to trust us — but our rules themselves stay strategically opaque (Rule 4). |

**The 11-rule ladder (first match wins):** classification-based opening (1) → post-forgiveness lockout (2) → final-tournament endgame exploit (3) → deceptive-message punisher (4) → late-defection lockout (5) → endgame-language suspicion (6) → round-1 opening (7) → early-defection retaliation (8) → early-defection forgiveness (9) → reciprocate-cooperation (10) → default cooperate (11).

We already have a slide for this, so use your judgement in combining both or using whatever you think is best. 
---

## Slide 4 — Tracing & LLM-as-Judge
**Title:** Closed-loop improvement: trace every decision, judge every match

- **Every step is logged** with its prompt, the model's reasoning, the rule it applied, and a classification signal → a complete, replayable trace per match.
- **LLM-as-judge** reviews each trace to localize errors (wrong rule fired, misclassification, missed move) and feeds reflections back into how we play the next match.
- **What this bought us:** between-match adaptation. Opponent classifications persist and *escalate* (sticky-hostile: once a DEFECTOR, always treated as one), and we caught/fixed an infrastructure bug *during* the live tournament (the move-window auto-cooperate — see Slide 6).

> **Honest scope:** the judge runs **post-match**, so this is **match-to-match** learning, not within-game real-time pivoting. The adaptation lever during a match is the deterministic threat signal + memory, not the judge.


---

## Slide 5 — Results: The Numbers
**Title:** Tournament results — net positive vs. the field

**[INSERT IMAGE: analysis/per_opponent_results.png]**

- **Placements (7 teams, 3 tournaments):** 2nd in T1 · tied 5–7th in T2 · 4th in T3.
- **Head-to-head vs. the 6 real teams (18 matches, 3 each): 5 W – 8 T – 5 L.**
- **Net margin: +0.108 avg payoff/round** over the field (our 1.032 vs. their 0.924).

| Opponent | Matches | W-T-L | Our avg | Opp avg | Margin |
|---|---|---|---|---|---|
| DeepSeek - RyVi | 3 | 1-2-0 | 1.04 | −0.21 | **+1.25** |
| liars dice test B | 3 | 2-0-1 | 1.21 | 0.66 | **+0.55** |
| AgenticArchitects | 3 | 1-1-1 | 1.47 | 1.47 | +0.00 |
| Amber-Agent2 | 3 | 1-1-1 | 1.01 | 1.36 | **−0.35** |
| koconnor_test | 3 | 0-2-1 | 0.87 | 1.27 | **−0.40** |
| Tit4Tat-Agent | 3 | 0-2-1 | 0.60 | 1.00 | **−0.40** |


---

## Slide 6 — Learnings: Strengths & the Two-Bug Diagnosis
**Title:** What worked, and two correctable weaknesses (one timing, one strategy)

**Strengths**
- Net positive vs. the field (**+0.108/round**) and we won our exploit opportunities — DeepSeek-RyVi (+1.25) and liars dice (+0.55).
- Stable cooperation with cooperators (multiple clean 2.0/2.0 results) — the "nice" core held.
- Deterministic guardrails (threat signal, sticky classification) kept us from being suckered repeatedly.

**Our three negative matchups split into just two root causes — and that distinction is the point:**

| | Amber-Agent2 (−0.35) | Tit4Tat-Agent (−0.40) **and** koconnor_test (−0.40) |
|---|---|---|
| **Type** | Timing / infrastructure | Strategy |
| **Cause** | 6 move-window misses → platform auto-cooperated us; planned-defect silently became cooperate. Plus first-contact tax of a nice opening vs a cold defector. | Classified a **reciprocator** as a permanent DEFECTOR → opened hostile → a tit-for-tat mirrors us into a DD grind we can't escape. |
| **Evidence** | Bug only ever flips planned-defect → cooperate (platform auto-cooperates on timeout); all 6 misses were vs Amber. | **Two independent witnesses.** Both opponents: read as COOPERATOR → we scored **2.0/2.0**; once flipped to DEFECTOR → **0.0 or worse**. ~2 pts/round swing, entirely self-inflicted. |
| **Fix** | **Done** — tightened poll cadence (move-window guard) + 429 backoff. | **Strategy change:** detect reciprocity (opp mirrors my last move) → lock to cooperation; never apply the sticky-hostile label to a mirror. |

**Takeaway:** the strategy bug isn't a fluke — the *same* misclassification cost us both Tit4Tat and koconnor. Fix the move window (shipped) and stop going hostile on reciprocators, and those three matchups alone swing well over **1 pt/round** in our favor — turning a middle-of-the-pack finish into a contender.

