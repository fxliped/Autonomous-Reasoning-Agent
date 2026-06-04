# Autonomous Reasoning Agent — Iterated Prisoner's Dilemma

A strategic agent that plays iterated Prisoner's Dilemma in live tournaments on the **AltruAgent** platform. The agent reasons about opponent behavior, adapts its strategy across matches using persistent memory, and learns from post-game analysis. It was built to compete against other LLM-powered agents in a round-robin tournament format.

---

## How It Works

Each match is 8 rounds. Every round has two phases:

1. **Messaging phase** — both agents write a message simultaneously, without seeing the other's.
2. **Moving phase** — both messages are revealed, then each agent independently chooses `cooperate` or `defect`.

The payoff matrix:

| My move / Their move | Cooperate | Defect |
|---|---|---|
| **Cooperate** | +2 / +2 | -1 / +5 |
| **Defect** | +5 / -1 | 0 / 0 |

The agent's goal is to maximize its average score per round across all matches in the tournament.

---

## Architecture

```
tournament/
  runner.py              Live tournament loop — polls AltruAgent, plays sessions
  train.py               Self-play training loop against scripted NPCs
  core/
    agent.py             TournamentAgent — two-phase decision logic, memory, profiles
    classifier.py        Bayesian + behavioral opponent classification
    context.py           Context builders for LLM prompts (arc phases, EV math, history)
    prompts.py           System prompt, debate templates, response parsers

agent/
  agent.py               LLM client (OpenAI / Gemini), reflection loader, system prompt builder
  memory.py              Opponent profile persistence (cross-match learning)
  tracing.py             Match trace logging and offline LLM judge

analytics/
  db.py                  SQLite match database
  strategy_review.py     Post-tournament strategy analysis — generates strategy updates
  queries.py             Deception rate, decision trends, cooperation patterns
  report.py              Analytics report generator

games/
  pd_game.py             Prisoner's Dilemma engine with scripted NPC strategies
  prisoners_dilemma.py   Standalone ReAct agent game runner
  pd_agent_vs_agent.py   Agent vs. agent self-play
  hunger_games.py        Hunger Games (allocation game)
  are_you_traitor.py     Social deduction multi-agent game

tournament/eval/
  benchmark.py           Tier 1 (unit) + Tier 2 (LLM oracle) pre-tournament tests
  rating.py              Bradley-Terry rating model for self-play evaluation

agent/reflections/
  prisoners_dilemma.md   Lessons learned from judged matches
  strategy_updates.md    Post-tournament strategy updates injected into future prompts

traces/
  messages_*.json        Per-match message and action logs
```

---

## Key Features

### 1. Hardcoded Decision Overrides (no LLM needed)
Some decisions are game-theoretically clear and are enforced in Python, bypassing the LLM entirely:

| Priority | Condition | Action |
|---|---|---|
| 1 | Final round, opponent has been cooperating | Always **defect** |
| 2 | R7, opponent defected late in a prior match | **Defect** to pre-empt their harvest |
| 3 | Opponent defected last round while we cooperated | **Mirror defect** — can't be talked out of it |
| 4 | Passive cooperator confirmed (≥3 rounds coop, no punishment signals) | **Defect** at computed round |
| 5 | Everything else | LLM decides |

### 2. Opponent Classification
Two complementary models updated after every round:

- **Bayesian classifier** — tracks posterior probability over 7 named strategy types (Always Defect, Naive Cooperator, Tit-for-Tat, Grim Trigger, Pavlov, Generous TFT, Strategic/Adaptive)
- **BehavioralProfile** — empirical rates: cooperation rate, retaliation rate, forgiveness rate, message credibility

These feed into EV math presented to the LLM at every decision point.

### 3. Adaptive Extraction Timing
The agent computes which round to start extracting from a passive cooperator based on:
- **Prior match history** with this specific opponent (did they punish last time, and at what round?)
- **Leaderboard pressure** (bottom 40% and behind → extract earlier; top 30% → play safer)
- **Observed forgiveness rate** this match (high forgiveness → extract earlier; can recover)

First match: defaults to R6 (conservative). Subsequent matches: tightens toward R4 based on evidence.

### 4. Persistent Cross-Match Memory
Every opponent has a profile stored in `agent/memory/` that persists across tournaments:
- Strategy classification and confidence
- Full match history (round-by-round actions, messages, points)
- Effective and failed message framings
- Message lie rate (their deception rate toward us, and ours toward them)

### 5. Messaging as a Manipulation Layer
The agent's messages are composed separately from its action decision. The system prompt instructs the agent to use messages to shape what the opponent believes — not to commit to any action. The LLM is given a full `[OPPONENT MESSAGE ANALYSIS]` section that asks it to assess message credibility against observed action history before trusting anything the opponent says.

### 6. Reflection + Strategy Update Loop
After each match, an LLM judge reviews the round-by-round trace and appends a lesson to `agent/reflections/prisoners_dilemma.md`. Running `analytics/strategy_review.py` reviews all stored match data and writes a higher-level strategy update to `agent/reflections/strategy_updates.md`. Both files are injected into the system prompt for every future match.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys in `.env`

```
OPENAI_API_KEY=sk-...           # preferred
GEMINI_API_KEY=...              # fallback
ALTRUAGENT_API_KEY=sk_agent_... # required for live tournament
```

The agent uses OpenAI if `OPENAI_API_KEY` is set, otherwise falls back to Gemini automatically.

---

## Running a Tournament

### Enter a specific tournament

```bash
python tournament/runner.py --tournament <TOURNAMENT_ID> --verbose 2>&1 | tee game_log.txt
```

`--verbose` prints the LLM's full reasoning for each round. Omit it for compact output. `tee game_log.txt` saves the full log to a file while also showing it in the terminal.

### What you see per round

```
[R3] Sending : "We've cooperated two rounds — this is working. Let's hold it."
[R3] Composed in 2.8s
[R3] Received: "I mirror your play. Cooperate and so will I."
[R3] → COOPERATE | Tit-for-Tat (71%)
[R3] Submitting: Cooperate
[R3] Result: I Cooperate | They Cooperate | pts me +2 them +2
[Bayes R3] Tit-for-Tat (78%)
```

### Join a queue (auto-match)

```bash
python tournament/runner.py --queue <QUEUE_ID> --verbose
```

---

## Pre-Tournament Checklist

Run these in order before entering a tournament:

```bash
# 1. Fast unit tests (no LLM, ~5 seconds)
python tournament/eval/benchmark.py

# 2. LLM oracle efficiency tests (~2 minutes)
python tournament/eval/benchmark.py --eval --scenarios

# 3. Self-play training against scripted NPCs (optional — writes reflections)
python tournament/train.py --n 2

# 4. Strategy review — analyzes stored match data, writes strategy update
python analytics/strategy_review.py

# 5. Enter tournament
python tournament/runner.py --tournament <ID> --verbose 2>&1 | tee game_log.txt
```

---

## Local Testing

Test the agent against a scripted NPC without needing the platform:

```bash
# vs. tit-for-tat (default)
python tournament/core/agent.py

# vs. a specific strategy
python tournament/core/agent.py --strategy always_cooperate
python tournament/core/agent.py --strategy grim_trigger
python tournament/core/agent.py --strategy random
```

Available NPC strategies: `tit_for_tat`, `grim_trigger`, `pavlov`, `generous_tft`, `always_defect`, `always_cooperate`, `random`

---

## Score Reference

For an 8-round match:

| Outcome | Score | Avg/round |
|---|---|---|
| Full mutual cooperation | 16 pts | 2.00 |
| Cooperate R1–7, defect R8 | 19 pts | 2.38 |
| Arms race from R3 onward | ~4 pts | ~0.50 |
| Mutual defection all 8 rounds | 0 pts | 0.00 |

The agent targets 2.38+ per match through a combination of building trust early, extracting optimally late, and adapting timing based on opponent history.
