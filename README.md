# Autonomous Reasoning Agent

This repo contains our team's game-playing agents. The section below documents
the **AltruAgent tournament agent** (Sarthak's branch): an LLM agent that plays
Repeated Prisoner's Dilemma on the class competition platform. The original
ReAct / Tracer framework that the rest of the project is built on is preserved
further down under [Background](#background-react-agent--tracer-framework).

---

# 🤖 AltruAgent Tournament Agent

An LLM-driven agent that connects to the AltruAgent platform over HTTP, joins a
queue, and plays every match of the auto-spawned round-robin tournament. Each
match is **8 rounds** of Repeated Prisoner's Dilemma with a blind messaging
phase before each move. Final score is **average payoff per round**, so the goal
is wide mutual cooperation, punishing only those who defect.

**Payoffs (you, opponent):** both cooperate `(+2,+2)` · you cooperate/they defect
`(-1,+5)` · you defect/they cooperate `(+5,-1)` · both defect `(0,0)`.

## Strategy: "Ruthless Reciprocator with Memory"

The agent is **nice but not naive**: it never defects first against an unknown
opponent, but once burned it retaliates immediately, and it remembers betrayers
across tournaments. Core principles:

1. **Nice but not naive** — never defect first against an unknown opponent.
2. **Retaliatory** — defect immediately after any defection.
3. **Conditional forgiveness** — *when* they first defected matters. An **early**
   defection (rounds 1–3) is treated as noise/probing and earns **one** forgiveness
   attempt. A **late** defection (round 4+) is treated as calculated exploitation
   and triggers **permanent lockout** — no forgiveness.
4. **Strategically opaque** — messages never reveal the agent's rules,
   classifications, or the move it is about to make.
5. **Memory-driven** — prior classifications decide the opening move. Against a
   known defector/exploiter, the agent opens with defection from round 1.

### Decision rules (evaluated in order, first match wins)

| # | Condition | Action |
|---|-----------|--------|
| 1 | Opponent is a known `DEFECTOR`/`EXPLOITER` from memory | **Defect** |
| 2 ⚠️ | Final-tournament endgame (Round 8) | **Defect** *(under review — see note)* |
| 3 ⚠️ | Final-tournament Round 7 vs a likely-defector | **Defect** *(under review)* |
| 4 | Their last message implied cooperation but they defected | **Defect permanently** (→ `EXPLOITER`) |
| 5 | First defection in round 4+ after sustained cooperation | **Defect permanently** (→ `EXPLOITER`) |
| 6 | Round 6–7, cooperative so far, but they use "endgame" language | **Defect** (pre-empt backward-induction) |
| 7 | Round 1 vs `UNKNOWN`/`COOPERATOR`/`RECIPROCATOR` | **Cooperate** |
| 8 | Early defection (1–3), not yet retaliated | **Defect** (one-round retaliation) |
| 9 | Early defection, already retaliated, forgiveness unused | **Cooperate** (one forgiveness) |
| 10 | They defected *after* a forgiveness attempt | **Defect permanently** (→ `EXPLOITER`) |
| 11 | They cooperated last round | **Cooperate** |
| 12 | Default | **Cooperate** |

> ⚠️ **Rules 2 & 3 are under review.** They assume a fixed "3 tournaments, T3 is
> the last" structure and tell the agent to defect (and send a baiting message)
> on the final round. The agent currently has **no runtime signal** for which
> tournament it is in, so it defaults to assuming the endgame — meaning it
> defects on Round 8 of *every* match. Whether this is correct depends on the
> real tournament structure, which we are confirming with the professor before
> finalizing. See `ALTRUAGENT_GUIDE.md`.

## Opponent classification

After every match the agent labels the opponent from their observed moves
(`agent/opponent_memory.py`):

| Label | How it's decided |
|-------|------------------|
| `COOPERATOR` | Never defected. |
| `EXPLOITER` | Defected while we cooperated, *after* 3+ rounds of mutual cooperation — a planned mid/late betrayal. |
| `DEFECTOR` | Defected on more than 50% of rounds. |
| `RECIPROCATOR` | Defected only in response (anything not matching the above). |

`DEFECTOR` and `EXPLOITER` are **hostile** labels and are **sticky**: once an
opponent earns one, a single well-behaved later match will not downgrade them.
This is what lets the agent open hard against a known betrayer.

## Cross-tournament memory

The platform reuses the same opponents across tournaments, so the agent keeps a
per-opponent record in `altruagent_opponent_memory.json` (one entry per opponent
name). Each record stores: matches played, their moves and our moves in the last
match, the per-round averages, the agent's last note, and the sticky
classification.

At the **start of each match** this record is injected into the prompt as an
`OPPONENT MEMORY` block, which is what drives the opening move (Rule 1). The file
persists across runs and tournaments. *(It is git-ignored — it is per-machine
runtime state, not code.)*

## How a round is played (two LLM calls)

1. **Message phase** — the agent decides its message *and* its intended action,
   then sends one short, opaque message and ends messaging. Messaging is *blind*:
   it cannot see the opponent's current-round message yet.
2. **Move phase** — after the phase flips, the agent sees the opponent's message
   and confirms or revises its action before submitting the move.

> **Known limitation (in progress):** reading the opponent's message reliably
> requires the platform's `/messages` history endpoint rather than the
> `new_messages` field; this fix is being finalized. Until then the agent plays
> on actions + memory and treats the opponent's message as unseen.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` (see `.env.example`) with your platform credentials and one LLM
key:

```text
ALTRUAGENT_API_KEY=your_platform_key_here
ALTRUAGENT_AGENT_NAME=your_agent_name_here
OPENAI_API_KEY=your_openai_key_here        # preferred
# gemeni_api_key=your_gemini_key_here      # fallback (Gemini)
```

> Never commit your real `.env` or `.bot_credentials.json` — both are git-ignored.

## Run it

```bash
# See available queues / tournaments
python games/altruagent_runner.py --list

# Join a specific queue and play (recommended)
python games/altruagent_runner.py --queue <queue_id> --verbose

# Or auto-join the first active queue
python games/altruagent_runner.py --queue auto

# Play a specific in-progress tournament you were enrolled in
python games/altruagent_runner.py --tournament <tournament_id>
```

A queue auto-spawns a round-robin tournament once it has 3+ agents. Each match
writes a full reasoning trace to `traces/`. Add `--judge` to localize reasoning
failures after play.

### Test opponents (for local / in-team testing)

`games/scripted_bot.py` runs a no-LLM bot with a fixed strategy
(`tit_for_tat` / `always_defect` / `always_cooperate` / `grim`). Point three
agents at the same queue to trigger a tournament:

```bash
python games/scripted_bot.py --name <bot_name> --strategy tit_for_tat --queue <queue_id>
```

Bot credentials are read from `.bot_credentials.json` (git-ignored).

## Files for the tournament agent

```text
agent/
  altruagent_client.py   HTTP client for the platform (auth, queues, gameplay)
  single_call_agent.py   The strategy "brain": prompt + message/action calls
  opponent_memory.py     Classification + cross-tournament memory store
  agent.py               LLM client wrapper (OpenAI preferred, Gemini fallback)
  tracing.py             Structured per-match reasoning traces
games/
  altruagent_runner.py   Main runnable: joins queue/tournament, plays matches
  scripted_bot.py        Scripted test opponents
ALTRUAGENT_GUIDE.md      Detailed operator guide (errors, debugging, internals)
```

For the full operator guide — error handling, debugging, and platform internals
— see [`ALTRUAGENT_GUIDE.md`](./ALTRUAGENT_GUIDE.md).

---

# Background: ReAct Agent & Tracer Framework

This project implements ReAct-style agents that play repeated games, currently:

- Prisoner's Dilemma
- Hunger Games, a human-vs-agent allocation game

The agent uses a `Thought -> Action -> Observation` loop. Game runs are logged as structured Tracer-style JSON traces, and a separate offline judge can inspect a selected trace file to localize reasoning failures and append game-specific reflections.

## Architecture

```text
games/
  React_agent_gameV1.py      Prisoner's Dilemma runner
  hungergames_V1.py          Hunger Games runner

agent/
  agent.py                   Gemini client, ReAct prompt, reflection loading
  tracing.py                 ReAct parser, trace logger, trace judge helpers
  judge_trace.py             CLI for judging one saved trace file
  reflections/
    prisoners_dilemma.md     Learned lessons for Prisoner's Dilemma
    hunger_games.md          Learned lessons for Hunger Games

traces/
  *.json                     Saved game traces
```

Each game runner owns the game rules and environment. The shared `agent/` package owns the LLM wrapper, prompt construction, reflection memory, trace logging, and offline judging.

## Reflection Flow

1. A game starts.
2. The runner calls `build_system_prompt(..., game_name="...")`.
3. The matching reflection file is loaded from `agent/reflections/`.
4. Any prior lessons are injected into the system prompt under `PRIOR REFLECTIONS FOR THIS GAME`.
5. The agent plays the game while `TraceLogger` records prompts, responses, parsed thoughts/actions, observations, and game state.
6. A JSON trace is saved under `traces/`.
7. Later, `agent/judge_trace.py` can judge a specific trace file.
8. The judge writes results back into the trace JSON and appends a reflection to the matching markdown file.

Reflections are game-specific. Hunger Games lessons do not automatically affect Prisoner's Dilemma, and vice versa.

## Tracer-Style Logging

This project adapts the Tracer idea from code error localization to reasoning error localization.

Instead of tracing code functions and variable states only, the project traces ReAct reasoning steps:

```json
{
  "event": "react_step",
  "round": 1,
  "step": 1,
  "prompt": "...",
  "response": "...",
  "parsed": {
    "thought": "...",
    "action": "make_move",
    "argument": "defect",
    "has_pause": true,
    "decision": null,
    "parse_error": null
  },
  "observation": "...",
  "state_before": {},
  "state_after": {}
}
```

This lets us localize failures such as:

- malformed ReAct output
- invalid moves
- misunderstood state
- weak opponent modeling
- poor strategy
- goal mismatch
- skipped or hallucinated tool calls

## Setup

Create a `.env` file with your Gemini API key:

```text
GEMINI_API_KEY=your_key_here
```

The existing code also supports the older project variable name:

```text
gemeni_api_key=your_key_here
```

Install dependencies in your virtual environment:

```bash
python -m pip install google-genai python-dotenv
```

If your shell has multiple Python environments active, use the project venv explicitly:

```bash
../.venv/Scripts/python.exe -m pip install google-genai python-dotenv
```

## Run Games

From the repo root:

```bash
python games/React_agent_gameV1.py
```

```bash
python games/hungergames_V1.py
```

After each game, a trace file is saved under `traces/`, for example:

```text
traces/prisoners_dilemma_20260429_132414_bf743097.json
```

Gameplay does not run the judge automatically. It only creates the trace.

## Judge A Trace

Run the judge on a specific trace file:

```bash
python agent/judge_trace.py traces/prisoners_dilemma_20260429_132414_bf743097.json
```

Equivalent module form:

```bash
python -m agent.judge_trace traces/prisoners_dilemma_20260429_132414_bf743097.json
```

If your dependencies are installed in the parent `.venv`, use:

```bash
../.venv/Scripts/python.exe agent/judge_trace.py traces/prisoners_dilemma_20260429_132414_bf743097.json
```

By default, judging:

- reads the selected trace file
- asks Gemini to localize reasoning failures
- writes the judge result into the trace JSON under `"judge"`
- appends the judge's reflection to `agent/reflections/<game_name>.md`

Useful options:

```bash
python agent/judge_trace.py traces/example.json --no-reflection
```

Judges the trace but does not update reflection memory.

```bash
python agent/judge_trace.py traces/example.json --no-write
```

Prints the judge result but does not modify the trace file.

## Demo Script

1. Run Prisoner's Dilemma:

```bash
python games/React_agent_gameV1.py
```

2. Open the generated JSON in `traces/` and show the step-by-step ReAct trace.

3. Judge that exact trace:

```bash
python agent/judge_trace.py traces/<trace_file>.json
```

4. Reopen the trace and show the `"judge"` section.

5. Open `agent/reflections/prisoners_dilemma.md` and show the new reflection headed by the source trace filename.

6. Run the game again and point out that the reflection is loaded into the prompt before the agent plays.

## Notes

- No second API key is needed for judging. The same Gemini key is used for gameplay and trace judging.
- Traces are JSON because they are meant for structured analysis.
- Reflections are Markdown because they are human-readable and easy to present.
- The judge is intentionally offline so game execution and trace evaluation stay separate.
