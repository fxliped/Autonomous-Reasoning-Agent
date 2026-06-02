# AltruAgent Platform — Operator's Guide

How our Prisoner's Dilemma agent runs on the class competition platform
(AltruAgent), what it sees, how its memory works, and how to operate and
debug it yourself.

Last updated: 2026-06-01.

---

## 0. The pieces — what each relevant file does

Only the files that matter for the tournament are listed (other-game and
local-only files like `prisoners_dilemma.py`, `test_harness.py`,
`hunger_games.py` are not used on the platform).

**Dependency chain, in one line:** you run `altruagent_runner.py` → it talks to
the platform via `altruagent_client.py`, gets decisions from
`single_call_agent.py` (which uses `agent.py` for the LLM), remembers opponents
via `opponent_memory.py`, and records everything via `tracing.py`.

### The one you actually run
- **`games/altruagent_runner.py`** — the orchestrator and entry point. Joins a
  queue/tournament, plays every match round-by-round, ties everything together.
  - `Runner` — logs in, checks claim status, loops over tournament matches
    (`run_queue` / `run_tournament` / `run_competition`).
  - `MatchPlayer` — drives a single 8-round match: detects message vs move
    phase, builds the agent's context (`_build_context`), calls the LLM, maps
    cooperate/defect → 0/1, sends moves/messages, saves trace + memory at the end.

### Code we built to support it
- **`agent/altruagent_client.py`** — the HTTP layer. Every platform call goes
  through here: login, claim check, list/join queues & tournaments, get game
  state, submit moves, send/terminate messages, resign. Handles the bearer token
  and the one-retry-on-401 rule. Knows nothing about strategy.
- **`agent/opponent_memory.py`** — the cross-tournament memory. `classify()`
  turns a match's move sequences into a label; `OpponentMemory` loads/saves the
  JSON file and produces the "OPPONENT MEMORY" text injected into the agent each
  match. Hostile labels stick.
- **`games/scripted_bot.py`** — **testing only, not for the real tournament.** A
  no-LLM opponent that plays a fixed strategy (tit-for-tat, always-defect, etc.)
  so we can fill a queue and watch our agent. Reuses the same HTTP client.

### Pre-existing code the agent depends on
- **`agent/single_call_agent.py`** — **the brain.** Holds the "Honest
  Reciprocator with Memory" strategy prompt (Rules 1–9) and makes the two LLM
  calls per round: `generate_message()` (blind) and `choose_action()` (after the
  opponent's message). *This is where the strategy lives — edit it to change how
  the agent plays.*
- **`agent/agent.py`** — the LLM plumbing. `LLMClient` / `create_client()`
  auto-detect OpenAI vs Gemini from `~/.env` and expose one `.complete()`
  method. Also loads `~/.env`.
- **`agent/tracing.py`** — `TraceLogger` (writes per-match JSON traces to
  `traces/`) and the optional LLM `judge_trace`. Used by `--verbose` and
  `--judge`.

### Data, output, and config (not code)
| Path | What it is |
|---|---|
| `~/.env` | Your credentials — `ALTRUAGENT_API_KEY`, `ALTRUAGENT_AGENT_NAME`, and your LLM key. **Edit this to point at your graded agent.** |
| `altruagent_opponent_memory.json` | The memory file. Auto-created/updated each match. Persists across the 3 tournaments. |
| `traces/` | One structured JSON trace per match — full prompts, untruncated reasoning, rules applied, game state. Post-game source of truth. |
| `logs/` | Plain-text run logs (only when you redirect output, e.g. `> logs/real_run.log`). |
| `.bot_credentials.json` | api_keys for the two throwaway test bots. Testing only. |
| `ALTRUAGENT_GUIDE.md` | This document. |

Credentials live in `~/.env` (home directory), loaded automatically:
- `ALTRUAGENT_API_KEY` — the agent's platform key
- `ALTRUAGENT_AGENT_NAME` — the agent's name (used for opponent-vs-self attribution)
- plus your LLM key (`GEMINI_API_KEY` / `gemeni_api_key`, or `OPENAI_API_KEY`)

---

## 1. How to run it yourself (for the tournament)

### One-time check
```bash
cd ~/Desktop/Autonomous-Reasoning-Agent-main
python3 -m agent.altruagent_client      # prints "Agent is CLAIMED and ready"
python3 games/altruagent_runner.py --list   # shows live queues/tournaments
```

### The three ways to enter a game

**A) Join a queue (the normal class path).** Everyone joins the same queue;
when 3+ agents are in it, a 120s timer fires and a round-robin tournament
spawns automatically (no admin needed).
```bash
python3 games/altruagent_runner.py --queue <QUEUE_ID> --verbose
# or let it pick the first active queue:
python3 games/altruagent_runner.py --queue auto --verbose
```

**B) Join a specific tournament** (if the professor hands you a `tournament_id`):
```bash
python3 games/altruagent_runner.py --tournament <TOURNAMENT_ID> --verbose
```

**C) Single competition** (one-off 2-player match by `session_id`):
```bash
python3 games/altruagent_runner.py --session <SESSION_ID> --verbose
```

### Recommended for the real tournament
Run it with logging so you have a record and can watch it live:
```bash
python3 -u games/altruagent_runner.py --queue <QUEUE_ID> --verbose > logs/real_run.log 2>&1 &
tail -f logs/real_run.log
```
- `-u` = unbuffered output (so the log updates live).
- `--verbose` = prints each decision + reasoning (truncated) to the log.
- Traces are saved to `traces/` **regardless** of `--verbose`.
- `&` runs it in the background; drop it to run in the foreground.

### ⚠️ Which agent are you running?
Right now `~/.env` points at **`sarthak_test_agent`** — a *test* agent whose
results do NOT count. For the **graded** tournament you must point
`ALTRUAGENT_API_KEY` (and `ALTRUAGENT_AGENT_NAME`) in `~/.env` at your **real,
claimed** competition agent. Re-run the one-time check above to confirm it says
"claimed" before the tournament starts.

### Things that will bite you if you forget
- **Keep the process running** the whole tournament. If it dies, matches get
  auto-cooperated (you become a free win for opponents). Re-launch with
  `--tournament <id>` to rejoin — membership survives by agent id.
- **Stable network + don't let the laptop sleep.** Each round needs 2 LLM calls
  within a **30-second** inactivity window or the platform auto-cooperates for
  you that round.
- **Don't delete the memory file between the 3 tournaments** — that's where its
  value is (see §4).

---

## 2. How memory is managed

- **File:** `altruagent_opponent_memory.json` in the project root (override with
  `--memory <path>`). Plain JSON, one record per opponent, keyed by opponent
  **name** (lowercased).
- **When it's written:** at the **end of every match**, the runner reads the
  final move sequences and updates that opponent's record, then saves the file.
- **What a record holds:**
  ```json
  "kogai-strat-2": {
    "name": "kogai-strat-2",
    "matches_played": 1,
    "classification": "COOPERATOR",
    "last_match_opp_moves": ["cooperate", ...],
    "last_match_my_moves":  ["cooperate", ...],
    "last_my_avg": 2.0,
    "last_opp_avg": 2.0
  }
  ```
- **How classification is decided** (`opponent_memory.classify`, deterministic):
  - 0 defections → `COOPERATOR`
  - defected while we cooperated after 3+ mutual-coop rounds (mid-game betrayal)
    → `EXPLOITER`
  - defected on >50% of rounds → `DEFECTOR`
  - otherwise (responsive defections) → `RECIPROCATOR`
- **Sticky hostility:** once an opponent is `DEFECTOR`/`EXPLOITER`, that label
  **sticks** even if they behave better later — we don't hand a known cheater a
  second free pass.
- **How it changes behavior:** at the start of every match the opponent's record
  is injected into the agent's context (the "OPPONENT MEMORY" block). The
  strategy's **Rule 1** then opens with **defection** against any
  `DEFECTOR`/`EXPLOITER`, skipping the costly "be nice first" probe.

**Reset before the graded run?** The file currently contains the two test bots
(`sarthak_tft_bot`, `sarthak_defect_bot`). They're harmless (you won't face
them), but if you want a clean slate, delete `altruagent_opponent_memory.json`
before the first real tournament. Then **leave it alone** across all 3.

---

## 3. What the agent sees each round

Each round has two LLM calls. The agent receives a text context built from the
live platform game state:

1. **Header** — opponent name, total rounds, current round, and that the score
   is a per-round average (so mutual cooperation is the goal).
2. **Payoff rules** — the full 2x2 matrix.
3. **OPPONENT MEMORY** — the cross-tournament record (or "No prior history…").
4. **ACTION HISTORY (this match)** — every completed round: your move, their
   move, and the points each side got. This is the authoritative signal.
5. **MESSAGE HISTORY** — messages exchanged per round (see limitation below).
6. **Cumulative raw score** — yours vs theirs so far this match.

Two-phase flow per round:
- **Message phase (blind):** the agent writes its message *without* seeing the
  opponent's message for the current round (platform enforces blind messaging).
- **Action phase:** the agent picks cooperate/defect. The opponent's message for
  the round becomes visible after the messaging phase closes.

### ⚠️ Known limitation: the agent currently plays "message-blind"
In testing, opponent **messages were not being surfaced into the context**
(the platform's `new_messages` field came back empty, and the dedicated
messages endpoint tags senders by integer index, which our attribution code
doesn't yet match). **Effect:** the agent decides almost entirely from the
**action history**, not from what opponents *say*.

This is mostly fine — the strategy is deliberately action-based and
message-skeptical, and it played correctly without messages in the live test.
But it means **Rule 2** ("opponent promised cooperation then defected → treat
as exploiter") can't fire yet. Wiring in real message ingestion is a queued
refinement (see §7).

---

## 4. What carries across tournaments

Only the **memory file**. Everything else (the current match's action history,
messages, scores) is per-match and resets each game.

Across the 3 graded tournaments, for each opponent the agent carries forward:
- their classification (`COOPERATOR`/`RECIPROCATOR`/`DEFECTOR`/`EXPLOITER`),
- their move sequence from the last match against them,
- the average payoffs from that match.

**Why this matters:** in a *single* tournament the memory does almost nothing —
every opponent starts `UNKNOWN`, so the agent pays the "be nice first" cost
against defectors. The payoff comes in tournaments **2 and 3**, when the agent
opens with defection against remembered cheaters and avoids those losses. (This
is exactly why our agent placed last in the single test tournament — see §5 —
and why that result understates it over a 3-tournament run.)

---

## 5. Did the agents run without error?

**Short version: yes, after one bug fix.** Here's the honest record from the
test run:

- **Initial crash (fixed):** the runner tried to *re-join* a tournament it was
  already auto-enrolled in via the queue; the platform rejected the redundant
  join and the agent exited. **Fixed** — the runner now skips the redundant
  join and treats join-failures as non-fatal. The two bots never hit this (their
  code already ignored that error).
- **After the fix:** the agent ran **start-to-finish with no crash** (exit code
  0), played all 4 of its matches, saved memory and traces.
- **Transient LLM hiccup (auto-recovered):** Gemini returned a 503 ("high
  demand") once mid-match; the client retried and the move went through fine.
- **One non-fatal warning:** a `wrong_phase` "can't move during messaging" —
  a latency race (the LLM call took long enough that the round's phase flipped
  underneath it). It self-corrected on the next poll. Queued for hardening (§7).

**Test tournament result:** 5 agents, round-robin. Our agent went
**0 wins / 3 draws / 1 loss**, average **1.4375**, placing **#5 of 5**. It
reached mutual cooperation (2.0) against both real classmates and the TFT bot,
and lost only to the pure always-defect bot (−0.25), which it correctly
retaliated against and saved as a `DEFECTOR`. (See §4 for why one tournament
understates the strategy.)

The bots (`sarthak_tft_bot`, `sarthak_defect_bot`) ran cleanly and exited 0.

---

## 6. How you'll know if there's an error

**Watch the log** (`logs/…log` if you redirected, or the terminal). Signals:

| What you see | Meaning | What to do |
|---|---|---|
| `Traceback (most recent call last)` | The agent crashed | Read the last line; re-launch with `--tournament <id>` to rejoin |
| process exited but tournament not done | Crash or kill | Re-launch with `--tournament <id>` |
| `[step failed: wrong_phase]` | Latency race (non-fatal) | Usually self-corrects; ignore unless frequent |
| `[step failed: invalid_action]` | Bad move (shouldn't happen) | Re-launch; report |
| `[state error: user_not_in_game]` | Not actually in that game | Verify the session/tournament id |
| `[API error, retrying…]` then continues | Transient LLM hiccup | Nothing — it auto-retries |
| `[LLM error … -> defaulting to cooperate]` | LLM fully failed a call | Non-fatal; that round defaults to cooperate |
| Agent silent, opponents' moves advancing | Process died or stalled | Check it's still running; re-launch |
| `Not claimed` on startup | Wrong/unclaimed credentials | Fix `~/.env`, claim the agent |

**Two ground-truth checks any time:**
```bash
# Is the agent process alive?
ps aux | grep altruagent_runner | grep -v grep

# What did it actually do? (per-match results + any errors)
grep -E "Match done|Traceback|step failed|state error|Tournament complete" logs/real_run.log
```

**Traces** in `traces/` are the post-game source of truth: each match file has
every prompt, the full untruncated LLM response, the rule it applied, and the
game state per round. Open the JSON to see exactly what happened and why.

---

## 7. Other things worth knowing

- **Game config (confirmed live):** `repeated_pd`, **8 rounds** (not the docs'
  default 10), payoffs CC=+2/+2, CD=−1/+5, DC=+5/−1, DD=0/0, blind messaging,
  1 chat per round (≤50 words), 30s inactivity timeout, **auto-cooperate** if
  you stall on a move.
- **Action mapping:** 0 = Cooperate, 1 = Defect.
- **Scoring:** final score is the **per-round average** (bounded −1…+5), and the
  tournament leaderboard ranks by mean per-round score across all your games.
- **Queues vs tournaments:** a queue is a permanent entry point that spawns a new
  tournament each time it fills (min 3, max 8 agents). While a queue's tournament
  is running, new joins are rejected until it finishes.
- **The two test bots still exist** as claimed agents on the platform. Harmless,
  but they're real identities tied to keys in `.bot_credentials.json`. To use
  them for more testing: `python3 games/scripted_bot.py --name sarthak_tft_bot
  --strategy tit_for_tat --queue <id>`.

### Queued refinements (not yet done)
1. **Harden the `wrong_phase` latency race** — re-poll and retry the move when
   the phase flips during a slow LLM call, instead of skipping it.
2. **Real message ingestion** — read the `/messages` endpoint and attribute
   senders by integer index so the agent actually *sees* opponent messages
   (enables Rule 2: catching "I'll cooperate" lies). Currently action-only.
3. **(Optional) reconsider forgiving a confirmed defector** — the one-time
   forgiveness costs an extra sucker payoff; weigh against its value vs
   reciprocators.

### Recommended pre-tournament checklist
- [ ] `~/.env` points at your **graded** agent, and the claim check passes.
- [ ] Decide whether to clear `altruagent_opponent_memory.json` (then leave it
      alone across all 3 tournaments).
- [ ] Run with `-u … --verbose > logs/real_run.log 2>&1` and `tail -f` it.
- [ ] Confirm the process stays alive; laptop won't sleep; stable network.
- [ ] After each tournament, skim `traces/` and the leaderboard.
