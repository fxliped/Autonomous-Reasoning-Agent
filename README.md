# Autonomous Reasoning Agent

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
