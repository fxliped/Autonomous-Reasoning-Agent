"""Quick latency benchmark: time the message + action phase on different Gemini
models, using REAL prompts (the system prompt + a real game context).

Run:  python3 bench_latency.py
Optionally compare more models by editing MODELS below.
"""

import os
import time
import json
import glob

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from agent.single_call_agent import ACTION_SYSTEM_PROMPT

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
TRIALS = 3

# --- Get real message/action prompts. Prefer a real trace; else synthesize. ---
def load_prompts():
    files = sorted(glob.glob("traces/altruagent_repeated_pd_*.json"), reverse=True)
    for f in files:
        try:
            steps = json.load(open(f)).get("steps", [])
            msg = next((s["prompt"] for s in steps if s["parsed"].get("phase") == "message"), None)
            act = next((s["prompt"] for s in steps if s["parsed"].get("phase") == "action"), None)
            if msg and act:
                print(f"Using real prompts from {f}")
                return msg, act
        except Exception:
            continue
    # Fallback synthetic context if no trace is available.
    print("No trace found — using a synthetic context.")
    ctx = (
        'You are playing Repeated Prisoner\'s Dilemma against "test_opp".\n'
        "This match is 8 rounds total. You are about to play round 8.\n"
        "TOURNAMENT CONTEXT: This is Tournament 2 of 3 — NOT the final tournament.\n"
        "OPPONENT MEMORY: Prior matches: 1. Classification: COOPERATOR.\n"
        "ACTION HISTORY: Rounds 1-7 all mutual cooperate.\n"
    )
    return ctx + "\nThis is the MESSAGE PHASE. Decide action + craft message. Output JSON.", \
           ctx + "\nThis is the ACTION PHASE. Execute your decided action. Output JSON."

msg_prompt, action_prompt = load_prompts()

key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)
cfg = types.GenerateContentConfig(system_instruction=ACTION_SYSTEM_PROMPT)

print(f"system prompt chars: {len(ACTION_SYSTEM_PROMPT)}")
print(f"message prompt chars: {len(msg_prompt)} | action prompt chars: {len(action_prompt)}")
print("=" * 70)


def timed(model, prompt, label):
    times = []
    for i in range(TRIALS):
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=cfg,
            )
            dt = time.time() - t0
            times.append(dt)
            ok = "ok" if (r.text or "").strip() else "EMPTY"
            print(f"  {model:<26} {label:<8} trial{i+1}: {dt:5.1f}s ({ok})")
        except Exception as e:
            dt = time.time() - t0
            print(f"  {model:<26} {label:<8} trial{i+1}: {dt:5.1f}s ERROR: {str(e)[:90]}")
    if times:
        print(f"  -> {model} {label} avg={sum(times)/len(times):.1f}s "
              f"min={min(times):.1f}s max={max(times):.1f}s")


for model in MODELS:
    print(f"--- {model} ---")
    timed(model, msg_prompt, "MESSAGE")
    timed(model, action_prompt, "ACTION")
    print()
