"""
Shared agent infrastructure for the autonomous game-playing system.

Provider priority: OpenAI (gpt-5.4-mini) → Gemini (gemini-2.5-flash).
Detection is automatic from environment variables — no code changes needed
when switching machines or teammates.
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5.4-mini"
GEMINI_MODEL = "gemini-2.5-flash"


# ── Provider detection ─────────────────────────────────────────────────────────

def detect_provider() -> tuple[str, str]:
    """
    Auto-detect which LLM provider to use based on available API keys.
    OpenAI takes priority. Falls back to Gemini if no OpenAI key is found.
    Returns (provider_name, model_name).
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai", OPENAI_MODEL
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemeni_api_key")
    if gemini_key:
        print("[LLMClient] No OPENAI_API_KEY found — falling back to Gemini.")
        return "gemini", GEMINI_MODEL
    raise EnvironmentError(
        "No LLM API key found.\n"
        "Set OPENAI_API_KEY (OpenAI) or GEMINI_API_KEY (Gemini) in your .env file."
    )


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified client for OpenAI and Gemini.
    Auto-detects provider from .env. Force a provider with provider='openai'|'gemini'.
    Call complete(system, messages) from any game — provider details are hidden.
    """

    def __init__(self, provider: Optional[str] = None):
        if provider:
            self.provider = provider
            self.model = OPENAI_MODEL if provider == "openai" else GEMINI_MODEL
        else:
            self.provider, self.model = detect_provider()
        self._raw = self._build_client()
        print(f"[LLMClient] {self.provider.upper()} / {self.model}")

    def _build_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "gemini":
            from google import genai
            key = os.getenv("GEMINI_API_KEY") or os.getenv("gemeni_api_key")
            return genai.Client(api_key=key)
        raise ValueError(f"Unknown provider: '{self.provider}'")

    def complete(self, system: str, messages: list[dict], retries: int = 3) -> str:
        """Complete a chat with automatic retry on API errors."""
        for attempt in range(retries):
            try:
                return self._call(system, messages)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  [API error ({attempt+1}/{retries}), retrying in 5s: {e}]")
                    time.sleep(5)
                else:
                    raise

    def _call(self, system: str, messages: list[dict]) -> str:
        if self.provider == "openai":
            full = [{"role": "system", "content": system}] + messages
            resp = self._raw.chat.completions.create(model=self.model, messages=full)
            return resp.choices[0].message.content

        elif self.provider == "gemini":
            from google.genai import types
            # Gemini expects alternating user/model roles
            gemini_msgs = [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ]
            cfg = types.GenerateContentConfig(system_instruction=system)
            resp = self._raw.models.generate_content(
                model=self.model, contents=gemini_msgs, config=cfg
            )
            return resp.text

        raise ValueError(f"Unknown provider: '{self.provider}'")


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Stateful ReAct agent backed by LLMClient.
    Maintains conversation history within a session.
    Call reset() to clear history between rounds or phases.
    """

    def __init__(self, client: LLMClient, system: str, name: str = "Agent"):
        self.client = client
        self.system = system
        self.name = name
        self.messages: list[dict] = []

    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.complete(self.system, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        """Clear conversation history (call between rounds or phases)."""
        self.messages = []


# ── Trace Logging ─────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    """One step in the ReAct loop — captured for debugging and Tracer consumption."""
    round: int
    step: int
    agent_name: str
    thought: str
    action: str
    tool: str
    arg: str
    observation: str
    raw_response: str = ""


@dataclass
class GameTrace:
    """Full game trace — all agent reasoning steps, outcome, and metadata."""
    game_name: str
    provider: str
    model: str
    steps: list[TraceStep] = field(default_factory=list)
    outcome: str = ""
    metadata: dict = field(default_factory=dict)

    def add(self, step: TraceStep):
        self.steps.append(step)

    def save(self, path: str):
        """Persist trace to JSON for analysis or Tracer consumption."""
        import dataclasses
        with open(path, "w") as f:
            json.dump(
                {
                    "game": self.game_name,
                    "provider": self.provider,
                    "model": self.model,
                    "outcome": self.outcome,
                    "metadata": self.metadata,
                    "steps": [dataclasses.asdict(s) for s in self.steps],
                },
                f,
                indent=2,
            )
        print(f"[Trace] Saved → {path}")
