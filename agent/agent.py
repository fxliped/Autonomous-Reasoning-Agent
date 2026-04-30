"""Shared ReAct agent utilities for game runners."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
REFLECTIONS_DIR = Path(__file__).resolve().parent / "reflections"

BASE_REACT_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output a Decision.

IMPORTANT: You MUST always start your response with 'Thought:' before any Action.

When writing your Thought, first reason out loud step by step: which action maximizes your probability of winning? Consider what your opponent is likely to do based on the history. Note that opponents sometimes make mistakes or act irrationally — a single unexpected move may be an error, not a permanent strategy shift. Weigh this before overreacting.
Use Action to call one of your available tools, then return PAUSE.
Observation will be the result of that tool.
""".strip()


# =============================================================================
# PROVIDER-AGNOSTIC LLM CLIENT
# Tries OpenAI first (OPENAI_API_KEY), then falls back to Gemini.
# =============================================================================

class LLMClient:
    """
    Provider-agnostic LLM client.
    Priority: OpenAI (OPENAI_API_KEY) → Gemini (GEMINI_API_KEY / gemeni_api_key).
    Exposes a single .complete(system, messages, model) method.
    messages format: [{"role": "user"|"assistant", "content": "..."}]
    """

    OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
    GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self):
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")

        if openai_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=openai_key)
                self.provider = "openai"
                self.default_model = self.OPENAI_DEFAULT_MODEL
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        elif gemini_key:
            try:
                from google import genai
                self._gemini = genai.Client(api_key=gemini_key)
                self.provider = "gemini"
                self.default_model = self.GEMINI_DEFAULT_MODEL
            except ImportError:
                raise ImportError("google-genai package not installed.")
        else:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY (preferred) or GEMINI_API_KEY in your .env file."
            )

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        """
        Send a conversation to the LLM and return the assistant's reply.
        messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        model = model or self.default_model
        for attempt in range(3):
            try:
                if self.provider == "openai":
                    return self._complete_openai(system, messages, model)
                else:
                    return self._complete_gemini(system, messages, model)
            except Exception as exc:
                if attempt < 2:
                    print(f"  [API error, retrying in 5s... ({attempt + 1}/3): {exc}]")
                    time.sleep(5)
                else:
                    raise

    def _complete_openai(self, system: str, messages: list[dict], model: str) -> str:
        all_messages = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        response = self._openai.chat.completions.create(model=model, messages=all_messages)
        return response.choices[0].message.content

    def _complete_gemini(self, system: str, messages: list[dict], model: str) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system)
        gemini_messages = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        completion = self._gemini.models.generate_content(
            model=model,
            contents=gemini_messages,
            config=config,
        )
        return completion.text


def create_client() -> LLMClient:
    """Create the shared LLM client (OpenAI preferred, Gemini fallback)."""
    return LLMClient()


# Keep for backward compatibility with Warren's existing game files
def create_gemini_client() -> LLMClient:
    return create_client()


# Default model constant — reflects whichever provider is active at import time.
# Prefer reading client.default_model at runtime over this module-level constant.
DEFAULT_MODEL = LLMClient.OPENAI_DEFAULT_MODEL  # overridden at runtime by client.default_model


# =============================================================================
# REFLECTION HELPERS
# =============================================================================

def reflection_path(game_name: str) -> Path:
    safe_name = game_name.strip().lower().replace(" ", "_")
    return REFLECTIONS_DIR / f"{safe_name}.md"


def load_reflections(game_name: str) -> str:
    path = reflection_path(game_name)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8").strip()
    if "No judged reflections yet." in content and "## Reflection" not in content:
        return ""
    return content


def append_reflection(
    game_name: str,
    reflection: str,
    source_file: str | Path | None = None,
) -> Path:
    REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = reflection_path(game_name)
    cleaned = reflection.strip()
    if not cleaned:
        return path
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    replace_placeholder = "No judged reflections yet." in existing and "## Reflection" not in existing
    mode = "w" if replace_placeholder else "a"
    source = Path(source_file).name if source_file else "manual"
    with path.open(mode, encoding="utf-8") as handle:
        if replace_placeholder:
            title = game_name.strip().replace("_", " ").title()
            handle.write(f"# {title} Reflections\n\n")
        elif path.stat().st_size:
            handle.write("\n\n")
        handle.write(f"## Reflection from {source}\n\n{cleaned}\n")
    return path


def build_system_prompt(
    game_prompt: str,
    game_name: str | None = None,
    use_reflections: bool = True,
) -> str:
    """Combine shared ReAct instructions, game instructions, and prior reflections."""
    sections = [BASE_REACT_PROMPT]
    if game_name and use_reflections:
        reflections = load_reflections(game_name)
        if reflections:
            sections.append(
                "PRIOR REFLECTIONS FOR THIS GAME:\n"
                f"{reflections}\n\n"
                "Use these lessons when they apply, but prioritize the current game state."
            )
    sections.append(game_prompt.strip())
    return "\n\n".join(sections)


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    """Stateful LLM agent that maintains conversation history within a round."""

    def __init__(self, client: LLMClient, system: str, model: str | None = None):
        self.client = client
        self.system = system
        self.model = model  # None → use client.default_model
        self.messages: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]

    def __call__(self, message: str):
        if not message:
            return None
        self.messages.append({"role": "user", "content": message})
        result = self.client.complete(self.system, self.messages, self.model)
        self.messages.append({"role": "assistant", "content": result})
        return result
