"""Shared ReAct agent utilities for game runners."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

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

# Single source of truth for default models — used by both detect_provider() and LLMClient.
_OPENAI_DEFAULT_MODEL = "gpt-4.1"
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


# =============================================================================
# PROVIDER DETECTION
# =============================================================================

def detect_provider() -> tuple[str, str]:
    """
    Auto-detect which LLM provider to use based on available API keys.
    OpenAI takes priority. Falls back to Gemini if no OpenAI key is found.
    Returns (provider_name, default_model_name).
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai", _OPENAI_DEFAULT_MODEL
    gemini_key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return "gemini", _GEMINI_DEFAULT_MODEL
    raise ValueError(
        "No API key found. Set OPENAI_API_KEY (preferred) or GEMINI_API_KEY in your .env file."
    )


# =============================================================================
# PROVIDER-AGNOSTIC LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Provider-agnostic LLM client.
    Priority: OpenAI (OPENAI_API_KEY) → Gemini (GEMINI_API_KEY / gemeni_api_key).
    Exposes a single .complete(system, messages, model) method.
    messages format: [{"role": "user"|"assistant", "content": "..."}]
    """

    OPENAI_DEFAULT_MODEL = _OPENAI_DEFAULT_MODEL
    GEMINI_DEFAULT_MODEL = _GEMINI_DEFAULT_MODEL

    def __init__(self):
        self.provider, self.default_model = detect_provider()
        self._build_client()

    def _build_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "gemini":
            from google import genai
            key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")
            self._gemini = genai.Client(api_key=key)

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


def create_gemini_client() -> LLMClient:
    """Backward-compatibility alias for create_client()."""
    return create_client()


# =============================================================================
# REFLECTION HELPERS
# =============================================================================

def reflection_path(game_name: str) -> Path:
    safe_name = game_name.strip().lower().replace(" ", "_")
    return REFLECTIONS_DIR / f"{safe_name}.md"


def load_reflections(game_name: str, max_reflections: int = 8) -> str:
    """Load the most recent N reflection entries for a game."""
    path = reflection_path(game_name)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8").strip()
    if "No judged reflections yet." in content and "## Reflection" not in content:
        return ""
    # Split on section headers and return the last max_reflections entries
    import re as _re
    sections = _re.split(r"(?=^## Reflection)", content, flags=_re.MULTILINE)
    # First element may be the title line (# Game Reflections) — keep it, cap the rest
    header = sections[0] if not sections[0].startswith("## Reflection") else ""
    entries = [s for s in sections if s.startswith("## Reflection")]
    recent = entries[-max_reflections:]
    parts = ([header.strip()] if header.strip() else []) + recent
    return "\n\n".join(parts).strip()


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


def load_strategy_updates() -> str:
    """Load the most recent strategy update from strategy_updates.md, if any."""
    path = REFLECTIONS_DIR / "strategy_updates.md"
    if not path.exists():
        return ""
    import re as _re
    content = path.read_text(encoding="utf-8").strip()
    # Return only the most recent update entry
    sections = _re.split(r"(?=^## Strategy Update)", content, flags=_re.MULTILINE)
    entries = [s for s in sections if s.startswith("## Strategy Update")]
    return entries[-1].strip() if entries else ""


def build_system_prompt(
    game_prompt: str,
    game_name: str | None = None,
    use_reflections: bool = True,
    use_react: bool = True,
) -> str:
    """
    Assemble a system prompt from optional components.

    use_react=True  (default): prepend BASE_REACT_PROMPT (Thought→Action→PAUSE→Observation loop).
                               Use for game runners that dispatch tool calls between turns.
    use_react=False:           omit BASE_REACT_PROMPT. Use for single-shot structured CoT agents
                               (e.g. TournamentAgent) where no tool loop runs.

    Injection order (when all present):
      BASE_REACT_PROMPT → STRATEGY UPDATE → PRIOR REFLECTIONS → game_prompt
    Strategy updates sit above per-match reflections so they frame how lessons are interpreted.
    """
    sections = []
    if use_react:
        sections.append(BASE_REACT_PROMPT)
    if game_name and use_reflections:
        strategy_update = load_strategy_updates()
        if strategy_update:
            sections.append(
                "STRATEGY UPDATE (from post-tournament review — supersedes older defaults):\n"
                f"{strategy_update}"
            )
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
    """
    Stateful LLM agent that maintains conversation history within a round.
    Create a new instance each round to keep context short, or call reset().
    """

    def __init__(
        self,
        client: LLMClient,
        system: str,
        model: str | None = None,
        name: str = "Agent",
    ):
        self.client = client
        self.system = system
        self.model = model  # None → use client.default_model
        self.name = name
        self.messages: list[dict] = []

    def __call__(self, message: str) -> Optional[str]:
        if not message:
            return None
        self.messages.append({"role": "user", "content": message})
        result = self.client.complete(self.system, self.messages, self.model)
        self.messages.append({"role": "assistant", "content": result})
        return result

    def reset(self):
        """Clear conversation history (use between rounds or phases)."""
        self.messages = []
