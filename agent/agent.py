"""Shared ReAct agent utilities for game runners."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash"
ROOT_DIR = Path(__file__).resolve().parents[1]
REFLECTIONS_DIR = Path(__file__).resolve().parent / "reflections"

BASE_REACT_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output a Decision.

IMPORTANT: You MUST always start your response with 'Thought:' before any Action.

Use Thought to reason about the current game state.
Use Action to call one of your available tools, then return PAUSE.
Observation will be the result of that tool.
""".strip()


def reflection_path(game_name: str) -> Path:
    """Return the reflection file path for a stable game name."""
    safe_name = game_name.strip().lower().replace(" ", "_")
    return REFLECTIONS_DIR / f"{safe_name}.md"


def load_reflections(game_name: str) -> str:
    """Load prior reflections for a game, or return an empty string."""
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
    """Append a reflection to the game's reflection memory file."""
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
    """Combine shared ReAct instructions, game instructions, and memory."""
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


def create_gemini_client():
    """Create a Gemini client using the existing project .env convention."""
    load_dotenv()
    api_key = os.getenv("gemeni_api_key") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


class Agent:
    """Stateful LLM agent that maintains conversation history within a round."""

    def __init__(self, client, system: str, model: str = DEFAULT_MODEL):
        self.client = client
        self.system = system
        self.model = model
        self.messages = []

    def __call__(self, message: str):
        if not message:
            return None

        self.messages.append({"role": "user", "parts": [{"text": message}]})
        result = self.execute()
        self.messages.append({"role": "model", "parts": [{"text": result}]})
        return result

    def execute(self) -> str:
        """Call Gemini API with retry logic for transient failures."""
        config = types.GenerateContentConfig(system_instruction=self.system)
        for attempt in range(3):
            try:
                completion = self.client.models.generate_content(
                    model=self.model,
                    contents=self.messages,
                    config=config,
                )
                return completion.text
            except Exception:
                if attempt < 2:
                    print(f"  [API busy, retrying in 5s... ({attempt + 1}/3)]")
                    time.sleep(5)
                    continue
                raise
