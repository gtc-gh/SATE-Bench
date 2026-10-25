"""Answer parsing: rule-based extraction first, LLM-based extraction as a fallback."""

import re
import time
from typing import Iterable, Optional, Tuple

from .config import EXTRACTOR_MAX_RETRIES, EXTRACTOR_MODEL, get_gemini_key
from .prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_SYSTEM_PROMPT_ACTIVE

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]
MOVE_LETTERS = ["X", "Y"]

LETTER_REGEX_PASSIVE = re.compile(r"\b([A-Ea-e])\b")
LETTER_REGEX_ACTIVE = re.compile(r"\b([A-Ea-eXxYy])\b")


def parse_passive_letter(text: str) -> Optional[str]:
    """Returns A-E or None."""
    if not text:
        return None
    text = text.strip()
    if len(text) == 1 and text.upper() in ANSWER_LETTERS:
        return text.upper()
    m = LETTER_REGEX_PASSIVE.search(text)
    return m.group(1).upper() if m else None


def parse_active_letter(text: str) -> Optional[str]:
    """Returns an answer (A-E), an exploration move (X/Y) or None."""
    if not text:
        return None
    text = text.strip()
    if len(text) == 1 and text.upper() in ANSWER_LETTERS + MOVE_LETTERS:
        return text.upper()
    m = LETTER_REGEX_ACTIVE.search(text)
    return m.group(1).upper() if m else None


def split_thinking(raw_text: str) -> Tuple[str, str]:
    """Separates a `<think>...</think>` reasoning trace from the final answer.

    Returns (answer_text, thought_text). Handles unclosed `<think>` tags and `\\boxed{}` answers.
    """
    if not raw_text:
        return "", ""
    text = raw_text.strip()

    if "</think>" in text:
        thought, answer = text.split("</think>", 1)
        return answer.strip(), thought.replace("<think>", "").strip()

    if "<think>" in text:
        content = text.replace("<think>", "").strip()
        box_pos = _rfind_boxed(content)
        if box_pos != -1:
            return content[box_pos:].strip(), content[:box_pos].strip()
        return content, content

    box_pos = _rfind_boxed(text)
    if box_pos > 0:
        return text[box_pos:].strip(), text[:box_pos].strip()
    return text, ""


def _rfind_boxed(text: str) -> int:
    pos = text.rfind(r"\boxed{")
    return pos if pos != -1 else text.rfind(r"\\boxed{")


def extract_boxed_content(text: str) -> Optional[str]:
    """Extracts the content of the last `\\boxed{...}` (handles nested braces)."""
    if not text:
        return None
    start = text.rfind(r"\boxed{")
    offset = len(r"\boxed{")
    if start == -1:
        start = text.rfind(r"\\boxed{")
        offset = len(r"\\boxed{")
    if start == -1:
        return None

    depth = 0
    content = text[start + offset:]
    for i, char in enumerate(content):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == -1:
            return content[:i].strip()
    return None


def extract_boxed_choice(text: str, valid_keys: Iterable[str]) -> Optional[str]:
    """Returns the option letter inside `\\boxed{}` if it is one of `valid_keys`."""
    valid = {k.upper() for k in valid_keys}
    boxed = extract_boxed_content(text)
    if not boxed:
        return None
    boxed = boxed.upper().strip()
    if boxed in valid:
        return boxed
    # e.g. "\boxed{Option A.}"
    for word in reversed(re.sub(r"[^A-Z0-9]", " ", boxed).split()):
        if len(word) == 1 and word in valid:
            return word
    letters = re.findall(f"([{''.join(sorted(valid))}])", boxed)
    return letters[-1] if letters else None


class FallbackExtractor:
    """Uses a lightweight Gemini model to extract the choice from irregular outputs.

    If no Gemini API key is available, the fallback is disabled and unparsable outputs
    are treated as invalid predictions.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = EXTRACTOR_MODEL):
        self.model = model
        self.client = None
        api_key = api_key or get_gemini_key()
        if api_key:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            print("[FallbackExtractor] GEMINI_API_KEY not set: LLM-based answer extraction is disabled.")

    def _query(self, prompt: str, parse_fn, system_prompt: str = EXTRACTOR_SYSTEM_PROMPT) -> Optional[str]:
        """Queries the extractor until `parse_fn` returns a valid letter (at most EXTRACTOR_MAX_RETRIES times)."""
        from google.genai import types
        for _ in range(EXTRACTOR_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0),
                )
                letter = parse_fn(response.text or "")
                if letter is not None:
                    return letter
            except Exception:
                time.sleep(2)
        return None

    def extract_passive(self, raw_text: str) -> Optional[str]:
        if self.client is None or not raw_text:
            return None
        prompt = f"Text to analyze:\n{raw_text}\n\nTask: Extract the final single answer letter (A, B, C, D, or E)."
        return self._query(prompt, parse_passive_letter)

    def extract_active(self, raw_text: str) -> Optional[str]:
        if self.client is None or not raw_text:
            return None
        prompt = f"Text to analyze:\n{raw_text}\n\nTask: Extract the single letter choice (A, B, C, D, E, X, or Y)."

        def exact_letter(text: str) -> Optional[str]:
            text = text.strip().upper()
            return text if text in ANSWER_LETTERS + MOVE_LETTERS else None

        return self._query(prompt, exact_letter, EXTRACTOR_SYSTEM_PROMPT_ACTIVE)
