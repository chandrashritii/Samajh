"""Swappable LLM interface. Defaults to Sarvam Chat; Gemini optional.

Keep this interface intentionally narrow — one call: `complete(system, user)
→ str`. Answering builds the structured-output prompt; this layer is dumb.
"""
from __future__ import annotations
from typing import Protocol

from . import config, sarvam_client


class LLM(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.1) -> str: ...


class SarvamLLM:
    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.1) -> str:
        return sarvam_client.chat_complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )


class GeminiLLM:
    """Optional. Drop-in for cost-free testing. Requires google-generativeai
    AND GEMINI_API_KEY in env. Lazily imported so the dependency stays optional.
    """
    def __init__(self) -> None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set.")
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._genai = genai
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.1) -> str:
        resp = self._model.generate_content(
            [system, user],
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return resp.text or ""


def get_llm() -> LLM:
    if config.LLM_PROVIDER == "gemini":
        return GeminiLLM()
    return SarvamLLM()
