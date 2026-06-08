"""Thin wrapper over Sarvam REST APIs (STT, Chat, Translate).

Kept deliberately small and explicit so each call site shows the contract.
Auth uses the api-subscription-key header — works for all three endpoints.

If a fallback key is configured, primary auth/quota failures (401/403/429)
transparently retry once on the fallback key.
"""
from __future__ import annotations
import json
from typing import Any, Callable
import requests

from . import config


class SarvamError(RuntimeError):
    pass


def friendly_error(exc: Exception) -> tuple[int, str]:
    """Map a Sarvam call failure to (http_status, human-readable message).

    POC-grade but specific: a billing problem says "recharge", a retired model
    says so, a slow model says "try again" — never a bare 500. Each cause maps to
    its OWN message; we don't, e.g., mislabel a token-truncation as a credit issue.
    """
    import requests as _rq
    if isinstance(exc, (_rq.Timeout,)):
        return 504, ("The Sarvam model took too long to respond. It may be under "
                     "load — please try again in a moment.")
    msg = str(exc).lower()
    if "deprecated" in msg or "has been deprecated" in msg:
        return 502, ("The configured Sarvam model is no longer available (it was "
                     "deprecated). Set SARVAM_CHAT_MODEL to a current model "
                     "(sarvam-30b or sarvam-105b).")
    if any(t in msg for t in ("insufficient", "quota", "credit", "payment", "billing",
                              " 402", " 403", "exhausted")):
        return 402, "Sarvam model credits expired. Please recharge to continue."
    if " 429" in msg or "rate limit" in msg or "too many requests" in msg:
        return 429, "Sarvam is rate-limiting requests right now — please try again shortly."
    return 502, "The answer service is temporarily unavailable. Please try again."


# HTTP statuses that indicate a key-specific problem (not a request problem)
# and so are worth retrying with a fallback key.
_KEY_LEVEL_STATUSES = {401, 403, 429}


def _keys() -> list[str]:
    keys = config.sarvam_api_keys()
    if not keys:
        raise SarvamError("SARVAM_API_KEY is not set. Add it to .env.")
    return keys


def _with_key_retry(do_request: Callable[[str], requests.Response], label: str) -> requests.Response:
    """Run `do_request(key)` against each available key in turn. Return the
    first response that is either successful or fails for a non-key reason.
    """
    keys = _keys()
    last: requests.Response | None = None
    for i, key in enumerate(keys):
        r = do_request(key)
        if r.status_code < 400 or r.status_code not in _KEY_LEVEL_STATUSES:
            return r
        last = r
        # try the next key, if any
    assert last is not None
    raise SarvamError(f"{label} {last.status_code} (all keys exhausted): {last.text[:300]}")


def stt_transcribe(audio_path: str, language_code: str = "en-IN",
                   model: str | None = None, with_timestamps: bool = True) -> dict[str, Any]:
    """POST /speech-to-text. Works on audio files <30s (REST limit).
    Returns the raw response dict: { transcript, language_code, timestamps?: {...} }.
    """
    url = f"{config.SARVAM_BASE_URL}/speech-to-text"
    model = model or config.SARVAM_STT_MODEL

    def do(key: str) -> requests.Response:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.rsplit("/", 1)[-1], f, "audio/mpeg")}
            data = {
                "model": model,
                "language_code": language_code,
                "with_timestamps": "true" if with_timestamps else "false",
            }
            return requests.post(
                url, headers={"api-subscription-key": key},
                files=files, data=data, timeout=120,
            )

    r = _with_key_retry(do, "STT")
    if r.status_code >= 400:
        raise SarvamError(f"STT {r.status_code}: {r.text[:300]}")
    return r.json()


def chat_complete(messages: list[dict[str, str]], *, model: str | None = None,
                  temperature: float = 0.1, max_tokens: int = 800) -> str:
    """POST /v1/chat/completions. Returns the assistant text content.

    Current Sarvam chat models (sarvam-30b/105b) are reasoning models: by default
    they emit a chain-of-thought into a separate message.reasoning_content which
    we never use but which still burns max_tokens (left on, it spent the whole
    budget reasoning and returned content=null). Per Sarvam's docs, reasoning is
    disabled by sending reasoning_effort=null — that gives only the final answer,
    ~20x fewer completion tokens, and ~3s instead of ~90s responses. Exactly what
    a grounded JSON-out tutor wants; we don't need the model to "think aloud".
    """
    url = f"{config.SARVAM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": model or config.SARVAM_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": None,  # JSON null → reasoning OFF (the string "none" is rejected)
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    def do(key: str) -> requests.Response:
        # Reasoning models are slower than the old sarvam-m; give them headroom.
        return requests.post(
            url,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            data=body_bytes, timeout=150,
        )

    r = _with_key_retry(do, "Chat")
    if r.status_code >= 400:
        raise SarvamError(f"Chat {r.status_code}: {r.text[:300]}")
    body = r.json()
    try:
        # Current Sarvam models (sarvam-30b/105b) are reasoning models: the answer
        # is in message.content and the chain-of-thought in a separate
        # message.reasoning_content (which we ignore). content can be null if the
        # token budget was exhausted on reasoning — coerce to "" so the caller
        # degrades to a graceful refusal/retry instead of a 500.
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise SarvamError(f"Unexpected chat response shape: {body}") from e


def text_to_speech(text: str, *, target_language_code: str, speaker: str | None = None,
                   model: str | None = None) -> bytes:
    """POST /text-to-speech (Bulbul). Returns decoded WAV audio bytes.

    Sarvam returns base64-encoded WAV chunks in `audios`; for long text it may
    return several. We concatenate by decoding each and stitching the PCM —
    but since each chunk is a self-contained WAV, the simplest robust approach
    is to send text already chunked under the model limit and use the first
    audio. Callers pre-chunk; here we just decode and join raw bytes.
    """
    import base64
    url = f"{config.SARVAM_BASE_URL}/text-to-speech"
    payload = {
        "text": text,
        "target_language_code": target_language_code,
        "model": model or config.SARVAM_TTS_MODEL,
        "speaker": speaker or config.SARVAM_TTS_SPEAKER,
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    def do(key: str) -> requests.Response:
        return requests.post(
            url,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            data=body_bytes, timeout=120,
        )

    r = _with_key_retry(do, "TTS")
    if r.status_code >= 400:
        raise SarvamError(f"TTS {r.status_code}: {r.text[:300]}")
    body = r.json()
    audios = body.get("audios") or []
    if not audios:
        raise SarvamError(f"Unexpected TTS response: {body}")
    # Each entry is a base64 WAV. Return the first; callers chunk text to fit.
    return base64.b64decode(audios[0])


def translate(text: str, *, source_language_code: str, target_language_code: str,
              mode: str = "code-mixed", model: str | None = None) -> str:
    """POST /translate. Mayura with mode=code-mixed natively keeps English
    technical terms in English when translating into Indic languages.
    """
    url = f"{config.SARVAM_BASE_URL}/translate"
    payload = {
        "input": text,
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
        "model": model or config.SARVAM_TRANSLATE_MODEL,
        "mode": mode,
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    def do(key: str) -> requests.Response:
        return requests.post(
            url,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            data=body_bytes, timeout=60,
        )

    r = _with_key_retry(do, "Translate")
    if r.status_code >= 400:
        raise SarvamError(f"Translate {r.status_code}: {r.text[:300]}")
    body = r.json()
    if "translated_text" not in body:
        raise SarvamError(f"Unexpected translate response: {body}")
    return body["translated_text"]
