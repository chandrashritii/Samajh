"""Voice layer: speech-in (Saaras STT) and speech-out (Bulbul TTS) + audio store.

Thin orchestration over sarvam_client — no answer/grounding logic lives here.
Synthesized audio is persisted to data/audio/{id}.wav and served by a GET route
so the text path, the voice path, and a future avatar layer all consume the
same retrievable file.
"""
from __future__ import annotations
import re
import uuid
import wave
from pathlib import Path

from . import config, sarvam_client

_AUDIO_DIR = config.DATA_DIR / "audio"
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def _audio_dir() -> Path:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIO_DIR


def audio_path(audio_id: str) -> Path:
    # audio_id is a hex token we mint ourselves; keep it filesystem-safe.
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", audio_id)
    return _audio_dir() / f"{safe}.wav"


def transcribe(upload_path: str, language_code: str = "unknown") -> dict:
    """STT a short uploaded clip. `unknown` lets Saaras auto-detect / code-mix.
    Returns {transcript, language_code, ...} (raw Saaras shape)."""
    return sarvam_client.stt_transcribe(upload_path, language_code=language_code)


def _tts_chunks(text: str, limit: int) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    out: list[str] = []
    buf = ""
    for s in _SENT_SPLIT.split(text):
        if not s:
            continue
        if len(buf) + len(s) + 1 <= limit:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                out.append(buf)
            # A single over-long sentence is hard-split on the char limit.
            buf = s if len(s) <= limit else s[:limit]
    if buf:
        out.append(buf)
    return out


def _concat_wavs(wav_blobs: list[bytes], dest: Path) -> None:
    """Stitch multiple self-contained WAV blobs into one WAV file by appending
    PCM frames. Assumes a common format (Bulbul returns consistent settings)."""
    import io
    if not wav_blobs:
        # Write a valid, silent 1-frame WAV so the file always exists.
        with wave.open(str(dest), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"")
        return
    params = None
    frames: list[bytes] = []
    for blob in wav_blobs:
        with wave.open(io.BytesIO(blob), "rb") as r:
            if params is None:
                params = r.getparams()
            frames.append(r.readframes(r.getnframes()))
    with wave.open(str(dest), "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)


def synthesize(text: str, language: str, *, speaker: str | None = None) -> str | None:
    """TTS `text` in `language` ('en'|'hi'|'ta'), persist a WAV, return its
    audio_id (None if there's nothing to say). The answer text is already in the
    target language (translation happened upstream), so we pass the matching
    Bulbul target-language code for correct prosody."""
    text = (text or "").strip()
    if not text:
        return None
    target = config.LANG_TO_SARVAM_CODE.get(language, "en-IN")
    blobs = [
        sarvam_client.text_to_speech(chunk, target_language_code=target, speaker=speaker)
        for chunk in _tts_chunks(text, config.TTS_CHAR_LIMIT)
    ]
    if not blobs:
        return None
    audio_id = uuid.uuid4().hex[:16]
    _concat_wavs(blobs, audio_path(audio_id))
    return audio_id


def speak_answer(answer_text: str, language: str, *, speaker: str | None = None) -> str | None:
    """Public hook used by /ask_voice and the viva. Kept as a named seam so the
    avatar layer can later intercept (text → audio → visemes) in one place."""
    return synthesize(answer_text, language, speaker=speaker)
