"""Cross-lingual retrieval support (retrieval / query-prep layer only).

The embedding model (MiniLM) is English-leaning, so an English query scores ~0
against a Hindi transcript and the refusal floor rejects everything. Fix:

  • At INGEST, a non-English lecture is embedded from English translations of its
    chunks (the ORIGINAL text is still stored for display/answering), so the
    index lives in one English-ish space.
  • At QUERY, a Devanagari/Tamil-script question is translated to English before
    embedding. English and romanized queries embed raw (they already match the
    English-embedded index).

This does NOT change answer generation, the 0.25 floor, or English lectures
(every hook is a no-op when the lecture's transcript_lang is 'en').
"""
from __future__ import annotations

from . import sarvam_client
from .indexing import Chunk

# Unicode block ranges for script detection.
_DEVANAGARI = (0x0900, 0x097F)
_TAMIL = (0x0B80, 0x0BFF)

# Mayura is capped ~1000 chars/call; bound the per-chunk translate input. The
# full original text is preserved in chunks.json regardless — this only bounds
# the string we embed.
_TRANSLATE_CAP = 950

# Small in-process cache so a repeated query (e.g. a suggested-question chip)
# doesn't re-hit the translate API.
_query_cache: dict[str, str] = {}


def _script_counts(text: str) -> tuple[int, int]:
    deva = tamil = 0
    for ch in text:
        o = ord(ch)
        if _DEVANAGARI[0] <= o <= _DEVANAGARI[1]:
            deva += 1
        elif _TAMIL[0] <= o <= _TAMIL[1]:
            tamil += 1
    return deva, tamil


def has_indic_script(text: str) -> bool:
    d, t = _script_counts(text)
    return (d + t) > 0


def detect_transcript_lang(chunks: list[Chunk]) -> str:
    """Sample the transcript and classify its dominant script → 'hi'|'ta'|'en'."""
    sample = " ".join(c.text for c in chunks[:8])
    deva, tamil = _script_counts(sample)
    if deva >= 20 and deva >= tamil:
        return "hi"
    if tamil >= 20:
        return "ta"
    return "en"


def _lang_to_code(lang: str) -> str:
    return {"hi": "hi-IN", "ta": "ta-IN", "en": "en-IN"}.get(lang, "en-IN")


def translate_chunks_to_english(chunks: list[Chunk], src_lang: str) -> list[str]:
    """Translate each chunk's text → English for embedding (1:1 with chunks).

    Firing ~100 translate calls back-to-back can trip transient rate limits, so
    each call gets a short retry with backoff and light pacing. On a hard
    per-chunk failure we fall back to the original text (degraded, never empty),
    and the caller logs how many rows fell back so an all-fallback (which would
    silently defeat cross-lingual retrieval) can't pass unnoticed."""
    import logging
    import time
    log = logging.getLogger("tutor.multilingual")
    src = _lang_to_code(src_lang)
    out: list[str] = []
    fell_back = 0
    for c in chunks:
        text = c.text[:_TRANSLATE_CAP]
        translated = None
        for attempt in range(3):
            try:
                translated = sarvam_client.translate(
                    text, source_language_code=src, target_language_code="en-IN",
                    mode="formal",
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log.warning("chunk translate failed after retries: %s", e)
                else:
                    time.sleep(0.6 * (attempt + 1))
        if translated is None:
            translated = c.text
            fell_back += 1
        out.append(translated)
        time.sleep(0.05)  # gentle pacing across the batch
    if fell_back:
        log.warning("translate_chunks_to_english: %d/%d chunks fell back to original",
                    fell_back, len(chunks))
    return out


def embedding_inputs(chunks: list[Chunk]) -> tuple[list[str] | None, str]:
    """Return (embed_texts_or_None, transcript_lang) for an ingest.
    English → (None, 'en') = embed the chunks' own text (unchanged behavior).
    Hindi/Tamil → (English translations, 'hi'|'ta')."""
    lang = detect_transcript_lang(chunks)
    if lang == "en":
        return None, "en"
    return translate_chunks_to_english(chunks, lang), lang


def _script_source_lang(text):
    """Pick a Sarvam source language from the query's dominant Indic script, or
    None for Latin/ASCII (which embeds fine against the English-ish index)."""
    deva, tamil = _script_counts(text)
    if deva == 0 and tamil == 0:
        return None
    return "hi" if deva >= tamil else "ta"


def prepare_query(question: str, transcript_lang: str) -> str:
    """The text to EMBED for retrieval.

    Every index embeds in an English-ish space — English lectures embed their own
    English text; non-English lectures embed English translations of their chunks.
    So ANY Indic-script query (Devanagari/Tamil) must be translated to English
    before embedding, regardless of which lecture it's asked against — otherwise a
    Hindi question scores ~0 even on an English lecture and is wrongly refused.
    Latin-script (English or romanized) queries pass through unchanged. The
    original question still drives answer generation; only the retrieval vector
    changes. transcript_lang is kept for signature stability but no longer gates."""
    src = _script_source_lang(question)
    if src is None:
        return question
    cached = _query_cache.get(question)
    if cached is not None:
        return cached
    try:
        en = sarvam_client.translate(
            question[:_TRANSLATE_CAP],
            source_language_code=_lang_to_code(src),
            target_language_code="en-IN", mode="formal",
        )
    except Exception:  # noqa: BLE001 — fall back to the raw query
        en = question
    _query_cache[question] = en
    return en


# Cache for the romanized/Hinglish refusal-path fallback (separate from the
# script-based query cache above).
_fallback_cache: dict[str, str] = {}


def english_fallback(question: str) -> str | None:
    """Translate a Latin-script query to English via Mayura auto-detect.

    Script-based `prepare_query` only catches Devanagari/Tamil; a romanized /
    Hinglish query like "column picture kya hota hai?" stays in Latin script,
    embeds raw, and MiniLM scores it just under the floor. This is the refusal-
    path rescue: auto-detect-translate the query to English so it can be
    re-retrieved. Returns the English string, or None if translation failed or
    produced no real change (e.g. the query was already English) — so the caller
    can skip a pointless re-retrieval. Cached per query.
    """
    q = question.strip()
    if not q:
        return None
    cached = _fallback_cache.get(question)
    if cached is not None:
        return cached or None
    try:
        en = sarvam_client.translate(
            question[:_TRANSLATE_CAP],
            source_language_code="auto", target_language_code="en-IN", mode="formal",
        )
    except Exception:  # noqa: BLE001
        en = ""
    en = (en or "").strip()
    result = en if (en and en.lower() != q.lower()) else ""
    _fallback_cache[question] = result
    return result or None
