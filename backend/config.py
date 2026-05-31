"""Central config. Reads env once, exposes typed constants."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_API_KEY_FALLBACK = os.getenv("SARVAM_API_KEY_FALLBACK", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def sarvam_api_keys() -> list[str]:
    """Ordered list of usable Sarvam keys (primary first), de-duped, no blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for k in (SARVAM_API_KEY, SARVAM_API_KEY_FALLBACK):
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "sarvam").lower()

SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
# Bulbul's per-call text cap; we sentence-chunk above this for TTS.
TTS_CHAR_LIMIT = int(os.getenv("TTS_CHAR_LIMIT", "450"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
GROUNDING_SIM_THRESHOLD = float(os.getenv("GROUNDING_SIM_THRESHOLD", "0.25"))
CHUNK_TARGET_SECONDS = float(os.getenv("CHUNK_TARGET_SECONDS", "45"))
CHUNK_OVERLAP_SEGMENTS = int(os.getenv("CHUNK_OVERLAP_SEGMENTS", "1"))

# All languages with both Sarvam translate (Mayura) and TTS (Bulbul) support,
# verified live. Answers translate + speak in any of these.
SUPPORTED_LANGUAGES = {"en", "hi", "ta", "bn", "gu", "kn", "ml", "mr", "od", "pa", "te"}
LANG_TO_SARVAM_CODE = {
    "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN", "gu": "gu-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "od": "od-IN", "pa": "pa-IN",
    "te": "te-IN",
}

# Bulbul v2 speakers accepted on this tier (verified live). Used to validate the
# optional per-request voice picker; falls back to SARVAM_TTS_SPEAKER otherwise.
SARVAM_TTS_SPEAKERS = {"anushka", "vidya", "arya", "manisha", "abhilash", "karun", "hitesh"}

REFUSAL_EN = "That isn't covered in this lecture."

# Concept map: target band for total concepts per lecture.
CONCEPT_MIN, CONCEPT_MAX = 5, 15

# Mastery scoring — coarse but explicit, easy to tune.
MASTERY_SCORE = {
    "unseen": 0.0,
    "engaged": 0.4,
    "shaky": 0.2,
    "demonstrated": 1.0,
}
