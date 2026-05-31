"""URL → timestamped transcript segments. Captions first, Saaras STT fallback."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
)

from . import config, transcription


_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Accept the common YouTube URL shapes plus a bare id."""
    url = url.strip()
    if _YT_ID.match(url):
        return url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.endswith("youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        if _YT_ID.match(vid):
            return vid
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
            if _YT_ID.match(vid):
                return vid
        for prefix in ("/shorts/", "/embed/", "/v/"):
            if parsed.path.startswith(prefix):
                vid = parsed.path[len(prefix):].split("/")[0]
                if _YT_ID.match(vid):
                    return vid
    raise ValueError(f"Could not extract a YouTube video id from: {url}")


def _segments_from_fetched(fetched) -> list[dict]:
    """Normalize a youtube-transcript-api FetchedTranscript into our segment dicts."""
    out: list[dict] = []
    for snippet in fetched:
        start = float(getattr(snippet, "start", 0.0))
        dur = float(getattr(snippet, "duration", 0.0))
        text = (getattr(snippet, "text", "") or "").replace("\n", " ").strip()
        if text:
            out.append({"start": start, "end": start + dur, "text": text})
    return out


def fetch_captions(video_id: str) -> list[dict] | None:
    """Return [{start, end, text}] from YouTube captions, or None if unavailable.
    Preference: manual English → any other English locale → auto-generated English
    → first available, translated to English if possible.
    """
    api = YouTubeTranscriptApi()
    try:
        listing = api.list(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        return None

    chosen = None
    try:
        chosen = listing.find_manually_created_transcript(["en", "en-US", "en-GB"])
    except Exception:
        pass
    if chosen is None:
        try:
            chosen = listing.find_generated_transcript(["en", "en-US", "en-GB"])
        except Exception:
            pass
    if chosen is None:
        for t in listing:
            try:
                chosen = t.translate("en") if t.is_translatable else t
                break
            except Exception:
                continue
    if chosen is None:
        return None

    try:
        fetched = chosen.fetch()
    except Exception:
        return None
    segments = _segments_from_fetched(fetched)
    return segments or None


def ingest(url: str) -> tuple[str, list[dict], Literal["captions", "stt"]]:
    """Resolve URL → (video_id, segments, source)."""
    video_id = extract_video_id(url)
    segments = fetch_captions(video_id)
    if segments:
        return video_id, segments, "captions"

    work = config.DATA_DIR / video_id / "audio_work"
    audio_path = transcription.download_audio(url, work)
    segments = transcription.transcribe_with_saaras(Path(audio_path))
    if not segments:
        raise RuntimeError("Saaras returned no segments — transcription failed.")
    return video_id, segments, "stt"
