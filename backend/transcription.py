"""Saaras STT fallback path: yt-dlp pulls audio → chunk → Sarvam STT → stitch.

Used only when a YouTube video has no captions. The Sarvam REST STT endpoint
is capped at ~30s per request, so we chunk via ffmpeg and offset timestamps.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path

from . import sarvam_client


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def download_audio(youtube_url: str, out_dir: Path) -> Path:
    """Download bestaudio as m4a using yt-dlp. Returns the audio path."""
    if not _have("yt-dlp"):
        raise RuntimeError("yt-dlp not found on PATH. Install it to use the STT fallback.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "audio.%(ext)s")
    cmd = [
        "yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio",
        "--no-playlist", "-o", out_template, youtube_url,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    for ext in ("m4a", "webm", "mp3", "opus"):
        candidate = out_dir / f"audio.{ext}"
        if candidate.exists():
            return candidate
    raise RuntimeError("yt-dlp finished but no audio file was produced.")


def _duration_seconds(audio_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(audio_path)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def _slice_audio(audio_path: Path, start: float, dur: float, dest: Path) -> None:
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(audio_path),
           "-t", f"{dur:.2f}", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_with_saaras(audio_path: Path, language_code: str = "en-IN",
                           chunk_seconds: float = 25.0) -> list[dict]:
    """Transcribe a long audio file by chunking. Returns segments with
    {start, end, text} keyed to original-audio timestamps.
    """
    if not _have("ffmpeg") or not _have("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe required for STT chunking. Install via brew/apt.")

    total = _duration_seconds(audio_path)
    work = audio_path.parent / "chunks"
    work.mkdir(exist_ok=True)

    segments: list[dict] = []
    t = 0.0
    idx = 0
    while t < total:
        dur = min(chunk_seconds, total - t)
        chunk_path = work / f"chunk_{idx:04d}.wav"
        _slice_audio(audio_path, t, dur, chunk_path)
        resp = sarvam_client.stt_transcribe(str(chunk_path), language_code=language_code)
        text = (resp.get("transcript") or "").strip()
        if text:
            segments.append({"start": t, "end": t + dur, "text": text})
        t += dur
        idx += 1
    return segments
