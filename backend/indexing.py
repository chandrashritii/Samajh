"""Chunk segments → embed → FAISS. Persisted under data/{video_id}/."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from . import config


@dataclass
class Chunk:
    idx: int
    start: float
    end: float
    text: str


# Lazy-loaded singleton — sentence-transformers loads ~80MB at first use.
_model = None


def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def chunk_segments(segments: list[dict],
                   target_seconds: float | None = None,
                   overlap_segments: int | None = None) -> list[Chunk]:
    """Coalesce caption segments into ~target_seconds windows with light overlap."""
    target = target_seconds if target_seconds is not None else config.CHUNK_TARGET_SECONDS
    overlap = overlap_segments if overlap_segments is not None else config.CHUNK_OVERLAP_SEGMENTS

    if not segments:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(segments):
        start = segments[i]["start"]
        j = i
        text_parts: list[str] = []
        while j < len(segments) and (segments[j]["end"] - start) < target:
            text_parts.append(segments[j]["text"])
            j += 1
        if j == i:
            j = i + 1
            text_parts = [segments[i]["text"]]
        end = segments[j - 1]["end"]
        text = " ".join(text_parts).strip()
        if text:
            chunks.append(Chunk(idx=len(chunks), start=start, end=end, text=text))
        if j >= len(segments):
            break
        # advance with overlap
        i = max(j - overlap, i + 1)
    return chunks


def _dir(video_id: str) -> Path:
    d = config.DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_paths(video_id: str) -> tuple[Path, Path, Path]:
    d = _dir(video_id)
    return d / "index.faiss", d / "chunks.json", d / "meta.json"


def has_cached_index(video_id: str) -> bool:
    fi, fc, fm = index_paths(video_id)
    return fi.exists() and fc.exists() and fm.exists()


def build_and_persist(video_id: str, chunks: list[Chunk], meta: dict,
                      embed_texts: list[str] | None = None) -> None:
    """Build + persist the FAISS index and chunk store.

    `embed_texts`, when given, are the strings actually embedded (must align 1:1
    with `chunks`). This lets a non-English lecture be embedded from English
    translations for cross-lingual retrieval while `chunks.json` keeps the
    ORIGINAL text for display/answering. Default embeds the chunks' own text.
    """
    model = _embedder()
    texts = embed_texts if embed_texts is not None else [c.text for c in chunks]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype="float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    fi, fc, fm = index_paths(video_id)
    faiss.write_index(index, str(fi))
    fc.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False))
    fm.write_text(json.dumps(meta, ensure_ascii=False))


def load(video_id: str) -> tuple[faiss.Index, list[Chunk], dict]:
    fi, fc, fm = index_paths(video_id)
    if not (fi.exists() and fc.exists() and fm.exists()):
        raise FileNotFoundError(f"No cached index for {video_id}")
    index = faiss.read_index(str(fi))
    raw = json.loads(fc.read_text())
    chunks = [Chunk(**c) for c in raw]
    meta = json.loads(fm.read_text())
    return index, chunks, meta


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed texts to normalized float32 vectors (one row per text)."""
    model = _embedder()
    v = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])
