"""Top-k retrieval over a persisted FAISS index."""
from __future__ import annotations
from dataclasses import dataclass

import faiss
import numpy as np

from .indexing import Chunk, embed_query


@dataclass
class Hit:
    chunk: Chunk
    score: float


def top_k(index: faiss.Index, chunks: list[Chunk], question: str, k: int) -> list[Hit]:
    qv = embed_query(question)
    k = min(k, len(chunks))
    if k == 0:
        return []
    scores, idxs = index.search(qv, k)
    out: list[Hit] = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        out.append(Hit(chunk=chunks[i], score=float(score)))
    return out
