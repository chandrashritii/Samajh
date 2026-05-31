"""CLI acceptance run for the P0 spine.

Usage:
    python scripts/test_p0.py [--url <YouTube URL>]

Runs the three F2 acceptance probes plus the F3 vernacular probe against the
running ingest → retrieve → answer → translate path. Prints concise PASS/FAIL.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# allow running as a script: python scripts/test_p0.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import answering, config, indexing, ingestion, retrieval, translation


DEFAULT_URL = "https://www.youtube.com/watch?v=jGwO_UgTS7I"  # Andrew Ng CS229 (2018) Lecture 1
IN_LECTURE_Q = "What is gradient descent?"
OUT_OF_LECTURE_Q = "What is the capital of France?"
HINDI_Q = "What is supervised learning?"


def header(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def ingest_or_load(url: str) -> str:
    vid = ingestion.extract_video_id(url)
    if indexing.has_cached_index(vid):
        print(f"[cache] using existing index for {vid}")
        return vid
    print(f"[ingest] fetching transcript and indexing {vid}…")
    vid, segments, source = ingestion.ingest(url)
    chunks = indexing.chunk_segments(segments)
    duration = float(segments[-1]["end"]) if segments else 0.0
    indexing.build_and_persist(vid, chunks, {
        "title": vid, "duration": duration, "transcript_source": source,
    })
    print(f"[ingest] {len(segments)} segments → {len(chunks)} chunks ({source}, {duration:.0f}s)")
    return vid


def ask(vid: str, question: str, language: str = "en"):
    index, chunks, _ = indexing.load(vid)
    hits = retrieval.top_k(index, chunks, question, k=config.RETRIEVAL_K)
    result = answering.answer(question, hits)
    final = translation.translate_answer(result.answer_en, language) if language != "en" else result.answer_en
    return result, final, hits


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()

    header(f"M1 — ingest {args.url}")
    vid = ingest_or_load(args.url)
    _, chunks, meta = indexing.load(vid)
    print(f"video_id={vid}  chunks={len(chunks)}  duration={meta.get('duration', 0):.0f}s  source={meta.get('transcript_source')}")
    print(f"first chunk: [{chunks[0].start:.1f}-{chunks[0].end:.1f}] {chunks[0].text[:120]}…")

    failures = 0

    header(f"M2.a — in-lecture question (expect grounded answer)")
    print(f"Q: {IN_LECTURE_Q}")
    result, final, hits = ask(vid, IN_LECTURE_Q, "en")
    print(f"top sim scores: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}  citations={result.citation_indices}")
    print(f"A: {final}")
    if not result.grounded:
        print("FAIL: expected grounded=True")
        failures += 1

    header(f"M2.b — out-of-lecture question (expect refusal)")
    print(f"Q: {OUT_OF_LECTURE_Q}")
    result, final, hits = ask(vid, OUT_OF_LECTURE_Q, "en")
    print(f"top sim scores: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}")
    print(f"A: {final}")
    if result.grounded:
        print("FAIL: expected refusal (grounded=False)")
        failures += 1

    header("M3 — Hindi answer with technical-term preservation")
    print(f"Q: {HINDI_Q}  lang=hi")
    result, final, hits = ask(vid, HINDI_Q, "hi")
    print(f"grounded={result.grounded}  citations={result.citation_indices}")
    print(f"A (en): {result.answer_en}")
    print(f"A (hi): {final}")

    print("\n" + ("ALL PASSED" if failures == 0 else f"{failures} FAILURE(S)"))
    return failures


if __name__ == "__main__":
    sys.exit(main())
