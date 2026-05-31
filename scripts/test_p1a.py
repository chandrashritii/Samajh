"""CLI acceptance run for the P1-A state (concept map, misconception, register
dial) + P0 regression (grounded answer, refusal, Hindi term preservation).

Usage:
    python scripts/test_p1a.py [--url <YouTube URL>] [--skip-vernacular]

Runs ONE end-to-end pass against the cached ML lecture. Designed to be the
single verification run after a build — does not call Sarvam more than needed
(concept extraction is cached on second run).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import (
    answering, concept_map, config, indexing, ingestion, mastery, retrieval, translation,
)
from backend.schemas import Register


DEFAULT_URL = "https://www.youtube.com/watch?v=jGwO_UgTS7I"  # Andrew Ng CS229 2018 L1


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
    print(f"[ingest] {len(segments)} segments → {len(chunks)} chunks ({source})")
    return vid


def ensure_concepts(vid: str):
    if concept_map.has_cached(vid):
        return concept_map.load(vid)
    _, chunks, _ = indexing.load(vid)
    print(f"[concepts] extracting (one LLM call)…")
    return concept_map.extract_and_persist(vid, chunks)


def ask(vid: str, question: str, language: str = "en",
        register: Register = Register.BALANCED):
    index, chunks, _ = indexing.load(vid)
    hits = retrieval.top_k(index, chunks, question, k=config.RETRIEVAL_K)
    result = answering.answer(question, hits)
    final = translation.translate_answer(result.answer_en, language, register) if language != "en" else result.answer_en
    misc_final = ""
    corr_final = ""
    if result.misconception.detected:
        misc_final = translation.translate_answer(
            result.misconception.misconception or "", language, register,
        ) if language != "en" else (result.misconception.misconception or "")
        corr_final = translation.translate_answer(
            result.misconception.correction or "", language, register,
        ) if language != "en" else (result.misconception.correction or "")
    return result, final, misc_final, corr_final, hits


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--skip-vernacular", action="store_true",
                   help="Skip the Hindi register comparison (saves translate credits).")
    args = p.parse_args()

    failures: list[str] = []

    header(f"N1 — concept map for {args.url}")
    vid = ingest_or_load(args.url)
    concepts = ensure_concepts(vid)
    print(f"concepts: {len(concepts)}")
    for c in concepts[:8]:
        print(f"  - {c['id']:<28}  ({len(c['chunk_ids'])} chunks, first @ {c['first_timestamp']:.0f}s)  {c['name']}")
    if len(concepts) < 5:
        failures.append(f"N1: expected ≥5 concepts, got {len(concepts)}")

    # ------- P0 regression + F3 + F4 attribution + mastery -----------------
    session_id = mastery.new_session_id()
    print(f"\n[mastery] session_id={session_id}")

    header("P0a — in-lecture question (expect grounded + concept attribution)")
    q = "What is gradient descent?"
    print(f"Q: {q}")
    result, final, misc_final, corr_final, hits = ask(vid, q)
    touched = concept_map.attribute([h.chunk.idx for h in hits], concepts)
    print(f"top sim: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}  citations={result.citation_indices}")
    print(f"concepts_touched={touched}")
    print(f"misconception.detected={result.misconception.detected}")
    print(f"A: {final[:400]}")
    if not result.grounded:
        failures.append("P0a: expected grounded=True")
    if not touched:
        failures.append("P0a: expected concept attribution")
    if result.misconception.detected:
        failures.append("P0a: false-positive misconception flag on a normal question")
    if result.grounded and touched:
        mastery.record_engagement(session_id, vid, touched, q)

    header("P0b — out-of-lecture question (expect refusal)")
    q = "What is the capital of France?"
    print(f"Q: {q}")
    result, final, _, _, hits = ask(vid, q)
    print(f"top sim: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}  misconception.detected={result.misconception.detected}")
    print(f"A: {final}")
    if result.grounded:
        failures.append("P0b: expected refusal")
    if result.misconception.detected:
        failures.append("P0b: should not flag misconception on off-topic")

    header("F3a — misconception (expect flag + grounded correction)")
    q = "Supervised learning doesn't need labeled data, right?"
    print(f"Q: {q}")
    result, final, misc_final, corr_final, hits = ask(vid, q)
    touched = concept_map.attribute([h.chunk.idx for h in hits], concepts)
    print(f"top sim: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}  citations={result.citation_indices}  concepts_touched={touched}")
    print(f"misconception.detected={result.misconception.detected}")
    print(f"misconception: {result.misconception.misconception}")
    print(f"correction:   {result.misconception.correction}")
    print(f"A: {final[:400]}")
    if not result.misconception.detected:
        failures.append("F3a: expected misconception flag")
    if result.misconception.detected and touched:
        mastery.record_misconception(session_id, vid, touched, result.misconception.misconception or "")

    header("F3b — misconception about something the lecture doesn't cover")
    q = "Quantum entanglement makes neural networks faster, doesn't it?"
    print(f"Q: {q}")
    result, final, misc_final, corr_final, hits = ask(vid, q)
    print(f"top sim: {[round(h.score, 3) for h in hits]}")
    print(f"grounded={result.grounded}  misconception.detected={result.misconception.detected}")
    print(f"A: {final[:300]}")
    if result.misconception.detected:
        # Allowed only if the correction is grounded — but for this off-topic question
        # we expect the lecture to lack the material, so detection should be false.
        failures.append("F3b: should not fabricate a correction for off-topic content")

    header("F4 — mastery snapshot after the ladder of questions above")
    rows = mastery.list_mastery(session_id, concepts)
    interesting = [r for r in rows if r["state"] != "unseen"]
    for r in interesting:
        print(f"  {r['concept_id']:<28}  state={r['state']:<13}  score={r['score']:.2f}")
    if not interesting:
        failures.append("F4: expected at least one concept to have moved off 'unseen'")

    # ------- F1+ register dial (vernacular comparison) ---------------------
    if not args.skip_vernacular:
        header("F1+ — register dial (Hindi): vernacular vs english")
        q = "What is supervised learning?"
        print(f"Q: {q}  lang=hi")
        result, _, _, _, _ = ask(vid, q, language="en")
        en_answer = result.answer_en
        print(f"A (en): {en_answer[:240]}")
        hi_v = translation.translate_answer(en_answer, "hi", Register.MORE_VERNACULAR)
        hi_b = translation.translate_answer(en_answer, "hi", Register.BALANCED)
        hi_e = translation.translate_answer(en_answer, "hi", Register.MORE_ENGLISH)
        print(f"\n--- more_vernacular ---\n{hi_v}")
        print(f"\n--- balanced ---\n{hi_b}")
        print(f"\n--- more_english ---\n{hi_e}")
        # Spec: more_vernacular vs more_english must yield a visibly different blend.
        # (balanced may land between them and that's fine.)
        if hi_v == hi_e:
            failures.append("F1+: more_vernacular and more_english produced identical output")
        for label, txt in (("more_vernacular", hi_v), ("balanced", hi_b), ("more_english", hi_e)):
            if "gradient" not in txt.lower() and "supervised" not in txt.lower() and "learning" not in txt.lower():
                # very loose check — at least one tech term should survive untranslated
                failures.append(f"F1+: no English technical term survived in {label}")

    print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILURE(S):"))
    for f in failures:
        print(f"  - {f}")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
