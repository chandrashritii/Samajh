"""Ingest a YouTube lecture locally (residential IP) and add it to the demo
library so the deployed host can serve it instantly without any YouTube access.

Runs the EXISTING pipeline (captions -> chunks -> MiniLM/FAISS -> concept map),
verifies a non-empty, sensible concept map, copies the built index into seed/
(bundled in the Docker image), and upserts the entry into frontend/samajh/library.json.

Usage:
  python scripts/add_to_library.py "<youtube-url-or-id>" \
      --subject "Biology" --lang en \
      --questions "Q1?" "Q2?" "Q3?" \
      [--title "..."] [--channel "..."] [--experimental] [--order N] [--no-commit-seed]

Idempotent: re-running updates the existing library.json entry and re-uses the
cached index. Use --force to re-ingest from scratch.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import concept_map, config, indexing, ingestion  # noqa: E402

LIBRARY_PATH = ROOT / "frontend" / "samajh" / "library.json"
SEED_DIR = ROOT / "seed"


def load_library() -> list[dict]:
    if LIBRARY_PATH.exists():
        try:
            return json.loads(LIBRARY_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_library(entries: list[dict]) -> None:
    entries.sort(key=lambda e: e.get("order", 999))
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")


def copy_to_seed(video_id: str) -> None:
    src = config.DATA_DIR / video_id
    dst = SEED_DIR / video_id
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("index.faiss", "chunks.json", "meta.json", "concepts.json"):
        s = src / fn
        if s.exists():
            shutil.copy2(s, dst / fn)


def ingest_video(url: str, force: bool) -> tuple[str, dict]:
    video_id = ingestion.extract_video_id(url)
    if force:
        shutil.rmtree(config.DATA_DIR / video_id, ignore_errors=True)

    if indexing.has_cached_index(video_id):
        _, chunks, meta = indexing.load(video_id)
        source = meta.get("transcript_source", "cached")
    else:
        video_id, segments, source = ingestion.ingest(url)
        chunks = indexing.chunk_segments(segments)
        if not chunks:
            raise RuntimeError("Transcript produced no usable chunks (sparse/no captions).")
        duration = float(segments[-1]["end"]) if segments else 0.0
        indexing.build_and_persist(
            video_id, chunks, {"title": video_id, "duration": duration, "transcript_source": source})

    # Concept map (extract if absent).
    concepts = concept_map.load(video_id)
    if not concepts:
        concepts = concept_map.extract_and_persist(video_id, chunks)

    return video_id, {
        "chunks": len(chunks), "concepts": concepts, "source": source,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--subject", required=True)
    p.add_argument("--lang", default="en")
    p.add_argument("--questions", nargs="*", default=[])
    p.add_argument("--title", default="")
    p.add_argument("--channel", default="")
    p.add_argument("--experimental", action="store_true")
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-commit-seed", action="store_true")
    args = p.parse_args()

    print(f"[ingest] {args.url}")
    try:
        video_id, info = ingest_video(args.url, args.force)
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    concepts = info["concepts"]
    n = len(concepts)
    auto = any(c.get("auto_derived") for c in concepts)
    print(f"  video_id={video_id} chunks={info['chunks']} concepts={n} "
          f"source={info['source']} auto_derived={auto}")
    for c in concepts[:8]:
        print(f"    - {c['id']}")

    # Sanity gate: a usable library entry needs a non-empty, mostly-real map.
    if n == 0:
        print("  REJECTED: empty concept map — do not add to library.")
        return 2
    if auto:
        print("  WARNING: concept map fell back to chunk-derived (LLM extraction failed). "
              "Review before relying on it.")

    # Upsert into library.json
    lib = load_library()
    existing = next((e for e in lib if e["video_id"] == video_id), None)
    order = args.order if args.order is not None else (
        existing["order"] if existing else (max([e["order"] for e in lib], default=0) + 1))
    entry = {
        "order": order,
        "video_id": video_id,
        "title": args.title or (existing or {}).get("title") or video_id,
        "channel": args.channel or (existing or {}).get("channel") or "",
        "subject": args.subject,
        "lang": args.lang,
        "experimental": args.experimental,
        "suggested_questions": args.questions or (existing or {}).get("suggested_questions", []),
    }
    lib = [e for e in lib if e["video_id"] != video_id] + [entry]
    save_library(lib)
    print(f"  library.json: upserted order={order} ({len(lib)} total)")

    if not args.no_commit_seed:
        copy_to_seed(video_id)
        print(f"  seed/{video_id}: copied for image bundling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
