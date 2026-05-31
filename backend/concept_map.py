"""Concept map extraction. One Sarvam chat call over the chunked transcript →
5–15 concepts with one-line descriptions, primary chunk ids, and (optional)
intra-lecture prerequisites. Persisted to data/{video_id}/concepts.json so
re-ingest is free.

The map is the question bank for the future viva (low-mastery → quiz target),
the attribution source for mastery, and the bones of an outline view in the UI.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from . import config, llm
from .indexing import Chunk


_SYSTEM = (
    "Extract a lightweight concept map (5–15 concepts) from a lecture "
    "transcript. For each concept: id (kebab-case), name, one_line_desc (≤18 "
    "words, grounded in the lecture), chunk_ids (integer indices where it is "
    "primarily taught, usually 1–5), prerequisites (ids of OTHER concepts in "
    "this list, intra-lecture only). Skip concepts that are only mentioned in "
    "passing. Use ONLY the transcript. Output ONLY a JSON object: "
    '{"concepts": [...]}. No prose, no fences.'
)


_CHUNK_PREVIEW_WORDS = 12  # keep prompt under sarvam-m's 7192-token window


def _fmt_ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _truncate(text: str, words: int = _CHUNK_PREVIEW_WORDS) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + "…"


# sarvam-m has an 8192-token context window; reserve ~2048 for the completion
# and a margin, leaving room for ~140 chunk-preview lines (~25 tokens each).
# Longer lectures are evenly downsampled so the prompt fits — the kept chunks
# keep their real idx values, so extracted chunk_ids stay valid for attribution.
_MAX_PROMPT_CHUNKS = 140


def _select_for_prompt(chunks: list[Chunk]) -> list[Chunk]:
    if len(chunks) <= _MAX_PROMPT_CHUNKS:
        return chunks
    step = len(chunks) / _MAX_PROMPT_CHUNKS
    idxs = sorted({int(i * step) for i in range(_MAX_PROMPT_CHUNKS)})
    return [chunks[i] for i in idxs if i < len(chunks)]


def _build_user_prompt(chunks: list[Chunk]) -> str:
    # Tight format: "#idx mm:ss preview". Saves ~10 tokens per chunk vs verbose.
    selected = _select_for_prompt(chunks)
    lines = ["CHUNKS (#idx mm:ss preview):"]
    for c in selected:
        lines.append(f"#{c.idx} {_fmt_ts(c.start)} {_truncate(c.text)}")
    lines.append(
        '\nJSON: {"concepts": [{"id","name","one_line_desc","chunk_ids":[int],'
        '"prerequisites":[id]}]}'
    )
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_KEBAB = re.compile(r"[^a-z0-9-]+")


def _slug(s: str) -> str:
    s = _KEBAB.sub("-", s.lower()).strip("-")
    return s or "concept"


def _strip_reasoning(raw: str) -> str:
    """sarvam-m is a reasoning model: it prepends a <think>…</think> block.
    Remove closed blocks; if the block is unclosed (truncated), drop everything
    before the first JSON brace. Also strip code fences.
    """
    s = _THINK.sub("", raw)
    if "<think>" in s.lower() and "}" in s:
        # Unclosed think tag — keep from the first '{' onward.
        i = s.find("{")
        if i >= 0:
            s = s[i:]
    return _FENCE.sub("", s).strip()


def _salvage_objects(s: str) -> list[dict]:
    """Extract complete top-level {...} objects from a (possibly truncated)
    JSON array body. A brace scanner that respects strings/escapes, so a JSON
    array cut off mid-object still yields every complete object before the cut.
    """
    # Narrow to the concepts array if present, else scan the whole string.
    key = s.find('"concepts"')
    start = s.find("[", key) if key >= 0 else s.find("[")
    region = s[start + 1:] if start >= 0 else s

    objs: list[dict] = []
    depth = 0
    in_str = False
    esc = False
    buf: list[str] = []
    for ch in region:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                frag = "".join(buf).strip()
                buf = []
                try:
                    obj = json.loads(frag)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
        elif depth > 0:
            buf.append(ch)
        # chars between objects (commas, whitespace, the trailing `]`) are ignored
    return objs


def _parse(raw: str, chunks: list[Chunk]) -> list[dict]:
    s = _strip_reasoning(raw)
    raw_concepts: list = []
    try:
        body = json.loads(s)
        raw_concepts = body.get("concepts") or []
    except json.JSONDecodeError:
        # Truncated or trailing-garbage JSON — salvage complete objects.
        raw_concepts = _salvage_objects(s)
    seen_ids: set[str] = set()
    valid_chunk_ids = {c.idx for c in chunks}
    out: list[dict] = []

    for c in raw_concepts:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        cid = _slug(str(c.get("id", name)))
        if cid in seen_ids:
            continue
        desc = str(c.get("one_line_desc", "")).strip()

        # Filter chunk_ids to those that actually exist.
        chunk_ids: list[int] = []
        for x in (c.get("chunk_ids") or []):
            try:
                xi = int(x)
                if xi in valid_chunk_ids:
                    chunk_ids.append(xi)
            except (TypeError, ValueError):
                continue
        if not chunk_ids:
            # Drop concepts the model couldn't anchor — they're not useful for attribution.
            continue

        prereqs_raw = c.get("prerequisites") or []
        prereqs = [_slug(str(p)) for p in prereqs_raw if isinstance(p, str)]

        first_ts = min(chunks[i].start for i in chunk_ids)
        out.append({
            "id": cid,
            "name": name,
            "one_line_desc": desc,
            "first_timestamp": first_ts,
            "chunk_ids": sorted(set(chunk_ids)),
            "prerequisites": prereqs,
        })
        seen_ids.add(cid)
        if len(out) >= config.CONCEPT_MAX:
            break

    # Strip prerequisites that don't resolve to a kept concept.
    keep_ids = {c["id"] for c in out}
    for c in out:
        c["prerequisites"] = [p for p in c["prerequisites"] if p in keep_ids and p != c["id"]]
    return out


def concepts_path(video_id: str) -> Path:
    return config.DATA_DIR / video_id / "concepts.json"


def has_cached(video_id: str) -> bool:
    return concepts_path(video_id).exists()


def load(video_id: str) -> list[dict]:
    p = concepts_path(video_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def _fallback_from_chunks(chunks: list[Chunk]) -> list[dict]:
    """Last-resort concept map when the LLM extraction yields nothing usable.
    Sample evenly-spaced chunks and derive a coarse concept from each so the
    map is never empty — the viva/mastery layers need a non-empty, anchored set.
    Marked auto-derived so the UI can distinguish them from LLM concepts.
    """
    if not chunks:
        return []
    n = max(config.CONCEPT_MIN, min(config.CONCEPT_MAX, max(1, len(chunks) // 4)))
    step = max(1, len(chunks) // n)
    out: list[dict] = []
    seen: set[str] = set()
    for i in range(0, len(chunks), step):
        c = chunks[i]
        name = " ".join(c.text.split()[:5]).strip().rstrip(".,;:")
        if not name:
            continue
        cid = _slug(name)
        if cid in seen:
            continue
        seen.add(cid)
        out.append({
            "id": cid,
            "name": name[:60],
            "one_line_desc": _truncate(c.text, 14),
            "first_timestamp": c.start,
            "chunk_ids": [c.idx],
            "prerequisites": [],
            "auto_derived": True,
        })
        if len(out) >= config.CONCEPT_MAX:
            break
    return out


# Map-reduce window size. sarvam-m is a reasoning model: on a single shot over
# a long transcript it spends its whole 2048-token output budget "thinking" and
# never emits the JSON (the V4/V7 silent-zero). Windows this size reliably leave
# room for the JSON after the <think> block — V2 (31 chunks) extracts cleanly.
_WINDOW_CHUNKS = 30


def _extract_once(chunks: list[Chunk]) -> list[dict]:
    """One extraction attempt over `chunks` (a single shot + stricter retry).
    Returns parsed concepts (possibly empty). Never raises — LLM/transport
    errors are swallowed and reported as an empty result for the caller to
    handle (windowed merge tolerates empty windows; the top level falls back).
    """
    import logging
    log = logging.getLogger("tutor.concepts")
    user_prompt = _build_user_prompt(chunks)
    try:
        raw = llm.get_llm().complete(_SYSTEM, user_prompt, max_tokens=2048, temperature=0.0)
        concepts = _parse(raw, chunks)
    except Exception as e:  # noqa: BLE001 — any LLM/transport error → retry then empty
        log.warning("concept extraction call failed: %s", e)
        concepts = []

    if not concepts:
        strict_system = _SYSTEM + (
            "\n\nIMPORTANT: Do NOT include any reasoning, <think> blocks, or prose. "
            "Output ONLY the JSON object, starting with '{' and ending with '}'. "
            "Keep it COMPACT so it is not truncated: short one_line_desc, fewer concepts."
        )
        try:
            raw = llm.get_llm().complete(strict_system, user_prompt, max_tokens=2048, temperature=0.0)
            concepts = _parse(raw, chunks)
        except Exception as e:  # noqa: BLE001
            log.warning("concept extraction retry failed: %s", e)
            concepts = []
    return concepts


def _merge_concepts(groups: list[list[dict]]) -> list[dict]:
    """Merge per-window concept lists: dedupe by id (union chunk_ids, keep first
    name/desc), sort by first appearance, cap at CONCEPT_MAX. Re-resolve
    prerequisites against the surviving set."""
    by_id: dict[str, dict] = {}
    for group in groups:
        for c in group:
            cid = c["id"]
            if cid not in by_id:
                by_id[cid] = {**c, "chunk_ids": list(c.get("chunk_ids", []))}
            else:
                merged = sorted(set(by_id[cid]["chunk_ids"]) | set(c.get("chunk_ids", [])))
                by_id[cid]["chunk_ids"] = merged
    out = sorted(by_id.values(), key=lambda c: c.get("first_timestamp", 0.0))[: config.CONCEPT_MAX]
    keep = {c["id"] for c in out}
    for c in out:
        c["prerequisites"] = [p for p in c.get("prerequisites", []) if p in keep and p != c["id"]]
    return out


def _extract_windowed(chunks: list[Chunk], window: int) -> list[dict]:
    """Map over fixed-size windows, reduce by merge+dedupe. Smaller windows keep
    each LLM call's JSON within sarvam-m's output budget."""
    import logging
    log = logging.getLogger("tutor.concepts")
    groups: list[list[dict]] = []
    for i in range(0, len(chunks), window):
        g = _extract_once(chunks[i:i + window])
        log.info("window %d-%d → %d concepts", i, i + min(window, len(chunks) - i), len(g))
        if g:
            groups.append(g)
    return _merge_concepts(groups) if groups else []


def extract_and_persist(video_id: str, chunks: list[Chunk]) -> list[dict]:
    """Extract a concept map and persist it. Single shot for short lectures;
    map-reduce over windows for long ones (so sarvam-m emits JSON within its
    output budget instead of exhausting it on reasoning). Never returns an empty
    map for a non-empty lecture: empty extraction falls back to chunk-derived
    concepts, logged loudly.
    """
    import logging
    log = logging.getLogger("tutor.concepts")
    if not chunks:
        return []

    if len(chunks) <= _WINDOW_CHUNKS:
        concepts = _extract_once(chunks)
        # A single shot can under-produce when sarvam-m spends its whole output
        # budget on an unclosed <think> block and never emits the JSON (seen on
        # some short lectures). Retrying with smaller windows reliably recovers,
        # because each window leaves room for JSON after the reasoning.
        if len(concepts) < config.CONCEPT_MIN and len(chunks) >= 4:
            windowed = _extract_windowed(chunks, max(4, len(chunks) // 2 + 1))
            if len(windowed) > len(concepts):
                log.info("single-shot gave %d; windowing recovered %d",
                         len(concepts), len(windowed))
                concepts = windowed
    else:
        concepts = _extract_windowed(chunks, _WINDOW_CHUNKS)

    if not concepts:
        log.error("concept extraction empty for %s — using chunk-derived fallback", video_id)
        concepts = _fallback_from_chunks(chunks)

    p = concepts_path(video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(concepts, ensure_ascii=False, indent=2))
    return concepts


# Rank weights for attribution: the top retrieved hit dominates; lower hits
# contribute progressively less. Length covers RETRIEVAL_K; beyond it → 0.
_RANK_WEIGHTS = [1.0, 0.6, 0.3, 0.15, 0.05]

# Blend of the two attribution signals. Semantic (query↔concept-text) leads,
# because a learner asking about concept A names A directly; chunk-overlap is
# the supporting signal that catches paraphrased questions whose retrieved
# chunks land on a concept the wording didn't mention.
_SEM_WEIGHT = 0.7
_CHUNK_WEIGHT = 0.3
# Minimum query↔concept cosine for a concept to be eligible at all. Off-topic
# asks score every concept below this; real asks clear it on the right one(s).
# Empirically: real ~0.30–0.70, off-topic <~0.15 on the MiniLM space here.
_SEM_MIN = 0.25

# Cache concept-text embeddings per video so /ask doesn't re-embed every call.
_concept_vec_cache: dict[str, tuple[tuple[str, ...], object]] = {}


def _concept_vectors(video_id: str | None, concepts: list[dict]):
    from .indexing import embed_texts  # local import: keep module import-light
    ids = tuple(c["id"] for c in concepts)
    if video_id is not None:
        cached = _concept_vec_cache.get(video_id)
        if cached is not None and cached[0] == ids:
            return cached[1]
    texts = [f"{c.get('name', '')}. {c.get('one_line_desc', '')}".strip() for c in concepts]
    vecs = embed_texts(texts)
    if video_id is not None:
        _concept_vec_cache[video_id] = (ids, vecs)
    return vecs


def _chunk_overlap_scores(chunk_indices: list[int], concepts: list[dict]) -> list[float]:
    rank_of: dict[int, int] = {}
    for rank, ci in enumerate(chunk_indices):
        rank_of.setdefault(ci, rank)  # keep best (lowest) rank per chunk
    out: list[float] = []
    for c in concepts:
        score = 0.0
        for cid_chunk in c.get("chunk_ids", []):
            r = rank_of.get(cid_chunk)
            if r is not None and r < len(_RANK_WEIGHTS):
                score += _RANK_WEIGHTS[r]
        out.append(score)
    return out


def attribute(question: str, chunk_indices: list[int], concepts: list[dict],
              *, video_id: str | None = None, top_n: int = 2, ratio: float = 0.6) -> list[str]:
    """Attribute an interaction to the DOMINANT concept(s) only.

    Two signals, blended:
      • semantic  — cosine between the question and each concept's "name. desc".
        This is what makes "Explain <concept A>" attribute to A itself, even when
        retrieval's top chunks for that phrasing belong to neighbouring concepts.
      • chunk overlap — rank-weighted overlap between the retrieved chunk indices
        (top hit first) and each concept's chunk_ids. Catches paraphrased asks.

    Both are normalised to [0,1], blended (semantic-led), then we keep the top
    concept plus any co-dominant concept within `ratio` of the top — so a genuine
    A+B question lights up both — capped at `top_n` and gated by an absolute floor
    so a refusal/off-topic ask attributes to nothing.
    """
    if not concepts:
        return []

    # --- semantic signal ---
    sem: list[float] = [0.0] * len(concepts)
    if question.strip():
        try:
            from .indexing import embed_query
            import numpy as np
            qv = embed_query(question)[0]
            cvecs = _concept_vectors(video_id, concepts)
            sims = (np.asarray(cvecs) @ qv)  # both normalised → cosine
            sem = [max(0.0, float(s)) for s in sims]
        except Exception:  # noqa: BLE001 — embedding must never break /ask
            sem = [0.0] * len(concepts)

    # --- chunk-overlap signal (normalised to its own max) ---
    chunk_raw = _chunk_overlap_scores(chunk_indices, concepts)
    cmax = max(chunk_raw) if chunk_raw else 0.0
    chunk_norm = [(s / cmax if cmax > 0 else 0.0) for s in chunk_raw]

    # Semantic ELIGIBILITY gate: chunk-overlap always normalises to 1.0 for the
    # best-overlapping concept, even when retrieval returned topical-but-wrong
    # chunks (e.g. an off-topic question whose nearest chunks still belong to
    # *some* concept). The query↔concept cosine, by contrast, stays low (<~0.15)
    # for genuinely off-topic asks and is 0.3–0.7 for real ones. So a concept is
    # only attributable if the question is semantically about it.
    eligible = [i for i, s in enumerate(sem) if s >= _SEM_MIN]
    if not eligible:
        return []

    scored: list[tuple[float, str]] = []
    for i in eligible:
        combined = _SEM_WEIGHT * sem[i] + _CHUNK_WEIGHT * chunk_norm[i]
        scored.append((combined, concepts[i]["id"]))

    scored.sort(key=lambda t: (-t[0], t[1]))
    best = scored[0][0]
    return [cid for score, cid in scored[:top_n] if score >= ratio * best]
