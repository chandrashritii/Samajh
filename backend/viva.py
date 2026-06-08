"""Adaptive viva + teach-back.

Orchestration only — every question and every evaluation is grounded in the
lecture's own chunks via the same discipline as /ask. A viva question can never
exceed the lecture; an evaluation never grades on outside knowledge (a
real-world-correct but lecture-unsupported answer is flagged `unsupported`).

State: one JSON file per viva at data/viva/{viva_id}.json.
Question bank: the concept map. Targeting: low / unseen / shaky concepts first.
"""
from __future__ import annotations
import json
import os
import re
import time
import uuid

from . import concept_map, config, indexing, llm, mastery
from .indexing import Chunk

_VIVA_DIR = config.DATA_DIR / "viva"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# How many concepts to drill before the viva auto-completes.
DEFAULT_MAX_QUESTIONS = 5
# Mastery-state priority for adaptive targeting (lower index = ask sooner).
_PRIORITY = {"shaky": 0, "unseen": 1, "engaged": 2, "demonstrated": 9}


# ---------- store -----------------------------------------------------------

def _dir():
    _VIVA_DIR.mkdir(parents=True, exist_ok=True)
    return _VIVA_DIR


def _path(viva_id: str):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", viva_id)
    return _dir() / f"{safe}.json"


def _save(v: dict) -> None:
    v["updated_at"] = time.time()
    p = _path(v["viva_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2))
    os.replace(tmp, p)


def load(viva_id: str) -> dict | None:
    p = _path(viva_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


# ---------- JSON-from-LLM helper (sarvam-m emits <think>) --------------------

def _parse_json(raw: str) -> dict | None:
    s = _THINK.sub("", raw)
    s = _FENCE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


# ---------- grounding context ----------------------------------------------

def _chunks_for_concept(chunks: list[Chunk], concept: dict) -> list[Chunk]:
    """The concept's own chunks are its grounding context — that's exactly the
    span where the lecture teaches it, so questions/eval can't drift off-topic."""
    by_idx = {c.idx: c for c in chunks}
    out = [by_idx[i] for i in concept.get("chunk_ids", []) if i in by_idx]
    return out[:6]  # keep the prompt tight


def _fmt_excerpts(ctx: list[Chunk]) -> str:
    lines = ["LECTURE EXCERPTS:"]
    for n, c in enumerate(ctx, start=1):
        lines.append(f"[{n}] {c.text}")
    return "\n".join(lines)


def _valid_cites(raw, n: int) -> list[int]:
    out = []
    for c in (raw or []):
        try:
            ci = int(c)
            if 1 <= ci <= n:
                out.append(ci)
        except (TypeError, ValueError):
            continue
    return out


# ---------- question generation --------------------------------------------

_Q_SYSTEM = (
    "You are a tutor running an oral viva. Write EXACTLY ONE short question about "
    "the named concept, answerable using ONLY the provided lecture excerpts. The "
    "question must stay within what the excerpts actually cover — never ask about "
    "anything not in them. Keep it to one sentence, conversational (it will be "
    "read aloud).\n"
    "Output a single JSON object: {\"question\": string}. Nothing else."
)

_TB_SYSTEM = (
    "You are a tutor running a teach-back. Write EXACTLY ONE short prompt asking "
    "the student to explain the named concept in their own words, scoped to what "
    "the provided excerpts cover. One sentence, conversational (read aloud).\n"
    "Output a single JSON object: {\"question\": string}. Nothing else."
)


def _generate_question(concept: dict, ctx: list[Chunk], mode: str) -> str:
    system = _TB_SYSTEM if mode == "teach_back" else _Q_SYSTEM
    user = (
        f"CONCEPT: {concept.get('name')}\n"
        f"{_fmt_excerpts(ctx)}\n\n"
        'Output JSON: {"question": "..."}'
    )
    raw = llm.get_llm().complete(system, user, max_tokens=800, temperature=0.2)
    parsed = _parse_json(raw) or {}
    q = str(parsed.get("question", "")).strip()
    if not q:
        # Deterministic grounded fallback so the viva never stalls.
        if mode == "teach_back":
            q = f"In your own words, explain {concept.get('name')} as covered in this lecture."
        else:
            q = f"What does this lecture say about {concept.get('name')}?"
    return q


# ---------- evaluation ------------------------------------------------------

_EVAL_SYSTEM = (
    "You grade a student's spoken answer about a concept STRICTLY against the "
    "provided lecture excerpts — never outside knowledge.\n"
    "Verdicts:\n"
    "- correct: the answer matches what the excerpts say.\n"
    "- partial: partially right but missing or muddling something the excerpts cover.\n"
    "- incorrect: contradicts the excerpts or misses the point.\n"
    "- unsupported: the answer may be true in general but is NOT supported by these "
    "excerpts — say so explicitly in the rationale.\n"
    "Rules: cite the excerpt numbers your judgement rests on. For partial/incorrect, "
    "the rationale must name the specific missing or wrong piece, grounded in the "
    "excerpts. Rationale <= 30 words, neutral and encouraging.\n"
    'Output one JSON object: {"verdict": "...", "rationale": "...", "citations": [int]}. '
    "Nothing else."
)

_TB_EVAL_SYSTEM = (
    "You evaluate a student's teach-back explanation of a concept STRICTLY against "
    "the provided lecture excerpts — never outside knowledge.\n"
    "Name what they got RIGHT and what was MISSING or WRONG versus the excerpts.\n"
    "Verdicts: correct (covered the key points), partial (right but incomplete), "
    "incorrect (wrong/off), unsupported (claims not backed by these excerpts).\n"
    "Rules: cite excerpt numbers. Rationale <= 35 words, name the missing/wrong piece. "
    "Encouraging tone.\n"
    'Output one JSON object: {"verdict": "...", "rationale": "...", "citations": [int]}. '
    "Nothing else."
)


def _evaluate(question: str, transcript: str, ctx: list[Chunk], mode: str) -> dict:
    system = _TB_EVAL_SYSTEM if mode == "teach_back" else _EVAL_SYSTEM
    user = (
        f"{_fmt_excerpts(ctx)}\n\n"
        f"VIVA QUESTION: {question}\n"
        f"STUDENT ANSWER (transcribed speech, may have minor STT errors): {transcript}\n\n"
        'Output JSON: {"verdict": "...", "rationale": "...", "citations": [int]}'
    )
    raw = llm.get_llm().complete(system, user, max_tokens=900, temperature=0.0)
    parsed = _parse_json(raw) or {}
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("correct", "partial", "incorrect", "unsupported"):
        verdict = "incorrect"
    rationale = str(parsed.get("rationale", "")).strip()
    cites = _valid_cites(parsed.get("citations"), len(ctx))
    # Discipline mirror of /ask: a correct/partial verdict with no citation isn't
    # grounded — demote it so we never "pass" an answer we can't anchor.
    if verdict in ("correct", "partial") and not cites:
        verdict = "unsupported"
        if not rationale:
            rationale = "Could not anchor this answer to the lecture excerpts."
    return {"verdict": verdict, "rationale": rationale, "citations": cites}


def _reexplain(concept: dict, ctx: list[Chunk]) -> str:
    """A concise grounded re-explanation for a wrong answer — strictly from the
    concept's excerpts."""
    system = (
        "Explain the named concept in 2-3 sentences using ONLY the provided "
        "excerpts. No outside knowledge. Plain prose for reading aloud."
    )
    user = f"CONCEPT: {concept.get('name')}\n{_fmt_excerpts(ctx)}"
    raw = llm.get_llm().complete(system, user, max_tokens=800, temperature=0.0)
    txt = _THINK.sub("", raw).strip()
    txt = _FENCE.sub("", txt).strip()
    return txt[:600]


# ---------- adaptive concept selection -------------------------------------

def _pick_concept(session_id: str, video_id: str, concepts: list[dict],
                  asked: list[str]) -> dict | None:
    """Lowest-mastery concept not yet asked this viva. Priority: shaky → unseen →
    engaged; demonstrated is skipped. Ties broken by ascending mastery score."""
    rows = {r["concept_id"]: r for r in mastery.list_mastery(session_id, concepts)}
    candidates = []
    for c in concepts:
        if c["id"] in asked:
            continue
        row = rows.get(c["id"], {"state": "unseen", "score": 0.0})
        if row["state"] == "demonstrated":
            continue
        candidates.append((_PRIORITY.get(row["state"], 5), float(row.get("score", 0.0)), c))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


# ---------- public API ------------------------------------------------------

def start(session_id: str, video_id: str, mode: str, max_questions: int | None = None) -> dict:
    """Create a viva and produce the first grounded question. Returns the viva
    dict with `current` populated (or done=True if there's nothing to ask)."""
    _, chunks, _meta = indexing.load(video_id)
    concepts = concept_map.load(video_id)
    mastery.get_or_create(session_id, video_id)

    total = min(max_questions or DEFAULT_MAX_QUESTIONS, len(concepts)) if concepts else 0
    viva = {
        "viva_id": uuid.uuid4().hex[:16],
        "session_id": session_id,
        "video_id": video_id,
        "mode": mode,
        "asked": [],
        "n_asked": 0,
        "max_questions": total,
        "current": None,
        "history": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    concept = _pick_concept(session_id, video_id, concepts, []) if concepts else None
    if concept is None or total == 0:
        viva["done"] = True
        _save(viva)
        return viva

    ctx = _chunks_for_concept(chunks, concept)
    question = _generate_question(concept, ctx, mode)
    viva["current"] = {
        "concept_id": concept["id"], "concept_name": concept["name"],
        "question": question, "chunk_ids": [c.idx for c in ctx],
    }
    viva["done"] = False
    _save(viva)
    return viva


def submit(viva_id: str, transcript: str) -> dict | None:
    """Evaluate the spoken answer to the current question, update mastery, and
    advance to the next concept. Returns a result dict (or None if viva absent).
    """
    viva = load(viva_id)
    if viva is None:
        return None
    cur = viva.get("current")
    if not cur:
        return {"_done": True, "viva": viva}

    _, chunks, _meta = indexing.load(viva["video_id"])
    concepts = concept_map.load(viva["video_id"])
    concept = next((c for c in concepts if c["id"] == cur["concept_id"]), None)
    by_idx = {c.idx: c for c in chunks}
    ctx = [by_idx[i] for i in cur.get("chunk_ids", []) if i in by_idx]

    # Guard: an empty / too-short transcript (mic opened but nothing said) must
    # NOT reach the grader — the LLM tends to hallucinate "correct" on empty
    # input. Treat it as no-answer-given (incorrect) deterministically.
    if len((transcript or "").strip()) < 3:
        ev = {
            "verdict": "incorrect",
            "rationale": "I didn't catch an answer — try saying it aloud.",
            "citations": [],
        }
    else:
        ev = _evaluate(cur["question"], transcript, ctx, viva["mode"])

    # Citations payload (start/end/snippet) from the concept's own chunks.
    citations = []
    for ci in ev["citations"]:
        c = ctx[ci - 1]
        snip = c.text if len(c.text) <= 220 else c.text[:217] + "…"
        citations.append({"start": c.start, "end": c.end, "snippet": snip})

    # Mastery transition.
    sid, vid, cid = viva["session_id"], viva["video_id"], cur["concept_id"]
    before = {r["concept_id"]: r["state"] for r in mastery.list_mastery(sid, concepts)}
    from_state = before.get(cid, "unseen")
    reexplanation = None
    if ev["verdict"] in ("correct", "partial"):
        mastery.mark_demonstrated(sid, vid, cid, evidence=f"viva:{ev['verdict']}")
        to_state = "demonstrated"
    else:
        # incorrect / unsupported → shaky, with a grounded re-explanation offered.
        mastery.record_misconception(sid, vid, [cid], f"viva miss: {ev['rationale'][:60]}")
        to_state = "shaky"
        if concept is not None and ctx:
            reexplanation = _reexplain(concept, ctx)

    viva["history"].append({
        "concept_id": cid, "question": cur["question"], "transcript": transcript,
        "verdict": ev["verdict"], "rationale": ev["rationale"],
    })
    viva["asked"].append(cid)
    viva["n_asked"] += 1

    # Next concept.
    nxt = None
    if viva["n_asked"] < viva["max_questions"]:
        nxt = _pick_concept(sid, vid, concepts, viva["asked"])
    if nxt is None:
        viva["current"] = None
        viva["done"] = True
    else:
        nctx = _chunks_for_concept(chunks, nxt)
        nq = _generate_question(nxt, nctx, viva["mode"])
        viva["current"] = {
            "concept_id": nxt["id"], "concept_name": nxt["name"],
            "question": nq, "chunk_ids": [c.idx for c in nctx],
        }
        viva["done"] = False
    _save(viva)

    return {
        "viva": viva,
        "verdict": ev["verdict"],
        "rationale": ev["rationale"],
        "citations": citations,
        "mastery_update": {"concept_id": cid, "from_state": from_state, "to_state": to_state},
        "reexplanation": reexplanation,
    }


def summary(viva_id: str) -> dict | None:
    viva = load(viva_id)
    if viva is None:
        return None
    concepts = concept_map.load(viva["video_id"])
    rows = mastery.list_mastery(viva["session_id"], concepts)
    solid = [r for r in rows if r["state"] == "demonstrated"]
    shaky = [r for r in rows if r["state"] == "shaky"]
    remaining = [r for r in rows if r["state"] in ("unseen", "engaged")]
    bits = []
    if solid:
        bits.append("solid on " + ", ".join(r["name"] for r in solid[:4]))
    if shaky:
        bits.append("shaky on " + ", ".join(r["name"] for r in shaky[:4]))
    headline = "; ".join(bits) if bits else "No concepts assessed yet."
    return {"viva": viva, "solid": solid, "shaky": shaky,
            "remaining": remaining, "headline": headline}
