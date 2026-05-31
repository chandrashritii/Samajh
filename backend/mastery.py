"""Per-session, per-concept mastery store.

State machine:
    unseen → engaged       (a question retrieved/touched this concept)
    engaged → shaky        (misconception detected on this concept)
    shaky → shaky          (subsequent engagements remain shaky until viva)
    {engaged|shaky} → demonstrated   (viva pass — the public hook below)

Storage: one JSON file per session at data/sessions/{session_id}.json.
Writes are best-effort but atomic (tempfile + os.replace).

Design note: this module exposes `mark_demonstrated()` as a clean extension
point for the upcoming voice viva — that path stays simple to wire later.
"""
from __future__ import annotations
import json
import os
import re
import time
import uuid
from pathlib import Path

from . import config


_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_EVIDENCE = 20  # keep the log bounded


def _dir() -> Path:
    d = config.DATA_DIR / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(session_id: str) -> Path:
    if not _SESSION_ID.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    return _dir() / f"{session_id}.json"


def _atomic_write(path: Path, body: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, path)


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def _load(session_id: str) -> dict | None:
    p = _path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save(session: dict) -> None:
    p = _path(session["session_id"])
    session["updated_at"] = time.time()
    _atomic_write(p, json.dumps(session, ensure_ascii=False, indent=2))


def get_or_create(session_id: str | None, video_id: str) -> dict:
    """Return the session dict, creating it if absent. Caller may supply None
    to get a freshly-issued id."""
    sid = session_id or new_session_id()
    s = _load(sid)
    if s is None:
        s = {
            "session_id": sid,
            "video_id": video_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "concepts": {},  # concept_id -> {state, score, evidence: [str]}
        }
        _save(s)
        return s
    # If a session was started against a different video, scope changes:
    # keep the file but record the new video_id so the UI can render correctly.
    if s.get("video_id") != video_id:
        s["video_id"] = video_id
    return s


def _entry(s: dict, concept_id: str) -> dict:
    return s["concepts"].setdefault(
        concept_id,
        {"state": "unseen", "score": config.MASTERY_SCORE["unseen"], "evidence": []},
    )


def _append_evidence(entry: dict, line: str) -> None:
    log = entry.setdefault("evidence", [])
    log.append(line)
    if len(log) > _MAX_EVIDENCE:
        del log[: len(log) - _MAX_EVIDENCE]


def record_engagement(session_id: str, video_id: str,
                      concept_ids: list[str], question: str) -> None:
    """A normal question that retrieved these concepts. Push unseen→engaged."""
    if not concept_ids:
        return
    s = get_or_create(session_id, video_id)
    for cid in concept_ids:
        e = _entry(s, cid)
        if e["state"] == "unseen":
            e["state"] = "engaged"
            e["score"] = config.MASTERY_SCORE["engaged"]
        elif e["state"] == "engaged":
            # bump score on repeat engagement, capped
            e["score"] = min(0.6, e["score"] + 0.05)
        # shaky and demonstrated unchanged by a normal ask
        _append_evidence(e, f"asked: {question[:80]}")
    _save(s)


def record_misconception(session_id: str, video_id: str,
                         concept_ids: list[str], misconception: str) -> None:
    """A misconception was detected. Mark shaky on the touched concepts."""
    if not concept_ids:
        return
    s = get_or_create(session_id, video_id)
    for cid in concept_ids:
        e = _entry(s, cid)
        e["state"] = "shaky"
        e["score"] = config.MASTERY_SCORE["shaky"]
        _append_evidence(e, f"misconception: {misconception[:80]}")
    _save(s)


def mark_demonstrated(session_id: str, video_id: str, concept_id: str,
                      evidence: str = "viva-pass") -> None:
    """Public hook for the future viva loop — flips a concept to demonstrated."""
    s = get_or_create(session_id, video_id)
    e = _entry(s, concept_id)
    e["state"] = "demonstrated"
    e["score"] = config.MASTERY_SCORE["demonstrated"]
    _append_evidence(e, evidence)
    _save(s)


def list_mastery(session_id: str, concepts: list[dict]) -> list[dict]:
    """Project the session's per-concept state onto the lecture's concept list.
    Concepts the session has never touched are returned as `unseen`.
    """
    s = _load(session_id)
    cs = (s or {}).get("concepts", {})
    out: list[dict] = []
    for c in concepts:
        cid = c["id"]
        entry = cs.get(cid, {
            "state": "unseen", "score": config.MASTERY_SCORE["unseen"], "evidence": [],
        })
        out.append({
            "concept_id": cid,
            "name": c["name"],
            "state": entry["state"],
            "score": float(entry["score"]),
            "evidence": list(entry.get("evidence", [])),
        })
    return out
