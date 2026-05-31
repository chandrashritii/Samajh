"""FastAPI app. P0 spine: ingest → retrieve → ground → translate.
P1-A: + concept map + mastery + misconception + register dial.
"""
from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import (
    answering, concept_map, config, indexing, ingestion, mastery, retrieval,
    translation, voice, viva as viva_mod,
)
from .schemas import (
    AskRequest, AskResponse, AskVoiceResponse, Citation, ConceptMapResponse, Health,
    IngestRequest, LectureMeta, MasteryEntry, MasteryState, MasteryUpdate,
    MisconceptionBlock, Register, SessionMasteryResponse, VivaAnswerResponse,
    VivaMode, VivaStartRequest, VivaStartResponse, VivaSummaryResponse, VivaSummaryRow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tutor")

app = FastAPI(title="Sarvam Video Tutor", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health()


def _ensure_concepts(video_id: str, chunks) -> list[dict]:
    """Load cached concepts; if absent, extract via LLM and persist. The LLM
    call here is the only one /ingest needs beyond local embedding work.
    """
    if concept_map.has_cached(video_id):
        return concept_map.load(video_id)
    try:
        concepts = concept_map.extract_and_persist(video_id, chunks)
        log.info("extracted %d concepts for %s", len(concepts), video_id)
        return concepts
    except Exception as e:
        # Concept extraction must not block ingest — index is still usable.
        log.warning("concept extraction failed for %s: %s", video_id, e)
        # Persist an empty file so we don't keep retrying on every /ingest call.
        concept_map.concepts_path(video_id).parent.mkdir(parents=True, exist_ok=True)
        concept_map.concepts_path(video_id).write_text("[]")
        return []


@app.post("/ingest", response_model=LectureMeta)
def ingest_endpoint(req: IngestRequest) -> LectureMeta:
    try:
        video_id = ingestion.extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if indexing.has_cached_index(video_id):
        _, chunks, meta = indexing.load(video_id)
        concepts = _ensure_concepts(video_id, chunks)
        return LectureMeta(
            video_id=video_id, title=meta.get("title", video_id),
            duration=float(meta.get("duration", 0.0)),
            segments=len(chunks), concepts=len(concepts), cached=True,
            transcript_source=meta.get("transcript_source", "unknown"),
        )

    try:
        video_id, segments, source = ingestion.ingest(req.url)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e}")

    chunks = indexing.chunk_segments(segments)
    if not chunks:
        raise HTTPException(status_code=422, detail="Transcript produced no usable chunks.")

    duration = float(segments[-1]["end"]) if segments else 0.0
    meta = {"title": video_id, "duration": duration, "transcript_source": source}
    indexing.build_and_persist(video_id, chunks, meta)
    concepts = _ensure_concepts(video_id, chunks)
    log.info("ingested %s: %d segs → %d chunks (%s); %d concepts",
             video_id, len(segments), len(chunks), source, len(concepts))
    return LectureMeta(
        video_id=video_id, title=meta["title"], duration=duration,
        segments=len(chunks), concepts=len(concepts), cached=False,
        transcript_source=source,
    )


@app.get("/lecture/{video_id}", response_model=LectureMeta)
def lecture_meta(video_id: str) -> LectureMeta:
    if not indexing.has_cached_index(video_id):
        raise HTTPException(status_code=404, detail="Lecture not indexed.")
    _, chunks, meta = indexing.load(video_id)
    concepts = concept_map.load(video_id)
    return LectureMeta(
        video_id=video_id, title=meta.get("title", video_id),
        duration=float(meta.get("duration", 0.0)),
        segments=len(chunks), concepts=len(concepts), cached=True,
        transcript_source=meta.get("transcript_source", "unknown"),
    )


@app.get("/lecture/{video_id}/concepts", response_model=ConceptMapResponse)
def lecture_concepts(video_id: str) -> ConceptMapResponse:
    if not concept_map.has_cached(video_id):
        if not indexing.has_cached_index(video_id):
            raise HTTPException(status_code=404, detail="Lecture not indexed.")
        # Index exists but concepts haven't been extracted yet — do it now.
        _, chunks, _ = indexing.load(video_id)
        _ensure_concepts(video_id, chunks)
    return ConceptMapResponse(video_id=video_id, concepts=concept_map.load(video_id))


@app.get("/session/{session_id}/mastery", response_model=SessionMasteryResponse)
def session_mastery(session_id: str) -> SessionMasteryResponse:
    try:
        from .mastery import _path  # validates the id shape
        _path(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # We need the concept list to project the mastery onto. Look up the
    # session's video_id; if the session doesn't exist yet we return empty.
    s = mastery._load(session_id)
    if s is None:
        return SessionMasteryResponse(session_id=session_id, video_id="", mastery=[])
    video_id = s.get("video_id", "")
    concepts = concept_map.load(video_id) if video_id else []
    rows = mastery.list_mastery(session_id, concepts)
    return SessionMasteryResponse(
        session_id=session_id, video_id=video_id,
        mastery=[MasteryEntry(**r) for r in rows],
    )


def _run_ask(req: AskRequest) -> AskResponse:
    """Shared grounded-answer core. /ask and /ask_voice both go through here so
    the voice path is a pure adapter over existing grounding/refusal logic —
    never a parallel answer path."""
    # An empty/whitespace question embeds to a zero vector (NaN once normalized),
    # which poisons retrieval — reject cleanly instead of 500-ing downstream.
    if not (req.question or "").strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if req.language not in config.SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")
    if not indexing.has_cached_index(req.video_id):
        raise HTTPException(status_code=404, detail="Lecture not indexed. Call /ingest first.")

    index, chunks, _meta = indexing.load(req.video_id)
    hits = retrieval.top_k(index, chunks, req.question, k=config.RETRIEVAL_K)
    result = answering.answer(req.question, hits)

    # Build citations payload from the hits indices.
    citations: list[Citation] = []
    for ci in result.citation_indices:
        h = hits[ci - 1]
        snippet = h.chunk.text
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
        citations.append(Citation(start=h.chunk.start, end=h.chunk.end, snippet=snippet))

    # Concept attribution — semantic (query↔concept) blended with chunk overlap,
    # so a question about concept A attributes to A itself, not merely whichever
    # concepts happen to share A's retrieved chunks.
    concepts = concept_map.load(req.video_id)
    hit_chunk_ids = [h.chunk.idx for h in hits]
    touched = (
        concept_map.attribute(req.question, hit_chunk_ids, concepts, video_id=req.video_id)
        if concepts else []
    )

    # Mastery: ensure a session exists; record engagement / misconception.
    session = mastery.get_or_create(req.session_id, req.video_id)
    sid = session["session_id"]

    if result.grounded and touched:
        mastery.record_engagement(sid, req.video_id, touched, req.question)
    if result.misconception.detected and touched:
        mastery.record_misconception(
            sid, req.video_id, touched, result.misconception.misconception or "",
        )

    # Translation — per register, with technical-term preservation throughout.
    final_answer = translation.translate_answer(result.answer_en, req.language, req.register)
    misc_block = MisconceptionBlock(detected=result.misconception.detected)
    if result.misconception.detected:
        misc_block.misconception = translation.translate_answer(
            result.misconception.misconception or "", req.language, req.register,
        ) or None
        misc_block.correction = translation.translate_answer(
            result.misconception.correction or "", req.language, req.register,
        ) or None

    # Single source of truth: the similarity the grounding gate actually used.
    nearest_concept = touched[0] if touched else None

    return AskResponse(
        answer=final_answer, language=req.language, grounded=result.grounded,
        citations=citations, session_id=sid, concepts_touched=touched,
        misconception=misc_block,
        top_sim=result.top_sim, nearest_concept=nearest_concept,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _run_ask(req)


# ---- Voice ----------------------------------------------------------------

def _save_upload(upload: UploadFile) -> str:
    """Persist an uploaded mic clip to a temp file; return its path."""
    import os
    suffix = Path(upload.filename or "clip.webm").suffix or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


@app.get("/audio/{audio_id}")
def get_audio(audio_id: str) -> FileResponse:
    p = voice.audio_path(audio_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(p, media_type="audio/wav")


@app.post("/ask_voice", response_model=AskVoiceResponse)
def ask_voice(
    audio: UploadFile = File(...),
    video_id: str = Form(...),
    session_id: Optional[str] = Form(None),
    language: str = Form("en"),
    register: str = Form("balanced"),
) -> AskVoiceResponse:
    """Saaras (STT) → existing grounded /ask core → Bulbul (TTS). The answer
    logic is unchanged; this is purely an adapter."""
    import os
    if not indexing.has_cached_index(video_id):
        raise HTTPException(status_code=404, detail="Lecture not indexed. Call /ingest first.")
    clip_path = _save_upload(audio)
    try:
        stt = voice.transcribe(clip_path, language_code="unknown")
        transcript = (stt.get("transcript") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(clip_path)
        except OSError:
            pass

    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe the audio (empty).")

    try:
        reg = Register(register)
    except ValueError:
        reg = Register.BALANCED
    ask_req = AskRequest(
        video_id=video_id, question=transcript, language=language,
        register=reg, session_id=session_id,
    )
    base = _run_ask(ask_req)

    # Speak the misconception correction first (if any), then the answer — same
    # order the UI renders, so audio and text agree.
    spoken = base.answer
    if base.misconception and base.misconception.detected and base.misconception.correction:
        spoken = f"{base.misconception.correction} {base.answer}"
    try:
        audio_id = voice.speak_answer(spoken, language)
    except Exception as e:
        log.warning("TTS failed (returning text-only): %s", e)
        audio_id = None

    return AskVoiceResponse(**base.model_dump(), transcript=transcript, audio=audio_id)


# ---- Viva -----------------------------------------------------------------

@app.post("/viva/start", response_model=VivaStartResponse)
def viva_start(req: VivaStartRequest) -> VivaStartResponse:
    if not indexing.has_cached_index(req.video_id):
        raise HTTPException(status_code=404, detail="Lecture not indexed. Call /ingest first.")
    if not concept_map.load(req.video_id):
        raise HTTPException(status_code=422, detail="No concept map for this lecture.")
    v = viva_mod.start(req.session_id, req.video_id, req.mode.value)
    cur = v.get("current")
    if not cur:
        return VivaStartResponse(
            viva_id=v["viva_id"],
            question="No concepts left to assess — you're all caught up!",
            done=True, asked=v["n_asked"], total=v["max_questions"],
        )
    spoken_q = translation.translate_answer(cur["question"], req.language, req.register)
    audio_id = None
    try:
        audio_id = voice.speak_answer(spoken_q, req.language)
    except Exception as e:
        log.warning("viva start TTS failed: %s", e)
    return VivaStartResponse(
        viva_id=v["viva_id"], concept_id=cur["concept_id"], concept_name=cur["concept_name"],
        question=spoken_q, audio=audio_id, done=False,
        asked=v["n_asked"], total=v["max_questions"],
    )


@app.post("/viva/answer", response_model=VivaAnswerResponse)
def viva_answer(
    audio: UploadFile = File(...),
    viva_id: str = Form(...),
    language: str = Form("en"),
    register: str = Form("balanced"),
) -> VivaAnswerResponse:
    import os
    v = viva_mod.load(viva_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Viva not found.")
    try:
        reg = Register(register)
    except ValueError:
        reg = Register.BALANCED

    clip_path = _save_upload(audio)
    try:
        stt = voice.transcribe(clip_path, language_code="unknown")
        transcript = (stt.get("transcript") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(clip_path)
        except OSError:
            pass

    res = viva_mod.submit(viva_id, transcript)
    if res is None:
        raise HTTPException(status_code=404, detail="Viva not found.")
    v = res["viva"]
    cur = v.get("current")

    rationale = translation.translate_answer(res["rationale"], language, reg)
    reexpl = (translation.translate_answer(res["reexplanation"], language, reg)
              if res.get("reexplanation") else None)
    next_q = translation.translate_answer(cur["question"], language, reg) if cur else None

    # Speak: verdict rationale (+ re-explanation if wrong) (+ next question).
    speak_parts = [p for p in (rationale, reexpl, next_q) if p]
    audio_id = None
    try:
        audio_id = voice.speak_answer(" ".join(speak_parts), language)
    except Exception as e:
        log.warning("viva answer TTS failed: %s", e)

    mu = res["mastery_update"]
    return VivaAnswerResponse(
        transcript=transcript, verdict=res["verdict"], rationale=rationale,
        citations=[Citation(**c) for c in res["citations"]],
        mastery_update=MasteryUpdate(**mu),
        reexplanation=reexpl, next_question=next_q,
        next_concept_id=cur["concept_id"] if cur else None,
        audio=audio_id, done=v.get("done", False),
        asked=v["n_asked"], total=v["max_questions"],
    )


@app.get("/viva/{viva_id}/summary", response_model=VivaSummaryResponse)
def viva_summary(viva_id: str) -> VivaSummaryResponse:
    s = viva_mod.summary(viva_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Viva not found.")
    v = s["viva"]

    def to_rows(rs):
        return [VivaSummaryRow(concept_id=r["concept_id"], name=r["name"],
                               state=r["state"], score=float(r["score"])) for r in rs]

    return VivaSummaryResponse(
        viva_id=v["viva_id"], session_id=v["session_id"], video_id=v["video_id"],
        mode=VivaMode(v["mode"]), asked=v["n_asked"],
        solid=to_rows(s["solid"]), shaky=to_rows(s["shaky"]),
        remaining=to_rows(s["remaining"]), headline=s["headline"],
    )


# ---- Startup: restore seed + warm the curated library --------------------

_SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


def _library_video_ids() -> list[str]:
    """The demo library (frontend/samajh/library.json) is the single source of
    truth for which lectures to make instantly available."""
    lib_path = Path(__file__).resolve().parent.parent / "frontend" / "samajh" / "library.json"
    try:
        import json as _json
        return [e["video_id"] for e in _json.loads(lib_path.read_text())]
    except Exception:  # noqa: BLE001
        return []


def _restore_seed() -> None:
    """Copy version-controlled lecture indices from seed/ into data/ when absent,
    so the curated library is instantly available on a fresh deploy (free-tier
    disks are ephemeral) with zero Sarvam cost and zero YouTube access."""
    import shutil
    if not _SEED_DIR.exists():
        return
    for d in _SEED_DIR.iterdir():
        if not d.is_dir():
            continue
        dest = config.DATA_DIR / d.name
        if dest.exists():
            continue
        try:
            shutil.copytree(d, dest)
            log.info("restored seed lecture %s", d.name)
        except Exception as e:  # noqa: BLE001
            log.warning("seed restore failed for %s: %s", d.name, e)


def _prewarm_chips() -> None:
    """Ensure each library lecture is indexed so cards load instantly. Seed
    restore usually satisfies this for free; this re-ingests any still missing.
    Best-effort — failures are logged, never block startup. On a datacenter host
    YouTube fetches will fail; that's why seed/ is bundled."""
    for vid in _library_video_ids():
        try:
            if indexing.has_cached_index(vid):
                _, chunks, _ = indexing.load(vid)
                _ensure_concepts(vid, chunks)
                continue
            vid2, segments, source = ingestion.ingest(f"https://www.youtube.com/watch?v={vid}")
            chunks = indexing.chunk_segments(segments)
            if not chunks:
                continue
            duration = float(segments[-1]["end"]) if segments else 0.0
            indexing.build_and_persist(
                vid2, chunks, {"title": vid2, "duration": duration, "transcript_source": source})
            _ensure_concepts(vid2, chunks)
            log.info("prewarmed library lecture %s", vid2)
        except Exception as e:  # noqa: BLE001 — best-effort warmup
            log.warning("prewarm failed for %s: %s", vid, e)


@app.on_event("startup")
def _startup_prewarm() -> None:
    import os
    import threading
    # Seed restore is cheap (file copy) — do it synchronously so the library is
    # ready immediately. Live re-ingest of anything missing runs off-thread.
    _restore_seed()
    if os.getenv("PREWARM_CHIPS", "1") == "1":
        threading.Thread(target=_prewarm_chips, daemon=True).start()


# ---- Static frontend: Samajh (public) at /, test bench at /dev -------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_SAMAJH_DIR = _FRONTEND_DIR / "samajh"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        # The polished learner-facing product.
        return FileResponse(_SAMAJH_DIR / "index.html")

    @app.get("/dev")
    def dev_bench() -> FileResponse:
        # The original test bench — raw JSON, top_sim, session ids, debug panels.
        return FileResponse(_FRONTEND_DIR / "index.html")
