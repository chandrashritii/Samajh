# Sarvam Video Tutor — P0 + P1-A

A grounded YouTube-lecture Q&A agent. You give it a lecture URL; it builds a transcript index and answers your doubts **strictly from that lecture** with timestamp citations. If the answer isn't in the lecture, it refuses rather than hallucinate. Vernacular (Hindi / Tamil) answering keeps technical terms in English.

**P0 spine (built & verified):** ingestion + grounding index + grounded English answer with citation/refusal + Hindi/Tamil code-switched answers with technical-term preservation + two-key Sarvam auth fallback.

**P1-A "Comprehension Intelligence Layer" (built & verified):** on-the-fly concept map + per-session mastery tracking + grounded misconception detection + English-density register dial.

**P1-B "Voice loop + Adaptive viva" (built & verified):** speak a doubt → hear the grounded answer back (Saaras STT → existing `/ask` → Bulbul TTS); adaptive spoken viva + teach-back that targets low-mastery concepts, evaluates spoken answers strictly against the lecture (correct/partial/incorrect/**unsupported**), and drives the mastery state machine.

Talking-head avatar and aggregate analytics are intentionally out of scope, but the synthesized audio is a retrievable file (`GET /audio/{id}`) so an avatar layer can lip-sync to it unchanged.

---

## What's built

**P0**
- **F1 — Ingestion + grounding index.** YouTube URL → captions (via `youtube-transcript-api`) → ~45s chunks with timestamps → MiniLM embeddings → FAISS index, persisted to `data/{video_id}/`. STT fallback (yt-dlp + Sarvam Saaras v3) when captions are missing.
- **F2 — Grounded English answering + refusal.** Top-k retrieval → strict citation-only prompt to Sarvam `sarvam-m` → structured JSON output → belt-and-suspenders refusal (low retrieval similarity *or* `grounded=false`).
- **F3 — Vernacular layer.** Same English-first answer → Sarvam Translate (`mayura:v1`, `mode=code-mixed`) → keeps English technical terms in English while the surrounding prose flows in Hindi / Tamil.

**P1-A — Comprehension Intelligence Layer**
- **F4 — Concept map + mastery.** A single Sarvam chat call at ingest extracts 5–15 lecture concepts with `chunk_ids`, persisted to `data/{video_id}/concepts.json`. A session mastery store at `data/sessions/{session_id}.json` tracks `unseen → engaged → shaky → demonstrated` per concept. Attribution is cheap: shared chunk indices between retrieval hits and concept maps. A public hook (`mark_demonstrated`) is ready for the upcoming viva to flip concepts to demonstrated.
- **F3 (misconception) — Grounded misconception detection.** The /ask path now does ONE combined LLM call that returns answer + grounded + citations + `{misconception_detected, misconception, correction, correction_citations}`. The flag is dropped unless the correction is itself grounded in the retrieved excerpts — no fabricated corrections, refusal discipline preserved. Calibrated to avoid false positives on normal well-posed questions.
- **F1+ — Register dial.** Optional `register: more_vernacular | balanced | more_english` on /ask. Maps to mayura mode (classic-colloquial / code-mixed / code-mixed-with-sentence-retention). Technical terms always stay English in all three settings.

---

## Architecture

```
YouTube URL
   │
   ▼
[ingestion]  captions → segments  (fallback: yt-dlp audio → Saaras)
   │
   ▼
[indexing]   chunk ~45s + 1-segment overlap → MiniLM → FAISS  →  data/{vid}/
   │
   ▼
question + language
   │
   ▼
[retrieval]  top-5 cosine  →  refusal if max sim < 0.25
   │
   ▼
[answering]  Sarvam Chat with ONLY retrieved chunks → JSON {answer, grounded, citations}
   │                                                  refusal if grounded=false
   ▼
[translation] if lang != en:  Sarvam Translate, mode=code-mixed (terms stay EN)
   │
   ▼
response  { answer, language, grounded, citations:[{start,end,snippet}] }
```

Module map (one module = one concern, all thin):

| File | Responsibility |
| --- | --- |
| `backend/ingestion.py` | URL → video_id, captions, STT fallback orchestration |
| `backend/transcription.py` | yt-dlp + ffmpeg chunking + Saaras STT loop |
| `backend/indexing.py` | Chunking, embeddings, FAISS persist/load |
| `backend/retrieval.py` | Top-k cosine over the persisted index |
| `backend/concept_map.py` | One-call lecture-scoped concept extraction + attribution |
| `backend/mastery.py` | Per-session, per-concept state machine + viva-pass hook |
| `backend/answering.py` | Citation prompt, JSON parsing, refusal + misconception logic |
| `backend/translation.py` | Sarvam Translate with register dial + term preservation |
| `backend/sarvam_client.py` | Thin HTTP wrapper (STT / Chat / Translate); two-key fallback |
| `backend/llm.py` | Swappable LLM interface (Sarvam default, Gemini optional) |
| `backend/main.py` | FastAPI routes + static frontend |

---

## Setup

```bash
# 1. Python env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env
# Edit .env and put your SARVAM_API_KEY

# 3. (Optional, only for caption-less videos) install ffmpeg
brew install ffmpeg     # macOS
# sudo apt install ffmpeg  # Linux
```

`sentence-transformers` will download the MiniLM model (~80 MB) on first ingest.

---

## Run

```bash
# Start the API + serve the minimal frontend
uvicorn backend.main:app --reload --port 8000

# Open http://localhost:8000
```

UI flow:
1. Paste a YouTube URL → **Ingest**.
2. Type a doubt, pick a language, **Ask**.
3. Citations are clickable — they jump to the exact moment in the YouTube video.

---

## Verify (without the UI)

P0 spine only:
```bash
python scripts/test_p0.py
```

P0 regression + P1-A (concept map, misconception, register dial):
```bash
python scripts/test_p1a.py
```

Both default to Andrew Ng's CS229 2018 Lecture 1 (clean English captions). Acceptance probes covered by `test_p1a.py`:

- **N1** — concept map for the lecture (≥5 concepts, anchored to chunk indices)
- **P0a** — "What is gradient descent?" → grounded + cited + concept attribution
- **P0b** — "What is the capital of France?" → refusal, no hallucination, no false-positive misconception
- **F3a** — "Supervised learning doesn't need labeled data, right?" → misconception flagged + grounded correction
- **F3b** — "Quantum entanglement makes neural networks faster, doesn't it?" → no flag (lecture can't ground a correction)
- **F4** — mastery snapshot shows concepts moved off `unseen` based on attribution
- **F1+** — register dial: same Hindi question at `more_vernacular`, `balanced`, `more_english` produces three visibly different blends; tech terms preserved in all three

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | liveness |
| `POST` | `/ingest` | `{url}` → indexes a lecture (transcript + FAISS + concept map), returns metadata |
| `GET`  | `/lecture/{video_id}` | metadata for an already-indexed lecture |
| `GET`  | `/lecture/{video_id}/concepts` | the concept map for that lecture |
| `POST` | `/ask` | `{video_id, question, language?, register?, session_id?}` → grounded answer + citations + misconception block + concepts touched |
| `GET`  | `/session/{session_id}/mastery` | per-concept mastery for the session |
| `POST` | `/ask_voice` | multipart `audio` + `{video_id, session_id?, language?, register?}` → Saaras STT → grounded `/ask` → Bulbul TTS. Returns `{transcript, answer, citations, grounded, top_sim, misconception, concepts_touched, audio}` |
| `GET`  | `/audio/{audio_id}` | the synthesized WAV for a spoken answer/question (retrievable file → future avatar can lip-sync to it) |
| `POST` | `/viva/start` | `{session_id, video_id, mode:"quiz"\|"teach_back", register?, language?}` → first grounded question targeting a low-mastery concept + spoken audio |
| `POST` | `/viva/answer` | multipart `audio` + `{viva_id, language?, register?}` → `{transcript, verdict, rationale, citations, mastery_update, reexplanation?, next_question?, audio, done}` |
| `GET`  | `/viva/{viva_id}/summary` | end-of-session read: `solid` / `shaky` / `remaining` concepts + headline |

Response shape for `/ask`:
```json
{
  "answer": "Supervised learning requires labeled data...",
  "language": "en",
  "grounded": true,
  "citations": [
    {"start": 2421.0, "end": 2465.3, "snippet": "...during training you are given inputs X together with..."}
  ],
  "session_id": "ada531b4b5cc46de",
  "concepts_touched": ["supervised-learning"],
  "misconception": {
    "detected": true,
    "misconception": "Supervised learning doesn't need labeled data",
    "correction": "Supervised learning requires labeled data (X and Y) during training to map inputs to outputs"
  }
}
```

`register` accepts `more_vernacular | balanced | more_english` (default `balanced`).

---

## Design choices & where they're made

- **Strict grounding > polish.** Two independent refusal gates — retrieval-similarity floor (`GROUNDING_SIM_THRESHOLD=0.25`) before we even call the LLM, then `grounded=false` from the LLM itself. Either fires → fixed refusal string. The LLM never sees the question without retrieved excerpts; the system prompt forbids outside knowledge. See `backend/answering.py`.
- **English-first, then translate.** Grounding fidelity is highest in English (better retrieval, better LLM behavior, easier to debug). Vernacular is a translation pass on top. Citations and `grounded` are language-independent, returned unchanged. See `backend/translation.py`.
- **Technical-term preservation via code-mixed translation.** Sarvam's Mayura `mode=code-mixed` is built for exactly this — Indic prose with English technical/domain terms preserved verbatim. Cleaner than a custom NER + dictionary patching layer, and stays inside the Sarvam stack.
- **Captions first.** Free, fast, accurate timestamps. STT fallback (Saaras v3 via yt-dlp + ffmpeg chunking, since the REST endpoint is capped at ~30 s) is implemented but secondary — most public educational lectures have captions.
- **Thin Sarvam wrapper, swappable LLM.** `sarvam_client.py` is one function per endpoint, no SDK. `llm.py` is a Protocol with a Sarvam default and a Gemini drop-in for credit-conscious testing — flip via `LLM_PROVIDER=gemini`.
- **FAISS over Chroma.** Single lecture per index, no metadata querying needed, persistence is two files. Simpler.
- **Cache by video_id.** Re-ingest is a `data/` read with no embedding work.

---

## Known limits / what to harden later

- **Saaras REST is <30 s per call.** Long caption-less lectures get chunked into 25 s windows; for very long videos, switch to Sarvam's batch STT endpoint.
- **Mayura's 1000-char limit** is handled by sentence-split chunking in `translation.py`.
- **Refusal threshold (0.25)** is empirical — tune against your specific lecture's chunk style if needed.
- **Single-lecture in-memory FAISS** is fine here; a multi-lecture deployment would want a real vector DB.

---

## Extension points (next states → not built yet, designed for)

- **Voice loop.** `answering.py` returns clean text + structured citations — a TTS layer (Sarvam Bulbul v3) can consume the answer text unchanged. STT-in feeds `/ask` as the `question` field; `sarvam_client.stt_transcribe` is ready.
- **Adaptive viva.** The concept map IS the question bank. `mastery.mark_demonstrated()` is the public hook that flips a concept to `demonstrated` after a successful viva turn. The viva picks low-mastery concepts.
- **Doubt vs Socratic.** Add a `mode` param to `/ask`; swap the system prompt in `answering.py`. Retrieval + refusal + misconception layers don't change.
- **Re-explain ladder.** Same answer, different system prompts ("simpler", "with an analogy") — no new infra; the misconception+answer composition is already separable.
- **Aggregate analytics.** Mastery is per-session today; promote `data/sessions/` to a SQLite/DB-backed store keyed by (learner_id, lecture_id) and aggregate.

---

## Approach & challenges (notes from building this)

- **The headline feature is the refusal, not the answer.** I spent most of the P0 design time on the refusal gates because a confidently-wrong answer is worse than no answer. Two independent checks (similarity + LLM self-report) catch different failure modes — similarity catches "we retrieved garbage", `grounded=false` catches "we retrieved adjacent material but it doesn't actually answer the question". In P1-A the misconception path inherits the same discipline: a flagged misconception must come with a correction grounded in the retrieved excerpts, otherwise the flag is silently dropped.
- **Sarvam's docs evolve fast.** I verified the endpoint URLs, model strings (`saaras:v3`, `sarvam-m`, `mayura:v1`) and auth header (`api-subscription-key`) against current `docs.sarvam.ai` before writing the client. The single auth header works across all three endpoints; the two-key fallback transparently retries on 401/403/429.
- **Technical-term preservation in vernacular.** Two options considered: (a) custom NER → wrap terms in markers → translate → unwrap, or (b) use Mayura's `code-mixed` mode. (b) is what the model is trained for; the prose flows better and there's nothing to maintain. Picked (b). For the register dial, `more_english` adds a sentence-level English-retention pass on top.
- **English-first answering, then translate.** Generating directly in Hindi via the chat model would mean less reliable grounding (the model's tendency to "complete" an answer from training-data knowledge gets stronger when generating Indic text). Translating a strictly-grounded English answer keeps the strictness intact. The same is true for the misconception block — composed in English, then translated.
- **One LLM call for concept extraction.** `sarvam-m` has a 7192-token context window. To fit a 75-minute lecture (113 chunks) plus a 1500-token output budget, I send a tight `#idx mm:ss preview` format with first-12-words previews. Tested at ~3.3K input tokens, leaves ample headroom. For substantially longer lectures the right move is a map-reduce pass; the extraction module is the single seam to touch.
- **One LLM call for /ask too.** The misconception assessment and the answer are produced in a single chat completion with a combined JSON schema. This keeps the per-ask cost the same as P0 and means the misconception correction shares grounding with the answer. The frontend renders the misconception as a distinct "let's clear this up first" block above the answer.
- **Mastery state machine is explicit and minimal.** `unseen → engaged → shaky → demonstrated` with documented transitions. State is filed under `data/sessions/{session_id}.json` (atomic writes, single JSON per session). Attribution uses the cheapest signal first (shared chunk indices) and exposes `mark_demonstrated()` as the future-viva hook.
- **Register dial is per-sentence, not per-call.** `more_english` is the only register that uses sentence-level logic: sentences with 2+ glossary hits stay verbatim English; the rest go through `code-mixed` translation. `more_vernacular` switches Mayura to `classic-colloquial`, which leans more purely Indic at the cost of occasionally transliterating a term — the user opted in. `balanced` is `code-mixed`.
- **STT fallback is implemented but not the demo path.** Caption-less educational lectures are rare; getting STT-with-timestamps right for hour-long videos is non-trivial (REST endpoint limit + chunk stitching). Implemented carefully but flagged as best-effort.
