"""Grounded answering + (optional) misconception detection in one LLM call.

Refusal gates (P0, unchanged):
  1. If top retrieval similarity is below GROUNDING_SIM_THRESHOLD → refuse.
  2. If the LLM signals grounded=false → refuse.

Misconception gate (P1-A):
  - The LLM may flag a misconception, but only if it can produce a CORRECTION
    that itself is supported by the retrieved excerpts (i.e. cites at least
    one excerpt). Otherwise we coerce the flag to false to keep refusal
    discipline intact — no inventing corrections from outside knowledge.
  - Calibration is via the system prompt: "highly confident, only when the
    question's PREMISE is contradicted by the excerpts".
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from . import config, llm
from .retrieval import Hit


_SYSTEM = (
    "You are a tutor that answers a student's question STRICTLY using only the "
    "provided lecture excerpts. You never use outside knowledge.\n\n"
    "You produce TWO things in one JSON object:\n"
    "(A) A misconception assessment.\n"
    "(B) A grounded answer.\n\n"
    "Rules for (A):\n"
    "1. Only flag a misconception when you are HIGHLY confident the question's "
    "phrasing contains a factually wrong premise that is DIRECTLY contradicted "
    "by the excerpts. A normal, well-posed question is NOT a misconception.\n"
    "2. If you flag a misconception, the correction MUST itself be grounded in "
    "the excerpts and you MUST cite the excerpt numbers used for the correction.\n"
    "3. If the excerpts do not contain material to correct the misconception, "
    "set misconception_detected=false. Do not invent a correction.\n"
    "4. NEVER flag a misconception for a proper-noun spelling, a name, or a "
    "transcription variant (auto-captions are noisy and mis-spell names). "
    "Misconceptions are about CONCEPTS and FACTS the student got wrong, never "
    "about how a name is spelled.\n\n"
    "Rules for (B) — CLAIM-LEVEL FAITHFULNESS:\n"
    "5. Answer ONLY what the excerpts EXPLICITLY state. Set grounded=true only "
    "if the excerpts explicitly answer the SPECIFIC thing asked — not merely the "
    "surrounding topic.\n"
    "6. If the excerpts discuss the surrounding topic but do NOT actually explain "
    "the specific mechanism, definition, or fact asked for (including topics the "
    "lecturer says will be 'covered later', 'in the next video', or defers), set "
    "grounded=false and answer in this form: \"This lecture references/discusses "
    "[topic] but doesn't explain [the specific thing asked] — that's not covered "
    "here.\" Do NOT infer or reconstruct the mechanism from adjacent material.\n"
    "7. If the excerpts DO explicitly contain the answer, answer concisely (2–5 "
    "sentences) and cite the excerpt numbers used.\n"
    "8. If the excerpts are unrelated to the question, set grounded=false and "
    "answer exactly: \"That isn't covered in this lecture.\"\n"
    "9. Never invent facts, definitions, mechanisms, or examples not in the "
    "excerpts, and never complete a deferred topic from outside knowledge.\n\n"
    "Output a single JSON object and nothing else.\n"
)

_JSON_SCHEMA_HINT = (
    'Output schema:\n'
    '{\n'
    '  "misconception_detected": boolean,\n'
    '  "misconception": string|null,            // ≤25 words, what the student got wrong\n'
    '  "correction": string|null,               // ≤40 words, grounded correction\n'
    '  "correction_citations": [integer],       // excerpt numbers used for correction\n'
    '  "answer": string,\n'
    '  "grounded": boolean,\n'
    '  "citations": [integer]                   // excerpt numbers used for the answer\n'
    '}'
)


def _fmt_ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _build_user_prompt(question: str, hits: list[Hit]) -> str:
    lines = ["LECTURE EXCERPTS:"]
    for n, h in enumerate(hits, start=1):
        ts = f"[{_fmt_ts(h.chunk.start)}–{_fmt_ts(h.chunk.end)}]"
        lines.append(f"[{n}] {ts}: {h.chunk.text}")
    lines.append("")
    lines.append(f"STUDENT QUESTION: {question}")
    lines.append("")
    lines.append(_JSON_SCHEMA_HINT)
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_json_loose(text: str) -> dict | None:
    # sarvam-m is a reasoning model — strip its <think> block first so a stray
    # '{' inside the reasoning can't derail the brace-matching fallback.
    s = _THINK.sub("", text)
    s = _FENCE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


@dataclass
class Misconception:
    detected: bool = False
    misconception: str | None = None
    correction: str | None = None


@dataclass
class AnsweringResult:
    answer_en: str
    grounded: bool
    citation_indices: list[int] = field(default_factory=list)  # 1-based, into hits
    misconception: Misconception = field(default_factory=Misconception)
    # The actual top retrieval similarity used for the grounding gate. Single
    # source of truth so the value the API reports can never drift from the
    # value the refusal decision was made on — on every path (grounded,
    # refused-by-floor, refused-by-LLM, empty-hits, cached, any language).
    top_sim: float = 0.0


def _refusal(misc: Misconception | None = None, top_sim: float = 0.0) -> AnsweringResult:
    return AnsweringResult(config.REFUSAL_EN, False, [], misc or Misconception(), top_sim)


def _valid_citation_list(raw, n: int) -> list[int]:
    out: list[int] = []
    for c in (raw or []):
        try:
            ci = int(c)
            if 1 <= ci <= n:
                out.append(ci)
        except (TypeError, ValueError):
            continue
    return out


def answer(question: str, hits: list[Hit]) -> AnsweringResult:
    if not hits:
        return _refusal(top_sim=0.0)

    # The actual top retrieval similarity — computed once here and carried on
    # the result so the API reports exactly the value the gate decided on.
    top_score = float(max(h.score for h in hits))
    if top_score < config.GROUNDING_SIM_THRESHOLD:
        return _refusal(top_sim=top_score)

    user_prompt = _build_user_prompt(question, hits)
    # temperature=0.0: a grounding/refusal decision should be deterministic.
    # At 0.1 the borderline "talks-around-it" cases (e.g. backprop deferred to a
    # later video) flipped between grounded and refused across identical runs.
    # Reasoning is disabled (see sarvam_client), so the whole budget goes to the
    # JSON answer — 900 is ample for a 2–5 sentence grounded answer + citations.
    raw = llm.get_llm().complete(_SYSTEM, user_prompt, max_tokens=900, temperature=0.0)
    parsed = _parse_json_loose(raw)

    if parsed is None:
        retry = llm.get_llm().complete(
            _SYSTEM,
            user_prompt + "\n\nReturn ONLY the JSON object. No prose, no code fences.",
            max_tokens=900, temperature=0.0,
        )
        parsed = _parse_json_loose(retry)

    if parsed is None:
        return _refusal(top_sim=top_score)

    # ---- (B) Answer / grounded flag ----------------------------------------
    grounded = bool(parsed.get("grounded", False))
    answer_text = str(parsed.get("answer", "")).strip()
    citations = _valid_citation_list(parsed.get("citations"), len(hits))

    if not grounded or not answer_text:
        answer_text = config.REFUSAL_EN
        grounded = False
        citations = []

    # ---- (A) Misconception, with grounded-correction enforcement ----------
    misc = Misconception()
    if bool(parsed.get("misconception_detected", False)):
        misc_text = (parsed.get("misconception") or "").strip()
        corr_text = (parsed.get("correction") or "").strip()
        corr_cites = _valid_citation_list(parsed.get("correction_citations"), len(hits))
        # Enforce: a flagged misconception MUST have a correction grounded in
        # the retrieved excerpts. Otherwise drop the flag silently.
        if misc_text and corr_text and corr_cites:
            misc = Misconception(detected=True, misconception=misc_text, correction=corr_text)

    # If misconception_detected but the answer ended up ungrounded, the
    # composed UI message would be: "we found a misconception about X you
    # can't actually verify here". Safer to drop the flag in that case too.
    if misc.detected and not grounded:
        misc = Misconception()

    return AnsweringResult(answer_text, grounded, citations, misc, top_sim=top_score)
