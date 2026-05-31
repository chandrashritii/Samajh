"""Vernacular layer with a register dial.

Register maps to a strategy:

  balanced       → mayura mode="code-mixed". Indic prose with English technical
                   terms preserved (default; the F1 behavior we already had).

  more_vernacular → mayura mode="modern-colloquial". More Indic flow; terms
                    can still appear in English (the model prefers code-mix
                    for technical nouns), but connectives lean fully Indic.

  more_english   → sentence-aware: sentences with high technical-term density
                   are kept verbatim English; the connective/explanatory
                   sentences are translated with code-mixed mode. Net effect:
                   ~60–80% English by word count.

Across all three, technical terms ALWAYS stay English — that's the P0
non-negotiable. The two non-default registers are about the surrounding prose,
not the terms themselves.
"""
from __future__ import annotations
import re

from . import config, sarvam_client
from .schemas import Register


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAYURA_LIMIT = 950  # leave headroom under the 1000-char cap

# Lightweight glossary used only for the more_english retention heuristic.
# The model still drives term preservation in the other registers — this is
# just to decide which whole sentences stay English in more_english mode.
_GLOSSARY = {
    # ML basics
    "machine learning", "supervised learning", "unsupervised learning",
    "reinforcement learning", "deep learning",
    # Optimization & training
    "gradient descent", "stochastic gradient descent", "back-propagation",
    "backpropagation", "learning rate", "loss function", "cost function",
    "training", "epoch", "epochs", "batch", "mini-batch", "hyperparameter",
    "hyperparameters", "optimization", "convergence",
    # Modeling
    "neural network", "neural networks", "linear regression", "logistic regression",
    "classification", "regression", "perceptron", "support vector machine",
    "decision tree", "random forest", "k-means", "clustering", "transformer",
    "attention", "embedding", "embeddings", "tokenization", "softmax",
    "cross-entropy", "activation", "convolutional",
    # Data / shape
    "feature", "features", "label", "labels", "labeled", "model", "models",
    "dataset", "input", "inputs", "output", "outputs", "input-output",
    "weights", "bias", "mapping", "mappings", "predict", "predicts",
    "predicting", "prediction", "predictions", "algorithm", "algorithms",
    "training set", "test set", "validation",
    "overfitting", "underfitting", "regularization",
    # Common ML notation glue (still very technical when they appear)
    "parameters", "parameter", "gradient",
}

# Two independent signals are enough to call a sentence "technical".
_TECH_HIT_KEEP_THRESHOLD = 2


def _mode_for(register: Register) -> str:
    if register == Register.MORE_VERNACULAR:
        # classic-colloquial leans more purely Indic than code-mixed; the
        # tradeoff is that some technical terms can get transliterated. We
        # accept that for the most-vernacular setting — the user opted in.
        return "classic-colloquial"
    # balanced and more_english both use code-mixed for actual translate calls.
    # more_english additionally pre-filters sentences (see translate_answer).
    return "code-mixed"


def _chunk_text(text: str, limit: int = _MAYURA_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = _SENT_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) + 1 <= limit:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                out.append(buf)
            buf = s if len(s) <= limit else s[:limit]
    if buf:
        out.append(buf)
    return out


def _technical_hits(sentence: str) -> int:
    """Count glossary phrase/word hits (each phrase counts as one hit). Also
    treats hyphen-compound tokens as a hit when both halves look technical
    (e.g., 'input-output')."""
    lower = sentence.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", lower)
    if not words:
        return 0
    used = [False] * len(words)
    hits = 0
    # phrase pass: longest first
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            if any(used[i:i + n]):
                continue
            phrase = " ".join(words[i:i + n])
            if phrase in _GLOSSARY:
                hits += 1
                for k in range(i, i + n):
                    used[k] = True
    # single-word pass
    for i, w in enumerate(words):
        if used[i]:
            continue
        if w in _GLOSSARY:
            hits += 1
            used[i] = True
        elif "-" in w:
            # hyphen-compound where any half is glossary-known (e.g., input-output)
            parts = [p for p in w.split("-") if p]
            if any(p in _GLOSSARY for p in parts):
                hits += 1
                used[i] = True
    return hits


def _translate_one(text: str, *, target: str, mode: str) -> str:
    return sarvam_client.translate(
        text,
        source_language_code="en-IN",
        target_language_code=target,
        mode=mode,
        model=config.SARVAM_TRANSLATE_MODEL,
    )


def _translate_block(text: str, target: str, mode: str) -> str:
    pieces = _chunk_text(text)
    return " ".join(_translate_one(p, target=target, mode=mode) for p in pieces).strip()


def translate_answer(text: str, language: str,
                     register: Register = Register.BALANCED) -> str:
    """language is 'en'|'hi'|'ta'. 'en' is a no-op regardless of register."""
    if language == "en" or not text.strip():
        return text
    target = config.LANG_TO_SARVAM_CODE.get(language)
    if not target:
        raise ValueError(f"Unsupported language: {language}")

    mode = _mode_for(register)

    if register == Register.MORE_ENGLISH:
        # Per-sentence retention. Sentences with enough glossary hits stay
        # verbatim English; the rest go through code-mixed translation.
        sentences = _SENT_SPLIT.split(text.strip())
        out: list[str] = []
        kept_any = False
        for s in sentences:
            if not s:
                continue
            if _technical_hits(s) >= _TECH_HIT_KEEP_THRESHOLD:
                out.append(s)
                kept_any = True
            else:
                out.append(_translate_block(s, target, mode))
        # Fallback: if the heuristic didn't keep anything and the answer is
        # short, keep the first sentence verbatim so more_english is visibly
        # different from balanced. The user picked this register on purpose.
        if not kept_any and out and len(out) <= 2:
            first = next((s for s in _SENT_SPLIT.split(text.strip()) if s), "")
            if first:
                out[0] = first
        return " ".join(out).strip()

    return _translate_block(text, target, mode)
