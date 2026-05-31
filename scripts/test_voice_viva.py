"""P1-B verification: voice loop + adaptive viva, against the live backend.

Uses pre-recorded fixtures under tests/fixtures (generated from Bulbul TTS) so
the path is verifiable without a live mic. STT on synthetic TTS is lossy, so the
voice asserts check the PLUMBING (transcript present, grounded flag, top_sim,
audio retrievable); the viva eval verdicts are asserted on CLEAN TEXT through the
eval core, where correctness is deterministic.

Run:  python scripts/test_voice_viva.py        (backend up on :8000)
Writes /tmp/voice_viva_report.md and prints PASS/FAIL counts.
"""
import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
VID = "J7DzL2_Na80"  # Strang LA L1 — clean English concept map (demo-critical)
FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

rows = []
def check(name, ok, detail=""):
    rows.append((name, "PASS" if ok else "FAIL", detail))


def ensure_fixtures():
    """Generate fixtures via Bulbul if missing (idempotent)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend import sarvam_client as sc
    FIX.mkdir(parents=True, exist_ok=True)
    clips = [("en_column", "What is the column picture?", "en-IN"),
             ("hi_column", "column picture kya hai", "hi-IN"),
             ("en_france", "What is the capital of France", "en-IN")]
    for name, text, lang in clips:
        p = FIX / f"{name}.wav"
        if not p.exists():
            p.write_bytes(sc.text_to_speech(text, target_language_code=lang))


def voice_ask(fixture, language="en", session="vv_test"):
    with open(FIX / fixture, "rb") as f:
        r = requests.post(f"{BASE}/ask_voice", files={"audio": f},
                          data={"video_id": VID, "session_id": session, "language": language},
                          timeout=120)
    return r.status_code, (r.json() if r.ok else {"detail": r.text[:200]})


def main():
    # health + ingest (cached)
    requests.post(f"{BASE}/ingest", json={"url": f"https://www.youtube.com/watch?v=J7DzL2_Na80"}, timeout=180)
    ensure_fixtures()

    # --- B1: voice in/out ---
    st, d = voice_ask("en_column.wav")
    check("ask_voice grounded: HTTP 200", st == 200, f"http={st}")
    check("ask_voice grounded: transcript present", bool(d.get("transcript")), repr(d.get("transcript")))
    check("ask_voice grounded: grounded=true", d.get("grounded") is True, f"grounded={d.get('grounded')} top_sim={d.get('top_sim')}")
    check("ask_voice grounded: citations>=1", len(d.get("citations") or []) >= 1, f"cites={len(d.get('citations') or [])}")
    check("ask_voice grounded: top_sim present", isinstance(d.get("top_sim"), (int, float)) and d.get("top_sim", 0) > 0, f"top_sim={d.get('top_sim')}")
    aid = d.get("audio")
    check("ask_voice grounded: audio id returned", bool(aid), f"audio={aid}")
    if aid:
        ar = requests.get(f"{BASE}/audio/{aid}", timeout=30)
        check("ask_voice grounded: audio retrievable WAV", ar.status_code == 200 and ar.headers.get("content-type") == "audio/wav" and len(ar.content) > 1000,
              f"http={ar.status_code} bytes={len(ar.content)}")

    st, d = voice_ask("en_france.wav")
    check("ask_voice off-topic: refused", st == 200 and d.get("grounded") is False, f"grounded={d.get('grounded')} top_sim={d.get('top_sim')}")

    # empty-ish robustness: unknown video
    with open(FIX / "en_column.wav", "rb") as f:
        r = requests.post(f"{BASE}/ask_voice", files={"audio": f}, data={"video_id": "zzzzzzzzzzz"}, timeout=60)
    check("ask_voice unknown video: 404", r.status_code == 404, f"http={r.status_code}")

    # --- B2/B3: viva flow ---
    r = requests.post(f"{BASE}/viva/start", json={"session_id": "vv_viva", "video_id": VID, "mode": "quiz"}, timeout=120)
    st = r.status_code; d = r.json() if r.ok else {}
    check("viva start: HTTP 200", st == 200, f"http={st}")
    check("viva start: targets a concept", bool(d.get("concept_id")), f"concept={d.get('concept_id')}")
    check("viva start: grounded question", bool(d.get("question")), repr((d.get("question") or "")[:60]))
    check("viva start: spoken audio", bool(d.get("audio")), f"audio={d.get('audio')}")
    viva_id = d.get("viva_id")

    if viva_id:
        with open(FIX / "en_column.wav", "rb") as f:
            r = requests.post(f"{BASE}/viva/answer", files={"audio": f}, data={"viva_id": viva_id}, timeout=120)
        st = r.status_code; d2 = r.json() if r.ok else {}
        check("viva answer: HTTP 200", st == 200, f"http={st}")
        check("viva answer: verdict present", d2.get("verdict") in ("correct", "partial", "incorrect", "unsupported"), f"verdict={d2.get('verdict')}")
        check("viva answer: mastery_update present", bool(d2.get("mastery_update")), str(d2.get("mastery_update")))
        check("viva answer: advances or done", bool(d2.get("next_question")) or d2.get("done"), f"next={bool(d2.get('next_question'))} done={d2.get('done')}")
        check("viva answer: spoken audio", bool(d2.get("audio")), f"audio={d2.get('audio')}")

        r = requests.get(f"{BASE}/viva/{viva_id}/summary", timeout=30)
        check("viva summary: HTTP 200 + headline", r.status_code == 200 and bool(r.json().get("headline")), r.json().get("headline", "")[:60])

    # --- B3 eval core on CLEAN TEXT (deterministic correctness) ---
    from backend import indexing, concept_map, viva as vmod
    _, chunks, _ = indexing.load(VID)
    concepts = concept_map.load(VID)
    cp = next((c for c in concepts if c["id"] == "column-picture"), concepts[0])
    ctx = vmod._chunks_for_concept(chunks, cp)

    good = ("The column picture views the system as a linear combination of the matrix's "
            "column vectors; you ask which combination of columns yields the vector b.")
    e = vmod._evaluate("What does the column picture represent?", good, ctx, "quiz")
    # correct OR partial both count as "passing + grounded" (both → demonstrated);
    # the grader's correct-vs-partial line is its judgment, not our contract.
    check("eval: good answer -> passing(correct/partial) + cited",
          e["verdict"] in ("correct", "partial") and bool(e["citations"]), str(e))

    uns = "The column picture works because a nonzero determinant guarantees invertibility via Cramer's rule."
    e = vmod._evaluate("What does the column picture represent?", uns, ctx, "quiz")
    check("eval: real-but-unsupported -> not correct", e["verdict"] in ("unsupported", "incorrect", "partial"), str(e))

    bad = "The column picture is about chemical reactions in biology."
    e = vmod._evaluate("What does the column picture represent?", bad, ctx, "quiz")
    check("eval: wrong answer -> incorrect", e["verdict"] == "incorrect", str(e))

    tb = "The column picture is about the columns of the matrix."
    e = vmod._evaluate("Explain the column picture.", tb, ctx, "teach_back")
    check("eval: teach-back partial -> partial/incorrect (named gap)", e["verdict"] in ("partial", "incorrect"), str(e))

    # --- render ---
    n_pass = sum(1 for _, v, _ in rows if v == "PASS")
    n_fail = sum(1 for _, v, _ in rows if v == "FAIL")
    out = [f"# P1-B Voice + Viva — verification", "",
           f"**{n_pass} PASS / {n_fail} FAIL**", "",
           "| check | verdict | detail |", "|---|---|---|"]
    for name, v, detail in rows:
        out.append(f"| {name} | **{v}** | {str(detail)[:80]} |")
    Path("/tmp/voice_viva_report.md").write_text("\n".join(out))
    print(f"PASS={n_pass} FAIL={n_fail}")
    for name, v, detail in rows:
        if v == "FAIL":
            print(f"  FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
