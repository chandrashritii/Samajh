// Samajh · समझ — learner-facing surface over the grounded-Q&A backend.
// Vanilla JS, no build step. Reuses existing endpoints; hides all debug scaffolding.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// The curated demo library is the single source of truth (frontend/samajh/library.json).
// Cards, ordering, suggested-question chips, and per-lecture default language all
// derive from it. Loaded at init.
let LIBRARY = [];

const STATE_LABEL = { demonstrated: "learned", shaky: "revisit", engaged: "exploring", unseen: "not yet" };
const STATE_CLASS = { demonstrated: "learned", shaky: "revisit", engaged: "exploring", unseen: "notyet" };
const REGISTER_BY_BLEND = ["more_vernacular", "balanced", "more_english"];

// Answer languages — all with Sarvam translate (Mayura) + voice (Bulbul) support.
const LANGS = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
  { code: "ta", label: "தமிழ்" },
  { code: "bn", label: "বাংলা" },
  { code: "gu", label: "ગુજરાતી" },
  { code: "mr", label: "मराठी" },
  { code: "te", label: "తెలుగు" },
  { code: "kn", label: "ಕನ್ನಡ" },
  { code: "ml", label: "മലയാളം" },
  { code: "pa", label: "ਪੰਜਾਬੀ" },
  { code: "od", label: "ଓଡ଼ିଆ" },
];
// Bulbul v2 voices available on this tier.
const SPEAKERS = [
  { id: "anushka", label: "Anushka (f)" },
  { id: "vidya", label: "Vidya (f)" },
  { id: "arya", label: "Arya (f)" },
  { id: "manisha", label: "Manisha (f)" },
  { id: "abhilash", label: "Abhilash (m)" },
  { id: "karun", label: "Karun (m)" },
  { id: "hitesh", label: "Hitesh (m)" },
];

const state = {
  videoId: null, sessionId: null, ytPlayer: null, ytLoading: false,
  concepts: [], lang: "en", title: null, suggestedQuestions: [], speaker: "anushka",
};

function libEntry(videoId) { return LIBRARY.find(e => e.video_id === videoId) || null; }

// ---- utilities -----------------------------------------------------------

function fmtTs(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? h + ":" : ""}${mm}:${String(r).padStart(2, "0")}`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
}
function newSession() {
  return "s_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `${r.status}`);
  return d;
}
async function getJSON(url) {
  const r = await fetch(url);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `${r.status}`);
  return d;
}
function currentRegister() {
  if (state.lang === "en") return "balanced";
  return REGISTER_BY_BLEND[Number($("#blend").value) || 1];
}

// ---- YouTube player ------------------------------------------------------

function loadYt() {
  return new Promise((res) => {
    if (window.YT && window.YT.Player) return res();
    if (state.ytLoading) {
      const t = setInterval(() => { if (window.YT && window.YT.Player) { clearInterval(t); res(); } }, 80);
      return;
    }
    state.ytLoading = true;
    window.onYouTubeIframeAPIReady = () => res();
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  });
}
async function mountPlayer(videoId) {
  await loadYt();
  const frame = $(".player-frame");
  frame.innerHTML = '<div id="yt-player"></div>';
  state.ytPlayer = new YT.Player("yt-player", {
    videoId, playerVars: { rel: 0, modestbranding: 1 },
  });
}
function seekTo(sec) {
  const p = state.ytPlayer;
  if (p && p.seekTo) { p.seekTo(sec, true); p.playVideo && p.playVideo(); }
  $(".player-frame").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function pauseVideo() { try { state.ytPlayer && state.ytPlayer.pauseVideo && state.ytPlayer.pauseVideo(); } catch (_) {} }

// ---- hero / chips --------------------------------------------------------

function renderChips() {
  $("#chip-row").innerHTML = LIBRARY.map((e, i) => `
    <button class="chip${i === 0 ? " suggested" : ""}" data-vid="${esc(e.video_id)}">
      <span class="chip-title">${esc(e.title)}</span>
      <span class="chip-sub">${esc(e.channel)}</span>
    </button>`).join("");
  $$(".chip").forEach(ch => ch.addEventListener("click", () => {
    const url = `https://www.youtube.com/watch?v=${ch.dataset.vid}`;
    $("#url").value = url;
    openLecture(url);
  }));
}

// Language + voice controls appear in two places (ask panel + viva). They share
// state.lang / state.speaker; helpers take element ids so each instance syncs.

const _speakerLabel = id => (SPEAKERS.find(s => s.id === id) || SPEAKERS[0]).label;
const _speakerShort = id => _speakerLabel(id).replace(/\s*\(.\)$/, "");

function renderLangs(segId = "lang-seg") {
  const seg = document.getElementById(segId);
  if (!seg) return;
  seg.innerHTML = LANGS.map(l =>
    `<button class="seg-opt${l.code === state.lang ? " active" : ""}" data-lang="${l.code}" lang="${l.code}">${esc(l.label)}</button>`
  ).join("");
  seg.querySelectorAll(".seg-opt").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
}

function renderSpeakers(triggerId = "voice-trigger", menuId = "voice-menu", currentId = "voice-current") {
  const trigger = document.getElementById(triggerId), menu = document.getElementById(menuId);
  const current = document.getElementById(currentId);
  if (!trigger || !menu) return;
  const dd = trigger.closest(".voice-dd");
  if (current) current.textContent = _speakerShort(state.speaker);

  menu.innerHTML = SPEAKERS.map(s =>
    `<li role="option" class="voice-opt${s.id === state.speaker ? " active" : ""}" data-id="${s.id}" aria-selected="${s.id === state.speaker}">${esc(s.label)}</li>`
  ).join("");

  const close = () => { menu.classList.add("hidden"); trigger.setAttribute("aria-expanded", "false"); };
  const open = () => { menu.classList.remove("hidden"); trigger.setAttribute("aria-expanded", "true"); };
  trigger.addEventListener("click", e => {
    e.stopPropagation();
    menu.classList.contains("hidden") ? open() : close();
  });
  menu.querySelectorAll(".voice-opt").forEach(li => li.addEventListener("click", () => {
    setSpeaker(li.dataset.id);
    close();
  }));
  document.addEventListener("click", e => { if (dd && !dd.contains(e.target)) close(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
}

// Apply a speaker choice to every voice dropdown instance.
function setSpeaker(id) {
  state.speaker = id;
  document.querySelectorAll(".voice-dd").forEach(dd => {
    const cur = dd.querySelector("[id$='voice-current']");
    if (cur) cur.textContent = _speakerShort(id);
    dd.querySelectorAll(".voice-opt").forEach(o => {
      const on = o.dataset.id === id;
      o.classList.toggle("active", on);
      o.setAttribute("aria-selected", on);
    });
  });
}

// Apply a language choice to every lang segmented control + the register dial.
function setLang(lang) {
  if (!lang) return;
  state.lang = lang;
  document.querySelectorAll(".lang-seg .seg-opt").forEach(x => x.classList.toggle("active", x.dataset.lang === lang));
  const blend = document.getElementById("blend-row");
  if (blend) {
    if (lang === "en") blend.classList.add("hidden");
    else {
      blend.classList.remove("hidden");
      const left = document.getElementById("blend-left");
      const label = (LANGS.find(l => l.code === lang) || {}).label || "vernacular";
      if (left) left.textContent = `more ${label}`;
    }
  }
}
$$("#lang-seg .seg-opt").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));

// ---- suggested-question chips (from the lecture's library entry) ----------

function renderSuggestedQuestions() {
  const box = $("#suggested-qs");
  if (!box) return;
  const qs = state.suggestedQuestions || [];
  if (!qs.length) { box.innerHTML = ""; box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.innerHTML = `<span class="sq-label">try asking</span>` +
    qs.map(q => `<button class="sq-chip" type="button">${esc(q)}</button>`).join("");
  $$(".sq-chip", box).forEach(c => c.addEventListener("click", () => {
    $("#question").value = c.textContent;
    askText();
  }));
}

// ---- ask: text -----------------------------------------------------------

$("#question").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askText(); }
});

async function askText() {
  const q = $("#question").value.trim();
  if (!q || !state.videoId) return;
  const turn = addPendingTurn(q);
  $("#question").value = "";
  try {
    const data = await postJSON("/ask", {
      video_id: state.videoId, session_id: state.sessionId,
      question: q, language: state.lang, register: currentRegister(),
    });
    state.sessionId = data.session_id;
    fillTurn(turn, data);
    await refreshMastery(data.concepts_touched || []);
  } catch (e) {
    fillTurnError(turn, e.message);
  }
}

// ---- ask: voice ----------------------------------------------------------

function makeRecorder() {
  let mr = null, chunks = [], stream = null;
  return {
    async start() {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = []; mr = new MediaRecorder(stream);
      mr.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      mr.start();
    },
    stop() {
      return new Promise(res => {
        if (!mr) return res(null);
        mr.onstop = () => { stream.getTracks().forEach(t => t.stop()); res(new Blob(chunks, { type: mr.mimeType || "audio/webm" })); };
        mr.stop();
      });
    },
    active: () => mr && mr.state === "recording",
  };
}
const askRec = makeRecorder();

$("#mic-btn").addEventListener("click", async () => {
  if (!state.videoId) return;
  const btn = $("#mic-btn"), hint = $("#ask-hint");
  if (!askRec.active()) {
    try {
      await askRec.start(); pauseVideo();
      btn.classList.add("recording");
      hint.className = "ask-hint live"; hint.textContent = "Listening… tap again when you're done.";
    } catch (e) { hint.className = "ask-hint error"; hint.textContent = `Mic unavailable: ${e.message}`; }
    return;
  }
  btn.classList.remove("recording");
  hint.className = "ask-hint live"; hint.textContent = "Transcribing and answering…";
  btn.disabled = true;
  const turn = addPendingTurn("…");
  try {
    const blob = await askRec.stop();
    const fd = new FormData();
    fd.append("audio", blob, "clip.webm");
    fd.append("video_id", state.videoId);
    fd.append("session_id", state.sessionId || "");
    fd.append("language", state.lang);
    fd.append("register", currentRegister());
    fd.append("speaker", state.speaker);
    const r = await fetch("/ask_voice", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `${r.status}`);
    state.sessionId = data.session_id;
    setTurnQuestion(turn, data.transcript || "(your question)");
    fillTurn(turn, data, /*autoplay*/ true);
    hint.className = "ask-hint"; hint.textContent = "Tap the mic and speak your doubt — in any language.";
    await refreshMastery(data.concepts_touched || []);
  } catch (e) {
    fillTurnError(turn, e.message);
    hint.className = "ask-hint error"; hint.textContent = `Voice failed: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
});

// ---- thread rendering ----------------------------------------------------

function addPendingTurn(q) {
  const el = document.createElement("div");
  el.className = "turn";
  el.innerHTML = `<p class="q">${esc(q)}</p><div class="answer-card"><p class="answer-body thinking">…thinking, only from this lecture.</p></div>`;
  $("#thread").prepend(el);
  return el;
}
function setTurnQuestion(turn, q) { turn.querySelector(".q").textContent = q; }

function fillTurn(turn, data, autoplay = false) {
  const grounded = !!data.grounded;
  const card = turn.querySelector(".answer-card");
  card.className = "answer-card" + (grounded ? "" : " refused");
  let html = "";

  if (data.audio) html += `<audio class="answer-audio" controls src="/audio/${esc(data.audio)}"></audio>`;

  const m = data.misconception;
  if (m && m.detected) {
    html += `<div class="misc-card">
      <div class="misc-label">let's clear this up first</div>
      ${m.misconception ? `<div class="misc-what">${esc(m.misconception)}</div>` : ""}
      ${m.correction ? `<div class="misc-fix">${esc(m.correction)}</div>` : ""}
    </div>`;
  }

  html += `<p class="answer-body${grounded ? "" : " refused"}">${esc(data.answer)}</p>`;

  if (grounded && data.citations && data.citations.length) {
    html += `<div class="cites">`;
    for (const c of data.citations) {
      html += `<a class="cite-chip" data-start="${c.start}"
        href="https://youtu.be/${esc(state.videoId)}?t=${Math.floor(c.start)}" target="_blank" rel="noreferrer">
        <span class="play-glyph">▶</span><span>from</span><span class="cite-time">${fmtTs(c.start)}</span></a>`;
    }
    html += `</div>`;
  }
  card.innerHTML = html;

  card.querySelectorAll(".cite-chip").forEach(a => a.addEventListener("click", e => {
    if (e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault(); seekTo(Number(a.dataset.start));
  }));
  if (autoplay && data.audio) { const au = card.querySelector(".answer-audio"); au && au.play().catch(() => {}); }
}
function fillTurnError(turn, msg) {
  const card = turn.querySelector(".answer-card");
  card.className = "answer-card refused";
  card.innerHTML = `<p class="answer-body refused">Something went wrong: ${esc(msg)}</p>`;
}

// ---- concepts + mastery (learner-facing) ---------------------------------

async function refreshConcepts() {
  try {
    const d = await getJSON(`/lecture/${state.videoId}/concepts`);
    state.concepts = d.concepts || [];
  } catch (_) { state.concepts = []; }
}
async function refreshMastery(touched = []) {
  let rows = [];
  if (state.sessionId) {
    try { rows = (await getJSON(`/session/${state.sessionId}/mastery`)).mastery || []; } catch (_) {}
  }
  const byId = new Map(rows.map(r => [r.concept_id, r]));
  const touchedSet = new Set(touched);
  const track = $("#concept-track");
  if (!state.concepts.length) { track.innerHTML = `<span class="ask-hint">Concept map is still warming up…</span>`; return; }
  track.innerHTML = state.concepts.map(c => {
    const st = (byId.get(c.id) || {}).state || "unseen";
    return `<span class="concept-pill${touchedSet.has(c.id) ? " touched" : ""}" data-start="${c.first_timestamp || 0}" title="${esc(STATE_LABEL[st])}">
      <i class="dot ${STATE_CLASS[st]}"></i>${esc(c.name)}</span>`;
  }).join("");
  $$("#concept-track .concept-pill").forEach(p => p.addEventListener("click", () => seekTo(Number(p.dataset.start))));
}

// ==========================================================================
//  Viva
// ==========================================================================

const viva = { id: null, mode: "quiz", started: false, recording: makeRecorder() };

$("#viva-entry-btn").addEventListener("click", () => {
  $("#viva-overlay").classList.remove("hidden");
  $("#viva-summary").classList.add("hidden");
  viva.started = false;
  setModeLock(false);
  resetVivaStage();
});
$("#viva-x").addEventListener("click", () => $("#viva-overlay").classList.add("hidden"));
$("#viva-overlay").addEventListener("click", e => { if (e.target === $("#viva-overlay")) $("#viva-overlay").classList.add("hidden"); });

// Once a viva is underway, the mode is fixed for that run — switching mid-session
// would silently strand the in-flight viva_id (the flaky tab-doesn't-change UX).
// We lock the toggle and offer a clean confirm-to-restart instead.
function setModeLock(locked) {
  $("#viva-mode").classList.toggle("locked", locked);
}
$$("#viva-mode .seg-opt").forEach(b => b.addEventListener("click", () => {
  const target = b.dataset.mode;
  if (target === viva.mode) return;
  if (viva.started) {
    const other = target === "quiz" ? "Quiz me" : "I'll explain";
    if (!confirm(`Switch to "${other}"? This restarts the session and your current progress in this viva is lost.`)) {
      return;  // keep the active tab as-is; no flaky half-switch
    }
    // Confirmed: reset and start fresh in the new mode.
    viva.mode = target;
    $$("#viva-mode .seg-opt").forEach(x => x.classList.toggle("active", x.dataset.mode === target));
    viva.started = false;
    viva.id = null;
    setModeLock(false);
    $("#viva-summary").classList.add("hidden");
    beginViva();
    return;
  }
  viva.mode = target;
  $$("#viva-mode .seg-opt").forEach(x => x.classList.toggle("active", x.dataset.mode === target));
}));

function resetVivaStage() {
  $("#viva-stage").innerHTML = `
    <p class="viva-lead">Pick a mode, then begin — Samajh asks about the ideas you haven't nailed yet, and listens to your spoken answer.</p>
    <button id="viva-begin" class="accent-btn">Begin</button>`;
  $("#viva-begin").addEventListener("click", beginViva);
}

async function beginViva() {
  $("#viva-stage").innerHTML = `<p class="viva-lead">Choosing a concept…</p>`;
  try {
    const d = await postJSON("/viva/start", {
      session_id: state.sessionId, video_id: state.videoId,
      mode: viva.mode, register: currentRegister(), language: state.lang,
      speaker: state.speaker,
    });
    viva.id = d.viva_id;
    if (d.done && !d.concept_id) { showVivaSummary(); return; }
    renderVivaQuestion(d);
  } catch (e) {
    $("#viva-stage").innerHTML = `<p class="viva-lead">Couldn't start: ${esc(e.message)}</p>`;
  }
}

function renderVivaQuestion(d) {
  // A question is on screen → the session is live; lock the mode toggle.
  viva.started = true;
  setModeLock(true);
  $("#viva-stage").innerHTML = `
    <div class="viva-progress">question ${(d.asked || 0) + 1} of ${d.total || "?"}</div>
    <p class="viva-question">${esc(d.question)}</p>
    ${d.audio ? `<audio class="viva-q-audio" controls autoplay src="/audio/${esc(d.audio)}"></audio>` : ""}
    <div class="viva-rec-row">
      <button id="viva-rec" class="accent-btn">🎤 Answer aloud</button>
      <span id="viva-rec-status" class="viva-rec-status"></span>
    </div>
    <div id="viva-result"></div>`;
  $("#viva-rec").addEventListener("click", vivaRecordToggle);
}

async function vivaRecordToggle() {
  const btn = $("#viva-rec"), st = $("#viva-rec-status");
  if (!viva.recording.active()) {
    try {
      await viva.recording.start();
      btn.textContent = "⏹ Stop & submit";
      st.className = "viva-rec-status live"; st.textContent = "listening…";
    } catch (e) { st.className = "viva-rec-status"; st.textContent = `mic: ${e.message}`; }
    return;
  }
  btn.textContent = "🎤 Answer aloud"; btn.disabled = true;
  st.className = "viva-rec-status live"; st.textContent = "checking against the lecture…";
  try {
    const blob = await viva.recording.stop();
    const fd = new FormData();
    fd.append("audio", blob, "answer.webm");
    fd.append("viva_id", viva.id);
    fd.append("language", state.lang);
    fd.append("register", currentRegister());
    fd.append("speaker", state.speaker);
    const r = await fetch("/viva/answer", { method: "POST", body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `${r.status}`);
    renderVivaResult(d);
    await refreshMastery();
  } catch (e) {
    st.className = "viva-rec-status"; st.textContent = `failed: ${e.message}`;
    btn.disabled = false;
  }
}

function renderVivaResult(d) {
  $("#viva-rec-status").textContent = "";
  const vb = $("#viva-rec").parentElement;
  vb && vb.remove();
  const res = $("#viva-result");
  let html = `<div class="verdict ${esc(d.verdict)}">${esc(d.verdict)}
    <span class="verdict-heard">you said: "${esc(d.transcript || "")}"</span></div>`;
  if (d.rationale) html += `<p class="viva-rationale">${esc(d.rationale)}</p>`;
  if (d.reexplanation) html += `<div class="viva-reexplain"><div class="label">let's revisit</div>${esc(d.reexplanation)}</div>`;
  if (d.citations && d.citations.length) {
    html += `<div class="cites">`;
    for (const c of d.citations)
      html += `<a class="cite-chip" data-start="${c.start}" href="https://youtu.be/${esc(state.videoId)}?t=${Math.floor(c.start)}" target="_blank" rel="noreferrer"><span class="play-glyph">▶</span><span>from</span><span class="cite-time">${fmtTs(c.start)}</span></a>`;
    html += `</div>`;
  }
  if (d.audio) html += `<audio class="viva-q-audio" controls autoplay src="/audio/${esc(d.audio)}"></audio>`;
  html += d.done
    ? `<div class="viva-rec-row"><button id="viva-finish" class="accent-btn">See how I did</button></div>`
    : `<div class="viva-rec-row"><button id="viva-next" class="accent-btn">Next question</button></div>`;
  res.innerHTML = html;
  res.querySelectorAll(".cite-chip").forEach(a => a.addEventListener("click", e => { e.preventDefault(); seekTo(Number(a.dataset.start)); }));
  if (d.done) {
    $("#viva-finish").addEventListener("click", showVivaSummary);
  } else {
    // The next question was already spoken together with the verdict audio above,
    // so advancing just re-renders the question text (no fresh audio fetch).
    $("#viva-next").addEventListener("click", () => renderVivaQuestion({
      question: d.next_question, audio: null, asked: d.asked, total: d.total,
    }));
  }
}

async function showVivaSummary() {
  // Session is over — unlock the mode toggle for the next run.
  viva.started = false;
  setModeLock(false);
  try {
    const s = await getJSON(`/viva/${viva.id}/summary`);
    $("#viva-stage").innerHTML = "";
    const sum = $("#viva-summary");
    sum.classList.remove("hidden");
    const pills = rows => rows.length
      ? rows.map(r => `<span class="concept-pill"><i class="dot ${STATE_CLASS[r.state]}"></i>${esc(r.name)}</span>`).join("")
      : `<span class="ask-hint">none</span>`;
    sum.innerHTML = `
      <h3>How you did</h3>
      <p class="viva-headline">${esc(s.headline)}</p>
      <div class="summary-group"><span class="sg-label">learned</span><div class="concept-track">${pills(s.solid)}</div></div>
      <div class="summary-group"><span class="sg-label">revisit</span><div class="concept-track">${pills(s.shaky)}</div></div>
      <div class="viva-rec-row"><button class="accent-btn" id="viva-done">Done</button></div>`;
    $("#viva-done").addEventListener("click", () => $("#viva-overlay").classList.add("hidden"));
  } catch (e) {
    $("#viva-stage").innerHTML = `<p class="viva-lead">Couldn't load summary: ${esc(e.message)}</p>`;
  }
}

// ---- init ----------------------------------------------------------------

state.sessionId = newSession();

async function init() {
  try {
    LIBRARY = await getJSON("/static/samajh/library.json");
  } catch (_) {
    LIBRARY = [];
  }
  renderChips();
  renderLangs();
  renderSpeakers();
  // Viva-scoped copies of the language + voice controls (same shared state).
  renderLangs("viva-lang");
  renderSpeakers("viva-voice-trigger", "viva-voice-menu", "viva-voice-current");
  // Deep link: ?url=<youtube> or ?v=<videoId> auto-opens that lecture (shareable,
  // and lets a reviewer land straight in a loaded surface).
  const p = new URLSearchParams(location.search);
  const url = p.get("url") || (p.get("v") ? `https://www.youtube.com/watch?v=${p.get("v")}` : null);
  if (url) { $("#url").value = url; openLecture(url); }
}
init();
