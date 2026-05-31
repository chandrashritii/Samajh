// Sarvam Video Tutor — manual testing shell.
// Single-file vanilla JS. Optimised for testing observability.

const $ = (s) => document.querySelector(s);

// ---- App state -----------------------------------------------------------

const state = {
  videoId: null,
  sessionId: null,
  concepts: [],        // [{id, name, one_line_desc, first_timestamp, chunk_ids[]}]
  ytPlayer: null,
  ytApiLoading: false,
};

// ---- Helpers -------------------------------------------------------------

function fmtTs(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  const h = Math.floor(m / 60);
  const mm = (m % 60).toString().padStart(2, '0');
  const ss = r.toString().padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

function escapeHtml(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

function setStatus(el, text, kind = '') {
  el.classList.remove('error', 'ok');
  if (kind) el.classList.add(kind);
  el.textContent = text;
}

function newSessionId() {
  // Reasonable client-side fallback; backend will accept and reuse this id.
  return 'sess_' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

// ---- YouTube iframe player ----------------------------------------------

function loadYtApi() {
  return new Promise((resolve) => {
    if (window.YT && window.YT.Player) return resolve();
    if (state.ytApiLoading) {
      const t = setInterval(() => {
        if (window.YT && window.YT.Player) { clearInterval(t); resolve(); }
      }, 100);
      return;
    }
    state.ytApiLoading = true;
    window.onYouTubeIframeAPIReady = () => resolve();
    const tag = document.createElement('script');
    tag.src = 'https://www.youtube.com/iframe_api';
    document.head.appendChild(tag);
  });
}

async function mountPlayer(videoId) {
  await loadYtApi();
  // Re-create the host element so we can re-mount cleanly on new ingests.
  const wrapper = $('#player-wrapper');
  wrapper.innerHTML = '<div id="yt-player"></div>';
  wrapper.classList.remove('hidden');
  state.ytPlayer = new YT.Player('yt-player', {
    videoId,
    playerVars: { rel: 0, modestbranding: 1 },
  });
}

function seekTo(seconds) {
  if (state.ytPlayer && typeof state.ytPlayer.seekTo === 'function') {
    state.ytPlayer.seekTo(seconds, true);
    if (typeof state.ytPlayer.playVideo === 'function') state.ytPlayer.playVideo();
    // Scroll the player into view so the seek is visible.
    $('#player-wrapper').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ---- Lecture loader ------------------------------------------------------

$('#ingest-btn').addEventListener('click', async () => {
  const url = $('#url').value.trim();
  if (!url) return;
  const btn = $('#ingest-btn');
  const status = $('#ingest-status');
  setStatus(status, 'Ingesting (captions → chunks → embeddings → concept map)…');
  btn.disabled = true;
  try {
    const meta = await postJSON('/ingest', { url });
    state.videoId = meta.video_id;
    renderLectureMeta(meta);
    await mountPlayer(meta.video_id);

    setStatus(status, `Loaded ${meta.video_id} ${meta.cached ? '(from cache)' : '(fresh)'}.`, 'ok');

    // Reveal downstream panels.
    $('#ask-card').classList.remove('hidden');
    $('#concepts-card').classList.remove('hidden');
    $('#mastery-card').classList.remove('hidden');

    if (!state.sessionId) state.sessionId = newSessionId();
    $('#session-id-display').textContent = state.sessionId;

    await refreshConcepts();
    await refreshMastery();
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
});

function renderLectureMeta(meta) {
  const el = $('#lecture-meta');
  el.classList.remove('hidden');
  el.innerHTML = `
    <span><strong>video_id</strong><code>${escapeHtml(meta.video_id)}</code></span>
    <span><strong>duration</strong>${fmtTs(meta.duration)}</span>
    <span><strong>chunks</strong>${meta.segments}</span>
    <span><strong>concepts</strong>${meta.concepts}</span>
    <span><strong>source</strong>${escapeHtml(meta.transcript_source)}</span>
    <span><strong>cached</strong>${meta.cached ? 'yes' : 'no'}</span>
  `;
}

// ---- Ask -----------------------------------------------------------------

$('#ask-btn').addEventListener('click', () => askCurrent());
$('#question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askCurrent();
  }
});

async function askCurrent() {
  if (!state.videoId) return;
  const question = $('#question').value.trim();
  if (!question) return;
  const language = $('#language').value;
  const register = document.querySelector('input[name="register"]:checked').value;
  const status = $('#ask-status');
  const btn = $('#ask-btn');

  setStatus(status, 'Thinking…');
  btn.disabled = true;
  try {
    const data = await postJSON('/ask', {
      video_id: state.videoId,
      session_id: state.sessionId,
      question, language, register,
    });
    state.sessionId = data.session_id;
    $('#session-id-display').textContent = state.sessionId;
    prependAnswerBlock(question, language, register, data);
    setStatus(status, '');
    await refreshMastery(data.concepts_touched || []);
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

function prependAnswerBlock(question, language, register, data) {
  const block = document.createElement('div');
  block.className = 'answer-block' + (data.grounded ? '' : ' refused');

  const groundedBadge = data.grounded
    ? '<span class="badge grounded">grounded</span>'
    : '<span class="badge refused">refused</span>';
  const scoreBadge = `<span class="badge score" title="Top retrieval cosine similarity. Refusal floor ≈ 0.25.">top sim ${(data.top_sim ?? 0).toFixed(3)}</span>`;
  const langBadge = `<span class="badge lang">${escapeHtml(language)}</span>`;
  const regBadge = `<span class="badge register">${escapeHtml(register.replace('_', ' '))}</span>`;

  let html = `<div class="answer-head">${groundedBadge}${scoreBadge}${langBadge}${regBadge}</div>`;
  html += `<p class="question-text">${escapeHtml(question)}</p>`;

  // Misconception block (separable so a TTS layer can ignore it later).
  if (data.misconception && data.misconception.detected) {
    html += `<div class="misconception">
      <div class="label">Let's clear this up first</div>
      <div class="misc-text">${escapeHtml(data.misconception.misconception || '')}</div>
      <div class="corr-text">${escapeHtml(data.misconception.correction || '')}</div>
    </div>`;
  }

  const refusalClass = data.grounded ? '' : ' refusal';
  html += `<div class="answer-text${refusalClass}">${escapeHtml(data.answer)}</div>`;

  // "Why did it refuse?" affordance.
  if (!data.grounded) {
    const nearest = data.nearest_concept
      ? `nearest concept: <code>${escapeHtml(data.nearest_concept)}</code>`
      : 'no concept hit by retrieval';
    html += `<p class="refusal-explainer">↳ Refused because top similarity ${(data.top_sim ?? 0).toFixed(3)} is below the floor; ${nearest}.</p>`;
  }

  // Citations (clickable; clicking seeks the embedded player).
  if (data.citations && data.citations.length) {
    html += '<div class="citations"><h3>From the lecture</h3>';
    for (const c of data.citations) {
      html += `<a class="cite" data-start="${c.start}" data-vid="${escapeHtml(state.videoId)}"
        href="https://youtu.be/${escapeHtml(state.videoId)}?t=${Math.floor(c.start)}"
        target="_blank" rel="noreferrer">
        <span class="cite-ts">${fmtTs(c.start)}–${fmtTs(c.end)}</span>
        <span class="cite-snippet">${escapeHtml(c.snippet)}</span>
        <span class="cite-raw">start=${c.start.toFixed(2)}s</span>
      </a>`;
    }
    html += '</div>';
  }

  // Concepts touched.
  if (data.concepts_touched && data.concepts_touched.length) {
    const chips = data.concepts_touched.map(c => `<span class="touched-chip">${escapeHtml(c)}</span>`).join('');
    html += `<div class="touched-chips"><span class="control-label">concepts touched</span> ${chips}</div>`;
  }

  // Raw JSON toggle.
  const rawId = 'raw-' + Math.random().toString(36).slice(2, 8);
  html += `<div class="raw-json-toggle" data-target="${rawId}">raw JSON</div>
           <pre id="${rawId}" class="hidden">${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;

  block.innerHTML = html;
  $('#answers').prepend(block);

  // Wire citation clicks to seek the embedded player (and prevent external nav).
  block.querySelectorAll('.cite').forEach(a => {
    a.addEventListener('click', (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; // honor modifier-open
      e.preventDefault();
      seekTo(Number(a.dataset.start));
    });
  });

  // Raw-JSON toggle.
  block.querySelector('.raw-json-toggle').addEventListener('click', (e) => {
    const target = block.querySelector('#' + e.currentTarget.dataset.target);
    target.classList.toggle('hidden');
    e.currentTarget.classList.toggle('open');
  });
}

// ---- Concepts panel ------------------------------------------------------

async function refreshConcepts() {
  if (!state.videoId) return;
  try {
    const data = await getJSON(`/lecture/${state.videoId}/concepts`);
    state.concepts = data.concepts || [];
    renderConcepts(state.concepts);
  } catch (e) {
    $('#concept-list').innerHTML =
      `<p class="status error">Failed to load concepts: ${escapeHtml(e.message)}</p>`;
  }
}

function renderConcepts(concepts) {
  if (!concepts.length) {
    $('#concept-list').innerHTML = '<p class="muted small">No concepts extracted yet.</p>';
    return;
  }
  const html = concepts.map(c => `
    <div class="concept-item" data-start="${c.first_timestamp}">
      <span class="concept-ts">${fmtTs(c.first_timestamp)}</span>
      <span class="concept-name">${escapeHtml(c.name)}</span>
      <span class="concept-desc">${escapeHtml(c.one_line_desc || '')}</span>
    </div>
  `).join('');
  const el = $('#concept-list');
  el.innerHTML = html;
  el.querySelectorAll('.concept-item').forEach(item => {
    item.addEventListener('click', () => seekTo(Number(item.dataset.start)));
  });
}

// ---- Mastery panel -------------------------------------------------------

async function refreshMastery(touched = []) {
  if (!state.sessionId || !state.concepts.length) {
    renderMastery([], touched);
    return;
  }
  try {
    const data = await getJSON(`/session/${state.sessionId}/mastery`);
    renderMastery(data.mastery || [], touched);
  } catch (e) {
    renderMastery([], touched);
  }
}

function renderMastery(rows, touched = []) {
  const byId = new Map(rows.map(r => [r.concept_id, r]));
  const touchedSet = new Set(touched);
  const states = { unseen: 0, engaged: 0, shaky: 0, demonstrated: 0 };
  const chips = state.concepts.map(c => {
    const row = byId.get(c.id) || { state: 'unseen', score: 0 };
    states[row.state] = (states[row.state] || 0) + 1;
    const cls = `mastery-chip${touchedSet.has(c.id) ? ' touched' : ''}`;
    const title = `${row.state} · score ${Number(row.score).toFixed(2)}`;
    return `<span class="${cls}" title="${escapeHtml(title)}">
      <span class="dot ${row.state}"></span>${escapeHtml(c.name)}
    </span>`;
  }).join('');
  $('#mastery-list').innerHTML = chips ||
    '<p class="muted small">Concept map empty — nothing to track.</p>';
  $('#mastery-count').textContent =
    `· ${states.engaged} engaged, ${states.shaky} shaky, ${states.demonstrated} demonstrated`;
}

// ---- Session control -----------------------------------------------------

$('#new-session-btn').addEventListener('click', () => {
  state.sessionId = newSessionId();
  $('#session-id-display').textContent = state.sessionId;
  $('#answers').innerHTML = '';
  refreshMastery();
});

$('#copy-session-btn').addEventListener('click', async () => {
  if (!state.sessionId) return;
  try {
    await navigator.clipboard.writeText(state.sessionId);
    const btn = $('#copy-session-btn');
    const orig = btn.textContent;
    btn.textContent = 'copied';
    setTimeout(() => { btn.textContent = orig; }, 1200);
  } catch (_) {}
});

// ==========================================================================
//  P1-B — Voice loop + adaptive viva
// ==========================================================================

// ---- Recording helper (MediaRecorder) ------------------------------------

function makeRecorder() {
  let mediaRecorder = null;
  let chunks = [];
  let stream = null;

  async function start() {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    mediaRecorder.start();
  }

  function stop() {
    return new Promise((resolve) => {
      if (!mediaRecorder) return resolve(null);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        if (stream) stream.getTracks().forEach((t) => t.stop());
        resolve(blob);
      };
      mediaRecorder.stop();
    });
  }

  const isRecording = () => mediaRecorder && mediaRecorder.state === 'recording';
  return { start, stop, isRecording };
}

function pauseVideo() {
  if (state.ytPlayer && typeof state.ytPlayer.pauseVideo === 'function') {
    try { state.ytPlayer.pauseVideo(); } catch (_) {}
  }
}

function playAudioId(audioId, mountEl) {
  if (!audioId) return;
  const el = mountEl || new Audio();
  el.src = `/audio/${audioId}`;
  if (mountEl) mountEl.classList.remove('hidden');
  el.play().catch(() => {});
  return el;
}

// ---- Voice ask (mic button on the Ask panel) -----------------------------

const askRecorder = makeRecorder();

$('#mic-btn').addEventListener('click', async () => {
  if (!state.videoId) return;
  const btn = $('#mic-btn');
  const status = $('#voice-status');

  if (!askRecorder.isRecording()) {
    try {
      await askRecorder.start();
      pauseVideo();
      btn.classList.add('recording');
      btn.textContent = '⏹';
      setStatus(status, 'Recording… click ⏹ to send.');
    } catch (e) {
      setStatus(status, `Mic error: ${e.message}`, 'error');
    }
    return;
  }

  // Stop + send.
  btn.classList.remove('recording');
  btn.textContent = '🎤';
  setStatus(status, 'Transcribing + answering…');
  btn.disabled = true;
  try {
    const blob = await askRecorder.stop();
    const fd = new FormData();
    fd.append('audio', blob, 'clip.webm');
    fd.append('video_id', state.videoId);
    fd.append('session_id', state.sessionId || '');
    fd.append('language', $('#language').value);
    fd.append('register', document.querySelector('input[name="register"]:checked').value);

    const res = await fetch('/ask_voice', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `${res.status}`);

    state.sessionId = data.session_id;
    $('#session-id-display').textContent = state.sessionId;
    $('#question').value = data.transcript || '';
    setStatus(status, `Heard: "${data.transcript || ''}"`, 'ok');
    prependAnswerBlock(data.transcript || '(voice)', $('#language').value,
      document.querySelector('input[name="register"]:checked').value, data);
    if (data.audio) {
      const block = $('#answers').firstChild;
      const player = document.createElement('audio');
      player.controls = true;
      player.className = 'answer-audio';
      block.prepend(player);
      playAudioId(data.audio, player);
    }
    await refreshMastery(data.concepts_touched || []);
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
});

// ---- Adaptive viva -------------------------------------------------------

const vivaRecorder = makeRecorder();
const viva = { id: null, lang: 'en', register: 'balanced' };

$('#start-viva-btn').addEventListener('click', startViva);
$('#viva-close-btn').addEventListener('click', () => $('#viva-card').classList.add('hidden'));

async function startViva() {
  if (!state.videoId || !state.sessionId) return;
  const mode = document.querySelector('input[name="viva-mode"]:checked').value;
  viva.lang = $('#language').value;
  viva.register = document.querySelector('input[name="register"]:checked').value;

  $('#viva-card').classList.remove('hidden');
  $('#viva-summary').classList.add('hidden');
  $('#viva-result').innerHTML = '';
  $('#viva-question').textContent = 'Picking a concept…';
  $('#viva-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const data = await postJSON('/viva/start', {
      session_id: state.sessionId, video_id: state.videoId,
      mode, register: viva.register, language: viva.lang,
    });
    viva.id = data.viva_id;
    renderVivaQuestion(data);
  } catch (e) {
    $('#viva-question').textContent = `Error: ${e.message}`;
  }
}

function renderVivaQuestion(data) {
  if (data.done && !data.concept_id) {
    $('#viva-concept').textContent = '';
    $('#viva-question').textContent = data.question || 'Viva complete.';
    $('#viva-record-btn').classList.add('hidden');
    loadVivaSummary();
    return;
  }
  $('#viva-record-btn').classList.remove('hidden');
  $('#viva-progress').textContent = `· ${data.asked}/${data.total}`;
  $('#viva-concept').innerHTML = `concept: <code>${escapeHtml(data.concept_id || '')}</code>`;
  $('#viva-question').textContent = data.question || '';
  if (data.audio) playAudioId(data.audio, $('#viva-audio'));
}

$('#viva-record-btn').addEventListener('click', async () => {
  if (!viva.id) return;
  const btn = $('#viva-record-btn');
  const status = $('#viva-rec-status');

  if (!vivaRecorder.isRecording()) {
    try {
      await vivaRecorder.start();
      btn.classList.add('recording');
      btn.textContent = '⏹ Stop & submit';
      setStatus(status, 'Recording your answer…');
    } catch (e) {
      setStatus(status, `Mic error: ${e.message}`, 'error');
    }
    return;
  }

  btn.classList.remove('recording');
  btn.textContent = '🎤 Record answer';
  setStatus(status, 'Evaluating…');
  btn.disabled = true;
  try {
    const blob = await vivaRecorder.stop();
    const fd = new FormData();
    fd.append('audio', blob, 'answer.webm');
    fd.append('viva_id', viva.id);
    fd.append('language', viva.lang);
    fd.append('register', viva.register);

    const res = await fetch('/viva/answer', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `${res.status}`);

    renderVivaResult(data);
    setStatus(status, '');
    await refreshMastery();

    if (data.done) {
      $('#viva-record-btn').classList.add('hidden');
      loadVivaSummary();
    } else if (data.next_question) {
      // Advance to the next question after a beat so the learner reads the verdict.
      setTimeout(() => {
        renderVivaQuestion({
          concept_id: data.next_concept_id, question: data.next_question,
          audio: data.audio, asked: data.asked, total: data.total,
        });
      }, 600);
    }
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
});

function renderVivaResult(data) {
  const verdictClass = {
    correct: 'ok', partial: 'warn', incorrect: 'error', unsupported: 'warn',
  }[data.verdict] || '';
  let html = `<div class="viva-verdict ${verdictClass}">
    <span class="badge ${verdictClass}">${escapeHtml(data.verdict)}</span>
    <span class="muted small">heard: "${escapeHtml(data.transcript || '')}"</span>
  </div>`;
  html += `<p class="viva-rationale">${escapeHtml(data.rationale || '')}</p>`;
  if (data.mastery_update) {
    html += `<p class="muted small">${escapeHtml(data.mastery_update.concept_id)}:
      ${escapeHtml(data.mastery_update.from_state)} → <strong>${escapeHtml(data.mastery_update.to_state)}</strong></p>`;
  }
  if (data.reexplanation) {
    html += `<div class="viva-reexplain"><div class="label">Let's revisit</div>${escapeHtml(data.reexplanation)}</div>`;
  }
  if (data.citations && data.citations.length) {
    html += '<div class="citations">';
    for (const c of data.citations) {
      html += `<a class="cite" data-start="${c.start}"
        href="https://youtu.be/${escapeHtml(state.videoId)}?t=${Math.floor(c.start)}" target="_blank" rel="noreferrer">
        <span class="cite-ts">${fmtTs(c.start)}–${fmtTs(c.end)}</span>
        <span class="cite-snippet">${escapeHtml(c.snippet)}</span></a>`;
    }
    html += '</div>';
  }
  const el = $('#viva-result');
  el.innerHTML = html;
  el.querySelectorAll('.cite').forEach(a => a.addEventListener('click', (e) => {
    if (e.metaKey || e.ctrlKey) return;
    e.preventDefault(); seekTo(Number(a.dataset.start));
  }));
  if (data.audio) playAudioId(data.audio, $('#viva-audio'));
}

async function loadVivaSummary() {
  if (!viva.id) return;
  try {
    const s = await getJSON(`/viva/${viva.id}/summary`);
    const sec = $('#viva-summary');
    sec.classList.remove('hidden');
    const list = (rows) => rows.map(r => `<span class="mastery-chip"><span class="dot ${r.state}"></span>${escapeHtml(r.name)}</span>`).join('') || '<span class="muted small">none</span>';
    sec.innerHTML = `<h3>Viva summary</h3>
      <p class="viva-headline">${escapeHtml(s.headline)}</p>
      <div class="summary-group"><span class="control-label">solid</span> ${list(s.solid)}</div>
      <div class="summary-group"><span class="control-label">shaky</span> ${list(s.shaky)}</div>
      <div class="summary-group"><span class="control-label">remaining</span> ${list(s.remaining)}</div>`;
  } catch (e) { /* non-fatal */ }
}
