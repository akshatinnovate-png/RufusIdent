// Enrolment station: continuous video. A student steps up → "new student
// detected" → the operator picks or types their details → PROCEED → look
// straight / left / right → frames uploaded → "next please". The backend
// picks the best photos and enrols them in the background (backend/enrolment.py).

import { state, search, studentsIn, classKey, upsertStudents, hasConsent } from './data.js';
import { loadEngine, landmarker, constraints } from './camera.js';
import { $, esc, toast } from './ui.js';

const PHASES = [
  { key: 'straight', say: 'LOOK STRAIGHT AT THE CAMERA', want: 12 },
  { key: 'left', say: '◂ SLOWLY TURN YOUR HEAD LEFT', want: 8 },
  { key: 'right', say: 'NOW TURN YOUR HEAD RIGHT ▸', want: 8 },
];
const MIN_FACE = 140;           // px: a student standing at the station
const TURN_MIN = 0.2, TURN_MAX = 0.5;
const SHOT_GAP_MS = 120;
const PHASE_TIMEOUT_MS = 9000;
const CROP_PX = 640;

let video, canvas, ctx, stream, raf, poll, running = false;
let lastVT = -1;
let mode = 'idle';              // idle | detected | capture | upload | done
let seenSince = 0, lostSince = 0;
let phase = 0, phaseStart = 0, lastShot = 0, leftSign = 0;
let shots = [];                 // { phase, blob }
let sessionCount = 0;
let status = { enrolled: {}, students: {}, jobs: [], queue: 0 };
let selectedClass = null;
let current = null;             // the student being enrolled
let outbox = [];                // captures saved on this laptop, not yet accepted by the backend
let flushing = false, flushTimer = null, backoff = 5000;

const base = () => state.settings.backendUrl?.replace(/\/$/, '');

export async function openStation() {
  video = $('#stVideo'); canvas = $('#stCanvas'); ctx = canvas.getContext('2d');
  $('#station').hidden = false;
  document.body.classList.add('hud-on');
  setMode('idle');
  setStatus('STARTING CAMERA', 'warn');
  if (!base()) { setStatus('NO BACKEND · SET IT IN SETTINGS → GATE', 'bad'); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints());
    video.srcObject = stream;
    await video.play();
    await loadEngine();
  } catch (e) {
    setStatus(`CAMERA ERROR · ${e.message}`, 'bad');
    return;
  }
  setStatus('READY', 'ok');
  running = true;
  resize();
  addEventListener('resize', resize);
  loop();
  refresh();
  poll = setInterval(refresh, 2000);
  outbox = await obAll().catch(() => []);
  renderOutbox();
  flush();
}

export function closeStation() {
  running = false;
  cancelAnimationFrame(raf);
  clearInterval(poll);
  removeEventListener('resize', resize);
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  $('#station').hidden = true;
  document.body.classList.remove('hud-on');
}

function setStatus(text, kind) {
  const el = $('#stStatus');
  el.textContent = text;
  el.className = `hud-status ${kind}`;
}

function say(text, kind = '') {
  const el = $('#stPrompt');
  if (el.textContent !== text) el.textContent = text;
  el.className = `hud-prompt ${kind}`;
}

function resize() {
  const dpr = devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ───────────── face reading ─────────────

function readFace() {
  const res = landmarker().detectForVideo(video, performance.now());
  const faces = (res.faceLandmarks || []).map(lm => {
    const vw = video.videoWidth, vh = video.videoHeight;
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of lm) { if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x; if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y; }
    const l = lm[33], r = lm[263], n = lm[1], ml = lm[61], mr = lm[291];
    const eyeW = Math.hypot((r.x - l.x) * vw, (r.y - l.y) * vh) || 1;
    const eyeY = (l.y + r.y) / 2, mouthY = (ml.y + mr.y) / 2;
    return {
      x: x0 * vw, y: y0 * vh, w: (x1 - x0) * vw, h: (y1 - y0) * vh,
      yaw: ((n.x - (l.x + r.x) / 2) * vw) / eyeW,
      pitch: (n.y - eyeY) / (mouthY - eyeY + 1e-6),
    };
  });
  faces.sort((a, b) => b.w - a.w);
  // Only someone actually close counts as a second student (not people passing far behind).
  const close = faces.filter(x => faces[0] && x.w >= faces[0].w * 0.5).length;
  return { face: faces[0], count: close };
}

// ───────────── main loop ─────────────

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  let f, count;
  try { ({ face: f, count } = readFace()); } catch { return; }
  const now = performance.now();
  const present = f && f.w >= MIN_FACE;
  if (present) { lostSince = 0; seenSince ||= now; lastFace = f; } else { seenSince = 0; lostSince ||= now; }
  draw(f);

  if (mode === 'idle') {
    say(f && !present ? 'COME A LITTLE CLOSER' : 'STEP UP TO THE CAMERA');
    if (present && now - seenSince > 600) setMode('detected');
  } else if (mode === 'detected') {
    if (!present && now - lostSince > 4000 && !$('#stSearch').value && !$('#stId').value) setMode('idle');
  } else if (mode === 'capture') {
    capture(f, count, present, now);
  } else if (mode === 'done') {
    if (!present && now - lostSince > 1200) setMode('idle');
  }
}

function draw(f) {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.clearRect(0, 0, cw, ch);
  if (!f) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(cw / vw, ch / vh), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  const x = cw - ((f.x + f.w) * s + ox), y = f.y * s + oy, w = f.w * s, h = f.h * s;
  const color = mode === 'capture' ? '#3dffa2' : f.w >= MIN_FACE ? '#4fd8ff' : 'rgba(255,255,255,.5)';
  const c = Math.min(w, h) * 0.22, pad = w * 0.12;
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.beginPath();
  const bx = x - pad, by = y - pad * 1.5, bw = w + pad * 2, bh = h + pad * 2.2;
  ctx.moveTo(bx, by + c); ctx.lineTo(bx, by); ctx.lineTo(bx + c, by);
  ctx.moveTo(bx + bw - c, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + c);
  ctx.moveTo(bx + bw, by + bh - c); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - c, by + bh);
  ctx.moveTo(bx + c, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - c);
  ctx.stroke();
}

function setMode(m) {
  mode = m;
  $('#stForm').hidden = m !== 'detected';
  $('#stSteps').hidden = m !== 'capture';
  if (m === 'idle') { say('STEP UP TO THE CAMERA'); clearForm(); }
  if (m === 'detected') {
    say('NEW STUDENT DETECTED · ENTER DETAILS', 'ok');
    setTimeout(() => (current ? $('#stGo') : $('#stSearch')).focus(), 50);
    if (!current) identify();
  }
}

// Already enrolled? One quick check when a student steps up.
let lastFace = null;
async function identify() {
  if (!lastFace) return;
  const c = document.createElement('canvas');
  const f = lastFace, vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(Math.max(f.w, f.h) * 2.6, vw, vh);
  c.width = c.height = Math.min(CROP_PX, Math.round(side));
  c.getContext('2d').drawImage(video, Math.min(Math.max(f.x + f.w / 2 - side / 2, 0), vw - side), Math.min(Math.max(f.y + f.h / 2 - side / 2, 0), vh - side), side, side, 0, 0, c.width, c.height);
  try {
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.9));
    const fd = new FormData();
    fd.append('image', blob, 'face.jpg');
    const j = await (await fetch(base() + '/enrol/identify', { method: 'POST', body: fd, signal: AbortSignal.timeout(8000) })).json();
    if (j.known && mode === 'detected' && !current && !$('#stId').value) {
      const s = state.byId.get(j.student_id);
      fillForm(s || { id: j.student_id, name: status.students[j.student_id]?.name || '', cls: '', section: '' });
      $('#stForm h3').textContent = `ALREADY ENROLLED · ${(s?.name || j.student_id).toUpperCase()}`;
      say(`ALREADY ENROLLED · ${(s?.name || j.student_id).toUpperCase()} · NEXT PLEASE (OR RE-TAKE)`, 'warn');
    }
  } catch {}
}

// ───────────── the form ─────────────

function clearForm() {
  current = null;
  $('#stForm h3').textContent = 'NEW STUDENT DETECTED';
  for (const id of ['stSearch', 'stId', 'stName', 'stCls', 'stSec']) $('#' + id).value = '';
  $('#stConsent').checked = false;
  $('#stResults').innerHTML = '';
  validate();
}

function fillForm(s) {
  current = s;
  $('#stId').value = s.id; $('#stName').value = s.name; $('#stCls').value = s.cls; $('#stSec').value = s.section;
  $('#stConsent').checked = hasConsent(s) || !!status.students[s.id]?.consent;
  $('#stSearch').value = ''; $('#stResults').innerHTML = '';
  validate();
  if (mode === 'idle') setMode('detected');
  $('#stGo').focus();
}

function formStudent() {
  return {
    id: $('#stId').value.trim(), name: $('#stName').value.trim(),
    cls: $('#stCls').value.trim().toUpperCase().replace(/^CLASS\s*/, ''), section: $('#stSec').value.trim().toUpperCase(),
  };
}

function validate() {
  const s = formStudent();
  const missing = ['id', 'name', 'cls', 'section'].filter(k => !s[k]);
  const note = [];
  if (missing.length) note.push('fill in admission no, name, class and section');
  else if (!$('#stConsent').checked) note.push('tick parent consent to continue');
  const existing = state.byId.get(s.id);
  if (s.id && existing && existing.name.toLowerCase() !== s.name.toLowerCase()) note.push(`⚠ ${s.id} is ${existing.name} in the roster`);
  if (s.id && (status.enrolled[s.id] || status.students[s.id]?.enrolled_at)) note.push('already enrolled: this will replace their station photos');
  $('#stNote').textContent = note.join(' · ');
  $('#stGo').disabled = missing.length > 0 || !$('#stConsent').checked;
}

// ───────────── capture ─────────────

function startCapture() {
  const s = formStudent();
  current = { ...(state.byId.get(s.id) || {}), ...s, consent: 'yes' };
  shots = []; phase = 0; leftSign = 0; phaseStart = performance.now(); lastShot = 0;
  setMode('capture');
  updateSteps();
}

function updateSteps() {
  document.querySelectorAll('#stSteps span').forEach((el, i) => {
    const p = PHASES[i];
    const n = shots.filter(x => x.phase === p.key).length;
    el.classList.toggle('on', i === phase);
    el.classList.toggle('ok', i < phase);
    el.style.setProperty('--p', `${Math.min(100, (n / p.want) * 100)}%`);
  });
}

function poseOk(f, key) {
  if (key === 'straight') return Math.abs(f.yaw) < 0.15 && f.pitch > 0.3 && f.pitch < 0.85;
  const a = Math.abs(f.yaw);
  if (a < TURN_MIN || a > TURN_MAX) return false;
  const sign = Math.sign(f.yaw);
  if (key === 'left') { leftSign ||= sign; return sign === leftSign; }   // whichever way they turn first…
  return sign === -leftSign;                                              // …then the other way
}

function capture(f, count, present, now) {
  const p = PHASES[phase];
  if (!present) {
    say('STAY IN FRONT OF THE CAMERA', 'warn');
    if (now - lostSince > 6000) { toast('Student left during capture: try again.', { kind: 'warn' }); setMode('detected'); }
    return;
  }
  if (count > 1) { say('ONLY ONE STUDENT IN FRONT OF THE CAMERA', 'warn'); return; }
  if (p.key !== 'straight' && Math.abs(f.yaw) > TURN_MAX) say('A LITTLE LESS', 'warn');
  else if (p.key !== 'straight' && Math.abs(f.yaw) < TURN_MIN && now - phaseStart > 2500) say(`${p.say} · A BIT MORE`, 'ok');
  else say(p.say, 'ok');

  if (poseOk(f, p.key) && now - lastShot > SHOT_GAP_MS) {
    lastShot = now;
    grab(f, p.key);
    updateSteps();
  }
  const have = shots.filter(x => x.phase === p.key).length;
  const timedOut = now - phaseStart > PHASE_TIMEOUT_MS;
  if (have >= p.want || (timedOut && have >= 3)) {
    phase++;
    phaseStart = now;
    if (phase >= PHASES.length) upload();
    else updateSteps();
  } else if (timedOut && now - phaseStart > PHASE_TIMEOUT_MS * 2) {
    phase++; phaseStart = now;                                            // can't get this pose: carry on with what we have
    if (phase >= PHASES.length) upload();
  }
}

// A crop around the face (with some context), at most 640 px.
function grab(f, key) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(Math.max(f.w, f.h) * 2.6, vw, vh);
  const x0 = Math.min(Math.max(f.x + f.w / 2 - side / 2, 0), vw - side);
  const y0 = Math.min(Math.max(f.y + f.h / 2 - side / 2, 0), vh - side);
  const c = document.createElement('canvas');
  c.width = c.height = Math.min(CROP_PX, Math.round(side));
  c.getContext('2d').drawImage(video, x0, y0, side, side, 0, 0, c.width, c.height);
  const slot = { phase: key, blob: null };
  slot.ready = new Promise(res => c.toBlob(b => { slot.blob = b; res(); }, 'image/jpeg', 0.92));
  shots.push(slot);
}

// ───────────── save locally first, then upload (never lose a capture) ─────────────

const OB_DB = 'rufusident-outbox', OB_STORE = 'captures';
let obDb;
function ob(mode, fn) {
  obDb ||= new Promise((res, rej) => {
    const r = indexedDB.open(OB_DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(OB_STORE, { keyPath: 'id' });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  return obDb.then(d => new Promise((res, rej) => {
    const t = d.transaction(OB_STORE, mode);
    const out = fn(t.objectStore(OB_STORE));
    t.oncomplete = () => res(out?.result);
    t.onerror = () => rej(t.error);
  }));
}
const obPut = item => ob('readwrite', s => s.put(item));
const obDel = id => ob('readwrite', s => s.delete(id));
const obAll = () => ob('readonly', s => s.getAll()).then(list => (list || []).sort((a, b) => a.created - b.created));

async function upload() {
  setMode('upload');
  say('SAVING…', 'ok');
  await Promise.all(shots.map(x => x.ready));                             // every frame fully encoded
  const item = {
    id: crypto.randomUUID(), created: Date.now(), attempts: 0, error: null,
    student: { id: current.id, name: current.name, cls: current.cls, section: current.section },
    shots: shots.filter(x => x.blob).map(x => ({ phase: x.phase, blob: x.blob })),
  };
  try {
    await obPut(item);                                                    // on this laptop's disk before anything else
  } catch (e) {
    say(`COULD NOT SAVE ON THIS LAPTOP · ${e.message.slice(0, 40)} · PLEASE RE-TAKE`, 'bad');
    setMode('detected');
    return;
  }
  outbox.push(item);
  await upsertStudents([{ ...item.student, consent: 'yes' }]).catch(() => {});   // roster + consent, even if offline
  sessionCount++;
  $('#stDone').textContent = sessionCount;
  say(`SAVED ✓ ${item.student.name.split(' ')[0].toUpperCase()} · NEXT PLEASE`, 'ok');
  setMode('done');
  renderOutbox();
  flush();
}

// Upload everything still on this laptop. A capture is removed only once the
// backend has written it to its own disk; retries are safe (same capture id).
async function flush() {
  clearTimeout(flushTimer);
  if (flushing || !running) return;
  flushing = true;
  let stalled = false;
  try {
    for (const item of [...outbox]) {
      if (item.error && item.permanent) continue;                        // needs the operator (see panel)
      const fd = new FormData();
      fd.append('capture_id', item.id);
      fd.append('student_id', item.student.id);
      fd.append('name', item.student.name);
      fd.append('cls', item.student.cls);
      fd.append('section', item.student.section);
      fd.append('consent', 'yes');
      item.shots.forEach((x, i) => fd.append('frames', x.blob, `${String(i).padStart(3, '0')}_${x.phase}.jpg`));
      let res, detail = '';
      try {
        res = await fetch(base() + '/enrol/submit', { method: 'POST', body: fd, signal: AbortSignal.timeout(60000) });
        if (!res.ok) detail = (await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`;
      } catch (e) {
        detail = e.name === 'TimeoutError' ? 'timed out' : 'backend not reachable';
      }
      if (res?.ok) {
        await obDel(item.id);
        outbox = outbox.filter(x => x.id !== item.id);
        backoff = 5000;
        continue;
      }
      item.attempts++;
      item.error = detail;
      item.permanent = res?.status === 400;                              // bad data: retrying won't help
      await obPut(item).catch(() => {});
      if (!item.permanent) { stalled = true; break; }                     // backend/network problem: try again later
    }
  } finally {
    flushing = false;
    renderOutbox();
    if (stalled || outbox.some(x => !x.permanent)) {
      backoff = stalled ? Math.min(backoff * 2, 60000) : 5000;
      flushTimer = setTimeout(flush, backoff);
    }
    refresh();
  }
}

function renderOutbox() {
  const el = $('#stOutbox');
  const waiting = outbox.filter(x => !x.permanent), stuck = outbox.filter(x => x.permanent);
  el.hidden = !outbox.length;
  $('#stOutboxCount').textContent = outbox.length ? `${outbox.length}` : '';
  $('#stOutboxList').innerHTML =
    waiting.map(x => `<li><b>${esc(x.student.name)}</b> ${esc(`${x.student.cls}-${x.student.section}`)}<span>${x.error ? `waiting to upload · ${esc(x.error)} · retrying automatically` : 'uploading…'}</span></li>`).join('') +
    stuck.map(x => `<li class="bad"><b>${esc(x.student.name)}</b><span>${esc(x.error)}</span><button data-discard="${esc(x.id)}">DISCARD</button></li>`).join('');
}

// ───────────── side panels ─────────────

async function refresh() {
  try {
    status = await (await fetch(base() + '/enrol/status', { signal: AbortSignal.timeout(4000) })).json();
    setStatus(status.calibrating ? 'RECALIBRATING THRESHOLDS' : status.queue ? `PROCESSING ${status.queue}` : 'READY', status.queue ? 'warn' : 'ok');
  } catch {
    setStatus('BACKEND NOT RUNNING', 'bad');
    return;
  }
  renderJobs();
  renderClasses();
  if (mode === 'detected') validate();
}

const isEnrolled = id => !!status.enrolled[id];

function renderJobs() {
  $('#stQueue').textContent = status.queue ? `${status.queue} WAITING` : 'IDLE';
  const icon = { queued: '…', processing: '⟳', done: '✓', failed: '✕' };
  $('#stJobs').innerHTML = status.jobs.length ? status.jobs.map(j => `
    <li class="${j.state}${j.warnings?.length ? ' warn' : ''}">
      <i>${icon[j.state] || '?'}</i>
      <div><b>${esc(j.student.name || j.student.id)}</b> ${esc(j.student.cls ? `${j.student.cls}-${j.student.section}` : '')}
        <span>${esc(j.message || j.state)}</span>
        ${j.reprocessable ? `<button data-reprocess="${esc(j.id)}">REPROCESS</button>` : j.state === 'failed' || j.warnings?.length ? `<button data-retake="${esc(j.student.id)}">RE-TAKE</button>` : ''}
      </div>
    </li>`).join('') : '<li class="dim">Nothing yet.</li>';
}

function renderClasses() {
  const classes = state.classes;
  const all = classes.map(k => {
    const list = studentsIn(k);
    return { k, total: list.length, done: list.filter(s => isEnrolled(s.id)).length };
  });
  const T = all.reduce((a, c) => a + c.total, 0), D = all.reduce((a, c) => a + c.done, 0);
  $('#stTotal').textContent = T ? `${D} / ${T}` : '';
  $('#stClasses').innerHTML = all.length ? all.map(c => `
    <button data-cls="${esc(c.k)}" class="${c.k === selectedClass ? 'on' : ''}">
      <b>${esc(c.k)}</b><i style="--w:${c.total ? (c.done / c.total) * 100 : 0}%"></i><em>${c.done}/${c.total}</em>
    </button>`).join('') : '<p class="dim">Import the roster (Settings → Data) to track classes.</p>';
  if (selectedClass) renderMissing();
}

function renderMissing() {
  const list = studentsIn(selectedClass).filter(s => !isEnrolled(s.id));
  $('#stMissingTitle').textContent = `${selectedClass} · ${list.length}`;
  $('#stMissing').innerHTML = list.length ? list.map(s => `
    <li><span><b>${esc(s.name)}</b> · ${esc(s.roll || s.id)}${hasConsent(s) || status.students[s.id]?.consent ? '' : ' · <em style="color:#ffb648;font-style:normal">no consent</em>'}</span>
      <button data-pick="${esc(s.id)}">ENROL</button></li>`).join('') : '<li class="dim">Everyone in this class is enrolled ✓</li>';
}

// End-of-day check: every roster student and where they stand.
function downloadReport() {
  if (!window.XLSX) return toast('Excel engine not loaded', { kind: 'warn' });
  const rows = state.classes.flatMap(k => studentsIn(k).map(s => {
    const m = status.students[s.id] || {};
    const enrolled = isEnrolled(s.id);
    const waiting = outbox.some(x => x.student.id === s.id);
    return {
      Class: k, Roll: Number(s.roll) || s.roll || '', 'Admission No': s.id, Name: s.name,
      Status: enrolled ? (m.verified === false ? 'Enrolled — re-take advised' : 'Enrolled') : waiting ? 'Captured, waiting to upload' : m.last_state === 'failed' ? 'FAILED — re-take' : 'Not enrolled',
      Photos: status.enrolled[s.id]?.photos ?? '', Verified: m.verified === true ? 'Yes' : m.verified === false ? 'No' : '',
      Consent: hasConsent(s) || m.consent ? 'Yes' : 'NO', Notes: (m.warnings || []).join('; ') || (m.last_state === 'failed' ? m.last_message : ''),
      'Enrolled at': m.enrolled_at || '',
    };
  }));
  const ws = XLSX.utils.json_to_sheet(rows);
  ws['!cols'] = [{ wch: 8 }, { wch: 6 }, { wch: 14 }, { wch: 26 }, { wch: 28 }, { wch: 7 }, { wch: 8 }, { wch: 8 }, { wch: 50 }, { wch: 20 }];
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Enrolment');
  const todo = rows.filter(r => !r.Status.startsWith('Enrolled') || r.Verified === 'No' || r.Consent === 'NO');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(todo.length ? todo : [{ Note: 'Everyone is enrolled, verified and has consent ✓' }]), 'To do');
  XLSX.writeFile(wb, `GHS_Enrolment_${new Date().toISOString().slice(0, 10)}.xlsx`);
}

// ───────────── wiring (once) ─────────────

export function wireStation() {
  const results = $('#stResults');
  $('#stSearch').addEventListener('input', e => {
    const list = search(e.target.value, 6);
    results.innerHTML = list.map(s => `<button data-pick="${esc(s.id)}"><b>${esc(s.name)}</b><span>${esc(classKey(s))} · ${esc(s.id)}${isEnrolled(s.id) ? ' · enrolled' : ''}</span></button>`).join('');
  });
  $('#stSearch').addEventListener('keydown', e => { if (e.key === 'Enter') results.querySelector('button')?.click(); });
  for (const id of ['stId', 'stName', 'stCls', 'stSec']) $('#' + id).addEventListener('input', validate);
  $('#stConsent').addEventListener('change', validate);
  $('#stGo').addEventListener('click', () => { if (!$('#stGo').disabled) startCapture(); });
  $('#stReport').addEventListener('click', downloadReport);
  $('#station').addEventListener('click', e => {
    const pick = e.target.closest('[data-pick]');
    if (pick) { const s = state.byId.get(pick.dataset.pick); if (s) fillForm(s); return; }
    const cls = e.target.closest('[data-cls]');
    if (cls) { selectedClass = cls.dataset.cls; renderClasses(); return; }
    const rep = e.target.closest('[data-reprocess]');
    if (rep) {
      fetch(base() + '/enrol/reprocess', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job: rep.dataset.reprocess }) })
        .then(r => r.json()).then(() => { toast('Re-queued'); refresh(); }).catch(() => toast('Backend not reachable', { kind: 'warn' }));
      return;
    }
    const discard = e.target.closest('[data-discard]');
    if (discard && confirm('Discard this capture? The student will need to be captured again.')) {
      obDel(discard.dataset.discard).then(() => { outbox = outbox.filter(x => x.id !== discard.dataset.discard); renderOutbox(); });
      return;
    }
    const retake = e.target.closest('[data-retake]');
    if (retake) {
      const s = state.byId.get(retake.dataset.retake) || status.jobs.find(j => j.student.id === retake.dataset.retake)?.student;
      if (s) fillForm({ ...s, section: s.section, cls: s.cls });
    }
  });
  addEventListener('keydown', e => {
    if ($('#station').hidden) return;
    if (e.key === 'Enter' && mode === 'detected' && document.activeElement?.id !== 'stSearch' && !$('#stGo').disabled) startCapture();
  });
  // Closing the station while captures are still waiting is safe (they're on disk), but say so.
  addEventListener('beforeunload', e => { if (outbox.length) { e.preventDefault(); e.returnValue = ''; } });
}
