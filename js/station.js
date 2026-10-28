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
let pendingRetry = null;        // frames kept if an upload failed

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
  shots.push(slot);
  c.toBlob(b => { slot.blob = b; }, 'image/jpeg', 0.92);
}

async function upload(retry = null) {
  setMode('upload');
  say('UPLOADING…', 'ok');
  const job = retry || { student: current, shots };
  await new Promise(r => setTimeout(r, 150));                             // let the last JPEGs finish
  const fd = new FormData();
  fd.append('student_id', job.student.id);
  fd.append('name', job.student.name);
  fd.append('cls', job.student.cls);
  fd.append('section', job.student.section);
  fd.append('consent', 'yes');
  job.shots.filter(x => x.blob).forEach((x, i) => fd.append('frames', x.blob, `${String(i).padStart(3, '0')}_${x.phase}.jpg`));
  try {
    const res = await fetch(base() + '/enrol/submit', { method: 'POST', body: fd, signal: AbortSignal.timeout(30000) });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
    pendingRetry = null;
    await upsertStudents([job.student]);                                  // new students join the roster; consent recorded
    sessionCount++;
    $('#stDone').textContent = sessionCount;
    say(`UPLOADED ✓ ${job.student.name.split(' ')[0].toUpperCase()} · NEXT PLEASE`, 'ok');
    setMode('done');
    refresh();
  } catch (e) {
    pendingRetry = job;
    say(`UPLOAD FAILED · ${e.message.slice(0, 50)} · PRESS R TO RETRY`, 'bad');
    mode = 'done';
  }
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
        ${j.state === 'failed' || j.warnings?.length ? `<button data-retake="${esc(j.student.id)}">RE-TAKE</button>` : ''}
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
  $('#station').addEventListener('click', e => {
    const pick = e.target.closest('[data-pick]');
    if (pick) { const s = state.byId.get(pick.dataset.pick); if (s) fillForm(s); return; }
    const cls = e.target.closest('[data-cls]');
    if (cls) { selectedClass = cls.dataset.cls; renderClasses(); return; }
    const retake = e.target.closest('[data-retake]');
    if (retake) {
      const s = state.byId.get(retake.dataset.retake) || status.jobs.find(j => j.student.id === retake.dataset.retake)?.student;
      if (s) fillForm({ ...s, section: s.section, cls: s.cls });
    }
  });
  addEventListener('keydown', e => {
    if ($('#station').hidden) return;
    if (e.key === 'Enter' && mode === 'detected' && document.activeElement?.id !== 'stSearch' && !$('#stGo').disabled) startCapture();
    if ((e.key === 'r' || e.key === 'R') && pendingRetry && document.activeElement?.tagName !== 'INPUT') upload(pendingRetry);
  });
}
