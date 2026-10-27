// Gate camera: live face tracking (MediaPipe, runs offline on this laptop),
// quality checks, and hand-off of good face crops to the recognition backend.

import { state, mark, undo, recordFor, arrivalStatus, daySummary, getDay, search, classKey, studentStats, onChange, STATUS } from './data.js';
import { $, esc, avatar, toast, pct } from './ui.js';

const MIN_FACE_PX = 90;       // face width in camera pixels before we try to identify (ArcFace uses 112px)
const MAX_YAW = 0.18;         // how far the head may turn (0 = straight on)
const LOW_LIGHT = 55, GLARE = 215;
const REQUEST_GAP_MS = 40;     // up to ~25 frames a second to the backend
const MAX_INFLIGHT = 2;        // keep the backend busy while the next frame is cropped
const RETRY_MS = 3000;         // after 'unknown'/'review'/'spoof', try again with a fresh track
const CROP_PX = 512;

let vision, engine, enginePromise, delegate = '', TESS = [];

export function loadEngine() {
  enginePromise ||= (async () => {
    vision = await import('../vendor/mediapipe/vision_bundle.mjs');
    const files = await vision.FilesetResolver.forVisionTasks(new URL('../vendor/mediapipe/wasm', import.meta.url).href);
    const opts = d => ({
      baseOptions: { modelAssetPath: new URL('../vendor/models/face_landmarker.task', import.meta.url).href, delegate: d },
      runningMode: 'VIDEO',
      numFaces: 4,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      engine = await vision.FaceLandmarker.createFromOptions(files, opts('GPU'));
      delegate = 'GPU · WebGL';
    } catch {
      engine = await vision.FaceLandmarker.createFromOptions(files, opts('CPU'));
      delegate = 'CPU · WASM';
    }
    TESS = vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION;
    return `face landmarker ready · ${delegate}`;
  })();
  enginePromise.catch(() => { enginePromise = null; });
  return enginePromise;
}

function constraints() {
  const id = state.settings.cameraId;
  return {
    audio: false,
    video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
  };
}

export async function probeCamera() {
  const s = await navigator.mediaDevices.getUserMedia(constraints());
  const track = s.getVideoTracks()[0];
  const { width, height } = track.getSettings();
  const label = track.label || 'camera';
  s.getTracks().forEach(t => t.stop());
  const text = `${label} · ${width}×${height}`;
  return height < 720 ? { warn: `${text} · below 720p, recognition will suffer` } : text;
}

export async function listCameras() {
  const devs = await navigator.mediaDevices.enumerateDevices();
  return devs.filter(d => d.kind === 'videoinput');
}

// ───────────── HUD runtime state ─────────────

let stream, raf, clockTimer, unsub;
let running = false;
let lastVT = -1, lastT = 0, fps = 0, frame = 0;
let light = 0;
const lightHist = [];
let inflight = 0, lastReq = 0;
const review = [];
let video, canvas, ctx;

export async function openCamera() {
  video = $('#hudVideo'); canvas = $('#hudCanvas'); ctx = canvas.getContext('2d');
  $('#hud').hidden = false;
  document.body.classList.add('hud-on');
  $('#hudGate').textContent = `GATE · ${state.settings.gateName.toUpperCase()}`;
  tickClock(); clockTimer = setInterval(tickClock, 1000);
  unsub = onChange(() => { renderToday(); renderLog(); });
  renderToday(); renderLog(); renderReview();
  setStatus('STARTING CAMERA', 'warn');

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints());
    video.srcObject = stream;
    await video.play();
    const { width, height, frameRate } = stream.getVideoTracks()[0].getSettings();
    $('#mRes').textContent = `${width}×${height} @ ${Math.round(frameRate || 0)}`;
  } catch (e) {
    setStatus('CAMERA ERROR', 'bad');
    prompt(`CAMERA UNAVAILABLE · ${e.message}`, 'bad');
    return;
  }

  setStatus('LOADING FACE ENGINE', 'warn');
  try { await loadEngine(); } catch (e) {
    setStatus('FACE ENGINE FAILED · MANUAL MODE', 'bad');
    prompt('USE MANUAL MARK BELOW', 'bad');
    return;
  }
  $('#mEngine').textContent = delegate;
  $('#mMatcher').textContent = state.settings.backendUrl ? 'BACKEND' : 'OFFLINE';
  $('#mLive').textContent = '—'; $('#mVerify').textContent = '—';
  setStatus(state.settings.backendUrl ? 'LIVE · SCANNING' : 'LIVE · MANUAL MODE', 'ok');
  running = true;
  resize();
  addEventListener('resize', resize);
  loop();
}

export function closeCamera() {
  running = false;
  cancelAnimationFrame(raf);
  clearInterval(clockTimer);
  unsub?.();
  removeEventListener('resize', resize);
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  $('#hud').hidden = true;
  document.body.classList.remove('hud-on');
}

function tickClock() {
  const d = new Date();
  $('#hudClock').textContent = d.toLocaleTimeString('en-GB');
  $('#hudDate').textContent = d.toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();
}

function setStatus(text, kind) {
  const el = $('#hudStatus');
  el.textContent = text;
  el.className = `hud-status ${kind}`;
}

function prompt(text, kind = '') {
  const el = $('#hudPrompt');
  if (el.textContent !== text) el.textContent = text;
  el.className = `hud-prompt ${kind}`;
}

function resize() {
  const dpr = devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ───────────── frame loop ─────────────

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  const now = performance.now();
  if (lastT) fps = fps ? fps * 0.9 + (1000 / (now - lastT)) * 0.1 : 1000 / (now - lastT);
  lastT = now;

  const z = zone();
  let res;
  try { res = engine.detectForVideo(zoneImage(z), now); } catch { return; }
  const faces = (res.faceLandmarks || []).map(lm => describe(lm, z));
  faces.sort((a, b) => b.w - a.w);
  if (++frame % 6 === 0) sampleLight(faces[0]);
  if (frame % 10 === 0) $('#mFps').textContent = fps.toFixed(1);

  $('#mFaces').textContent = faces.length;
  evaluate(faces);
  draw(faces);
}

// The detector only looks at the gate lane: a centre square of the frame.
// The browser's face model is short-range, so cropping makes a student 2 m
// away look ~1.8x bigger to it.
const ZONES = { square: 1, tight: 0.6 };

function zone() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const k = ZONES[state.settings.zone];
  if (!k) return { x: 0, y: 0, w: vw, h: vh, full: true };
  const side = Math.min(vw, vh) * k;
  return { x: (vw - side) / 2, y: (vh - side) / 2, w: side, h: side };
}

let zoneCanvas;
function zoneImage(z) {
  if (z.full) return video;
  zoneCanvas ||= document.createElement('canvas');
  const px = Math.round(Math.min(z.w, 960));
  if (zoneCanvas.width !== px) zoneCanvas.width = zoneCanvas.height = px;
  zoneCanvas.getContext('2d').drawImage(video, z.x, z.y, z.w, z.h, 0, 0, px, px);
  return zoneCanvas;
}

// Landmarks converted to camera pixels.
function describe(lm, z) {
  const pts = lm.map(p => ({ x: z.x + p.x * z.w, y: z.y + p.y * z.h }));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) { if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x; if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y; }
  const l = pts[33], r = pts[263], nose = pts[1];
  const eyeDist = Math.hypot(r.x - l.x, r.y - l.y) || 1e-6;
  const yaw = (nose.x - (l.x + r.x) / 2) / eyeDist;
  return { lm: pts, x: x0, y: y0, w: x1 - x0, h: y1 - y0, yaw };
}

// Light is measured on the face when there is one (a dark corridor behind a
// well-lit student is fine), otherwise on the whole frame.
function sampleLight(face) {
  const c = sampleLight.c ||= Object.assign(document.createElement('canvas'), { width: 64, height: 36 });
  const g = c.getContext('2d', { willReadFrequently: true });
  if (face) g.drawImage(video, face.x + face.w * 0.2, face.y + face.h * 0.15, face.w * 0.6, face.h * 0.7, 0, 0, 64, 36);
  else g.drawImage(video, 0, 0, 64, 36);
  const d = g.getImageData(0, 0, 64, 36).data;
  let sum = 0;
  for (let i = 0; i < d.length; i += 4) sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  light = sum / (d.length / 4);
  lightHist.push(light);
  if (lightHist.length > 60) lightHist.shift();
  $('#mLight').textContent = `${Math.round((light / 255) * 100)}%`;
  drawLightGraph();
}

function drawLightGraph() {
  const g = $('#lightGraph').getContext('2d');
  const W = 260, H = 70;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = 'rgba(255,182,72,.35)'; g.setLineDash([3, 4]); g.lineWidth = 1;
  for (const t of [LOW_LIGHT, GLARE]) { const y = H - (t / 255) * H; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  g.setLineDash([]);
  if (lightHist.length < 2) return;
  const step = W / 59;
  g.beginPath();
  lightHist.forEach((v, i) => { const x = i * step, y = H - (v / 255) * H; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = '#4fd8ff'; g.lineWidth = 1.6; g.stroke();
  g.lineTo((lightHist.length - 1) * step, H); g.lineTo(0, H); g.closePath();
  g.fillStyle = 'rgba(79,216,255,.12)'; g.fill();
}

// ───────────── drawing ─────────────

let banner = null; // { text, kind, until }

function mapper() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(cw / vw, ch / vh);
  const ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  // The video is shown mirrored, so flip x.
  return { s, pt: (x, y) => [cw - (x * s + ox), y * s + oy] };
}

function drawZone(pt) {
  const z = zone();
  if (z.full) return;
  const [ax, ay] = pt(z.x + z.w, z.y), [bx, by] = pt(z.x, z.y + z.h);
  const c = (bx - ax) * 0.06;
  ctx.strokeStyle = 'rgba(79,216,255,.35)'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
  ctx.beginPath();
  for (const [x, y, dx, dy] of [[ax, ay, 1, 1], [bx, ay, -1, 1], [bx, by, -1, -1], [ax, by, 1, -1]]) {
    ctx.moveTo(x, y + dy * c); ctx.lineTo(x, y); ctx.lineTo(x + dx * c, y);
  }
  ctx.stroke();
}

function draw(faces) {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.clearRect(0, 0, cw, ch);
  const { s, pt } = mapper();
  drawZone(pt);
  if (!faces.length) return;

  faces.forEach((f, i) => {
    const primary = i === 0;
    const [ax, ay] = pt(f.x + f.w, f.y);
    const w = f.w * s, h = f.h * s;
    const pad = w * 0.12;
    const bx = ax - pad, by = ay - pad * 1.6, bw = w + pad * 2, bh = h + pad * 2.4;
    const color = primary ? (f.quality === 'ok' ? '#3dffa2' : '#ffb648') : 'rgba(255,255,255,.45)';

    if (primary && TESS.length) {
      ctx.strokeStyle = 'rgba(61,255,162,.16)';
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (const { start, end } of TESS) {
        const a = f.lm[start], b = f.lm[end];
        const [x1, y1] = pt(a.x, a.y), [x2, y2] = pt(b.x, b.y);
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      }
      ctx.stroke();
      ctx.fillStyle = '#ff5ad1';
      for (const k of [33, 263, 1, 61, 291, 152, 10]) {
        const [x, y] = pt(f.lm[k].x, f.lm[k].y);
        ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Corner brackets.
    const c = Math.min(bw, bh) * 0.2;
    ctx.strokeStyle = color; ctx.lineWidth = primary ? 3 : 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, by + c); ctx.lineTo(bx, by); ctx.lineTo(bx + c, by);
    ctx.moveTo(bx + bw - c, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + c);
    ctx.moveTo(bx + bw, by + bh - c); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - c, by + bh);
    ctx.moveTo(bx + c, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - c);
    ctx.stroke();

    if (primary) {
      const label = banner && performance.now() < banner.until ? banner.text : f.label;
      ctx.font = '600 13px "JetBrains Mono", monospace';
      const tw = ctx.measureText(label).width + 16;
      ctx.fillStyle = 'rgba(2,14,9,.78)';
      ctx.fillRect(bx, by - 30, tw, 22);
      ctx.fillStyle = color;
      ctx.fillRect(bx, by - 30, 3, 22);
      ctx.fillText(label, bx + 9, by - 14);
    }
  });
}

// ───────────── quality + recognition ─────────────

// One track = one person walking up to the camera. Every good frame of the
// track is streamed to the backend, which fuses them and answers 'match'
// only when several frames agree (see backend/fusion.py).
let track = null;

function trackFor(f) {
  const now = performance.now();
  const cx = f.x + f.w / 2;
  const fresh = !track
    || now - track.seen > 1500                         // face left the frame
    || Math.abs(cx - track.cx) > f.w * 0.6             // a different person stepped in
    || f.w < track.w * 0.5
    || (track.decided && track.decided !== 'match' && now - track.decidedAt > RETRY_MS);
  if (fresh) track = { id: Math.random().toString(36).slice(2, 10), cx, w: f.w, seen: now, frames: 0, need: 3, decided: null, decidedAt: 0 };
  track.cx = cx; track.w = f.w; track.seen = now;
  return track;
}

function evaluate(faces) {
  if (!faces.length) {
    $('#mSize').textContent = '—'; $('#mPose').textContent = '—'; $('#mQuality').textContent = 'NO FACE';
    if (!(banner && performance.now() < banner.until)) prompt('STAND IN FRONT OF THE CAMERA');
    return;
  }
  const f = faces[0];
  const t = trackFor(f);
  const turned = Math.abs(f.yaw) > MAX_YAW;
  let q = 'ok', msg;
  if (f.w < MIN_FACE_PX) { q = 'far'; msg = 'STEP CLOSER'; }
  else if (turned) { q = 'pose'; msg = 'LOOK STRAIGHT AT THE CAMERA'; }
  else if (light < LOW_LIGHT) { q = 'dark'; msg = 'TOO DARK · CHECK GATE LIGHTING'; }
  else if (light > GLARE) { q = 'glare'; msg = 'TOO BRIGHT · CAMERA FACING LIGHT'; }
  f.quality = q;
  const backend = !!state.settings.backendUrl;
  f.label = q !== 'ok' ? msg
    : !backend ? 'FACE OK'
    : t.decided === 'match' && t.name ? t.name.toUpperCase()
    : t.decided ? t.decided.toUpperCase()
    : `VERIFYING ${Math.min(t.frames, t.need)}/${t.need}`;

  $('#mSize').textContent = `${Math.round(f.w)} px`;
  $('#mPose').textContent = turned ? `TURNED ${f.yaw > 0 ? 'L' : 'R'}` : 'FRONTAL';
  $('#mQuality').textContent = q === 'ok' ? 'GOOD' : q.toUpperCase();
  $('#mVerify').textContent = backend ? (t.decided ? t.decided.toUpperCase() : `${t.frames}/${t.need} FRAMES`) : '—';

  if (banner && performance.now() < banner.until) { prompt(banner.text, banner.kind); return; }
  if (q !== 'ok') { prompt(msg, 'warn'); return; }
  if (faces.length > 1 && faces[1].w > MIN_FACE_PX) prompt('ONE STUDENT AT A TIME, PLEASE', 'warn');
  else if (!backend) prompt('FACE OK · RECOGNITION OFFLINE · USE MANUAL MARK', 'warn');
  else if (!t.decided) prompt(t.frames ? `VERIFYING ${Math.min(t.frames, t.need)}/${t.need} · HOLD STILL` : 'HOLD STILL…', 'ok');
  else if (t.decided === 'match') prompt('MARKED ✓ · PLEASE MOVE ON', 'ok');

  if (backend && !t.decided && inflight < MAX_INFLIGHT && performance.now() - lastReq > REQUEST_GAP_MS) recognise(f, t);
}

// A square crop with plenty of context around the face: the anti-spoof
// models look at the area around the face (up to 4x its size).
function cropFace(f) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(Math.max(f.w, f.h) * 3.2, vw, vh);
  const cx = f.x + f.w / 2, cy = f.y + f.h / 2;
  const x0 = Math.min(Math.max(cx - side / 2, 0), vw - side);
  const y0 = Math.min(Math.max(cy - side / 2, 0), vh - side);
  const c = document.createElement('canvas');
  c.width = c.height = CROP_PX;
  c.getContext('2d').drawImage(video, x0, y0, side, side, 0, 0, CROP_PX, CROP_PX);
  return c;
}

async function recognise(f, t) {
  inflight++;
  lastReq = performance.now();
  try {
    const crop = cropFace(f);
    const blob = await new Promise(r => crop.toBlob(r, 'image/jpeg', 0.9));
    const fd = new FormData();
    fd.append('image', blob, 'face.jpg');
    fd.append('track', t.id);
    fd.append('gate', state.settings.gateName);
    const res = await fetch(state.settings.backendUrl.replace(/\/$/, '') + '/recognize', { method: 'POST', body: fd, signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    $('#mMatcher').textContent = `BACKEND · ${j.ms ?? '?'} ms`;
    if (j.live != null) $('#mLive').textContent = `${Math.round(j.live * 100)}%`;
    if (t !== track || t.decided) return;              // an answer for someone who already left
    if (j.frames) { t.frames = j.frames; t.need = j.need_frames || t.need; }
    if (['match', 'review', 'unknown', 'spoof'].includes(j.status)) {
      t.decided = j.status;
      t.decidedAt = performance.now();
      await handleResult(j, t, crop);
    }
  } catch (e) {
    $('#mMatcher').textContent = `BACKEND · ${e.name === 'TimeoutError' ? 'TIMEOUT' : 'ERROR'}`;
  } finally {
    inflight--;
  }
}

async function handleResult(j, t, crop) {
  const img = () => crop.toDataURL('image/jpeg', 0.8);
  const s = j.student_id && state.byId.get(String(j.student_id));

  if (j.status === 'spoof') { flash('LIVE FACE NEEDED · PHOTO OR SCREEN DETECTED', 'bad', 2500); buzz(); return; }
  if (j.status === 'unknown') { flash('NOT RECOGNISED · PLEASE SEE THE TEACHER', 'bad', 2000); return; }
  if (!s) { flash(`ID ${j.student_id} IS ENROLLED BUT NOT IN THE ROSTER`, 'bad', 2500); return; }

  if (j.status === 'review') {
    if (!review.some(r => r.id === s.id)) review.unshift({ id: s.id, conf: j.confidence ?? 0, img: img(), reason: j.reason || '' });
    renderReview();
    flash('PLEASE WAIT · TEACHER WILL CONFIRM', 'warn', 1800);
    return;
  }

  t.name = s.name;
  const existing = recordFor(s.id);
  if (existing && existing.s !== 'A') {
    showIdent(s, existing, j.confidence, img(), true);
    flash(`${s.name.toUpperCase()} · ALREADY MARKED ${existing.t}`, 'ok', 2000);
    return;
  }
  const status = arrivalStatus();
  await mark([s.id], status, { source: 'cam', confidence: j.confidence });
  showIdent(s, recordFor(s.id), j.confidence, img(), false);
  flash(`WELCOME, ${s.name.split(' ')[0].toUpperCase()} · ${STATUS[status].toUpperCase()}`, 'ok', 2200);
  beep();
}

function flash(text, kind, ms) {
  banner = { text, kind, until: performance.now() + ms };
  prompt(text, kind);
}

let audio;
function tone(freq, secs, vol = 0.08) {
  try {
    audio ||= new AudioContext();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq; g.gain.value = vol;
    o.connect(g).connect(audio.destination);
    o.start(); o.stop(audio.currentTime + secs);
  } catch {}
}
const beep = () => tone(880, 0.12);
const buzz = () => tone(180, 0.35, 0.12);

// ───────────── side panels ─────────────

function showIdent(s, rec, conf, img, already) {
  const st = studentStats(s.id);
  $('#ident').innerHTML = `
    <div class="ident-top">
      ${img ? `<img class="ident-img" src="${img}" alt="">` : avatar(s, 'ident-img')}
      <div>
        <b>${esc(s.name)}</b>
        <span>${esc(classKey(s))} · ROLL ${esc(s.roll || '—')} · ${esc(s.id)}</span>
        <em class="badge ${rec.s}">${already ? 'ALREADY ' : ''}${STATUS[rec.s].toUpperCase()} · ${esc(rec.t)}</em>
      </div>
    </div>
    ${conf != null ? `<div class="conf"><span>MATCH</span><i style="--w:${Math.round(conf * 100)}%"></i><b>${(conf * 100).toFixed(1)}%</b></div>` : ''}
    <div class="conf"><span>ATTEND.</span><i style="--w:${st?.pct ?? 0}%"></i><b>${st?.pct != null ? st.pct + '%' : '—'}</b></div>`;
}

function renderToday() {
  if ($('#hud').hidden) return;
  const d = daySummary();
  const here = d.P + d.L;
  const p = pct(here, d.total);
  const C = 2 * Math.PI * 52;
  const ring = $('#todayRing');
  ring.style.strokeDasharray = C;
  ring.style.strokeDashoffset = C * (1 - p / 100);
  $('#todayPct').textContent = `${p}%`;
  $('#todayCount').textContent = `${here.toLocaleString('en-IN')} / ${d.total.toLocaleString('en-IN')}`;
  $('#tP').textContent = d.P; $('#tL').textContent = d.L; $('#tA').textContent = d.A;
}

function renderLog() {
  if ($('#hud').hidden) return;
  const day = getDay();
  const rows = Object.entries(day)
    .filter(([, r]) => r.s !== 'A')
    .sort((a, b) => b[1].t.localeCompare(a[1].t))
    .slice(0, 12);
  $('#gateLog').innerHTML = rows.length ? rows.map(([id, r]) => {
    const s = state.byId.get(id);
    return s ? `<li><time>${esc(r.t)}</time><b>${esc(s.name)}</b><span>${esc(classKey(s))}</span><em class="badge ${r.s}">${r.s}</em></li>` : '';
  }).join('') : '<li class="dim">No arrivals yet today.</li>';
}

function renderReview() {
  $('#reviewPanel').hidden = !review.length;
  $('#reviewCount').textContent = review.length ? `(${review.length})` : '';
  $('#reviewList').innerHTML = review.map((r, i) => {
    const s = state.byId.get(r.id);
    return `<div class="rv">
      <img src="${r.img}" alt="">
      <div><b>${esc(s.name)}</b><span>${esc(classKey(s))} · ${(r.conf * 100).toFixed(0)}%${r.reason ? ` · ${esc(r.reason)}` : ''}</span></div>
      <button data-rv="ok" data-i="${i}" title="Confirm">✓</button>
      <button data-rv="no" data-i="${i}" title="Not this student">✕</button>
    </div>`;
  }).join('');
}

// ───────────── wiring (once) ─────────────

export function wireCamera() {
  $('#reviewList').addEventListener('click', async e => {
    const b = e.target.closest('button[data-rv]');
    if (!b) return;
    const [r] = review.splice(Number(b.dataset.i), 1);
    if (b.dataset.rv === 'ok') {
      const s = state.byId.get(r.id);
      await mark([r.id], arrivalStatus(), { source: 'cam', confidence: r.conf });
      showIdent(s, recordFor(r.id), r.conf, r.img, false);
      toast(`${s.name} marked ${STATUS[recordFor(r.id).s].toLowerCase()}`);
    }
    renderReview();
  });

  const input = $('#manualInput'), out = $('#manualResults');
  input.addEventListener('input', () => {
    const list = search(input.value, 6);
    out.innerHTML = list.map(s => {
      const rec = recordFor(s.id);
      return `<button data-id="${esc(s.id)}">${avatar(s, 'mini')}<b>${esc(s.name)}</b><span>${esc(classKey(s))} · ${esc(s.id)}</span>${rec ? `<em class="badge ${rec.s}">${rec.s} ${esc(rec.t)}</em>` : ''}</button>`;
    }).join('');
  });
  out.addEventListener('click', async e => {
    const b = e.target.closest('button[data-id]');
    if (!b) return;
    const s = state.byId.get(b.dataset.id);
    const u = await mark([s.id], arrivalStatus(), { source: 'gate-manual' });
    showIdent(s, recordFor(s.id), null, null, false);
    toast(`${s.name} marked ${STATUS[recordFor(s.id).s].toLowerCase()} (manual)`, { action: 'Undo', onAction: () => undo(u) });
    input.value = ''; out.innerHTML = '';
    input.blur();
  });
}
