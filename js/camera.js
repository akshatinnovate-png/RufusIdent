// Gate camera. Two things run side by side:
//  * MediaPipe in the browser (25 fps): face mesh, pose and light hints for
//    students close to the camera.
//  * The recognition backend: whole frames are streamed to it; it finds every
//    face (near or 5-6 m away), tracks each person and says who they are.

import { state, mark, undo, recordFor, arrivalStatus, daySummary, getDay, search, classKey, studentStats, onChange, STATUS } from './data.js';
import { $, esc, avatar, toast, pct } from './ui.js';

const MAX_YAW = 0.18;          // how far the head may turn (0 = straight on)
const LOW_LIGHT = 55, GLARE = 215;
const FACE_M = 0.15;           // average face width in metres, for the distance estimate
const STALE_MS = 1200;         // hide backend boxes older than this (or 2.5x the backend's frame time)

let vision, engine, enginePromise, delegate = '', TESS = [];

export const landmarker = () => engine;

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

export function constraints() {
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
let inflight = false;
let lastNear = [];
let backend = { faces: [], at: 0, scale: 1, z: null, ms: 0 };
const handled = new Map();     // track -> status already acted on
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
  backend = { faces: [], at: 0, scale: 1, z: null, ms: 0 };
  resize();
  addEventListener('resize', resize);
  loop();
  if (state.settings.backendUrl) pump();
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

// ───────────── frame loop (browser, every camera frame) ─────────────

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  const now = performance.now();
  if (lastT) fps = fps ? fps * 0.9 + (1000 / (now - lastT)) * 0.1 : 1000 / (now - lastT);
  lastT = now;

  // With the backend connected the browser model only draws the mesh and gives
  // pose hints, so it runs on every other frame to leave the CPU to recognition.
  const z = zone();
  let near = lastNear;
  if (!state.settings.backendUrl || frame % 2 === 0) {
    let res;
    try { res = engine.detectForVideo(zoneImage(z), now); } catch { return; }
    near = lastNear = (res.faceLandmarks || []).map(lm => describe(lm, z)).sort((a, b) => b.w - a.w);
  }
  if (++frame % 6 === 0) sampleLight(near[0] || biggestBackendFace());
  if (frame % 10 === 0) $('#mFps').textContent = fps.toFixed(1);

  const far = liveBackendFaces();
  $('#mFaces').textContent = state.settings.backendUrl ? far.length : near.length;
  evaluate(near, far);
  draw(near, far);
}

// The browser's face model only looks at the gate lane (a centre square):
// it is short-range, and cropping makes a student 2 m away look ~1.8x bigger.
// The same lane is what gets streamed to the backend.
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

// Rough distance from face width, using the camera's field of view (Settings → Gate).
function distance(px) {
  const fov = (Number(state.settings.fov) || 78) * Math.PI / 180;
  return (FACE_M * video.videoWidth) / (2 * Math.tan(fov / 2) * px);
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

// ───────────── backend stream (as fast as the backend answers) ─────────────

// Backend faces in camera pixels, while they are fresh.
function liveBackendFaces() {
  if (!backend.z || performance.now() - backend.at > Math.max(STALE_MS, backend.ms * 2.5)) return [];
  const { z, scale } = backend;
  return backend.faces.map(f => {
    const [x1, y1, x2, y2] = f.box.map(v => v * scale);
    return { ...f, x: z.x + x1, y: z.y + y1, w: x2 - x1, h: y2 - y1, px: f.width * scale };
  });
}

function biggestBackendFace() {
  return liveBackendFaces().sort((a, b) => b.w - a.w)[0];
}

async function pump() {
  if (!running || inflight) return;
  inflight = true;
  try {
    if (video.readyState < 2) { await new Promise(r => setTimeout(r, 200)); return; }
    const z = zone();
    // Native resolution (capped at 4K-lane size): far faces need every pixel.
    const px = Math.round(Math.min(z.w, 2160));
    const c = document.createElement('canvas');
    c.width = px; c.height = Math.round(z.h * px / z.w);
    c.getContext('2d').drawImage(video, z.x, z.y, z.w, z.h, 0, 0, c.width, c.height);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.88));
    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    fd.append('gate', state.settings.gateName);
    const res = await fetch(state.settings.backendUrl.replace(/\/$/, '') + '/frame', { method: 'POST', body: fd, signal: AbortSignal.timeout(8000) });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${detail.slice(0, 80)}`);
    }
    const j = await res.json();
    backend = { faces: j.faces || [], at: performance.now(), scale: z.w / c.width, z, ms: j.ms };
    $('#mMatcher').textContent = `BACKEND · ${j.ms} ms${j.far_scan ? ' · FAR' : ''}`;
    for (const f of backend.faces) {
      if (!['match', 'review', 'unknown', 'spoof'].includes(f.status)) continue;
      if (handled.get(f.track) === f.status) continue;     // already acted on this verdict
      handled.set(f.track, f.status);
      await handleResult(f, c);
    }
    if (handled.size > 500) handled.clear();
  } catch (e) {
    const why = e.name === 'TimeoutError' ? 'TIMEOUT' : e.message.startsWith('HTTP') ? e.message.split(' ').slice(0, 2).join(' ') : 'NOT RUNNING';
    $('#mMatcher').textContent = `BACKEND · ${why}`;
    flash(why === 'NOT RUNNING' ? `BACKEND NOT RUNNING AT ${state.settings.backendUrl}` : `BACKEND ERROR · ${e.message.slice(0, 60)}`, 'bad', 1500);
    console.error('Backend /frame failed:', e);
    await new Promise(r => setTimeout(r, 1000));
  } finally {
    inflight = false;
    if (running) setTimeout(pump, 0);
  }
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

const COLORS = { match: '#3dffa2', pending: '#4fd8ff', too_far: 'rgba(255,255,255,.55)', low_quality: '#ffb648', review: '#ffb648', unknown: '#ff5a36', spoof: '#ff5a36', multiple_faces: '#ffb648' };

function faceLabel(f) {
  const s = f.student_id && state.byId.get(String(f.student_id));
  const d = `${distance(f.px).toFixed(1)}m`;
  switch (f.status) {
    case 'match': return `${s ? s.name.toUpperCase() : f.student_id} ✓`;
    case 'pending': return f.reason?.includes('closer') ? `${s ? s.name.split(' ')[0].toUpperCase() : 'KNOWN'} · COME CLOSER · ${d}` : `VERIFYING ${Math.min(f.frames || 0, f.need_frames || 3)}/${f.need_frames || 3} · ${d}`;
    case 'too_far': return `DETECTED · ${d}`;
    case 'low_quality': return `${(f.reason || 'unclear').toUpperCase()} · ${d}`;
    case 'review': return 'TEACHER CHECK';
    case 'spoof': return 'PHOTO / SCREEN?';
    case 'unknown': return 'NOT RECOGNISED';
    default: return f.status.toUpperCase();
  }
}

function brackets(bx, by, bw, bh, color, width) {
  const c = Math.min(bw, bh) * 0.2;
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(bx, by + c); ctx.lineTo(bx, by); ctx.lineTo(bx + c, by);
  ctx.moveTo(bx + bw - c, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + c);
  ctx.moveTo(bx + bw, by + bh - c); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - c, by + bh);
  ctx.moveTo(bx + c, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - c);
  ctx.stroke();
}

function tag(x, y, text, color) {
  ctx.font = '600 12px "JetBrains Mono", monospace';
  const tw = ctx.measureText(text).width + 14;
  ctx.fillStyle = 'rgba(2,14,9,.8)';
  ctx.fillRect(x, y - 24, tw, 20);
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 24, 3, 20);
  ctx.fillText(text, x + 8, y - 10);
}

function draw(near, far) {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.clearRect(0, 0, cw, ch);
  const { s, pt } = mapper();
  drawZone(pt);

  // Face mesh on the closest student (cosmetic, from the browser model).
  const f0 = near[0];
  if (f0 && TESS.length) {
    ctx.strokeStyle = 'rgba(61,255,162,.16)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (const { start, end } of TESS) {
      const [x1, y1] = pt(f0.lm[start].x, f0.lm[start].y), [x2, y2] = pt(f0.lm[end].x, f0.lm[end].y);
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    ctx.fillStyle = '#ff5ad1';
    for (const k of [33, 263, 1, 61, 291, 152, 10]) {
      const [x, y] = pt(f0.lm[k].x, f0.lm[k].y);
      ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill();
    }
  }

  if (state.settings.backendUrl) {
    // Every face the backend is tracking, near or far, each with its own verdict.
    for (const f of far) {
      const [ax, ay] = pt(f.x + f.w, f.y);
      const w = f.w * s, h = f.h * s, pad = Math.max(w * 0.15, 4);
      const color = COLORS[f.status] || '#fff';
      brackets(ax - pad, ay - pad, w + pad * 2, h + pad * 2, color, f.w > 80 ? 3 : 2);
      tag(ax - pad, ay - pad, faceLabel(f), color);
    }
  } else {
    near.forEach((f, i) => {
      const [ax, ay] = pt(f.x + f.w, f.y);
      const w = f.w * s, h = f.h * s, pad = w * 0.12;
      brackets(ax - pad, ay - pad * 1.6, w + pad * 2, h + pad * 2.4, i ? 'rgba(255,255,255,.45)' : '#ffb648', i ? 1.5 : 3);
      if (!i) tag(ax - pad, ay - pad * 1.6, 'FACE OK · MANUAL MODE', '#ffb648');
    });
  }
}

// ───────────── prompts + readouts ─────────────

function evaluate(near, far) {
  const f = near[0];
  const turned = f && Math.abs(f.yaw) > MAX_YAW;
  $('#mPose').textContent = f ? (turned ? `TURNED ${f.yaw > 0 ? 'L' : 'R'}` : 'FRONTAL') : '—';

  const main = [...far].sort((a, b) => b.w - a.w)[0];
  if (main) {
    $('#mSize').textContent = `${Math.round(main.px)} px · ${distance(main.px).toFixed(1)} m`;
    $('#mQuality').textContent = main.status === 'too_far' ? 'TOO FAR' : main.range === 'far' ? 'FAR' : main.status === 'low_quality' ? 'LOW' : 'GOOD';
    $('#mLive').textContent = main.live != null ? `${Math.round(main.live * 100)}%` : main.px < 48 ? 'NEEDS CLOSER' : '—';
    $('#mVerify').textContent = main.status === 'pending' ? `${main.frames || 0}/${main.need_frames || 3} FRAMES` : main.status.toUpperCase().replace('_', ' ');
  } else {
    $('#mSize').textContent = f ? `${Math.round(f.w)} px` : '—';
    $('#mQuality').textContent = f ? 'GOOD' : 'NO FACE';
    $('#mLive').textContent = '—'; $('#mVerify').textContent = '—';
  }

  if (banner && performance.now() < banner.until) { prompt(banner.text, banner.kind); return; }
  if (f && f.w > 90) {
    if (turned) return prompt('LOOK STRAIGHT AT THE CAMERA', 'warn');
    if (light < LOW_LIGHT) return prompt('TOO DARK · CHECK GATE LIGHTING', 'warn');
    if (light > GLARE) return prompt('TOO BRIGHT · CAMERA FACING LIGHT', 'warn');
  }
  if (!state.settings.backendUrl) return prompt(f ? 'RECOGNITION OFFLINE · USE MANUAL MARK' : 'STAND IN FRONT OF THE CAMERA', f ? 'warn' : '');
  if (!far.length) return prompt('WALK TOWARDS THE CAMERA');
  const pending = far.filter(x => x.status === 'pending');
  if (pending.some(x => x.reason?.includes('closer'))) return prompt('KEEP WALKING · LOOK AT THE CAMERA', 'ok');
  if (pending.length) return prompt(`VERIFYING ${pending.length} STUDENT${pending.length > 1 ? 'S' : ''} · KEEP LOOKING AHEAD`, 'ok');
  if (far.every(x => x.status === 'too_far')) return prompt('WALK TOWARDS THE CAMERA');
  if (far.some(x => x.status === 'match')) return prompt('MARKED ✓ · PLEASE MOVE ON', 'ok');
  prompt('WALK TOWARDS THE CAMERA');
}

// ───────────── acting on decisions ─────────────

// The face cut out of the frame that was sent, for the side panels.
function snapshot(f, frameCanvas) {
  const [x1, y1, x2, y2] = f.box;
  const side = Math.max(x2 - x1, y2 - y1) * 1.8;
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  c.getContext('2d').drawImage(frameCanvas, (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2, side, side, 0, 0, 128, 128);
  return c.toDataURL('image/jpeg', 0.85);
}

async function handleResult(f, frameCanvas) {
  const img = () => snapshot(f, frameCanvas);
  const s = f.student_id && state.byId.get(String(f.student_id));

  if (f.status === 'unknown') { flash('NOT RECOGNISED · PLEASE SEE THE TEACHER', 'bad', 2000); return; }
  if (!s) { flash(`ID ${f.student_id} IS ENROLLED BUT NOT IN THE ROSTER`, 'bad', 2500); return; }

  if (f.status === 'review' || f.status === 'spoof') {
    // Never marked without a person checking: one tap on the gate screen.
    const reason = f.status === 'spoof' ? 'possible photo / screen' : (f.reason || '');
    if (!review.some(r => r.id === s.id)) review.unshift({ id: s.id, conf: f.confidence ?? 0, img: img(), reason, track: f.track, kind: f.status });
    renderReview();
    flash('PLEASE WAIT · TEACHER WILL CONFIRM', 'warn', 1800);
    if (f.status === 'spoof') buzz();
    return;
  }

  const existing = recordFor(s.id);
  if (existing && existing.s !== 'A') {
    showIdent(s, existing, f.confidence, img(), true);
    flash(`${s.name.toUpperCase()} · ALREADY MARKED ${existing.t}`, 'ok', 1500);
    return;
  }
  const status = arrivalStatus();
  await mark([s.id], status, { source: 'cam', confidence: f.confidence });
  showIdent(s, recordFor(s.id), f.confidence, img(), false);
  flash(`WELCOME, ${s.name.split(' ')[0].toUpperCase()} · ${STATUS[status].toUpperCase()}`, 'ok', 1800);
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
      // The teacher's ✓ teaches the gallery this face (not for a possible photo/screen).
      if (r.track && r.kind === 'review' && state.settings.backendUrl) {
        fetch(state.settings.backendUrl.replace(/\/$/, '') + '/review/confirm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ track: r.track, student_id: r.id }) }).catch(() => {});
      }
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
