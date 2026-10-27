// Boot screen: every line is a real check, not a fake progress bar.

import { state, load } from './data.js';
import { probeCamera, loadEngine } from './camera.js';
import { checkGroq } from './assistant.js';
import { $, esc } from './ui.js';

const warn = msg => ({ warn: msg });

function withTimeout(promise, ms, what) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what} timed out after ${ms / 1000}s`)), ms)),
  ]);
}

async function checkBackend() {
  const url = state.settings.backendUrl;
  if (!url) return warn('not configured · gate runs in manual-mark mode');
  try {
    // start.bat launches the backend at the same time; give it a few seconds to load its models.
    let res, lastErr;
    for (let i = 0; i < 8 && !res; i++) {
      try { res = await withTimeout(fetch(url.replace(/\/$/, '') + '/health'), 2000, 'backend'); }
      catch (e) { lastErr = e; await new Promise(r => setTimeout(r, 1000)); }
    }
    if (!res) throw lastErr;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const h = await res.json();
    if (!h.students) return warn('online but no students enrolled · run backend/enroll.py');
    return `online · ${h.students.toLocaleString('en-IN')} students · ${h.photos.toLocaleString('en-IN')} photos · anti-spoof ${h.antispoof ? 'on' : 'off'}`;
  } catch (e) {
    return warn(`unreachable (${e.message}) · manual mode`);
  }
}

async function checkReport() {
  const url = state.settings.backendUrl?.replace(/\/$/, '');
  if (!url) return warn('needs the recognition backend');
  try {
    const st = await (await withTimeout(fetch(url + '/report/status'), 3000, 'report')).json();
    if (st.problem) return warn(`not set up (${st.problem.toLowerCase()}) · Settings → Daily report`);
    if (!st.next_run) return warn('automatic email is off · Settings → Daily report');
    const next = new Date(st.next_run).toLocaleString('en-IN', { weekday: 'short', hour: '2-digit', minute: '2-digit' });
    return st.last_error ? warn(`last send failed: ${st.last_error}`) : `next email ${next}`;
  } catch {
    return warn('backend offline');
  }
}

const STEPS = [
  ['Runtime', async () => {
    if (!window.isSecureContext) throw new Error('open the app with start.bat (localhost), not by double-clicking the file');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('this browser has no camera API');
    return `${navigator.hardwareConcurrency || '?'} CPU threads · secure context`;
  }],
  ['Local database', async () => {
    await load();
    return `${state.days.size} day(s) of attendance on disk`;
  }],
  ['Student roster', async () => {
    if (!state.roster.length) return warn('empty · import the roster in Settings → Data');
    return `${state.roster.length.toLocaleString('en-IN')} students · ${state.classes.length} classes${state.demo ? ' · DEMO DATA' : ''}`;
  }],
  ['Excel engine', async () => {
    if (!window.XLSX) throw new Error('vendor/xlsx.full.min.js did not load');
    return `SheetJS ${XLSX.version}`;
  }],
  ['Gate camera', () => withTimeout(probeCamera(), 15000, 'camera')],
  ['Face engine', () => withTimeout(loadEngine(), 30000, 'face engine')],
  ['Recognition backend', checkBackend],
  ['Daily report', checkReport],
  ['AI assistant', () => withTimeout(checkGroq(), 8000, 'Groq')],
];

function typeTitle(text) {
  const el = $('#bootTitle');
  let i = 0;
  return new Promise(resolve => {
    const tick = () => {
      el.textContent = text.slice(0, ++i);
      if (i < text.length) setTimeout(tick, 45); else resolve();
    };
    tick();
  });
}

function petals() {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const box = $('.petals');
  for (let i = 0; i < 16; i++) {
    const p = document.createElement('i');
    p.style.left = `${Math.random() * 100}%`;
    p.style.animationDelay = `${Math.random() * 8}s`;
    p.style.animationDuration = `${8 + Math.random() * 7}s`;
    p.style.setProperty('--s', (0.6 + Math.random() * 0.8).toFixed(2));
    p.style.setProperty('--drift', `${Math.round(Math.random() * 160 - 80)}px`);
    box.append(p);
  }
}

export async function runBoot() {
  petals();
  const log = $('#bootLog');
  const ring = $('#bootRing');
  const C = 2 * Math.PI * 92;
  ring.style.strokeDasharray = C;
  ring.style.strokeDashoffset = C;

  await typeTitle('GULMOHUR HIGH SCHOOL');

  let failed = 0;
  for (let i = 0; i < STEPS.length; i++) {
    const [label, fn] = STEPS[i];
    const li = document.createElement('li');
    li.innerHTML = `<span class="tag run">[ .. ]</span><span class="lbl">${esc(label)}</span><span class="dots"></span><span class="val">checking…</span>`;
    log.append(li);
    let tag = 'ok', text;
    try {
      const out = await fn();
      if (out && out.warn) { tag = 'warn'; text = out.warn; } else text = out;
    } catch (e) {
      tag = 'fail'; text = e.message || String(e); failed++;
    }
    li.querySelector('.tag').className = `tag ${tag}`;
    li.querySelector('.tag').textContent = { ok: '[ OK ]', warn: '[WARN]', fail: '[FAIL]' }[tag];
    li.querySelector('.val').textContent = text;
    ring.style.strokeDashoffset = C * (1 - (i + 1) / STEPS.length);
  }

  const hint = $('#bootHint');
  hint.innerHTML = failed
    ? `<b class="fail">${failed} check(s) failed</b> · the app still opens, fix the red lines first · click to continue`
    : '<b>SYSTEM READY</b> · click or press any key';
  document.body.classList.add('boot-done');

  await new Promise(resolve => {
    const boot = $('#boot');
    let done = false;
    const go = () => {
      if (done) return;
      done = true;
      removeEventListener('keydown', go);
      boot.removeEventListener('click', go);
      resolve();
    };
    addEventListener('keydown', go);
    boot.addEventListener('click', go);
    if (!failed) setTimeout(go, 1800);
  });

  const boot = $('#boot');
  boot.classList.add('leaving');
  setTimeout(() => boot.remove(), 700);
}
