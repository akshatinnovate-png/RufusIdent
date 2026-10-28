// App shell: boot, routing and the dashboard screens.

import * as data from './data.js';
import { state, STATUS } from './data.js';
import { runBoot } from './boot.js';
import { openCamera, closeCamera, wireCamera, listCameras } from './camera.js';
import { ask, resetChat, wireMic } from './assistant.js';
import { startSync, syncNow } from './sync.js';
import { openStation, closeStation, wireStation } from './station.js';
import { $, $$, esc, avatar, toast, pct, prettyDate, STATUS_CLASS } from './ui.js';

let route = { view: 'home', id: null };
let classesDate = data.today();

// ───────────── routing ─────────────

function parseHash() {
  const [, view = 'home', id] = location.hash.split('/');
  return { view: view || 'home', id: id ? decodeURIComponent(id) : null };
}

function go() {
  const next = parseHash();
  if (route.view === 'camera' && next.view !== 'camera') closeCamera();
  if (route.view === 'enrol' && next.view !== 'enrol') closeStation();
  route = next;
  if (route.view === 'camera') { openCamera(); return; }
  if (route.view === 'enrol') { openStation(); return; }
  $$('.view').forEach(v => { v.hidden = v.dataset.view !== route.view; });
  $$('.rail [data-nav]').forEach(a => a.classList.toggle('on', a.dataset.nav === route.view || (route.view === 'class' && a.dataset.nav === 'classes')));
  render();
  $('.shell').scrollTo?.(0, 0);
}

function render() {
  $('#demoBanner').hidden = !state.demo;
  $('#topDate').textContent = `· ${new Date().toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })}`;
  ({ home: renderHome, classes: renderClasses, class: renderClass, student: renderStudent, leaderboard: renderLeaderboard }[route.view] || renderHome)();
}

// ───────────── home ─────────────

const ICONS = {
  gate: '<svg viewBox="0 0 48 48"><rect x="4" y="8" width="40" height="32" rx="9" fill="#e9f6ee"/><path d="M14 18v-4h4M30 14h4v4M34 30v4h-4M18 34h-4v-4" stroke="#136b35" stroke-width="3" fill="none" stroke-linecap="round"/><circle cx="24" cy="23" r="5" fill="#e8502a"/></svg>',
  classes: '<svg viewBox="0 0 48 48"><rect x="6" y="6" width="16" height="16" rx="5" fill="#136b35"/><rect x="26" y="6" width="16" height="16" rx="5" fill="#f4b740"/><rect x="6" y="26" width="16" height="16" rx="5" fill="#1d2160"/><rect x="26" y="26" width="16" height="16" rx="5" fill="#e8502a"/></svg>',
  trophy: '<svg viewBox="0 0 48 48"><path d="M15 8h18v9a9 9 0 0 1-18 0z" fill="#f4b740"/><path d="M15 11H8v3a7 7 0 0 0 7 7M33 11h7v3a7 7 0 0 1-7 7" stroke="#f4b740" stroke-width="3" fill="none"/><rect x="21" y="26" width="6" height="8" fill="#136b35"/><rect x="14" y="34" width="20" height="6" rx="3" fill="#1d2160"/></svg>',
};

function renderHome() {
  const h = new Date().getHours();
  $('#greet').textContent = h < 12 ? 'Good morning,' : h < 17 ? 'Good afternoon,' : 'Good evening,';
  const d = data.daySummary();
  const top = data.leaderboard({ limit: 1 })[0];
  const here = d.P + d.L;
  $('#homeCards').innerHTML = `
    <a class="card" href="#/camera">
      ${ICONS.gate}
      <p class="big">${here.toLocaleString('en-IN')}<small> / ${d.total.toLocaleString('en-IN')}</small></p>
      <p>students through the gate today, ${pct(here, d.total)}% so far. ${d.L} late.</p>
      <span class="tag-line">Live gate attendance</span>
    </a>
    <a class="card" href="#/classes">
      ${ICONS.classes}
      <p class="big">${d.classesComplete}<small> / ${state.classes.length}</small></p>
      <p>class registers complete. ${d.unmarked.toLocaleString('en-IN')} students still not marked.</p>
      <span class="tag-line">Class registers</span>
    </a>
    <a class="card" href="${top ? `#/student/${encodeURIComponent(top.id)}` : '#/leaderboard'}">
      ${ICONS.trophy}
      <p class="big">${top ? `${top.pct}%` : '—'}</p>
      <p>${top ? `${esc(top.name)} (${esc(data.classKey(top))}) leads the attendance leaderboard.` : 'The leaderboard fills up after the first day of attendance.'}</p>
      <span class="tag-line">Leaderboard</span>
    </a>`;

  const firstClass = state.classes[0] || 'X-A';
  const chips = [
    ['Today’s summary', 'Give me today’s attendance summary, class-wise.'],
    [`Absent in ${firstClass}`, `Who is absent in ${firstClass} today?`],
    ['Not marked yet', 'Which students are not marked yet today? Group by class.'],
    ['Top 10', 'Show the top 10 attendance leaderboard.'],
    ['Export Excel', 'Export today’s attendance Excel.'],
  ];
  $('#chips').innerHTML = chips.map(([label, q]) => `<button type="button" class="chip" data-q="${esc(q)}">${esc(label)}</button>`).join('');
}

// ───────────── classes ─────────────

function renderClasses() {
  $('#classesDate').value = classesDate;
  const d = data.daySummary(classesDate);
  $('#classesSub').textContent = `${prettyDate(classesDate)} · ${d.P + d.L} present of ${d.total} · ${d.A} absent · ${d.unmarked} not marked`;
  if (!state.classes.length) {
    $('#classGrid').innerHTML = `<div class="empty">No roster yet. Open <b>Settings → Data</b> and import your student list (.xlsx or .csv).</div>`;
    return;
  }
  $('#classGrid').innerHTML = d.classes.map(c => {
    const done = c.total - c.unmarked;
    return `<a class="class-tile" href="#/class/${encodeURIComponent(c.cls)}">
      <div class="ct-head"><b>${esc(c.cls)}</b><span>${c.total} students</span></div>
      <div class="bar"><i class="p" style="width:${pct(c.P, c.total)}%"></i><i class="l" style="width:${pct(c.L, c.total)}%"></i><i class="a" style="width:${pct(c.A, c.total)}%"></i></div>
      <div class="ct-nums"><span class="st-p"><small>P</small>${c.P}</span><span class="st-l"><small>L</small>${c.L}</span><span class="st-a"><small>A</small>${c.A}</span><span class="muted">${c.unmarked ? `${c.unmarked} left` : 'complete ✓'}</span></div>
      <div class="ct-pct">${pct(c.P + c.L, c.total)}%<small>${done === c.total ? 'final' : 'so far'}</small></div>
    </a>`;
  }).join('');
}

function renderClass() {
  const key = route.id;
  const list = data.studentsIn(key);
  const c = data.classSummary(key, classesDate);
  $('#classTitle').textContent = `Class ${key}`;
  $('#classSub').textContent = `${prettyDate(classesDate)} · ${c.P} present · ${c.L} late · ${c.A} absent · ${c.unmarked} not marked`;
  const f = $('#classFilter').value.trim().toLowerCase();
  const day = data.getDay(classesDate);
  const rows = list.filter(s => !f || s.name.toLowerCase().includes(f) || String(s.roll) === f || s.id.includes(f));
  $('#register').innerHTML = `
    <div class="reg-row head"><span>Roll</span><span>Student</span><span>Adm. no</span><span>Time</span><span>Status</span></div>
    ${rows.map(s => {
      const r = day[s.id];
      return `<div class="reg-row">
        <span class="roll">${esc(s.roll)}</span>
        <a class="who" href="#/student/${encodeURIComponent(s.id)}">${avatar(s, 'mini')}<b>${esc(s.name)}</b></a>
        <span class="muted">${esc(s.id)}</span>
        <span class="muted">${r ? esc(r.t) : '—'}</span>
        <span class="seg" data-id="${esc(s.id)}">
          ${['P', 'L', 'A'].map(k => `<button class="${r?.s === k ? `on ${STATUS_CLASS[k]}` : ''}" data-s="${k}" title="${STATUS[k]}">${k}</button>`).join('')}
        </span>
      </div>`;
    }).join('') || '<div class="empty">No students match.</div>'}`;
}

// ───────────── student datasheet ─────────────

function renderStudent() {
  const s = state.byId.get(route.id);
  if (!s) { $('#sheet').innerHTML = '<div class="empty">Student not found.</div>'; return; }
  const st = data.studentStats(s.id);
  const rec = data.recordFor(s.id);
  const key = data.classKey(s);
  const [firstName, ...rest] = s.name.split(' ');
  const board = data.leaderboard({ limit: 5 });
  const classBoard = data.leaderboard({ cls: key, limit: 5 });
  const classRank = data.leaderboard({ cls: key, limit: 9999 }).findIndex(x => x.id === s.id) + 1;
  const lbList = (list, rankKey) => list.map(x => `
    <a class="lb-mini ${x.id === s.id ? 'me' : ''}" href="#/student/${encodeURIComponent(x.id)}">
      <span class="r">${x[rankKey]}</span>${avatar(x, 'mini')}<b>${esc(x.name)}</b><span class="muted">${esc(data.classKey(x))}</span><em>${x.pct}%</em>
    </a>`).join('') || '<p class="muted">No data yet.</p>';

  $('#sheet').innerHTML = `
  <article class="datasheet">
    <div class="ds-accent"></div>
    <header class="ds-nav">
      <a href="#/class/${encodeURIComponent(key)}" class="back">← Class ${esc(key)}</a>
      <nav><span>DATASHEET</span><span>ATTENDANCE</span><span>HISTORY</span></nav>
      <img src="assets/crest.png" alt="" class="ds-crest">
    </header>

    <div class="ds-body">
      <div class="ds-left">
        <span class="today-pill ${rec ? STATUS_CLASS[rec.s] : ''}">${rec ? `Today: ${STATUS[rec.s]} · ${esc(rec.t)}` : 'Today: not marked yet'}</span>
        <h2 class="ds-name">${esc(firstName)}<br>${esc(rest.join(' '))}</h2>
        <p class="ds-team"><img src="assets/crest.png" alt="">Class ${esc(s.cls)} · Section ${esc(s.section)}</p>
        ${s.note ? `<p class="ds-note">★ ${esc(s.note)}</p>` : ''}
        <dl class="ds-facts">
          <div><dt>ADMISSION NO</dt><dd>${esc(s.id)}</dd></div>
          <div><dt>ROLL</dt><dd>${esc(s.roll || '—')}</dd></div>
          <div><dt>HOUSE</dt><dd>${esc(s.house || '—')}</dd></div>
          <div><dt>CLASS RANK</dt><dd>${classRank ? `${classRank} / ${data.studentsIn(key).length}` : '—'}</dd></div>
          ${s.blood ? `<div><dt>BLOOD GROUP</dt><dd>${esc(s.blood)}</dd></div>` : ''}
          ${s.father ? `<div class="wide"><dt>FATHER'S NAME</dt><dd class="txt">${esc(s.father)}</dd></div>` : ''}
          ${s.phone ? `<div class="wide"><dt>PHONE</dt><dd class="txt"><a href="tel:${esc(s.phone.replace(/[^\d+]/g, ''))}">${esc(s.phone)}</a></dd></div>` : ''}
        </dl>
      </div>

      <div class="ds-center">
        <div class="ds-watermark">GULMOHUR</div>
        <div class="ds-orbit"></div>
        <div class="ds-rank">
          <small>SCHOOL RANK</small>
          <b><sup>#</sup>${st.rank || '—'}</b>
          <span>of ${st.rankOf.toLocaleString('en-IN')} on the attendance leaderboard</span>
        </div>
        <div class="ds-photo">${avatar(s, 'ds-avatar')}</div>
      </div>

      <div class="ds-right">
        <div class="lb-card">
          <div class="lb-tabs"><button class="on" data-lb="school">School top 5</button><button data-lb="class">${esc(key)} top 5</button></div>
          <div data-lbp="school">${lbList(board, 'rank')}</div>
          <div data-lbp="class" hidden>${lbList(classBoard, 'classRank')}</div>
        </div>
        <div class="history">
          <h5>Last ${st.history.length} school days</h5>
          <div class="dots">${st.history.map(([d, c]) => `<i class="${c ? STATUS_CLASS[c] : 'none'}" title="${d}: ${c ? STATUS[c] : 'no record'}"></i>`).join('')}</div>
          <div class="legend"><span class="st-p">● Present</span><span class="st-l">● Late</span><span class="st-a">● Absent</span></div>
        </div>
      </div>
    </div>

    <div class="ds-tiles">
      <div class="tile green"><small>ATTENDANCE</small><b>${st.pct ?? '—'}<sup>${st.pct != null ? '%' : ''}</sup></b><span>${st.streak} day streak</span></div>
      <div class="tile navy"><small>DAYS PRESENT</small><b>${st.present + st.late}</b><span>${st.late} of them late</span></div>
      <div class="tile flame"><small>DAYS ABSENT</small><b>${st.absent}</b><span>of ${st.days} school days</span></div>
    </div>

    <aside class="shoutout">
      <span class="so-badge">DEV SHOUTOUT</span>
      <b>Akshat Sarkar</b>
      <span>Built RufusIdent for Gulmohur High School · GHS AI Team · MIT License</span>
    </aside>
  </article>`;
}

// ───────────── leaderboard ─────────────

function renderLeaderboard() {
  const sel = $('#lbClass');
  const cur = sel.value;
  sel.innerHTML = `<option value="">Whole school</option>${state.classes.map(c => `<option ${c === cur ? 'selected' : ''}>${esc(c)}</option>`).join('')}`;
  const cls = sel.value || null;
  const list = data.leaderboard({ cls, limit: 50 });
  const rk = x => (cls ? x.classRank : x.rank);
  const [a, b, c] = list;
  $('#podium').innerHTML = list.length ? [b, a, c].map((x, i) => x ? `
    <a class="pod p${[2, 1, 3][i]}" href="#/student/${encodeURIComponent(x.id)}">
      ${avatar(x, 'pod-av')}<b>${esc(x.name)}</b><span>${esc(data.classKey(x))}</span>
      <div class="step"><em>#${rk(x)}</em><strong>${x.pct}%</strong></div>
    </a>` : '<div></div>').join('') : '<div class="empty">No attendance recorded yet.</div>';
  $('#lbTable').innerHTML = list.slice(3).map(x => `
    <a class="lb-row" href="#/student/${encodeURIComponent(x.id)}">
      <span class="r">${rk(x)}</span>${avatar(x, 'mini')}<b>${esc(x.name)}</b><span class="muted">${esc(data.classKey(x))}</span>
      <span>${x.present + x.late} days</span><span class="muted">${x.streak}🔥</span><em>${x.pct}%</em>
    </a>`).join('');
}

// ───────────── search palette ─────────────

function openPalette() {
  $('#palette').hidden = false;
  const i = $('#paletteInput');
  i.value = ''; $('#paletteResults').innerHTML = '<p class="muted pad">Type a name, admission number or a class like “X-A”.</p>';
  i.focus();
}
function closePalette() { $('#palette').hidden = true; }

function paletteSearch() {
  const q = $('#paletteInput').value;
  const cls = data.resolveClass(q);
  const list = data.search(q, 8);
  $('#paletteResults').innerHTML =
    (cls ? `<a class="pr" href="#/class/${encodeURIComponent(cls)}"><span class="mini mono">⌗</span><b>Class ${esc(cls)}</b><span class="muted">open register</span></a>` : '') +
    list.map(s => `<a class="pr" href="#/student/${encodeURIComponent(s.id)}">${avatar(s, 'mini')}<b>${esc(s.name)}</b><span class="muted">${esc(data.classKey(s))} · ${esc(s.id)}</span></a>`).join('') ||
    '<p class="muted pad">No match.</p>';
}

// ───────────── settings ─────────────

async function openSettings() {
  const f = $('#settingsForm');
  for (const [k, v] of Object.entries(state.settings)) if (f.elements[k] && k !== 'cameraId') f.elements[k].value = v;
  try {
    const cams = await listCameras();
    f.elements.cameraId.innerHTML = '<option value="">Default camera</option>' + cams.map(c => `<option value="${esc(c.deviceId)}">${esc(c.label || 'Camera')}</option>`).join('');
  } catch {}
  f.elements.cameraId.value = state.settings.cameraId;
  updateDataStats();
  $('#settings').showModal();
  loadReportSettings();
}

// ───────────── daily report (settings live on the backend) ─────────────

const PROVIDERS = {
  gmail: { host: 'smtp.gmail.com', port: 587, security: 'starttls' },
  outlook: { host: 'smtp.office365.com', port: 587, security: 'starttls' },
};
const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

async function backendJson(path, body) {
  const url = state.settings.backendUrl?.replace(/\/$/, '');
  if (!url) throw new Error('Set the recognition backend URL (Settings → Gate) first.');
  const res = await fetch(url + path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
  return j;
}

function reportStatusLine(st) {
  const el = $('#rpStatus');
  const last = st.history?.filter(h => !h.test).at(-1);
  const parts = [];
  if (st.problem) parts.push(`Not ready: ${st.problem}`);
  else if (st.next_run) parts.push(`Next report: ${new Date(st.next_run).toLocaleString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`);
  else parts.push('Automatic sending is off.');
  if (last) parts.push(`last ${last.ok ? 'sent' : 'FAILED'} ${new Date(last.at).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}${last.ok ? '' : ': ' + last.detail}`);
  el.textContent = parts.join(' · ');
  el.className = `note ${st.problem || st.last_error ? 'bad' : 'ok'}`;
}

async function loadReportSettings() {
  $('#rpDays').innerHTML = DAY_NAMES.map((d, i) => `<label><input type="checkbox" value="${i}">${d}</label>`).join('');
  try {
    const [s, st] = await Promise.all([backendJson('/report/settings'), backendJson('/report/status')]);
    $('#rpEnabled').checked = s.enabled;
    $('#rpRecipients').value = s.recipients.join('\n');
    $('#rpTime').value = s.time;
    $$('#rpDays input').forEach(c => { c.checked = s.days.includes(Number(c.value)); });
    const prov = Object.entries(PROVIDERS).find(([, p]) => p.host === s.smtp.host)?.[0] || 'custom';
    $('#rpProvider').value = prov;
    $('#rpCustom').hidden = prov !== 'custom';
    $('#rpHost').value = s.smtp.host; $('#rpPort').value = s.smtp.port; $('#rpSec').value = s.smtp.security;
    $('#rpUser').value = s.smtp.username;
    $('#rpPass').value = '';
    $('#rpPass').placeholder = s.smtp.has_password ? '•••••••• (saved, leave blank to keep)' : 'app password';
    $('#rpDir').value = s.reports_dir || '';
    reportStatusLine(st);
  } catch (e) {
    $('#rpStatus').textContent = `Backend not reachable: ${e.message}`;
    $('#rpStatus').className = 'note bad';
  }
}

function reportForm() {
  const prov = $('#rpProvider').value;
  const smtp = prov === 'custom'
    ? { host: $('#rpHost').value.trim(), port: Number($('#rpPort').value) || 587, security: $('#rpSec').value }
    : { ...PROVIDERS[prov] };
  return {
    enabled: $('#rpEnabled').checked,
    recipients: $('#rpRecipients').value.split(/[\n,;]+/).map(x => x.trim()).filter(Boolean),
    time: $('#rpTime').value || '09:30',
    days: $$('#rpDays input').filter(c => c.checked).map(c => Number(c.value)),
    reports_dir: $('#rpDir').value.trim(),
    smtp: { ...smtp, username: $('#rpUser').value.trim(), password: $('#rpPass').value },
  };
}

async function reportAction(act) {
  const status = $('#rpStatus');
  try {
    const form = reportForm();
    const bad = form.recipients.find(r => !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(r));
    if (bad) throw new Error(`“${bad}” is not an email address.`);
    await backendJson('/report/settings', form);
    if (act === 'rp-save') toast('Report settings saved');
    if (act === 'rp-test' || act === 'rp-now') {
      status.textContent = 'Sending…'; status.className = 'note';
      await syncNow();                      // make sure the latest marks are on the backend
      const r = await backendJson('/report/send', { test: act === 'rp-test' });
      toast(`Mail server accepted it for ${r.recipients.join(', ')}. It can take a few minutes to arrive.`);
      if (r.refused?.length) toast(`Refused (check the spelling): ${r.refused.join(', ')}`, { kind: 'warn' });
    }
  } catch (e) {
    toast(e.message, { kind: 'warn' });
  }
  loadReportSettings();
}

function updateDataStats() {
  $('#dataStats').textContent = `${state.roster.length} students · ${state.classes.length} classes · ${state.days.size} day(s) of records${state.demo ? ' · DEMO' : ''}`;
}

function saveSettingsForm() {
  const f = $('#settingsForm');
  const v = n => f.elements[n].value.trim();
  data.saveSettings({
    groqKey: v('groqKey'), model: v('model') || 'openai/gpt-oss-120b',
    gateName: v('gateName') || 'Main Gate', cameraId: f.elements.cameraId.value, zone: f.elements.zone.value, fov: Number(v('fov')) || 78,
    lateAfter: v('lateAfter') || '08:00', backendUrl: v('backendUrl') || 'http://127.0.0.1:8899',
  });
  toast('Settings saved');
}

function download(name, text, type) {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ───────────── events ─────────────

function wire() {
  addEventListener('hashchange', go);

  document.addEventListener('click', async e => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'new-chat') { resetChat(); location.hash = '#/home'; }
    if (act === 'search') openPalette();
    if (act === 'settings') openSettings();
    if (act === 'export') {
      try { toast(`Downloaded ${data.exportExcel(classesDate)}`); } catch (err) { toast(err.message, { kind: 'warn' }); }
    }
    if (act === 'finalize-all' || act === 'finalize-class') {
      const key = act === 'finalize-class' ? route.id : null;
      const n = (key ? data.studentsIn(key) : state.roster).filter(s => !data.recordFor(s.id, classesDate)).length;
      if (!n) return toast('Everyone is already marked.');
      if (!confirm(`Mark ${n} unmarked student(s) ${key ? `in ${key} ` : ''}as absent for ${classesDate}?`)) return;
      const { undo } = await data.markRemainingAbsent(key, classesDate);
      toast(`${n} marked absent`, { action: 'Undo', onAction: () => data.undo(undo) });
    }
    if (act === 'demo') {
      if (state.roster.length && !state.demo && !confirm('This replaces the real roster and attendance with demo data. Continue?')) return;
      await data.loadDemo(); updateDataStats(); toast('Demo data loaded (640 fake students)');
    }
    if (act.startsWith('rp-')) reportAction(act);
    if (act === 'backup') download(`rufusident-backup-${data.today()}.json`, data.backupJson(), 'application/json');
    if (act === 'wipe') {
      if (!confirm('Erase the roster and ALL attendance history on this laptop? Download a backup first.')) return;
      await data.wipe(); updateDataStats(); toast('All data erased');
    }
  });

  // composer
  const promptEl = $('#prompt');
  const grow = () => { promptEl.style.height = 'auto'; promptEl.style.height = `${Math.min(promptEl.scrollHeight, 160)}px`; };
  promptEl.addEventListener('input', grow);
  promptEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); } });
  $('#composer').addEventListener('submit', e => {
    e.preventDefault();
    const t = promptEl.value;
    promptEl.value = ''; grow();
    ask(t);
  });
  $('#chips').addEventListener('click', e => { const c = e.target.closest('[data-q]'); if (c) ask(c.dataset.q); });
  wireMic((text, final) => { promptEl.value = text; grow(); if (final) $('#composer').requestSubmit(); });

  // classes
  $('#classesDate').addEventListener('change', e => { classesDate = e.target.value || data.today(); render(); });
  $('#classFilter').addEventListener('input', renderClass);
  $('#register').addEventListener('click', async e => {
    const b = e.target.closest('.seg button');
    if (!b) return;
    const id = b.parentElement.dataset.id;
    await data.mark([id], b.dataset.s, { date: classesDate, source: 'manual' });
  });
  $('#lbClass').addEventListener('change', renderLeaderboard);
  $('#sheet').addEventListener('click', e => {
    const t = e.target.closest('[data-lb]');
    if (!t) return;
    $$('[data-lb]', $('#sheet')).forEach(b => b.classList.toggle('on', b === t));
    $$('[data-lbp]', $('#sheet')).forEach(p => { p.hidden = p.dataset.lbp !== t.dataset.lb; });
  });

  // palette
  $('#paletteInput').addEventListener('input', paletteSearch);
  $('#paletteInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') { $('#paletteResults a')?.click(); }
  });
  $('#palette').addEventListener('click', e => { if (e.target.id === 'palette' || e.target.closest('a')) closePalette(); });

  addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); }
    if (e.key === 'Escape') {
      if (!$('#palette').hidden) closePalette();
      else if (route.view === 'camera' && document.activeElement !== $('#manualInput')) location.hash = '#/home';
      else if (route.view === 'enrol' && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) location.hash = '#/home';
    }
  });

  // settings
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    $$('.tab').forEach(x => x.classList.toggle('on', x === t));
    $$('.tab-body').forEach(b => { b.hidden = b.dataset.body !== t.dataset.tab; });
  }));
  $('#settings').addEventListener('close', () => { if ($('#settings').returnValue === 'save') saveSettingsForm(); });
  $('#rpProvider').addEventListener('change', e => { $('#rpCustom').hidden = e.target.value !== 'custom'; });
  $('#rosterFile').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const { students, problems } = await data.importRosterFile(file);
      toast(`Imported ${students.length} students${problems.length ? ` · ${problems.length} row(s) skipped` : ''}`);
      if (problems.length) console.warn('Roster rows skipped:', problems);
      updateDataStats();
    } catch (err) {
      toast(`Import failed: ${err.message}`, { kind: 'warn' });
    }
    e.target.value = '';
  });

  wireCamera();
  wireStation();
  data.onChange(() => { if (route.view !== 'camera' && route.view !== 'enrol') render(); });
}

// ───────────── start ─────────────

(async () => {
  wire();
  await runBoot();
  $('#app').hidden = false;
  go();
  startSync();
})();
