// Local data layer. Everything the UI and the assistant read or write goes
// through this module, so the backend can replace it later without touching
// the screens.

const DB_NAME = 'rufusident';
const STORE = 'kv';

export const STATUS = { P: 'Present', L: 'Late', A: 'Absent' };

const DEFAULT_SETTINGS = {
  groqKey: '',
  model: 'openai/gpt-oss-120b',
  gateName: 'Main Gate',
  cameraId: '',
  lateAfter: '08:00',
  zone: 'square',
  fov: 78,
  backendUrl: 'http://localhost:8000',
};

export const state = {
  roster: [],          // [{id, name, cls, section, roll, house, photo}]
  byId: new Map(),
  classes: [],         // ['IX-A', 'IX-B', …] in school order
  days: new Map(),     // 'YYYY-MM-DD' -> { [studentId]: {s, t, src, c} }
  demo: false,
  settings: { ...DEFAULT_SETTINGS },
};

// ───────────── IndexedDB (tiny key/value wrapper) ─────────────

let dbPromise;
function db() {
  dbPromise ||= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(mode, fn) {
  return db().then(d => new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(out?.result);
    t.onerror = () => reject(t.error);
  }));
}

const idbGet = key => tx('readonly', s => s.get(key));
const idbSet = (key, val) => tx('readwrite', s => s.put(val, key));
const idbKeys = () => tx('readonly', s => s.getAllKeys());
const idbClear = () => tx('readwrite', s => s.clear());

// ───────────── change notifications ─────────────

const listeners = new Set();
let version = 0;
export function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function changed() { version++; listeners.forEach(fn => fn()); }

// ───────────── loading ─────────────

export async function load() {
  try { Object.assign(state.settings, JSON.parse(localStorage.getItem('ri.settings') || '{}')); } catch {}
  const meta = (await idbGet('meta')) || {};
  state.demo = !!meta.demo;
  setRoster((await idbGet('roster')) || []);
  state.days.clear();
  for (const key of await idbKeys()) {
    if (String(key).startsWith('day:')) state.days.set(key.slice(4), await idbGet(key));
  }
  changed();
}

export function saveSettings(patch) {
  Object.assign(state.settings, patch);
  try { localStorage.setItem('ri.settings', JSON.stringify(state.settings)); } catch {}
  changed();
}

function setRoster(list) {
  state.roster = list;
  state.byId = new Map(list.map(s => [s.id, s]));
  state.classes = [...new Set(list.map(classKey))].sort(compareClass);
}

// ───────────── helpers ─────────────

export function today() {
  const d = new Date();
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
export function nowTime() {
  const d = new Date();
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
const pad = n => String(n).padStart(2, '0');

export const classKey = s => `${s.cls}-${s.section}`;

const ROMAN = { I: 1, II: 2, III: 3, IV: 4, V: 5, VI: 6, VII: 7, VIII: 8, IX: 9, X: 10, XI: 11, XII: 12 };
function gradeNum(cls) {
  const c = String(cls).toUpperCase().trim();
  return ROMAN[c] ?? (parseInt(c, 10) || 99);
}
function compareClass(a, b) {
  const [ca, sa = ''] = a.split('-');
  const [cb, sb = ''] = b.split('-');
  return gradeNum(ca) - gradeNum(cb) || sa.localeCompare(sb);
}
const byRoll = (a, b) => (Number(a.roll) || 0) - (Number(b.roll) || 0) || a.name.localeCompare(b.name);

export function studentsIn(key) {
  return state.roster.filter(s => classKey(s) === key).sort(byRoll);
}

export function resolveClass(input) {
  if (!input) return null;
  const norm = String(input).toUpperCase().replace(/^CLASS\s*/, '').replace(/[\s_/]+/g, '-');
  return state.classes.find(k => k.toUpperCase() === norm) || null;
}

export function getDay(date = today()) {
  return state.days.get(date) || {};
}

export function recordFor(id, date = today()) {
  return getDay(date)[id] || null;
}

// ───────────── marking ─────────────

// Marks students and returns an undo token (their previous records).
export async function mark(ids, status, { date = today(), source = 'manual', confidence, time } = {}) {
  if (!STATUS[status]) throw new Error(`Unknown status ${status}`);
  const day = { ...getDay(date) };
  const prev = {};
  const t = time || nowTime();
  for (const id of ids) {
    if (!state.byId.has(id)) continue;
    prev[id] = day[id] || null;
    day[id] = { s: status, t, src: source, ...(confidence != null ? { c: +confidence.toFixed(3) } : {}) };
  }
  state.days.set(date, day);
  await idbSet(`day:${date}`, day);
  changed();
  return { date, prev };
}

export async function undo({ date, prev }) {
  const day = { ...getDay(date) };
  for (const [id, rec] of Object.entries(prev)) {
    if (rec) day[id] = rec; else delete day[id];
  }
  state.days.set(date, day);
  await idbSet(`day:${date}`, day);
  changed();
}

// Status for a gate arrival right now: Present, or Late after the cut-off.
export function arrivalStatus() {
  return nowTime() > (state.settings.lateAfter || '23:59') ? 'L' : 'P';
}

export async function markRemainingAbsent(key = null, date = today()) {
  const day = getDay(date);
  const pool = key ? studentsIn(key) : state.roster;
  const ids = pool.filter(s => !day[s.id]).map(s => s.id);
  if (!ids.length) return { count: 0, undo: null };
  const u = await mark(ids, 'A', { date, source: 'finalize' });
  return { count: ids.length, undo: u };
}

// ───────────── summaries ─────────────

export function classSummary(key, date = today()) {
  const day = getDay(date);
  const out = { cls: key, total: 0, P: 0, L: 0, A: 0, unmarked: 0 };
  for (const s of studentsIn(key)) {
    out.total++;
    const r = day[s.id];
    if (r) out[r.s]++; else out.unmarked++;
  }
  return out;
}

export function daySummary(date = today()) {
  const classes = state.classes.map(k => classSummary(k, date));
  const tot = { date, total: 0, P: 0, L: 0, A: 0, unmarked: 0 };
  for (const c of classes) for (const k of ['total', 'P', 'L', 'A', 'unmarked']) tot[k] += c[k];
  tot.classesComplete = classes.filter(c => c.total && !c.unmarked).length;
  return { ...tot, classes };
}

// Stats for every student, rebuilt only when data changes.
let statsCache = { v: -1, map: new Map(), ranked: [] };
function allStats() {
  if (statsCache.v === version) return statsCache;
  const t = today();
  const dates = [...state.days.keys()].sort().reverse();
  const map = new Map(state.roster.map(s => [s.id, { present: 0, late: 0, absent: 0, days: 0, streak: 0, open: true }]));
  for (const date of dates) {
    const day = state.days.get(date);
    if (!Object.keys(day).length) continue;
    for (const [id, st] of map) {
      const r = day[id];
      if (!r) { if (date !== t) st.open = false; continue; }
      st.days++;
      if (r.s === 'P') st.present++;
      else if (r.s === 'L') st.late++;
      else st.absent++;
      if (st.open) { if (r.s === 'A') st.open = false; else st.streak++; }
    }
  }
  for (const st of map.values()) {
    st.pct = st.days ? Math.round(((st.present + st.late) / st.days) * 1000) / 10 : null;
    delete st.open;
  }
  const ranked = state.roster
    .filter(s => map.get(s.id).days)
    .sort((a, b) => {
      const x = map.get(a.id), y = map.get(b.id);
      return y.pct - x.pct || (y.present + y.late) - (x.present + x.late) || y.streak - x.streak || a.name.localeCompare(b.name);
    });
  ranked.forEach((s, i) => { map.get(s.id).rank = i + 1; });
  statsCache = { v: version, map, ranked };
  return statsCache;
}

export function studentStats(id) {
  const { map, ranked } = allStats();
  const st = map.get(id);
  if (!st) return null;
  const history = [...state.days.keys()].sort().slice(-30).map(d => [d, state.days.get(d)[id]?.s || null]);
  return { ...st, rankOf: ranked.length, history };
}

export function leaderboard({ cls = null, limit = 10 } = {}) {
  const { map, ranked } = allStats();
  return ranked
    .filter(s => !cls || classKey(s) === cls)
    .slice(0, limit)
    .map((s, i) => ({ ...s, ...map.get(s.id), classRank: i + 1 }));
}

export function search(q, limit = 8) {
  const n = String(q || '').trim().toLowerCase();
  if (!n) return [];
  const cls = resolveClass(n);
  if (cls) return studentsIn(cls).slice(0, limit);
  const scored = [];
  for (const s of state.roster) {
    const name = s.name.toLowerCase();
    let score = -1;
    if (String(s.id).toLowerCase() === n) score = 100;
    else if (name === n) score = 90;
    else if (name.startsWith(n)) score = 70;
    else if (name.split(' ').some(w => w.startsWith(n))) score = 60;
    else if (name.includes(n) || String(s.id).toLowerCase().includes(n)) score = 40;
    if (score >= 0) scored.push([score, s]);
  }
  return scored.sort((a, b) => b[0] - a[0] || a[1].name.localeCompare(b[1].name)).slice(0, limit).map(x => x[1]);
}

// ───────────── roster import ─────────────

const COLS = {
  id: ['admission_no', 'admission no', 'adm no', 'adm_no', 'admno', 'admission number', 'id', 'student id', 'student_id'],
  name: ['name', 'student name', 'full name', 'student_name'],
  cls: ['class', 'std', 'standard', 'grade'],
  section: ['section', 'sec', 'div', 'division'],
  roll: ['roll', 'roll no', 'roll_no', 'roll number', 'rollno'],
  house: ['house'],
  photo: ['photo', 'photo_url', 'image', 'picture'],
  blood: ['blood', 'blood group', 'blood_group', 'bloodgroup'],
  father: ['father', 'father name', 'father_name', "father's name", 'fathers name', 'guardian', 'parent'],
  phone: ['phone', 'phone no', 'phone_no', 'mobile', 'mobile no', 'contact', 'contact no'],
  note: ['note', 'notes', 'remarks', 'about'],
};

export function normaliseRoster(rows) {
  if (!rows.length) throw new Error('The file has no rows.');
  const headers = Object.keys(rows[0]);
  const pick = {};
  for (const [field, names] of Object.entries(COLS)) {
    pick[field] = headers.find(h => names.includes(h.trim().toLowerCase()));
  }
  const missing = ['id', 'name', 'cls', 'section'].filter(f => !pick[f]);
  if (missing.length) throw new Error(`Missing column(s): ${missing.map(f => ({ id: 'admission_no', cls: 'class' }[f] || f)).join(', ')}`);

  const seen = new Set();
  const out = [];
  const problems = [];
  rows.forEach((r, i) => {
    const s = {};
    for (const f of Object.keys(COLS)) s[f] = pick[f] ? String(r[pick[f]] ?? '').trim() : '';
    if (!s.id || !s.name) { problems.push(`row ${i + 2}: missing admission no or name`); return; }
    if (seen.has(s.id)) { problems.push(`row ${i + 2}: duplicate admission no ${s.id}`); return; }
    seen.add(s.id);
    s.cls = s.cls.toUpperCase().replace(/^CLASS\s*/, '');
    s.section = s.section.toUpperCase();
    out.push(s);
  });
  return { students: out, problems };
}

export async function importRosterFile(file) {
  if (!window.XLSX) throw new Error('Excel engine not loaded.');
  const wb = XLSX.read(await file.arrayBuffer());
  const rows = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: '' });
  const res = normaliseRoster(rows);
  await replaceRoster(res.students, false);
  return res;
}

async function replaceRoster(list, demo) {
  setRoster(list);
  state.demo = demo;
  await idbSet('roster', list);
  await idbSet('meta', { demo });
  changed();
}

export async function wipe() {
  await idbClear();
  state.days.clear();
  setRoster([]);
  state.demo = false;
  changed();
}

export function backupJson() {
  return JSON.stringify({ roster: state.roster, days: Object.fromEntries(state.days), exported: new Date().toISOString() });
}

// ───────────── demo data (clearly flagged in the UI) ─────────────

export async function loadDemo() {
  const first = ['Aarav', 'Vivaan', 'Aditya', 'Ananya', 'Diya', 'Ishaan', 'Kavya', 'Riya', 'Arjun', 'Saanvi', 'Rohan', 'Priya', 'Aryan', 'Meera', 'Kabir', 'Tanvi', 'Siddharth', 'Nisha', 'Yash', 'Pooja', 'Aniket', 'Sneha', 'Rahul', 'Shreya', 'Dev', 'Ishita', 'Karan', 'Neha', 'Manav', 'Simran'];
  const last = ['Sharma', 'Singh', 'Kumar', 'Mahato', 'Das', 'Gupta', 'Banerjee', 'Mishra', 'Prasad', 'Sinha', 'Agarwal', 'Murmu', 'Sahu', 'Verma', 'Roy', 'Jha', 'Ghosh', 'Pandey', 'Tiwari', 'Oraon'];
  const houses = ['Tagore', 'Raman', 'Bose', 'Kalam'];
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const list = [];
  let adm = 24001;
  for (const cls of ['IX', 'X', 'XI', 'XII']) {
    for (const section of ['A', 'B', 'C', 'D']) {
      for (let roll = 1; roll <= 40; roll++) {
        list.push({
          id: String(adm++),
          name: `${first[Math.floor(rnd() * first.length)]} ${last[Math.floor(rnd() * last.length)]}`,
          cls, section, roll: String(roll),
          house: houses[Math.floor(rnd() * houses.length)],
          photo: '',
        });
      }
    }
  }
  await wipe();
  await replaceRoster(list, true);

  // 20 past school days (Mon–Sat) of history.
  const d = new Date();
  let made = 0;
  while (made < 20) {
    d.setDate(d.getDate() - 1);
    if (d.getDay() === 0) continue;
    const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
    const day = {};
    for (const s of list) {
      const base = 0.8 + (Number(s.id) % 17) / 85;
      const x = rnd();
      const mins = 7 * 60 + 15 + Math.floor(rnd() * 50);
      const t = `${pad(Math.floor(mins / 60))}:${pad(mins % 60)}`;
      day[s.id] = x < base ? { s: t > '08:00' ? 'L' : 'P', t, src: 'demo' } : { s: 'A', t: '09:15', src: 'demo' };
    }
    state.days.set(date, day);
    await idbSet(`day:${date}`, day);
    made++;
  }
  changed();
}

// ───────────── Excel export ─────────────

export function exportExcel(date = today()) {
  if (!window.XLSX) throw new Error('Excel engine not loaded.');
  const wb = XLSX.utils.book_new();
  const sum = daySummary(date);
  const summaryRows = sum.classes.map(c => ({
    Class: c.cls, Strength: c.total, Present: c.P, Late: c.L, Absent: c.A, 'Not marked': c.unmarked,
    'Attendance %': c.total ? Math.round(((c.P + c.L) / c.total) * 1000) / 10 : 0,
  }));
  summaryRows.push({
    Class: 'TOTAL', Strength: sum.total, Present: sum.P, Late: sum.L, Absent: sum.A, 'Not marked': sum.unmarked,
    'Attendance %': sum.total ? Math.round(((sum.P + sum.L) / sum.total) * 1000) / 10 : 0,
  });
  const ws = XLSX.utils.json_to_sheet(summaryRows);
  ws['!cols'] = [{ wch: 10 }, { wch: 10 }, { wch: 9 }, { wch: 7 }, { wch: 8 }, { wch: 11 }, { wch: 13 }];
  XLSX.utils.sheet_add_aoa(ws, [[`Gulmohur High School · Attendance · ${date}`]], { origin: { r: summaryRows.length + 2, c: 0 } });
  XLSX.utils.book_append_sheet(wb, ws, 'Summary');

  const day = getDay(date);
  const src = { cam: 'Gate camera', 'gate-manual': 'Gate (manual)', manual: 'Register', assistant: 'AI assistant', finalize: 'Auto (not seen)', demo: 'Demo' };
  for (const key of state.classes) {
    const rows = studentsIn(key).map(s => {
      const r = day[s.id];
      return {
        Roll: Number(s.roll) || s.roll, 'Admission No': s.id, Name: s.name,
        Status: r ? STATUS[r.s] : 'Not marked', Time: r?.t || '', Source: r ? (src[r.src] || r.src) : '',
        Confidence: r?.c ?? '',
      };
    });
    const sheet = XLSX.utils.json_to_sheet(rows);
    sheet['!cols'] = [{ wch: 6 }, { wch: 13 }, { wch: 26 }, { wch: 11 }, { wch: 7 }, { wch: 16 }, { wch: 11 }];
    XLSX.utils.book_append_sheet(wb, sheet, key.replace(/[\\/?*[\]:]/g, '').slice(0, 31));
  }
  const name = `GHS_Attendance_${date}.xlsx`;
  XLSX.writeFile(wb, name);
  return name;
}
