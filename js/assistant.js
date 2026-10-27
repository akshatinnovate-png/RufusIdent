// RufusIdent AI: an agent on Groq (gpt-oss-120b) that answers questions and
// acts on attendance through tools. Every action it takes shows up in the
// chat with an Undo button.

import * as data from './data.js';
import { state, STATUS } from './data.js';
import { $, esc, toast } from './ui.js';
import { syncNow } from './sync.js';

const API = 'https://api.groq.com/openai/v1';
const MAX_STEPS = 8;

export async function checkGroq() {
  const key = state.settings.groqKey;
  if (!key) return { warn: 'no Groq API key · add it in Settings → Assistant' };
  const res = await fetch(`${API}/models`, { headers: { Authorization: `Bearer ${key}` } });
  if (res.status === 401) throw new Error('Groq rejected the API key');
  if (!res.ok) throw new Error(`Groq HTTP ${res.status}`);
  const { data: models = [] } = await res.json();
  const model = state.settings.model;
  return models.some(m => m.id === model) ? `Groq online · ${model}` : { warn: `Groq online but ${model} not listed for this key` };
}

// ───────────── tools ─────────────

const dateParam = { type: 'string', description: 'YYYY-MM-DD. Omit for today.' };

const TOOLS = [
  {
    name: 'get_day_summary',
    description: 'Whole-school attendance totals for a date, with a per-class breakdown (present, late, absent, not marked).',
    parameters: { type: 'object', properties: { date: dateParam } },
    run: ({ date }) => data.daySummary(date || data.today()),
  },
  {
    name: 'get_class_attendance',
    description: 'Every student in one class with their status for a date. Class looks like "X-A" or "IX-B".',
    parameters: { type: 'object', properties: { class_section: { type: 'string' }, date: dateParam }, required: ['class_section'] },
    run: ({ class_section, date }) => {
      const key = data.resolveClass(class_section);
      if (!key) return { error: `No class "${class_section}". Classes: ${state.classes.join(', ')}` };
      const d = date || data.today();
      const day = data.getDay(d);
      return {
        class: key, date: d, summary: data.classSummary(key, d),
        students: data.studentsIn(key).map(s => ({ id: s.id, roll: s.roll, name: s.name, status: day[s.id] ? STATUS[day[s.id].s] : 'Not marked', time: day[s.id]?.t || null })),
      };
    },
  },
  {
    name: 'list_students_by_status',
    description: 'Students with a given status on a date, optionally limited to one class. Use for "who is absent/late/not marked".',
    parameters: {
      type: 'object',
      properties: { status: { type: 'string', enum: ['present', 'late', 'absent', 'not_marked'] }, class_section: { type: 'string' }, date: dateParam },
      required: ['status'],
    },
    run: ({ status, class_section, date }) => {
      const key = class_section ? data.resolveClass(class_section) : null;
      if (class_section && !key) return { error: `No class "${class_section}".` };
      const day = data.getDay(date || data.today());
      const code = { present: 'P', late: 'L', absent: 'A' }[status];
      const pool = key ? data.studentsIn(key) : state.roster;
      const hits = pool.filter(s => (code ? day[s.id]?.s === code : !day[s.id]));
      return {
        count: hits.length,
        students: hits.slice(0, 120).map(s => ({ id: s.id, name: s.name, class: data.classKey(s), roll: s.roll, time: day[s.id]?.t || null })),
        truncated: hits.length > 120,
      };
    },
  },
  {
    name: 'find_students',
    description: 'Search students by name, admission number, or class. Returns up to 10.',
    parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
    run: ({ query }) => data.search(query, 10).map(s => ({ id: s.id, name: s.name, class: data.classKey(s), roll: s.roll, today: data.recordFor(s.id)?.s ? STATUS[data.recordFor(s.id).s] : 'Not marked' })),
  },
  {
    name: 'get_student_profile',
    description: 'Attendance record for one student: percentage, days present/late/absent, current streak, leaderboard rank, last 30 days.',
    parameters: { type: 'object', properties: { student_id: { type: 'string' } }, required: ['student_id'] },
    run: ({ student_id }) => {
      const s = state.byId.get(String(student_id));
      if (!s) return { error: `No student with admission no ${student_id}` };
      const { history, ...st } = data.studentStats(s.id);
      return { ...s, class: data.classKey(s), ...st, last_30_days: history.map(([d, c]) => `${d}:${c || '-'}`).join(' ') };
    },
  },
  {
    name: 'get_leaderboard',
    description: 'Top students by attendance %, whole school or one class.',
    parameters: { type: 'object', properties: { class_section: { type: 'string' }, limit: { type: 'integer', minimum: 1, maximum: 50 } } },
    run: ({ class_section, limit = 10 }) => {
      const key = class_section ? data.resolveClass(class_section) : null;
      return data.leaderboard({ cls: key, limit: Math.min(limit, 50) }).map(s => ({ rank: key ? s.classRank : s.rank, id: s.id, name: s.name, class: data.classKey(s), pct: s.pct, present: s.present + s.late, absent: s.absent, streak: s.streak }));
    },
  },
  {
    name: 'mark_attendance',
    description: 'Set attendance for one or more students (by admission number). Only do this when the user clearly asks to change attendance.',
    parameters: {
      type: 'object',
      properties: {
        student_ids: { type: 'array', items: { type: 'string' } },
        status: { type: 'string', enum: ['present', 'late', 'absent'] },
        date: dateParam,
      },
      required: ['student_ids', 'status'],
    },
    run: async ({ student_ids, status, date }, ui) => {
      const ids = student_ids.map(String);
      const unknown = ids.filter(id => !state.byId.has(id));
      const ok = ids.filter(id => state.byId.has(id));
      if (!ok.length) return { error: 'None of those admission numbers exist.', unknown };
      const code = { present: 'P', late: 'L', absent: 'A' }[status];
      const u = await data.mark(ok, code, { date: date || data.today(), source: 'assistant' });
      const names = ok.map(id => state.byId.get(id).name);
      ui.action(`Marked ${names.length > 3 ? `${names.length} students` : names.join(', ')} ${status}${date ? ` on ${date}` : ''}`, u);
      return { marked: ok.length, status, unknown };
    },
  },
  {
    name: 'mark_unmarked_absent',
    description: 'Close the register: mark everyone not yet marked as absent, for one class or the whole school. Only when the user asks.',
    parameters: { type: 'object', properties: { class_section: { type: 'string' }, date: dateParam } },
    run: async ({ class_section, date }, ui) => {
      const key = class_section ? data.resolveClass(class_section) : null;
      if (class_section && !key) return { error: `No class "${class_section}".` };
      const { count, undo } = await data.markRemainingAbsent(key, date || data.today());
      if (count) ui.action(`Marked ${count} unmarked student(s) absent${key ? ` in ${key}` : ''}`, undo);
      return { marked_absent: count, scope: key || 'whole school' };
    },
  },
  {
    name: 'export_excel',
    description: 'Download the class-wise attendance Excel workbook (summary sheet + one sheet per class) for a date.',
    parameters: { type: 'object', properties: { date: dateParam } },
    run: ({ date }, ui) => {
      const file = data.exportExcel(date || data.today());
      ui.action(`Downloaded ${file}`);
      return { file };
    },
  },
  {
    name: 'email_report',
    description: 'Email the daily attendance Excel report now to the configured recipients (the same report that is sent automatically every school day). Only when the user asks to send/email it.',
    parameters: { type: 'object', properties: { date: dateParam } },
    run: async ({ date }, ui) => {
      const url = state.settings.backendUrl?.replace(/\/$/, '');
      if (!url) return { error: 'No backend configured.' };
      await syncNow();
      const res = await fetch(url + '/report/send', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ date: date || data.today() }) });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) return { error: j.detail || `HTTP ${res.status}` };
      ui.action(`Emailed the ${j.date} report to ${j.recipients.join(', ')}`);
      return j;
    },
  },
  {
    name: 'report_schedule',
    description: 'When the daily report email goes out, to whom, and whether the last one was sent.',
    parameters: { type: 'object', properties: {} },
    run: async () => {
      const url = state.settings.backendUrl?.replace(/\/$/, '');
      if (!url) return { error: 'No backend configured.' };
      const [s, st] = await Promise.all([fetch(url + '/report/settings').then(r => r.json()), fetch(url + '/report/status').then(r => r.json())]);
      return { enabled: s.enabled, recipients: s.recipients, time: s.time, days: s.days.map(d => ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d]), next_run: st.next_run, problem: st.problem, recent: st.history.slice(-5) };
    },
  },
  {
    name: 'open_screen',
    description: 'Open a screen for the user: gate camera, class register, student datasheet, leaderboard, or class list.',
    parameters: {
      type: 'object',
      properties: { screen: { type: 'string', enum: ['camera', 'classes', 'class', 'student', 'leaderboard'] }, id: { type: 'string', description: 'Class (for class) or admission no (for student).' } },
      required: ['screen'],
    },
    run: ({ screen, id }, ui) => {
      let hash = `#/${screen}`;
      if (screen === 'class') { const k = data.resolveClass(id); if (!k) return { error: 'Unknown class' }; hash += `/${encodeURIComponent(k)}`; }
      if (screen === 'student') { if (!state.byId.has(String(id))) return { error: 'Unknown student' }; hash += `/${encodeURIComponent(id)}`; }
      ui.action(`Opened ${screen}${id ? ` ${id}` : ''}`);
      setTimeout(() => { location.hash = hash; }, 600);
      return { opened: hash };
    },
  },
];

const TOOL_MAP = new Map(TOOLS.map(t => [t.name, t]));
const TOOL_DEFS = TOOLS.map(({ name, description, parameters }) => ({ type: 'function', function: { name, description, parameters } }));

function systemPrompt() {
  const d = new Date();
  return `You are RufusIdent AI, the attendance assistant of Gulmohur High School (High School section), Telco Town, Jamshedpur, Jharkhand.
Today is ${d.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })} (${data.today()}), time ${data.nowTime()}. Students arriving after ${state.settings.lateAfter} are marked Late.
The school has ${state.roster.length} students in ${state.classes.length} classes: ${state.classes.join(', ') || 'none imported yet'}.${state.demo ? ' NOTE: the roster is DEMO data, say so if asked about real students.' : ''}

Rules:
- Always use tools for numbers, names and statuses. Never invent or estimate attendance.
- Change attendance only when the user clearly asks. After an action, say exactly what changed.
- If a name matches several students, list them and ask which one.
- Be brief. Use markdown tables for lists of students. Use Indian number formatting.
- You were built by the GHS AI Team; lead developer Akshat Sarkar.`;
}

// ───────────── chat state + UI ─────────────

let messages = [];
let busy = false;

export function resetChat() {
  messages = [];
  $('#thread').innerHTML = '';
  $('#thread').hidden = true;
  $('#homeHero').hidden = false;
}

function bubble(role, html) {
  const thread = $('#thread');
  thread.hidden = false;
  $('#homeHero').hidden = true;
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = html;
  thread.append(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

const ui = {
  action(text, undoToken) {
    const el = bubble('action', `<span class="act-ico">⚡</span><span>${esc(text)}</span>`);
    if (undoToken) {
      const b = document.createElement('button');
      b.textContent = 'Undo';
      b.onclick = async () => { await data.undo(undoToken); b.replaceWith(Object.assign(document.createElement('em'), { textContent: 'undone' })); };
      el.append(b);
    }
  },
};

async function callGroq() {
  const res = await fetch(`${API}/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${state.settings.groqKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: state.settings.model || 'openai/gpt-oss-120b',
      messages: [{ role: 'system', content: systemPrompt() }, ...messages],
      tools: TOOL_DEFS,
      tool_choice: 'auto',
      reasoning_effort: 'low',
      temperature: 0.3,
    }),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error?.message || msg; } catch {}
    throw new Error(msg);
  }
  return (await res.json()).choices[0].message;
}

export async function ask(text) {
  text = text.trim();
  if (!text || busy) return;
  if (!state.settings.groqKey) {
    toast('Add your Groq API key in Settings → Assistant first.', { kind: 'warn' });
    document.querySelector('[data-act="settings"]').click();
    return;
  }
  busy = true;
  $('#sendBtn').disabled = true;
  bubble('user', esc(text));
  const start = messages.length;
  messages.push({ role: 'user', content: text });
  const thinking = bubble('bot thinking', '<span class="dots3"><i></i><i></i><i></i></span>');

  try {
    for (let step = 0; step < MAX_STEPS; step++) {
      const msg = await callGroq();
      const calls = msg.tool_calls || [];
      messages.push({ role: 'assistant', content: msg.content || '', ...(calls.length ? { tool_calls: calls } : {}) });
      if (!calls.length) {
        thinking.remove();
        bubble('bot', md(msg.content || '(no answer)'));
        return;
      }
      for (const call of calls) {
        const tool = TOOL_MAP.get(call.function.name);
        let result;
        try {
          const args = JSON.parse(call.function.arguments || '{}');
          result = tool ? await tool.run(args, ui) : { error: `Unknown tool ${call.function.name}` };
        } catch (e) {
          result = { error: e.message };
        }
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
      }
      thinking.parentElement?.append(thinking); // keep the dots at the bottom
    }
    thinking.remove();
    bubble('bot', md('I stopped after several steps without a final answer. Try asking more specifically.'));
  } catch (e) {
    thinking.remove();
    bubble('bot error', `Couldn't reach Groq: ${esc(e.message)}`);
    messages.length = start;
  } finally {
    busy = false;
    $('#sendBtn').disabled = false;
  }
}

// ───────────── tiny, safe markdown ─────────────

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/(^|[^*])\*([^*]+)\*/g, '$1<i>$2</i>');
}

export function md(src) {
  const lines = esc(src).split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || '')) {
      const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<div class="tbl"><table><thead><tr>${head.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    if (/^\s*[-*•]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) items.push(inline(lines[i++].replace(/^\s*[-*•]\s+/, '')));
      out.push(`<ul>${items.map(x => `<li>${x}</li>`).join('')}</ul>`);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) items.push(inline(lines[i++].replace(/^\s*\d+\.\s+/, '')));
      out.push(`<ol>${items.map(x => `<li>${x}</li>`).join('')}</ol>`);
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) out.push(`<h4>${inline(h[2])}</h4>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    i++;
  }
  return out.join('');
}

// ───────────── voice input (Chrome / Edge) ─────────────

export function wireMic(onText) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $('#micBtn');
  if (!SR) { btn.hidden = true; return; }
  let rec = null;
  btn.addEventListener('click', () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = 'en-IN';
    rec.interimResults = true;
    btn.classList.add('on');
    rec.onresult = e => onText([...e.results].map(r => r[0].transcript).join(''), e.results[e.results.length - 1].isFinal);
    rec.onend = () => { btn.classList.remove('on'); rec = null; };
    rec.start();
  });
}
