// Keeps the backend's copy of the roster and attendance up to date, so it can
// email the daily report on its own. Only changed days are sent: each day's
// hash is compared with the hashes the backend already has.

import { state, onChange } from './data.js';

const hashes = new WeakMap();   // day object -> hash (days are replaced, never mutated)
let known = null;               // { roster_hash, days: {date: hash} } from the backend
let timer = null, busy = false;
export const syncStatus = { ok: null, at: null, error: null };

function fnv(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16) + ':' + str.length;
}

function hashOf(obj) {
  if (!hashes.has(obj)) hashes.set(obj, fnv(JSON.stringify(obj)));
  return hashes.get(obj);
}

const base = () => state.settings.backendUrl?.replace(/\/$/, '');

async function post(path, body) {
  const res = await fetch(base() + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}

export async function syncNow() {
  if (!base()) return;
  if (busy) { schedule(); return; }
  busy = true;
  try {
    if (!known) known = await (await fetch(base() + '/sync/state', { signal: AbortSignal.timeout(5000) })).json();
    const rosterHash = fnv(JSON.stringify(state.roster)) + (state.demo ? ':demo' : '');
    if (state.roster.length && known.roster_hash !== rosterHash) {
      await post('/sync/roster', { roster: state.roster, demo: state.demo, hash: rosterHash });
      known.roster_hash = rosterHash;
    }
    for (const [date, day] of state.days) {
      const h = hashOf(day);
      if (known.days[date] !== h) {
        await post(`/sync/day/${date}`, { records: day, hash: h });
        known.days[date] = h;
      }
    }
    Object.assign(syncStatus, { ok: true, at: new Date(), error: null });
  } catch (e) {
    known = null;                                  // re-read the backend's state next time
    Object.assign(syncStatus, { ok: false, error: e.message });
    clearTimeout(timer);
    timer = setTimeout(syncNow, 15000);
  } finally {
    busy = false;
  }
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(syncNow, 2000);
}

export function startSync() {
  onChange(schedule);
  syncNow();
}
