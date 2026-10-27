// Small DOM helpers shared by every screen.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function initials(name) {
  return String(name).split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join('').toUpperCase();
}

export function pct(n, d) {
  return d ? Math.round((n / d) * 1000) / 10 : 0;
}

export function prettyDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
}

// Photo if the roster has one, otherwise a monogram.
export function avatar(s, cls = 'avatar') {
  if (s.photo) return `<img class="${cls}" src="${esc(s.photo)}" alt="" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'${cls} mono',textContent:'${esc(initials(s.name))}'}))">`;
  return `<span class="${cls} mono">${esc(initials(s.name))}</span>`;
}

export function toast(msg, { action, onAction, kind = '' } = {}) {
  const box = document.createElement('div');
  box.className = `toast ${kind}`;
  box.innerHTML = `<span>${esc(msg)}</span>`;
  if (action) {
    const b = document.createElement('button');
    b.textContent = action;
    b.onclick = () => { onAction?.(); box.remove(); };
    box.append(b);
  }
  $('#toasts').append(box);
  setTimeout(() => box.classList.add('out'), 5200);
  setTimeout(() => box.remove(), 5600);
}

export const STATUS_CLASS = { P: 'st-p', L: 'st-l', A: 'st-a' };
