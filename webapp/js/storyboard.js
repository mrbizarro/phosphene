// webapp/js/storyboard.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ============================================================================
// STORYBOARD
// ============================================================================
// One concept in, a film's worth of shots out. Everything here talks to
// /storyboard/*; nothing here renders, queues, or tracks progress on its own —
// a shot is an ordinary job and the bottom pane already knows how to show one.
//
// The board poller lives here and ONLY runs while the tab is open (2 s, its own
// timer). poll() at 1.5 s is already doing a lot and this must not ride on it.
// The single cross-tab cost is one call at the end of poll(): sbPollHook().

globalThis.SB = {
  id: '',                 // the open board, '' = list/empty
  payload: null,          // the last /storyboard/get reply
  timer: null,            // the 2 s board poller
  saveTimer: null,
  saveInFlight: false,
  saveAgain: false,
  // 'auto' (the list) | 'list' | 'player'. Restored from the session so a reload does
  // not re-dock a preview the user closed two minutes ago.
  stageMode: (function () {
    try { return sessionStorage.getItem('phos.sb.stageMode') || 'auto'; }
    catch (e) { return 'auto'; }
  })(),
  primed: false,          // brief defaults are read from BOOT once, not per entry
  boards: [],             // /status.storyboards, refreshed by poll()
  lastUndo: null,
  // Which of the seven stage states is on screen. sbShow() is the only writer.
  // The rail reads it to say where you are, and sbLoad() reads it so a 2 s poll
  // cannot yank someone off the film screen back onto the shot list.
  stage: '',
  films: [],              // /storyboard/films for the open board, newest first
  filmsFor: '',           // which board `films` belongs to — never another one
  filmDir: '',
  filmShort: '',
  filmOpen: '',           // the film being played, by name
  boardsSig: '',          // last painted board-row signature (see sbPollHook)
};
const SB_BOOT = (typeof BOOT !== 'undefined' && BOOT.storyboard) ? BOOT.storyboard : {};

// The validator's structured errors, in human. Keyed on `code`, never on the
// message text — that is the whole reason validate_storyboard_detail() exists.
// `fix` names an action sbFixError() knows how to run.
const SB_ERR_COPY = {
  schema_version:      { html: '<b>This storyboard was written by a different version of Phosphene.</b> Nothing here can be rendered safely. Plan a new one.' },
  board_id_empty:      { html: '<b>This storyboard file is damaged</b> and can\'t be repaired from the panel.' },
  no_shots:            { html: '<b>No shots.</b> Re-plan, or add one by hand.', fix: 'add', label: 'Add a shot' },
  shot_not_object:     { html: (e) => `<b>Shot ${e.n} is unreadable.</b> Delete it and add a new one.`, fix: 'delete', label: 'Delete shot' },
  shot_number:         { html: (e) => `<b>Shot ${e.n} lost its number.</b>`, fix: 'renumber', label: 'Renumber all shots' },
  shot_duplicate:      { html: (e) => `<b>Two shots are both numbered ${e.n}.</b>`, fix: 'renumber', label: 'Renumber all shots' },
  bad_mode:            { html: (e) => `<b>Shot ${e.n} asks for a shot type this build doesn't have.</b> Set it to Text or Character.`, fix: 'text', label: 'Make it Text' },
  empty_prompt:        { html: (e) => `<b>Shot ${e.n} has no prompt.</b> Write what happens in it.`, fix: 'focus', label: 'Write it' },
  unknown_character:   { html: (e) => `<b>Shot ${e.n} casts <code>${escapeHtml(e.data.character_id || '')}</code>, who isn't on this Mac.</b> Pick someone installed, or train them first.`, fix: 'pickchar', label: 'Pick someone' },
  missing_trigger:     { html: (e) => `<b>Shot ${e.n}'s prompt doesn't say <code>${escapeHtml(e.data.trigger || '')}</code>,</b> so the trained face won't load and a stranger renders instead.`, fix: 'trigger', label: 'Put it back' },
  character_without_id:{ html: (e) => `<b>Shot ${e.n} is a Character shot with nobody cast.</b>`, fix: 'pickchar', label: 'Pick someone' },
  bad_duration:        { html: (e) => `<b>Shot ${e.n} is ${escapeHtml(String(e.data.duration_s))} seconds long.</b> Pick something between 1 and 60.`, fix: 'dur5', label: 'Make it 5 s' },
  refs_not_list:       { html: (e) => `<b>Shot ${e.n}'s reference images are damaged.</b>`, fix: 'clearrefs', label: 'Clear them' },
  ref_missing:         { html: (e) => `<b>Shot ${e.n} wants a reference image that isn't there any more:</b> <code>${escapeHtml(e.data.name || '')}</code>`, fix: 'clearrefs', label: 'Clear them' },
  remix_needs_ref:     { html: (e) => `<b>Shot ${e.n} is a Remix shot with no reference image.</b>`, fix: 'text', label: 'Make it Text' },
  // The offer is the server's own `fit_*` — the largest canvas THIS Mac may
  // render at the quality already chosen. It used to read "Use 1024×576" on
  // every machine, which on a Mac that caps at 768 offered a size the validator
  // rejects, and the button's guard (`> 1024`) meant it wrote nothing at all:
  // the error never cleared and Render stayed disabled forever. (GitHub #71)
  over_cap:            { html: (e) => `<b>${e.data.pass_name === 'final' ? 'Delivery' : 'Draft'} is set to ${e.data.width}×${e.data.height}; this Mac caps at ${e.data.max_dim}.</b> It'll shrink to fit unless you lower it.`,
                         fix: 'cap',
                         label: (e) => `Use ${e.data.fit_width}×${e.data.fit_height}` },
};

function sbEl(id) { return document.getElementById(id); }

// Wall clock, at the scale a film is measured in. Wraps fmtMin so it never
// shows seconds when the answer is hours.
function sbFmtWall(secs) {
  if (!secs || secs < 0) return '—';
  if (secs < 90) return `${Math.round(secs)} s`;
  const m = Math.round(secs / 60);
  if (m < 90) return `about ${m} m`;
  const h = Math.floor(m / 60), rem = m % 60;
  return rem ? `about ${h} h ${rem} m` : `about ${h} h`;
}
function sbFmtRuntime(secs) {
  if (!secs) return '';
  if (secs <= 90) return `${Math.round(secs)} s of film`;
  const m = Math.floor(secs / 60), s = Math.round(secs % 60);
  return `${m} m ${s} s of film`;
}
function sbShotEst(secs) {
  if (!secs) return '';
  return secs < 90 ? `~${Math.round(secs)} s` : `~${Math.round(secs / 60)} m`;
}
// A film's length is read off a player, so it is written the way a player
// writes it — 0:31, 2:04 — not "31 s of film".
function sbFmtClock(secs) {
  const t = Math.max(0, Math.round(Number(secs) || 0));
  const m = Math.floor(t / 60), s = t % 60;
  return m + ':' + (s < 10 ? '0' : '') + s;
}
function sbFmtBytes(n) {
  const b = Number(n) || 0;
  if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
  if (b >= 1048576) return Math.round(b / 1048576) + ' MB';
  if (b >= 1024) return Math.round(b / 1024) + ' KB';
  return b + ' B';
}
// "when was this made" in the only units that answer it without arithmetic.
// `now` is a parameter so the gate can pin it.
function sbFmtAgo(ts, now) {
  const t = Number(ts) || 0;
  if (!t) return '';
  const secs = Math.max(0, ((now || Date.now()) / 1000) - t);
  if (secs < 90) return 'just now';
  const m = Math.round(secs / 60);
  if (m < 60) return m + (m === 1 ? ' minute ago' : ' minutes ago');
  const h = Math.round(m / 60);
  if (h < 24) return h + (h === 1 ? ' hour ago' : ' hours ago');
  const d = Math.round(h / 24);
  if (d < 30) return d + (d === 1 ? ' day ago' : ' days ago');
  try { return new Date(t * 1000).toLocaleDateString(); } catch (e) { return ''; }
}
// Which button made this file. Two mp4s in one folder otherwise look identical.
function sbFilmKind(f) {
  const k = (f || {}).kind;
  if (k === 'timeline') return 'Timeline render';
  if (k === 'export') return 'Export';
  return 'Film';
}
// The film on screen: the one the caller asked for by name, else the newest.
// Pure, because "which film am I looking at" is exactly the kind of thing that
// silently picks the wrong one.
function sbFilmPick(films, want) {
  const list = films || [];
  if (!list.length) return null;
  if (want) {
    const hit = list.filter(f => f.name === want || f.path === want)[0];
    if (hit) return hit;
  }
  return list[0];
}

// ---- tab lifecycle ---------------------------------------------------------
// The delivery-pass chips, from the registry. Three static buttons carried
// hardcoded canvases (two of them wrong — Standard is 1280×704) and could never
// show a tier the registry grew. Filtered to the server's own final_qualities
// list, labelled and sized from the cell, and stamped with the pack so the
// q4-tier CSS gate hides every q8 tier rather than the one called "high".
// The sub-label is the storyboard's OWN canvas for that tier, clamped to this
// Mac (SB_BOOT.canvases) — not the registry's `c.canvas`, which describes a
// single Manual render and disagrees (Standard is 1280×704 there, 1024×576
// here). The chip must say what the click writes: printing one number and
// writing another is how a 24 GB Mac ended up with an over-cap delivery pass
// it never chose. (GitHub #71)
function sbQualityChips(boxId, allowed, current) {
  const box = sbEl(boxId);
  if (!box) return;
  const cells = ((BOOT.ltx || {}).qualities) || [];
  const cv = SB_BOOT.canvases || {};
  const list = (Array.isArray(cells) ? cells : Object.values(cells))
    .filter(c => c && (allowed || []).indexOf(c.key) !== -1);
  if (!list.length) return;
  box.innerHTML = list.map(c => {
    const on = c.key === current;
    const size = cv[c.key] ? `${cv[c.key].width}×${cv[c.key].height}` : c.canvas;
    const sub = `${size}${c.pack === 'q8' ? ' · Q8' : ''}`;
    return `<button type="button" class="pill-btn${on ? ' active' : ''}" `
         + `data-q="${escapeHtml(c.key)}" data-pack="${escapeHtml(c.pack || '')}">`
         + `${escapeHtml(c.label)}<span class="sub">${escapeHtml(sub)}</span></button>`;
  }).join('');
}

// The ACTIVE chip is applied later from the board's own policy (sbRenderPlan),
// so these only need a sane pre-board default.
function sbRenderFinalQualities() {
  sbQualityChips('sbFinalQuality', SB_BOOT.final_qualities,
                 (SB_BOOT.defaults || {}).final_quality || 'standard');
}
function sbRenderDraftQualities() {
  sbQualityChips('sbDraftQuality', SB_BOOT.draft_qualities,
                 (SB_BOOT.defaults || {}).draft_quality || 'quick');
}

function sbInit() {
  sbRenderDraftQualities();
  sbRenderFinalQualities();
  const help = sbEl('sbRamHelpNote');
  if (help && !help.textContent) help.textContent = SB_BOOT.ram_help || '';
  // Draft restore — a restart must not eat what someone typed.
  try {
    const draft = localStorage.getItem('phos_sb_draft');
    const box = sbEl('sbConcept');
    if (draft && box && !box.value) box.value = draft;
  } catch (e) {}
  // Defaults are read from the bootstrap ONCE. sbInit() runs on every tab entry,
  // and SB_BOOT was captured at page load — re-applying it on re-entry silently
  // reset Shots and Engine to stale values while disk held the real ones, and
  // the UI is what sbPlan() submits. After the first entry the in-session choice
  // IS the truth.
  if (!SB.primed) {
    const d = SB_BOOT.defaults || {};
    _sbShots = Number(d.shots) || 12;
    _sbEngineMode = d.engine || 'auto';
    SB.primed = true;
  }
  sbSetShots(_sbShots, false);
  sbRenderEnginePicker();
  if (typeof refreshManualCharacters === 'function') {
    Promise.resolve(refreshManualCharacters()).then(sbRenderCast).catch(() => sbRenderCast());
  } else { sbRenderCast(); }
  sbConceptInput();
  sbMustInput();
  // Restore the last board the user was on, same idiom as phos_workflow.
  let last = '';
  try { last = localStorage.getItem('phos_sb_open') || ''; } catch (e) {}
  sbRefreshBoards().then(() => {
    if (last && SB.boards.some(b => b.id === last)) sbOpen(last);
    else if (SB.boards.length) sbShow('list');
    else sbShow('empty');
  });
  if (SB.timer) clearInterval(SB.timer);
  SB.timer = setInterval(sbTick, 2000);
}

function sbTeardown() {
  // The board poller, and nothing else. It used to tear the EDITOR down too,
  // back when the timeline was a state of this tab — so leaving the storyboard
  // closed a document that had nothing to do with it.
  if (SB.timer) { clearInterval(SB.timer); SB.timer = null; }
}

async function sbRefreshBoards() {
  try {
    const r = await (await fetch('/storyboard/list')).json();
    SB.boards = r.boards || [];
  } catch (e) { SB.boards = []; }
  sbRenderBoardLists();
  return SB.boards;
}

// ---- the editing guard -----------------------------------------------------
// The shot list is innerHTML-rebuilt on every repaint, and the board poller
// repaints every 2 s. Without these two guards, typing into a prompt loses
// focus and then loses the text: the tick lands INSIDE the 800 ms save debounce,
// replaces the textarea with the server's copy, and the debounce then writes the
// reverted value back to disk. Measured 5/5 by the validator, and "change any
// prompt" is one of the four promises printed on the empty state.
//
// Two different questions, deliberately:
//   sbTypingInShots() — is the user mid-word? Blocks the DOM rebuild only. A
//     click on ↑ / a select / a grade button must still repaint, and by then the
//     focused element is a button or a closed select, not a text field.
//   sbHoldingShots()  — is the card busy at all (text OR an open select menu)?
//     Blocks the POLL, so nothing is fetched-and-repainted under an open menu
//     and no server copy can clobber an edit that hasn't been saved yet.
function sbTypingInShots() {
  const a = document.activeElement;
  if (!a || !a.closest || !a.closest('#sbShots')) return false;
  const t = (a.tagName || '').toUpperCase();
  return t === 'TEXTAREA' || (t === 'INPUT' && a.type !== 'button');
}
// AN OPEN <select> IS NOT RELIABLY document.activeElement. In the Pinokio app
// window the dropdown is a NATIVE menu in its own window, so the page loses
// focus the instant it opens and activeElement falls back to <body> — the guard
// below then read "nobody is holding", the 2 s poll rebuilt the shot list, and
// the menu the user was reading blinked shut. Every time. In a browser tab the
// select keeps document focus, which is exactly why the same build worked at
// 127.0.0.1:8198 and not in the app. (GitHub #71)
//
// So we track the interaction ourselves rather than asking the document who has
// focus. Set on pointerdown, cleared by the change (a choice) or by the next
// pointerdown anywhere else (a dismissal); the ceiling exists only so a lost
// clear cannot freeze the board poller for good.
let _sbSelectOpenUntil = 0;
const SB_SELECT_HOLD_MS = 20000;
function sbSelectOpen() { return Date.now() < _sbSelectOpenUntil; }
document.addEventListener('pointerdown', (ev) => {
  const t = ev.target;
  _sbSelectOpenUntil = (t && t.closest && t.closest('#sbShots select'))
    ? Date.now() + SB_SELECT_HOLD_MS : 0;
}, true);
document.addEventListener('change', (ev) => {
  const t = ev.target;
  if (t && t.closest && t.closest('#sbShots select')) _sbSelectOpenUntil = 0;
}, true);

function sbHoldingShots() {
  if (sbSelectOpen()) return true;
  const a = document.activeElement;
  if (a && a.closest && a.closest('#sbShots')) {
    const t = (a.tagName || '').toUpperCase();
    if (t === 'TEXTAREA' || t === 'SELECT' || t === 'INPUT') return true;
  }
  // An edit typed but not yet flushed is exactly as unsafe to overwrite.
  return !!SB.saveTimer || SB.saveInFlight;
}

async function sbTick() {
  if (document.body.dataset.workflow !== 'storyboard') return;
  // The global poll skips hidden tabs; this one should too.
  if (document.hidden) return;
  // The timeline has its own clock and its own document. Repainting the shot
  // list under it would be work nobody can see, and sbLoad() reaches into the
  // player — which the timeline is currently using.
  if (typeof SBE !== 'undefined' && SBE.open) return;
  if (sbHoldingShots()) {
    // Don't fetch over someone's typing. The run bar is the one thing that
    // moves on its own, and it lives outside #sbShots, so keep it live.
    if (SB.id && SB.payload) { try { sbRenderRunBar(SB.payload); } catch (e) {} }
    return;
  }
  if (SB.id) await sbLoad(SB.id, true);
  else await sbRefreshBoards();
}

// ---- the brief -------------------------------------------------------------
let _sbDraftTimer = null;
function sbConceptInput() {
  const box = sbEl('sbConcept');
  const btn = sbEl('sbPlanBtn');
  if (btn) {
    const empty = !box || !box.value.trim();
    btn.disabled = empty || btn.dataset.busy === '1';
    if (empty) btn.title = 'Write a couple of sentences about the film first.';
    else if (btn.dataset.busy !== '1') btn.title = '';
  }
  if (_sbDraftTimer) clearTimeout(_sbDraftTimer);
  _sbDraftTimer = setTimeout(() => {
    try { localStorage.setItem('phos_sb_draft', (box && box.value) || ''); } catch (e) {}
  }, 400);
}

function sbMustInput() {
  const box = sbEl('sbMust');
  const meta = sbEl('sbMustMeta');
  if (!meta) return;
  const n = ((box && box.value) || '').split('\n').filter(x => x.trim()).length;
  meta.textContent = n === 0 ? 'none' : (n === 1 ? '1 shot' : `${n} shots`);
}

// LOCATIONS. Free text, one per line, `name: description`. A list of objects
// behind a form builder would be more correct and nobody would fill it in;
// this is the shape people already type into the Look field.
function sbParseLocations(text) {
  const out = [];
  const seen = {};
  ((text || '').split('\n')).forEach(line => {
    const raw = line.trim();
    if (!raw) return;
    const at = raw.indexOf(':');
    // No colon = the whole line is the description and the name is derived.
    // Refusing the line instead would lose what somebody just typed.
    const name = (at > 0 ? raw.slice(0, at) : raw.split(/[,.]/)[0]).trim().slice(0, 60);
    const desc = (at > 0 ? raw.slice(at + 1) : raw).trim();
    if (!name || !desc) return;
    let id = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);
    if (!id) id = 'loc' + (out.length + 1);
    if (!/^[a-z0-9]/.test(id)) id = 'l' + id;
    while (seen[id]) id = id.replace(/\d*$/, m => String((parseInt(m || '1', 10) || 1) + 1));
    seen[id] = 1;
    out.push({ id: id, name: name, description: desc });
  });
  return out;
}

function sbLocInput() {
  const meta = sbEl('sbLocMeta');
  if (!meta) return;
  const locs = sbParseLocations((sbEl('sbLocations') || {}).value);
  meta.textContent = locs.length === 0
    ? 'no locations — every shot invents its own'
    : locs.map(l => l.name).join(' · ') +
      (locs.length === 1 ? ' — 1 place, pinned on every shot that uses it'
                         : ` — ${locs.length} places, pinned on every shot that uses them`);
}

let _sbShots = 12;
// ONE TAKE is the fifth answer on the Shots row: a film that is one shot,
// planned as beats. `_sbTake` is 0 (off) or the take's length in seconds.
let _sbTake = 0;
function sbSetShots(n, persist) {
  const take = (n === 'take');
  if (take) { if (!_sbTake) _sbTake = 60; }
  else { _sbTake = 0; _sbShots = Number(n) || 12; }
  document.querySelectorAll('#sbLengthGroup .q-chip').forEach(b =>
    b.classList.toggle('active', take ? b.dataset.sbShots === 'take' : Number(b.dataset.sbShots) === _sbShots));
  const row = sbEl('sbTakeRow');
  if (row) row.hidden = !take;
  if (take) sbSetTake(_sbTake);
  if (persist !== false && !take) sbSaveSetting('storyboard_shots', _sbShots);
}
function sbSetTake(secs) {
  _sbTake = Number(secs) || 60;
  document.querySelectorAll('#sbTakeGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', Number(b.dataset.sbTake) === _sbTake));
}
function sbShotsValue() {
  if (_sbTake) return Math.round(_sbTake / 5);
  const on = document.querySelector('#sbLengthGroup .q-chip.active');
  return on ? Number(on.dataset.sbShots) : _sbShots;
}
async function sbSaveSetting(key, value) {
  try {
    const fd = new URLSearchParams(); fd.set(key, String(value));
    await fetch('/settings', { method: 'POST', body: fd });
  } catch (e) {}
}

function sbToggleSwitchHelp() {
  const btn = sbEl('sbSwitchHelpBtn'), note = sbEl('sbSwitchHelpNote');
  if (!btn || !note) return;
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  note.hidden = open;
}
// The long-shot switch needs the Q8 pack (the chain runs on the dev
// transformer). Say so on the switch, from the status the panel already has,
// rather than at render time per shot.
function sbSyncBriefGates() {
  const el = sbEl('sbLongWindows');
  if (!el) return;
  const s = globalThis.LAST_STATUS;
  if (!s) return;
  const ok = !!s.q8_available;
  el.disabled = !ok;
  if (!ok && el.checked) el.checked = false;
  const lab = el.closest('label');
  if (lab) lab.title = ok
    ? 'A shot longer than 5 seconds renders as a chain of 5-second passes, each continuing from the last, instead of being shortened to fit one. Slower. LTX shots only.'
    : 'Needs the Q8 pack — download it from the launcher menu first.';
}
async function sbRestill(n) {
  const fd = new FormData();
  fd.set('id', SB.id); fd.set('n', String(n));
  let r;
  try { r = await (await fetch('/storyboard/restill', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'Could not start a new still.', { kind: 'danger', duration: 6000 }); return; }
  phosToast(`A new still for shot ${n} is being made; the shot renders from it when it lands.`, { duration: 6000 });
  sbLoad(SB.id, true);
}

function sbToggleRamHelp() {
  const btn = sbEl('sbRamHelpBtn'), note = sbEl('sbRamHelpNote');
  if (!btn || !note) return;
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  note.hidden = open;
}
function sbToggleEngineHelp() {
  const btn = sbEl('sbEngineHelpBtn'), note = sbEl('sbEngineHelpNote');
  if (!btn || !note) return;
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  note.hidden = open;
  if (!note.textContent) note.textContent = SB_BOOT.engine_help || '';
}

// A small .eng-seg: same glyph, same label, same accent variables, read off the
// SAME ENGINES registry row the header switcher renders from — so a shot card
// and the header can never disagree about what an engine looks like.
function sbEngineChip(id) {
  const e = (typeof ENGINES !== 'undefined' ? ENGINES : []).find(x => x.id === id)
         || { id: id, label: (id || 'ltx').toUpperCase(), mark: 'eng-mark-ltx',
              accent: '', accent_dim: '', accent_soft: '', tagline: '' };
  const why = id === 'h3'
    ? 'Renders on Hailuo H3 — video, dialogue and sound together. A 75 GB install.'
    : 'Renders on LTX — ships with the panel, and the only engine that loads a trained character.';
  return `<span class="sb-chip sb-chip-engine" data-engine="${escapeHtml(e.id)}"
      style="--eng-accent:${escapeHtml(e.accent)};--eng-dim:${escapeHtml(e.accent_dim)};--eng-soft:${escapeHtml(e.accent_soft)}"
      title="${escapeHtml(why)}"><span class="eng-mark"><svg class="ph" aria-hidden="true"><use href="#${escapeHtml(e.mark)}"/></svg></span>${escapeHtml(e.label)}</span>`;
}

// Mirrors _renderManualCharactersList() against the same _manualCharacters
// array. Single select in v1; click the lit avatar to deselect.
let _sbCastId = '';
function sbRenderCast() {
  const wrap = sbEl('sbCharsList'), empty = sbEl('sbCharsEmpty');
  if (!wrap) return;
  const list = (typeof _manualCharacters !== 'undefined' && _manualCharacters) || [];
  if (!list.length) {
    wrap.innerHTML = '';
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  wrap.innerHTML = list.map(c => {
    const active = c.id === _sbCastId;
    const name = c.name || c.trigger || c.id;
    const avatar = c.sample_image_url
      ? `<img class="chars-avatar-img" src="${escapeHtml(c.sample_image_url)}" alt="">`
      : `<span class="chars-avatar-ph">${escapeHtml((name || '?').charAt(0).toUpperCase())}</span>`;
    return `<button type="button" class="chars-avatar-chip ${active ? 'active' : ''}"
              onclick="sbPickCast(${JSON.stringify(c.id).replace(/"/g, '&quot;')})"
              title="${escapeHtml(name)}${active ? ' · click to deselect' : ''}">
              ${avatar}<span class="chars-avatar-name">${escapeHtml(name)}</span></button>`;
  }).join('');
}
function sbPickCast(id) {
  _sbCastId = (_sbCastId === id) ? '' : id;
  // Casting under an H3-only film is a promise the engine can't keep, so the
  // pair snaps back to Auto rather than being quietly ignored at plan time.
  if (_sbCastId && _sbEngineMode === 'h3') {
    sbSetEngineMode('auto');
    phosToast('Auto: Hailuo H3 can’t load a trained character, so their shots go to LTX.',
              {duration: 6000});
  }
  sbRenderCast();
  sbRenderEnginePicker();
}

// ---- the film-level engine -------------------------------------------------
// The owner's question, verbatim: "I don't get why I cannot select in the
// storyboard if I'm going to send to LTX or to Hailuo." So he can. Three
// options, once per film, asked BEFORE the plan exists — because the planner
// writes H3's three-field dialect or LTX prose depending on the answer, and a
// prompt written for one engine is not a prompt for the other. Changing it on
// an existing film is therefore a re-plan, and the Re-plan modal carries the
// same control.
let _sbEngineMode = 'auto';
const SB_ENGINE_OPTS = [
  { key: 'auto', label: 'Auto',      sub: 'per shot' },
  { key: 'h3',   label: 'Hailuo H3', sub: 'voice + sound', engine: 'h3' },
  { key: 'ltx',  label: 'LTX',       sub: 'characters',    engine: 'ltx' },
];

function sbH3Installed() {
  const p = (window._ENGINE_PROBES || {}).h3 || {};
  return { capable: !!p.capable, available: !!p.available };
}

function sbEnginePickerHtml(active) {
  const h3 = sbH3Installed();
  return SB_ENGINE_OPTS.map(o => {
    const e = o.engine ? (ENGINES || []).find(x => x.id === o.engine) : null;
    const blocked = (o.key === 'h3') && !h3.available;
    const offer = blocked && h3.capable;
    const cls = 'pill-btn' + (o.key === active ? ' active' : '')
              + (offer ? ' needs-install' : '');
    const style = e ? `--eng-accent:${escapeHtml(e.accent)};--eng-dim:${escapeHtml(e.accent_dim)};--eng-soft:${escapeHtml(e.accent_soft)}` : '';
    const title = o.key === 'auto'
        ? 'The plan decides per shot: a shot with one of your trained characters goes to LTX, everything else to Hailuo H3.'
      : offer ? 'Hailuo H3 isn’t installed yet — install it from the Phosphene sidebar in Pinokio.'
      : blocked ? 'This Mac can’t run Hailuo H3.'
      : o.key === 'h3' ? 'Every shot on Hailuo H3 — it renders dialogue, voices and sound with the picture. No trained characters.'
      : 'Every shot on LTX — the built-in engine, and the only one that loads a trained character.';
    const glyph = e ? `<span class="eng-mark"><svg class="ph" aria-hidden="true"><use href="#${escapeHtml(e.mark)}"/></svg></span>` : '';
    return `<button type="button" class="${cls}" data-sb-engine="${o.key}" style="${style}"
        ${blocked && !offer ? 'disabled' : ''} title="${escapeHtml(title)}">
      ${glyph}${escapeHtml(o.label)}<span class="sub">${escapeHtml(o.sub)}</span></button>`;
  }).join('');
}

function sbEnginePickerNote(active) {
  const h3 = sbH3Installed();
  if (!h3.available) {
    return h3.capable
      ? 'Hailuo H3 isn’t installed — <b>Install Hailuo H3</b> in the Phosphene sidebar in Pinokio, and this film can use it. Until then every shot renders on LTX.'
      : 'This Mac can’t run Hailuo H3, so every shot renders on LTX.';
  }
  if (active === 'h3') return 'Every shot on Hailuo H3 — dialogue, voices and sound rendered with the picture. <b>A trained character can’t come along</b>: H3 loads no LoRAs.';
  if (active === 'ltx') return 'Every shot on LTX — every mode, and the only engine that loads a trained character.';
  return 'A shot with one of your trained characters goes to <b>LTX</b>; every other shot goes to <b>Hailuo H3</b>.';
}

function sbRenderEnginePicker() {
  const g = sbEl('sbEngineGroup');
  if (g) g.innerHTML = sbEnginePickerHtml(_sbEngineMode);
  const n = sbEl('sbEnginePickNote');
  if (n) n.innerHTML = sbEnginePickerNote(_sbEngineMode);
}

function sbSetEngineMode(key, persist) {
  const h3 = sbH3Installed();
  if (key === 'h3' && !h3.available) {
    // Not installed is not dead — the click IS the install, same as the High
    // quality chip's needs-install state.
    // The registry names the install affordance (ENGINES[h3].install_card), so
    // this opens the SAME card the header switcher opens.
    const card = ((ENGINES || []).find(x => x.id === 'h3') || {}).install_card;
    if (h3.capable && card && typeof window[card] === 'function') { try { window[card](); } catch (e) {} }
    else if (h3.capable) phosToast('Install Hailuo H3 from the Phosphene sidebar in Pinokio.', {duration: 6000});
    return;
  }
  if (key === 'h3' && _sbCastId) {
    _sbCastId = '';
    sbRenderCast();
    phosToast('Cast cleared — Hailuo H3 can’t load a trained character.', {duration: 6000});
  }
  _sbEngineMode = key;
  sbRenderEnginePicker();
  if (persist !== false) sbSaveSetting('storyboard_engine', key);
}

// The Re-plan modal's copy of the control. Separate value so opening the modal
// and cancelling can't change the brief.
let _sbReplanEngineMode = 'auto';
function sbRenderReplanEnginePicker() {
  const g = sbEl('sbReplanEngineGroup');
  if (g) g.innerHTML = sbEnginePickerHtml(_sbReplanEngineMode);
  const n = sbEl('sbReplanEngineNote');
  if (n) n.innerHTML = sbEnginePickerNote(_sbReplanEngineMode) + sbEngineSnapWarning(_sbReplanEngineMode);
}

// Flipping a film's engine RE-SNAPS every shot, because H3 and LTX do not share
// a length axis: H3 renders in 3, 5, 10 or 15-second beats and LTX in 3, 5, 7,
// 10. A 7 s shot moving to H3 becomes 5 s. That is a real change to the plan,
// so it is announced rather than discovered — the same class of silence the
// 7-second lie was, one level up. Counted against the shots that are actually
// on the board, so it says "two shots" only when two shots really move.
function sbEngineSnapWarning(mode) {
  const shots = ((SB.board || {}).shots) || [];
  if (!shots.length) return '';
  const target = (mode === 'h3') ? 'h3' : (mode === 'ltx' ? 'ltx' : '');
  if (!target) return '';                       // 'auto' changes nothing by itself
  const lens = ((target === 'h3' ? (BOOT.h3 || {}).lengths : (BOOT.ltx || {}).lengths) || [])
    .filter(l => l.offered !== false).map(l => Number(l.seconds));
  if (!lens.length) return '';
  const moving = shots.filter(s => lens.indexOf(Math.round(s.duration_s)) === -1).length;
  if (!moving) return '';
  const beats = lens.slice(0, -1).join(', ') + ' or ' + lens[lens.length - 1];
  const eng = (target === 'h3') ? 'Hailuo H3' : 'LTX';
  const n = (moving === 1) ? 'One shot' : `${moving} shots`;
  return `<div class="sb-enginepick-note" style="margin-top:4px">`
       + escapeHtml(`${eng} renders in ${beats}-second beats. ${n} will move to the nearest one.`)
       + `</div>`;
}

// ---- plan ------------------------------------------------------------------
async function sbPlan() {
  const concept = (sbEl('sbConcept') || {}).value || '';
  if (!concept.trim()) { phosToast('Write a couple of sentences about the film first.'); return; }
  const btn = sbEl('sbPlanBtn');
  if (btn) { btn.dataset.busy = '1'; btn.disabled = true; btn.textContent = 'Planning…'; }
  const fd = new URLSearchParams();
  fd.set('concept', concept);
  fd.set('shots', String(sbShotsValue()));
  fd.set('take_seconds', String(_sbTake || 0));
  fd.set('style', (sbEl('sbStyle') || {}).value || '');
  fd.set('must', (sbEl('sbMust') || {}).value || '');
  fd.set('locations', (sbEl('sbLocations') || {}).value || '');
  fd.set('wardrobe', (sbEl('sbWardrobe') || {}).value || '');
  fd.set('engine', _sbEngineMode);
  if (_sbCastId) fd.set('character_id', _sbCastId);
  // The director's two fields travel whenever the row exists, so clearing
  // the path is a real "no soundtrack" and not an absent field the server
  // would read as "keep what the board had".
  if (sbEl('sbTrack')) {
    fd.set('soundtrack', (sbEl('sbTrack').value || '').trim());
    fd.set('bars_per_shot', (sbEl('sbTrackBars') || {}).value || '2');
  }
  if (sbEl('sbAuto')) fd.set('auto', sbEl('sbAuto').checked ? '1' : '0');
  if (sbEl('sbAnchorStills')) fd.set('anchor_stills', sbEl('sbAnchorStills').checked ? '1' : '0');
  if (sbEl('sbLongWindows')) fd.set('long_windows', sbEl('sbLongWindows').checked ? '1' : '0');
  let r;
  try {
    r = await (await fetch('/storyboard/plan', { method: 'POST', body: fd })).json();
  } catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) {
    sbPlanBtnReset();
    phosToast(r.error || 'Planning could not start.', { kind: 'danger', duration: 6000 });
    return;
  }
  SB.id = r.id;
  try { localStorage.setItem('phos_sb_open', SB.id); } catch (e) {}
  if (sbEl('sbAuto') && sbEl('sbAuto').checked) {
    phosToast('Auto is on: after the plan, every shot renders, the cut is made and the film is assembled on their own. The Film screen shows it when it lands.',
              { duration: 9000 });
  }
  sbShow('planning');
  sbSetPlanningStage('load');
  sbLoad(SB.id);
}
// THE SHOTS CHIPS STAND DOWN WHEN A TRACK IS ON THE BRIEF — the grid decides
// the count — and the line under the field says what will happen instead.
function sbTrackInput() {
  const path = ((sbEl('sbTrack') || {}).value || '').trim();
  const bars = (sbEl('sbTrackBars') || {}).value || '2';
  const meta = sbEl('sbTrackMeta');
  const row = sbEl('sbLengthRow');
  if (meta) {
    meta.textContent = path
      ? 'music video — one shot per ' + bars + ' bar' + (bars === '1' ? '' : 's')
        + ' on the downbeat; the track replaces the clips\' own sound and sets the shot count'
      : 'optional — a track makes it a music video cut to the beat';
  }
  if (row) row.classList.toggle('is-standing-down', !!path);
}

function sbPlanBtnReset() {
  const btn = sbEl('sbPlanBtn');
  if (!btn) return;
  delete btn.dataset.busy;
  btn.textContent = 'Plan film';
  sbConceptInput();
}

async function sbCancelPlan() {
  const fd = new URLSearchParams(); fd.set('id', SB.id);
  try { await fetch('/storyboard/cancel', { method: 'POST', body: fd }); } catch (e) {}
  phosToast('Planning cancelled.');
  sbPlanBtnReset();
  sbLoad(SB.id);
}

function sbSetPlanningStage(stage) {
  const map = {
    load:   ['Loading the planner', 'About a minute. Nothing renders yet.'],
    grid:   ['Reading the beat', 'Finding the downbeats the shots will cut on.'],
    write:  ['Writing the plan', 'About a minute. Nothing renders yet.'],
    check:  ['Checking the plan', 'About a minute. Nothing renders yet.'],
    repair: ['Fixing the plan', 'It came back slightly malformed. One retry.'],
    unload: ['Giving the memory back', 'The renderer gets it now.'],
  };
  const order = ['load', 'grid', 'write', 'check', 'repair', 'unload'];
  const at = order.indexOf(stage);
  const t = map[stage] || map.load;
  const ttl = sbEl('sbPlanningTitle'), sub = sbEl('sbPlanningSub');
  if (ttl) ttl.textContent = t[0];
  if (sub) sub.textContent = t[1];
  document.querySelectorAll('#sbPlanningSteps .sb-step').forEach(el => {
    const i = order.indexOf(el.dataset.step);
    el.classList.toggle('is-now', i === at);
    el.classList.toggle('is-done', i >= 0 && at >= 0 && i < at);
    // The repair row is revealed only when the retry actually fired —
    // advertising a failure that usually doesn't happen is worse than silence.
    if (el.dataset.step === 'repair') el.hidden = (stage !== 'repair');
  });
}

function sbOpenReplan() {
  _sbReplanEngineMode = ((SB.payload || {}).engine_mode) || 'auto';
  sbRenderReplanEnginePicker();
  sbEl('sbReplanModal').classList.add('show');
}
function sbCloseReplan() { sbEl('sbReplanModal').classList.remove('show'); }
async function sbReplan() {
  const notes = (sbEl('sbReplanNotes') || {}).value || '';
  sbCloseReplan();
  const b = (SB.payload || {}).board || {};
  const fd = new URLSearchParams();
  fd.set('id', SB.id);
  fd.set('concept', b.concept || '');
  fd.set('style', b.style || '');
  fd.set('shots', String(b.shots_target || (b.shots || []).length || 12));
  fd.set('must', (b.must || []).join('\n'));
  fd.set('notes', notes);
  fd.set('engine', _sbReplanEngineMode);
  const cast = (b.cast || [])[0];
  // An H3-only film carries no cast — the server refuses the pair, so don't
  // send one it would have to reject.
  if (cast && cast.id && _sbReplanEngineMode !== 'h3') fd.set('character_id', cast.id);
  let r;
  try { r = await (await fetch('/storyboard/plan', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'Re-plan could not start.', { kind: 'danger', duration: 6000 }); return; }
  sbShow('planning'); sbSetPlanningStage('load'); sbLoad(SB.id);
}

// Try again re-plans THIS board from the brief the board itself stores. It used
// to call sbPlan(), which sends no id and reads the left-hand form — so a failed
// "deep-sea welder" board plus an unrelated concept still sitting in the brief
// produced a second, different film and left the failed one as a dead row.
async function sbTryAgain() {
  const b = ((SB.payload || {}).board) || {};
  if (!SB.id) return sbPlan();
  const emode = b.engine_mode || 'auto';
  const fd = new URLSearchParams();
  fd.set('id', SB.id);
  fd.set('concept', b.concept || '');
  fd.set('shots', String(b.shots_target || (b.shots || []).length || 12));
  fd.set('style', b.style || '');
  fd.set('must', (b.must || []).join('\n'));
  fd.set('engine', emode);
  const cast = (b.cast || [])[0];
  if (cast && cast.id && emode !== 'h3') fd.set('character_id', cast.id);
  let r;
  try { r = await (await fetch('/storyboard/plan', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'Planning could not start.', { kind: 'danger', duration: 6000 }); return; }
  sbShow('planning'); sbSetPlanningStage('load'); sbLoad(SB.id);
}

function sbShowRaw() {
  const raw = ((SB.payload || {}).planner || {}).raw || '';
  sbEl('sbRawText').textContent = raw || '(nothing captured)';
  sbEl('sbRawModal').classList.add('show');
}

// ---- board open / load -----------------------------------------------------
// RETURNS THE LOAD. It used to swallow it, and `await sbOpen(id)` therefore
// awaited `undefined` — so the row's "Edit" button opened the timeline and then
// the load that was still in flight called sbShow('plan') and tore it straight
// back down. Clicking Edit landed on the shot list. Handing the promise back is
// the whole fix, and sbOpenAt() is the caller that needed it.
function sbOpen(id) {
  // Another board's films are not this board's. Without this the rail carried
  // the last film you looked at onto the next storyboard's step 4.
  if (id !== SB.id) { SB.films = []; SB.filmsFor = ''; SB.filmOpen = ''; }
  SB.id = id;
  try { localStorage.setItem('phos_sb_open', id); } catch (e) {}
  return sbLoad(id);
}
// Open a board AND land on a named step. The board list's rows use it, so a
// film with a finished cut is one click from the cut, not four.
async function sbOpenAt(id, step) {
  await sbOpen(id);
  if (step) sbGo(step);
}
function sbBackToList() {
  SB.id = '';
  SB.payload = null;
  SB.films = [];
  SB.filmsFor = '';
  SB.filmOpen = '';
  try { localStorage.removeItem('phos_sb_open'); } catch (e) {}
  sbRefreshBoards().then(() => sbShow(SB.boards.length ? 'list' : 'empty'));
}

// ---- the rail: plan → shots → arrange → film -------------------------------
// One model function, pure, so the thing that decides what is reachable is
// testable without a browser. `snap` is {clips, film, stage, shots, done}.
function sbRailModel(snap) {
  const s = snap || {};
  const clips = Number(s.clips || 0);
  const stage = s.stage || '';
  const film = s.film || null;
  // A step is 'now' when you are standing in it, 'locked' when it cannot do
  // anything yet (and says what would unlock it), 'done' when it has produced
  // what it produces, 'ready' otherwise. Locked is never hidden: a path you
  // cannot see the end of is not a path.
  const shotsSub = !s.shots ? ''
    : (clips ? clips + ' of ' + s.shots + ' rendered' : s.shots + ' planned');
  const arrangeLocked = !clips;
  return [
    { key: 'plan', label: 'Plan', n: 1, state: 'done', sub: '',
      hint: 'The brief this film was planned from — open it to re-plan.' },
    { key: 'shots', label: 'Shots', n: 2,
      state: stage === 'plan' ? 'now' : (clips ? 'done' : 'ready'),
      sub: shotsSub,
      hint: 'Read the shot list, fix a prompt, render, grade what came back.' },
    // Step 3 is a DOOR now, not a room: it switches to the Editor tab
    // carrying this film. The word is Editor everywhere — tab, header, this
    // step, and the board row's button.
    { key: 'edit', label: 'Edit', n: 3,
      state: stage === 'editor' ? 'now'
           : arrangeLocked ? 'locked' : (film ? 'done' : 'ready'),
      sub: '',
      hint: arrangeLocked
        ? 'Render a shot first — there is nothing to edit yet.'
        : 'Open this film in the Editor: trim, reorder, cut to the beat, then render the film.' },
    { key: 'film', label: 'Film', n: 4,
      state: stage === 'film' ? 'now'
           : arrangeLocked ? 'locked' : (film ? 'done' : 'ready'),
      sub: film && film.duration ? sbFmtClock(film.duration) : '',
      hint: arrangeLocked ? 'Render a shot first.'
          : film ? 'Watch the finished film.'
          : 'Nothing rendered yet — arrange the shots and render the timeline.' },
  ];
}
function sbRailPaint() {
  const rail = sbEl('sbRail');
  if (!rail) return;
  const open = !!SB.id && ['plan', 'film'].indexOf(SB.stage) !== -1;
  rail.hidden = !open;
  if (!open) return;
  const b = ((SB.payload || {}).board) || {};
  const row = SB.boards.filter(x => x.id === SB.id)[0] || {};
  const shots = (b.shots || []).length;
  const clips = (b.shots || []).filter(s => s.draft_output || s.final_output).length;
  // The open board's own film list wins over the poll's summary — after a
  // render the screen must not still be saying "no film".
  const mine = SB.filmsFor === SB.id ? (SB.films || []) : [];
  const film = mine[0] || row.film || null;
  // "You are here" survives the tab move: while the Editor holds THIS film,
  // step 3 is the step you are standing in, wherever the body happens to be.
  const inEditor = typeof SBE !== 'undefined' && SBE.open && SBE.id === SB.id;
  const model = sbRailModel({ clips: clips, shots: shots, film: film,
                              stage: inEditor ? 'editor' : SB.stage });
  model.forEach(step => {
    const el = rail.querySelector('.sb-rail-step[data-step="' + step.key + '"]');
    if (!el) return;
    el.classList.toggle('is-now', step.state === 'now');
    el.classList.toggle('is-done', step.state === 'done');
    el.classList.toggle('is-locked', step.state === 'locked');
    el.disabled = step.state === 'locked';
    el.title = step.hint;
    const sub = el.querySelector('.sb-rail-sub');
    if (sub) sub.textContent = step.sub ? '· ' + step.sub : '';
  });
}
// Every door in one function, so "go to the editor" means the same thing
// wherever it is asked from.
function sbGo(step) {
  if (!SB.id) return;
  if (step === 'plan') {
    sbOpenReplan();
    return;
  }
  if (step === 'shots') {
    sbShow('plan');
    if (SB.payload) sbRenderPlan(SB.payload);
    return;
  }
  if (step === 'edit') {
    // Not a stage swap any more — a tab switch that carries the film. The
    // Editor is its own window; this is the door into it from a board.
    edOpenBoard(SB.id);
    return;
  }
  if (step === 'film') sbFilmOpen();
}

async function sbLoad(id, quiet) {
  let r;
  try {
    r = await (await fetch('/storyboard/get?id=' + encodeURIComponent(id))).json();
  } catch (e) { return; }
  if (!r || !r.ok) {
    if (!quiet) phosToast('That storyboard is gone.', { kind: 'danger' });
    sbBackToList();
    return;
  }
  if (SB.payload && SB.payload.board && r.board && sbTypingInShots()) {
    // A refresh that lands mid-word keeps what is on screen.
    sbAdoptLiveEdits(r);
  }
  SB.payload = r;
  const st = (r.planner || {}).state;
  if (r.planning || st === 'running') {
    sbShow('planning');
    sbSetPlanningStage((r.planner || {}).stage || 'load');
    return;
  }
  if (st === 'failed') {
    sbPlanBtnReset();
    sbShow('planfail');
    sbRenderPlanFail(r.planner || {});
    return;
  }
  sbPlanBtnReset();
  // A 2 s poll must not yank someone off the film screen and back onto the
  // shot list. The shot list is still repainted underneath — it just isn't
  // shoved in front of the film the user asked to watch.
  sbShow(SB.stage === 'film' ? 'film' : 'plan');
  sbRenderPlan(r);
  if (SB.stage === 'film') return;
  // The player keeps whatever was last selected globally, so opening a film
  // showed an unrelated Video-tab render under the film's own title. Put the
  // film's newest clip up — but only when the current one isn't already one of
  // this film's, so it never yanks a shot the user just clicked.
  const mine = (r.board.shots || [])
    .map(x => x.final_output || x.draft_output).filter(Boolean);
  if (mine.length && mine.indexOf(activePath) === -1) {
    try { selectOutput(mine[mine.length - 1]); } catch (e) {}
  }
}

function sbRenderPlanFail(p) {
  const map = {
    download: 'The planner model didn\'t finish downloading. Check your connection and try again — it resumes where it stopped.',
    oom: 'Not enough free memory to load the planner. Close some apps and try again.',
    invalid: 'It produced something this build can\'t read, twice. Trying again usually works — the model isn\'t deterministic. If it keeps failing, shorten the concept.',
    busy: 'Something else is using the GPU. Wait for the queue to empty and try again.',
    // The planner never got to finish, so "it couldn't write a usable plan"
    // would be untrue. The heal path writes this kind at boot.
    restarted: 'The panel restarted while the planner was running, so the plan never finished. Nothing was lost — press Try again.',
  };
  // An unmapped kind falls back to what the server actually said rather than to
  // a shrug: the sentence was written, it just had nowhere to appear.
  sbEl('sbPlanFailMsg').textContent = map[p.error_kind]
    || (p.error ? ('Something went wrong while planning: ' + p.error + '. Try again.')
                : 'Something went wrong while planning. Try again.');
  const ttl = document.querySelector('#sbPlanFail .sb-empty-title');
  if (ttl) ttl.textContent = (p.error_kind === 'restarted')
    ? 'The plan didn\'t finish.' : 'The planner couldn\'t write a usable plan.';
  sbEl('sbPlanFailRaw').hidden = !p.raw;
}

// Exactly one of the five stage states is visible at a time, and body.sb-full
// is on while there is nothing for the player to show — the same trick the
// Ideogram layout editor uses to take the whole column.
function sbShow(which) {
  // Five states now, not six: `timeline` moved out to the Editor tab, which
  // owns its own visibility. A stage state was the wrong home for a document.
  const states = { empty: 'sbEmpty', list: 'sbList', planning: 'sbPlanning',
                   planfail: 'sbPlanFail', plan: 'sbPlan',
                   film: 'sbFilm' };
  SB.stage = which;
  Object.keys(states).forEach(k => {
    const el = sbEl(states[k]);
    if (el) el.hidden = (k !== which);
  });
  // A <video> left playing behind a hidden div is a decoder and a soundtrack
  // nobody can see. Leaving the film screen stops it.
  if (which !== 'film') {
    const v = sbEl('sbFilmVideo');
    if (v) { try { v.pause(); } catch (e) {} }
  }
  // The brief's board list is a way BACK to another film while one is open.
  // While the stage is already showing the list, it would be the same list
  // twice in one screen — so it folds away.
  const boards = sbEl('sbBoards');
  if (boards) {
    boards.hidden = !SB.boards.length || which === 'list' || which === 'empty';
  }
  sbRailPaint();
  sbSyncStage();
}

function sbHasClip() {
  const shots = (((SB.payload || {}).board) || {}).shots || [];
  return shots.some(s => s.draft_output || s.final_output);
}

function sbSyncStage() {
  const on = document.body.dataset.workflow === 'storyboard';
  const hasClip = on && sbHasClip();
  // The editor no longer lives in this column, so this function no longer
  // decides anything about it — `body.sbe-open` and the layout takeover it
  // drove are both gone with the move to the Editor tab.
  // The film screen IS a player, so it takes the column on those terms.
  if (on && SB.stage === 'film') {
    document.body.classList.add('sb-full');
    const t = sbEl('sbStageToggle');
    if (t) t.hidden = true;
    return;
  }
  let full;
  if (!on) full = false;
  else if (SB.stageMode === 'list') full = true;
  else if (SB.stageMode === 'player') full = false;
  // AUTO NO LONGER MEANS "the moment a clip exists, take half the column".
  //
  // It used to read `full = !hasClip`, so the instant the first draft landed the player
  // docked itself over the work — 414 px of a 812 px column, measured, with twelve shots
  // left sharing the remainder. Nobody asked for it: the user was reading the shot list
  // and a render finished. That is the whole of the report.
  //
  // Auto now stays on the list, and the player appears when the user ASKS for a clip
  // (sbOpenShotClip, the per-shot thumbnail). The clip is never hidden — the thumbnail on
  // each shot card is the affordance, and it was already there.
  else full = true;
  document.body.classList.toggle('sb-full', !!full);
  const tog = sbEl('sbStageToggle');
  if (tog) {
    tog.hidden = !hasClip;
    tog.querySelectorAll('.smt-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.sbStage === (full ? 'list' : 'player')));
  }
}
function sbSetStage(mode) {
  SB.stageMode = mode;
  // Session-scoped, so a reload mid-film does not re-dock a player the user closed.
  // sessionStorage, not localStorage: this is a working preference for this sitting, not
  // a setting the user should have to find and undo next week.
  try { sessionStorage.setItem('phos.sb.stageMode', mode); } catch (e) {}
  sbSyncStage();
}

// Esc collapses the docked preview, matching the lightbox's own Esc. Deliberately narrow:
// only on the storyboard tab, only when the player is actually docked, and only when the
// lightbox is closed (its Esc wins) and the user is not typing in a shot's textarea.
document.addEventListener('keydown', function (e) {
  if (e.key !== 'Escape') return;
  if (document.body.dataset.workflow !== 'storyboard') return;
  if (document.body.classList.contains('sb-full')) return;
  const lb = document.getElementById('expandLightbox');
  if (lb && getComputedStyle(lb).display !== 'none') return;
  const t = document.activeElement;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  sbSetStage('list');
});

// ---- the plan screen -------------------------------------------------------
// The last shot-list HTML this module wrote. Nothing else writes #sbShots, so
// "same string" means "the DOM already says this" — see sbRenderPlan.
let _sbShotsHtml = '';

function sbRenderPlan(r) {
  const b = r.board || {};
  const est = r.estimate || {};
  const shots = b.shots || [];
  const title = sbEl('sbTitle');
  if (title && document.activeElement !== title) title.value = b.title || '';
  // The three brief switches read back from the board, so reopening a film
  // shows how it is being made — not whatever the last brief had ticked.
  for (const [id, key] of [['sbAuto', 'auto'], ['sbAnchorStills', 'anchor_stills'], ['sbLongWindows', 'long_windows']]) {
    const el = sbEl(id);
    if (el && key in b) el.checked = !!b[key];
  }
  if ('take_seconds' in b) {
    if (b.take_seconds) { sbSetShots('take', false); sbSetTake(b.take_seconds); }
    else sbSetShots(b.shots_target || _sbShots, false);
  }
  sbSyncBriefGates();

  // --- summary ---
  // Per PASS, not per `status`: during delivery, `status === 'done'` still
  // reads true from the draft that already landed.
  const outKey = r.pass === 'final' ? 'final_output' : 'draft_output';
  const done = shots.filter(s => s[outKey]).length;
  const failed = shots.filter(s => s.status === 'failed').length;
  const rendering = !!r.rendering;
  let status;
  if (r.pass === 'final') {
    // "Delivery rendering · 4 of 4" is a sentence about a thread that hasn't
    // released its slot yet, not about the film. Once every shot has its
    // delivery clip the film is finished, whatever the dispatcher is doing.
    status = (rendering && done < shots.length)
      ? `Delivery rendering · ${done} of ${shots.length}`
      : `Finished · ${done} shot${done === 1 ? '' : 's'}`;
  } else if (rendering) {
    status = `Drafts rendering · ${done} of ${shots.length}`;
  } else if (done && done + failed >= shots.length) {
    status = failed ? `Drafts done · ${done} of ${shots.length}, ${failed} failed`
                    : `Drafts done · ${done} of ${shots.length}`;
  } else if (done) {
    // Partway through and NOT running — stopped, or a single shot retried.
    // "Drafts rendering" here would be a sentence that isn't true.
    status = `Drafts · ${done} of ${shots.length}`;
  } else {
    status = 'Draft plan · not rendered';
  }
  sbEl('sbPlanStatus').textContent = status;

  // The first two cells describe the work THIS PASS still has to do. With none
  // left they were reading "0 shots" and "—", which says nothing about the film
  // you are looking at — so at zero they switch to describing the film itself.
  const nothingLeft = !est.shots;
  const runtimeAll = shots.reduce((a, s) => a + (Number(s.duration_s) || 0), 0);
  sbEl('sbSumShots').textContent = nothingLeft
    ? (shots.length === 1 ? '1 shot' : `${shots.length} shots`)
    : (est.shots === 1 ? '1 shot' : `${est.shots} shots`);
  sbEl('sbSumRuntime').textContent = sbFmtRuntime(nothingLeft ? runtimeAll : est.runtime_secs);
  sbEl('sbSumTime').textContent = nothingLeft ? 'all rendered' : sbFmtWall(est.total_secs);
  sbEl('sbSumTimeSub').textContent = nothingLeft
    ? 'nothing left to render'
    : (r.pass === 'final' ? 'delivery' : 'drafts') + ', this Mac';
  const loads = est.pipeline_loads || 0;
  sbEl('sbSumLoads').textContent = loads === 1 ? '1 model load' : `${loads} model loads`;
  let loadsSub = loads === 1 ? 'every shot renders back to back' : 'grouped, not in story order';
  if (est.saved_secs > 0) loadsSub += ` — rendering grouped saves ${sbFmtWall(est.saved_secs).replace('about ', '')}`;
  sbEl('sbSumLoadsSub').textContent = loadsSub;
  const freeGb = (r.disk || {}).free_gb;
  sbEl('sbSumDisk').textContent = (freeGb == null) ? '—' : `${freeGb} GB free`;
  const tight = (freeGb != null && freeGb < 20);
  sbEl('sbSumDiskCell').classList.toggle('is-warn', tight);
  sbEl('sbSumDiskSub').textContent = tight
    ? 'clips land in mlx_outputs/ — this is tight' : 'clips land in mlx_outputs/';

  // Run strip — one segment per bucket, sized by shot count. A single band
  // explaining nothing is chrome, so it hides below 4 shots on one bucket.
  const strip = sbEl('sbRunStrip');
  const buckets = est.buckets || [];
  const NAMES = { t2v: 'Text & Character', remix: 'Remix', keyframe: 'Keyframes',
                  extend: 'Extend', a2v: 'Audio' };
  if (buckets.length <= 1 && shots.length < 4) {
    strip.hidden = true;
  } else {
    strip.hidden = false;
    strip.innerHTML = buckets.map(bk => {
      const ns = bk.shots || [];
      // "2–6" for [2,4,6] would claim five shots that aren't in this bucket.
      // A range only when the run really is contiguous; otherwise say which.
      const contiguous = ns.length > 1 && ns[ns.length - 1] - ns[0] === ns.length - 1;
      const range = !ns.length ? ''
        : ns.length === 1 ? `${ns[0]}`
        : contiguous ? `${ns[0]}–${ns[ns.length - 1]}`
        : (ns.length > 4 ? `${ns.length} shots` : ns.join(', '));
      const label = (bk.engine === 'h3' ? 'H3 · ' : '') + (NAMES[bk.kind] || bk.kind);
      return `<div class="sb-runseg" style="flex-grow:${ns.length}" data-shots="${ns.join(',')}">
        ${bk.engine === 'h3' ? '' : '<span class="sb-runseg-load" title="One model load, about 90 s">load</span>'}
        <span class="sb-runseg-name">${escapeHtml(label)}</span>
        <span class="sb-runseg-shots">${range}</span></div>`;
    }).join('');
  }

  // --- quality ---
  const pol = b.policy || {};
  const dq = (pol.draft || {}).quality || 'quick';
  const fq = (pol.final || {}).quality || 'standard';
  document.querySelectorAll('#sbDraftQuality .pill-btn').forEach(x =>
    x.classList.toggle('active', x.dataset.q === dq));
  document.querySelectorAll('#sbFinalQuality .pill-btn').forEach(x =>
    x.classList.toggle('active', x.dataset.q === fq));
  sbEl('sbQualityMeta').textContent =
    `Draft ${(pol.draft || {}).width}×${(pol.draft || {}).height} · ` +
    `Delivery ${(pol.final || {}).width}×${(pol.final || {}).height}`;

  // --- board-level errors ---
  const errs = r.errors || [];
  const boardErrs = errs.filter(e => !e.n);
  const errBox = sbEl('sbErrors');
  errBox.hidden = !boardErrs.length;
  errBox.innerHTML = boardErrs.map(sbErrRow).join('');

  // --- shot cards ---
  // Rebuilding the list under a cursor is what makes editing impossible, so a
  // repaint that arrives mid-word repaints everything EXCEPT the cards. The
  // next repaint after blur (<=2 s) brings the server's copy in. Same for a
  // dropdown someone has open (see sbSelectOpen).
  //
  // AND: only paint when the list actually CHANGED. A board that nobody is
  // touching renders byte-identical cards on every 2 s tick, and this line was
  // still destroying and rebuilding every node in it ~30 times a minute — for
  // nothing. Anything the DOM was holding died with it, an open <select> most
  // visibly. Cheap, too: the string is built either way, the compare is the
  // only new work.
  const box = sbEl('sbShots');
  if (!sbTypingInShots() && !sbSelectOpen()) {
    const html = shots.map(s => sbShotCard(s, r, errs)).join('');
    if (html !== _sbShotsHtml || box.childElementCount !== shots.length) {
      box.innerHTML = html;
      _sbShotsHtml = html;
    }
  }

  // --- which engine, and why ---
  // There is NO engine selection on this tab, and saying so out loud is the
  // fix for the first thing the owner asked about it. The line states the rule;
  // the ? carries the Python-owned paragraph; the chips on the cards carry the
  // per-shot answer.
  const mix = est.engine_mix || {};
  const mode = r.engine_mode || 'auto';
  let engText;
  if (!r.h3_available) {
    engText = SB_BOOT.engine_note_no_h3 || '';
  } else if (mode === 'h3') {
    engText = 'You set this film to <b>Hailuo H3</b> — every shot.';
  } else if (mode === 'ltx') {
    engText = 'You set this film to <b>LTX</b> — every shot.';
  } else {
    engText = SB_BOOT.engine_note || '';
    if (mix.h3 && mix.ltx) engText += ` <b>${mix.h3} on Hailuo H3, ${mix.ltx} on LTX.</b>`;
    else if (mix.h3) engText += ` <b>All ${mix.h3} on Hailuo H3.</b>`;
    else if (mix.ltx) engText += ` <b>All ${mix.ltx} on LTX.</b>`;
  }
  if (r.h3_available && mode !== 'auto') engText += ' Change it in Re-plan.';
  sbEl('sbEngineNoteText').innerHTML = engText;

  // --- run bar / action bar ---
  sbRenderRunBar(r);
  const graded = shots.filter(s => s.grade).length;
  const anyClip = sbHasClip();
  sbEl('sbActionBar').hidden = anyClip;
  sbEl('sbTally').hidden = !anyClip;
  const btn = sbEl('sbRenderBtn');
  btn.disabled = errs.length > 0 || rendering;
  btn.title = errs.length ? `Fix the ${errs.length} problem${errs.length > 1 ? 's' : ''} above first.` : '';
  sbEl('sbActionNote').textContent =
    `${est.shots || 0} shot${est.shots === 1 ? '' : 's'} · ${sbFmtWall(est.total_secs)}`;
  if (anyClip) sbRenderTally(shots, r);
  if (!sbTypingInShots()) sbAutoGrowPrompts();
  sbSyncStage();
}

// 3-10 rows, sized to the text. A planner prompt is 70-140 words and a fixed
// 3-row box hides most of it behind a scrollbar — which matters here, because
// reading the plan before spending render hours is the entire point.
function sbAutoGrowPrompts(one) {
  const els = one ? [one] : document.querySelectorAll('#sbShots .sb-shot-prompt');
  els.forEach(el => {
    const line = 19.4;                     // 12.5px * 1.55 line-height
    el.style.height = 'auto';
    const want = Math.min(Math.max(el.scrollHeight, line * 3), line * 10 + 16);
    el.style.height = Math.ceil(want) + 'px';
  });
}

function sbErrRow(e) {
  const copy = SB_ERR_COPY[e.code];
  const html = !copy ? escapeHtml(e.message)
             : (typeof copy.html === 'function' ? copy.html(e) : copy.html);
  const label = copy && (typeof copy.label === 'function' ? copy.label(e) : copy.label);
  const fix = copy && copy.fix
    ? `<button type="button" class="sb-err-fix" onclick="sbFixError('${copy.fix}',${e.n || 0},'${escapeHtml(e.code)}')">${escapeHtml(label)}</button>`
    : '';
  return `<div class="sb-err-row"><span class="sb-err-dot"></span><span>${html}</span>${fix}</div>`;
}

function sbShotCard(s, r, errs) {
  const n = s.n;
  const mine = errs.filter(e => e.n === n);
  const est = (r.per_shot_est || {})[String(n)];
  const chars = r.characters || [];
  const locked = (s.status === 'rendering' || s.status === 'queued' || s.status === 'done');
  // The chip says what will ACTUALLY render, not what the plan wrote. With no
  // H3 pack the server forces every shot to LTX at enqueue, so a pink Hailuo
  // chip on a machine that has no Hailuo would be the UI contradicting itself.
  const engine = r.h3_available ? (s.engine || 'ltx') : 'ltx';
  const clip = s.final_output || s.draft_output;
  // A shot whose prompt was just rewritten has no clip of its own yet, but the
  // take it replaced is still on disk. Show it under the "old take" wash so the
  // card isn't suddenly blank — and NOT gradeable, because it is not the shot
  // the plan now describes.
  const stale = !clip ? (s.stale_output || '') : '';
  const shots = (r.board || {}).shots || [];
  const isChar = !!s.character_id;
  const opts = ['<option value="">— nobody —</option>'].concat(chars.map(c =>
    `<option value="${escapeHtml(c.id)}" ${c.id === s.character_id ? 'selected' : ''}>${escapeHtml(c.name || c.id)}</option>`)).join('');
  // With no trained characters on this Mac the cast select can only ever say
  // "nobody", so it isn't shown at all and the Character button says why.
  const noCast = !chars.length;
  // THE 7-SECOND LIE, and what it actually was.
  //
  // This menu was `[3, 5, 7, 10]` for both engines. A shot set to 7 s whose
  // engine is h3 reaches storyboard.h3_length_for(7.0), which snaps to the
  // nearest of {3,5,10,15} with ties to the shorter — |5-7| beats |10-7| — so
  // it rendered 5s, 124 frames, about 5.2 seconds. The select still read "7 s".
  // estimate() priced it at 5 s too, so nothing anywhere disagreed and the user
  // had no way to find out except by watching the clip. It generalised: any
  // planner-written duration was pushed into the menu, so an 8 s H3 shot
  // silently became 10 s. And in the other direction H3's 15 s was UNREACHABLE
  // from a menu that stopped at 10.
  //
  // The menu is now the engine's own table — the same one the Manual strips,
  // the Characters tab and every estimate read. An H3 shot offers 3/5/10/15 and
  // has no 7; an LTX shot offers 3/5/7/10 (and 20s, which the table itself
  // restricts to Quick). One duration vocabulary per engine, server-owned.
  const _lenTable = (engine === 'h3' ? (BOOT.h3 || {}).lengths : (BOOT.ltx || {}).lengths) || [];
  // `offered` is the GLOBAL flag; a length can also be restricted to certain
  // canvases, and 20s is — it holds together at 640x448 and comes apart around
  // frame 454 at 1024x576. Filtering on `offered` alone put 20s on every LTX
  // shot, and shot_to_job then enqueued {quality: balanced, frames: 481}: the
  // exact cell the Manual strip greys out with a reason. The per-quality gate
  // is the same one the tier table already carries, asked here too.
  const _sbQual = (engine === 'h3')
    ? null
    : (((r.board || {}).policy || {})[r.pass === 'final' ? 'final' : 'draft'] || {}).quality
      || ((BOOT.ltx || {}).default_quality);
  const _lens = _lenTable.filter(l => {
    if (l.offered === false) return false;
    const allowed = l.qualities || [];
    return !(allowed.length && _sbQual && allowed.indexOf(_sbQual) === -1);
  });
  const _cur = Math.round(s.duration_s);
  let durOpts = _lens.map(l =>
    `<option value="${l.seconds}" ${_cur === Number(l.seconds) ? 'selected' : ''}>${escapeHtml(l.label)}</option>`).join('');
  // A board carrying an OFF-AXIS duration keeps round-tripping — a hand-edited
  // board, or one planned before this table existed, must not silently lose its
  // value. But on the H3 lane the extra option says what it will ACTUALLY do,
  // because that is the whole bug: it is the snap that was invisible.
  if (_lens.length && !_lens.some(l => Number(l.seconds) === _cur)) {
    let label = `${_cur} s`;
    if (engine === 'h3') {
      const near = _lens.reduce((a, b) =>
        (Math.abs(b.seconds - _cur) < Math.abs(a.seconds - _cur)
          || (Math.abs(b.seconds - _cur) === Math.abs(a.seconds - _cur) && b.seconds < a.seconds)) ? b : a);
      label = `${_cur} s · nearest ${near.label}`;
    }
    durOpts += `<option value="${_cur}" selected>${escapeHtml(label)}</option>`;
  }
  const passLabel = r.pass === 'final' ? 'Delivery' : 'Draft';
  const seedSet = (s.seed != null && s.seed !== -1);

  // ANCHOR STILL. The image the shot starts from, when the board anchors
  // shots; a failed still says so and the shot renders unanchored.
  const stillPending = !s.still && !s.still_error && s.still_job_id && s.still_job_id !== 'skipped';
  const stillBlock = s.still ? `
    <div class="sb-still" title="The still this shot starts from">
      <a href="/image?path=${encodeURIComponent(s.still)}" target="_blank" rel="noopener" title="Open the still">
        <img src="/image?w=480&path=${encodeURIComponent(s.still)}" alt="Anchor still for shot ${n}" loading="lazy"></a>
      <span class="sb-still-tag">starts from this still</span>
      <button type="button" class="sb-err-fix sb-still-redo" data-act="restill" title="Make a new still and render this shot from it">New still</button>
    </div>` : stillPending ? `
    <div class="sb-still is-pending"><span class="sb-still-spin" aria-hidden="true"></span><span class="sb-still-tag">making the still…</span></div>`
    : (s.still_error ? `<div class="sb-still sb-still-failed" title="${escapeHtml(s.still_error)}"><span class="sb-still-tag">no still — rendered unanchored</span>
      <button type="button" class="sb-err-fix sb-still-redo" data-act="restill" title="Try the still again">Try again</button></div>` : '');
  const outBlock = stale ? `
    <div class="sb-shot-out car-card is-stale">
      <div class="car-thumb-wrap" onclick="selectOutput('${escapeHtml(stale)}')">
        <video class="car-thumb" preload="metadata" muted playsinline
               src="/file?path=${encodeURIComponent(stale)}"></video>
      </div>
      <div class="info"><span class="name">${escapeHtml(stale.split('/').pop())}</span>
        <span class="sub"><span class="sb-stale-tag">old take</span> · rewritten, not rendered yet</span></div>
    </div>` : clip ? `
    <div class="sb-shot-out car-card">
      <div class="car-thumb-wrap" onclick="sbOpenShotClip(${n})">
        <video class="car-thumb" preload="metadata" muted playsinline
               src="/file?path=${encodeURIComponent(clip)}"
               onmouseenter="this.currentTime=0;this.playbackRate=0.6;this.play().catch(()=>{})"
               onmouseleave="this.pause();this.currentTime=2.5"></video>
      </div>
      <div class="info"><span class="name">${escapeHtml(clip.split('/').pop())}</span>
        <span class="sub">${s.final_output ? 'delivery' : 'draft'}</span></div>
      <div class="sb-grade" role="group" aria-label="Grade shot ${n}">
        <button type="button" class="sb-grade-btn ${s.grade === 'keep' ? 'active' : ''}" data-act="grade" data-g="keep" aria-pressed="${s.grade === 'keep'}">KEEP</button>
        <button type="button" class="sb-grade-btn ${s.grade === 'reroll' ? 'active' : ''}" data-act="grade" data-g="reroll" aria-pressed="${s.grade === 'reroll'}">RE-ROLL</button>
        <button type="button" class="sb-grade-btn ${s.grade === 'cut' ? 'active' : ''}" data-act="grade" data-g="cut" aria-pressed="${s.grade === 'cut'}">CUT</button>
      </div>
      <textarea class="sb-note" rows="2" data-act="note" ${s.grade === 'reroll' ? '' : 'hidden'}
        placeholder="What should change? (goes back to the planner)">${escapeHtml(s.note || '')}</textarea>
    </div>` : '';

  const failBlock = (s.status === 'failed') ? (() => {
    const fe = friendlyJobError(s.error || '');
    return `<div class="sb-shot-err"><div class="sb-err-row"><span class="sb-err-dot"></span>
      <span><b>Shot ${n} failed.</b> ${escapeHtml(fe.friendly)} — ${escapeHtml(fe.hint)}</span>
      <button type="button" class="sb-err-fix" data-act="retry">Retry this shot</button>
      <button type="button" class="sb-err-fix" data-act="cut">Cut it</button></div></div>`;
  })() : '';

  return `<li class="sb-shot ${mine.length ? 'has-error' : ''} ${locked ? 'is-locked' : ''}"
      data-n="${n}" draggable="${locked ? 'false' : 'true'}" tabindex="0">
    <div class="sb-shot-head">
      <span class="sb-shot-n">${String(n).padStart(2, '0')}</span>
      <div class="sb-seg" role="group" aria-label="Shot type">
        <button type="button" class="sb-seg-btn ${isChar ? '' : 'active'}" data-act="mode" data-mode="text" ${locked ? 'disabled' : ''}>Text</button>
        <button type="button" class="sb-seg-btn ${isChar ? 'active' : ''}" data-act="mode" data-mode="character"
                ${locked || noCast ? 'disabled' : ''} ${noCast ? 'title="No trained characters on this Mac yet — train one in the Train tab."' : ''}>Character</button>
      </div>
      <select class="sb-select sb-shot-char" data-act="char" aria-label="Who's in this shot" ${locked ? 'disabled' : ''} ${noCast ? 'hidden' : ''}>${opts}</select>
      <select class="sb-select sb-shot-dur" data-act="dur" aria-label="Length" ${locked ? 'disabled' : ''}>${durOpts}</select>
      ${sbEngineChip(engine)}
      ${isChar && /<d>\s*(?!<\/\s*d\s*>)\S/i.test(s.prompt || '')
        ? `<span class="sb-chip sb-chip-voice" title="This shot has spoken lines, so the character's voice loads.">voice</span>`
        : ''}
      <span class="sb-chip sb-chip-pass" data-act="pass" title="Quality is set for the whole film — click to change it">${passLabel}</span>
      <span class="sb-shot-est">${sbShotEst(est)}</span>
      <span class="sb-shot-spacer"></span>
      <span class="sb-seedwrap" hidden>
        <input type="number" class="sb-seed" data-act="seed" value="${seedSet ? s.seed : ''}"
               aria-label="Seed for shot ${n}" ${locked ? 'disabled' : ''}></span>
      <button type="button" class="sb-icon" data-act="up" title="Move up" aria-label="Move shot ${n} up" ${n === 1 || locked ? 'disabled' : ''}>↑</button>
      <button type="button" class="sb-icon" data-act="down" title="Move down" aria-label="Move shot ${n} down" ${n === shots.length || locked ? 'disabled' : ''}>↓</button>
      <button type="button" class="sb-icon ${seedSet ? 'is-set' : ''}" data-act="seedtoggle"
              title="${seedSet ? `Seed ${s.seed} — same number, same roll of the dice. Fix it so the delivery render matches the draft you approved.` : 'Seed — same number, same roll of the dice. Fix it so the delivery render matches the draft you approved.'}"
              aria-label="Seed for shot ${n}">🎲</button>
      <button type="button" class="sb-icon sb-icon-danger" data-act="del" title="${locked ? "This one's already rendering." : 'Delete this shot'}" aria-label="Delete shot ${n}" ${locked && s.status !== 'done' ? 'disabled' : ''}>✕</button>
      <span class="sb-grip" title="Drag to reorder" aria-hidden="true">⠿</span>
    </div>
    <textarea class="sb-shot-prompt" rows="3" spellcheck="false" data-act="prompt"
      ${locked ? 'readonly' : ''} title="${escapeHtml(s.title || '')}">${escapeHtml(s.prompt || '')}</textarea>
    ${s.take_seconds ? `
    <div class="sb-beats-label">One take · ${s.take_seconds} s · ${Math.round(s.take_seconds / 5)} beats <span class="sub">one line per 5 seconds · the first beat is the prompt above</span></div>
    <textarea class="sb-shot-prompt sb-beats" rows="${Math.min(12, Math.round(s.take_seconds / 5))}" spellcheck="false" data-act="beats"
      ${locked ? 'readonly' : ''}>${escapeHtml((s.beats || []).join('\n'))}</textarea>` : ''}
    ${mine.length ? `<div class="sb-shot-err">${mine.map(sbErrRow).join('')}</div>` : ''}
    ${failBlock}
    ${stillBlock}${outBlock}
  </li>`;
}

function sbRenderTally(shots, r) {
  const keep = shots.filter(s => s.grade === 'keep').length;
  const rr = shots.filter(s => s.grade === 'reroll').length;
  const cut = shots.filter(s => s.grade === 'cut').length;
  const graded = keep + rr + cut;
  const un = shots.length - graded;
  sbEl('sbTallyText').innerHTML = graded === 0 ? 'nothing graded yet'
    : `<b>${keep} keep</b> · ${rr} re-roll · ${cut} cut${un ? ` · ${un} ungraded` : ''}`;
  // The tally bar REPLACES the action bar the moment a clip exists — which,
  // after a Stop or a single retry, would leave no way to render the shots
  // that never got a draft. So it carries that button when there are any.
  const pending = shots.filter(s => s.status !== 'skipped' && !s.draft_output).length;
  const resume = sbEl('sbResumeBtn');
  resume.hidden = !pending || !!(r && r.rendering);
  resume.textContent = `Render ${pending} remaining`;
  const rw = sbEl('sbRewriteBtn');
  rw.hidden = !rr;
  rw.textContent = rr === 1 ? 'Rewrite 1 shot' : `Rewrite ${rr} shots`;
  const fin = sbEl('sbFinishBtn');
  fin.textContent = keep ? `Finish ${keep} keeper${keep === 1 ? '' : 's'}` : 'Finish keepers';
  fin.disabled = !keep;
  fin.title = keep ? '' : 'Mark at least one shot KEEP first.';
  const canExport = shots.some(s => s.final_output);
  sbEl('sbExportBtn').hidden = !canExport;
  sbEl('sbExportNote').hidden = !canExport;
  // The timeline works off ANY rendered clip — draft or delivery — because
  // that is the same selection the exporter makes (_sbe_board_clips).
  const tl = sbEl('sbTimelineBtn');
  if (tl) tl.hidden = !shots.some(s => s.draft_output || s.final_output);
}

// The shots that never got a draft — after a Stop, or when a shot was added by
// hand to a film that already rendered.
function sbRenderRemaining() {
  const shots = (((SB.payload || {}).board) || {}).shots || [];
  const ns = shots.filter(s => s.status !== 'skipped' && !s.draft_output).map(s => s.n);
  if (!ns.length) return;
  const est = (SB.payload || {}).estimate || {};
  if (!confirm(
      `Render ${ns.length} draft${ns.length === 1 ? '' : 's'}?\n\n` +
      `${sbFmtWall(est.total_secs).replace(/^about /, 'About ')} on this Mac. ` +
      'The shots you already have are untouched.\n\n' +
      'You can pause or stop after any shot.')) return;
  sbRenderPass('draft', ns);
}

function sbRenderRunBar(r) {
  const bar = sbEl('sbRunBar');
  const shots = (r.board || {}).shots || [];
  if (!r.rendering) { bar.hidden = true; return; }
  bar.hidden = false;
  const cur = (LAST_STATUS && LAST_STATUS.current) || null;
  const tag = cur ? _sbTagOf(cur) : null;
  const active = (tag && tag.id === SB.id) ? shots.find(s => s.n === tag.n) : null;
  const outKey = (r.pass === 'final') ? 'final_output' : 'draft_output';
  const done = shots.filter(s => s[outKey]).length;
  const paused = LAST_STATUS && LAST_STATUS.paused;
  sbEl('sbRunTitle').textContent = paused
    ? `Paused · ${done} of ${shots.length} done`
    : (active ? (tag.still ? `Still for shot ${active.n} of ${shots.length}` : `Shot ${active.n} of ${shots.length}`)
              : `${done} of ${shots.length} done`);
  let sub = '';
  if (active) sub = (tag.still ? 'making the still · ' : '') + `S${String(active.n).padStart(2, '0')} · ${snippet(active.prompt, 40)}`;
  // The remaining time is a SERVER number. /status.eta_sec is the sum of
  // per-job ETAs and is trustworthy when every queued job is this film's.
  const q = (LAST_STATUS && LAST_STATUS.queue) || [];
  const allMine = q.length && q.every(j => { const t = _sbTagOf(j); return t && t.id === SB.id; });
  if (allMine && LAST_STATUS.eta_sec) sub += ` · ${sbFmtWall(LAST_STATUS.eta_sec)} left`;
  else {
    // THE FILM'S OWN NUMBER when the queue's is not ours to read: the sum
    // of the per-shot estimates for what this pass has not landed yet,
    // from the same cost model the summary cell prices the film with.
    const per = r.per_shot_est || {};
    const left = shots.filter(s => !s[outKey] && s.status !== 'skipped')
      .reduce((a, s) => a + (Number(per[String(s.n)]) || 0), 0);
    if (left > 0) sub += ` · ${sbFmtWall(left)} left`;
  }
  sbEl('sbRunSub').textContent = sub;
  sbEl('sbPauseBtn').textContent = paused ? 'Resume' : 'Pause';
  sbEl('sbRunDots').innerHTML = shots.map(s => {
    const cls = s[outKey] ? 'is-done'
              : s.status === 'failed' ? 'is-failed'
              : s.status === 'rendering' ? 'is-running'
              : s.status === 'skipped' ? 'is-cut' : '';
    return `<button type="button" class="sb-dot ${cls}" onclick="sbScrollToShot(${s.n})"
      title="S${String(s.n).padStart(2, '0')} · ${escapeHtml(snippet(s.prompt, 40))}"
      aria-label="Shot ${s.n}"></button>`;
  }).join('');
  // Designed now, empty today: fill from the server's preview_url the day one
  // exists. No layout shift either way — the box is always the same size.
  const pv = ((cur || {}).progress || {}).preview_url;
  sbEl('sbRunThumb').innerHTML = pv
    ? `<img src="${escapeHtml(pv)}?t=${Date.now()}" alt="">`
    : `<svg width="26" height="26" viewBox="0 0 256 256" style="opacity:.18" aria-hidden="true"><use href="#ph-film-slate"/></svg>`;
}

function _sbTagOf(job) {
  // `sb:<board>#<n>` is a shot's clip; `sb:<board>#<n>:still` is the anchor
  // still made before it. Both belong to the film.
  const m = /^sb:([^#]+)#(\d+)(:still)?$/.exec(((job || {}).params || {}).session_tag || '');
  return m ? { id: m[1], n: Number(m[2]), still: !!m[3] } : null;
}
function sbScrollToShot(n) {
  const el = document.querySelector(`.sb-shot[data-n="${n}"]`);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
function sbOpenShotClip(n) {
  const s = (((SB.payload || {}).board) || {}).shots.find(x => x.n === n);
  const clip = s && (s.final_output || s.draft_output);
  if (!clip) return;
  // Opening a clip IS the request for the player, so it says so outright rather than
  // dropping back to 'auto' and hoping auto agrees. Since auto now means "stay on the
  // list", the old line would have opened a clip into a hidden player.
  sbSetStage('player');
  selectOutput(clip);
}

// ---- editing ---------------------------------------------------------------
function sbShotById(n) {
  return (((SB.payload || {}).board) || {}).shots.find(s => s.n === n);
}

// Every edit patches the board client-side then POSTs it. The server
// re-validates, re-injects triggers, renumbers and returns a fresh estimate —
// there is NO client-side validation at all, which is what stops the panel's
// copy from drifting away from storyboard.py.
function sbQueueSave(immediate) {
  if (SB.saveTimer) clearTimeout(SB.saveTimer);
  SB.saveTimer = setTimeout(() => { SB.saveTimer = null; sbFlushSave(); },
                            immediate ? 0 : 600);
}

// While a save is in flight the user can keep typing, and the reply carries the
// board as it was when we SENT it. Replacing SB.payload with that reply would
// silently roll those keystrokes back — the same class of bug as the tick
// clobbering the textarea. So: whatever is in a focused editor right now is the
// truth, and it is re-applied over the reply (and re-saved).
function sbAdoptLiveEdits(payload) {
  let dirty = false;
  document.querySelectorAll('#sbShots .sb-shot').forEach(li => {
    const n = Number(li.dataset.n);
    const shot = (payload.board.shots || []).find(x => x.n === n);
    if (!shot) return;
    const box = li.querySelector('textarea[data-act="prompt"]');
    if (box && document.activeElement === box && box.value !== shot.prompt) {
      shot.prompt = box.value; dirty = true;
    }
    const note = li.querySelector('textarea[data-act="note"]');
    if (note && document.activeElement === note && note.value !== (shot.note || '')) {
      shot.note = note.value; dirty = true;
    }
  });
  return dirty;
}

async function sbFlushSave() {
  if (SB.saveInFlight) { SB.saveAgain = true; return; }
  const board = ((SB.payload || {}).board);
  if (!board) return;
  SB.saveInFlight = true;
  try {
    const r = await (await fetch('/storyboard/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: SB.id, board: board }),
    })).json();
    if (r && r.ok) {
      SB.payload = r;
      if (sbAdoptLiveEdits(r)) SB.saveAgain = true;
      sbRenderPlan(r);
    }
  } catch (e) {
  } finally {
    SB.saveInFlight = false;
    if (SB.saveAgain) { SB.saveAgain = false; sbQueueSave(true); }
  }
}

function sbShotAction(n, act, el, ev) {
  const board = ((SB.payload || {}).board);
  const s = sbShotById(n);
  if (!board || !s) return;
  switch (act) {
    case 'mode': {
      const m = el.dataset.mode;
      if (m === 'character') {
        // Nobody cast — take the first installed character rather than saving
        // an incoherent shot. The server injects the trigger on save.
        if (!s.character_id) {
          const first = ((SB.payload.characters || [])[0] || {}).id;
          if (!first) { phosToast('No trained characters on this Mac yet.'); return; }
          s.character_id = first;
        }
        s.mode = 'character';
      } else {
        // Text clears the cast but LEAVES THE PROMPT ALONE — deleting someone's
        // words is never the right default.
        delete s.character_id; delete s.trigger;
        s.mode = 'text';
      }
      break;
    }
    case 'char': {
      const v = el.value;
      if (v) { s.character_id = v; s.mode = 'character'; }
      else { delete s.character_id; delete s.trigger; s.mode = 'text'; }
      break;
    }
    case 'dur': s.duration_s = Number(el.value); break;
    case 'prompt': s.prompt = el.value; break;
    case 'seed': s.seed = el.value === '' ? -1 : Number(el.value); break;
    case 'seedtoggle': {
      const wrap = el.closest('.sb-shot-head').querySelector('.sb-seedwrap');
      if (wrap) { wrap.hidden = !wrap.hidden; if (!wrap.hidden) wrap.querySelector('.sb-seed').focus(); }
      return;
    }
    case 'pass':
      sbEl('sbQualitySection').open = true;
      sbEl('sbQualitySection').scrollIntoView({ behavior: 'smooth', block: 'center' });
      phosToast('Quality is set for the whole film.');
      return;
    case 'up': case 'down': {
      const i = board.shots.indexOf(s);
      const j = act === 'up' ? i - 1 : i + 1;
      if (j < 0 || j >= board.shots.length) return;
      board.shots.splice(j, 0, board.shots.splice(i, 1)[0]);
      board.shots.forEach((x, k) => { x.n = k + 1; });
      break;
    }
    case 'del': {
      if (s.draft_output || s.final_output) {
        if (!confirm(`Remove shot ${n} from the film?\n\nThe clip stays in mlx_outputs/.`)) return;
      } else {
        SB.lastUndo = { index: board.shots.indexOf(s), shot: JSON.parse(JSON.stringify(s)) };
        const t = phosToast(`Shot ${n} removed.`, { kind: 'success', duration: 6000 });
        if (t) {
          const u = document.createElement('button');
          u.className = 'phos-toast-undo'; u.textContent = 'Undo';
          u.style.pointerEvents = 'auto';
          u.onclick = () => { sbUndoDelete(); t.remove(); };
          t.appendChild(u);
        }
      }
      board.shots = board.shots.filter(x => x !== s);
      board.shots.forEach((x, k) => { x.n = k + 1; });
      break;
    }
    case 'grade': {
      const g = (s.grade === el.dataset.g) ? null : el.dataset.g;
      sbGrade(n, g, s.note || '');
      return;
    }
    case 'note': s.note = el.value; sbGrade(n, s.grade, el.value); return;
    case 'retry': sbRenderPass(SB.payload.pass || 'draft', [n]); return;
    case 'restill': sbRestill(n); return;
    case 'cut': sbGrade(n, 'cut', s.note || ''); return;
    default: return;
  }
  sbRenderPlan(SB.payload);
  sbQueueSave(act !== 'prompt');
}

function sbUndoDelete() {
  const board = ((SB.payload || {}).board);
  if (!board || !SB.lastUndo) return;
  board.shots.splice(SB.lastUndo.index, 0, SB.lastUndo.shot);
  board.shots.forEach((x, k) => { x.n = k + 1; });
  SB.lastUndo = null;
  sbRenderPlan(SB.payload);
  sbQueueSave(true);
}

function sbAddShot() {
  const board = ((SB.payload || {}).board);
  if (!board) return;
  // Follow the film's own rule rather than hardcoding LTX: in an Auto film on a
  // machine with the H3 pack, an uncast shot belongs on H3 like every other
  // uncast shot — otherwise the card contradicts the note printed above it.
  const em = (SB.payload || {}).engine_mode || 'auto';
  const eng = !((SB.payload || {}).h3_available) ? 'ltx' : (em === 'ltx' ? 'ltx' : 'h3');
  board.shots.push({ n: board.shots.length + 1, mode: 'text', engine: eng,
                     prompt: '', duration_s: 5, seed: -1, refs: [], status: 'pending' });
  sbRenderPlan(SB.payload);
  sbQueueSave(true);
  const last = document.querySelector('.sb-shot:last-child .sb-shot-prompt');
  if (last) last.focus();
}

function sbTitleSave() {
  const board = ((SB.payload || {}).board);
  if (!board) return;
  board.title = sbEl('sbTitle').value;
  sbQueueSave(true);
}

async function sbGrade(n, grade, note) {
  const fd = new URLSearchParams();
  fd.set('id', SB.id); fd.set('n', String(n));
  if (grade) fd.set('grade', grade);
  fd.set('note', note || '');
  try {
    const r = await (await fetch('/storyboard/grade', { method: 'POST', body: fd })).json();
    if (r && r.ok) { SB.payload = r; sbRenderPlan(r); }
  } catch (e) {}
}

function sbFixError(fix, n, code) {
  const board = ((SB.payload || {}).board);
  if (!board) return;
  const s = n ? sbShotById(n) : null;
  if (fix === 'add') return sbAddShot();
  if (fix === 'renumber') {
    board.shots.forEach((x, k) => { x.n = k + 1; });
  } else if (fix === 'delete' && n) {
    board.shots.splice(n - 1, 1);
    board.shots.forEach((x, k) => { x.n = k + 1; });
  } else if (fix === 'text' && s) {
    s.mode = 'text'; delete s.character_id; delete s.trigger; s.refs = [];
  } else if (fix === 'focus' && n) {
    const el = document.querySelector(`.sb-shot[data-n="${n}"] .sb-shot-prompt`);
    if (el) el.focus();
    return;
  } else if (fix === 'pickchar' && n) {
    const el = document.querySelector(`.sb-shot[data-n="${n}"] .sb-shot-char`);
    if (el) { el.focus(); if (el.showPicker) { try { el.showPicker(); } catch (e) {} } }
    return;
  } else if (fix === 'trigger' && s) {
    // The server owns ensure_trigger(); a save with the character still set is
    // all it takes, and the canonical text comes back in the reply.
    sbQueueSave(true);
    return;
  } else if (fix === 'dur5' && s) {
    s.duration_s = 5;
  } else if (fix === 'clearrefs' && s) {
    s.refs = [];
  } else if (fix === 'cap') {
    // Both passes, to whatever THIS Mac delivers at the quality each already
    // asks for — looked up in the server's clamped table, never computed here.
    const cv = SB_BOOT.canvases || {};
    ['draft', 'final'].forEach(k => {
      const p = (board.policy || {})[k];
      if (!p) return;
      let fit = cv[p.quality];
      if (!fit) {                                  // a hand-edited quality name
        p.quality = (k === 'draft' ? 'quick' : 'balanced');
        fit = cv[p.quality];
      }
      if (fit) { p.width = fit.width; p.height = fit.height; }
    });
  }
  sbRenderPlan(SB.payload);
  sbQueueSave(true);
}

// ---- render ----------------------------------------------------------------
function sbRenderDrafts() {
  const est = (SB.payload || {}).estimate || {};
  const nshots = est.shots || 0;
  const loads = est.pipeline_loads || 1;
  if (!confirm(
      `Render ${nshots} draft${nshots === 1 ? '' : 's'}?\n\n` +
      `${sbFmtWall(est.total_secs).replace(/^about /, 'About ')} on this Mac. ` +
      `${loads === 1 ? 'One model load, then every shot back to back.' : `${loads} model loads, grouped.`}\n` +
      'Clips land in mlx_outputs/ and show up in your gallery like any other render.\n\n' +
      'You can pause or stop after any shot.')) return;
  sbRenderPass('draft', null);
}

async function sbRenderPass(passName, only) {
  const fd = new URLSearchParams();
  fd.set('id', SB.id); fd.set('pass', passName);
  if (only && only.length) fd.set('only', only.join(','));
  let r;
  try { r = await (await fetch('/storyboard/render', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) {
    phosToast(r.error || 'Could not start the render.', { kind: 'danger', duration: 6000 });
    return;
  }
  sbLoad(SB.id);
  poll();
}

async function sbFinish() {
  const shots = (((SB.payload || {}).board) || {}).shots || [];
  const keep = shots.filter(s => s.grade === 'keep').map(s => s.n);
  if (!keep.length) return;
  const fd = new URLSearchParams();
  fd.set('id', SB.id); fd.set('pass', 'final'); fd.set('only', keep.join(','));
  let est = {};
  try { est = ((await (await fetch('/storyboard/estimate', { method: 'POST', body: fd })).json()) || {}).estimate || {}; }
  catch (e) {}
  const pol = (((SB.payload || {}).board) || {}).policy || {};
  const f = pol.final || {};
  if (!confirm(
      `Finish ${keep.length} shot${keep.length === 1 ? '' : 's'}?\n\n` +
      `Delivery pass, ${f.width}×${f.height}, ${(f.quality || '').replace(/^./, c => c.toUpperCase())}. ` +
      `${sbFmtWall(est.total_secs).replace(/^about /, 'About ')} on this Mac.\n` +
      "Each shot re-renders at its draft's seed, so you get the take you approved — bigger.\n" +
      'The drafts stay in your gallery.')) return;
  sbRenderPass('final', keep);
}

async function sbRewrite() {
  const shots = (((SB.payload || {}).board) || {}).shots || [];
  const ns = shots.filter(s => s.grade === 'reroll').map(s => s.n);
  if (!ns.length) return;
  if (!confirm(
      `Rewrite ${ns.length} shot${ns.length === 1 ? '' : 's'}?\n\n` +
      `The planner loads again (~1 min) and rewrites just ${ns.length === 1 ? 'this one' : 'these'}, using your notes.\n` +
      'The rest of the film is untouched. Nothing renders while it runs.')) return;
  const fd = new URLSearchParams();
  fd.set('id', SB.id); fd.set('ns', ns.join(','));
  let r;
  try { r = await (await fetch('/storyboard/replan-shots', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'Could not start the rewrite.', { kind: 'danger', duration: 6000 }); return; }
  sbShow('planning'); sbSetPlanningStage('load'); sbLoad(SB.id);
}

function sbStopShot() { api('/stop', 'POST').then(poll); }

async function sbStopFilm() {
  const shots = (((SB.payload || {}).board) || {}).shots || [];
  const waiting = shots.filter(s => s.status === 'queued').length;
  if (!confirm('Stop this storyboard?\n\n' +
      `The shot that's rendering is cancelled — its clip is lost. The ${waiting} shot${waiting === 1 ? '' : 's'} still waiting ${waiting === 1 ? 'is' : 'are'} removed from the queue.\n` +
      'Everything already rendered stays.')) return;
  const fd = new URLSearchParams(); fd.set('id', SB.id);
  try { await fetch('/storyboard/stop', { method: 'POST', body: fd }); } catch (e) {}
  sbLoad(SB.id); poll();
}

async function sbExport() {
  // Export used to be a handful of file copies and came back instantly. It now
  // ALSO encodes the whole film, which takes as long as an encode takes — so
  // the button has to hold itself shut, or an impatient second click starts a
  // second ffmpeg writing the same file as the first.
  const btn = sbEl('sbExportBtn');
  if (btn.dataset.busy === '1') return;
  const prev = btn.textContent;
  btn.dataset.busy = '1';
  btn.disabled = true;
  btn.textContent = 'Assembling…';
  const fd = new URLSearchParams(); fd.set('id', SB.id);
  let r;
  try { r = await (await fetch('/storyboard/export', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  finally { btn.dataset.busy = ''; btn.disabled = false; btn.textContent = prev; }
  if (!r.ok) { phosToast(r.error || 'Export failed.', { kind: 'danger' }); return; }
  // The film is best-effort: the folder + shot list always land, so a failed
  // assembly is a NOTE on a successful export, never a failed export.
  if (r.film_name) {
    phosToast(`Exported ${r.files.length} clips + ${r.film_name} `
              + `(${Math.round(r.film_duration || 0)} s) to ${r.dir}`,
              { kind: 'success', duration: 8000 });
    // AND SHOW IT. A toast that fades is what made the finished film invisible:
    // the panel wrote a film, said so for eight seconds, and then left the user
    // on a list of individual shots. The export ends on the film it made.
    sbFilmOpen({ focus: r.film_name });
  } else {
    phosToast(`Exported ${r.files.length} clips + a shot list to ${r.dir}. `
              + `The single film could not be assembled: ${r.film_error || 'unknown reason'}`,
              { duration: 9000 });
  }
}

// ---- THE FILM SCREEN -------------------------------------------------------
// The seventh stage state, and the one the other six are for. Reads
// /storyboard/films — the folder both assemblers already write into — so this
// screen invents no storage and can never disagree with what is on disk.
async function sbFilmOpen(opts) {
  if (!SB.id) return;
  const o = opts || {};
  if (o.focus) SB.filmOpen = String(o.focus).split('/').pop();
  // Never show the LAST board's film while this board's list is in flight.
  if (SB.filmsFor !== SB.id) { SB.films = []; sbEl('sbFilmBody').innerHTML = ''; }
  sbShow('film');
  const stage = sbEl('sbStage');
  if (stage) stage.scrollTop = 0;
  sbEl('sbFilmStatus').textContent = 'looking…';
  await sbFilmLoad();
}
async function sbFilmLoad() {
  let r;
  try {
    r = await (await fetch('/storyboard/films?id=' + encodeURIComponent(SB.id))).json();
  } catch (e) { r = null; }
  if (!r || !r.ok) {
    SB.films = [];
    sbEl('sbFilmStatus').textContent = '';
    sbEl('sbFilmBody').innerHTML =
      '<div class="sb-empty"><div class="sb-empty-title">Could not read the films folder.</div>'
      + '<div class="sb-empty-sub">' + escapeHtml((r && r.error) || 'the panel did not answer')
      + '</div></div>';
    return;
  }
  SB.films = r.films || [];
  SB.filmsFor = SB.id;
  SB.filmDir = r.dir || '';
  SB.filmShort = r.dir_short || r.dir || '';
  sbFilmPaint();
  sbRailPaint();
}
function sbFilmPaint() {
  const body = sbEl('sbFilmBody');
  const status = sbEl('sbFilmStatus');
  if (!body) return;
  const film = sbFilmPick(SB.films, SB.filmOpen);
  if (!film) {
    // Honest, and it names the button that would fix it. The old behaviour was
    // to show nothing at all and let the user conclude the film didn't exist.
    const shots = ((((SB.payload || {}).board) || {}).shots) || [];
    const clips = shots.filter(s => s.draft_output || s.final_output).length;
    status.textContent = 'nothing rendered yet';
    body.innerHTML = `
      <div class="sb-empty">
        <div class="sb-empty-icon"><svg width="56" height="56" viewBox="0 0 256 256" aria-hidden="true"><use href="#ph-film-slate"/></svg></div>
        <div class="sb-empty-title">No film yet.</div>
        <div class="sb-empty-sub">${clips
          ? 'Arrange the ' + clips + ' clip' + (clips === 1 ? '' : 's') + ' you have on the timeline and render. The finished film lands in <code>' + escapeHtml(SB.filmShort || 'mlx_outputs/storyboards/') + '</code> and appears here.'
          : 'Render some shots first — a film is the shots, joined. Nothing to arrange yet.'}</div>
        ${clips ? '<div class="sb-film-actions"><button type="button" class="primary" onclick="sbGo(\'arrange\')">Open the timeline</button></div>' : ''}
      </div>`;
    return;
  }
  SB.filmOpen = film.name;
  const dims = (film.width && film.height) ? `${film.width}×${film.height}` : '';
  // The header stays quiet once there is a film on screen — the facts row below
  // says all of it, and saying it twice is how a screen starts feeling padded.
  status.textContent = '';
  const others = (SB.films || []).filter(f => f.name !== film.name);
  body.innerHTML = `
    <div class="sb-film-stage">
      <video id="sbFilmVideo" controls playsinline preload="metadata"
             src="${escapeHtml(film.url)}"></video>
    </div>
    <div class="sb-film-name">
      <span class="sb-film-kind">${escapeHtml(sbFilmKind(film))}</span>
      <code>${escapeHtml(film.name)}</code>
    </div>
    <div class="sb-summary">
      <div class="sb-sum-cell"><b>${film.duration ? escapeHtml(sbFmtClock(film.duration)) : '—'}</b><span>${film.clips ? film.clips + ' shots joined' : 'runtime'}</span></div>
      <div class="sb-sum-cell"><b>${escapeHtml(dims || '—')}</b><span>${escapeHtml(sbFmtBytes(film.bytes))} on disk</span></div>
      <div class="sb-sum-cell"><b>${escapeHtml(sbFmtAgo(film.at) || '—')}</b><span>${escapeHtml(sbFilmKind(film).toLowerCase())}</span></div>
    </div>
    <div class="sb-film-where">Lives in <code>${escapeHtml(SB.filmShort || SB.filmDir)}</code></div>
    <div class="sb-film-actions">
      <button type="button" class="ghost-btn" onclick="sbFilmReveal()">
        <svg class="ph" aria-hidden="true"><use href="#ph-folder-simple"/></svg>Show in Finder</button>
      <button type="button" class="ghost-btn" onclick="sbGo('edit')"
              title="Back to the cut this was assembled from — re-cut it in the Editor and render again.">Re-cut in the Editor</button>
    </div>
    ${others.length ? `
    <header class="carousel-head" style="margin-top:6px"><h3>Earlier films</h3></header>
    <ul class="row-list sb-filmlist">
      ${others.map(f => `
      <li onclick="sbFilmSelect('${escapeHtml(f.name)}')">
        <span class="ttl">${escapeHtml(f.name)}</span>
        <span class="params">${f.duration ? escapeHtml(sbFmtClock(f.duration)) + ' · ' : ''}${escapeHtml(sbFmtBytes(f.bytes))}</span>
        <span class="badge">${escapeHtml(sbFmtAgo(f.at))}</span>
      </li>`).join('')}
    </ul>` : ''}`;
}
function sbFilmSelect(name) {
  SB.filmOpen = name;
  sbFilmPaint();
}
async function sbFilmReveal() {
  const fd = new URLSearchParams();
  fd.set('id', SB.id);
  if (SB.filmOpen) fd.set('name', SB.filmOpen);
  let r;
  try { r = await (await fetch('/storyboard/reveal', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) phosToast(r.error || 'Could not open the folder.', { kind: 'danger' });
}

// ---- board lists -----------------------------------------------------------
function sbBoardChip(b) {
  if (b.planning) return 'planning';
  if (b.running) return 'rendering';
  if (b.failed) return `${b.failed} failed`;
  // `done` counts shots THIS PANEL rendered as jobs; `clips` counts what is on
  // disk. A board of imported or restored clips has done === 0, and the chip
  // called it "plan only" while seven finished shots sat in the folder.
  if (!b.done) return b.clips ? `${b.clips} clips` : 'plan only';
  if (b.done >= b.shots) return 'drafts done';
  // Partway and idle — stopped, or one shot retried. Saying "rendering" here
  // would be the chip claiming something the machine isn't doing.
  return `${b.done} of ${b.shots}`;
}
// The row's trailing slot names the FURTHEST place this board has got to, and
// goes there. One slot, never two.
//
// The owner could not find the editor ("it doesn't even have a button") and
// could not find his film ("where is the finalized clip?"). Both answers are
// the same answer: from the list of films, the next thing you want is one
// click away, and the row says which thing that is.
function sbRowAction(b) {
  if (b.film) {
    const len = b.film.duration ? ' · ' + sbFmtClock(b.film.duration) : '';
    return `<button class="sb-row-go" title="Watch the finished film"
        onclick="event.stopPropagation();sbOpenAt('${escapeHtml(b.id)}','film')">
        <svg class="ph" aria-hidden="true"><use href="#ph-film-slate"/></svg>Film${escapeHtml(len)}</button>`;
  }
  // `clips` is what is on disk; `done` is only what this panel rendered as a
  // job. A board of imported clips is editable and must show the way in.
  if (b.clips || b.done) {
    return `<button class="sb-row-go" title="Open this film in the Editor"
        onclick="event.stopPropagation();sbOpenAt('${escapeHtml(b.id)}','edit')">
        <svg class="ph" aria-hidden="true"><use href="#ph-scissors"/></svg>Open in Editor</button>`;
  }
  return '';
}
function sbRenderBoardLists() {
  const rows = SB.boards.map(b => `
    <li data-id="${escapeHtml(b.id)}" class="${b.running || b.planning ? 'is-live' : ''}"
        onclick="sbOpen('${escapeHtml(b.id)}')">
      <span class="ttl">${escapeHtml(b.title || 'Untitled film')}</span>
      <span class="params">${b.shots} shots · ${b.done} rendered</span>
      <span class="badge">${escapeHtml(sbBoardChip(b))}</span>
      ${sbRowAction(b)}
      <button title="Delete this storyboard" onclick="event.stopPropagation();sbDeleteBoard('${escapeHtml(b.id)}','${escapeHtml((b.title || '').replace(/'/g, ''))}')"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>
    </li>`).join('');
  const full = sbEl('sbBoardList');
  if (full) full.innerHTML = rows || '<li class="empty-state"><span></span><span>No storyboards yet</span><span></span><span></span></li>';
  const mini = sbEl('sbBoardListMini');
  if (mini) mini.innerHTML = rows;
}
async function sbDeleteBoard(id, title) {
  if (!confirm(`Delete "${title || 'this storyboard'}"?\n\nDeletes the plan. The clips it already rendered stay in mlx_outputs/.`)) return;
  const fd = new URLSearchParams(); fd.set('id', id);
  const r = await (await fetch('/storyboard/delete', { method: 'POST', body: fd })).json();
  if (!r.ok) { phosToast(r.error || 'Could not delete.', { kind: 'danger' }); return; }
  if (SB.id === id) { SB.id = ''; SB.payload = null; try { localStorage.removeItem('phos_sb_open'); } catch (e) {} }
  await sbRefreshBoards();
  sbShow(SB.boards.length ? 'list' : 'empty');
}

// ---- shared state: pull a normal generation INTO a film ---------------------
async function sbAddActiveToBoard(chosen) {
  if (!activePath) return;
  const sel = sbEl('sbAddSelect');
  if (!chosen && SB.boards.length > 1 && sel && sel.style.display === 'none') {
    sel.innerHTML = SB.boards.map(b =>
      `<option value="${escapeHtml(b.id)}">${escapeHtml(b.title || 'Untitled film')}</option>`).join('')
      + '<option value="new">— new storyboard —</option>';
    sel.style.display = '';
    return;
  }
  const id = chosen || (SB.boards.length === 1 ? SB.boards[0].id : (SB.boards[0] || {}).id) || 'new';
  if (sel) sel.style.display = 'none';
  const fd = new URLSearchParams();
  fd.set('id', id); fd.set('path', activePath);
  let r;
  try { r = await (await fetch('/storyboard/add-shot', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'Could not add the clip.', { kind: 'danger' }); return; }
  phosToast(`Added to "${r.title || 'the film'}" as shot ${r.n}.`, { kind: 'success' });
  _flashActionDone('sbAddBtn', 'Added');
  sbRefreshBoards();
}

// One badge, one click, no new surface: a gallery card that is a shot of a film
// opens that film.
function sbOpenFromClip(id) {
  workflowSwitch('storyboard');
  setTimeout(() => sbOpen(id), 30);
}

// ---- the one cross-tab hook, called at the end of poll() -------------------
function sbPollHook(s) {
  SB.boards = s.storyboards || [];
  // Repaint the rows only when the rows changed. /status carries the film
  // summary now, so "this board has a film" lights up the moment ffmpeg
  // finishes — without rebuilding a list under the cursor twice a second.
  const sig = JSON.stringify(SB.boards.map(b => [b.id, b.title, b.shots, b.done,
    b.clips, b.failed, b.running, b.planning,
    b.film ? [b.film.name, b.film.at, b.film.count] : 0]));
  if (sig !== SB.boardsSig) {
    SB.boardsSig = sig;
    sbRenderBoardLists();
    if (SB.id) sbRailPaint();
  }
  const live = SB.boards.filter(b => b.running);
  const count = sbEl('sbTabCount');
  if (count) {
    if (live.length) {
      const b = live[0];
      count.hidden = false;
      count.textContent = `${b.done}/${b.shots}`;
    } else { count.hidden = true; count.textContent = ''; }
  }
  const add = sbEl('sbAddBtn');
  if (add) add.style.display = SB.boards.length ? '' : 'none';
  // Plan film is blocked while the worker is busy — constraint 1, made visible.
  const btn = sbEl('sbPlanBtn');
  if (btn && btn.dataset.busy !== '1') {
    const busy = !!s.running || !!(s.queue || []).length;
    const empty = !((sbEl('sbConcept') || {}).value || '').trim();
    btn.disabled = busy || empty;
    btn.title = busy ? 'The renderer is using the memory. Planning can start when the queue is empty.'
              : empty ? 'Write a couple of sentences about the film first.' : '';
  }
  if (document.body.dataset.workflow === 'storyboard' && SB.id) sbRenderRunBar(SB.payload || {});
}

// ---- wiring ----------------------------------------------------------------
document.addEventListener('click', (ev) => {
  const chip = ev.target.closest && ev.target.closest('#sbLengthGroup .q-chip');
  if (chip) { sbSetShots(chip.dataset.sbShots === 'take' ? 'take' : Number(chip.dataset.sbShots)); return; }
  const tk = ev.target.closest && ev.target.closest('#sbTakeGroup [data-sb-take]');
  if (tk) { sbSetTake(tk.dataset.sbTake); return; }
  const eng = ev.target.closest && ev.target.closest('#sbEngineGroup [data-sb-engine]');
  if (eng) { sbSetEngineMode(eng.dataset.sbEngine); return; }
  const reng = ev.target.closest && ev.target.closest('#sbReplanEngineGroup [data-sb-engine]');
  if (reng) {
    const h3 = sbH3Installed();
    if (reng.dataset.sbEngine === 'h3' && !h3.available) { sbSetEngineMode('h3'); return; }
    _sbReplanEngineMode = reng.dataset.sbEngine;
    sbRenderReplanEnginePicker();
    return;
  }
  const q = ev.target.closest && ev.target.closest('#sbDraftQuality .pill-btn, #sbFinalQuality .pill-btn');
  if (q) {
    const board = ((SB.payload || {}).board);
    if (!board) return;
    const which = q.closest('#sbDraftQuality') ? 'draft' : 'final';
    // Server-owned, already clamped to this Mac. The literal table that used to
    // live here wrote 1024×576 for Standard on a machine that caps at 768 — an
    // over_cap error the moment you touched the control, with Render disabled
    // and a fix button that did nothing. (GitHub #71)
    const dims = (SB_BOOT.canvases || {})[q.dataset.q];
    if (!dims) return;
    board.policy = board.policy || {};
    board.policy[which] = Object.assign({}, board.policy[which],
      { quality: q.dataset.q, width: dims.width, height: dims.height });
    sbSaveSetting(which === 'draft' ? 'storyboard_draft_quality' : 'storyboard_final_quality', q.dataset.q);
    sbRenderPlan(SB.payload);
    sbQueueSave(true);
    return;
  }
  const act = ev.target.closest && ev.target.closest('#sbShots [data-act]');
  if (act) {
    const li = act.closest('.sb-shot');
    if (li) sbShotAction(Number(li.dataset.n), act.dataset.act, act, ev);
    return;
  }
  const seg = ev.target.closest && ev.target.closest('.sb-runseg');
  if (seg) {
    const ns = (seg.dataset.shots || '').split(',');
    document.querySelectorAll('.sb-shot').forEach(el =>
      el.classList.toggle('is-lit', ns.indexOf(el.dataset.n) !== -1));
  }
});
document.addEventListener('mouseover', (ev) => {
  const seg = ev.target.closest && ev.target.closest('.sb-runseg');
  if (!seg) return;
  const ns = (seg.dataset.shots || '').split(',');
  document.querySelectorAll('.sb-shot').forEach(el =>
    el.classList.toggle('is-lit', ns.indexOf(el.dataset.n) !== -1));
});
document.addEventListener('mouseout', (ev) => {
  if (ev.target.closest && ev.target.closest('.sb-runseg'))
    document.querySelectorAll('.sb-shot.is-lit').forEach(el => el.classList.remove('is-lit'));
});
document.addEventListener('change', (ev) => {
  const el = ev.target.closest && ev.target.closest('#sbShots [data-act]');
  if (!el) return;
  const li = el.closest('.sb-shot');
  if (li) sbShotAction(Number(li.dataset.n), el.dataset.act, el, ev);
});
// Leaving a card is the moment the guard lifts — repaint straight away rather
// than leaving the user looking at a stale card for up to 2 s.
document.addEventListener('focusout', (ev) => {
  if (!ev.target.closest || !ev.target.closest('#sbShots')) return;
  setTimeout(() => {
    if (sbTypingInShots() || !SB.payload) return;
    if (document.body.dataset.workflow !== 'storyboard') return;
    try { sbRenderPlan(SB.payload); } catch (e) {}
  }, 60);
});

let _sbPromptTimer = null;
document.addEventListener('input', (ev) => {
  const bel = ev.target.closest && ev.target.closest('#sbShots textarea[data-act="beats"]');
  if (bel) {
    // The beats of a take: one line per 5 s, kept as a list the length of
    // the take; beat 1 is also the shot's prompt.
    const li = bel.closest('.sb-shot');
    const s = sbShotById(Number(li.dataset.n));
    if (!s) return;
    const n = Math.max(1, Math.round((s.take_seconds || 0) / 5));
    const lines = String(bel.value || '').split('\n').map(x => x.trim()).slice(0, n);
    while (lines.length < n) lines.push('');
    s.beats = lines;
    if (lines[0]) { s.prompt = lines[0]; const pe = li.querySelector('textarea[data-act="prompt"]'); if (pe && pe.value !== lines[0]) pe.value = lines[0]; }
    if (_sbPromptTimer) clearTimeout(_sbPromptTimer);
    _sbPromptTimer = setTimeout(() => sbQueueSave(true), 800);
    return;
  }
  const el = ev.target.closest && ev.target.closest('#sbShots textarea[data-act="prompt"]');
  if (!el) return;
  const li = el.closest('.sb-shot');
  const s = sbShotById(Number(li.dataset.n));
  if (!s) return;
  s.prompt = el.value;
  if (s.take_seconds && Array.isArray(s.beats)) s.beats[0] = el.value;
  sbAutoGrowPrompts(el);
  if (_sbPromptTimer) clearTimeout(_sbPromptTimer);
  _sbPromptTimer = setTimeout(() => sbQueueSave(true), 800);
});
// Grade keys: K / R / C with a card focused. Surfaced in the tally bar's title,
// not as visible chrome.
document.addEventListener('keydown', (ev) => {
  if (document.body.dataset.workflow !== 'storyboard') return;
  const li = document.activeElement && document.activeElement.closest
           && document.activeElement.closest('.sb-shot');
  if (!li || /^(INPUT|TEXTAREA|SELECT)$/.test((ev.target.tagName || ''))) return;
  const g = { k: 'keep', r: 'reroll', c: 'cut' }[ev.key.toLowerCase()];
  if (!g) return;
  const n = Number(li.dataset.n);
  const s = sbShotById(n);
  if (!s || !(s.draft_output || s.final_output)) return;
  ev.preventDefault();
  sbGrade(n, s.grade === g ? null : g, s.note || '');
});
// Drag to reorder — ~20 lines, no library. The ↑ / ↓ buttons are the primary
// affordance (keyboard- and touch-reachable); drag is the enhancement.
let _sbDragN = null;
document.addEventListener('dragstart', (ev) => {
  const li = ev.target.closest && ev.target.closest('.sb-shot');
  if (!li || li.getAttribute('draggable') === 'false') return;
  _sbDragN = Number(li.dataset.n);
  li.classList.add('is-dragging');
  try { ev.dataTransfer.setData('text/plain', String(_sbDragN)); } catch (e) {}
});
document.addEventListener('dragover', (ev) => {
  const li = ev.target.closest && ev.target.closest('.sb-shot');
  if (!li || _sbDragN == null) return;
  ev.preventDefault();
  const r = li.getBoundingClientRect();
  const after = (ev.clientY - r.top) > r.height / 2;
  document.querySelectorAll('.sb-shot').forEach(el =>
    el.classList.remove('sb-drop-before', 'sb-drop-after'));
  li.classList.add(after ? 'sb-drop-after' : 'sb-drop-before');
});
document.addEventListener('drop', (ev) => {
  const li = ev.target.closest && ev.target.closest('.sb-shot');
  if (!li || _sbDragN == null) return;
  ev.preventDefault();
  const board = ((SB.payload || {}).board);
  const from = board.shots.findIndex(s => s.n === _sbDragN);
  let to = board.shots.findIndex(s => s.n === Number(li.dataset.n));
  const r = li.getBoundingClientRect();
  if ((ev.clientY - r.top) > r.height / 2) to += 1;
  if (from < 0 || to < 0) return;
  const moved = board.shots.splice(from, 1)[0];
  board.shots.splice(to > from ? to - 1 : to, 0, moved);
  board.shots.forEach((x, k) => { x.n = k + 1; });
  sbRenderPlan(SB.payload);
  sbQueueSave(true);
});
document.addEventListener('dragend', () => {
  _sbDragN = null;
  document.querySelectorAll('.sb-shot').forEach(el =>
    el.classList.remove('is-dragging', 'sb-drop-before', 'sb-drop-after'));
});

// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  sbSetTake,
  sbToggleSwitchHelp, sbSyncBriefGates, sbRestill,
  sbTrackInput,
  sbEl, sbFmtClock, sbFmtBytes, sbFmtAgo,
  sbFilmKind, sbFilmPick, sbRenderFinalQualities, sbInit,
  sbTeardown, sbConceptInput, sbMustInput, sbLocInput,
  sbToggleRamHelp, sbToggleEngineHelp, sbEngineChip, sbSetEngineMode,
  sbPlan, sbCancelPlan, sbOpenReplan, sbCloseReplan,
  sbReplan, sbTryAgain, sbShowRaw, sbOpen,
  sbOpenAt, sbBackToList, sbRailModel, sbRailPaint,
  sbGo, sbLoad, sbShow, sbSyncStage,
  sbSetStage, sbRenderPlan, sbRenderRemaining, sbAddShot,
  sbTitleSave, sbRenderDrafts, sbFinish, sbRewrite,
  sbStopShot, sbStopFilm, sbExport, sbFilmOpen,
  sbFilmPaint, sbBoardChip, sbRowAction, sbAddActiveToBoard,
  sbOpenFromClip, sbPollHook,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  sbDeleteBoard, sbFilmReveal, sbFilmSelect, sbFixError,
  sbOpenShotClip, sbPickCast, sbScrollToShot,
});
