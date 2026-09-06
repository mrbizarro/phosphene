// webapp/js/loras.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ====== LoRA picker ======
//
// State model: an in-memory list of LoRA entries the user has added.
// Adding can come from "Use" on a CivitAI install or from clicking a
// row in the local list. Each entry:
//   { path, name, strength, trigger_words, civitai_url }
// On every change we mirror the list into the hidden #lorasJson field
// so make_job's parse_loras_from_form picks them up at submit time.

globalThis._activeLoras = [];   // [{path, name, strength, trigger_words, compatible_modes, ...}]
globalThis._knownUserLoras = []; // last list_user_loras() snapshot, for the picker

// Resolve the compat tag the picker should filter by, based on the
// CURRENT (mode + engine) selection. The unified picker lives in ONE
// place but its library list re-filters when the user switches mode
// (Manual T2V/I2V/keyframe/extend → "video") or when they pick a
// different engine in Image Studio (Qwen-Edit → "image:qwen", FLUX.2
// Klein-Edit → "image:flux2", etc.). Returns "" to mean "no filter,
// show everything" — the back-compat path when we can't determine
// the active engine.
function _currentLoraModeFilter() {
  // `currentMode` is a `let` at the top of this <script> block (line ~15016),
  // sharing the same IIFE scope as setMode() and renderLorasList(). It is
  // NOT a property of `window` — `let`/`const` at the script's top level
  // are not auto-attached to globalThis in modern browsers. Reading
  // `window.currentMode` was always undefined, falling through to the
  // hidden #mode input which the image branch of setMode deliberately
  // doesn't update (to avoid an accidental form submit firing a video
  // render with stale fields). Net effect: the picker filter always
  // saw 't2v' even after the user clicked Studio. Read currentMode
  // directly via lexical capture so the mode-aware filter actually
  // works.
  const mode = (currentMode || document.getElementById('mode')?.value || 't2v');
  if (mode !== 'image') {
    // Video modes are no longer one lane: LTX and Hailuo H3 have separate
    // libraries in separate directories, and neither can load the other's
    // adapters. The tag comes off the ENGINES table (`lora_tag`) so a third
    // video engine is one registry entry, not another branch here.
    try {
      const e = (typeof engineById === 'function') ? engineById(currentEngine()) : null;
      if (e && e.lora_tag) return e.lora_tag;
    } catch (_) { /* fall through to the built-in lane */ }
    return 'video';
  }
  // Image Studio: read the engine override + map to compat tag.
  const eng = (document.getElementById('imgStudioEngine')?.value || 'auto').toLowerCase();
  if (eng.startsWith('qwen_edit')) return 'image:qwen';
  if (eng.startsWith('ideogram'))  return 'image:ideogram';
  if (eng.startsWith('flux2_edit') || eng.startsWith('flux2'))  return 'image:flux2';
  if (eng.startsWith('flux1') || eng === 'flux1_inline')        return 'image:flux1';
  if (eng.startsWith('kontext'))   return 'image:kontext';
  if (eng.startsWith('z_image'))   return 'image:z_image';
  if (eng.startsWith('hidream'))   return 'image:hidream';
  // Mock engine is image-lane — narrow off LTX video LoRAs the same way
  // 'auto' does. Returning '' here skipped the filter entirely (same bug
  // ab50f12 fixed for 'auto'); video LTX LoRAs would re-appear in the
  // Image Studio picker when the user picked Mock for testing.
  if (eng.startsWith('mock'))      return 'image';
  // 'auto' → fall back to the user's saved engine. Without a server
  // round-trip to /agent/config we don't know exactly which family
  // they saved; show ALL image-lane LoRAs (image:* + unknown) but
  // STILL hide LTX video LoRAs which can never run on mflux. Returning
  // 'image' as a meta-tag — the row filter below treats it as a
  // wildcard matching anything starting with `image:` plus `unknown`,
  // so we narrow off the LTX side without false-hiding the rest.
  // Returning '' here would (and did) skip the filter entirely
  // because `if (modeTag)` is falsy on empty string — that's the
  // bug the user hit when Studio + auto showed all 5 of their LTX
  // LoRAs in the picker.
  return 'image';
}

// Friendly label for the active filter, surfaced in the picker banner
// so the user understands "why don't I see all my LoRAs". Mirrors the
// _currentLoraModeFilter return values.
function _loraFilterLabel(tag) {
  switch (tag) {
    case 'video':         return 'LTX-Video LoRAs (active video mode)';
    case 'video:h3':      return 'Hailuo H3 LoRAs (MiniMax H3 base — separate library)';
    case 'image':         return 'image LoRAs (any mflux family — auto engine)';
    case 'image:qwen':    return 'Qwen-Image / Qwen-Image-Edit LoRAs';
    case 'image:ideogram': return 'Ideogram 4 LoRAs (community formats — none validated yet)';
    case 'image:flux2':   return 'FLUX.2 LoRAs';
    case 'image:flux1':   return 'FLUX.1 LoRAs';
    case 'image:kontext': return 'FLUX.1 Kontext LoRAs';
    case 'image:z_image': return 'Z-Image LoRAs (none expected — most LoRAs trained for other families)';
    case 'image:sdxl':    return 'SDXL LoRAs';
    case 'image:hidream': return 'HiDream LoRAs (LoRA support not yet implemented — picked LoRAs are ignored)';
    default:              return 'all installed LoRAs';
  }
}

// Which LoRA library directory the picker is currently pointed at. Filled by
// refreshLoras() from /loras — one fetch carries both, so flipping engine
// re-labels the empty state without a round-trip.
globalThis._lorasDirs = { ltx: 'mlx_models/loras/', h3: '' };
// Whether the INSTALLED H3 pack's runner takes `--lora` at all. An older
// checkout renders every tier and cannot take an adapter, and offering the
// picker there would be a lie the user only discovers 30 s into a render.
globalThis._h3LoraSupported = false;

// Show / hide the shared LoRA picker for the ACTIVE engine, and re-point its
// "drop files here" line at that engine's directory.
//
// This used to be a CSS fold (`data-ltx-only` on #loraPickerVideoSlot). A fold
// rule can say "one engine"; it cannot say "any engine whose lora_tag is set
// AND, if that engine is H3, whose installed pack has --lora" — which is the
// real condition. So the markup lost its attribute and the decision moved
// here, reading the same ENGINES table the rest of the engine surface does.
function _syncLoraPickerForEngine() {
  const slot = document.getElementById('loraPickerVideoSlot');
  if (!slot) return;
  let e = null;
  try { e = (typeof engineById === 'function') ? engineById(currentEngine()) : null; }
  catch (_) { e = null; }
  const tag = e ? e.lora_tag : 'video';
  let show = !!tag;
  if (show && e && e.id === 'h3') {
    // BOOT.h3.loras.supported is refreshed on every /status tick, so updating
    // an old pack unlocks the picker without a panel restart — the same
    // contract Turbo's download already has.
    show = !!((H3 && H3.loras && H3.loras.supported) || _h3LoraSupported);
  }
  slot.hidden = !show;
  slot.style.display = show ? '' : 'none';
  const dirEl = document.getElementById('lorasDir');
  if (dirEl) {
    dirEl.textContent = (tag === 'video:h3')
      ? (_lorasDirs.h3 || 'the Hailuo H3 pack’s loras/ folder')
      : (_lorasDirs.ltx || 'mlx_models/loras/');
  }
  const importBtn = document.getElementById('h3LoraImportBtn');
  if (importBtn) importBtn.hidden = tag !== 'video:h3';
  if (typeof renderLorasList === 'function') { try { renderLorasList(); } catch (_) {} }
  // Re-serialize: the lane guard in _serializeLoras is what keeps the other
  // engine's chips off the wire, and it has to re-run when the lane changes.
  if (typeof _serializeLoras === 'function') { try { _serializeLoras(); } catch (_) {} }
  if (typeof renderH3LoraSlot === 'function') { try { renderH3LoraSlot(); } catch (_) {} }
}

async function importH3Lora(file) {
  if (!file) return;
  if (!/\.safetensors$/i.test(file.name || '')) {
    alert('Choose a .safetensors Hailuo H3 LoRA file.');
    return;
  }
  const btn = document.getElementById('h3LoraImportBtn');
  // innerHTML, not textContent: the button carries an inline <svg> icon, and
  // restoring a string would leave the icon permanently gone after the first
  // import — a one-way UI decay that only shows up on the SECOND use.
  const original = btn ? btn.innerHTML : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Importing…'; }
  const fd = new FormData();
  fd.append('file', file, file.name);
  try {
    const r = await fetch('/h3/loras/import', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
    await refreshLoras();
    const pairs = `${data.pairs} module pair${data.pairs === 1 ? '' : 's'}`;
    const converted = data.converted ? ' Key namespace converted safely.' : '';
    // The H3 loader applies no alpha, so when a file's own scale isn't 1.0 the
    // strength control is where it gets applied — say the number rather than
    // leaving it in the sidecar for nobody to find.
    const strength = (typeof data.recommended_strength === 'number'
                      && Math.abs(data.recommended_strength - 1) > 1e-6)
      ? ` Recommended strength ${Number(data.recommended_strength.toFixed(4))}.` : '';
    alert(`Imported ${data.filename} (${pairs}).${converted}${strength}`);
  } catch (e) {
    alert(`H3 LoRA import failed: ${e.message || e}`);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = original || 'Import H3 LoRA'; }
  }
}

// The H3-lane LoRA (at most one) currently picked, as a picker row — or null.
function _h3ActiveUserLora() {
  if (!Array.isArray(_activeLoras)) return null;
  for (const a of _activeLoras) {
    const u = Array.isArray(_knownUserLoras)
      ? _knownUserLoras.find(x => x.path === a.path) : null;
    if (u && u.lane === 'h3') return Object.assign({}, u, { strength: a.strength });
  }
  return null;
}

// Adapter-slot control. Only rendered when the conflict is REAL — Turbo
// selected AND an H3 LoRA picked — because H3's runner has one `--lora` and
// there is nothing to choose until both want it. The copy comes from
// BOOT.h3.loras.note so the sentence lives next to the constant that encodes
// the constraint.
function renderH3LoraSlot() {
  const row = document.getElementById('h3LoraSlotRow');
  if (!row) return;
  const lora = _h3ActiveUserLora();
  const turboOn = (document.getElementById('h3_turbo') || {}).value === '1';
  const onH3 = (typeof currentEngine === 'function') && currentEngine() === 'h3';
  // A runner that stacks takes Turbo AND the LoRA — no slot to choose.
  const stacks = !!(H3 && H3.loras && Number(H3.loras.max_stack || 1) > 1);
  const conflict = !!(onH3 && lora && turboOn && !stacks);
  row.hidden = !conflict;
  if (!conflict) {
    // No conflict = no choice to remember. Reset so a later render can't
    // inherit a stale "user" from a LoRA the user has since un-picked.
    const inp = document.getElementById('h3_lora_slot');
    if (inp) inp.value = 'turbo';
    document.querySelectorAll('#h3LoraSlotGroup .pill-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.h3LoraSlot === 'turbo'));
    return;
  }
  const sub = document.getElementById('h3LoraSlotUserSub');
  if (sub) sub.textContent = lora.name || lora.filename || '';
  const note = document.getElementById('h3LoraSlotNote');
  if (note) {
    note.textContent = ((H3 && H3.loras && H3.loras.note) ||
      "H3's runner has one adapter slot, so a LoRA and Turbo can't both run.")
      + ' Choosing your LoRA turns Turbo off and renders at this shape’s own step count.';
  }
  const cur = (document.getElementById('h3_lora_slot') || {}).value || 'turbo';
  document.querySelectorAll('#h3LoraSlotGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.h3LoraSlot === cur));
}

function setH3LoraSlot(slot) {
  const v = (slot === 'user') ? 'user' : 'turbo';
  const inp = document.getElementById('h3_lora_slot');
  if (inp) inp.value = v;
  document.querySelectorAll('#h3LoraSlotGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.h3LoraSlot === v));
  renderH3LoraSlot();
}

function _serializeLoras() {
  // What the helper actually needs is path + strength. Keep the rest in
  // the in-memory list for UI rendering, drop it on the wire. Summary
  // count is updated by renderLorasList() which has fuller state — we
  // don't touch it here to avoid two functions stomping each other.
  //
  // Lane guard (2026-08-09): _activeLoras SURVIVES an engine switch on
  // purpose — flipping LTX → H3 → LTX must not silently unpick the user's
  // chips. But the hidden field is what gets POSTed, so anything from the
  // other video engine's library has to be withheld from THIS submit or an
  // LTX render would be handed an H3 adapter (and vice versa). The server
  // scrubs the same way in make_job; this copy keeps the UI's own summary and
  // the wire in agreement instead of relying on a correction it can't see.
  const _tag = (typeof _currentLoraModeFilter === 'function')
    ? _currentLoraModeFilter() : '';
  const _laneOf = (p) => {
    const u = Array.isArray(_knownUserLoras)
      ? _knownUserLoras.find(x => x.path === p) : null;
    return (u && u.lane) || 'ltx';
  };
  const _inLane = (l) => (_tag === 'video:h3')
    ? _laneOf(l.path) === 'h3'
    : _laneOf(l.path) !== 'h3';
  const slim = _activeLoras.filter(_inLane)
    .map(l => ({ path: l.path, strength: l.strength }));
  document.getElementById('lorasJson').value = JSON.stringify(slim);
  // Train-Character LoRAs are trained against the dev transformer (HQ
  // path) — Quick/Standard run the distilled model at 8 steps, which is
  // a different fine-tune of the same architecture. Fusion succeeds
  // numerically but deltas don't transfer cleanly; the user sees muddy
  // output. Disable those chips when any trained LoRA is attached so
  // the wrong-path landing becomes visible.
  try { updateQualityChipsForLora(); } catch (_) {}
  // Picking (or un-picking) an H3 LoRA is the other half of the
  // single-adapter-slot conflict, so the Adapter row follows the selection.
  if (typeof renderH3LoraSlot === 'function') { try { renderH3LoraSlot(); } catch (_) {} }
}

function updateQualityChipsForLora() {
  const trained = Array.isArray(_activeLoras) && _activeLoras.some(a => {
    const meta = Array.isArray(_knownUserLoras)
      ? _knownUserLoras.find(u => u.path === a.path)
      : null;
    // _knownUserLoras carries the raw sidecar kind ('train_character');
    // 'trained' is only a display-side alias built inside renderLorasList.
    // Match the raw value so the chip disable actually fires.
    return meta && meta.kind === 'train_character';
  });
  document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => {
    const q = b.dataset.quality;
    const incompat = trained && (q === 'quick' || q === 'standard');
    b.classList.toggle('disabled', incompat);
    if (incompat) {
      b.title = 'Trained character LoRA needs the dev transformer (High quality). Quick/Standard run the distilled model — different fine-tune, so the LoRA can\'t reproduce the character faithfully.';
    } else if (b.dataset.tooltipDefault) {
      b.title = b.dataset.tooltipDefault;
    } else {
      b.removeAttribute('title');
    }
  });
  // If the currently-selected preset just became incompatible, bump to High.
  const cur = document.getElementById('quality').value;
  if (trained && (cur === 'quick' || cur === 'standard')) {
    setQuality('high');
  }
}

function addLoraToActive(entry) {
  // Idempotent: same path twice = update strength only.
  const existing = _activeLoras.find(l => l.path === entry.path);
  if (existing) {
    existing.strength = entry.strength;
  } else {
    _activeLoras.push(entry);
  }
  renderLorasList();
  _serializeLoras();
}

function removeLoraFromActive(path) {
  _activeLoras = _activeLoras.filter(l => l.path !== path);
  renderLorasList();
  _serializeLoras();
}

function setLoraStrength(path, strength) {
  const e = _activeLoras.find(l => l.path === path);
  if (!e) return;
  e.strength = Math.max(-2, Math.min(2, parseFloat(strength) || 0));
  _serializeLoras();
}

// --- Ingredients × Character picker -------------------------------------
// Fills the optional character dropdown in Ingredients mode from the trained
// characters in _knownUserLoras (kind=train_character). Selecting one stacks
// the character LoRA on top of the Ingredients IC-LoRA server-side so the SAME
// trained face lands in every composed scene. The character's trigger word
// rides a hidden field; the server prepends it to the Action if missing.
function populateIngredientCharLoras() {
  const sel = document.getElementById('ingredient_char_lora');
  const empty = document.getElementById('ingredientCharEmpty');
  if (!sel) return;
  const chars = (Array.isArray(_knownUserLoras) ? _knownUserLoras : [])
    .filter(u => u && u.kind === 'train_character' && u.ltx_compatible !== false);
  const prev = sel.value;
  sel.innerHTML = '<option value="">None — compose from the reference images only</option>';
  for (const c of chars) {
    const o = document.createElement('option');
    o.value = c.path;
    const trig = (Array.isArray(c.trigger_words) && c.trigger_words[0]) || '';
    o.dataset.trigger = trig;
    o.textContent = (c.name || c.filename || c.path) + (trig ? ` · "${trig}"` : '');
    sel.appendChild(o);
  }
  if (prev && chars.some(c => c.path === prev)) sel.value = prev;
  sel.style.display = chars.length ? '' : 'none';
  if (empty) empty.style.display = chars.length ? 'none' : '';
  onIngredientCharChange();
}

function onIngredientCharChange() {
  const sel = document.getElementById('ingredient_char_lora');
  const tune = document.getElementById('ingredientCharTune');
  const trigHint = document.getElementById('ingredientCharTrigHint');
  const trigField = document.getElementById('ingredient_char_trigger');
  if (!sel) return;
  const opt = sel.options[sel.selectedIndex];
  const trigger = (opt && opt.dataset.trigger) || '';
  if (trigField) trigField.value = trigger;
  const picked = !!sel.value;
  if (tune) tune.style.display = picked ? '' : 'none';
  if (trigHint) {
    trigHint.innerHTML = picked
      ? (trigger
          ? 'Trigger word <code>' + escapeHtml(trigger) + '</code> is added to your Action automatically so the character fires.'
          : 'No trigger word recorded for this character — it may still fire from the LoRA alone.')
      : '';
  }
}

function onIngredientCharStrength(v) {
  const lab = document.getElementById('ingCharStrLabel');
  const field = document.getElementById('ingredient_char_strength');
  const f = parseFloat(v);
  if (lab) lab.textContent = (isNaN(f) ? 1.8 : f).toFixed(1);
  if (field) field.value = String(v);
}

// ---- Updates from CivitAI -------------------------------------------------
// `LORA_UPDATES` maps an installed LoRA's path to the newer version the
// check found. The badge and the Update button read it; the install is the
// same /civitai/download the browser tab uses, so the new file lands with a
// sidecar and the picker learns it on the next refresh.
const LORA_UPDATES = {};
// The check is remembered per browser (with when it ran) so the badges
// survive a reload; a new check replaces it.
try {
  const saved = JSON.parse(localStorage.getItem('phos_lora_updates') || 'null');
  if (saved && saved.items && (Date.now() - (saved.at || 0)) < 7 * 86400e3) {
    saved.items.forEach(it => { LORA_UPDATES[it.path] = it; });
  }
} catch (e) {}
async function checkLoraUpdates() {
  const btn = document.getElementById('loraUpdatesBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Checking…'; }
  try {
    const r = await (await fetch('/loras/updates')).json();
    Object.keys(LORA_UPDATES).forEach(k => delete LORA_UPDATES[k]);
    (r.items || []).forEach(it => { LORA_UPDATES[it.path] = it; });
    try { localStorage.setItem('phos_lora_updates', JSON.stringify({ at: Date.now(), items: r.items || [] })); } catch (e) {}
    const n = (r.items || []).length;
    if (typeof phosToast === 'function') {
      phosToast(!r.checked ? 'No installed LoRA came from CivitAI, so there is nothing to check.'
        : n ? `${n} LoRA${n > 1 ? 's have' : ' has'} a newer version on CivitAI — marked in the list.`
        : `All ${r.checked} CivitAI LoRA${r.checked > 1 ? 's are' : ' is'} up to date.`
        + ((r.errors || []).length ? ` ${r.errors.length} could not be checked.` : ''), { duration: 6000 });
    }
    renderLorasList();
  } catch (e) {
    if (typeof phosToast === 'function') phosToast('CivitAI could not be reached.', { kind: 'danger' });
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Check for updates'; }
  }
}
async function updateLora(path) {
  const it = LORA_UPDATES[path];
  if (!it) return;
  if (typeof phosToast === 'function') phosToast(`Downloading ${it.name} · ${it.latest_version_name || 'new version'}…`, { duration: 5000 });
  const fd = new URLSearchParams();
  fd.set('download_url', it.download_url);
  fd.set('meta', JSON.stringify(it.meta || {}));
  try {
    const r = await (await fetch('/civitai/download', { method: 'POST', body: fd })).json();
    if (r.ok) {
      delete LORA_UPDATES[path];
      try { localStorage.setItem('phos_lora_updates', JSON.stringify({ at: Date.now(), items: Object.values(LORA_UPDATES) })); } catch (e) {}
      if (typeof phosToast === 'function') phosToast(`${it.name} updated. The old file is still installed — delete it from the list when you are done comparing.`, { duration: 8000 });
      await refreshLoras();
    } else if (typeof phosToast === 'function') phosToast(r.error || 'The download failed.', { kind: 'danger', duration: 7000 });
  } catch (e) {
    if (typeof phosToast === 'function') phosToast('The download failed.', { kind: 'danger' });
  }
}

// ---- Guides -----------------------------------------------------------------
// One paragraph per LoRA, written by the planner model from the sidecar
// (name, description, trigger words, base model) and saved back into it.
async function writeLoraGuide(path, btn) {
  // Busy state on the button itself: the planner model loads for tens of
  // seconds the first time, and a second click would queue a second write.
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Writing…'; }
  const fd = new URLSearchParams(); fd.set('path', path);
  try {
    const r = await (await fetch('/loras/guide', { method: 'POST', body: fd })).json();
    if (r.ok) { await refreshLoras(); }
    else if (typeof phosToast === 'function') phosToast(r.error || 'The guide could not be written.', { kind: 'danger', duration: 7000 });
  } catch (e) {
    if (typeof phosToast === 'function') phosToast('The guide could not be written.', { kind: 'danger' });
  } finally {
    if (btn && btn.isConnected) { btn.disabled = false; btn.textContent = btn.dataset.label || 'Write a guide'; }
  }
}
// Guides are refused while a render runs (the planner is a 12B model); say
// so on the button instead of after the click.
function _guideBusyAttrs() {
  const busy = !!(globalThis.LAST_STATUS && globalThis.LAST_STATUS.current);
  return busy ? 'disabled title="Wait for the render to finish — the guide is written by the planner model"' : '';
}

async function refreshLoras() {
  // Pull the FULL library (no mode filter) so _knownUserLoras keeps every
  // entry — that lets refreshLoras() also serve as the "deleted on disk"
  // garbage collector below. The visible filter is applied client-side
  // in renderLorasList() via _currentLoraModeFilter(), which keeps the
  // round-trip count to one and avoids re-fetching every time the
  // user flips mode/engine.
  //
  // 2026-05-20: dropped `exclude_characters=1` (Mr Bizarro report:
  // "LoRA for character disappears. Also, normal video loras are not
  // loading again"). The exclude flag was added 2026-05-17 to avoid
  // duplicate entries between the new Characters tab and the regular
  // LoRA picker — but it also hid trained character bundles from T2V
  // and I2V modes, where users legitimately want to stack them on top
  // of a non-character render. The Characters tab still has its own
  // avatar picker for the curated character UX; the regular picker now
  // ALSO shows the trained bundles so users can pick them manually
  // anywhere LoRAs are allowed. Per-row "trained" kind badge already
  // distinguishes them visually in renderLorasList.
  let data;
  try {
    data = await (await fetch('/loras')).json();
  } catch (e) {
    return;
  }
  // One list, BOTH video libraries, each row carrying `lane`. The client
  // filters per engine (renderLorasList) so an engine switch costs no
  // round-trip — the same reason this fetch has never sent a mode filter.
  _knownUserLoras = data.user || [];
  if (data.loras_dir) _lorasDirs.ltx = data.loras_dir;
  if (data.h3_loras_dir) _lorasDirs.h3 = data.h3_loras_dir;
  _h3LoraSupported = !!data.h3_lora_supported;
  // Update displayed loras dir — whichever library the ACTIVE engine reads.
  {
    const dirEl = document.getElementById('lorasDir');
    const tag = (typeof _currentLoraModeFilter === 'function')
      ? _currentLoraModeFilter() : 'video';
    if (dirEl) {
      dirEl.textContent = (tag === 'video:h3')
        ? (_lorasDirs.h3 || 'the Hailuo H3 pack’s loras/ folder')
        : (_lorasDirs.ltx || 'mlx_models/loras/');
    }
  }
  // Backfill compatible_modes on any active LoRA whose entry we now
  // have full metadata for — addLoraToActive() may have stored a sparse
  // {path, strength, name} from a CivitAI install before /loras was
  // refreshed.
  for (const a of _activeLoras) {
    if (!a.compatible_modes) {
      const ul = _knownUserLoras.find(u => u.path === a.path);
      if (ul) a.compatible_modes = ul.compatible_modes || ['unknown'];
    }
  }
  // If a row was previously active but the file is gone (deleted on
  // disk), drop it from the active set so we don't submit a stale path.
  const knownPaths = new Set(_knownUserLoras.map(l => l.path));
  _activeLoras = _activeLoras.filter(l =>
    knownPaths.has(l.path) || l.path.includes('/'));   // keep HF ids (no dir slash)
  renderLorasList();
  _serializeLoras();
  // The picker's own visibility depends on the H3 gate we just refreshed.
  if (typeof _syncLoraPickerForEngine === 'function') {
    try { _syncLoraPickerForEngine(); } catch (_) {}
  }
  // Refill the Ingredients-mode character dropdown from the same library
  // (kind=train_character). Runs here so a newly-trained character appears
  // the moment /loras is re-fetched after training.
  try { populateIngredientCharLoras(); } catch (_) {}
}

function _loraGenerationCompatible(row, modeTag) {
  return !(modeTag === 'video' && row && row.ltx_compatible === false);
}

function renderLorasList() {
  const wrap = document.getElementById('lorasList');
  const empty = document.getElementById('lorasEmpty');
  const filterRow = document.getElementById('lorasFilterRow');
  const filterInput = document.getElementById('lorasFilter');
  const banner = document.getElementById('loraModeBanner');
  if (!wrap) return;

  // Resolve current (mode + engine) → compat tag. Drives both the
  // library filter AND the per-active-chip "wrong family" warning.
  const modeTag = _currentLoraModeFilter();

  // Combine: user-installed LoRAs (from /loras) plus any active LoRAs
  // that aren't user-installed (HF repo paths, e.g. from the HDR toggle).
  const allRows = [];
  const seen = new Set();
  for (const ul of _knownUserLoras) {
    const active = _activeLoras.find(a => a.path === ul.path);
    seen.add(ul.path);
    allRows.push({
      path: ul.path,
      name: ul.name,
      trigger_words: ul.trigger_words || [],
      recommended_strength: ul.recommended_strength || 1.0,
      filename: ul.filename,
      civitai_url: ul.civitai_url,
      guide: ul.guide || '',
      compatible_modes: ul.compatible_modes || ['unknown'],
      // Which engine DIRECTORY this file came from. Checked before
      // compatible_modes in the filter below, because 'unknown' is
      // deliberately permissive and must not be permissive ACROSS engines.
      lane: ul.lane || 'ltx',
      // H3 lane only: 'bare' / 'comfyui' load, 'diffusers' / 'kohya' don't.
      layout: ul.layout || null,
      layout_ok: (ul.layout_ok === undefined) ? true : !!ul.layout_ok,
      layout_reason: ul.layout_reason || '',
      ltx_compatible: ul.ltx_compatible,
      ltx_compat_reason: ul.ltx_compat_reason || '',
      active: !!active,
      strength: active ? active.strength : (ul.recommended_strength || 1.0),
      // 'user' = downloaded/installed (CivitAI or manual);
      // 'trained' = produced by Phosphene's in-app Train Character pipeline
      // ('kind' on the sidecar is 'train_character'). The picker badges
      // these separately so the user finds their own characters fast.
      kind: ul.kind === 'train_character' ? 'trained' : 'user',
    });
  }
  for (const a of _activeLoras) {
    if (seen.has(a.path)) continue;
    allRows.push({
      path: a.path,
      name: a.name || a.path,
      trigger_words: a.trigger_words || [],
      recommended_strength: 1.0,
      filename: null,
      civitai_url: null,
      compatible_modes: a.compatible_modes || ['unknown'],
      lane: a.lane || 'ltx',
      layout: null,
      layout_ok: true,
      layout_reason: '',
      ltx_compatible: null,
      ltx_compat_reason: '',
      active: true,
      strength: a.strength,
      kind: 'remote',
    });
  }

  // Empty state — collapse the filter box too.
  if (allRows.length === 0) {
    wrap.innerHTML = '';
    if (empty) empty.style.display = '';
    if (filterRow) filterRow.style.display = 'none';
    if (banner) banner.textContent = '';
    return;
  }
  if (empty) empty.style.display = 'none';

  // Apply mode-aware filter FIRST. Hard-filter EVERY row (active OR not)
  // whose compatible_modes don't intersect the active engine. This is the
  // tight-integration version the user explicitly asked for — the prior
  // behavior of "keep mismatched active rows visible with ⚠" was
  // confusing ("now I see the LoRA, but there is no filtering by selected
  // model"). Mismatched-active rows aren't deleted from `_activeLoras` —
  // they're just hidden from this list while the engine is set this way,
  // so flipping back to a compatible engine restores them. The "+ N from
  // other modes" pill at the bottom of the banner is the escape hatch
  // for the rare user who wants to manage a hidden chip without
  // switching engines first.
  //
  // "unknown" compatible_modes still pass every filter — sidecar-less
  // LoRAs need to be reachable, the picker indicator is the user's cue
  // that the family wasn't auto-detected.
  let rows = allRows;
  let hiddenCount = 0;
  const showOtherModes = !!window._loraShowOtherModes;
  if (modeTag) {
    // Heuristic: trained character bundles (e.g. bizarrotrn_v2.safetensors,
    // .audio.safetensors, .voice.wav) are video-only conditioning even
    // when their sidecar tagged them ['unknown']. Without this rule, the
    // 'unknown' branch below would surface them in image-mode pickers —
    // exactly the LTX-LoRA-leakage Mr Bizarro complained about:
    // "they show the LTX LoRA's. You need to separate both galleries."
    const _looksVideoOnly = (r) => {
      if (r.kind === 'trained') return true;
      const path = (r.path || r.filename || r.name || '').toLowerCase();
      if (/_v2\.safetensors$/.test(path)) return true;       // character convention
      if (/\.audio\.safetensors$/.test(path)) return true;   // character audio companion
      if (/\.voice\./.test(path)) return true;               // character voice clip
      if (/\.style\.safetensors$/.test(path)) return true;   // style LoRA convention
      return false;
    };
    const _matches = (r) => {
      const tags = r.compatible_modes || ['unknown'];
      const isImageMode = (modeTag === 'image' || modeTag.startsWith('image:'));
      if (isImageMode) {
        // Image modes only accept LoRAs whose sidecar declared an
        // image:* family. The 'unknown' branch is still permissive
        // (sidecar-less drops should still surface) but EXCLUDES
        // anything that looks like a character/style video bundle.
        if (tags.some(t => typeof t === 'string' && t.startsWith('image:'))) return true;
        if (tags.includes('unknown') && !_looksVideoOnly(r)) return true;
        return false;
      }
      // ---- Video lanes: LTX and Hailuo H3, by DIRECTORY -----------------
      // 2026-08-09: the video side stopped being one family the moment H3
      // learned to take LoRAs, and the two are not interchangeable in
      // either direction — an LTX adapter handed to H3's loader matches
      // zero modules (silent no-op), an H3 one handed to LTX fails inside
      // the fuser. The 'unknown' escape hatch above is what makes this a
      // lane check rather than a tag check: a sidecar-less .safetensors
      // dropped into mlx_models/loras/ classifies as 'unknown' and would
      // otherwise surface in the H3 picker, which is the exact
      // cross-gallery leakage the image split was built to stop.
      if (modeTag === 'video:h3') return r.lane === 'h3';
      if (r.lane === 'h3') return false;
      return tags.includes(modeTag) || tags.includes('unknown');
    };
    rows = allRows.filter(r => {
      // A model-family mismatch may be revealed with "Show other modes";
      // an adapter proven unable to attach to the active LTX generation may
      // not. The helper would refuse it too, but the library must not offer a
      // button that can only lead to an error.
      if (!_loraGenerationCompatible(r, modeTag)) {
        return false;
      }
      const m = _matches(r);
      if (!m && !showOtherModes) hiddenCount++;
      return m || showOtherModes;
    });
    // 2026-05-20: removed the auto-fallback that surfaced LTX LoRAs in
    // image-mode pickers when no image-family LoRAs were installed.
    // Mr Bizarro: "they show the LTX LoRA's. You need to separate both
    // galleries because one is for LoRA's and one is for, and you
    // keep getting confused between them." Strict per-mode separation
    // wins — the empty state below tells the user where to install
    // image LoRAs from instead.
  }
  // Surface the filter input only when 5+ LoRAs (post-mode-filter)
  // remain; below that it's just visual noise.
  if (filterRow) filterRow.style.display = (rows.length >= 5) ? '' : 'none';

  // Then text filter (case-insensitive substring on name + trigger words).
  const q = (filterInput && filterInput.value || '').trim().toLowerCase();
  if (q) {
    rows = rows.filter(r => {
      if (r.name && r.name.toLowerCase().includes(q)) return true;
      for (const t of (r.trigger_words || [])) {
        if (String(t).toLowerCase().includes(q)) return true;
      }
      return false;
    });
  }
  // Sort: active rows first (so the user's selection floats to the top),
  // then alphabetical by name. Stable enough for a UI list.
  rows.sort((a, b) => {
    if (a.active !== b.active) return a.active ? -1 : 1;
    return (a.name || '').localeCompare(b.name || '');
  });

  // Mode banner — explains the active filter so users grok "why don't
  // I see all my LoRAs". When mismatched LoRAs exist (active OR not),
  // exposes a "Show N from other modes" toggle so the user has an
  // escape hatch to manage them without switching engines first.
  if (banner) {
    let bannerHtml = `Library filter: <strong>${escapeHtml(_loraFilterLabel(modeTag))}</strong>`;
    if (hiddenCount > 0) {
      bannerHtml += ` · <a href="#" style="color:var(--muted)" onclick="event.preventDefault(); window._loraShowOtherModes = true; renderLorasList()">Show ${hiddenCount} from other modes</a>`;
    } else if (showOtherModes) {
      bannerHtml += ` · <a href="#" style="color:var(--muted)" onclick="event.preventDefault(); window._loraShowOtherModes = false; renderLorasList()">Hide other modes</a>`;
    }
    banner.innerHTML = bannerHtml;
  }

  // Update header summary.
  const summary = document.getElementById('lorasSummaryCount');
  if (summary) {
    const total = allRows.length;
    const active = allRows.filter(r => r.active).length;
    summary.textContent = `${total} installed · ${active} active${q ? ` · ${rows.length} match` : ''}`;
  }

  if (rows.length === 0) {
    // Helpful empty state. The library is structurally split: this
    // mode shows zero matches because the user has no LoRAs tagged
    // for it. Surface the install path directly so users don't think
    // the picker is broken. Three sub-cases:
    //   1. Search active (q non-empty) — text filter cleared everything.
    //   2. Image mode + no image LoRAs in library.
    //   3. Video mode + no video LoRAs in library.
    if (q) {
      wrap.innerHTML = `<div class="hint" style="padding:8px 0;">No LoRAs match "${escapeHtml(q)}".</div>`;
    } else if (typeof modeTag === 'string' && modeTag.startsWith('image')) {
      wrap.innerHTML = `
        <div class="hint" style="padding:14px 8px;text-align:center;line-height:1.6;">
          <div style="margin-bottom:4px;color:var(--fg);"><strong>No image LoRAs in your library.</strong></div>
          <div>The Library tab shows LoRAs scoped to the active engine — your installed LoRAs are LTX-Video LoRAs.</div>
          <div style="margin-top:6px;">Install a Qwen-Image or Flux LoRA via the <strong>Browse CivitAI</strong> button above (filter base model to Qwen / Flux).</div>
        </div>`;
    } else if (modeTag === 'video') {
      wrap.innerHTML = `
        <div class="hint" style="padding:14px 8px;text-align:center;line-height:1.6;">
          <div style="margin-bottom:4px;color:var(--fg);"><strong>No video LoRAs in your library.</strong></div>
          <div>Install an LTX-Video LoRA via the <strong>Browse CivitAI</strong> button above (filter base model to LTX 2.3), or train one in the <strong>Train Character</strong> tab.</div>
        </div>`;
    } else if (modeTag === 'video:h3') {
      // Its own empty state, not the LTX one: the two libraries are separate
      // directories. CivitAI is convenient, but a custom adapter someone
      // already has must be just as discoverable: it only needs the runner's
      // lora_A/lora_B layout in the H3 library. The layout gate below keeps a
      // raw Kohya file visible with its conversion reason instead of letting a
      // bad adapter reach a long render.
      wrap.innerHTML = `
        <div class="hint" style="padding:14px 8px;text-align:center;line-height:1.6;">
          <div style="margin-bottom:4px;color:var(--fg);"><strong>No Hailuo H3 LoRAs in your library.</strong></div>
          <div>H3 has its own library — your LTX LoRAs can't load here, and H3's can't load on LTX.</div>
          <div style="margin-top:6px;">Already have one? <strong>Import H3 LoRA</strong> above takes a <code>.safetensors</code> file and checks it against your installed H3 transformer before it lands in the library.</div>
          <div style="margin-top:6px;">Drop a converted H3 <code>.safetensors</code> with <code>lora_A</code> / <code>lora_B</code> tensors into <code>${escapeHtml(_lorasDirs.h3 || 'the H3 pack’s loras/ folder')}</code>, then press Rescan — same result, no size limit.</div>
          <div style="margin-top:6px;">Or install one via <strong>Browse CivitAI</strong> above using the <strong>Hailuo H3</strong> CivitAI filter.</div>
        </div>`;
    } else {
      wrap.innerHTML = `<div class="hint" style="padding:8px 0;">No LoRAs available.</div>`;
    }
    return;
  }
  wrap.innerHTML = rows.map(r => loraRowHtml(r, modeTag)).join('');
}

// Build a single compact LoRA row. Inactive rows are ~36px tall (just
// name + meta + corner actions). Active rows expand inline with the
// strength slider and trigger chips. Click anywhere on the main row to
// toggle activation.
//
// modeTag is the active engine compat tag (e.g. "image:qwen") — used to
// render the per-chip warning when an ACTIVE LoRA doesn't fit the
// current engine, and a `?` indicator for unclassified LoRAs.
function loraRowHtml(r, modeTag) {
  const pathHtml = escapeHtml(r.path);
  const pathAttr = JSON.stringify(r.path).replace(/"/g, '&quot;');
  const nameHtml = escapeHtml(r.name);
  const nameAttr = JSON.stringify(r.name).replace(/"/g, '&quot;');
  // Per-row family badges. `?` for unknown family (unclassified
  // base_model), `⚠` for an ACTIVE LoRA whose compat tags don't match
  // the current engine — clear visible signal that "this chip stays
  // selected but won't actually fire on the engine you've chosen".
  const tags = r.compatible_modes || ['unknown'];
  const isUnknown = tags.length === 1 && tags[0] === 'unknown';
  const familyMismatch = !!(modeTag && r.active
    && !tags.includes(modeTag) && !tags.includes('unknown'));
  const familyBadges = [];
  if (isUnknown) {
    familyBadges.push(`<span class="badge" title="Unknown LoRA family — sidecar didn't list a base_model. May or may not fire on this engine.">?</span>`);
  }
  if (familyMismatch) {
    familyBadges.push(`<span class="badge" style="background:rgba(232,179,65,0.15);color:var(--warn,#e8b341)" title="This LoRA was trained for a different family (${tags.join(', ')}) — it will be passed to mflux but probably won't influence the output. Switch the Engine dropdown to a matching family, or remove this LoRA."><svg class="ph" aria-hidden="true"><use href="#ph-warning-fill"/></svg></span>`);
  }
  // H3 lane: a file whose KEY LAYOUT the runner can't read (diffusers /
  // kohya). Shown rather than hidden, with the reason on the badge, because
  // "my LoRA vanished" is a worse bug report than "my LoRA says why it can't
  // run" — and the reason is the actionable half (it names the conversion).
  if (r.layout_ok === false) {
    familyBadges.push(`<span class="badge" style="background:rgba(248,81,73,0.15);color:var(--danger,#f85149)" title="${escapeHtml(r.layout_reason || 'This LoRA is in a key layout the H3 runner cannot read.')}"><svg class="ph" aria-hidden="true"><use href="#ph-warning-fill"/></svg></span>`);
  }
  // Trigger summary line under the name (when not expanded). Truncated.
  const trigs = r.trigger_words || [];
  const trigSummary = trigs.length
    ? trigs.slice(0, 4).join(' · ') + (trigs.length > 4 ? ` +${trigs.length - 4}` : '')
    : 'no trigger word';
  // Corner actions: rename → download → external link → delete.
  // Order is safe-first, destructive-last so users don't fat-finger the X.
  // Rename + download are local-only (need an on-disk file), so they're
  // gated to `user` / `trained` rows; remote (HF) rows skip them.
  const corner = [];
  if (r.kind === 'user' || r.kind === 'trained') {
    corner.push(`<button class="lora-icon-btn" type="button" title="Rename (display name only — filename stays the same)"
                         onclick="event.stopPropagation(); renameLora(${pathAttr}, ${nameAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-pencil-simple"/></svg></button>`);
    corner.push(`<button class="lora-icon-btn" type="button" title="Download the .safetensors file"
                         onclick="event.stopPropagation(); downloadLora(${pathAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-download-simple"/></svg></button>`);
  }
  if (LORA_UPDATES[r.path]) {
    const up = LORA_UPDATES[r.path];
    corner.push(`<button class="lora-icon-btn lora-update-btn" type="button" title="Newer version on CivitAI: ${escapeHtml(up.latest_version_name || String(up.latest_version_id))} — download it"
                         onclick="event.stopPropagation(); updateLora(${pathAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-download-simple"/></svg></button>`);
  }
  if (r.civitai_url) {
    corner.push(`<a class="lora-icon-btn" href="${escapeHtml(r.civitai_url)}" target="_blank" rel="noopener" title="Open on CivitAI" onclick="event.stopPropagation()"><svg class="ph" aria-hidden="true"><use href="#ph-arrow-square-out"/></svg></a>`);
  }
  if (r.kind === 'user' || r.kind === 'trained') {
    corner.push(`<button class="lora-icon-btn danger" type="button" title="Delete from disk"
                         onclick="event.stopPropagation(); deleteLora(${pathAttr}, ${nameAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>`);
  } else {
    corner.push(`<button class="lora-icon-btn" type="button" title="Remove from active set"
                         onclick="event.stopPropagation(); removeLoraFromActive(${pathAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>`);
  }
  // Trigger chips for the expanded section. Same click-to-append behavior
  // as before — chips prepend the trigger to the prompt textarea.
  const chipsHtml = trigs.length
    ? trigs.slice(0, 12).map(w => {
        const wAttr = JSON.stringify(w).replace(/"/g, '&quot;');
        return `<span class="trigger-chip" title="Click to add to prompt"
                       onclick="event.stopPropagation(); appendTriggerToPrompt(${wAttr})">${escapeHtml(w)}</span>`;
      }).join('')
    : `<span class="trigger-chip empty">style-only LoRA — no trigger word needed</span>`;

  return `
    <div class="lora-row ${r.active ? 'active' : ''}" data-path="${pathHtml}">
      <div class="lora-row-main"
           onclick="toggleLora(${pathAttr}, ${!r.active}, ${r.recommended_strength}, ${nameAttr})">
        <div class="lora-toggle-dot"></div>
        <div class="lora-text">
          <div class="lora-name" title="${pathHtml}">
            ${nameHtml}${r.kind === 'remote' ? '<span class="badge">HF</span>' : ''}${r.kind === 'trained' ? '<span class="badge badge-trained" title="Trained in Phosphene’s Train Character workflow">Trained</span>' : ''}${familyBadges.join('')}${LORA_UPDATES[r.path] ? `<button type="button" class="badge badge-update lora-update-badge" title="A newer version is on CivitAI — click to download it" onclick="event.stopPropagation(); updateLora(${pathAttr})">Update</button>` : ''}
          </div>
          <div class="lora-name-meta" title="${escapeHtml(trigs.join(', '))}">${escapeHtml(trigSummary)}</div>
        </div>
        <div class="lora-row-actions">${corner.join('')}</div>
      </div>
      <div class="lora-row-extra">
        <div class="lora-strength-row">
          <label>strength</label>
          <input type="range" min="-2" max="2" step="0.05" value="${r.strength}"
                 onclick="event.stopPropagation()"
                 oninput="this.nextElementSibling.value = this.value; setLoraStrength(${pathAttr}, this.value)">
          <input type="number" min="-2" max="2" step="0.05" value="${r.strength}"
                 onclick="event.stopPropagation()"
                 oninput="this.previousElementSibling.value = this.value; setLoraStrength(${pathAttr}, this.value)">
        </div>
        <div class="trigger-chips">${chipsHtml}</div>
        ${(r.kind === 'user' || r.kind === 'trained') ? `
        <div class="lora-guide-actions">
          ${r.guide ? '' : `<button type="button" class="ghost-btn small" ${_guideBusyAttrs()} onclick="event.stopPropagation(); writeLoraGuide(${pathAttr}, this)">Write a guide</button>
          <span class="muted small">What it does, how to prompt it, a strength to start from — written by the planner model from the LoRA's own notes.</span>`}
        </div>
        ${r.guide ? `<div class="lora-guide">${escapeHtml(r.guide)}
          <div class="lora-guide-actions"><button type="button" class="ghost-btn small" ${_guideBusyAttrs()} onclick="event.stopPropagation(); writeLoraGuide(${pathAttr}, this)">Rewrite</button></div></div>` : ''}` : ''}
      </div>
    </div>`;
}

function toggleLora(path, on, recommended, name) {
  if (on) {
    addLoraToActive({ path, strength: recommended, name });
    // Auto-insert the LoRA's first trigger word into the prompt so the
    // user doesn't have to remember + retype it. The biggest source of
    // "I activated the LoRA but it didn't fire" reports is the trigger
    // missing from the prompt entirely. appendTriggerToPrompt is
    // idempotent — duplicate clicks are no-ops. Skips silently if the
    // LoRA has no trigger_words (style-only LoRAs).
    try {
      const meta = _knownUserLoras.find(u => u.path === path);
      const triggers = (meta && meta.trigger_words) || [];
      if (triggers.length) {
        appendTriggerToPrompt(triggers[0]);
      }
    } catch (e) { /* picker race during boot — ignore */ }
  } else {
    removeLoraFromActive(path);
  }
}

// Append a LoRA's trigger word to the prompt textarea. Most LTX LoRAs
// only fully activate when their trigger word is somewhere in the prompt,
// and asking users to remember + type a string like "DISPSTYLE" exactly
// is friction. Click the chip → it goes in. Idempotent: if the word is
// already present (case-insensitive substring), do nothing so users can
// click freely without piling duplicates.
function appendTriggerToPrompt(word) {
  const ta = document.getElementById('prompt');
  if (!ta) return;
  const cur = ta.value || '';
  if (cur.toLowerCase().includes(String(word).toLowerCase())) {
    // Brief visual ping so the click feels acknowledged even though we
    // didn't change anything — otherwise users repeat-click thinking
    // it's broken.
    ta.classList.add('flash-ok');
    setTimeout(() => ta.classList.remove('flash-ok'), 250);
    return;
  }
  // If the user has typed nothing, drop the trigger in alone. Otherwise
  // prepend to the existing prompt: many LoRA authors put the trigger
  // FIRST in their examples, and quality often degrades when the trigger
  // is buried at the end past 20+ tokens of unrelated context.
  if (cur.trim() === '') {
    ta.value = String(word);
  } else {
    ta.value = String(word) + ', ' + cur;
  }
  ta.focus();
  ta.dispatchEvent(new Event('input', { bubbles: true }));
}

// Inline rename. Click the pencil icon → the LoRA's name turns into an
// <input> prefilled with the current name. Enter / blur commits, Escape
// cancels. Only the sidecar's display-name changes — the .safetensors
// filename is left alone so anything that references the on-disk path
// (saved job sidecars, persisted picker selection, etc.) keeps working.
async function renameLora(path, currentName) {
  const row = document.querySelector(`.lora-row[data-path="${CSS.escape(path)}"]`);
  const nameEl = row && row.querySelector('.lora-name');
  if (!nameEl || nameEl.dataset.editing === '1') return;
  nameEl.dataset.editing = '1';
  // Save the rendered HTML (name + badges) so Escape / failure can restore
  // it byte-for-byte. The badge cluster (HF, Trained, ? unknown-family etc.)
  // is part of nameEl — preserving the full innerHTML is simpler than
  // re-rendering it manually.
  const original = nameEl.innerHTML;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = currentName || '';
  input.maxLength = 120;
  input.className = 'lora-name-edit';
  input.onclick = (e) => e.stopPropagation();
  nameEl.replaceChildren(input);
  input.focus();
  input.select();

  let done = false;
  const finish = () => { delete nameEl.dataset.editing; done = true; };
  const cancel = () => {
    if (done) return;
    nameEl.innerHTML = original;
    finish();
  };
  const commit = async () => {
    if (done) return;
    const newName = input.value.trim();
    if (!newName || newName === currentName) { cancel(); return; }
    finish();  // mark done BEFORE the async hop so blur can't re-fire commit
    const fd = new FormData();
    fd.set('path', path);
    fd.set('name', newName);
    try {
      const r = await fetch('/loras/rename', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams(fd),
      });
      const data = await r.json();
      if (!r.ok || !data.ok) {
        nameEl.innerHTML = original;
        alert('Rename failed: ' + (data.error || `HTTP ${r.status}`));
        return;
      }
      refreshLoras();
    } catch (e) {
      nameEl.innerHTML = original;
      alert('Rename failed: ' + (e.message || e));
    }
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
    else if (e.key === 'Escape') { e.preventDefault(); cancel(); input.blur(); }
  });
  input.addEventListener('blur', commit);
}

// Trigger the browser's native download dialog for a LoRA's .safetensors.
// The /loras/download endpoint sets Content-Disposition: attachment so the
// browser pops Save instead of navigating to the bytes. We use a temp <a>
// rather than window.location.href to avoid creating a navigation history
// entry; the `download` attr is a belt-and-braces fallback for the
// disposition header.
function downloadLora(path) {
  const a = document.createElement('a');
  a.href = '/loras/download?path=' + encodeURIComponent(path);
  // The BASENAME, not the empty string. `download=''` delegates the name to
  // the server's Content-Disposition, which this endpoint does send — but the
  // empty form silently falls back to the URL-derived name ("download") on any
  // path where the header is missing or stripped by an extension. We know the
  // name here; saying it costs nothing and removes the failure mode.
  a.download = String(path || '').split('/').pop() || '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function deleteLora(path, name) {
  if (!confirm(`Delete the LoRA file for "${name}" from disk? This is permanent.`)) {
    return;
  }
  const fd = new FormData();
  fd.set('path', path);
  try {
    const r = await fetch('/loras/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      alert('Delete failed: ' + (data.error || `HTTP ${r.status}`));
      return;
    }
    removeLoraFromActive(path);
    refreshLoras();
  } catch (e) {
    alert('Delete failed: ' + (e.message || e));
  }
}

// ====== Manual-tab Characters picker ======
//
// Single-select character chip strip. Sits above the LoRA picker; visible
// only in T2V mode. Selecting a character writes its id to the hidden
// #characterIdInput which rides the form on submit — make_job() expands
// character_id into face+audio LoRA additions (see the character_id
// branch in the video make_job path).
//
// The Characters TAB (separate top-level workflow) is left untouched —
// its compose state still works exactly as before, and the bundles it
// reads come from the SAME /characters endpoint this picker calls.
globalThis._manualCharacters = [];     // [{id, name, trigger, has_voice, sample_image_url, ...}]
globalThis._selectedCharacterId = '';  // '' means none selected

function _updateCharsPickerVisibility(mode) {
  // 2026-05-17 (Codex C+ pass 3): the chip strip lives in 'character'
  // mode only — its own pill in the mode bar. T2V no longer hosts it
  // (clean separation: T2V is for non-character text-to-video; Character
  // is for the trained-LoRA face/voice cascade).
  // Hide on every other mode — Image / FFLF / Extend / Train / Studio
  // hide #genForm or have a different mental model anyway, but the
  // helper stays in sync defensively.
  const show = (mode === 'character');
  const slot = document.getElementById('manualCharactersPickerSlot');
  const divider = document.getElementById('charactersPickerDivider');
  if (slot) slot.classList.toggle('show', show);
  if (divider) divider.classList.toggle('show', show);
  // When hidden, don't carry a stale selection on submit — clear the
  // hidden input. The user can re-select when they next visit T2V.
  if (!show) {
    _selectedCharacterId = '';
    const inp = document.getElementById('characterIdInput');
    if (inp) inp.value = '';
    // Re-render the avatar strip so the .active ring drops in lockstep with
    // the cleared selection. Without this the old avatar stayed highlighted;
    // returning to Character mode then showed a character that LOOKED selected
    // while the hidden field was empty -> Generate shipped character_id="" ->
    // plain T2V with no LoRA (the desync bug, present since fd59a2c).
    if (typeof _renderManualCharactersList === 'function') {
      try { _renderManualCharactersList(); } catch (_) {}
    }
    _renderCharsAppliedNote();
    // Restore the default quality strip when leaving T2V — the
    // character-only strip should never be visible in I2V / FFLF /
    // Extend (no character context).
    if (typeof _applyCharacterQualityStripVisibility === 'function') {
      try { _applyCharacterQualityStripVisibility(); } catch (_) {}
    }
  }
}

async function refreshManualCharacters() {
  // /characters returns { characters: [...] } — same payload the Characters
  // tab consumes, so the source of truth stays single.
  let data;
  try {
    data = await (await fetch('/characters')).json();
  } catch (e) {
    return;
  }
  _manualCharacters = (data && Array.isArray(data.characters)) ? data.characters : [];
  // If the previously-selected character is no longer on disk (renamed,
  // deleted), drop the selection silently rather than carrying a stale
  // id that would 404 the lookup at submit.
  if (_selectedCharacterId &&
      !_manualCharacters.some(c => c.id === _selectedCharacterId)) {
    _selectedCharacterId = '';
    const inp = document.getElementById('characterIdInput');
    if (inp) inp.value = '';
  }
  _renderManualCharactersList();
}

function _renderManualCharactersList() {
  const wrap = document.getElementById('charsList');
  const empty = document.getElementById('charsEmpty');
  const meta = document.getElementById('charsSummaryMeta');
  if (!wrap) return;
  if (!_manualCharacters.length) {
    wrap.innerHTML = '';
    if (empty) empty.hidden = false;
    if (meta) meta.textContent = 'no bundles on disk';
    _renderCharsAppliedNote();
    return;
  }
  if (empty) empty.hidden = true;
  const chips = [];
  // 2026-05-18 round 3: no more "None" chip. Mr Bizarro's point — no
  // character is just plain video gen, which is what the Text mode
  // already covers; an explicit None chip was redundant chrome.
  // Deselect is now: click the active avatar again (toggles off).
  //
  // Avatars are 44 px (up from 32) so the faces are readable at a
  // glance. The row wraps to multiple lines via flex-wrap on the
  // container, so 20+ characters stack cleanly without a horizontal-
  // scroll trap.
  for (const c of _manualCharacters) {
    const active = (c.id === _selectedCharacterId);
    const name = c.name || c.trigger || c.id;
    const trigger = c.trigger || c.id || '';
    const initial = (name || '?').charAt(0).toUpperCase();
    const avatar = c.sample_image_url
      ? `<img class="chars-avatar-img" src="${escapeHtml(c.sample_image_url)}" alt="">`
      : `<span class="chars-avatar-ph">${escapeHtml(initial)}</span>`;
    const hasVoice = !!c.has_voice;
    // Voice indicator — a small music-note badge for characters that
    // have a voice LoRA. Silent characters get no badge (absence reads
    // as 'silent' more cleanly than a gray badge fighting for
    // attention). Tooltip on the avatar already mentions silent state.
    const voiceBadge = hasVoice
      ? `<span class="chars-avatar-voice" title="Has voice LoRA + reference clip"><svg class="ph"><use href="#ph-music-notes"/></svg></span>`
      : '';
    const idAttr = JSON.stringify(c.id).replace(/"/g, '&quot;');
    const tt = `${escapeHtml(name)} · ${escapeHtml(trigger)}${hasVoice ? ' · has voice' : ' · silent'}${active ? ' · click to deselect' : ''}`;
    chips.push(`
      <button type="button" class="chars-avatar-chip ${active ? 'active' : ''}"
              onclick="selectManualCharacter(${idAttr})"
              title="${tt}">
        ${avatar}
        <span class="chars-avatar-name">${escapeHtml(name)}</span>
        ${voiceBadge}
      </button>
    `);
  }
  wrap.innerHTML = chips.join('');
  if (meta) {
    if (_selectedCharacterId) {
      const cur = _manualCharacters.find(c => c.id === _selectedCharacterId);
      meta.textContent = cur ? `selected: ${cur.name || cur.trigger}` : '1 selected';
    } else {
      meta.textContent = `${_manualCharacters.length} available`;
    }
  }
  // Strength row stays in the DOM as a hidden carrier; the actual
  // slider lives inline inside charsAppliedNote (rendered by
  // _renderCharsAppliedNote). This keeps the picker compact — no
  // separate full-width strength row pushing other controls down.
  const strengthRow = document.getElementById('charsStrengthRow');
  if (strengthRow) strengthRow.hidden = !_selectedCharacterId;
  _renderCharsAppliedNote();
}

function _renderCharsAppliedNote() {
  // Compact under-strip meta line (2026-05-18 redesign). Shows:
  //   <name> · <trigger> · [strength slider 0-2] · [silent? indicator]
  // The inline strength slider replaces the old full-width chars-
  // strength-row; everything fits on one tight horizontal line under
  // the avatars. Hidden when no character is selected.
  //
  // Side effect: keeps the No-voice toggle visibility in sync with the
  // selection. The pill only makes sense for characters that HAVE a
  // voice LoRA in the first place; for silent characters there's no
  // voice to skip.
  const note = document.getElementById('charsAppliedNote');
  const voicePill = document.getElementById('noVoicePill');
  // A hidden No-voice pill must never keep a live checkbox: leaving the
  // check behind let a character-mode "No voice" ride into a later PLAIN
  // render's prompt as a no-speech constraint (journey audit, High). The
  // silent-character branch below already cleared it; these two paths —
  // no selection, selection no longer on disk — did not.
  const _clearNoVoice = () => {
    const cb = document.getElementById('noVoice');
    if (cb) cb.checked = false;
  };
  if (!note) return;
  if (!_selectedCharacterId) {
    note.hidden = true;
    note.innerHTML = '';
    if (voicePill) { voicePill.hidden = true; _clearNoVoice(); }
    return;
  }
  const c = _manualCharacters.find(x => x.id === _selectedCharacterId);
  if (!c) {
    note.hidden = true;
    note.innerHTML = '';
    if (voicePill) { voicePill.hidden = true; _clearNoVoice(); }
    return;
  }
  // Show the No-voice pill only when the active character actually has
  // a voice LoRA. Silent characters can't be made more silent. Also
  // reset the checkbox when hiding so a stale check from a previous
  // selection doesn't accidentally suppress voice on a silent character.
  if (voicePill) {
    voicePill.hidden = !c.audio_lora_path;
    if (voicePill.hidden) {
      const cb = document.getElementById('noVoice');
      if (cb) cb.checked = false;
    } else if (typeof refreshNoVoiceAuto === 'function') {
      // Casting a character with a voice is the moment the default becomes
      // meaningful, so evaluate it here as well as on prompt input.
      refreshNoVoiceAuto();
    }
  }
  const name = escapeHtml(c.name || c.trigger || c.id);
  const triggerCode = c.trigger
    ? ` · trigger <code>${escapeHtml(c.trigger)}</code>`
    : '';
  const silentBadge = (!c.audio_lora_path)
    ? ` · <em title="No audio LoRA on disk — face only">silent</em>`
    : '';
  const cur = parseFloat(document.getElementById('characterStrength')?.value || '1.0');
  const voi = parseFloat(document.getElementById('characterVoiceStrength')?.value || '1.0');
  const hasVoice = !!c.audio_lora_path;
  // ONE slider stays the default surface; the split lives one disclosure down.
  // A character is two files, but the panel is not asking anyone to think about
  // two numbers before they have a reason to. `split` is a text affordance, not
  // a chevron — it sits inside a one-line control — and its label carries the
  // pair once they differ, so a non-default voice is visible without opening it.
  const splitLabel = (Math.abs(voi - cur) > 0.001)
    ? `split · ${cur.toFixed(1)} / ${voi.toFixed(1)}`
    : 'split';
  note.innerHTML = `
    <span><strong>${name}</strong>${triggerCode}${silentBadge}</span>
    <span class="chars-inline-strength">
      <span>strength</span>
      <input type="range" min="0" max="2" step="0.05" value="${cur.toFixed(2)}"
             oninput="setCharStrength('face', this.value)">
      <output id="charFaceOut">${cur.toFixed(1)}</output>
      ${hasVoice ? `<button type="button" class="chars-split-toggle" id="charSplitBtn"
              aria-expanded="false" aria-controls="charSplitRow"
              title="Set the face and the voice separately"
              onclick="toggleCharSplit()">${escapeHtml(splitLabel)}</button>` : ''}
    </span>
    ${hasVoice ? `
    <div class="chars-split" id="charSplitRow" hidden>
      <div class="chars-split-line">
        <span>face</span>
        <input type="range" min="0" max="2" step="0.05" value="${cur.toFixed(2)}"
               oninput="setCharStrength('face', this.value)">
        <output id="charFaceSplitOut">${cur.toFixed(1)}</output>
      </div>
      <div class="chars-split-line">
        <span>voice</span>
        <input type="range" min="0" max="2" step="0.05" value="${voi.toFixed(2)}"
               oninput="setCharStrength('voice', this.value)">
        <output id="charVoiceOut">${voi.toFixed(1)}</output>
        <button type="button" class="help-dot" id="charVoiceHelpBtn" aria-expanded="false"
                aria-controls="charVoiceHelpNote" title="Why the voice has its own number"
                onclick="toggleCharVoiceHelp()">?</button>
      </div>
      <div class="h3-winhelp" id="charVoiceHelpNote" hidden></div>
    </div>` : ''}
  `;
  note.hidden = false;
}

// The character strip, filled from the server-resolved table. LTX-2.3 gets the
// two graded-default canvases; LTX-2.5 gets those plus both real HQ qualities.
// Runs again whenever pack presence changes because every q8 row is an install
// CTA when its own weights are absent.
function renderCharacterStrip() {
  const cfg = ((BOOT.ltx || {}).character) || {};
  const group = document.getElementById('qualityGroupCharacter');
  if (!group) return;
  const rows = Array.isArray(cfg.tokens) && cfg.tokens.length
    ? cfg.tokens
    : [cfg.draft, cfg.pro].filter(Boolean);
  if (!rows.length) return;
  const previous = group.querySelector('.char-quality.active')?.dataset.charQuality;
  const keys = rows.map(row => row.key);
  const current = keys.includes(previous) ? previous
    : (keys.includes(cfg.default) ? cfg.default : keys[0]);
  group.style.gridTemplateColumns = `repeat(${rows.length}, 1fr)`;
  group.innerHTML = rows.map(row => {
    const needsInstall = typeof ltxCellNeedsInstall === 'function'
      && ltxCellNeedsInstall(row);
    const foot = needsInstall && typeof ltxCellInstallLabel === 'function'
      ? ltxCellInstallLabel(row) : (row.tier || '');
    const cls = 'q-chip pill-btn pill-quality char-quality'
      + (row.key === current ? ' active' : '')
      + (needsInstall ? ' needs-install' : '');
    return `<button type="button" class="${cls}"
              data-char-quality="${escapeHtml(row.key)}"
              data-quality="${escapeHtml(row.quality || cfg.quality || 'high')}"
              data-width="${Number(row.width || 1024)}"
              data-height="${Number(row.height || 576)}"
              data-pack="${escapeHtml(row.pack || 'q8')}"
              data-pipeline="${escapeHtml(row.pipeline || '')}"
              title="${escapeHtml(row.title || '')}">
        <span class="ql-name">${escapeHtml(row.label || row.key)}</span>
        <span class="q-spec ql-spec sub">${Number(row.width)}×${Number(row.height)}</span>
        <span class="ql-tier">${escapeHtml(foot)}</span>
      </button>`;
  }).join('');
}


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  checkLoraUpdates, updateLora, writeLoraGuide,
  _currentLoraModeFilter, _syncLoraPickerForEngine, importH3Lora, renderH3LoraSlot,
  setH3LoraSlot, _serializeLoras, addLoraToActive, populateIngredientCharLoras,
  onIngredientCharChange, onIngredientCharStrength, refreshLoras, _loraGenerationCompatible,
  renderLorasList, appendTriggerToPrompt, _updateCharsPickerVisibility, refreshManualCharacters,
  _renderManualCharactersList, _renderCharsAppliedNote, renderCharacterStrip,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  deleteLora, downloadLora, removeLoraFromActive, renameLora,
  setLoraStrength, toggleLora,
});
