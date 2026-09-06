// webapp/js/engines.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ============================================================================
// Engine registry — the header switcher and every gate behind it
// ============================================================================
// An engine is not a quality setting: different model, its own venv, its own
// subprocess, its own geometry rules, its own set of modes. The registry
// (BOOT.engines, from the server-side ENGINES table) is the single source for
// all of it, so adding a third engine is one Python entry plus one <symbol> —
// nothing below learns any engine's name.
//
// The four gate states a segment has to carry, unchanged from the two-engine
// version and now generic:
//   not capable   → not rendered at all (a 32 GB Mac never learns H3 exists,
//                   and with nothing left to choose the switcher disappears)
//   capable, not installed → dashed segment + a cost badge; the click opens
//                   the install card. An offer, never a dead button.
//   installed but broken   → same shape, badge says "repair" and the card
//                   never mentions a download (the v3.4.0 report, verbatim)
//   installed, wrong mode  → inert; the reason lands in #engineRowNote
//
// The truth is still server-side: make_job re-runs every one of these against
// the same table, so a stale tab, a Load-Params replay of an old sidecar, or a
// direct curl to /queue/add can never render on an engine this Mac (or this
// mode) can't actually run.
globalThis.ENGINES = (BOOT.engines || []);
globalThis.ENGINE_DEFAULT = (BOOT.default_engine || 'ltx');
globalThis.H3_ENGINE_LS_KEY = 'phos_video_engine';

function engineById(id) {
  return ENGINES.find(e => e.id === id) || null;
}
function defaultEngine() {
  return engineById(ENGINE_DEFAULT) || ENGINES[0]
      || { id: 'ltx', label: 'LTX', builtin: true, strip_label: 'Quality' };
}

// An engine's live capability block. `probe` names the bootstrap key that
// carries it (BOOT.h3 for Hailuo H3), so this function never mentions an
// engine by name. The built-in engine has nothing to probe: it IS the panel.
// An `announced` engine has no weights yet, so it is capable of nothing.
function engineStatus(e) {
  if (!e) return { capable: false, available: false };
  if (e.state === 'announced') return { capable: false, available: false, announced: true };
  if (!e.probe) return { capable: true, available: true };
  return (window._ENGINE_PROBES && window._ENGINE_PROBES[e.probe]) || { capable: false, available: false };
}

// Modes this engine may serve. `excluded_modes` exists because some UI intents
// resolve to a mode an engine nominally supports but must not get: 'character'
// submits mode=t2v but stacks LTX LoRAs, and 'i2v_clean_audio' muxes an
// external track onto LTX video. Mirrors engine_serves_mode() in Python, which
// is the copy that decides.
function engineServesMode(e, mode) {
  if (!e) return false;
  // 'oneshot' is a UI mode that submits t2v (or i2v with an anchor) plus
  // take_seconds; both engines render it — H3 as 15 s parts, LTX as 10 s —
  // so it is asked about as the mode it ships under. Without this the chip
  // would be folded as eng-foreign on H3, the engine the feature was built on.
  if (mode === 'oneshot') mode = 't2v';
  if ((e.excluded_modes || []).indexOf(mode) !== -1) return false;
  if (!e.modes) return true;
  return e.modes.indexOf(mode) !== -1;
}

// Which workflow tab an engine belongs to. The switcher is a Video-form
// control; showing it while the user is in Images or Train Character would be
// offering a choice that changes nothing.
function _currentSurface() {
  const wf = (document.body.dataset.workflow || 'manual').toLowerCase();
  // 'storyboard' MUST be in this map. Without it the `|| 'video'` fallback
  // leaves the engine switcher visible in a tab that has no engine choice to
  // offer — the film decides per shot, not a global toggle.
  return ({ manual: 'video', studio: 'image', audio: 'audio', train: 'train',
            storyboard: 'storyboard' })[wf] || 'video';
}
function engineOnSurface(e) {
  return (e.surfaces || ['video']).indexOf(_currentSurface()) !== -1;
}

// An engine gets a segment when this Mac could plausibly run it one day: the
// built-in always, an announced one always (that IS the news), and an optional
// pack only when the hardware is capable. Not-capable is not rendered — a
// permanently dead engine in the chrome is noise for the ~80% under 64 GB.
function engineRenderable(e) {
  if (!engineOnSurface(e)) return false;
  if (e.builtin || e.state === 'announced') return true;
  return !!engineStatus(e).capable;
}

// ---- Capability probes ------------------------------------------------------
// One entry per engine whose registry row names a `probe`. /status carries a
// fresh block every tick, so an install finishing in the Pinokio sidebar
// unlocks its engine without a panel restart (the same contract the Q8
// download already has with the High pill). A new engine adds a key here the
// day it has a status function; nothing else in this file changes.
window._ENGINE_PROBES = {
  h3: (BOOT.h3 || { capable: false, available: false, tiers: [] }),
};
// H3 keeps its own binding because the H3 tier / Turbo / export code below is
// H3-specific by nature (its own geometry table, its own adapter) and reads it
// on nearly every line. It is the SAME object the probe map holds.
globalThis.H3 = window._ENGINE_PROBES.h3;

// ---- H3 render shape: two axes, one cell -----------------------------------
// The panel speaks (quality, length); the wire speaks a composite `h3_tier`
// key. These four helpers are the whole translation layer, and none of them
// invents a shape — every one of them looks up a cell the server built.

// A tier key of ANY vintage → the cell key it means. The alias map is the
// server's (H3_TIER_ALIASES), shipped in the status block, so the legacy names
// exist in exactly one place: hq_5s → standard_5s, wide_5s → high_5s,
// long_10s → standard_10s, and so on.
function h3ResolveTierKey(key) {
  const k = String(key || '').trim().toLowerCase();
  if (!k) return '';
  return ((H3.aliases || {})[k]) || k;
}
function h3TierByKey(key) {
  const k = h3ResolveTierKey(key);
  return (H3.tiers || []).find(t => t.key === k) || (H3.tiers || [])[0] || null;
}
// The cell at (quality, length), or null. This is what both strips read: a chip
// prints the eta of the cell it WOULD select, which is how the estimate stays
// live as either axis moves.
function h3CellFor(quality, length) {
  return (H3.tiers || []).find(t => t.quality === quality && t.length === length) || null;
}
function h3CurrentQuality() {
  return (document.getElementById('h3_quality') || {}).value
      || H3.default_quality || 'draft';
}
function h3CurrentLength() {
  return (document.getElementById('h3_length') || {}).value
      || H3.default_length || '3s';
}
// The cell the form is currently pointing at. Falls back through the composite
// key so a state restored from an old sidecar (which has no axes) still lands.
function h3CurrentCell() {
  return h3CellFor(h3CurrentQuality(), h3CurrentLength())
      || h3TierByKey((document.getElementById('h3_tier') || {}).value);
}

// Restore the saved shape into the hidden inputs HERE, at parse time — before
// the boot sequence at the bottom of this script runs setMode('t2v'), which
// reaches setEngine → setH3Tier and would otherwise persist the HTML default
// over the user's choice. (Caught in validation: the tier reset to Draft on
// every reload.)
(function _restoreH3TierEarly() {
  const inp = document.getElementById('h3_tier');
  if (!inp) return;
  // Two axes now, but the SINGLE key is still what older installs persisted, so
  // read the axes first and fall back to it. A user who reloads after this
  // update keeps the shape they were on rather than being reset to Draft 3s.
  let savedQ = null, savedL = null, saved = null;
  try {
    savedQ = localStorage.getItem('phos_h3_quality');
    savedL = localStorage.getItem('phos_h3_length');
    saved = localStorage.getItem('phos_h3_tier');
  } catch (e) {}
  let cell = (savedQ && savedL) ? h3CellFor(savedQ, savedL) : null;
  if (!cell && saved) {
    const k = h3ResolveTierKey(saved);
    cell = (H3.tiers || []).find(t => t.key === k) || null;
  }
  if (cell) {
    inp.value = cell.key;
    const q = document.getElementById('h3_quality');
    const l = document.getElementById('h3_length');
    if (q) q.value = cell.quality;
    if (l) l.value = cell.length;
  }
  // Same treatment for the export canvas, and for the same reason.
  const up = document.getElementById('h3_upscale');
  if (!up) return;
  let savedUp = null;
  try { savedUp = localStorage.getItem('phos_h3_upscale'); } catch (e) {}
  const allowed = H3.upscale_modes || ['off', 'fit_720p', 'fit_1080p'];
  if (savedUp && allowed.indexOf(savedUp) !== -1) up.value = savedUp;
  // And again for the sampler-depth pills, same reason.
  const st = document.getElementById('h3_steps');
  if (!st) return;
  let savedSt = null;
  try { savedSt = localStorage.getItem('phos_h3_steps'); } catch (e) {}
  if (savedSt && ['auto', '12', '16', '20'].indexOf(savedSt) !== -1) st.value = savedSt;
  // And Turbo — but only restore an ON state when this install can actually
  // serve it. A user who downloaded the adapter, deleted it and reloaded must
  // come back on Standard, not on a mode that would fail at queue time.
  const tb = document.getElementById('h3_turbo');
  if (!tb) return;
  let savedTb = null;
  try { savedTb = localStorage.getItem('phos_h3_turbo'); } catch (e) {}
  // Turbo is the DEFAULT when the adapter is installed (owner's call): it is
  // ~half the wall clock and graded better at the mouth than the full-step
  // path, so the fast one should be the one you land on and Standard the one
  // you reach for. An explicit "0" is still honoured — only the ABSENCE of a
  // preference defaults on. Never defaults on when the adapter isn't there.
  const turboOk = !!(H3.turbo && H3.turbo.available);
  tb.value = (turboOk && savedTb !== '0') ? '1' : '0';
})();

// What the export pass will DO to the selected tier's canvas, in one line under
// the Export row. Every tier used to advertise itself as "768×448 · 124f" and
// nothing said which of them came back with black bars on it — the answer
// depends on BOTH the tier's aspect and the export target, so it is rendered
// here rather than baked into a chip. The strings come from the server
// (`tier.export_note[mode]`, generated from compute_upscale_plan itself), so
// this function only picks one; it never phrases anything.
// Reads the QUALITY's export_note, not the cell's: what the export pass does to
// a frame depends only on the canvas, so the sentence lives on the canvas and
// /status doesn't repeat three strings across twelve cells on every tick.
function h3QualityByKey(key) {
  return (H3.qualities || []).find(q => q.key === key) || (H3.qualities || [])[0] || null;
}
function h3LengthByKey(key) {
  return (H3.lengths || []).find(l => l.key === key) || (H3.lengths || [])[0] || null;
}
function _h3SyncExportNote() {
  const el = document.getElementById('h3ExportNote');
  if (!el) return;
  const q = h3QualityByKey(h3CurrentQuality());
  const mode = (document.getElementById('h3_upscale') || {}).value || 'off';
  const txt = (q && q.export_note && q.export_note[mode]) || '';
  el.textContent = txt;
  el.hidden = !txt;
}

// Export canvas for an H3 render. Separate from the LTX `upscale` control
// (which is data-ltx-only and folds away on H3) so one pill never means two
// things. Server-side make_job re-validates — a stale tab must never win.
function setH3Upscale(mode) {
  const allowed = H3.upscale_modes || ['off', 'fit_720p', 'fit_1080p'];
  const v = allowed.indexOf(mode) !== -1 ? mode : (H3.default_upscale || 'fit_720p');
  const inp = document.getElementById('h3_upscale');
  if (inp) inp.value = v;
  document.querySelectorAll('#h3UpscaleGroup [data-h3-upscale]').forEach(b =>
    b.classList.toggle('active', b.dataset.h3Upscale === v));
  try { localStorage.setItem('phos_h3_upscale', v); } catch (e) {}
  _h3SyncExportNote();
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
}
document.querySelectorAll('#h3UpscaleGroup [data-h3-upscale]').forEach(b => {
  b.onclick = () => setH3Upscale(b.dataset.h3Upscale);
});

// H3 orientation — a per-render flip of the resolved cell's canvas, not a new
// tier. The chips' estimates stay valid because a rotation changes no pixel
// count. Persisted like the other H3 sub-preferences.
function setH3Orientation(v) {
  const val = (v === 'portrait') ? 'portrait' : 'landscape';
  const inp = document.getElementById('h3_orientation');
  if (inp) inp.value = val;
  document.querySelectorAll('#h3OrientationGroup [data-h3-orientation]').forEach(b =>
    b.classList.toggle('active', b.dataset.h3Orientation === val));
  try { localStorage.setItem('phos_h3_orientation', val); } catch (e) {}
  _h3SyncOrientationSubs();
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
}
// Say the ACTUAL canvases the current cell would produce, both ways round.
function _h3SyncOrientationSubs() {
  const cell = (typeof h3CurrentCell === 'function') ? h3CurrentCell() : null;
  const land = document.getElementById('h3OrientLandSub');
  const port = document.getElementById('h3OrientPortSub');
  if (!cell) return;
  if (land) land.textContent = `${cell.width}×${cell.height}`;
  if (port) port.textContent = `${cell.height}×${cell.width} · vertical`;
}
document.querySelectorAll('#h3OrientationGroup [data-h3-orientation]').forEach(b => {
  b.onclick = () => setH3Orientation(b.dataset.h3Orientation);
});
(function _restoreH3Orientation() {
  let v = 'landscape';
  try { v = localStorage.getItem('phos_h3_orientation') || 'landscape'; } catch (e) {}
  const apply = () => setH3Orientation(v);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply);
  } else { apply(); }
})();

// Sampler depth for an H3 render. 'auto' = the tier's tuned count (stamped in
// H3_TIERS); a number overrides it for every window of the job. Server-side
// make_job clamps to 4-30 and re-validates — a stale tab must never win.
function setH3Steps(v) {
  const allowed = ['auto', '12', '16', '20'];
  v = allowed.indexOf(String(v)) !== -1 ? String(v) : 'auto';
  const inp = document.getElementById('h3_steps');
  if (inp) inp.value = v;
  document.querySelectorAll('#h3StepsGroup [data-h3-steps]').forEach(b =>
    b.classList.toggle('active', b.dataset.h3Steps === v));
  // Mirror the resolved count into the shared hidden `steps` so the queue
  // card and the estimate line read the truth (make_job re-stamps anyway).
  const tier = h3CurrentCell();
  const s = document.getElementById('steps');
  if (s && tier) s.value = (v === 'auto') ? tier.steps : parseInt(v, 10);
  try { localStorage.setItem('phos_h3_steps', v); } catch (e) {}
  // A pinned depth changes the wall clock of EVERY cell, so both strips have to
  // re-price. This is the whole point of showing an estimate on the chips: it
  // has to answer "what would this cost me, as things stand right now".
  if (typeof renderH3Axes === 'function') { try { renderH3Axes(); } catch (e) {} }
  if (typeof renderH3Turbo === 'function') { try { renderH3Turbo(); } catch (e) {} }
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
  if (typeof updateCustomizeSummary === 'function') { try { updateCustomizeSummary(); } catch (e) {} }
}
document.querySelectorAll('#h3StepsGroup [data-h3-steps]').forEach(b => {
  b.onclick = () => setH3Steps(b.dataset.h3Steps);
});

// ---- Turbo: the 4-step distillation LoRA ------------------------------------
// A speed MODE, not a tier: same model, same geometry, fewer denoise passes.
// Three states, and the UI has to say which one it is in:
//   runner has no --lora  → the whole row is hidden (an old pack never learns
//                           Turbo exists, exactly like chained tiers)
//   supported, not installed  → dashed pill; click explains/fetches the asset
//   available             → a normal pill, and picking it pins steps at 4
function h3TurboState() {
  return (H3 && H3.turbo) || { available: false, supported: false, downloaded: false };
}

// The per-tier estimate comes from the server's tier table (turbo_eta), built
// from that tier's own GEOMETRY — Turbo runs 3 forwards whatever the tier bakes
// and the fixed per-window cost doesn't shrink, so there is no single ratio
// that could be right for every tier (it is 0.45 on an 8-forward one and 0.59
// on a 6-forward one). The retired adapter has end-to-end measurements in the
// changelog, but LightX2V v1.0 does not yet; its active cells remain derived
// rather than inheriting a measurement from different weights.
// The pill's SECOND line, in the same grammar every other .pill-btn in
// Customize uses (name on top, one spec line under it): the cost of turning it
// on, or the cost of getting it at all.
// Both Speed segments print an ABSOLUTE wall clock, in the same shape, so the
// two are directly comparable. Two mistakes were baked into the old copy and
// the owner hit both: the Standard segment described the sampler ("the tier's
// own sampler") while Turbo quoted a number, so there was nothing to compare
// against; and the number was rendered as "~4 min", whose tilde reads as a
// MINUS at this size — he read "-4 min at this tier" as four minutes being
// ADDED. No tildes here, and never a delta: just "8 min" vs "4 min".
function _h3EtaPlain(s) {
  // "~17-19 min" -> "17-19 min", "~4 min" -> "4 min", "~27 min · batch" kept.
  return String(s || '').replace(/[~≈]/g, '').trim();
}
// Minutes for a cell under the CURRENT sampler state. The server pre-computes
// the two states that matter (the cell's own steps, and Turbo's 3 forwards) and
// ships the two model outputs — per_forward_sec and fixed_sec — so a PINNED
// Steps override can be priced in the browser through the same arithmetic
// rather than through a second, drifting cost model.
function h3CellEtaMin(cell, opts) {
  if (!cell) return 0;
  opts = opts || {};
  const turbo = (opts.turbo != null) ? opts.turbo
    : ((document.getElementById('h3_turbo') || {}).value === '1');
  if (turbo) return cell.turbo_min;
  const ov = (opts.steps != null) ? String(opts.steps)
    : ((document.getElementById('h3_steps') || {}).value || 'auto');
  if (ov !== 'auto' && /^\d+$/.test(ov)) {
    const win = Math.max(1, cell.chain_windows || 1);
    const fwd = Math.max(1, parseInt(ov, 10) - 1);
    return (win * fwd * cell.per_forward_sec + win * cell.fixed_sec) / 60;
  }
  return cell.eta_min;
}
function h3FmtEtaMin(m) {
  return '~' + Math.max(1, Math.round(m)) + ' min' + (m >= 25 ? ' · batch' : '');
}
// The eta STRING for a cell in the current state. Prefers the server's own
// string whenever the state is one the server priced (default steps, or Turbo),
// because that is where a MEASURED wall clock lives — "~8-9 min" on the one
// Turbo run that has actually been rendered end to end is not a number this
// function should be regenerating.
function h3CellEta(cell, opts) {
  if (!cell) return '';
  opts = opts || {};
  const turbo = (opts.turbo != null) ? opts.turbo
    : ((document.getElementById('h3_turbo') || {}).value === '1');
  if (turbo) return cell.turbo_eta;
  const ov = (document.getElementById('h3_steps') || {}).value || 'auto';
  if (ov === 'auto' || !/^\d+$/.test(ov)) return cell.eta;
  return h3FmtEtaMin(h3CellEtaMin(cell, { turbo: false }));
}
function h3SpeedSub(which) {
  const cell = h3CurrentCell();
  if (which === 'standard') {
    const eta = cell ? _h3EtaPlain(h3CellEta(cell, { turbo: false })) : '';
    return eta || 'this shape as tuned';
  }
  const t = h3TurboState();
  if (!t.downloaded && !t.install_available) return 'adapter asset pending';
  if (!t.downloaded) return (t.download_gb || 2.0) + ' GB download';
  const eta = cell && cell.turbo_eta ? _h3EtaPlain(cell.turbo_eta) : '';
  return eta || '4-step adapter';
}
function h3TurboPillSub() {
  return h3SpeedSub('turbo');
}
// Kept for anything (and anyone) still reading the one-line form.
function h3TurboPillLabel() {
  return 'Turbo · ' + h3TurboPillSub();
}

function renderH3Turbo() {
  const row = document.getElementById('h3TurboRow');
  const pill = document.getElementById('h3TurboPill');
  const sub = document.getElementById('h3TurboPillSub');
  const t = h3TurboState();
  if (row) row.hidden = !t.supported;
  if (!pill) return;
  if (sub) sub.textContent = h3SpeedSub('turbo');
  const stdSub = document.getElementById('h3StdPillSub');
  if (stdSub) stdSub.textContent = h3SpeedSub('standard');
  pill.classList.toggle('needs-download', !t.downloaded);
  // Two different kinds of number, and the tooltip must not blur them: the ONE
  // shape that has actually been rendered with the adapter end to end says so,
  // everything else says out loud that its figure is derived from geometry.
  const tier = h3CurrentCell();
  const basis = (tier && tier.turbo_measured)
    ? ' Measured end to end at this exact canvas and length — not derived.'
    : ' Estimated for this shape: Turbo runs ' + ((tier && tier.turbo_forwards) || 3)
      + ' forwards instead of ' + ((tier && tier.forwards) || 8)
      + ', over the same fixed load/decode time. Not measured at this canvas.';
  const fallbackNote = (t.downloaded && t.fallback && t.adapter_version)
    ? ' Running the ' + t.adapter_version + ' fallback adapter — the v1.0 '
      + 'download replaces it.'
    : '';
  pill.title = t.downloaded
    ? (t.note || '') + basis + fallbackNote
    : (t.install_available
      ? 'Downloads the LightX2V v1.0 runner-layout adapter (~'
        + (t.download_gb || 2.0) + ' GB) into the H3 pack.'
      : (t.install_note || 'The runner-layout adapter release asset is pending.'));
  // The pack could have gone away (or arrived) since boot without a reload.
  if (!t.available && (document.getElementById('h3_turbo') || {}).value === '1') {
    setH3Turbo(false);
  }
}

// Steps and Turbo are the same axis, so only one of them can be in charge.
// Turbo is, and the pills go visibly dead rather than silently ignored — the
// server drops the override too (make_job), this just stops it looking live.
function _h3SyncStepsEnabled() {
  const on = (document.getElementById('h3_turbo') || {}).value === '1';
  const row = document.getElementById('h3StepsRow');
  document.querySelectorAll('#h3StepsGroup [data-h3-steps]').forEach(b => {
    b.disabled = on;
    b.classList.toggle('disabled', on);
  });
  if (row) {
    row.style.opacity = on ? '.5' : '';
    row.title = on ? 'Turbo pins the sampler at 4 steps.' : '';
  }
}

function setH3Turbo(on) {
  const t = h3TurboState();
  const v = (on && t.available) ? '1' : '0';
  const inp = document.getElementById('h3_turbo');
  if (inp) inp.value = v;
  document.querySelectorAll('#h3TurboGroup [data-h3-turbo]').forEach(b =>
    b.classList.toggle('active', b.dataset.h3Turbo === v));
  const note = document.getElementById('h3TurboNote');
  if (note) {
    note.hidden = (v !== '1');
    note.textContent = t.note || '';
  }
  if (v === '1') {
    // ORDER IS LOAD-BEARING, and it was wrong until Steps became a primary
    // control and the lie got visible: setH3Steps('auto') re-derives the
    // shared hidden `steps` from the TIER (9), so running it after the pin
    // overwrote the 4 and the derived line read "Steps 9" on a Turbo render.
    // Release the pill override first, THEN pin. make_job stamps
    // H3_TURBO_STEPS server-side either way — this only ever affected what the
    // form told the user it was about to do, which is the whole reason Speed
    // and Steps were moved out where the user can see them.
    if (typeof setH3Steps === 'function') { try { setH3Steps('auto'); } catch (e) {} }
    const s = document.getElementById('steps');
    if (s) s.value = t.steps || 4;
  }
  _h3SyncStepsEnabled();
  // Turbo is one half of the single-adapter-slot conflict, so the Adapter row
  // appears / disappears with it.
  if (typeof renderH3LoraSlot === 'function') { try { renderH3LoraSlot(); } catch (e) {} }
  try { localStorage.setItem('phos_h3_turbo', v); } catch (e) {}
  // Turbo roughly halves every cell, so the chips have to re-price — a strip
  // still advertising the 9-step wall clock while Turbo is lit is the same lie
  // the Speed pill's absolute times were added to kill.
  if (typeof renderH3Axes === 'function') { try { renderH3Axes(); } catch (e) {} }
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
  if (typeof updateCustomizeSummary === 'function') { try { updateCustomizeSummary(); } catch (e) {} }
}

// One click, two behaviours, because the pill has two jobs: select Turbo when
// it is ready, or fetch it when it isn't. Never both — a click that starts a
// download must not also arm a render that has nothing to render with.
async function h3TurboClick() {
  const t = h3TurboState();
  if (t.available) { setH3Turbo(true); return; }
  if (!t.supported) {
    if (typeof phosToast === 'function') {
      phosToast('This Hailuo H3 pack predates Turbo. Re-run "Install Hailuo H3" '
                + 'in the Pinokio sidebar to update the clone — your weights stay.',
                { kind: 'danger' });
    }
    return;
  }
  if (!t.install_available) {
    if (typeof phosToast === 'function') {
      phosToast(t.install_note || 'The H3 Turbo runner-layout adapter release asset is pending.',
                { kind: 'danger' });
    }
    return;
  }
  const gb = t.download_gb || 2.0;
  if (!confirm('Download the H3 Turbo adapter?\n\n'
             + '~' + gb + ' GB, into the H3 pack’s models folder.\n'
             + 'The LightX2V source adapter is Apache-2.0.\n\n'
             + 'Progress streams to the log at the bottom of the page.')) return;
  try {
    const r = await fetch('/h3/turbo/install', { method: 'POST' });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    if (typeof phosToast === 'function') {
      phosToast('Turbo download started — watch the log. The pill turns on by '
                + 'itself when the adapter lands.', { kind: 'ok' });
    }
  } catch (e) {
    if (typeof phosToast === 'function') {
      phosToast('Turbo download: ' + (e.message || 'failed'), { kind: 'danger' });
    }
  }
}

document.querySelectorAll('#h3TurboGroup [data-h3-turbo]').forEach(b => {
  b.onclick = () => (b.dataset.h3Turbo === '1') ? h3TurboClick() : setH3Turbo(false);
});

// Adapter-slot pills. Bound at parse time like the Turbo / Steps groups —
// the row itself is hidden until Turbo and a LoRA actually collide.
document.querySelectorAll('#h3LoraSlotGroup [data-h3-lora-slot]').forEach(b => {
  b.onclick = () => setH3LoraSlot(b.dataset.h3LoraSlot);
});

// The switcher earns its place only when there is a real choice to make: at
// least two ACTIONABLE engines on the current workflow. One engine means the
// control and its divider both disappear — which is exactly what the ~80% of
// users under 64 GB have always seen, preserved through the move to the
// header. An `announced` engine is news, not a choice: it rides along once
// the switcher is already up, but it must never bring it up on its own, or a
// 32 GB Mac gets a permanently dead chip — the exact noise the capability
// gate exists to prevent. (Old name kept: it is what the rest of this file
// calls.)
function _engineRowVisible() {
  return ENGINES.filter(e => engineRenderable(e) && e.state !== 'announced').length > 1;
}

// ---- The header switcher ----------------------------------------------------
// Rendered entirely from the registry. Every visual decision — label, mark,
// accent, badge, tooltip, order — comes off the engine's own row, so this
// Scope the mode strip to what the ACTIVE engine actually renders. Each
// engine's form should be its own surface: a model that can't do Extend
// shouldn't advertise Extend. Purely additive to setEngine — the authority
// is still engineServesMode() (and make_job server-side); this only stops the
// UI from offering a mode that would silently bounce the user to the other
// engine. Registry-driven, so a third engine scopes itself for free.
// The Remix parent pill is a UI GROUP whose children are real modes, so it
// hides only when the engine serves NONE of them.
function syncModeStripToEngine() {
  const e = engineById(currentEngine());
  if (!e) return;
  document.querySelectorAll('#modeGroup .mode-chip').forEach(chip => {
    const mode = chip.dataset.mode;
    if (!mode) return;
    let ok;
    if (mode === 'remix') {
      const kids = (typeof REMIX_MODES !== 'undefined' && REMIX_MODES)
        ? REMIX_MODES : ['ingredients'];
      ok = kids.some(m => engineServesMode(e, m));
    } else {
      ok = engineServesMode(e, mode);
    }
    chip.classList.toggle('eng-foreign', !ok);
  });
}

// function has no idea which engines exist. Re-run on: boot, every setEngine,
// a workflow-tab change, and any /status tick that moves an install.
// The picker is a DROPDOWN, not a segmented row. Two reasons, and the second
// is the one that forced it:
//   1. Header room. Two full segments plus the health cluster already
//      overflowed a 14" window; a third engine (Flux Video) could not fit at
//      any width. A trigger showing only the ACTIVE engine costs one slot no
//      matter how many engines exist.
//   2. "Which one is selected" has to survive being glanced at. In a segmented
//      row that is carried entirely by a highlight; a dropdown states it.
//
// The MENU IS PORTALED TO <body> and positioned fixed, because <header> is
// overflow:hidden — the same clipping that produced the "avatar cut off"
// report. A menu rendered inside the header would be sliced at its edge.
window._engineMenuOpen = false;

function _engineMenuEl() {
  let m = document.getElementById('engineMenu');
  if (!m) {
    m = document.createElement('div');
    m.id = 'engineMenu';
    m.setAttribute('role', 'listbox');
    m.hidden = true;
    document.body.appendChild(m);     // portal: escape header overflow:hidden
  }
  return m;
}

function closeEngineMenu() {
  window._engineMenuOpen = false;
  const m = document.getElementById('engineMenu');
  if (m) m.hidden = true;
  const t = document.querySelector('#engineSwitch .eng-trigger');
  if (t) t.setAttribute('aria-expanded', 'false');
}

function toggleEngineMenu() {
  const m = _engineMenuEl();
  if (window._engineMenuOpen) { closeEngineMenu(); return; }
  const trig = document.querySelector('#engineSwitch .eng-trigger');
  if (!trig) return;
  const r = trig.getBoundingClientRect();
  m.style.top = (r.bottom + 6) + 'px';
  m.style.left = r.left + 'px';
  m.style.minWidth = Math.max(r.width, 240) + 'px';
  m.hidden = false;
  window._engineMenuOpen = true;
  trig.setAttribute('aria-expanded', 'true');
}

document.addEventListener('click', (ev) => {
  if (!window._engineMenuOpen) return;
  if (ev.target.closest('#engineMenu') || ev.target.closest('#engineSwitch')) return;
  closeEngineMenu();
}, true);
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && window._engineMenuOpen) closeEngineMenu();
});
window.addEventListener('resize', closeEngineMenu);

function renderEngineSwitch() {
  const box = document.getElementById('engineSwitch');
  const div = document.getElementById('engineSwitchDivider');
  if (!box) return;
  const show = _engineRowVisible();
  box.hidden = !show;
  if (div) div.hidden = !show;
  if (!show) { box.innerHTML = ''; closeEngineMenu(); return; }

  const active = currentEngine();
  const list = ENGINES.filter(engineRenderable);

  // ---- The trigger: the active engine, and nothing else ----
  const act = list.find(e => e.id === active) || list[0];
  if (act) {
    box.innerHTML = `<button type="button" class="eng-trigger" aria-haspopup="listbox"
        aria-expanded="false"
        style="--eng-accent:${escapeHtml(act.accent)};--eng-dim:${escapeHtml(act.accent_dim)};--eng-soft:${escapeHtml(act.accent_soft)}"
        title="${escapeHtml(_engineTooltip(act, engineStatus(act), engineServesMode(act, currentMode)))}">
      <span class="eng-mark"><svg class="ph" aria-hidden="true"><use href="#${escapeHtml(act.mark)}"/></svg></span>
      <span class="eng-seg-name">${escapeHtml(act.label)}</span>${
      act.generation ? `<span class="eng-seg-gen">${escapeHtml(act.generation)}</span>` : ''}
      <svg class="ph eng-caret" aria-hidden="true"><use href="#ph-caret-down-bold"/></svg>
    </button>`;
    const t = box.querySelector('.eng-trigger');
    if (t) t.onclick = (ev) => { ev.stopPropagation(); toggleEngineMenu(); };
  }

  // ---- The menu: every engine, each stating its own state ----
  const menu = _engineMenuEl();
  menu.innerHTML = list.map(e => {
    const st = engineStatus(e);
    const offer = !e.builtin && !st.announced && st.capable && !st.available;
    const modeOk = engineServesMode(e, currentMode);
    // Four badges, four different sentences. An OFFER (download / repair) is
    // worth the engine's own colour; a constraint is not.
    let badge = '', badgeClass = '';
    if (st.announced) { badge = 'soon'; }
    else if (offer) {
      badge = st.repairable ? 'repair' : (e.install_size || '').replace(/^~/, '');
      badgeClass = ' offer';
    } else if (st.available && !modeOk) { badge = (e.serves_label || '').toLowerCase().replace(' and ', ' · '); }
    // Inert = real but unreachable RIGHT NOW. Distinct from needs-install,
    // which IS reachable — that click is the install.
    const inert = st.announced || (st.available && !modeOk);
    const cls = 'eng-opt'
      + (e.id === active ? ' active' : '')
      + (offer ? ' needs-install' : '')
      + (inert ? ' inert' : '');
    // The tagline earns its place here in a way it never could in a segment:
    // the menu is where someone decides BETWEEN engines, so each row says what
    // it is for rather than only what it is called.
    return `<button type="button" class="${cls}" data-engine="${escapeHtml(e.id)}"
        role="option" aria-selected="${e.id === active ? 'true' : 'false'}"
        style="--eng-accent:${escapeHtml(e.accent)};--eng-dim:${escapeHtml(e.accent_dim)};--eng-soft:${escapeHtml(e.accent_soft)}"
        title="${escapeHtml(_engineTooltip(e, st, modeOk))}">
      <span class="eng-mark"><svg class="ph" aria-hidden="true"><use href="#${escapeHtml(e.mark)}"/></svg></span>
      <span class="eng-opt-body">
        <span class="eng-opt-top">
          <span class="eng-seg-name">${escapeHtml(e.label)}</span>${
          e.generation ? `<span class="eng-seg-gen">${escapeHtml(e.generation)}</span>` : ''}${
          badge ? `<span class="eng-badge${badgeClass}">${escapeHtml(badge)}</span>` : ''}
        </span>
        <span class="eng-opt-sub">${escapeHtml(e.tagline || e.sublabel || '')}</span>
      </span>${
      e.id === active ? '<svg class="ph eng-tick" aria-hidden="true"><use href="#ph-check-bold"/></svg>' : ''}
    </button>`;
  }).join('');

  menu.querySelectorAll('.eng-opt').forEach(b => {
    b.onclick = () => {
      closeEngineMenu();
      engineSegClick(b.dataset.engine);
    };
  });
}

// One sentence per state, and it has to be the RIGHT one — telling a user with
// 75 GB of weights on disk that the engine "isn't installed" is the v3.4.0
// regression report, verbatim.
function _engineTooltip(e, st, modeOk) {
  // The generation belongs in the TITLE too: the compact breakpoint
  // (<=1500 px — a 14" MBP window) hides .eng-seg-name and .eng-seg-gen, so on
  // a laptop the release's headline fix had no surface at all and #modelTag was
  // the only thing naming the build.
  const name = e.label + (e.generation ? ' ' + e.generation : '')
             + ' · ' + (e.sublabel || '');
  if (st.announced) return name + ' — ' + (e.tagline || 'not released yet');
  if (!e.builtin && !st.capable) {
    // The floor is the LOWEST lane's, served by the panel. This read
    // `st.min_ram_gb || 64` — the bf16 number, or a literal 64 that no floor
    // in this codebase has ever been — on a machine whose real bar is 46.
    return name + ' — needs ' + (st.ram_floor_gb || st.min_ram_gb || 46)
                + ' GB unified memory';
  }
  if (!e.builtin && !st.available) {
    return st.repairable
      ? name + ' — needs repair; your weights are still on disk. Click for the one-click fix.'
      : name + ' — available to install (' + (e.install_size || '')
        + '). Click to see what it does.';
  }
  if (!modeOk) {
    return name + ' — renders ' + (e.serves_label || 'other modes')
         + " only, and this mode isn't one of them.";
  }
  return name + ' — ' + (e.tagline || '');
}

// One click, several jobs, and never two at once: select the engine when it is
// ready, open its install/repair card when it isn't, and do nothing at all
// when it is inert. A click that starts a download must not also arm a render
// that has nothing to render with.
function engineSegClick(id) {
  const e = engineById(id);
  if (!e) return;
  const st = engineStatus(e);
  if (st.announced) return;
  if (!e.builtin && st.capable && !st.available) {
    const fn = e.install_card && window[e.install_card];
    // 'chip' tells the card this click came from the switcher, where a repeat
    // click means "yes, I know" rather than "explain it again". The explicit
    // buttons elsewhere pass nothing and always get the full card.
    if (typeof fn === 'function') fn('chip');
    return;
  }
  if (!e.builtin && !st.capable) return;
  if (!engineServesMode(e, currentMode)) return;
  setEngine(id);
}

// ---- The two strips ---------------------------------------------------------
// Each chip prints the eta of the cell it WOULD select — the Quality chips at
// the current length, the Length chips at the current quality — so the estimate
// is live on both axes at once and changing either one re-prices the other. The
// numbers are the server's: a chip looks up a cell, it never computes a canvas
// or a duration of its own.
// Per-engine differences, and ONLY the differences. Everything else about a
// tier chip — the classes, the three spans, the unavailable state, the title
// rules — is shared, because it was already right for H3 and re-deriving it for
// LTX is how two strips start disagreeing about what "unavailable" looks like.
//
// The engine argument is DEFAULTED to 'h3' throughout this section on purpose:
// every existing H3 call site then stays byte-identical, which is the property
// the refactor is checked against (560 chips, diffed before and after).
const TIER_ENGINES = {
  h3: {
    boot: () => H3,
    // H3's canvases are several different aspect ratios, so the ratio is the
    // fact worth printing beside the canvas.
    qualitySpec: (item, cell) => `${item.canvas} · ${item.aspect}`,
    lengthSpec: (item, cell) => (cell
      ? `${cell.frames}f` + (cell.chain_windows > 1 ? ` · ${cell.chain_windows}×5s` : '')
      : `${item.frames}f`),
    eta: (cell) => h3CellEta(cell),
    cellFor: (q, l) => h3CellFor(q, l),
    currentQuality: () => h3CurrentQuality(),
    currentLength: () => h3CurrentLength(),
    setQuality: (k) => setH3Quality(k),
    setLength: (k) => setH3Length(k),
    qStripId: 'h3QualityGroup', lStripId: 'h3LengthGroup',
    metaId: 'h3LengthMeta', noteId: '',
    qAttr: 'h3-quality', lAttr: 'h3-length',
  },
  ltx: {
    boot: () => (BOOT.ltx || {}),
    // LTX is 16:9 or 4:3 and the user picked the canvas by name; what changes
    // under them as they move the Length axis is the FRAME COUNT, so that is
    // what the quality chip prints. It is the current cell's frames, never a
    // fixed number — a chip that said "73f" while rendering 121 would be the
    // same class of lie as the engine label this release is fixing.
    qualitySpec: (item, cell) => `${item.canvas} · ${cell ? cell.frames : '—'}f`,
    lengthSpec: (item, cell) => (cell ? `${cell.frames}f` : `${item.frames}f`),
    eta: (cell) => ltxCellEta(cell),
    // A cell whose weights are not on disk is an INSTALL OFFER, not a choice.
    // `offered` is the RAM question and is resolved at import time; this is the
    // pack question and it can only be answered at runtime, so it lives here.
    needsInstall: (cell) => ltxCellNeedsInstall(cell),
    installLabel: (cell) => ltxCellInstallLabel(cell),
    cellFor: (q, l) => ltxCellFor(q, l),
    currentQuality: () => ltxCurrentQuality(),
    currentLength: () => ltxCurrentLength(),
    setQuality: (k) => setLtxQuality(k),
    setLength: (k) => setLtxLength(k),
    qStripId: 'qualityGroup', lStripId: 'ltxLengthGroup',
    metaId: 'ltxLengthMeta', noteId: 'ltxTierNote',
    // The CANVAS chips keep the SHIPPED `data-quality` attribute rather than
    // taking an engine-prefixed one. Five surfaces already read it — the click
    // handler, setQuality's active toggle, applyTierTimes, the trained-LoRA
    // compatibility gate and the character-strip swap — and renaming it to
    // match a naming scheme would have broken all five for tidiness. H3's
    // chips are new markup and have no such history, so they keep theirs.
    qAttr: 'quality', lAttr: 'ltx-length',
    // A duration that is not on the axis is not an error and is not hidden —
    // power users have shipped work with the raw Frames field. It lights no
    // chip and says what it is: `custom · 337f · 14 s`.
    customMeta: () => {
      const f = parseInt((document.getElementById('frames') || {}).value, 10);
      if (!Number.isFinite(f)) return '';
      const secs = Math.round(((f - 1) / 24) * 10) / 10;
      return `custom · ${f}f · ${secs} s`;
    },
  },
};

function _tierChipHtml(engine, kind, item, cell, active) {
  const E = TIER_ENGINES[engine] || TIER_ENGINES.h3;
  const ok = !cell ? false : (cell.available !== false);
  // NEEDS-INSTALL is a third state, distinct from unavailable: the cell is
  // renderable on this Mac, the weights just aren't here yet. It reads as a
  // CTA (dashed + download glyph via .needs-install) and its third slot names
  // the download instead of an ETA the user cannot have yet.
  const needsInstall = ok && E.needsInstall ? !!E.needsInstall(cell) : false;
  const cls = 'q-chip pill-btn pill-quality'
    + (active ? ' active' : '') + (ok ? '' : ' unavailable')
    + (needsInstall ? ' needs-install' : '');
  const spec = (kind === 'quality') ? E.qualitySpec(item, cell)
                                    : E.lengthSpec(item, cell);
  const foot = !ok ? 'unavailable'
             : needsInstall ? E.installLabel(cell)
             : E.eta(cell);
  const title = ok
    ? ((cell && cell.blurb) || item.blurb || '')
    : (cell && cell.unavailable_reason) || 'Not available on this install.';
  const attr = (kind === 'quality') ? (E.qAttr || `${engine}-quality`)
                                    : (E.lAttr || `${engine}-length`);
  return `
    <button type="button" class="${cls}" data-${attr}="${escapeHtml(item.key)}"
            ${ok ? '' : 'aria-disabled="true"'} title="${escapeHtml(title)}">
      <span class="ql-name">${escapeHtml(item.label)}</span>
      <span class="q-spec ql-spec sub">${escapeHtml(spec)}</span>
      <span class="ql-tier">${escapeHtml(foot)}</span>
    </button>`;
}
// The shipped name, kept as a one-line shim so every H3 call site is unchanged.
function _h3ChipHtml(kind, item, cell, active) {
  return _tierChipHtml('h3', kind, item, cell, active);
}

function renderTierAxes(engine) {
  engine = engine || 'h3';
  const E = TIER_ENGINES[engine] || TIER_ENGINES.h3;
  const B = E.boot() || {};
  const qStrip = document.getElementById(E.qStripId);
  const lStrip = document.getElementById(E.lStripId);
  if (!qStrip || !lStrip) return;
  const qualities = B.qualities || [];
  const lengths = B.lengths || [];
  const curQ = E.currentQuality();
  const curL = E.currentLength();

  qStrip.style.gridTemplateColumns = `repeat(${Math.max(1, qualities.length)}, 1fr)`;
  qStrip.innerHTML = qualities.map(q =>
    _tierChipHtml(engine, 'quality', q, E.cellFor(q.key, curL), q.key === curQ)).join('');
  // Past four lengths (the lab dense pass turns a fifth on) the chips would
  // squeeze; wrap to three per row and let the strip grow a line instead.
  lStrip.style.gridTemplateColumns =
    `repeat(${lengths.length > 4 ? 3 : Math.max(1, lengths.length)}, 1fr)`;
  lStrip.innerHTML = lengths.map(l =>
    _tierChipHtml(engine, 'length', l, E.cellFor(curQ, l.key), l.key === curL)).join('');

  const qAttr = E.qAttr || `${engine}-quality`;
  const lAttr = E.lAttr || `${engine}-length`;
  qStrip.querySelectorAll(`[data-${qAttr}]`).forEach(b => {
    b.onclick = () => {
      // A needs-install chip is a DOWNLOAD BUTTON, not a tier choice. Binding
      // setQuality directly here bypassed the shipped .needs-install routing:
      // clicking High selected a tier the machine could not render, the
      // pre-render block explained why for ~1.5 s, and the next poll silently
      // moved the form to a bigger, slower canvas nobody asked for.
      if (b.classList.contains('needs-install')) {
        if (typeof openModelsModal === 'function') openModelsModal();
        return;
      }
      E.setQuality(b.getAttribute(`data-${qAttr}`));
    };
  });
  lStrip.querySelectorAll(`[data-${lAttr}]`).forEach(b => {
    b.onclick = () => E.setLength(b.getAttribute(`data-${lAttr}`));
  });
  // The right-hand meta on the Length label: the combination, in one line.
  const meta = E.metaId ? document.getElementById(E.metaId) : null;
  const cell = E.cellFor(curQ, curL);
  if (meta) {
    // A synthesised custom cell (length '') gets the custom line, which is the
    // one that can state the DURATION a raw frame count works out to — the
    // number the user actually wanted to know when they typed it.
    const custom = !cell || cell.length === '';
    meta.textContent = custom
      ? (E.customMeta ? E.customMeta() : '')
      : `${cell.width}×${cell.height} · ${cell.frames}f · ${E.eta(cell)}`;
  }
  // The selected cell's honest notes, joined. H3 has no note element and
  // passes noteId '' — the lookup is skipped rather than guarded downstream.
  const note = E.noteId ? document.getElementById(E.noteId) : null;
  if (note) {
    const txt = (cell && cell.note) || '';
    note.textContent = txt;
    note.hidden = !txt;
  }
  // Every LTX repaint settles the Speed row too — one hook covers boot,
  // shape changes, Load Params and engine switches, and the function
  // deliberately clears the preset without re-entering this repaint.
  if (engine === 'ltx' && typeof _applySchedPresetRowVisibility === 'function') {
    try { _applySchedPresetRowVisibility(); } catch (e) {}
  }
}
// The shipped name, kept so every H3 call site is unchanged.
function renderH3Axes() { return renderTierAxes('h3'); }

// ---- LTX's side of the shared axis machinery --------------------------------
// The mirror of h3CellFor / h3CurrentQuality / h3CellEta, and nothing more. The
// LTX form already owns #quality (the shipped hidden select every mode reads),
// so the CANVAS axis writes the field that already exists rather than a second
// one — a new hidden input here would be a second definition of "what will
// render", which is the bug class this release keeps closing.
function ltxCellFor(quality, length) {
  const cell = ((BOOT.ltx || {}).tiers || [])
    .find(t => t.quality === quality && t.length === length) || null;
  if (cell || length !== '') return cell;
  // CUSTOM DURATION. The canvas is still perfectly available — only the length
  // is off the axis — so the quality chips must not go grey and claim
  // otherwise. Synthesise a cell that carries the real canvas and the real
  // frame count, and leave the eta as the word `custom` rather than inventing
  // a number: pricing an arbitrary shape in the browser would be the second
  // cost model this whole table exists to avoid.
  const q = ((BOOT.ltx || {}).qualities || []).find(x => x.key === quality);
  if (!q) return null;
  const f = parseInt((document.getElementById('frames') || {}).value, 10);
  if (!Number.isFinite(f)) return null;
  return {
    quality: q.key, quality_label: q.label, length: '', length_label: 'custom',
    width: q.width, height: q.height, frames: f,
    pack: q.pack, pipeline: q.pipeline,
    eta: 'custom', eta_measured: false, available: true, note: '',
    blurb: q.blurb || '',
  };
}
function ltxCurrentQuality() {
  const v = (document.getElementById('quality') || {}).value;
  return v || (BOOT.ltx || {}).default_quality || 'balanced';
}
// DERIVED FROM #frames, not from the hidden field, and that direction matters.
// #frames is what actually renders; the chip is a label on top of it. Reading
// the label would let the two disagree the moment somebody types 337 into
// Customize — the chip would still show 5s while a 14-second clip rendered,
// which is precisely the 7-second lie this release exists to fix, relocated.
// An off-axis frame count matches no rung, lights no chip, and prints itself.
function ltxCurrentLength() {
  const f = parseInt((document.getElementById('frames') || {}).value, 10);
  const lens = (BOOT.ltx || {}).lengths || [];
  const hit = lens.find(l => Number(l.frames) === f);
  if (hit) return hit.key;
  if (Number.isFinite(f)) return '';        // custom — no chip is active
  return (document.getElementById('ltx_length') || {}).value
      || (BOOT.ltx || {}).default_length || '5s';
}
function ltxCurrentCell() {
  return ltxCellFor(ltxCurrentQuality(), ltxCurrentLength());
}
// The eta STRING for a cell. The server's own string wins whenever the state is
// one the server priced, because that is where a MEASURED wall clock lives —
// and where it is NOT measured the string is still the server's model, so the
// browser never carries a second cost model. The only case it recomputes is a
// custom duration, which by definition has no cell.
function ltxCellEta(cell) {
  if (!cell) return '';
  if (cell.eta === 'custom') return 'custom';
  // Chips must not advertise the tuned wall clock while the Fast draft
  // schedule is armed — the same lie the H3 Turbo repaint exists to kill.
  // fast_eta exists only on cells that can run the preset (server-stamped),
  // so HQ cells keep their own number untouched.
  if (cell.fast_eta && typeof schedPresetActive === 'function'
      && schedPresetActive()) {
    return cell.fast_eta + ' · fast draft';
  }
  const tail = (cell.pack === 'q8' && cell.pipeline === 'hq') ? ' · Q8 HQ' : '';
  return (cell.eta || '') + tail;
}
// Are this cell's weights on disk? A RUNTIME question — the tier table is built
// at import time and cannot know. `offered` answers "can this Mac's RAM serve
// this tier"; both must be able to be true (§11-E).
function ltxCellNeedsInstall(cell) {
  if (!cell || cell.pack !== 'q8') return false;
  const s = window.__phosLastStatus || {};
  // UNKNOWN IS NOT MISSING. Before the first /status lands there is no pack
  // answer at all, and `!undefined` is `true` — so the chip was born claiming
  // weights were absent on a machine that has them, and clicking it opened a
  // download for a 30 GB pack the user already owned. A tier is only an install
  // OFFER once we have actually been told something is missing; until then it
  // is an ordinary chip. Being briefly wrong in the harmless direction beats
  // being confidently wrong in the expensive one.
  if (s.q8_available === undefined && s.q8_pack_available === undefined) return false;
  const packOk = (s.q8_pack_available !== undefined) ? s.q8_pack_available : s.q8_available;
  if (cell.pipeline === 'hq') {
    // High needs the pack AND the add-on.
    return !(packOk && (s.hq_addon_missing || []).length === 0);
  }
  return !packOk;
}
// What the chip says instead of an ETA, and it names the download that is
// actually missing rather than a fixed string.
function ltxCellInstallLabel(cell) {
  const s = window.__phosLastStatus || {};
  const P = ((BOOT.ltx || {}).packs) || {};
  const packOk = (s.q8_pack_available !== undefined) ? s.q8_pack_available : s.q8_available;
  // NAME THE DOWNLOAD THAT IS MISSING, not the family it belongs to. With the
  // 30 GB pack present and only the 29.5 GB add-on absent, offering "Q8
  // weights" is the exact conflation `q8_pack_available` was introduced to
  // kill — telling a user to re-buy what they already have.
  const addonMissing = (s.hq_addon_missing || []).length > 0;
  // The pack comes first: without it the add-on has nowhere to land. Once it is
  // present, High's only remaining gap is the add-on.
  let want = P.q8;
  if (cell && cell.pipeline === 'hq' && packOk && addonMissing) want = P.hq_addon;
  if (!want) return 'install needed';
  const short = String(want.name || '').replace(/^LTX-2\.5\s*/, '');
  return `Install ${short} · ${want.size}`;
}
// Set the CANVAS. Length is untouched — the whole point of the two axes.
function setLtxQuality(key) {
  const q = ((BOOT.ltx || {}).qualities || []).find(x => x.key === key);
  if (!q) return;
  _ltxApplyShape(q.key, ltxCurrentLength());
}
// Set the DURATION. Canvas is untouched.
function setLtxLength(key) {
  const l = ((BOOT.ltx || {}).lengths || []).find(x => x.key === key);
  if (!l) return;
  _ltxApplyShape(ltxCurrentQuality(), l.key);
}
// The ONE place the form's LTX shape is written — the mirror of _h3ApplyShape.
// Every other caller (the strips, Load Params, a restored state) routes through
// here, so there is exactly one definition of what the form now says it will
// render, and #duration / #frames stay in agreement with the chips instead of
// drifting from them.
function _ltxApplyShape(qKey, lKey) {
  const cell = ltxCellFor(qKey, lKey);
  if (!cell) return;
  // A cell this canvas does not serve is REFUSED, not silently redirected: the
  // chip is already greyed with the reason in its title, so a click that
  // quietly rendered something else would be worse than a click that does
  // nothing.
  //
  // ...but "does nothing" was the whole complaint on the HQ chips. Now that
  // they are shown-disabled instead of hidden (owner ruling 2026-08-23), a
  // click has to SAY the reason rather than swallow it — a tooltip is not
  // discoverable on a chip a user has just tapped. Same #engineRowNote line
  // _h3ApplyShape writes into, and it deliberately does not set
  // `dataset.packNote`, so applyPackIncompleteGate never clears a note it
  // did not write and never has this one clobbered.
  if (cell.available === false) {
    const note = document.getElementById('engineRowNote');
    const reason = cell.unavailable_reason || '';
    if (note && reason && note.dataset.packNote !== '1') {
      note.textContent = reason;
      note.hidden = false;
    }
    renderTierAxes('ltx');
    return;
  }
  const setv = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.value = v; };
  // The DURATION half is ours. The CANVAS half is setQuality()'s — it has
  // owned #quality, #width, #height, the aspect row and the upscale default
  // since long before this table existed, and a second writer here would be a
  // second definition of "what will render". So we hand it the key and let it
  // do its job; it calls back into renderTierAxes('ltx') when it is done.
  setv('ltx_length', cell.length);
  setv('duration', cell.seconds);
  setv('frames', cell.frames);
  if (typeof setQuality === 'function') setQuality(cell.quality);
  else renderTierAxes('ltx');
}

// Set the CANVAS. Length is untouched — that is the entire point of the
// refactor: "somebody may want to run the HQ 5s version for 10s".
function setH3Quality(key) {
  const q = h3QualityByKey(key);
  if (!q) return;
  _h3ApplyShape(q.key, h3CurrentLength());
}
// Set the DURATION. Canvas is untouched.
function setH3Length(key) {
  const l = h3LengthByKey(key);
  if (!l) return;
  _h3ApplyShape(h3CurrentQuality(), l.key);
}

// The nearest renderable cell to one this install can't serve. Mirrors
// h3_fallback_tier() in Python exactly — keep the CANVAS, give up length, never
// longer than was asked for, never the lab dense path — so a restored state and
// a queued job land on the same shape rather than disagreeing about it.
function _h3FallbackCell(cell) {
  const want = Number((h3LengthByKey(cell.length) || {}).seconds || 0);
  const lens = (H3.lengths || [])
    .filter(l => !l.dense && Number(l.seconds) <= want)
    .sort((a, b) => (b.seconds - a.seconds) || (b.order - a.order));
  for (const l of lens) {
    const c = h3CellFor(cell.quality, l.key);
    if (c && c.available !== false) return c;
  }
  return h3TierByKey(H3.default_tier);
}

// The one place the form's H3 state is written. Everything else — the strips,
// setH3Tier, Load Params, Draft→Finish — routes through here so there is exactly
// one definition of "what the form now says it will render".
//
// An unavailable cell is handled two different ways, and the difference matters:
//   a CLICK is refused (opts.fallback falsy). The chip is already greyed with
//     the reason, and quietly rendering something else is how a user ends up
//     with a 5 s clip they didn't ask for.
//   a RESTORE is redirected (opts.fallback true) — a persisted shape, a Load
//     Params replay, a Finish, a boot on a pack that has since lost chaining.
//     Refusing there would leave the form pointing at something unrenderable,
//     so it degrades along the same path make_job would take and says so.
// Either way make_job re-runs the gate server-side, because a stale tab must
// never win.
function _h3ApplyShape(qualityKey, lengthKey, opts) {
  opts = opts || {};
  let cell = h3CellFor(qualityKey, lengthKey);
  if (!cell) cell = h3CellFor(qualityKey, H3.default_length || '5s')
                || h3TierByKey(H3.default_tier);
  if (!cell) return;
  if (cell.available === false) {
    const note = document.getElementById('engineRowNote');
    const reason = cell.unavailable_reason || '';
    if (!opts.fallback) {
      if (note) { note.textContent = reason; note.hidden = !reason; }
      // Re-render so the click leaves the strips exactly as they were.
      renderH3Axes();
      return;
    }
    const fb = _h3FallbackCell(cell);
    if (!fb || fb.available === false) return;
    if (note) {
      note.textContent = reason + ' Falling back to ' + fb.label + '.';
      note.hidden = false;
    }
    cell = fb;
  }
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set('h3_quality', cell.quality);
  set('h3_length', cell.length);
  set('h3_tier', cell.key);
  // Mirror the cell geometry into the shared hidden fields so the queue card,
  // the "Generate" estimate and Load Params all read the truth. make_job
  // re-stamps these server-side too — a stale tab must never win.
  set('width', cell.width);
  set('height', cell.height);
  set('frames', cell.frames);
  const s = document.getElementById('steps');
  if (s) {
    // Respect a pinned Steps override across shape switches; 'auto' follows
    // the cell's own count.
    const ov = (document.getElementById('h3_steps') || {}).value || 'auto';
    s.value = (ov !== 'auto' && /^\d+$/.test(ov)) ? parseInt(ov, 10) : cell.steps;
  }
  // Cells carry every honest note they owe, and the server owns the text. Two
  // exist today and they now COMPOSE: a chained length warns that one prompt is
  // asked of every window (so a scripted line lands once per window) at ANY
  // quality, and the Draft canvas says out loud that its 0.25 MP pass resolves
  // composition and timing but not faces. Surfaced where the user is choosing,
  // not in a tooltip.
  const noteEl = document.getElementById('h3TierNote');
  if (noteEl) {
    const n = cell.note || '';
    noteEl.textContent = n;
    noteEl.hidden = !n;
  }
  // The export line is per (canvas × target): switching quality changes whether
  // this canvas exports clean or padded, so it has to be re-read here too.
  _h3SyncExportNote();
  // Per-window prompts exist only for a CHAINED length and there is one box per
  // window, so the control has to follow the Length axis live: 5s hides it, 10s
  // shows two boxes, 15s shows three (and the third is seeded, not blank).
  if (typeof renderH3WindowPrompts === 'function') {
    try { renderH3WindowPrompts(); } catch (e) {}
  }
  try {
    localStorage.setItem('phos_h3_quality', cell.quality);
    localStorage.setItem('phos_h3_length', cell.length);
    localStorage.setItem('phos_h3_tier', cell.key);
  } catch (e) {}
  renderH3Axes();
  // Turbo's pill carries THIS shape's estimate, so it has to be re-labelled
  // whenever either axis moves.
  if (typeof renderH3Turbo === 'function') { try { renderH3Turbo(); } catch (e) {} }
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
  if (typeof updateCustomizeSummary === 'function') { try { updateCustomizeSummary(); } catch (e) {} }
}

// ---- Per-window prompts -----------------------------------------------------
// A 10 s clip is TWO chained 5-second windows and a 15 s clip is three, and by
// default every window is handed the same prompt — which is why a one-off
// action reads as a loop even though the frames are genuinely unique. These
// five functions are the whole control; the runner has taken `--chain-prompts`
// since before v3.4.1 and nothing but the UI was missing.
//
// The default is untouched: with the toggle off the hidden field posts an empty
// string, make_job normalises that to [], and run_h3_job_inner takes exactly the
// argv it always did.

// How many windows the CURRENTLY SELECTED cell renders. Read off the server's
// cell, never computed here — the browser looks a cell up, it never invents one.
function h3ChainWindows() {
  const cell = h3CellFor(h3CurrentQuality(), h3CurrentLength());
  return Math.max(1, Number((cell && cell.chain_windows) || 1));
}

// Whether the INSTALLED runner can take a shot list at all. A SECOND probe, not
// a synonym for H3.chain: chaining and --chain-prompts landed on the runner at
// different times, so a pack exists that renders 10 s / 15 s and cannot be told
// what each window should do. That user keeps the honest warning on the cell
// (swapped in server-side) and never sees this control.
function h3ChainPromptsSupported() {
  return !!(H3 && H3.chain_prompts);
}

function _h3WinValues() {
  return Array.prototype.slice
    .call(document.querySelectorAll('#h3WinPromptsList .h3-win-ta'))
    .map(t => String(t.value || ''));
}

// The one place the hidden field is written. Anything that isn't a live, on,
// chained control posts '' — a stale list left over from a 15 s selection must
// never ride along on a 5 s render (make_job clamps it too, but the form should
// not be lying about what it is submitting).
function _h3SerializeWindowPrompts() {
  const inp = document.getElementById('h3_chain_prompts');
  if (!inp) return;
  const row = document.getElementById('h3WindowPromptsRow');
  const on = !!(document.getElementById('h3WinPromptsToggle') || {}).checked;
  if (!row || row.hidden || !on) { inp.value = ''; return; }
  const vals = _h3WinValues().slice(0, h3ChainWindows());
  // All-blank is the default by another name; post nothing rather than an array
  // of empty strings.
  inp.value = vals.some(v => v.trim()) ? JSON.stringify(vals) : '';
}

// Build (or rebuild) one labelled textarea per window. Existing text survives a
// rebuild — switching 10s → 15s keeps both beats and adds a third — and a box
// that did not exist before is seeded with the main prompt whenever the control
// is on, so the user is always editing a shot list rather than staring at a
// blank box wondering what belongs in it.
function renderH3WindowPrompts() {
  const row = document.getElementById('h3WindowPromptsRow');
  const list = document.getElementById('h3WinPromptsList');
  const toggle = document.getElementById('h3WinPromptsToggle');
  if (!row || !list || !toggle) return;
  if (!toggle.dataset.init) {
    toggle.dataset.init = '1';
    let saved = null;
    try { saved = localStorage.getItem('phos_h3_winprompts'); } catch (e) {}
    toggle.checked = (saved === '1');
    toggle.addEventListener('change', () => toggleH3WindowPrompts(toggle.checked));
  }
  const help = document.getElementById('h3WinHelpNote');
  if (help && H3 && H3.chain_prompt_help && !help.textContent) {
    help.textContent = H3.chain_prompt_help;
  }
  const windows = h3ChainWindows();
  const engineIsH3 = (((document.getElementById('engine') || {}).value) === 'h3');
  // Three conditions, and each hides it for a different honest reason: a
  // non-H3 engine has no windows, a 3 s / 5 s length is ONE window (the box
  // would change nothing), and a pack without --chain-prompts cannot use one.
  const show = engineIsH3 && windows > 1 && h3ChainPromptsSupported();
  row.hidden = !show;
  if (!show) { _h3SerializeWindowPrompts(); return; }
  const hint = document.getElementById('h3WinPromptsHint');
  if (hint) hint.textContent = windows + ' × 5s windows';
  const prev = _h3WinValues();
  const main = String((document.getElementById('prompt') || {}).value || '');
  const parts = [];
  for (let i = 0; i < windows; i++) {
    const from = i * 5;
    const v = (prev[i] != null) ? prev[i] : (toggle.checked ? main : '');
    parts.push(
      '<div class="h3-win">'
      + '<div class="h3-win-label"><span>Window ' + (i + 1) + '</span>'
      + '<span class="h3-win-span">' + from + '–' + (from + 5) + 's</span></div>'
      + '<textarea class="h3-win-ta" data-win="' + (i + 1) + '" rows="2"'
      + ' placeholder="Beat ' + (i + 1) + ' — leave empty to use the main prompt">'
      + escapeHtml(v) + '</textarea></div>');
  }
  list.innerHTML = parts.join('');
  list.hidden = !toggle.checked;
  list.querySelectorAll('.h3-win-ta').forEach(t => {
    t.addEventListener('input', _h3SerializeWindowPrompts);
  });
  _h3SerializeWindowPrompts();
}

// The toggle. Turning it ON seeds every EMPTY box with the main prompt: the
// user should be editing what they already wrote, not retyping it three times.
function toggleH3WindowPrompts(on) {
  const toggle = document.getElementById('h3WinPromptsToggle');
  const list = document.getElementById('h3WinPromptsList');
  if (!toggle) return;
  toggle.checked = !!on;
  if (toggle.checked) {
    const main = String((document.getElementById('prompt') || {}).value || '');
    document.querySelectorAll('#h3WinPromptsList .h3-win-ta').forEach(t => {
      if (!String(t.value || '').trim()) t.value = main;
    });
  }
  if (list) list.hidden = !toggle.checked;
  try {
    localStorage.setItem('phos_h3_winprompts', toggle.checked ? '1' : '0');
  } catch (e) {}
  _h3SerializeWindowPrompts();
}

// The `?`. Reveals the sentence in place rather than in a title= tooltip, and
// says so to assistive tech via aria-expanded.
function toggleH3WindowHelp() {
  const btn = document.getElementById('h3WinHelpBtn');
  const note = document.getElementById('h3WinHelpNote');
  if (!btn || !note) return;
  if (!note.textContent && H3 && H3.chain_prompt_help) {
    note.textContent = H3.chain_prompt_help;
  }
  const open = !!note.hidden;
  note.hidden = !open;
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// Restore path: a sidecar's raw `params.h3_chain_prompts` (blanks included) →
// the boxes. Used by Load Params and Draft→Finish. MUST run after the shape has
// been applied, because the number of boxes is a property of the length.
function setH3ChainPrompts(listValue) {
  const toggle = document.getElementById('h3WinPromptsToggle');
  const list = document.getElementById('h3WinPromptsList');
  if (!toggle) return;
  const arr = Array.isArray(listValue)
    ? listValue.map(x => String(x == null ? '' : x))
    : [];
  const on = arr.some(v => v.trim());
  toggle.checked = on;
  renderH3WindowPrompts();
  Array.prototype.slice
    .call(document.querySelectorAll('#h3WinPromptsList .h3-win-ta'))
    .forEach((t, i) => { t.value = (arr[i] != null) ? arr[i] : ''; });
  if (list) list.hidden = !on;
  _h3SerializeWindowPrompts();
}

// Kept as the single-key entry point, because a tier key is what every SIDECAR
// carries: Load Params, Draft→Finish, setQuality's H3 re-assertion and the boot
// path all hand this a key (possibly a legacy one) and expect the form to land
// on that shape. It decomposes into the two axes and delegates.
function setH3Tier(key) {
  const tier = h3TierByKey(key);
  if (!tier) return;
  // fallback:true — every caller of this is a RESTORE (boot, Load Params,
  // Finish, setQuality's H3 re-assertion), and a restore that lands on a shape
  // this install can't serve must degrade, not leave the form armed with it.
  _h3ApplyShape(tier.quality, tier.length, { fallback: true });
}


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  engineById, defaultEngine, engineStatus, engineServesMode,
  h3ResolveTierKey, h3TierByKey, h3CellFor, h3CurrentCell,
  setH3Upscale, setH3Orientation, setH3Steps, h3TurboPillSub,
  renderH3Turbo, setH3Turbo, syncModeStripToEngine, renderEngineSwitch,
  renderTierAxes, renderH3Axes, ltxCellFor, ltxCurrentQuality,
  ltxCurrentLength, ltxCurrentCell, ltxCellEta, ltxCellNeedsInstall,
  ltxCellInstallLabel, _ltxApplyShape, setH3Quality, _h3ApplyShape,
  renderH3WindowPrompts, toggleH3WindowHelp, setH3ChainPrompts, setH3Tier,
});
