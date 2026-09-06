// webapp/js/queue.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ============================================================================
// Draft → Finish — re-render this clip at a HIGHER QUALITY, same length
// ============================================================================
// Under fixed tiers "Finish" meant "pick another tier", and a tier bundled the
// canvas WITH the duration — so finishing a 3 s draft at "HQ 5s" silently made
// the clip two seconds longer, and finishing a 10 s draft was not expressible
// at all. With the two axes it means exactly one thing:
//
//     SAME LENGTH. HIGHER QUALITY.
//
// The length is the shot the user judged; changing it would not be a finish, it
// would be a different clip. The canvas is the thing they deferred paying for.
// So the picker lists the CANVASES above the clip's own, at the clip's own
// length, and the button is offered on any clip that has a rung above it — a
// Standard 10 s clip can now be committed to High 10 s, which the old
// draft-only gate had no way to express.
//
// Everything else about the mechanism is unchanged, and has to be: the finish
// render inherits the draft's ACTUAL seed (sidecar `seed_used`, never the `-1`
// the user submitted) along with the prompt, the first frame, the export canvas,
// Turbo and any pinned step count. Change any of those and the user gets a
// different shot, which would make the button a lie.
//
// Mechanically this is still the Load Params pattern: read the clip's sidecar,
// restore it into #genForm, and let the ONE submit path run (it owns the
// double-submit guard, the LoRA-orphan check and the prompt modifiers). The only
// divergences are that the quality is swapped on the way in, and that the form
// is submitted for the user instead of being left for them to press Generate —
// "iterate cheap, then commit" is a single decision, and the restored form is
// still sitting there afterwards to tweak and re-run.

// A cell by EXACT key, legacy keys resolved. h3TierByKey() falls back to the
// first cell when the key is unknown, which is right for a picker but wrong
// here: a clip rendered at a shape this install no longer offers
// (LTX_H3_DENSE_10S turned back off, an older pack) must read as "nothing to
// offer", not as Draft 3s.
function h3TierByKeyExact(key) {
  if (!key) return null;
  const k = h3ResolveTierKey(key);
  return (H3.tiers || []).find(t => t.key === k) || null;
}

// The cells a clip can be finished at: the SAME length, at every canvas above
// the one it was rendered on, that this install can actually render. Server-side
// H3_TIERS stays the single source of truth — this filters, it never invents a
// shape or an eta.
function h3FinishTargets(srcCell) {
  if (!srcCell) return [];
  const order = (H3.qualities || []).map(q => q.key);
  const from = order.indexOf(srcCell.quality);
  if (from < 0) return [];
  return order.slice(from + 1)
    .map(q => h3CellFor(q, srcCell.length))
    .filter(c => c && c.available !== false);
}

// The user's chosen finish CANVAS, sanity-checked against what this install can
// actually render at this clip's length. Order of preference: their persisted
// choice, the server's pin if it has one, then one rung up — which is the same
// instinct the old `hq_5s` default encoded (the cheap next step, never a
// surprise commitment to the most expensive thing on the table).
function h3FinishTierKey(srcCell) {
  const targets = h3FinishTargets(srcCell);
  if (!targets.length) return null;
  let saved = null;
  try { saved = localStorage.getItem('phos_h3_finish_quality'); } catch (e) {}
  for (const q of [saved, H3.finish_quality_default]) {
    if (!q) continue;
    const hit = targets.find(t => t.quality === q);
    if (hit) return hit.key;
  }
  return targets[0].key;   // one rung up
}

function h3FinishSetTier(key) {
  const cell = h3TierByKeyExact(key);
  if (!cell) return;
  try { localStorage.setItem('phos_h3_finish_quality', cell.quality); } catch (e) {}
  // Re-label in place. The clip stays selected — picking a quality is choosing
  // what the button will do, not doing it.
  _syncH3FinishAffordance(findOutputByPath(activePath));
}

// PURE. Given a completed H3 render's sidecar `params` and the tier to finish
// at, return the exact #genForm field→value map that reproduces that clip at
// the new tier — or null when the sidecar isn't a finishable H3 draft.
//
// No DOM, no globals: this is the piece that has to be RIGHT (it is where the
// seed carry-over lives), so it is kept callable in isolation and unit-tested
// against real sidecar shapes.
//
// Every key it emits is in the make_job allowlist — engine, h3_tier,
// h3_quality, h3_length, h3_upscale, h3_steps, mode, prompt, negative_prompt,
// seed, image. A field that isn't in that dict silently no-ops on /queue/add,
// which is the known trap in this codebase; adding a key here means adding it
// there too.
function h3FinishFieldsFromSidecar(p, tierKey) {
  if (!p || typeof p !== 'object') return null;
  if (p.engine !== 'h3') return null;
  if (!tierKey) return null;
  // The target may be given as a legacy key by an old caller; resolve it, and
  // emit the two axes alongside it so make_job takes the axis path (which wins)
  // rather than re-deriving them from the string.
  const target = h3TierByKeyExact(tierKey);
  if (!target) return null;
  // seed_used is the integer the H3 path resolved and recorded at render time.
  // `seed` is what the user SUBMITTED and is '-1' whenever they left it random,
  // so preferring it would hand the finish render a fresh roll — the exact bug
  // the Manual loadParams fix (b024bb5) had to correct. Same contract here.
  const seedRaw = (p.seed_used != null && String(p.seed_used) !== ''
                   && String(p.seed_used) !== '-1')
    ? p.seed_used
    : p.seed;
  const seed = (seedRaw == null || String(seedRaw) === '') ? '-1' : String(seedRaw);
  // H3 serves t2v and i2v only; anything else in the sidecar is a corrupt or
  // hand-edited file, and t2v is the safe read (a bogus mode would be snapped
  // back server-side anyway, but then the first frame would be silently lost).
  const mode = (p.mode === 'i2v') ? 'i2v' : 't2v';
  const fields = {
    mode: mode,
    engine: 'h3',
    h3_tier: target.key,
    h3_quality: target.quality,
    h3_length: target.length,
    prompt: String(p.prompt || ''),
    negative_prompt: String(p.negative_prompt || ''),
    seed: seed,
    // Export canvas: '' lets the caller keep whatever the panel has, which
    // matters for sidecars written before h3_upscale existed.
    h3_upscale: (typeof p.h3_upscale === 'string' && p.h3_upscale) ? p.h3_upscale : '',
    // The sidecar stores the resolved override as an int, 0 = "the cell's own
    // count". The form's pills speak 'auto' | '12' | '16' | '20'; anything
    // outside that set (an older sidecar, a curl'd job) reads as auto so the
    // target canvas's tuned count is used rather than a depth no pill can show.
    h3_steps: (['12', '16', '20'].indexOf(String(p.h3_steps)) !== -1)
      ? String(p.h3_steps) : 'auto',
    // Turbo carries over: a draft judged with the 4-step sampler should be
    // finished with it too, or the Finish render is a different recipe as well
    // as a different canvas. setH3Turbo re-checks availability, so a sidecar
    // from an install that has since lost the files lands on Standard, not on a
    // mode that would fail at queue time.
    h3_turbo: !!p.h3_turbo,
    // The shot list carries over, and it is safe to: Finish means SAME LENGTH,
    // higher quality, so the window count is identical by construction and
    // entry i still means window i. Dropping it would silently finish a
    // two-beat draft as the same beat twice — the exact bug this control fixes.
    h3_chain_prompts: Array.isArray(p.h3_chain_prompts)
      ? p.h3_chain_prompts.map(x => String(x == null ? '' : x)) : [],
    // First frame. i2v without one is not renderable, so it is carried
    // verbatim; on t2v it is deliberately empty — H3 ignores it and leaving a
    // stale path in the picker would misrepresent what was queued.
    image: (mode === 'i2v' && typeof p.image === 'string') ? p.image : '',
  };
  if (mode === 'i2v' && !fields.image) return null;
  return fields;
}

// Show/hide + re-label the Finish control for the selected output. Called from
// selectOutput on every selection. Reads o.engine / o.h3_tier, which
// list_outputs() lifts out of the sidecar read it already performs — no fetch
// on the selection path.
function _syncH3FinishAffordance(o) {
  const wrap = document.getElementById('h3FinishWrap');
  if (!wrap) return;
  const label = document.getElementById('h3FinishLabel');
  const btn = document.getElementById('h3FinishBtn');
  const sel = document.getElementById('h3FinishTier');
  const srcTier = (o && o.engine === 'h3') ? h3TierByKeyExact(o.h3_tier) : null;
  const targetKey = srcTier ? h3FinishTierKey(srcTier) : null;
  const target = targetKey ? h3TierByKeyExact(targetKey) : null;
  // Two ways to be uninteresting: not an H3 clip, or an H3 clip already on the
  // top canvas (nothing above it to finish AT). Note the gate is no longer
  // `draft` — Finish means "same length, higher quality", and a Standard 10 s
  // clip has a higher quality to go to just as much as a Draft 3 s one does.
  if (!srcTier || !target) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = '';
  // The eta is the cell's own string from the server — never a number computed
  // here, so the measured wall times stay in one place. The cell's label has
  // its own "·" ("High · 10s"), which is flattened to "High 10s" so the one dot
  // left separates the shape from its cost.
  const name = String(target.label).replace(/\s*·\s*/g, ' ').trim();
  if (label) label.textContent = `Finish at ${name} · ${target.eta}`;
  if (btn) {
    btn.title = `Re-render this ${srcTier.quality_label} clip at `
              + `${target.label} (${target.spec}, ${target.eta}) — same `
              + `${target.length_label} length, same prompt, same seed, same `
              + `first frame`;
  }
  if (sel) {
    const opts = h3FinishTargets(srcTier)
      .map(t => `<option value="${escapeHtml(t.key)}"${t.key === target.key ? ' selected' : ''}>`
              + `${escapeHtml(t.label)} · ${escapeHtml(t.eta)}</option>`)
      .join('');
    // Only rebuild when the option set actually changed — this runs on every
    // gallery click and blowing away the <select> each time would fight the
    // native dropdown if it happened to be open.
    if (sel.dataset.built !== opts) {
      sel.innerHTML = opts;
      sel.dataset.built = opts;
    }
    sel.value = target.key;
  }
}

// Commit the selected clip at a higher quality. Restores the sidecar into the
// form (Load Params pattern) with ONLY the quality swapped, then submits through
// #genForm's own handler.
async function h3FinishActive() {
  if (!activePath) return;
  const o = findOutputByPath(activePath);
  const srcTier = (o && o.engine === 'h3') ? h3TierByKeyExact(o.h3_tier) : null;
  if (!srcTier) return;                     // button shouldn't be visible
  const targetKey = h3FinishTierKey(srcTier);
  const target = targetKey ? h3TierByKeyExact(targetKey) : null;
  if (!target) return;

  let p = null;
  try {
    const r = await fetch('/sidecar?path=' + encodeURIComponent(activePath));
    if (!r.ok) throw new Error('no sidecar (older output?)');
    const data = await r.json();
    p = data && data.params;
  } catch (e) {
    if (typeof phosToast === 'function') {
      phosToast('Finish: ' + (e.message || 'failed to read sidecar'), { kind: 'danger' });
    }
    return;
  }
  const fields = h3FinishFieldsFromSidecar(p, target.key);
  if (!fields) {
    if (typeof phosToast === 'function') {
      phosToast('Finish: this clip\'s sidecar is missing what the re-render needs.',
                { kind: 'danger' });
    }
    return;
  }

  // ---- restore into the form -------------------------------------------
  // Leave Image Studio / Train first, or setMode alone looks like a no-op
  // (body[data-workflow] keeps #genForm hidden) — same reason animateFromPhoto
  // calls this.
  if (typeof workflowSwitch === 'function') { try { workflowSwitch('manual'); } catch (e) {} }
  setMode(fields.mode);
  if (fields.mode === 'i2v') {
    // setMode('i2v') copies the #i2vMode SELECT into the hidden #mode, and
    // that select may still be sitting on 'i2v_clean_audio' from an earlier
    // LTX render. H3 doesn't serve that mode — make_job would silently demote
    // the job to LTX and render the wrong engine at H3 geometry. Pin both,
    // same as loadParams does when restoring an i2v sidecar.
    const i2vSel = document.getElementById('i2vMode');
    if (i2vSel) i2vSel.value = 'i2v';
    document.getElementById('mode').value = 'i2v';
  }
  // setMode() runs _syncEngineForMode(), which re-applies the PERSISTED
  // engine — so the engine has to be forced AFTER it, not before. setEngine
  // re-runs every gate (RAM, install, mode) and returns what it actually
  // landed on; if that isn't h3 the job would silently queue on LTX at H3
  // geometry, so bail loudly instead.
  const engine = (typeof setEngine === 'function') ? setEngine('h3') : 'ltx';
  if (engine !== 'h3') {
    const note = (document.getElementById('engineRowNote') || {}).textContent || '';
    if (typeof phosToast === 'function') {
      phosToast('Finish needs the Hailuo H3 engine. ' + note, { kind: 'danger' });
    }
    return;
  }
  document.getElementById('prompt').value = fields.prompt;
  document.getElementById('negative_prompt').value = fields.negative_prompt;
  if (typeof syncAvoidRowFromValue === 'function') {
    try { syncAvoidRowFromValue(); } catch (e) {}
  }
  if (fields.image) {
    // pickerSetImage keeps the preview tile + recent-strip selection in sync
    // with the hidden input. snapAspect:false because the H3 cell owns the
    // geometry (setH3Tier below stamps width/height/frames).
    if (typeof pickerSetImage === 'function') {
      pickerSetImage('image', fields.image, { snapAspect: false });
    } else {
      document.getElementById('image').value = fields.image;
    }
  }
  if (fields.h3_upscale && typeof setH3Upscale === 'function') setH3Upscale(fields.h3_upscale);
  if (typeof setH3Steps === 'function') setH3Steps(fields.h3_steps);
  // Turbo before the shape: setH3Turbo forces the Steps pills back to 'auto',
  // and setH3Tier below re-reads them when it stamps the resolved count.
  if (typeof setH3Turbo === 'function') setH3Turbo(!!fields.h3_turbo);
  // Shape LAST of the H3 controls: setH3Tier stamps width/height/frames/steps
  // from the cell table, so anything geometry-related set after it would be
  // fighting the source of truth. The LENGTH inside `fields.h3_tier` is the
  // clip's own — only the quality moved.
  setH3Tier(fields.h3_tier);
  // And the shot list after the shape, for the same reason it comes after in
  // loadParams: the box count is a property of the length.
  if (typeof setH3ChainPrompts === 'function') {
    try { setH3ChainPrompts(fields.h3_chain_prompts); } catch (e) {}
  }
  // Seed after the shape for the same reason it comes last in loadParams:
  // nothing downstream may quietly re-randomise it.
  document.getElementById('seed').value = fields.seed;
  if (typeof updateCustomizeSummary === 'function') { try { updateCustomizeSummary(); } catch (e) {} }
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
  const formPane = document.querySelector('aside.form-pane');
  if (formPane) formPane.scrollTop = 0;

  // ---- queue it ---------------------------------------------------------
  // requestSubmit(), not submit(): the latter bypasses the submit listener
  // that owns the double-click guard and the prompt modifiers.
  const form = document.getElementById('genForm');
  if (!form || typeof form.requestSubmit !== 'function') {
    if (typeof phosToast === 'function') {
      phosToast('Finish: form not ready — press Generate.', { kind: 'danger' });
    }
    return;
  }
  if (typeof phosToast === 'function') {
    phosToast(`Finishing at ${target.label} · ${target.eta} · seed ${fields.seed}`,
              { kind: 'success' });
  }
  form.requestSubmit();
}

// Modes H3 can serve. Thin wrapper over the registry so the H3-specific code
// below (and anything else that grew to call this) keeps working unchanged.
function _h3ServesMode(mode) {
  return engineServesMode(engineById('h3'), mode);
}

function setEngine(engine, opts) {
  opts = opts || {};
  const fallback = defaultEngine();
  const note = document.getElementById('engineRowNote');
  let e = engineById(engine);
  let target = e ? e.id : fallback.id;
  let reason = '';

  // ---- gates ---------------------------------------------------------------
  // Every one of these is re-run server-side in make_job. This copy exists so
  // the UI can say WHY, not so it can decide.
  if (e && !e.builtin) {
    const st = engineStatus(e);
    if (st.announced) {
      target = fallback.id;
      reason = e.label + " isn't released yet.";
    } else if (!st.capable) {
      target = fallback.id;
      // The lowest lane's floor, not the bf16 one and not a literal 64.
      reason = e.label + ' needs '
             + (st.ram_floor_gb || st.min_ram_gb || 46) + ' GB unified memory.';
    } else if (!st.available) {
      target = fallback.id;
      // Distinguish "you never installed this" from "you DID install this and
      // something broke it". Telling a user with 75 GB of H3 weights on disk
      // that H3 "isn't installed" is the v3.4.0 regression report, verbatim.
      // A third case joined them: the Mac whose only gap is the local Q8
      // engine build. The server writes that sentence; do not paraphrase it.
      reason = st.needs_q8_dit
        ? (st.ram_note || (e.label + ' needs its low-RAM engine built.'))
        : st.repairable
        ? e.label + ' needs repair — your weights are still on disk. Click the chip.'
        : e.label + " isn't installed yet.";
    } else if (!engineServesMode(e, currentMode)) {
      target = fallback.id;
      reason = e.label + ' renders ' + (e.serves_label || 'other modes')
             + ' only — back on ' + fallback.label + ' for this mode.';
    } else if (currentMode === 'i2v' && st.first_frame === false) {
      target = fallback.id;
      reason = 'This ' + e.label + ' build has no first-frame support — '
             + 'update the pack to use Image mode.';
    }
  }
  if (target !== (e || {}).id) e = engineById(target) || fallback;

  const inp = document.getElementById('engine');
  if (inp) inp.value = target;
  // The fold rules (`body:not([data-engine="ltx"]) [data-ltx-only]`) and the
  // --eng-* accent variables both key off this one attribute, and both are
  // emitted from the same ENGINES table that produced `target`.
  document.body.dataset.engine = target;
  if (note) { note.textContent = reason; note.hidden = !reason; }
  renderEngineSwitch();
  try { syncModeStripToEngine(); } catch (e) {}

  // ---- surface swap --------------------------------------------------------
  // Each engine's registry row NAMES the elements it owns, so this is generic:
  // hide every engine's strip and standing hint, then reveal the active one's.
  // An engine with no `strip` (LTX) is left alone — its strip visibility
  // belongs to _applyCharacterQualityStripVisibility, which has to choose
  // between the default and character strips.
  ENGINES.forEach(x => {
    if (x.strip) {
      const s = document.getElementById(x.strip);
      if (s) s.hidden = (x.id !== target);
    }
    if (x.hint) {
      const h = document.getElementById(x.hint);
      if (h) h.hidden = (x.id !== target);
    }
  });
  const qLabel = document.getElementById('qualityLabelName');
  if (qLabel) qLabel.textContent = e.strip_label || 'Quality';
  if (target === 'h3') {
    renderH3Axes();
    setH3Tier((document.getElementById('h3_tier') || {}).value || H3.default_tier);
    setH3Upscale((document.getElementById('h3_upscale') || {}).value
                 || H3.default_upscale || 'fit_720p');
    setH3Steps((document.getElementById('h3_steps') || {}).value || 'auto');
    renderH3Turbo();
    setH3Turbo((document.getElementById('h3_turbo') || {}).value === '1');
    // After the shape, because the number of window boxes IS the shape.
    renderH3WindowPrompts();
    // LTX post-processing doesn't run on an H3 render (make_job neutralises
    // all three server-side). Mirror that in the UI or the derived line lies:
    // it was reading "768×448 → 1280×720 fit" for a render that ships 768×448.
    if (typeof setUpscale === 'function') { try { setUpscale('off'); } catch (e) {} }
    if (typeof setAccel === 'function') { try { setAccel('off'); } catch (e) {} }
    if (typeof setTemporalMode === 'function') { try { setTemporalMode('native'); } catch (e) {} }
  } else {
    // Coming back from H3: its frame counts live on the 17n+5 grid (124, 243),
    // which LTX rejects — it needs 8k+1. Snap on the way out so the field the
    // user is now looking at is a value LTX will actually accept, and the
    // bound Duration stays truthful.
    if (typeof snapFramesTo8kPlus1 === 'function') {
      try { snapFramesTo8kPlus1(); } catch (e) {}
    }
    // Give the active quality preset its upscale back (H3 forced it off).
    if (typeof setUpscale === 'function' && typeof QUALITY_PRESETS === 'object') {
      const _qp = QUALITY_PRESETS[(document.getElementById('quality') || {}).value];
      if (_qp) { try { setUpscale(_qp.upscale || 'off'); } catch (e) {} }
    }
    // LTX's own two strips, the mirror of the renderH3Axes() call above.
    if (typeof renderTierAxes === 'function') {
      try { renderTierAxes('ltx'); } catch (e) {}
    }
    if (typeof _applyCharacterQualityStripVisibility === 'function') {
      // Restore whichever LTX strip the current selection calls for.
      try { _applyCharacterQualityStripVisibility(); } catch (e) {}
    }
  }
  if (opts.persist !== false) {
    try { localStorage.setItem(H3_ENGINE_LS_KEY, target); } catch (e) {}
  }
  if (typeof updatePromptPlaceholder === 'function') {
    try { updatePromptPlaceholder(); } catch (e) {}
  }
  try { _syncEnginePromptTools(); } catch (e) {}
  // The LoRA picker is SHARED by the video engines and each has its own
  // library, so an engine switch re-points it, re-filters it and re-serializes
  // the hidden field (the lane guard). Last, after the surface swap, so it
  // reads the engine this call actually settled on rather than the one that
  // was requested — a gate may have bounced it back to the built-in.
  try { _syncLoraPickerForEngine(); } catch (e) {}
  // A take's estimate is per engine (parts on H3, a chain on LTX).
  try { if (typeof takeRefresh === 'function') takeRefresh(); } catch (e) {}
  return target;
}

// The composer tools that SURVIVE an engine switch but change meaning across
// it. Everything else in that strip is folded by a data-<engine>-only rule and
// needs no JS at all — this exists only for the one control whose mechanism
// differs per engine rather than existing on one engine.
//
// "No music": both engines volunteer a score and neither lets you strip one
// afterwards, so the control is right on both. What it DOES differs — LTX gets
// a prose audio directive, H3 gets `non_diegetic_music: N/A`, its trained field
// value (H3_PROMPTING_GUIDE §2.5). The tooltip has to follow the mechanism or
// it describes a render that isn't happening; both strings live on the element
// as data-title-* so the copy stays in the markup with the control.
function _syncEnginePromptTools() {
  const eng = (document.body.dataset.engine || ENGINE_DEFAULT);
  const pill = document.getElementById('noMusicPill');
  if (pill) {
    const t = pill.getAttribute('data-title-' + eng)
           || pill.getAttribute('data-title-ltx');
    if (t) pill.title = t;
  }
}

function currentEngine() {
  return (document.getElementById('engine') || {}).value || ENGINE_DEFAULT;
}

// Called from setMode(). Re-applies the user's PERSISTED engine choice rather
// than whatever is currently selected, so the snap-back is temporary: flipping
// Text → FFLF drops to the built-in engine with a note, and flipping back to
// Text restores the user's choice instead of silently leaving them on the
// other engine. setEngine() re-runs every gate, so an unsupported mode still
// lands on the built-in. An engine that has since left the registry (an old
// localStorage value, a preview flag turned back off) falls through to the
// default rather than resolving to nothing.
function _syncEngineForMode() {
  let want = null;
  try { want = localStorage.getItem(H3_ENGINE_LS_KEY); } catch (e) {}
  setEngine(engineById(want) ? want : ENGINE_DEFAULT, { persist: false });
}

// /status carries a fresh h3 block every tick, so an install finishing in the
// Pinokio sidebar unlocks the engine without a panel restart (same contract
// the Q8 download already has with the High pill).
function updateH3Availability(s) {
  const next = s && s.h3;
  if (!next) return;
  const changed = (next.available !== H3.available)
               || (next.capable !== H3.capable)
               || (next.first_frame !== H3.first_frame)
               // `chain` gates the 10 s / 15 s tiers, so the strip has to be
               // re-rendered when an H3 pack update brings --chain-windows in.
               || (next.chain !== H3.chain)
               // `chain_prompts` gates the Per-window prompts control AND the
               // sentence on the chained cells, so a pack update that brings
               // --chain-prompts in has to re-render both without a reload.
               || (next.chain_prompts !== H3.chain_prompts)
               // Install→repair→install flips the pill's copy and the models
               // card even when `available` itself hasn't moved yet.
               || (next.repairable !== H3.repairable)
               || (next.reason !== H3.reason)
               // Turbo's release adapter may arrive from inside the panel, so
               // this turns the dashed pill live as soon as it lands.
               || (((next.turbo || {}).available) !== ((H3.turbo || {}).available))
               || (((next.turbo || {}).supported) !== ((H3.turbo || {}).supported))
               || ((next.tiers || []).length !== (H3.tiers || []).length);
  // BOTH bindings, and they must stay the same object: the H3-specific code
  // reads `H3` on nearly every line, the registry reads the probe map. A
  // divergence here is a switcher showing "not installed" over a working
  // engine (or worse, the reverse).
  H3 = next;
  window._ENGINE_PROBES.h3 = next;
  if (changed) {
    setEngine(currentEngine(), { persist: false });
    // The Finish button's label and its picker are both derived from H3.tiers,
    // so a pack install/repair that changes the offered tiers has to re-label
    // the currently-selected clip's affordance too — otherwise it keeps
    // advertising a tier this install just stopped (or started) offering.
    if (typeof _syncH3FinishAffordance === 'function') {
      try { _syncH3FinishAffordance(findOutputByPath(activePath)); } catch (e) {}
    }
  }
}

// Install card — H3 is a Pinokio-script install (clone + venv + ~75 GB of
// weights), not an in-panel `hf download`, so the panel explains the one
// sidebar click rather than pretending it can do it itself. Same shape the
// Sharp/Qwen packs use.
//
// SHOWN ONCE, THEN GET OUT OF THE WAY. Session-scoped: the first click on the
// H3 segment earns the full explainer; every click after that gets the nudge
// below instead. A modal that re-pops identically on every click stops being
// an explanation and becomes a wall — and the one thing a user who just read
// it wants to do is click the engine again.
//
// `source` is 'chip' from the engine switcher and undefined from the explicit
// "How to install" / "How to repair" buttons in the Models list, which must
// ALWAYS open the card: the user asked for it by name there.
let _h3CardSeen = false;
function openH3InstallCard(source) {
  if (source === 'chip' && _h3CardSeen) { _h3NudgeEngineOffer(); return; }
  const m = document.getElementById('h3InstallModal');
  const body = document.getElementById('h3InstallBody');
  if (body) {
    const missing = (H3.missing || []);
    // REPAIR is a different story from INSTALL and has to read like one. A
    // user whose 75 GB of weights are still on disk must not be shown a "~75
    // GB download" card — that is what made the v3.4.0 report read as "H3
    // vanished and Reset didn't bring it back".
    // WHAT actually broke, from the probe's own missing-list — NOT from the
    // two-way venv_broken guess. venv_broken means "a venv was BUILT and its
    // interpreter dangles"; a venv that was never built at all is reason
    // missing_venv with venv_broken false, and the old two-way branch rendered
    // that as "the code checkout is missing" — issue #68, where a user with a
    // complete external-SSD install chased a phantom checkout problem while
    // the real gap (venv never built) sat folded away in a <details>.
    const _missing = (H3.missing || []).map(String);
    const _missVenv = _missing.some(m => m.includes('venv'));
    const _missRunner = _missing.some(m => m.includes('runner') || m.includes('scripts/'));
    const diagnosis = H3.venv_broken
        ? 'What broke: H3’s Python environment points at a moved or deleted '
          + 'interpreter. Rebuilding the environment takes about two minutes.'
        : _missVenv
        ? 'What broke: H3’s Python environment was never built (or was '
          + 'removed). Building it takes about two minutes and re-downloads '
          + 'nothing.'
        : _missRunner
        ? 'What broke: the engine’s code checkout is missing or incomplete. '
          + 'Re-cloning it takes about a minute.'
        : 'What broke: ' + (_missing[0] || 'a component the probe lists below')
          + '.';
    const intro = H3.repairable ? `
      <p style="margin:0 0 10px">
        <b>Hailuo H3 is installed — it just needs repairing.</b> Your weights
        (~75 GB) are still on disk and are <em>not</em> re-downloaded.
      </p>
      <p style="margin:0 0 10px;color:var(--muted)">
        ${diagnosis}
      </p>
      <p style="margin:0 0 10px;color:var(--muted);font-size:12px">
        ${_missing.length ? 'Missing: ' + _missing.map(escapeHtml).join(' · ') : ''}
      </p>
      <p style="margin:0 0 10px">
        Fix it from Pinokio: open the <b>Phosphene</b> entry in the Pinokio
        sidebar and click <b>“Repair Hailuo H3”</b> (it appears in place of the
        install entry). The step is idempotent — it skips every weight already
        on disk. The panel picks the engine back up within a couple of seconds,
        no restart.
      </p>` : `
      <p style="margin:0 0 10px">
        <b>Hailuo H3 is Phosphene's second video engine</b>, a peer of LTX
        rather than an add-on to it: one prompt in, video <em>and</em> synced
        dialogue <em>and</em> sound out, in a single pass. It runs fully
        locally and sits beside LTX in the engine switcher — installing it
        changes nothing about your existing renders, and either engine can
        drive any render you start.
      </p>
      <p style="margin:0 0 10px;color:var(--muted)">
        ${escapeHtml(H3.size_note || '')}
      </p>
      <p style="margin:0 0 10px">
        Install it from Pinokio, not from here: open the <b>Phosphene</b> entry
        in the Pinokio sidebar and click
        <b>“Install Hailuo H3 (second video engine, ~75 GB)”</b>. The panel
        picks it up within a couple of seconds — no restart.
      </p>`;
    // THE DIAGNOSTIC DUMP GOES BEHIND A DISCLOSURE. `missing` is a list of raw
    // absolute paths — indispensable when an install half-lands, and pure
    // noise the other 99% of the time. Printing it under a two-sentence pitch
    // made the offer read like an error report. Kept verbatim, folded shut.
    body.innerHTML = intro + `
      ${missing.length ? `<details class="h3-diag">
        <summary>Details for troubleshooting</summary>
        <p>Currently missing: ${escapeHtml(missing.join('; '))}</p>
      </details>` : ''}`;
  }
  if (m) m.style.display = 'flex';
  _h3CardSeen = true;
}
function closeH3InstallCard() {
  const m = document.getElementById('h3InstallModal');
  if (m) m.style.display = 'none';
}

// The lighter affordance, for every click after the first. Two jobs:
//
//   1. POINT AT THE OFFER. Pulse the H3 segment itself, not only its badge —
//      the badge is `display:none` under 1500 px (the header runs out of room
//      on a laptop long before the desktop does), so a badge-only highlight
//      would be an invisible answer to a click on exactly the machines most
//      likely to be running this.
//   2. LEAVE A DOOR OPEN. The nudge carries its own link back to the full
//      explainer, so dismissing the modal once never costs the user the
//      explanation permanently.
//
// Rides the panel's own phosToast rather than a second notification surface —
// the storyboard's engine picker already answers this exact situation with one
// (sbSetEngineMode), and two toasts that look different for the same event is
// how a UI starts feeling assembled rather than designed. The reopen link is
// appended to the element phosToast hands back.
function _h3NudgeEngineOffer() {
  const seg = document.querySelector('.eng-seg[data-engine="h3"]');
  if (seg) {
    seg.classList.remove('offer-nudge');
    void seg.offsetWidth;            // restart the animation on a repeat click
    seg.classList.add('offer-nudge');
    setTimeout(() => seg.classList.remove('offer-nudge'), 1200);
  }
  // ONE LINE, and it has to FIT one: .phos-toast-msg is nowrap + ellipsis
  // inside a 480px cap, so a sentence that runs long is not a long toast, it
  // is a truncated one — "…install it from the Phosphene entry i…" was the
  // first draft, and it clipped exactly where the instruction lived.
  const el = phosToast('Hailuo H3 · 75 GB — install from the Pinokio sidebar.',
                       { icon: 'ph-info', duration: 5000 });
  if (!el) return;
  const a = document.createElement('a');
  a.href = '#';
  a.className = 'phos-toast-action';
  a.textContent = 'What it is';
  a.onclick = (ev) => { ev.preventDefault(); el.remove(); openH3InstallCard(); };
  el.appendChild(a);
}

// Prompt enhancement via Gemma — wraps the upstream CLI's `enhance`
// subcommand. Cold start ~12-15s (Gemma load), warm ~5s. Blocks the UI
// during the request (just the button — rest of the form stays usable).
async function enhancePrompt() {
  const ta = document.getElementById('prompt');
  const original = ta.value.trim();
  if (!original) { alert('Type a prompt before enhancing it.'); return; }
  const mode = (currentMode === 'i2v' || currentMode === 'keyframe' || currentMode === 'extend') ? 'i2v' : 't2v';
  const btn = document.getElementById('enhanceBtn');
  // Snapshot the FULL inner markup, not just text — the button carries
  // an inline ph-sparkle-fill SVG that textContent would strip.
  const originalLabel = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<svg class="ph" aria-hidden="true" style="margin-right:6px;vertical-align:-2px"><use href="#ph-sparkle-fill"/></svg>Loading Gemma… (~15s on cold start)';
  let res;
  try {
    // Collect trigger tokens to preserve case-exact through enhance.
    // Source: every active LoRA's trigger_words + the active character
    // trigger (if Character mode). Server unions these with any
    // matching character ids found in the raw prompt as a defense in
    // depth — see /prompt/enhance handler for the merge.
    const preserveTokens = [];
    if (typeof _activeLoras !== 'undefined' && Array.isArray(_activeLoras)) {
      for (const l of _activeLoras) {
        for (const t of (l.trigger_words || [])) {
          const s = String(t || '').trim();
          if (s) preserveTokens.push(s);
        }
      }
    }
    // The hidden field's element id is "characterIdInput" (its NAME is
    // "character_id"); getElementById('character_id') always returned null,
    // so the active character trigger was never preserved through Enhance.
    const charIdEl = document.getElementById('characterIdInput');
    if (charIdEl && charIdEl.value) preserveTokens.push(charIdEl.value);
    const fd = new URLSearchParams({ prompt: original, mode });
    if (preserveTokens.length) fd.set('preserve_tokens', JSON.stringify(preserveTokens));
    const r = await fetch('/prompt/enhance', { method: 'POST', body: fd });
    res = await r.json();
  } catch (e) {
    alert('Enhance request failed: ' + (e.message || e));
    btn.disabled = false; btn.innerHTML = originalLabel;
    return;
  }
  btn.disabled = false; btn.innerHTML = originalLabel;
  if (res.error) { alert('Enhance failed: ' + res.error); return; }
  // Show diff in a confirm so the user can decide whether to accept.
  const accept = confirm(
    `Original:\n${res.original}\n\nEnhanced:\n${res.enhanced}\n\nReplace your prompt with the enhanced version?`
  );
  if (accept) {
    ta.value = res.enhanced;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

// Extend duration: user types seconds, we convert to latent frames behind
// the scenes. Each latent = 8 video frames; at 24 fps that's 0.333 s.
// Round-up so the user gets at least the seconds they asked for.
//   seconds → latents: ceil(seconds * 24 / 8)
//   latents → actual seconds: latents * 8 / 24
// Hint line shows both numbers so the conversion isn't a black box.
function syncExtendDuration() {
  const secInput = document.getElementById('extend_seconds');
  const hidden = document.getElementById('extend_frames');
  const hint = document.getElementById('extendDurationHint');
  if (!secInput || !hidden || !hint) return;
  const seconds = parseFloat(secInput.value) || 0;
  const latents = Math.max(1, Math.ceil(seconds * 24 / 8));
  const actualSec = (latents * 8 / 24);
  hidden.value = String(latents);
  hint.textContent = `≈ ${actualSec.toFixed(2)} s of new content (${latents} latent frames × 8 video frames at 24 fps)`;
}
document.getElementById('extend_seconds').addEventListener('input', syncExtendDuration);
syncExtendDuration();   // initialize on load
document.getElementById('i2vMode').addEventListener('change', () => {
  document.getElementById('audioSection').classList.toggle('show', document.getElementById('i2vMode').value === 'i2v_clean_audio');
  if (currentMode === 'i2v') document.getElementById('mode').value = document.getElementById('i2vMode').value;
  updateTemporalAvailability();
  updateCustomizeSummary();
  updateDerived();
});

function applyAspect(key) {
  if (!ASPECTS[key]) return;
  document.getElementById('aspect').value = key;
  // Aspect controls dimensions only when the active preset has a choice
  // (Standard / High at 1280×704 vs 704×1280). Quick is fixed 4:3 and
  // ignores the aspect picker (the row is hidden in that state, so this
  // path normally won't fire — defensive in case of programmatic calls).
  const q = document.getElementById('quality').value;
  if (q === 'quick') return;
  const preset = QUALITY_PRESETS[q] || QUALITY_PRESETS['standard'];
  const vertical = (key === 'vertical');
  document.getElementById('width').value  = vertical ? preset.h : preset.w;
  document.getElementById('height').value = vertical ? preset.w : preset.h;
  updateCustomizeSummary();
  updateDerived();
}

// applyQuality is kept as a tiny shim — old call sites (mode switching,
// etc.) call it expecting "set steps for the active quality." The
// dimensions are now owned by setQuality / applyAspect.
function applyQuality() {
  const q = document.getElementById('quality').value;
  if (_qualityUsesHq(q)) {
    document.getElementById('steps').value = 18;
  } else {
    document.getElementById('steps').value = 8;       // every distilled canvas
  }
  updateCustomizeSummary();
  updateDerived();
}

function durationToFrames(s) {
  const k = Math.max(0, Math.round(s * FPS / 8));
  return k * 8 + 1;
}
function framesToDuration(f) { return ((f - 1) / FPS).toFixed(2); }

// LTX 2.3 requires frame counts in the form 1 + 8k (one keyframe + N
// VAE-temporal blocks of 8 frames each). Typing "100" or "240" wastes
// compute on partially-filled trailing latents — the pipeline rounds
// up internally but charges for the empty slots. Snap on blur to the
// nearest valid value below + 1 (so we never silently render *more*
// than the user asked for, only less or equal).
function snapFramesTo8kPlus1() {
  const el = document.getElementById('frames');
  if (!el) return;
  const v = parseInt(el.value) || 0;
  if (v < 1) { el.value = 9; return; }
  // FLOOR to the 8k+1 grid — never up. Math.round could snap 100 → 105,
  // which broke the "never render more than you asked for" promise the
  // hint makes; the server (make_job) floors the same way, so what the box
  // settles on is what the job records and what the engine delivers.
  const k = Math.max(1, Math.floor((v - 1) / 8));
  const snapped = k * 8 + 1;
  if (snapped !== v) {
    el.value = snapped;
    // Reflect the change in duration too, since they're bound.
    document.getElementById('duration').value = framesToDuration(snapped);
  }
}

function updateDerived() {
  // The per-window hint counts windows for the CURRENT length, so it has to
  // move when the length does, not only when the pill is clicked.
  if (typeof windowPromptsInput === 'function') { try { windowPromptsInput(); } catch (e) {} }
  const mode = document.getElementById('mode').value;
  const w = parseInt(document.getElementById('width').value || 0);
  const h = parseInt(document.getElementById('height').value || 0);
  const f = parseInt(document.getElementById('frames').value || 0);
  const dur = framesToDuration(f);

  const upscale = document.getElementById('upscale')?.value || 'off';
  let finalRes = `<strong>${w}×${h}</strong>`;
  if (upscale === 'fit_720p') {
    const tw = w >= h ? 1280 : 720;
    const th = w >= h ? 720 : 1280;
    finalRes = `${w}×${h} → <strong>${tw}×${th}</strong> fit`;
  } else if (upscale === 'x2') {
    finalRes = `${w}×${h} → <strong>${w * 2}×${h * 2}</strong>`;
  } else {
    let pw = w, ph = h;
    if (w === 704 && h % 16 === 0) pw = 720;
    if (h === 704 && w % 16 === 0) ph = 720;
    const padded = (pw !== w || ph !== h) && mode === 'i2v_clean_audio';
    finalRes = padded ? `${w}×${h} → <strong>${pw}×${ph}</strong>` : `<strong>${w}×${h}</strong>`;
  }
  const accel = document.getElementById('accel')?.value || 'off';
  const accelText = accel === 'off' ? '' : ` · ${accel === 'turbo' ? 'Turbo' : 'Boost'}`;
  const temporal = document.getElementById('temporal_mode')?.value || 'native';
  let temporalText = '';
  if (temporal === 'fps12_interp24') {
    const intervalSec = Math.max(0, (f - 1) / FPS);
    const sourceFrames = Math.max(1, Math.round(intervalSec * 12 / 8)) * 8 + 1;
    temporalText = ` · LTX ${sourceFrames}f @ 12fps → ${FPS}fps`;
  }

  document.getElementById('derived').innerHTML = `Duration <strong>${dur}s</strong> @ ${FPS}fps${temporalText} · ${finalRes} · Steps ${document.getElementById('steps').value}${accelText}`;

  // Compact derived line in the sticky action footer — same info, tighter
  // typography. Lets the user see what they're about to render WITHOUT
  // scrolling back to the customize disclosure. Mirrors the Customize
  // summary style (just dimensions + duration; full details stay in the
  // expanded Customize body).
  const derivedFooter = document.getElementById('derivedFooter');
  if (derivedFooter) {
    derivedFooter.innerHTML = `<strong>${dur}s</strong> · ${finalRes}${temporalText}${accelText}`;
  }
  // Also update the Quality strip's right-side meta line (e.g. "5s · 1024×576")
  // so the Quality picker block reads as a self-contained summary.
  const qualityMeta = document.getElementById('qualityMeta');
  if (qualityMeta) {
    const qBare = `${w}×${h}`;
    qualityMeta.textContent = `${dur}s · ${qBare}`;
  }

  const warns = [];
  // Hailuo H3 counts frames on a 17n+5 grid, not LTX's 8k+1, and its tiers
  // are fixed — so the 8k+1 nudge is not just irrelevant there, it's wrong
  // (it told the user 124 was a mistake when 124 is the HQ·5s tier). H3's
  // canvas rule is /32 (its runner's own grid, stamped from the tier cell
  // anyway); every LTX lane is TWO-STAGE and floors the canvas to /64 —
  // warning about /32 there let a 1000-wide request pass while the engine
  // rendered 960 with a sidecar claiming 1000 (the CUSTOMIZE audit's
  // "Width × Height LIES" row). Each warning states what WILL render;
  // make_job normalizes the job to the same numbers.
  const _h3Active = document.body.dataset.engine === 'h3';
  const _grid = _h3Active ? 32 : 64;
  if (w % _grid !== 0) {
    const eff = Math.max(_grid, Math.floor(w / _grid) * _grid);
    warns.push(_h3Active
      ? `Width ${w} isn't a multiple of 32 (closest ${Math.round(w / 32) * 32})`
      : `Width ${w} renders at ${eff} — the two-stage engine needs multiples of 64`);
  }
  if (h % _grid !== 0) {
    const eff = Math.max(_grid, Math.floor(h / _grid) * _grid);
    warns.push(_h3Active
      ? `Height ${h} isn't a multiple of 32 (closest ${Math.round(h / 32) * 32})`
      : `Height ${h} renders at ${eff} — the two-stage engine needs multiples of 64`);
  }
  if (!_h3Active && f > 1 && (f - 1) % 8 !== 0) {
    const eff = Math.max(1, Math.floor((f - 1) / 8) * 8 + 1);
    warns.push(`Frames snap down to ${eff} (the 8k+1 grid)`);
  }
  if (temporal === 'fps12_interp24') {
    warns.push('12→24fps is experimental; check dialogue lip-sync and fast motion');
  }
  const banner = document.getElementById('warnBanner');
  if (warns.length) { banner.innerHTML = '<svg class="ph" aria-hidden="true" style="margin-right:6px;vertical-align:-2px"><use href="#ph-warning-fill"/></svg>' + warns.join(' · '); banner.classList.add('show'); }
  else banner.classList.remove('show');

  // Mode-aware visibility
  const inI2V = mode === 'i2v' || mode === 'i2v_clean_audio';
  const inImageFlow = inI2V || currentMode === 'keyframe';
  document.getElementById('imageSection').classList.toggle('show', inI2V && currentMode !== 'keyframe');
  // The reference-use row lives inside that section and follows the same
  // mode question, plus the server's 2.5-only availability flag.
  if (typeof _applyI2vRefModeVisibility === 'function') {
    try { _applyI2vRefModeVisibility(); } catch (_) {}
  }
  document.getElementById('extendSection').classList.toggle('show', currentMode === 'extend');
  // Colorize (restore) shows its own source-video picker. Unlike Extend it
  // KEEPS the sizing + quick-metrics rows below (the source's own dims/length
  // drive the output, but the prompt + seed still apply).
  const _restoreSection = document.getElementById('restoreSection');
  if (_restoreSection) _restoreSection.classList.toggle('show', currentMode === 'restore');
  // Upscale ×2 — its own source picker + "keep the shot" slider. The source
  // drives dims (×2, capped) and length, so the sizing/quick-metrics rows
  // are hidden below, like Extend.
  const _upscaleSection = document.getElementById('upscaleSection');
  if (_upscaleSection) _upscaleSection.classList.toggle('show', currentMode === 'upscale');
  // Ingredients (multi-reference) — its own multi-image picker + action field.
  // Like Colorize it KEEPS the sizing/quick-metrics rows (frames apply; the
  // sheet drives the rest).
  const _ingredientsSection = document.getElementById('ingredientsSection');
  if (_ingredientsSection) _ingredientsSection.classList.toggle('show', currentMode === 'ingredients');
  // Control (Union) — its own control-video picker. Like Colorize it KEEPS the
  // sizing/quick-metrics rows (the control clip drives dims/length; prompt +
  // seed still apply).
  const _controlSection = document.getElementById('controlSection');
  if (_controlSection) _controlSection.classList.toggle('show', currentMode === 'control');
  document.getElementById('keyframeSection').classList.toggle('show', currentMode === 'keyframe');
  // Keyframe toggle row — visible only in keyframe mode
  const kfToggleRow = document.getElementById('kfToggleRow');
  if (kfToggleRow) kfToggleRow.style.display = currentMode === 'keyframe' ? '' : 'none';
  renderKeyframeDynamicSlots();
  if (currentMode === 'keyframe') {
    maybeScaleTouchedKeyframeTiming(window._kfTimingLastFrames, f);
  }
  syncKeyframeTiming();
  document.getElementById('sizingSection').classList.toggle('show', currentMode !== 'extend' && currentMode !== 'upscale');
  // quickMetricsRow (Duration / Frames / Seed) doesn't apply to Extend
  // (extend_seconds drives the new content; the source video provides
  // the rest) nor to Upscale ×2 (source dims ×2 + source length). Hide it
  // there, show otherwise.
  const qmr = document.getElementById('quickMetricsRow');
  if (qmr) qmr.classList.toggle('show', currentMode !== 'extend' && currentMode !== 'upscale');
  document.getElementById('audioSection').classList.toggle('show', mode === 'i2v_clean_audio');
  // I2V audio source picker (Advanced) — only relevant in I2V flow.
  // In T2V/Extend/FFLF the model generates audio jointly; there's nothing
  // to swap out, so the dropdown is just noise.
  const i2vAudioSec = document.getElementById('i2vAudioModeSection');
  if (i2vAudioSec) i2vAudioSec.classList.toggle('show', inI2V);
  // Width/height stays visible in image flows too. (Restored 2026-06-03: the
  // 2026-05-17 simplification hid it for I2V/FFLF, which cost users the custom
  // I2V sizing they relied on.) The image still drives the DEFAULT —
  // snapAspectToImage() auto-snaps aspect+dims on upload so the common case
  // can't cover-crop a 16:9 photo into 9:16 — but the inputs are now editable
  // so power users can set an exact custom size. Safe on low-RAM Macs: the
  // server-side per-tier clamp (tier_max_dim → make_job, ~line 6806) caps base
  // (<48 GB) I2V at 768 regardless of what's typed, so this can't push to swap.
  const dimsRow = document.getElementById('dimsRow');
  if (dimsRow) dimsRow.style.display = '';

  // Image previews are now part of the picker component itself — the
  // preview <img> + clear button live inside .picker-drop and are toggled
  // by pickerSetImage(). No per-mode preview management here anymore;
  // the old imagePreview / startImagePreview / endImagePreview elements
  // are gone.
}

['width','height','frames','duration'].forEach(id => {
  const el = document.getElementById(id);
  // THE OTHER DIRECTION OF THE 7-SECOND LIE. The chips write these fields
  // correctly; these fields never repainted the chips. Type 14 into Duration
  // and the Length strip kept showing `5s` / `121f` / `~2 min` over a form
  // that would render 337 frames — the exact defect ltxCurrentLength()'s own
  // comment predicts, arriving from the side nobody wired.
  const repaintTiers = () => {
    if (typeof renderTierAxes === 'function') {
      try { renderTierAxes('ltx'); } catch (e) {}
    }
  };
  if (id === 'duration') {
    el.addEventListener('input', e => { document.getElementById('frames').value = durationToFrames(parseFloat(e.target.value) || 0); updateDerived(); repaintTiers(); });
  } else if (id === 'frames') {
    el.addEventListener('input', e => { document.getElementById('duration').value = framesToDuration(parseInt(e.target.value) || 0); updateDerived(); repaintTiers(); });
    el.addEventListener('blur', () => { snapFramesTo8kPlus1(); updateDerived(); repaintTiers(); });
  } else {
    // width / height: also refresh the Customize summary so "custom" flags
    // appear/disappear as the user types away from the preset values.
    el.addEventListener('input', () => { updateCustomizeSummary(); updateDerived(); repaintTiers(); });
  }
});
document.getElementById('keyframe_mid_seconds')?.addEventListener('input', () => {
  window._kfMidTouched = true;
  window._kfTimingTouched.mid = true;
  syncKeyframeTiming();
});
['02', '04', '05'].forEach(key => {
  document.getElementById(`keyframe_${key}_seconds`)?.addEventListener('input', () => {
    window._kfTimingTouched[key] = true;
    syncKeyframeTiming();
  });
});
// Picker hidden inputs no longer take user input — their value changes
// via pickerSetImage(), which already calls updateDerived(). No per-input
// listeners needed.

// Auto-snap the aspect picker based on an image's actual dimensions.
// Avoids the 16:9-source-cropped-to-9:16-strip footgun.
function snapAspectToImage(path) {
  const probe = new Image();
  probe.onload = () => {
    const r = probe.naturalWidth / probe.naturalHeight;
    const target = r >= 1 ? 'landscape' : 'vertical';
    if (document.getElementById('aspect').value !== target) setAspect(target);
  };
  probe.src = '/image?path=' + encodeURIComponent(path);
}

// uploadImage() / uploadKeyframe() were replaced by the unified picker
// component (pickerUploadFile + refreshUploadsStrip). The /upload endpoint
// still drives the actual transfer; the only change is which JS calls it.

// ====== Image picker component ======
// One implementation, five call sites: I2V image, keyframe start/mid/end,
// and A2V a2v_image. Each picker carries a `key` (the hidden
// field's name); every DOM element it owns is suffixed with `_<key>` so
// we can wire listeners by lookup instead of a per-instance closure.
// `a2v_image` was added 2026-05-20 to replace the old bespoke
// studio-ref-slot in the Audio-to-Video tab with the same drop +
// preview + clear + recent-uploads-strip surface every other mode uses.
globalThis.PICKERS = [
  'image',
  'start_image',
  'keyframe_02_image',
  'mid_image',
  'keyframe_04_image',
  'keyframe_05_image',
  'end_image',
  'a2v_image',
];

function pickerEls(key) {
  return {
    drop:    document.getElementById(`picker_drop_${key}`),
    file:    document.getElementById(`picker_file_${key}`),
    hidden:  document.getElementById(key),
    preview: document.getElementById(`picker_preview_${key}`),
    clear:   document.getElementById(`picker_clear_${key}`),
    empty:   document.querySelector(`#picker_drop_${key} .picker-empty`),
    recentWrap:  document.getElementById(`picker_recent_${key}_wrap`),
    recentStrip: document.getElementById(`picker_recent_${key}`),
  };
}

function pickerSetImage(key, path, opts = {}) {
  const els = pickerEls(key);
  if (!els.hidden) return;
  els.hidden.value = path;
  if (path) {
    // A DEAD PATH USED TO LOOK EXACTLY LIKE A LIVE ONE. The tile got
    // `.has-image`, the × appeared, and the broken <img> read as "an image is
    // selected" — so a Load Params replay, an Animate on a photo since
    // deleted, or an H3 first-frame scratch file that was tidied away all
    // submitted happily and failed 30 s into the render. The picker is the
    // place that knows; validating here is the same shape as the refusal
    // gates. Confirmed with a second request so a transient hiccup cannot
    // throw away a good pick: only a 404 (the server saying the file is not
    // there) clears it.
    els.preview.onerror = () => {
      const dead = els.hidden.value;
      if (!dead || dead !== path) return;
      fetch(`/image?path=${encodeURIComponent(path)}&w=16`)
        .then(r => {
          if (r.status !== 404) return;
          if (els.hidden.value !== path) return;
          pickerSetImage(key, '');
          const name = path.split('/').pop();
          if (typeof phosToast === 'function') {
            phosToast(`${name} is no longer on disk — pick another image`,
                      { kind: 'warning' });
          }
        })
        .catch(() => {});
    };
    els.preview.src = `/image?path=${encodeURIComponent(path)}&w=480`;
    els.preview.style.display = 'block';
    els.empty.style.display = 'none';
    els.clear.style.display = 'flex';
    els.drop.classList.add('has-image');
    // Highlight the matching thumbnail in the recent strip if visible.
    if (els.recentStrip) {
      els.recentStrip.querySelectorAll('img').forEach(img => {
        img.classList.toggle('selected', img.dataset.path === path);
      });
    }
    // FFLF anchors framing on the start frame; I2V anchors on its single
    // image. End frame doesn't drive aspect (would override the start
    // frame). a2v_image lives in the Audio→Video tab which has its own
    // width/height inputs — calling snapAspectToImage would change the
    // GLOBAL #aspect selector, not the A2V one, so skip it for a2v_image
    // to avoid silently mutating an unrelated mode.
    if ((key === 'image' || key === 'start_image') && opts.snapAspect !== false) {
      snapAspectToImage(path);
    }
  } else {
    // Drop the handler with the pick, or a later `removeAttribute('src')`
    // fires it against a path that is no longer selected.
    els.preview.onerror = null;
    els.preview.removeAttribute('src');
    els.preview.style.display = 'none';
    els.empty.style.display = '';
    els.clear.style.display = 'none';
    els.drop.classList.remove('has-image');
    if (els.recentStrip) {
      els.recentStrip.querySelectorAll('img').forEach(img => img.classList.remove('selected'));
    }
  }
  updateDerived();
}

async function pickerUploadFile(key, file) {
  const els = pickerEls(key);
  if (!file || !els.drop) return;
  // Inline progress overlay on the drop tile while the upload runs.
  let busy = els.drop.querySelector('.picker-uploading');
  if (!busy) {
    busy = document.createElement('div');
    busy.className = 'picker-uploading';
    busy.textContent = `Uploading ${file.name}…`;
    els.drop.appendChild(busy);
  }
  try {
    const fd = new FormData(); fd.append('image', file);
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'upload failed');
    pickerSetImage(key, data.path);
    // Refresh the "Recent uploads" strip so the just-uploaded file shows
    // up immediately for the other slots too.
    refreshUploadsStrip();
  } catch (e) {
    alert(`Upload failed: ${e.message || e}`);
  } finally {
    busy.remove();
  }
}

function pickerWire(key) {
  const els = pickerEls(key);
  if (!els.drop) return;
  // Click → file dialog. Skip when the click came from the clear button.
  els.drop.addEventListener('click', (e) => {
    if (e.target.closest('.picker-clear')) return;
    els.file.click();
  });
  els.file.addEventListener('change', () => {
    if (els.file.files[0]) pickerUploadFile(key, els.file.files[0]);
    els.file.value = '';   // allow re-uploading the same file
  });
  els.clear.addEventListener('click', (e) => { e.stopPropagation(); pickerSetImage(key, ''); });
  // Drag-drop. preventDefault on dragover is what enables drop.
  els.drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.drop.classList.add('dragover');
  });
  els.drop.addEventListener('dragleave', () => els.drop.classList.remove('dragover'));
  els.drop.addEventListener('drop', (e) => {
    e.preventDefault();
    els.drop.classList.remove('dragover');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) pickerUploadFile(key, f);
  });
}

globalThis._uploadsCache = [];   // last fetched list, kept module-level so all
                          //   three pickers render the same source data.
async function refreshUploadsStrip() {
  let data;
  try { data = await api('/uploads?limit=24'); }
  catch (e) { return; }
  _uploadsCache = Array.isArray(data && data.uploads) ? data.uploads : [];
  PICKERS.forEach(key => {
    const els = pickerEls(key);
    if (!els.recentStrip) return;
    if (!_uploadsCache.length) {
      els.recentWrap.style.display = 'none';
      els.recentStrip.innerHTML = '';   // no stale thumbs behind a hidden strip
      return;
    }
    els.recentWrap.style.display = '';
    const currentPath = els.hidden.value;
    // Each thumbnail carries its own "×" (a Pinokio ask): the strip had no
    // way to remove an imported image, and files removed by hand in Finder
    // kept their thumbnails. Listeners, not inline onclick — generated
    // markup resolves inline handlers through the global scope, which is
    // the v4.9.0 regression class (see scripts/lint_webapp.mjs).
    els.recentStrip.innerHTML = _uploadsCache.map(u => `
      <span class="picker-recent-wrap">
        <img class="picker-recent-thumb${u.path === currentPath ? ' selected' : ''}"
             src="${escapeHtml(_thumbUrl(u.url, 128))}"
             data-path="${escapeHtml(u.path)}"
             title="${escapeHtml(u.name)} · ${u.size_kb} KB · ${escapeHtml(u.mtime)}"
             alt="">
        <button type="button" class="picker-recent-x" data-path="${escapeHtml(u.path)}"
                title="Delete this imported image"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>
      </span>
    `).join('');
    els.recentStrip.querySelectorAll('img').forEach(img => {
      img.addEventListener('click', () => pickerSetImage(key, img.dataset.path));
    });
    els.recentStrip.querySelectorAll('.picker-recent-x').forEach(btn => {
      btn.addEventListener('click', (e) => { e.stopPropagation(); deleteUpload(btn.dataset.path); });
    });
  });
}

// Delete an imported reference image from disk (plus its thumbnails), then
// repaint every strip. A picker currently pointing at the deleted file is
// cleared so a queued job cannot reference a path that no longer exists.
async function deleteUpload(path) {
  if (!path) return;
  const name = String(path).split('/').pop();
  if (!confirm(`Delete "${name}" from your imported images? This removes the file.`)) return;
  let r;
  try {
    r = await api('/upload/delete', 'POST', new URLSearchParams({path}));
  } catch (e) {
    phosToast('Could not delete: ' + (e && e.message ? e.message : e), {});
    return;
  }
  if (!r || r.ok === false) {
    phosToast('Could not delete: ' + ((r && r.error) || 'unknown error'), {});
    return;
  }
  PICKERS.forEach(key => {
    const els = pickerEls(key);
    if (els && els.hidden && els.hidden.value === path) pickerSetImage(key, '');
  });
  await refreshUploadsStrip();
  if (typeof refreshIngredientRecent === 'function') { try { refreshIngredientRecent(); } catch (e) {} }
}

// ====== Ingredients (multi-reference) picker ======
// Unlike the single-image pickers above, Ingredients holds an ORDERED LIST of
// 2-8 server-side paths. The list is mirrored into the hidden
// #ingredient_images_json input that make_job reads (and that's in the
// allowlist — see the make_job params dict). Uploads go through the same
// /upload endpoint; selecting from "Recent uploads" appends a path.
let _ingredientPaths = [];   // array of server-side panel_uploads/* paths

function _ingredientSync() {
  const hidden = document.getElementById('ingredient_images_json');
  if (hidden) hidden.value = JSON.stringify(_ingredientPaths);
  _ingredientRenderThumbs();
  // Keep the recent strip's "added" highlight in sync.
  if (typeof refreshIngredientRecent === 'function') refreshIngredientRecent();
}

function _ingredientRenderThumbs() {
  const wrap = document.getElementById('ingredients_thumbs');
  if (!wrap) return;
  if (!_ingredientPaths.length) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = _ingredientPaths.map((p, i) => `
    <div class="ingredient-thumb" data-idx="${i}">
      <img src="${escapeHtml(_thumbUrl('/image?path=' + encodeURIComponent(p), 160))}" alt="">
      <button type="button" class="ingredient-thumb-x" data-idx="${i}" title="Remove"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>
      <span class="ingredient-thumb-n">${i + 1}</span>
    </div>
  `).join('');
  wrap.querySelectorAll('.ingredient-thumb-x').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.idx, 10);
      _ingredientPaths.splice(idx, 1);
      _ingredientSync();
    });
  });
}

function ingredientAddPath(path) {
  if (!path) return;
  if (_ingredientPaths.includes(path)) return;   // no dupes
  if (_ingredientPaths.length >= 8) { alert('Ingredients takes at most 8 images.'); return; }
  _ingredientPaths.push(path);
  _ingredientSync();
}

async function ingredientUploadFile(file) {
  const drop = document.getElementById('ingredients_drop');
  if (!file || !drop) return;
  let busy = drop.querySelector('.picker-uploading');
  if (!busy) {
    busy = document.createElement('div');
    busy.className = 'picker-uploading';
    drop.appendChild(busy);
  }
  busy.textContent = `Uploading ${file.name}…`;
  try {
    const fd = new FormData(); fd.append('image', file);
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'upload failed');
    ingredientAddPath(data.path);
    if (typeof refreshUploadsStrip === 'function') refreshUploadsStrip();
  } catch (e) {
    alert(`Upload failed: ${e.message || e}`);
  } finally {
    busy.remove();
  }
}

async function refreshIngredientRecent() {
  const wrap = document.getElementById('ingredients_recent_wrap');
  const strip = document.getElementById('ingredients_recent');
  if (!wrap || !strip) return;
  // Reuse the module-level uploads cache the single pickers already fill.
  let list = _uploadsCache;
  if (!list || !list.length) {
    try { const d = await api('/uploads?limit=24'); list = (d && d.uploads) || []; _uploadsCache = list; }
    catch (e) { list = []; }
  }
  if (!list.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  strip.innerHTML = list.map(u => `
    <img class="picker-recent-thumb${_ingredientPaths.includes(u.path) ? ' selected' : ''}"
         src="${escapeHtml(_thumbUrl(u.url, 128))}"
         data-path="${escapeHtml(u.path)}"
         title="${escapeHtml(u.name)} · ${u.size_kb} KB"
         alt="">
  `).join('');
  strip.querySelectorAll('img').forEach(img => {
    img.addEventListener('click', () => ingredientAddPath(img.dataset.path));
  });
}

function ingredientPickerWire() {
  const drop = document.getElementById('ingredients_drop');
  const file = document.getElementById('ingredients_file');
  if (!drop || !file || drop.__wired) return;
  drop.__wired = true;
  drop.addEventListener('click', (e) => {
    if (e.target.closest('.ingredient-thumb-x')) return;
    file.click();
  });
  file.addEventListener('change', () => {
    Array.from(file.files || []).forEach(f => ingredientUploadFile(f));
    file.value = '';   // allow re-uploading the same file
  });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragover');
    Array.from((e.dataTransfer && e.dataTransfer.files) || []).forEach(f => ingredientUploadFile(f));
  });
  _ingredientSync();
}

// ====== Format helpers ======
// Memory badge formatting. 2026-05-20 — switched from "· swap N.N" to
// "· N% pressure". macOS swap accounting is sticky: once swap pages are
// allocated they stay flagged-used even after the in-RAM copies are
// paged back, so the swap number only ever decreases on reboot. Users
// would see "swap 8.5" hours after the system stopped thrashing and
// assume Phosphene was leaking. Memory pressure (active + wired +
// compressed pages over total) is what actually drives system stress,
// it tracks real-time, and it's what macOS's own Activity Monitor
// surfaces in the Memory Pressure indicator. The swap value is still
// included in /status payloads for any external tooling that wants it.
function fmtMem(m) {
  // Compact on purpose. "21.0 / 64 GB · 33% pressure" is five pieces of
  // information for a glance that only ever asks one question — am I near the
  // ceiling. The decimal on used never changed a decision, and the word
  // "pressure" is carried by the tooltip and the pill's own colour. The full
  // sentence still lives in the title attribute for anyone who wants it.
  const p = (m.pressure_pct != null) ? `${m.pressure_pct}%` : `swap ${(m.swap_gb || 0).toFixed(1)}`;
  return `${Math.round(m.used_gb)}/${m.total_gb.toFixed(0)} GB · ${p}`;
}
function fmtMemTitle(m) {
  const used = (m.used_gb != null) ? m.used_gb.toFixed(1) : '?';
  const tot = (m.total_gb != null) ? m.total_gb.toFixed(0) : '?';
  const pr = (m.pressure_pct != null) ? `${m.pressure_pct}% memory pressure` : null;
  const sw = (m.swap_gb != null) ? `swap ${m.swap_gb.toFixed(1)} GB` : null;
  return [`${used} of ${tot} GB in use`, pr, sw].filter(Boolean).join(' · ');
}
function fmtMin(s) { if (!s || s < 0) return '—'; const m = Math.floor(s/60); const sec = Math.round(s%60); return m > 0 ? `${m}m ${sec}s` : `${sec}s`; }
function snippet(s, n = 70) { if (!s) return ''; s = s.replace(/\s+/g,' ').trim(); return s.length > n ? s.slice(0, n-1)+'…' : s; }
function escapeHtml(s) { if (!s) return ''; return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

// Append `w=<px>` to an /image URL so the server returns a resized
// JPEG thumbnail instead of the full PNG. HiDream Quality renders are
// 2560x1440 (14 MB decoded each) — painting them at 200px carousel
// thumbs filled Chrome with >1 GB of decoded bitmaps. Thumbs cut that
// ~25-30x. Pass-through for non-`/image?` URLs (videos served via
// /file, civitai previews on a remote CDN, etc).
function _thumbUrl(url, w) {
  if (!url || typeof url !== 'string') return url;
  if (!url.startsWith('/image?')) return url;
  // Preserve existing query string (path, mtime cache-bust v=N, etc).
  const sep = url.includes('w=') ? '' : (url.includes('?') ? '&' : '?');
  return sep ? `${url}&w=${w | 0}` : url;
}

async function api(path, method = 'GET', body = null) {
  // The second argument is a METHOD STRING, not a fetch options object — and
  // that is a genuinely surprising signature, because everything else in the
  // browser takes `{method, body}`. Both of v4.0's new actions (Stop early,
  // Storage → Remove) were written against the shape everyone expects, and
  // `fetch(path, {method: {method:'POST'}})` THROWS before the request leaves
  // the page: "'[object Object]' is not a valid HTTP method." Two headline
  // features were dead in the UI while both servers behaved perfectly, because
  // each was proved with curl instead of with the button.
  //
  // So the helper now accepts either shape. This is not politeness towards a
  // typo: the options-object form is the conventional one, and a helper that
  // silently throws on it will keep collecting this bug forever.
  if (method && typeof method === 'object') {
    body = (method.body != null) ? method.body : body;
    method = method.method || 'GET';
  }
  const opts = { method };
  if (body) {
    opts.body = body instanceof FormData ? new URLSearchParams(body) : body;
    opts.headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
  }
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 409) {
    // Prefer the SERVER'S sentence. /models/remove's three refusals are
    // carefully written ("That's the model this build renders with.") and
    // throwing a bare status turned every one of them into "…: 400".
    let msg = '';
    try { msg = ((await r.json()) || {}).error || ''; } catch (e) {}
    throw new Error(msg || `${path}: ${r.status}`);
  }
  return r.status === 409 ? { error: 'busy' } : r.json().catch(() => ({}));
}

// ====== Poll ======
// Cache of the latest /status response so non-poll callers (setMode,
// setQuality) can refresh tier-gated UI without waiting for the next tick.
globalThis.LAST_STATUS = null;

// Tracks consecutive /status failures so we can surface a panel-offline
// banner instead of silently freezing the UI. Two-strike threshold so a
// single transient hiccup (network blip, panel reload) doesn't flash.
globalThis._POLL_FAILS = 0;

// Last 8 banner messages stay accessible from devtools at
// `window._panelBannerLog` so a user who saw a flash but couldn't
// read it can recover the message after the fact. Cheap insurance.
window._panelBannerLog = window._panelBannerLog || [];

function _setOfflineBanner(visible, msg) {
  let bar = document.getElementById('panelOfflineBanner');
  if (visible) {
    const text = msg || "uploads, chat & renders are paused";
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'panelOfflineBanner';
      bar.className = 'panel-offline-banner';
      bar.innerHTML =
        '<span class="icon"><img src="/assets/favicon-64.png" alt=""></span>' +
        '<span class="label">Phosphene offline</span>' +
        '<span class="text"></span>' +
        '<span class="hint">restart from Pinokio</span>';
      document.body.appendChild(bar);
    }
    bar.classList.remove('reconnected');
    bar.querySelector('.label').textContent = 'Phosphene offline';
    bar.querySelector('.text').textContent = text;
    bar.querySelector('.hint').textContent = 'restart from Pinokio';
    const entry = `${new Date().toLocaleTimeString()} offline · ${text}`;
    window._panelBannerLog.push(entry);
    if (window._panelBannerLog.length > 8) window._panelBannerLog.shift();
    console.warn('[phosphene] offline banner:', text);
  } else if (bar && !bar.classList.contains('reconnected')) {
    // Linger for 3s in a "back online" state instead of removing
    // instantly. Mr Bizarro flagged that he saw a banner flash during a
    // restart but couldn't read it before it disappeared — this gives
    // the eye time to register and gives the user a chance to scroll
    // back in window._panelBannerLog if they want details.
    bar.classList.add('reconnected');
    bar.querySelector('.label').textContent = 'Phosphene reconnected';
    bar.querySelector('.text').textContent = 'queue + renders resumed';
    bar.querySelector('.hint').textContent = '';
    const entry = `${new Date().toLocaleTimeString()} online · reconnected`;
    window._panelBannerLog.push(entry);
    if (window._panelBannerLog.length > 8) window._panelBannerLog.shift();
    console.info('[phosphene] panel reconnected');
    setTimeout(() => {
      const b = document.getElementById('panelOfflineBanner');
      if (b && b.classList.contains('reconnected')) b.remove();
    }, 3000);
  }
}

// Combine the cheap boot header+size scan (model_integrity) with the on-demand
// deep checksum scan (deep_verify.result) into one {ok, bad[]} for the banner,
// so a checksum mismatch lights the same red Repair banner.
function mergeIntegrity(integ, deep) {
  const bad = []; const seen = new Set();
  const add = (arr) => (arr || []).forEach(b => {
    const k = b.repo + '/' + b.file;
    if (!seen.has(k)) { seen.add(k); bad.push(b); }
  });
  if (integ && !integ.ok) add(integ.bad);
  const dv = deep && deep.result;
  if (dv && !dv.ok) add(dv.bad);
  // Render-level codec audit (the v3.8.1 class) rides model_integrity as its
  // own sub-block. Flagged `codec:true` so the banner never offers Repair for
  // it — re-downloading weights cannot fix an unapplied codec patch. See
  // _output_codec_report() in the backend.
  const oc = integ && integ.output_codec;
  if (oc && oc.ok === false) {
    bad.push({ repo: 'output-codec', file: oc.file || '', codec: true,
               reason: (oc.problems || []).join('; ') });
  }
  return { ok: bad.length === 0, bad };
}

// Deep (checksum) verify — Settings → Model files. Hashes every weight vs the
// published upstream SHA-256; catches right-size/wrong-content "mosaic" weights.
async function startDeepVerify() {
  const btn = document.getElementById('deepVerifyBtn');
  const st = document.getElementById('deepVerifyStatus');
  if (btn) btn.disabled = true;
  if (st) st.textContent = 'starting…';
  try {
    await fetch('/models/verify-deep', { method: 'POST' });
    if (st) st.textContent = 'verifying… (this can take 1–2 min)';
  } catch (e) {
    if (st) st.textContent = 'failed to start: ' + e;
    if (btn) btn.disabled = false;
  }
}

function renderDeepVerifyStatus(deep) {
  const btn = document.getElementById('deepVerifyBtn');
  const st = document.getElementById('deepVerifyStatus');
  if (!st || !btn) return;
  if (deep && deep.active) {
    btn.disabled = true;
    st.textContent = 'verifying ' + (deep.progress || '') + '… (1–2 min)';
    return;
  }
  btn.disabled = false;
  const r = deep && deep.result;
  if (!r) return;
  if (r.error) { st.textContent = 'verify error: ' + r.error; return; }
  if (!r.ok) {
    const nMisplaced = (r.bad || []).filter(b => b.placement).length;
    const n = (r.bad || []).length - nMisplaced;
    st.innerHTML = '<span style="color:#e06666">✗ '
      + [n ? n + ' file(s) corrupt/stale' : '',
         nMisplaced ? nMisplaced + ' file(s) in the wrong place' : ''].filter(Boolean).join(', ')
      + ' — see the red banner up top.</span>';
  } else {
    const uv = (r.unverified || []).length;
    // §3.5: "unverifiable" is not a failure, and the user deserves to know WHY
    // some files cannot be checked rather than being left to assume the worst.
    // An absent or malformed manifest reports unverified and stays ok,
    // precisely so it never triggers a spurious multi-GB re-download.
    st.innerHTML = '<span style="color:#5bbf7b">✓ all ' + r.checked + ' file(s) match upstream'
      + (uv ? ' (' + uv + ' unverifiable — the LTX-2.5 packs carry their own '
            + 'checksum list; an older copy may not have one)' : '') + '.</span>';
  }
}

// 2026-06-04: model-integrity banner. The backend flags corrupt/partial weight
// files (a garbage-decode "mosaic" cause) in /status.model_integrity; surface a
// one-click Repair so users self-heal instead of staring at broken renders.
function renderIntegrityBanner(integ) {
  const bad = (integ && !integ.ok) ? (integ.bad || []) : [];
  let el = document.getElementById('integrityBanner');
  if (!bad.length) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('div');
    el.id = 'integrityBanner';
    el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#7a1f1f;'
      + 'color:#fff;padding:10px 16px;font-size:13px;line-height:1.4;display:flex;'
      + 'align-items:center;gap:12px;flex-wrap:wrap;box-shadow:0 2px 10px rgba(0,0,0,.45)';
    document.body.appendChild(el);
  }
  // Repair (re-download) only makes sense for weight files — codec-audit rows
  // are excluded from the button list; their cure is Update / reinstall.
  const repos = [...new Set(bad.filter(b => !b.codec).map(b => b.repo))];
  // Placement errors (right content, wrong path) are a different failure from a
  // corrupt download, and the cure is usually a move rather than a re-fetch — so
  // they get their own headline and their reason printed in full (it names both
  // the found-at and expected-at paths). See _placement_errors() in the backend.
  const misplaced = bad.filter(b => b.placement);
  const codecBad = bad.filter(b => b.codec);
  const corrupt = bad.filter(b => !b.placement && !b.codec);
  el.innerHTML =
    '<span style="font-weight:700">'
    + (corrupt.length ? 'Model files look incomplete / corrupt'
       : misplaced.length ? 'Model files are in the wrong place'
       : 'Renders are not being encoded as requested') + '</span>'
    + (corrupt.length
        ? '<span style="opacity:.92">' + escapeHtml(corrupt.map(b => b.file).join(', '))
          + ' — this produces garbled / "mosaic" output (usually an interrupted '
          + 'download).</span>'
        : '')
    + (misplaced.length
        ? '<span style="opacity:.92;flex-basis:100%">'
          + misplaced.map(b => escapeHtml(b.reason)).join('<br>')
          + '</span>'
        : '')
    + (codecBad.length
        ? '<span style="opacity:.92;flex-basis:100%">'
          + codecBad.map(b => escapeHtml(b.file + ' — ' + b.reason)).join('<br>')
          + '. The codec patch may not be applied — click Update in Pinokio '
          + '(if it persists, reinstall). Clips rendered like this carry '
          + 'avoidable compression on faces.</span>'
        : '')
    + '<span style="margin-left:auto;display:flex;gap:8px">'
    + repos.map(k => '<button class="btn btn-primary" onclick="repairModel(\'' + escapeHtml(k)
        + '\')">Repair ' + escapeHtml(k.toUpperCase()) + ' (re-download)</button>').join('')
    + '<button class="btn" onclick="this.closest(\'#integrityBanner\').remove()">Dismiss</button>'
    + '</span>';
}

async function repairModel(key) {
  if (!confirm('Re-download the corrupt ' + key.toUpperCase() + ' model file(s)?\n\n'
      + 'This deletes the bad files and fetches fresh copies (resumable).')) return;
  try {
    const r = await fetch('/models/repair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'repo_key=' + encodeURIComponent(key),
    });
    const data = await r.json();
    if (data.ok) {
      alert('Repair started — re-downloading ' + ((data.deleted || []).length) + ' file(s). '
        + 'Watch the download progress / Logs; the banner clears once the files verify clean.');
    } else {
      alert('Repair failed: ' + (data.error || r.status));
    }
  } catch (e) { alert('Repair failed: ' + e); }
}

// Translate a cryptic engine error into actionable user guidance. Extracted
// from the Now card 2026-08-11 so a storyboard shot's failure reads EXACTLY
// like a manual one — one if/else, two callers, no way for the two to drift.
// "helper died mid-job (no event)" is the SIGKILL-by-jetsam signature on
// memory-pressured Macs: the helper subprocess gets killed by the OS for using
// too much RAM and we never get an event back. Tell the user how to recover
// instead of leaving them with the engine wording.
function friendlyJobError(raw) {
  raw = raw || 'unknown error';
  const rawLower = String(raw).toLowerCase();
  if (rawLower.includes('sigkill')) {
    return { friendly: 'Helper killed by the OS — out of memory (jetsam).',
             hint: 'Close memory-heavy apps (Chrome, Slack, iOS Simulator) and try again, ' +
                   'or switch Quality to Quick (about half the RAM).' };
  }
  if (rawLower.includes('sigsegv') || rawLower.includes('sigbus')) {
    return { friendly: 'Helper crashed at the native level (MLX/Metal fault).',
             hint: 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
                   'on github.com/mrbizarro/phosphene/issues so we can fix it.' };
  }
  if (rawLower.includes('sigabrt')) {
    return { friendly: 'Helper hit a C-level assertion and aborted.',
             hint: 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
                   'on github.com/mrbizarro/phosphene/issues.' };
  }
  if (rawLower.includes('helper exited from') || rawLower.includes('helper pipe closed') ||
      rawLower.includes('helper died') || rawLower.includes('helper exited')) {
    return { friendly: 'Helper exited unexpectedly.',
             hint: 'Check the log for the last "step:*" breadcrumb (tells us which ' +
                   'phase died). If memory-pressured, close other apps and retry.' };
  }
  // Layer 2. Layer 1 (applyPackIncompleteGate) should have stopped this before
  // the render started; this fires when it could not — a file removed while the
  // job sat in the queue, or a surface that reaches make_job without the form.
  // The RuntimeError text itself is deliberately unchanged upstream: it is what
  // the log carries and what a bug report quotes. This translates it, it does
  // not replace it.
  if (/model is\s+incomplete\. Missing/i.test(raw)) {
    const m = /Missing \d+ file\(s\) in [^:]+: ([^.]+)\./.exec(raw);
    return {
      friendly: 'Stopped before rendering — the model weights are incomplete.',
      hint: (m ? 'Missing: ' + m[1].trim() + '. ' : '')
          + 'Rendering with a file missing produces garbled "mosaic" video rather '
          + 'than an error, so Phosphene stops instead. Settings → Models → Resume.',
    };
  }
  if (rawLower.includes('q8') || rawLower.includes('keyframe')) {
    return { friendly: 'This mode needs the Q8 model.', hint: raw };
  }
  return { friendly: 'Job failed.', hint: raw };
}

// ---- Completion alerts ------------------------------------------------------
// A render is minutes long. When one finishes — or fails — the tab says so
// with a short chime, and the browser says so when the tab is in the
// background and notifications were allowed. Keyed on the history: a job id
// that was not done on the previous poll and is now. The first poll of a
// page load only records what is already done, so opening the panel is never
// a burst of alerts for last night.
let _doneSeen = null;
function notifyJobsDone(s) {
  const hist = Array.isArray(s && s.history) ? s.history : [];
  const now = new Set();
  for (const j of hist) if (j && j.id && (j.status === 'done' || j.status === 'failed')) now.add(j.id);
  if (_doneSeen === null) { _doneSeen = now; return; }
  const cur = (globalThis._settingsCache && _settingsCache.settings) || {};
  const on = cur.notify_done !== false;
  for (const j of hist) {
    if (!j || !j.id || !now.has(j.id) || _doneSeen.has(j.id)) continue;
    if (on) notifyOneJob(j);
  }
  _doneSeen = now;
}
function notifyOneJob(j) {
  const failed = j.status === 'failed';
  const what = (j.params && (j.params.label || j.params.preset_label)) ||
               (j.params && j.params.prompt ? String(j.params.prompt).slice(0, 60) : '') ||
               (j.params && j.params.mode) || 'a render';
  playDoneChime(failed);
  try {
    if (document.hidden && typeof window.Notification !== 'undefined' && window.Notification.permission === 'granted') {
      const n = new window.Notification(failed ? 'Phosphene — a render failed' : 'Phosphene — render done',
                                 { body: what, tag: 'phos-' + j.id, silent: true });
      n.onclick = () => { try { window.focus(); n.close(); } catch (e) {} };
    }
  } catch (e) {}
}
// Two short tones from the Web Audio API — no asset, no download, and a
// falling pair for a failure so the ear knows without looking.
let _chimeCtx = null;
function playDoneChime(failed) {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!_chimeCtx) _chimeCtx = new AC();
    const ctx = _chimeCtx;
    if (ctx.state === 'suspended') { ctx.resume().catch(() => {}); }
    const t0 = ctx.currentTime + 0.01;
    const notes = failed ? [[440, 0], [330, 0.16]] : [[660, 0], [880, 0.14]];
    for (const [f, dt] of notes) {
      const o = ctx.createOscillator(); const g = ctx.createGain();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t0 + dt);
      g.gain.exponentialRampToValueAtTime(0.12, t0 + dt + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dt + 0.22);
      o.connect(g); g.connect(ctx.destination);
      o.start(t0 + dt); o.stop(t0 + dt + 0.25);
    }
  } catch (e) {}
}

async function poll() {
  // Reflect HDR-vs-character mutual exclusion every poll cycle. character_id
  // gets set from multiple code paths (manual chip click, load-params,
  // localStorage rehydrate, charactersGenerate sub-form, retry); keeping
  // the HDR pill in sync via a single per-poll re-eval is cheaper than
  // tracking down every assignment site.
  if (typeof _applyHdrPillAvailability === 'function') {
    try { _applyHdrPillAvailability(); } catch (_) {}
  }
  let s;
  const url = '/status' + (filterMode === 'hidden' ? '?include_hidden=1' : '');
  try {
    s = await (await fetch(url)).json();
    notifyJobsDone(s);
    _POLL_FAILS = 0;
    _setOfflineBanner(false);
  } catch (e) {
    _POLL_FAILS += 1;
    // Suppress the offline banner while a known long-running endpoint is
    // in flight (e.g. /version/pull which blocks the server for ~30s on
    // git fetch + git pull). Without this suppression the banner flashes
    // "Phosphene offline → reconnected" at the tail of every successful
    // Update click — looks like an error to the user when nothing is
    // actually wrong.
    if (_POLL_FAILS >= 2 && !window._suppressOfflineBanner) _setOfflineBanner(true);
    return;
  }
  LAST_STATUS = s;
  // PUBLISHED HERE, AT THE TOP, and that position is the fix for R2-B1.
  // It used to be assigned ~230 lines further down — AFTER the block that
  // repaints the tier strips. So the one repaint that fires (the first poll,
  // when the pack signature changes from undefined) read `{}` and re-rendered
  // the chips from a status that had not arrived yet; every later poll saw an
  // unchanged signature and never repainted again. The High chip was born an
  // install offer on a fully-equipped machine and stayed one until the user
  // happened to click another quality chip.
  //
  // Anything derived from /status has to be able to see /status.
  window.__phosLastStatus = s;

  // Corrupt/partial-weight banner (mosaic self-heal).
  try { renderIntegrityBanner(mergeIntegrity(s.model_integrity, s.deep_verify)); } catch (_) {}
  try { renderDeepVerifyStatus(s.deep_verify); } catch (_) {}

  // Memory
  const m = s.memory;
  const memPill = document.getElementById('memPill');
  memPill.innerHTML = `<span class="dot"></span>${fmtMem(m)}`;
  memPill.title = fmtMemTitle(m);
  // 2026-05-20: color the badge by real pressure, not by sticky swap.
  // Same reason fmtMem dropped swap from the visible label — swap is
  // a high-water mark that only decreases on reboot, so keying the
  // danger/warn colors off it kept the badge red long after pressure
  // had fully recovered.
  let memCls = 'pill-good';
  if (m.pressure_pct > 90) memCls = 'pill-danger';
  else if (m.pressure_pct > 75) memCls = 'pill-warn';
  memPill.className = 'pill ' + memCls;
  // The chip DERIVES from the pills, so it must refresh after the colour
  // classes land — calling it before memCls kept the chip one poll stale.
  try { updateHealthChip(); } catch (e) {}

  // Comfy (hidden when not running). Drives three things in lockstep —
  // the status pill, the global Stop Comfy button, and the per-render
  // "Stop ComfyUI before render" checkbox in the form. The checkbox row
  // stays hidden when Comfy isn't running so users who don't have Comfy
  // installed never see a cryptic toggle.
  const cp = document.getElementById('comfyPill');
  const stopBtn = document.getElementById('stopComfyBtn');
  const comfyRow = document.getElementById('comfyKillRow');
  const comfyToggle = document.getElementById('stop_comfy');
  if (s.comfy_pids.length) {
    cp.innerHTML = `<span class="dot"></span>Comfy ${s.comfy_pids.join(', ')}`;
    cp.className = 'pill pill-warn'; cp.style.display = '';
    stopBtn.style.display = '';
    if (comfyRow) comfyRow.style.display = '';
  } else {
    cp.style.display = 'none';
    stopBtn.style.display = 'none';
    if (comfyRow) comfyRow.style.display = 'none';
    // When Comfy isn't running, also force the form value off so the
    // submission doesn't carry a meaningless `stop_comfy=on` server-side.
    if (comfyToggle) comfyToggle.checked = false;
  }

  // Helper
  const hp = document.getElementById('helperPill');
  if (s.helper && s.helper.alive) {
    hp.innerHTML = `<span class="dot"></span>helper warm`;
    hp.className = 'pill pill-good';
    hp.title = 'Helper subprocess is loaded with pipelines and ready.';
  } else {
    // Helper auto-respawns on the next job (see WarmHelper._ensure). "Cold"
    // is normal after the idle timeout, not an error — first job after a
    // cold start eats a ~30s pipeline-load cost.
    hp.innerHTML = `<span class="dot"></span>helper idle`;
    hp.className = 'pill';
    hp.title = 'Helper is idle (auto-exited after the idle timeout). The next queued job will respawn it; expect a one-time ~30s pipeline-load delay.';
  }

  // Tier pill — what this Mac's RAM tier allows. Click to open the
  // explanation modal. Color is informational, not warning: the tier is
  // what it is, not "wrong".
  const tp = document.getElementById('tierPill');
  if (s.tier) {
    const t = s.tier;
    const cls = t.key === 'base' ? 'pill-warn'
              : (t.key === 'pro' ? 'pill-good' : '');
    // Show the friendly label ("Compact" / "Comfortable" / "Roomy" /
    // "Studio") not the internal key. Click opens the explanation modal.
    tp.innerHTML = `<span class="dot"></span>${escapeHtml(t.label || t.key)}`;
    tp.className = 'pill ' + cls;
    tp.title = `${t.label} (${t.ram_label}) · ${t.tagline} · click for details`;
    // Apply tier-driven enabled/disabled state to mode + quality pills.
    // Done here in poll() so a tier override (env var) flips state on
    // panel restart without needing to also change a separate setMode call.
    applyTierGates(t);
  }

  // Models pill — roll-up status: base ready / Q8 ready, plus active download.
  // Renders as one of:
  //   "models ↓ Q4 12%"   while a download streams (live progress, last hf line)
  //   "models 3/3"        all on disk
  //   "models 2/3"        base ready, Q8 missing → warn color
  //   "models 0/3"        base incomplete → bad color
  const mp = document.getElementById('modelsPill');
  const dl = s.download && s.download.active ? s.download : null;
  if (dl) {
    const elapsed = Math.max(0, Math.round(s.server_now - (dl.started_ts || s.server_now)));
    mp.innerHTML = `<span class="dot"></span>↓ ${dl.key} · ${elapsed}s`;
    mp.className = 'pill pill-running';
    mp.title = `Downloading ${dl.repo_id} — ${dl.last_line || 'starting…'}`;
  } else {
    // Per-repo ready/total counts, matches what the modal shows (3 rows by
    // default: Q4 + Gemma + Q8). base_available is a roll-up bool that
    // honors the HF-id env-var short-circuit; we use it for the color
    // hint, not the count itself.
    const baseOk = s.base_available;
    const q8Ok = s.q8_available;
    const ready = s.repos_ready ?? 0;
    const total = s.repos_total ?? 0;
    mp.innerHTML = `<span class="dot"></span>models ${ready}/${total}`;
    mp.className = 'pill ' + (!baseOk ? 'pill-warn' : (q8Ok ? 'pill-good' : ''));
    mp.title = !baseOk
      ? 'Base models incomplete — click to download'
      : (q8Ok ? 'All models on disk' : 'Q8 not installed (optional — needed for High quality + FFLF)');
  }
  // If the modal is open, refresh its rows on each poll so progress updates.
  if (document.getElementById('modelsModal').style.display !== 'none') {
    refreshModelsModal({ silent: true });
  }
  // Inline models card — top-of-form, big, can't miss it. State logic
  // lives in updateModelsCard so we don't bloat poll() further.
  updateModelsCard(s);
  // Hailuo H3 install state — refreshes the engine pill in place when the
  // pack lands (or disappears), same live-unlock contract Q8 already has.
  if (typeof updateH3Availability === 'function') updateH3Availability(s);
  // The pack-incomplete gate, BEFORE the render rather than 30 s into it.
  if (typeof applyPackIncompleteGate === 'function') {
    try { applyPackIncompleteGate(s); } catch (e) {}
  }

  // Queue pill + tab badge. Animate the bottom-pane Queue badge with
  // a brief scale-up "pop" when the count goes up — draws the eye to
  // the badge so the user notices that a new job got queued without
  // having to watch the strip. CSS .badge.bump handles the keyframes;
  // we just toggle the class for ~280 ms.
  const qp = document.getElementById('queuePill');
  qp.innerHTML = `<span class="dot"></span>queue ${s.queue.length}${s.paused ? ' · paused' : ''}`;
  qp.className = 'pill ' + (s.paused ? 'pill-warn' : (s.queue.length ? 'pill-running' : ''));
  const qb = document.getElementById('queueBadge');
  const prevQueueLen = window._lastQueueLen ?? 0;
  if (s.queue.length) {
    qb.textContent = s.queue.length;
    qb.style.display = '';
    if (s.queue.length > prevQueueLen) {
      qb.classList.remove('bump');
      // Re-trigger the CSS animation by toggling the class after a
      // reflow. Without the rAF the second add doesn't re-fire.
      requestAnimationFrame(() => {
        qb.classList.add('bump');
        setTimeout(() => qb.classList.remove('bump'), 320);
      });
    }
  } else {
    qb.style.display = 'none';
  }
  window._lastQueueLen = s.queue.length;

  // Job pill
  const jp = document.getElementById('jobPill');
  if (s.running && s.current) {
    const elapsed = Math.max(0, Math.round(s.server_now - s.current.started_ts));
    jp.innerHTML = `<span class="dot"></span>${s.current.params.label || s.current.params.mode} · ${elapsed}s`;
    jp.className = 'pill pill-running';
  } else {
    jp.innerHTML = `<span class="dot"></span>idle`;
    jp.className = 'pill';
  }

  document.getElementById('pauseBtn').textContent = s.paused ? 'Resume queue' : 'Pause queue';

  // Q8 / High enable.
  //
  // This block used to WRITE the chip's subtitle as well — "Q8 Pro · 7 min",
  // "Install Q8 (37 GB)" — one of two hand-rolled updaters printing numbers
  // that no measurement on 2.5 supports (the 2.5 Q8 pack is 30.02 GB and the
  // High add-on is a separate 29.50 GB). The tier table fills every chip's
  // third slot now, the way H3's already did, so what is left here is the
  // STATE: is the pack on disk, and is the chip therefore an install CTA.
  // PACK PRESENCE, as distinct from cap-tier. `data-cap-tier` answers "what can
  // this Mac's RAM serve"; this answers "are the weights on disk". On 2.5 those
  // two diverge for the first time — a 64 GB Mac holding only q4_25 resolves
  // cap_tier=q8 — so every surface that used to key off cap-tier to mean "no
  // Q8" has to read this instead. Both must be able to be true.
  // Reads q8_PACK_available, NOT q8_available: characters render on q8 +
  // distilled and do not touch the HQ add-on, so a user with the full 30 GB
  // pack and no add-on must not be told to install the pack they already have.
  const q8PackOk = (s.q8_pack_available !== undefined) ? s.q8_pack_available : s.q8_available;
  document.body.dataset.q8Pack = q8PackOk ? 'ready' : 'missing';
  // The High chip's install state is rendered by renderTierAxes now
  // (ltxCellNeedsInstall / ltxCellInstallLabel), so the chip's third slot
  // names the download instead of an ETA the user cannot have yet, and a click
  // opens the Models modal instead of selecting a tier that cannot render.
  //
  // THE SILENT BOUNCE IS GONE. This block used to `setQuality('standard')`
  // whenever Q8 was absent and High was selected — so clicking High moved the
  // form, by itself, to a BIGGER and SLOWER canvas the user never asked for,
  // one and a half seconds after showing them the explanation. The pre-render
  // gate (applyPackIncompleteGate) already disables Generate and names the
  // missing file; the user keeps the tier they chose and is told why it cannot
  // run, which is the honest version of the same information.
  //
  // Repaint so the chips follow pack state as a download lands or a file goes
  // missing, without a reload.
  const packSig = `${s.q8_available}|${q8PackOk}|${(s.q8_missing || []).length}|${(s.hq_addon_missing || []).length}`;
  if (window._lastPackSig !== packSig) {
    window._lastPackSig = packSig;
    if (typeof renderTierAxes === 'function' && document.body.dataset.engine !== 'h3') {
      try { renderTierAxes('ltx'); } catch (e) {}
    }
    try { renderCharacterStrip(); } catch (e) {}
    try { charactersRenderChips(); } catch (e) {}
  }

  // Balanced subtitle: on the "standard" (48–79 GB) tier with Q8 installed,
  // the Balanced chip auto-routes to Q8 Fast (safe_a config) — but ONLY for
  // mode ∈ {t2v, a2v} and frames ≤ 121, which is what the 64 GB memory
  // budget actually fits. Other combos fall back to Q4 Balanced. The
  // subtitle reflects the user's current selection so they see what they
  // will actually get when they hit Generate. Re-evaluated on every poll
  // and on mode/frames change (see the listener wired at DOMContentLoaded
  // below this function).
  // Keyframe (FFLF) and Extend both require Q8 — server enforces it (see
  // run_job_inner). The UI was previously silently downgrading the user to
  // Standard when they picked keyframe with Q8 missing, then the server
  // would 500 on submit. Disable Generate + show a clear reason while in
  // that state. Y1.036 added Extend to the same gate after the Y1.024
  // download trim exposed that Extend is structurally Q8-class.
  const genBtn = document.getElementById('genBtn');
  const q8GatedMode = (currentMode === 'keyframe' || currentMode === 'extend');
  if (currentMode === 'ingredients' && !ingredientsServed()) {
    // Belt to setMode's braces. setMode should make this unreachable, but
    // this is the state the button is in if anything ever gets there again
    // — and "Generate is lit, click it, get an error" is exactly the
    // experience being removed.
    genBtn.disabled = true;
    genBtn.title = 'Ingredients needs the LTX-2.3 generation — its reference '
                 + 'adapter has no 2.5 release. Use Image mode with Inspire, '
                 + 'or install the 2.3 pack from the Train tab.';
    genBtn.textContent = 'Generate · needs LTX-2.3';
  } else if (genBtn.disabled
             && genBtn.textContent.startsWith('Generate · needs LTX-2.3')) {
    genBtn.disabled = false;
    genBtn.title = '';
    genBtn.textContent = 'Generate';
  } else if (q8GatedMode && !s.q8_available) {
    genBtn.disabled = true;
    const modeName = currentMode === 'keyframe' ? 'Keyframe (FFLF)' : 'Extend';
    const left = (s.q8_missing || []).length;
    genBtn.title = left > 0 && left < 6
      ? `${modeName} needs Q8 — ${left} file(s) still downloading.`
      : (() => {
          // Registry-driven: the pack this generation actually needs, at the
          // size the registry actually records. "37 GB" was 2.3's number.
          const P = ((BOOT.ltx || {}).packs) || {};
          const w = P.q8;
          return w ? `${modeName} needs ${w.name} (~${w.size}) — install it from Settings → Models.`
                   : `${modeName} needs the Q8 weights — install them from Settings → Models.`;
        })();
    genBtn.textContent = 'Generate · Q8 required';
  } else if (genBtn.disabled && genBtn.textContent.startsWith('Generate · Q8')) {
    // Restore — only do so if WE were the ones who disabled it, otherwise
    // some future code path that disables Generate for a different reason
    // would get clobbered here.
    genBtn.disabled = false;
    genBtn.title = '';
    genBtn.textContent = 'Generate';
  }

  // Now card
  // Y1.039 — bar + meta line driven by server-computed progress (phase-aware,
  // config-bucketed ETA, denoise per-step extrapolation). Falls back to the
  // old elapsed/global-avg behavior if the server didn't ship a progress
  // block (e.g. mid-deploy where the server is older than the JS).
  const nowCard = document.getElementById('nowCard');
  const fill = document.getElementById('progressFill');
  // Normalize once per /status response. The compact Now card and the main
  // stage are two views over this object, not two readers inventing their own
  // schema/rules (and emphatically not two polling loops).
  let livePreviewData = null;
  // The Now-card Stop button is gone — both the video form's Stop and
  // the Image Studio's Stop now live in the form-pane and stay
  // visible across mode switches, so a Now-card duplicate is no
  // longer needed.
  if (s.running && s.current) {
    nowCard.classList.remove('idle', 'failed');
    const prog = s.current.progress || null;
    const elapsedFallback = Math.max(0, s.server_now - s.current.started_ts);
    let pct, elapsed, phaseLabel, timing;
    if (prog) {
      pct = Math.min(99, Math.max(0, prog.pct ?? 0));
      elapsed = prog.elapsed_sec ?? elapsedFallback;
      phaseLabel = prog.phase_label || 'Working';
      if (prog.remaining_sec != null && prog.remaining_sec > 0) {
        timing = `<strong>${fmtMin(elapsed)}</strong> in · ~${fmtMin(prog.remaining_sec)} left`;
      } else if (prog.eta_sec) {
        timing = `<strong>${fmtMin(elapsed)}</strong> / ~${fmtMin(prog.eta_sec)}`;
      } else {
        timing = `<strong>${fmtMin(elapsed)}</strong> elapsed`;
      }
    } else {
      // Legacy fallback path
      const avg = s.avg_elapsed_sec || 420;
      pct = Math.min(99, Math.round(elapsedFallback / avg * 100));
      elapsed = elapsedFallback;
      phaseLabel = '';
      timing = `<strong>${fmtMin(elapsed)}</strong> elapsed${avg ? ' / ~'+fmtMin(avg)+' avg' : ''}`;
    }
    fill.style.width = pct + '%';
    nowCard.querySelector('.ttl').textContent = snippet(s.current.params.label || s.current.params.prompt, 80);
    // Image jobs don't have width/height/frames; show n × aspect instead.
    // Falls through to the video shape when those fields ARE present.
    //
    // TRAINING has no frames either, and printed the literal string
    // "undefinedf" for the whole run — reported in #61 as
    // `train · 512×512 · undefinedf · 5m 39s`. A training job's shape is its
    // canvas; the step count is already carried by phaseLabel underneath.
    // Anything else missing frames drops the token rather than inventing one.
    const cur = s.current.params;
    const shape = [cur.width && cur.height ? `${cur.width}×${cur.height}` : null,
                   (cur.frames != null && cur.frames !== '') ? `${cur.frames}f` : null]
                  .filter(Boolean).join(' · ');
    const baseMeta = (cur.mode === 'image')
      ? `image · ${cur.aspect || '?'} · n=${cur.n || '?'} · ${cur.engine_override || 'auto'} · ${timing}`
      : [cur.mode, shape, timing].filter(Boolean).join(' · ');
    nowCard.querySelector('.meta').innerHTML = phaseLabel
      ? `${baseMeta}<br><span style="color:var(--muted)">${escapeHtml(phaseLabel)}</span>`
      : baseMeta;
    livePreviewData = normalizeLivePreview(s, prog);
    renderNowPreview(s, prog, livePreviewData);
  } else {
    renderNowPreview(s, null, null);
    // Idle state. If the LAST history entry was a failure (helper crash,
    // OOM, etc.) surface it loud-and-clear here — otherwise users like
    // cocktailpeanut just see "Idle" and assume "the panel did nothing."
    // We hold the failure visible until the user starts a new job.
    fill.style.width = '0%';
    const last = (s.history || [])[0];
    // Mr Bizarro flagged that a stuck "Job failed" card with no close button
    // is disruptive — once a render fails, the surface holds the message
    // visible until the next job runs, with no way to acknowledge it.
    // _dismissedFailureId remembers which job id the user chose to
    // dismiss so we don't keep nagging. Clears on next job (id changes).
    const showFailure = last && last.status === 'failed' && !s.queue.length
                        && last.id !== window._dismissedFailureId;
    const actionsEl = document.getElementById('nowCardActions');
    if (showFailure) {
      nowCard.classList.remove('idle');
      nowCard.classList.add('failed');
      // Translate cryptic engine errors into actionable user guidance.
      // "helper died mid-job (no event)" is the SIGKILL-by-jetsam
      // signature on memory-pressured Macs — the helper subprocess gets
      // killed by the OS for using too much RAM and we never get an
      // event back. Tell the user how to recover instead of leaving them
      // with the engine wording.
      const { friendly, hint } = friendlyJobError(last.error || 'unknown error');
      nowCard.querySelector('.ttl').innerHTML =
        `<span style="color: var(--danger, #f85149)"><svg class="ph" aria-hidden="true" style="margin-right:4px;vertical-align:-2px"><use href="#ph-warning-fill"/></svg>${escapeHtml(friendly)}</span>`;
      nowCard.querySelector('.meta').innerHTML =
        `<span style="color: var(--muted)">${escapeHtml(snippet(last.params.label || last.params.prompt, 80))}</span>` +
        ` <span style="color: var(--muted)">· ${escapeHtml(last.params.mode)} · ${last.params.width}×${last.params.height}</span>` +
        `<br><span style="color: var(--text)">${escapeHtml(hint)}</span>`;
      // Action row: Retry (re-submit same params via /queue/retry) +
      // Dismiss (mark this id as handled so the next idle poll clears
      // the card). Both buttons live in a stable sibling element of
      // .ttl so the click handlers survive every poll-driven rewrite.
      // A single delegated listener on document (installed once at
      // boot) catches the clicks via data-action so we never lose them
      // to an inline-handler race.
      if (actionsEl) {
        actionsEl.dataset.jobId = String(last.id);
        actionsEl.innerHTML =
          `<button type="button" class="now-card-retry" data-action="retry" ` +
          `title="Re-submit this job with the same params">` +
          `<svg class="ph" aria-hidden="true"><use href="#ph-arrow-clockwise"/></svg>` +
          `<span>Retry</span></button>` +
          `<button type="button" class="now-card-dismiss" data-action="dismiss" ` +
          `title="Dismiss this failure" aria-label="Dismiss this failure">` +
          `<svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>`;
      }
    } else if (last && last.status === 'stopped' && !s.queue.length
               && last.id !== window._dismissedFailureId) {
      // STOPPED EARLY — a sibling of the failed card, muted rather than red.
      // The user looked at the preview, saw the wrong shot and stopped it;
      // calling that an error would be the panel second-guessing a decision it
      // exists to enable.
      nowCard.classList.remove('idle', 'failed');
      nowCard.classList.add('stopped');
      nowCard.querySelector('.ttl').textContent = 'Stopped early';
      nowCard.querySelector('.meta').innerHTML =
        `<span style="color: var(--muted)">${escapeHtml(snippet(last.params.label || last.params.prompt, 80))}</span>` +
        `<br><span style="color: var(--text)">Nothing was saved.</span>`;
      if (actionsEl) {
        actionsEl.dataset.jobId = String(last.id);
        actionsEl.innerHTML =
          `<button type="button" class="now-card-retry" data-action="retry" ` +
          `title="Run this again with the same params">` +
          `<svg class="ph" aria-hidden="true"><use href="#ph-arrow-clockwise"/></svg>` +
          `<span>Try again</span></button>` +
          `<button type="button" class="now-card-dismiss" data-action="dismiss" ` +
          `title="Dismiss" aria-label="Dismiss"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>`;
      }
    } else {
      nowCard.classList.add('idle');
      nowCard.classList.remove('failed', 'stopped');
      nowCard.querySelector('.ttl').textContent = s.paused ? 'Paused' : 'Idle';
      nowCard.querySelector('.meta').textContent = s.paused
        ? 'Worker paused — current job (if any) finishes, queue waits for resume.'
        : (s.queue.length ? 'Worker about to pick up next queued job.' : 'No jobs queued. Generate something on the left.');
      if (actionsEl) { actionsEl.innerHTML = ''; actionsEl.dataset.jobId = ''; }
    }
  }

  // Y2.001 — bottom-pane live dot mirrors worker state. Tooltip sets
  // a one-line status for accessibility / hover. The dot animates
  // when running, switches red on the failed state.
  const liveDot = document.getElementById('bpLiveDot');
  if (liveDot) {
    liveDot.classList.remove('live', 'failed');
    if (s.running && s.current) {
      liveDot.classList.add('live');
      liveDot.title = 'Rendering';
    } else {
      const last = (s.history || [])[0];
      const showFailure = last && last.status === 'failed' && !s.queue.length;
      if (showFailure) { liveDot.classList.add('failed'); liveDot.title = 'Last job failed'; }
      else if (s.paused) liveDot.title = 'Paused';
      else if (s.queue.length) liveDot.title = `${s.queue.length} queued`;
      else liveDot.title = 'Idle';
    }
  }

  // (The in-player progress overlay used to be wired here; it was
  // removed 2026-05-12 because it duplicated the bottom Now strip and
  // covered the playing video. Bottom strip is the single source.)

  // Logs
  const log = document.getElementById('log');
  const wasNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  const _logText = s.log.length ? s.log.join('\n') : 'No log yet.';
  // Only touch the DOM when the text changed: an unconditional write replaced
  // the text node 40x/min, which destroyed the user's selection — the exact
  // gesture of copying a traceback into a bug report.
  if (log.textContent !== _logText) log.textContent = _logText;
  if (wasNearBottom) log.scrollTop = log.scrollHeight;

  // Queue list — memoized on a signature of the IDs + ordinals so
  // identical queue data doesn't trigger a full innerHTML replacement
  // every 1.5 s. Pre-fix every poll() rebuilt all <li> nodes which
  // caused a perceptible flicker on the Queue tab during long batches.
  const ql = document.getElementById('queueList');
  const qSig = !s.queue.length ? '__empty__'
             : s.queue.map(j => j.id).join('|');
  if (qSig !== window._lastQueueSig) {
    if (!s.queue.length) ql.innerHTML = '<li class="empty-state"><span></span><span>Queue empty</span><span></span><span></span></li>';
    else ql.innerHTML = s.queue.map((j, i) => {
      // Image jobs don't have width/height/frames; show n × aspect instead.
      const params = (j.params.mode === 'image')
        ? `image · ${j.params.aspect || '?'} · n=${j.params.n || '?'}`
        : `${j.params.mode} · ${j.params.width}×${j.params.height} · ${j.params.frames}f`;
      // Which film this job is a shot of. A pure function of immutable params,
      // so the qSig memoisation above needs no change.
      const sb = /^sb:([^#]+)#(\d+)$/.exec(j.params.session_tag || '');
      const sbTotal = sb ? ((SB.boards.find(b => b.id === sb[1]) || {}).shots || '?') : '';
      // A queued shot wears the film badge AND its engine: a film's jobs are
      // not all on one engine, and the queue is where you find out what is
      // about to run. Same chip, same registry row, as the card and the header.
      const sbBadge = sb
        ? `<span class="sb-rowtags"><span class="badge sb-badge" title="${escapeHtml(j.params.label || '')}">S${sb[2].padStart(2,'0')}/${sbTotal}</span>`
          + sbEngineChip(j.params.engine || 'ltx') + `</span>`
        : '';
      return `
      <li>
        <span class="pos">#${i+1}</span>
        <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
        ${sbBadge}
        <span class="params">${params}</span>
        <button title="Remove" onclick="removeJob('${j.id}')"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>
      </li>`;
    }).join('');
    window._lastQueueSig = qSig;
  }

  // History — failed jobs show the error inline in the title slot, so
  // users can see WHY without having to scroll the log to find it.
  // mode='image' jobs render with a thumbnail of the first candidate
  // and an Animate button that pre-fills the i2v form (does NOT
  // auto-submit — user keeps the chance to tweak prompt/seed).
  const hl = document.getElementById('historyList');
  // Apply the All / Videos / Photos filter (set by setRecentFilter).
  const filterPhotos = (window.recentFilter || 'all');
  const filtered = s.history.filter(j => {
    if (filterPhotos === 'all') return true;
    const isPhoto = (j.params && j.params.mode === 'image');
    return filterPhotos === 'photos' ? isPhoto : !isPhoto;
  });
  // Memoized on the same principle as the queue list above: identical data
  // must not trigger an innerHTML replacement every 1.5 s. Unmemoized, the 20
  // <li> (with photo thumbs re-decoded each time) flickered, dropped :hover
  // state, and could swap a Retry button out from under an in-flight click.
  // NOTE: this block is INLINE IN poll(), not a function — an early return
  // here would abort the whole poll (outputs, version, banners). Guard, never
  // return, exactly like the queue memo above.
  const hSig = filterPhotos + '|' + filtered.slice(0, 20)
    .map(j => j.id + '|' + j.status + '|' + (j.output_path || '')).join(';');
  if (window._lastHistorySig !== hSig) {
  window._lastHistorySig = hSig;
  if (!filtered.length) {
    const empty = filterPhotos === 'photos' ? 'No photo renders yet'
                : filterPhotos === 'videos' ? 'No video renders yet'
                : 'No history yet';
    hl.innerHTML = `<li class="empty-state"><span></span><span>${empty}</span><span></span><span></span></li>`;
  }
  else hl.innerHTML = filtered.slice(0, 20).map(j => {
    const titleText = escapeHtml(j.params.label || snippet(_displayPromptFor(j.params.prompt), 60));
    const titleAttr = escapeHtml(_displayPromptFor(j.params.prompt));
    let titleHtml;
    if (j.status === 'failed' && j.error) {
      titleHtml = `${titleText} ` +
        `<span class="err-inline" title="${escapeHtml(j.error)}">— ${escapeHtml(snippet(j.error, 70))}</span>`;
    } else if (j.status === 'done' && j.warning) {
      // A training run that finished WEAK is "done" to the queue and a
      // failure to the person. The row said "done" and nothing else, so the
      // one number that explained it lived only in a log line (#62).
      titleHtml = `${titleText} ` +
        `<span class="warn-inline" title="${escapeHtml(j.warning)}">— ${escapeHtml(snippet(j.warning, 70))}</span>`;
    } else {
      titleHtml = titleText;
    }
    // Image rows get a thumbnail + Animate button; video rows keep
    // the existing 4-column shape so the layout doesn't shift.
    const isPhoto = (j.params && j.params.mode === 'image');
    if (isPhoto && j.status === 'done' && j.output_path) {
      const cands = (j.params.candidate_paths && j.params.candidate_paths.length)
        ? j.params.candidate_paths.length : 1;
      const engineLabel = escapeHtml(_imgEngineLabel(j.params.engine));
      // Reuse the server-built mtime-versioned URL when available
      // (currentOutputs has each entry's `&v=<mtime>` cache-bust),
      // falling back to a plain URL otherwise. Each job's output_path
      // is immutable post-completion, so no Date.now() bust is needed
      // — and it WAS the source of the carousel hot-loop: this list
      // re-renders every ~1s poll tick, so a `&t=${Date.now()}` on
      // every photo row caused ~16 thumbnail re-fetches per tick
      // (~50-60 req/s steady-state idle) that defeated the cache.
      const matchedOutput = (currentOutputs || []).find(x => x.path === j.output_path);
      const thumbSrc = _thumbUrl(matchedOutput
        ? matchedOutput.url
        : `/image?path=${encodeURIComponent(j.output_path)}`, 200);
      // escapeHtml (not a bare " -> &quot; replace) so apostrophes/&/< in an
      // Ideogram caption can't terminate the single-quoted onclick attribute,
      // and pre-fill the readable description rather than the raw caption JSON.
      const animateArgs = escapeHtml(JSON.stringify({
        path: j.output_path, prompt: _displayPromptFor(j.params.prompt)
      }));
      return `
      <li class="${j.status}" data-photo="1">
        <span class="badge">photo</span>
        <span class="ttl photo-ttl" title="${titleAttr}">
          <img class="photo-thumb" src="${thumbSrc}" alt=""
               style="width:36px;height:36px;object-fit:cover;border-radius:4px;vertical-align:middle;margin-right:6px">
          ${titleHtml}
          <span class="photo-meta hint" style="margin-left:6px;font-size:11px;opacity:0.7">${cands}× · ${engineLabel}</span>
        </span>
        <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
        <span><button class="animate-btn" type="button"
              onclick='animateFromPhoto(${animateArgs})' title="Pre-fill i2v with this image (does not auto-submit)">Animate</button></span>
      </li>`;
    }
    // Failed jobs get a Retry button in the action column. Clicking it
    // re-submits the same params via /queue/add so the user doesn't have
    // to rebuild the form. Cancelled jobs get the same treatment — a
    // cancellation often means "wrong intent, try again with edits" but
    // sometimes "ran out of RAM, want to retry as-is."
    const isRetryable = j.status === 'failed' || j.status === 'cancelled';
    const actionHtml = isRetryable
      ? `<button class="retry-btn" type="button"
                 title="Re-submit this job with the same params"
                 onclick='retryJob(${JSON.stringify(j.id)})'>Retry</button>`
      : '';
    // Same film badge the queue rows carry, so a shot is identifiable
    // wherever the bottom pane shows it.
    const hsb = /^sb:([^#]+)#(\d+)$/.exec((j.params || {}).session_tag || '');
    const hsbTotal = hsb ? ((SB.boards.find(b => b.id === hsb[1]) || {}).shots || '?') : '';
    const hsbBadge = hsb
      ? `<span class="sb-rowtags"><span class="badge sb-badge" title="${escapeHtml(j.params.label || '')}">S${hsb[2].padStart(2,'0')}/${hsbTotal}</span>`
        + sbEngineChip((j.params || {}).engine || 'ltx') + `</span>`
      : '';
    return `
    <li class="${j.status}">
      <span class="badge">${j.status}</span>
      <span class="ttl" title="${titleAttr}">${titleHtml}</span>
      ${hsbBadge}
      <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
      <span>${actionHtml}</span>
    </li>`;
  }).join('');
  }

  // Outputs / carousel
  if (JSON.stringify(currentOutputs) !== JSON.stringify(s.outputs)) {
    currentOutputs = s.outputs;
    if (typeof updateModelCredit === 'function') { try { updateModelCredit(); } catch (e) {} }
    renderCarousel();
    // Pick the first FILTERED entry as the default selection so the
    // viewer agrees with what the gallery is showing — without this,
    // landing on Photos with no active selection would auto-pick a
    // video and fight the filter.
    // A first-ever render can finish while activePath is still null because
    // the live stage, not an output, owns the player. Let renderLiveStage()
    // below perform its preview-backed handoff; auto-selecting here would
    // destroy the last estimate one line before the seamless swap can use it.
    if (stageMayAutoSelectOutput()) {
      const visible = filteredMainOutputs();
      if (visible.length) selectOutput(visible[0].path, { autoplay: false });
    }
    // If the saved filter (from localStorage) selects a kind that's not
    // in /status's top-60, the carousel is empty on boot and the user
    // never clicks the chip (it's already active). Kick the auto-fetch
    // here so a hard-refresh with a saved Photos filter still populates.
    // _maybeAutoLoadAllForEmptyFilter is idempotent and re-entrant safe
    // via outputsLoadAll's _outputsLoadAllInFlight guard.
    if (typeof _maybeAutoLoadAllForEmptyFilter === 'function') {
      _maybeAutoLoadAllForEmptyFilter(mainOutputsFilter);
    }
    // The Extend source dropdown is video-only — Extend mode operates on
    // mp4 input. Filter to videos so the user can't accidentally pick a
    // .png as an Extend source (which would 400 server-side).
    const sel = document.getElementById('extendSrcSelect');
    const videoOutputs = currentOutputs.filter(o => !isPhotoOutputMain(o));
    const _videoOpts = '<option value="">— pick an output below or paste a path —</option>' +
      videoOutputs.slice(0, 40).map(o => `<option value="${escapeHtml(o.path)}">${escapeHtml(o.name)}</option>`).join('');
    sel.innerHTML = _videoOpts;
    // Colorize (restore) source dropdown — same video-only list as Extend.
    const restoreSel = document.getElementById('restoreSrcSelect');
    if (restoreSel) restoreSel.innerHTML = _videoOpts;
    // Upscale ×2 source dropdown — same video-only list.
    const upscaleSel = document.getElementById('upscaleSrcSelect');
    if (upscaleSel) upscaleSel.innerHTML = _videoOpts;
    // Control (Union) control-video dropdown — same video-only list.
    const controlSel = document.getElementById('controlSrcSelect');
    if (controlSel) controlSel.innerHTML = _videoOpts;
  }
  // OUTSIDE #queueList's qSig memoisation, and after currentOutputs refreshes:
  // a completed job can therefore hand off from its last preview frame to the
  // newly-listed mp4 in this same poll without rebuilding the queue DOM.
  renderLiveStage(s, livePreviewData);
  // (The old "Hidden (N)" pill that lived here was retired with the
  // Visible/Hidden segmented control — the carousel-head comment above
  // documents the removal. Setting textContent on the missing element
  // was firing a TypeError every poll cycle.)
  // Title: count is post-filter so the badge agrees with the rendered
  // cells. "Outputs · 23 photos" when Photos is active, plain "Outputs · N"
  // when All. Hidden override stays unchanged.
  const _visible = filteredMainOutputs();
  document.getElementById('carouselTitle').textContent =
    filterMode === 'hidden' ? 'Hidden outputs' : outputsTitleText();

  // "Show all (N)" button — reveal whenever the server reports more
  // outputs total than the polling fast path returned, and the user
  // hasn't already loaded the full list. Hides itself again once
  // outputsLoadAll() has merged the older entries into _olderOutputs.
  const _showAllBtn = document.getElementById('outputsShowAllBtn');
  if (_showAllBtn) {
    const _total = typeof s.outputs_total === 'number' ? s.outputs_total : (s.outputs ? s.outputs.length : 0);
    const _polledShown = currentOutputs.length;
    const _hasOlder = _total > _polledShown;
    if (window._showingAllOutputs || !_hasOlder) {
      _showAllBtn.style.display = 'none';
    } else {
      _showAllBtn.textContent = `Show all (${_total})`;
      _showAllBtn.style.display = '';
    }
  }

  // Train Character integration: detect newly-completed train jobs and
  // refresh both the trained-LoRA list AND the global LoRA picker so a
  // freshly-trained character appears in T2V/I2V's picker without a manual
  // reload. We watch history for entries with mode === 'train' that
  // weren't there last poll; cheap and survives the user being on any
  // tab/mode at the time training finishes.
  if (Array.isArray(s.history)) {
    const trainDoneCount = s.history.filter(h =>
      h && h.params && h.params.mode === 'train' && h.status === 'done'
    ).length;
    if (typeof window._lastTrainDoneCount === 'undefined') {
      window._lastTrainDoneCount = trainDoneCount;
    } else if (trainDoneCount > window._lastTrainDoneCount) {
      window._lastTrainDoneCount = trainDoneCount;
      // New train job(s) completed since last poll — re-sync everything.
      if (typeof refreshLoras === 'function') {
        try { refreshLoras(); } catch (e) {}
      }
      if (typeof trainRefreshLoraList === 'function') {
        try { trainRefreshLoraList(); } catch (e) {}
      }
      // Also refresh the Manual-tab Characters picker — a newly-finished
      // character training drops a bundle.json into mlx_models/characters/,
      // and the picker should pick it up without a manual page reload.
      if (typeof refreshManualCharacters === 'function') {
        try { refreshManualCharacters(); } catch (e) {}
      }
    }
  }

  // Storyboard — ONE call, no new timer. It sets the tab count during an
  // overnight run, gates Plan film on the worker being idle, and decides
  // whether `To film` exists at all. All three are inert with no boards.
  if (typeof sbPollHook === 'function') { try { sbPollHook(s); } catch (e) {} }
}

// ---- Layer 1 of the pack-incomplete story, and the one that matters ---------
//
// `ltx_pack_preflight()` already refuses a render whose weights are incomplete
// and names the file. It is the right refusal in the wrong PLACE: it raises
// mid-render, after Gemma has loaded and the user has watched a progress bar
// for half a minute. (Worse, on dev it had been dead code since 2.5 became the
// default — see DEFECT-1 — so the actual behaviour was the June-2026 rainbow
// mosaic with no error at all.)
//
// /status already carries q8_missing, hq_addon_missing and base_missing. So the
// same question the server asks at job time is asked here every poll, against
// the tier the form is actually pointing at, and the answer disables Generate
// with the filenames in the note instead of spending the user's minute first.
//
// Scoped to the SELECTED tier on purpose. A T2V user on Balanced is not nagged
// about weights High would need — the existing #engineRowNote contract is "the
// one-line reason a gate fired", not a standing inventory.
function applyPackIncompleteGate(s) {
  const note = document.getElementById('engineRowNote');
  const genBtn = document.getElementById('genBtn');
  if (!note || !genBtn) return;
  // H3 has its own install card and its own gates; this is the LTX lane.
  const engine = document.body.dataset.engine || 'ltx';
  const clear = () => {
    if (genBtn.dataset.packBlocked === '1') {
      genBtn.disabled = false;
      genBtn.removeAttribute('title');
      delete genBtn.dataset.packBlocked;
    }
    if (note.dataset.packNote === '1') {
      note.hidden = true; note.innerHTML = '';
      delete note.dataset.packNote;
    }
  };
  if (engine !== 'ltx') { clear(); return; }
  let cell = (typeof ltxCurrentCell === 'function') ? ltxCurrentCell() : null;
  const charId = (document.getElementById('characterIdInput') || {}).value || '';
  const charChip = charId
    ? document.querySelector('#qualityGroupCharacter .char-quality.active') : null;
  if (charChip) {
    cell = {
      pack: charChip.dataset.pack || 'q8',
      pipeline: charChip.dataset.pipeline || '',
      quality_label: charChip.querySelector('.ql-name')?.textContent || 'This tier',
    };
  }
  // The cell knows its pack. Falling back to the raw quality field keeps this
  // correct before the table has rendered, and on any surface that sets
  // #quality directly.
  const pack = (cell && cell.pack)
    || (_qualityUsesHq((document.getElementById('quality') || {}).value) ? 'q8' : 'q4');
  // Only the q8 lane can be half-installed in a way the user can act on: an
  // incomplete BASE pack is already a hard block in the models card above, and
  // duplicating it here would give the same fact two voices.
  if (pack !== 'q8') { clear(); return; }
  const missing = [].concat(
    s.q8_missing || [],
    cell && cell.pipeline === 'hq' ? (s.hq_addon_missing || []) : [],
  );
  if (!missing.length) { clear(); return; }
  // A pack with NOTHING installed is an install offer, not an incomplete
  // download — the High chip's .needs-install state and the Models modal
  // already say so, and "finish the download" would be the wrong verb.
  if (!s.q8_available && missing.length > 6) { clear(); return; }
  const label = (cell && cell.quality_label) || 'This tier';
  // Relative, as §3.6's verbatim has it: an absolute path is this machine's
  // detail, and the note has to fit on one line beside the Generate button.
  const dir = String(s.q8_path || 'the Q8 pack').replace(/^.*?(?=mlx_models\/)/, '');
  const shown = missing.slice(0, 3).join(', ') + (missing.length > 3 ? ' …' : '');
  note.innerHTML = escapeHtml(
      `${label} needs ${missing.length} more file(s) in ${dir} — ${shown}. `
      + `Finish the download first.`)
    + ` <a href="#" onclick="openModelsModal();return false;">Finish the download →</a>`;
  note.hidden = false;
  note.dataset.packNote = '1';
  genBtn.disabled = true;
  genBtn.title = "The weights this tier needs aren't all there yet.";
  genBtn.dataset.packBlocked = '1';
}

// DELETED in v4.0: window.refreshBalancedSubtitle.
//
// It wrote the Balanced chip's third slot ("Q4 · 5 min" / "Q8 Fast · 6 min" /
// "Q4 fallback") from a rule it re-derived in JS, and it wired itself to
// #frames input and wrapped setMode() to stay current. Every number in it was
// a 2.3 figure that no measurement on 2.5 supports, and the wrapper made
// setMode a function two files had opinions about.
//
// The tier table fills that slot now, for all four chips, from a cost model
// anchored on measured renders — the same mechanism H3's chips have always
// used. One writer, one source of truth, and re-pricing is a Python edit.

// Recent-tab type filter (All / Videos / Photos). Stored on window so
// the value survives across renders without polluting the existing
// `filterMode` global (which is for the hidden/visible carousel filter,
// a different concept). Defaults to 'all'.
window.recentFilter = window.recentFilter || 'all';
function setRecentFilter(mode) {
  window.recentFilter = mode;
  document.getElementById('recentFilterAll').classList.toggle('active', mode === 'all');
  document.getElementById('recentFilterVideos').classList.toggle('active', mode === 'videos');
  document.getElementById('recentFilterPhotos').classList.toggle('active', mode === 'photos');
  // Re-poll so the filter's effect is visible immediately. /status is
  // a localhost no-op, so the user perceives no latency.
  poll();
}

// Right-pane gallery filter — All / Videos / Photos, persisted via
// localStorage so picking "Photos" once sticks across reloads. Photo
// discrimination uses params.mode === 'image' first, then falls back
// to filename suffix for older sidecar-less entries.
try {
  const stored = localStorage.getItem('phos_outputs_filter');
  if (stored === 'all' || stored === 'videos' || stored === 'photos') {
    window.outputsFilter = stored;
  }
} catch (e) {}
window.outputsFilter = window.outputsFilter || 'all';
function setOutputsFilter(mode) {
  if (mode !== 'all' && mode !== 'videos' && mode !== 'photos') mode = 'all';
  window.outputsFilter = mode;
  try { localStorage.setItem('phos_outputs_filter', mode); } catch (e) {}
  const a = document.getElementById('outputsFilterAll');
  const v = document.getElementById('outputsFilterVideos');
  const p = document.getElementById('outputsFilterPhotos');
  if (a) a.classList.toggle('active', mode === 'all');
  if (v) v.classList.toggle('active', mode === 'videos');
  if (p) p.classList.toggle('active', mode === 'photos');
}
// Apply the persisted filter on first paint so the chip styling
// matches the live `window.outputsFilter` before the user clicks.
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('outputsFilterAll')) {
    setOutputsFilter(window.outputsFilter || 'all');
  }
  // Same first-paint sync for the main right-pane chip strip — the
  // localStorage value was loaded into `mainOutputsFilter` at script
  // top, but the chip DOM hadn't been built yet, so the .active class
  // is still on All. Re-apply now that the strip exists.
  if (document.getElementById('mainOutputsFilterAll')) {
    _updateMainFilterChips();
  }
});

// Animate button — pre-fills the i2v form with the given still as the
// reference image and the same prompt. Does NOT auto-submit; the user
// keeps the chance to tweak prompt/seed/quality before clicking
// Generate. Reversible (changing the image picker resets it).
async function retryJob(jobId) {
  // Re-submit a failed/cancelled job by its id. Server side endpoint
  // /queue/retry takes the original job's id, copies its params verbatim
  // into a fresh queue entry, and returns the new id. Toast on success
  // (the Now/Queue pane will pick the new entry up on the next poll).
  if (!jobId) return;
  try {
    const fd = new URLSearchParams();
    fd.set('id', jobId);
    const r = await fetch('/queue/retry', { method: 'POST', body: fd });
    let data = {};
    try { data = await r.json(); } catch (e) { /* keep empty */ }
    if (!r.ok || !data.ok) {
      alert('Retry failed: ' + (data.error || ('HTTP ' + r.status)));
      return;
    }
    // Force one immediate poll so the queue badge updates without the
    // 1.5s wait. poll() is idempotent.
    if (typeof poll === 'function') poll();
  } catch (e) {
    alert('Retry error: ' + (e.message || 'unknown'));
  }
}

// Re-run a previously-generated still through HiDream Quality (the
// 12-step + light FBCache recipe). Reads the source's sidecar to pull
// the original prompt + seed + refs, then drives the Image Studio form
// (engine swapped to hidream_quality_inline, n=1) and submits to the
// queue. Used from the photo card hover button.
//
// Why no manual confirmation step: Mr Bizarro's intent is "pick best of 4
// Fast/Medium previews → bake one at Quality" — adding a dialog would
// burn a click for no information. The Image Studio form is left
// pre-filled so the user can tweak + resubmit if they want.
async function remakeInQuality(payload) {
  if (!payload || !payload.path) {
    if (typeof phosToast === 'function') {
      phosToast('Remake: missing path', { kind: 'danger' });
    }
    return;
  }
  // Pull the source's generation params from its sidecar JSON.
  let sidecar;
  try {
    const r = await fetch('/sidecar?path=' + encodeURIComponent(payload.path));
    if (!r.ok) throw new Error('no sidecar (older output?)');
    sidecar = await r.json();
  } catch (e) {
    if (typeof phosToast === 'function') {
      phosToast('Remake: ' + (e.message || 'failed to read sidecar'),
                { kind: 'danger' });
    }
    return;
  }
  // Switch to Image Studio mode so the form is visible.
  if (typeof setMode === 'function') setMode('image');

  // Pre-fill the form. Prompt + seed + aspect carry over verbatim.
  // Refs go into the studio's 3-slot model.
  const promptEl = document.getElementById('imgStudioPrompt');
  if (promptEl) promptEl.value = sidecar.prompt || '';
  const seedEl = document.getElementById('imgStudioSeed');
  if (seedEl && sidecar.seed != null) seedEl.value = String(sidecar.seed);
  const nEl = document.getElementById('imgStudioN');
  if (nEl) nEl.value = '1';   // remake just the picked one
  const engineEl = document.getElementById('imgStudioEngine');
  if (engineEl) engineEl.value = 'hidream_quality_inline';
  const aspectEl = document.getElementById('imgStudioAspect');
  if (aspectEl && sidecar.aspect) aspectEl.value = sidecar.aspect;

  // Refs — clear the 3 slots, then re-populate from sidecar.refs.
  if (typeof IMG_STUDIO !== 'undefined' && IMG_STUDIO.refs) {
    for (let i = 0; i < IMG_STUDIO.refs.length; i++) {
      IMG_STUDIO.refs[i] = null;
      if (typeof imgStudioRenderSlot === 'function') imgStudioRenderSlot(i);
    }
    const refs = Array.isArray(sidecar.refs) ? sidecar.refs : [];
    refs.slice(0, IMG_STUDIO.refs.length).forEach((path, i) => {
      const fname = String(path).split('/').pop();
      IMG_STUDIO.refs[i] = { path, name: fname };
      if (typeof imgStudioRenderSlot === 'function') imgStudioRenderSlot(i);
    });
  }

  // Refresh validity + estimate UI so the Generate button enables.
  if (typeof imgStudioUpdateValidity === 'function') imgStudioUpdateValidity();
  if (typeof imgStudioUpdateEstimate === 'function') imgStudioUpdateEstimate();
  if (typeof imgStudioRefreshEngineStatus === 'function') imgStudioRefreshEngineStatus();

  // Auto-submit. imgStudioGenerate reads the now-prefilled fields.
  if (typeof imgStudioGenerate === 'function') {
    if (typeof phosToast === 'function') {
      phosToast('Remaking in Quality (12-step) · watch Recent → Photos',
                { kind: 'success' });
    }
    imgStudioGenerate();
  }
}

// Ideogram 4 stores params.prompt as a structured caption JSON string (root
// keys high_level_description / style_description / compositional_deconstruction)
// because mflux reads it from a --prompt-file verbatim. Every UI surface that
// shows a prompt or pre-fills it into a PLAIN box (Animate→i2v, titles) wants a
// readable string, never the raw JSON. Parse it and prefer high_level_description;
// any other engine's plain-text prompt just round-trips unchanged.
function _parseIdeoCaption(s) {
  if (typeof s !== 'string') return null;
  const t = s.trim();
  if (t.charAt(0) !== '{') return null;
  try {
    const o = JSON.parse(t);
    if (o && typeof o === 'object' && typeof o.high_level_description === 'string') return o;
  } catch (_) {}
  return null;
}
function _displayPromptFor(prompt) {
  const cap = _parseIdeoCaption(prompt);
  return cap ? cap.high_level_description : (prompt || '');
}
// Friendly engine label from the internal token stamped on image jobs
// (params.engine = "mflux/ideogram" etc.). Falls back to the raw token.
function _imgEngineLabel(token) {
  const map = {
    'mflux/ideogram': 'Ideogram 4', 'mflux/hidream': 'HiDream',
    'mflux/qwen_edit': 'Qwen-Edit', 'mflux/flux2': 'FLUX.2', 'mflux': 'mflux'
  };
  return map[token] || (token || 'image');
}

function animateFromPhoto(payload) {
  if (!payload || !payload.path) return;
  // Leave the Studio/Ideogram workflow first — otherwise body[data-workflow]
  // stays "studio" and its CSS keeps the video form (#genForm) hidden, so
  // setMode('i2v') alone would look like nothing happened. Back to Video.
  if (typeof workflowSwitch === 'function') workflowSwitch('manual');
  // Switch to i2v mode (pill + form fields). setMode('i2v') hides the
  // Studio pane, shows the video form, and applies the i2v-specific
  // dropdown selection.
  setMode('i2v');
  // Use the existing image picker helper so the preview tile updates
  // alongside the hidden form field. Pass snapAspect:false to keep
  // whatever aspect the user already had selected (matches loadParams).
  if (typeof pickerSetImage === 'function') {
    pickerSetImage('image', payload.path, { snapAspect: false });
  } else {
    document.getElementById('image').value = payload.path;
  }
  if (payload.prompt) {
    document.getElementById('prompt').value = payload.prompt;
  }
  // Scroll the form pane to the top so the image picker + prompt are
  // immediately visible.
  const formPane = document.querySelector('aside.form-pane');
  if (formPane) formPane.scrollTop = 0;
  if (typeof updateDerived === 'function') updateDerived();
  if (typeof updateCustomizeSummary === 'function') updateCustomizeSummary();
}

// Format render duration for the gallery card sub-line. Falls back to
// the time-of-day when the sidecar is missing (older outputs that
// pre-date the elapsed_sec field, or outputs whose sidecar got
// deleted) so the slot is never empty.
function _outputDurationLabel(o) {
  // Lead with what the file IS (its length), then how long it took, labeled.
  // "1 h 20 m" alone on a 10-second clip read as a broken duration
  // (Mr Bizarro 2026-08-10: "preview is not accurate") — same confusion as
  // the 2026-05-21 mtime incident, opposite direction. Numbers get names.
  const clip = (o && typeof o.clip_sec === 'number' && o.clip_sec > 0)
    ? (o.clip_sec >= 60 ? `${Math.floor(o.clip_sec / 60)}m ${Math.round(o.clip_sec % 60)}s clip`
                        : `${Math.round(o.clip_sec)}s clip`)
    : null;
  const s = (o && typeof o.elapsed_sec === 'number') ? o.elapsed_sec : null;
  if (clip && s != null) return `${clip} · ${_fmtRenderTime(s)}`;
  if (clip) return clip;
  if (s == null) {
    // No render-elapsed in sidecar — show a relative timestamp instead
    // of HH:MM, which looked like a render duration and was confusing
    // (Mr Bizarro 2026-05-21: pointed at a "21:19" label that looked
    // like a 21-minute render but was actually the file's wall-clock
    // mtime). Empty card is better than misleading.
    return (o && o.mtime && typeof _relTimeFromMtime === 'function')
      ? _relTimeFromMtime(o.mtime) : '—';
  }
  return _fmtRenderTime(s);
}

function _fmtRenderTime(s) {
  if (s < 60)    return `rendered ${Math.round(s)} s`;
  if (s < 3600)  return `rendered ${Math.floor(s / 60)} m ${Math.round(s % 60)} s`;
  return `rendered ${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} m`;
}

function renderCarousel() {
  const el = document.getElementById('carousel');
  const visible = filteredMainOutputs();
  if (!visible.length) {
    const q = (typeof outputsQueryText === 'function') ? outputsQueryText() : '';
    const msg = q ? ('No matches for \u201c' + escapeHtml(q) + '\u201d.')
              : mainOutputsFilter === 'photos' ? 'No photo outputs yet.'
              : mainOutputsFilter === 'videos' ? 'No video outputs yet.'
              : 'No outputs in this view yet.';
    el.innerHTML = `<div class="empty-msg">${msg}</div>`;
    return;
  }
  // PERF: cap rendered DOM cards. The auto-fetch (d29de9c) started
  // landing the full /outputs list (~600+ entries) into the carousel
  // on first poll, which made Chrome hold the entire DOM tree resident
  // — Mr Bizarro saw ~10 GB Chrome RSS + 62 GB swap during renders
  // (2026-05-20). 108d41b deferred video metadata fetches via
  // IntersectionObserver but didn't shrink the DOM itself.
  //
  // Cap at CAROUSEL_RENDER_CAP visible cards (240 — generous for normal
  // scroll, ~94% smaller than the 658-card worst case). When more
  // exist, append a "Show N more" button at the end. Click bumps the
  // cap by another batch and re-renders. The cap is per-render — it
  // resets to default when the filter changes or new outputs land.
  const CAROUSEL_RENDER_CAP = 240;
  const _renderLimit = window._carouselRenderLimit || CAROUSEL_RENDER_CAP;
  const _visibleSlice = visible.slice(0, _renderLimit);
  const _hiddenCount = visible.length - _visibleSlice.length;
  el.innerHTML = _visibleSlice.map(o => {
    const pathAttr = JSON.stringify(o.path).replace(/"/g, '&quot;');
    const isPhoto = isPhotoOutputMain(o);
    // Thumbnail markup branches on kind. Videos use <video> with a
    // mid-clip seek (2.5s — LTX clips are 5s at 24fps and the first
    // half-second is often a dark fade-in, so seeking to the middle
    // gets a representative frame). Photos use <img> directly with
    // the same /image?path=… cache-bust URL the server stamped.
    //
    // PERF: with the auto-fetch landing 658+ entries into the carousel,
    // `preload="metadata"` on every <video> stalled the page — each
    // metadata fetch downloads the moov atom + enough bytes to render
    // the t=2.5 poster frame (~hundreds of KB), and 586 of those
    // saturate the browser's 6-connection limit. The user's click-to-
    // play request then queues behind ~580 thumbnail fetches and looks
    // "stuck."
    //
    // Fix: ship the <video> with `data-src` instead of `src`. An
    // IntersectionObserver (wired below renderCarousel) promotes
    // `data-src` to `src` only when the card is within the viewport's
    // ~2-screen-tall preload margin. Off-screen cards stay completely
    // dormant. <img> already has loading="lazy" so it's fine; we keep
    // the existing markup for photos.
    const thumbHtml = isPhoto
      ? `<img class="car-thumb" src="${_thumbUrl(o.url, 480)}" alt="${escapeHtml(o.name)}" loading="lazy">`
      // Hover-scrub: on enter, jump to 0 and play silently at 0.6×;
      // on leave, pause + snap back to the static 2.5s preview frame.
      // The play() promise can reject during a fast scrub (browser
      // says "play interrupted by pause") — swallow it.
      // src is deferred — the IntersectionObserver below promotes
      // data-src → src when the card scrolls into view. preload stays
      // metadata so once src is set, the t=2.5 poster frame renders
      // without the user having to hover.
      : `<video data-src="${o.url}#t=2.5" preload="metadata" muted playsinline
                onmouseenter="if (!this.src && this.dataset.src) this.src = this.dataset.src; this.currentTime=0; this.playbackRate=0.6; this.play().catch(()=>{})"
                onmouseleave="this.pause(); this.currentTime=2.5; this.playbackRate=1"></video>`;
    // Per-card actions (revealed on hover) — kept deliberately minimal:
    //   * Photos get a small "Animate" chip (turns the still into i2v).
    //   * Everything gets a delete (×) chip.
    // The previous Hide + Extend buttons were dropped per Mr Bizarro:
    //   - Hide was useless (clutter, never used in practice).
    //   - Extend wasn't practical on Mac.
    // Folder-reveal moved to a global button in the carousel header.
    const animateArgs = JSON.stringify({path: o.path, prompt: ''}).replace(/"/g, '&quot;');
    const animateChip = isPhoto
      ? `<button class="card-action card-action-photo" type="button"
                 title="Pre-fill i2v with this image (does not auto-submit)"
                 onclick="event.stopPropagation(); animateFromPhoto(${animateArgs})">Animate</button>`
      : '';
    // Remake-in-Quality chip — only on photos that have a sidecar (we need the
    // original prompt + seed + refs to re-run). Pre-fills the Image Studio
    // form with the source's exact params, swaps the engine to
    // hidream_quality_inline, sets n=1, and auto-submits. Useful flow:
    // generate 4 candidates in Fast/Medium, pick the best, hit ✦ to bake the
    // same composition at Quality fidelity.
    const remakeArgs = JSON.stringify({path: o.path}).replace(/"/g, '&quot;');
    const remakeChip = (isPhoto && o.has_sidecar)
      ? `<button class="card-action card-action-photo" type="button"
                 title="Re-run this prompt + seed + refs through HiDream Quality (auto-submits)"
                 onclick="event.stopPropagation(); remakeInQuality(${remakeArgs})">✦ Quality</button>`
      : '';
    return `
    <div class="car-card${o.path === activePath ? ' active' : ''}"
         data-path="${escapeHtml(o.path)}" onclick="selectOutput(${pathAttr})">
      <div class="car-thumb-wrap">
        ${thumbHtml}
        ${o.has_sidecar
          ? `<button class="car-info-btn" type="button" title="Show generation info"
                     onclick="event.stopPropagation(); openOutputInfoModal(${pathAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-info"/></svg></button>`
          : ''}
        <div class="card-chrome">
          ${remakeChip}
          ${animateChip}
          <button class="card-action card-action-danger" type="button" title="Delete this file from disk"
                  onclick="event.stopPropagation(); deleteOutput(${pathAttr})"><svg class="ph" aria-hidden="true"><use href="#ph-trash-simple"/></svg></button>
        </div>
      </div>
      <div class="info">
        <div class="name" title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</div>
        <div class="sub" title="Render time · file size">
          ${o.sb ? `<span class="badge sb-badge" title="Shot ${o.sb.n} of a storyboard — click to open it"
                 onclick="event.stopPropagation(); sbOpenFromClip('${escapeHtml(o.sb.id)}')">S${String(o.sb.n).padStart(2,'0')}</span> · ` : ''}${_outputDurationLabel(o)} · ${o.size_mb.toFixed(1)} MB
        </div>
      </div>
    </div>`;
  }).join('');
  // Append the "Show more" trailer when the render was capped. Each
  // click bumps the cap by another full batch (240 cards), so users
  // with thousands of outputs still pay only ~240 cards' worth of DOM
  // per page-load by default. The trailer's onclick is inline so we
  // don't need a separate listener wiring step.
  if (_hiddenCount > 0) {
    el.insertAdjacentHTML('beforeend',
      `<div class="carousel-more-trailer" style="grid-column:1/-1;display:flex;justify-content:center;padding:14px 8px 6px;">
         <button type="button" class="ghost-btn" onclick="window._carouselRenderLimit=(window._carouselRenderLimit||${CAROUSEL_RENDER_CAP})+${CAROUSEL_RENDER_CAP};renderCarousel();">
           Show ${Math.min(_hiddenCount, CAROUSEL_RENDER_CAP)} more · ${_hiddenCount} hidden
         </button>
       </div>`);
  }
  // Lazy-load the just-rendered video thumbnails. <img> entries already
  // have native loading="lazy" but <video> has no equivalent, so we
  // observe each carousel card and promote `data-src` → `src` only when
  // it crosses into a 2-screen-tall preload margin. Without this, all
  // 586 videos start downloading their poster-frame bytes at once when
  // _showingAllOutputs is true, and the user's click-to-play stalls
  // behind the queue. The observer is single-shot per card (unobserve
  // after promotion) and uses a shared instance reset on each render
  // so previously-observed nodes (now detached) get GC'd.
  if (window._carThumbObserver) {
    try { window._carThumbObserver.disconnect(); } catch (_e) {}
  }
  window._carThumbObserver = new IntersectionObserver((entries, obs) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const v = e.target.querySelector('video[data-src]');
      if (v && !v.src) v.src = v.dataset.src;
      obs.unobserve(e.target);
    }
  }, { root: el, rootMargin: '200% 0px 200% 0px', threshold: 0 });
  el.querySelectorAll('.car-card').forEach(c => {
    if (c.querySelector('video[data-src]:not([src])')) {
      window._carThumbObserver.observe(c);
    }
  });
}

// Relative-time helper for the player overlay meta line. Takes the
// server's "YYYY-MM-DD HH:MM:SS" mtime string and returns a humanlike
// "2 min ago" / "3 h ago" / "yesterday" so the meta strip carries
// recency info without forcing the user to do date math.
function _relTimeFromMtime(mtime) {
  if (!mtime) return '';
  // Treat the server timestamp as local time (it's the panel host's
  // wall clock). Date parses "YYYY-MM-DD HH:MM:SS" inconsistently
  // across browsers, so re-format to ISO-with-T which all engines
  // accept as local.
  const iso = String(mtime).replace(' ', 'T');
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return mtime;
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 45)    return 'just now';
  if (sec < 90)    return '1 min ago';
  if (sec < 3600)  return Math.round(sec / 60) + ' min ago';
  if (sec < 5400)  return '1 h ago';
  if (sec < 86400) return Math.round(sec / 3600) + ' h ago';
  if (sec < 172800) return 'yesterday';
  return Math.round(sec / 86400) + ' d ago';
}

// Helper — look up an output entry by path across BOTH the polled
// top-60 (currentOutputs) and the older pool loaded via /outputs (when
// Show all or the auto-fetch has fired). Three callers needed this
// after d29de9c started surfacing older photos automatically — without
// it they hit the currentOutputs-only path, which returns undefined
// for any older entry and falls through to broken codepaths:
// - selectOutput: falls back to /file?path=, wraps a PNG in <video>,
//   browser stalls trying to play a still image as video (the "super
//   slow preview" report immediately after d29de9c shipped).
// - openExpandLightbox: bails out, lightbox never opens.
// - animateActive: animates with no sidecar prompt for older photos.
function findOutputByPath(path) {
  let o = currentOutputs.find(x => x.path === path);
  if (!o && window._olderOutputs && window._olderOutputs.length) {
    o = window._olderOutputs.find(x => x.path === path);
  }
  return o || null;
}
// Persistent stage mute (fuschichou, Pinokio, 2026-09-01). Every finished
// render lands in a FRESH <video> element, so muting via the native control
// only ever silenced that one clip — the next overnight completion started
// audible again. The native mute button is the toggle; this makes it stick:
// each new stage/lightbox video starts in the last chosen state, and any
// mute/unmute the user makes is written back.
const STAGE_MUTE_KEY = 'ph_stage_muted';
function _stageMutePreferred() {
  try { return localStorage.getItem(STAGE_MUTE_KEY) === '1'; } catch (e) { return false; }
}
function _wireStageMutePersistence(video) {
  if (!video) return;
  video.muted = _stageMutePreferred();
  video.addEventListener('volumechange', () => {
    try { localStorage.setItem(STAGE_MUTE_KEY, video.muted ? '1' : '0'); } catch (e) {}
  });
}

function stageMayAutoSelectOutput() {
  return !activePath && !window._liveStageOwnsPlayer;
}
function selectOutput(path, options) {
  options = options || {};
  // AUTOPLAY IS FOR A CLICK, NOT FOR BOOT. The stage selects the newest
  // output on load and after every refresh or filter change; those used to
  // build the player with `autoplay`, so the panel opened already playing
  // the last clip, sound and all. A selection the person did not make shows
  // the clip paused on its first frame; a click plays.
  const autoplay = options.autoplay !== false;
  activePath = path;
  const _uev = (typeof window !== 'undefined') ? window.event : null;
  const userSelected = !!(_uev && _uev.isTrusted);
  if (userSelected) window._stagePlaybackIntentAt = Date.now();
  // The credit names the weights that made THIS clip.
  if (typeof updateModelCredit === 'function') { try { updateModelCredit(path); } catch (e) {} }
  // If the Ideogram editor canvas is holding the stage, a USER click on an
  // output means they want to see the render — flip the stage to Result.
  // Gate on isTrusted so the boot-time auto-select of the newest output (and
  // any other programmatic selection) can't yank someone out of mid-edit;
  // the Generate flow flips explicitly in imgStudioGenerate.
  if (userSelected &&
      typeof ideoInLayout === 'function' && ideoInLayout() &&
      typeof stageSetMode === 'function') {
    stageSetMode('result');
  }
  document.querySelectorAll('.car-card').forEach(el => el.classList.toggle('active', el.dataset.path === path));
  const wrap = document.getElementById('playerWrap');
  // A user can pick an output while the forming take owns the stage. Preserve
  // the render in status/Now, but immediately yield the hero to the explicit
  // click; the next poll offers the small LIVE return chip instead of stealing
  // the clip back.
  const liveBackdrop = options.liveHandoff
    ? ((wrap.querySelector('.live-stage-image') || {}).src || '') : '';
  window._liveStageOwnsPlayer = false;
  const liveOverlay = document.getElementById('liveStageOverlay');
  const liveChip = document.getElementById('liveReturnChip');
  if (liveOverlay) liveOverlay.hidden = true;
  if (liveChip) liveChip.hidden = true;
  wrap.classList.remove('live-stage', 'is-warming', 'is-aborting');
  delete wrap.dataset.liveJobId;
  delete wrap.dataset.liveState;
  wrap.classList.remove('empty');
  // Y1.039 — use the server-provided URL (which includes the mtime
  // cache-bust v=N param) instead of reconstructing from path. Otherwise
  // the player ends up on the cached stale-bytes URL and re-shows black
  // until the browser cache expires.
  const o = findOutputByPath(path);
  const isPhoto = isPhotoOutputMain(o);
  // Photo entries don't go through /file (which is OUTPUT-bound and
  // serves video with Range headers). Use /image which supports both
  // OUTPUT and UPLOADS roots, with the right MIME headers. Server-side
  // list_outputs() already builds the right URL per kind, so o.url
  // is correct for both — the fallback path below only runs when an
  // unknown path is requested (e.g. clicked from a stale modal link).
  let playerSrc;
  if (o) {
    playerSrc = o.url;
  } else if (isPhoto) {
    playerSrc = `/image?path=${encodeURIComponent(path)}`;
  } else {
    playerSrc = `/file?path=${encodeURIComponent(path)}`;
  }
  // Photo viewer is a static <img> — no controls, no autoplay (would
  // be a no-op on an image element anyway). Video viewer keeps the
  // existing controls + autoplay behaviour.
  if (isPhoto) {
    wrap.innerHTML = `<img src="${escapeHtml(playerSrc)}" alt="${o ? escapeHtml(o.name) : ''}">`;
  } else if (liveBackdrop) {
    wrap.innerHTML =
      `<img class="player-handoff-backdrop" src="${escapeHtml(liveBackdrop)}" alt="">` +
      `<video class="player-handoff-media" controls${autoplay ? ' autoplay' : ''} src="${escapeHtml(playerSrc)}"></video>`;
    const handoffVideo = wrap.querySelector('.player-handoff-media');
    _wireStageMutePersistence(handoffVideo);
    const revealFinished = () => {
      if (!handoffVideo || !handoffVideo.isConnected) return;
      handoffVideo.classList.add('is-ready');
      setTimeout(() => {
        const back = wrap.querySelector('.player-handoff-backdrop');
        if (back) back.remove();
        handoffVideo.classList.remove('player-handoff-media', 'is-ready');
      }, 240);
    };
    if (handoffVideo.readyState >= 2) requestAnimationFrame(revealFinished);
    else handoffVideo.addEventListener('loadeddata', revealFinished, { once: true });
    // A valid local mp4 normally fires loadeddata immediately. This ceiling is
    // the escape hatch for a browser that withholds it while autoplay policy
    // settles; controls must never stay transparent forever.
    setTimeout(revealFinished, 4000);
  } else {
    wrap.innerHTML = `<video controls${autoplay ? ' autoplay' : ''} src="${escapeHtml(playerSrc)}"></video>`;
    _wireStageMutePersistence(wrap.querySelector('video'));
  }
  // Surface aspect adapts to actual media dimensions so vertical clips
  // render vertically (previous hardcoded 16:9 surface + object-fit:cover
  // cropped head/feet off any 9:16 clip, and also showed horizontal
  // clips wrong if the surface had been switched). Read intrinsic w/h
  // on metadata load and set --media-aspect (which the CSS rule reads).
  // For vertical clips we also flip data-orient so the surface switches
  // to height-driven sizing rather than overflowing the stage.
  const surface = wrap.closest('.player-surface');
  if (surface) {
    surface.removeAttribute('data-orient');
    surface.style.removeProperty('--media-aspect');
    const media = wrap.querySelector(isPhoto ? 'img' : 'video');
    if (media) {
      const apply = () => {
        const w = isPhoto ? media.naturalWidth  : media.videoWidth;
        const h = isPhoto ? media.naturalHeight : media.videoHeight;
        if (!w || !h) return;
        surface.style.setProperty('--media-aspect', `${w} / ${h}`);
        if (h > w) surface.setAttribute('data-orient', 'vertical');
        else       surface.removeAttribute('data-orient');
      };
      if (isPhoto) {
        if (media.complete && media.naturalWidth) apply();
        else media.addEventListener('load', apply, { once: true });
      } else {
        if (media.readyState >= 1 && media.videoWidth) apply();
        else media.addEventListener('loadedmetadata', apply, { once: true });
      }
    }
  }

  // Y2.001 — populate the new overlays. The legacy #playerName / #playerMeta
  // are kept hidden but the writes still happen so any test/external code
  // reading them gets the same data.
  document.getElementById('playerMeta').style.display = '';
  const legacyMetaText = o
    ? `${o.name} · ${o.mtime} · ${o.size_mb.toFixed(1)} MB`
    : '';
  document.getElementById('playerName').textContent = legacyMetaText;
  // Visible top overlay — title + meta line with relative time.
  const overlayTop = document.getElementById('playerOverlayTop');
  const overlayActions = document.getElementById('playerOverlayActions');
  if (overlayTop) {
    overlayTop.style.display = '';
    document.getElementById('playerOverlayName').textContent = o ? o.name : '';
    const rel = o ? _relTimeFromMtime(o.mtime) : '';
    const sizeLbl = o ? `${o.size_mb.toFixed(1)} MB` : '';
    const kindLbl = isPhoto ? 'Photo' : 'Video';
    document.getElementById('playerOverlayMeta').innerHTML = o
      ? `<span>${kindLbl}</span><span class="po-dot"></span>` +
        `<span>${escapeHtml(rel)}</span><span class="po-dot"></span>` +
        `<span>${sizeLbl}</span>`
      : '';
  }
  if (overlayActions) overlayActions.style.display = '';

  // Load params is video-only — image sidecars use the library@1 schema
  // which doesn't carry the i2v/t2v form fields the loader expects.
  document.getElementById('loadParamsBtn').disabled = !(o && o.has_sidecar) || isPhoto;
  // Action button row: swap "Use as Extend" for "Animate" on photo
  // entries (Extend is video-only, but the still can be the seed for
  // an i2v render).
  const useExtBtn = document.getElementById('useAsExtendBtn');
  const animBtn = document.getElementById('animateBtn');
  // Extend is hidden on TWO kinds of output, for two different reasons:
  //   * a photo — Extend is a video pipeline (Animate takes its slot instead)
  //   * an H3 clip — Extend is an LTX pipeline. It runs the LTX Q8 extend
  //     sampler over a source clip, which would take an H3 render and continue
  //     it with a different model: different weights, different geometry grid
  //     (17n+5 vs 8k+1), and no audio branch at all, so the joint soundtrack
  //     that is the whole point of H3 would simply stop. H3's own answer to
  //     "make it longer" is window chaining, and chaining is the LENGTH AXIS —
  //     a choice made before the render, not an action on the result. There is
  //     nothing coherent to offer here, so the button goes; #h3Hint on the form
  //     says where length comes from, next to the Length strip that sets it.
  //   Scoped off the SELECTED OUTPUT's engine (sidecar-derived, already in the
  //   /status payload as o.engine) — not the form's current engine, which is
  //   about the next render and says nothing about this clip.
  const outIsH3 = !!(o && o.engine === 'h3');
  if (useExtBtn) useExtBtn.style.display = (isPhoto || outIsH3) ? 'none' : '';
  // Upscale ×2 is video-only but engine-agnostic: an H3 draft is exactly the
  // clip it was built for.
  const useUpBtn = document.getElementById('useAsUpscaleBtn');
  if (useUpBtn) useUpBtn.style.display = isPhoto ? 'none' : '';
  if (animBtn) animBtn.style.display = isPhoto ? '' : 'none';
  // "Finish at …" — for a completed H3 render that has a higher canvas to be
  // committed at. Decided from o.engine / o.h3_tier (both sidecar-derived,
  // already in the /status payload and resolved server-side through the legacy
  // alias map), so no extra request rides the selection path.
  if (typeof _syncH3FinishAffordance === 'function') {
    try { _syncH3FinishAffordance(isPhoto ? null : o); } catch (e) {}
  }
}

// Expand lightbox — full-viewport viewer for the active output. Reuses
// the active entry's URL / kind detection so a single button works for
// both image and video. Closed by Esc, backdrop click, or the × button.
function openExpandLightbox() {
  if (!activePath) return;
  const o = findOutputByPath(activePath);
  if (!o) return;
  const lb = document.getElementById('expandLightbox');
  const stage = document.getElementById('expandStage');
  const meta = document.getElementById('expandMeta');
  if (!lb || !stage) return;
  const isPhoto = isPhotoOutputMain(o);
  // Build the media element fresh each time so the previous selection's
  // <video> stops decoding immediately.
  stage.innerHTML = isPhoto
    ? `<img src="${o.url}" alt="${escapeHtml(o.name)}">`
    : `<video src="${o.url}" controls autoplay></video>`;
  _wireStageMutePersistence(stage.querySelector('video'));
  if (meta) {
    const sizeLbl = `${o.size_mb.toFixed(1)} MB`;
    meta.textContent = `${o.name} · ${sizeLbl}`;
  }
  lb.style.display = 'flex';
}

function closeExpandLightbox() {
  const lb = document.getElementById('expandLightbox');
  const stage = document.getElementById('expandStage');
  if (!lb) return;
  if (stage) stage.innerHTML = '';  // stops video playback
  lb.style.display = 'none';
}

// Esc + F shortcuts for the expand lightbox. Bound once at module init.
(function _wireExpandLightboxKeys() {
  document.addEventListener('keydown', (e) => {
    const lb = document.getElementById('expandLightbox');
    if (!lb) return;
    const isOpen = lb.style.display === 'flex';
    // Don't steal keystrokes from inputs/textareas.
    const tag = (e.target && e.target.tagName) || '';
    const inField = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable);
    if (e.key === 'Escape' && isOpen) {
      closeExpandLightbox();
      e.preventDefault();
    } else if ((e.key === 'f' || e.key === 'F') && !isOpen && !inField && activePath) {
      openExpandLightbox();
      e.preventDefault();
    } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && !inField && activePath
               && typeof filteredMainOutputs === 'function') {
      // Carousel keyboard nav. Wraps at the edges so power users can
      // scrub through the whole gallery without lifting hands off the
      // keyboard. Works whether the lightbox is open or not — if it
      // IS open, the lightbox content updates with the new selection.
      const list = filteredMainOutputs();
      if (!list.length) return;
      const idx = list.findIndex(o => o.path === activePath);
      const nextIdx = e.key === 'ArrowLeft'
        ? (idx <= 0 ? list.length - 1 : idx - 1)
        : (idx < 0 || idx >= list.length - 1 ? 0 : idx + 1);
      const next = list[nextIdx];
      if (next && typeof selectOutput === 'function') {
        selectOutput(next.path);
        if (isOpen) openExpandLightbox();   // re-bind media to new pick
      }
      e.preventDefault();
    }
  });
})();

// Toast helper — non-blocking confirmation pattern, stacks at the
// bottom-center of the viewport. Use for delete confirmation, "moved
// to Trash" feedback, save success, etc. Auto-dismisses after
// `duration` ms (default 3 s). Pass `kind: "success" | "danger"` to
// tint the border + icon.
function phosToast(message, opts) {
  opts = opts || {};
  const c = document.getElementById('phosToast');
  if (!c) return null;
  const el = document.createElement('div');
  el.className = 'phos-toast';
  if (opts.kind === 'success' || opts.kind === 'danger') {
    el.classList.add('phos-toast-' + opts.kind);
  }
  const icon = opts.icon
            || (opts.kind === 'success' ? 'ph-check-bold'
            :  opts.kind === 'danger'  ? 'ph-x-circle'
                                       : 'ph-info');
  el.innerHTML =
    `<svg class="ph" aria-hidden="true"><use href="#${icon}"/></svg>` +
    `<span class="phos-toast-msg"></span>`;
  el.querySelector('.phos-toast-msg').textContent = String(message);
  c.appendChild(el);
  const duration = (opts.duration === 0) ? 0 : (opts.duration || 3000);
  if (duration > 0) {
    setTimeout(() => {
      if (!el.isConnected) return;
      el.classList.add('is-leaving');
      el.addEventListener('animationend', () => el.remove(), { once: true });
    }, duration);
  }
  return el;
}

// Animate the active output (photo only). Mirrors animateFromPhoto's
// shape — pre-fills the i2v form from the active entry's path and the
// sidecar's prompt if there is one. Sidecar fetch is best-effort; if it
// fails (or there isn't one), we still animate with an empty prompt so
// the user can type their own.
async function animateActive() {
  if (!activePath) return;
  const o = findOutputByPath(activePath);
  let prompt = '';
  if (o && o.has_sidecar) {
    try {
      const r = await fetch('/sidecar?path=' + encodeURIComponent(activePath));
      if (r.ok) {
        const data = await r.json();
        // Image sidecars use schema "phosphene/library/image@1" with
        // a top-level `prompt`; video sidecars nest it under `params`.
        // Cover both shapes so the UI works regardless of source.
        prompt = _displayPromptFor((data && (data.prompt || (data.params && data.params.prompt))) || '');
      }
    } catch (e) {}
  }
  if (typeof animateFromPhoto === 'function') {
    animateFromPhoto({path: activePath, prompt: prompt});
  }
}

async function hide(path) { await fetch('/output/hide?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }
async function unhide(path) { await fetch('/output/show?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }

async function deleteOutput(path) {
  // Per-card × button. Moves the media (and any sibling sidecar JSON)
  // to the macOS Trash via /output/delete — files are recoverable
  // from Finder Trash via Cmd-Z right after, or by dragging them out
  // of the Trash bin later. Toast confirms the move; if the user was
  // viewing this clip in the expand lightbox, that closes too.
  if (!path) return;
  const base = path.split('/').pop();
  if (!confirm('Move to Trash?\n\n' + base + '\n\nRestore from Finder if needed.')) return;
  try {
    const fd = new URLSearchParams();
    fd.set('path', path);
    const r = await fetch('/output/delete', { method: 'POST', body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data.ok) {
      phosToast('Delete failed: ' + (data.error || ('HTTP ' + r.status)),
                { kind: 'danger', duration: 5000 });
      return;
    }
    // If the deleted clip was open in the expand lightbox, close it —
    // otherwise the lightbox sits there pointing at a 404 URL.
    if (activePath === path) {
      activePath = null;
      const lb = document.getElementById('expandLightbox');
      if (lb && lb.style.display === 'flex' && typeof closeExpandLightbox === 'function') {
        closeExpandLightbox();
      }
    }
    currentOutputs = [];
    poll();
    phosToast('Moved to Trash · ' + base, { kind: 'success' });
  } catch (e) {
    phosToast('Delete error: ' + (e.message || 'unknown'),
              { kind: 'danger', duration: 5000 });
  }
}

async function openOutputsFolder() {
  // One-click Reveal in Finder. Global, lives in the Outputs header so
  // it isn't duplicated per-card. macOS `open <dir>` — Phosphene is
  // Apple Silicon-only, no cross-platform branch.
  try {
    const r = await fetch('/output/open_folder', { method: 'POST' });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data.ok) {
      alert('Open folder failed: ' + (data.error || ('HTTP ' + r.status)));
    }
  } catch (e) {
    alert('Open folder error: ' + (e.message || 'unknown'));
  }
}
function hideActive() { if (activePath) hide(activePath); }

function useAsExtendSourcePath(path) {
  setMode('extend');
  document.getElementById('video_path').value = path;
  document.getElementById('extendSrcSelect').value = path;
  updateDerived();
  document.querySelector('aside.form-pane').scrollTop = 0;
}
function useAsExtendSource() { if (!activePath) return alert('Pick an output first.'); useAsExtendSourcePath(activePath); }
// Upscale ×2 — same hand-off shape as Extend: switch to the Remix tool,
// point the picker at this clip, scroll the form to the top.
function useAsUpscaleSourcePath(path) {
  setMode('upscale');
  const inp = document.getElementById('upscale_source_path');
  if (inp) inp.value = path;
  const sel = document.getElementById('upscaleSrcSelect');
  if (sel) sel.value = path;
  updateDerived();
  const pane = document.querySelector('aside.form-pane');
  if (pane) pane.scrollTop = 0;
}
function useAsUpscaleSource() { if (!activePath) return alert('Pick an output first.'); useAsUpscaleSourcePath(activePath); }
// Upscale ×2 presets — one hidden number (keep_shot) the server maps to how
// the render starts; the pills are the only thing the user touches.
function setUpscalePreset(btn) {
  const v = (btn && btn.dataset && btn.dataset.keep) || '1.0';
  const inp = document.getElementById('keep_shot');
  if (inp) inp.value = v;
  document.querySelectorAll('#upscalePresetGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', b === btn));
}

async function loadParams() {
  if (!activePath) return;
  const r = await fetch('/sidecar?path='+encodeURIComponent(activePath));
  if (!r.ok) return;
  const data = await r.json();
  const p = data.params;
  // Image-Studio sidecars (Ideogram 4 etc.) carry a TOP-LEVEL prompt that is a
  // structured caption JSON, not the video params shape below — the p.mode
  // dispatch would dump JSON into the i2v prompt box (or throw when p is
  // undefined). Detect an Ideogram caption and restore it into Image Studio +
  // its visual canvas instead of the video form.
  const _ideoSrc = _parseIdeoCaption((data && (data.prompt || (p && p.prompt))) || '');
  if (_ideoSrc) {
    const wf = document.querySelector('[data-workflow="studio"]');
    if (wf) wf.click();                                          // switch to Images
    const eng = document.getElementById('imgStudioEngine');
    if (eng) {
      eng.value = 'ideogram4_inline';
      if (typeof ideoSyncVisibility === 'function') ideoSyncVisibility();   // reveal the canvas
    }
    const pe = document.getElementById('imgStudioPrompt');
    if (pe) pe.value = _ideoSrc.high_level_description || '';
    const ta = document.getElementById('ideoRawJson');
    if (ta) {
      ta.value = JSON.stringify(_ideoSrc, null, 2);
      if (typeof ideoApplyRaw === 'function') { try { ideoApplyRaw(); } catch (_) {} }  // rehydrate boxes
    }
    return;
  }
  // If this clip came from the Characters tab, restore the Characters
  // compose state instead of dumping the user into Manual. The sidecar
  // carries the source flag + the original compose chips verbatim
  // (character_id, framing, duration, prompt_body, quality_choice).
  if (p && p.source === 'characters' && p.character_id) {
    try {
      await charactersLoadParams(p);
      return;
    } catch (e) {
      // Fall through to manual-form population on any unexpected error
      // so the user still gets *something* useful.
      console.warn('characters loadParams failed, falling back to Manual:', e);
    }
  }
  // Character renders are stored with mode='t2v' server-side (the
  // helper actually runs t2v under the hood; 'character' is a UI-only
  // intent that dispatches on character_id presence). Without this
  // check Load Params on a character clip drops the user into plain
  // Text mode and the picker stays hidden — confusing because the
  // strength/quality/loras restore happens but the mode pill says
  // Text. Fixed 2026-05-18: snap to Character mode when the sidecar
  // carries a character_id so the form's UI state matches the saved
  // intent, not the under-the-hood implementation.
  if (p.mode === 'extend') setMode('extend');
  else if (p.mode === 'keyframe') setMode('keyframe');
  else if (p.mode === 'i2v_clean_audio' || p.mode === 'i2v') { setMode('i2v'); document.getElementById('i2vMode').value = p.mode; document.getElementById('mode').value = p.mode; }
  else if (p.character_id) setMode('character');
  else setMode('t2v');
  // Apply quality + aspect FIRST (these stomp on width/height), then
  // override with explicit sidecar values so any custom dims survive.
  if (p.quality) setQuality(p.quality);
  // Snap aspect from the sidecar's recorded dims; only call when quality
  // isn't 'quick' (Quick has no aspect choice and the row is hidden).
  if (p.quality !== 'quick' && p.width && p.height) {
    for (const [k, a] of Object.entries(ASPECTS)) {
      if ((a.w === p.width && a.h === p.height) ||
          (a.h === p.width && a.w === p.height)) { setAspect(k); break; }
    }
  }
  // Now load explicit dims — overrides whatever the preset/aspect set.
  if (p.width) document.getElementById('width').value = p.width;
  if (p.height) document.getElementById('height').value = p.height;
  if (p.accel) setAccel(p.accel);
  // Schedule preset round-trips from day one (the journey audit's rule: a
  // new field ships WITH its restore path or it joins the 20 orphans).
  // Always applied — a clip rendered on Tuned must clear a leftover Fast.
  if (typeof setSchedPreset === 'function') {
    try { setSchedPreset(p.schedule_preset || ''); } catch (e) {}
  }
  // Reference use round-trips from day one; always applied, so a clip
  // rendered on Anchor clears a leftover Inspire.
  if (typeof setI2vRefMode === 'function') {
    try { setI2vRefMode(p.i2v_reference_mode || 'anchor'); } catch (e) {}
  }
  if (p.temporal_mode) setTemporalMode(p.temporal_mode);
  // A take restores as a take (length + beats), not as the fields it was
  // turned into — those are derived, and would re-derive differently.
  if (p.take && p.take.seconds && typeof setTakeSeconds === 'function') setTakeSeconds(p.take.seconds, p.take.beat_prompts || p.take.beats || null);
  else if (typeof setTakeSeconds === 'function') setTakeSeconds(0);
  if (p.upscale) setUpscale(p.upscale);
  if (p.upscale_method) setUpscaleMethod(p.upscale_method);
  document.getElementById('prompt').value = p.prompt || '';
  document.getElementById('negative_prompt').value = p.negative_prompt || '';
  // If the loaded sidecar carried an Avoid value, surface the row so the
  // user can see it without having to click the toggle.
  syncAvoidRowFromValue();
  if (p.frames) { document.getElementById('frames').value = p.frames; document.getElementById('duration').value = framesToDuration(p.frames); }
  if (p.steps) document.getElementById('steps').value = p.steps;
  // Prefer seed_used (the actual integer the helper picked at gen time)
  // over seed (what the user submitted — often `-1` for random). Without
  // this, Load Params on a -1 submission restores -1 and the next render
  // gets a fresh random seed instead of reproducing the original clip.
  // The user can still flip it back to -1 manually if they want a fresh
  // random; the goal is reproducibility by default. Fixed 2026-05-18.
  const seedToRestore = (p.seed_used != null && p.seed_used !== '')
    ? p.seed_used
    : p.seed;
  if (seedToRestore != null) {
    document.getElementById('seed').value = seedToRestore;
  }
  // Image / keyframes go through pickerSetImage so the preview tile
  // and recent-strip selection state update along with the hidden input.
  let restoredStartImage = p.start_image || '';
  let restoredEndImage = p.end_image || '';
  if (p.mode === 'keyframe') {
    let restoredKeyframes = null;
    try {
      const kfs = p.keyframes_json ? JSON.parse(p.keyframes_json) : null;
      if (Array.isArray(kfs) && kfs.length >= 3) {
        restoredKeyframes = kfs;
        restoredStartImage = restoredStartImage || (kfs[0] && kfs[0].image_path) || '';
        restoredEndImage = restoredEndImage || (kfs[kfs.length - 1] && kfs[kfs.length - 1].image_path) || '';
        setKeyframeMode(kfs.length);
      } else {
        setKeyframeMode(p.mid_image ? 3 : 2);
      }
    } catch (_) {
      setKeyframeMode(p.mid_image ? 3 : 2);
    }
    if (Array.isArray(restoredKeyframes) && restoredKeyframes.length >= 3) {
      renderKeyframeDynamicSlots();
      const slots = keyframeTimingSlots(restoredKeyframes.length);
      slots.forEach((slot, i) => {
        const kf = restoredKeyframes[i + 1];
        if (kf && kf.image_path) pickerSetImage(slot.imageKey, kf.image_path, { snapAspect: false });
        const secInp = document.getElementById(slot.secId);
        const frameIndex = kf ? parseInt(kf.frame_index, 10) : NaN;
        if (secInp && Number.isFinite(frameIndex)) {
          window._kfTimingTouched[slot.key] = true;
          secInp.value = (frameIndex / FPS).toFixed(2);
        }
      });
      syncKeyframeTiming();
    } else if (p.mid_image) {
      renderKeyframeDynamicSlots();
      const slot = keyframeTimingSlots(3)[0];
      if (slot) {
        pickerSetImage(slot.imageKey, p.mid_image, { snapAspect: false });
        const secInp = document.getElementById(slot.secId);
        const sec = parseFloat(p.keyframe_mid_seconds || '');
        if (secInp && Number.isFinite(sec)) {
          window._kfTimingTouched[slot.key] = true;
          secInp.value = sec.toFixed(2);
          syncKeyframeTiming();
        }
      }
    }
  }
  if (p.image)       pickerSetImage('image', p.image, { snapAspect: false });
  if (restoredStartImage) pickerSetImage('start_image', restoredStartImage, { snapAspect: false });
  if (restoredEndImage)   pickerSetImage('end_image', restoredEndImage, { snapAspect: false });
  if (p.audio) document.getElementById('audio').value = p.audio;
  // Extend-specific: restore the WHOLE request, not just the source path.
  // Duration, direction, sampler depth and CFG were all saved to the sidecar
  // and never read back — a Load Params + Generate on an Extend silently
  // re-rendered with the defaults (journey audit, High).
  if (p.video_path) document.getElementById('video_path').value = p.video_path;
  if (p.mode === 'extend') {
    const extFrames = parseInt(p.extend_frames, 10);
    if (Number.isFinite(extFrames) && extFrames > 0) {
      const fEl = document.getElementById('extend_frames');
      const sEl = document.getElementById('extend_seconds');
      if (fEl) fEl.value = String(extFrames);
      // extend_frames counts LATENT frames (8 video frames each at 24 fps);
      // the visible seconds field derives from it, then the sync refreshes
      // the hint line with the same arithmetic the input handler uses.
      if (sEl) sEl.value = String((extFrames * 8) / 24);
      if (typeof syncExtendDuration === 'function') {
        try { syncExtendDuration(); } catch (_) {}
      }
    }
    if (p.extend_direction) {
      const dEl = document.getElementById('extend_direction');
      if (dEl) dEl.value = p.extend_direction;
    }
    const extSteps = parseInt(p.extend_steps, 10);
    const extCfg = parseFloat(p.extend_cfg);
    if (Number.isFinite(extSteps)) {
      const el = document.getElementById('extend_steps');
      if (el) el.value = String(extSteps);
    }
    if (Number.isFinite(extCfg)) {
      const el = document.getElementById('extend_cfg');
      if (el) el.value = String(extCfg);
    }
    // Reflect the Fast/Quality pills: light the preset that matches the
    // restored values; a hand-set API pair lights neither (it IS custom).
    const extMode = (extSteps === 30 && extCfg === 3.0) ? 'quality'
                  : (extSteps === 12 && extCfg === 1.0) ? 'fast' : '';
    document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b =>
      b.classList.toggle('active', !!extMode && b.dataset.extendMode === extMode));
  }
  if (p.label) document.getElementById('preset_label').value = p.label;

  // Manual-tab Characters picker — restore the selection if the sidecar
  // recorded a character_id (set by make_job when the form carried one).
  // We do this BEFORE the LoRA-list restore so the face/audio LoRAs that
  // make_job re-expands on the next submit aren't accidentally surfaced
  // as plain LoRA chips in the picker too. Strip them out of the loras
  // list (they'll be re-added by the backend expansion) so the picker
  // shows only the user-stacked style LoRAs — same shape as the original
  // submission.
  let lorasForPicker = Array.isArray(p.loras) ? p.loras : null;
  if (p.character_id && typeof refreshManualCharacters === 'function') {
    // AWAITED, and through the real cascade (v4.0.5, owner-reported): the
    // old restore was a fire-and-forget IIFE that wrote _selectedCharacterId
    // and the hidden input directly — no avatar ring on slow registry loads,
    // no quality-strip swap, "Loaded" flashed before the cast existed, and
    // a character deleted since the render restored to silence. loadParams
    // is async; there is nothing to gain by not waiting.
    try { await refreshManualCharacters(); } catch (_) {}
    const exists = (_manualCharacters || []).some(c => c.id === p.character_id);
    if (exists) {
      // The non-toggling hydrator: full cascade, no trigger injection (the
      // sidecar prompt, already restored above, carries the trigger).
      try { applyCharacterSelection(p.character_id); } catch (_) {}
      // The cascade snaps the character-quality strip to its default chip,
      // which stomps quality + dims — re-assert the CLIP's own record.
      // Chip first (by the UI token the sidecar carries), then dims.
      try {
        const chipKey = String(p.quality_choice || '').toLowerCase();
        const strip = document.getElementById('qualityGroupCharacter');
        const chip = chipKey && strip
          ? strip.querySelector(`[data-char-quality="${chipKey}"]`) : null;
        if (chip && typeof _setCharacterQuality === 'function') {
          _setCharacterQuality(chip, { allowMissing: true });
        }
      } catch (_) {}
      if (p.quality) document.getElementById('quality').value = p.quality;
      if (p.width) document.getElementById('width').value = p.width;
      if (p.height) document.getElementById('height').value = p.height;
      if (p.frames) document.getElementById('frames').value = p.frames;
      // Explicit No-voice restore (provenance: the sidecar's own value).
      // Marking it touched stops the prompt auto-default from overriding
      // a value the user explicitly rendered with.
      const nv = document.getElementById('noVoice');
      if (nv && p.no_voice != null && p.no_voice !== '') {
        nv.checked = (p.no_voice === 'on' || p.no_voice === true
                      || p.no_voice === 'true');
        if (typeof markNoVoiceTouched === 'function') markNoVoiceTouched();
      }
    } else {
      // FAIL VISIBLY. The saved cast is gone from disk; restoring the rest
      // of the form silently is how the owner's clip reopened castless with
      // a green "Loaded" flash on top.
      if (typeof phosToast === 'function') {
        phosToast(`This clip's character "${p.character_id}" is no longer in `
                  + 'the library — everything else was restored, but the '
                  + 'cast could not be. Re-train or re-download it to '
                  + 're-render this clip faithfully.', { kind: 'danger' });
      }
    }
    // BOTH strengths, because the character has two and the sidecar records
    // two. Only the face was restored here, so a clip rendered with a
    // deliberately hotter or colder voice reopened with the voice silently
    // back at its default — Load Params quietly changing the render it claims
    // to reload. The hidden inputs are the source the strip re-renders from,
    // so setting them and re-rendering restores both sliders and the value the
    // next submit will send. AFTER the selection cascade, which resets both.
    _restoreCharacterStrengths(p);
    // Strip the character's face/audio LoRA paths out of the loras list
    // so the picker doesn't show duplicate state. The backend will
    // re-expand on the next submit. We can't read list_characters() from
    // the client cheaply mid-loadParams, so we use a name heuristic
    // identical to the server's `_is_character_lora` rule: filename
    // matches `<trigger>_v2.safetensors` or `<trigger>.audio.safetensors`.
    if (lorasForPicker) {
      const trig = String(p.character_id).toLowerCase();
      lorasForPicker = lorasForPicker.filter(l => {
        const fname = String(l.path || '').split('/').pop().toLowerCase();
        if (fname === `${trig}_v2.safetensors`) return false;
        if (fname === `${trig}.audio.safetensors`) return false;
        return true;
      });
    }
  }

  // Restore the LoRA picker state. Without this, Load Params would silently
  // drop every LoRA the prior render used — prompt + seed + dims all match
  // but `loras=0` would go on the wire and the model would re-render
  // without fusion (no face/style transfer). Wire shape on the sidecar is
  // a list of {path, strength}; we re-decorate with `name` + `trigger_words`
  // from _knownUserLoras when available so the chip renders nicely.
  if (lorasForPicker) {
    _activeLoras = lorasForPicker.map(l => {
      const path = l.path;
      const strength = (typeof l.strength === 'number') ? l.strength : 1.0;
      const meta = (Array.isArray(_knownUserLoras)
                    ? _knownUserLoras.find(u => u.path === path)
                    : null) || {};
      return {
        path,
        strength,
        name: meta.name || (path ? path.split('/').pop() : 'LoRA'),
        trigger_words: meta.trigger_words || [],
        compatible_modes: meta.compatible_modes || ['unknown'],
      };
    }).filter(x => x.path);
    if (typeof renderLorasList === 'function') {
      try { renderLorasList(); } catch (_) {}
    }
    if (typeof _serializeLoras === 'function') {
      try { _serializeLoras(); } catch (_) {}
    }
  }

  // ---- Engine + engine-specific settings ----------------------------------
  // Params restored geometry and prompt but NOT the engine, so loading an LTX
  // clip while the H3 surface was active left LTX dimensions sitting under an
  // H3 tier strip — the state the owner hit and rightly called confusing. The
  // sidecar has carried `engine` (and the H3 fields) since the engine landed;
  // this just reads them. Order matters: engine first (it swaps the surface),
  // tier LAST (setH3Tier stamps width/height/frames), and the seed re-applied
  // after the tier so the tier can't clobber the seed we just loaded.
  const _seedBefore = (document.getElementById('seed') || {}).value;
  const _eng = String((p && p.engine) || (data && data.engine) || 'ltx').toLowerCase();
  if (typeof setEngine === 'function' && typeof engineById === 'function' && engineById(_eng)) {
    try { setEngine(_eng, { persist: false }); } catch (e) {}
  }
  if (_eng === 'h3') {
    if (typeof setH3Upscale === 'function' && p.h3_upscale) {
      try { setH3Upscale(p.h3_upscale); } catch (e) {}
    }
    if (typeof setH3Orientation === 'function') {
      try { setH3Orientation(p.h3_orientation || 'landscape'); } catch (e) {}
    }
    if (typeof setH3Turbo === 'function') {
      try { setH3Turbo(!!Number(p.h3_turbo || 0)); } catch (e) {}
    }
    if (typeof setH3Steps === 'function') {
      try { setH3Steps(Number(p.h3_steps || 0) > 0 ? String(p.h3_steps) : 'auto'); } catch (e) {}
    }
    // The render shape. Prefer the two axes when the sidecar carries them; fall
    // back to the composite key, which every sidecar has always carried and
    // which h3TierByKey resolves even when it is a pre-two-axis name (hq_5s,
    // wide_5s, long_10s). Either way one call, and the CELL stamps the geometry.
    if (typeof setH3Quality === 'function' && p.h3_quality && p.h3_length
        && h3CellFor(p.h3_quality, p.h3_length)) {
      try { _h3ApplyShape(p.h3_quality, p.h3_length, { fallback: true }); } catch (e) {}
    } else if (typeof setH3Tier === 'function' && p.h3_tier) {
      try { setH3Tier(p.h3_tier); } catch (e) {}
    }
    // Per-window prompts AFTER the shape: the number of boxes is a property of
    // the length, so restoring the list before the cell is applied would drop
    // every entry past the previously-selected window count. The sidecar carries
    // the RAW list the user typed (blanks included), which is what makes a
    // reload land on the same form rather than on a resolved one.
    if (typeof setH3ChainPrompts === 'function') {
      try { setH3ChainPrompts(p.h3_chain_prompts); } catch (e) {}
    }
    const _seedEl = document.getElementById('seed');
    if (_seedEl && _seedBefore != null) _seedEl.value = _seedBefore;
  }

  // STG round-trips (journey audit, High): the slider only means anything on
  // an effective HQ pipeline, and a sidecar without the field means the
  // render ran without it — so absence resets to Off instead of leaving a
  // stale slider armed for the next submit.
  const _stgEl = document.getElementById('stgScale');
  if (_stgEl) {
    const stg = parseFloat(p.stg_scale);
    const val = Number.isFinite(stg) ? Math.max(0, Math.min(4, stg)) : 0;
    _stgEl.value = String(val);
    const lbl = document.getElementById('stgScaleValue');
    if (lbl) lbl.textContent = val === 0 ? 'Off' : val.toFixed(1);
    if (typeof _applyStgRowVisibility === 'function') {
      try { _applyStgRowVisibility(); } catch (_) {}
    }
  }
  // Repaint the LTX strips AFTER frames landed (journey audit, High): the
  // programmatic frame assignment fires no input event, so a loaded
  // 10-second clip used to sit under a Length chip still lit at 5 s.
  if (_eng !== 'h3' && typeof renderTierAxes === 'function') {
    try { renderTierAxes('ltx'); } catch (_) {}
  }

  updateCustomizeSummary();
  updateDerived();
  // Say it out loud. This whole function ran silently before — the form
  // changed somewhere off-screen and the click read as a dead button.
  _flashActionDone('loadParamsBtn', 'Loaded');
  // On the Storyboard surface the Video form isn't on screen at all, so the
  // flash happens somewhere the user can't see and the click reads as dead.
  // Take them to the form that just changed.
  if (document.body.dataset.workflow === 'storyboard') {
    phosToast('Loaded into the Video form.', { kind: 'success' });
    if (typeof workflowSwitch === 'function') workflowSwitch('manual');
  }
}

// Momentary "it worked" state on an action button. The panel has no toast
// primitive, and an action that succeeds silently is indistinguishable from
// one that failed — which is exactly how Params was being read.
function _flashActionDone(btnId, word) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const label = btn.querySelector('.po-act-label');
  if (!label) return;
  if (btn.dataset.flashing === '1') return;
  const prev = label.textContent;
  btn.dataset.flashing = '1';
  btn.classList.add('po-act-done');
  label.textContent = word;
  setTimeout(() => {
    label.textContent = prev;
    btn.classList.remove('po-act-done');
    delete btn.dataset.flashing;
  }, 1400);
}

// ====== Output info modal ======
//
// Opened by the ⓘ button on each gallery card. Shows the full sidecar
// (.mp4.json) we wrote at render time: prompt, seed, mode, dimensions,
// frames, steps, LoRAs used (with display names + strengths), elapsed
// time, queue id, model. Plus per-field copy buttons for the things
// users actually want to reuse (prompt + seed).
//
// Why a modal and not inline detail-on-hover: the prompt alone can be
// 1000+ chars; trying to render it inline next to the thumbnail would
// blow up the gallery layout. Modal lets us scroll comfortably.

let _outputInfoLastPath = null;

async function openOutputInfoModal(path) {
  _outputInfoLastPath = path;
  const modal = document.getElementById('outputInfoModal');
  const body = document.getElementById('outputInfoBody');
  const title = document.getElementById('outputInfoTitle');
  modal.style.display = 'flex';
  body.innerHTML = '<div class="hint">Loading…</div>';
  // Display the filename in the modal title for quick orientation.
  const fname = path.split('/').pop();
  if (title) title.textContent = `Generation info · ${fname}`;
  let data;
  try {
    const r = await fetch('/sidecar?path=' + encodeURIComponent(path));
    if (!r.ok) {
      body.innerHTML = `<div class="hint">No sidecar metadata for this output (older generation, or sidecar was deleted).</div>`;
      return;
    }
    data = await r.json();
  } catch (e) {
    body.innerHTML = `<div class="hint">Couldn't load info: ${escapeHtml(e.message || String(e))}</div>`;
    return;
  }
  body.innerHTML = renderOutputInfoBody(path, data);
}

function closeOutputInfoModal() {
  document.getElementById('outputInfoModal').style.display = 'none';
}

function _copyToClipboard(text, btn) {
  // Best-effort copy with visual feedback. Falls back silently when the
  // clipboard API is blocked (e.g. iframe sandboxes without permissions).
  try {
    navigator.clipboard.writeText(text);
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1200);
    }
  } catch (e) { /* swallow */ }
}

function _humanSize(b) {
  if (b == null) return '';
  if (b < 1024) return `${b} B`;
  if (b < 1024*1024) return `${(b/1024).toFixed(1)} KB`;
  if (b < 1024*1024*1024) return `${(b/1024/1024).toFixed(1)} MB`;
  return `${(b/1024/1024/1024).toFixed(2)} GB`;
}

function _humanDuration(s) {
  if (s == null) return '';
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60); const r = (s - m*60).toFixed(0);
  return `${m} min ${r} s`;
}

function renderOutputInfoBody(path, data) {
  const p = (data && data.params) || {};
  const loras = Array.isArray(p.loras) ? p.loras : [];

  // Look up each LoRA's display name from the installed-LoRAs cache so
  // the modal shows "Claymation Style" instead of the raw safetensors
  // path. Falls back gracefully when a LoRA was deleted or is an HF id.
  const lookupLoraName = (loraPath) => {
    if (!loraPath) return '?';
    const known = (_knownUserLoras || []).find(l => l.path === loraPath);
    if (known) return known.name;
    if (loraPath.includes('/') && !loraPath.endsWith('.safetensors')) return loraPath;
    return loraPath.split('/').pop().replace(/\.safetensors$/, '');
  };

  const promptText = p.prompt || '';
  const promptAttr = JSON.stringify(promptText).replace(/"/g, '&quot;');
  const seedVal = String(p.seed_used != null ? p.seed_used : p.seed || '');
  const seedAttr = JSON.stringify(seedVal).replace(/"/g, '&quot;');
  const pathAttr = JSON.stringify(path).replace(/"/g, '&quot;');
  const accelMetrics = (data && data.accel_metrics) || null;
  const keyframeModeLabel = (() => {
    if (!p.keyframes_json) return 'FFLF (first + last frame)';
    try {
      const kfs = JSON.parse(p.keyframes_json);
      return Array.isArray(kfs) && kfs.length >= 3
        ? `${kfs.length} Keyframes`
        : 'Multi-keyframe';
    } catch (_) {
      return 'Multi-keyframe';
    }
  })();
  const modeLabel = ({
    t2v: 'Text → Video',
    i2v: 'Image → Video',
    i2v_clean_audio: 'Image → Video (clean audio)',
    keyframe: keyframeModeLabel,
    extend: 'Extend',
  })[p.mode] || (p.mode || '—');

  // Compose the dimensions + duration into a single "Format" line — fewer
  // grid rows, easier to scan. We separate technical metadata (Format,
  // Frames) from generation parameters (Mode, Quality, Seed, Steps).
  const formatBits = [];
  if (p.width && p.height) formatBits.push(`${p.width} × ${p.height}`);
  if (data.video_duration_sec != null) formatBits.push(`${data.video_duration_sec.toFixed(2)} s @ ${data.fps || 24} fps`);

  let html = '';

  // ---- Output (technical) ----
  html += `<div class="oi-section">
    <div class="oi-section-title"><span>Output</span></div>
    <dl class="oi-grid">
      ${formatBits.length ? `<dt>Format</dt><dd>${formatBits.join('  ·  ')}</dd>` : ''}
      ${p.frames != null ? `<dt>Frames</dt><dd>${p.frames}</dd>` : ''}
    </dl>
  </div>`;

  // ---- Generation parameters ----
  const genRows = [];
  genRows.push(`<dt>Mode</dt><dd>${escapeHtml(modeLabel)}</dd>`);
  // A Hailuo H3 render has no LTX quality preset. `params.quality` is just
  // whatever the quality strip happened to hold when Generate was pressed —
  // the H3 TIER defined this render's geometry — so printing "Quality:
  // Balanced" on an H3 clip states something that was never true of it.
  // Exact-match the tier rather than reusing h3TierByKey(), which falls back
  // to the first tier and would relabel an unknown key as "Draft · 3s"; on a
  // machine with no H3 pack installed (H3.tiers empty) the raw key is printed,
  // which is honest, where a fallback label would not be.
  if (String(p.engine || 'ltx').toLowerCase() === 'h3') {
    const h3RawKey = p.h3_tier || (data && data.h3 && data.h3.tier) || '';
    // RESOLVE first: a clip rendered before the two-axis refactor carries
    // `hq_5s` / `wide_5s` / `long_10s`, and those still name a real shape. The
    // raw key is what prints when nothing resolves (no H3 pack installed, a
    // hand-edited sidecar) — honest, where a fallback label would not be.
    const h3TierDef = (typeof h3TierByKeyExact === 'function')
      ? h3TierByKeyExact(h3RawKey) : null;
    genRows.push(`<dt>Engine</dt><dd>Hailuo H3</dd>`);
    // Two axes, two rows. The frozen `h3_tier_label` from render time wins over
    // today's label if the sidecar has one — a clip should keep the name it was
    // rendered under even if the presets are renamed again.
    const qLabel = h3TierDef ? h3TierDef.quality_label : (p.h3_quality || '');
    const lLabel = h3TierDef ? h3TierDef.length_label : (p.h3_length || '');
    if (qLabel || lLabel) {
      genRows.push(`<dt>Quality</dt><dd>${escapeHtml(qLabel || '—')}${
        h3TierDef ? ' · ' + escapeHtml(h3TierDef.width + '×' + h3TierDef.height
                                      + ' · ' + h3TierDef.aspect) : ''}</dd>`);
      genRows.push(`<dt>Length</dt><dd>${escapeHtml(lLabel || '—')}${
        p.frames != null ? ' · ' + escapeHtml(String(p.frames)) + 'f' : ''}</dd>`);
    } else {
      genRows.push(`<dt>H3 tier</dt><dd>${escapeHtml(p.h3_tier_label || h3RawKey || '—')}</dd>`);
    }
    // Turbo changed the sampler recipe, so it earns a row on every clip that
    // used it — it is the difference between two otherwise identical renders.
    if (p.h3_turbo) {
      const tinfo = (data && data.h3 && data.h3.turbo) || null;
      const applied = tinfo && tinfo.applied && tinfo.applied.applied;
      genRows.push(`<dt>Turbo</dt><dd>on · ${escapeHtml(String(p.steps || 4))}-step `
        + `distill LoRA${applied ? ' · ' + escapeHtml(String(applied)) + ' modules applied' : ''}</dd>`);
    } else if (Number(p.h3_steps || 0) > 0) {
      // Steps only earns a row when the user pinned a depth — the tier default
      // is already implied by the tier row.
      genRows.push(`<dt>Steps</dt><dd>${escapeHtml(String(p.steps || p.h3_steps))} · tier default overridden</dd>`);
    }
  } else {
    genRows.push(`<dt>Quality</dt><dd>${escapeHtml((p.quality || 'standard').replace(/^./, c => c.toUpperCase()))}</dd>`);
  }
  if (p.accel && p.accel !== 'off') {
    genRows.push(`<dt>Speed</dt><dd>${escapeHtml(p.accel.replace(/^./, c => c.toUpperCase()))}</dd>`);
  }
  // A non-default schedule changed the take, so the record must name it.
  if (p.schedule_preset && p.schedule_preset !== 'default') {
    genRows.push(`<dt>Schedule</dt><dd>${escapeHtml(String(p.schedule_preset))} · draft schedule — a different take than Tuned</dd>`);
  }
  if (p.i2v_reference_mode === 'inspire') {
    genRows.push(`<dt>Reference use</dt><dd>Inspire — the image guided subject and style; the shot was composed fresh (not animated from it)</dd>`);
  }
  if (accelMetrics && p.accel && p.accel !== 'off') {
    const cachedCount = accelMetrics.cached_steps_count || 0;
    const totalSteps = accelMetrics.total_steps || p.steps || 0;
    const savings = accelMetrics.estimated_denoise_call_savings_pct;
    const cachedList = Array.isArray(accelMetrics.cached_steps) && accelMetrics.cached_steps.length
      ? ` · cached steps ${escapeHtml(accelMetrics.cached_steps.join(', '))}`
      : '';
    const savingsText = savings != null ? ` · ~${escapeHtml(String(savings))}% denoise calls saved` : '';
    genRows.push(`<dt>Accel metrics</dt><dd>${cachedCount}/${totalSteps} cached${savingsText}${cachedList}</dd>`);
  }
  if (data.memory_policy) {
    const mp = data.memory_policy;
    const req = mp.requested || p.memory_policy || 'auto';
    const eff = mp.effective || req;
    const decode = mp.vae_decode || (mp.helper && mp.helper.vae_decode) || '';
    const fullMax = mp.full_decode_max_frames != null ? ` · full≤${escapeHtml(String(mp.full_decode_max_frames))}f` : '';
    const reason = mp.reason ? ` · ${escapeHtml(mp.reason)}` : '';
    genRows.push(`<dt>Memory</dt><dd>${escapeHtml(req)}${eff !== req ? ` → ${escapeHtml(eff)}` : ''}${decode ? ` · VAE ${escapeHtml(decode)}` : ''}${fullMax}${reason}</dd>`);
  }
  if (p.temporal_mode === 'fps12_interp24' || data.temporal) {
    const t = data.temporal || {};
    const sourceFrames = t.source_frames || p.model_frames || '—';
    const deliveryFrames = t.delivery_frames || p.frames || '—';
    const sourceFps = t.model_fps || p.model_fps || 12;
    const deliveryFps = t.delivery_fps || p.delivery_fps || 24;
    genRows.push(`<dt>Long clips</dt><dd>12 → 24fps · LTX ${escapeHtml(String(sourceFrames))}f @ ${escapeHtml(String(sourceFps))}fps → ${escapeHtml(String(deliveryFrames))}f @ ${escapeHtml(String(deliveryFps))}fps</dd>`);
  }
  if (p.upscale && p.upscale !== 'off') {
    const up = data.upscale || {};
    const target = up.target_w && up.target_h ? ` → ${up.target_w} × ${up.target_h}` : '';
    const isSharp = p.upscale_method === 'pipersr' || p.upscale_method === 'model' || (data.upscale && (data.upscale.method === 'pipersr_coreml' || data.upscale.pre_pass === 'pipersr_x2' || data.upscale.method === 'ltx_latent_x2' || data.upscale.pre_pass === 'ltx_latent_x2'));
    const baseLabel = p.upscale === 'fit_720p' ? '720p fit (no crop)' : (p.upscale === 'x2' ? '2×' : p.upscale);
    const label = isSharp ? `${baseLabel} · Sharp (PiperSR)` : `${baseLabel} · Fast (Lanczos)`;
    genRows.push(`<dt>Upscale</dt><dd>${escapeHtml(label + target)}</dd>`);
  }
  // H3's own export pass. `p.upscale` is always 'off' on an H3 render (the LTX
  // knobs are neutralised server-side), so it needs its own row rather than
  // silently claiming no post-process happened.
  if (p.h3_upscale && p.h3_upscale !== 'off') {
    const up = data.upscale || {};
    const target = up.target_w && up.target_h ? ` → ${up.target_w} × ${up.target_h}` : '';
    const base = p.h3_upscale === 'fit_1080p' ? '1080p fit (no crop)' : '720p fit (no crop)';
    genRows.push(`<dt>Export</dt><dd>${escapeHtml(base + ' · Fast (Lanczos)' + target)}</dd>`);
  }
  // Chained windows — the honest shape of a 10 s / 15 s H3 clip.
  if (data.h3 && Number(data.h3.chain_windows || 1) > 1) {
    const c = data.h3;
    genRows.push(`<dt>Chained</dt><dd>${escapeHtml(String(c.chain_windows))} × ${escapeHtml(String(c.window_frames))}f windows → ${escapeHtml(String(c.delivered_frames || p.frames || '—'))}f · ${escapeHtml(String(c.seams || (c.chain_windows - 1)))} join(s)</dd>`);
    // What each window was ACTUALLY asked for. The resolved list — blanks
    // already filled from the main prompt — so a clip whose second beat didn't
    // land can be read straight off the modal. Absent = one prompt, every
    // window (the default), and the Chained row above already says how many.
    const wp = Array.isArray(c.chain_prompts) ? c.chain_prompts : [];
    if (wp.length) {
      genRows.push(`<dt>Window prompts</dt><dd>${wp.map((t, i) =>
        `<div><b>${i + 1}</b> · ${escapeHtml(snippet(String(t == null ? '' : t), 110))}</div>`
      ).join('')}</dd>`);
    }
  }
  const codec = data.output_codec || (data.upscale && data.upscale.codec);
  if (codec && codec.pix_fmt && codec.crf != null) {
    const preset = codec.preset ? ` · ${codec.preset}` : '';
    genRows.push(`<dt>Output codec</dt><dd>${escapeHtml(codec.pix_fmt)} · CRF ${escapeHtml(String(codec.crf))}${escapeHtml(preset)}</dd>`);
  }
  if (p.negative_prompt) {
    genRows.push(`<dt>Avoid</dt><dd>${escapeHtml(snippet(p.negative_prompt, 90))}</dd>`);
  }
  if (seedVal) {
    genRows.push(`<dt>Seed</dt><dd>
      <code>${escapeHtml(seedVal)}</code>
      <button class="oi-copy" type="button" onclick="_copyToClipboard(${seedAttr}, this)">Copy</button>
    </dd>`);
  }
  if (p.steps != null) genRows.push(`<dt>Steps</dt><dd>${p.steps}</dd>`);
  if (p.hdr) genRows.push(`<dt>HDR</dt><dd>On</dd>`);
  if (p.label) genRows.push(`<dt>Label</dt><dd>${escapeHtml(p.label)}</dd>`);

  html += `<div class="oi-section">
    <div class="oi-section-title"><span>Generation</span></div>
    <dl class="oi-grid">${genRows.join('')}</dl>
  </div>`;

  // ---- Prompt ----
  if (promptText) {
    html += `<div class="oi-section">
      <div class="oi-section-title">
        <span>Prompt</span>
        <button class="oi-copy" type="button" onclick="_copyToClipboard(${promptAttr}, this)">Copy</button>
      </div>
      <div class="oi-prompt">${escapeHtml(promptText)}</div>
    </div>`;
  }

  // ---- LoRAs (flat list, hairline-separated) ----
  if (loras.length) {
    const rows = loras.map(l => {
      const name = lookupLoraName(l.path);
      const strength = (l.strength != null ? l.strength : 1).toFixed(2);
      return `<div class="oi-lora-row">
        <span class="oi-lora-name" title="${escapeHtml(l.path || '')}">${escapeHtml(name)}</span>
        <span class="oi-lora-strength">strength ${strength}</span>
      </div>`;
    }).join('');
    html += `<div class="oi-section">
      <div class="oi-section-title">
        <span>LoRAs used</span>
        <span class="oi-count">${loras.length}</span>
      </div>
      <div class="oi-lora-list">${rows}</div>
    </div>`;
  }

  // ---- Timing + provenance ----
  const timingRows = [];
  if (data.started) timingRows.push(`<dt>Started</dt><dd>${escapeHtml(data.started)}</dd>`);
  if (data.elapsed_sec != null) timingRows.push(`<dt>Elapsed</dt><dd>${_humanDuration(data.elapsed_sec)}</dd>`);
  if (data.queue_id) timingRows.push(`<dt>Queue ID</dt><dd><code>${escapeHtml(data.queue_id)}</code></dd>`);
  if (data.model) timingRows.push(`<dt>Model</dt><dd><code>${escapeHtml(data.model.split('/').pop())}</code></dd>`);
  if (timingRows.length) {
    html += `<div class="oi-section">
      <div class="oi-section-title"><span>Timing</span></div>
      <dl class="oi-grid">${timingRows.join('')}</dl>
    </div>`;
  }

  // ---- Action row ----
  html += `<div class="oi-actions">
    <button class="ghost-btn" type="button" onclick="closeOutputInfoModal()">Close</button>
    <button class="oi-primary" type="button"
            onclick="closeOutputInfoModal(); selectOutput(${pathAttr}); loadParams()">
      Load params into form
    </button>
  </div>`;

  return html;
}

async function removeJob(id) { await fetch('/queue/remove?id='+encodeURIComponent(id),{method:'POST'}); poll(); }
async function togglePause() {
  const s = await (await fetch('/status')).json();
  await api(s.paused ? '/queue/resume' : '/queue/pause', 'POST');
  poll();
}

// ====== Tabs ======
document.querySelectorAll('.tabs button[data-tab]').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button[data-tab]').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.toggle('show', t.id === 'tab-'+b.dataset.tab));
});

// ====== Batch modal ======
function openBatch() { document.getElementById('batchModal').classList.add('show'); }
function closeBatch() { document.getElementById('batchModal').classList.remove('show'); }
async function queueBatch() {
  const fd = new FormData(document.getElementById('genForm'));
  fd.set('prompts', document.getElementById('batchPrompts').value);
  const r = await api('/queue/batch','POST',fd);
  if (r && r.error) { alert('Batch error: '+r.error); return; }
  if (r && r.added) { document.getElementById('batchPrompts').value = ''; poll(); }
}

// ====== "No music" toggle pill ======
//
// Custom pill replacing the default checkbox. Click anywhere on the pill
// to flip the hidden checkbox + reflect state in the UI (.on class drives
// the accent fill from the toggle-pill CSS). Backed by a real <input
// type=checkbox> inside the label, so FormData still picks it up the
// normal way and screen readers / keyboard nav still work.
(function () {
  const pill = document.getElementById('noMusicPill');
  const cb = document.getElementById('noMusic');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', e => {
    // <label> already toggles the checkbox; we just need to refresh the
    // visual state on the next tick AFTER the native toggle has fired.
    setTimeout(sync, 0);
  });
  sync();
})();

// ====== Form submit ======
//
// "No music" toggle: appends a clear audio constraint to the prompt
// before submission so the LTX 2.3 vocoder skips the soundtrack/score it
// otherwise tends to add. Music is hard to remove cleanly from a stem
// after the fact (it shares spectral space with dialogue), so users who
// plan to score the clip themselves want voice + ambient only.
//
// We modify the FormData copy, not the textarea value — so the user's
// original prompt stays untouched in the UI.
document.getElementById('genForm').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);

  // ---- engine payload scrub ------------------------------------------------
  // The fold rules hide an LTX-only control on H3; they do not empty the hidden
  // input behind it, and FormData reads the input. So a user who picks a
  // character, types an Avoid line, then switches to H3 was still POSTing
  // character_id + a LoRA stack + negative_prompt on a job that reads none of
  // them. Nothing broke — run_h3_job ignores all three — but the queue card,
  // the ⓘ modal and the sidecar all then describe a render that didn't happen,
  // and Load Params replays the fiction. Clear them here so what we send
  // matches what the surface says. `engine` itself is re-validated in make_job;
  // this is about honesty of the record, not safety.
  //
  // NOT scrubbed: `seed` (H3 resolves -1 itself), `image` (first-frame
  // conditioning is real on H3), `frames`/`width`/`height` (make_job stamps the
  // selected cell's own geometry over them).
  if ((fd.get('engine') || 'ltx') === 'h3') {
    fd.set('negative_prompt', '');   // guidance-distilled: no unconditional branch
    fd.set('character_id', '');      // character LoRAs are an LTX construct
    fd.set('no_voice', '');          // only ever meant "skip the character's voice LoRA"
    fd.set('schedule_preset', '');   // an LTX-2.5 distilled schedule; H3 has its own axes
    fd.set('i2v_reference_mode', 'anchor');  // an LTX-2.5 sampler behaviour
    // `loras` IS NOT SCRUBBED, and the line that used to scrub it said "the H3
    // runner stacks nothing" — true when it was written, false since the H3
    // LoRA import shipped in 3.7.0. H3 takes ONE adapter from its own family
    // through the same `--lora` flag Turbo rides, the picker offers exactly
    // that family (it filters on the engine's own `lora_tag`, "video:h3"), and
    // `h3_lora_slot` exists for the sole purpose of deciding whether that slot
    // spends on Turbo or on the user's pick. Blanking the field here made the
    // whole picker a control that renders, accepts a selection, and posts
    // nothing — the silent-no-op class this release keeps closing.
    //
    // Cross-lane safety is the SERVER's job and it already does it: make_job
    // filters the stack by the directory each file lives in and logs
    // "Dropped N LoRA(s) that belong to the other video engine". That is
    // strictly better than blanking here, because the user is told why.
  }

  // Disable the Generate button while we POST to /queue/add so a fast
  // double-click doesn't queue the same job twice. The button is
  // re-enabled after poll() returns (or on error). Keep this ABOVE the
  // LoRA-orphan confirm() so a user who cancels the confirm doesn't
  // leave the button stuck disabled.
  const genBtn = document.getElementById('genBtn');
  const reenable = () => { if (genBtn) genBtn.disabled = false; };

  // LAST LINE OF DEFENCE. FormData reads the hidden #mode input directly —
  // it never consults currentMode — so every guard above this point is a
  // guard on how #mode got its value, and this is the only one that reads
  // what is actually about to be POSTed. Cheap, and it is the assertion the
  // other three are trying to keep true.
  if (fd.get('mode') === 'ingredients' && !ingredientsServed()) {
    alert('Ingredients needs the LTX-2.3 generation — its reference adapter '
        + 'has no 2.5 release yet, so on 2.5 the references are ignored and '
        + 'the clip costs full two-stage time.\n\n'
        + 'For reference-guided work on 2.5, use Image mode with Inspire — '
        + 'or install the 2.3 pack from the Train tab.');
    reenable();
    return;
  }

  // Image mode with nothing in the Image slot used to queue, fail on the
  // server, and show up as a red card — one person hit it six times in a
  // row (fleet, 2026-09-02). Say it before the click costs anything.
  const _modeNow = String(fd.get('mode') || '');
  if ((_modeNow === 'i2v' || _modeNow === 'i2v_clean_audio')
      && !String(fd.get('image') || '').trim()) {
    alert('Image mode needs a reference image — drop one into the Image slot, '
        + 'or switch to Text mode and render from the prompt alone.');
    reenable();
    return;
  }

  // Safety net: if the prompt mentions a trigger word from a LoRA the user
  // has installed but NOT toggled active for this render, ask before
  // submitting. The #1 silent-failure mode is "I typed my LoRA's trigger
  // word but forgot to switch on the picker chip" — the render then runs
  // without fusion, and the face/style doesn't reproduce. Compare prompt
  // vs. (_knownUserLoras minus _activeLoras) and surface a confirm() so
  // the mismatch isn't invisible.
  try {
    const promptRaw = (fd.get('prompt') || '').toString();
    const promptLower = promptRaw.toLowerCase();
    // Skipped on H3: the scrub above already dropped the LoRA stack, so the
    // confirm would be warning about fusion that was never going to happen on
    // an engine that stacks nothing.
    if (promptLower.trim() &&
        (fd.get('engine') || 'ltx') !== 'h3' &&
        Array.isArray(_knownUserLoras) &&
        Array.isArray(_activeLoras)) {
      // COVERED-NESS IS A PROPERTY OF THE TRIGGER, NOT OF A PATH. Two blind
      // spots made this warn about triggers that were fully attached:
      //
      //   1. It only looked at _activeLoras — the user-LoRA picker. A CAST
      //      CHARACTER's stack is expanded server-side from character_id and
      //      never appears there, so casting bizarrotrn and writing
      //      "bizarrotrn" warned that bizarrotrn was not attached while it was
      //      about to be fused first in the stack.
      //   2. It matched attached-ness by PATH. A library holding two entries
      //      for one trigger — the bundle copy and a "bizarrotrn (high)" variant
      //      — flagged the one the user had NOT toggled even though the trigger
      //      was already covered by the one they had.
      //
      // So: build the set of triggers ANY attached source carries, and only
      // warn about a trigger that appears in none of them.
      const coveredTriggers = new Set();
      const addTrigger = (w) => {
        const t = String(w || '').toLowerCase().trim();
        if (t) coveredTriggers.add(t);
      };
      for (const l of _activeLoras) {
        for (const w of (l.trigger_words || [])) addTrigger(w);
        // A user LoRA row may carry only a path; recover its triggers from the
        // library entry so an attached-by-path LoRA still covers its words.
        const known = (_knownUserLoras || []).find(k => k.path === l.path);
        if (known) for (const w of (known.trigger_words || [])) addTrigger(w);
      }
      // The cast character's own trigger(s) — the registry knows them, and the
      // backend expands them into the stack, so they are attached by definition.
      const castId = (fd.get('character_id') || '').toString().trim();
      if (castId) {
        addTrigger(castId);
        const chars = (typeof _manualCharacters !== 'undefined' && Array.isArray(_manualCharacters))
          ? _manualCharacters : [];
        const cast = chars.find(c => c && c.id === castId);
        if (cast) {
          addTrigger(cast.trigger);
          for (const w of (cast.trigger_words || [])) addTrigger(w);
        }
      }
      const orphans = [];   // [{name, trigger}]
      const seenTrigger = new Set();
      const wordRe = /[a-z0-9]+/g;
      const promptTokens = new Set(promptLower.match(wordRe) || []);
      for (const ul of _knownUserLoras) {
        for (const w of (ul.trigger_words || [])) {
          if (!w) continue;
          const wLower = String(w).toLowerCase().trim();
          if (!wLower || wLower.length < 4) continue;   // skip 1-3 char tokens, too common
          if (coveredTriggers.has(wLower)) continue;    // attached by SOME source
          if (seenTrigger.has(wLower)) continue;        // one warning per trigger
          if (promptTokens.has(wLower)) {
            seenTrigger.add(wLower);
            orphans.push({ name: ul.name || ul.filename || ul.path.split('/').pop(), trigger: w });
            break;   // one match per LoRA is enough
          }
        }
      }
      if (orphans.length) {
        const lines = orphans.map(o => `  • "${o.trigger}" → ${o.name}`).join('\n');
        const ok = window.confirm(
          'Your prompt mentions trigger word(s) from LoRA(s) that are NOT attached:\n\n' +
          lines + '\n\n' +
          'The model will NOT reproduce these characters/styles. Toggle the LoRA on in the picker, then Generate.\n\n' +
          'Generate without the LoRA anyway?'
        );
        if (!ok) { reenable(); return; }
      }
    }
  } catch (_) { /* never block submit on this check */ }

  if (genBtn) genBtn.disabled = true;

  const noMusic = document.getElementById('noMusic');
  if (noMusic && noMusic.checked) {
    const original = fd.get('prompt') || '';
    const lower = original.toLowerCase();
    if ((fd.get('engine') || 'ltx') === 'h3') {
      // H3's trained mechanism, not LTX's phrasing. `non_diegetic_music` is one
      // of the three fields its encoder was trained on and `N/A` is the trained
      // value for "no score" (H3_PROMPTING_GUIDE §2.5, and §7.4 lists it as one
      // of the three ways a refusal can work at all on a model with no
      // unconditional branch). Appending a CFG-era "no music, no soundtrack, no
      // score" list here would be the wrong idiom AND would spend prompt budget
      // naming the thing we don't want — which §7.6 warns can summon it.
      // Skipped when the prompt already sets the field: a user writing the
      // three-field format owns that line, and a second one would contradict it.
      if (!lower.includes('non_diegetic_music')) {
        fd.set('prompt', original.trim() + '\n\nnon_diegetic_music: N/A');
      }
    } else {
      const constraint = ' Audio: voice and ambient sounds only, no music, no soundtrack, no score, no melody.';
      if (!lower.includes('no music')) {
        fd.set('prompt', original.trim() + constraint);
      }
    }
  }
  // No voice — drops the character's audio LoRA server-side (see
  // make_job character_id branch) AND nudges the prompt toward ambient
  // audio so the model doesn't fill the silence with generic speech.
  // GATED on a cast character: "only visible when a character is selected"
  // was the old justification for not checking, and the journey audit
  // showed the hidden checkbox could stay checked across a mode switch —
  // quietly rewriting a plain T2V prompt with a no-speech constraint.
  const noVoice = document.getElementById('noVoice');
  if (noVoice && noVoice.checked && (fd.get('character_id') || '')) {
    const original = fd.get('prompt') || '';
    const constraint = ' Audio: ambient sounds and environmental noise only, no speech, no dialogue, no narration.';
    if (!original.toLowerCase().includes('no speech') &&
        !original.toLowerCase().includes('no dialogue')) {
      fd.set('prompt', original.trim() + constraint);
    }
  }
  try {
    // Keyframe mode: FFLF keeps the legacy start/end shape. Dynamic
    // multi-anchor mode serializes all anchors to keyframes_json so the
    // helper's native multi-keyframe path receives explicit frame indices.
    const kfMode = (fd.get('mode') || '').toString();
    if (kfMode === 'keyframe') {
      const startImg = (fd.get('start_image') || '').toString().trim();
      const endImg = (fd.get('end_image') || '').toString().trim();
      if (!startImg || !endImg) {
        alert('Pick both a start frame and an end frame before generating.');
        reenable();
        return;
      }
      const frames = parseInt((fd.get('frames') || '121').toString()) || 121;
      if (window._kfMode >= 3) {
        renderKeyframeDynamicSlots();
        const slots = [
          { image: startImg, frame: 0, label: 'Start' },
          ...keyframeTimingSlots().map(slot => ({
            image: (document.getElementById(slot.imageKey)?.value || '').toString().trim(),
            frameId: slot.frameId,
            label: slot.label,
          })),
          { image: endImg, frame: frames - 1, label: 'End' },
        ];
        const missing = slots.filter(s => !s.image).map(s => s.label || 'Start/End');
        if (missing.length) {
          alert(`Pick all ${window._kfMode} keyframe images before generating: ` + missing.join(', '));
          reenable();
          return;
        }
        syncKeyframeTiming();
        const kfList = slots.map((slot, i) => {
          const idx = (slot.frame != null)
            ? slot.frame
            : parseInt((document.getElementById(slot.frameId)?.value || '').toString(), 10);
          return { image_path: slot.image, frame_index: idx };
        });
        const idxs = kfList.map(k => k.frame_index);
        if (idxs.some(i => !Number.isFinite(i)) || idxs.some((i, n) => n > 0 && i <= idxs[n - 1])) {
          alert('Keyframe times must be strictly increasing. Check the Beat at(s) values.');
          reenable();
          return;
        }
        fd.set('keyframes_json', JSON.stringify(kfList));
        fd.set('keyframes_total_frames', String(frames));
        fd.set('keyframe_count', String(window._kfMode));
      } else {
        fd.delete('keyframes_json');
        fd.delete('keyframes_total_frames');
        fd.delete('keyframe_count');
      }
    }
    // STG "detail guidance" — explicit fd.set so the value is unambiguous
    // and so it only rides along when it can actually do something. STG acts
    // only on the Q8 HQ path (quality=high); the Q4 distilled paths ignore it
    // entirely (DistilledPipeline runs no guider). On any non-high quality we
    // drop the field so a stale slider value can't reach the worker. The range
    // input already carries name="stg_scale", but the FormData copy is
    // post-processed here so this set() wins regardless of input order.
    {
      const _q = (fd.get('quality') || '').toString();
      const _stgEl = document.getElementById('stgScale');
      if (_qualityUsesHq(_q) && _stgEl) {
        fd.set('stg_scale', _stgEl.value || '0');
      } else {
        fd.delete('stg_scale');
      }
    }
    // Hailuo H3: send the selected CELL's geometry explicitly. make_job
    // re-stamps it server-side too (a stale tab must never win), but posting the
    // truth means the queue card is right the instant the job lands, and it
    // can't carry an LTX 8k+1 frame count into a runner that snaps to 17n+5.
    // The cell is looked up from the two axes — the same resolution order
    // make_job uses — and the composite `h3_tier` is re-set from it so the three
    // fields can never disagree on the wire.
    if ((fd.get('engine') || 'ltx').toString() === 'h3') {
      const _t = (typeof h3CellFor === 'function')
        ? (h3CellFor((fd.get('h3_quality') || '').toString(),
                     (fd.get('h3_length') || '').toString())
           || h3TierByKey((fd.get('h3_tier') || '').toString()))
        : null;
      if (_t) {
        fd.set('h3_tier', String(_t.key));
        fd.set('h3_quality', String(_t.quality));
        fd.set('h3_length', String(_t.length));
        fd.set('width', String(_t.width));
        fd.set('height', String(_t.height));
        fd.set('frames', String(_t.frames));
        fd.set('steps', String(_t.steps));
      }
    }
    await api('/queue/add','POST',fd);
  } finally {
    // Re-enable on the next event-loop tick so the button visibly
    // bounces rather than feeling "stuck on click". poll() refreshes
    // the queue right after, picking up the new job in the next ~1.5s.
    setTimeout(reenable, 200);
  }
  poll();
});


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  notifyJobsDone, notifyOneJob, playDoneChime,
  h3FinishSetTier, h3FinishActive, setEngine, _syncEnginePromptTools,
  currentEngine, _syncEngineForMode, openH3InstallCard, closeH3InstallCard,
  enhancePrompt, applyAspect, applyQuality, updateDerived,
  pickerSetImage, pickerUploadFile, pickerWire, refreshUploadsStrip,
  refreshIngredientRecent, ingredientPickerWire, fmtMin, snippet,
  escapeHtml, api, _setOfflineBanner, startDeepVerify,
  friendlyJobError, poll, applyPackIncompleteGate, setRecentFilter,
  retryJob, renderCarousel, findOutputByPath, stageMayAutoSelectOutput,
  selectOutput, openExpandLightbox, closeExpandLightbox, phosToast,
  animateActive, hide, openOutputsFolder, hideActive,
  useAsExtendSource, useAsUpscaleSource, useAsUpscaleSourcePath, setUpscalePreset, loadParams, _flashActionDone, closeOutputInfoModal,
  togglePause, openBatch, closeBatch, queueBatch,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  _copyToClipboard, animateFromPhoto, deleteOutput, openOutputInfoModal,
  remakeInQuality, removeJob, repairModel,
});
