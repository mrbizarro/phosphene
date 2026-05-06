const BOOT = window.BOOT;
const ASPECTS = BOOT.aspects;
const FPS = BOOT.fps;
const MODEL_UPSCALE_ENABLED = !!BOOT.model_upscale_enabled;
const PIPERSR_UPSCALE_ENABLED = !!BOOT.pipersr_upscale_enabled;

// Apply tier-aware time estimates to the Quality pill subtitles. The HTML
// ships with the Comfortable-tier (M4 Studio 64 GB) numbers as defaults;
// users on Compact / Roomy / Studio tiers see realistic estimates instead
// of the optimistic baseline. Runs once on boot, plus when the tier modal
// reports new info (rare — tier is fixed for a given Mac).
function applyTierTimes() {
  const qt = (BOOT.quality_times || {});
  document.querySelectorAll('#qualityGroup .pill-quality').forEach(btn => {
    const key = btn.dataset.quality;
    const time = qt[key];
    const spec = btn.querySelector('.ql-spec');
    if (!spec) return;
    const dimsMatch = spec.textContent.match(/^([0-9]+×[0-9]+(\s+→\s+[0-9p]+)?)/);
    const dims = dimsMatch ? dimsMatch[1] : '';
    if (time && dims) {
      spec.textContent = `${dims} · ${time}`;
    } else if (time) {
      spec.textContent = time;
    }
  });
}

let filterMode = 'visible';
let activePath = null;
let currentOutputs = [];
let currentMode = 't2v';

// Model tag in the bottom-pane nav links to dgrauet's repo. Strip an
// absolute filesystem path back to the HF repo id form for display
// (the panel sets LTX_MODEL to a local path in Pinokio installs).
(function () {
  const m = String(BOOT.model || '');
  let label = m;
  const idx = m.indexOf('mlx_models/');
  if (idx >= 0) label = m.slice(idx + 'mlx_models/'.length);
  if (label.startsWith('/')) label = label.split('/').slice(-2).join('/');
  document.getElementById('modelTag').textContent = label;
})();
// `audio` is still a free-text input (advanced section); `image` is now a
// picker — leave the picker empty by default and let the user pick or
// drop. Pre-filling examples/reference.png surprised users into rendering
// the demo image when they meant to leave it blank.
document.getElementById('audio').value = BOOT.default_audio;

// ====== Pill-button group helpers ======
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode').value = mode;
  document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  // For i2v, switch the actual mode based on the i2vMode select
  if (mode === 'i2v') {
    document.getElementById('mode').value = document.getElementById('i2vMode').value;
  }
  // Keyframe REQUIRES Q8 (uses dev transformer); force quality=high.
  // If Q8 isn't available the High pill stays disabled and the user gets the
  // same "Q8 not installed" hint as elsewhere.
  if (mode === 'keyframe') {
    setQuality('high');
  }
  updateAccelAvailability();
  updateTemporalAvailability();
  updateDerived();
  // Refresh the inline models card immediately — switching to FFLF when
  // Q8 is missing should surface the Download Q8 CTA without waiting for
  // the next 1.5s poll tick.
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
// Quality presets (Y1.013) — each one bundles the backend quality value
// (which selects the model + sampler) with the canonical dimensions.
// Backend still routes only on `quality == 'high'` vs anything else, so
  // 'quick', 'balanced', and 'standard' all run Q4 distilled — they differ in
// pixel count. The richer label is preserved in the sidecar so the
// info modal can show "Quick" / "Standard" / "High" verbatim.
const QUALITY_PRESETS = {
  quick:    { w: 640,  h: 480, upscale: 'off' },        // 4:3, fastest sanity check
  balanced: { w: 1024, h: 576, upscale: 'fit_720p' },   // exact 16:9 → 1280×720
  standard: { w: 1280, h: 704, upscale: 'off' },        // LTX-wide canonical render
  high:     { w: 1280, h: 704, upscale: 'off' },        // same dims, different model (Q8)
};

function setQuality(q) {
  // Tolerate legacy values from old sidecars: 'draft' → 'standard'.
  if (q === 'draft' || !QUALITY_PRESETS[q]) q = 'standard';
  document.getElementById('quality').value = q;
  document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.quality === q));
  // Set canonical dimensions for the preset, respecting the current
  // aspect choice. Quick is 4:3 only — landscape orientation only.
  const preset = QUALITY_PRESETS[q];
  const aspect = document.getElementById('aspect').value || 'landscape';
  const vertical = (aspect === 'vertical' && q !== 'quick');
  document.getElementById('width').value  = vertical ? preset.h : preset.w;
  document.getElementById('height').value = vertical ? preset.w : preset.h;
  setUpscale(preset.upscale || 'off');
  // Hide the Aspect row when Quick is active (only 4:3 supported); show
  // it for Standard/High where 16:9 vs 9:16 is a real choice.
  const aspectRow = document.getElementById('aspectRow');
  if (aspectRow) aspectRow.style.display = (q === 'quick') ? 'none' : '';
  applyQuality();
  updateAccelAvailability();
  updateTemporalAvailability();
  updateCustomizeSummary();
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
function setAccel(a) {
  const allowed = document.getElementById('quality').value !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe';
  const v = allowed ? a : 'off';
  document.getElementById('accel').value = v;
  document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.accel === v));
  updateCustomizeSummary();
  updateDerived();
}
function temporalModeAllowed() {
  const q = document.getElementById('quality').value;
  const mode = document.getElementById('mode').value;
  return q !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe' && (mode === 't2v' || mode === 'i2v');
}
function setTemporalMode(t) {
  const allowed = temporalModeAllowed();
  const v = (allowed && t === 'fps12_interp24') ? 'fps12_interp24' : 'native';
  document.getElementById('temporal_mode').value = v;
  document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.temporal === v));
  updateCustomizeSummary();
  updateDerived();
}
function setUpscale(u) {
  const v = ['off', 'fit_720p', 'x2'].includes(u) ? u : 'off';
  document.getElementById('upscale').value = v;
  document.querySelectorAll('#upscaleGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.upscale === v));
  // Show / hide the Method pill row — only relevant when an upscale is
  // actually being applied. When toggled to "off", revert method to Fast
  // so a later toggle back to fit_720p starts from the safe default.
  const methodRow = document.getElementById('upscaleMethodRow');
  if (methodRow) methodRow.style.display = (v === 'off' || !PIPERSR_UPSCALE_ENABLED) ? 'none' : '';
  if (v === 'off' || !PIPERSR_UPSCALE_ENABLED) setUpscaleMethod('lanczos');
  updateCustomizeSummary();
  updateDerived();
}
function setUpscaleMethod(m) {
  if (m === 'model') m = 'pipersr'; // legacy sidecars from the retired LTX Sharp path
  const v = (PIPERSR_UPSCALE_ENABLED && m === 'pipersr') ? 'pipersr' : 'lanczos';
  document.getElementById('upscale_method').value = v;
  document.querySelectorAll('#upscaleMethodGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.method === v));
  updateCustomizeSummary();
  updateDerived();
}
function updateAccelAvailability() {
  const allowed = document.getElementById('quality').value !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe';
  document.querySelectorAll('#accelGroup .pill-btn').forEach(b => {
    const disabled = !allowed && b.dataset.accel !== 'off';
    b.classList.toggle('disabled', disabled);
  });
  if (!allowed && document.getElementById('accel').value !== 'off') setAccel('off');
}
function updateTemporalAvailability() {
  const allowed = temporalModeAllowed();
  document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => {
    const disabled = !allowed && b.dataset.temporal !== 'native';
    b.classList.toggle('disabled', disabled);
    if (b.dataset.temporal === 'fps12_interp24') {
      b.title = allowed
        ? 'Generate at 12fps, then interpolate to a normal 24fps export.'
        : 'Available for Q4 Text/Image renders. Off for High, FFLF, Extend, and external-audio I2V.';
    }
  });
  if (!allowed && document.getElementById('temporal_mode').value !== 'native') setTemporalMode('native');
}
function setAspect(a) {
  if (!ASPECTS[a]) return;
  document.getElementById('aspect').value = a;
  document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.aspect === a));
  applyAspect(a);
}

// Compose the right-aligned line in the Customize summary. Reflects the
// current effective state: aspect, custom-dims callout, speed setting.
function updateCustomizeSummary() {
  const el = document.getElementById('customizeSummary');
  if (!el) return;
  const q = document.getElementById('quality').value;
  const w = parseInt(document.getElementById('width').value || 0);
  const h = parseInt(document.getElementById('height').value || 0);
  const aspect = document.getElementById('aspect').value || 'landscape';
  const accel = document.getElementById('accel').value || 'off';
  const upscale = document.getElementById('upscale').value || 'off';
  const parts = [];
  // Aspect (Quick is fixed 4:3, no choice; Standard/High show landscape/vertical).
  if (q === 'quick') parts.push('4:3 · 640×480');
  else parts.push(aspect === 'vertical' ? '9:16' : '16:9');
  // Flag custom dims if they don't match the preset.
  const preset = QUALITY_PRESETS[q] || QUALITY_PRESETS['standard'];
  const vertical = (aspect === 'vertical' && q !== 'quick');
  const expectedW = vertical ? preset.h : preset.w;
  const expectedH = vertical ? preset.w : preset.h;
  if (q !== 'quick' && (w !== expectedW || h !== expectedH)) {
    parts.push(`${w}×${h} custom`);
  }
  // Speed
  parts.push(accel === 'off' ? 'exact speed' : (accel === 'boost' ? 'boost' : 'turbo'));
  if ((document.getElementById('temporal_mode')?.value || 'native') === 'fps12_interp24') {
    parts.push('12→24fps long clip');
  }
  const method = (document.getElementById('upscale_method')?.value || 'lanczos');
  const methodTag = method === 'pipersr' || method === 'model' ? ' sharp' : '';
  if (upscale === 'fit_720p') parts.push('720p export' + methodTag);
  else if (upscale === 'x2') parts.push('2× export' + methodTag);
  el.textContent = parts.join(' · ');
}
function setExtendMode(m) {
  // Fast = no-CFG path, fits in 64 GB at 1280×704. Quality = upstream
  // defaults, requires headroom. Both are exposed on the form via hidden
  // inputs; this just flips the values + active pill.
  const steps = m === 'quality' ? 30  : 12;
  const cfg   = m === 'quality' ? 3.0 : 1.0;
  document.getElementById('extend_steps').value = String(steps);
  document.getElementById('extend_cfg').value   = String(cfg);
  document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.extendMode === m));
}

document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setQuality(b.dataset.quality); });
document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setAccel(b.dataset.accel); });
document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setTemporalMode(b.dataset.temporal); });
document.querySelectorAll('#upscaleGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscale(b.dataset.upscale); });
document.querySelectorAll('#upscaleMethodGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscaleMethod(b.dataset.method); });
document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.onclick = () => setAspect(b.dataset.aspect));
document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b => b.onclick = () => setExtendMode(b.dataset.extendMode));

// Prompt enhancement via Gemma — wraps the upstream CLI's `enhance`
// subcommand. Cold start ~12-15s (Gemma load), warm ~5s. Blocks the UI
// during the request (just the button — rest of the form stays usable).
async function enhancePrompt() {
  const ta = document.getElementById('prompt');
  const original = ta.value.trim();
  if (!original) { alert('Type a prompt before enhancing it.'); return; }
  const mode = (currentMode === 'i2v' || currentMode === 'keyframe' || currentMode === 'extend') ? 'i2v' : 't2v';
  const btn = document.getElementById('enhanceBtn');
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = '✨ Loading Gemma… (~15s on cold start)';
  let res;
  try {
    const fd = new URLSearchParams({ prompt: original, mode });
    const r = await fetch('/prompt/enhance', { method: 'POST', body: fd });
    res = await r.json();
  } catch (e) {
    alert('Enhance request failed: ' + (e.message || e));
    btn.disabled = false; btn.textContent = originalLabel;
    return;
  }
  btn.disabled = false; btn.textContent = originalLabel;
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
  if (q === 'high') {
    document.getElementById('steps').value = 18;
  } else {
    document.getElementById('steps').value = 8;       // quick + balanced + standard
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
  // Nearest 8k+1: round (v-1)/8 to nearest int, multiply back, +1.
  const k = Math.max(1, Math.round((v - 1) / 8));
  const snapped = k * 8 + 1;
  if (snapped !== v) {
    el.value = snapped;
    // Reflect the change in duration too, since they're bound.
    document.getElementById('duration').value = framesToDuration(snapped);
  }
}

function updateDerived() {
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

  const warns = [];
  if (w % 32 !== 0) warns.push(`Width ${w} isn't a multiple of 32 (closest ${Math.round(w/32)*32})`);
  if (h % 32 !== 0) warns.push(`Height ${h} isn't a multiple of 32 (closest ${Math.round(h/32)*32})`);
  if (f > 1 && (f - 1) % 8 !== 0) {
    const closest = Math.max(1, Math.round((f - 1) / 8) * 8 + 1);
    warns.push(`Frames work best as 8k+1 (closest ${closest})`);
  }
  if (temporal === 'fps12_interp24') {
    warns.push('12→24fps is experimental; check dialogue lip-sync and fast motion');
  }
  const banner = document.getElementById('warnBanner');
  if (warns.length) { banner.innerHTML = '⚠ ' + warns.join(' · '); banner.classList.add('show'); }
  else banner.classList.remove('show');

  // Mode-aware visibility
  const inI2V = mode === 'i2v' || mode === 'i2v_clean_audio';
  const inImageFlow = inI2V || currentMode === 'keyframe';
  document.getElementById('imageSection').classList.toggle('show', inI2V && currentMode !== 'keyframe');
  document.getElementById('extendSection').classList.toggle('show', currentMode === 'extend');
  document.getElementById('keyframeSection').classList.toggle('show', currentMode === 'keyframe');
  document.getElementById('sizingSection').classList.toggle('show', currentMode !== 'extend');
  document.getElementById('audioSection').classList.toggle('show', mode === 'i2v_clean_audio');
  // I2V audio source picker (Advanced) — only relevant in I2V flow.
  // In T2V/Extend/FFLF the model generates audio jointly; there's nothing
  // to swap out, so the dropdown is just noise.
  const i2vAudioSec = document.getElementById('i2vAudioModeSection');
  if (i2vAudioSec) i2vAudioSec.classList.toggle('show', inI2V);
  // In image flows the aspect picker is the only sizing control. Width/height
  // auto-derive from aspect+quality so the source image drives the framing
  // and we don't accidentally cover-crop a 16:9 photo into 9:16.
  const dimsRow = document.getElementById('dimsRow');
  if (dimsRow) dimsRow.style.display = inImageFlow ? 'none' : '';

  // Image previews are now part of the picker component itself — the
  // preview <img> + clear button live inside .picker-drop and are toggled
  // by pickerSetImage(). No per-mode preview management here anymore;
  // the old imagePreview / startImagePreview / endImagePreview elements
  // are gone.
}

['width','height','frames','duration'].forEach(id => {
  const el = document.getElementById(id);
  if (id === 'duration') {
    el.addEventListener('input', e => { document.getElementById('frames').value = durationToFrames(parseFloat(e.target.value) || 0); updateDerived(); });
  } else if (id === 'frames') {
    el.addEventListener('input', e => { document.getElementById('duration').value = framesToDuration(parseInt(e.target.value) || 0); updateDerived(); });
    el.addEventListener('blur', () => { snapFramesTo8kPlus1(); updateDerived(); });
  } else {
    // width / height: also refresh the Customize summary so "custom" flags
    // appear/disappear as the user types away from the preset values.
    el.addEventListener('input', () => { updateCustomizeSummary(); updateDerived(); });
  }
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
// One implementation, three call sites: I2V image, FFLF start_image,
// FFLF end_image. Each picker carries a `key` (the hidden field's name);
// every DOM element it owns is suffixed with `_<key>` so we can wire
// listeners by lookup instead of a per-instance closure.
const PICKERS = ['image', 'start_image', 'end_image'];

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
    els.preview.src = `/image?path=${encodeURIComponent(path)}`;
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
    // image. End frame doesn't drive aspect (would override the start frame).
    if (key !== 'end_image' && opts.snapAspect !== false) {
      snapAspectToImage(path);
    }
  } else {
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

let _uploadsCache = [];   // last fetched list, kept module-level so all
                          //   three pickers render the same source data.
async function refreshUploadsStrip() {
  let data;
  try { data = await api('/uploads?limit=24'); }
  catch (e) { return; }
  _uploadsCache = data.uploads || [];
  PICKERS.forEach(key => {
    const els = pickerEls(key);
    if (!els.recentStrip) return;
    if (!_uploadsCache.length) {
      els.recentWrap.style.display = 'none';
      return;
    }
    els.recentWrap.style.display = '';
    const currentPath = els.hidden.value;
    els.recentStrip.innerHTML = _uploadsCache.map(u => `
      <img class="picker-recent-thumb${u.path === currentPath ? ' selected' : ''}"
           src="${escapeHtml(u.url)}"
           data-path="${escapeHtml(u.path)}"
           title="${escapeHtml(u.name)} · ${u.size_kb} KB · ${escapeHtml(u.mtime)}"
           alt="">
    `).join('');
    els.recentStrip.querySelectorAll('img').forEach(img => {
      img.addEventListener('click', () => pickerSetImage(key, img.dataset.path));
    });
  });
}

// ====== Format helpers ======
function fmtMem(m) { return `${m.used_gb.toFixed(1)} / ${m.total_gb.toFixed(0)} GB · swap ${m.swap_gb.toFixed(1)}`; }
function fmtMin(s) { if (!s || s < 0) return '—'; const m = Math.floor(s/60); const sec = Math.round(s%60); return m > 0 ? `${m}m ${sec}s` : `${sec}s`; }
function snippet(s, n = 70) { if (!s) return ''; s = s.replace(/\s+/g,' ').trim(); return s.length > n ? s.slice(0, n-1)+'…' : s; }
function escapeHtml(s) { if (!s) return ''; return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function api(path, method = 'GET', body = null) {
  const opts = { method };
  if (body) {
    opts.body = body instanceof FormData ? new URLSearchParams(body) : body;
    opts.headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
  }
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 409) throw new Error(`${path}: ${r.status}`);
  return r.status === 409 ? { error: 'busy' } : r.json().catch(() => ({}));
}

// ====== Poll ======
// Cache of the latest /status response so non-poll callers (setMode,
// setQuality) can refresh tier-gated UI without waiting for the next tick.
let LAST_STATUS = null;

// Tracks consecutive /status failures so we can surface a panel-offline
// banner instead of silently freezing the UI. Two-strike threshold so a
// single transient hiccup (network blip, panel reload) doesn't flash.
let _POLL_FAILS = 0;

function _setOfflineBanner(visible, msg) {
  let bar = document.getElementById('panelOfflineBanner');
  if (visible) {
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
    bar.querySelector('.text').textContent =
      msg || "uploads, chat & renders are paused";
  } else if (bar) {
    bar.remove();
  }
}

async function poll() {
  let s;
  const url = '/status' + (filterMode === 'hidden' ? '?include_hidden=1' : '');
  try {
    s = await (await fetch(url)).json();
    _POLL_FAILS = 0;
    _setOfflineBanner(false);
  } catch (e) {
    _POLL_FAILS += 1;
    if (_POLL_FAILS >= 2) _setOfflineBanner(true);
    return;
  }
  LAST_STATUS = s;

  // Memory
  const m = s.memory;
  const memPill = document.getElementById('memPill');
  memPill.innerHTML = `<span class="dot"></span>${fmtMem(m)}`;
  let memCls = 'pill-good';
  if (m.swap_gb > 8 || m.pressure_pct > 90) memCls = 'pill-danger';
  else if (m.swap_gb > 4 || m.pressure_pct > 75) memCls = 'pill-warn';
  memPill.className = 'pill ' + memCls;

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

  // Queue pill + tab badge
  const qp = document.getElementById('queuePill');
  qp.innerHTML = `<span class="dot"></span>queue ${s.queue.length}${s.paused ? ' · paused' : ''}`;
  qp.className = 'pill ' + (s.paused ? 'pill-warn' : (s.queue.length ? 'pill-running' : ''));
  const qb = document.getElementById('queueBadge');
  if (s.queue.length) { qb.textContent = s.queue.length; qb.style.display = ''; } else { qb.style.display = 'none'; }

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

  // Q8 / High enable
  const highBtn = document.getElementById('qualityHigh');
  const highSub = document.getElementById('highSub');
  if (s.q8_available) {
    highBtn.classList.remove('disabled');
    highSub.textContent = 'Q8 + TeaCache';
  } else {
    highBtn.classList.add('disabled');
    const missing = (s.q8_missing || []).length;
    highSub.textContent = missing > 0 && missing < 6 ? `Q8 downloading · ${missing} files left` : 'Q8 not installed';
    if (document.getElementById('quality').value === 'high') setQuality('standard');
  }

  // Keyframe (FFLF) and Extend both require Q8 — server enforces it (see
  // run_job_inner). The UI was previously silently downgrading the user to
  // Standard when they picked keyframe with Q8 missing, then the server
  // would 500 on submit. Disable Generate + show a clear reason while in
  // that state. Y1.036 added Extend to the same gate after the Y1.024
  // download trim exposed that Extend is structurally Q8-class.
  const genBtn = document.getElementById('genBtn');
  const q8GatedMode = (currentMode === 'keyframe' || currentMode === 'extend');
  if (q8GatedMode && !s.q8_available) {
    genBtn.disabled = true;
    const modeName = currentMode === 'keyframe' ? 'Keyframe (FFLF)' : 'Extend';
    const left = (s.q8_missing || []).length;
    genBtn.title = left > 0 && left < 6
      ? `${modeName} needs Q8 — ${left} file(s) still downloading.`
      : `${modeName} needs the Q8 model. Click "Download Q8 (~37 GB)" in Pinokio.`;
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
    const baseMeta = `${s.current.params.mode} · ${s.current.params.width}×${s.current.params.height} · ${s.current.params.frames}f · ${timing}`;
    nowCard.querySelector('.meta').innerHTML = phaseLabel
      ? `${baseMeta}<br><span style="color:var(--muted)">${escapeHtml(phaseLabel)}</span>`
      : baseMeta;
  } else {
    // Idle state. If the LAST history entry was a failure (helper crash,
    // OOM, etc.) surface it loud-and-clear here — otherwise users like
    // cocktailpeanut just see "Idle" and assume "the panel did nothing."
    // We hold the failure visible until the user starts a new job.
    fill.style.width = '0%';
    const last = (s.history || [])[0];
    const showFailure = last && last.status === 'failed' && !s.queue.length;
    if (showFailure) {
      nowCard.classList.remove('idle');
      nowCard.classList.add('failed');
      // Translate cryptic engine errors into actionable user guidance.
      // "helper died mid-job (no event)" is the SIGKILL-by-jetsam
      // signature on memory-pressured Macs — the helper subprocess gets
      // killed by the OS for using too much RAM and we never get an
      // event back. Tell the user how to recover instead of leaving them
      // with the engine wording.
      const raw = (last.error || 'unknown error');
      const rawLower = raw.toLowerCase();
      let friendly, hint;
      if (rawLower.includes('sigkill')) {
        friendly = 'Helper killed by the OS — out of memory (jetsam).';
        hint = 'Close memory-heavy apps (Chrome, Slack, iOS Simulator) and try again, ' +
               'or switch Quality to Quick (about half the RAM).';
      } else if (rawLower.includes('sigsegv') || rawLower.includes('sigbus')) {
        friendly = 'Helper crashed at the native level (MLX/Metal fault).';
        hint = 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
               'on github.com/mrbizarro/phosphene/issues so we can fix it.';
      } else if (rawLower.includes('sigabrt')) {
        friendly = 'Helper hit a C-level assertion and aborted.';
        hint = 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
               'on github.com/mrbizarro/phosphene/issues.';
      } else if (rawLower.includes('helper exited from') || rawLower.includes('helper pipe closed') ||
                 rawLower.includes('helper died') || rawLower.includes('helper exited')) {
        friendly = 'Helper exited unexpectedly.';
        hint = 'Check the log for the last "step:*" breadcrumb (tells us which ' +
               'phase died). If memory-pressured, close other apps and retry.';
      } else if (rawLower.includes('q8') || rawLower.includes('keyframe')) {
        friendly = 'This mode needs the Q8 model.';
        hint = raw;
      } else {
        friendly = 'Job failed.';
        hint = raw;
      }
      nowCard.querySelector('.ttl').innerHTML =
        `<span style="color: var(--danger, #f85149)">⚠ ${escapeHtml(friendly)}</span>`;
      nowCard.querySelector('.meta').innerHTML =
        `<span style="color: var(--muted)">${escapeHtml(snippet(last.params.label || last.params.prompt, 80))}</span>` +
        ` <span style="color: var(--muted)">· ${escapeHtml(last.params.mode)} · ${last.params.width}×${last.params.height}</span>` +
        `<br><span style="color: var(--text)">${escapeHtml(hint)}</span>`;
    } else {
      nowCard.classList.add('idle');
      nowCard.classList.remove('failed');
      nowCard.querySelector('.ttl').textContent = s.paused ? 'Paused' : 'Idle';
      nowCard.querySelector('.meta').textContent = s.paused
        ? 'Worker paused — current job (if any) finishes, queue waits for resume.'
        : (s.queue.length ? 'Worker about to pick up next queued job.' : 'No jobs queued. Generate something on the left.');
    }
  }

  // Logs
  const log = document.getElementById('log');
  const wasNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  log.textContent = s.log.length ? s.log.join('\n') : 'No log yet.';
  if (wasNearBottom) log.scrollTop = log.scrollHeight;

  // Queue list
  const ql = document.getElementById('queueList');
  if (!s.queue.length) ql.innerHTML = '<li class="empty-state"><span></span><span>Queue empty</span><span></span><span></span></li>';
  else ql.innerHTML = s.queue.map((j, i) => `
    <li>
      <span class="pos">#${i+1}</span>
      <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
      <span class="params">${j.params.mode} · ${j.params.width}×${j.params.height} · ${j.params.frames}f</span>
      <button title="Remove" onclick="removeJob('${j.id}')">×</button>
    </li>`).join('');

  // History — failed jobs show the error inline in the title slot, so
  // users can see WHY without having to scroll the log to find it.
  const hl = document.getElementById('historyList');
  if (!s.history.length) hl.innerHTML = '<li class="empty-state"><span></span><span>No history yet</span><span></span><span></span></li>';
  else hl.innerHTML = s.history.slice(0, 20).map(j => {
    const titleText = escapeHtml(j.params.label || snippet(j.params.prompt, 60));
    const titleAttr = escapeHtml(j.params.prompt || '');
    let titleHtml;
    if (j.status === 'failed' && j.error) {
      titleHtml = `${titleText} ` +
        `<span class="err-inline" title="${escapeHtml(j.error)}">— ${escapeHtml(snippet(j.error, 70))}</span>`;
    } else {
      titleHtml = titleText;
    }
    return `
    <li class="${j.status}">
      <span class="badge">${j.status}</span>
      <span class="ttl" title="${titleAttr}">${titleHtml}</span>
      <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
      <span></span>
    </li>`;
  }).join('');

  // Outputs / carousel
  if (JSON.stringify(currentOutputs) !== JSON.stringify(s.outputs)) {
    currentOutputs = s.outputs;
    renderCarousel();
    if (!activePath && currentOutputs.length) selectOutput(currentOutputs[0].path);
    const sel = document.getElementById('extendSrcSelect');
    sel.innerHTML = '<option value="">— pick an output below or paste a path —</option>' +
      currentOutputs.slice(0, 40).map(o => `<option value="${escapeHtml(o.path)}">${escapeHtml(o.name)}</option>`).join('');
  }
  document.getElementById('filterHidden').textContent = `Hidden${s.hidden_count ? ' ('+s.hidden_count+')' : ''}`;
  document.getElementById('carouselTitle').textContent = filterMode === 'hidden' ? 'Hidden outputs' : `Outputs · ${currentOutputs.length}`;
}

function setFilter(mode) {
  filterMode = mode;
  document.getElementById('filterAll').classList.toggle('active', mode === 'visible');
  document.getElementById('filterHidden').classList.toggle('active', mode === 'hidden');
  poll();
}

// Format render duration for the gallery card sub-line. Falls back to
// the time-of-day when the sidecar is missing (older outputs that
// pre-date the elapsed_sec field, or outputs whose sidecar got
// deleted) so the slot is never empty.
function _outputDurationLabel(o) {
  const s = (o && typeof o.elapsed_sec === 'number') ? o.elapsed_sec : null;
  if (s == null) {
    // Fallback: show time-of-day from mtime so empty cards aren't worse.
    return o.mtime ? o.mtime.slice(11, 16) : '—';
  }
  if (s < 60)    return `${Math.round(s)} s`;
  if (s < 3600)  return `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} m`;
}

function renderCarousel() {
  const el = document.getElementById('carousel');
  if (!currentOutputs.length) { el.innerHTML = '<div class="empty-msg">No outputs in this view yet.</div>'; return; }
  el.innerHTML = currentOutputs.map(o => {
    const pathAttr = JSON.stringify(o.path).replace(/"/g, '&quot;');
    // Thumbnail seek point: 2.5s is the midpoint of an LTX 5s clip (121
    // frames at 24fps ≈ 5.04s). The first half-second of LTX renders is
    // often dark/static (model fades into the scene), so #t=0.5 produced
    // "black thumbnail" complaints — the video was fine, the seek point
    // was the darkest moment. Mid-clip is reliably the visual peak.
    return `
    <div class="car-card${o.hidden ? ' hidden-card' : ''}${o.path === activePath ? ' active' : ''}"
         data-path="${escapeHtml(o.path)}" onclick="selectOutput(${pathAttr})">
      <video src="${o.url}#t=2.5" preload="metadata" muted></video>
      ${o.has_sidecar
        ? `<button class="car-info-btn" type="button" title="Show generation info"
                   onclick="event.stopPropagation(); openOutputInfoModal(${pathAttr})">ⓘ</button>`
        : ''}
      <div class="info">
        <div class="name" title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</div>
        <div class="sub" title="Render time · file size">
          ${_outputDurationLabel(o)} · ${o.size_mb.toFixed(1)} MB
        </div>
      </div>
      <div class="row-btns">
        <button onclick="event.stopPropagation(); ${o.hidden ? 'unhide' : 'hide'}(${pathAttr})">${o.hidden ? 'Show' : 'Hide'}</button>
        <button onclick="event.stopPropagation(); useAsExtendSourcePath(${pathAttr})">Extend</button>
      </div>
    </div>`;
  }).join('');
}

function selectOutput(path) {
  activePath = path;
  document.querySelectorAll('.car-card').forEach(el => el.classList.toggle('active', el.dataset.path === path));
  const wrap = document.getElementById('playerWrap');
  wrap.classList.remove('empty');
  // Y1.039 — use the server-provided URL (which includes the mtime
  // cache-bust v=N param) instead of reconstructing from path. Otherwise
  // the player ends up on the cached stale-bytes URL and re-shows black
  // until the browser cache expires.
  const o = currentOutputs.find(x => x.path === path);
  const playerSrc = o ? o.url : `/file?path=${encodeURIComponent(path)}`;
  wrap.innerHTML = `<video controls autoplay src="${playerSrc}"></video>`;
  document.getElementById('playerMeta').style.display = '';
  document.getElementById('playerName').innerHTML = o ? `<strong>${escapeHtml(o.name)}</strong> · ${o.mtime} · ${o.size_mb.toFixed(1)} MB` : '';
  document.getElementById('loadParamsBtn').disabled = !(o && o.has_sidecar);
}

async function hide(path) { await fetch('/output/hide?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }
async function unhide(path) { await fetch('/output/show?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }
function hideActive() { if (activePath) hide(activePath); }

function useAsExtendSourcePath(path) {
  setMode('extend');
  document.getElementById('video_path').value = path;
  document.getElementById('extendSrcSelect').value = path;
  updateDerived();
  document.querySelector('aside.form-pane').scrollTop = 0;
}
function useAsExtendSource() { if (!activePath) return alert('Pick an output first.'); useAsExtendSourcePath(activePath); }

async function loadParams() {
  if (!activePath) return;
  const r = await fetch('/sidecar?path='+encodeURIComponent(activePath));
  if (!r.ok) return;
  const data = await r.json();
  const p = data.params;
  if (p.mode === 'extend') setMode('extend');
  else if (p.mode === 'keyframe') setMode('keyframe');
  else if (p.mode === 'i2v_clean_audio' || p.mode === 'i2v') { setMode('i2v'); document.getElementById('i2vMode').value = p.mode; document.getElementById('mode').value = p.mode; }
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
  if (p.temporal_mode) setTemporalMode(p.temporal_mode);
  if (p.upscale) setUpscale(p.upscale);
  if (p.upscale_method) setUpscaleMethod(p.upscale_method);
  document.getElementById('prompt').value = p.prompt || '';
  document.getElementById('negative_prompt').value = p.negative_prompt || '';
  if (p.frames) { document.getElementById('frames').value = p.frames; document.getElementById('duration').value = framesToDuration(p.frames); }
  if (p.steps) document.getElementById('steps').value = p.steps;
  if (p.seed != null) document.getElementById('seed').value = p.seed;
  // Image / start / end go through pickerSetImage so the preview tile
  // and recent-strip selection state update along with the hidden input.
  if (p.image)       pickerSetImage('image', p.image, { snapAspect: false });
  if (p.start_image) pickerSetImage('start_image', p.start_image, { snapAspect: false });
  if (p.end_image)   pickerSetImage('end_image', p.end_image, { snapAspect: false });
  if (p.audio) document.getElementById('audio').value = p.audio;
  // Extend-specific: restore source video path
  if (p.video_path) document.getElementById('video_path').value = p.video_path;
  if (p.label) document.getElementById('preset_label').value = p.label;
  updateCustomizeSummary();
  updateDerived();
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
  const modeLabel = ({
    t2v: 'Text → Video',
    i2v: 'Image → Video',
    i2v_clean_audio: 'Image → Video (clean audio)',
    keyframe: 'FFLF (first + last frame)',
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
  genRows.push(`<dt>Quality</dt><dd>${escapeHtml((p.quality || 'standard').replace(/^./, c => c.toUpperCase()))}</dd>`);
  if (p.accel && p.accel !== 'off') {
    genRows.push(`<dt>Speed</dt><dd>${escapeHtml(p.accel.replace(/^./, c => c.toUpperCase()))}</dd>`);
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
  const noMusic = document.getElementById('noMusic');
  if (noMusic && noMusic.checked) {
    const original = fd.get('prompt') || '';
    const constraint = ' Audio: voice and ambient sounds only, no music, no soundtrack, no score, no melody.';
    if (!original.toLowerCase().includes('no music')) {
      fd.set('prompt', original.trim() + constraint);
    }
  }
  await api('/queue/add','POST',fd);
  poll();
});

// ====== Inline models card (top of form) ======
// Displays what the current install needs RIGHT WHERE the user is about
// to act. Beats burying the download CTA in a header modal. State picked
// from /status: base missing → red blocker, current-mode-needs-Q8 →
// amber prompt, downloading → animated progress, all-good → hidden.
function updateModelsCard(s) {
  const card     = document.getElementById('modelsInline');
  const icon     = document.getElementById('modelsInlineIcon');
  const title    = document.getElementById('modelsInlineTitle');
  const sub      = document.getElementById('modelsInlineSub');
  const progress = document.getElementById('modelsInlineProgress');
  const fill     = document.getElementById('modelsInlineFill');
  const last     = document.getElementById('modelsInlineLast');
  const actions  = document.getElementById('modelsInlineActions');
  if (!card) return;

  const baseOk = !!s.base_available;
  const q8Ok   = !!s.q8_available;
  const dl     = s.download && s.download.active ? s.download : null;
  const tier   = s.tier || {};
  const dismissed = !!(s.settings && s.settings.models_card_dismissed);

  // Reset state classes — we set the right one below.
  card.classList.remove('state-missing', 'state-warn', 'state-downloading', 'dismissible');
  progress.style.display = 'none';

  // ----- Active download takes precedence over everything ------------------
  if (dl) {
    card.style.display = '';
    card.classList.add('state-downloading');
    icon.textContent = '↓';
    const labelByKey = { q4: 'Q4 base model', gemma: 'Gemma text encoder', q8: 'Q8 high-quality model' };
    title.textContent = `Downloading ${labelByKey[dl.key] || dl.repo_id}`;
    const elapsed = Math.max(0, Math.round((Date.now()/1000) - (dl.started_ts || 0)));
    sub.textContent = `${elapsed}s elapsed · resumable if interrupted`;
    progress.style.display = '';
    // Try to extract a percent from the last hf line (tqdm format).
    const m = (dl.last_line || '').match(/\b(\d{1,3})%/);
    fill.style.width = m ? `${Math.min(100, parseInt(m[1]))}%` : '15%';
    last.textContent = dl.last_line || 'starting…';
    actions.innerHTML = `<button class="danger" onclick="cancelDownload()">Cancel</button>`;
    return;
  }

  // ----- Base missing — hard block, the panel can't render anything --------
  if (!baseOk) {
    card.style.display = '';
    card.classList.add('state-missing');
    icon.textContent = '⚠';
    title.textContent = 'Base models needed before you can render';
    const missing = (s.base_missing || []).length;
    sub.innerHTML = `Q4 (~20 GB) and Gemma (~6 GB) are required. Click below — downloads resume if interrupted.${
      missing ? ` <span style="color:var(--muted)">(${missing} files left)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q4')">Download Q4 (20 GB)</button>`
      : `<button disabled title="hf binary not found — reinstall via Pinokio">hf missing</button>`;
    return;
  }

  // ----- User picked a mode that needs Q8, but Q8 isn't there --------------
  // FFLF + Extend + High quality all need Q8. Surface the CTA *only* when
  // the user is about to do one of those — no point nagging a T2V user
  // about Q8 if they'll never use it.
  // Dismissible: a user who deliberately doesn't want Q8 (storage budget,
  // they only do T2V Quick/Standard) can × this away and we'll respect it
  // until either model state changes or they re-summon the modal.
  // Y1.036 — Extend joins FFLF and High in needing Q8. The Extend pipeline
  // loads `transformer-dev.safetensors` for CFG-guided denoise; Q4 doesn't
  // ship it after the Y1.024 download trim, so surface the same CTA here.
  const needsQ8 = (currentMode === 'keyframe')
                || (currentMode === 'extend')
                || (document.getElementById('quality').value === 'high');
  if (needsQ8 && !q8Ok && tier.allows_q8 !== false) {
    if (dismissed) { card.style.display = 'none'; return; }
    card.style.display = '';
    card.classList.add('state-warn', 'dismissible');
    icon.textContent = '⬇';
    const reason = currentMode === 'keyframe' ? 'FFLF needs the Q8 model'
                : currentMode === 'extend'    ? 'Extend needs the Q8 model'
                                              : 'High quality needs the Q8 model';
    title.textContent = reason;
    const missing = (s.q8_missing || []).length;
    sub.innerHTML = `Q8 (~37 GB) is a separate one-time download. Resumable.${
      missing && missing < 8 ? ` <span style="color:var(--muted)">(${missing} files left — partial install detected)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q8')">Download Q8 (37 GB)</button>`
      : `<button disabled>hf missing</button>`;
    return;
  }

  // ----- All good — hide the card completely -------------------------------
  // Per user feedback: the "Models ready · 3/3" status was visual noise once
  // everything was downloaded. Hide the card on full readiness; the header
  // models pill still gives a way to reopen the modal if the user wants to
  // manage repos. If state regresses (a file gets deleted, partial download
  // appears), one of the branches above re-shows it automatically.
  const allReady = baseOk && q8Ok;
  if (allReady) {
    card.style.display = 'none';
    actions.innerHTML = '';
    return;
  }
  // ----- Partial-OK quiet state ---------------------------------------------
  // Base OK but Q8 missing on a tier that supports it AND the user hasn't
  // picked a Q8-needing mode — gentle nudge in neutral colours, dismissible.
  if (dismissed) { card.style.display = 'none'; return; }
  card.style.display = '';
  card.classList.add('dismissible');
  icon.textContent = '✓';
  const ready = s.repos_ready ?? 0;
  const total = s.repos_total ?? 0;
  title.textContent = `Models ready · ${ready}/${total}`;
  const partialNote = (q8Ok && baseOk) ? '' : ` · ${total - ready} optional missing`;
  sub.innerHTML =
    `All installed weights detected${partialNote}. ` +
    `<a style="color:var(--accent-bright,#7e98ff); cursor:pointer; text-decoration:underline" onclick="openModelsModal()">Manage models →</a>`;
  actions.innerHTML = '';
}

// Persist the "user dismissed the models card" flag. POSTs to /settings
// and re-runs updateModelsCard with the latest status so the card hides
// immediately (not after the next /status poll cycle, ~5s away).
async function dismissModelsCard() {
  try {
    const fd = new URLSearchParams();
    fd.set('models_card_dismissed', 'true');
    await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: fd,
    });
  } catch (e) { /* best effort — UI still hides locally on next poll */ }
  // Optimistically hide right now without waiting for the poll round-trip.
  const card = document.getElementById('modelsInline');
  if (card) card.style.display = 'none';
  // Patch LAST_STATUS so subsequent updateModelsCard calls before the next
  // /status fetch agree with the on-disk state.
  if (LAST_STATUS && LAST_STATUS.settings) {
    LAST_STATUS.settings.models_card_dismissed = true;
  }
}

// ====== Tier gating ======
// Disables the FFLF / Extend mode pills and the High quality pill when
// the detected hardware tier doesn't support them. Visual state +
// tooltip + intercepted clicks. Run from the poll handler so an env
// override flips state on restart.
function applyTierGates(tier) {
  // Mode pills
  document.querySelectorAll('#modeGroup .pill-btn').forEach(b => {
    const m = b.dataset.mode;
    const allowed = (m === 'keyframe') ? tier.allows_keyframe
                  : (m === 'extend')   ? tier.allows_extend
                  : true;
    b.classList.toggle('disabled', !allowed);
    if (!allowed) {
      const need = m === 'keyframe'
        ? 'first/last-frame interpolation needs more memory than this Mac has — try Image → Video instead'
        : 'extending an existing clip needs more memory than this Mac has — try Image → Video instead';
      b.title = `Off on the ${tier.label} tier · ${need}`;
    } else {
      b.title = '';
    }
  });
  // Quality: High requires Q8. We already disable based on q8_available
  // for the no-download case; this layer enforces the RAM tier on top.
  // Both layers can disable — we OR them together via a class.
  const highBtn = document.getElementById('qualityHigh');
  if (highBtn) {
    if (!tier.allows_q8) {
      highBtn.classList.add('disabled');
      highBtn.title = `Off on the ${tier.label} tier · the high-quality model needs more memory than this Mac has`;
    } else {
      // Don't unconditionally clear .disabled — the Q8-not-installed code
      // path also sets it. Only clear if the tier is the only reason.
      // The poll() code that checks q8_available re-applies that state
      // every cycle, so this branch is safe to unset.
      highBtn.title = '';
    }
  }
}
// Intercept clicks on disabled mode pills so users get a helpful message
// instead of a broken-feeling no-op.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('#modeGroup .pill-btn.disabled');
  if (btn) {
    e.stopPropagation();
    e.preventDefault();
    alert(btn.title || 'This mode is not supported on this hardware tier.');
  }
}, true);

// ====== Tier modal ======
function openTierModal() {
  const modal = document.getElementById('tierModal');
  modal.style.display = 'flex';
  // Defensive: show "loading" state immediately so the modal never appears
  // with the body completely blank (which is what happens if the panel
  // process is dead and fetch fails — looked like a "buggy bug" to a user
  // who was just kicked off a stale browser view).
  document.getElementById('tierModalTitle').textContent = 'Hardware tier';
  document.getElementById('tierModalBlurb').innerHTML = '<em>Loading…</em>';
  document.getElementById('tierCapsList').innerHTML = '';
  // Tier doesn't change at runtime — RAM is fixed at boot — so a single
  // fetch on open is plenty. No need for live polling here.
  fetch('/status').then(r => r.json()).then(s => {
    const t = s.tier || {};
    const tt = t.times || {};
    // Helper: a row is "available" if it's allowed; "max" is the friendly
    // size limit (or "Any size" / "—" when there is no limit / disabled).
    const sizeLine = (on, maxDim, fallback) => {
      if (!on) return fallback || 'Not available on this Mac';
      if (!maxDim) return 'Any size';
      return `Up to ${maxDim} pixels on the longer side`;
    };
    document.getElementById('tierModalTitle').textContent = `What this Mac can do · ${t.label || 'unknown'}`;
    document.getElementById('tierModalBlurb').innerHTML = `
      <div style="margin-bottom: 6px"><strong>${escapeHtml(t.label || '')}</strong> · ${escapeHtml(t.ram_label || '')} of memory</div>
      <div>${escapeHtml(t.blurb || '')}</div>`;
    // One row per mode/option, with three pieces of info each:
    //   - is it available? (✓ / ✗)
    //   - what's the size limit? (plain English)
    //   - how long does a typical 5-second render take? (rough estimate)
    const items = [
      {
        title: 'Text → video',
        desc: 'Type a prompt, get a clip. The default mode.',
        on: true,
        size: sizeLine(true, t.t2v_max_dim),
        time: tt.t2v_standard,
      },
      {
        title: 'Image → video',
        desc: 'Drop in a still, get it animated. Same speed as text → video.',
        on: true,
        size: sizeLine(true, t.i2v_max_dim),
        time: tt.i2v_standard,
      },
      {
        title: 'Quick (640×480)',
        desc: 'Smaller preview to scout prompts and seeds before a full-size render.',
        on: true,
        size: 'Always smaller than Standard',
        time: tt.t2v_draft,
      },
      {
        title: 'High quality',
        desc: 'Bigger model, two-stage denoising, sharper faces. Needs the optional Q8 download.',
        on: !!t.allows_q8,
        size: sizeLine(!!t.allows_q8, 0, 'Needs more memory than this Mac has'),
        time: tt.high,
      },
      {
        title: 'First / last frame (FFLF)',
        desc: 'Pick a start image and an end image, the model fills the motion between.',
        on: !!t.allows_keyframe,
        size: sizeLine(!!t.allows_keyframe, t.keyframe_max_dim, 'Needs more memory than this Mac has'),
        time: tt.keyframe,
      },
      {
        title: 'Extend an existing clip',
        desc: 'Pick a video you already rendered, the model adds more time onto either end.',
        on: !!t.allows_extend,
        size: sizeLine(!!t.allows_extend, t.extend_max_dim, 'Needs more memory than this Mac has'),
        time: tt.extend,
      },
    ];
    document.getElementById('tierCapsList').innerHTML = items.map(it => `
      <li class="${it.on ? 'ready' : 'missing'}">
        <span class="icon">${it.on ? '✓' : '✗'}</span>
        <div class="meta">
          <span class="ttl">${escapeHtml(it.title)}</span>
          <span class="sub">${escapeHtml(it.desc)}</span>
          <span class="sub" style="margin-top:2px">
            <span style="color:var(--fg,#d8e0ee)">${escapeHtml(it.size)}</span>${
              it.time ? ` · <span style="color:var(--accent-bright,#7e98ff)">~ ${escapeHtml(it.time)} for a 5-second clip</span>` : ''
            }
          </span>
        </div>
        <span></span>
      </li>`).join('');
  }).catch(err => {
    // Panel might be dead, status endpoint unreachable, or response not JSON.
    // Replace the loading state with a visible error so the modal doesn't
    // look "broken" with empty content.
    document.getElementById('tierModalBlurb').innerHTML =
      '<div style="color: var(--danger, #f85149)">Could not load tier info — the panel server may have stopped responding. Check the Pinokio terminal and restart the panel if needed.</div>';
    document.getElementById('tierCapsList').innerHTML = '';
    console.error('tier modal fetch failed:', err);
  });
}
function closeTierModal() { document.getElementById('tierModal').style.display = 'none'; }

// ====== Settings modal ======
// Single-shot fetch on open (settings change rarely). The modal hydrates
// preset cards from the /settings response so the labels and blurbs
// match the server-side OUTPUT_PRESETS table — no preset content
// duplicated in JS.
let _settingsCache = null;

async function openSettingsModal() {
  const modal = document.getElementById('settingsModal');
  modal.style.display = 'flex';
  document.getElementById('settingsStatus').textContent = '';
  document.getElementById('settingsStatus').className = 'settings-status';
  try {
    const r = await fetch('/settings');
    _settingsCache = await r.json();
  } catch (e) {
    document.getElementById('settingsStatus').textContent = 'Could not load settings.';
    document.getElementById('settingsStatus').className = 'settings-status err';
    return;
  }
  const cur = _settingsCache.settings;
  const presets = _settingsCache.presets;
  // Render preset cards (Standard, Video production, Web, Custom).
  // Display order matches the typical user journey: most users want
  // Standard, video pros pick Video production, web preview folks pick Web.
  const order = ['standard', 'archival', 'web'];
  const grid = document.getElementById('settingsPresets');
  grid.innerHTML = '';
  for (const key of order) {
    const p = presets[key];
    const active = cur.output_preset === key ? 'active' : '';
    const card = document.createElement('label');
    card.className = `preset-card ${active}`;
    card.dataset.preset = key;
    card.innerHTML = `
      <input type="radio" name="settingsPreset" value="${key}" ${cur.output_preset === key ? 'checked' : ''}>
      <div class="preset-text">
        <div class="preset-label">${escapeHtml(p.label)}</div>
        <div class="preset-blurb">${escapeHtml(p.blurb)}</div>
        <div class="preset-spec">pix_fmt=${p.pix_fmt} · crf=${p.crf}</div>
      </div>`;
    card.addEventListener('click', () => selectPreset(key));
    grid.appendChild(card);
  }
  // Custom row.
  const customActive = cur.output_preset === 'custom' ? 'active' : '';
  const custom = document.createElement('label');
  custom.className = `preset-card ${customActive}`;
  custom.dataset.preset = 'custom';
  custom.innerHTML = `
    <input type="radio" name="settingsPreset" value="custom" ${cur.output_preset === 'custom' ? 'checked' : ''}>
    <div class="preset-text">
      <div class="preset-label">Custom</div>
      <div class="preset-blurb">Set pix_fmt and CRF manually. For unusual workflows: 10-bit HDR, format-specific delivery, or non-standard CRF for video production work.</div>
      <div class="preset-spec">pix_fmt=${cur.output_pix_fmt} · crf=${cur.output_crf}</div>
    </div>`;
  custom.addEventListener('click', () => selectPreset('custom'));
  grid.appendChild(custom);
  // Pre-fill custom inputs with current values
  document.getElementById('settingsPixFmt').value = cur.output_pix_fmt;
  document.getElementById('settingsCrfRange').value = cur.output_crf;
  document.getElementById('settingsCrfNum').value = cur.output_crf;
  document.getElementById('settingsCustomSection').style.display =
    cur.output_preset === 'custom' ? 'block' : 'none';

  // Token rows. We never receive the actual key from the server (the
  // /settings GET returns has_X booleans only), so we display either
  // "set" with an empty placeholder input, or "—" with the placeholder.
  // Inputs start empty on every modal open; user pastes when they want
  // to change.
  setTokenStatus('civitaiKey', cur.has_civitai_key);
  setTokenStatus('hfToken', cur.has_hf_token);
  // Placeholders reflect the saved state so an empty input doesn't read
  // as "no token here" when there is one. The asterisks make it clear
  // something's persisted; the hint reminds users they paste to replace.
  const civInput = document.getElementById('civitaiKeyInput');
  const hfInput = document.getElementById('hfTokenInput');
  civInput.value = '';
  hfInput.value = '';
  civInput.placeholder = cur.has_civitai_key
    ? '•••••••••• saved — paste new to replace'
    : '32-char API key';
  hfInput.placeholder = cur.has_hf_token
    ? '•••••••••• saved — paste new to replace'
    : 'hf_…';
  document.getElementById('civitaiKeyClear').style.display = cur.has_civitai_key ? '' : 'none';
  document.getElementById('hfTokenClear').style.display = cur.has_hf_token ? '' : 'none';

  // Spicy mode — render current state. _spicyArmed is the mid-confirm
  // state (clicked once, waiting for the second click). It lives only
  // on the JS side; only ON/OFF gets persisted.
  _spicyArmed = false;
  renderSpicyState(!!cur.spicy_mode);
}

let _spicyArmed = false;

function renderSpicyState(isOn) {
  const badge = document.getElementById('spicyStateBadge');
  const btn = document.getElementById('spicyToggleBtn');
  const hint = document.getElementById('spicyHint');
  if (!badge || !btn) return;
  badge.classList.remove('on', 'armed');
  if (_spicyArmed) {
    badge.textContent = 'ARMED';
    badge.classList.add('armed');
    btn.textContent = 'Click again to confirm';
    btn.classList.remove('ghost-btn');
    btn.classList.add('primary-btn');
    hint.style.display = '';
    hint.textContent = 'Confirms turning Spicy mode ON. NSFW LoRAs will be available in the CivitAI browser. Cancel by closing the modal.';
  } else if (isOn) {
    badge.textContent = 'ON';
    badge.classList.add('on');
    btn.textContent = 'Disable';
    btn.classList.remove('primary-btn');
    btn.classList.add('ghost-btn');
    hint.style.display = '';
    hint.textContent = 'Spicy mode is ON. NSFW LoRAs are visible in the CivitAI browser when you tick "Show NSFW".';
  } else {
    badge.textContent = 'OFF';
    btn.textContent = 'Enable Spicy mode';
    btn.classList.remove('primary-btn');
    btn.classList.add('ghost-btn');
    hint.style.display = 'none';
    hint.textContent = '';
  }
}

async function toggleSpicyMode() {
  // Two-click confirm to turn ON, single-click to turn OFF.
  // Easy to disable, deliberate to enable — matches the user spec
  // ("don't want people to turn it on by mistake, or kids").
  const cur = (_settingsCache && _settingsCache.settings) || {};
  const isOn = !!cur.spicy_mode;
  if (isOn) {
    // Single-click off, no confirm.
    await _persistSpicyMode(false);
    return;
  }
  if (!_spicyArmed) {
    _spicyArmed = true;
    renderSpicyState(false);
    // Auto-disarm after 6 s if the user doesn't confirm — prevents
    // the "click again" state lingering across an unrelated tab return.
    setTimeout(() => {
      if (_spicyArmed) {
        _spicyArmed = false;
        renderSpicyState(!!(_settingsCache?.settings?.spicy_mode));
      }
    }, 6000);
    return;
  }
  // Second click — actually persist.
  _spicyArmed = false;
  await _persistSpicyMode(true);
}

async function _persistSpicyMode(target) {
  const status = document.getElementById('settingsStatus');
  try {
    const fd = new URLSearchParams();
    fd.set('spicy_mode', target ? 'true' : 'false');
    const r = await fetch('/settings', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    if (_settingsCache && _settingsCache.settings) {
      _settingsCache.settings.spicy_mode = !!target;
    }
    renderSpicyState(!!target);
    if (status) {
      status.textContent = target ? 'Spicy mode ON · NSFW LoRAs unlocked' : 'Spicy mode OFF · NSFW LoRAs hidden';
      status.className = 'settings-status ok';
    }
    // Refresh the CivitAI panel so the "Show NSFW" toggle appears /
    // disappears immediately without a full page reload.
    if (typeof refreshCivitaiAccessUI === 'function') refreshCivitaiAccessUI();
  } catch (e) {
    if (status) {
      status.textContent = 'Could not change Spicy mode: ' + (e.message || e);
      status.className = 'settings-status err';
    }
  }
}

function setTokenStatus(prefix, isSet, dirty) {
  const el = document.getElementById(prefix + 'Status');
  if (!el) return;
  el.classList.remove('set', 'dirty');
  if (dirty) {
    el.textContent = '✎ unsaved';
    el.classList.add('dirty');
  } else if (isSet) {
    el.textContent = '✓ saved';
    el.classList.add('set');
  } else {
    el.textContent = 'not set';
  }
}

function onTokenInput(which) {
  const prefix = which === 'civitai' ? 'civitaiKey' : 'hfToken';
  const inp = document.getElementById(prefix + 'Input');
  setTokenStatus(prefix, false, !!inp.value);
}

function toggleTokenVisibility(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (!inp) return;
  if (inp.type === 'password') {
    inp.type = 'text';
    btn.textContent = 'hide';
  } else {
    inp.type = 'password';
    btn.textContent = 'show';
  }
}

// Save-then-test in one click. The /civitai/test and /hf/test endpoints
// use the saved key, not the current input field. Pre-Y1.023 the user
// had to: paste → click Apply → click Test. That left a footgun: users
// pasted, clicked Test, saw it fail (because nothing was saved yet),
// closed the modal thinking the panel was broken. The token never
// landed in panel_settings.json, gated downloads kept failing.
//
// Now Test does Save first when the input has a value, so a single
// click works. If the save fails (validator rejects malformed token),
// we surface the error inline next to the field instead of just at
// the bottom of the modal.
async function testToken(which) {
  const path = which === 'civitai' ? '/civitai/test' : '/hf/test';
  const resultId = which === 'civitai' ? 'civitaiTestResult' : 'hfTestResult';
  const inputId = which === 'civitai' ? 'civitaiKeyInput' : 'hfTokenInput';
  const fieldName = which === 'civitai' ? 'civitai_api_key' : 'hf_token';
  const statusPrefix = which === 'civitai' ? 'civitaiKey' : 'hfToken';
  const clearBtnId = which === 'civitai' ? 'civitaiKeyClear' : 'hfTokenClear';
  const result = document.getElementById(resultId);
  if (!result) return;
  result.textContent = 'Testing…';
  result.style.color = 'var(--muted)';

  // If the input has content, save it first. Empty input means "test
  // the already-saved token" — the legitimate use after the panel is
  // configured.
  const inputEl = document.getElementById(inputId);
  const inputValue = inputEl ? inputEl.value.trim() : '';
  if (inputValue) {
    try {
      const fd = new URLSearchParams();
      fd.set(fieldName, inputValue);
      const saveResp = await fetch('/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: fd,
      });
      const saveData = await saveResp.json();
      if (!saveResp.ok || saveData.error) {
        result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> ${escapeHtml(saveData.error || `HTTP ${saveResp.status}`)}`;
        return;
      }
      // Save succeeded — reflect the persisted state in the UI.
      inputEl.value = '';
      _settingsCache = { ...(_settingsCache || {}), settings: saveData.settings };
      setTokenStatus(statusPrefix, true);
      const clearBtn = document.getElementById(clearBtnId);
      if (clearBtn) clearBtn.style.display = '';
    } catch (e) {
      result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> Save failed: ${escapeHtml(e.message || String(e))}`;
      return;
    }
  }

  // Now hit the test endpoint, which reads the freshly-saved token.
  try {
    const r = await fetch(path);
    const data = await r.json();
    if (data.ok) {
      result.innerHTML = `<strong style="color: var(--success, #3fb950)">✓</strong> ${escapeHtml(data.message)}`;
    } else {
      result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> ${escapeHtml(data.error)}`;
    }
  } catch (e) {
    result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> Network error: ${escapeHtml(e.message || String(e))}`;
  }
}

async function clearToken(which) {
  const fd = new FormData();
  if (which === 'civitai') fd.set('civitai_api_key', '');
  if (which === 'hf')      fd.set('hf_token', '');
  try {
    // urlencoded body — see applySettings for why.
    const r = await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      alert('Could not clear: ' + (data.error || `HTTP ${r.status}`));
      return;
    }
    // Refresh the modal so the status flips back to "not set".
    openSettingsModal();
  } catch (e) {
    alert('Network error: ' + (e.message || e));
  }
}

function closeSettingsModal() {
  document.getElementById('settingsModal').style.display = 'none';
}

function selectPreset(key) {
  document.querySelectorAll('#settingsPresets .preset-card').forEach(c => {
    c.classList.toggle('active', c.dataset.preset === key);
    const r = c.querySelector('input[type="radio"]');
    if (r) r.checked = (c.dataset.preset === key);
  });
  document.getElementById('settingsCustomSection').style.display =
    key === 'custom' ? 'block' : 'none';
  // Clear status so it doesn't claim "saved" after a fresh selection.
  document.getElementById('settingsStatus').textContent = '';
  document.getElementById('settingsStatus').className = 'settings-status';
}

async function applySettings() {
  const status = document.getElementById('settingsStatus');
  const btn = document.getElementById('settingsApplyBtn');
  status.textContent = 'Saving…';
  status.className = 'settings-status';
  btn.disabled = true;
  // Read which preset is selected. Custom path also sends pix_fmt + crf.
  const checked = document.querySelector('#settingsPresets input[type="radio"]:checked');
  const preset = checked ? checked.value : 'standard';
  const fd = new FormData();
  fd.set('output_preset', preset);
  if (preset === 'custom') {
    fd.set('output_pix_fmt', document.getElementById('settingsPixFmt').value);
    fd.set('output_crf', document.getElementById('settingsCrfNum').value);
  }
  // Tokens — only send a key when the input has a value. Empty input
  // means "leave as-is" (clearing is explicit via the Clear button).
  // This protects against accidentally wiping a saved key by clicking
  // Apply on an unchanged form.
  const civInput = document.getElementById('civitaiKeyInput').value.trim();
  if (civInput) fd.set('civitai_api_key', civInput);
  const hfInput = document.getElementById('hfTokenInput').value.trim();
  if (hfInput)  fd.set('hf_token', hfInput);
  try {
    // Convert FormData → URLSearchParams so the body is sent as
    // x-www-form-urlencoded — the panel's parse_qs only understands
    // that wire format, NOT the multipart/form-data fetch sends by
    // default with FormData. This bug silently turned every settings
    // save into a no-op (server saw empty payload) until caught.
    const r = await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      status.textContent = data.error || `HTTP ${r.status}`;
      status.className = 'settings-status err';
      btn.disabled = false;
      return;
    }
    status.textContent = data.helper_restarted
      ? 'Saved. Helper restarted — takes effect on the next render.'
      : 'Saved.';
    status.className = 'settings-status ok';
    btn.disabled = false;
    // Refresh cache so a re-open shows the new values without a stale flash.
    _settingsCache = { ...(_settingsCache || {}), settings: data.settings };
  } catch (e) {
    status.textContent = 'Network error: ' + (e.message || e);
    status.className = 'settings-status err';
    btn.disabled = false;
  }
}

// ====== HDR toggle pill (header pill behavior, same as No-music) ======
(function () {
  const pill = document.getElementById('hdrPill');
  const cb = document.getElementById('hdr');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', () => setTimeout(sync, 0));
  sync();
})();

// ====== CivitAI NSFW toggle pill (mirrors HDR toggle UX) ======
(function () {
  const pill = document.getElementById('civitaiNsfwPill');
  const cb = document.getElementById('civitaiNsfw');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', () => setTimeout(sync, 0));
  sync();
})();

// ====== LoRA picker ======
//
// State model: an in-memory list of LoRA entries the user has added.
// Adding can come from "Use" on a CivitAI install or from clicking a
// row in the local list. Each entry:
//   { path, name, strength, trigger_words, civitai_url }
// On every change we mirror the list into the hidden #lorasJson field
// so make_job's parse_loras_from_form picks them up at submit time.

let _activeLoras = [];   // [{path, name, strength, trigger_words, ...}]
let _knownUserLoras = []; // last list_user_loras() snapshot, for the picker

function _serializeLoras() {
  // What the helper actually needs is path + strength. Keep the rest in
  // the in-memory list for UI rendering, drop it on the wire. Summary
  // count is updated by renderLorasList() which has fuller state — we
  // don't touch it here to avoid two functions stomping each other.
  const slim = _activeLoras.map(l => ({ path: l.path, strength: l.strength }));
  document.getElementById('lorasJson').value = JSON.stringify(slim);
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

async function refreshLoras() {
  let data;
  try {
    data = await (await fetch('/loras')).json();
  } catch (e) {
    return;
  }
  _knownUserLoras = data.user || [];
  // Update displayed loras dir
  if (data.loras_dir) {
    const dirEl = document.getElementById('lorasDir');
    if (dirEl) dirEl.textContent = data.loras_dir;
  }
  // If a row was previously active but the file is gone (deleted on
  // disk), drop it from the active set so we don't submit a stale path.
  const knownPaths = new Set(_knownUserLoras.map(l => l.path));
  _activeLoras = _activeLoras.filter(l =>
    knownPaths.has(l.path) || l.path.includes('/'));   // keep HF ids (no dir slash)
  renderLorasList();
  _serializeLoras();
}

function renderLorasList() {
  const wrap = document.getElementById('lorasList');
  const empty = document.getElementById('lorasEmpty');
  const filterRow = document.getElementById('lorasFilterRow');
  const filterInput = document.getElementById('lorasFilter');
  if (!wrap) return;
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
      active: !!active,
      strength: active ? active.strength : (ul.recommended_strength || 1.0),
      kind: 'user',
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
    return;
  }
  if (empty) empty.style.display = 'none';
  // Surface the filter input only when 5+ LoRAs are installed; below that
  // the box is just visual noise.
  if (filterRow) filterRow.style.display = (allRows.length >= 5) ? '' : 'none';

  // Apply filter (case-insensitive substring on name + trigger words).
  let rows = allRows;
  const q = (filterInput && filterInput.value || '').trim().toLowerCase();
  if (q) {
    rows = allRows.filter(r => {
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

  // Update header summary.
  const summary = document.getElementById('lorasSummaryCount');
  if (summary) {
    const total = allRows.length;
    const active = allRows.filter(r => r.active).length;
    summary.textContent = `${total} installed · ${active} active${q ? ` · ${rows.length} match` : ''}`;
  }

  if (rows.length === 0) {
    wrap.innerHTML = `<div class="hint" style="padding:8px 0;">No LoRAs match "${escapeHtml(q)}".</div>`;
    return;
  }
  wrap.innerHTML = rows.map(r => loraRowHtml(r)).join('');
}

// Build a single compact LoRA row. Inactive rows are ~36px tall (just
// name + meta + corner actions). Active rows expand inline with the
// strength slider and trigger chips. Click anywhere on the main row to
// toggle activation.
function loraRowHtml(r) {
  const pathHtml = escapeHtml(r.path);
  const pathAttr = JSON.stringify(r.path).replace(/"/g, '&quot;');
  const nameHtml = escapeHtml(r.name);
  const nameAttr = JSON.stringify(r.name).replace(/"/g, '&quot;');
  // Trigger summary line under the name (when not expanded). Truncated.
  const trigs = r.trigger_words || [];
  const trigSummary = trigs.length
    ? trigs.slice(0, 4).join(' · ') + (trigs.length > 4 ? ` +${trigs.length - 4}` : '')
    : 'no trigger word';
  // Corner actions — link to civitai page + delete (or remove for HF/remote).
  const corner = [];
  if (r.civitai_url) {
    corner.push(`<a class="lora-icon-btn" href="${escapeHtml(r.civitai_url)}" target="_blank" rel="noopener" title="Open on CivitAI" onclick="event.stopPropagation()">↗</a>`);
  }
  if (r.kind === 'user') {
    corner.push(`<button class="lora-icon-btn danger" type="button" title="Delete from disk"
                         onclick="event.stopPropagation(); deleteLora(${pathAttr}, ${nameAttr})">×</button>`);
  } else {
    corner.push(`<button class="lora-icon-btn" type="button" title="Remove from active set"
                         onclick="event.stopPropagation(); removeLoraFromActive(${pathAttr})">×</button>`);
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
            ${nameHtml}${r.kind === 'remote' ? '<span class="badge">HF</span>' : ''}
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
      </div>
    </div>`;
}

function toggleLora(path, on, recommended, name) {
  if (on) {
    addLoraToActive({ path, strength: recommended, name });
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

// ====== CivitAI modal ======

let _civitaiCursor = '';
let _civitaiSearching = false;

function openCivitaiModal() {
  document.getElementById('civitaiModal').style.display = 'flex';
  // Pull /loras to populate the dir text and the auth-banner state.
  fetch('/loras').then(r => r.json()).then(d => {
    const dirEl = document.getElementById('civitaiTargetDir');
    if (dirEl && d.loras_dir) dirEl.textContent = d.loras_dir;
    renderCivitaiAuthBanner(!!d.civitai_auth);
  }).catch(() => { renderCivitaiAuthBanner(false); });
  document.getElementById('civitaiQuery').value = '';
  _civitaiCursor = '';
  // Pull current Spicy mode state so the "Show NSFW" toggle hides when off.
  refreshCivitaiAccessUI();
  civitaiSearch();
}

// Hide / show the "Show NSFW" toggle in the CivitAI browser based on the
// Spicy mode setting. Called on modal open and after toggleSpicyMode flips
// the value, so the UI tracks the gate without a page reload.
async function refreshCivitaiAccessUI() {
  let spicy = false;
  try {
    const r = await fetch('/settings');
    const j = await r.json();
    spicy = !!(j && j.settings && j.settings.spicy_mode);
  } catch (_) { /* default off */ }
  const pill = document.getElementById('civitaiNsfwPill');
  const cb = document.getElementById('civitaiNsfw');
  if (pill) pill.style.display = spicy ? '' : 'none';
  if (!spicy && cb) cb.checked = false;  // force off when spicy mode is off
}

// Render the inline API-key banner at the top of the CivitAI browser.
// Three states: set (✓ small green), missing (amber, prompts for key),
// editing (input visible while user is changing/setting the key). The
// banner is the primary surface for the key now — Settings still has
// the field but most users won't need to dig there.
function renderCivitaiAuthBanner(haveKey, mode) {
  const box = document.getElementById('civitaiAuthBanner');
  if (!box) return;
  // Three visual modes: 'view' (default), 'edit' (showing input), 'err' (last save failed).
  const m = mode || (haveKey ? 'view' : 'edit');
  box.style.display = '';
  box.classList.remove('missing','set','err');
  if (m === 'view' && haveKey) {
    box.classList.add('set');
    box.innerHTML = `
      <span><strong style="color:var(--success,#3fb950)">✓</strong> CivitAI API key set —
      LoRA downloads will work.</span>
      <span class="grow"></span>
      <a class="changekey" onclick="renderCivitaiAuthBanner(true,'edit')">change key</a>`;
    return;
  }
  // edit / missing mode — render input + Save.
  box.classList.add(m === 'err' ? 'err' : 'missing');
  const intro = m === 'err'
    ? `<strong>That key didn't work.</strong> Double-check it from <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a> and try again.`
    : haveKey
      ? `Replace your CivitAI API key. The current one stays active until you save a new one.`
      : `<strong>CivitAI requires an API key</strong> to download LoRAs. Get one at <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a> and paste it here:`;
  box.innerHTML = `
    <div class="grow" style="flex-basis:100%; margin-bottom:6px;">${intro}</div>
    <input type="password" id="civitaiAuthInput" placeholder="paste API key — usually 32 hex chars"
           autocomplete="off" spellcheck="false">
    <button type="button" id="civitaiAuthSave" onclick="civitaiAuthSave()">Save & test</button>
    ${haveKey ? '<a class="changekey" onclick="renderCivitaiAuthBanner(true,\'view\')">cancel</a>' : ''}`;
  // Pressing Enter inside the input triggers save.
  const inp = document.getElementById('civitaiAuthInput');
  if (inp) inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); civitaiAuthSave(); }
  });
}

async function civitaiAuthSave() {
  const inp = document.getElementById('civitaiAuthInput');
  const btn = document.getElementById('civitaiAuthSave');
  if (!inp) return;
  const key = (inp.value || '').trim();
  if (!key) { inp.focus(); return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
  // Save via /settings (single source of truth for tokens). After save
  // we hit /civitai/test to verify before flipping the banner to "set" —
  // that catches the most common error (typo'd key) right at the moment
  // the user pasted it, instead of failing later on the first download.
  try {
    const fd = new URLSearchParams();
    fd.set('civitai_api_key', key);
    const r = await fetch('/settings', { method: 'POST',
      headers: {'Content-Type':'application/x-www-form-urlencoded'}, body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
    // Verify
    if (btn) btn.textContent = 'Testing…';
    const t = await fetch('/civitai/test');
    const td = await t.json();
    if (!td.ok) {
      renderCivitaiAuthBanner(false, 'err');
      return;
    }
    renderCivitaiAuthBanner(true, 'view');
    // Re-run the search so any 401-blocked thumbnails reload as authed.
    civitaiSearch();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Save & test'; }
    renderCivitaiAuthBanner(false, 'err');
  }
}

function closeCivitaiModal() {
  document.getElementById('civitaiModal').style.display = 'none';
}

async function civitaiSearch() {
  if (_civitaiSearching) return;
  _civitaiSearching = true;
  const grid = document.getElementById('civitaiGrid');
  const status = document.getElementById('civitaiStatus');
  const loadMore = document.getElementById('civitaiLoadMore');
  status.textContent = '';
  status.className = 'civitai-status-line';
  grid.innerHTML = '<div class="hint">Loading…</div>';
  loadMore.style.display = 'none';
  _civitaiCursor = '';
  try {
    const params = new URLSearchParams();
    const q = document.getElementById('civitaiQuery').value.trim();
    if (q) params.set('query', q);
    if (document.getElementById('civitaiNsfw').checked) params.set('nsfw', 'true');
    params.set('limit', '24');
    const r = await fetch('/civitai/search?' + params.toString());
    const data = await r.json();
    if (data.error) {
      grid.innerHTML = '';
      status.textContent = data.error;
      status.className = 'civitai-status-line err';
      return;
    }
    renderCivitaiGrid(data.items, /* append */ false);
    _civitaiCursor = data.next_cursor || '';
    if (data.has_more) loadMore.style.display = '';
    if ((data.items || []).length === 0) {
      grid.innerHTML = `<div class="hint">No LTX 2.3 LoRAs match "${escapeHtml(q || '')}"${document.getElementById('civitaiNsfw').checked ? '' : ' (try Show NSFW for more)'}.</div>`;
    }
  } catch (e) {
    status.textContent = 'Network error: ' + (e.message || e);
    status.className = 'civitai-status-line err';
  } finally {
    _civitaiSearching = false;
  }
}

async function civitaiLoadMore() {
  if (_civitaiSearching || !_civitaiCursor) return;
  _civitaiSearching = true;
  const loadMore = document.getElementById('civitaiLoadMore');
  loadMore.disabled = true;
  loadMore.textContent = 'Loading…';
  try {
    const params = new URLSearchParams();
    const q = document.getElementById('civitaiQuery').value.trim();
    if (q) params.set('query', q);
    if (document.getElementById('civitaiNsfw').checked) params.set('nsfw', 'true');
    params.set('limit', '24');
    params.set('cursor', _civitaiCursor);
    const r = await fetch('/civitai/search?' + params.toString());
    const data = await r.json();
    if (data.error) {
      document.getElementById('civitaiStatus').textContent = data.error;
      document.getElementById('civitaiStatus').className = 'civitai-status-line err';
      return;
    }
    renderCivitaiGrid(data.items, /* append */ true);
    _civitaiCursor = data.next_cursor || '';
    loadMore.style.display = data.has_more ? '' : 'none';
  } catch (e) {
    document.getElementById('civitaiStatus').textContent = 'Network error: ' + (e.message || e);
    document.getElementById('civitaiStatus').className = 'civitai-status-line err';
  } finally {
    _civitaiSearching = false;
    loadMore.disabled = false;
    loadMore.textContent = 'Load more';
  }
}

function renderCivitaiGrid(items, append) {
  const grid = document.getElementById('civitaiGrid');
  if (!append) grid.innerHTML = '';
  if (!items || items.length === 0) return;
  const frag = document.createDocumentFragment();
  for (const it of items) {
    const card = document.createElement('div');
    card.className = 'civitai-card';
    const sizeMb = it.size_kb ? (it.size_kb / 1024).toFixed(1) : '?';
    const dl = it.downloads ? new Intl.NumberFormat().format(it.downloads) : '?';
    const triggers = (it.trigger_words || []).slice(0, 3).join(', ');
    // LTX is a video model so most LoRAs ship animated previews. Render
    // <video> for videos (autoplay muted loop = looks like an animated
    // GIF, no user interaction needed) and <img> for stills. Both share
    // the .preview class so the card height is stable while images
    // load. CivitAI's CDN sets `Access-Control-Allow-Origin: *` so
    // cross-origin loads work without a panel-side proxy.
    let previewHtml;
    if (!it.preview_url) {
      previewHtml = `<div class="preview-empty">no preview</div>`;
    } else if (it.preview_type === 'video' || /\.mp4($|\?)/i.test(it.preview_url)) {
      previewHtml = `<video class="preview" src="${escapeHtml(it.preview_url)}"
                            autoplay muted loop playsinline preload="metadata"></video>`;
    } else {
      previewHtml = `<img class="preview" src="${escapeHtml(it.preview_url)}" alt="" loading="lazy">`;
    }
    card.innerHTML = `
      ${previewHtml}
      <div class="body">
        <div class="ttl" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</div>
        <div class="meta">
          <span>by ${escapeHtml(it.creator)}</span>
          <span>↓ ${dl}</span>
          <span>${sizeMb} MB</span>
          ${it.nsfw ? '<span class="nsfw-badge">NSFW</span>' : ''}
        </div>
        ${triggers ? `<div class="meta"><span title="trigger words">trigger: ${escapeHtml(triggers)}</span></div>` : ''}
        ${it.civitai_url
          ? `<div class="meta"><a class="civitai-source-link" href="${escapeHtml(it.civitai_url)}" target="_blank" rel="noopener" title="Open the original CivitAI page — usage notes, examples, comments">Read instructions on CivitAI ↗</a></div>`
          : ''}
      </div>
      <div class="actions">
        <button type="button" class="primary-btn" data-id="${it.id}">Install</button>
      </div>`;
    const btn = card.querySelector('button[data-id]');
    btn.addEventListener('click', () => civitaiInstall(btn, it));
    frag.appendChild(card);
  }
  grid.appendChild(frag);
}

async function civitaiInstall(btn, item) {
  btn.disabled = true;
  const origLabel = btn.textContent;
  btn.textContent = 'Downloading…';
  const fd = new FormData();
  fd.set('download_url', item.download_url);
  fd.set('meta', JSON.stringify(item));
  try {
    const r = await fetch('/civitai/download', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      const status = document.getElementById('civitaiStatus');
      status.textContent = `Download failed: ${data.error || 'HTTP ' + r.status}`;
      status.className = 'civitai-status-line err';
      btn.disabled = false;
      btn.textContent = origLabel;
      return;
    }
    btn.textContent = data.skipped ? 'Already installed ✓' : 'Installed ✓';
    const status = document.getElementById('civitaiStatus');
    status.textContent = data.skipped
      ? `Already in ${data.path} — auto-enabled below.`
      : `Saved to ${data.path}. Auto-enabled below.`;
    status.className = 'civitai-status-line ok';
    // Refresh the local picker so the new LoRA appears, then auto-enable.
    // Auto-enable applies on BOTH the fresh-download AND the
    // already-installed paths — clicking Install on a CivitAI card
    // should always result in "this LoRA is now usable in the next
    // render," regardless of whether it was already on disk. Earlier
    // build only auto-enabled fresh downloads, leaving repeat clicks
    // looking like a no-op even though the file was sitting right
    // there in the picker.
    await refreshLoras();
    addLoraToActive({
      path: data.path,
      name: data.name || item.name,
      strength: item.recommended_strength || 1.0,
      trigger_words: item.trigger_words || [],
    });
    // Open the LoRAs disclosure so the user sees the entry without
    // hunting for it after the modal closes.
    const det = document.getElementById('lorasDetails');
    if (det) det.open = true;
  } catch (e) {
    document.getElementById('civitaiStatus').textContent = 'Network error: ' + (e.message || e);
    document.getElementById('civitaiStatus').className = 'civitai-status-line err';
    btn.disabled = false;
    btn.textContent = origLabel;
  }
}

// Boot: load the local LoRA list on page load so the picker isn't empty
// when the user expands it for the first time.
document.addEventListener('DOMContentLoaded', () => { refreshLoras(); });

// ====== Models modal ======
// Opens to /models snapshot. While open, the main poll() refreshes the
// list every cycle so download progress appears live. Each row shows:
//   ✓ ready (green)             — all repo files present
//   ◐ partial (amber)           — some files there, some missing (e.g. interrupted)
//   ⊘ missing (red)             — nothing on disk
//   ↻ downloading (blue, anim)  — hf is currently fetching this repo
function openModelsModal() {
  document.getElementById('modelsModal').style.display = 'flex';
  refreshModelsModal();
}
function closeModelsModal() {
  document.getElementById('modelsModal').style.display = 'none';
}
async function refreshModelsModal({ silent = false } = {}) {
  const list = document.getElementById('modelsList');
  const hint = document.getElementById('modelsHint');
  const foot = document.getElementById('modelsFoot');
  let data;
  try { data = await api('/models'); }
  catch (e) {
    if (!silent) hint.textContent = 'Failed to load models. Panel might be restarting — try again.';
    return;
  }
  const repos = data.repos || [];
  const active = data.active_download;
  hint.innerHTML = data.hf_available
    ? `Each row shows what's on disk. Click <b>Download</b> to fetch missing files via <code>hf download</code>; progress streams to the log at the bottom of the page.`
    : `<span style="color:var(--warning,#d29922)">⚠ <code>hf</code> not found</span> — this Pinokio install doesn't have <code>huggingface_hub&gt;=1.0</code> in the venv. Run Update from Pinokio, then come back.`;
  const rows = repos.map(r => {
    let cls, icon, statusText, btnHtml;
    if (active && active.key === r.key) {
      cls = 'downloading';
      icon = '↻';
      const elapsed = Math.max(0, Math.round((Date.now()/1000) - (active.started_ts || 0)));
      const last = active.last_line ? `<div class="progress">${escapeHtml(active.last_line)}</div>` : '';
      statusText = `Downloading · ${elapsed}s${last}`;
      btnHtml = `<button class="ghost" onclick="cancelDownload()">Cancel</button>`;
    } else if (r.complete) {
      cls = 'ready'; icon = '✓';
      // `where: 'hf_cache'` means the files were resolved from
      // ~/.cache/huggingface/ rather than the canonical mlx_models/
      // dir. Common on manual / dev installs that pre-existed Pinokio
      // and pulled the model via `huggingface-cli` or first-run helper.
      const tag = r.where === 'hf_cache' ? 'HF cache' : 'local';
      statusText = `Ready · ${r.total_files} files · ~${r.size_gb || '?'} GB · ${tag}`;
      btnHtml = `<button class="ghost" disabled>Installed</button>`;
    } else if (r.present_files > 0) {
      cls = 'partial'; icon = '◐';
      const left = r.total_files - r.present_files;
      statusText = `Partial · ${r.present_files}/${r.total_files} files · ${left} missing — resume to finish`;
      btnHtml = data.hf_available
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Resume</button>`
        : `<button disabled>Resume</button>`;
    } else {
      cls = 'missing'; icon = '⊘';
      statusText = `Not installed · ~${r.size_gb || '?'} GB`;
      btnHtml = data.hf_available
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Download</button>`
        : `<button disabled>Download</button>`;
    }
    const kindBadge = r.kind === 'optional'
      ? `<span style="color:var(--muted)">optional</span>`
      : `<span style="color:var(--success,#3fb950)">required</span>`;
    return `
      <li class="${cls}">
        <span class="icon">${icon}</span>
        <div class="meta">
          <span class="ttl">${escapeHtml(r.name)} · ${kindBadge}</span>
          <span class="sub">${escapeHtml(r.repo_id)} → ${escapeHtml(r.local_dir)}</span>
          <span class="sub">${statusText}${r.blurb ? ' · ' + escapeHtml(r.blurb) : ''}</span>
        </div>
        ${btnHtml}
      </li>`;
  }).join('');
  list.innerHTML = rows || `<li class="empty-state">No model manifest found — required_files.json is missing or unreadable.</li>`;
  // Footer summarises required vs optional counts.
  const reqRepos = repos.filter(r => r.kind !== 'optional');
  const optRepos = repos.filter(r => r.kind === 'optional');
  const reqReady = reqRepos.filter(r => r.complete).length;
  const optReady = optRepos.filter(r => r.complete).length;
  foot.innerHTML = `
    <div>Required: ${reqReady}/${reqRepos.length} ready &nbsp;·&nbsp; Optional: ${optReady}/${optRepos.length} ready</div>
    <div style="margin-top:4px">Tip: downloads resume on retry — closing this dialog mid-download keeps it running in the background.</div>`;
}
async function startDownload(key) {
  let res;
  try {
    res = await api('/models/download', 'POST', `repo_key=${encodeURIComponent(key)}`);
  } catch (e) {
    alert('Download failed to start: ' + (e?.message || e));
    return;
  }
  // The api() helper coerces 409 (busy) to { error: 'busy' } — surface that
  // to the user instead of silently no-op'ing the click.
  if (res && res.error) {
    alert(`Can't start download: ${res.error}`);
  }
  refreshModelsModal();
  poll();
}
async function cancelDownload() {
  if (!confirm('Cancel the active download? Partial files stay on disk; clicking Download/Resume later picks up where you left off.')) return;
  try { await api('/models/cancel', 'POST'); } catch (e) {}
  refreshModelsModal();
}

// ====== Version pill (the "magic button") ======
//
// One always-visible pill in the header that changes content + colour
// based on /version state. Clicking it does the right thing for the
// current state — no modal, no nested click flows.
//
// Backend: a daemon thread polls GitHub every 30 minutes (commits API
// for the SHA + raw VERSION file for the human-friendly Y1.NNN label)
// and exposes the result at /version. The JS polls /version every 5
// minutes (cheap; pre-computed dict read). When the user clicks while
// behind, /version/pull does the actual `git pull` server-side; the
// user still has to Stop+Start phosphene in Pinokio to apply.
//
// Rationale: users keep telling us "I clicked Update but I don't see
// the new features" — by the time the feedback reaches us we've usually
// pushed three more commits. The pill turns the loop from
// "hope-they-noticed" into a literal one-click action.

let _versionState = null;
let _versionRestartPending = false;   // set after a successful /version/pull;
                                      // pill turns into a "restart" reminder.

async function refreshVersionPill() {
  try {
    const r = await fetch('/version');
    _versionState = await r.json();
  } catch (e) {
    return;             // network blip; don't blow away last good state
  }
  renderVersionPill();
}

function _versionDisplayLabel(s) {
  // Prefer the human Y1.NNN VERSION file label. Fall back to the short
  // SHA for older checkouts that predate the VERSION file. Last-resort
  // ellipsis when nothing's known yet.
  return s.local_version || s.local_short || '…';
}

function _versionRemoteLabel(s) {
  return s.remote_version || s.remote_short || 'latest';
}

function renderVersionPill() {
  const pill = document.getElementById('versionPill');
  if (!pill) return;
  const s = _versionState || {};
  const local = _versionDisplayLabel(s);
  const remote = _versionRemoteLabel(s);
  // Reset every state class; exactly one is added below.
  pill.classList.remove('pill-update','pill-current','pill-dev','pill-checking','pill-restart','pill-busy');
  pill.style.display = '';

  // Pill text leads with the MEANING of the state, not the version code.
  // Earlier build showed bare "Y1.005" which read as a label rather than
  // a status — users didn't realize they could click it. Now every state
  // uses plain English so a user glancing at the header understands at
  // a glance whether they're current, behind, or need to restart.

  // Highest-priority state: a pull just happened and the panel needs a
  // restart to load the new code.
  if (_versionRestartPending) {
    pill.classList.add('pill-restart');
    pill.textContent = '↻ Restart Phosphene';
    const v = s.pull_pulled_to_version || s.pull_pulled_to_short || 'the new code';
    pill.title = s.pull_requires_full_update
      ? `Pulled ${v}. This update touched dependencies — use Pinokio's Update button (not just Stop+Start).`
      : `Pulled ${v}. Click Stop → Start in Pinokio to apply.`;
    return;
  }
  // Suppressed (dev branch / dirty tree / no git).
  if (s.suppress_reason) {
    pill.classList.add('pill-dev');
    pill.textContent = `${local} · dev`;
    pill.title = `Update check paused: ${s.suppress_reason}.`;
    return;
  }
  // Behind origin/main — eye-catching action prompt.
  if (!s.error && s.checked_ts && (s.behind_by | 0) > 0) {
    pill.classList.add('pill-update');
    pill.textContent = `↑ Update to ${remote}`;
    pill.title = `You're on ${local}; latest is ${remote}. Click to pull the update.`;
    return;
  }
  // Last check errored (offline).
  if (s.error) {
    pill.classList.add('pill-dev');
    pill.textContent = `${local} · offline`;
    pill.title = `Couldn't reach github.com (${s.error}). Click to retry.`;
    return;
  }
  // Current with origin/main.
  if (s.checked_ts && (s.behind_by | 0) === 0) {
    pill.classList.add('pill-current');
    pill.textContent = `✓ Up to date · ${local}`;
    pill.title = `You're on ${local}, the latest version. Click to re-check now.`;
    return;
  }
  // First poll hasn't landed yet.
  pill.classList.add('pill-checking');
  pill.textContent = `Checking · ${local}`;
  pill.title = 'Checking for updates…';
}

// One click — does the right thing for the current state. Magic button.
async function versionPillClick() {
  if (_versionRestartPending) {
    // Educational click: tell the user what's needed.
    const s = _versionState || {};
    const tip = s.pull_requires_full_update
      ? "Pulled. Because this update touched Python deps / patches, use Pinokio's Update button (it reinstalls + reapplies patches). After that click Start."
      : "Pulled. Click Stop, then Start in Pinokio to apply (your queue and settings are preserved).";
    alert(tip);
    return;
  }
  const s = _versionState || {};
  if (s.suppress_reason) {
    alert(`Update check is paused: ${s.suppress_reason}.\n\n` +
          `Phosphene only checks GitHub when you're on a clean main branch. ` +
          `Commit your local changes (or switch back to main) to re-enable updates.`);
    return;
  }
  // Behind: pull the update.
  if (!s.error && s.checked_ts && (s.behind_by | 0) > 0) {
    await versionDoPull();
    return;
  }
  // Current OR error OR pre-first-poll: re-check now.
  await versionDoRefresh();
}

async function versionDoRefresh() {
  const pill = document.getElementById('versionPill');
  pill.classList.add('pill-busy');
  const origText = pill.textContent;
  pill.textContent = '⟳ checking…';
  try {
    const r = await fetch('/version/check', { method: 'POST' });
    const data = await r.json();
    if (data && data.state) _versionState = data.state;
  } catch (e) {
    // Leave _versionState as-is so the pill returns to the prior render
    // instead of flashing to "unknown".
  }
  pill.classList.remove('pill-busy');
  renderVersionPill();
}

async function versionDoPull() {
  const s = _versionState || {};
  const target = _versionRemoteLabel(s);
  const local = _versionDisplayLabel(s);
  const ok = confirm(
    `Pull update from ${local} → ${target}?\n\n` +
    `This runs git pull on your phosphene install. After it succeeds, ` +
    `you'll need to click Stop, then Start in Pinokio to load the new code. ` +
    `Your queue and settings are preserved across restarts.`
  );
  if (!ok) return;
  const pill = document.getElementById('versionPill');
  pill.classList.add('pill-busy');
  pill.textContent = '⟳ pulling…';
  try {
    const r = await fetch('/version/pull', { method: 'POST' });
    const data = await r.json();
    if (data && data.state) _versionState = data.state;
    if (!r.ok || !data.ok) {
      pill.classList.remove('pill-busy');
      renderVersionPill();
      alert(`Pull failed:\n\n${(data && data.error) || 'unknown error'}\n\n` +
            `Tip: try the full Pinokio Update button instead — it also handles ` +
            `cases where you have local changes that block a fast-forward.`);
      return;
    }
    _versionRestartPending = true;
    pill.classList.remove('pill-busy');
    renderVersionPill();
    const newVersion = (data.state && (data.state.pull_pulled_to_version || data.state.pull_pulled_to_short)) || 'new version';
    const fullUpdateNote = data.state && data.state.pull_requires_full_update
      ? `\n\n⚠ This update touched Python dependencies / patches. Use ` +
        `Pinokio's Update button (not just Stop+Start) so deps reinstall.`
      : '';
    alert(`Pulled to ${newVersion}.\n\nClick Stop, then Start in Pinokio to apply.${fullUpdateNote}`);
  } catch (e) {
    pill.classList.remove('pill-busy');
    renderVersionPill();
    alert(`Pull failed: ${e.message || e}`);
  }
}

// Boot: first /version read happens 2 seconds after DOM ready (gives the
// panel's startup-delay thread time to complete its first remote check),
// then every 5 minutes thereafter.
setTimeout(refreshVersionPill, 2000);
setInterval(refreshVersionPill, 5 * 60 * 1000);

// ====== Init ======
setInterval(poll, 1500);
poll();
setMode('t2v');
setAspect('landscape');         // sets aspect first so the default preset orients correctly
setQuality('balanced');         // bundles quality + dims; respects current aspect
applyTierTimes();               // rewrite Quality pill subtitles to match this Mac
updateCustomizeSummary();
updateDerived();

// Wire the picker components (I2V image + FFLF start/end) and seed the
// "Recent uploads" strip. The strip is shared across all three pickers,
// so dropping a new image in one slot makes it instantly clickable in
// the other two.
PICKERS.forEach(pickerWire);
refreshUploadsStrip();
// Refresh the strip whenever a render finishes (queue/history changes
// don't fire here), and whenever the user opens FFLF — covers the case
// where they uploaded something via I2V, then switched to FFLF.
document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.addEventListener('click', refreshUploadsStrip));


// ============================================================================
// AGENTIC FLOWS — chat UI (rich version)
// ============================================================================
//
// Pieces:
//   - Workflow tab switch (Manual / Agentic Flows).
//   - Chat with avatar bubbles, markdown rendering, expandable tool cards.
//   - Composer with auto-resize textarea + circular send button (disabled
//     when empty, Cmd/Ctrl+Enter to fire).
//   - Sessions dropdown (from the header session-title click).
//   - Engine settings drawer (modal).
//
// Server-authoritative: after every user message we re-render the whole
// thread from /agent/sessions/<id>'s `rendered_messages` payload. The
// optimistic user bubble + typing row are inserted immediately for snap.

window.AGENT = {
  sessionId: null,
  config: null,
  busy: false,
  models: [],
  sessions: [],            // cached list for the dropdown
  selectedAnchors: {},     // {shot_label: candidate_obj} — synced from session.tool_state
  imageConfig: null,       // {kind, has_bfl_api_key, ...}
};

function workflowSwitch(name) {
  document.querySelectorAll('#workflowTabs button[data-workflow]')
    .forEach(b => b.classList.toggle('active', b.dataset.workflow === name));
  const manual = document.getElementById('genForm');
  const agent = document.getElementById('agentPane');
  // Set body data attribute so CSS can switch the layout (wider form-pane,
  // show agent-stage-pane on the right).
  document.body.setAttribute('data-workflow', name);
  if (name === 'agent') {
    if (manual) manual.style.display = 'none';
    if (agent) agent.hidden = false;
    agentRefreshConfig();
    if (!window.AGENT.sessionId) {
      const stored = localStorage.getItem('phos_agent_session');
      if (stored) agentLoadSession(stored);
      else agentLoadMostRecent();
    } else {
      agentLoadSession(window.AGENT.sessionId);
    }
    agentStageStart();
    setTimeout(() => {
      const ta = document.getElementById('agentInput');
      if (ta) { agentAutoResize(ta); ta.focus(); }
    }, 50);
  } else {
    if (manual) manual.style.display = '';
    if (agent) agent.hidden = true;
    agentStageStop();
  }
  try { localStorage.setItem('phos_workflow', name); } catch(e) {}
}

document.querySelectorAll('#workflowTabs button[data-workflow]').forEach(b => {
  b.addEventListener('click', () => workflowSwitch(b.dataset.workflow));
});

// ---- Engine status (in the header pill) -----------------------------------
// Human-readable model name. mlx-community ships ugly slugs like
// "Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4Bit" or
// "gemma-3-12b-it-4bit" that fill the pill. Keep the "family · size"
// signal and drop the rest.
function formatModelName(raw) {
  if (!raw) return '';
  let s = String(raw);
  // Strip common quantization / format suffixes anywhere in the slug.
  s = s.replace(/-(MLX|mlx|GGUF|gguf|AWQ|GPTQ)\b.*$/, '');
  s = s.replace(/-(it|chat|instruct|inst)\b/i, '');
  s = s.replace(/-?(4bit|8bit|q4|q8|bf16|fp16|int4|int8)$/i, '');
  // If it still has flavor tags like "-Abliterated-Heretic", drop those
  // tail tokens unless they're size-bearing.
  // First collapse the family + size if we can detect them.
  const fam = (s.match(/^([A-Za-z][A-Za-z0-9.]*)/) || [])[1] || '';
  const size = (s.match(/(\d+(?:\.\d+)?)[Bb]\b/) || [])[1] || '';
  if (fam && size) {
    // Special-case Gemma's "gemma-3-12b" → "Gemma 3 12B".
    if (/^gemma$/i.test(fam)) {
      const ver = (s.match(/^gemma-(\d)/i) || [])[1];
      return ver ? `Gemma ${ver} ${size}B` : `Gemma ${size}B`;
    }
    // Default: "Qwen3.6 35B"
    return `${fam} ${size}B`;
  }
  // Fallback — truncate the raw slug.
  return s.length > 22 ? s.slice(0, 20) + '…' : s;
}

async function agentRefreshConfig() {
  try {
    const r = await fetch('/agent/config');
    const j = await r.json();
    window.AGENT.config = j;
    window.AGENT.models = j.available_models || [];
    const eng = j.engine || {};
    const local = j.local_server || {};
    const dot = document.getElementById('agentEngineDot');
    const label = document.getElementById('agentEngineLabel');
    let live = false;
    let summary = '';
    if (eng.kind === 'phosphene_local') {
      live = !!local.running;
      const modelName = formatModelName(eng.model || '');
      // Look up the resident size from the discovered models so the user
      // sees "22 GB" inline next to the name — no surprise loads.
      let sizeBit = '';
      try {
        const path = eng.local_model_path || '';
        const list = j.available_models || [];
        const hit = list.find(m => m.path === path);
        if (hit && hit.size_gb) sizeBit = ` · ${hit.size_gb.toFixed(1)} GB`;
      } catch (e) {}
      summary = live
        ? `${modelName || 'Local'}${sizeBit} · live`
        : `${modelName || 'Local'}${sizeBit} · click to start`;
    } else {
      const u = (eng.base_url || '').replace(/^https?:\/\//, '').replace(/\/v1$/, '');
      summary = `${eng.model || 'remote'} · ${u}`;
      live = !!eng.has_api_key;
    }
    if (dot) {
      dot.classList.remove('live', 'warn', 'bad');
      dot.classList.add(live ? 'live' : 'warn');
    }
    if (label) label.textContent = summary;
    agentRenderModePill();
    // Stop-engine quick action: only meaningful when a local engine is
    // actually running. Hide it for remote engines and when nothing is
    // resident (clicking would no-op).
    const stopBtn = document.getElementById('agentEngineStopBtn');
    if (stopBtn) {
      const isLocal = eng.kind === 'phosphene_local';
      stopBtn.hidden = !(isLocal && local && local.running);
    }
  } catch (e) {
    const label = document.getElementById('agentEngineLabel');
    if (label) label.textContent = 'engine unavailable';
  }
}

function agentRenderModePill() {
  const pill = document.getElementById('agentModePill');
  const icon = document.getElementById('agentModeIcon');
  const label = document.getElementById('agentModeLabel');
  if (!pill) return;
  const mode = ((window.AGENT.config && window.AGENT.config.engine
                 && window.AGENT.config.engine.mode) || 'plan_sleep');
  if (mode === 'interactive') {
    pill.classList.add('is-interactive');
    if (icon) icon.textContent = '💬';
    if (label) label.textContent = 'Interactive';
    pill.title = 'Interactive: chat model stays resident across finishes (uses ~22 GB for Qwen 35B).\nClick to switch to Plan & sleep mode.';
  } else {
    pill.classList.remove('is-interactive');
    if (icon) icon.textContent = '🌙';
    if (label) label.textContent = 'Plan & sleep';
    pill.title = 'Plan & sleep: auto-stop the chat model after the agent finishes (frees RAM for renders).\nClick to switch to Interactive mode.';
  }
}

async function agentToggleMode() {
  const cur = ((window.AGENT.config && window.AGENT.config.engine
                && window.AGENT.config.engine.mode) || 'plan_sleep');
  const next = cur === 'interactive' ? 'plan_sleep' : 'interactive';
  try {
    const r = await fetch('/agent/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: next}),
    });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
    await agentRefreshConfig();
  } catch (e) {
    alert('Could not switch mode: ' + e.message);
  }
}

async function agentEngineStopNow() {
  const btn = document.getElementById('agentEngineStopBtn');
  if (btn) btn.disabled = true;
  try {
    const r = await fetch('/agent/local/stop', {method: 'POST'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    // Reflect the new state in the header without waiting for the next
    // /status poll.
    await agentRefreshConfig();
  } catch (e) {
    alert('Stop failed: ' + e.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ---- Sessions -------------------------------------------------------------
async function agentNewSession(initialMessage) {
  const title = (initialMessage || '').slice(0, 60) || 'New chat';
  const r = await fetch('/agent/sessions/new', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title}),
  });
  const j = await r.json();
  if (!j.ok) {
    alert('Could not create session: ' + (j.error || 'unknown'));
    return null;
  }
  window.AGENT.sessionId = j.session.session_id;
  try { localStorage.setItem('phos_agent_session', j.session.session_id); } catch(e) {}
  agentSetSessionTitle(j.session.title || 'New chat', j.session.session_id);
  agentRender([]);
  return j.session;
}

async function agentLoadMostRecent() {
  try {
    const r = await fetch('/agent/sessions');
    const j = await r.json();
    const sessions = j.sessions || [];
    window.AGENT.sessions = sessions;
    if (sessions.length === 0) {
      agentRender([]);
      return;
    }
    const mostRecent = sessions[0];
    await agentLoadSession(mostRecent.session_id);
    try { localStorage.setItem('phos_agent_session', mostRecent.session_id); } catch(e) {}
  } catch (e) { console.warn('agentLoadMostRecent', e); }
}

async function agentLoadSession(sid) {
  try {
    const r = await fetch('/agent/sessions/' + encodeURIComponent(sid));
    if (!r.ok) {
      try { localStorage.removeItem('phos_agent_session'); } catch(e) {}
      window.AGENT.sessionId = null;
      window.AGENT.selectedAnchors = {};
      agentSetSessionTitle('New chat', null);
      agentRender([]);
      return;
    }
    const j = await r.json();
    window.AGENT.sessionId = sid;
    const sess = j.session || {};
    window.AGENT.selectedAnchors = (sess.tool_state || {}).selected_anchors || {};
    agentSetSessionTitle(sess.title || 'Untitled', sid);
    agentRender(j.rendered_messages || []);
  } catch (e) { console.error('agentLoadSession', e); }
}

function agentSetSessionTitle(title, sid) {
  const el = document.getElementById('agentSessionTitle');
  if (!el) return;
  el.innerHTML = '';
  const t = document.createElement('span');
  t.textContent = title;
  el.appendChild(t);
  if (sid) {
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = '· ' + sid.slice(0, 8);
    el.appendChild(m);
  }
}

// ============================================================================
// AGENT SESSIONS SIDEBAR (Cmd+K)
// ============================================================================
// Replaces the old absolute-positioned popover with a slide-in sidebar.
// Searchable, keyboard-navigable, time-bucketed. Pinning persists in
// localStorage and reflows the form-pane when active.
window.ASP = {
  open: false,
  pinned: false,
  query: '',
  all: [],          // raw /agent/sessions response
  filtered: [],     // bucketed for render
  focusIndex: -1,   // arrow-key cursor
};

function aspInit() {
  // Restore pinned state.
  try {
    if (localStorage.getItem('phos_asp_pinned') === '1') {
      document.body.classList.add('asp-pinned');
      window.ASP.pinned = true;
    }
  } catch (e) {}
  // Update the trigger count badge from any cached list.
  aspUpdateTriggerCount();
  // Search input.
  const inp = document.getElementById('aspSearchInput');
  if (inp) {
    let to = null;
    inp.addEventListener('input', () => {
      window.ASP.query = inp.value;
      window.ASP.focusIndex = -1;
      clearTimeout(to);
      to = setTimeout(aspRefilterAndRender, 80);
    });
  }
  // Global keyboard shortcuts.
  document.addEventListener('keydown', (e) => {
    // Cmd/Ctrl+K toggles the sidebar.
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      aspToggle();
      return;
    }
    if (window.ASP.open) {
      // "/" focuses the search input.
      if (e.key === '/' && document.activeElement !== inp) {
        e.preventDefault();
        inp && inp.focus();
        return;
      }
      // Esc closes (only when not pinned).
      if (e.key === 'Escape' && !window.ASP.pinned) {
        const modal = document.getElementById('agentSettingsModal');
        const browser = document.getElementById('modelBrowserModal');
        const lb = document.getElementById('agentStageLightbox');
        // Don't steal Esc from layered modals.
        if ((modal && modal.classList.contains('open')) ||
            (browser && browser.classList.contains('open')) ||
            (lb && lb.classList.contains('open'))) return;
        aspClose();
        return;
      }
      // Arrow nav + Enter on the list.
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        aspMoveFocus(1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        aspMoveFocus(-1);
      } else if (e.key === 'Enter') {
        const items = window.ASP.filtered;
        if (window.ASP.focusIndex >= 0 && window.ASP.focusIndex < items.length) {
          e.preventDefault();
          aspLoad(items[window.ASP.focusIndex].session_id);
        }
      }
    }
  });
}

async function aspToggle() {
  if (window.ASP.open) aspClose();
  else await aspOpen();
}

async function aspOpen() {
  const panel = document.getElementById('agentSessionsPanel');
  const backdrop = document.getElementById('aspBackdrop');
  if (!panel) return;
  // Refresh list before opening.
  try {
    const r = await fetch('/agent/sessions');
    const j = await r.json();
    window.ASP.all = j.sessions || [];
    window.AGENT.sessions = window.ASP.all;  // keep legacy var in sync
  } catch (e) { /* swallow — show whatever's cached */ }
  aspRefilterAndRender();
  panel.dataset.state = 'open';
  panel.setAttribute('aria-hidden', 'false');
  if (backdrop && !window.ASP.pinned) backdrop.classList.add('is-shown');
  if (backdrop) backdrop.hidden = false;
  window.ASP.open = true;
  aspUpdateTriggerCount();
  // Focus search after slide-in animation.
  setTimeout(() => {
    const inp = document.getElementById('aspSearchInput');
    if (inp) inp.focus();
  }, 120);
}

function aspClose() {
  if (window.ASP.pinned) return;   // pin overrides close
  const panel = document.getElementById('agentSessionsPanel');
  const backdrop = document.getElementById('aspBackdrop');
  if (!panel) return;
  panel.dataset.state = 'closed';
  panel.setAttribute('aria-hidden', 'true');
  if (backdrop) backdrop.classList.remove('is-shown');
  window.ASP.open = false;
  window.ASP.focusIndex = -1;
}

function aspTogglePin() {
  const next = !window.ASP.pinned;
  window.ASP.pinned = next;
  document.body.classList.toggle('asp-pinned', next);
  const backdrop = document.getElementById('aspBackdrop');
  if (backdrop) {
    if (next) backdrop.classList.remove('is-shown');
    else if (window.ASP.open) backdrop.classList.add('is-shown');
  }
  try { localStorage.setItem('phos_asp_pinned', next ? '1' : ''); } catch (e) {}
  // When pinning, ensure the panel is open (otherwise pinning a closed
  // panel does nothing visible).
  if (next && !window.ASP.open) aspOpen();
}

function aspRelTime(ts) {
  if (!ts) return '';
  const now = Date.now() / 1000;
  const d = now - ts;
  if (d < 60) return 'just now';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  if (d < 86400 * 2) return 'Yesterday';
  if (d < 86400 * 7) return Math.floor(d / 86400) + 'd ago';
  const dt = new Date(ts * 1000);
  return dt.toLocaleString(undefined, {month: 'short', day: 'numeric'});
}

function aspBucket(ts) {
  if (!ts) return 'Earlier';
  const now = Date.now() / 1000;
  const d = now - ts;
  if (d < 86400) return 'Today';
  if (d < 86400 * 2) return 'Yesterday';
  if (d < 86400 * 7) return 'This week';
  return 'Earlier';
}

function aspRefilterAndRender() {
  const q = (window.ASP.query || '').toLowerCase().trim();
  const all = (window.ASP.all || []).slice();
  // Sort newest first.
  all.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
  // Filter on title + session id prefix (preview from messages would
  // require a full session fetch per row — defer to v2).
  const filtered = q
    ? all.filter(s => (s.title || '').toLowerCase().includes(q) ||
                      (s.session_id || '').toLowerCase().includes(q))
    : all;
  window.ASP.filtered = filtered;
  aspRender();
  aspUpdateTriggerCount();
}

function aspRender() {
  const list = document.getElementById('aspList');
  const empty = document.getElementById('aspEmpty');
  if (!list || !empty) return;
  list.innerHTML = '';
  const items = window.ASP.filtered || [];
  if (items.length === 0) {
    empty.hidden = false;
    if (window.ASP.query) {
      empty.querySelector('.asp-empty-title').textContent = 'No matches';
      empty.querySelector('.asp-empty-hint').textContent = 'Try a different query, or clear the search.';
    } else {
      empty.querySelector('.asp-empty-title').textContent = 'No sessions yet';
      empty.querySelector('.asp-empty-hint').textContent = 'Type a prompt to start your first one.';
    }
    return;
  }
  empty.hidden = true;

  // Pinned sessions surface at the top regardless of recency. Then the
  // standard time buckets in order. All rows still share one keyboard
  // focus index so arrow nav flows naturally across sections.
  const pinnedItems = items.filter(s => s.pinned);
  const restItems = items.filter(s => !s.pinned);
  const buckets = {Today: [], Yesterday: [], 'This week': [], Earlier: []};
  for (const s of restItems) {
    const b = aspBucket(s.updated_at);
    if (!buckets[b]) buckets[b] = [];
    buckets[b].push(s);
  }
  let flatIdx = 0;
  if (pinnedItems.length) {
    const head = document.createElement('div');
    head.className = 'asp-section-label';
    head.textContent = 'Pinned';
    list.appendChild(head);
    for (const s of pinnedItems) { list.appendChild(aspMakeItem(s, flatIdx)); flatIdx++; }
  }
  for (const label of ['Today', 'Yesterday', 'This week', 'Earlier']) {
    const bucket = buckets[label];
    if (!bucket || bucket.length === 0) continue;
    const head = document.createElement('div');
    head.className = 'asp-section-label';
    head.textContent = label;
    list.appendChild(head);
    for (const s of bucket) {
      list.appendChild(aspMakeItem(s, flatIdx));
      flatIdx++;
    }
  }
}

function aspMakeItem(s, flatIdx) {
  const item = document.createElement('div');
  item.className = 'asp-item';
  if (s.session_id === window.AGENT.sessionId) item.classList.add('is-active');
  if (window.ASP.focusIndex === flatIdx) item.classList.add('is-focused');
  if (s.pinned) item.classList.add('is-pinned');
  item.dataset.sid = s.session_id;
  // Cross-reference live queue: if any of this session's submitted shots
  // are currently in the panel queue or running, show a live dot.
  const queueIds = (window.AGENT && window.AGENT.lastStatus
                    ? (((window.AGENT.lastStatus.queue || [])
                       .concat(window.AGENT.lastStatus.current ? [window.AGENT.lastStatus.current] : [])).map(j => j.id))
                    : []);
  const sessionShotIds = (s.submitted_shot_ids || []);   // optional field
  const isRunning = sessionShotIds.some(id => queueIds.includes(id));
  const time = aspRelTime(s.updated_at);
  const preview = (s.preview || '').trim();
  item.innerHTML = `
    <div class="asp-row">
      <div class="asp-item-title">${escapeHtml(s.title || 'Untitled')}</div>
      <div class="asp-item-time">${escapeHtml(time)}</div>
    </div>
    ${preview ? `<div class="asp-item-preview">${escapeHtml(preview)}</div>` : ''}
    <div class="asp-item-meta">
      <span class="asp-chip">${s.messages || 0} msg</span>
      <span class="asp-chip asp-chip-shots">${s.shots_submitted || 0} shots</span>
      ${isRunning ? '<span class="asp-status-running" title="Active in queue"></span>' : ''}
    </div>
    <div class="asp-actions">
      <button type="button" class="asp-action-btn" data-action="pin" title="${s.pinned ? 'Unpin' : 'Pin'}">
        <svg class="asp-pin-icon" viewBox="0 0 24 24" fill="${s.pinned ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="17" x2="12" y2="22"/>
          <path d="M5 17h14l-2-7V4H7v6z"/>
        </svg>
      </button>
      <button type="button" class="asp-action-btn" data-action="rename" title="Rename">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
        </svg>
      </button>
      <button type="button" class="asp-action-btn danger" data-action="delete" title="Delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
          <path d="M10 11v6M14 11v6"/>
        </svg>
      </button>
    </div>
  `;
  item.addEventListener('click', (e) => {
    // If user clicked one of the action buttons, dispatch the action
    // instead of loading the session.
    const btn = e.target.closest('[data-action]');
    if (btn) {
      e.stopPropagation();
      const action = btn.dataset.action;
      if (action === 'delete')      aspDeleteRow(s, item);
      else if (action === 'rename') aspBeginRename(s, item);
      else if (action === 'pin')    aspTogglePinRow(s);
      return;
    }
    aspLoad(s.session_id);
  });
  return item;
}

async function aspDeleteRow(s, itemEl) {
  if (!confirm(`Delete "${s.title || 'Untitled'}"? This can't be undone.`)) return;
  try {
    const r = await fetch('/agent/sessions/' + encodeURIComponent(s.session_id) + '/delete', {method: 'POST'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
  } catch (e) { alert('Delete failed: ' + e.message); return; }
  // Remove from in-memory list and re-render. If the deleted session
  // was the active one, fall back to the most-recent remaining.
  window.ASP.all = (window.ASP.all || []).filter(x => x.session_id !== s.session_id);
  if (s.session_id === window.AGENT.sessionId) {
    window.AGENT.sessionId = null;
    try { localStorage.removeItem('phos_agent_session'); } catch (e) {}
    const remaining = (window.ASP.all || []).slice().sort((a,b) => (b.updated_at||0) - (a.updated_at||0));
    if (remaining.length) await agentLoadSession(remaining[0].session_id);
    else {
      // Empty UI — clear the chat surface.
      const chat = document.getElementById('agentChat');
      if (chat) chat.innerHTML = '';
      agentSetSessionTitle('New chat', null);
    }
  }
  aspRefilterAndRender();
}

function aspBeginRename(s, itemEl) {
  const titleEl = itemEl.querySelector('.asp-item-title');
  if (!titleEl) return;
  const original = s.title || '';
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'asp-item-title-edit';
  input.value = original;
  titleEl.replaceWith(input);
  input.focus();
  input.select();
  const finish = async (commit) => {
    if (!input.parentNode) return;
    const next = (input.value || '').trim();
    const wrap = document.createElement('div');
    wrap.className = 'asp-item-title';
    wrap.textContent = (commit && next) ? next : original;
    input.replaceWith(wrap);
    if (!commit || !next || next === original) return;
    try {
      const r = await fetch('/agent/sessions/' + encodeURIComponent(s.session_id) + '/rename', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title: next}),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      s.title = next;
      // Update the in-memory cache + active-session header if relevant.
      const cached = (window.ASP.all || []).find(x => x.session_id === s.session_id);
      if (cached) cached.title = next;
      if (window.AGENT.sessionId === s.session_id) {
        agentSetSessionTitle(next, s.session_id);
      }
    } catch (e) { alert('Rename failed: ' + e.message); }
  };
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); finish(true); }
    else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    e.stopPropagation();                // don't bubble to global asp shortcuts
  });
}

async function aspTogglePinRow(s) {
  const next = !s.pinned;
  try {
    const r = await fetch('/agent/sessions/' + encodeURIComponent(s.session_id) + '/pin', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pinned: next}),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    s.pinned = next;
    const cached = (window.ASP.all || []).find(x => x.session_id === s.session_id);
    if (cached) cached.pinned = next;
    aspRefilterAndRender();
  } catch (e) { alert('Pin failed: ' + e.message); }
}

function aspMoveFocus(delta) {
  const items = window.ASP.filtered || [];
  if (items.length === 0) return;
  const next = Math.max(0, Math.min(items.length - 1, window.ASP.focusIndex + delta));
  window.ASP.focusIndex = next;
  aspRender();
  // Scroll into view.
  const list = document.getElementById('aspList');
  if (list) {
    const focused = list.querySelector('.asp-item.is-focused');
    if (focused) focused.scrollIntoView({block: 'nearest'});
  }
}

async function aspLoad(sid) {
  if (!window.ASP.pinned) aspClose();
  await agentLoadSession(sid);
  // Re-render to update the active state styling.
  aspRender();
}

function aspUpdateTriggerCount() {
  const el = document.getElementById('aspTriggerCount');
  if (!el) return;
  const n = (window.ASP.all || []).length;
  if (n > 0) {
    el.textContent = String(n);
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

// Refresh the cached list periodically while the agent tab is active —
// keeps the trigger count and "running" dots honest without forcing
// re-fetches on every keystroke.
setInterval(() => {
  if (document.body.getAttribute('data-workflow') !== 'agent') return;
  fetch('/agent/sessions').then(r => r.json()).then(j => {
    window.ASP.all = j.sessions || [];
    aspUpdateTriggerCount();
    if (window.ASP.open) aspRefilterAndRender();
  }).catch(() => {});
}, 10000);

// Boot.
document.addEventListener('DOMContentLoaded', aspInit);
if (document.readyState !== 'loading') aspInit();

// ---- Markdown rendering ---------------------------------------------------
// Tight subset: headers, bold, italic, inline code, code blocks, lists,
// tables, blockquotes, hr, paragraphs. Escapes HTML first; processes
// markdown on already-safe text. Code blocks are pulled out of the way
// before other regexes run, then restored.
function mdToHtml(src) {
  if (!src) return '';
  let s = String(src);
  // Strip the fenced ```action ...``` blocks we use for tool calls — they're
  // already represented as tool-call cards, no need to show the JSON twice.
  // (Same prefilter as before; runs before marked sees the source.)
  s = s.replace(/```(?:action|tool|json action|action_json)\s*\n[\s\S]*?\n```/gi, '').trim();
  if (!s) return '';
  // marked + DOMPurify are loaded as <script> globals from /webapp/vendor/.
  // Fall back to escaped plaintext if either is missing so chat bubbles
  // never render blank.
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    return `<p>${escapeHtml(s).replace(/\n/g, '<br>')}</p>`;
  }
  const html = marked.parse(s, { gfm: true, breaks: false });
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

// ---- Chat rendering -------------------------------------------------------
function agentRender(messages) {
  const chat = document.getElementById('agentChat');
  if (!chat) return;
  chat.innerHTML = '';
  if (!messages || messages.length === 0) {
    chat.appendChild(renderEmpty());
    return;
  }
  for (const m of messages) chat.appendChild(renderMessage(m));
  // Hand-off the scroll on next animation frame so freshly-inserted nodes
  // have measured heights.
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function renderEmpty() {
  const wrap = document.createElement('div');
  wrap.className = 'agent-empty';
  wrap.innerHTML = `
    <div class="badge">Beta · Phosphene Agentic Flows</div>
    <h3>Plan a film overnight</h3>
    <p>Paste a script or describe a piece. I'll plan the shots, estimate the wall time, and queue the renders.</p>
    <p>You wake up to mp4s and a manifest.json.</p>
    <div class="examples">
      <div class="example" data-prompt="I want to make a short movie. Help me plan it shot by shot, then we'll generate it together.">
        <span>I want to make a short movie</span>
        <span class="arrow">→</span>
      </div>
      <div class="example" data-prompt="I want to make a 30-minute video. Walk me through how to break it into renderable shots given LTX 2.3's per-clip limits.">
        <span>I want to make a 30-minute video</span>
        <span class="arrow">→</span>
      </div>
      <div class="example" data-prompt="I want to make clips for my existing project. Ask me about the project first, then we'll plan the next batch of shots together.">
        <span>I want to make clips for my existing project</span>
        <span class="arrow">→</span>
      </div>
    </div>
  `;
  wrap.querySelectorAll('.example').forEach(b => {
    b.addEventListener('click', () => {
      const ta = document.getElementById('agentInput');
      if (!ta) return;
      ta.value = b.dataset.prompt;
      agentAutoResize(ta);
      agentUpdateSendState();
      ta.focus();
    });
  });
  return wrap;
}

function renderMessage(m) {
  if (m.kind === 'tool_result') return renderToolResultCard(m.result || {});
  if (m.kind === 'system_note') return renderSystemNote(m.content || '');

  const row = document.createElement('div');
  row.className = 'agent-msg-row';

  const av = document.createElement('div');
  av.className = `agent-avatar ${m.kind === 'user' ? 'user' : 'claude'}`;
  av.textContent = m.kind === 'user' ? 'U' : 'C';

  const body = document.createElement('div');
  body.className = 'agent-msg-body';

  const name = document.createElement('div');
  name.className = 'agent-msg-name';
  name.textContent = m.kind === 'user' ? 'You' : 'Claude';

  body.appendChild(name);

  if (m.attachments && m.attachments.length) {
    body.appendChild(renderAttachmentChips(m.attachments));
  }

  const content = document.createElement('div');
  content.className = 'agent-msg-content agent-md';
  content.innerHTML = mdToHtml(m.content || '');
  body.appendChild(content);

  if (m.tool_call) body.appendChild(renderToolCallCard(m.tool_call));

  row.appendChild(av);
  row.appendChild(body);
  return row;
}

function renderAttachmentChips(attachments) {
  const wrap = document.createElement('div');
  wrap.className = 'agent-msg-attachments';
  for (const a of attachments) {
    const kind = AGENT_ATTACH_KIND(a.mime, a.name);
    const link = document.createElement('a');
    link.className = 'agent-msg-attach';
    link.href = '/image?path=' + encodeURIComponent(a.path);
    link.target = '_blank';
    link.rel = 'noopener';
    link.title = a.path;

    const thumb = document.createElement(kind === 'image' ? 'img' : 'div');
    thumb.className = 'thumb';
    if (kind === 'image') {
      thumb.src = '/image?path=' + encodeURIComponent(a.path);
      thumb.alt = a.name;
    } else {
      thumb.textContent = agentAttachIcon(kind);
    }
    const nameEl = document.createElement('span');
    nameEl.className = 'name';
    nameEl.textContent = a.name;
    link.appendChild(thumb);
    link.appendChild(nameEl);
    wrap.appendChild(link);
  }
  return wrap;
}

function renderSystemNote(text) {
  const div = document.createElement('div');
  div.style.cssText = 'text-align:center; padding:8px; font-size:11px; color:var(--muted); font-style:italic;';
  div.textContent = text;
  return div;
}

function renderToolCallCard(call) {
  const card = document.createElement('div');
  card.className = 'agent-tool-card pending';
  const head = document.createElement('div');
  head.className = 'head';
  const summary = summarizeToolCall(call);
  head.innerHTML = `
    <span class="icon">⚙</span>
    <span class="name">${escapeHtml(call.tool || '?')}</span>
    <span class="summary">${escapeHtml(summary)}</span>
    <span class="chevron">›</span>
  `;
  const body = document.createElement('div');
  body.className = 'body';
  const pre = document.createElement('pre');
  try { pre.textContent = JSON.stringify(call.args || {}, null, 2); }
  catch(e) { pre.textContent = String(call.args); }
  body.appendChild(pre);
  card.appendChild(head);
  card.appendChild(body);
  head.addEventListener('click', () => card.classList.toggle('open'));
  return card;
}

function renderToolResultCard(result) {
  const card = document.createElement('div');
  const ok = result.ok !== false && !result.error;
  card.className = 'agent-tool-card ' + (ok ? 'success' : 'error');
  const head = document.createElement('div');
  head.className = 'head';
  const inner = result.result;
  const summary = ok ? summarizeToolResult(inner) : (result.error || 'failed');
  head.innerHTML = `
    <span class="icon ${ok ? 'success' : 'error'}">${ok ? '✓' : '✗'}</span>
    <span class="name">${ok ? 'result' : 'error'}</span>
    <span class="summary">${escapeHtml(summary)}</span>
    <span class="chevron">›</span>
  `;

  const body = document.createElement('div');
  body.className = 'body';
  const pre = document.createElement('pre');
  try { pre.textContent = JSON.stringify(ok ? inner : result, null, 2); }
  catch(e) { pre.textContent = String(result); }
  body.appendChild(pre);

  card.appendChild(head);
  card.appendChild(body);
  head.addEventListener('click', () => card.classList.toggle('open'));

  // Phase B of the director workflow: when the result carries
  // `candidates`, render an interactive thumbnail grid below the head.
  // The card stays expanded by default so the user can immediately pick.
  if (ok && inner && Array.isArray(inner.candidates) && inner.candidates.length > 0) {
    card.classList.add('open');                    // open by default
    const grid = renderAnchorGrid(inner);
    card.appendChild(grid);
  }
  return card;
}

function renderAnchorGrid(payload) {
  const wrap = document.createElement('div');
  wrap.className = 'anchor-grid-wrap';
  const label = payload.shot_label || 'shot';
  const prompt = payload.prompt || '';
  const engine = payload.engine || '';

  const meta = document.createElement('div');
  meta.className = 'anchor-grid-meta';
  meta.innerHTML = `
    <span class="label-pill">${escapeHtml(label)}</span>
    <span>${payload.candidates.length} candidates · ${escapeHtml(engine)}</span>
    <span style="flex:1"></span>
    <span style="font-size:10px">click to pick</span>
  `;
  wrap.appendChild(meta);

  if (prompt) {
    const p = document.createElement('div');
    p.className = 'anchor-prompt';
    p.textContent = prompt;
    wrap.appendChild(p);
  }

  const grid = document.createElement('div');
  grid.className = 'anchor-grid';

  const selected = window.AGENT.selectedAnchors || {};
  const currentPick = (selected[label] || {}).png_path;

  for (const cand of payload.candidates) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'anchor-cell' + (cand.png_path === currentPick ? ' selected' : '');
    cell.dataset.shotLabel = label;
    cell.dataset.pngPath = cand.png_path;

    const img = document.createElement('img');
    img.src = '/image?path=' + encodeURIComponent(cand.png_path);
    img.alt = label + ' candidate';
    img.loading = 'lazy';
    cell.appendChild(img);

    const check = document.createElement('span');
    check.className = 'check';
    check.textContent = '✓';
    cell.appendChild(check);

    if (typeof cand.seed === 'number' && cand.seed >= 0) {
      const s = document.createElement('span');
      s.className = 'seed';
      s.textContent = 'seed ' + cand.seed;
      cell.appendChild(s);
    }
    if (cand.engine) {
      const e = document.createElement('span');
      e.className = 'engine-tag';
      e.textContent = cand.engine;
      cell.appendChild(e);
    }

    cell.addEventListener('click', () => agentPickAnchor(label, cand, grid));
    grid.appendChild(cell);
  }

  wrap.appendChild(grid);
  return wrap;
}

async function agentPickAnchor(label, cand, gridEl) {
  if (!window.AGENT.sessionId) return;
  // Optimistic UI: mark this cell selected, deselect siblings
  if (gridEl) {
    gridEl.querySelectorAll('.anchor-cell').forEach(c => c.classList.remove('selected'));
    const me = gridEl.querySelector(`[data-png-path="${CSS.escape(cand.png_path)}"]`);
    if (me) me.classList.add('selected');
  }
  window.AGENT.selectedAnchors[label] = cand;

  try {
    const r = await fetch(
      '/agent/sessions/' + encodeURIComponent(window.AGENT.sessionId) + '/anchors/select',
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shot_label: label, png_path: cand.png_path}),
      }
    );
    const j = await r.json();
    if (!r.ok || j.error) {
      alert('Could not save selection: ' + (j.error || ('HTTP ' + r.status)));
      return;
    }
    window.AGENT.selectedAnchors = j.all_selected || window.AGENT.selectedAnchors;
  } catch (e) {
    console.error('agentPickAnchor', e);
  }
}

function summarizeToolCall(call) {
  const t = call.tool || '';
  const a = call.args || {};
  if (t === 'submit_shot') {
    return `${a.label || a.preset_label || 'unnamed'} — ${a.duration_seconds || '?'}s ${a.mode || 't2v'} ${a.quality || 'balanced'}`;
  }
  if (t === 'estimate_shot') {
    return `${a.duration_seconds || '?'}s ${a.mode || 't2v'} ${a.quality || 'balanced'} ${a.accel || ''}`;
  }
  if (t === 'extract_frame') {
    return `${a.which || 'last'} of ${(a.job_id || '').slice(0, 12)}…`;
  }
  if (t === 'wait_for_shot') {
    return (a.job_id || '').slice(0, 14) + '…';
  }
  if (t === 'get_queue_status') return 'queue snapshot';
  if (t === 'write_session_manifest') {
    return a.title || 'manifest';
  }
  if (t === 'finish') return a.summary ? a.summary.slice(0, 80) : 'done';
  if (t === 'upload_image') return (a.attachment_id || '').split('/').pop() || '';
  return Object.keys(a).slice(0, 3).join(', ');
}

function summarizeToolResult(inner) {
  if (typeof inner !== 'object' || inner === null) return String(inner ?? '').slice(0, 100);
  if ('job_id' in inner && 'estimated_wall_human' in inner) {
    return `queued ${inner.job_id} · ETA ${inner.estimated_wall_human}`;
  }
  if ('manifest_path' in inner) {
    return `manifest written · ${inner.shot_count || '?'} shots`;
  }
  if ('estimate_wall_human' in inner) return `ETA ${inner.estimate_wall_human}`;
  if ('png_path' in inner) {
    return `frame ${inner.frame_index} → ${(inner.png_path || '').split('/').pop()}`;
  }
  if ('summary' in inner) return inner.summary.slice(0, 100);
  if ('queue_depth' in inner) {
    return `queue ${inner.queue_depth}, total ${inner.total_estimated_wall_human || '?'}`;
  }
  if ('status' in inner && 'output_path' in inner) {
    return `${inner.status} · ${(inner.output_path || '').split('/').pop() || '-'}`;
  }
  if ('absolute_path' in inner) {
    return (inner.name || inner.absolute_path);
  }
  // Fallback: show top 2 keys
  return Object.entries(inner).slice(0, 2)
    .map(([k, v]) => `${k}=${typeof v === 'object' ? '...' : String(v).slice(0, 40)}`).join(', ');
}

function renderTypingRow(msg) {
  const row = document.createElement('div');
  row.className = 'agent-typing-row';
  row.id = 'agentTypingRow';
  const av = document.createElement('div');
  av.className = 'agent-avatar claude';
  av.textContent = 'C';
  const bubble = document.createElement('div');
  bubble.className = 'agent-typing-bubble';
  bubble.innerHTML = `
    <span class="agent-typing-dots">
      <span class="agent-typing-dot"></span>
      <span class="agent-typing-dot"></span>
      <span class="agent-typing-dot"></span>
    </span>
    <span id="agentTypingText">${escapeHtml(msg || 'Thinking')}</span>
  `;
  row.appendChild(av);
  row.appendChild(bubble);
  return row;
}

// ---- Send -----------------------------------------------------------------
async function agentSend() {
  if (window.AGENT.busy) return;
  const input = document.getElementById('agentInput');
  const btn = document.getElementById('agentSendBtn');
  const text = (input.value || '').trim();

  // Wait for any in-flight uploads so the message arrives WITH its
  // attachments. UX: don't block the user — if uploads error we surface
  // them in the chip row and the user can remove/retry.
  const stillUploading = (window.AGENT.pendingAttachments || []).some(a => a.status === 'uploading');
  if (stillUploading) {
    btn.disabled = true;
    let waited = 0;
    while ((window.AGENT.pendingAttachments || []).some(a => a.status === 'uploading')) {
      await new Promise(r => setTimeout(r, 120));
      waited += 120;
      if (waited > 60000) break;          // 60 s safety bail
    }
  }
  const ready = agentReadyAttachments();
  if (!text && !ready.length) return;

  if (!window.AGENT.sessionId) {
    const sess = await agentNewSession(text || ready[0].name);
    if (!sess) return;
  }

  // If a Refine reference is set, prepend "Refine <jobid> (<label>): " to
  // the user's message so the agent picks it up as a variation request.
  // Clear the chip after so the next message is a normal one.
  let outgoing = text;
  if (window.AGENT_REFINE) {
    const r = window.AGENT_REFINE;
    const ref = r.jobId || r.clipPath;
    const lbl = r.label ? ` (${r.label})` : '';
    outgoing = `Refine ${ref}${lbl}: ${text}`;
    agentClearRefine();
  }

  const chat = document.getElementById('agentChat');
  // Clear empty-state if present, then append user bubble + typing
  const empty = chat.querySelector('.agent-empty');
  if (empty) empty.remove();
  chat.appendChild(renderMessage({
    kind: 'user',
    content: outgoing,
    attachments: ready.map(a => ({path: a.path, name: a.name, mime: a.mime, size: a.size})),
  }));
  chat.appendChild(renderTypingRow('Thinking'));
  chat.scrollTop = chat.scrollHeight;

  input.value = '';
  agentAutoResize(input);
  // Snapshot then clear before send — if the round-trip succeeds we don't
  // want stale chips. If it fails the user can re-attach.
  const sentAttachments = ready.map(a => ({path: a.path, name: a.name, mime: a.mime, size: a.size}));
  agentClearAttachments();
  agentUpdateSendState();
  window.AGENT.busy = true;
  btn.disabled = true;

  // Streaming-feel: while the message round-trip is in flight (which can
  // take minutes when the agent makes many tool calls on a local model),
  // poll the server-side session every 2 s and re-render as new messages
  // land. The typing indicator stays visible until the round-trip
  // completes (since `busy` doesn't flip until then).
  const sid = window.AGENT.sessionId;
  // Track when we started the current LLM "thinking" wait so the typing
  // indicator can surface "Thinking · 18s" rather than a static phrase
  // during long Qwen 35B inferences. Reset every time a new message lands.
  let lastMsgLen = 0;
  let waitingSince = Date.now();
  let poller = setInterval(async () => {
    if (!window.AGENT.busy) return;
    try {
      const sr = await fetch('/agent/sessions/' + encodeURIComponent(sid));
      if (!sr.ok) return;
      const sj = await sr.json();
      const msgs = sj.rendered_messages || [];
      const cur = chat.querySelectorAll('.agent-msg-row, .agent-tool-card').length;
      if (msgs.length > cur) {
        // Re-render only when the message count actually grew — avoids
        // flickery rebuilds during lulls.
        const typingTextEl = document.getElementById('agentTypingText');
        const phase = typingTextEl ? typingTextEl.textContent : 'Working';
        agentRender(msgs);
        chat.appendChild(renderTypingRow(_phaseFor(msgs, phase, 0)));
        chat.scrollTop = chat.scrollHeight;
        lastMsgLen = msgs.length;
        waitingSince = Date.now();
      } else {
        // No growth — still refresh the typing label so the user sees
        // "Thinking · 12s" instead of a frozen "Drafting plan". Rewriting
        // just the text node avoids re-rendering the whole chat.
        const typingTextEl = document.getElementById('agentTypingText');
        if (typingTextEl) {
          const elapsed = Math.round((Date.now() - waitingSince) / 1000);
          typingTextEl.textContent = _phaseFor(msgs, 'Thinking', elapsed);
        }
      }
    } catch(e) {}
  }, 1500);

  try {
    const r = await fetch(
      '/agent/sessions/' + encodeURIComponent(sid) + '/message',
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: outgoing, attachments: sentAttachments}),
      }
    );
    const j = await r.json();
    if (!r.ok || j.error) {
      const typing = document.getElementById('agentTypingRow');
      if (typing) {
        typing.querySelector('.agent-typing-bubble').innerHTML =
          `<span style="color:#f49a9e">⚠ Error: ${escapeHtml(j.error || ('HTTP ' + r.status))}</span>`;
      }
    } else {
      agentRender(j.rendered_messages || []);
    }
  } catch (e) {
    const typing = document.getElementById('agentTypingRow');
    if (typing) {
      typing.querySelector('.agent-typing-bubble').innerHTML =
        `<span style="color:#f49a9e">⚠ Network error: ${escapeHtml(String(e))}</span>`;
    }
  } finally {
    clearInterval(poller);
    window.AGENT.busy = false;
    btn.disabled = false;
    // The auto-start backend logic may have just spawned the local
    // engine — the engine pill in the header was 'click to start'
    // when the user hit Send. Now it should read 'live'. Refresh
    // the config view so the pill catches up.
    agentRefreshConfig();
    requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
  }
}

// Pick a contextual typing-indicator phrase based on what just happened.
// `elapsedSec` is how long we've been waiting since the last message landed
// — surface it once the wait runs long enough that a frozen label feels
// stuck (Qwen 35B easily takes 30s for a complex turn).
function _phaseFor(messages, fallback, elapsedSec) {
  const tail = (s) => (elapsedSec && elapsedSec >= 6) ? `${s} · ${elapsedSec}s` : s;
  if (!messages || messages.length === 0) return tail(fallback || 'Thinking');
  const last = messages[messages.length - 1];
  if (last.kind === 'user') {
    // We just sent the user's message; the model is composing its reply.
    return tail('Thinking');
  }
  if (last.kind === 'tool_result') {
    const r = last.result || {};
    const inner = r.result || {};
    if (typeof inner === 'object' && inner && 'job_id' in inner) {
      return tail(`Queued ${String(inner.job_id || '').slice(-3) || ''}, planning next`);
    }
    return tail('Reading result, drafting next step');
  }
  if (last.kind === 'assistant') {
    if (last.tool_call) return tail(`Calling ${last.tool_call.tool}`);
    return tail('Drafting next step');
  }
  return tail(fallback || 'Working');
}

// ---- Composer plumbing ----------------------------------------------------
function agentAutoResize(ta) {
  if (!ta) return;
  ta.style.height = 'auto';
  ta.style.height = Math.min(220, Math.max(48, ta.scrollHeight)) + 'px';
}

function agentUpdateSendState() {
  const input = document.getElementById('agentInput');
  const btn = document.getElementById('agentSendBtn');
  if (!input || !btn) return;
  // Allow send when there's text OR at least one ready attachment — the
  // agent can act on attachments alone (e.g. "use this image as the lead").
  const hasText = !!input.value.trim();
  const hasAttach = (window.AGENT.pendingAttachments || []).some(a => a.path);
  btn.disabled = (!hasText && !hasAttach) || window.AGENT.busy;
}

// ---- Attachments ----------------------------------------------------------
// `pendingAttachments` is the staged list before send. Each entry:
//   { id, name, size, mime, path, status: 'uploading'|'ready'|'error', error }
// `path` is filled in once /upload returns. Send is allowed when at least
// one attachment is `ready` (or there is text).
window.AGENT.pendingAttachments = window.AGENT.pendingAttachments || [];

const AGENT_ATTACH_KIND = (mime, name) => {
  const m = (mime || '').toLowerCase();
  const n = (name || '').toLowerCase();
  if (m.startsWith('image/')) return 'image';
  if (m === 'application/pdf' || n.endsWith('.pdf')) return 'pdf';
  if (m.startsWith('text/') || n.endsWith('.md') || n.endsWith('.txt')) return 'text';
  return 'file';
};

const AGENT_ATTACH_MAX_BYTES = 64 * 1024 * 1024;     // mirrors backend cap

function agentFmtBytes(n) {
  if (!n && n !== 0) return '';
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(0) + ' KB';
  return (n/(1024*1024)).toFixed(1) + ' MB';
}

function agentAttachIcon(kind) {
  if (kind === 'image') return '🖼';
  if (kind === 'pdf')   return '📄';
  if (kind === 'text')  return '📝';
  return '📎';
}

function agentRenderAttachRow() {
  const row = document.getElementById('agentAttachRow');
  const btn = document.getElementById('agentAttachBtn');
  const cnt = document.getElementById('agentAttachCount');
  if (!row) return;
  const list = window.AGENT.pendingAttachments;
  if (btn) btn.classList.toggle('has-attach', list.length > 0);
  if (cnt) cnt.textContent = String(list.length);
  if (!list.length) { row.style.display = 'none'; row.innerHTML = ''; return; }
  row.style.display = 'flex';
  row.innerHTML = '';
  for (const a of list) {
    const chip = document.createElement('div');
    chip.className = 'agent-attach-chip';
    if (a.status === 'uploading') chip.classList.add('pending');
    if (a.status === 'error')     chip.classList.add('error');
    const kind = AGENT_ATTACH_KIND(a.mime, a.name);
    const thumb = document.createElement(kind === 'image' && a.path ? 'img' : 'div');
    thumb.className = 'thumb';
    if (kind === 'image' && a.path) {
      thumb.src = '/image?path=' + encodeURIComponent(a.path);
      thumb.alt = a.name;
    } else {
      thumb.textContent = agentAttachIcon(kind);
    }
    const name = document.createElement('span');
    name.className = 'name';
    name.title = a.name;
    name.textContent = a.name;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = a.status === 'uploading' ? 'uploading…'
                     : a.status === 'error'     ? (a.error || 'failed')
                     : agentFmtBytes(a.size);
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'remove';
    x.title = 'Remove';
    x.textContent = '×';
    x.addEventListener('click', () => agentRemoveAttachment(a.id));
    chip.appendChild(thumb); chip.appendChild(name); chip.appendChild(meta); chip.appendChild(x);
    row.appendChild(chip);
  }
}

function agentRemoveAttachment(id) {
  window.AGENT.pendingAttachments =
    window.AGENT.pendingAttachments.filter(a => a.id !== id);
  agentRenderAttachRow();
  agentUpdateSendState();
}

function agentClearAttachments() {
  window.AGENT.pendingAttachments = [];
  agentRenderAttachRow();
  agentUpdateSendState();
}

async function agentUploadFile(att) {
  // POST one file to /upload. The backend already exists for the legacy
  // gallery picker; we reuse it. Returns { ok, path } on success.
  const fd = new FormData();
  fd.append('image', att._file, att.name);
  try {
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
    att.path = j.path;
    att.status = 'ready';
  } catch (e) {
    att.status = 'error';
    att.error  = String(e.message || e);
  } finally {
    delete att._file;                // free the File reference
    agentRenderAttachRow();
    agentUpdateSendState();
  }
}

function agentAddFiles(fileList) {
  if (!fileList || !fileList.length) return;
  for (const f of Array.from(fileList)) {
    if (f.size > AGENT_ATTACH_MAX_BYTES) {
      alert(`"${f.name}" is too large (max ${(AGENT_ATTACH_MAX_BYTES/1024/1024).toFixed(0)} MB)`);
      continue;
    }
    const att = {
      id: 'att_' + Math.random().toString(36).slice(2, 10),
      name: f.name,
      size: f.size,
      mime: f.type || '',
      path: '',
      status: 'uploading',
      _file: f,
    };
    window.AGENT.pendingAttachments.push(att);
    agentUploadFile(att);                 // fires async; UI updates on settle
  }
  agentRenderAttachRow();
  agentUpdateSendState();
}

function agentOnFilesPicked(ev) {
  agentAddFiles(ev.target.files);
  ev.target.value = '';                   // allow picking the same file again
}

function agentReadyAttachments() {
  return (window.AGENT.pendingAttachments || []).filter(a => a.status === 'ready' && a.path);
}

// ---- Drag-drop on the chat surface ---------------------------------------
function agentInstallDragDrop() {
  const pane = document.getElementById('agentPane');
  const overlay = document.getElementById('agentDropOverlay');
  if (!pane || !overlay) return;
  let dragDepth = 0;
  pane.addEventListener('dragenter', e => {
    if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes('Files')) return;
    e.preventDefault();
    dragDepth += 1;
    overlay.classList.add('visible');
  });
  pane.addEventListener('dragover', e => {
    if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  pane.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.remove('visible');
  });
  pane.addEventListener('drop', e => {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
    e.preventDefault();
    dragDepth = 0;
    overlay.classList.remove('visible');
    agentAddFiles(e.dataTransfer.files);
  });
}

// ---- Paste-from-clipboard on the textarea --------------------------------
function agentInstallPasteHandler() {
  const ta = document.getElementById('agentInput');
  if (!ta) return;
  ta.addEventListener('paste', e => {
    if (!e.clipboardData) return;
    const items = e.clipboardData.items || [];
    const files = [];
    for (const it of Array.from(items)) {
      if (it.kind === 'file') {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length) {
      e.preventDefault();
      agentAddFiles(files);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('agentInput');
  if (ta) {
    ta.addEventListener('input', () => {
      agentAutoResize(ta);
      agentUpdateSendState();
    });
    ta.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        agentSend();
      }
    });
    agentAutoResize(ta);
    agentUpdateSendState();
  }
  agentInstallDragDrop();
  agentInstallPasteHandler();
});

// ---- Engine settings drawer -----------------------------------------------
async function agentRefreshImageConfig() {
  try {
    const r = await fetch('/agent/image/config');
    const j = await r.json();
    window.AGENT.imageConfig = j;
    return j;
  } catch (e) {
    return null;
  }
}

function openAgentSettings() {
  Promise.all([agentRefreshConfig(), agentRefreshImageConfig()]).then(() => {
    const modal = document.getElementById('agentSettingsModal');
    if (!modal) return;
    const cfg = (window.AGENT.config && window.AGENT.config.engine) || {};
    const local = (window.AGENT.config && window.AGENT.config.local_server) || {};
    const imgCfg = (window.AGENT.imageConfig && window.AGENT.imageConfig.image_engine) || {};
    document.getElementById('agentKind').value = cfg.kind || 'phosphene_local';
    document.getElementById('agentBaseUrl').value = cfg.base_url || '';
    document.getElementById('agentRemoteModel').value = cfg.kind === 'custom' ? (cfg.model || '') : '';
    if ((cfg.kind || 'phosphene_local') === 'ollama') agentOllamaRefresh();
    document.getElementById('agentApiKey').value = '';
    document.getElementById('agentApiKey').placeholder =
      cfg.has_api_key ? '(saved key — leave blank to keep)' : 'Paste API key';
    document.getElementById('agentTemp').value = cfg.temperature ?? 0.4;
    document.getElementById('agentMaxTokens').value = cfg.max_tokens ?? 3072;

    // Image-engine fields
    document.getElementById('agentImageKind').value = imgCfg.kind || 'mock';
    // mflux
    const namedMfluxModels = ['krea-dev', 'dev', 'schnell'];
    const mfModel = imgCfg.mflux_model || 'krea-dev';
    if (namedMfluxModels.includes(mfModel)) {
      document.getElementById('agentMfluxModel').value = mfModel;
      document.getElementById('agentMfluxCustomPath').value = '';
    } else {
      document.getElementById('agentMfluxModel').value = '__custom__';
      document.getElementById('agentMfluxCustomPath').value = mfModel;
    }
    document.getElementById('agentMfluxBaseModel').value = imgCfg.mflux_base_model || 'krea-dev';
    document.getElementById('agentMfluxSteps').value = imgCfg.mflux_steps || 25;
    document.getElementById('agentMfluxQuantize').value = String(imgCfg.mflux_quantize || 4);
    // BFL
    document.getElementById('agentBflModel').value = imgCfg.bfl_model || 'flux-dev';
    document.getElementById('agentBflKey').value = '';
    document.getElementById('agentBflKey').placeholder =
      imgCfg.has_bfl_api_key ? '(saved key — leave blank to keep)' : 'Paste BFL API key';
    const imgPill = document.getElementById('agentImagePill');
    if (imgPill) {
      const okMsg = window.AGENT.imageConfig || {};
      imgPill.textContent = okMsg.ok === false ? 'needs config' : (imgCfg.kind || 'mock');
      imgPill.style.color = okMsg.ok === false ? '#f49a9e' : '#9be7a4';
      imgPill.style.borderColor = okMsg.ok === false ? 'rgba(207,34,46,0.5)' : 'rgba(46,160,67,0.5)';
      imgPill.title = okMsg.message || '';
    }
    agentImageKindChanged();

    const sel = document.getElementById('agentLocalModel');
    sel.innerHTML = '';
    const models = (window.AGENT.config && window.AGENT.config.available_models) || [];
    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No chat-capable models found in mlx_models/';
      sel.appendChild(opt);
    } else {
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.textContent = `${m.name} · ${m.size_gb} GB`;
        if ((cfg.local_model_path || '') === m.path) opt.selected = true;
        sel.appendChild(opt);
      }
    }
    agentKindChanged();
    agentLocalRefreshRow(local);
    modal.classList.add('open');
  });
}

function closeAgentSettings() {
  document.getElementById('agentSettingsModal').classList.remove('open');
}

// ---- Project notes modal --------------------------------------------------
async function openProjectNotes() {
  const modal = document.getElementById('agentNotesModal');
  const ta = document.getElementById('agentNotesTextarea');
  const st = document.getElementById('agentNotesStatus');
  if (!modal || !ta) return;
  st.textContent = 'Loading…';
  ta.value = '';
  modal.classList.add('open');
  try {
    const r = await fetch('/agent/notes');
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    ta.value = j.content || '';
    const n = (j.content || '').length;
    st.textContent = n ? `${n.toLocaleString()} chars` : 'Empty — type to start.';
  } catch (e) {
    st.textContent = 'Load failed: ' + e.message;
  }
  ta.focus();
}

function closeProjectNotes() {
  document.getElementById('agentNotesModal').classList.remove('open');
}

async function saveProjectNotes() {
  const ta = document.getElementById('agentNotesTextarea');
  const st = document.getElementById('agentNotesStatus');
  if (!ta) return;
  st.textContent = 'Saving…';
  try {
    const r = await fetch('/agent/notes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({content: ta.value}),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    st.textContent = `Saved · ${(j.bytes || 0).toLocaleString()} bytes`;
    setTimeout(closeProjectNotes, 350);
  } catch (e) {
    st.textContent = 'Save failed: ' + e.message;
  }
}

function agentKindChanged() {
  const kind = document.getElementById('agentKind').value;
  document.getElementById('agentLocalModelField').style.display = kind === 'phosphene_local' ? '' : 'none';
  document.getElementById('agentLocalRow').style.display = kind === 'phosphene_local' ? '' : 'none';
  document.getElementById('agentOllamaField').style.display = kind === 'ollama' ? '' : 'none';
  document.getElementById('agentBaseUrlField').style.display = kind === 'custom' ? '' : 'none';
  document.getElementById('agentApiKeyField').style.display = kind === 'custom' ? '' : 'none';
  document.getElementById('agentRemoteModelField').style.display = kind === 'custom' ? '' : 'none';
  if (kind === 'ollama') agentOllamaRefresh();
}

async function agentOllamaRefresh() {
  const sel = document.getElementById('agentOllamaModel');
  const hint = document.getElementById('agentOllamaHint');
  if (!sel) return;
  sel.innerHTML = '<option>Probing Ollama at 127.0.0.1:11434…</option>';
  try {
    const r = await fetch('/agent/ollama/status');
    const j = await r.json();
    sel.innerHTML = '';
    if (!j.running) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Ollama not running on 127.0.0.1:11434';
      sel.appendChild(opt);
      if (hint) hint.innerHTML = `<strong>Ollama is not running.</strong> Start it with <code>ollama serve</code>, then click Refresh. Install models with <code>ollama pull qwen2.5-coder:32b</code>.`;
      return;
    }
    const cfg = (window.AGENT.config && window.AGENT.config.engine) || {};
    if ((j.models || []).length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No Ollama models installed';
      sel.appendChild(opt);
    }
    for (const m of (j.models || [])) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name}${m.size_gb ? ' · ' + m.size_gb + ' GB' : ''}${m.parameter_size ? ' · ' + m.parameter_size : ''}${m.quantization ? ' · ' + m.quantization : ''}`;
      if (cfg.kind === 'ollama' && cfg.model === m.name) opt.selected = true;
      sel.appendChild(opt);
    }
    if (hint) hint.innerHTML = `Talks to <code>${escapeHtml(j.openai_url || j.base_url + '/v1')}/chat/completions</code>. Tool calling works on models whose Modelfile declares it (Qwen3 Coder, Llama 3.x, Devstral, Mistral, Granite). Run <code>ollama show &lt;model&gt;</code> to verify.`;
  } catch(e) {
    sel.innerHTML = '<option>Failed to probe — see console</option>';
  }
}

function agentImageKindChanged() {
  const kind = document.getElementById('agentImageKind').value;
  // mflux fields
  const isMflux = kind === 'mflux';
  document.getElementById('agentMfluxModelField').style.display = isMflux ? '' : 'none';
  document.getElementById('agentMfluxParamsField').style.display = isMflux ? '' : 'none';
  document.getElementById('agentMfluxInstallHint').style.display = isMflux ? '' : 'none';
  if (isMflux) agentMfluxModelChanged();
  else {
    document.getElementById('agentMfluxCustomField').style.display = 'none';
    document.getElementById('agentMfluxBaseField').style.display = 'none';
  }
  // BFL fields
  document.getElementById('agentBflModelField').style.display = kind === 'bfl' ? '' : 'none';
  document.getElementById('agentBflKeyField').style.display = kind === 'bfl' ? '' : 'none';
}

function agentMfluxModelChanged() {
  const v = document.getElementById('agentMfluxModel').value;
  const isCustom = v === '__custom__';
  document.getElementById('agentMfluxCustomField').style.display = isCustom ? '' : 'none';
  document.getElementById('agentMfluxBaseField').style.display = isCustom ? '' : 'none';
  // For schnell, drop the recommended steps to 4
  const stepsInput = document.getElementById('agentMfluxSteps');
  if (stepsInput && !stepsInput.dataset.userTouched) {
    if (v === 'schnell') stepsInput.value = 4;
    else if (v === 'krea-dev' || v === 'dev') stepsInput.value = 25;
  }
}

function agentLocalRefreshRow(local) {
  const pill = document.getElementById('agentLocalPill');
  const detail = document.getElementById('agentLocalDetail');
  const btn = document.getElementById('agentLocalToggleBtn');
  if (!pill || !detail || !btn) return;
  if (local.running) {
    pill.textContent = 'live';
    pill.classList.add('live'); pill.classList.remove('bad');
    detail.textContent = `mlx-lm.server pid ${local.pid} on :${local.port}`;
    btn.textContent = 'Stop';
  } else {
    pill.textContent = 'stopped';
    pill.classList.remove('live');
    if (local.last_error) pill.classList.add('bad');
    detail.textContent = local.last_error || 'mlx-lm.server (will spawn on Start)';
    btn.textContent = 'Start';
  }
}

async function agentLocalToggle() {
  const local = (window.AGENT.config && window.AGENT.config.local_server) || {};
  if (local.running) {
    await fetch('/agent/local/stop', {method: 'POST'});
  } else {
    const sel = document.getElementById('agentLocalModel');
    const modelPath = sel ? sel.value : '';
    await fetch('/agent/local/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_path: modelPath || undefined}),
    });
  }
  await new Promise(r => setTimeout(r, 300));
  await agentRefreshConfig();
  const local2 = (window.AGENT.config && window.AGENT.config.local_server) || {};
  agentLocalRefreshRow(local2);
}

async function agentSaveSettings() {
  const kind = document.getElementById('agentKind').value;
  const payload = {
    kind,
    temperature: parseFloat(document.getElementById('agentTemp').value || '0.4'),
    max_tokens: parseInt(document.getElementById('agentMaxTokens').value || '3072', 10),
  };
  if (kind === 'phosphene_local') {
    const sel = document.getElementById('agentLocalModel');
    if (sel && sel.value) {
      payload.local_model_path = sel.value;
      const parts = sel.value.split('/');
      payload.model = parts[parts.length - 1] || sel.value;
    }
  } else if (kind === 'ollama') {
    // Ollama bridge: same OpenAI-compat shape, just talks to 127.0.0.1:11434/v1.
    // No api_key. The model field is the Ollama tag (e.g. "qwen2.5-coder:32b").
    const sel = document.getElementById('agentOllamaModel');
    payload.base_url = 'http://127.0.0.1:11434/v1';
    payload.model = sel ? sel.value : '';
    payload.local_model_path = '';
  } else {
    payload.base_url = (document.getElementById('agentBaseUrl').value || '').trim();
    payload.model = (document.getElementById('agentRemoteModel').value || '').trim();
    const ak = (document.getElementById('agentApiKey').value || '').trim();
    if (ak) payload.api_key = ak;
  }
  const r = await fetch('/agent/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok || j.error) {
    alert('Could not save: ' + (j.error || ('HTTP ' + r.status)));
    return;
  }

  // Save image-engine config too (separate file).
  const imgKind = document.getElementById('agentImageKind').value;
  const imgPayload = {
    kind: imgKind,
    bfl_model: document.getElementById('agentBflModel').value,
  };
  const bk = (document.getElementById('agentBflKey').value || '').trim();
  if (bk) imgPayload.bfl_api_key = bk;
  // mflux fields (only meaningful when kind === 'mflux', but we save them
  // either way so the form retains the user's previous setup when they
  // toggle backends back and forth).
  const mfSel = document.getElementById('agentMfluxModel').value;
  if (mfSel === '__custom__') {
    const cp = (document.getElementById('agentMfluxCustomPath').value || '').trim();
    if (cp) {
      imgPayload.mflux_model = cp;
      imgPayload.mflux_base_model = document.getElementById('agentMfluxBaseModel').value;
    }
  } else {
    imgPayload.mflux_model = mfSel;
    imgPayload.mflux_base_model = '';
  }
  imgPayload.mflux_steps = parseInt(document.getElementById('agentMfluxSteps').value || '25', 10);
  imgPayload.mflux_quantize = parseInt(document.getElementById('agentMfluxQuantize').value || '4', 10);
  try {
    const ir = await fetch('/agent/image/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(imgPayload),
    });
    const ij = await ir.json();
    if (!ir.ok || ij.error) {
      alert('Image config could not be saved: ' + (ij.error || ('HTTP ' + ir.status)));
    }
  } catch(e) {
    alert('Image config save failed: ' + e);
  }

  closeAgentSettings();
  await agentRefreshConfig();
  await agentRefreshImageConfig();
}

// ============================================================================
// HF MODEL BROWSER — search + install MLX chat models
// ============================================================================
window.MODEL_BROWSER = {
  pollerId: null,
  lastResults: [],
};

function openModelBrowser() {
  const m = document.getElementById('modelBrowserModal');
  if (!m) return;
  m.classList.add('open');
  // Auto-focus the query input on open.
  setTimeout(() => {
    const q = document.getElementById('modelBrowserQuery');
    if (q) q.focus();
  }, 50);
  // If we already have results from a previous search, leave them; else
  // run a default search ("qwen") so the user sees something useful.
  if (!window.MODEL_BROWSER.lastResults.length) {
    document.getElementById('modelBrowserQuery').value = 'qwen';
    modelBrowserSearch();
  }
  // Start polling install status in case a download is already in flight.
  modelBrowserStartPolling();
}

function closeModelBrowser() {
  document.getElementById('modelBrowserModal').classList.remove('open');
  modelBrowserStopPolling();
}

async function modelBrowserSearch() {
  const q = (document.getElementById('modelBrowserQuery').value || '').trim();
  const abliterated = document.getElementById('modelBrowserAbliterated').checked;
  const results = document.getElementById('modelBrowserResults');
  const btn = document.getElementById('modelBrowserSearchBtn');
  results.innerHTML = '<div class="model-browser-empty">Searching…</div>';
  btn.disabled = true;
  try {
    const url = '/agent/models/search?q=' + encodeURIComponent(q)
              + '&abliterated=' + (abliterated ? '1' : '0')
              + '&limit=40';
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok || j.error) {
      results.innerHTML = '<div class="model-browser-empty" style="color:#f49a9e">Error: ' + escapeHtml(j.error || ('HTTP ' + r.status)) + '</div>';
      return;
    }
    window.MODEL_BROWSER.lastResults = j.results || [];
    modelBrowserRender(j.results || []);
  } catch(e) {
    results.innerHTML = '<div class="model-browser-empty" style="color:#f49a9e">Error: ' + escapeHtml(String(e)) + '</div>';
  } finally {
    btn.disabled = false;
  }
}

function modelBrowserRender(results) {
  const wrap = document.getElementById('modelBrowserResults');
  if (!results.length) {
    wrap.innerHTML = '<div class="model-browser-empty">No matches. Try a different query.</div>';
    return;
  }
  wrap.innerHTML = '';
  for (const m of results) {
    const row = document.createElement('div');
    row.className = 'model-result';
    const isAbliterated = (m.repo_id || '').toLowerCase().includes('abliterated')
                       || (m.repo_id || '').toLowerCase().startsWith('huihui-ai/');
    const dl = (m.downloads || 0).toLocaleString();
    const lk = (m.likes || 0).toLocaleString();
    const tags = [];
    if (m.gated) tags.push('<span class="tag gated">gated</span>');
    if (isAbliterated) tags.push('<span class="tag abliterated">abliterated</span>');
    if (m.library_name) tags.push(`<span class="tag">${escapeHtml(m.library_name)}</span>`);
    if (m.pipeline_tag) tags.push(`<span class="tag">${escapeHtml(m.pipeline_tag)}</span>`);
    row.innerHTML = `
      <div class="info">
        <div class="name">${escapeHtml(m.repo_id)}</div>
        <div class="meta">
          <span>↓ ${dl}</span>
          <span>♥ ${lk}</span>
          ${tags.join('')}
        </div>
      </div>
      <div class="actions">
        <button class="info-btn" onclick="modelBrowserInfo('${escapeHtml(m.repo_id)}')">Info</button>
        <button class="install-btn" onclick="modelBrowserInstall('${escapeHtml(m.repo_id)}', this)">Install</button>
      </div>
    `;
    wrap.appendChild(row);
  }
}

async function modelBrowserInfo(repoId) {
  // Lightweight inline info — pop a confirm with size + file count + gated state.
  try {
    const r = await fetch('/agent/models/info?repo_id=' + encodeURIComponent(repoId));
    const j = await r.json();
    if (j.gated || j.error) {
      alert('Repo info:\n' + (j.error || ('Gated. Open https://huggingface.co/' + repoId + ' and accept the terms first.')));
      return;
    }
    alert(`${repoId}\n\nFiles: ${j.file_count}\nTotal: ${j.total_size_gb} GB\n\nClick Install to download into mlx_models/.`);
  } catch(e) {
    alert('Could not fetch info: ' + e);
  }
}

async function modelBrowserInstall(repoId, btn) {
  if (!confirm(`Install ${repoId}?\n\nDownloads to mlx_models/. Files are large — first run can take 5-30 min depending on size and network.`)) {
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Queuing…';
  try {
    const r = await fetch('/agent/models/install', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({repo_id: repoId}),
    });
    const j = await r.json();
    if (!r.ok || j.error) {
      alert('Install failed to start: ' + (j.error || ('HTTP ' + r.status)));
      btn.disabled = false;
      btn.textContent = 'Install';
      return;
    }
    btn.textContent = 'Downloading…';
    modelBrowserStartPolling();
  } catch(e) {
    alert('Install error: ' + e);
    btn.disabled = false;
    btn.textContent = 'Install';
  }
}

function modelBrowserStartPolling() {
  modelBrowserStopPolling();
  modelBrowserPollOnce();
  window.MODEL_BROWSER.pollerId = setInterval(modelBrowserPollOnce, 1500);
}

function modelBrowserStopPolling() {
  if (window.MODEL_BROWSER.pollerId) {
    clearInterval(window.MODEL_BROWSER.pollerId);
    window.MODEL_BROWSER.pollerId = null;
  }
}

async function modelBrowserPollOnce() {
  try {
    const r = await fetch('/agent/models/install/status');
    const j = await r.json();
    const status = document.getElementById('modelBrowserStatus');
    const lbl = document.getElementById('modelBrowserStatusLabel');
    const summ = document.getElementById('modelBrowserStatusSummary');
    const line = document.getElementById('modelBrowserStatusLine');
    if (!status || !lbl) return;
    if (j.active) {
      status.classList.add('visible');
      lbl.textContent = 'Downloading';
      lbl.style.color = '';
      const elapsed = j.elapsed_s ? ` · ${Math.round(j.elapsed_s)}s` : '';
      summ.textContent = `${j.repo_id || ''}${elapsed}`;
      line.textContent = j.last_line || '';
    } else if (j.done) {
      status.classList.add('visible');
      lbl.textContent = 'Installed';
      lbl.style.color = '#9be7a4';
      summ.textContent = `${j.repo_id || ''} → ${j.target_dir || ''}`;
      line.textContent = '';
      modelBrowserStopPolling();
      // Refresh the local-model picker in the background so the user
      // sees the new model on next settings-modal open.
      try { agentRefreshConfig(); } catch(e) {}
    } else if (j.error) {
      status.classList.add('visible');
      lbl.textContent = 'Failed';
      lbl.style.color = '#f49a9e';
      summ.textContent = j.repo_id || '';
      line.textContent = j.error;
      modelBrowserStopPolling();
    } else {
      status.classList.remove('visible');
      modelBrowserStopPolling();
    }
  } catch(e) {
    /* swallow */
  }
}

// Esc closes the model browser, before falling through to settings/fullscreen.
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const m = document.getElementById('modelBrowserModal');
    if (m && m.classList.contains('open')) {
      closeModelBrowser();
      e.stopPropagation();
    }
  }
}, true);

// Cmd/Ctrl+Enter in the search box runs the search.
document.addEventListener('DOMContentLoaded', () => {
  const q = document.getElementById('modelBrowserQuery');
  if (q) {
    q.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); modelBrowserSearch(); }
    });
  }
});

// ============================================================================
// AGENT STAGE PANE — live canvas on the right side
// ============================================================================
// Shows the agent's video work as it happens: currently rendering job,
// session outputs (with click-to-play lightbox), and a recent-activity
// feed of tool calls. Polls /status + the active session every ~1.5 s
// while the agent workflow is selected.

window.AGENT_STAGE = {
  pollerId: null,
  lastSessionShotIds: [],   // submitted_shots from session.tool_state, ordered
  lastEventCount: 0,        // for animating new entries in the activity feed
};

function agentStageStart() {
  agentStageStop();
  // Tick once immediately, then every 1500 ms.
  agentStageTick();
  window.AGENT_STAGE.pollerId = setInterval(agentStageTick, 1500);
}

function agentStageStop() {
  if (window.AGENT_STAGE.pollerId) {
    clearInterval(window.AGENT_STAGE.pollerId);
    window.AGENT_STAGE.pollerId = null;
  }
}

async function agentStageTick() {
  const pane = document.querySelector('.agent-stage-pane');
  // Don't bother polling when the pane is hidden.
  if (!pane || getComputedStyle(pane).display === 'none') return;
  try {
    const [statusResp, sessResp] = await Promise.all([
      fetch('/status').then(r => r.ok ? r.json() : null).catch(() => null),
      window.AGENT.sessionId
        ? fetch('/agent/sessions/' + encodeURIComponent(window.AGENT.sessionId))
            .then(r => r.ok ? r.json() : null).catch(() => null)
        : null,
    ]);
    agentStageRender(statusResp, sessResp);
    agentRenderRamChip(statusResp);
  } catch(e) {
    /* swallow — next tick retries */
  }
}

// RAM headroom chip in the agent header. The single answer to
// "can I send a message right now without putting the Mac in swap?"
// — colored green when comfortable, amber when tight, red when not.
function agentRenderRamChip(status) {
  const chip = document.getElementById('agentRamChip');
  const label = document.getElementById('agentRamLabel');
  if (!chip || !label || !status) return;
  const m = status.memory || {};
  const total = Number(m.total_gb) || 0;
  const used = Number(m.used_gb) || 0;
  const free = Math.max(0, total - used);
  // Pull the configured chat model's resident size if we know it. The
  // /agent/config response carries `available_models[*].size_gb`; match
  // by `path` for local engines.
  let modelGb = null;
  let modelName = '';
  let isLocal = false;
  let isLoaded = false;
  try {
    const cfg = window.AGENT && window.AGENT.config;
    const eng = cfg && cfg.engine;
    if (eng && eng.kind === 'phosphene_local') {
      isLocal = true;
      isLoaded = !!(cfg.local_server && cfg.local_server.running);
      const path = eng.local_model_path || '';
      const list = cfg.available_models || [];
      const hit = list.find(m => m.path === path);
      if (hit && hit.size_gb) {
        modelGb = Number(hit.size_gb);
        modelName = hit.name || '';
      }
    }
  } catch (e) {}

  // If the chat model is already resident, its weights are already in
  // `used` — the question is simply "is there room left to keep going?"
  // If it's NOT loaded yet, we need to subtract its expected size from
  // `free` to predict whether spawning would push us into swap.
  const swapGb = Number(m.swap_gb) || 0;
  let cls = 'is-roomy';
  let text = `${free.toFixed(0)} GB free`;
  let tip = `Total ${total.toFixed(0)} GB · Used ${used.toFixed(1)} GB · Free ${free.toFixed(1)} GB`;
  if (swapGb >= 1) tip += ` · Swap ${swapGb.toFixed(1)} GB`;

  if (isLocal && modelGb && !isLoaded) {
    const after = free - modelGb;
    tip += `\n${modelName || 'Selected model'}: ~${modelGb.toFixed(1)} GB`;
    tip += `\nIf started: ${after.toFixed(1)} GB would remain for renders`;
    if (after < 2 || swapGb >= 8) {
      cls = 'is-bad';
      text = `Tight · ${free.toFixed(0)} GB free`;
    } else if (after < 8) {
      cls = 'is-tight';
      text = `${free.toFixed(0)} GB free · model fits`;
    } else {
      cls = 'is-roomy';
      text = `${free.toFixed(0)} GB free · plenty`;
    }
  } else if (isLocal && isLoaded) {
    tip += `\nChat model loaded${modelName ? ' (' + modelName + ', ~' + (modelGb||0).toFixed(1) + ' GB)' : ''}`;
    if (free < 4 || swapGb >= 8) { cls = 'is-bad'; text = `Tight · ${free.toFixed(0)} GB free`; }
    else if (free < 12)            { cls = 'is-tight'; text = `${free.toFixed(0)} GB free`; }
    else                            { cls = 'is-roomy'; text = `${free.toFixed(0)} GB free`; }
  } else {
    // Remote engine — the chat model isn't on this Mac. Just report
    // raw headroom for the renderer's benefit.
    if (free < 4 || swapGb >= 8) { cls = 'is-bad'; text = `Tight · ${free.toFixed(0)} GB free`; }
    else if (free < 12)            { cls = 'is-tight'; text = `${free.toFixed(0)} GB free`; }
    else                            { cls = 'is-roomy'; text = `${free.toFixed(0)} GB free`; }
  }
  chip.className = 'agent-ram-chip ' + cls;
  chip.title = tip;
  label.textContent = text;
}

function agentStageRender(status, sess) {
  const dot = document.getElementById('agentStageDot');
  const sessionPill = document.getElementById('agentStageSession');
  const nowEl = document.getElementById('agentStageNow');
  const outputsEl = document.getElementById('agentStageOutputs');
  const outputsCountEl = document.getElementById('agentStageOutputsCount');
  const activityEl = document.getElementById('agentStageActivity');
  const activityCountEl = document.getElementById('agentStageActivityCount');
  if (!dot || !nowEl || !outputsEl || !activityEl) return;

  const running = !!(status && status.running);
  dot.classList.toggle('live', running);
  if (sessionPill) {
    if (window.AGENT.sessionId) {
      sessionPill.textContent = window.AGENT.sessionId.slice(0, 10);
    } else {
      sessionPill.textContent = 'no session';
    }
  }

  // Now rendering. /status returns `current.progress` as an object now:
  //   { phase, phase_label, pct (0-100), elapsed_sec, eta_sec, denoise_step, ... }
  // The legacy code path here was calling Number(progressObject) which yields
  // NaN, so the bar always read 0% with phase "rendering" — making the user
  // think nothing was happening even mid-batch. Read the structured fields
  // and fall back gracefully if a future status format ships flat numbers.
  const cur = status && status.current;
  if (running && cur) {
    const p = cur.params || {};
    const label = p.label || (p.preset_label || cur.id || 'render');
    const prog = cur.progress;
    let pct = 0;
    let phase = (cur.status === 'running' ? 'rendering' : '');
    let eta = cur.eta_seconds || cur.eta || null;
    let elapsed = null;
    let denoiseStep = null, denoiseTotal = null;
    if (prog && typeof prog === 'object') {
      pct = Math.max(0, Math.min(100, Number(prog.pct) || 0));
      phase = prog.phase_label || prog.phase || phase;
      eta = prog.eta_sec || prog.remaining_sec || eta;
      elapsed = prog.elapsed_sec || null;
      denoiseStep = prog.denoise_step;
      denoiseTotal = prog.denoise_total;
    } else {
      pct = Math.max(0, Math.min(1, Number(prog) || 0)) * 100;
    }
    // Append step counter when in the denoise phase so progress feels
    // alive even when pct ticks slowly.
    let phaseDetail = phase || 'rendering';
    if (denoiseStep && denoiseTotal) {
      phaseDetail += ` · step ${denoiseStep}/${denoiseTotal}`;
    } else if (elapsed && elapsed >= 5) {
      phaseDetail += ` · ${agentFmtDur(elapsed)} in`;
    }
    nowEl.classList.remove('idle');
    nowEl.innerHTML = `
      <div class="stage-now-label">${escapeHtml(label)}</div>
      <div class="stage-now-meta">${escapeHtml(p.mode || 't2v')} · ${escapeHtml(p.quality || 'balanced')} · ${escapeHtml(p.frames || '?')}f</div>
      <div class="stage-progress-bar">
        <div class="stage-progress-fill" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div class="stage-progress-text">
        <span>${escapeHtml(phaseDetail)}</span>
        <span>${pct.toFixed(0)}%${eta ? ' · ETA ' + agentFmtDur(eta) : ''}</span>
      </div>
    `;
  } else {
    nowEl.className = 'stage-now-card idle';
    nowEl.innerHTML = `<div>Idle. Ask the agent to plan a shot to see it render here.</div>`;
  }

  // Session outputs — pull from session.tool_state.submitted_shots (ordered)
  // and look up each in /status.history for current state + output_path.
  const tool = (sess && sess.session && sess.session.tool_state) || {};
  const submitted = tool.submitted_shots || [];
  const allJobs = []
    .concat(status && status.queue ? status.queue : [])
    .concat(cur ? [cur] : [])
    .concat(status && status.history ? status.history : []);
  const byId = new Map(allJobs.map(j => [j.id, j]));
  const outputs = submitted.map(s => {
    const j = byId.get(s.job_id) || s;
    const p = (j.params || {});
    return {
      id: s.job_id,
      label: s.label || p.label || s.job_id,
      status: j.status || 'unknown',
      output_path: j.output_path || null,
      mode: p.mode || s.mode || 't2v',
      duration: s.duration_seconds || null,
    };
  });
  outputsCountEl.textContent = outputs.length;
  if (outputs.length === 0) {
    outputsEl.innerHTML = `<div class="stage-empty">No mp4s rendered yet. Submit a shot from the chat.</div>`;
  } else {
    outputsEl.innerHTML = '';
    for (const o of outputs.slice(0, 24)) {
      const cell = document.createElement('div');
      const failed = o.status === 'error' || o.status === 'failed' || o.status === 'cancelled';
      cell.className = 'stage-output-cell' + (failed ? ' failed' : '');
      cell.title = o.label + ' · ' + o.status;
      // If we have a finished output_path, show the video as a thumbnail.
      // Otherwise show a status badge.
      if (o.output_path && o.status === 'done') {
        const v = document.createElement('video');
        v.className = 'vid';
        v.src = '/file?path=' + encodeURIComponent(o.output_path);
        v.preload = 'metadata';
        v.muted = true;
        cell.appendChild(v);
        cell.addEventListener('click', () => agentStageLightboxOpen(o.output_path, o.label, o.id));
        // Refine button (overlay top-right): "give me a variation of this clip"
        const refine = document.createElement('button');
        refine.type = 'button';
        refine.className = 'refine-btn';
        refine.title = 'Refine this clip — start a variation in the chat';
        refine.textContent = '↻';
        refine.addEventListener('click', (e) => {
          e.stopPropagation();
          agentSetRefine({jobId: o.id, label: o.label, clipPath: o.output_path});
        });
        cell.appendChild(refine);
      } else {
        cell.style.display = 'flex';
        cell.style.alignItems = 'center';
        cell.style.justifyContent = 'center';
        const span = document.createElement('span');
        span.style.cssText = 'color:var(--muted); font-size:11px; font-style:italic;';
        span.textContent = failed ? 'failed' : (o.status === 'running' ? 'rendering…' : 'queued');
        cell.appendChild(span);
      }
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = failed ? 'fail' : (o.status === 'done' ? '✓' : o.status.slice(0, 4));
      cell.appendChild(badge);
      const lbl = document.createElement('span');
      lbl.className = 'label';
      lbl.textContent = o.label;
      cell.appendChild(lbl);
      outputsEl.appendChild(cell);
    }
  }

  // Activity feed — derive from rendered_messages: each tool_call/tool_result
  // becomes one row with an icon. Newest at the top.
  const rendered = (sess && sess.rendered_messages) || [];
  const events = [];
  for (let i = rendered.length - 1; i >= 0 && events.length < 40; i--) {
    const m = rendered[i];
    if (m.kind === 'tool_result') {
      const r = m.result || {};
      const ok = r.ok !== false && !r.error;
      const inner = r.result || {};
      let txt;
      if (typeof inner === 'object' && 'job_id' in inner) {
        txt = `→ queued ${inner.job_id} · ${inner.estimated_wall_human || '?'}`;
      } else if (typeof inner === 'object' && 'manifest_path' in inner) {
        txt = `→ manifest written (${inner.shot_count} shots)`;
      } else if (typeof inner === 'object' && 'png_path' in inner) {
        txt = `→ frame extracted (${inner.frame_index})`;
      } else if (typeof inner === 'object' && 'candidates' in inner) {
        txt = `→ ${inner.candidates.length} candidates ready`;
      } else if (!ok) {
        txt = `✗ ${(r.error || 'failed').slice(0, 80)}`;
      } else {
        txt = '→ result';
      }
      events.push({ kind: ok ? 'ok' : 'fail', text: txt });
    } else if (m.kind === 'assistant' && m.tool_call) {
      events.push({ kind: 'run', text: '⚙ ' + m.tool_call.tool });
    }
  }
  activityCountEl.textContent = events.length;
  if (events.length === 0) {
    activityEl.innerHTML = `<div class="stage-empty">No tool calls yet.</div>`;
  } else {
    activityEl.innerHTML = '';
    for (const ev of events) {
      const row = document.createElement('div');
      row.className = 'stage-activity-row ' + ev.kind;
      const icon = (ev.kind === 'ok') ? '✓' : (ev.kind === 'fail') ? '✗' : '⚙';
      row.innerHTML = `
        <span class="icon">${icon}</span>
        <span class="text">${escapeHtml(ev.text)}</span>
      `;
      activityEl.appendChild(row);
    }
  }
}

function agentFmtDur(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60), rem = s % 60;
  if (m < 60) return m + 'm ' + (rem < 10 ? '0' : '') + rem + 's';
  const h = Math.floor(m / 60), mr = m % 60;
  return h + 'h ' + (mr < 10 ? '0' : '') + mr + 'm';
}

// Track which clip is currently in the lightbox so the Refine button
// has something to reference when clicked.
window.AGENT_STAGE.lightboxCurrent = null;

function agentStageLightboxOpen(path, label, jobId) {
  const lb = document.getElementById('agentStageLightbox');
  const v = document.getElementById('agentStageLightboxVideo');
  if (!lb || !v) return;
  v.src = '/file?path=' + encodeURIComponent(path);
  v.title = label || '';
  window.AGENT_STAGE.lightboxCurrent = {jobId: jobId || null, label: label || '', clipPath: path};
  lb.classList.add('open');
  v.play().catch(() => {});
}

function agentStageLightboxClose() {
  const lb = document.getElementById('agentStageLightbox');
  const v = document.getElementById('agentStageLightboxVideo');
  if (!lb || !v) return;
  v.pause();
  v.src = '';
  window.AGENT_STAGE.lightboxCurrent = null;
  lb.classList.remove('open');
}

function agentStageLightboxRefine() {
  const cur = window.AGENT_STAGE.lightboxCurrent;
  if (!cur) return;
  agentStageLightboxClose();
  agentSetRefine(cur);
}

// ---- Composer reference chip — "Refine this clip" -----------------------
// When the user clicks ↻ on a stage output (or the Refine button in the
// lightbox), we set a refine reference. The chip shows above the textarea;
// on next Send, the user's message is prepended with "Refine <job_id>: "
// so the agent calls inspect_clip and treats the rest as the requested
// modification. Clear with × on the chip.
window.AGENT_REFINE = null;          // {jobId, label, clipPath}

function agentSetRefine(ref) {
  window.AGENT_REFINE = ref;
  const chip = document.getElementById('agentRefChip');
  const lbl = document.getElementById('agentRefChipLabel');
  if (!chip || !lbl) return;
  lbl.textContent = ref.label || ref.jobId || ref.clipPath || 'clip';
  lbl.title = ref.jobId ? `${ref.label || ''} · ${ref.jobId}` : ref.clipPath;
  chip.classList.add('visible');
  // Bring focus to the composer so the user can type their refinement.
  const ta = document.getElementById('agentInput');
  if (ta) {
    ta.placeholder = 'How should this clip be different? (e.g. "more pause", "warmer light", "longer take")';
    setTimeout(() => ta.focus(), 50);
  }
}

function agentClearRefine() {
  window.AGENT_REFINE = null;
  const chip = document.getElementById('agentRefChip');
  if (chip) chip.classList.remove('visible');
  const ta = document.getElementById('agentInput');
  if (ta) ta.placeholder = 'Paste a script, or describe a piece. The agent will plan, estimate the wall time, and queue overnight.';
}

// Esc closes the stage lightbox first (before falling through to the
// settings modal / fullscreen exit handlers).
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const lb = document.getElementById('agentStageLightbox');
    if (lb && lb.classList.contains('open')) {
      agentStageLightboxClose();
      e.stopPropagation();
    }
  }
}, true);

// ---- Pop out to system browser -------------------------------------------
// The pop-out is a real <a target="_blank"> link (set up during boot via
// agentInitPopOut). The browser handles the actual navigation as a user
// gesture so popup-blockers don't kick in. This handler just sets the
// localStorage flags so the new tab boots straight into Agentic Flows +
// fullscreen.
function agentPopOutFlagsBeforeNavigate() {
  try {
    localStorage.setItem('phos_workflow', 'agent');
    localStorage.setItem('phos_agent_fullscreen', '1');
  } catch(e) {}
  // The <a href> handles navigation. Don't preventDefault.
}

function agentInitPopOut() {
  const a = document.getElementById('agentPopOutBtn');
  if (!a) return;
  a.href = window.location.origin || ('http://127.0.0.1:' + window.location.port);
}
// Run on every page load; cheap and idempotent.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', agentInitPopOut);
} else {
  agentInitPopOut();
}

// ---- Fullscreen / focus mode ---------------------------------------------
function agentToggleFullscreen(force) {
  // `force === true` to enter, `false` to exit, undefined to toggle
  const cur = document.body.classList.contains('agent-fullscreen');
  const next = (typeof force === 'boolean') ? force : !cur;
  document.body.classList.toggle('agent-fullscreen', next);
  const btn = document.getElementById('agentFullscreenBtn');
  if (btn) btn.title = next ? 'Exit fullscreen (Esc)' : 'Expand to fullscreen';
  const ix = document.getElementById('agentFullscreenIconExpand');
  const iy = document.getElementById('agentFullscreenIconCollapse');
  if (ix) ix.style.display = next ? 'none' : '';
  if (iy) iy.style.display = next ? '' : 'none';
  try { localStorage.setItem('phos_agent_fullscreen', next ? '1' : ''); } catch(e) {}
  // After collapse animation settles, scroll chat to bottom so the
  // user keeps their place.
  requestAnimationFrame(() => {
    const chat = document.getElementById('agentChat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  });
}

// Esc handler: priority is modal-close > fullscreen-exit. So Esc to
// dismiss the settings drawer doesn't drop the user out of fullscreen
// at the same time.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  const modal = document.getElementById('agentSettingsModal');
  if (modal && modal.classList.contains('open')) {
    closeAgentSettings();
    return;
  }
  if (document.body.classList.contains('agent-fullscreen')) {
    agentToggleFullscreen(false);
  }
});

// Initial workflow tab restore from localStorage.
try {
  const saved = localStorage.getItem('phos_workflow');
  if (saved === 'agent') workflowSwitch('agent');
  // Restore fullscreen state, but ONLY when the agent tab is the
  // active workflow — otherwise we'd hide the manual form too.
  if (localStorage.getItem('phos_agent_fullscreen') && saved === 'agent') {
    agentToggleFullscreen(true);
  }
} catch(e) {}
