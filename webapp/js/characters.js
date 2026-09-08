// webapp/js/characters.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ====== Audio → Video pane ===================================================
// State for the Audio workflow tab. Routes to make_job with mode='a2v'.
// One drop-zone for audio (required) is owned here. The optional
// reference image was migrated to the shared .picker component
// (key='a2v_image') 2026-05-20 — its state lives in the hidden input
// #a2v_image, wired automatically via PICKERS.forEach(pickerWire) at
// boot, so this struct no longer carries an imagePath field.
const AUDIO_STUDIO = {
  busy: false,
  audioPath: null,      // server-side path returned by /upload
  audioName: null,      // display name for the drop-zone tag
  audioDuration: null,  // probed duration (seconds) if ffprobe returns it
  wired: false,         // init() runs once
};

function audioStudioInit() {
  if (AUDIO_STUDIO.wired) return;
  AUDIO_STUDIO.wired = true;
  const audioSlot = document.getElementById('audioStudioAudioSlot');
  const audioInput = document.getElementById('audioStudioAudioInput');

  // Audio drop-zone wiring — click to pick, drag-and-drop, paste.
  // (Image drop-zone wiring lives in pickerWire('a2v_image'), called
  // once at module boot from PICKERS.forEach(pickerWire).)
  if (audioSlot && audioInput) {
    audioSlot.addEventListener('click', () => audioInput.click());
    audioSlot.addEventListener('dragover', (e) => {
      e.preventDefault();
      audioSlot.classList.add('drop-active');
    });
    audioSlot.addEventListener('dragleave', () => audioSlot.classList.remove('drop-active'));
    audioSlot.addEventListener('drop', (e) => {
      e.preventDefault();
      audioSlot.classList.remove('drop-active');
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        audioStudioUploadAudio(e.dataTransfer.files[0]);
      }
    });
    audioInput.addEventListener('change', () => {
      if (audioInput.files && audioInput.files[0]) {
        audioStudioUploadAudio(audioInput.files[0]);
      }
    });
  }
  audioStudioRenderSlots();
}

async function audioStudioUploadAudio(file) {
  const status = document.getElementById('audioStudioStatus');
  if (status) status.textContent = 'Uploading audio…';
  try {
    const fd = new FormData();
    // /upload accepts both `image` and `audio` field names.
    fd.append('audio', file);
    const r = await fetch('/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    AUDIO_STUDIO.audioPath = data.path || null;
    AUDIO_STUDIO.audioName = file.name;
    AUDIO_STUDIO.audioDuration = (data.duration_sec != null) ? Number(data.duration_sec) : null;
    audioStudioRenderSlots();
    if (status) status.textContent = '';
  } catch (e) {
    if (status) status.textContent = 'Audio upload failed: ' + (e.message || 'unknown');
  }
}

// audioStudioUploadImage / audioStudioClearImage were retired
// 2026-05-20 when the image slot was migrated to the shared .picker
// component (key='a2v_image'). pickerUploadFile + pickerSetImage now
// handle upload and clear, and the hidden input #a2v_image carries the
// path that audioStudioGenerate() reads.

function audioStudioClearAudio() {
  AUDIO_STUDIO.audioPath = null;
  AUDIO_STUDIO.audioName = null;
  AUDIO_STUDIO.audioDuration = null;
  audioStudioRenderSlots();
}

function audioStudioRenderSlots() {
  // Only the audio slot is rendered here. The image slot is now a
  // shared .picker (key='a2v_image') wired by pickerWire — it manages
  // its own preview, clear button, and recent-uploads strip.
  const audioSlot = document.getElementById('audioStudioAudioSlot');
  if (!audioSlot) return;
  if (AUDIO_STUDIO.audioPath) {
    const dur = (AUDIO_STUDIO.audioDuration != null)
      ? ' · ' + AUDIO_STUDIO.audioDuration.toFixed(1) + ' s'
      : '';
    // escapeHtml — user-supplied filename can contain &, <, >, ", '
    // that would otherwise break out of innerHTML into attribute/tag
    // context. Same defense the rest of the panel uses for any user
    // string that ends up in innerHTML.
    const safeName = escapeHtml(AUDIO_STUDIO.audioName || 'audio');
    audioSlot.innerHTML =
      '<span class="ref-tag">Audio</span>' +
      '<div style="padding:18px 12px;text-align:center;font-size:12px;color:var(--text);overflow:hidden;text-overflow:ellipsis;">' +
      '<svg class="ph" aria-hidden="true" style="width:18px;height:18px;vertical-align:-3px;margin-right:6px;"><use href="#ph-music-notes"/></svg>' +
      safeName + dur +
      '<div class="hint" style="margin-top:8px;">' +
        '<a href="#" onclick="event.stopPropagation();audioStudioClearAudio();return false;">Remove</a>' +
      '</div>' +
      '</div>';
  } else {
    audioSlot.innerHTML =
      '<span class="ref-tag">Audio</span>' +
      '<div class="hint" style="padding:24px 12px;text-align:center;color:var(--muted);font-size:12px;">' +
      'Drop a WAV / MP3 / M4A / FLAC here, or click to pick a file.' +
      '</div>';
  }
}

// Duration slider readout + length warning.
//
// The frame arithmetic here is the same one the submit path uses (24 fps,
// rounded up to the 8k+1 grid), so the number shown is the number the model
// is actually asked for — not a rounded-down approximation of it.
//
// This warns rather than blocks on purpose, and as of 3.8.3 it warns on the
// right quantity.
//
// It used to fire on FRAME COUNT alone, past ~257. @blackest then answered the
// two questions that were open in #46, and both answers moved the target:
//
//   1. The AUDIO survives. "I am certain it continues right to the end of the
//      20 seconds" — so this is the picture running out of coherence, not the
//      joint sequence failing.
//   2. It is not a frame cap. Same 20 s request, same machine (M2 Max 96 GB):
//        640x480   481 frames   fine, all 20 s
//        1024x576  481 asked    picture dies at frame ~454
//        1280x720  481 asked    white out by 481
//      "I think its not so much a frame cap but a limit on how many frames at
//      x size can be generated."
//
// So the budget is frames x AREA, and the old warning was simply wrong in the
// case that matters most: it shouted at a 640x480 20 s render that works.
//
// AND THEN THE REPORTER'S OWN FOUR DATAPOINTS REFUTED THAT. No frames x area
// constant can separate them: 832x480 x 721 = 287.9 Mpx holds together, while
// 1024x576 x 481 = 283.7 Mpx falls apart — a SMALLER product failing. A budget
// that calls the working one worse than the failing one is not a budget, it is
// a coin toss with a number on it.
//
// They separate cleanly on PER-FRAME AREA. Clean: 0.307 and 0.399 Mpx/frame,
// at 721 and 481 frames. Failing: 0.590 and 0.922 Mpx/frame, both giving out
// around frame 450. So the lever is the CANVAS, and length is nearly free
// below the knee — which is the opposite of what the old warning told people,
// and it told them so at exactly the 640x480 20 s render that works.
//
// The knee sits at ~0.45 Mpx a frame, between the highest clean reading and
// the lowest failing one. Below it, nothing is warned about within the
// slider's range; above it, the warning names the canvas as the lever and
// ~450 frames as where the picture gives out. Still field reports from other
// people's machines, not a limit measured here, and the copy still says so.
const A2V_AREA_KNEE = 0.45e6;      // pixels per frame
const A2V_KNEE_FRAMES = 450;       // where the reports say the picture goes
function _a2vFramesForSeconds(sec) {
  const target = Math.max(1, Math.round(sec * 24));
  return ((target - 1 + 7) >> 3 << 3) + 1;   // round up to 8k+1
}
function audioStudioDurationChanged(val) {
  const slider = document.getElementById('audioStudioDuration');
  // Called with no argument from the Width/Height inputs, so the canvas and
  // the length always warn against each other rather than in isolation.
  const sec = parseInt((val !== undefined && val !== null ? val
                        : (slider ? slider.value : '7')) || '7', 10);
  const out = document.getElementById('audioStudioDurationVal');
  if (out) out.textContent = sec + ' s';
  const warn = document.getElementById('audioStudioDurationWarn');
  if (!warn) return;
  const frames = _a2vFramesForSeconds(sec);
  const w = parseInt((document.getElementById('audioStudioWidth') || {}).value || '1024', 10);
  const h = parseInt((document.getElementById('audioStudioHeight') || {}).value || '576', 10);
  const area = (w > 0 && h > 0) ? w * h : 1024 * 576;
  // THE CANVAS IS THE LEVER, NOT THE LENGTH. Below the knee the reports run
  // clean to 721 frames, so there is nothing to say; above it they give out
  // around frame 450 whatever the length asked for.
  if (area > A2V_AREA_KNEE && frames > A2V_KNEE_FRAMES) {
    const kneeSec = Math.max(1, Math.floor((A2V_KNEE_FRAMES - 1) / 24));
    warn.style.display = '';
    warn.innerHTML = '<b>' + w + '\u00d7' + h + '</b> is past the canvas where '
      + 'long Audio \u2192 Video renders have been reported holding together. '
      + 'What gives out is the picture, not the sound: the audio plays to the '
      + 'end while anatomy drifts and the frame washes out, and the reports put '
      + 'that around <b>frame ' + A2V_KNEE_FRAMES + '</b> (~' + kneeSec + ' s) '
      + 'whatever length was asked for. It is the <b>canvas</b>, not the length: '
      + '832\u00d7480 has been reported clean at 721 frames, while 1024\u00d7576 '
      + 'died around frame 454 on the same machine '
      + '(<a href="https://github.com/mrbizarro/Phosphene/issues/46" target="_blank" rel="noopener">#46</a> '
      + '\u2014 field reports, not a limit measured here). '
      + 'Drop to about 832\u00d7480 or smaller and the length comes back. Longer still renders.';
  } else {
    warn.style.display = 'none';
    warn.textContent = '';
  }
}

async function audioStudioEnhancePrompt() {
  const ta = document.getElementById('audioStudioPrompt');
  const original = ta.value.trim();
  if (!original) { alert('Type a prompt before enhancing it.'); return; }
  const btn = document.getElementById('audioStudioEnhanceBtn');
  const originalLabel = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<svg class="ph" aria-hidden="true" style="margin-right:6px;vertical-align:-2px"><use href="#ph-sparkle-fill"/></svg>Loading Gemma\u2026 (~15s)';
  try {
    const r = await fetch('/prompt/enhance', { method: 'POST', body: new URLSearchParams({ prompt: original, mode: 't2v' }) });
    const res = await r.json();
    if (res.error) { alert('Enhance failed: ' + res.error); return; }
    if (confirm('Original:\n' + res.original + '\n\nEnhanced:\n' + res.enhanced + '\n\nReplace your prompt with the enhanced version?'))
      { ta.value = res.enhanced; ta.dispatchEvent(new Event('input', { bubbles: true })); }
  } catch (e) { alert('Enhance request failed: ' + (e.message || e)); }
  finally { btn.disabled = false; btn.innerHTML = originalLabel; }
}

async function audioStudioGenerate() {
  if (AUDIO_STUDIO.busy) return;
  const status = document.getElementById('audioStudioStatus');
  const btn = document.getElementById('audioStudioGenBtn');
  const prompt = (document.getElementById('audioStudioPrompt').value || '').trim();
  if (!AUDIO_STUDIO.audioPath) {
    if (status) status.textContent = 'Audio file is required.';
    return;
  }
  if (!prompt) {
    if (status) status.textContent = 'Prompt is required.';
    return;
  }
  // Width / height from free-form inputs; snap to 32 px multiple
  // (LTX latent grid requirement).
  const w = Math.max(32, Math.round(parseInt(document.getElementById('audioStudioWidth').value || '1024', 10) / 32) * 32);
  const h = Math.max(32, Math.round(parseInt(document.getElementById('audioStudioHeight').value || '576', 10) / 32) * 32);
  // Duration from slider → frames at 8k+1 cadence the model expects.
  const dur = parseInt(document.getElementById('audioStudioDuration').value || '7', 10);
  const frames = _a2vFramesForSeconds(dur);
  const seed = parseInt(document.getElementById('audioStudioSeed').value || '-1', 10);
  const audioConditioningScale = parseFloat(document.getElementById('audioConditioningScale').value || '1.0');
  // Offset into the source file. Clamped at 0 here as well as server-side —
  // a negative start would be silently swallowed by load_audio.
  const audioStartEl = document.getElementById('audioStudioStart');
  const audioStart = Math.max(0, parseFloat((audioStartEl && audioStartEl.value) || '0') || 0);

  AUDIO_STUDIO.busy = true;
  if (btn) btn.disabled = true;
  if (status) status.textContent = 'Queueing…';
  try {
    // Image path lives in the shared .picker hidden input (#a2v_image),
    // not AUDIO_STUDIO.imagePath anymore. Empty string when the user
    // didn't pick one — pure A2V flow.
    const a2vImageEl = document.getElementById('a2v_image');
    const a2vImagePath = (a2vImageEl && a2vImageEl.value) || '';
    const fd = new URLSearchParams();
    fd.set('mode', 'a2v');
    fd.set('prompt', prompt);
    fd.set('audio', AUDIO_STUDIO.audioPath);
    if (a2vImagePath) fd.set('image', a2vImagePath);
    fd.set('width', String(w));
    fd.set('height', String(h));
    fd.set('frames', String(frames));
    fd.set('seed', String(seed));
    fd.set('audio_conditioning_scale', String(audioConditioningScale));
    fd.set('audio_start_time', String(audioStart));
    fd.set('quality', 'high');  // A2V is always pipeline-class (Q8 dev or Q4 distilled)
    // No accel, no enhance — A2V uses A2VidPipelineTwoStage's own walks.
    fd.set('accel', 'off');
    fd.set('enhance', 'off');
    // Forward picked LoRAs if any are active in the unified picker.
    if (typeof _activeLoras !== 'undefined' && Array.isArray(_activeLoras) && _activeLoras.length) {
      const slim = _activeLoras.map(l => ({ path: l.path, strength: l.strength }));
      fd.set('loras', JSON.stringify(slim));
    }
    const r = await fetch('/queue/add', { method: 'POST', body: fd });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error('HTTP ' + r.status + ' ' + txt);
    }
    if (status) status.textContent = 'Submitted. Watch Now / Recent.';
    if (typeof phosToast === 'function') {
      phosToast('Queued Audio → Video clip · watch Now', { kind: 'success' });
    }
    const nowTab = document.querySelector('.tabs button[data-tab="now"]');
    if (nowTab) {
      nowTab.classList.add('flash');
      setTimeout(() => nowTab.classList.remove('flash'), 1200);
    }
  } catch (e) {
    if (status) status.textContent = 'Submit failed: ' + (e.message || 'unknown');
    if (typeof phosToast === 'function') {
      phosToast('A2V submit failed: ' + (e.message || 'unknown'),
                { kind: 'danger', duration: 5000 });
    }
  } finally {
    AUDIO_STUDIO.busy = false;
    if (btn) btn.disabled = false;
  }
}

// ====== Train Character pane ================================================
// In-progress dataset state. The job_id is minted by the server on the FIRST
// upload of a session and echoed back on every subsequent /train/upload so
// the whole batch lands in one folder under state/train_character/<id>/.
globalThis.TRAIN = {
  jobId: null,
  // Mirror of the server-side image list. Each: {filename, path, src}.
  images: [],
  // Mirrors the server's TRAIN_DEFAULT_PRESET. Character defaults to HIGH:
  // it is the only recipe ever graded on a face, and for two months the panel
  // pre-selected + badged Quick instead, which is how a first-time user spent
  // his GPU hours on a 1.98e-04 adapter (#62). Style defaults to Quick — a
  // look is what rank 8/16 is genuinely good at.
  defaultPresets: BOOT.train_default_preset || { character: 'high', style: 'quick' },
  preset: (BOOT.train_default_preset && BOOT.train_default_preset.character) || 'high',
  // True once the user has clicked a preset pill themselves. Until then the
  // train-type toggle is free to snap the preset to that type's default; once
  // it is true, the panel never overrides a deliberate choice.
  presetTouched: false,
  // Train type — 'character' (face + optional voice) or 'style' (no voice,
  // aesthetic LoRA). Drives the preset table lookup + UI visibility (Voice
  // card hides for style; guidance + labels swap). Mirrors the server-side
  // TRAIN_TYPES tuple in mlx_ltx_panel.py.
  trainType: 'character',
  // Local mirror of the server preset table. BOOT carries the authoritative,
  // hardware-adjusted version so 48 GB Macs see the compact training profile
  // instead of the old 64 GB-class defaults. Static fallbacks below are only
  // for damaged/old boot payloads.
  //
  // Schema change 2026-05-19: presets describe EPOCHS, not steps. Actual
  // step count is computed from `epochs × image_count` at the consumer
  // (trainComputeSteps below) so the same preset auto-scales with
  // dataset size. Advanced trainSteps override still wins.
  presets: BOOT.train_presets || {
    quick:  { epochs:  30, rank: 8,  resolution: 512, seconds_per_step: 1.5, ram_peak_gb: 12,
              label: 'Quick',  subtitle: '~30 epochs · rank 8 · 512px', max_steps: 3000 },
    medium: { epochs:  60, rank: 16, resolution: 576, seconds_per_step: 2.2, ram_peak_gb: 18,
              label: 'Medium', subtitle: '~60 epochs · rank 16 · 576px', max_steps: 5000 },
    high:   { epochs: 100, rank: 32, resolution: 512, seconds_per_step: 2.0, ram_peak_gb: 28,
              label: 'High',   subtitle: '~100 epochs · rank 32 · 512px (v2 recipe)', max_steps: 7000 },
  },
  // Mirror of the server-side TRAIN_STYLE_PRESETS. Style table differs from
  // character: quick uses rank 16 (not rank 8), and "high" adds epochs not
  // rank (rank 32 is the validated capacity ceiling for our LTX-2.3 stack).
  stylePresets: BOOT.train_style_presets || {
    quick:  { epochs:  30, rank: 16, resolution: 512, seconds_per_step: 1.5, ram_peak_gb: 12,
              label: 'Quick',  subtitle: '~30 epochs · rank 16 · 512px', max_steps: 3000 },
    medium: { epochs:  60, rank: 32, resolution: 512, seconds_per_step: 2.0, ram_peak_gb: 18,
              label: 'Medium', subtitle: '~60 epochs · rank 32 · 512px', max_steps: 5000 },
    high:   { epochs: 100, rank: 32, resolution: 512, seconds_per_step: 2.0, ram_peak_gb: 28,
              label: 'High',   subtitle: '~100 epochs · rank 32 · 512px', max_steps: 7000 },
  },
  trainProfile: BOOT.train_profile || {},
  // Voice (optional) state. `voiceFile` is the server-saved record once
  // an upload completes; `voiceEnabled` mirrors the toggle.
  //
  // Default voiceEnabled = TRUE so the checkbox reads "audio LoRA: on"
  // even before the user uploads a clip. The checkbox is still HTML-
  // disabled until upload, so the user can't actually submit
  // train_audio=true without a voice file — trainStart() gates on
  // `voiceEnabled && voiceFile` (line ~19334), so this default is a
  // visual signal ("we'll train audio if you give us a clip") rather
  // than a behavioral trap. After upload the disabled flag flips off
  // and the user can uncheck if they want face-only.
  voiceFile: null,           // {filename, path, size, audioUrl}
  voiceEnabled: true,         // default ON visually; behaviorally gated on voiceFile
  voicePreset: 'standard',    // 'smoke' | 'standard' | 'long'
  voicePresets: {
    smoke:    { steps: 100, label: 'Smoke',    sub: '~7 min wall' },
    standard: { steps: 250, label: 'Standard', sub: '~17 min wall' },
    long:     { steps: 500, label: 'Long',     sub: '~33 min wall' },
  },
  // ~30 s/step on M4 Max. Mirrors TRAIN_AUDIO_SECONDS_PER_STEP py-side;
  // these labels are guidance not contract.
  voiceSecondsPerStep: 30,
  initialised: false,
};

// Helper — returns the right preset table for the current train type.
// Centralizes the lookup so trainUpdateEstimate / trainUpdateButtonState
// don't each have to know about both tables.
function trainActivePresets() {
  return TRAIN.trainType === 'style' ? TRAIN.stylePresets : TRAIN.presets;
}

// The preset this panel recommends for the active train type. Served by the
// backend (TRAIN_DEFAULT_PRESET) so the badge, the pre-selection and make_job's
// fallback are one decision in one place.
function trainRecommendedPreset() {
  const map = TRAIN.defaultPresets || {};
  const key = map[TRAIN.trainType];
  const table = trainActivePresets();
  if (key && table && table[key]) return key;
  return (table && table.high) ? 'high' : 'quick';
}

function trainActivePreset() {
  const table = trainActivePresets();
  return table[TRAIN.preset] || table[trainRecommendedPreset()] || {};
}

function trainUpdatePresetButtons() {
  const table = trainActivePresets();
  const recommended = trainRecommendedPreset();
  document.querySelectorAll('#trainPresetGroup .pill-btn').forEach((b) => {
    const key = b.dataset.trainPreset;
    b.classList.toggle('active', key === TRAIN.preset);
    // The badge follows the recommendation instead of being nailed to one
    // pill in the HTML. A badge on a preset that cannot do the job the tab
    // is for is the interface lying, and that is the whole of #62.
    const badge = b.querySelector('[data-rec-slot]');
    if (badge) badge.hidden = (key !== recommended);
  });
  const subIds = {
    quick: 'trainPresetQuickSub',
    medium: 'trainPresetMediumSub',
    high: 'trainPresetHighSub',
  };
  Object.keys(subIds).forEach((key) => {
    const el = document.getElementById(subIds[key]);
    const p = table[key];
    if (el && p && p.subtitle) el.textContent = p.subtitle;
  });
  trainUpdatePresetNote();
}

// The sentence under the pills. Says what the number means, and on a sub-64 GB
// Mac says the thing no pill NAME can say: "High" there is rank 8 / 500 steps /
// 448px on two projections, so the graded rank-32 recipe is unreachable from
// this menu and "just use High" is advice this machine cannot honour (#62).
function trainUpdatePresetNote() {
  const el = document.getElementById('trainPresetNote');
  if (!el) return;
  const profile = TRAIN.trainProfile || {};
  const compact = !!profile.compact;
  if (TRAIN.trainType === 'style') {
    el.textContent = compact
      ? 'This Mac trains on the compact profile (rank 4–8, 384–448px, capped steps), so a style adapter here will be a light touch. No style preset has been graded.'
      : 'A style adapter teaches a look, not a person, so Quick is a fair place to start; Medium and High buy capacity for a more complex look. None of the style presets has been graded.';
    return;
  }
  if (compact) {
    const ram = profile.ram_gb ? `${profile.ram_gb} GB` : 'under 64 GB';
    el.innerHTML =
      `<strong>This Mac has ${escapeHtml(String(ram))}, so training runs the compact profile — and "High" here is not the graded recipe.</strong> ` +
      'It trains rank 8 / 500 steps / 448px on two of the four attention projections. The recipe that has carried a face is rank 32 / ~100 epochs / 512px on all four, ' +
      'and no preset on this machine can reach it. Expect a weak adapter and treat the result as a look rather than an identity — that is the hardware, not your photos.';
    return;
  }
  el.innerHTML =
    '<strong>High is the only recipe ever graded on a face.</strong> ' +
    'Measured with <code>lora_compat.py</code>, rank-32 adapters that carry an identity sit at 5.4e-04 to 1.6e-03 delta_rms; ' +
    'Quick\'s rank 8 has measured 1.54e-04 and 1.98e-04 on real datasets, at or under the 2.0e-04 floor no working adapter has been below. ' +
    'Pick Quick for a fast look or a style, not for a person.';
}

function trainDisableSelectAbove(selectId, maxValue) {
  const sel = document.getElementById(selectId);
  const max = Number(maxValue || 0);
  if (!sel || !max) return;
  Array.from(sel.options).forEach((opt) => {
    if (!opt.value) {
      opt.disabled = false;
      return;
    }
    const n = Number(opt.value);
    opt.disabled = Number.isFinite(n) && n > max;
  });
  const selected = Number(sel.value || 0);
  if (selected && selected > max) sel.value = '';
}

function trainApplyAdvancedLimits() {
  const preset = trainActivePreset();
  const maxRank = Number(preset.max_rank || TRAIN.trainProfile.max_rank || 0);
  const maxResolution = Number(preset.max_resolution || TRAIN.trainProfile.max_resolution || 0);
  const maxSteps = Number(preset.max_steps || TRAIN.trainProfile.max_steps || 0);
  if (maxRank) trainDisableSelectAbove('trainRank', maxRank);
  if (maxResolution) trainDisableSelectAbove('trainResolution', maxResolution);
  const stepsInput = document.getElementById('trainSteps');
  if (stepsInput && maxSteps) {
    stepsInput.max = String(maxSteps);
    const current = Number(stepsInput.value || 0);
    if (current && current > maxSteps) stepsInput.value = '';
  }
}

function trainUpdateAdvancedFields() {
  trainApplyAdvancedLimits();
}

// ============================================================================
// CHARACTERS TAB — discover paired LoRAs, pick one, type the scene, ship.
// ============================================================================
//
// Server endpoints:
//   GET  /characters                  — discover bundles, returns list
//   GET  /characters/<id>/preview     — serve the sample training image
//   POST /characters/<id>/generate    — assemble prompt + queue render job
//
// Recipe defaults (TC=1.8, stage1=10/stage2=3, cfg=3.0, seed=-1,
// enhance=false, video_skip=1+audio_skip=1) are applied server-side per
// docs/API.md. The UI collects prompt (trigger pre-filled) + duration +
// quality. Draft/Pro use the generation's graded default; LTX-2.5 also offers
// the two-stage High canvases. All four tokens come from BOOT.ltx.character.

window.CHARACTERS = {
  list: [],            // [{id, name, trigger, pronoun, subject_noun, sample_image_url, ...}]
  selected: null,      // currently-composing character (object from list)
  duration: '7s',      // 5s | 7s | 10s | 15s
  quality: 'pro',      // token: draft | pro | high | high720 (last two: 2.5)
  // ONE CHARACTER, ONE PAIR OF STRENGTHS, WHICHEVER SURFACE LAUNCHED IT.
  // These were 0.8 and "applied to both face_lora and audio_lora" — a 2.3-era
  // correction for over-baked quirks at 5000 steps, and a comment that stopped
  // being true when the server split the two files. The server has defaulted
  // face 1.0 / voice 1.0 since; this surface kept sending 0.8 and never sent a
  // voice value at all, so the same character rendered face 0.8 / voice 1.0
  // here and face 1.0 / voice 1.0 from the Manual tab. Same numbers on both
  // lanes or the surfaces disagree again.
  charStrength: 1.0,
  voiceStrength: 1.0,
  // Reference audio for i2v_clean_audio mode (character lip-syncs to
  // this clip). Image-to-video deliberately omitted on the Characters
  // surface per Mr Bizarro 2026-05-16.
  audioPath: null,     // server-side path returned by /upload
  audioName: null,     // original filename for display
  // Extra LoRAs are NOT tracked here — the Characters tab portals the
  // Manual tab's existing #lorasDetails picker into its compose card
  // and reads window._activeLoras directly on submit. One source of
  // truth, no duplicated picker UI.
  loading: false,
  initialised: false,
};

// Triples: [value, short chip label, full descriptive text used in
// the assembled prompt + as a tooltip on the chip].
const CHARACTERS_FRAMING = [
  ['ECU', 'ECU', 'extreme close-up'],
  ['CU',  'CU',  'close-up'],
  ['MCU', 'MCU', 'medium close-up'],
  ['MS',  'MS',  'medium shot'],
  ['LS',  'LS',  'long shot'],
];
// DERIVED from the engine's own length table, not typed. This was the fourth
// independent duration vocabulary in the panel (after the storyboard card's
// [3,5,7,10], the Manual strip and H3's table), and four tables for one concept
// is how the 7-second lie happened. The blurb is the table's own.
const CHARACTERS_DURATION = (((BOOT.ltx || {}).lengths) || [])
  .filter(l => l.offered !== false)
  .map(l => [l.key, l.label, `${l.seconds} seconds — ${l.blurb}`]);
// DERIVED from BOOT.ltx.character — the same server-resolved table the
// composer's character strip renders, so the two surfaces cannot disagree
// about what a character render is.
//
// It used to be a hardcoded pair, `draft` / `high`, carrying 2.3-era wall times
// and "Turbo". Two problems at once: there was no way for the tab to ASK for
// the graded 2.5 path (q8 + distilled), and `high` on 2.5 meant the two-stage
// HQ pipeline plus the 29.5 GB add-on — so the tab was offering the only two
// choices that were wrong.
//
// Each tuple's fourth item is the server row. That makes install routing use
// the same pack/pipeline fields as the main High chips.
const CHARACTERS_QUALITY = (() => {
  const c = ((BOOT.ltx || {}).character) || {};
  const rows = Array.isArray(c.tokens) && c.tokens.length
    ? c.tokens
    : [c.draft, c.pro].filter(Boolean);
  if (!rows.length) {
    return [['pro', 'Q8 Pro', 'Q8 Pro — the graded default character recipe', null]];
  }
  return rows.map(row => [
    row.key,
    row.label,
    `${row.label} — ${row.width}×${row.height}, ${row.tier}`,
    row,
  ]);
})();
// Look-up by value → full descriptive text (the third tuple slot).
const CHARACTERS_FRAMING_TEXT = Object.fromEntries(
  CHARACTERS_FRAMING.map(([v, , full]) => [v, full])
);

async function downloadSampleCharacter() {
  // Class-based (not id) so the button can live in more than one empty state
  // (the Character-mode strip AND the standalone Characters grid) and both
  // update together.
  const btns = Array.from(document.querySelectorAll('.js-get-sample-char'));
  const statuses = Array.from(document.querySelectorAll('.js-get-sample-char-status'));
  if (!btns.length) return;
  const setStatus = (msg) => statuses.forEach(s => { s.hidden = false; s.textContent = msg; });
  const setDisabled = (v) => btns.forEach(b => { b.disabled = v; });
  setDisabled(true);
  setStatus('Starting… (~817 MB, one-time)');
  const finishOk = async (msg) => {
    setStatus(msg);
    // Re-init the picker(s) so the new character appears + the empty state hides.
    try { if (typeof charactersInit === 'function') await charactersInit(); } catch (_) {}
    try { if (typeof refreshManualCharacters === 'function') await refreshManualCharacters(); } catch (_) {}
    setDisabled(false);
  };
  try {
    const r = await fetch('/characters/download-sample', { method: 'POST' });
    const j = await r.json().catch(() => ({}));
    if (j && j.already) { await finishOk('Already installed — "Bizarro" is in your characters.'); return; }
    if (!r.ok && r.status !== 202) { throw new Error((j && j.error) || ('HTTP ' + r.status)); }
    const poll = setInterval(async () => {
      let s;
      try { s = await (await fetch('/characters/download-sample/status')).json(); }
      catch (_) { return; }
      if (s.status === 'downloading') {
        setStatus('Downloading… ' + (s.mb || 0) + ' / ' + (s.total_mb || '?') + ' MB');
      } else if (s.status === 'done' || s.present) {
        clearInterval(poll);
        await finishOk('Installed! "Bizarro" is in your characters now — pick it to start.');
      } else if (s.status === 'error') {
        clearInterval(poll);
        setStatus('Download failed: ' + (s.error || 'unknown') + ' — try again.');
        setDisabled(false);
      }
    }, 2500);
  } catch (e) {
    setStatus('Could not start: ' + (e.message || e));
    setDisabled(false);
  }
}

async function charactersInit() {
  // Idempotent. The grid is the default view; compose state only
  // appears after the user picks a card.
  if (!window.CHARACTERS.initialised) {
    window.CHARACTERS.initialised = true;
    charactersRenderChips();
    charactersSyncStrengthControls();   // the slider shows what we will submit
    charactersBackToGrid();   // ensure grid state visible
  }
  await charactersLoadList();
}

async function charactersLoadList() {
  if (window.CHARACTERS.loading) return;
  window.CHARACTERS.loading = true;
  const grid = document.getElementById('charactersGrid');
  const empty = document.getElementById('charactersEmpty');
  if (grid) grid.innerHTML = '<div class="characters-loading">Scanning <code>mlx_models/loras/</code>…</div>';
  if (empty) empty.hidden = true;
  try {
    const r = await fetch('/characters', { credentials: 'same-origin' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    window.CHARACTERS.list = Array.isArray(j.characters) ? j.characters : [];
    charactersRenderGrid();
  } catch (e) {
    if (grid) grid.innerHTML = `<div class="characters-loading">Couldn't load characters: ${charactersEscapeHtml(String(e.message || e))}</div>`;
  } finally {
    window.CHARACTERS.loading = false;
  }
}

function charactersRenderGrid() {
  const grid = document.getElementById('charactersGrid');
  const empty = document.getElementById('charactersEmpty');
  if (!grid) return;
  const list = window.CHARACTERS.list || [];
  if (list.length === 0) {
    grid.innerHTML = '';
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  const cards = list.map(c => {
    const img = c.sample_image_url
      ? `<img class="characters-card-img" src="${charactersEscapeAttr(c.sample_image_url)}" alt="${charactersEscapeAttr(c.name || c.trigger)}" loading="lazy">`
      : `<div class="characters-card-placeholder">${charactersEscapeHtml((c.name || c.trigger || '?').slice(0, 1).toUpperCase())}</div>`;
    // Silent characters: muted-speaker overlay on the avatar + a small
    // "Silent" pill in the chip row. has_voice is set server-side by
    // list_characters() based on whether <trigger>.audio.safetensors
    // exists alongside the face LoRA.
    const silentOverlay = c.has_voice
      ? ''
      : `<span class="characters-card-silent-overlay" title="No voice LoRA — character is silent">
           <svg class="ph" aria-hidden="true"><use href="#ph-speaker-slash"/></svg>
         </span>`;
    const silentBadge = c.has_voice
      ? ''
      : `<span class="characters-silent-badge" title="No voice LoRA on disk — train one via the Train tab">
           <svg class="ph" aria-hidden="true"><use href="#ph-speaker-slash"/></svg>Silent
         </span>`;
    return `
      <div class="characters-card" role="button" tabindex="0"
           data-character-id="${charactersEscapeAttr(c.id)}"
           onclick="charactersOpenCompose('${charactersEscapeAttr(c.id)}')"
           onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();charactersOpenCompose('${charactersEscapeAttr(c.id)}')}">
        ${img}
        ${silentOverlay}
        <div class="characters-card-meta">
          <div class="characters-card-name">${charactersEscapeHtml(c.name || c.trigger)}</div>
          <div class="characters-card-chips">
            <code class="characters-trigger-chip">${charactersEscapeHtml(c.trigger)}</code>
            <span class="characters-pronoun-chip">${charactersEscapeHtml((c.pronoun || 'they') + ' · ' + (c.subject_noun || 'person'))}</span>
            ${silentBadge}
          </div>
        </div>
      </div>`;
  }).join('');
  grid.innerHTML = cards;
}

function charactersRenderChips() {
  // Each option is [value, shortLabel, fullText, serverRow]. The chip face shows
  // the short label (so 5 framing options fit on one row at ~420px);
  // the full text lives on `title` for accessibility / hover discovery.
  const renderGroup = (containerId, options, current, fieldName) => {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = options.map(([val, label, full, row]) => {
      const needsInstall = fieldName === 'quality' && row
        && typeof ltxCellNeedsInstall === 'function' && ltxCellNeedsInstall(row);
      const title = needsInstall && typeof ltxCellInstallLabel === 'function'
        ? ltxCellInstallLabel(row) : (full || label);
      return (
      `<button type="button" class="characters-chip${val === current ? ' active' : ''}${needsInstall ? ' needs-install' : ''}"
               data-val="${charactersEscapeAttr(val)}"
               data-pipeline="${charactersEscapeAttr((row && row.pipeline) || '')}"
               title="${charactersEscapeAttr(title)}"
               onclick="charactersPickChip('${fieldName}', '${charactersEscapeAttr(val)}')">${charactersEscapeHtml(label)}</button>`
      );
    }).join('');
  };
  renderGroup('charactersDurationChips', CHARACTERS_DURATION, window.CHARACTERS.duration, 'duration');
  renderGroup('charactersQualityChips',  CHARACTERS_QUALITY,  window.CHARACTERS.quality,  'quality');
}

function charactersPickChip(field, val) {
  if (field !== 'duration' && field !== 'quality') return;
  if (field === 'quality') {
    const option = CHARACTERS_QUALITY.find(([key]) => key === val);
    const row = option && option[3];
    if (row && typeof ltxCellNeedsInstall === 'function'
        && ltxCellNeedsInstall(row)) {
      if (typeof openModelsModal === 'function') openModelsModal();
      return;
    }
  }
  window.CHARACTERS[field] = val;
  charactersRenderChips();
}

function charactersOpenCompose(id) {
  const c = (window.CHARACTERS.list || []).find(x => x.id === id);
  if (!c) return;
  window.CHARACTERS.selected = c;
  // Every entry to the compose view repaints the strength control from state,
  // so a value restored by Load Params is on screen before the user can submit.
  charactersSyncStrengthControls();

  // Toggle states.
  const grid = document.getElementById('charactersGridState');
  const compose = document.getElementById('charactersComposeState');
  if (grid) grid.hidden = true;
  if (compose) compose.hidden = false;

  // Populate the avatar.
  const av = document.getElementById('charactersComposeAvatar');
  const placeholder = document.getElementById('charactersComposeAvatarPlaceholder');
  if (av) {
    if (c.sample_image_url) {
      av.innerHTML = `<img src="${charactersEscapeAttr(c.sample_image_url)}" alt="${charactersEscapeAttr(c.name || c.trigger)}">`;
    } else {
      av.innerHTML = `<div class="characters-avatar-placeholder">${charactersEscapeHtml((c.name || c.trigger || '?').slice(0, 1).toUpperCase())}</div>`;
    }
  }

  // Populate the titles + chips.
  const nameEl = document.getElementById('charactersComposeName');
  if (nameEl) nameEl.textContent = c.name || c.trigger;
  const trigEl = document.getElementById('charactersComposeTrigger');
  if (trigEl) trigEl.textContent = c.trigger;
  const pronounEl = document.getElementById('charactersComposePronoun');
  if (pronounEl) pronounEl.textContent = `${c.pronoun || 'they'} · ${c.subject_noun || 'person'}`;

  // Silent indicator in the compose head. Reuses the same badge style as
  // the grid card so the state is obvious wherever the character is shown.
  let silentEl = document.getElementById('charactersComposeSilent');
  const chipRow = pronounEl ? pronounEl.parentElement : null;
  if (!c.has_voice) {
    if (!silentEl && chipRow) {
      silentEl = document.createElement('span');
      silentEl.id = 'charactersComposeSilent';
      silentEl.className = 'characters-silent-badge';
      silentEl.title = 'No voice LoRA — train one via the Train tab';
      silentEl.innerHTML =
        '<svg class="ph" aria-hidden="true"><use href="#ph-speaker-slash"/></svg>Silent';
      chipRow.appendChild(silentEl);
    } else if (silentEl) {
      silentEl.hidden = false;
    }
  } else if (silentEl) {
    silentEl.hidden = true;
  }

  // Silent characters can't benefit from audio_skip_step (no audio LoRA
  // to skip-guidance). Force Turbo off and disable the checkbox so users
  // aren't confused by a no-op. The label gets a tooltip explaining why.
  const turboCb = document.getElementById('charactersTurbo');
  if (turboCb) {
    turboCb.disabled = !c.has_voice;
    if (!c.has_voice) {
      turboCb.checked = false;
      turboCb.title =
        'This character has no voice LoRA — Turbo (audio skip-guidance) ' +
        'is unavailable. Train a voice via the Train tab to enable.';
    } else {
      turboCb.title = '';
    }
  }

  // Pre-fill the prompt textarea with the trigger word as a starting
  // point so the user keeps it in their prompt. They can edit anything
  // around it but the trigger is what binds identity. Cursor lands at
  // the end so they can immediately type the rest of their scene.
  const ta = document.getElementById('charactersPrompt');
  if (ta) {
    ta.value = `${c.trigger} `;
    setTimeout(() => {
      ta.focus();
      const len = ta.value.length;
      ta.setSelectionRange(len, len);
    }, 60);
  }
  const toast = document.getElementById('charactersToast');
  if (toast) { toast.hidden = true; toast.textContent = ''; toast.classList.remove('error'); }
  // Reset audio state when switching characters.
  charactersClearAudio();
  // Portal the unified LoRA picker into this compose card. Same picker
  // the Manual tab uses (#lorasDetails) — moved here while the user is
  // composing a character clip, so stack-on-top-of-character LoRAs
  // (cinematronx style, etc.) use the same UI everywhere.
  if (typeof _portalLoraPicker === 'function') {
    _portalLoraPicker('characters');
  }
}

function charactersBackToGrid() {
  window.CHARACTERS.selected = null;
  const grid = document.getElementById('charactersGridState');
  const compose = document.getElementById('charactersComposeState');
  if (grid) grid.hidden = false;
  if (compose) compose.hidden = true;
  // Move the LoRA picker back to its declared home (Manual tab #genForm)
  // so that tab's FormData submit still picks up #lorasJson.
  if (typeof _portalLoraPicker === 'function') {
    _portalLoraPicker('video');
  }
}

// Preview removed — the user's prompt IS what the model sees, verbatim.
// Stub kept so legacy oninput hooks calling it don't error.
function charactersUpdatePreview() { /* no-op (preview block removed) */ }

// ---- Reference audio upload (Characters compose card) -----------------

async function charactersHandleAudioUpload(ev) {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const fnameEl = document.getElementById('charactersAudioFname');
  const clearEl = document.getElementById('charactersAudioClear');
  if (fnameEl) fnameEl.textContent = 'uploading…';
  try {
    const fd = new FormData();
    fd.append('audio', file);  // /upload accepts both `image` and `audio` field names
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    window.CHARACTERS.audioPath = j.path;
    window.CHARACTERS.audioName = file.name;
    if (fnameEl) fnameEl.textContent = file.name;
    if (clearEl) clearEl.hidden = false;
  } catch (e) {
    if (fnameEl) fnameEl.textContent = 'upload failed: ' + (e.message || e);
  }
  ev.target.value = '';
}

function charactersClearAudio() {
  window.CHARACTERS.audioPath = null;
  window.CHARACTERS.audioName = null;
  const fnameEl = document.getElementById('charactersAudioFname');
  const clearEl = document.getElementById('charactersAudioClear');
  if (fnameEl) fnameEl.textContent = 'none';
  if (clearEl) clearEl.hidden = true;
}

function charactersUpdateStrengthDisplay(val) {
  const v = parseFloat(val);
  if (!isFinite(v)) return;
  window.CHARACTERS.charStrength = v;
  const out = document.getElementById('charactersStrengthValue');
  if (out) out.textContent = v.toFixed(2);
}

// THE CONTROL IS RENDERED FROM THE STATE, NEVER THE OTHER WAY AROUND.
// The slider's `value` was hardcoded in the markup, so when the state default
// moved to 1.0 the two silently disagreed: an untouched form DISPLAYED 0.80 and
// SUBMITTED 1.0. The user reads one number and gets another, and no amount of
// care in the submit path can fix a control that was never told what it holds.
// Called on every entry to the compose view and after Load Params.
function charactersSyncStrengthControls() {
  const v = Number(window.CHARACTERS.charStrength ?? 1.0);
  const slider = document.getElementById('charactersStrength');
  if (slider) {
    // The markup's min/max must not silently clamp the state either.
    const lo = parseFloat(slider.min), hi = parseFloat(slider.max);
    if (isFinite(lo) && v < lo) slider.min = String(v);
    if (isFinite(hi) && v > hi) slider.max = String(v);
    slider.value = String(v);
  }
  const out = document.getElementById('charactersStrengthValue');
  if (out) out.textContent = v.toFixed(2);
}

async function charactersGenerate() {
  const c = window.CHARACTERS.selected;
  if (!c) return;
  const qualityRow = (CHARACTERS_QUALITY.find(
    ([key]) => key === window.CHARACTERS.quality) || [])[3];
  if (qualityRow && typeof ltxCellNeedsInstall === 'function'
      && ltxCellNeedsInstall(qualityRow)) {
    if (typeof openModelsModal === 'function') openModelsModal();
    return;
  }
  const btn = document.getElementById('charactersGenerateBtn');
  const labelSpan = btn ? btn.querySelector('.characters-generate-label') : null;
  const busySpan  = btn ? btn.querySelector('.characters-generate-busy')  : null;
  const promptEl = document.getElementById('charactersPrompt');
  const toast = document.getElementById('charactersToast');
  const prompt_body = (promptEl?.value || '').trim();
  if (btn) btn.disabled = true;
  if (labelSpan) labelSpan.hidden = true;
  if (busySpan)  busySpan.hidden  = false;
  if (toast) { toast.hidden = true; toast.textContent = ''; toast.classList.remove('error'); }

  const fd = new URLSearchParams();
  // The textarea content IS the prompt — passed verbatim. The backend
  // no longer composes anything around it (no framing prefix, no
  // 'Photorealistic, cinematic, atmospheric.' suffix). The trigger
  // word was pre-filled in the textarea so it's already in here.
  fd.set('prompt_body', prompt_body);
  fd.set('duration', window.CHARACTERS.duration);
  fd.set('quality',  window.CHARACTERS.quality);
  // Character LoRA strength — passed through so the backend can apply
  // it to both face_lora and audio_lora at job-build time.
  // BOTH strengths, always. Sending only the face left the voice to the
  // server's default, which meant this surface could not express the pair it
  // was showing and a sidecar could never round-trip it.
  fd.set('character_strength', String(window.CHARACTERS.charStrength ?? 1.0));
  fd.set('character_voice_strength', String(window.CHARACTERS.voiceStrength ?? 1.0));
  // Reference audio (i2v_clean_audio mode) — optional.
  if (window.CHARACTERS.audioPath) fd.set('audio', window.CHARACTERS.audioPath);
  // Extra LoRAs come from the portaled #lorasDetails picker — read
  // _activeLoras directly so we share state with the Manual tab. The
  // character's OWN face + audio LoRAs are auto-stacked on the
  // backend, so only the picker-selected ones go in extra_loras.
  if (typeof _activeLoras !== 'undefined' && Array.isArray(_activeLoras) && _activeLoras.length) {
    const extras = _activeLoras
      .filter(l => l && l.path)
      .map(l => ({ path: l.path, strength: (typeof l.strength === 'number' ? l.strength : 1.0) }));
    if (extras.length) fd.set('extra_loras', JSON.stringify(extras));
  }

  try {
    const r = await fetch(`/characters/${encodeURIComponent(c.id)}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      credentials: 'same-origin',
      body: fd.toString(),
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    if (toast) {
      toast.hidden = false;
      toast.textContent = `Queued ${c.name || c.trigger} · ${window.CHARACTERS.duration} · ${window.CHARACTERS.quality} → job ${j.job_id}`;
    }
    // Refresh the queue/recent display so the user sees the new job
    // land. The panel's status loop polls every couple of seconds; nudge
    // it for an immediate refresh so the new card pops without delay.
    if (typeof poll === 'function') {
      try { poll(); } catch (e) {}
    }
  } catch (e) {
    if (toast) {
      toast.hidden = false;
      toast.classList.add('error');
      toast.textContent = `Couldn't queue: ${String(e.message || e)}`;
    }
  } finally {
    if (btn) btn.disabled = false;
    if (labelSpan) labelSpan.hidden = false;
    if (busySpan)  busySpan.hidden  = true;
  }
}

function charactersEscapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}
function charactersEscapeAttr(s) { return charactersEscapeHtml(s); }

// Restore the Characters compose state from a sidecar `params` dict that
// was written by /characters/<id>/generate (source === 'characters').
// Called from loadParams() when the clip's sidecar carries that flag.
//
// Sequence:
//   1. Switch workflow tab to Characters (this calls charactersInit which
//      lazy-loads the list if needed).
//   2. Wait for the character list to be ready (charactersLoadList is
//      idempotent — we just await it).
//   3. Call charactersOpenCompose(character_id) — same code path the
//      grid card uses, so all the avatar / chip / preview wiring is
//      reused.
//   4. Restore framing / duration / quality / turbo chip state from the
//      sidecar values. Re-render chips to update .active classes.
//   5. Repopulate the prompt textarea with prompt_body verbatim and
//      refresh the preview.
// ONE RESTORER, BOTH LOAD PATHS. Load Params has two: the Manual path, and
// this Characters branch, which returns early and therefore never reached the
// Manual path's strength restoration at all — so a character clip reloaded
// with BOTH strengths silently back at their defaults. The external review
// found the Manual half fixed and this half untouched, which is exactly what a
// second copy of a rule buys you. There is one copy now.
function _restoreCharacterStrengths(p) {
  const put = (value, inputId) => {
    if (typeof value === 'undefined' || value === null) return;
    const num = parseFloat(value);
    if (Number.isNaN(num)) return;
    const inp = document.getElementById(inputId);
    if (inp) inp.value = num;
  };
  put(p.character_strength, 'characterStrength');
  put(p.character_voice_strength, 'characterVoiceStrength');
  if (typeof _renderCharsAppliedNote === 'function') _renderCharsAppliedNote();
  // The Characters-tab state and its visible slider follow the same numbers,
  // so a clip reopened there shows what it will re-submit.
  if (window.CHARACTERS) {
    const f = parseFloat(p.character_strength);
    const v = parseFloat(p.character_voice_strength);
    if (!Number.isNaN(f)) window.CHARACTERS.charStrength = f;
    if (!Number.isNaN(v)) window.CHARACTERS.voiceStrength = v;
    if (typeof charactersSyncStrengthControls === 'function') {
      charactersSyncStrengthControls();
    }
  }
}

async function charactersLoadParams(p) {
  // 2026-05-17 — Characters is no longer its own workflow tab. Load Params
  // on a Characters-origin sidecar restores EVERYTHING into the Manual
  // tab's NEW Character mode (Codex C+ pass 3: Character is a first-
  // class mode pill, not a chip strip inside T2V). selectManualCharacter
  // sets up the rest: chip strip, Q8 quality strip swap, auto-stacked
  // face+audio LoRAs at submit.
  // Mr Bizarro's intent: "When you click load params, you get everything
  // exactly the same, so you can replicate clips. Else, what for?"
  workflowSwitch('manual');
  if (typeof setMode === 'function') setMode('character');

  // Wait for the Manual chip strip to be populated so selectManualCharacter
  // can find the character. refreshManualCharacters is idempotent + safe
  // to call multiple times.
  if (typeof refreshManualCharacters === 'function') {
    try { await refreshManualCharacters(); } catch (_) {}
  }
  const charList = (typeof _manualCharacters !== 'undefined' && Array.isArray(_manualCharacters))
    ? _manualCharacters : [];
  const found = charList.find(c => c.id === p.character_id);
  if (!found) {
    throw new Error(`character ${p.character_id} not in list`);
  }

  // (Skip-step restore removed in v4.0.5 — the control was dead at the
  // engine boundary, so an old sidecar's value has nothing to restore into.)

  // Pre-select character. selectManualCharacter() writes the id to the
  // hidden #characterIdInput, appends the trigger to the prompt
  // (idempotent), and (via _applyCharacterQualityStripVisibility) swaps
  // the quality strip to Q8 Draft / Q8 Pro.
  if (typeof selectManualCharacter === 'function') {
    selectManualCharacter(p.character_id);
  }

  // Restore the UI TOKEN, not merely the canvas. Pro and High are both
  // 1024×576, so dimensions alone cannot reproduce the selected pipeline.
  // New sidecars carry quality_choice; older ones fall back to their real
  // quality and finally to geometry. There is no schema/version marker on the
  // old dc0051c sidecars that mapped `high` to Pro, so on LTX-2.5 an old bare
  // `quality=high` now reopens as the newly first-class High token.
  const sidecarW = parseInt(p.width || '0', 10);
  const sidecarH = parseInt(p.height || '0', 10);
  const charCfg = ((BOOT.ltx || {}).character) || {};
  const charRows = Array.isArray(charCfg.tokens) ? charCfg.tokens : [];
  const charKeys = charRows.map(row => row.key);
  const aliases = {
    high_720p: 'high720', balanced: 'pro', standard: 'pro', quick: 'draft'
  };
  let charChoice = String(p.quality_choice || '').toLowerCase();
  charChoice = aliases[charChoice] || charChoice;
  if (!charKeys.includes(charChoice)) {
    const pipelineQuality = String(p.quality || '').toLowerCase();
    charChoice = aliases[pipelineQuality] || pipelineQuality;
  }
  const _draftPairs = [[704, 384], [736, 416]];   // current, then legacy
  const isDraft = _draftPairs.some(([w, h]) =>
    (sidecarW === w && sidecarH === h) || (sidecarW === h && sidecarH === w));
  if (!charKeys.includes(charChoice)) charChoice = isDraft ? 'draft' : 'pro';
  const charQualityGroup = document.getElementById('qualityGroupCharacter');
  if (charQualityGroup) {
    const sel = `[data-char-quality="${charChoice}"]`;
    const btn = charQualityGroup.querySelector(sel);
    if (btn && typeof _setCharacterQuality === 'function') {
      _setCharacterQuality(btn, { allowMissing: true });
    }
  }
  if (window.CHARACTERS) window.CHARACTERS.quality = charChoice;

  // Restore aspect (landscape vs vertical). If the sidecar's recorded
  // h > w, it was a vertical render. setAspect already exists and
  // re-derives width/height via QUALITY_PRESETS — but for character mode
  // we want to keep the exact 736×416 / 1024×576 from the sidecar, so
  // we set the hidden aspect input + active chip ourselves and then
  // restore w/h verbatim.
  const aspectInp = document.getElementById('aspect');
  if (aspectInp) {
    const wantVertical = sidecarH > sidecarW;
    aspectInp.value = wantVertical ? 'vertical' : 'landscape';
    document.querySelectorAll('#aspectGroup .pill-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.aspect === aspectInp.value));
  }
  // Now restore the verbatim dims so any later setQuality / setAspect
  // callbacks don't overwrite them.
  const wInp = document.getElementById('width');
  const hInp = document.getElementById('height');
  if (wInp && sidecarW > 0) wInp.value = sidecarW;
  if (hInp && sidecarH > 0) hInp.value = sidecarH;

  // Both strengths — this branch returns before the Manual path's restoration.
  _restoreCharacterStrengths(p);

  // Restore frames + duration. Frames is the source of truth for the
  // render; the duration field is metadata that drives the UI estimate.
  const sidecarFrames = parseInt(p.frames || '0', 10);
  if (sidecarFrames > 0) {
    const framesInp = document.getElementById('frames');
    if (framesInp) framesInp.value = sidecarFrames;
    // Compute duration from frames (8k+1 → seconds at 24fps).
    const durSec = Math.round(((sidecarFrames - 1) / 24) * 10) / 10;
    const durInp = document.getElementById('duration');
    if (durInp) durInp.value = durSec;
  }

  // Restore seed — prefer seed_used (the integer the helper actually
  // picked at generation time) over seed (often '-1' when the user let
  // the panel randomize). Without this preference, Load Params on a
  // -1 submission restored -1 and the next render got a fresh random
  // seed instead of reproducing the source clip. Mirrors the Manual
  // loadParams fix (commit b024bb5, 2026-05-18); the Character branch
  // was missed in that pass and reported by Mr Bizarro on the panel.
  const _seedRaw = (p.seed_used != null && String(p.seed_used) !== '' && String(p.seed_used) !== '-1')
                    ? p.seed_used
                    : p.seed;
  if (typeof _seedRaw !== 'undefined' && _seedRaw !== null && String(_seedRaw) !== '-1') {
    const seedInp = document.getElementById('seed');
    if (seedInp) seedInp.value = String(_seedRaw);
  }

  // Prompt textarea — verbatim. Use prompt_body if present (legacy
  // sidecars), else the full prompt.
  const ta = document.getElementById('prompt');
  if (ta) {
    ta.value = (typeof p.prompt_body === 'string' && p.prompt_body)
      ? p.prompt_body
      : (p.prompt || '');
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }

  // Restore extra LoRAs (style stacks on top of character). The sidecar
  // stores `loras` as the FULL stack (face + audio + extras). Filter out
  // the character's own face/audio paths to recover only the extras.
  if (Array.isArray(p.loras) && p.loras.length > 0) {
    const facePath  = (found.face_lora_path  || '');
    const audioPath = (found.audio_lora_path || '');
    const extras = p.loras.filter(l => {
      const lp = l && l.path;
      return lp && lp !== facePath && lp !== audioPath;
    });
    if (extras.length && typeof _activeLoras !== 'undefined') {
      _activeLoras.length = 0;
      for (const l of extras) {
        _activeLoras.push({ path: l.path, strength: l.strength });
      }
      if (typeof renderLorasList === 'function') renderLorasList();
    }
  }

  // Scroll to top so the user sees the restored state.
  const formPane = document.querySelector('aside.form-pane');
  if (formPane) formPane.scrollTop = 0;
  // Done — early-exit before the old dead-UI code that followed.
  // The Manual tab IS the character surface now, and the restoration above is
  // the whole of it. What used to follow this point was 37 lines targeting the
  // retired Characters-tab UI, unreachable behind a `return;` and labelled
  // "DEAD CODE BELOW (kept temporarily to avoid bigger diff)". It outlived the
  // follow-up that was meant to prune it and became actively dangerous: it held
  // NEWER-looking quality-mapping logic than the live path above, so the next
  // reader had two plausible implementations and only one that runs. Deleted.
}


// ============================================================================
// TRAIN CHARACTER tab
// ============================================================================
const TRAIN_MIN = Number(BOOT.train_min_images || 15);
const TRAIN_MAX = Number(BOOT.train_max_images || 50);

function trainInit() {
  // Idempotent — safe to call on every setMode('train') without re-binding
  // the drop zone. Triggers a list refresh too so the user sees the LoRAs
  // they trained in earlier sessions.
  if (!TRAIN.initialised) {
    TRAIN.initialised = true;
    trainWireDropZone();
    trainWireBundleDropZone();
    // Note: train-type pill clicks are now handled by `setTrainType()`
    // wired directly via the buttons' onclick=, not via this init
    // path. See setTrainType definition below for context.
    trainWirePresetButtons();
    trainWireAdvancedFields();
    trainWireVoice();
    // Initial trigger value — JS-side generator (instant); /train/suggest-trigger
    // exists for non-JS callers.
    const t = document.getElementById('trainTrigger');
    if (t && !t.value) t.value = trainGenerateTriggerJS();
    document.getElementById('trainTriggerExample').textContent =
      (document.getElementById('trainTrigger').value || 'mrztrn');
    trainTriggerDigitWarn();
    document.getElementById('trainTrigger').addEventListener('input', () => {
      document.getElementById('trainTriggerExample').textContent =
        (document.getElementById('trainTrigger').value || 'mrztrn');
      trainTriggerDigitWarn();
      trainUpdateButtonState();
      trainUpdateEstimate();
    });
  }
  trainUpdatePresetButtons();
  trainUpdateAdvancedFields();
  trainRefreshLoraList();
  trainUpdateEstimate();
  trainUpdateButtonState();
  trainUpdateStartLabel();
  trainCheckPreflight();
  trainGuidanceRestore();
}

// Train-type pill click handler. Top-level (not behind trainInit's
// initialised-once gate) so it always works regardless of how the user
// got to the Train tab — including page reloads where the JS engine
// kept a previous TRAIN.initialised=true in memory. Same pattern as
// setMode / setQuality / setAccel (direct onclick from the HTML).
//
// First attempt (d1a8f9e, addEventListener inside trainInit) was
// reported as still-not-clickable on 2026-05-20; suspect the gate
// kept it from firing on subsequent train-tab opens. Direct onclick
// at the button is bulletproof.
window.setTrainType = function(t) {
  if (t !== 'character' && t !== 'style') return;
  // 1. Toggle .active across the pill group
  const group = document.getElementById('trainTypeGroup');
  if (group) {
    group.querySelectorAll('.pill-btn[data-train-type]').forEach((p) => {
      p.classList.toggle('active', p.dataset.trainType === t);
    });
  }
  // 2. State
  if (typeof TRAIN === 'object' && TRAIN) {
    TRAIN.trainType = t;
    // Snap to the new type's recommended preset — but only while the user
    // has not picked one themselves. Character wants High (the graded
    // identity recipe), style wants Quick (a look, cheaply); silently
    // carrying a 3-hour High over to a style run, or a rank-8 Quick over to
    // a face, is the same class of mistake #62 is about.
    const table = (t === 'style') ? TRAIN.stylePresets : TRAIN.presets;
    const want = (TRAIN.defaultPresets || {})[t];
    if (!TRAIN.presetTouched && want && table && table[want]) {
      TRAIN.preset = want;
    } else if (table && !table[TRAIN.preset]) {
      TRAIN.preset = (want && table[want]) ? want : 'quick';
    }
  }
  // 3. Swap guidance bodies (both shipped in HTML; one starts hidden)
  const gc = document.getElementById('trainGuidanceBodyCharacter');
  const gs = document.getElementById('trainGuidanceBodyStyle');
  if (gc) gc.hidden = (t === 'style');
  if (gs) gs.hidden = (t !== 'style');
  // 4. Hide voice card for style (styles don't have voices)
  const vc = document.getElementById('trainVoiceCard');
  if (vc) vc.hidden = (t === 'style');
  // 5. Re-render downstream UI bits if they exist
  if (typeof trainUpdatePresetButtons === 'function') trainUpdatePresetButtons();
  if (typeof trainUpdateAdvancedFields === 'function') trainUpdateAdvancedFields();
  if (typeof trainUpdateEstimate === 'function') trainUpdateEstimate();
  if (typeof trainUpdateButtonState === 'function') trainUpdateButtonState();
  if (typeof trainUpdateStartLabel === 'function') trainUpdateStartLabel();
};

// "How to train well" guidance — open by default, dismissible. The dismissed
// state persists in localStorage so power users don't see the panel every
// session. New users always see it on first visit.
function trainGuidanceRestore() {
  const el = document.getElementById('trainGuidance');
  if (!el) return;
  try {
    if (localStorage.getItem('phos_train_guidance_dismissed') === '1') {
      el.removeAttribute('open');
    }
  } catch (_) { /* private browsing — leave open */ }
}

function trainGuidanceDismiss() {
  const el = document.getElementById('trainGuidance');
  if (el) el.removeAttribute('open');
  try { localStorage.setItem('phos_train_guidance_dismissed', '1'); } catch (_) {}
}

// Hits /train/preflight and surfaces a banner inside the Train form when a
// required model is missing. Today the only on-demand download is the LTX-2.3
// full-precision dev transformer (~21 GB, Q8 repo) — Phosphene's default
// install ships the distilled transformer only, and the Q4 repo's smaller
// transformer-dev is quantized (trainer refuses it, #35). Banner offers a
// one-click Download button that triggers /train/install + redirects to
// /status for progress.
async function trainCheckPreflight() {
  const box = document.getElementById('trainPreflight');
  if (!box) return;
  try {
    const r = await fetch('/train/preflight');
    const data = await r.json();
    if (!data || !data.ok) { box.style.display = 'none'; return; }
    const missing = (data.required || []).filter(m => !m.ready);
    if (missing.length === 0) {
      box.style.display = 'none';
      return;
    }
    box.style.display = 'block';
    box.innerHTML = `
      <div class="train-preflight-card">
        <div class="train-preflight-title">
          <svg class="ph" aria-hidden="true" style="margin-right:6px;vertical-align:-2px"><use href="#ph-warning-fill"/></svg>Required model${missing.length > 1 ? 's' : ''} not downloaded
        </div>
        <div class="train-preflight-list">
          ${missing.map(m => `
            <div class="train-preflight-row">
              <div class="train-preflight-label">${m.label}
                <span class="train-preflight-size">~${m.size_gb} GB</span></div>
              <div class="train-preflight-blurb">${m.blurb}</div>
              <button type="button" class="btn btn-primary"
                      onclick="trainInstall('${m.key}')">Download</button>
            </div>
          `).join('')}
        </div>
        <div class="train-preflight-foot">
          Phosphene installs only what it renders with. Training runs against
          LTX-2.3 and needs its own weights, so they are downloaded on demand
          rather than shipped to everyone. Each download is resumable, and
          nothing above is needed to render.
        </div>
      </div>
    `;
  } catch (e) {
    box.style.display = 'none';
  }
}

async function trainInstall(key) {
  // The panel parses POST bodies with parse_qs (urlencoded ONLY); a FormData
  // body serializes as multipart and reads back empty, so /train/install saw
  // key='' and returned "unknown install key" (reported by @cocktailpeanut,
  // 2026-06-04). Send urlencoded like every other POST in this panel.
  try {
    const r = await fetch('/train/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'key=' + encodeURIComponent(key),
    });
    const data = await r.json();
    if (!data.ok) {
      alert('Download failed: ' + (data.error || r.status));
      return;
    }
    // Progress streams to STATUS log. Re-poll preflight every 5s; banner
    // self-hides when the file lands.
    const watch = setInterval(async () => {
      try {
        const r2 = await fetch('/train/preflight');
        const d2 = await r2.json();
        const stillMissing = (d2.required || []).filter(m => !m.ready).length;
        if (stillMissing === 0) {
          clearInterval(watch);
          trainCheckPreflight();
        }
      } catch (e) { /* ignore */ }
    }, 5000);
  } catch (e) {
    alert('Download request failed: ' + e.message);
  }
}

// Trigger generator — `<3 consonants><2 digits>` for rare/uncommon tokens.
// Skip vowels + ambiguous letters (l, i, o → 1, 0). Mirrors the Python
// _suggest_trigger_token; not a security boundary so the algorithms can
// drift slightly — server has its own generator for non-JS callers.
// Letters-only, `<3 consonants>trn` — the shape of every trigger that has
// carried a face here (bizarrotrn, elontrn, ariatrn). The old `mrz07` shape
// tokenizes to m / rz / 0 / 7: two single-digit tokens with a huge prior of
// their own. Mirrors _suggest_trigger_token() server-side (#62).
function trainGenerateTriggerJS() {
  const cons = 'bcdfghjkmnpqrstvwxyz';
  let letters = '';
  for (let i = 0; i < 3; i++) letters += cons[Math.floor(Math.random() * cons.length)];
  return letters + 'trn';
}

// Warn (never block) when a typed trigger carries digits — see above.
function trainTriggerDigitWarn() {
  const t = document.getElementById('trainTrigger');
  const hint = document.getElementById('trainTriggerHint');
  if (!t || !hint) return;
  const bad = /\d/.test((t.value || '').trim());
  hint.textContent = bad
    ? 'digits split into their own very common tokens — every trigger that has carried a face here was letters-only (e.g. mrztrn)'
    : 'a rare, letters-only token the model will associate with this character';
  hint.classList.toggle('warn', bad);
}

function trainSuggestTrigger() {
  const t = document.getElementById('trainTrigger');
  t.value = trainGenerateTriggerJS();
  document.getElementById('trainTriggerExample').textContent = t.value;
  trainTriggerDigitWarn();
}

function trainWireDropZone() {
  const drop = document.getElementById('trainDrop');
  const input = document.getElementById('trainFileInput');
  if (!drop || !input) return;
  drop.addEventListener('click', (e) => {
    // Avoid re-firing the picker when the user clicks an existing thumb's
    // controls (× / num pill).
    if (e.target.closest('.train-thumb')) return;
    input.click();
  });
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('dragover');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragover');
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) trainUploadFiles(files);
  });
  input.addEventListener('change', () => {
    if (input.files && input.files.length) trainUploadFiles(input.files);
    input.value = '';
  });
}

function trainWireBundleDropZone() {
  const drop = document.getElementById('trainBundleDrop');
  const input = document.getElementById('trainBundleInput');
  if (!drop || !input) return;
  drop.addEventListener('click', () => { if (!drop.classList.contains('busy')) input.click(); });
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('dragover');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragover');
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) trainUploadBundle(f);
  });
  input.addEventListener('change', () => {
    if (input.files && input.files.length) trainUploadBundle(input.files[0]);
    input.value = '';
  });
}

async function trainUploadBundle(file) {
  const drop = document.getElementById('trainBundleDrop');
  const status = document.getElementById('trainStatus');
  if (!file) return;
  if (!/\.zip$/i.test(file.name)) {
    if (status) status.textContent = 'Bundle must be a .zip file.';
    return;
  }
  if (drop) drop.classList.add('busy');
  if (status) status.textContent = `Unpacking ${file.name}…`;
  try {
    const fd = new FormData();
    fd.append('file', file, file.name);
    if (TRAIN.jobId) fd.append('job_id', TRAIN.jobId);
    const r = await fetch('/train/upload-bundle', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) {
      if (status) status.textContent = 'Bundle error: ' + (j.error || r.status);
      return;
    }
    TRAIN.jobId = j.job_id;
    await trainRefreshDataset();
    const bits = [
      `${j.image_count} image(s) staged`,
      `${j.caption_count} caption(s)`,
      `${j.paired_count} paired`,
    ];
    if (Array.isArray(j.unpaired_warnings) && j.unpaired_warnings.length) {
      bits.push(`warnings: ${j.unpaired_warnings.slice(0, 2).join(' · ')}${j.unpaired_warnings.length > 2 ? '…' : ''}`);
    }
    if (status) status.textContent = bits.join(' · ');
  } catch (e) {
    if (status) status.textContent = 'Bundle upload failed: ' + (e.message || 'unknown');
  } finally {
    if (drop) drop.classList.remove('busy');
  }
}

function trainWirePresetButtons() {
  document.querySelectorAll('#trainPresetGroup .pill-btn').forEach(b => {
    b.addEventListener('click', () => {
      TRAIN.preset = b.dataset.trainPreset;
      TRAIN.presetTouched = true;
      document.querySelectorAll('#trainPresetGroup .pill-btn').forEach(x =>
        x.classList.toggle('active', x === b));
      trainUpdateAdvancedFields();
      trainUpdateEstimate();
    });
  });
  // Crop strategy pills — same click pattern. Updates the hidden
  // #trainCropStrategy input which rides the form on submit. No estimate
  // change because letterbox uses the same square canvas as center crop;
  // wall time + memory are unaffected.
  document.querySelectorAll('#trainCropStrategyGroup .pill-btn').forEach(b => {
    b.addEventListener('click', () => {
      const v = b.dataset.trainCrop;
      const inp = document.getElementById('trainCropStrategy');
      if (inp) inp.value = v;
      document.querySelectorAll('#trainCropStrategyGroup .pill-btn').forEach(x =>
        x.classList.toggle('active', x === b));
    });
  });
}

function trainWireAdvancedFields() {
  ['trainRank', 'trainSteps', 'trainLR', 'trainResolution', 'trainCaptionStrategy']
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', () => {
        trainApplyAdvancedLimits();
        trainUpdateEstimate();
      });
      if (el) el.addEventListener('input', () => {
        trainApplyAdvancedLimits();
        trainUpdateEstimate();
      });
    });
  trainApplyAdvancedLimits();
}

async function trainUploadFiles(fileList) {
  const status = document.getElementById('trainStatus');
  // Partition incoming files into images and captions. The user may drop
  // either type into the same zone — image_001.png + image_001.txt is the
  // intended workflow.
  const all = Array.from(fileList);
  const imgs = all.filter(f =>
    /^image\/(png|jpe?g|webp)$/.test(f.type) ||
    /\.(png|jpe?g|webp)$/i.test(f.name));
  const caps = all.filter(f =>
    /\.txt$/i.test(f.name) || /^text\/plain$/.test(f.type));
  if (!imgs.length && !caps.length) {
    if (status) status.textContent = 'No supported files in that selection (need PNG / JPG / WEBP / TXT).';
    return;
  }
  const total = imgs.length + caps.length;
  if (status) status.textContent = `Uploading 0 / ${total}…`;

  let done = 0;
  let capThinWarnings = [];

  // ----- IMAGES (optimistic placeholders so the grid lights up immediately) -----
  for (let i = 0; i < imgs.length; i++) {
    const f = imgs[i];
    if (TRAIN.images.length >= TRAIN_MAX) {
      if (status) status.textContent = `At max ${TRAIN_MAX} images. Stopping upload.`;
      break;
    }
    const localUrl = await trainFileToDataURL(f).catch(() => null);
    const placeholderIdx = TRAIN.images.length;
    TRAIN.images.push({
      filename: '__uploading__' + placeholderIdx,
      path: '',
      src: localUrl || '',
      uploading: true,
      captioned: false,
    });
    trainRenderThumbs();

    try {
      const fd = new FormData();
      fd.append('file', f, f.name);
      if (TRAIN.jobId) fd.append('job_id', TRAIN.jobId);
      const r = await fetch('/train/upload', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok || !j.ok) {
        TRAIN.images.splice(placeholderIdx, 1);
        if (status) status.textContent = 'Upload error: ' + (j.error || r.status);
        trainRenderThumbs();
        continue;
      }
      TRAIN.jobId = j.job_id;
      TRAIN.images[placeholderIdx] = {
        filename: j.filename,
        path: j.path,
        src: '/train/file?job_id=' + encodeURIComponent(j.job_id) +
             '&filename=' + encodeURIComponent(j.filename) +
             '&v=' + Date.now(),
        uploading: false,
        captioned: !!j.captioned,
        original_stem: j.original_stem || null,
      };
      done += 1;
      trainRenderThumbs();
      if (status) status.textContent = `Uploaded ${done} / ${total}…`;
    } catch (e) {
      TRAIN.images.splice(placeholderIdx, 1);
      if (status) status.textContent = 'Upload failed: ' + (e.message || 'unknown');
      trainRenderThumbs();
    }
  }

  // ----- CAPTIONS (server pairs by stem against caption_map.json) -----
  for (let i = 0; i < caps.length; i++) {
    const f = caps[i];
    try {
      const fd = new FormData();
      fd.append('file', f, f.name);
      if (TRAIN.jobId) fd.append('job_id', TRAIN.jobId);
      const r = await fetch('/train/upload', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok || !j.ok) {
        if (status) status.textContent = 'Caption upload error: ' + (j.error || r.status);
        continue;
      }
      TRAIN.jobId = j.job_id;
      if (typeof j.word_count === 'number' && j.word_count < 10) {
        capThinWarnings.push(`${f.name} (${j.word_count} words)`);
      }
      done += 1;
      if (status) status.textContent = `Uploaded ${done} / ${total}…`;
    } catch (e) {
      if (status) status.textContent = 'Caption upload failed: ' + (e.message || 'unknown');
    }
  }
  if (caps.length) {
    // Pull canonical state so the captioned badges on existing thumbs
    // refresh (a caption may have just paired with an already-uploaded image).
    await trainRefreshDataset();
  }

  if (status) {
    const n = TRAIN.images.length;
    const captioned = TRAIN.images.filter(x => x.captioned).length;
    const bits = [];
    if (n < TRAIN_MIN) bits.push(`${n} image${n === 1 ? '' : 's'} — need ${TRAIN_MIN - n} more to train`);
    else bits.push(`${n} images ready`);
    if (caps.length) bits.push(`${captioned} captioned`);
    if (capThinWarnings.length) {
      bits.push(`thin captions: ${capThinWarnings.slice(0, 3).join(', ')}${capThinWarnings.length > 3 ? '…' : ''}`);
    }
    status.textContent = bits.join(' · ');
  }
  trainUpdateEstimate();
  trainUpdateButtonState();
}

function trainFileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function trainRenderThumbs() {
  const wrap = document.getElementById('trainThumbs');
  const empty = document.getElementById('trainDropEmpty');
  const drop = document.getElementById('trainDrop');
  const clearAllBtn = document.getElementById('trainClearAllBtn');
  if (!wrap) return;
  if (TRAIN.images.length === 0) {
    wrap.hidden = true;
    if (empty) empty.style.display = '';
    if (drop) drop.classList.remove('has-images');
    if (clearAllBtn) clearAllBtn.hidden = true;
    trainUpdateCounter();
    return;
  }
  wrap.hidden = false;
  if (empty) empty.style.display = 'none';
  if (drop) drop.classList.add('has-images');
  if (clearAllBtn) clearAllBtn.hidden = false;
  wrap.innerHTML = TRAIN.images.map((img, idx) => {
    const cls = 'train-thumb' + (img.uploading ? ' uploading' : '');
    const removeAttr = img.uploading
      ? ''
      : `onclick="trainRemoveImage('${img.filename.replace(/'/g, "\\'")}')"`;
    const capCls = img.uploading ? '' :
      (img.captioned ? 'captioned' : 'uncaptioned');
    const capLabel = img.uploading ? '' :
      (img.captioned ? '✓ cap' : 'no cap');
    const capTitle = img.uploading ? '' :
      (img.captioned ? 'caption paired' : 'no caption — will fall back to trigger_simple');
    return `<div class="${cls}" data-idx="${idx}">
      <img src="${img.src || ''}" alt="char ${idx + 1}" loading="lazy">
      <span class="train-thumb-num">${String(idx + 1).padStart(3, '0')}</span>
      ${capLabel ? `<span class="train-thumb-cap ${capCls}" title="${capTitle}">${capLabel}</span>` : ''}
      <button type="button" class="train-thumb-x" ${removeAttr} title="Remove"><svg class="ph" aria-hidden="true"><use href="#ph-x-bold"/></svg></button>
    </div>`;
  }).join('');
  trainUpdateCounter();
}

function trainUpdateCounter() {
  const counter = document.getElementById('trainCounter');
  const hint = document.getElementById('trainCounterHint');
  const capChip = document.getElementById('trainCaptionCounter');
  if (!counter) return;
  const n = TRAIN.images.length;
  const captioned = TRAIN.images.filter(x => x.captioned).length;
  counter.textContent = `${n} / ${TRAIN_MAX} images`;
  counter.classList.toggle('ok', n >= TRAIN_MIN);
  counter.classList.toggle('short', n > 0 && n < TRAIN_MIN);
  if (hint) {
    if (n === 0) hint.textContent = `need at least ${TRAIN_MIN} to train`;
    else if (n < TRAIN_MIN) hint.textContent = `need ${TRAIN_MIN - n} more`;
    else if (n < TRAIN_MAX) hint.textContent = `ready · up to ${TRAIN_MAX - n} more if you want variety`;
    else hint.textContent = `at the ${TRAIN_MAX}-image limit`;
  }
  if (capChip) {
    if (captioned === 0) {
      capChip.hidden = true;
    } else {
      capChip.hidden = false;
      capChip.textContent = (captioned === n)
        ? `${captioned} captioned · all paired`
        : `${captioned} / ${n} captioned`;
      capChip.classList.toggle('ok', captioned === n && n > 0);
      capChip.classList.toggle('partial', captioned > 0 && captioned < n);
    }
  }
}

async function trainRemoveImage(filename) {
  if (!TRAIN.jobId || !filename) return;
  try {
    const fd = new URLSearchParams();
    fd.set('train_job_id', TRAIN.jobId);
    fd.set('filename', filename);
    const r = await fetch('/train/remove-image', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) {
      console.warn('remove-image failed', j);
      return;
    }
    // Server renumbers char_001…NN; re-pull the canonical list so our
    // src URLs match the new filenames on disk.
    await trainRefreshDataset();
  } catch (e) {
    console.warn('remove-image error', e);
  }
}

async function trainClearAll() {
  if (!TRAIN.images.length) return;
  if (!confirm(`Remove all ${TRAIN.images.length} uploaded images?`)) return;
  const toRemove = TRAIN.images.filter(x => !x.uploading).map(x => x.filename);
  for (const f of toRemove) {
    await trainRemoveImage(f);
  }
  TRAIN.images = [];
  trainRenderThumbs();
  trainUpdateEstimate();
  trainUpdateButtonState();
}

async function trainRefreshDataset() {
  if (!TRAIN.jobId) {
    TRAIN.images = [];
    trainRenderThumbs();
    return;
  }
  try {
    const r = await fetch('/train/dataset?job_id=' + encodeURIComponent(TRAIN.jobId));
    const j = await r.json();
    if (!j.ok) return;
    TRAIN.images = (j.images || []).map(im => ({
      filename: im.filename,
      path: im.path,
      src: '/train/file?job_id=' + encodeURIComponent(TRAIN.jobId) +
           '&filename=' + encodeURIComponent(im.filename) + '&v=' + Date.now(),
      uploading: false,
      captioned: !!im.captioned,
      caption_words: im.caption_words || null,
      original_stem: im.original_stem || null,
    }));
    if (Array.isArray(j.parked_captions) && j.parked_captions.length) {
      const status = document.getElementById('trainStatus');
      if (status) {
        const msg = `${j.parked_captions.length} caption(s) parked without matching images: ${j.parked_captions.slice(0, 3).join(', ')}${j.parked_captions.length > 3 ? '…' : ''}`;
        status.textContent = msg;
      }
    }
    trainRenderThumbs();
    trainUpdateEstimate();
    trainUpdateButtonState();
  } catch (e) { console.warn('train dataset refresh failed', e); }
}

// ====== Auto-caption (Gemma 3) ======
//
// One-click captioning of every image in the current dataset. POSTs to
// /train/auto-caption which spawns caption_with_gemma.py as a subprocess;
// progress streams to STATE.log and is mirrored on /train/auto-caption/status
// (i / n / current file). We poll that status endpoint every 1 sec while a
// run is active, update the inline progress bar, and refresh the dataset
// view on completion so the user sees the new captioned count + per-image
// caption_words update without a manual reload.
//
// Memory note: Gemma 3 12B (~6 GB at 4-bit) loads in the subprocess and
// frees on exit, so it doesn't accumulate on top of the dev transformer
// the trainer needs next. The backend refuses to start a second run while
// one is in flight + refuses if a training job is currently running.
let _trainCaptionPoll = null;

async function trainAutoCaption() {
  const btn = document.getElementById('trainAutoCaptionBtn');
  const label = document.getElementById('trainAutoCaptionLabel');
  const prog = document.getElementById('trainAutoCaptionProgress');
  const status = document.getElementById('trainAutoCaptionStatus');
  const fill = document.getElementById('trainAutoCaptionBarFill');
  if (!btn || btn.disabled) return;
  if (!TRAIN.jobId) {
    if (status) status.textContent = 'No dataset yet — drop some images first.';
    return;
  }
  const trig = (document.getElementById('trainTrigger').value || '').trim();
  if (!trig) {
    if (status) status.textContent = 'Set a trigger word first.';
    return;
  }
  const n = TRAIN.images.filter(x => !x.uploading).length;
  // Confirmation if there are already captions — auto-caption OVERWRITES.
  // We treat any caption count > 0 as "user has invested in captions" and
  // ask before clobbering. Cheap safety net.
  const existingCaps = TRAIN.images.filter(x => x.captioned).length;
  if (existingCaps > 0) {
    if (!confirm(
      `Auto-caption will OVERWRITE ${existingCaps} existing caption(s) ` +
      `for "${trig}". Continue?`
    )) return;
  }
  btn.disabled = true;
  if (label) label.textContent = 'Captioning…';
  if (prog) prog.hidden = false;
  if (fill) fill.style.width = '0%';
  if (status) status.textContent = `Loading Gemma 3 (~3s)…`;
  const fd = new FormData();
  fd.set('train_job_id', TRAIN.jobId);
  fd.set('trigger', trig);
  let r;
  try {
    r = await fetch('/train/auto-caption', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(fd),
    });
  } catch (e) {
    if (status) status.textContent = 'Network error: ' + (e.message || e);
    trainAutoCaptionFinish();
    return;
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.ok) {
    if (status) status.textContent = 'Failed: ' + (j.error || `HTTP ${r.status}`);
    trainAutoCaptionFinish();
    return;
  }
  // Backend accepted — start polling for progress.
  _trainCaptionPoll = setInterval(trainAutoCaptionPollOnce, 1000);
}

async function trainAutoCaptionPollOnce() {
  let s;
  try {
    const r = await fetch('/train/auto-caption/status');
    s = await r.json();
  } catch (e) { return; }
  const status = document.getElementById('trainAutoCaptionStatus');
  const fill = document.getElementById('trainAutoCaptionBarFill');
  const i = s.i || 0, n = s.n || 0;
  if (fill) fill.style.width = (n > 0 ? (i / n) * 100 : 0).toFixed(1) + '%';
  if (status) {
    if (s.error) {
      status.textContent = `Error: ${s.error}`;
    } else if (s.running) {
      const f = s.current_file ? ` · ${s.current_file}` : '';
      status.textContent = `${i} / ${n}${f} · ${s.elapsed_sec}s elapsed`;
    } else {
      status.textContent = `Done · ${i} captions in ${s.elapsed_sec}s. Refreshing…`;
    }
  }
  if (!s.running) {
    if (_trainCaptionPoll) { clearInterval(_trainCaptionPoll); _trainCaptionPoll = null; }
    // Refresh the dataset so the captioned-count + per-image word_count
    // updates in the thumb grid, then re-enable the button.
    await trainRefreshDataset();
    setTimeout(trainAutoCaptionFinish, 600);
  }
}

function trainAutoCaptionFinish() {
  const btn = document.getElementById('trainAutoCaptionBtn');
  const label = document.getElementById('trainAutoCaptionLabel');
  if (btn) btn.disabled = false;
  if (label) label.textContent = 'Auto-caption with Gemma 3';
  // Leave the progress strip visible at 100% so the user has a record
  // of the run; it'll reset to 0% next time the button is clicked.
  trainUpdateButtonState();
}

// Reflect upload/trigger state into the auto-caption button. Called from
// the existing trainUpdateButtonState() so we don't add another mutation
// observer.
function trainUpdateAutoCaptionState() {
  const btn = document.getElementById('trainAutoCaptionBtn');
  if (!btn) return;
  const n = TRAIN.images.filter(x => !x.uploading).length;
  const trig = (document.getElementById('trainTrigger').value || '').trim();
  // Don't toggle while a run is in flight — the runner manages the button.
  if (_trainCaptionPoll) return;
  const ok = n >= 1 && trig.length >= 3 && trig.length <= 32;
  btn.disabled = !ok;
  if (!ok) {
    if (n < 1) btn.title = 'Drop at least one image first.';
    else if (trig.length < 3) btn.title = 'Trigger word must be 3+ characters.';
    else btn.title = 'Trigger word is too long (max 32).';
  } else {
    btn.title = `Run Gemma 3 on all ${n} image(s) — ~${(2.5 * n).toFixed(0)}s`;
  }
}

function trainEffectiveSteps() {
  const v = (document.getElementById('trainSteps').value || '').trim();
  if (!v) return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

// Mirrors py-side _preset_steps_for(): steps = epochs × image_count, capped
// by preset.max_steps when the active hardware profile supplies one.
// Used by trainUpdateEstimate to keep the ETA chip honest as the user
// drops more photos in. Floor of 1 image so n=0 doesn't yield zero
// steps (matches server behavior).
function trainComputeSteps(preset, imageCount) {
  const epochs = parseInt(preset.epochs, 10) || 0;
  let steps = Math.max(1, epochs * Math.max(1, imageCount | 0));
  const maxSteps = Number(preset.max_steps || 0);
  if (maxSteps > 0) steps = Math.min(steps, maxSteps);
  return steps;
}

function trainUpdateEstimate() {
  trainApplyAdvancedLimits();
  const preset = trainActivePreset();
  const stepOverride = trainEffectiveSteps();
  const n = TRAIN.images.length;
  let steps = stepOverride || trainComputeSteps(preset, n);
  const maxSteps = Number(preset.max_steps || TRAIN.trainProfile.max_steps || 0);
  if (maxSteps > 0) steps = Math.min(steps, maxSteps);
  const sec = Math.round(3 * Math.max(0, n) + steps * preset.seconds_per_step + 30);
  // Estimate row is hidden until the user has dropped at least one
  // image. Before that, the row reads as missing data — not a useful
  // signal. Once there's data, reveal the row + populate it.
  const row = document.getElementById('trainEstimate');
  if (row) row.style.display = n === 0 ? 'none' : '';
  const ramRow = document.getElementById('trainEstimateRam');
  const timeRow = document.getElementById('trainEstimateTime');
  const outRow = document.getElementById('trainEstimateOut');
  if (timeRow && n > 0) {
    const profile = TRAIN.trainProfile && TRAIN.trainProfile.label
      ? ` · ${TRAIN.trainProfile.label}`
      : '';
    timeRow.textContent = trainFmtDuration(sec) + ` · ${steps} steps${profile}`;
  }
  if (ramRow) ramRow.textContent = `~${preset.ram_peak_gb} GB peak`;
  if (outRow) {
    const trig = (document.getElementById('trainTrigger').value || 'mrz07');
    outRow.textContent = `mlx_models/loras/${(TRAIN.jobId || 'trn-<new>')}.safetensors · trigger "${trig}"`;
  }
}

function trainFmtDuration(sec) {
  if (sec < 60) return `${sec} s`;
  const m = Math.round(sec / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h} h ${rem} min` : `${h} h`;
}

function trainUpdateButtonState() {
  const btn = document.getElementById('trainStartBtn');
  if (!btn) return;
  const n = TRAIN.images.filter(x => !x.uploading).length;
  const trig = (document.getElementById('trainTrigger').value || '').trim();
  const ok = n >= TRAIN_MIN && trig.length >= 3 && trig.length <= 32;
  btn.disabled = !ok;
  if (!ok) {
    if (n < TRAIN_MIN) btn.title = `Need at least ${TRAIN_MIN} images (have ${n}).`;
    else if (trig.length < 3) btn.title = 'Trigger word must be 3+ characters.';
    else btn.title = 'Trigger word is too long (max 32).';
  } else {
    btn.title = 'Queue this training job.';
  }
  // The auto-caption button has a LOWER bar (1 image + 3-char trigger,
  // versus 15 images for training) so it's tracked separately. Folded
  // into this same controller so we don't add another mutation observer
  // — every place that calls trainUpdateButtonState() now also reflects
  // the auto-caption state.
  trainUpdateAutoCaptionState();
}

async function trainStart() {
  const btn = document.getElementById('trainStartBtn');
  const status = document.getElementById('trainStatus');
  if (!TRAIN.jobId || TRAIN.images.length < TRAIN_MIN) {
    if (status) status.textContent = `Need at least ${TRAIN_MIN} images.`;
    return;
  }
  const trig = (document.getElementById('trainTrigger').value || '').trim();
  if (!trig) {
    if (status) status.textContent = 'Trigger word required.';
    return;
  }
  // Caption warnings: thin captions (< 10 words) or user_provided strategy
  // with no .txt files at all. Surface as a one-time confirm so power users
  // can override (e.g. they want bare trigger_simple anyway).
  const captionStrategy = document.getElementById('trainCaptionStrategy').value || 'user_provided';
  const captionedCount = TRAIN.images.filter(x => x.captioned).length;
  const thinCaps = TRAIN.images.filter(x =>
    x.captioned && typeof x.caption_words === 'number' && x.caption_words < 10);
  if (captionStrategy === 'user_provided' && captionedCount === 0) {
    if (!confirm('Caption strategy is "user_provided" but no .txt captions were uploaded. ' +
                 'Continue with the trigger_simple fallback for every image?')) return;
  } else if (thinCaps.length) {
    if (!confirm(`${thinCaps.length} caption(s) are very short (< 10 words). ` +
                 'Short captions usually train weaker LoRAs. Continue anyway?')) return;
  }
  const fd = new URLSearchParams();
  fd.set('train_job_id', TRAIN.jobId);
  // train_type drives whether the /train/start server treats this as
  // a character (face + optional voice) or a style (look + color +
  // lighting) training. The pill at the top of the Train tab sets
  // TRAIN.trainType; without sending it here, the server would default
  // to 'character' for every submission and clicking Style would
  // silently train a character LoRA against style frames.
  fd.set('train_type', TRAIN.trainType || 'character');
  fd.set('trigger', trig);
  fd.set('preset', TRAIN.preset);
  fd.set('image_count', String(TRAIN.images.length));
  fd.set('caption_strategy', captionStrategy);
  // Crop strategy chip — center (default) or letterbox.
  const cropInp = document.getElementById('trainCropStrategy');
  if (cropInp && cropInp.value) fd.set('crop_strategy', cropInp.value);
  const rank = document.getElementById('trainRank').value;
  if (rank) fd.set('rank', rank);
  const stepsVal = document.getElementById('trainSteps').value;
  if (stepsVal) fd.set('steps', stepsVal);
  const lrVal = document.getElementById('trainLR').value;
  if (lrVal) fd.set('lr', lrVal);
  const resVal = document.getElementById('trainResolution').value;
  if (resVal) fd.set('resolution', resVal);

  // Voice phase — only sent when the toggle is on AND a clip is uploaded.
  // /train/start re-validates voice presence server-side and 400s if the
  // clip is missing.
  if (TRAIN.voiceEnabled && TRAIN.voiceFile) {
    fd.set('train_audio', 'true');
    const vp = TRAIN.voicePresets[TRAIN.voicePreset]
            || TRAIN.voicePresets.standard;
    fd.set('audio_steps', String(vp.steps));
  }

  const restoreLabel = () => { if (btn) btn.textContent = trainStartLabelText(); };
  if (btn) { btn.disabled = true; btn.textContent = 'Submitting…'; }
  try {
    const r = await fetch('/train/start', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) {
      if (status) status.textContent = 'Failed to enqueue: ' + (j.error || r.status);
      if (btn) { btn.disabled = false; restoreLabel(); }
      return;
    }
    if (status) status.textContent = `Queued · job ${j.queued_id}. Watch the Now / Queue pane for progress.`;
    // Reset the form so the user can stage another dataset. The current
    // job's files stay on disk — only our local mirror is cleared.
    TRAIN.jobId = null;
    TRAIN.images = [];
    TRAIN.voiceFile = null;
    TRAIN.voiceEnabled = false;
    trainRenderThumbs();
    trainRenderVoice();
    trainUpdateEstimate();
    restoreLabel();
    // Refresh queue poll immediately so the new job card appears without
    // waiting for the next 1.5s tick.
    if (typeof poll === 'function') poll();
  } catch (e) {
    if (status) status.textContent = 'Enqueue failed: ' + (e.message || 'unknown');
    if (btn) { btn.disabled = false; restoreLabel(); }
  }
}

// ============================================================
// Voice (optional) — drop zone + toggle + audio-step presets
// ============================================================

function trainStartLabelText() {
  if (TRAIN.trainType === 'style') return 'Train Style';
  return (TRAIN.voiceEnabled && TRAIN.voiceFile)
    ? 'Train Character + Voice'
    : 'Train Character';
}

function trainUpdateStartLabel() {
  const btn = document.getElementById('trainStartBtn');
  if (!btn) return;
  // Don't override 'Submitting…' mid-request.
  if (btn.textContent === 'Submitting…') return;
  btn.textContent = trainStartLabelText();
}

function trainWireVoice() {
  const drop = document.getElementById('trainVoiceDrop');
  const input = document.getElementById('trainVoiceFileInput');
  if (drop && input) {
    drop.addEventListener('click', (e) => {
      // Don't re-open the picker if the user clicks the remove button or
      // interacts with the <audio> element.
      if (e.target.closest('.train-voice-remove')) return;
      if (e.target.closest('audio')) return;
      input.click();
    });
    drop.addEventListener('dragover', (e) => {
      e.preventDefault();
      drop.classList.add('dragover');
    });
    drop.addEventListener('dragleave', () =>
      drop.classList.remove('dragover'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('dragover');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) trainVoiceUpload(files[0]);
    });
    input.addEventListener('change', () => {
      if (input.files && input.files.length) trainVoiceUpload(input.files[0]);
      input.value = '';
    });
  }
  // Voice preset chips.
  document.querySelectorAll('#trainVoicePresetGroup .pill-btn').forEach(b => {
    b.addEventListener('click', () => {
      TRAIN.voicePreset = b.dataset.voicePreset || 'standard';
      document.querySelectorAll('#trainVoicePresetGroup .pill-btn').forEach(x =>
        x.classList.toggle('active', x === b));
    });
  });
  trainRenderVoice();
}

async function trainVoiceUpload(file) {
  if (!file) return;
  const status = document.getElementById('trainStatus');
  // The voice endpoint requires a dataset (job_id) — the user must drop
  // at least one image first. Guard with a friendly message.
  if (!TRAIN.jobId) {
    if (status) status.textContent =
      'Upload at least one training image before adding a voice clip.';
    return;
  }
  // Quick client-side extension + size check (server re-validates).
  const ext = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
  if (!['.wav', '.mp3', '.m4a', '.flac'].includes(ext)) {
    if (status) status.textContent =
      `Unsupported audio type ${ext} — use WAV / MP3 / M4A / FLAC.`;
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    if (status) status.textContent =
      `Voice clip too large (${(file.size / 1024 / 1024).toFixed(1)} MB) — max 50 MB.`;
    return;
  }
  try {
    const fd = new FormData();
    fd.append('job_id', TRAIN.jobId);
    fd.append('file', file, file.name);
    const r = await fetch('/train/upload-voice', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) {
      if (status) status.textContent =
        'Voice upload failed: ' + (j.error || r.status);
      return;
    }
    // Build a local object URL for the audio preview so we can play
    // immediately without round-tripping through /train/voice-file (which
    // doesn't exist yet — preview is local-only).
    const audioUrl = URL.createObjectURL(file);
    TRAIN.voiceFile = {
      filename: j.filename,
      path: j.path,
      size: j.size,
      // Server-side ffprobe duration — authoritative since it uses the
      // same ffmpeg the trainer uses to decode. The client-side
      // <audio>.duration reading is still a fallback for browsers that
      // can't decode the format inline (rare but possible for FLAC).
      durationSeconds: typeof j.duration_seconds === 'number' ? j.duration_seconds : null,
      audioUrl,
      originalName: file.name,
    };
    TRAIN.voiceEnabled = true;     // auto-on after a successful upload
    trainRenderVoice();
    trainUpdateStartLabel();
    if (status) {
      const sizeMB = (j.size / 1024 / 1024).toFixed(2);
      const dur = TRAIN.voiceFile.durationSeconds;
      // Soft warning band: anything outside the 10–25 s sweet spot but
      // still inside [min, max] passes the upload but gets a friendly
      // nudge. Under min already 400'd at the server.
      let durMsg = '';
      if (typeof dur === 'number') {
        durMsg = ` · ${dur.toFixed(1)} s`;
        if (dur > 30) durMsg += ' (long — consider trimming to 10–25 s)';
        else if (dur < 5) durMsg += ' (short — recommend ≥5 s)';
      }
      status.textContent =
        `Voice clip ready (${sizeMB} MB${durMsg}). ` +
        'Press play above to preview, then start training.';
    }
  } catch (e) {
    if (status) status.textContent =
      'Voice upload failed: ' + (e.message || 'unknown');
  }
}

async function trainVoiceRemove(ev) {
  if (ev) { ev.preventDefault(); ev.stopPropagation(); }
  const f = TRAIN.voiceFile;
  // Free the local object URL before dropping the reference.
  if (f && f.audioUrl) {
    try { URL.revokeObjectURL(f.audioUrl); } catch (_) {}
  }
  if (TRAIN.jobId) {
    try {
      const fd = new URLSearchParams();
      fd.set('train_job_id', TRAIN.jobId);
      await fetch('/train/remove-voice', { method: 'POST', body: fd });
    } catch (e) {
      console.warn('remove-voice error', e);
    }
  }
  TRAIN.voiceFile = null;
  TRAIN.voiceEnabled = false;
  trainRenderVoice();
  trainUpdateStartLabel();
}

function trainVoiceToggleChanged() {
  const cb = document.getElementById('trainVoiceToggle');
  TRAIN.voiceEnabled = !!(cb && cb.checked);
  trainRenderVoice();
  trainUpdateStartLabel();
}

function trainRenderVoice() {
  const drop = document.getElementById('trainVoiceDrop');
  const empty = document.getElementById('trainVoiceEmpty');
  const loaded = document.getElementById('trainVoiceLoaded');
  const audio = document.getElementById('trainVoiceAudio');
  const nameEl = document.getElementById('trainVoiceFilename');
  const metaEl = document.getElementById('trainVoiceMeta');
  const toggle = document.getElementById('trainVoiceToggle');
  const toggleHint = document.getElementById('trainVoiceToggleHint');
  const stepsRow = document.getElementById('trainVoiceStepsRow');
  const f = TRAIN.voiceFile;
  if (f) {
    if (drop) drop.classList.add('has-file');
    if (empty) empty.hidden = true;
    if (loaded) loaded.hidden = false;
    if (audio) {
      if (audio.src !== f.audioUrl) audio.src = f.audioUrl;
    }
    if (nameEl) nameEl.textContent = f.originalName || f.filename;
    if (metaEl) {
      const sizeKB = (f.size / 1024).toFixed(0);
      const sizeStr = f.size > 1024 * 1024
        ? `${(f.size / 1024 / 1024).toFixed(2)} MB`
        : `${sizeKB} KB`;
      // Once the <audio> metadata loads we can append duration.
      metaEl.textContent = sizeStr;
      if (audio) {
        audio.addEventListener('loadedmetadata', () => {
          if (Number.isFinite(audio.duration)) {
            metaEl.textContent = `${audio.duration.toFixed(1)} s · ${sizeStr}`;
          }
        }, { once: true });
      }
    }
    if (toggle) {
      toggle.disabled = false;
      toggle.checked = !!TRAIN.voiceEnabled;
    }
    if (toggleHint) {
      toggleHint.textContent = TRAIN.voiceEnabled
        ? '(adds ~25–125 min depending on preset)'
        : '(uploaded but training skipped)';
    }
    if (stepsRow) stepsRow.hidden = !TRAIN.voiceEnabled;
  } else {
    if (drop) drop.classList.remove('has-file');
    if (empty) empty.hidden = false;
    if (loaded) loaded.hidden = true;
    if (audio) audio.removeAttribute('src');
    if (toggle) {
      toggle.disabled = true;
      toggle.checked = false;
    }
    if (toggleHint) toggleHint.textContent = 'upload a clip to enable';
    if (stepsRow) stepsRow.hidden = true;
  }
}

async function trainRefreshLoraList() {
  const list = document.getElementById('trainLoraList');
  if (!list) return;
  // Keep the global picker in sync. /loras scans the same directory but
  // also rebuilds _knownUserLoras + drops stale entries from _activeLoras.
  // Cheap (one fs scan) and harmless if it races with a render. Letting
  // it run unconditionally so a freshly-trained LoRA appears in T2V's
  // picker the moment the trained list updates — no manual refresh.
  try { if (typeof refreshLoras === 'function') refreshLoras(); } catch (e) {}
  try {
    const r = await fetch('/train/list');
    const j = await r.json();
    const items = (j.loras || []);
    trainRenderVerdictBanner(items);
    if (!items.length) {
      list.innerHTML = '<div class="hint" style="padding:12px 0">No trained LoRAs yet. Start your first run above.</div>';
      return;
    }
    // Compact name-only chips (rewritten 2026-05-18 per Mr Bizarro). Each
    // chip is a link that switches to the Video tab + drops the LoRA
    // into the picker. The previous per-row "Use in T2V / I2V / Copy
    // trigger / delete" chrome moved off this card; deletion lives in
    // the LoRA picker's row controls and the trigger is one click away
    // on the Video tab.
    list.innerHTML = items.map(it => {
      const safeTrig = (it.trigger || '').replace(/[<>&"']/g, c =>
        ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'})[c]);
      const safePath = (it.path || '').replace(/'/g, "\\'");
      const displayName = it.name || it.filename;
      // The verdict rides on the chip so a dead LoRA is visible in the list,
      // not only in the log of the run that made it (#62). `unknown` — every
      // LoRA older than the measurement — is deliberately undecorated:
      // silence is not weakness.
      const verdict = String(it.adapter_verdict || 'unknown').toLowerCase();
      const flagged = (verdict === 'weak' || verdict === 'inert');
      const advice = it.adapter_advice || '';
      const title = flagged
        ? `"${safeTrig}" — ${verdict.toUpperCase()}: ${advice}`
        : `"${safeTrig}" — ${it.size_mb} MB · ${trainFmtAge(Date.now() - (Number(it.created_at) || 0) * 1000)}`;
      const badge = flagged
        ? `<span class="train-lora-verdict">${verdict === 'inert' ? 'DEAD' : 'WEAK'}</span>`
        : '';
      return `<a href="#" class="train-lora-chip${flagged ? ' is-weak' : ''}" title="${escapeHtml(title)}"
        onclick="trainUseInVideo('${safePath}','${safeTrig}','t2v'); return false;">${displayName}${badge}</a>`;
    }).join('');
  } catch (e) {
    list.innerHTML = '<div class="hint">Load failed: ' + (e.message || 'unknown') + '</div>';
  }
}

// The banner above the chips. Speaks only when the most recent training is
// weak or inert — the moment a user is standing in this tab wondering why the
// character they just spent hours on does nothing.
function trainRenderVerdictBanner(items) {
  const el = document.getElementById('trainVerdictBanner');
  if (!el) return;
  const rows = Array.isArray(items) ? items.slice() : [];
  rows.sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0));
  const newest = rows[0];
  const verdict = newest ? String(newest.adapter_verdict || 'unknown').toLowerCase() : 'unknown';
  if (!newest || (verdict !== 'weak' && verdict !== 'inert')) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  const name = escapeHtml(newest.name || newest.filename || 'this adapter');
  const advice = escapeHtml(newest.adapter_advice ||
    'Its learned deltas are too small to visibly change a render.');
  const head = verdict === 'inert'
    ? `<strong>${name} carries nothing.</strong>`
    : `<strong>${name} came out weak.</strong>`;
  el.innerHTML =
    `<div class="train-verdict-head">${head}</div>` +
    `<div class="train-verdict-body">${advice}</div>` +
    `<div class="train-verdict-body train-verdict-how">Measured by Phosphene when the run finished. ` +
    `You can re-check any adapter yourself: <code>./ltx-2-mlx/env/bin/python3.11 lora_compat.py &lt;file&gt;.safetensors</code></div>`;
  el.hidden = false;
}

function trainFmtAge(ms) {
  if (!Number.isFinite(ms) || ms < 0) return 'just now';
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function trainUseInVideo(loraPath, trigger, targetMode) {
  // Switch the panel into the requested video mode + pre-fill the LoRA
  // picker by emitting an "add" against the unified picker's state. The
  // picker's _activeLoras list is what the form serialises into the
  // hidden #lorasJson input on submit. Falls back to copying the path
  // to clipboard if the picker hooks aren't present.
  try { setMode(targetMode); } catch (e) {}
  // Pre-fill the prompt textarea with the trigger word as a starter so
  // the user sees how it's used.
  const prompt = document.getElementById('prompt');
  if (prompt && !prompt.value) {
    prompt.value = `${trigger} man, cinematic medium shot, soft natural light`;
  }
  // Use the existing picker plumbing if available. _activeLoras and
  // renderLorasList come from the unified LoRA picker module.
  if (typeof window._activeLoras !== 'undefined' && Array.isArray(window._activeLoras)) {
    // Go through the picker's own add path: it renders the chip AND
    // serialises #lorasJson, which is what the form actually posts. The
    // previous hand-rolled push rendered the chip but called a serialiser
    // that never existed (_syncLorasJsonField), so a freshly trained
    // character LOOKED applied and rendered without it (review 2026-09-02).
    addLoraToActive({ path: loraPath, strength: 1.0, name: loraPath.split('/').pop() });
  } else {
    // Fall back: copy the path so the user can paste it into the LoRA
    // picker manually.
    if (typeof imgStudioCopyPath === 'function') imgStudioCopyPath(loraPath);
  }
}

function trainCopyTriggerCmd(trigger) {
  if (!trigger) return;
  const text = `${trigger} man, cinematic medium shot, soft natural light`;
  if (typeof imgStudioCopyPath === 'function') imgStudioCopyPath(text);
}

async function trainDeleteLora(loraPath) {
  if (!loraPath) return;
  if (!confirm('Delete this LoRA?\n\n' + loraPath)) return;
  try {
    const fd = new URLSearchParams();
    fd.set('path', loraPath);
    const r = await fetch('/train/delete', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok) {
      alert('Delete failed: ' + (j.error || r.status));
      return;
    }
    trainRefreshLoraList();
  } catch (e) {
    alert('Delete error: ' + (e.message || 'unknown'));
  }
}

// Quality presets (Y1.013) — each one bundles the backend quality value
// (which selects the model + sampler) with the canonical dimensions.
// Backend routes on the cell's `pipeline`, not the key name, so
  // 'quick', 'balanced', and 'standard' all run Q4 distilled — they differ in
// pixel count. The richer label is preserved in the sidecar so the
// info modal can show "Quick" / "Standard" / "High" verbatim.
globalThis.QUALITY_PRESETS = {
  quick:    { w: 640,  h: 448, upscale: 'off' },        // 10:7, fastest sanity check. 448 not 480: the two-stage lane snaps to multiples of 64, so 480 was never delivered.
  balanced: { w: 1024, h: 576, upscale: 'fit_720p' },   // exact 16:9 → 1280×720
  standard: { w: 1280, h: 704, upscale: 'off' },        // LTX-wide canonical render
  // High = Q8 quality at 1024×576. Lab finding 2026-05-09: at 1024×576
  // Q8 produces "outstanding" output at ~7:48 wall vs ~11:51 at
  // 1280×704 — Q8's quality differentiator is per-token detail
  // capacity, not raw pixels. User can still pick 1280×704 explicitly
  // via the resolution chip; this preset just sets the saner default.
  high:     { w: 1024, h: 576, upscale: 'off' },
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
  if (typeof _applyStgRowVisibility === 'function') {
    try { _applyStgRowVisibility(); } catch (_) {}
  }
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
  // H3 owns the render shape while it's the active engine, and setQuality()
  // has several callers (boot, aspect changes, Load Params, the workflow-tab
  // restore) that would otherwise stomp the tier geometry with LTX preset dims
  // and re-arm the LTX upscale. Re-applying here makes every path
  // self-correcting instead of leaving the quick-settings advertising
  // "1024×576 → 1280×720 fit" for a render that ships 768×448.
  if (document.body.dataset.engine === 'h3' && typeof setH3Tier === 'function') {
    try { setH3Tier((document.getElementById('h3_tier') || {}).value); } catch (_) {}
    if (typeof setUpscale === 'function') { try { setUpscale('off'); } catch (_) {} }
  }
  // Repaint the LTX strips LAST, after every field this function owns is
  // settled, so the chips print the shape that is now actually in the form.
  // Both axes are repainted because moving the canvas re-prices every length.
  if (typeof renderTierAxes === 'function') {
    try { renderTierAxes('ltx'); } catch (_) {}
  }
}
// Does the selected quality run the two-stage HQ lane? Reads the SAME registry
// cell the server does (BOOT.ltx.qualities[].pipeline) instead of comparing to
// the literal 'high' — which is what made these gates blind to a second HQ tier
// and would have offered accel/interpolation on High · 720p, where the backend
// refuses them.
function _qualityUsesHq(q) {
  const cells = ((BOOT.ltx || {}).qualities) || [];
  const cell = Array.isArray(cells)
    ? cells.find(c => c && c.key === String(q || ''))
    : cells[String(q || '')];
  return !!(cell && cell.pipeline === 'hq');
}
function setAccel(a) {
  const allowed = !_qualityUsesHq(document.getElementById('quality').value) && currentMode !== 'extend' && currentMode !== 'keyframe';
  const v = allowed ? a : 'off';
  document.getElementById('accel').value = v;
  document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.accel === v));
  updateCustomizeSummary();
  updateDerived();
}
function temporalModeAllowed() {
  const q = document.getElementById('quality').value;
  const mode = document.getElementById('mode').value;
  return !_qualityUsesHq(q) && currentMode !== 'extend' && currentMode !== 'keyframe' && (mode === 't2v' || mode === 'i2v');
}
// ---- ONE SHOT ---------------------------------------------------------------
// One Shot is its own mode (the chip in #modeGroup, setMode('oneshot') in
// boot.js): one continuous shot of 30 s – 2 min written as beats, one per five
// seconds, rendered as parts that continue from each other's last frame. The
// server does the arithmetic (make_job → take_plan); this keeps the form
// honest: the engine's own length pills fold while the mode is open, the
// beats box prefills from the prompt the first time, and the estimate comes
// from /take/estimate. Leaving the mode calls setTakeSeconds(0) so a normal
// clip never carries take_seconds.
const TAKE_CHOICES = [0, 30, 45, 60, 90, 120];
// The last length the user chose, so leaving the mode (which zeroes the
// hidden field — a normal clip never carries take_seconds) and coming back
// reopens on THEIR choice, not on the 1 min default every time.
let _oneshotLastSeconds = 60;
// A part is what continues from the last frame of the one before it: 15 s
// (3 beats) on H3, 10 s (2 beats) on LTX. Read from the body's engine so the
// labels follow the switcher; the server's own number wins in the estimate.
function takePartSeconds(engine) {
  return (engine || document.body.dataset.engine || 'ltx') === 'h3' ? 15 : 10;
}
function setTakeSeconds(s, beats) {
  s = parseInt(s || 0, 10) || 0;
  if (!TAKE_CHOICES.includes(s)) s = 0;
  if (s) _oneshotLastSeconds = s;
  const hid = document.getElementById('take_seconds');
  if (hid) hid.value = String(s);
  document.querySelectorAll('#takeGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.take, 10) === s));
  const row = document.getElementById('beatsRow');
  if (row) row.hidden = !s;
  ['h3LengthGroup', 'ltxLengthGroup', 'temporalGroup', 'windowsRow'].forEach(id => {
    const g = document.getElementById(id);
    if (g) g.classList.toggle('take-locked', !!s);
  });
  const ta = document.getElementById('beats_text');
  if (s && ta) {
    // Only an explicit list (Load Params, the Storyboard) fills the box.
    // It used to prefill from the prompt on its own, which put ONE written
    // beat and five blank holds in front of a user who had only typed a
    // prompt — half a shot of prompt, half of hold. An empty box now means
    // "the prompt carries the whole shot" (make_job), and the Split button
    // is the deliberate way to start from the prompt's sentences.
    if (Array.isArray(beats)) ta.value = beats.join('\n');
    beatsInput();
    takeRefresh();
  } else {
    const b = document.getElementById('beats');
    if (b) b.value = '';
    const est = document.getElementById('takeEstimate');
    if (est) est.textContent = '';
  }
  if (typeof updateCustomizeSummary === 'function') updateCustomizeSummary();
  if (typeof updateDerived === 'function') updateDerived();
}
// The prompt's sentences, one per beat — the Split button's starting point.
function takePrefill(s) {
  const prompt = (document.getElementById('prompt') || {}).value || '';
  const n = Math.max(1, Math.round(s / 5));
  const sents = prompt.split(/(?<=[.!?])\s+/).map(x => x.trim()).filter(Boolean);
  const out = [];
  for (let i = 0; i < n; i++) out.push(sents[i] || '');
  while (out.length && !out[out.length - 1]) out.pop();
  return out.join('\n');
}
// The "Split my prompt into beats" button: the same prefill, on demand, over
// whatever is in the box. Explicit because the automatic one only fills an
// EMPTY box — a user who typed three lines and changed the prompt has no
// other way to start again from it.
function takePrefillClick() {
  const ta = document.getElementById('beats_text');
  const s = parseInt((document.getElementById('take_seconds') || {}).value || '0', 10) || 0;
  if (!ta || !s) return;
  ta.value = takePrefill(s);
  beatsInput();
}
function beatsInput() {
  const ta = document.getElementById('beats_text');
  const out = document.getElementById('beats');
  const hint = document.getElementById('beatsHint');
  if (!ta || !out) return;
  const lines = String(ta.value || '').split('\n').map(x => x.trim());
  while (lines.length && !lines[lines.length - 1]) lines.pop();
  out.value = lines.length ? JSON.stringify(lines) : '';
  const s = parseInt((document.getElementById('take_seconds') || {}).value || '0', 10) || 0;
  const n = Math.round(s / 5);
  if (hint && n) {
    const filled = lines.filter(Boolean).length;
    const extra = lines.length - n;
    // Zero written is not an error: the prompt is the whole shot. Say so,
    // because a box that reads "0 written" next to a Generate button looks
    // like something the user forgot (field report, 2026-09-08).
    hint.textContent = filled === 0
      ? `${n} lines of 5 s · none written — the prompt carries the whole shot; write a line to direct a moment`
      : `${n} lines of 5 s · ${filled} written`
        + (filled < n ? ' · leave a line blank to hold on the scene' : '')
        + (extra > 0 ? ` · ${extra} extra line${extra === 1 ? '' : 's'} will be dropped` : '');
  }
}
let _takeRefreshSeq = 0;
async function takeRefresh() {
  const el = document.getElementById('takeEstimate');
  const s = parseInt((document.getElementById('take_seconds') || {}).value || '0', 10) || 0;
  if (!el || !s) return;
  const engine = document.body.dataset.engine || 'ltx';
  const quality = engine === 'h3'
    ? ((document.getElementById('h3_quality') || {}).value || 'standard')
    : ((document.getElementById('quality') || {}).value || 'balanced');
  const seq = ++_takeRefreshSeq;
  el.textContent = 'working out the time…';
  try {
    const r = await (await fetch(`/take/estimate?engine=${encodeURIComponent(engine)}&quality=${encodeURIComponent(quality)}&seconds=${s}`)).json();
    if (seq !== _takeRefreshSeq) return;
    if (!r.ok) { el.textContent = r.error || ''; return; }
    const eng = r.engine || engine;
    const partSec = takePartSeconds(eng);
    // The server's count when it sends one; the same arithmetic otherwise.
    const parts = parseInt(r.parts, 10) || Math.ceil(s / partSec);
    const bits = [`${r.beats || Math.round(s / 5)} beats`];
    bits.push(`${parts} parts of ${partSec} s that continue from ` + (eng === 'h3' ? 'each other' : 'the last frame'));
    if (r.eta) bits.push(`about ${r.eta.replace(/^~/, '').replace(' · batch', '')} on this Mac`);
    else if (eng === 'ltx') bits.push('about 7 min per 5 s at Quick on this Mac');
    el.textContent = bits.join(' · ');
  } catch (e) { if (seq === _takeRefreshSeq) el.textContent = ''; }
}
// ---- the One Shot panel ----------------------------------------------------
// What setMode('oneshot') opens and every other setMode closes. The mode-level
// bits (the chip, currentMode, the hidden #mode) stay in boot.js; this owns
// the panel: its visibility, the per-engine part labels, the anchor image and
// the two continuity toggles.
function oneshotActive() {
  return (typeof currentMode !== 'undefined') && currentMode === 'oneshot';
}
// The backend mode a One Shot ships under: i2v when an anchor image is set,
// t2v otherwise. The same rule Image mode's submit guard applies (an i2v job
// needs an image), read from the SAME hidden field.
function oneshotBackendMode() {
  const img = (document.getElementById('image') || {}).value || '';
  return String(img).trim() ? 'i2v' : 't2v';
}
function oneshotEnter() {
  const panel = document.getElementById('takeAxes');
  if (panel) panel.hidden = false;
  document.body.classList.add('oneshot-mode');
  oneshotRefreshLabels();
  oneshotSyncAnchor();
  setTakeLightLock((document.getElementById('take_light_lock') || {}).value || 'on');
  setTakeRetake((document.getElementById('take_retake') || {}).value || 'on');
  const cur = parseInt((document.getElementById('take_seconds') || {}).value || '0', 10) || 0;
  setTakeSeconds(cur || _oneshotLastSeconds || 60);
}
function oneshotLeave() {
  const panel = document.getElementById('takeAxes');
  if (panel) panel.hidden = true;
  document.body.classList.remove('oneshot-mode');
  const cur = parseInt((document.getElementById('take_seconds') || {}).value || '0', 10) || 0;
  if (cur) setTakeSeconds(0);
  // The two continuity fields are hidden inputs in the video form, so
  // FormData posts them with EVERY clip. Back to their defaults on the way
  // out, or a normal clip's sidecar carries an "off" that meant nothing.
  setTakeLightLock('on');
  setTakeRetake('on');
}
// "1 min", "1½ min", "30 s" — the length as the chips name it, for the
// footer strip and anywhere else the shot is summarised in one line.
function takeLengthLabel(s) {
  s = parseInt(s || 0, 10) || 0;
  if (s < 60) return `${s} s`;
  const whole = Math.floor(s / 60);
  const rem = s - whole * 60;
  if (!rem) return `${whole} min`;
  if (rem === 30) return `${whole}½ min`;
  return `${whole} min ${rem} s`;
}
// The one-line summary of a One Shot for the footer strip: "1 min · 6 parts
// of 10 s". The parts follow the engine (15 s on H3, 10 s on LTX); the
// server's own count wins in the estimate, this is the label.
function oneshotSummary(s, engine) {
  s = parseInt(s || 0, 10) || 0;
  if (!s) return '';
  const partSec = takePartSeconds(engine);
  const parts = Math.ceil(s / partSec);
  return `${takeLengthLabel(s)} · ${parts} part${parts === 1 ? '' : 's'} of ${partSec} s`;
}
// The part labels on the length chips and the one-line engine note. Called on
// entry and from setEngine, because a part is 15 s on H3 and 10 s on LTX.
function oneshotRefreshLabels() {
  const engine = document.body.dataset.engine || 'ltx';
  const partSec = takePartSeconds(engine);
  document.querySelectorAll('#takeGroup .pill-btn').forEach(b => {
    const s = parseInt(b.dataset.take, 10) || 0;
    const el = b.querySelector('.take-parts');
    if (el && s) {
      const n = Math.ceil(s / partSec);
      // One line, so the chip keeps the Quality strip's height.
      el.textContent = `${n} × ${partSec} s`;
    }
  });
  const note = document.getElementById('takeEngineNote');
  if (note) {
    note.textContent = engine === 'h3'
      ? 'Hailuo H3 — 15-second parts that continue from each other.'
      : 'LTX — 10-second parts that continue from the last frame.';
  }
}
// The anchor image. One hidden field (#image) for Image mode and One Shot,
// one upload path (pickerUploadFile), so the server sees exactly the i2v it
// already knows. This only mirrors that field into the panel's own thumbnail
// and flips the hidden #mode while the mode is open.
function oneshotSyncAnchor() {
  const path = String((document.getElementById('image') || {}).value || '').trim();
  const thumb = document.getElementById('oneshotAnchorThumb');
  const clear = document.getElementById('oneshotAnchorClear');
  const pick = document.getElementById('oneshotAnchorPick');
  const name = document.getElementById('oneshotAnchorName');
  if (thumb) {
    if (path) { thumb.src = `/image?path=${encodeURIComponent(path)}&w=160`; thumb.hidden = false; }
    else { thumb.removeAttribute('src'); thumb.hidden = true; }
  }
  if (clear) clear.hidden = !path;
  if (pick) pick.textContent = path ? 'Change the image' : 'Choose an image';
  if (name) name.textContent = path ? path.split('/').pop() : '';
  if (oneshotActive()) {
    const modeInp = document.getElementById('mode');
    if (modeInp) modeInp.value = oneshotBackendMode();
  }
}
async function oneshotAnchorPicked(input) {
  const f = input && input.files && input.files[0];
  if (!f) return;
  if (typeof pickerUploadFile === 'function') await pickerUploadFile('image', f);
  input.value = '';
  oneshotSyncAnchor();
}
function oneshotClearAnchor() {
  if (typeof pickerSetImage === 'function') pickerSetImage('image', '');
  oneshotSyncAnchor();
}
function _setTakeToggle(hiddenId, groupId, attr, v) {
  v = (String(v || 'on').toLowerCase() === 'off') ? 'off' : 'on';
  const hid = document.getElementById(hiddenId);
  if (hid) hid.value = v;
  document.querySelectorAll(`#${groupId} .pill-btn`).forEach(b =>
    b.classList.toggle('active', b.dataset[attr] === v));
  return v;
}
function setTakeLightLock(v) { return _setTakeToggle('take_light_lock', 'takeLightLockGroup', 'takeLight', v); }
function setTakeRetake(v) { return _setTakeToggle('take_retake', 'takeRetakeGroup', 'takeRetake', v); }

function setTemporalMode(t) {
  const allowed = temporalModeAllowed();
  const v = (allowed && (t === 'fps12_interp24' || t === 'windows')) ? t : 'native';
  document.getElementById('temporal_mode').value = v;
  document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.temporal === v));
  // The per-window prompts belong to ONE of the three answers.
  const row = document.getElementById('windowsRow');
  if (row) row.hidden = (v !== 'windows');
  if (v === 'windows') windowPromptsInput();
  updateCustomizeSummary();
  updateDerived();
}

// One line per window on screen, a JSON array on the wire — the same shape
// the H3 chain posts, so a curl and the form agree. The hint counts the
// windows the current length needs (ltx_windows: 121-frame windows, 112 new
// frames each) so the box says how many lines mean something.
function windowPromptsInput() {
  const ta = document.getElementById('window_prompts_text');
  const out = document.getElementById('window_prompts');
  const hint = document.getElementById('windowsHint');
  if (!ta || !out) return;
  const lines = String(ta.value || '').split('\n').map(s => s.trim());
  while (lines.length && !lines[lines.length - 1]) lines.pop();
  out.value = lines.length ? JSON.stringify(lines) : '';
  if (hint) {
    const f = parseInt(document.getElementById('frames')?.value || '121', 10) || 121;
    const n = f <= 121 ? 1 : 1 + Math.ceil((f - 121) / 112);
    hint.textContent = n <= 1
      ? 'this length fits one window — pick a longer clip'
      : n + ' windows · one line per window · blank holds the last moment';
  }
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
  const allowed = !_qualityUsesHq(document.getElementById('quality').value) && currentMode !== 'extend' && currentMode !== 'keyframe';
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
  // Same guard as setQuality: the H3 tier pins the canvas, so an orientation
  // change (the row is hidden under H3, but boot and Load Params still call
  // this) must not leave the quick-settings advertising dims the render won't
  // use.
  if (document.body.dataset.engine === 'h3' && typeof setH3Tier === 'function') {
    try { setH3Tier((document.getElementById('h3_tier') || {}).value); } catch (_) {}
  }
}

// Compose the right-aligned line in the Customize summary. Reflects the
// current effective state: aspect, custom-dims callout, speed setting.
function updateCustomizeSummary() {
  const el = document.getElementById('customizeSummary');
  if (!el) return;
  // H3 renders on its own geometry and its own export control; every LTX knob
  // summarised below is folded away in that state, so summarising them would
  // describe a render that isn't happening.
  if (document.body.dataset.engine === 'h3') {
    const tier = h3CurrentCell();
    const up = (document.getElementById('h3_upscale') || {}).value || 'off';
    const parts = [tier ? tier.spec : 'Hailuo H3'];
    // Turbo and Steps used to be summarised here. They moved onto the primary
    // surface (#h3PrimaryControls), and a disclosure summary exists to reveal
    // what the disclosure is HIDING — restating two chips the user can see two
    // inches above it makes the line longer and tells them nothing. What is
    // actually folded away on H3 is the export target, so that is what it says.
    if (up === 'fit_720p') parts.push('720p export');
    else if (up === 'fit_1080p') parts.push('1080p export');
    else if (up === 'ltx_x2') parts.push('LTX ×2 after');
    else parts.push('native export');
    el.textContent = parts.join(' · ');
    return;
  }
  const q = document.getElementById('quality').value;
  const w = parseInt(document.getElementById('width').value || 0);
  const h = parseInt(document.getElementById('height').value || 0);
  const aspect = document.getElementById('aspect').value || 'landscape';
  const accel = document.getElementById('accel').value || 'off';
  const upscale = document.getElementById('upscale').value || 'off';
  const parts = [];
  // Aspect (Quick is fixed 4:3, no choice; Standard/High show landscape/vertical).
  if (q === 'quick') parts.push('10:7 · 640×448');
  else parts.push(aspect === 'vertical' ? '9:16' : '16:9');
  // Flag custom dims if they don't match the preset.
  const preset = QUALITY_PRESETS[q] || QUALITY_PRESETS['standard'];
  const vertical = (aspect === 'vertical' && q !== 'quick');
  const expectedW = vertical ? preset.h : preset.w;
  const expectedH = vertical ? preset.w : preset.h;
  if (q !== 'quick' && (w !== expectedW || h !== expectedH)) {
    parts.push(`${w}×${h} custom`);
  }
  // Speed — say something ONLY when a non-default accel is set (an old
  // sidecar restore). "exact speed" used to print whenever accel was off,
  // while the separate HQ Speed control defaulted to Fast — the summary
  // could contradict the open panel. That control is gone (v4.0.5, dead at
  // the engine boundary), and a default deserves no words.
  if (accel === 'boost' || accel === 'turbo') parts.push(accel);
  // The REAL speed control: the folded-away disclosure must name a
  // non-default schedule, because it changes the take, not just the clock.
  if (typeof schedPresetActive === 'function' && schedPresetActive()) {
    parts.push('fast draft · different take');
  }
  // Inspire changes what the reference DOES; the folded-away summary owes
  // that sentence as much as it owes the schedule.
  if (typeof i2vInspireActive === 'function' && i2vInspireActive()
      && currentMode === 'i2v') {
    parts.push('inspire · new shot from the image');
  }
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

function updatePromptPlaceholder() {
  const prompt = document.getElementById('prompt');
  if (!prompt) return;
  const base = 'Describe the scene AND the sound — e.g. wizard in a forest clearing, fireflies spiraling up · low whispered chant, ember crackle, distant owl. Audio is generated jointly with video; without sound cues the model outputs near-silent ambient.';
  const keyframeTwo = 'Describe the full first-to-last transition in one prompt. Include motion, camera, mood, and audio cues; the start/end images anchor the visual endpoints.';
  const keyframeMulti = `One prompt controls the whole ${window._kfMode}-keyframe shot; the Beat at(s) controls define segment timing. Write one continuous action with the beats described in order, plus audio cues.`;
  if (currentMode === 'keyframe') {
    prompt.placeholder = window._kfMode >= 3 ? keyframeMulti : keyframeTwo;
  } else if (currentMode === 'i2v') {
    prompt.placeholder = 'Describe how the reference image should move, plus sound cues. The image anchors frame 0; the prompt directs the full clip.';
  } else if (currentMode === 'oneshot') {
    prompt.placeholder = 'Describe the whole shot once — the subject, the world, the time of day and the sound. Then write one beat per five seconds below; the camera never cuts.';
  } else if (currentMode === 'ingredients') {
    prompt.placeholder = "Describe WHAT'S in the reference sheet — each character, prop, and the location. e.g. a friendly cartoon hedgehog with rounded chestnut fur; a green coiled garden hose; the bright interior of a 'Greenfield' garden store. (The Action field above describes the shot itself.)";
  } else if (currentMode === 'control') {
    prompt.placeholder = "Describe the NEW subject/scene to paint onto the control clip's motion and structure — plus sound cues. e.g. a red origami crane unfolding on a black table · soft paper rustle. The control video drives the composition; this prompt swaps what's in it.";
  } else {
    prompt.placeholder = base;
  }
}

// Mode chip click — keyframe has two visible chips backed by one backend
// mode. The click handler chooses the 2- or 3-frame UI after setMode()
// restores the shared keyframe screen.
document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.onclick = () => {
  setMode(b.dataset.mode);
  if (b.dataset.mode === 'keyframe') {
    const def = b.dataset.kfDefault || '2';
    const fallback = parseInt(document.getElementById('keyframe_count')?.value || '6', 10);
    setKeyframeMode(def === 'multi' ? fallback : parseInt(def, 10));
  }
});
// Remix sub-tool clicks set the REAL backend mode (ingredients/control/restore);
// setMode keeps the parent Remix pill lit + this sub-pill active + the section
// shown. Wired here alongside the #modeGroup handler so both rows behave alike.
document.querySelectorAll('#remixSubGroup .pill-btn').forEach(b => b.onclick = () => {
  // Ingredients needs the 2.3 generation: its IC-LoRA is 2.3-trained and no
  // 2.5 one exists, so on 2.5 the references are silently ignored and the
  // clip costs full two-stage time (owner-reproduced 2026-08-15). Say that
  // instead of letting someone spend 11 GPU-minutes finding out. The server
  // refuses too — this is the polite half.
  if (b.dataset.remix === 'ingredients'
      && (BOOT.ltx || {}).ingredients_available === false) {
    if (typeof phosToast === 'function') {
      phosToast('Ingredients needs the LTX-2.3 generation — its reference '
        + 'adapter has no 2.5 release yet, so on 2.5 your references would '
        + 'be ignored. For reference-guided work here, use Image mode with '
        + 'Inspire.', { kind: 'danger' });
    }
    return;
  }
  setMode(b.dataset.remix);
});
// Paint the Ingredients chip as unavailable on a generation that cannot
// serve it, so the state is visible before the click.
(function _markIngredientsAvailability() {
  const apply = () => {
    if ((BOOT.ltx || {}).ingredients_available !== false) return;
    const chip = document.querySelector('#remixSubGroup [data-remix="ingredients"]');
    if (!chip) return;
    chip.classList.add('disabled');
    chip.title = 'Needs LTX-2.3 — the 2.5 reference adapter is not published '
               + 'yet. Use Image mode with Inspire for reference-guided work '
               + 'on 2.5, or install the 2.3 pack from the Train tab. '
               + '(The same two routes the server names if you get here.)';
    chip.setAttribute('aria-disabled', 'true');
    const sub = chip.querySelector('.mc-sub');
    if (sub) sub.textContent = 'needs LTX-2.3';
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply);
  } else { apply(); }
})();
document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.onclick = () => {
  // Disabled-but-actionable: the High pill becomes a "click to install Q8"
  // CTA when Q8 is missing. Routes to the Models modal so the user lands
  // on the download button with full context (size, current state, etc.).
  if (b.classList.contains('disabled')) {
    if (b.classList.contains('needs-install') && typeof openModelsModal === 'function') {
      openModelsModal();
    }
    return;
  }
  setQuality(b.dataset.quality);
});
document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setAccel(b.dataset.accel); });
document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setTemporalMode(b.dataset.temporal); });
document.querySelectorAll('#upscaleGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscale(b.dataset.upscale); });
document.querySelectorAll('#upscaleMethodGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscaleMethod(b.dataset.method); });
document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.onclick = () => setAspect(b.dataset.aspect));
document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b => b.onclick = () => setExtendMode(b.dataset.extendMode));


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  setTakeSeconds, takePrefill, takePrefillClick, beatsInput, takeRefresh,
  takePartSeconds, takeLengthLabel, oneshotSummary,
  oneshotEnter, oneshotLeave, oneshotBackendMode, oneshotRefreshLabels, oneshotSyncAnchor,
  oneshotAnchorPicked, oneshotClearAnchor, setTakeLightLock, setTakeRetake,
  windowPromptsInput,
  audioStudioInit, audioStudioDurationChanged, audioStudioEnhancePrompt, audioStudioGenerate,
  trainRecommendedPreset, trainUpdatePresetButtons, trainUpdatePresetNote, downloadSampleCharacter,
  charactersInit, charactersRenderChips, charactersOpenCompose, charactersBackToGrid,
  charactersHandleAudioUpload, charactersClearAudio, charactersUpdateStrengthDisplay, charactersSyncStrengthControls,
  charactersGenerate, charactersEscapeHtml, charactersEscapeAttr, _restoreCharacterStrengths,
  charactersLoadParams, trainInit, trainGuidanceDismiss, trainCheckPreflight,
  trainSuggestTrigger, trainClearAll, trainAutoCaption, trainStart,
  trainVoiceRemove, trainVoiceToggleChanged, trainRefreshLoraList, trainRenderVerdictBanner,
  setQuality, _qualityUsesHq, setAccel, setTemporalMode,
  setUpscale, setUpscaleMethod, updateAccelAvailability, updateTemporalAvailability,
  setAspect, updateCustomizeSummary, updatePromptPlaceholder,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  audioStudioClearAudio, charactersPickChip, trainInstall, trainRemoveImage,
  trainUseInVideo,
});
