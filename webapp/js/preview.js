// webapp/js/preview.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ---- One live-preview model, two consumers ----------------------------------
//
// LTX publishes `progress.preview.{url,estimate,total,meaningful,...}`. H3's
// runner-facing progress is intentionally a different schema and can surface
// the same facts as `progress.live_preview`, `preview_url`, `preview_step`,
// etc. Normalize at the poll boundary ONCE. Neither consumer counts estimates
// or owns a timer; `meaningful` remains the engine/server's decision.
function normalizeLivePreview(s, prog) {
  const cur = (s || {}).current || {};
  const params = cur.params || {};
  const engine = String(params.engine || '').toLowerCase();
  let raw = null;
  if (prog && prog.preview && typeof prog.preview === 'object') raw = prog.preview;
  else if (prog && prog.live_preview && typeof prog.live_preview === 'object') raw = prog.live_preview;
  else if (cur.preview && typeof cur.preview === 'object') raw = cur.preview;
  else if (prog && typeof prog.preview === 'string') raw = { url: prog.preview };

  const firstNumber = (...values) => {
    for (const value of values) {
      if (value === null || value === undefined || value === '') continue;
      const n = Number(value);
      if (Number.isFinite(n)) return n;
    }
    return null;
  };
  const url = String(
    (raw && (raw.url || raw.preview_url || raw.image_url || raw.src)) ||
    (prog && prog.preview_url) || cur.preview_url || ''
  );
  const estimate = firstNumber(
    raw && (raw.estimate ?? raw.step ?? raw.preview_step ?? raw.frame),
    prog && prog.preview_step
  );
  const total = firstNumber(
    raw && (raw.total ?? raw.steps ?? raw.preview_total ?? raw.frame_total),
    prog && prog.preview_total
  );
  let meaningful = null;
  if (raw && typeof raw.meaningful === 'boolean') meaningful = raw.meaningful;
  else if (prog && typeof prog.preview_meaningful === 'boolean') meaningful = prog.preview_meaningful;
  // A schema that only publishes preview_url after its own quality gate is
  // already making the meaningful decision server-side. URL presence is the
  // published fact; the browser still does not infer a threshold.
  else if (url) meaningful = true;

  // Both engines can promise a warming state before status.json exists, but
  // only when their bootstrap explicitly declares the installed runner ready.
  // Engines/lanes with no preview stay untouched.
  let eligible = !!(raw || url);
  if (engine === 'ltx') {
    const pstate = ((BOOT.ltx || {}).preview_state) || {};
    const previewModes = ((BOOT.ltx || {}).preview_modes) || [];
    const qualities = ((BOOT.ltx || {}).qualities) || [];
    const cell = Array.isArray(qualities)
      ? qualities.find(c => c && c.key === String(params.quality || ''))
      : qualities[String(params.quality || '')];
    const laneRuns = cell ? Number(cell.preview_every || 0) > 0 : false;
    const mode = String(params.mode || '').toLowerCase();
    const laneDeclared = Array.isArray(previewModes) && previewModes.includes(mode);
    eligible = eligible || (pstate.on !== false && laneRuns && laneDeclared);
  } else if (engine === 'h3') {
    const h3Preview = ((BOOT.h3 || {}).live_preview) || {};
    const h3Modes = ((BOOT.h3 || {}).modes) || [];
    const mode = String(params.mode || '').toLowerCase();
    eligible = eligible || (h3Preview.on === true &&
      Array.isArray(h3Modes) && h3Modes.includes(mode));
  }

  const remaining = firstNumber(
    prog && prog.remaining_sec,
    raw && raw.remaining_sec,
    raw && raw.saves_sec
  );
  const eta = firstNumber(prog && prog.eta_sec, raw && raw.eta_sec);
  const elapsed = firstNumber(prog && prog.elapsed_sec, 0) || 0;
  return {
    eligible,
    available: !!(raw || url),
    engine,
    help: engine === 'h3'
      ? ((((BOOT.h3 || {}).live_preview) || {}).help || '')
      : ((((BOOT.ltx || {}).help) || {}).preview || ''),
    url,
    estimate,
    total,
    meaningful: meaningful === true,
    abortable: !!(raw && raw.abortable),
    saves_sec: firstNumber(raw && raw.saves_sec, remaining),
    remaining_sec: remaining != null ? remaining
      : (eta != null ? Math.max(0, eta - elapsed) : null),
  };
}

function _liveStageMediaHeld() {
  const wrap = document.getElementById('playerWrap');
  const video = wrap ? wrap.querySelector('video') : null;
  const playing = !!(video && !video.paused && !video.ended);
  const recent = Date.now() - Number(window._stagePlaybackIntentAt || 0)
                 < LIVE_STAGE_PLAYBACK_HOLD_MS;
  return playing || recent;
}

function _showLiveReturnChip(label, outputPath) {
  const chip = document.getElementById('liveReturnChip');
  if (!chip) return;
  chip.textContent = label;
  chip.hidden = false;
  chip.classList.toggle('is-done', !!outputPath);
  if (outputPath) chip.dataset.outputPath = outputPath;
  else delete chip.dataset.outputPath;
}

function _hideLiveStageChrome() {
  const overlay = document.getElementById('liveStageOverlay');
  const chip = document.getElementById('liveReturnChip');
  if (overlay) overlay.hidden = true;
  if (chip) { chip.hidden = true; chip.classList.remove('is-done'); delete chip.dataset.outputPath; }
}

function _restoreSelectedOutputAfterLive() {
  const wrap = document.getElementById('playerWrap');
  window._liveStageOwnsPlayer = false;
  _hideLiveStageChrome();
  if (activePath && findOutputByPath(activePath)) {
    selectOutput(activePath);
    return;
  }
  if (!wrap) return;
  wrap.className = 'player-wrap empty';
  wrap.innerHTML = `<div class="ps-empty">
    <div class="ps-empty-icon" aria-hidden="true">
      <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
        <circle cx="28" cy="28" r="22" stroke="currentColor" stroke-width="1.5" stroke-opacity="0.35"/>
        <path d="M23 19 L37 28 L23 37 Z" fill="currentColor" fill-opacity="0.45"/>
      </svg>
    </div>
    <div class="ps-empty-title">No outputs yet</div>
    <div class="ps-empty-sub">Generate something on the left and the result lands here.</div>
  </div>`;
  const surface = wrap.closest('.player-surface');
  if (surface) { surface.removeAttribute('data-orient'); surface.style.removeProperty('--media-aspect'); }
}

function _handoffLiveStageToOutput(path) {
  if (!path || !findOutputByPath(path)) return false;
  window._liveStagePendingOutput = null;
  window._liveStageJobId = null;
  window._liveStageForcedJobId = null;
  // Give the finished take the same grace window as a user-started clip. This
  // matters when the queue has already advanced: the next job's first poll
  // must not replace the completion frame before anyone has seen it.
  window._stagePlaybackIntentAt = Date.now();
  selectOutput(path, { liveHandoff: !!window._liveStageOwnsPlayer });
  return true;
}

function returnToLiveRender() {
  const chip = document.getElementById('liveReturnChip');
  const donePath = chip && chip.dataset.outputPath;
  const video = document.querySelector('#playerWrap video');
  if (video) video.pause();
  window._stagePlaybackIntentAt = 0;
  if (donePath) {
    _handoffLiveStageToOutput(donePath);
    return;
  }
  const s = window.__phosLastStatus || {};
  const cur = s.current;
  if (!cur) return;
  window._liveStageForcedJobId = cur.id;
  renderLiveStage(s, normalizeLivePreview(s, cur.progress || null));
}

function _renderLiveStageFrame(s, preview) {
  const cur = s.current;
  const wrap = document.getElementById('playerWrap');
  const surface = wrap && wrap.closest('.player-surface');
  const overlay = document.getElementById('liveStageOverlay');
  const chip = document.getElementById('liveReturnChip');
  if (!cur || !wrap || !surface || !overlay) return;

  window._liveStageOwnsPlayer = true;
  window._liveStageJobId = cur.id;
  if (chip) chip.hidden = true;
  document.getElementById('playerOverlayTop').style.display = 'none';
  document.getElementById('playerOverlayActions').style.display = 'none';
  // The stage takes the RENDER'S OWN shape, not a hardcoded 16:9 — the job's
  // geometry is known before the first frame exists, so a vertical take gets
  // the tall portrait stage from the warming state on, exactly the shape the
  // finished clip will claim (same data-orient/--media-aspect idiom the
  // metadata-load path uses; owner-reported: vertical previews rendered as a
  // small strip inside a landscape box).
  const _lw = parseInt((cur.params || {}).width, 10) || 16;
  const _lh = parseInt((cur.params || {}).height, 10) || 9;
  surface.style.setProperty('--media-aspect', `${_lw} / ${_lh}`);
  if (_lh > _lw) surface.setAttribute('data-orient', 'vertical');
  else surface.removeAttribute('data-orient');
  wrap.classList.remove('empty');
  wrap.classList.add('live-stage');
  wrap.dataset.liveJobId = String(cur.id);

  const stopping = window._stopEarlyRequested === cur.id;
  wrap.classList.toggle('is-aborting', stopping);
  if (!preview.meaningful || !preview.url) {
    if (wrap.dataset.liveState !== 'warming') {
      wrap.innerHTML = `<div class="live-stage-warming">
        <svg class="live-stage-warming-mark" viewBox="0 0 24 16" fill="none" aria-hidden="true">
          <rect x="1" y="1" width="22" height="14" rx="2" stroke="currentColor" stroke-width="1"/>
          <path d="M1 5h22M1 11h22M6 1v14M18 1v14" stroke="currentColor" stroke-width=".8" opacity=".72"/>
        </svg>
        <strong>Finding the shot…</strong>
        <span>The first useful estimate will appear here.</span>
      </div>`;
      wrap.dataset.liveState = 'warming';
    }
  } else {
    let img = wrap.querySelector('.live-stage-image');
    if (!img) {
      wrap.innerHTML = '<img class="live-stage-image" alt="Live render preview">';
      img = wrap.querySelector('.live-stage-image');
    }
    if (img.getAttribute('src') !== preview.url) img.setAttribute('src', preview.url);
    wrap.dataset.liveState = 'meaningful';
  }

  const badge = document.getElementById('liveStageBadge');
  const title = document.getElementById('liveStageTitle');
  const eta = document.getElementById('liveStageEta');
  const stop = document.getElementById('liveStageStop');
  overlay.hidden = false;
  if (badge) {
    badge.textContent = 'LIVE';
    badge.title = preview.help || 'Live render preview';
  }
  const step = preview.estimate != null && preview.total != null
    ? ` · step ${preview.estimate}/${preview.total}` : '';
  if (title) title.textContent = preview.meaningful
    ? `forming take${step}` : 'forming take · warming';
  if (eta) {
    eta.textContent = stopping ? 'Finishing the current step, then stopping.'
      : (preview.remaining_sec != null && preview.remaining_sec > 0
          ? `~${fmtMin(preview.remaining_sec)} left`
          : 'ETA settling…');
  }
  if (stop) {
    stop.hidden = !(preview.meaningful && preview.abortable);
    stop.disabled = stopping;
    stop.textContent = stopping ? 'Stopping…' : 'Stop early';
  }
}

function renderLiveStage(s, preview) {
  const currentId = (s.running && s.current) ? s.current.id : null;
  const priorId = window._liveStageJobId;

  // Capture the output before considering the next queued job. If the user is
  // still watching another clip, hold it as a DONE chip; otherwise hand off
  // immediately from the last estimate to the real mp4.
  if (priorId && priorId !== currentId) {
    const ended = (s.history || []).find(j => j && j.id === priorId);
    if (ended && ended.status === 'done' && ended.output_path) {
      window._liveStagePendingOutput = { id: priorId, path: ended.output_path, since: Date.now() };
    } else if (ended && window._liveStageOwnsPlayer) {
      _restoreSelectedOutputAfterLive();
    }
    if (ended) window._liveStageJobId = null;
  }

  const pending = window._liveStagePendingOutput;
  if (pending) {
    // GIVE-UP PATH. Every branch below returns, and the flag was cleared only
    // by a SUCCESSFUL handoff — so a finished clip that never surfaces in the
    // outputs list (hidden, deleted, or outside the 60-item /status window)
    // parked the stage on "preparing finished take" for the rest of the
    // session, and suppressed the live preview of every later render in the
    // batch. 20 s is ten polls: far beyond the deliberate two-second
    // list_outputs cutoff this state exists to bridge, and short enough that
    // a batch's next render loses at most its warm-up to the stale card.
    if (Date.now() - (pending.since || 0) > 20000) {
      window._liveStagePendingOutput = null;
      if (window._liveStageOwnsPlayer) _restoreSelectedOutputAfterLive();
      else _hideLiveStageChrome();
    } else if (window._liveStageOwnsPlayer || !_liveStageMediaHeld()) {
      if (_handoffLiveStageToOutput(pending.path)) {
        // A newly-queued job may already be running. Its preview is handled on
        // the next poll, after the finished video's real playback state exists.
        return;
      }
      // list_outputs intentionally withholds a freshly-written mp4 for two
      // seconds. Keep the last forming pixels mounted during that safety gap;
      // restoring the old selected clip here would create the exact black
      // flash this handoff exists to remove.
      if (window._liveStageOwnsPlayer) {
        const badge = document.getElementById('liveStageBadge');
        const title = document.getElementById('liveStageTitle');
        const eta = document.getElementById('liveStageEta');
        const stop = document.getElementById('liveStageStop');
        if (badge) badge.textContent = 'DONE';
        if (title) title.textContent = 'preparing finished take';
        if (eta) eta.textContent = 'Loading the full clip…';
        if (stop) stop.hidden = true;
      }
      return;
    } else {
      _showLiveReturnChip('DONE · view finished take', pending.path);
      return;
    }
  }

  if (!currentId || !preview || !preview.eligible) {
    if (window._liveStageOwnsPlayer) _restoreSelectedOutputAfterLive();
    else _hideLiveStageChrome();
    return;
  }

  window._liveStageJobId = currentId;
  const forced = window._liveStageForcedJobId === currentId;
  if (!window._liveStageOwnsPlayer && !forced && _liveStageMediaHeld()) {
    const label = preview.meaningful ? 'LIVE · return to render'
                                     : 'LIVE · render warming';
    _showLiveReturnChip(label, '');
    return;
  }
  window._liveStageForcedJobId = null;
  _renderLiveStageFrame(s, preview);
}

// Native video controls do not call selectOutput() when playback begins. A
// delegated intent stamp covers play/pause/seeking clicks and keyboard use;
// the actual `video.paused` state remains the stronger hold while it plays.
document.addEventListener('pointerdown', (event) => {
  if (event.target && event.target.matches('#playerWrap video')) {
    window._stagePlaybackIntentAt = Date.now();
  }
}, true);
document.addEventListener('keydown', (event) => {
  if (event.target && event.target.matches('#playerWrap video') &&
      [' ', 'Enter', 'k', 'K', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
    window._stagePlaybackIntentAt = Date.now();
  }
}, true);

// ---- The live preview, in the Now card --------------------------------------
//
// Four states, and the transitions between them are what the copy is for:
//
//   warming          a pulsing film-frame glyph. The first estimates are still
//                    essentially noise and there is nothing worth judging, so
//                    there is deliberately NO Stop-early button yet — a stop
//                    over a noise field is a trap that aborts a good take.
//   first-meaningful the live image, and Stop early appears with what it saves.
//   aborting         frozen on the last frame, the button disabled.
//   aborted          the card takes over (see the `stopped` branch in poll()).
//
// `meaningful` is the SERVER's call — see _preview_progress(). The client
// renders it; it never counts estimates.
function renderNowPreview(s, prog, previewData) {
  const box = document.getElementById('nowThumb');
  const actions = document.getElementById('nowCardActions');
  if (!box) return;
  const prev = previewData && previewData.available ? previewData : null;
  // THE THUMB FOLLOWS THE RENDER'S SHAPE. A fixed 16:9 box center-cropped a
  // vertical (9:16) preview into a landscape strip of torso — the render's own
  // params say what shape it is, so the box matches it, clamped so a portrait
  // thumb widens the card by nothing and grows it by at most ~2x.
  {
    const jp = ((s.current || {}).params) || {};
    const jw = parseInt(jp.width, 10), jh = parseInt(jp.height, 10);
    if (jw > 0 && jh > 0) {
      const ar = Math.min(16 / 9, Math.max(9 / 16, jw / jh));
      box.style.aspectRatio = String(ar);
    } else {
      box.style.aspectRatio = '';
    }
  }
  // A MISSING DECODER MUST SAY SO. Without this, a render on an install whose
  // 22 MB decoder never arrived looks identical to one where the user switched
  // the preview off: nothing appears, and a feature the release announced simply
  // does not happen. Only the missing-decoder case speaks — "off" was the user's
  // own choice and needs no announcement.
  const pstate = ((BOOT.ltx || {}).preview_state) || {};
  // SCOPED TO THE JOB IT IS ABOUT. renderNowPreview() runs for EVERY active job,
  // so this condition — "something is running, there is no preview, the decoder
  // is missing" — announced an LTX live-preview failure over H3 renders, image
  // jobs and training runs, complete with an Install link for a decoder those
  // jobs would never have used. The capability declares which engines it serves
  // (required_files.json → capabilities.live_preview.engines) and the running
  // job says which engine it is, so the notice only appears where it is true.
  const capEngines = (((BOOT.ltx || {}).capabilities || {}).live_preview || {}).engines || ['ltx'];
  const _p = ((s.current || {}).params) || {};
  // A POSITIVE MATCH, never a default. `engine || 'ltx'` meant every job that
  // does not carry the field — image jobs never do, it is a video field —
  // counted as LTX and got an LTX live-preview failure notice with an Install
  // link for a decoder they would never use. Missing now means "not LTX".
  const jobEngine = String(_p.engine || '').toLowerCase();
  const jobMode = String(_p.mode || '').toLowerCase();
  // And only the modes that actually run the preview lane. Image and training
  // jobs are not video renders; `preview_every` on the quality cell is the
  // server's own answer for the LTX paths (HQ tiers that never call
  // _live_preview_params say so here) — when the table is absent we do not
  // guess, we stay quiet.
  const _qs = ((BOOT.ltx || {}).qualities) || [];
  const _previewModes = ((BOOT.ltx || {}).preview_modes) || [];
  const _cell = Array.isArray(_qs)
    ? _qs.find(c => c && c.key === String(_p.quality || ''))
    : _qs[String(_p.quality || '')];
  const laneRuns = _cell ? Number(_cell.preview_every || 0) > 0 : false;
  const previewServesThisJob =
        jobEngine !== '' && capEngines.indexOf(jobEngine) !== -1
     && Array.isArray(_previewModes) && _previewModes.indexOf(jobMode) !== -1
     && laneRuns;
  // A REASON SWITCH, not a decoder special-case. There are two silent absences
  // now, and they need DIFFERENT actions: a missing decoder is a download, a
  // stale engine is an Update. Sending someone whose engine predates the
  // feature to the Models modal would have them install a 22 MB decoder they
  // already have and watch nothing change.
  //
  // `off` is still not in this table on purpose — that absence was the user's
  // own choice and needs no announcement.
  // A REASON TABLE, not a decoder special-case. There are two silent absences
  // now and they need DIFFERENT actions: a missing decoder is a download, a
  // stale engine is an Update. Sending someone whose engine predates the
  // feature to the Models modal would have them install a 22 MB decoder they
  // already have and watch nothing change.
  //
  // A reason MAY have no CTA. `stale_engine` deliberately has none: the Update
  // button lives in the Pinokio sidebar, not in this page, so every link the
  // panel could offer is a dead end — and a dead-end link reads as "we handled
  // it" when nothing has been handled. The note carries the instruction.
  //
  // `off` is absent on purpose: that absence was the user's own choice.
  const PREVIEW_ABSENCE_CTA = {
    missing_decoder: { label: 'Install it', run: 'openModelsModal()' },
    stale_engine: null,
  };
  // H3 gets the SAME courtesy, from its own bootstrap block. Its absence has
  // a different cause and therefore a different sentence: the panel side is
  // complete (schema, adapter, per-job live dir) but no published H3 runner
  // implements `--live-preview`, so nothing can publish frames yet. Silence
  // here read as "the theater preview is broken on MiniMax" — it is not
  // broken, it is not built on that half yet, and saying so is the fix
  // available today. No CTA: there is nothing for the user to click.
  const h3state = ((BOOT.h3 || {}).live_preview) || {};
  const h3Modes = ((BOOT.h3 || {}).modes) || [];
  const h3Speaks = s.running && !prev && jobEngine === 'h3'
                && Array.isArray(h3Modes) && h3Modes.indexOf(jobMode) !== -1
                && h3state.on !== true && !!h3state.note;
  const speaks = (s.running && !prev && previewServesThisJob
              && Object.prototype.hasOwnProperty.call(PREVIEW_ABSENCE_CTA,
                                                      pstate.reason))
              || h3Speaks;
  let miss = document.getElementById('nowPreviewMissing');
  if (speaks) {
    if (!miss) {
      miss = document.createElement('div');
      miss.id = 'nowPreviewMissing';
      miss.className = 'now-preview-missing';
      box.parentNode.insertBefore(miss, box.nextSibling);
    }
    const cta = h3Speaks ? null : PREVIEW_ABSENCE_CTA[pstate.reason];
    miss.innerHTML = escapeHtml((h3Speaks ? h3state.note : pstate.note) || '') + (cta
      ? ` <a href="#" onclick="event.preventDefault();${cta.run}">`
        + escapeHtml(cta.label) + '</a>'
      : '');
  } else if (miss) {
    miss.remove();
  }
  if (!s.running || !prev) {
    box.hidden = true;
    box.className = 'now-thumb';
    box.innerHTML = '';
    const oldDot = document.querySelector('.now-thumb-help');
    if (oldDot) oldDot.remove();
    const oldNote = document.getElementById('nowThumbHelpNote');
    if (oldNote) oldNote.remove();
    if (actions && actions.dataset.stopEarly === '1') {
      actions.innerHTML = ''; delete actions.dataset.stopEarly;
    }
    return;
  }
  box.hidden = false;
  const stopping = window._stopEarlyRequested === s.current.id;
  // The copy constraint, on the element it constrains. §4.1: "The UI must never
  // invite a face judgement from it." Rendered once per render rather than per
  // poll so the button does not flicker under the 1.5 s cadence.
  if (prev.help && !box.parentNode.querySelector('.now-thumb-help')) {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'help-dot now-thumb-help';
    dot.title = 'What am I looking at?';
    dot.setAttribute('aria-expanded', 'false');
    dot.setAttribute('aria-controls', 'nowThumbHelpNote');
    dot.textContent = '?';
    dot.onclick = toggleNowThumbHelp;
    box.parentNode.insertBefore(dot, box.nextSibling);
    const note = document.createElement('div');
    note.className = 'h3-winhelp';
    note.id = 'nowThumbHelpNote';
    note.hidden = true;
    note.textContent = prev.help;
    box.parentNode.appendChild(note);
  } else if (!prev.help) {
    const oldDot = box.parentNode.querySelector('.now-thumb-help');
    const oldNote = document.getElementById('nowThumbHelpNote');
    if (oldDot) oldDot.remove();
    if (oldNote) oldNote.remove();
  } else {
    const note = document.getElementById('nowThumbHelpNote');
    if (note && note.textContent !== prev.help) note.textContent = prev.help;
  }
  if (!prev.meaningful) {
    box.className = 'now-thumb is-warming';
    box.innerHTML = '';
  } else {
    box.className = 'now-thumb' + (stopping ? ' is-aborting' : '');
    // Replace the src in place rather than the node, so the browser does not
    // flash white between estimates.
    let img = box.querySelector('img');
    if (!img) { img = document.createElement('img'); img.alt = ''; box.appendChild(img); }
    if (img.getAttribute('src') !== prev.url) img.setAttribute('src', prev.url);
  }
  // The second line of #nowDetail, and the button, both follow `meaningful`.
  const meta = document.querySelector('#nowCard .meta');
  if (meta) {
    const saves = (prev.saves_sec != null && prev.saves_sec > 0)
      ? ` — saves about ${fmtMin(prev.saves_sec)}` : '';   // absent -> drop the sentence
    const line = stopping
      ? 'Finishing the current step, then stopping.'
      : (prev.meaningful
          ? `This is the shot it's making. Stop now if it's wrong${saves}.`
          : 'Finding the shot…');
    meta.innerHTML += `<br><span style="color:var(--muted)">${escapeHtml(line)}</span>`;
  }
  if (actions && prev.abortable) {
    if (stopping) {
      actions.dataset.stopEarly = '1';
      actions.innerHTML = `<button type="button" class="qchip" disabled>Stopping…</button>`;
    } else if (actions.dataset.stopEarly !== '1' || !actions.querySelector('[data-action="stop-early"]')) {
      actions.dataset.stopEarly = '1';
      actions.innerHTML =
        `<button type="button" class="qchip" data-action="stop-early" ` +
        `title="Stops this render now. Nothing is saved — the clip was never finished.">Stop early</button>`;
    }
  }
}

// Fills lazily from the Python-owned constant, the same four-hop path the voice
// help takes.
function toggleNowThumbHelp() {
  const note = document.getElementById('nowThumbHelpNote');
  const btn = document.querySelector('.now-thumb-help');
  if (!note || !btn) return;
  if (!note.textContent) {
    note.textContent = ((BOOT.ltx || {}).help || {}).preview || '';
  }
  const open = note.hidden;
  note.hidden = !open;
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// The house rule: native confirm() for anything destructive, and it says what
// is lost. The minutes come from the server's own remaining estimate; when it
// has none, the sentence is DROPPED rather than guessed.
async function stopEarly() {
  const s = window.__phosLastStatus || {};
  const cur = s.current;
  if (!cur) return;
  const prev = normalizeLivePreview(s, cur.progress || null) || {};
  const saves = (prev.saves_sec != null && prev.saves_sec > 0)
    ? `About ${fmtMin(prev.saves_sec)} of work is dropped. ` : '';
  if (!confirm('Stop this render?\n\n' +
      "The shot you're looking at is a preview — it was never finished, so nothing is saved.\n" +
      saves + 'The queue carries on with the next job.')) return;
  window._stopEarlyRequested = cur.id;
  try {
    await api('/stop?mode=early', 'POST');
  } catch (e) {
    window._stopEarlyRequested = null;
    phosToast(String((e && e.message) || e), { kind: 'danger', duration: 6000 });
  }
}

// ---- No voice, defaulted rather than guessed --------------------------------
//
// The storyboard lane DERIVES this exactly, from the planner's <d> tags. The
// Manual tab cannot: a typed prompt has no markup, and a panel that claimed to
// know whether you meant someone to speak would be wrong often enough to be
// worse than useless. So it does the one honest thing available — it sets a
// default and SAYS IT DID.
//
// NO TRI-STATE. A checkbox that is checked is checked; the ` · auto` suffix
// says who checked it, and one click removes the suffix permanently for the
// session. That is the whole disclosure.
globalThis.SPEECH = /(^|\s)says\b|<d>|["“](?=[^"”]{3,})/i;
let _noVoiceTouched = false;
function markNoVoiceTouched() {
  _noVoiceTouched = true;
  const lbl = document.getElementById('noVoiceLabel');
  if (lbl) lbl.textContent = 'No voice';
  const pill = document.getElementById('noVoicePill');
  if (pill) pill.title = "Skip the character's voice LoRA for this render. The face still locks, but audio stays ambient — no speech, no gibberish.";
}
function refreshNoVoiceAuto() {
  // Only while the pill is untouched THIS SESSION. Once the user has an
  // opinion, their choice is frozen and the prompt stops moving it.
  if (_noVoiceTouched) return;
  const cb = document.getElementById('noVoice');
  const pill = document.getElementById('noVoicePill');
  const lbl = document.getElementById('noVoiceLabel');
  if (!cb || !pill || !lbl || pill.hidden) return;
  const prompt = (document.getElementById('prompt') || {}).value || '';
  const hasSpeech = SPEECH.test(prompt);
  cb.checked = !hasSpeech;
  lbl.textContent = hasSpeech ? 'No voice' : 'No voice · auto';
  pill.title = "Skip the character's voice LoRA for this render. The face still locks, but audio stays ambient — no speech, no gibberish."
    + (hasSpeech ? '' : '\nSet on its own because there are no spoken lines in the prompt — click to override.');
}
(function wireNoVoiceAuto() {
  const attach = () => {
    const p = document.getElementById('prompt');
    if (!p || p.__noVoiceWired) return;
    let t = null;
    p.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(refreshNoVoiceAuto, 500);
    });
    p.__noVoiceWired = true;
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else { attach(); }
})();

// ---- The two halves of a character's strength -------------------------------
// One writer for both hidden fields, so the collapsed slider, the two split
// sliders and every `output` beside them can never disagree about what will
// render.
//
// COLLAPSED BEHAVIOUR, and it is deliberate: moving the single slider moves the
// FACE only. The voice keeps its own value and does not track. A user who never
// opens the disclosure gets face 1.0 / voice 1.0 — the graded pair — and
// dragging "strength" changes the thing that word means to them (how much like
// the trained person it looks), not the audio. The hotter voice is one
// disclosure away, and the label carries the pair the moment they differ.
function setCharStrength(which, value) {
  const v = Math.max(0, Math.min(2, parseFloat(value)));
  if (!Number.isFinite(v)) return;
  const id = (which === 'voice') ? 'characterVoiceStrength' : 'characterStrength';
  const el = document.getElementById(id);
  if (el) el.value = v.toFixed(2);
  // Every visible control for this half, wherever it is on screen.
  const sel = (which === 'voice')
    ? ['#charVoiceOut']
    : ['#charFaceOut', '#charFaceSplitOut'];
  sel.forEach(s => { const o = document.querySelector(s); if (o) o.value = v.toFixed(1); });
  document.querySelectorAll('.chars-inline-strength input[type="range"]').forEach(r => {
    if (which === 'face' && parseFloat(r.value) !== v) r.value = v.toFixed(2);
  });
  const row = document.getElementById('charSplitRow');
  if (row) {
    const lines = row.querySelectorAll('.chars-split-line input[type="range"]');
    const idx = (which === 'voice') ? 1 : 0;
    if (lines[idx] && parseFloat(lines[idx].value) !== v) lines[idx].value = v.toFixed(2);
  }
  // The toggle's own label carries the pair once they differ, so a non-default
  // voice is legible without opening the disclosure.
  const btn = document.getElementById('charSplitBtn');
  if (btn) {
    const f = parseFloat((document.getElementById('characterStrength') || {}).value || '1.0');
    const s = parseFloat((document.getElementById('characterVoiceStrength') || {}).value || '1.0');
    btn.textContent = (Math.abs(s - f) > 0.001)
      ? `split · ${f.toFixed(1)} / ${s.toFixed(1)}` : 'split';
  }
}
function toggleCharSplit() {
  const row = document.getElementById('charSplitRow');
  const btn = document.getElementById('charSplitBtn');
  if (!row || !btn) return;
  const open = row.hidden;
  row.hidden = !open;
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
// The doctrine, filled lazily from the Python-owned string so the sentence
// explaining the mechanism ships with the mechanism.
function toggleCharVoiceHelp() {
  const note = document.getElementById('charVoiceHelpNote');
  const btn = document.getElementById('charVoiceHelpBtn');
  if (!note || !btn) return;
  if (!note.textContent) {
    note.textContent = ((BOOT.ltx || {}).help || {}).voice_strength || '';
  }
  const open = note.hidden;
  note.hidden = !open;
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// ============== Manage characters modal (2026-05-18) ==============
// Pencil icon in the avatar strip opens this modal. Lists every
// trained character with rename + delete actions. Rename only touches
// the bundle.json display name; delete removes face+audio LoRAs +
// sidecars + voice clip + avatar cache (NOT the training dataset).
function openCharactersManageModal() {
  const modal = document.getElementById('charactersManageModal');
  if (!modal) return;
  modal.style.display = 'flex';
  // Use a fresh /characters fetch so deletes/renames in another tab
  // are reflected immediately.
  _renderCharactersManageList();
}

function closeCharactersManageModal() {
  const modal = document.getElementById('charactersManageModal');
  if (modal) modal.style.display = 'none';
}

async function _renderCharactersManageList() {
  const list = document.getElementById('charactersManageList');
  if (!list) return;
  list.innerHTML = '<div class="hint">Loading…</div>';
  try {
    const data = await (await fetch('/characters')).json();
    const chars = (data && Array.isArray(data.characters)) ? data.characters : [];
    if (!chars.length) {
      list.innerHTML = `<div class="hint" style="padding:18px 4px">
        No trained characters yet — train one in the
        <a href="#" onclick="closeCharactersManageModal(); workflowSwitch('train'); return false;">Train tab</a>.
      </div>`;
      return;
    }
    list.innerHTML = chars.map(c => {
      const name = c.name || c.trigger || c.id;
      const trigger = c.trigger || c.id || '';
      const initial = (name || '?').charAt(0).toUpperCase();
      const avatar = c.sample_image_url
        ? `<img class="cm-avatar" src="${escapeHtml(c.sample_image_url)}" alt="">`
        : `<span class="cm-avatar-ph">${escapeHtml(initial)}</span>`;
      const idAttr = JSON.stringify(c.id).replace(/"/g, '&quot;');
      return `<div class="chars-manage-row" data-cid="${escapeHtml(c.id)}">
        ${avatar}
        <input type="text" class="cm-name-input"
               value="${escapeHtml(name)}" maxlength="120"
               data-original="${escapeHtml(name)}">
        <span class="cm-trigger" title="Trigger word — baked into the trained weights, can't be renamed">${escapeHtml(trigger)}</span>
        <span class="cm-actions">
          <button type="button" class="cm-btn cm-btn-save"
                  onclick="_charactersManageSave(${idAttr}, this)">Save</button>
          <button type="button" class="cm-btn cm-btn-delete"
                  onclick="_charactersManageDelete(${idAttr}, this)">Delete</button>
        </span>
      </div>`;
    }).join('');
  } catch (e) {
    list.innerHTML = `<div class="hint">Load failed: ${escapeHtml(e.message || 'unknown')}</div>`;
  }
}

async function _charactersManageSave(cid, btn) {
  const row = btn.closest('.chars-manage-row');
  const input = row && row.querySelector('.cm-name-input');
  if (!row || !input) return;
  const newName = (input.value || '').trim();
  if (!newName) {
    alert('Name cannot be empty.');
    input.focus();
    return;
  }
  if (newName === input.dataset.original) return;  // no-op
  btn.disabled = true;
  try {
    const fd = new URLSearchParams();
    fd.set('name', newName);
    const r = await fetch(`/characters/${encodeURIComponent(cid)}/rename`,
                         { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      alert('Rename failed: ' + (data.error || `HTTP ${r.status}`));
      btn.disabled = false;
      return;
    }
    input.dataset.original = newName;
    btn.textContent = 'Saved';
    setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 900);
    // Refresh the inline avatar strip + LoRA picker so the new name
    // appears immediately on the main form.
    try { refreshManualCharacters(); } catch (_) {}
  } catch (e) {
    alert('Rename failed: ' + (e.message || e));
    btn.disabled = false;
  }
}

async function _charactersManageDelete(cid, btn) {
  const row = btn.closest('.chars-manage-row');
  const nameInput = row && row.querySelector('.cm-name-input');
  const displayName = nameInput ? nameInput.value : cid;
  if (!confirm(`Delete "${displayName}"?\n\nThis removes the face LoRA, audio LoRA, voice clip, and avatar cache. Your training dataset under state/train_character/ is kept so you can retrain.`)) {
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Deleting…';
  try {
    const r = await fetch(`/characters/${encodeURIComponent(cid)}/delete`,
                         { method: 'POST' });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      alert('Delete failed: ' + (data.error || `HTTP ${r.status}`));
      btn.disabled = false;
      btn.textContent = 'Delete';
      return;
    }
    // Remove the row from the modal + refresh the avatar strip.
    if (row) row.remove();
    // If the deleted character was the active selection, clear it so
    // the form doesn't ship a stale character_id on next submit.
    if (_selectedCharacterId === cid) {
      _selectedCharacterId = '';
      const inp = document.getElementById('characterIdInput');
      if (inp) inp.value = '';
    }
    try { refreshManualCharacters(); } catch (_) {}
  } catch (e) {
    alert('Delete failed: ' + (e.message || e));
    btn.disabled = false;
    btn.textContent = 'Delete';
  }
}

// The NON-TOGGLING setter — the one safe hydrator (journey audit's fix for
// the owner-reported cast-restore failure). Sets the selection to exactly
// `id` ('' clears), runs the FULL cascade — hidden input, avatar ring,
// quality-strip swap, applied-note/No-voice sync — and never injects the
// trigger into the prompt (Load Params restores the prompt verbatim from
// the sidecar, which already carries it). Click handlers decide toggle
// semantics; hydrators call this directly.
function applyCharacterSelection(id, opts) {
  opts = opts || {};
  _selectedCharacterId = id || '';
  const inp = document.getElementById('characterIdInput');
  if (inp) inp.value = _selectedCharacterId;
  // Trigger injection is CLICK behaviour, not hydration behaviour.
  if (_selectedCharacterId && opts.injectTrigger) {
    const c = _manualCharacters.find(x => x.id === _selectedCharacterId);
    if (c && c.trigger && typeof appendTriggerToPrompt === 'function') {
      try { appendTriggerToPrompt(c.trigger); } catch (_) {}
    }
  }
  _renderManualCharactersList();
  // Swap quality strips: when a character is active, the Q4 distilled
  // tiers (Quick / Balanced / Standard) don't match the dev-trained
  // LoRA and produce a soft, generic-Trump-ish render. Only the Q8 HQ
  // path lines up. Hide the default 4-chip strip + reveal the 2-chip
  // character-only strip (Q8 Draft / Q8 Pro). When the character is
  // deselected, restore the default and revert to Balanced. Called
  // here AND after refreshManualCharacters() boot so the visibility
  // state is correct even on first paint.
  _applyCharacterQualityStripVisibility();
}

function selectManualCharacter(id) {
  // Click-to-toggle: clicking the currently-active avatar deselects it
  // (2026-05-18 round 3 — the explicit "None" chip is gone; this is
  // how the user clears a selection now). Passing id === '' also
  // deselects, kept for callsites that always want to clear. The actual
  // state change + cascade live in applyCharacterSelection.
  const next = (id === _selectedCharacterId) ? '' : (id || '');
  applyCharacterSelection(next, { injectTrigger: true });
}

// Toggle the visibility of the default vs character-only quality strips
// based on whether a character is currently selected. Idempotent — safe
// to call any time the selection state changes.
function _applyCharacterQualityStripVisibility() {
  // Q4 tier keeps the regular quality pills (Quick/Balanced/Standard) —
  // no Q8-only character quality strip. The character LoRA fuses into
  // the Q4 distilled base; identity match is imperfect but it renders.
  if (window.PHOSPHENE_CAP_TIER === 'q4') return;
  const def  = document.getElementById('qualityGroup');
  const char = document.getElementById('qualityGroupCharacter');
  const skipWrap = document.getElementById('charSkipstepToggleWrap');
  if (!def || !char) return;
  if (_selectedCharacterId) {
    def.hidden = true;
    char.hidden = false;
    if (skipWrap) skipWrap.hidden = false;
    // Snap to Q8 Pro on first switch — but only if no char-quality
    // chip is currently active (preserves user's choice if they had
    // picked Draft earlier and switched between characters).
    const anyActive = char.querySelector('.char-quality.active');
    const target = anyActive || char.querySelector('[data-char-quality="pro"]');
    if (target) _setCharacterQuality(target, { allowMissing: true });
  } else {
    def.hidden = false;
    char.hidden = true;
    if (skipWrap) skipWrap.hidden = true;
    // Restore Balanced as the default video preset — the same value
    // the page boots with — so the user lands in a sensible state
    // when they deselect the character.
    if (typeof setQuality === 'function') {
      try { setQuality('balanced'); } catch (_) {}
    }
    // (History: a skip-step reset used to live here; the whole HQ Speed
    // control was removed in v4.0.5 after the audit proved it dead at
    // the engine boundary, so there is no skip-step state to manage.)
  }
}

// Apply one server-resolved character row. ``data-quality`` is the real
// pipeline token; ``data-char-quality`` is the UI token recorded in the
// sidecar. Keeping both is what distinguishes Pro from High at 1024×576.
function _setCharacterQuality(btn, opts) {
  if (!btn) return;
  opts = opts || {};
  if (btn.classList.contains('needs-install') && !opts.allowMissing) {
    if (typeof openModelsModal === 'function') openModelsModal();
    return;
  }
  const group = document.getElementById('qualityGroupCharacter');
  if (group) {
    group.querySelectorAll('.char-quality').forEach(b =>
      b.classList.toggle('active', b === btn));
  }
  const qInp = document.getElementById('quality');
  const choiceInp = document.getElementById('quality_choice');
  const wInp = document.getElementById('width');
  const hInp = document.getElementById('height');
  const aspect = document.getElementById('aspect');
  const w = parseInt(btn.dataset.width || '1024', 10);
  const h = parseInt(btn.dataset.height || '576', 10);
  if (qInp) qInp.value = btn.dataset.quality
    || ((BOOT.ltx || {}).character || {}).quality || 'high';
  if (choiceInp) choiceInp.value = btn.dataset.charQuality || 'pro';
  // Honor the orientation chip — swap w/h for vertical renders.
  const vertical = aspect && aspect.value === 'vertical';
  if (wInp) wInp.value = vertical ? h : w;
  if (hInp) hInp.value = vertical ? w : h;
  // (Skip-step application removed in v4.0.5 — the HQ Speed control was
  // dead at the engine boundary and no longer exists.)
  if (typeof setUpscale === 'function') {
    try { setUpscale('fit_720p'); } catch (_) {}
  }
  if (typeof updateCustomizeSummary === 'function') {
    try { updateCustomizeSummary(); } catch (_) {}
  }
  // Quality is now 'high'; reveal the STG slider.
  if (typeof _applyStgRowVisibility === 'function') {
    try { _applyStgRowVisibility(); } catch (_) {}
  }
}

// (v4.0.5) _setSkipStepEnabled, _wireHqSpeedPills and
// _applyHqSpeedRowVisibility are gone with the HQ Speed control: the
// hidden inputs they wrote were dropped at the engine boundary, so the
// entire cluster managed state that never reached a render.

// ---- Speed (LTX-2.5 distilled schedule preset) -----------------------------
// The REAL speed control that replaced the dead one. Server-owned gating:
// the row shows only when the current tier cell carries `fast_eta`, which
// the registry stamps only on 2.5 distilled cells.
function setSchedPreset(v) {
  const val = (v === 'fast') ? 'fast' : '';
  const inp = document.getElementById('schedule_preset');
  if (inp) inp.value = val;
  document.querySelectorAll('#schedPresetGroup [data-sched-preset]').forEach(b =>
    b.classList.toggle('active', (b.dataset.schedPreset || '') === val));
  // Chips re-price under the preset (ltxCellEta reads the hidden input), the
  // summary names the state — same discipline as H3's Turbo repaint.
  if (typeof renderTierAxes === 'function') { try { renderTierAxes('ltx'); } catch (e) {} }
  if (typeof updateDerived === 'function') { try { updateDerived(); } catch (e) {} }
  if (typeof updateCustomizeSummary === 'function') { try { updateCustomizeSummary(); } catch (e) {} }
}
function schedPresetActive() {
  return (document.getElementById('schedule_preset') || {}).value === 'fast';
}
function _applySchedPresetRowVisibility() {
  const row = document.getElementById('schedPresetRow');
  if (!row) return;
  let cell = null;
  try { cell = ltxCellFor(ltxCurrentQuality(), ltxCurrentLength()); } catch (e) {}
  const offered = !!(cell && cell.fast_eta)
    && document.body.dataset.engine !== 'h3';
  row.hidden = !offered;
  // A preset the current lane cannot run must not ride on the form — the
  // server would drop it anyway (make_job gates it), this keeps the UI and
  // the wire agreeing. Moving to an HQ tier or H3 resets the pill to Tuned.
  // Direct clear (not setSchedPreset) — this runs from renderTierAxes and
  // the setter repaints the axes, which would recurse.
  if (!offered && schedPresetActive()) {
    const inp = document.getElementById('schedule_preset');
    if (inp) inp.value = '';
    document.querySelectorAll('#schedPresetGroup [data-sched-preset]').forEach(b =>
      b.classList.toggle('active', (b.dataset.schedPreset || '') === ''));
  }
  if (offered && cell) {
    const fastSub = document.getElementById('schedFastSub');
    if (fastSub) fastSub.textContent = cell.fast_eta + ' · different take';
    const tunedSub = document.getElementById('schedTunedSub');
    if (tunedSub) tunedSub.textContent = cell.eta || 'default schedule';
  }
}
// ---- i2v reference use (Anchor / Inspire) ----------------------------------
// Server-owned availability: BOOT.ltx.inspire_available is true only where the
// engine's masked-sample re-pin is version-resolved on (2.5). The UI never
// parses a generation label.
function setI2vRefMode(v) {
  const val = (v === 'inspire') ? 'inspire' : 'anchor';
  const inp = document.getElementById('i2v_reference_mode');
  if (inp) inp.value = val;
  document.querySelectorAll('#i2vRefModeGroup [data-i2v-ref]').forEach(b =>
    b.classList.toggle('active', (b.dataset.i2vRef || '') === val));
  if (typeof updateCustomizeSummary === 'function') {
    try { updateCustomizeSummary(); } catch (e) {}
  }
}
function i2vInspireActive() {
  return (document.getElementById('i2v_reference_mode') || {}).value === 'inspire';
}
function _applyI2vRefModeVisibility() {
  const row = document.getElementById('i2vRefModeRow');
  if (!row) return;
  const isI2v = (currentMode === 'i2v');
  const offered = !!((BOOT.ltx || {}).inspire_available)
    && isI2v && document.body.dataset.engine !== 'h3';
  row.hidden = !offered;
  // A mode the current lane cannot honor must not ride on the form — the
  // server drops it anyway (make_job gates it); this keeps UI and wire
  // agreeing. Direct clear, no setter (avoids a summary repaint loop).
  if (!offered && i2vInspireActive()) {
    const inp = document.getElementById('i2v_reference_mode');
    if (inp) inp.value = 'anchor';
    document.querySelectorAll('#i2vRefModeGroup [data-i2v-ref]').forEach(b =>
      b.classList.toggle('active', (b.dataset.i2vRef || '') === 'anchor'));
  }
}
function _wireI2vRefModePills() {
  const group = document.getElementById('i2vRefModeGroup');
  if (!group || group.dataset.wired === '1') return;
  group.dataset.wired = '1';
  group.querySelectorAll('[data-i2v-ref]').forEach(b => {
    b.addEventListener('click', (e) => {
      e.preventDefault();
      setI2vRefMode(b.dataset.i2vRef || 'anchor');
    });
  });
}

function _wireSchedPresetPills() {
  const group = document.getElementById('schedPresetGroup');
  if (!group || group.dataset.wired === '1') return;
  group.dataset.wired = '1';
  group.querySelectorAll('[data-sched-preset]').forEach(b => {
    b.addEventListener('click', (e) => {
      e.preventDefault();
      setSchedPreset(b.dataset.schedPreset || '');
    });
  });
}

// Show the STG "detail guidance" slider only when quality=high (Q8 HQ).
// STG is a no-op on the Q4 distilled paths, so there's nothing to expose
// for Quick/Standard/Balanced. Hiding the row does NOT reset stg_scale —
// the slider's own value persists; the make_job clamp + the helper's
// stg_scale>0 gate mean a stale non-zero value can't engage on a Q4 render
// anyway (the Q4 dispatch never reads stg_scale).
function _applyStgRowVisibility() {
  const row = document.getElementById('stgRow');
  if (!row) return;
  const q = document.getElementById('quality')?.value || '';
  row.hidden = !_qualityUsesHq(q);
}

// Delegation survives renderCharacterStrip replacing all four buttons when a
// pack finishes installing. Idempotent via the data-wired flag.
function _wireCharacterQualityChips() {
  const group = document.getElementById('qualityGroupCharacter');
  if (!group || group.dataset.wired === '1') return;
  group.dataset.wired = '1';
  group.addEventListener('click', (e) => {
    const b = e.target.closest('.char-quality');
    if (!b || !group.contains(b)) return;
    e.preventDefault();
    _setCharacterQuality(b);
  });
}

// ====== CivitAI modal ======

globalThis._civitaiCursor = '';
globalThis._civitaiSearching = false;
// Search context — 'video' (LTX-2.3) or 'image' (Qwen + HiDream). Picked
// at modal-open time from the active workflow tab so the user sees LoRAs
// that match the engine they're about to use. 2026-05-18.
globalThis._civitaiContext = 'video';
// Family pill state for context=image. Empty / 'all' shows every image
// family; 'qwen' or 'hidream' narrows to just that engine. Re-derived
// from the available_families echo in each /civitai/search response.
globalThis._civitaiFamily = '';

// Returns the search context the CivitAI modal should use given which
// workflow tab the user is in. Studio (Images) → image; everything else
// → video. The body[data-workflow] attribute is set by workflowSwitch
// so this works for any callsite.
function _civitaiContextForCurrentWorkflow() {
  const wf = (document.body.dataset.workflow || '').toLowerCase();
  if (wf === 'studio' || wf === 'image') return 'image';
  return 'video';
}

// The video context's family pill IS the lane selector — the two video
// engines cannot load each other's adapters — so it is preselected from the
// engine the user is actually on rather than defaulting to "All". Net effect
// for an LTX user: the modal opens showing exactly what it always showed.
function _civitaiFamilyForCurrentEngine() {
  if (_civitaiContext !== 'video') return '';
  try {
    return (currentEngine() === 'h3') ? 'h3' : 'ltx';
  } catch (_) { return 'ltx'; }
}

function _civitaiContextMeta(ctx, fam) {
  if (ctx === 'image') {
    return {
      title: 'Browse CivitAI for image LoRAs',
      hint: 'Qwen-Image-Edit, HiDream-O1.',
      empty: 'No image LoRAs match',
    };
  }
  if (fam === 'h3') {
    return {
      title: 'Browse CivitAI for Hailuo H3 LoRAs',
      hint: 'MiniMax H3.',
      empty: 'No MiniMax H3 LoRAs match',
    };
  }
  if (fam === 'ltx') {
    return {
      title: 'Browse CivitAI for LTX 2.3 LoRAs',
      hint: 'LTX-Video 2.3.',
      empty: 'No LTX 2.3 LoRAs match',
    };
  }
  return {
    title: 'Browse CivitAI for video LoRAs',
    hint: 'LTX-Video 2.3 and MiniMax H3.',
    empty: 'No video LoRAs match',
  };
}

async function openCivitaiModal(context) {
  // Pick context from the active workflow if not explicitly passed.
  _civitaiContext = context || _civitaiContextForCurrentWorkflow();
  // Family BEFORE the title: on video the title names the engine's family.
  _civitaiFamily = _civitaiFamilyForCurrentEngine();
  const meta = _civitaiContextMeta(_civitaiContext, _civitaiFamily);
  const titleEl = document.getElementById('civitaiModalTitle');
  if (titleEl) titleEl.textContent = meta.title;
  // Static markup is hidden, and this synchronous pass keeps it fail-closed
  // while the authoritative Settings response is in flight.
  renderSpicyAccess();
  document.getElementById('civitaiModal').style.display = 'flex';
  // Pull /loras to populate the dir text and the auth-banner state. The dir
  // shown is the one this family's downloads will actually land in — the
  // server routes by the item's base model, so browsing H3 while on H3 has to
  // name the H3 library or the line would be a lie.
  fetch('/loras').then(r => r.json()).then(d => {
    if (d.loras_dir) _lorasDirs.ltx = d.loras_dir;
    if (d.h3_loras_dir) _lorasDirs.h3 = d.h3_loras_dir;
    _h3LoraSupported = !!d.h3_lora_supported;
    _civitaiSyncTargetDir();
    renderCivitaiAuthBanner(!!d.civitai_auth);
  }).catch(() => { renderCivitaiAuthBanner(false); });
  document.getElementById('civitaiQuery').value = '';
  _civitaiCursor = '';
  // Hide the family row by default; the first search response fills it and
  // shows it for every context that HAS families (image, and video since H3).
  const famRow = document.getElementById('civitaiFamilyRow');
  if (famRow) famRow.style.display = 'none';
  // Resolve the gate before searching so a stale checked box can never add
  // nsfw=true while Settings is still loading.
  await refreshCivitaiAccessUI();
  await civitaiSourceRowSync();
  civitaiSearch();
}

// Render the family-filter pill row when the response carries
// available_families. Pills are simple buttons styled to match the
// rest of the panel's pill UX. Click sets _civitaiFamily and re-runs
// the search.
function civitaiRenderFamilyPills(available, active) {
  const row = document.getElementById('civitaiFamilyRow');
  if (!row) return;
  if (!available || !Array.isArray(available) || available.length === 0) {
    row.style.display = 'none';
    row.innerHTML = '';
    return;
  }
  // Friendly labels per family id. Keep this map small and explicit.
  const labels = {
    qwen:    'Qwen-Image',
    hidream: 'HiDream',
    ltx:     'LTX',
    h3:      'Hailuo H3',
  };
  const all = ['all', ...available];
  row.style.display = 'flex';
  row.innerHTML = all.map(f => {
    const label = f === 'all' ? 'All' : (labels[f] || f);
    const isActive = (f === 'all' ? !active || active === 'all' : active === f);
    return `<button type="button"
              class="pill-btn${isActive ? ' active' : ''}"
              data-family="${escapeHtml(f)}"
              onclick="civitaiSetFamily('${escapeHtml(f)}')">${escapeHtml(label)}</button>`;
  }).join('');
}

// Point the "LoRAs land in …" line at the directory the CURRENT family's
// downloads will actually be written to. The server routes by the item's own
// base model, so on the video surface this tracks the family pill.
function _civitaiSyncTargetDir() {
  const dirEl = document.getElementById('civitaiTargetDir');
  if (!dirEl) return;
  const ltx = _lorasDirs.ltx || 'mlx_models/loras/';
  const h3 = _lorasDirs.h3 || 'the Hailuo H3 pack’s loras/ folder';
  if (_civitaiContext !== 'video') { dirEl.textContent = ltx; return; }
  if (_civitaiFamily === 'h3') { dirEl.textContent = h3; return; }
  if (_civitaiFamily === 'ltx') { dirEl.textContent = ltx; return; }
  // "All" spans both video engines and the server routes each download by its
  // OWN base model, so naming one directory here would be a coin flip. Name
  // both, and say what decides.
  dirEl.textContent = `${ltx} · MiniMax H3 → ${h3}`;
}

// Click handler for a family pill — set state + re-search.
function civitaiSetFamily(family) {
  _civitaiFamily = (family === 'all') ? '' : family;
  const _m = _civitaiContextMeta(_civitaiContext, _civitaiFamily);
  const _t = document.getElementById('civitaiModalTitle');
  if (_t) _t.textContent = _m.title;
  _civitaiSyncTargetDir();
  // Optimistically toggle the active class so the click feels instant
  // (the re-render after the fetch will reaffirm it).
  const row = document.getElementById('civitaiFamilyRow');
  if (row) {
    row.querySelectorAll('.pill-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.family === family);
    });
  }
  civitaiSearch();
}

// One UI predicate serves both LTX and H3. The active engine only changes the
// LoRA family; it never changes whether NSFW controls/data are authorized.
function spicyModeEnabled() {
  return !!(_settingsCache && _settingsCache.settings &&
            _settingsCache.settings.spicy_mode === true);
}

function renderSpicyAccess() {
  const enabled = spicyModeEnabled();
  document.querySelectorAll('[data-spicy-only]').forEach(el => {
    el.hidden = !enabled;
  });
  const cb = document.getElementById('civitaiNsfw');
  if (!enabled && cb) cb.checked = false;
  return enabled;
}

function civitaiNsfwRequested() {
  const cb = document.getElementById('civitaiNsfw');
  return spicyModeEnabled() && !!(cb && cb.checked);
}

// Refresh the shared Settings cache, then render from the one predicate.
// Any fetch/shape failure explicitly records OFF instead of trusting stale
// state from an earlier session.
async function refreshCivitaiAccessUI() {
  try {
    const r = await fetch('/settings');
    const j = await r.json();
    if (!r.ok || !j || !j.settings) throw new Error('invalid settings response');
    if (!_settingsCache) _settingsCache = {};
    _settingsCache.settings = j.settings;
  } catch (_) {
    if (!_settingsCache) _settingsCache = {};
    _settingsCache.settings = Object.assign(
      {}, _settingsCache.settings || {}, { spicy_mode: false }
    );
  }
  return renderSpicyAccess();
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
      <span><svg class="ph" aria-hidden="true" style="color:var(--success,#3fb950);margin-right:4px;vertical-align:-2px"><use href="#ph-check-bold"/></svg><strong style="color:var(--success,#3fb950)"></strong> CivitAI API key set —
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

// ---- the source: CivitAI, or a Hugging Face org --------------------------------
let _civitaiSource = 'civitai';
let _civitaiRerun = false;      // a source or family change landed mid-search
function civitaiSetSource(src) {
  _civitaiSource = (src === 'huggingface') ? 'huggingface' : 'civitai';
  document.querySelectorAll('#civitaiSourceRow [data-civitai-source]').forEach(b =>
    b.classList.toggle('active', b.dataset.civitaiSource === _civitaiSource));
  const q = document.getElementById('civitaiQuery');
  if (q) q.placeholder = _civitaiSource === 'huggingface' ? 'Search Hugging Face — a name, author:someone, or owner/repo' : 'Search by name, style, creator…';
  // The heading and the CivitAI key banner belong to CivitAI; Hugging Face
  // needs neither.
  const title = document.getElementById('civitaiModalTitle');
  const banner = document.getElementById('civitaiAuthBanner');
  if (_civitaiSource === 'huggingface') {
    if (title) title.textContent = `Browse Hugging Face for ${_civitaiFamily === 'h3' ? 'Hailuo H3' : 'LTX'} LoRAs`;
    if (banner) banner.style.display = 'none';
  } else {
    if (title) title.textContent = `Browse CivitAI for ${_civitaiFamily === 'h3' ? 'Hailuo H3' : 'LTX 2.3'} LoRAs`;
    try { refreshCivitaiAccessUI(); } catch (_) {}
  }
  if (_civitaiSearching) { _civitaiRerun = true; return; }
  civitaiSearch();
}
function civitaiSourceRowSync() {
  const row = document.getElementById('civitaiSourceRow');
  if (!row) return;
  const show = _civitaiContext === 'video';
  row.style.display = show ? 'flex' : 'none';
  if (!show) _civitaiSource = 'civitai';
}
// Hugging Face source: the whole catalog for the lane comes at once (no
// paging), filtered by the query on the server.
async function _hfLoraSearch(grid, status, loadMore) {
  const params = new URLSearchParams();
  params.set('lane', _civitaiFamily === 'h3' ? 'h3' : 'ltx');
  const q = document.getElementById('civitaiQuery').value.trim();
  if (q) params.set('q', q);
  const r = await fetch('/hf/loras?' + params.toString());
  const data = await r.json();
  if (!data.ok) {
    grid.innerHTML = '';
    status.textContent = data.error || 'Could not load the catalog.';
    status.className = 'civitai-status-line err';
    return;
  }
  renderCivitaiGrid(data.items, false);
  loadMore.style.display = 'none';
  if ((data.items || []).length === 0) {
    grid.innerHTML = `<div class="hint">Nothing on Hugging Face for ${_civitaiFamily === 'h3' ? 'Hailuo H3' : 'LTX'}${q ? ` matching "${escapeHtml(q)}"` : ''}. Try a name, <code>author:someone</code>, or <code>owner/repo</code>.</div>`;
  } else {
    status.textContent = `${data.items.length} LoRA${data.items.length === 1 ? '' : 's'} on Hugging Face — a card plays the repo's own example when it has one. Read the repo before you install; Phosphene lists what matches, nothing more.`;
    status.className = 'civitai-status-line';
  }
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
    // `_civitaiSource` lives at module level; the contract tests extract this
    // function alone, so read it defensively rather than assume the module.
    const _src = (typeof _civitaiSource === 'string') ? _civitaiSource : 'civitai';
    if (_src !== 'civitai') {
      await _hfLoraSearch(grid, status, loadMore);
      return;
    }
    const params = new URLSearchParams();
    const q = document.getElementById('civitaiQuery').value.trim();
    if (q) params.set('query', q);
    if (civitaiNsfwRequested()) params.set('nsfw', 'true');
    params.set('limit', '24');
    params.set('context', _civitaiContext);
    if (_civitaiFamily) params.set('family', _civitaiFamily);
    const r = await fetch('/civitai/search?' + params.toString());
    const data = await r.json();
    if (data.error) {
      grid.innerHTML = '';
      status.textContent = data.error;
      status.className = 'civitai-status-line err';
      return;
    }
    // Render family pills from the server's echoed catalog (only set
    // for context=image). Reaffirms the active selection after each
    // search so the active class is always in sync with state.
    civitaiRenderFamilyPills(data.available_families, data.family);
    renderCivitaiGrid(data.items, /* append */ false);
    _civitaiCursor = data.next_cursor || '';
    if (data.has_more) loadMore.style.display = '';
    if ((data.items || []).length === 0) {
      const meta = _civitaiContextMeta(_civitaiContext, _civitaiFamily);
      grid.innerHTML = `<div class="hint">${meta.empty} "${escapeHtml(q || '')}"${civitaiNsfwRequested() ? '' : ' (try Show NSFW for more)'}.</div>`;
    }
  } catch (e) {
    status.textContent = 'Network error: ' + (e.message || e);
    status.className = 'civitai-status-line err';
  } finally {
    _civitaiSearching = false;
    if (typeof _civitaiRerun !== 'undefined' && _civitaiRerun) { _civitaiRerun = false; civitaiSearch(); }
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
    if (civitaiNsfwRequested()) params.set('nsfw', 'true');
    params.set('limit', '24');
    params.set('cursor', _civitaiCursor);
    params.set('context', _civitaiContext);
    if (_civitaiFamily) params.set('family', _civitaiFamily);
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
          ${it.source === 'huggingface' ? `<span title="likes on Hugging Face">♥ ${Number(it.likes || 0)}</span>` : `<span>↓ ${dl}</span>`}
          <span>${sizeMb} MB</span>
          ${it.nsfw ? '<span class="nsfw-badge">NSFW</span>' : ''}
        </div>
        ${triggers ? `<div class="meta"><span title="trigger words">trigger: ${escapeHtml(triggers)}</span></div>` : ''}
        ${it.civitai_url
          ? `<div class="meta"><a class="civitai-source-link" href="${escapeHtml(it.civitai_url)}" target="_blank" rel="noopener" title="Open the original CivitAI page — usage notes, examples, comments">Read instructions on CivitAI <svg class="ph" aria-hidden="true" style="margin-left:3px;vertical-align:-2px"><use href="#ph-arrow-square-out"/></svg></a></div>`
          : (it.hf_url
              ? `<div class="meta"><a class="civitai-source-link" href="${escapeHtml(it.hf_url)}" target="_blank" rel="noopener" title="Open the Hugging Face repo — the author's example clip and notes">Open on Hugging Face <svg class="ph" aria-hidden="true" style="margin-left:3px;vertical-align:-2px"><use href="#ph-arrow-square-out"/></svg></a></div>`
              : '')}
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
  const fromHf = item.source === 'huggingface';
  if (fromHf) {
    fd.set('repo', item.id);
    fd.set('filename', item.filename);
  } else {
    fd.set('download_url', item.download_url);
  }
  fd.set('meta', JSON.stringify(item));
  try {
    const r = await fetch(fromHf ? '/hf/loras/download' : '/civitai/download', {
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
    // The server routes by the item's OWN base model, so a MiniMax H3 LoRA
    // lands in the H3 library even if it was found while the browser showed
    // "All". Auto-enable only when that library is the one the active engine
    // reads — attaching an H3 adapter to an LTX render would fail inside the
    // fuser, and attaching an LTX one to H3 would match zero modules and
    // render as though nothing were attached at all.
    const lane = data.lane || 'ltx';
    let activeTag = 'video';
    try {
      activeTag = (typeof _currentLoraModeFilter === 'function')
        ? _currentLoraModeFilter() : 'video';
    } catch (_) {}
    const laneMatches = (lane === 'h3') ? (activeTag === 'video:h3')
                                        : (activeTag !== 'video:h3');
    const status = document.getElementById('civitaiStatus');
    const where = data.skipped ? `Already in ${data.path}` : `Saved to ${data.path}`;
    status.textContent = where + (
      laneMatches
        ? ' — auto-enabled below.'
        : (lane === 'h3'
            ? ' — it is a Hailuo H3 LoRA, so switch the engine to Hailuo H3 to use it.'
            : ' — it is an LTX LoRA, so switch the engine to LTX to use it.'))
      + (data.converted
          ? ' Converted from the ComfyUI key layout on install (a key rename;'
            + ' the tensors are untouched).'
          : '');
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
    if (laneMatches) {
      addLoraToActive({
        path: data.path,
        name: data.name || item.name,
        strength: item.recommended_strength || 1.0,
        trigger_words: item.trigger_words || [],
        lane: lane,
      });
    }
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
document.addEventListener('DOMContentLoaded', () => {
  refreshLoras();
  // Manual-tab Characters picker — load the list once at boot so the
  // chips are visible immediately when the user opens the section.
  // (The Characters tab UI is no longer reachable from the nav as of
  // 2026-05-17 — chip strip in Manual is the only character surface.)
  if (typeof refreshManualCharacters === 'function') {
    try { refreshManualCharacters(); } catch (e) {}
  }
  // Apply initial picker visibility (T2V only) based on the default mode.
  if (typeof _updateCharsPickerVisibility === 'function') {
    try { _updateCharsPickerVisibility(currentMode || 't2v'); } catch (e) {}
  }
  // Bind the Q8 Draft / Q8 Pro chip clicks for the character-only
  // quality strip. Idempotent.
  if (typeof _wireCharacterQualityChips === 'function') {
    try { _wireCharacterQualityChips(); } catch (e) {}
  }
  // (HQ Speed pill wiring + skip-step boot init removed in v4.0.5 —
  // the control was dead at the engine boundary.)
  // Bind the Speed pills (LTX-2.5 distilled schedule preset). Visibility
  // settles on every renderTierAxes('ltx') repaint, including boot's.
  if (typeof _wireSchedPresetPills === 'function') {
    try { _wireSchedPresetPills(); } catch (e) {}
  }
  // Bind the reference-use pills; visibility follows mode in updateDerived.
  if (typeof _wireI2vRefModePills === 'function') {
    try { _wireI2vRefModePills(); } catch (e) {}
  }
  // Apply correct quality-strip visibility based on whether a character
  // is already selected (e.g. restored from sidecar / Load Params).
  if (typeof _applyCharacterQualityStripVisibility === 'function') {
    try { _applyCharacterQualityStripVisibility(); } catch (e) {}
  }
  // Apply correct STG-slider visibility based on initial quality.
  if (typeof _applyStgRowVisibility === 'function') {
    try { _applyStgRowVisibility(); } catch (e) {}
  }
});

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
    ? `Each row shows what's on disk. Click <b>Download</b> to fetch the missing files; progress streams to the log at the bottom of the page. Everything is resumable and checksum-verified.`
    : `<span style="color:var(--warning,#d29922)"><svg class="ph" aria-hidden="true" style="margin-right:4px;vertical-align:-2px"><use href="#ph-warning-fill"/></svg><code>hf</code> not found</span> — this Pinokio install doesn't have <code>huggingface_hub&gt;=1.0</code> in the venv. Run Update from Pinokio, then come back. The LTX-2.5 rows do not need it — they download from a GitHub release.`;
  // A row whose HOST pack is missing cannot usefully be downloaded: the add-on
  // lands INSIDE that pack's directory and loads from there by name, so
  // fetching it alone produces a folder holding two files and nothing else. The
  // dependency is STATED, not enforced silently.
  const completeByKey = {};
  repos.forEach(r => { completeByKey[r.key] = !!r.complete; });
  const rows = repos.map(r => {
    let cls, icon, statusText, btnHtml;
    // Per ROW, not per install. The 2.5 packs download from a GitHub release
    // and need no `hf` at all — gating them on it disabled a button that would
    // have worked, on the exact install the modal's own header tells to go
    // ahead. `needs_hf` is derived from the registry's mirror block.
    const hfOk = (data.hf_available ?? true) || (r.needs_hf === false);
    const hostMissing = r.host_key && !completeByKey[r.host_key];
    if (active && active.key === r.key) {
      cls = 'downloading';
      icon = '<svg class="ph" aria-hidden="true"><use href="#ph-arrow-clockwise-bold"/></svg>';
      const elapsed = Math.max(0, Math.round((Date.now()/1000) - (active.started_ts || 0)));
      const last = active.last_line ? `<div class="progress">${escapeHtml(active.last_line)}</div>` : '';
      statusText = `Downloading · ${elapsed}s${last}`;
      btnHtml = `<button class="ghost" onclick="cancelDownload()">Cancel</button>`;
    } else if (r.complete) {
      cls = 'ready'; icon = '<svg class="ph" aria-hidden="true"><use href="#ph-check-bold"/></svg>';
      // `where: 'hf_cache'` means the files were resolved from
      // ~/.cache/huggingface/ rather than the canonical mlx_models/
      // dir. Common on manual / dev installs that pre-existed Pinokio
      // and pulled the model via `huggingface-cli` or first-run helper.
      const tag = r.where === 'hf_cache' ? 'HF cache' : 'local';
      statusText = `Ready · ${r.total_files} files · ~${r.size_gb || '?'} GB · ${tag}`;
      btnHtml = `<button class="ghost" disabled>Installed</button>`;
    } else if (r.present_files > 0) {
      cls = 'partial'; icon = '<svg class="ph" aria-hidden="true"><use href="#ph-download-simple"/></svg>';
      const left = r.total_files - r.present_files;
      statusText = `Partial · ${r.present_files}/${r.total_files} files · ${left} missing — resume to finish`;
      btnHtml = hostMissing
        ? `<button disabled title="Install the Q8 weights first — the add-on lives inside that folder.">Needs Q8</button>`
        : hfOk
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Resume</button>`
        : `<button disabled>Resume</button>`;
    } else {
      cls = 'missing'; icon = '<svg class="ph" aria-hidden="true"><use href="#ph-x-circle"/></svg>';
      statusText = `Not installed · ~${r.size_gb || '?'} GB`;
      btnHtml = hostMissing
        ? `<button disabled title="Install the Q8 weights first — the add-on lives inside that folder.">Needs Q8</button>`
        : hfOk
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Download</button>`
        : `<button disabled>Download</button>`;
    }
    // "required" means required BY THIS BUILD. A base pack from the retired
    // generation is neither required nor optional — it is previous, and saying
    // so is the difference between an inventory and a claim.
    const kindBadge = (r.active === false)
      ? `<span style="color:var(--muted)">previous generation</span>`
      : r.kind === 'optional'
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
  // Hailuo H3 — an optional PACK, not an `hf download` repo (clone + its own
  // venv + ~75 GB of weights), so it can't come from the manifest loop above.
  // It still belongs in this list: this is where users look for "what else can
  // I install". Rendered from the live /status snapshot, with a button that
  // routes to the same install card the engine pill opens.
  let h3Row = '';
  {
    const h3 = (LAST_STATUS && LAST_STATUS.h3) || (typeof H3 === 'object' ? H3 : null);
    if (h3 && h3.capable) {
      const ready = !!h3.available;
      const cls = ready ? 'ready' : 'missing';
      const icon = ready
        ? '<svg class="ph" aria-hidden="true"><use href="#ph-check-bold"/></svg>'
        : '<svg class="ph" aria-hidden="true"><use href="#ph-x-circle"/></svg>';
      const statusText = ready
        ? `Ready · engine picker unlocked · ${escapeHtml(h3.root || '')}`
        : 'Available to install · one click in the Pinokio sidebar';
      const btn = ready
        ? `<button class="ghost" disabled>Installed</button>`
        : `<button onclick="openH3InstallCard()">How to install</button>`;
      h3Row = `
        <li class="${cls}">
          <span class="icon">${icon}</span>
          <div class="meta">
            <span class="ttl">Hailuo H3 (MiniMax-H3 FL2VA) · <span style="color:var(--muted)">second video engine</span></span>
            <span class="sub">A peer of LTX — one prompt in, video + synced dialogue + sound out</span>
            <span class="sub">${statusText} · ${escapeHtml(h3.size_note || '')}</span>
          </div>
          ${btn}
        </li>`;
    }
  }
  list.innerHTML = (rows + h3Row) || `<li class="empty-state">No model manifest found — required_files.json is missing or unreadable.</li>`;
  // Footer summarises required vs optional counts.
  // Count only what THIS BUILD needs. 2.3's two base rows were inside
  // "Required: N/M ready", which made §7.1's first-visit contract
  // ("Required: 2/2 ready") unreachable on any machine that also has 2.3.
  const reqRepos = repos.filter(r => r.kind !== 'optional' && r.active !== false);
  const optRepos = repos.filter(r => r.kind === 'optional' && r.active !== false);
  const prevRepos = repos.filter(r => r.active === false);
  const reqReady = reqRepos.filter(r => r.complete).length;
  const optReady = optRepos.filter(r => r.complete).length;
  foot.innerHTML = `
    <div>Required: ${reqReady}/${reqRepos.length} ready &nbsp;·&nbsp; Optional: ${optReady}/${optRepos.length} ready${
      prevRepos.length ? ` &nbsp;·&nbsp; <span style="color:var(--muted)">${prevRepos.length} from a previous generation (Settings → Storage)</span>` : ''}</div>
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


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup and the other files resolve these through
// the global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  civitaiSetSource, civitaiSourceRowSync,
  normalizeLivePreview, _liveStageMediaHeld, _showLiveReturnChip, _hideLiveStageChrome,
  _restoreSelectedOutputAfterLive, _handoffLiveStageToOutput, returnToLiveRender, _renderLiveStageFrame,
  renderLiveStage, renderNowPreview, stopEarly, markNoVoiceTouched,
  refreshNoVoiceAuto, setCharStrength, toggleCharSplit, toggleCharVoiceHelp,
  openCharactersManageModal, closeCharactersManageModal, applyCharacterSelection, selectManualCharacter,
  _applyCharacterQualityStripVisibility, _setCharacterQuality, setSchedPreset, schedPresetActive,
  _applySchedPresetRowVisibility, setI2vRefMode, i2vInspireActive, _applyI2vRefModeVisibility,
  _applyStgRowVisibility, _civitaiContextMeta, openCivitaiModal, civitaiRenderFamilyPills,
  spicyModeEnabled, renderSpicyAccess, civitaiNsfwRequested, renderCivitaiAuthBanner,
  closeCivitaiModal, civitaiSearch, civitaiLoadMore, renderCivitaiGrid,
  openModelsModal, closeModelsModal, refreshModelsModal, startDownload,
  cancelDownload,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  _charactersManageDelete, _charactersManageSave, civitaiAuthSave, civitaiSetFamily,
});
