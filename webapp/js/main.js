// webapp/js/main.js — the page's kickoff sequence (the old '====== Init
// ======' tail of the inline block), extracted in slice 3 of
// docs/ARCHITECTURE.md. Its <script type="module"> tag is deliberately
// the LAST one: in the single-block days every function was hoisted, so
// the first poll() could never race a definition. Split into modules,
// the only way to keep that guarantee is for the kickoffs to run after
// every module has evaluated — which 'last module tag' provides. Any
// future 'call this once at startup' line belongs HERE, not at the top
// level of a feature module.
// ====== Init ======
// Skip poll when the tab is backgrounded — at 1.5s cadence with a fan-
// spinning render in the background, every saved request matters. Pinokio
// users park the panel in a tab and switch to other apps for the 5–20 min
// a render takes; nothing in the UI needs updating until they come back.
// `visibilitychange` fires immediately when the user returns so the chrome
// catches up on the first frame.
setInterval(() => { if (!document.hidden) poll(); }, 1500);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

// Delegated click handler for the failed Now-card action buttons.
// Inline `onclick` attributes on these buttons were fragile — the
// failed branch of poll() rewrites .ttl's innerHTML every 1.5s, and a
// click that landed mid-rewrite could be lost. A single delegated
// listener on document survives every rewrite + costs nothing.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action="retry"], [data-action="dismiss"], [data-action="stop-early"]');
  if (!btn) return;
  e.stopPropagation();
  e.preventDefault();
  // A third delegated action in the same row, for the same reason the other
  // two are delegated: poll() rewrites this element every 1.5 s, and an inline
  // handler would be lost to that race mid-click.
  if (btn.dataset.action === 'stop-early') { stopEarly(); return; }
  const actions = btn.closest('.now-card-actions');
  const id = actions ? actions.dataset.jobId : '';
  if (!id) return;
  if (btn.dataset.action === 'retry') {
    if (typeof retryJob === 'function') retryJob(id);
  } else {
    window._dismissedFailureId = id;
    if (typeof poll === 'function') poll();
  }
});

poll();
setMode('t2v');
setAspect('landscape');         // sets aspect first so the default preset orients correctly
setQuality('balanced');         // bundles quality + dims; respects current aspect
applyTierTimes();               // no-op for LTX since v4.0 — the tier table owns those subtitles
renderCharacterStrip();         // the generation-scoped character quality ladder
// Engine picker — re-apply the last-used engine after the boot sequence above
// has settled the mode. setEngine() re-runs every gate (capable / installed /
// mode), so a stale localStorage value from a machine that has since lost the
// pack just lands back on LTX. The tier was restored at parse time (see
// _restoreH3TierEarly).
(function restoreEngineChoice() {
  let engine = null;
  try { engine = localStorage.getItem(H3_ENGINE_LS_KEY); } catch (e) {}
  setEngine(engine === 'h3' ? 'h3' : 'ltx', { persist: false });
})();
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
// Workflow tabs — Manual / Characters / Train
// ============================================================================
// The in-panel agentic chat surface was removed 2026-05-15 (tag
// pre-agent-removal-2026-05-15). External agents drive Phosphene via the
// HTTP API in docs/API.md. The tab switcher now flips between three
// surfaces: the manual generate form, the Characters tab (LoRA pair +
// prompt + ship), and Train (character LoRA training).


// The completion-alert switch is read by the poller from the settings cache,
// which used to be filled only when the Settings modal opened. One fetch at
// boot; the modal refreshes it as before.
(async () => {
  try {
    if (!globalThis._settingsCache) {
      const r = await fetch('/settings');
      if (r.ok) globalThis._settingsCache = await r.json();
    }
  } catch (e) {}
})();

// The appearance is applied at boot, before anyone looks — a stored "light"
// that arrived after the first paint would flash the dark palette first.
try { if (typeof applyAppearance === 'function') applyAppearance(); } catch (e) {}
