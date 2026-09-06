#!/usr/bin/env python3
"""Contract gate for the two refusals the fleet actually hits.

A refusal is the panel saying no on purpose (see docs/ANALYTICS.md → "a
refusal is not a failure"). The guards themselves are correct and stay; what
this file gates is that a user is never OFFERED the control that leads to one.

The 14-day fleet read on 2026-08-23 found both of these at the top of the
render_failed list, which is how they were found at all:

  * "Ingredients needs the LTX-2.3 generation…"  — 65 events, 16 people.
    The sub-chip was disabled and its click handler blocked, but the parent
    "Remix" pill in the main mode bar resolved to 'ingredients' by default
    and was never gated.
  * "High quality (Q8 two-stage) isn't supported on the … hardware tier" —
    23 events, 6 people. `applyTierGates` looked up `#qualityHigh`, an id no
    markup carries, so its High branch was dead code; and the CSS that hides
    High on the Q4 surface matched `high` exactly, so `high_720p` (v4.0.2,
    same `hq` pipeline) stayed clickable.

The server-side facts are executed. The client-side gate is a JS function in
an embedded page, so it is asserted structurally — enough to catch a
regression that deletes the guard, and paired with a browser pass on a real
2.5 install for the behaviour itself.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = Path(tempfile.mkdtemp(prefix="phos-refusal-gates-"))
os.environ["LTX_STATE_DIR"] = str(STATE)
os.environ["PHOSPHENE_ANALYTICS_DISABLED"] = "1"
os.environ["PHOSPHENE_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("LTX_PORT", "8297")
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as P  # noqa: E402

# Both halves — the Python server and the page (markup + JS), which lives
# at webapp/index.html since slice 2 of the extraction (docs/ARCHITECTURE.md).
PANEL_SRC = ((ROOT / "mlx_ltx_panel.py").read_text(encoding="utf-8")
             + "\n" + (ROOT / "webapp" / "index.html").read_text(encoding="utf-8"))
for _m in sorted((ROOT / "webapp" / "js").glob("*.js")):
    PANEL_SRC += "\n" + _m.read_text(encoding="utf-8")
# The stylesheet the browser loads — CSS moved out of the Python source to
# webapp/style/panel.css in the slice-1 extraction (docs/ARCHITECTURE.md).
# CSS-rule assertions must scan THIS, or they pass vacuously against a
# source file that no longer contains any CSS at all.
PANEL_CSS = (ROOT / "webapp" / "style" / "panel.css").read_text(encoding="utf-8")


class TestIngredientsGenerationGate(unittest.TestCase):
    """One predicate, read by the worker, the bootstrap and the tests — so
    "which generations serve Ingredients" cannot drift between the three."""

    def setUp(self):
        self.old_version = P.ACTIVE_MODEL_VERSION

    def tearDown(self):
        P.ACTIVE_MODEL_VERSION = self.old_version

    def test_the_predicate_is_generation_scoped(self):
        self.assertFalse(P.ltx_generation_serves_ingredients("ltx25"))
        self.assertTrue(P.ltx_generation_serves_ingredients("ltx23"))

    def test_the_flag_reaches_the_page_on_both_generations(self):
        P.ACTIVE_MODEL_VERSION = "ltx25"
        self.assertIs(
            P.ltx_tiers_payload()["ingredients_available"], False,
            "the UI cannot gate what the bootstrap never told it")
        P.ACTIVE_MODEL_VERSION = "ltx23"
        self.assertIs(P.ltx_tiers_payload()["ingredients_available"], True)

    def test_the_worker_gate_and_the_ui_flag_are_the_same_predicate(self):
        """If these two ever come from different expressions, one of them
        will be updated the day a 2.5 adapter ships and the other will not."""
        self.assertIn("ltx_generation_serves_ingredients()",
                      PANEL_SRC.split("def ltx_tiers_payload")[1][:4000])
        self.assertIn("if not ltx_generation_serves_ingredients():", PANEL_SRC)

    def test_the_refusal_is_raised_as_a_refusal_not_a_runtime_error(self):
        gate = PANEL_SRC.split(
            "if not ltx_generation_serves_ingredients():")[1][:400]
        self.assertIn('raise RenderRefused(', gate)
        self.assertIn('"ingredients_generation"', gate)

    def test_the_parent_remix_pill_no_longer_defaults_into_the_refusal(self):
        """THE actual hole. setMode('remix') used to resolve to a literal
        'ingredients'; _lastRemixMode is never persisted, so every fresh page
        load took that default and one click on an enabled pill put
        `ingredients` into the submitted form."""
        self.assertNotIn("window._lastRemixMode || 'ingredients'", PANEL_SRC,
                         "the parent Remix pill defaults straight into a "
                         "mode this generation may not serve")
        self.assertIn("window._lastRemixMode || defaultRemixMode()", PANEL_SRC)

    def test_setMode_snaps_away_from_an_unservable_ingredients(self):
        """The braces: a remembered _lastRemixMode, a stale localStorage or
        any direct caller must not be able to land on it either."""
        body = PANEL_SRC.split("function setMode(mode) {")[1][:2400]
        self.assertIn("if (mode === 'ingredients' && !ingredientsServed())",
                      body)

    def test_the_submit_handler_checks_what_is_actually_posted(self):
        """FormData reads the hidden #mode input, never currentMode — so
        this is the only guard that sees the value about to go on the wire."""
        self.assertIn(
            "if (fd.get('mode') === 'ingredients' && !ingredientsServed())",
            PANEL_SRC)

    def test_the_generate_button_carries_the_reason(self):
        self.assertIn("Generate · needs LTX-2.3", PANEL_SRC)

    def test_the_gate_names_both_routes_the_server_names(self):
        """The refusal text offers two ways out. A dead-end that does not
        repeat them is still a dead end."""
        for surface in ("Image mode with Inspire", "2.3 pack from the Train tab"):
            self.assertGreaterEqual(
                PANEL_SRC.count(surface), 2,
                f"{surface!r} should appear in the server refusal AND in the "
                f"UI that now prevents reaching it")

    def test_one_reader_for_the_flag(self):
        """Four surfaces now care. They must all read the same helper, or
        the day a 2.5 adapter ships one of them will still say no."""
        self.assertIn("function ingredientsServed()", PANEL_SRC)
        raw = len(re.findall(r"ingredients_available", PANEL_SRC))
        # Python: the payload key. JS: the helper's own read, plus the
        # pre-existing sub-chip toast + painter. Nothing new may read it raw.
        self.assertLessEqual(raw, 5, "a new raw read of the flag crept in — "
                                     "route it through ingredientsServed()")


class TestHqHardwareTierGate(unittest.TestCase):
    """High quality on a Compact Mac: the chip must arrive disabled, with the
    reason, on BOTH members of the HQ lane."""

    def setUp(self):
        self.old_caps = dict(P.SYSTEM_CAPS)

    def tearDown(self):
        P.SYSTEM_CAPS.clear()
        P.SYSTEM_CAPS.update(self.old_caps)

    def _hq_cells(self):
        return [t for t in P.ltx_tiers_payload()["tiers"]
                if t.get("pipeline") == "hq"]

    def test_the_lane_has_more_than_one_member(self):
        """The premise of the bug: `high_720p` was added on the same
        pipeline as `high`, and the exact-match gate only knew about one."""
        qualities = {c["quality"] for c in self._hq_cells()}
        self.assertIn("high", qualities)
        self.assertGreater(len(qualities), 1,
                           "if the HQ lane is one quality again, the "
                           "prefix-matching gate below is over-built, not "
                           "wrong — but check the CSS selector too")

    def test_every_hq_cell_is_unavailable_when_this_mac_cannot_run_q8(self):
        P.SYSTEM_CAPS["allows_q8"] = False
        cells = self._hq_cells()
        self.assertTrue(cells)
        for c in cells:
            self.assertIs(c["available"], False, c["key"])
            self.assertIn("hardware", c["unavailable_reason"].lower(), c["key"])
            self.assertTrue(c["unavailable_reason"].strip(),
                            f"{c['key']} is disabled with no reason shown")

    def test_nothing_outside_the_hq_lane_is_touched(self):
        P.SYSTEM_CAPS["allows_q8"] = False
        others = [t for t in P.ltx_tiers_payload()["tiers"]
                  if t.get("pipeline") != "hq"]
        self.assertTrue(others)
        # Quick / Standard still run on a Compact Mac. The only cells that may
        # be unavailable here are the pre-existing canvas-vs-length ones.
        for c in others:
            if c["available"] is False:
                self.assertNotIn("hardware", c["unavailable_reason"].lower(),
                                 c["key"])

    def test_a_q8_capable_mac_is_unaffected(self):
        P.SYSTEM_CAPS["allows_q8"] = True
        for c in self._hq_cells():
            self.assertNotIn("hardware tier",
                             (c.get("unavailable_reason") or "").lower(),
                             c["key"])

    def test_the_hq_chips_are_disabled_with_a_reason_not_hidden(self):
        """Owner ruling 2026-08-23. Hiding High on a Compact Mac left the user
        with three columns and no story — no way to learn that High exists,
        why it is missing, or what would bring it back. The cell already
        carries a written reason, so the honest state is disabled-with-reason.
        """
        # No display:none rule may target the HQ lane any more — neither the
        # exact-match one (which let `high_720p` through) nor a prefix one.
        self.assertNotIn('#qualityGroup [data-quality="high"]', PANEL_CSS)
        self.assertNotIn('#qualityGroup [data-quality^="high"]', PANEL_CSS)
        # The greyed/struck-through unavailable style has to reach the LTX
        # strips, or an un-hidden chip looks clickable and does nothing.
        self.assertIn("#qualityGroup .q-chip.unavailable", PANEL_CSS)
        self.assertIn("#ltxLengthGroup .q-chip.unavailable", PANEL_CSS)
        # And a click has to SAY the reason: a tooltip is not discoverable on
        # a chip somebody has just tapped.
        gate = PANEL_SRC.split("function _ltxApplyShape(")[1][:1400]
        self.assertIn("cell.unavailable_reason", gate)
        self.assertIn("engineRowNote", gate)

    def test_the_refusal_is_raised_as_a_refusal(self):
        gate = PANEL_SRC.split(
            "High quality (Q8 two-stage) isn't supported")[0][-300:]
        self.assertIn('raise RenderRefused(', gate)
        self.assertIn('"hardware_tier"', gate)


class TestTheImageDefaultThatNeverExisted(unittest.TestCase):
    """`image not found: <path>` — 35 render failures across 22 distinct
    people in 14 days, the widest-spread real failure in the fleet, at ~1.6
    events per person: everybody hits it once.

    The cause was a server-side default. `make_job` filled an empty `image`
    field with `examples/reference.png`, a demo file that has never shipped —
    not in git, not in required_files.json, created by no installer. The
    client-side pre-fill of the same path was deliberately removed when the
    control became a picker; the server's was left behind.
    """

    def test_the_demo_reference_is_not_a_file_this_repo_ships(self):
        self.assertFalse(
            (ROOT / "examples" / "reference.png").exists(),
            "examples/reference.png now exists — if it is genuinely shipped, "
            "the default may come back; if it is a stray, delete it.")

    def test_an_empty_image_field_stays_empty(self):
        self.assertEqual(P.default_reference_image(), "")

    def test_an_explicit_default_that_really_exists_is_still_honoured(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as fh:
            saved = P.REFERENCE
            try:
                P.REFERENCE = Path(fh.name)
                self.assertEqual(P.default_reference_image(), fh.name)
            finally:
                P.REFERENCE = saved

    def test_make_job_no_longer_invents_a_reference_image(self):
        self.assertEqual(
            PANEL_SRC.count('"image": f("image", str(REFERENCE))'), 0)
        self.assertIn('"image": f("image", default_reference_image())',
                      PANEL_SRC)

    def test_the_two_image_modes_validate_their_input_like_every_other_mode(self):
        """extend / control / restore / ingredients / keyframe / a2v and H3's
        own i2v all check the file exists before spending a render. The two
        plain LTX image modes were the only ones that did not, which is why
        this arrived as a bare helper error 30 s in."""
        gate = PANEL_SRC.split(
            'if mode in ("i2v", "i2v_clean_audio"):')[1][:900]
        self.assertIn("Image mode needs a reference image", gate)
        self.assertIn("no longer on disk", gate)
        self.assertIn("Path(_img).exists()", gate)

    def test_a_dead_pick_clears_itself_instead_of_looking_selected(self):
        """A broken preview read as 'an image is selected', so a Load Params
        replay of a deleted photo submitted happily. Only a 404 clears it —
        a transient must never throw away a good pick."""
        picker = PANEL_SRC.split("function pickerSetImage(")[1][:1600]
        self.assertIn("els.preview.onerror", picker)
        self.assertIn("r.status !== 404", picker)
        self.assertIn("pickerSetImage(key, '')", picker)


class TestH3RamRefusalStatesTheRealFloor(unittest.TestCase):
    """The `h3_ram` refusal said "about 64 GB of unified memory".

    No floor in this codebase has ever been 64. `H3_MIN_RAM_GB` is 60 (a 64 GB
    Mac reports ~63.x after firmware reservations) and `H3_MIN_RAM_GB_Q8` is
    46 — the entire point of building the Q8 DiT pack was, in its own comment,
    to "put H3 in reach of 48 GB Macs". So a 48 GB owner who could run H3 was
    told their machine needed 64, which is the kind of wrong that ends a
    feature for somebody permanently.

    Worse, `h3_capable()` gates the engine switcher and returns False on a
    sub-60 GB Mac until the Q8 pack EXISTS ON DISK — `h3_build_q8.sh`'s own
    header spells out the dead end: "a 48 GB Mac with no pack gets no Engine
    switcher". The door is a 5-minute local build with no download, and
    nothing on screen mentioned it.
    """

    def setUp(self):
        # h3_status() memoizes for 3 s (poll-load fix); these tests patch
        # RAM/paths state between calls and need fresh computation.
        P.h3_status_invalidate()
        self.old_ram = P.SYSTEM_RAM_GB
        self.old_dir = P._h3_q8_dit_dir

    def tearDown(self):
        P.SYSTEM_RAM_GB = self.old_ram
        P._h3_q8_dit_dir = self.old_dir

    def _band(self, ram: float, q8_pack: bool):
        P.SYSTEM_RAM_GB = ram
        P._h3_q8_dit_dir = (lambda: Path("/tmp/h3-dit-q8")) if q8_pack \
            else (lambda: None)
        return P.h3_ram_verdict()

    def test_the_two_floors_are_the_ones_the_product_believes(self):
        # The Q8 floor moved 46 -> 36 on 2026-08-29, when the full phase profile
        # replaced the single "27.3 GiB" figure it had been guessed from and the
        # panel started leaving the user real headroom. The bf16 floor is
        # unchanged. These are asserted as ORDER and RELATION, not as literals:
        # pinning the number is what made this test fail for a correct change,
        # and a stale literal here is the same defect the class exists to catch.
        self.assertEqual(P.H3_MIN_RAM_GB, 60.0)
        self.assertLess(P.H3_MIN_RAM_GB_Q8, P.H3_MIN_RAM_GB)
        self.assertGreater(P.H3_MIN_RAM_GB_Q8, 32.0)

    def test_a_big_mac_is_not_blocked(self):
        v = self._band(64.0, False)
        self.assertEqual(v["lane"], "bf16")
        self.assertFalse(v["blocked"])

    def test_a_48gb_mac_with_the_pack_runs_on_the_q8_lane(self):
        v = self._band(48.0, True)
        self.assertEqual(v["lane"], "q8")
        self.assertFalse(v["blocked"])
        self.assertFalse(v["needs_q8_dit"])

    def test_a_48gb_mac_without_the_pack_is_told_it_CAN_run_h3(self):
        v = self._band(48.0, False)
        self.assertTrue(v["needs_q8_dit"])
        self.assertNotIn("64", v["message"])
        self.assertIn("runs on this Mac", v["message"])
        self.assertIn("Install Hailuo H3", v["message"])

    def test_a_small_mac_is_told_the_real_floor_not_64(self):
        """The message must quote the floor the code actually gates on."""
        v = self._band(32.0, False)
        self.assertIsNone(v["lane"])
        self.assertTrue(v["blocked"])
        self.assertFalse(v["needs_q8_dit"])
        self.assertIn(str(int(P.H3_MIN_RAM_GB_Q8)), v["message"])
        self.assertNotIn("about 64", v["message"])
        self.assertIn("LTX", v["message"])

    def test_the_stale_sentence_is_no_longer_raised(self):
        """It survives in exactly one place — as a TEXTUAL needle in
        `_ANALYTICS_REFUSAL_REASONS`, so a replayed pre-2026-08-28 usage log
        still classifies as `refused` instead of rejoining `other`. Nothing
        raises it."""
        self.assertNotIn("Hailuo H3 needs about 64 GB", PANEL_SRC)
        self.assertEqual(
            PANEL_SRC.count("needs about 64 gb of unified memory"), 1,
            "the retired sentence is back somewhere other than the "
            "analytics needle table")

    def test_the_refusal_and_the_fallback_read_the_same_verdict(self):
        """Four surfaces stated one wrong number. They now share one."""
        gate = PANEL_SRC.split("def run_h3_job_inner")[1][:3000]
        self.assertIn('raise RenderRefused("h3_ram", h3_ram_verdict()', gate)
        self.assertIn('push("engine=h3 requested — falling back to LTX. "',
                      PANEL_SRC)

    def test_the_refusal_slug_still_matches_the_new_text(self):
        """`_ANALYTICS_REFUSAL_REASONS` matches refusals textually as a
        fallback. Rewriting the sentence must not drop it back into `other`
        — the exact regression docs/ANALYTICS.md was written about."""
        for ram, pack in ((48.0, False), (32.0, False)):
            msg = self._band(ram, pack)["message"]
            self.assertEqual(P._analytics_refusal_reason(msg), "h3_ram",
                             msg)

    def test_the_switcher_can_see_a_mac_that_is_one_build_away(self):
        """not-capable is not RENDERED, so a 48 GB Mac never learned H3
        existed. It is capable-and-not-available: the dashed offer segment
        whose click opens the install card."""
        P.SYSTEM_RAM_GB = 48.0
        P._h3_q8_dit_dir = lambda: None
        s = P.h3_status()
        self.assertTrue(s["capable"])
        self.assertFalse(s["available"])
        self.assertTrue(s["needs_q8_dit"])
        self.assertTrue(s["ram_note"])
        self.assertEqual(s["ram_floor_gb"], P.H3_MIN_RAM_GB_Q8)

    def test_a_fully_installed_pack_minus_the_q8_build_is_repairable(self):
        """The state a 48 GB owner who installed H3 before the Q8 build step
        existed is actually in. Nothing is missing and nothing has to be
        re-downloaded, so the card must offer the build, not a 75 GB fetch."""
        old_paths = P.h3_paths
        P.SYSTEM_RAM_GB = 48.0
        P._h3_q8_dit_dir = lambda: None
        P.h3_paths = lambda: {"missing": [], "reason": "ok", "root": "/x",
                              "models": "/x", "repairable": False,
                              "venv_broken": False, "weights_ok": True}
        try:
            s = P.h3_status()
            self.assertTrue(s["capable"])
            self.assertFalse(s["available"])
            self.assertTrue(s["repairable"],
                            "nothing has to be re-downloaded")
            self.assertEqual(s["reason"], "missing_q8_dit")
        finally:
            P.h3_paths = old_paths

    def test_a_capable_mac_is_unaffected(self):
        P.SYSTEM_RAM_GB = 64.0
        P._h3_q8_dit_dir = lambda: None
        s = P.h3_status()
        self.assertFalse(s["needs_q8_dit"])
        self.assertEqual(s["ram_note"], "")
        self.assertEqual(s["ram_floor_gb"], P.H3_MIN_RAM_GB)

    def test_the_chrome_reads_the_lane_floor_not_a_literal_64(self):
        self.assertNotIn("(st.min_ram_gb || 64)", PANEL_SRC)
        self.assertIn("st.ram_floor_gb", PANEL_SRC)

    def test_the_install_card_has_its_own_branch_for_this_band(self):
        """The repair copy below it says the checkout is missing or
        incomplete. Nothing is missing here."""
        self.assertIn("if (h3s.needs_q8_dit) {", PANEL_SRC)
        branch = PANEL_SRC.split("if (h3s.needs_q8_dit) {")[1][:900]
        self.assertIn("h3s.ram_note", branch)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ImageStudioMemoryGuardIsARefusal(unittest.TestCase):
    """Review 2026-09-02: 15 of the 16 unclassified failures on 4.9.0 were
    the Image Studio pre-flight refusing 8 GB / 24 GB Macs an engine that
    holds ~24 GB of weights, with a message pointing at a developer env
    var. The guard is now a refusal (closed slug `image_ram`) that names
    the Mac size that runs the engine, or says to close apps when the Mac
    could fit it but is busy."""

    def _cfg(self):
        return P._build_image_engine_config("qwen_edit_lightning_inline")

    def test_engine_that_can_never_fit_names_the_mac_size(self):
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 24.0), \
             unittest.mock.patch.object(P, "get_memory",
                                        lambda: {"total_gb": 24.0, "used_gb": 6.0}):
            with self.assertRaises(P.RenderRefused) as cm:
                P._preflight_image_job(self._cfg(), engine_override="qwen_edit_lightning_inline")
        self.assertEqual(cm.exception.reason, "image_ram")
        self.assertIn("needs a 32 GB Mac", str(cm.exception))
        self.assertIn("Auto", str(cm.exception))
        self.assertNotIn("PHOSPHENE_SKIP_PREFLIGHT", str(cm.exception))
        self.assertEqual(P._analytics_refusal_reason(str(cm.exception)), "image_ram")

    def test_busy_mac_that_could_fit_is_told_to_close_apps(self):
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 64.0), \
             unittest.mock.patch.object(P, "get_memory",
                                        lambda: {"total_gb": 64.0, "used_gb": 54.0}):
            with self.assertRaises(P.RenderRefused) as cm:
                P._preflight_image_job(self._cfg(), engine_override="qwen_edit_lightning_inline")
        self.assertEqual(cm.exception.reason, "image_ram")
        self.assertIn("free right now", str(cm.exception))

    def test_fit_table_rounds_to_a_mac_size_that_exists(self):
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 64.0):
            self.assertEqual(P._image_engine_fit(24.0), (True, 32))
            self.assertEqual(P._image_engine_fit(32.0), (True, 36))
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 16.0):
            self.assertEqual(P._image_engine_fit(24.0)[0], False)
            self.assertEqual(P._image_engine_fit(8.0), (True, 16))

    def test_pack_missing_is_a_refusal_slug(self):
        msg = ("High quality needs the LTX-2.5 High add-on (the Q8 model), which "
               "isn't downloaded on this Mac yet. Missing 12 file(s): x.")
        self.assertEqual(P._analytics_refusal_reason(msg), "pack_missing")


class StepCountsNeverPadTheTable(unittest.TestCase):
    """Fleet 2026-09-03: an explicit stage-2 count of 12 on 2.5's 4-point
    table loaded the models and then died on "cannot thin … up to 12 steps".
    The panel now caps explicit counts at what the table holds."""

    def test_stage2_is_capped_and_stage1_on_the_distilled_lane(self):
        job = {"params": {"quality": "balanced", "mode": "t2v",
                          "stage1_steps": 40, "stage2_steps": 12}}
        with unittest.mock.patch.object(P, "model_version", lambda *a, **k: {"id": "ltx25"}):
            P._clamp_stage_steps_to_tables(job)
        self.assertEqual(job["params"]["stage2_steps"], 3)
        self.assertLessEqual(job["params"]["stage1_steps"], 8)

    def test_hq_lane_stage1_is_left_alone(self):
        job = {"params": {"quality": "high", "mode": "t2v",
                          "stage1_steps": 15, "stage2_steps": 12}}
        with unittest.mock.patch.object(P, "model_version", lambda *a, **k: {"id": "ltx25"}):
            P._clamp_stage_steps_to_tables(job)
        self.assertEqual(job["params"]["stage1_steps"], 15)
        self.assertEqual(job["params"]["stage2_steps"], 3)

    def test_counts_within_the_table_are_untouched(self):
        job = {"params": {"quality": "balanced", "mode": "t2v",
                          "stage1_steps": 6, "stage2_steps": 2}}
        with unittest.mock.patch.object(P, "model_version", lambda *a, **k: {"id": "ltx25"}):
            P._clamp_stage_steps_to_tables(job)
        self.assertEqual((job["params"]["stage1_steps"], job["params"]["stage2_steps"]), (6, 2))


class LowRamStreamingDecision(unittest.TestCase):
    """v4.9.9: Macs with 24 GB or less stream transformer blocks from disk;
    the environment can force either way for testing or opting out."""

    def test_small_macs_stream_big_ones_do_not(self):
        with unittest.mock.patch.dict(os.environ, {"LTX_LOW_RAM_STREAM": ""}):
            with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 16.0):
                self.assertTrue(P.low_ram_streaming_enabled())
            with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 24.0):
                self.assertTrue(P.low_ram_streaming_enabled())
            with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 32.0):
                self.assertFalse(P.low_ram_streaming_enabled())
            with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 64.0):
                self.assertFalse(P.low_ram_streaming_enabled())

    def test_environment_forces_either_way(self):
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 64.0):
            with unittest.mock.patch.dict(os.environ, {"LTX_LOW_RAM_STREAM": "1"}):
                self.assertTrue(P.low_ram_streaming_enabled())
        with unittest.mock.patch.object(P, "SYSTEM_RAM_GB", 16.0):
            with unittest.mock.patch.dict(os.environ, {"LTX_LOW_RAM_STREAM": "0"}):
                self.assertFalse(P.low_ram_streaming_enabled())
