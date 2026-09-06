#!/usr/bin/env python3
"""Contract gate for Motion Control — the shipped feature nobody could find.

Union Control has worked in this panel for months: official un-gated
Lightricks weights, in `required_files.json`, wired to a real pipeline. Two
long-time users still asked for it as a MISSING feature — @sohaibpp on Pinokio
("Motion control") and a user on X asking for "IC-LoRA support (especially
Union Control / Pose / Depth) exposed in the UI". Both were right about the
naming and wrong about the capability.

Nothing on screen contained the words they were searching for: the mode was
"Control", one click inside a group called "Remix" whose only visible sub-line
said "your media → new video". So what this file gates is the COPY, because
the copy is what was broken — plus the two facts the copy must keep telling
the truth about:

  1. Phosphene has NO preprocessor. Pose / depth / canny SEQUENCES work today
     if the user brings one; deriving one from an ordinary clip is not a thing
     this install does, and a label implying otherwise is the same defect
     wearing the opposite sign.
  2. The adapter is LTX-2.3-trained and the default generation is 2.5. On 2.5
     the motion still transfers (the reference latent is pinned regardless)
     and the prompt's grip on the new subject does not — measured 2026-08-28,
     same clip, same prompt, same seed, both generations.

The markup is asserted structurally (it is an embedded page, like the
Ingredients gate in test_refusal_gates.py); the server-owned facts are
executed.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = Path(tempfile.mkdtemp(prefix="phos-control-disco-"))
os.environ["LTX_STATE_DIR"] = str(STATE)
os.environ["PHOSPHENE_ANALYTICS_DISABLED"] = "1"
os.environ["PHOSPHENE_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("LTX_PORT", "8296")
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as P  # noqa: E402

# Both halves of the panel — the Python server and the page (markup + JS),
# which lives at webapp/index.html since slice 2 of the extraction
# (docs/ARCHITECTURE.md).
PANEL_SRC = ((ROOT / "mlx_ltx_panel.py").read_text(encoding="utf-8")
             + "\n" + (ROOT / "webapp" / "index.html").read_text(encoding="utf-8"))
for _m in sorted((ROOT / "webapp" / "js").glob("*.js")):
    PANEL_SRC += "\n" + _m.read_text(encoding="utf-8")
DOC = ROOT / "docs" / "MOTION_CONTROL.md"


def _remix_sub_chip(remix: str) -> str:
    """The one <button data-remix="…"> line, as written."""
    for line in PANEL_SRC.splitlines():
        if f'data-remix="{remix}"' in line:
            return line
    raise AssertionError(f'no data-remix="{remix}" chip in the markup')


class TestTheNameSaysWhatItIs(unittest.TestCase):
    """The rename, and the words that had to appear somewhere clickable."""

    def test_the_sub_chip_is_called_motion_control(self):
        chip = _remix_sub_chip("control")
        self.assertIn(">Motion Control<", chip,
                      "the chip is the thing two users failed to find; "
                      '"Control" alone does not name what it controls')

    def test_the_chip_names_the_adapter_on_hover(self):
        """A user who read about LTX IC-LoRAs elsewhere has to recognise this
        chip as the thing they read about."""
        chip = _remix_sub_chip("control")
        self.assertIn("title=", chip)
        self.assertIn("Union Control", chip)

    def test_the_parent_pill_names_the_tools_it_hides(self):
        """The Remix pill's sub-line is the ONLY Remix copy visible without a
        click. It described the group and named nothing inside it."""
        pill = next(l for l in PANEL_SRC.splitlines()
                    if 'data-mode="remix"' in l and "mode-chip" in l)
        self.assertIn("motion control", pill.lower())
        self.assertNotIn("your media → new video", pill,
                         "the old sub-line is back; it names no tool")

    def test_every_remix_sub_chip_explains_itself_on_hover(self):
        for remix in ("ingredients", "control", "restore"):
            self.assertIn("title=", _remix_sub_chip(remix), remix)

    def test_the_section_header_names_the_mode_and_the_adapter(self):
        section = PANEL_SRC.split('id="controlSection"')[1][:4000]
        self.assertIn("Motion Control", section)
        self.assertIn("Union Control IC-LoRA", section)
        self.assertNotIn("<h2>Control video</h2>", PANEL_SRC)

    def test_the_copy_says_what_transfers(self):
        section = PANEL_SRC.split('id="controlSection"')[1][:4000].lower()
        for word in ("motion", "camera move", "composition", "pose"):
            self.assertIn(word, section,
                          f"the section never says it transfers {word}")

    def test_the_wiring_ids_survived_the_rename(self):
        """FormData reads #control_video_path and updateDerived toggles
        #controlSection. A copy pass that renames an id renames a feature
        into a 404."""
        for hook in ('id="controlSection"', 'id="controlSrcSelect"',
                     'id="control_video_path"', 'name="control_video_path"',
                     'data-remix="control"'):
            self.assertIn(hook, PANEL_SRC, hook)
        self.assertIn("currentMode === 'control'", PANEL_SRC)


class TestTheHonestHalf(unittest.TestCase):
    """Union Control follows whatever control signal it is given — so pose and
    depth SEQUENCES work, and there is nothing here that makes one."""

    def setUp(self):
        self.section = PANEL_SRC.split('id="controlSection"')[1][:4000]

    def test_the_missing_preprocessor_is_stated_in_the_ui(self):
        self.assertIn("no preprocessor", self.section.lower())

    def test_pose_and_depth_are_offered_as_inputs_you_bring(self):
        low = self.section.lower()
        self.assertIn("pose", low)
        self.assertIn("depth", low)
        self.assertTrue(
            "bring your own" in low or "you have to bring" in low,
            "pose/depth are named without saying who has to produce them")

    def test_the_chip_sub_line_does_not_advertise_pose_or_depth(self):
        """A four-word sub-line cannot carry the caveat, so it must not carry
        the promise either — that is how "Union Control / Pose / Depth" reads
        as three turnkey modes."""
        sub = re.search(r'<span class="mc-sub sub">([^<]*)</span>',
                        _remix_sub_chip("control"))
        self.assertIsNotNone(sub)
        self.assertNotIn("pose", sub.group(1).lower())
        self.assertNotIn("depth", sub.group(1).lower())

    def test_no_preprocessor_is_promised_anywhere(self):
        for banned in ("openpose", "depth estimator", "canny pass will",
                       "we will derive", "auto-derive"):
            self.assertNotIn(banned, PANEL_SRC.lower(), banned)

    def test_the_token_free_fact_matches_required_files(self):
        """The README used to file Control under "gated LoRAs (HDR and
        Lightricks Control)". The Union repo is public and un-gated, and a
        user who believes they need a token does not try the feature."""
        import json
        repos = json.loads((ROOT / "required_files.json").read_text())["repos"]
        union = next(r for r in repos if r.get("key") == "ic_union_control")
        self.assertIs(union.get("gated", False), False)
        self.assertIn("no Hugging Face token",
                      PANEL_SRC.split('id="controlSection"')[1][:4000])


class TestTheGenerationFact(unittest.TestCase):
    """The 2.3 lane, and the half of the mode that stops working on 2.5."""

    def setUp(self):
        self.old_version = P.ACTIVE_MODEL_VERSION

    def tearDown(self):
        P.ACTIVE_MODEL_VERSION = self.old_version

    def test_the_predicate_is_generation_scoped(self):
        self.assertFalse(P.ltx_control_full_repaint("ltx25"))
        self.assertTrue(P.ltx_control_full_repaint("ltx23"))

    def test_control_has_its_own_predicate_not_the_ingredients_one(self):
        """Same answer today, different failures, different fix dates. One
        shared call would silently re-gate Motion Control the day a 2.5
        Ingredients adapter ships."""
        self.assertIn("def ltx_control_full_repaint(", PANEL_SRC)
        body = PANEL_SRC.split("def ltx_control_full_repaint(")[1][:1500]
        self.assertNotIn("ltx_generation_serves_ingredients(", body)

    def test_both_flag_and_sentence_reach_the_page(self):
        P.ACTIVE_MODEL_VERSION = "ltx25"
        payload = P.ltx_tiers_payload()
        self.assertIs(payload["control_full_repaint"], False)
        self.assertEqual(payload["control_generation_note"],
                         P.LTX_CONTROL_GENERATION_NOTE)
        P.ACTIVE_MODEL_VERSION = "ltx23"
        self.assertIs(P.ltx_tiers_payload()["control_full_repaint"], True)

    def test_the_note_says_which_half_still_works(self):
        note = P.LTX_CONTROL_GENERATION_NOTE.lower()
        self.assertIn("ltx-2.3", note)
        self.assertIn("2.5", note)
        # The distinction that makes this diagnosable at all.
        self.assertTrue("still transfer" in note or "still work" in note,
                        "the note must say the motion DOES transfer on 2.5 — "
                        "otherwise it reads as 'broken' and the user stops")
        self.assertIn("prompt", note)

    def test_the_note_names_the_way_out(self):
        note = P.LTX_CONTROL_GENERATION_NOTE
        self.assertIn("LTX_MODEL_VERSION=ltx23", note)
        self.assertIn("2.3 pack from the Train tab", note)

    def test_motion_control_is_NOT_refused(self):
        """Deliberate, and the opposite of the Ingredients ruling: on 2.5 this
        still delivers the camera work, so refusing it would take a working
        feature off the default lane. If a refusal is ever added, this test is
        the place to record the decision — not a silent flip."""
        self.assertNotIn('"control_generation"', PANEL_SRC)
        control_branch = PANEL_SRC.split('if mode == "control":')[1][:3000]
        self.assertNotIn("RenderRefused", control_branch)

    def test_one_reader_for_the_flag(self):
        self.assertIn("function _paintControlGenNote()", PANEL_SRC)
        raw = len(re.findall(r"control_full_repaint", PANEL_SRC))
        # Python: the predicate's def, its own return, the payload key.
        # JS: the single read inside _paintControlGenNote.
        self.assertLessEqual(raw, 6, "a new raw read of the flag crept in — "
                                     "route it through _paintControlGenNote()")

    def test_the_note_is_hidden_where_it_is_untrue(self):
        painter = PANEL_SRC.split("function _paintControlGenNote()")[1][:900]
        self.assertIn("control_full_repaint === false", painter)
        self.assertIn("display = 'none'", painter)


class TestSourceDerivedCanvasRidesTheEngineGrid(unittest.TestCase):
    """Control and Colorize compute their canvas from the SOURCE CLIP inside
    run_job_inner — after make_job's /64 normalisation has already run — so
    they had their own /32 copy of the rule. The two-stage pipeline snaps to
    64 anyway: a 768x416 control clip rendered 768x384 with the log line and
    the sidecar both saying 416, and the reference got crushed 8% vertically
    onto the canvas we named."""

    def test_the_grid_floors_never_raises(self):
        self.assertEqual(P.ltx_floor_canvas(768, 416), (768, 384))
        self.assertEqual(P.ltx_floor_canvas(1000, 500), (960, 448))

    def test_on_grid_dimensions_are_untouched(self):
        for wh in ((1024, 576), (768, 384), (1280, 704), (896, 512)):
            self.assertEqual(P.ltx_floor_canvas(*wh), wh)

    def test_a_tiny_canvas_floors_to_64_not_zero(self):
        self.assertEqual(P.ltx_floor_canvas(60, 40), (64, 64))

    def test_both_source_derived_lanes_use_the_shared_helper(self):
        for marker, name in (('if mode == "control":', "Control"),
                             ('if mode == "restore":', "Colorize")):
            branch = PANEL_SRC.split(marker)[1][:6000]
            self.assertIn("ltx_floor_canvas(", branch, name)
            self.assertNotIn("// 32) * 32", branch,
                             f"{name} still floors its canvas to /32")

    def test_make_job_shares_the_same_helper(self):
        """One rule, one function — or the day the grid changes, two of the
        three lanes learn about it."""
        # one def + five call sites: make_job, Control, Colorize, Upscale ×2,
        # and the H3 form's "LTX ×2" export note (same rule, same helper).
        self.assertEqual(PANEL_SRC.count("ltx_floor_canvas("), 6)


class TestTheDocumentation(unittest.TestCase):
    """A feature two people could not find in the UI is also a feature nobody
    could find in the docs — there was no page for it at all."""

    def test_the_doc_exists(self):
        self.assertTrue(DOC.is_file(), "docs/MOTION_CONTROL.md")

    def test_the_doc_states_the_one_2_5_ic_lora_fact(self):
        text = DOC.read_text(encoding="utf-8").lower()
        self.assertIn("exactly one", text)
        self.assertIn("upscaler", text)
        self.assertIn("2026-08-11", text)

    def test_the_doc_states_the_preprocessor_gap(self):
        text = DOC.read_text(encoding="utf-8").lower()
        self.assertIn("no preprocessor", text)

    def test_the_readme_points_at_it(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Motion Control", readme)
        self.assertIn("docs/MOTION_CONTROL.md", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
