#!/usr/bin/env python3
"""Editor v2 — speed on the clip, titles on the overlay lane, transitions on
the cut. The server half, plus the client half run side by side.

THREE THINGS ARE LOCKED HERE, and each is the failure that would otherwise
be silent:

1. **A transition never widens the picture lane.** It is a typed object on a
   BOUNDARY (`transitions[]`, `after_clip`), the clips' slots do not move, the
   film stays exactly as long as the timeline says, and a boundary with no
   source handles is REFUSED with a sentence naming the side and the shortfall
   — never rendered as a hard cut. The graph is split at the boundary and
   joined with `xfade`, centred on the cut; the sound takes the lane path and
   is byte-for-byte what it was.

2. **Speed is one term on each clock.** `setpts` on the picture, `atempo`
   (chained past 0.5–2.0) on the sound, `(end - start) / speed` on the film.
   Every strip length, drift, resync and envelope reads the PLAYED length, and
   the client computes the same numbers as the server for the same document.

3. **A title is a card the render draws.** Explicit `kind: "text"` wins in
   `overlay_kind`, every existing overlay still reads from its suffix, the
   font is a FILE resolved and verified before ffmpeg is built, and the raster
   is frame-sized RGBA fed through the same overlay chain as an uploaded PNG.

Run:  ./ltx-2-mlx/env/bin/python3.11 -m pytest -q test_editor_v2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as panel                                        # noqa: E402
import storyboard_editor as sedit                                    # noqa: E402
from test_storyboard_editor_ui import (FUNCTIONS, NODE, SHIM,        # noqa: E402
                                       extract_function, panel_source)


def _clip(cid, path, start, end, film_start, **kw):
    c = sedit.new_clip(path, start, end, film_start, id=cid,
                       duration=kw.pop("duration", 10.0), source="human")
    c.update(kw)
    return c


def _doc(clips, **kw):
    d = {"version": sedit.EDIT_VERSION, "board_id": "sb_v2", "revision": 0,
         "source": "human", "audio": None, "beats": None, "clips": clips,
         "settings": {}}
    d.update(kw)
    return d


def _two(tail=2.0, head=2.0, **kw):
    """Two touching clips: A plays 1..4 of a 10 s take (spare tail = 10-4),
    B plays `head`..`head`+3 of its own 10 s take (spare head = `head`)."""
    a = _clip("a", "/x/a.mp4", 1.0, 4.0, 0.0, duration=4.0 + tail)
    b = _clip("b", "/x/b.mp4", head, head + 3.0, 3.0)
    return _doc([a, b], **kw)


def _codes(doc):
    return [e["code"] for e in sedit.validate_edit(doc)]


def _seg(idx, start, end, *, speed=None, duration=10.0, has_audio=True, **kw):
    s = {"kind": "video", "input": idx,
         "info": {"has_audio": has_audio, "duration": duration, "w": 768,
                  "h": 416, "sample_rate": 48000},
         "window": {"start": start, "end": end}, "adjust": None,
         "duration": (end - start) / (speed or 1.0), "path": f"/x/{idx}.mp4"}
    if speed:
        s["speed"] = speed
    s.update(kw)
    return s


# =============================================================================
# TRANSITIONS
# =============================================================================
class ATransitionOwnsABoundary(unittest.TestCase):
    def test_a_valid_dissolve_validates_and_the_clips_do_not_move(self):
        doc = _two(transitions=[{"id": "t1", "after_clip": "a",
                                 "kind": "dissolve", "duration": 0.5}])
        self.assertEqual(_codes(doc), [])
        norm = sedit.normalise_edit(doc)
        self.assertEqual([c["film_start"] for c in norm["clips"]], [0.0, 3.0])
        self.assertEqual([c["film_end"] for c in norm["clips"]], [3.0, 6.0])
        self.assertEqual(sedit.edit_duration(norm), 6.0)
        r = sedit.resolve_transitions(norm)[0]
        self.assertIsNone(r["problem"])
        self.assertEqual(r["before_clip"], "b")
        self.assertEqual(r["at"], 3.0)
        self.assertEqual(r["duration"], 0.5)         # 12 frames at 24 fps
        self.assertEqual(r["half"], 0.25)

    def test_the_picture_lane_rule_is_untouched(self):
        # The invariant this whole design exists to keep: pictures may not
        # overlap, and WARNING_CODES still holds exactly the one code.
        self.assertEqual(sedit.WARNING_CODES, frozenset({"clips_audio_overlap"}))
        doc = _two()
        doc["clips"][1]["film_start"] = 2.5
        doc["clips"][1]["film_end"] = 5.5
        self.assertIn("clips_overlap", _codes(doc))

    def test_every_transition_code_is_an_error_never_a_warning(self):
        for code in ("transitions_shape", "transition_shape",
                     "transition_unknown_clip", "transition_duplicate_boundary",
                     "transition_no_handles", "transition_kind",
                     "transition_duration", "transition_last_clip"):
            self.assertNotIn(code, sedit.WARNING_CODES)

    def test_shape_and_unknown_clip(self):
        self.assertIn("transitions_shape", _codes(_two(transitions="no")))
        self.assertIn("transition_shape", _codes(_two(transitions=["no"])))
        self.assertIn("transition_unknown_clip",
                      _codes(_two(transitions=[{"after_clip": "zz",
                                                "kind": "dissolve",
                                                "duration": 0.5}])))

    def test_one_per_boundary(self):
        doc = _two(transitions=[
            {"id": "t1", "after_clip": "a", "kind": "dissolve", "duration": 0.5},
            {"id": "t2", "after_clip": "a", "kind": "fade_black", "duration": 0.5}])
        self.assertIn("transition_duplicate_boundary", _codes(doc))

    def test_the_last_clip_has_nothing_to_dissolve_into(self):
        doc = _two(transitions=[{"id": "t1", "after_clip": "b",
                                 "kind": "dissolve", "duration": 0.5}])
        self.assertIn("transition_last_clip", _codes(doc))

    def test_no_handles_names_the_side_and_the_shortfall(self):
        # A has 0.1 s past its out-point; a 0.5 s dissolve needs 0.25 s.
        doc = _two(tail=0.1, transitions=[{"id": "t1", "after_clip": "a",
                                           "kind": "dissolve", "duration": 0.5}])
        errs = [e for e in sedit.validate_edit(doc)
                if e["code"] == "transition_no_handles"]
        self.assertEqual(len(errs), 1)
        self.assertIn("clip 1 has only 0.10s beyond its out-point", errs[0]["message"])
        self.assertIn("needs 0.25s", errs[0]["message"])
        self.assertEqual(errs[0]["where"], 0)
        # ...and the incoming side, in its own words.
        doc = _two(head=0.0, transitions=[{"id": "t1", "after_clip": "a",
                                           "kind": "dissolve", "duration": 0.5}])
        msg = [e for e in sedit.validate_edit(doc)
               if e["code"] == "transition_no_handles"][0]["message"]
        self.assertIn("clip 2 has only 0.00s before its in-point", msg)

    def test_an_unknown_source_length_is_refused_not_guessed(self):
        doc = _two(transitions=[{"id": "t1", "after_clip": "a",
                                 "kind": "dissolve", "duration": 0.5}])
        doc["clips"][0]["duration"] = None
        msg = [e for e in sedit.validate_edit(doc)
               if e["code"] == "transition_no_handles"][0]["message"]
        self.assertIn("not known", msg)

    def test_a_still_and_a_slug_have_infinite_handles(self):
        a = {"id": "a", "kind": "slug", "path": None, "start": 0, "end": 3,
             "film_start": 0.0, "film_end": 3.0, "source": "human",
             "locked": False}
        b = _clip("b", "/x/b.mp4", 2.0, 5.0, 3.0)
        doc = _doc([a, b], transitions=[{"id": "t1", "after_clip": "a",
                                         "kind": "fade_black",
                                         "duration": 1.0}])
        self.assertEqual(_codes(doc), [])

    def test_the_duration_is_clamped_to_half_the_shorter_side_and_2s(self):
        doc = _two(tail=5.0, head=5.0,
                   transitions=[{"id": "t1", "after_clip": "a",
                                 "kind": "dissolve", "duration": 9.0}])
        r = sedit.resolve_transitions(doc)[0]
        self.assertEqual(r["duration"], 1.5)         # half of the 3 s B
        # ...and to even frames: 0.8 s is 19.2 frames, which becomes 20.
        doc["transitions"][0]["duration"] = 0.8
        self.assertAlmostEqual(sedit.resolve_transitions(doc)[0]["duration"],
                               20 / 24, places=6)
        # The document keeps what the user typed.
        self.assertEqual(sedit.normalise_edit(doc)["transitions"][0]["duration"], 0.8)

    def test_normalise_is_stable_and_absent_when_empty(self):
        doc = _two(transitions=[{"after_clip": "a", "kind": "dissolve",
                                 "duration": 0.5}])
        once = sedit.normalise_edit(doc)
        self.assertTrue(once["transitions"][0]["id"])
        twice = sedit.normalise_edit(json.loads(json.dumps(once)))
        self.assertEqual(once["transitions"], twice["transitions"])
        self.assertNotIn("transitions", sedit.normalise_edit(_two(transitions=[])))
        self.assertNotIn("transitions", sedit.normalise_edit(_two()))

    def test_the_cut_list_stamps_the_pair_and_only_a_resolved_pair(self):
        doc = _two(transitions=[{"id": "t1", "after_clip": "a",
                                 "kind": "dissolve", "duration": 0.5}])
        cuts = sedit.edit_to_cuts(doc)
        self.assertEqual(cuts[0]["transition"], {"kind": "dissolve", "duration": 0.5})
        self.assertEqual(cuts[1]["tx_in"], 0.25)
        bad = _two(tail=0.0, transitions=[{"id": "t1", "after_clip": "a",
                                           "kind": "dissolve", "duration": 0.5}])
        cuts = sedit.edit_to_cuts(bad)
        self.assertNotIn("transition", cuts[0])
        self.assertNotIn("tx_in", cuts[1])

    def test_a_document_without_transitions_builds_the_identical_cut_list(self):
        self.assertEqual(sedit.edit_to_cuts(_two()),
                         sedit.edit_to_cuts(_two(transitions=[])))
        for e in sedit.edit_to_cuts(_two()):
            self.assertNotIn("transition", e)
            self.assertNotIn("tx_in", e)
            self.assertNotIn("speed", e)


class TheGraphSplitsAtTheBoundary(unittest.TestCase):
    def graph(self, segs, **kw):
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            return panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p",
                                              segments=segs, **kw)

    def test_no_transition_is_the_concat_it_always_was(self):
        g, lbl = self.graph([_seg(0, 1.0, 4.0), _seg(1, 2.0, 5.0)])
        self.assertIn("[v0][a0][v1][a1]concat=n=2:v=1:a=1[vcat][aout]", g)
        self.assertNotIn("xfade", g)
        self.assertEqual(lbl, "[vcat]")

    def test_the_two_runs_are_crossfaded_centred_on_the_cut(self):
        segs = [_seg(0, 1.0, 4.0, transition={"kind": "dissolve", "duration": 0.5}),
                _seg(1, 2.0, 5.0, tx_in=0.25)]
        g, _ = self.graph(segs)
        # The outgoing picture pulls 0.25 s past its out-point, the incoming
        # 0.25 s before its in-point; the audio windows are UNTOUCHED.
        self.assertIn("[0:v]trim=start=1.000000:end=4.250000,setpts=PTS-STARTPTS", g)
        self.assertIn("[1:v]trim=start=1.750000:end=5.000000,setpts=PTS-STARTPTS", g)
        self.assertIn("[0:a]atrim=start=1.000000:end=4.000000,", g)
        self.assertIn("[1:a]atrim=start=2.000000:end=5.000000,", g)
        self.assertIn("[v0]concat=n=1:v=1:a=0[pg0]", g)
        self.assertIn("[v1]concat=n=1:v=1:a=0[pg1]", g)
        # offset = len(run 0) - duration = 3.25 - 0.5
        self.assertIn("[pg0][pg1]xfade=transition=fade:duration=0.500000:"
                      "offset=2.750000[vcat]", g)
        # ...and the sound takes the lane path, end to end, gapless.
        self.assertIn("[a0][a1]concat=n=2:v=0:a=1[aout]", g)
        self.assertIn("apad,atrim=0:3.000000,asetpts=PTS-STARTPTS[a0]", g)

    def test_fade_black_is_ffmpegs_fadeblack(self):
        segs = [_seg(0, 1.0, 4.0, transition={"kind": "fade_black", "duration": 1.0}),
                _seg(1, 2.0, 5.0, tx_in=0.5)]
        g, _ = self.graph(segs)
        self.assertIn("xfade=transition=fadeblack:duration=1.000000", g)

    def test_three_clips_two_boundaries_chain_offsets(self):
        segs = [_seg(0, 1.0, 4.0, transition={"kind": "dissolve", "duration": 0.5}),
                _seg(1, 2.0, 5.0, tx_in=0.25,
                     transition={"kind": "dissolve", "duration": 1.0}),
                _seg(2, 1.0, 3.0, tx_in=0.5)]
        g, _ = self.graph(segs)
        self.assertIn("[pg0][pg1]xfade=transition=fade:duration=0.500000:offset=2.750000[px1]", g)
        # after the first join the picture is 3.25 + 3.75 - 0.5 = 6.5 long
        self.assertIn("[px1][pg2]xfade=transition=fade:duration=1.000000:offset=5.500000[vcat]", g)

    def test_a_dangling_transition_is_reconciled_by_the_timeline_pass(self):
        cuts = [{"path": "/x/a.mp4", "start": 1.0, "end": 4.0, "film_start": 0.0,
                 "transition": {"kind": "dissolve", "duration": 0.5}},
                {"path": "/x/missing.mp4", "start": 2.0, "end": 5.0,
                 "film_start": 3.0, "tx_in": 0.25}]
        info = {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                "sample_rate": 48000}
        with mock.patch.object(panel, "_sb_probe_clip",
                               side_effect=lambda p: None if "missing" in str(p) else info):
            segs, unreadable, _ = panel._sb_timeline_segments(cuts)
        self.assertEqual(unreadable, ["missing.mp4"])
        self.assertNotIn("transition", segs[0])

    def test_a_still_input_loops_for_its_handles_too(self):
        cuts = [{"path": "/x/a.mp4", "start": 1.0, "end": 4.0, "film_start": 0.0,
                 "transition": {"kind": "dissolve", "duration": 0.5}},
                {"path": "/x/card.png", "kind": "still", "start": 0.0, "end": 3.0,
                 "film_start": 3.0, "tx_in": 0.25}]
        info = {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                "sample_rate": 48000}
        with mock.patch.object(panel, "_sb_probe_clip", return_value=info), \
                mock.patch.object(panel, "_sb_probe_still",
                                  return_value={"w": 768, "h": 416, "duration": 0,
                                                "has_audio": False, "sample_rate": 0}):
            segs, _, inputs = panel._sb_timeline_segments(cuts)
        self.assertEqual(inputs[1][5], "3.250000")
        self.assertEqual(segs[1]["duration"], 3.0)

    def test_the_render_refuses_with_the_validators_sentence(self):
        doc = _two(tail=0.0, transitions=[{"id": "t1", "after_clip": "a",
                                           "kind": "dissolve", "duration": 0.5}])
        board = {"id": "sb_v2", "title": "v2"}
        with mock.patch.object(panel, "_sb_assemble_film") as asm:
            res = panel._sbe_render_edit(board, doc)
        self.assertFalse(res["ok"])
        self.assertEqual(res["status"], 400)
        self.assertIn("beyond its out-point", res["error"])
        asm.assert_not_called()


# =============================================================================
# SPEED
# =============================================================================
class SpeedOnTheClip(unittest.TestCase):
    def test_absent_is_1x_and_stills_have_no_clock(self):
        self.assertEqual(sedit.clip_speed(_clip("a", "/x/a.mp4", 0, 3, 0)), 1.0)
        self.assertEqual(sedit.clip_speed({"kind": "still", "speed": 2.0}), 1.0)
        self.assertEqual(sedit.clip_speed(_clip("a", "/x/a.mp4", 0, 3, 0, speed=9)), 4.0)
        self.assertEqual(sedit.clip_speed(_clip("a", "/x/a.mp4", 0, 3, 0, speed=0.01)), 0.25)

    def test_the_slot_is_the_window_over_the_speed(self):
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=2.0)
        c["film_end"] = 2.0
        self.assertEqual(_codes(_doc([c])), [])
        c["film_end"] = 4.0
        self.assertIn("clip_length_mismatch", _codes(_doc([c])))
        self.assertEqual(sedit.clip_audio(c)["len"], 2.0)
        self.assertEqual(sedit.clip_audio(c)["speed"], 2.0)

    def test_range_and_kind(self):
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=8.0)
        c["film_end"] = 0.5
        self.assertIn("clip_speed_range", _codes(_doc([c])))
        s = {"id": "s", "kind": "still", "path": str(ROOT / "assets" / "icon.png"),
             "start": 0, "end": 2, "film_start": 0, "film_end": 2,
             "source": "human", "locked": False, "speed": 2.0}
        self.assertIn("clip_speed_kind", _codes(_doc([s])))

    def test_normalise_writes_only_a_non_unity_speed(self):
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=1.0)
        self.assertNotIn("speed", sedit.normalise_edit(_doc([c]))["clips"][0])
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=0.5)
        c["film_end"] = 8.0
        self.assertEqual(sedit.normalise_edit(_doc([c]))["clips"][0]["speed"], 0.5)
        self.assertEqual(sedit.edit_to_cuts(_doc([c]))[0]["speed"], 0.5)

    def test_drift_and_resync_are_on_the_films_clock(self):
        c = _clip("a", "/x/a.mp4", 2.0, 6.0, 10.0, speed=2.0)
        c["film_end"] = 12.0
        # The strip starts at source 3.0, which the picture plays at 10.5.
        c["audio"] = {"start": 3.0, "end": 6.0, "film_start": 10.5}
        self.assertEqual(sedit.clip_audio_drift(c), 0.0)
        c["audio"]["film_start"] = 11.0
        self.assertEqual(sedit.clip_audio_drift(c), 0.5)
        self.assertEqual(sedit.clip_audio_resync(c), 10.5)

    def test_the_envelope_reads_the_played_length(self):
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=2.0,
                  afx={"fade_out": 1.0})
        c["film_end"] = 2.0
        cut = sedit.edit_to_cuts(_doc([c]))[0]
        self.assertEqual(cut["gain"], [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
        self.assertEqual(sedit.audible_strips(_doc([c])), [[0.0, 2.0]])

    def test_atempo_is_chained_past_the_window(self):
        self.assertEqual(panel._sb_atempo_term(1.0), "")
        self.assertEqual(panel._sb_atempo_term(2.0), "atempo=2.000000,")
        self.assertEqual(panel._sb_atempo_term(4.0), "atempo=2.000000,atempo=2.000000,")
        self.assertEqual(panel._sb_atempo_term(0.25), "atempo=0.500000,atempo=0.500000,")
        self.assertEqual(panel._sb_atempo_term(3.0), "atempo=2.000000,atempo=1.500000,")

    def test_the_graph_retimes_picture_and_sound_together(self):
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g, _ = panel._sb_film_filtergraph(
                [], 768, 416, 48000, "yuv420p",
                segments=[_seg(0, 1.0, 5.0, speed=2.0)])
        self.assertIn("[0:v]trim=start=1.000000:end=5.000000,"
                      "setpts=(PTS-STARTPTS)/2.000000,fps=24,", g)
        self.assertIn("[0:a]atrim=start=1.000000:end=5.000000,asetpts=PTS-STARTPTS,"
                      "atempo=2.000000,aresample=48000,", g)
        self.assertIn("apad,atrim=0:2.000000,asetpts=PTS-STARTPTS[a0]", g)

    def test_the_split_plan_lays_a_retimed_strip_at_its_played_length(self):
        segs = [_seg(0, 0.0, 4.0, speed=2.0, mute=False),
                _seg(1, 0.0, 2.0, audio={"start": 0.0, "end": 2.0, "delta": 0.0})]
        plan = panel._sb_split_audio_plan(segs)
        self.assertTrue(plan["split"])
        self.assertEqual(plan["total"], 4.0)
        self.assertEqual(plan["lanes"][0]["len"], 2.0)
        self.assertEqual(plan["lanes"][0]["end"], 4.0)     # source seconds
        self.assertEqual(plan["lanes"][0]["speed"], 2.0)
        self.assertEqual(plan["lanes"][1]["at"], 2.0)


# =============================================================================
# TITLES
# =============================================================================
class ATitleIsACardTheRenderDraws(unittest.TestCase):
    def _title(self, **kw):
        o = {"id": "o1", "kind": "text", "text": "FIN", "film_start": 1.0,
             "film_end": 3.0}
        o.update(kw)
        return o

    def test_explicit_kind_wins_and_the_suffix_still_decides_the_rest(self):
        self.assertEqual(sedit.overlay_kind(self._title()), "text")
        self.assertEqual(sedit.overlay_kind({"path": "/x/card.png"}), "still")
        self.assertEqual(sedit.overlay_kind({"path": "/x/loop.mp4"}), "video")
        self.assertEqual(sedit.overlay_kind({"kind": "still", "path": "/x/a.mp4"}), "still")

    def test_validate_and_normalise(self):
        doc = _doc([_clip("a", "/x/a.mp4", 0, 5, 0)], overlays=[self._title()])
        self.assertEqual(_codes(doc), [])
        n = sedit.normalise_edit(doc)["overlays"][0]
        self.assertEqual(n["kind"], "text")
        self.assertIsNone(n["path"])
        self.assertEqual([n["start"], n["end"]], [0.0, 2.0])
        self.assertNotIn("style", n)                      # defaults are absent
        doc["overlays"][0]["text"] = "  "
        self.assertIn("overlay_text_empty", _codes(doc))
        doc["overlays"][0]["text"] = "x"
        doc["overlays"][0]["style"] = {"font_size": 9999, "x": 3, "align": "up",
                                       "color": "red", "box": "yes"}
        codes = _codes(doc)
        self.assertIn("overlay_text_style_range", codes)
        self.assertIn("overlay_text_style", codes)

    def test_the_accessor_clamps_and_writes_only_what_differs(self):
        o = self._title(style={"font_size": 9999, "x": 3, "color": "#fc0",
                               "box": True, "box_opacity": 2})
        t = sedit.overlay_text(o)
        self.assertEqual(t["style"]["font_size"], 400)
        self.assertEqual(t["style"]["x"], 1.0)
        self.assertEqual(t["style"]["color"], "#ffcc00")
        self.assertTrue(t["style"]["box"])
        self.assertEqual(t["style"]["box_opacity"], 1.0)
        doc = _doc([_clip("a", "/x/a.mp4", 0, 5, 0)], overlays=[o])
        n = sedit.normalise_edit(doc)["overlays"][0]
        self.assertEqual(n["style"], {"font_size": 400, "x": 1.0, "color": "#ffcc00",
                                      "box": True, "box_opacity": 1.0})

    def test_the_font_is_a_file_and_a_missing_one_is_a_sentence(self):
        with mock.patch.dict(os.environ, {"LTX_TITLE_FONT": "/nope/none.ttf"}):
            self.assertIsNone(sedit.title_font_path())
            self.assertIn("/nope/none.ttf", sedit.title_font_problem())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LTX_TITLE_FONT", None)
            with mock.patch.object(sedit, "TITLE_FONT_CANDIDATES", ()):
                self.assertIn("no font for titles", sedit.title_font_problem())

    def test_the_raster_is_frame_sized_rgba_with_ink_where_the_anchor_is(self):
        if sedit.title_font_path() is None:
            raise unittest.SkipTest("no system font on this machine")
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p = sedit.render_title(self._title(style={"y": 0.9, "box": True}),
                                   640, 360, Path(d) / "t.png")
            im = Image.open(p)
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.size, (640, 360))
            a = im.getchannel("A")
            self.assertEqual(a.getpixel((320, 20)), 0)          # sky is clear
            self.assertGreater(a.getpixel((320, 324)), 0)       # ink at y=0.9

    def test_the_assembler_feeds_the_raster_through_the_overlay_chain(self):
        if sedit.title_font_path() is None:
            raise unittest.SkipTest("no system font on this machine")
        info = {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                "sample_rate": 48000}
        seen = {}

        def fake_ffmpeg(cmd, label):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"x")
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(panel, "_sb_probe_clip", return_value=info), \
                mock.patch.object(panel, "run_ffmpeg_tracked", side_effect=fake_ffmpeg), \
                mock.patch.object(panel, "_sb_write_film_sidecar"):
            res = panel._sb_assemble_film(
                ["/x/a.mp4"], Path(d) / "f.mp4",
                timeline=[{"path": "/x/a.mp4", "start": 0.0, "end": 4.0,
                           "film_start": 0.0}],
                overlays=[self._title()])
            self.assertTrue(res["ok"], res)
            cmd = seen["cmd"]
            i = cmd.index("-filter_complex")
            graph = cmd[i + 1]
            png = [a for a in cmd if a.endswith(".png")][0]
            self.assertTrue(Path(png).is_file())
            self.assertIn("/.titles/title_", png)
            self.assertIn("-loop", cmd)
            self.assertIn("[1:v]format=rgba,scale=768:416:flags=lanczos,format=rgba", graph)
            self.assertIn("enable='between(t,1.000000,3.000000)'", graph)

    def test_the_assembler_refuses_without_a_font(self):
        info = {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                "sample_rate": 48000}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(panel, "_sb_probe_clip", return_value=info), \
                mock.patch.object(panel, "run_ffmpeg_tracked") as ff, \
                mock.patch.dict(os.environ, {"LTX_TITLE_FONT": "/nope/none.ttf"}):
            res = panel._sb_assemble_film(
                ["/x/a.mp4"], Path(d) / "f.mp4",
                timeline=[{"path": "/x/a.mp4", "start": 0.0, "end": 4.0,
                           "film_start": 0.0}],
                overlays=[self._title()])
        self.assertFalse(res["ok"])
        self.assertIn("/nope/none.ttf", res["error"])
        ff.assert_not_called()


# =============================================================================
# THE CLIENT AGREES WITH THE SERVER
# =============================================================================
BODY = r"""
const out = {};
const A = { id: 'a', path: '/x/a.mp4', start: 1, end: 4, film_start: 0, film_end: 3,
            duration: 6, source: 'human', locked: false };
const B = { id: 'b', path: '/x/b.mp4', start: 2, end: 5, film_start: 3, film_end: 6,
            duration: 10, source: 'human', locked: false };
const TX = [{ id: 't1', after_clip: 'a', kind: 'dissolve', duration: 0.5 }];
out.resolved = sbeTxResolve([A, B], TX, 24);
out.short = sbeTxResolve([Object.assign({}, A, { duration: 4.1 }), B], TX, 24);
out.clamp = sbeTxDuration({ duration: 9 }, 3, 3, 24);
out.even = sbeTxDuration({ duration: 0.8 }, 3, 3, 24);
out.edges = [sbeTxEdges([A, B], TX, 'a', 24), sbeTxEdges([A, B], TX, 'b', 24)];
out.prune = sbeTxPrune(TX.concat([{ id: 't2', after_clip: 'b', kind: 'dissolve', duration: 1 },
                                   { id: 't3', after_clip: 'zz', kind: 'dissolve', duration: 1 }]),
                       [A, B]).map(t => t.id);
out.repoint = sbeTxRepoint(TX, 'a', 'a2')[0].after_clip;
out.set = sbeTxSet([], 'a', 'fade_black', 9).transitions[0];
out.del = sbeTxDelete(TX, 'a').transitions.length;
out.opacity = [2.5, 2.875, 3].map(t => sbeFadeOpacityAt(A, t, { head: 0, tail: 0.25 }));
// speed
const C = { id: 'c', path: '/x/c.mp4', start: 0, end: 4, film_start: 0, film_end: 4,
            duration: 10, source: 'human', locked: false };
const D = { id: 'd', path: '/x/d.mp4', start: 0, end: 2, film_start: 4, film_end: 6,
            duration: 10, source: 'human', locked: false };
const r = sbeSetSpeed([C, D], 'c', 2);
out.speed = [r.ok, sbeById(r.clips, 'c').speed, sbeById(r.clips, 'c').film_end,
             sbeById(r.clips, 'd').film_start, sbeLen(sbeById(r.clips, 'c'))];
const c2 = sbeById(r.clips, 'c');
out.audioLen = sbeClipAudio(c2).len;
const tr = sbeTrim(r.clips.map(x => Object.assign({}, x)), 'c', 'r', 3, { ripple: true });   // ⌘: grow past the neighbour; the speed arithmetic is the subject
out.trimR = [sbeById(tr.clips, 'c').end, sbeById(tr.clips, 'c').film_end];
const sp = sbeSplitAt(r.clips.map(x => Object.assign({}, x)), 1, 'new',
                      [{ id: 't9', after_clip: 'c', kind: 'dissolve', duration: 0.5 }]);
out.split = [sp.ok, sbeById(sp.clips, 'c').end, sbeById(sp.clips, 'new').start,
             sbeById(sp.clips, 'new').film_start, sp.transitions[0].after_clip];
const dr = Object.assign({}, c2, { audio: { start: 1, end: 3, film_start: 0.75 } });
out.drift = [sbeAudioDrift(dr), sbeAudioDrift(Object.assign({}, dr, { audio: { start: 1, end: 3, film_start: 0.5 } }))];
out.clean = [sbeCleanClip(c2).speed, sbeCleanClip(C).speed === undefined];
// framing: clamped, neutral absent, a slug refuses
out.frame = (() => {
  const r = sbeSetFraming([Object.assign({}, C)], 'c', 'zoom', 2.5);
  const r2 = sbeSetFraming(r.clips, 'c', 'x', 1.7);
  const r3 = sbeSetFraming(r2.clips, 'c', 'zoom', 1);
  const slug = sbeSetFraming([{ id: 's', kind: 'slug', start: 0, end: 2, film_start: 0, film_end: 2 }], 's', 'zoom', 2);
  return [sbeById(r.clips, 'c').frame, sbeById(r2.clips, 'c').frame, sbeById(r3.clips, 'c').frame === undefined,
          slug.ok, sbeCleanClip(sbeById(r2.clips, 'c')).frame, sbeFraming({}).zoom];
})();
// duplicate: same window, speed and fades, linked sound, right after, ripple
const dup = sbeDuplicate([Object.assign({}, c2, { fx: { fade_in: 0.5 }, audio: { start: 0, end: 4, film_start: 0.5 } }), Object.assign({}, D)], 'c');
out.dup = [dup.ok, dup.clips.length, dup.added.start, dup.added.end, dup.added.speed,
           dup.added.fx.fade_in, dup.added.audio === undefined, dup.added.film_start,
           sbeById(dup.clips, 'd').film_start];
// titles
out.kind = [sbeOvKind({ kind: 'text', text: 'x' }), sbeOvKind({ path: '/x/a.png' }), sbeOvKind({ path: '/x/a.mp4' })];
out.text = sbeOvText({ kind: 'text', text: 'A\r\nB', style: { font_size: 9999, x: 3, color: '#fc0', box: true, box_opacity: 2 } });
out.add = sbeOvAdd([], { kind: 'text', text: 'FIN', duration_s: 2 }, 5).added;
out.rgba = sbeRgba('#ff0000', 0.5);
process.stdout.write(JSON.stringify(out));
"""


def run_client() -> dict:
    if NODE is None:
        raise unittest.SkipTest("node not on PATH")
    source = panel_source()
    script = (SHIM + "\n".join(extract_function(n, source) for n in FUNCTIONS)
              + "\n(async () => {\n" + BODY + "\n})();\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = Path(fh.name)
    try:
        result = subprocess.run([NODE, str(path)], capture_output=True,
                                text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + "\n" + result.stderr)
        return json.loads(result.stdout)
    finally:
        path.unlink(missing_ok=True)


class TheClientAgrees(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_client()

    def test_transitions_resolve_the_same_way(self):
        r = self.r["resolved"][0]
        self.assertIsNone(r["problem"])
        self.assertEqual([r["before_clip"], r["at"], r["duration"], r["half"]],
                         ["b", 3.0, 0.5, 0.25])
        s = self.r["short"][0]
        self.assertEqual(s["problem"]["code"], "transition_no_handles")
        self.assertIn("only 0.10s beyond its out-point", s["problem"]["message"])
        self.assertEqual(self.r["clamp"], 1.5)
        self.assertAlmostEqual(self.r["even"], 20 / 24, places=6)
        # ...the same numbers Python produces for the same document.
        a = _clip("a", "/x/a.mp4", 1.0, 4.0, 0.0, duration=6.0)
        b = _clip("b", "/x/b.mp4", 2.0, 5.0, 3.0)
        py = sedit.resolve_transitions(_doc([a, b], transitions=[
            {"id": "t1", "after_clip": "a", "kind": "dissolve", "duration": 0.5}]))[0]
        self.assertEqual([py["duration"], py["half"], py["at"]],
                         [r["duration"], r["half"], r["at"]])

    def test_edges_prune_repoint_set_delete(self):
        self.assertEqual(self.r["edges"], [{"head": 0, "tail": 0.25}, {"head": 0.25, "tail": 0}])
        self.assertEqual(self.r["prune"], ["t1"])     # t2 is on the last clip
        self.assertEqual(self.r["repoint"], "a2")
        self.assertEqual([self.r["set"]["kind"], self.r["set"]["duration"]], ["fade_black", 2.0])
        self.assertEqual(self.r["del"], 0)
        self.assertEqual(self.r["opacity"], [1, 0.5, 0])

    def test_speed_reflows_and_retimes(self):
        self.assertEqual(self.r["speed"], [True, 2, 2, 2, 2])
        self.assertEqual(self.r["audioLen"], 2)
        # a right trim of one film second at 2x moves the out-point two
        # source seconds
        self.assertEqual(self.r["trimR"], [6, 3])
        self.assertEqual(self.r["split"], [True, 2, 2, 1, "new"])
        self.assertEqual(self.r["drift"], [0.25, 0])
        self.assertEqual(self.r["clean"], [2, True])

    def test_framing_is_clamped_and_neutral_is_absent(self):
        f = self.r["frame"]
        self.assertEqual(f[0], {"zoom": 2.5, "x": 0.5, "y": 0.5})
        self.assertEqual(f[1], {"zoom": 2.5, "x": 1, "y": 0.5})
        self.assertTrue(f[2])
        self.assertFalse(f[3])
        self.assertEqual(f[4], {"zoom": 2.5, "x": 1, "y": 0.5})
        self.assertEqual(f[5], 1)
        # ...and the same numbers Python clamps to
        self.assertEqual(sedit.clip_frame({"frame": {"zoom": 2.5, "x": 1.7}}),
                         {"zoom": 2.5, "x": 1.0, "y": 0.5})

    def test_duplicate_is_the_same_shot_again_right_after(self):
        # c is 0–4 of the take at 2x = 2 s on the film; the copy starts at 2
        # and d slides from 4 to 6
        self.assertEqual(self.r["dup"], [True, 3, 0, 4, 2, 0.5, True, 2, 4])

    def test_titles_read_the_same_defaults_and_clamps(self):
        self.assertEqual(self.r["kind"], ["text", "still", "video"])
        t = self.r["text"]
        self.assertEqual(t["text"], "A\nB")
        py = sedit.overlay_text({"kind": "text", "text": "A\r\nB",
                                 "style": {"font_size": 9999, "x": 3, "color": "#fc0",
                                           "box": True, "box_opacity": 2}})
        self.assertEqual(t["style"], py["style"])
        self.assertEqual(self.r["add"]["kind"], "text")
        self.assertIsNone(self.r["add"]["path"])
        self.assertEqual(self.r["rgba"], "rgba(255,0,0,0.500)")

    def test_the_constants_are_the_panels(self):
        import re
        src = (ROOT / "webapp" / "js" / "editor.js").read_text()
        for name, val in (("SBE_SPEED_MIN", sedit.SPEED_MIN),
                          ("SBE_SPEED_MAX", sedit.SPEED_MAX),
                          ("SBE_TX_MAX", sedit.TRANSITION_MAX),
                          ("SBE_TEXT_MAX", sedit.TEXT_MAX_CHARS)):
            m = re.search(rf"^const {name} = ([0-9.]+);", src, re.M)
            self.assertIsNotNone(m, name)
            self.assertEqual(float(m.group(1)), float(val), name)


if __name__ == "__main__":
    unittest.main()


class TheExportStillOpens(unittest.TestCase):
    """The NLE export neither carries a title nor a transition yet, and it
    must say nothing false: a retimed clip rides as in/out vs start/end, a
    title (no path) is skipped, and nothing raises."""

    def test_export_with_a_title_and_a_retimed_clip(self):
        c = _clip("a", "/x/a.mp4", 0.0, 4.0, 0.0, speed=2.0)
        c["film_end"] = 2.0
        doc = sedit.normalise_edit(_doc([c], overlays=[
            {"id": "o1", "kind": "text", "text": "FIN", "film_start": 0.5,
             "film_end": 1.5}]))
        probe = lambda p: {"w": 768, "h": 416, "duration": 10.0, "has_audio": True}
        rows = sedit._nle_segments(doc["clips"], probe=probe)
        self.assertEqual(rows[0]["speed"], 2.0)
        self.assertEqual([rows[0]["start"], rows[0]["end"]], [0.0, 4.0])
        self.assertEqual([rows[0]["film_start"], rows[0]["film_end"]], [0.0, 2.0])
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sedit.Path, "is_file", return_value=True), \
                mock.patch.object(sedit, "_link_or_copy", return_value="link"):
            res = sedit.export_nle(doc["clips"], d, name="v2", probe=probe,
                                   overlays=sedit.overlay_items(doc))
            self.assertTrue(res["ok"])
            self.assertEqual(res["clips"], 1)
            self.assertNotIn("FIN", Path(res["xml"]).read_text())
