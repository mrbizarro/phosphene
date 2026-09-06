#!/usr/bin/env python3
"""Framing — a zoom and a reframe per clip, honest in all three outputs.

Locked: neutral is absent; the window is clamped; a slug has nothing to
reframe; the render crops the SOURCE's own pixels before the fit so one
string is right at every size and composes with trims, speed and fades;
the FCP7 export carries it as Basic Motion (scale + centre) and the AE
script as scale + position — the decision travels, never baked pixels."""
from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as panel                                        # noqa: E402
import storyboard_editor as sedit                                    # noqa: E402


def _clip(cid, path, start, end, film_start, **kw):
    c = sedit.new_clip(path, start, end, film_start, id=cid, duration=10.0, source="human")
    c.update(kw)
    return c


def _doc(clips, **kw):
    d = {"version": sedit.EDIT_VERSION, "board_id": "sb_f", "revision": 0, "source": "human",
         "audio": None, "beats": None, "clips": clips, "settings": {}}
    d.update(kw)
    return d


def _codes(doc):
    return [e["code"] for e in sedit.validate_edit(doc)]


class TheModel(unittest.TestCase):
    def test_absent_is_the_whole_frame(self):
        self.assertEqual(sedit.clip_frame({}), {"zoom": 1.0, "x": 0.5, "y": 0.5})
        self.assertTrue(sedit.clip_frame_is_neutral({"frame": {"zoom": 1.0, "x": 0.2}}))
        f = sedit.clip_frame({"frame": {"zoom": 9, "x": -1, "y": 2}})
        self.assertEqual(f, {"zoom": 3.0, "x": 0.0, "y": 1.0})

    def test_validation(self):
        c = _clip("a", "/x/a.mp4", 0, 4, 0, frame={"zoom": 2.0, "x": 0.3, "y": 0.7})
        self.assertEqual(_codes(_doc([c])), [])
        c["frame"] = {"zoom": 5}
        self.assertIn("clip_frame_range", _codes(_doc([c])))
        c["frame"] = {"zoom": "big"}
        self.assertIn("clip_frame", _codes(_doc([c])))
        c["frame"] = "no"
        self.assertIn("clip_frame", _codes(_doc([c])))
        s = {"id": "s", "kind": "slug", "path": None, "start": 0, "end": 2, "film_start": 0,
             "film_end": 2, "source": "human", "locked": False, "frame": {"zoom": 2}}
        self.assertIn("clip_frame_kind", _codes(_doc([s])))

    def test_normalise_writes_only_a_real_reframe_and_clamps_it(self):
        c = _clip("a", "/x/a.mp4", 0, 4, 0, frame={"zoom": 1.0, "x": 0.1})
        self.assertNotIn("frame", sedit.normalise_edit(_doc([c]))["clips"][0])
        c = _clip("a", "/x/a.mp4", 0, 4, 0, frame={"zoom": 2.5, "x": 1.4, "y": 0.25})
        got = sedit.normalise_edit(_doc([c]))["clips"][0]["frame"]
        self.assertEqual(got, {"zoom": 2.5, "x": 1.0, "y": 0.25})
        cut = sedit.edit_to_cuts(_doc([c]))[0]
        self.assertEqual(cut["frame"], {"zoom": 2.5, "x": 1.0, "y": 0.25})
        self.assertNotIn("frame", sedit.edit_to_cuts(_doc([_clip("b", "/x/b.mp4", 0, 4, 0)]))[0])


class TheRender(unittest.TestCase):
    def test_no_reframe_adds_no_filter(self):
        self.assertEqual(panel._sb_frame_term(None), "")
        self.assertEqual(panel._sb_frame_term({"zoom": 1.0}), "")
        self.assertEqual(panel._sb_frame_term({"zoom": "x"}), "")

    def test_the_crop_is_the_sources_own_pixels_centred_on_the_anchor(self):
        t = panel._sb_frame_term({"zoom": 2.0, "x": 0.25, "y": 0.5})
        self.assertTrue(t.startswith("crop=w=2*floor(iw/2.000000/2):h=2*floor(ih/2.000000/2):"))
        self.assertIn("x=(iw-2*floor(iw/2.000000/2))*0.250000", t)
        self.assertIn("y=(ih-2*floor(ih/2.000000/2))*0.500000", t)
        self.assertTrue(t.endswith(","))

    def test_the_crop_sits_after_the_trim_and_before_the_fit(self):
        seg = {"kind": "video", "input": 0,
               "info": {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                        "sample_rate": 48000},
               "window": {"start": 1.0, "end": 4.0}, "adjust": None, "duration": 3.0,
               "path": "/x/a.mp4", "frame": {"zoom": 2.0, "x": 0.5, "y": 0.5}, "speed": 2.0}
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g, _ = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p", segments=[seg])
        chain = [c for c in g.split(";") if c.startswith("[0:v]")][0]
        self.assertLess(chain.index("setpts="), chain.index("crop="))
        self.assertLess(chain.index("crop="), chain.index("fps="))
        self.assertLess(chain.index("fps="), chain.index("scale="))

    def test_the_timeline_pass_carries_it_on_stills_too(self):
        cuts = [{"path": "/x/card.png", "kind": "still", "start": 0.0, "end": 3.0,
                 "film_start": 0.0, "frame": {"zoom": 1.5, "x": 0.5, "y": 0.5}}]
        with mock.patch.object(panel, "_sb_probe_still",
                               return_value={"w": 768, "h": 416, "duration": 0,
                                             "has_audio": False, "sample_rate": 0}):
            segs, _, _ = panel._sb_timeline_segments(cuts)
        self.assertEqual(segs[0]["frame"]["zoom"], 1.5)
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g, _ = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p", segments=segs)
        self.assertIn("setpts=PTS-STARTPTS,crop=", g)


class TheExport(unittest.TestCase):
    def test_fcp7_carries_basic_motion(self):
        seg = {"frame": {"zoom": 2.0, "x": 0.25, "y": 0.5}}
        xml = sedit._fcp7_motion(seg)
        root = ET.fromstring(xml)
        self.assertEqual(root.find("effect/effectid").text, "basic")
        params = {p.find("parameterid").text: p for p in root.findall("effect/parameter")}
        self.assertEqual(params["scale"].find("value").text, "200.00")
        self.assertEqual(params["center"].find("value/horiz").text, "0.5000")
        self.assertEqual(params["center"].find("value/vert").text, "0.0000")
        self.assertEqual(sedit._fcp7_motion({"frame": {"zoom": 1.0}}), "")

    def test_ae_scales_and_moves_the_layer(self):
        c = _clip("a", "/x/a.mp4", 0, 4, 0, frame={"zoom": 2.0, "x": 0.25, "y": 0.5})
        rows = sedit._nle_segments([c], probe=lambda p: {"w": 768, "h": 416, "duration": 10.0,
                                                          "has_audio": True})
        jsx = sedit.ae_jsx(rows, name="f", media={"/x/a.mp4": "a.mp4"}, width=1920,
                           height=1080, fps=24, audio=None)
        self.assertIn("('ADBE Scale').setValue([200.000, 200.000])", jsx)
        # x = 0.25 at 2x: the layer's centre moves right by 0.5 frame widths
        self.assertIn("('ADBE Position').setValue([1920.000, 540.000])", jsx)

    def test_the_project_rows_carry_the_frame(self):
        c = _clip("a", "/x/a.mp4", 0, 4, 0, frame={"zoom": 2.0})
        rows = sedit._nle_segments([c], probe=lambda p: {"w": 768, "h": 416, "duration": 10.0,
                                                          "has_audio": True})
        self.assertEqual(rows[0]["frame"]["zoom"], 2.0)


if __name__ == "__main__":
    unittest.main()
