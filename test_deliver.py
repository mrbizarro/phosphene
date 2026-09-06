#!/usr/bin/env python3
"""Deliver as — what the Editor's film is encoded as, and how big.

Locked: the default is byte-for-byte the H.264 the assembler always wrote;
HEVC goes through VideoToolbox with the `hvc1` tag Safari needs; ProRes is a
10-bit 4:2:2 .mov with PCM sound; size is UP only and appended as one Lanczos
scale after the overlays; a different delivery is a different file name; and
the route accepts `.mov`."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as panel                                        # noqa: E402


def _seg(idx, start, end):
    return {"kind": "video", "input": idx,
            "info": {"has_audio": True, "duration": 10.0, "w": 768, "h": 416,
                     "sample_rate": 48000},
            "window": {"start": start, "end": end}, "adjust": None,
            "duration": end - start, "path": f"/x/{idx}.mp4"}


class TheChoice(unittest.TestCase):
    def test_resolution_and_defaults(self):
        d = panel._sb_deliver(None, None)
        self.assertEqual([d["format"], d["size"], d["height"], d["ext"]],
                         ["h264", "native", 0, ".mp4"])
        d = panel._sb_deliver("PRORES", "2160p")
        self.assertEqual([d["format"], d["height"], d["ext"], d["label"]],
                         ["prores", 2160, ".mov", "ProRes 422 HQ · 2160p"])
        self.assertEqual(panel._sb_deliver("gif", "8k")["format"], "h264")

    def test_the_names(self):
        b = {"title": "The car wash"}
        self.assertEqual(panel._sb_film_name(b), "the-car-wash_film.mp4")
        self.assertEqual(panel._sb_film_name(b, panel._sb_deliver("hevc", "1080p")),
                         "the-car-wash_film_hevc_1080p.mp4")
        self.assertEqual(panel._sb_film_name(b, panel._sb_deliver("prores", None)),
                         "the-car-wash_film_prores.mov")

    def test_the_encoder_args(self):
        codec = {"pix_fmt": "yuv420p", "crf": "18"}
        h = panel._sb_encode_args(panel._sb_deliver("h264", None), codec)
        self.assertEqual(h[:6], ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        v = panel._sb_encode_args(panel._sb_deliver("hevc", None), codec)
        self.assertIn("hevc_videotoolbox", v)
        self.assertEqual(v[v.index("-tag:v") + 1], "hvc1")
        p = panel._sb_encode_args(panel._sb_deliver("prores", None), codec)
        self.assertIn("prores_ks", p)
        self.assertEqual(p[p.index("-pix_fmt") + 1], "yuv422p10le")
        self.assertEqual(p[p.index("-c:a") + 1], "pcm_s16le")
        self.assertNotIn("+faststart", p)


class TheGraph(unittest.TestCase):
    def test_no_size_is_the_graph_it_always_was(self):
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g0, _ = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p",
                                               segments=[_seg(0, 0, 4)])
            g1, _ = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p",
                                               segments=[_seg(0, 0, 4)], scale_to=0)
        self.assertEqual(g0, g1)
        self.assertNotIn("scale=-2", g0)

    def test_the_size_is_one_lanczos_scale_after_everything(self):
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g, lbl = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p",
                                                segments=[_seg(0, 0, 4)], scale_to=1080)
        self.assertTrue(g.endswith("[vcat]scale=-2:1080:flags=lanczos[vdl]"))
        self.assertEqual(lbl, "[vdl]")


class TheAssembler(unittest.TestCase):
    def _run(self, deliver):
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
                           "film_start": 0.0}], deliver=deliver)
        return res, seen["cmd"]

    def test_default_is_the_old_command(self):
        res, cmd = self._run(None)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
        self.assertEqual([res["width"], res["height"]], [768, 416])
        self.assertEqual(res["deliver"]["format"], "h264")

    def test_hevc_at_1080p_scales_and_reports_the_new_size(self):
        res, cmd = self._run({"format": "hevc", "size": "1080p"})
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "hevc_videotoolbox")
        self.assertIn("scale=-2:1080:flags=lanczos", cmd[cmd.index("-filter_complex") + 1])
        self.assertEqual([res["width"], res["height"]], [1994, 1080])
        self.assertEqual(res["deliver"]["label"], "HEVC · 1080p")

    def test_prores_asks_the_graph_for_10_bit_422(self):
        res, cmd = self._run({"format": "prores", "size": "native"})
        graph = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("format=yuv422p10le[v0]", graph)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "prores_ks")

    def test_a_1080_cut_delivered_at_1080p_is_not_scaled(self):
        info = {"has_audio": True, "duration": 10.0, "w": 1920, "h": 1080,
                "sample_rate": 48000}
        seen = {}

        def fake_ffmpeg(cmd, label):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"x")
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(panel, "_sb_probe_clip", return_value=info), \
                mock.patch.object(panel, "run_ffmpeg_tracked", side_effect=fake_ffmpeg), \
                mock.patch.object(panel, "_sb_write_film_sidecar"):
            panel._sb_assemble_film(
                ["/x/a.mp4"], Path(d) / "f.mp4",
                timeline=[{"path": "/x/a.mp4", "start": 0.0, "end": 4.0,
                           "film_start": 0.0}], deliver={"format": "h264", "size": "1080p"})
        self.assertNotIn("scale=-2", seen["cmd"][seen["cmd"].index("-filter_complex") + 1])


class TheRoute(unittest.TestCase):
    def test_the_route_names_the_file_by_delivery_and_allows_mov(self):
        from test_storyboard_editor_api import FakeHandler
        board = {"id": "sb_t", "title": "The car wash", "shots": []}
        with mock.patch.object(panel.storyboard, "load_storyboard", return_value=board), \
                mock.patch("storyboard_editor.load_edit", return_value={"version": 2, "clips": [{"id": "a"}]}), \
                mock.patch.object(panel, "_sbe_render_edit",
                                  return_value={"ok": True, "path": "/x/f.mov", "clips": 1,
                                                "duration": 4.0}) as rend:
            h = FakeHandler()
            h.post("edit/render", {"id": "sb_t", "format": "prores", "size": "2160p"})
        self.assertEqual(h.status, 200, h.payload)
        kw = rend.call_args.kwargs
        self.assertEqual(kw["out_name"], "the-car-wash_film_prores_2160p.mov")
        self.assertEqual(kw["deliver"], {"format": "prores", "size": "2160p", "finish": ""})


if __name__ == "__main__":
    unittest.main()


class TheFinish(unittest.TestCase):
    def test_grain_is_added_after_the_size_and_names_the_file(self):
        d = panel._sb_deliver("h264", "1080p", "heavy_grain")
        self.assertEqual([d["finish"], d["grain"], d["label"]],
                         ["heavy_grain", 18, "H.264 · 1080p · heavy grain"])
        self.assertEqual(panel._sb_deliver(None, None, "snow")["finish"], "none")
        self.assertEqual(panel._sb_film_name({"title": "x"}, d), "x_film_1080p_heavy_grain.mp4")
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g, lbl = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p",
                                                segments=[_seg(0, 0, 4)], scale_to=1080, grain=9)
        self.assertTrue(g.endswith("[vcat]scale=-2:1080:flags=lanczos[vdl];[vdl]noise=alls=9:allf=t+u[vgr]"))
        self.assertEqual(lbl, "[vgr]")
        with mock.patch.object(panel, "bt709_vf", return_value=""):
            g0, _ = panel._sb_film_filtergraph([], 768, 416, 48000, "yuv420p", segments=[_seg(0, 0, 4)])
        self.assertNotIn("noise=", g0)
