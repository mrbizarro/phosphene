#!/usr/bin/env python3
"""The sliding-window planner for LTX: geometry and the per-window prompt
contract. Pure arithmetic, so a wrong stride is a red test and not a
minute-long render that stutters at every seam."""
from __future__ import annotations

import math
import unittest

import ltx_windows as lw


class TheGeometry(unittest.TestCase):
    def test_one_pass_when_the_window_covers_it(self):
        p = lw.plan_windows(121)
        self.assertEqual(p["count"], 1)
        self.assertEqual(p["delivered_frames"], 121)
        self.assertEqual(p["windows"][0], {"index": 0, "start_frame": 0, "frames": 121,
                                           "new_frames": 121, "kept_frames": 121,
                                           "start_sec": 0.0, "end_sec": 5.0})

    def test_the_formula(self):
        # stride = window - discard - overlap = 121 - 0 - 9 = 112
        # count  = 1 + ceil((total - window + discard) / stride)
        for total in (241, 481, 721, 1441):
            p = lw.plan_windows(total)
            self.assertEqual(p["stride"], 112)
            self.assertEqual(p["count"], 1 + math.ceil((total - 121) / 112))
            self.assertGreaterEqual(p["delivered_frames"], total)
            self.assertEqual(p["delivered_frames"], 121 + (p["count"] - 1) * 112)
            # every later window adds exactly one stride of NEW picture
            for w in p["windows"][1:]:
                self.assertEqual(w["new_frames"], 112)
            # ...and re-sees the previous tail
            self.assertEqual(p["windows"][1]["start_frame"], 121 - 9)

    def test_discard_shortens_what_is_kept(self):
        p = lw.plan_windows(481, discard=16)
        self.assertEqual(p["stride"], 96)
        self.assertEqual(p["windows"][1]["start_frame"], 121 - 16 - 9)
        # a later window generates stride + discard and keeps stride
        self.assertEqual(p["windows"][1]["new_frames"], 112)
        self.assertEqual(p["windows"][1]["kept_frames"], 105 + 96)
        self.assertEqual(p["delivered_frames"], 121 - 16 + (p["count"] - 1) * 96)
        self.assertGreaterEqual(p["delivered_frames"], 481)

    def test_the_latent_grid_is_kept_and_every_rounding_is_said(self):
        p = lw.plan_windows(250, window=120, overlap=10, discard=5)
        self.assertEqual(p["total_frames"], 249)
        self.assertEqual(p["window"], 121)
        self.assertEqual(p["overlap"], 9)
        self.assertEqual(p["discard"], 0)
        self.assertEqual(p["stride"] % 8, 0)
        self.assertEqual(len(p["notes"]), 4)
        self.assertEqual(lw.extend_latents(112), 14)

    def test_nonsense_is_refused(self):
        with self.assertRaises(ValueError):
            lw.plan_windows(481, overlap=120)
        with self.assertRaises(ValueError):
            lw.plan_windows(24 * 60 * 10)          # ten minutes

    def test_a_minute(self):
        p = lw.plan_windows(24 * 60 + 1)
        self.assertEqual(p["count"], 13)
        self.assertIn("13 window(s)", lw.describe(p))


class ThePromptContract(unittest.TestCase):
    def test_window_one_is_the_prompt_itself(self):
        self.assertEqual(lw.window_prompts("A man walks in.", count=1),
                         ["A man walks in."])

    def test_a_later_window_leads_with_its_own_line_then_the_invariants(self):
        out = lw.window_prompts("A man walks in.", ["", "He sits down."],
                                invariants="one man, a grey room, soft window light",
                                count=3)
        self.assertEqual(out[0], "A man walks in.")
        self.assertTrue(out[1].startswith("He sits down. "))
        self.assertIn(lw.CONTINUE, out[1])
        self.assertTrue(out[1].endswith("Throughout: one man, a grey room, soft window light."))
        # the third window has no line of its own: it HOLDS, it does not
        # replay the shot
        self.assertTrue(out[2].startswith(lw.HOLD))
        self.assertNotIn("walks in", out[2])

    def test_a_settle_is_what_a_blank_window_holds(self):
        out = lw.window_prompts("She turns.", count=2, settle="she faces the window")
        self.assertTrue(out[1].startswith("she faces the window, held."))

    def test_the_first_line_may_override_the_base(self):
        self.assertEqual(lw.window_prompts("base", ["own"], count=1), ["own"])


if __name__ == "__main__":
    unittest.main()


# =============================================================================
# THE PANEL: the allowlist, the refusals, and the chain
# =============================================================================
import sys as _sys                                                    # noqa: E402
from pathlib import Path as _Path                                     # noqa: E402
from unittest import mock as _mock                                    # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import mlx_ltx_panel as panel                                         # noqa: E402


class ThePanel(unittest.TestCase):
    def test_make_job_carries_the_windows_fields(self):
        job = panel.make_job({"mode": "t2v", "prompt": "a man walks",
                              "frames": "481", "temporal_mode": "windows",
                              "window_prompts": '["", "he sits", ""]',
                              "window_invariants": "one man, grey room"})
        p = job["params"]
        self.assertEqual(p["long_mode"], "windows")
        self.assertEqual(p["temporal_mode"], "native")
        self.assertEqual(p["window_prompts"], ["", "he sits", ""])
        self.assertEqual(p["window_invariants"], "one man, grey room")
        # a plain render carries the neutral values, not an absence
        q = panel.make_job({"mode": "t2v", "prompt": "x"})["params"]
        self.assertEqual(q["long_mode"], "native")
        self.assertEqual(q["window_prompts"], [])
        # newline-separated is accepted too
        r = panel.make_job({"mode": "t2v", "prompt": "x", "temporal_mode": "windows",
                            "window_prompts": "a\nb"})["params"]
        self.assertEqual(r["window_prompts"], ["a", "b"])

    def test_the_chain_extends_on_the_kept_tail_and_trims_to_length(self):
        plan = lw.plan_windows(241)
        calls, ffm = [], []

        def fake_run(spec):
            calls.append(spec)
            _Path(spec["params"]["output_path"]).write_bytes(b"x")
            return {"elapsed_sec": 1}

        def fake_ffmpeg(cmd, label):
            ffm.append((label, cmd))
            _Path(cmd[-1]).write_bytes(b"y")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            first = _Path(d) / "clip_w0.mp4"
            first.write_bytes(b"w0")
            raw = _Path(d) / "clip.mp4"
            with _mock.patch.object(panel.HELPER, "run", side_effect=fake_run), \
                    _mock.patch.object(panel, "run_ffmpeg_tracked", side_effect=fake_ffmpeg), \
                    _mock.patch.object(panel, "set_hidden") as hidden, \
                    _mock.patch.object(panel, "pack_path", return_value=_Path("/q8")), \
                    _mock.patch.object(panel, "hq_weights",
                                       return_value={"dev_transformer": "dev.safetensors"}):
                out = panel._run_windows_chain(
                    {"id": "j1"}, {"prompt": "A man walks in.", "seed": 7,
                                   "window_prompts": ["", "He sits."],
                                   "window_invariants": "grey room"},
                    plan, first, raw, 241)
            # 241 frames = 3 windows: two extends on top of the first pass
            self.assertEqual(len(calls), 2)
            spec = calls[0]
            self.assertEqual(spec["action"], "extend")
            # the extend sees the TAIL of the previous output, never the whole
            # clip — that is what keeps every window the same price
            self.assertTrue(spec["params"]["video_path"].endswith("_w0t.mp4"))
            tail0 = [c for l, c in ffm if l == "Windows: tail 0"][0]
            self.assertIn("select='between(n\\,0\\,120)',setpts=N/FRAME_RATE/TB", tail0)
            self.assertEqual(spec["params"]["extend_frames"], 14)     # 112 / 8
            self.assertEqual(spec["params"]["seed"], 8)
            self.assertTrue(spec["params"]["prompt"].startswith("He sits. "))
            self.assertIn("Throughout: grey room.", spec["params"]["prompt"])
            # the third window has no line: it holds, on the second's output
            self.assertTrue(calls[1]["params"]["prompt"].startswith(lw.HOLD))
            self.assertTrue(calls[1]["params"]["video_path"].endswith("_w1t.mp4"))
            # window 2's output is 121 + 112 frames; its tail is the last 121
            tail1 = [c for l, c in ffm if l == "Windows: tail 1"][0]
            self.assertIn("select='between(n\\,112\\,232)',setpts=N/FRAME_RATE/TB", tail1)
            # and only the 112 new frames of each window go into the join
            piece1 = [c for l, c in ffm if l == "Windows: piece 1"][0]
            self.assertIn("select='gte(n\\,121)',setpts=N/FRAME_RATE/TB", piece1)
            self.assertEqual(spec["params"]["model_dir"], "/q8")
            # the final trim lands on raw_out at the asked length
            self.assertEqual(ffm[-1][0], "Windows: final join")
            self.assertEqual(ffm[-1][1][-1], str(raw))
            self.assertIn(f"{241 / 24:.6f}", ffm[-1][1])
            self.assertEqual(out["output_frames"], 241)
            self.assertEqual(len(out["files"]), 3)
            self.assertEqual(len(out["pieces"]), 3)          # w0 + two new-frame pieces
            hidden_paths = {c.args[0] for c in hidden.call_args_list}
            for f in out["files"]:
                self.assertIn(f, hidden_paths)                # every window output hidden

    def test_the_refusals_name_the_reason(self):
        job = panel.make_job({"mode": "t2v", "prompt": "x", "frames": "481",
                              "temporal_mode": "windows"})
        with _mock.patch.dict(panel.SYSTEM_CAPS, {"allows_extend": False, "label": "Compact"}), \
                _mock.patch.object(panel.HELPER, "run") as run, \
                _mock.patch.object(panel, "_apply_generation_profile_to_job"):
            with self.assertRaises(panel.RenderRefused) as cm:
                panel.run_job_inner(job)
        self.assertIn("Sliding windows", str(cm.exception))
        run.assert_not_called()
