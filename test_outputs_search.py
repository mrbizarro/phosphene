#!/usr/bin/env python3
"""Gallery search: every output row carries `q`, the searchable words of its
sidecar, built in the loop that already reads it."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as panel                                        # noqa: E402


class TheWords(unittest.TestCase):
    def test_prompt_mode_size_loras_and_character(self):
        q = panel._output_search_text(
            {"engine": "ltx", "model": "/m/ltx-2.5-mlx-q8", "temporal": {"mode": "fps12_interp24"}},
            {"prompt": "A woman turns to the window", "mode": "i2v", "quality": "balanced",
             "width": 1280, "height": 704, "frames": 121, "seed_used": 4242,
             "character_id": "ariatrn",
             "loras": [{"path": "/l/Crisp_Enhance.safetensors", "strength": 0.8}, "/l/hdr.safetensors"]})
        for w in ("a woman turns", "i2v", "balanced", "1280x704", "121f", "4242",
                  "ariatrn", "crisp_enhance", "hdr", "ltx-2.5-mlx-q8", "fps12_interp24"):
            self.assertIn(w, q, w)
        self.assertEqual(panel._output_search_text({}, {}), "")
        self.assertEqual(panel._output_search_text({"bad": object()}, {"prompt": None}), "")

    def test_list_outputs_carries_q(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            clip = out / "clip.mp4"
            clip.write_bytes(b"x")
            (out / "clip.mp4.json").write_text(json.dumps(
                {"params": {"prompt": "the grinning man", "mode": "t2v", "width": 768,
                            "height": 416}, "engine": "ltx"}))
            plain = out / "plain.mp4"
            plain.write_bytes(b"y")
            # The gallery skips files ffmpeg may still be writing (a 2 s
            # mtime cutoff), so these are aged before they are listed.
            import os
            old = time.time() - 30
            for f in (clip, plain):
                os.utime(f, (old, old))
            lib = out / "lib"
            lib.mkdir()
            with mock.patch.object(panel, "OUTPUT", out), \
                    mock.patch.object(panel, "UPLOADS", lib):
                rows = panel.list_outputs(limit=0)
        by = {r["name"]: r for r in rows}
        self.assertIn("the grinning man", by["clip.mp4"]["q"])
        self.assertIn("768x416", by["clip.mp4"]["q"])
        self.assertEqual(by["plain.mp4"]["q"], "")


if __name__ == "__main__":
    unittest.main()
