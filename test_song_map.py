#!/usr/bin/env python3
"""The song map: sections and energy on the bar grid, and the pacing rule
the Director derives from them. A synthesised track — a quiet intro, a loud
bright chorus, a quiet outro at 120 bpm — so the labels can be asserted
rather than eyeballed."""
from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import storyboard_edit as se

SR = 22050
BPM = 120.0
BAR = 60.0 / BPM * 4          # 2.0 s


def _track(path: Path, bars=(("q", 8), ("loud", 8), ("q", 8))) -> float:
    """Kick on every beat so beat_map finds the grid; a bright loud pad in the
    loud bars, a faint dull one elsewhere."""
    total = sum(n for _, n in bars) * BAR
    t = np.arange(int(total * SR)) / SR
    y = np.zeros_like(t, dtype=np.float32)
    # the kick: a 60 Hz thump decaying over 120 ms, every beat
    beat = 60.0 / BPM
    for k in range(int(total / beat)):
        i0 = int(k * beat * SR)
        n = int(0.12 * SR)
        tt = np.arange(n) / SR
        y[i0:i0 + n] += (0.35 * np.sin(2 * math.pi * 60 * tt) * np.exp(-tt * 30)).astype(np.float32)
    cursor = 0.0
    for kind, n in bars:
        a, b = int(cursor * SR), int((cursor + n * BAR) * SR)
        tt = t[a:b]
        if kind == "loud":
            y[a:b] += (0.35 * np.sin(2 * math.pi * 440 * tt)
                       + 0.25 * np.sin(2 * math.pi * 3520 * tt)).astype(np.float32)
        else:
            y[a:b] += (0.04 * np.sin(2 * math.pi * 110 * tt)).astype(np.float32)
        cursor += n * BAR
    y = np.clip(y, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return total


class TheSongMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "t.wav"
        cls.total = _track(cls.path)
        cls.map = se.song_map(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bars_sit_on_downbeats_and_carry_energy(self):
        m = self.map
        self.assertGreaterEqual(len(m["bars"]), 20)
        self.assertLessEqual(len(m["bars"]), 26)
        for a, b in zip(m["bars"], m["bars"][1:]):
            self.assertAlmostEqual(b["start"], a["end"], places=6)
            self.assertAlmostEqual(b["start"] - a["start"], BAR, delta=0.15)
        e = [x["energy"] for x in m["bars"]]
        self.assertGreater(max(e), 0.9)
        self.assertLess(min(e), 0.3)

    def test_three_sections_labelled_by_position_and_energy(self):
        secs = self.map["sections"]
        self.assertEqual([s["label"] for s in secs], ["intro", "chorus", "outro"])
        # the boundaries are bar edges, close to 16 s and 32 s
        self.assertAlmostEqual(secs[1]["start"], 8 * BAR, delta=BAR)
        self.assertAlmostEqual(secs[2]["start"], 16 * BAR, delta=BAR)
        self.assertGreater(secs[1]["energy"], secs[0]["energy"] * 3)
        self.assertEqual(secs[0]["bars"][0], 0)
        self.assertEqual(secs[-1]["bars"][1], len(self.map["bars"]))

    def test_a_beat_map_is_reused_when_given(self):
        beats = se.beat_map(str(self.path))
        m2 = se.song_map(str(self.path), beats)
        self.assertEqual([s["label"] for s in m2["sections"]],
                         [s["label"] for s in self.map["sections"]])

    def test_a_jingle_is_one_section_never_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "j.wav"
            _track(p, bars=(("loud", 5),))
            m = se.song_map(str(p))
        self.assertEqual(len(m["sections"]), 1)
        self.assertIn(m["sections"][0]["label"], se.SECTION_LABELS)


class ThePacingRule(unittest.TestCase):
    def test_chorus_cuts_twice_as_often_and_the_ends_half_as_often(self):
        self.assertEqual(se.director_pacing("chorus", 0.9, 0.5, 2), 1)
        self.assertEqual(se.director_pacing("chorus", 0.9, 0.5, 1), 1)
        self.assertEqual(se.director_pacing("intro", 0.1, 0.5, 2), 4)
        self.assertEqual(se.director_pacing("outro", 0.1, 0.5, 2), 4)
        self.assertEqual(se.director_pacing("bridge", 0.3, 0.5, 2), 2)
        self.assertEqual(se.director_pacing("verse", 0.5, 0.5, 2), 2)
        # an unlabelled hot section cuts like a chorus
        self.assertEqual(se.director_pacing("", 0.9, 0.5, 4), 2)


if __name__ == "__main__":
    unittest.main()
