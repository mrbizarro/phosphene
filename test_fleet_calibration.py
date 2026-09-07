"""Estimates follow the chip; the queue path never routes to an engine that is not there.

Fleet, 30 days to 2026-09-07: the same tier renders 2-4x slower on an M4 Pro
than on the M4 Max every estimate was measured on, and faster on the M5 and
Ultra parts. The chips promised the M4 Max number to everyone.
"""
import os, unittest
from unittest import mock

import mlx_ltx_panel as p


class SpeedFactor(unittest.TestCase):
    def test_unknown_chip_prices_as_m4_max(self):
        with mock.patch.object(p, "_hw_chip_family", lambda: "unknown"), \
             mock.patch.dict(os.environ, {"PHOSPHENE_SPEED_FACTOR": ""}):
            self.assertEqual(p._hw_speed_factor("ltx"), 1.0)
            self.assertEqual(p._hw_speed_factor("h3"), 1.0)

    def test_m4_pro_is_slower_and_m5_max_faster(self):
        with mock.patch.dict(os.environ, {"PHOSPHENE_SPEED_FACTOR": ""}):
            with mock.patch.object(p, "_hw_chip_family", lambda: "M4 Pro"):
                self.assertGreater(p._hw_speed_factor("ltx"), 1.4)
                self.assertGreater(p._hw_speed_factor("h3"), 1.8)
            with mock.patch.object(p, "_hw_chip_family", lambda: "M5 Max"):
                self.assertLess(p._hw_speed_factor("ltx"), 0.5)

    def test_estimates_scale_with_the_factor(self):
        import inspect
        n_ltx = len(inspect.signature(p.ltx_estimate_minutes).parameters)
        with mock.patch.dict(os.environ, {"PHOSPHENE_SPEED_FACTOR": "1"}):
            h3_one = p.h3_estimate_minutes(768, 448, 124, 1, 3)
            ltx_one = p.ltx_estimate_minutes(1024, 576, 121, *([8] * (n_ltx - 3)))
        with mock.patch.dict(os.environ, {"PHOSPHENE_SPEED_FACTOR": "2"}):
            self.assertAlmostEqual(p.h3_estimate_minutes(768, 448, 124, 1, 3), h3_one * 2, places=4)
            self.assertAlmostEqual(p.ltx_estimate_minutes(1024, 576, 121, *([8] * (n_ltx - 3))), ltx_one * 2, places=4)

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"PHOSPHENE_SPEED_FACTOR": "3.5"}):
            self.assertEqual(p._hw_speed_factor("h3"), 3.5)


class QueuePathRouting(unittest.TestCase):
    def test_hidream_pick_falls_back_to_auto_when_not_installed(self):
        with mock.patch.object(p, "_hidream_available", lambda: False):
            job = p.make_job({"mode": "image", "prompt": "a lamp", "engine_override": "hidream_inline", "n": "1"})
        self.assertEqual(job["params"].get("engine_override"), "auto")

    def test_train_floor_is_declared(self):
        self.assertGreaterEqual(p.TRAIN_MIN_RAM_GB, 24)


if __name__ == "__main__":
    unittest.main()
