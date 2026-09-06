#!/usr/bin/env python3
"""Contract gate for the ordered H3 Turbo adapter resolver + pinned installer."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = Path(tempfile.mkdtemp(prefix="phos-h3-turbo-contract-"))
os.environ["LTX_STATE_DIR"] = str(STATE)
os.environ["PHOSPHENE_ANALYTICS_DISABLED"] = "1"
os.environ["PHOSPHENE_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("LTX_PORT", "8297")
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as P  # noqa: E402


def _reset_dl_state():
    P._set_h3_turbo_dl(status="idle", mb=0,
                       total_mb=P.H3_TURBO_ASSET_BYTES // (1 << 20),
                       error=None)


class TestH3TurboResolver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="phos-turbo-files-")
        self.directory = Path(self.tmp.name)
        self.old_dir = P._h3_turbo_dir
        self.old_floor = P.H3_TURBO_LORA_MIN_BYTES
        P._h3_turbo_dir = lambda: self.directory
        P.H3_TURBO_LORA_MIN_BYTES = 1

    def tearDown(self):
        P._h3_turbo_dir = self.old_dir
        P.H3_TURBO_LORA_MIN_BYTES = self.old_floor
        self.tmp.cleanup()

    def put(self, filename: str) -> Path:
        path = self.directory / filename
        path.write_bytes(b"runner-layout fixture")
        return path

    def test_v4_600_ema_is_preferred_when_present_and_runs_six_forwards(self):
        v4 = self.put(P.H3_TURBO_V4_FILE)
        self.put(P.H3_TURBO_LORA_FILE)
        self.put(P.H3_TURBO_CKPT500_FILE)
        resolved = P.h3_turbo_paths()
        self.assertEqual(resolved["lora"], v4)
        self.assertEqual(resolved["version"], "v4-600-EMA")
        self.assertFalse(resolved["fallback"])
        self.assertEqual(P.h3_turbo_argv(resolved), ["--lora", f"{v4}:1.0"])
        self.assertEqual(P.h3_turbo_steps(resolved), 7)          # 6 forwards
        self.assertEqual(P.h3_turbo_steps({"version": "v1.0"}), 4)
        self.assertEqual(P.h3_turbo_steps({"version": "ckpt500-EMA"}), 4)

    def test_the_managed_download_is_the_v4_adapter_from_its_author(self):
        a = P._h3_turbo_asset()
        self.assertEqual(a["file"], P.H3_TURBO_V4_FILE)
        self.assertTrue(a["url"].startswith("https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/"))
        self.assertEqual(a["sha256"], "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3")
        self.assertEqual(a["bytes"], 779849816)
        # the LightX2V release asset stays reachable by key
        self.assertEqual(P._h3_turbo_asset("v1.0")["url"], P.H3_TURBO_ASSET_URL)

    def test_v1_is_preferred_when_all_three_exist(self):
        primary = self.put(P.H3_TURBO_LORA_FILE)
        self.put(P.H3_TURBO_FALLBACK_LORA_FILE)
        self.put(P.H3_TURBO_CKPT500_FILE)
        resolved = P.h3_turbo_paths()
        self.assertEqual(resolved["lora"], primary)
        self.assertEqual(resolved["version"], "v1.0")
        self.assertFalse(resolved["fallback"])
        self.assertEqual(
            P.h3_turbo_argv(resolved), ["--lora", f"{primary}:1.0"],
        )

    def test_alpha_folded_v01_beats_ckpt500(self):
        fallback = self.put(P.H3_TURBO_FALLBACK_LORA_FILE)
        self.put(P.H3_TURBO_CKPT500_FILE)
        resolved = P.h3_turbo_paths()
        self.assertEqual(resolved["lora"], fallback)
        self.assertEqual(resolved["version"], "v0.1")
        self.assertTrue(resolved["fallback"])
        self.assertEqual(
            P.h3_turbo_argv(resolved), ["--lora", f"{fallback}:1.0"],
        )

    def test_ckpt500_is_the_last_resort_and_labels_itself(self):
        # The retired adapter alone keeps Turbo alive (the v4.0.4 regression:
        # installs that rendered fine on it were un-Turboed by its removal),
        # but it must resolve LAST and carry the honest fallback flag.
        retired = self.put(P.H3_TURBO_CKPT500_FILE)
        resolved = P.h3_turbo_paths()
        self.assertEqual(resolved["lora"], retired)
        self.assertEqual(resolved["version"], "ckpt500-EMA")
        self.assertTrue(resolved["fallback"])
        self.assertEqual(
            P.h3_turbo_argv(resolved), ["--lora", f"{retired}:1.0"],
        )

    def test_raw_v01_is_never_selected(self):
        self.put(P.H3_TURBO_RAW_V01_FILE)
        resolved = P.h3_turbo_paths()
        self.assertFalse(resolved["files_ok"])
        self.assertIsNone(resolved["lora"])
        with self.assertRaisesRegex(RuntimeError, "not available"):
            P.h3_turbo_argv(resolved)


class TestH3TurboInstallContract(unittest.TestCase):
    def setUp(self):
        self.old_paths = P.h3_paths
        self.old_supported = P.h3_supports_lora
        self.old_dir = P._h3_turbo_dir
        self.tmp = tempfile.TemporaryDirectory(prefix="phos-turbo-install-")
        self.target = Path(self.tmp.name)
        P.h3_paths = lambda: {"missing": []}
        P.h3_supports_lora = lambda: True
        P._h3_turbo_dir = lambda: self.target
        _reset_dl_state()

    def tearDown(self):
        P.h3_paths = self.old_paths
        P.h3_supports_lora = self.old_supported
        P._h3_turbo_dir = self.old_dir
        self.tmp.cleanup()
        _reset_dl_state()

    def test_install_starts_the_pinned_download(self):
        started = threading.Event()
        calls = []

        def fake_download(target_dir, push_log):
            calls.append(target_dir)
            started.set()

        logs = []
        result = P._h3_install_turbo(logs.append, download_fn=fake_download)
        self.assertTrue(result["ok"])
        self.assertTrue(result["started"])
        self.assertEqual(result["asset"], P._h3_turbo_asset()["url"])
        self.assertEqual(result["sha256"], P._h3_turbo_asset()["sha256"])
        self.assertEqual(result["bytes"], P._h3_turbo_asset()["bytes"])
        self.assertTrue(started.wait(timeout=5))
        self.assertEqual(calls, [self.target])

    def test_second_install_while_active_is_refused(self):
        gate = threading.Event()

        def blocked_download(target_dir, push_log):
            gate.wait(timeout=10)

        first = P._h3_install_turbo(lambda _m: None,
                                    download_fn=blocked_download)
        self.assertTrue(first["ok"])
        second = P._h3_install_turbo(lambda _m: None,
                                     download_fn=blocked_download)
        gate.set()
        self.assertFalse(second["ok"])
        self.assertIn("active", second["error"])

    def test_present_adapter_short_circuits(self):
        (self.target / P.H3_TURBO_LORA_FILE).write_bytes(b"fixture")
        result = P._h3_install_turbo(lambda _m: None,
                                     download_fn=lambda *a: None)
        self.assertTrue(result["ok"])
        self.assertFalse(result["started"])
        self.assertTrue(result["already_installed"])

    def test_missing_h3_and_old_runner_still_refuse(self):
        P.h3_paths = lambda: {"missing": ["dit (weights absent)"]}
        result = P._h3_install_turbo(lambda _m: None,
                                     download_fn=lambda *a: None)
        self.assertFalse(result["ok"])
        self.assertIn("isn't fully installed", result["error"])

        P.h3_paths = lambda: {"missing": []}
        P.h3_supports_lora = lambda: False
        result = P._h3_install_turbo(lambda _m: None,
                                     download_fn=lambda *a: None)
        self.assertFalse(result["ok"])
        self.assertIn("predates Turbo", result["error"])

    def test_status_offers_the_install(self):
        status = P.h3_turbo_status()
        self.assertTrue(status["install_available"])
        self.assertFalse(status["installing"])
        self.assertIn(P.H3_TURBO_LORA_FILE, status["install_note"])


class TestH3TurboPins(unittest.TestCase):
    def _load_fetch_script(self):
        spec = importlib.util.spec_from_file_location(
            "fetch_h3_turbo", ROOT / "scripts" / "fetch_h3_turbo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_fetch_script_pins_match_the_panel(self):
        script = self._load_fetch_script()
        self.assertEqual(script.ASSET_NAME, P.H3_TURBO_LORA_FILE)
        self.assertEqual(script.ASSET_URL, P.H3_TURBO_ASSET_URL)
        self.assertEqual(script.ASSET_SHA256, P.H3_TURBO_ASSET_SHA256)
        self.assertEqual(script.ASSET_BYTES, P.H3_TURBO_ASSET_BYTES)

    def test_installer_runs_the_digest_checked_fetch(self):
        installer = (ROOT / "install_h3.js").read_text(encoding="utf-8")
        self.assertIn("scripts/fetch_h3_turbo.py", installer)
        self.assertIn(P.H3_TURBO_LORA_FILE, installer)
        self.assertIn(P.H3_TURBO_ASSET_SHA256, installer)
        # Provenance of the repack stays recorded.
        self.assertIn(P.H3_TURBO_SOURCE_FILE, installer)
        self.assertIn(P.H3_TURBO_SOURCE_SHA256, installer)
        self.assertIn(P.H3_TURBO_FALLBACK_LORA_FILE, installer)
        # The publication TODO is history; a revert would resurrect it.
        self.assertNotIn("REQUIRED BEFORE WIRING", installer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
