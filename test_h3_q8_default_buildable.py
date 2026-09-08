"""The default's weights must be installable wherever the default applies.

v4.8.0 made the compact Q8 engine H3's AUTO default on every machine — but
`h3_build_q8.sh` still skipped the build on 64 GB+ Macs from the era when
bf16 was their default. Result, field-reported the day after release: a
fully-updated install rendering H3 at 47.76 GB resident, because AUTO wanted
a pack that Install had refused to build and fell back to bf16 without a
word. This suite pins the three parts of the fix:

  1. the build script has NO memory gate — every machine builds the pack;
  2. the dispatch says OUT LOUD when it falls back to bf16 un-asked;
  3. /status names the situation (`dit_choice.reason = auto_no_q8_pack`)
     instead of making the reader derive it from three booleans.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
BUILD_SH = ROOT / "scripts" / "pinokio" / "h3_build_q8.sh"
PANEL = ROOT / "mlx_ltx_panel.py"


class BuildScriptBuildsEverywhere(unittest.TestCase):
    def setUp(self):
        self.src = BUILD_SH.read_text()

    def test_no_memory_gate(self):
        self.assertNotIn("hw.memsize", self.src,
                         "the Q8 build is gated on RAM again — but Q8 is the "
                         "AUTO default on every machine, so every machine "
                         "must be able to build the pack")
        self.assertNotIn("bf16 engine is the default", self.src)

    def test_still_idempotent(self):
        # The re-run gate is the pack's own quant_config.json; losing it
        # would re-quantize 22 GB on every Install click.
        self.assertIn("quant_config.json", self.src)
        self.assertIn("already built", self.src)


class TheFallbackIsLoud(unittest.TestCase):
    def setUp(self):
        self.src = PANEL.read_text()

    def test_dispatch_announces_unasked_bf16(self):
        # The push() at the H3 dispatch. Silent bf16 is the bug that made
        # "I updated but the memory is the same" undiagnosable from the log.
        self.assertIn("the compact Q8 engine is not built on this install",
                      self.src)

    def test_status_names_the_reason(self):
        self.assertIn("auto_no_q8_pack", self.src)

    def test_hint_no_longer_promises_bf16_to_big_macs(self):
        self.assertNotIn("Automatic gives a 64 GB Mac the full model", self.src)


if __name__ == "__main__":
    unittest.main()


class TheMarkerMeansValidated(unittest.TestCase):
    """#78: on a 36 GB Mac the quantizer is SIGKILLed during its own in-process
    validation after writing a valid pack, and the old marker (quant_config.json,
    written before that validation) made the retry say "already built"."""
    def setUp(self):
        self.src = BUILD_SH.read_text()

    def test_validation_runs_in_its_own_process(self):
        self.assertIn("validate_pack_fresh", self.src)
        self.assertIn("load_dit(sys.argv[1]", self.src)

    def test_marker_written_only_after_validation(self):
        self.assertIn('.built_ok', self.src)
        # the marker is created inside the branch that follows a successful validation
        self.assertRegex(self.src, r'validate_pack_fresh; then\s*\n\s*: > "\$PACK/\.built_ok"')

    def test_idempotence_checks_the_shards_not_just_the_config(self):
        self.assertIn("pack_shards_present", self.src)
        self.assertIn("model.safetensors.index.json", self.src)
        self.assertRegex(self.src, r'\[ -f "\$PACK/\.built_ok" \] && pack_shards_present')

    def test_a_pack_that_fails_validation_is_rebuilt_next_time(self):
        self.assertIn('rm -f "$PACK/quant_config.json" "$PACK/.built_ok"', self.src)
