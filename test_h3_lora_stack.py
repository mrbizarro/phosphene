"""H3 LoRA stacking is a RUNNER capability, reported, never assumed.

A pack that predates the repeatable `--lora` has one adapter slot and the
panel keeps its Turbo-or-LoRA arbitration for it; a pack that carries the
stack marker in its `--lora` help takes Turbo and up to four user adapters in
one render. Both answers come from the installed script's text, so an old
install never posts a second `--lora` into an argparse error.
"""
import unittest
from unittest import mock

import mlx_ltx_panel as p


def _flags(*present):
    def probe(flag):
        return any(flag == f or flag in f for f in present)
    return probe


class StackCapability(unittest.TestCase):
    def test_single_slot_runner_keeps_the_old_contract(self):
        with mock.patch.object(p, "_h3_runner_has_flag", _flags("--lora")):
            self.assertTrue(p.h3_supports_lora())
            self.assertFalse(p.h3_supports_lora_stack())
            self.assertEqual(p.h3_lora_max_stack(), 1)
            self.assertIn("ONE adapter slot", p.h3_lora_stack_note())

    def test_stacking_runner_reports_the_limit_and_the_advice(self):
        with mock.patch.object(p, "_h3_runner_has_flag", _flags("--lora", p.H3_LORA_STACK_MARKER)):
            self.assertTrue(p.h3_supports_lora_stack())
            self.assertEqual(p.h3_lora_max_stack(), p.H3_LORA_STACK_LIMIT)
            self.assertGreater(p.H3_LORA_STACK_LIMIT, 1)
            self.assertIn("stack", p.h3_lora_stack_note())
            self.assertIn("1.5", p.h3_lora_stack_note())

    def test_marker_is_the_runner_help_text(self):
        """The marker must be the sentence the stackable runner prints, not a flag."""
        self.assertTrue(p.H3_LORA_STACK_MARKER.startswith("Repeat the flag"))

    def test_status_block_carries_the_live_limit(self):
        with mock.patch.object(p, "_h3_runner_has_flag", _flags("--lora", p.H3_LORA_STACK_MARKER)), \
             mock.patch.object(p, "list_h3_user_loras", lambda: []):
            st = p.h3_loras_status()
            self.assertEqual(st["max_stack"], p.H3_LORA_STACK_LIMIT)
            self.assertEqual(st["note"], p.H3_LORA_STACK_OK_NOTE)


if __name__ == "__main__":
    unittest.main()
