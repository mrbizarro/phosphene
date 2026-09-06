"""kohya-format H3 adapters are converted, not refused.

The character LoRAs people train for H3 come out of kohya-style trainers:
`lora_unet_blocks_N_attn_qkv_proj.lora_down.weight` / `.lora_up.weight` and a
`.alpha` scalar per module, applied as (alpha/rank)·B@A. The runner reads
lora_A / lora_B and applies no alpha, so the panel renames the modules onto
the runner's tree and folds alpha/rank into lora_B — strength 1.0 then means
"as trained", the same contract as every other file in the lane.
"""
import json, struct, tempfile, unittest
from pathlib import Path

import mlx.core as mx

import mlx_ltx_panel as P


def _kohya_file(path: Path, alpha: float, rank: int = 4, extra_unknown: bool = False, meta=None):
    t = {}
    for name in ("lora_unet_blocks_3_attn_qkv_proj", "lora_unet_blocks_3_attn_out_proj",
                 "lora_unet_token_refiner_blocks_0_mlp_fc1"):
        t[f"{name}.lora_down.weight"] = mx.random.normal((rank, 8)).astype(mx.bfloat16)
        t[f"{name}.lora_up.weight"] = mx.random.normal((12, rank)).astype(mx.bfloat16)
        t[f"{name}.alpha"] = mx.array(alpha, dtype=mx.bfloat16)
    if extra_unknown:
        t["lora_unet_blocks_3_adaln_proj.lora_down.weight"] = mx.zeros((rank, 8))
        t["lora_unet_blocks_3_adaln_proj.lora_up.weight"] = mx.zeros((12, rank))
    mx.save_safetensors(str(path), t, metadata=meta or {"modelspec.trigger_phrase": "Some Person", "ss_network_alpha": str(alpha)})
    return t


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


class KohyaConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="phos-kohya-")
        self.path = Path(self.tmp.name) / "Person.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_names_kohya_as_convertible(self):
        _kohya_file(self.path, alpha=4.0)
        info = P._h3_lora_layout(self.path)
        self.assertEqual(info["layout"], "kohya")
        self.assertTrue(info["convertible"])
        self.assertEqual(info["pairs"], 3)

    def test_keys_map_onto_the_runner_modules(self):
        _kohya_file(self.path, alpha=4.0)
        P._h3_lora_prepare(self.path)
        keys = sorted(k for k in _header(self.path) if k != "__metadata__")
        self.assertIn("blocks.3.attn.qkv_proj.lora_A.weight", keys)
        self.assertIn("blocks.3.attn.out_proj.lora_B.weight", keys)
        self.assertIn("token_refiner.blocks.0.mlp.fc1.lora_A.weight", keys)
        self.assertFalse(any("lora_down" in k or ".alpha" in k for k in keys))
        self.assertEqual(P._h3_lora_layout(self.path)["layout"], "bare")

    def test_alpha_over_rank_is_folded_into_b(self):
        src = _kohya_file(self.path, alpha=2.0, rank=4)      # quotient 0.5
        P._h3_lora_prepare(self.path)
        out = mx.load(str(self.path))
        want = (src["lora_unet_blocks_3_attn_qkv_proj.lora_up.weight"].astype(mx.float32) * 0.5)
        got = out["blocks.3.attn.qkv_proj.lora_B.weight"].astype(mx.float32)
        self.assertTrue(mx.allclose(got, want, atol=2e-2, rtol=2e-2).item())
        a_same = mx.array_equal(out["blocks.3.attn.qkv_proj.lora_A.weight"],
                                src["lora_unet_blocks_3_attn_qkv_proj.lora_down.weight"])
        self.assertTrue(a_same.item())
        self.assertEqual(_header(self.path)["__metadata__"]["alpha_over_rank"], "0.5")

    def test_alpha_equal_rank_leaves_b_untouched(self):
        src = _kohya_file(self.path, alpha=4.0, rank=4)
        P._h3_lora_prepare(self.path)
        out = mx.load(str(self.path))
        self.assertTrue(mx.array_equal(out["blocks.3.attn.qkv_proj.lora_B.weight"],
                                       src["lora_unet_blocks_3_attn_qkv_proj.lora_up.weight"]).item())

    def test_metadata_and_trigger_phrase_survive_and_reach_the_sidecar(self):
        _kohya_file(self.path, alpha=4.0)
        sc = self.path.with_suffix(".json")
        sc.write_text(json.dumps({"name": "Person", "trigger_words": []}))
        done = P._h3_lora_prepare(self.path)
        self.assertEqual(done["trigger_words"], ["Some Person"])
        meta = _header(self.path)["__metadata__"]
        self.assertEqual(meta["converted_from"], "kohya")
        self.assertEqual(meta["modelspec.trigger_phrase"], "Some Person")
        self.assertEqual(json.loads(sc.read_text())["trigger_words"], ["Some Person"])

    def test_unknown_modules_are_dropped_and_counted(self):
        _kohya_file(self.path, alpha=4.0, extra_unknown=True)
        done = P._h3_lora_prepare(self.path)
        self.assertEqual(done["pairs"], 3)
        self.assertEqual(_header(self.path)["__metadata__"]["dropped_modules"], "1")

    def test_conversion_is_idempotent(self):
        _kohya_file(self.path, alpha=2.0)
        P._h3_lora_prepare(self.path)
        first = self.path.read_bytes()
        again = P._h3_lora_prepare(self.path)
        self.assertEqual(again["layout"], "bare")
        self.assertEqual(self.path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
