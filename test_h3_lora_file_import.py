#!/usr/bin/env python3
"""Contract tests for the local H3 LoRA file-import path."""
from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE = Path(tempfile.mkdtemp(prefix="phos-h3-file-import-"))
os.environ["LTX_STATE_DIR"] = str(STATE)
os.environ["PHOSPHENE_ANALYTICS_DISABLED"] = "1"
os.environ["PHOSPHENE_DISABLE_VERSION_CHECK"] = "1"

import mlx_ltx_panel as P


def _safetensors_lora(*, alpha: float | None = None,
                      module: str = "blocks.24.attn.qkv_proj",
                      metadata: dict | None = None) -> bytes:
    """A tiny but complete F32 A/B safetensors payload for import validation."""
    names = (
        module + ".lora_A.weight",
        module + ".lora_B.weight",
    )
    values = [(name, struct.pack("<f", 0.0)) for name in names]
    if alpha is not None:
        values.append((module + ".alpha", struct.pack("<f", alpha)))
    header = {"__metadata__": metadata} if metadata is not None else {}
    offset = 0
    for key, raw in values:
        header[key] = {"dtype": "F32",
                       "shape": [1] if key.endswith(".alpha") else [1, 1],
                       "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(raw for _, raw in values)


def _two_rank_lora(*, metadata: dict | None = None) -> bytes:
    """Two modules whose `lora_A` first dims differ — so `alpha / rank` has no
    single answer. Kijai's repacks carry 72 and 88 distinct ranks each."""
    values, header = [], ({"__metadata__": metadata} if metadata is not None else {})
    for module, rank in (("blocks.0.attn.qkv_proj", 2), ("blocks.1.attn.qkv_proj", 8)):
        values.append((module + ".lora_A.weight", b"\0" * (4 * rank), [rank, 1]))
        values.append((module + ".lora_B.weight", b"\0" * (4 * rank), [1, rank]))
    offset = 0
    for key, raw, shape in values:
        header[key] = {"dtype": "F32", "shape": shape,
                       "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(r for _, r, _ in values)


def _lora_with_orphans(*, pairs: int, orphans: int) -> bytes:
    """`pairs` complete A/B modules plus `orphans` lone `lora_A` tensors."""
    values, header = [], {}
    for i in range(pairs):
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            values.append((f"blocks.{i}.attn.qkv_proj" + suffix, struct.pack("<f", 0.0)))
    for i in range(orphans):
        values.append((f"blocks.{900 + i}.attn.qkv_proj.lora_A.weight",
                       struct.pack("<f", 0.0)))
    offset = 0
    for key, raw in values:
        header[key] = {"dtype": "F32", "shape": [1, 1],
                       "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(r for _, r in values)


def _prefixed_lora(prefix: str = "diffusion_model.",
                   module: str = "blocks.24.attn.qkv_proj") -> bytes:
    """A ComfyUI repack: the bare layout with every key under a namespace.

    THE fixture this suite was missing. Every other payload here is already
    bare, which is why an import gate that ran BEFORE `_h3_lora_strip_prefix`
    could compare `diffusion_model.blocks.…` against bare DiT module stems, go
    green in CI, and still refuse four of the five real ComfyUI repacks on a
    machine that had them. The repo's own note above `H3_LORAS_DIRNAME` says
    most H3 LoRAs on CivitAI arrive in exactly this shape.
    """
    return _safetensors_lora(module=prefix + module)


def _kohya_header() -> bytes:
    header = {
        "lora_unet_blocks_24_attn_qkv_proj.lora_down.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
        "lora_unet_blocks_24_attn_qkv_proj.lora_up.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
        "lora_unet_blocks_24_attn_qkv_proj.alpha": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
    }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded


def _diffusers_header() -> bytes:
    """The one layout the lane still refuses: split to_q/to_k/to_v with no alpha in the file."""
    header = {
        "transformer.blocks.0.attn.to_q.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
        "transformer.blocks.0.attn.to_q.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]},
    }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + b"\0" * 8


class TestH3LoraFileImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="phos-h3-lora-import-")
        self.dir = Path(self.tmp.name)
        self.old_dir = P._safe_h3_loras_dir
        P._safe_h3_loras_dir = lambda: self.dir

    def tearDown(self):
        P._safe_h3_loras_dir = self.old_dir
        self.tmp.cleanup()

    def test_imports_runner_layout_into_h3_library(self):
        payload = _safetensors_lora()
        result = P.import_h3_lora_file("MysticXXX_MMH3-V4.safetensors", payload)

        target = self.dir / "MysticXXX_MMH3-V4.safetensors"
        self.assertEqual(Path(result["path"]), target)
        self.assertTrue(target.is_file())
        self.assertEqual(result["layout"], "bare")
        self.assertFalse(result["converted"])

    def test_a_kohya_file_is_converted_on_import(self):
        """kohya is CONVERTED now, not refused (test_h3_lora_kohya.py pins the
        arithmetic). The import lands a bare-layout file plus its sidecar."""
        result = P.import_h3_lora_file("raw-kohya.safetensors", _kohya_header())
        self.assertEqual(result["layout"], "kohya")
        self.assertTrue(result["converted"])
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(names, ["raw-kohya.json", "raw-kohya.safetensors"])
        self.assertEqual(P._h3_lora_layout(self.dir / "raw-kohya.safetensors")["layout"], "bare")

    def test_refuses_non_safetensors_uploads(self):
        with self.assertRaisesRegex(ValueError, "\.safetensors"):
            P.import_h3_lora_file("adapter.zip", b"not an adapter")

    def test_non_unit_alpha_becomes_a_recommended_strength_not_a_refusal(self):
        """alpha != rank is a NUMBER, not a defect.

        An earlier revision refused here. It was wrong twice over. Wrong on
        policy: the picker has a per-LoRA strength control, and this project's
        own shipped position is that lightx2v's 8-step adapter (alpha 8 over
        rank 128) is correct and is loaded at 0.0625 — so "the H3 loader does
        not apply alpha" is the reason to WRITE the number down, not to reject
        the file. Wrong on arithmetic too: the number it refused on came from
        pairing one metadata alpha with whichever `lora_A` dict iteration
        reached first, and the two Kijai repacks carry 72 and 88 distinct
        per-module ranks.
        """
        result = P.import_h3_lora_file("needs-scale.safetensors",
                                       _safetensors_lora(alpha=8.0))

        self.assertTrue((self.dir / "needs-scale.safetensors").is_file())
        self.assertEqual(result["scale_source"], "per_module")
        # rank is lora_A's first dim (1 in this fixture), so alpha 8 -> 8.0.
        self.assertAlmostEqual(result["recommended_strength"], 8.0)
        sidecar = json.loads((self.dir / "needs-scale.json").read_text())
        self.assertAlmostEqual(sidecar["recommended_strength"], 8.0)

    def test_import_runs_after_the_comfyui_prefix_is_stripped(self):
        """REGRESSION GATE for the render-path break (PR #65 review, P0-1).

        The import checks compare module names against the installed DiT, whose
        stems are bare. Run them before `_h3_lora_strip_prefix` and every
        ComfyUI repack is refused for targeting `diffusion_model.blocks.0.…`.
        Stub the DiT with the BARE name only: the import must still succeed,
        which is only possible if the strip happened first.
        """
        old_targets = P._h3_lora_target_modules
        P._h3_lora_target_modules = lambda: {"blocks.24.attn.qkv_proj"}
        try:
            result = P.import_h3_lora_file("comfy-repack.safetensors",
                                           _prefixed_lora())
        finally:
            P._h3_lora_target_modules = old_targets

        self.assertEqual(result["layout"], "comfyui")
        self.assertTrue(result["converted"])
        self.assertTrue((self.dir / "comfy-repack.safetensors").is_file())

    def test_render_path_prepare_accepts_a_repack_with_no_dit_installed(self):
        """The other half of P0-1: `_h3_lora_prepare` is the RENDER path.

        It is called at CivitAI install and again on every render dispatch, and
        the CivitAI path DELETES the file when it raises. It must not acquire
        upload-shaped opinions. A repack goes through it and comes out bare.
        """
        staged = self.dir / "render-path.safetensors"
        staged.write_bytes(_prefixed_lora())

        def explode():
            raise AssertionError("_h3_lora_prepare must not consult the DiT")

        old_targets = P._h3_lora_target_modules
        P._h3_lora_target_modules = explode
        try:
            info = P._h3_lora_prepare(staged)
        finally:
            P._h3_lora_target_modules = old_targets

        self.assertEqual(info["layout"], "comfyui")
        self.assertTrue(info["converted"])

    def test_folded_scale_markers_are_believed_over_a_metadata_alpha(self):
        """The three spellings the real converted adapters actually carry.

        Kijai's 8-step repack states `alpha: "8"` AND `baked_scale: "0.0625"`;
        the scale is already inside lora_B, so 1.0 is correct and the bare
        `alpha` is provenance, not instruction.
        """
        for key, value in (("baked_scale", "0.0625"),
                           ("peft_scale_folded_into_B", "1"),
                           ("training_scale", "1.0")):
            with self.subTest(key=key):
                name = f"folded-{key}.safetensors"
                result = P.import_h3_lora_file(
                    name, _safetensors_lora(metadata={"alpha": "8", key: value}))
                self.assertEqual(result["scale_source"], "folded")
                self.assertAlmostEqual(result["recommended_strength"], 1.0)
                self.assertIn(key, result["scale_evidence"])

    def test_metadata_alpha_over_one_rank_is_divided_but_many_ranks_is_not(self):
        """A single metadata alpha is only divisible when the rank is unambiguous."""
        uniform = P.import_h3_lora_file("uniform.safetensors",
                                        _safetensors_lora(metadata={"alpha": "4"}))
        self.assertEqual(uniform["scale_source"], "uniform")
        self.assertAlmostEqual(uniform["recommended_strength"], 4.0)  # rank 1

        mixed = P.import_h3_lora_file("mixed.safetensors",
                                      _two_rank_lora(metadata={"alpha": "4"}))
        self.assertEqual(mixed["scale_source"], "advisory")
        self.assertAlmostEqual(mixed["recommended_strength"], 1.0)
        self.assertIn("ambiguous", mixed["scale_evidence"])

    def test_one_orphan_tensor_is_ignored_but_a_broken_file_is_refused(self):
        """0.90, the same ratio `lora_compat` has used on the LTX lane."""
        ok = P.import_h3_lora_file("one-orphan.safetensors",
                                   _lora_with_orphans(pairs=12, orphans=1))
        self.assertEqual(ok["pairs"], 12)

        with self.assertRaisesRegex(RuntimeError, "unmatched"):
            P.import_h3_lora_file("mostly-orphans.safetensors",
                                  _lora_with_orphans(pairs=1, orphans=4))

    def test_a_refusal_names_the_users_file_not_the_staging_temp(self):
        """Staging under `.name.<pid>.<tid>.uploading` leaked that string into
        an alert(). The staging DIRECTORY keeps the user's own filename on the
        staged file, so every message names what they picked."""
        with self.assertRaises(RuntimeError) as ctx:
            P.import_h3_lora_file("my-adapter.safetensors",
                                  _safetensors_lora()[:-8])
        message = str(ctx.exception)
        self.assertIn("my-adapter.safetensors", message)
        self.assertNotIn("uploading", message)
        self.assertNotIn(str(os.getpid()), message)

    def test_a_refusal_leaves_no_staging_directory_behind(self):
        with self.assertRaises(Exception):
            P.import_h3_lora_file("doomed.safetensors", _diffusers_header())
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_successful_import_writes_a_sidecar_like_the_civitai_path(self):
        P.import_h3_lora_file("My_Cool_LoRA.safetensors", _safetensors_lora())
        sidecar = json.loads((self.dir / "My_Cool_LoRA.json").read_text())
        self.assertEqual(sidecar["name"], "My Cool LoRA")
        self.assertEqual(sidecar["base_model"], "MiniMax H3")
        self.assertEqual(sidecar["kind"], "import")
        self.assertAlmostEqual(sidecar["recommended_strength"], 1.0)
        # The picker reads through _read_lora_sidecar; it must find the row.
        self.assertEqual(
            P._read_lora_sidecar(self.dir / "My_Cool_LoRA.safetensors")["name"],
            "My Cool LoRA")

    def test_imports_explicit_scale_one_conversion_metadata(self):
        result = P.import_h3_lora_file(
            "converted.safetensors",
            _safetensors_lora(metadata={"alpha": "alpha == rank; scale = 1.0"}))

        self.assertEqual(result["filename"], "converted.safetensors")

    def test_refuses_unreadable_alpha_without_leaving_a_file(self):
        payload = _safetensors_lora(alpha=1.0)[:-4]
        with self.assertRaisesRegex(RuntimeError, "alpha|truncated|offsets"):
            P.import_h3_lora_file("bad-alpha.safetensors", payload)

        self.assertEqual(list(self.dir.iterdir()), [])

    def test_refuses_modules_missing_from_the_installed_transformer(self):
        old_targets = P._h3_lora_target_modules
        P._h3_lora_target_modules = lambda: {"blocks.24.attn.qkv_proj"}
        try:
            with self.assertRaisesRegex(RuntimeError, "does not have"):
                P.import_h3_lora_file(
                    "wrong-model.safetensors",
                    _safetensors_lora(module="blocks.24.attn.typo"))
        finally:
            P._h3_lora_target_modules = old_targets

        self.assertEqual(list(self.dir.iterdir()), [])

    def test_refuses_a_header_only_adapter(self):
        header_only = _safetensors_lora()[:-8]
        with self.assertRaisesRegex(RuntimeError, "truncated|offsets"):
            P.import_h3_lora_file("truncated.safetensors", header_only)

        self.assertEqual(list(self.dir.iterdir()), [])

    def test_refuses_duplicate_name_without_replacing_first_import(self):
        first = _safetensors_lora()
        replacement = _safetensors_lora(alpha=1.0)
        P.import_h3_lora_file("keep-me.safetensors", first)

        with self.assertRaises(FileExistsError):
            P.import_h3_lora_file("keep-me.safetensors", replacement)

        self.assertEqual((self.dir / "keep-me.safetensors").read_bytes(), first)

    def test_picker_markup_has_a_real_import_control(self):
        panel = ((ROOT / "mlx_ltx_panel.py").read_text(encoding="utf-8")
                 + "\n" + (ROOT / "webapp" / "index.html").read_text(encoding="utf-8"))
        for _m in sorted((ROOT / "webapp" / "js").glob("*.js")):
            panel += "\n" + _m.read_text(encoding="utf-8")
        self.assertIn("Import H3 LoRA", panel)
        self.assertIn("/h3/loras/import", panel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
