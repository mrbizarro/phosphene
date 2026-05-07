# Phosphene Image-Gen Research — Apple Silicon, May 2026

_Tier-aware research for the agent's anchor-still pipeline (i2v inputs).
LTX 2.3's t2v is weak; image-gen quality is the dominant lever for video output._

## TL;DR

- **"Flux Klein"** = **FLUX.2 [klein] 4B** by Black Forest Labs (Jan 2026). Apache 2.0, mflux-compatible at 4-bit (~4.3 GB), 4-step inference, ~12-18 s/image on Comfortable Mac. **Ship as default.**
- **mflux 0.17.x is now multi-family.** Legacy `mflux-generate` is FLUX.1-only. New CLIs: `mflux-generate-flux2`, `mflux-generate-z-image-turbo`, `mflux-generate-fibo`, `mflux-generate-qwen`, `mflux-generate-kontext`. Panel currently calls legacy → silently misses every new family.
- **Z-Image-Turbo (Tongyi/Alibaba, Nov 2025)** is the Compact-tier default — 6B model, ~5.9 GB at 4-bit, 8-9 steps, fits 16 GB.

## The single default to ship

**FLUX.2 [klein] 4B at 4-bit, 4 steps** via:

```
mflux-generate-flux2 \
  --model Runpod/FLUX.2-klein-4B-mflux-4bit \
  --base-model flux2-klein-4b \
  --steps 4 \
  --quantize 4 \
  --guidance 1.0
```

Apache 2.0 → no gating, commercial OK. ~4.3 GB on disk, fits Compact tier with LTX dormant. 4 candidates per shot in 50-75 s on Comfortable. Multi-reference editing built-in.

## Tier-aware defaults

| Tier | RAM | Default model | Quant | Steps | Wall time |
|---|---|---|---|---|---|
| Compact | 16-32 GB | `Tongyi-MAI/Z-Image-Turbo` (mflux-4bit) | 4 | 9 | ~15-25 s |
| Comfortable | 32-79 GB | `Runpod/FLUX.2-klein-4B-mflux-4bit` | 4 | 4 | ~12-18 s |
| Roomy | 80-119 GB | klein-4B 8-bit OR Krea-dev 8-bit | 4-8 | 4-25 | ~18-80 s |
| Studio | 120 GB+ | klein-4B 4-bit OR FLUX.2-dev 4-bit | 4 | 4 | ~12 s |

Auto-detect snippet:
```python
def _default_image_engine_for_tier(tier: str) -> dict:
    if tier == "base":          # Compact
        return {"kind": "mflux", "family": "z_image_turbo",
                "model_path": "filipstrand/Z-Image-Turbo-mflux-4bit",
                "steps": 9, "quantize": 4, "guidance": 0.0}
    if tier in ("standard", "high", "pro"):
        return {"kind": "mflux", "family": "flux2",
                "model_path": "Runpod/FLUX.2-klein-4B-mflux-4bit",
                "steps": 4, "quantize": 4, "guidance": 1.0}
    return {"kind": "mock"}
```

## Model landscape (mflux-friendly repos)

| Model | Repo | Params | 4-bit GB | License | Steps |
|---|---|---|---|---|---|
| **FLUX.2-klein-4B** | `Runpod/FLUX.2-klein-4B-mflux-4bit` | 4B | **4.3** | **Apache 2.0** | **4** |
| FLUX.2-klein-9B | community ports forming | 9B | ~9-11 | Non-commercial | 4-8 |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | 32B | ~16-18 | Non-commercial | 25-50 |
| **Z-Image-Turbo** | **`filipstrand/Z-Image-Turbo-mflux-4bit`** | 6B | **5.9** | **Apache 2.0** | **8-9** |
| Z-Image (base) | `Tongyi-MAI/Z-Image` | 6B | ~6.5 | Apache 2.0 | 25 |
| FLUX.1-Krea-dev | `filipstrand/FLUX.1-Krea-dev-mflux-4bit` | 12B | 9.61 | Non-commercial (gated) | 25 |
| FLUX.1-dev | `madroid/flux.1-dev-mflux-4bit` | 12B | ~9.5 | Non-commercial (gated) | 25-50 |
| FLUX.1-schnell | `dhairyashil/FLUX.1-schnell-mflux-4bit` | 12B | ~9 | Apache 2.0 | 2-4 |
| FIBO | `briaai/FIBO` | 8B | ~10-12 | Open-RAIL-ish | 30 |
| Qwen-Image | `filipstrand/Qwen-Image-mflux-6bit` | 20B | ~16 (6-bit) | Apache 2.0 | 25-50 |
| Flex.1-alpha | `ostris/Flex.1-alpha` | 8B | ~9-10 | Apache 2.0 | 25 |

## mflux 0.17.x per-family CLIs

| Command | Family |
|---|---|
| `mflux-generate` (legacy) | flux1 |
| `mflux-generate-flux2` | flux2 (klein-4B, klein-9B, dev) |
| `mflux-generate-flux2-edit` | flux2 image-edit |
| `mflux-generate-z-image` | z_image base |
| `mflux-generate-z-image-turbo` | z_image_turbo |
| `mflux-generate-fibo` | fibo |
| `mflux-generate-fibo-edit` | fibo edit (incl. remove-bg) |
| `mflux-generate-qwen` | qwen |
| `mflux-generate-kontext` | flux1-kontext |
| `mflux-generate-fill` | flux1 inpaint |
| `mflux-generate-controlnet` | flux1 controlnet |
| `mflux-upscale-seedvr2` | seedvr2 upscaler (could replace pipersr!) |

All share the same argparse mixins — `--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, `--guidance`, `--seed`, `--quantize`, `--model`, `--base-model`, `--output`, `--metadata`, `--lora-paths`, `--lora-scales`, `--image-path`, `--image-strength`.

## Implementation plan (20 ranked items)

1. **(M)** Family-aware `_resolve_mflux_bin` in `agent/image_engine.py:131` + new `MFLUX_FAMILY_BIN` lookup table. Without this, none of the 0.17 families are reachable.
2. **(M)** Per-family default-flag table in `_generate_mflux`. klein-4B uses 4 steps + guidance 1; Z-Image-Turbo uses 9 steps + guidance 0; Krea-dev keeps 25 + guidance 4.5.
3. **(L)** Tier auto-detect via `_default_image_engine_for_tier` in `mlx_ltx_panel.py:~2042`. Apply at first boot when `agent_image_config.json` doesn't exist.
4. **(M)** Add FLUX.2-klein-4B to the model dropdown — `mlx_ltx_panel.py:10881`. Add `<optgroup>` per family.
5. **(M)** Add Z-Image-Turbo to the model dropdown.
6. **(M)** "Quality dial" pill group above the model dropdown — Fast (Z-Image-Turbo) / Balanced (klein-4B 4-bit) / Max (Krea-dev 8-bit on Roomy+, klein-4B 4-bit elsewhere).
7. **(M)** Reuse `_hf_model_install_async` for image weights — "Download" button in image drawer. Pre-flight via `_hf_size_estimate`.
8. **(S)** `health_check` extension reporting `weights_downloaded: bool` per family. Surface in Settings drawer pill.
9. **(M)** "Test current pipeline" button — runs portrait prompt at n=1, shows wall-time inline.
10. **(M)** OOM pre-flight: `if (model_gb + LTX_peak_gb > tier_ram): refuse + suggest 4-bit`.
11. **(M)** Tier-aware subprocess timeout — 240s for klein-4B/Z-Image, 600s for Krea-dev/dev.
12. **(L)** `mflux_warm_helper.py` — long-lived subprocess that loads the model once and accepts prompts via stdin. Mirror `mlx_warm_helper.py`. Saves 25-50s cold-load per shot batch.
13. **(M)** Sibling venv `image-gen/env/` so mflux deps don't conflict with LTX's transformers/mlx pins. `_resolve_mflux_bin` checks there first.
14. **(S)** HF gating UX — when `_hf_size_estimate` returns `gated: true`, surface "Open HF model card → accept terms → paste HF_TOKEN" inline.
15. **(S)** Update `docs/AGENTIC_FLOWS.md § Image generation backends` — document family dispatch, tier defaults, deprecate legacy `mflux-generate`-only path.
16. **(S)** Update `agent/image_engine.py:1-28` module docstring — current still says "Two backends ship in v1: mock, bfl" with mflux as "future".
17. **(M)** Tier-aware default for `ImageEngineConfig.mflux_model` (was `"krea-dev"` — wrong for 16 GB Macs).
18. **(L)** "Compare models" Settings page — runs test prompt across 3 picks side-by-side, click to set as default.
19. **(L)** Add `mflux-upscale-seedvr2` as post-render option for image candidates.
20. **(S)** Surface `--metadata` flag (writes JSON sidecar with prompt+seed+steps+model). Save next to candidate PNG.

## Sources

- [filipstrand/mflux on GitHub](https://github.com/filipstrand/mflux)
- [mflux releases page](https://github.com/filipstrand/mflux/releases)
- [black-forest-labs/FLUX.2-klein-4B on HF](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Runpod/FLUX.2-klein-4B-mflux-4bit on HF (~4.3 GB)](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit)
- [BFL FLUX.2 [klein] launch blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
- [Tongyi-MAI/Z-Image-Turbo on HF](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [filipstrand/Z-Image-Turbo-mflux-4bit on HF (~5.9 GB)](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit)
- [filipstrand/FLUX.1-Krea-dev-mflux-4bit on HF (9.61 GB)](https://huggingface.co/filipstrand/FLUX.1-Krea-dev-mflux-4bit)
- [Apatero — Flux on Apple Silicon performance guide](https://www.apatero.com/blog/flux-apple-silicon-m1-m2-m3-m4-complete-performance-guide-2025)
