"""Image generation engines for the Agentic Flows planner.

The director-collaboration loop:

  1. Agent plans a multi-shot piece in plain text.
  2. Agent calls `generate_shot_images(shot_label, prompt, n=4)` per shot.
     Each call returns a list of candidate PNGs saved on disk.
  3. UI renders the candidates as a thumbnail grid in the tool-result
     card. User clicks the best one — that POSTs to
     `/agent/sessions/<id>/anchors/select` which records the choice in
     `session.tool_state["selected_anchors"]`.
  4. User types "render". Agent reads the selected anchors and submits
     i2v shots with each anchor as `ref_image_path`. The video model
     fills the motion between known frames; the look is locked.

This module is the dispatch layer. Three backends ship today:

  - **mock** — flat-colored PNGs drawn with PIL. Zero deps beyond what
    LTX already needs. Used for testing the UX without spending API
    credits and as a fallback when no API key is set.
  - **bfl**  — Black Forest Labs (api.bfl.ml). Async submit + poll.
    Requires `bfl_api_key`. Models: flux-dev (cheap, 25 steps),
    flux-pro (better, slower), flux-schnell (4 steps, fastest).
  - **mflux** — fully-local Mac generation via filipstrand/mflux. Now
    multi-family: mflux 0.17.x ships per-family CLI commands
    (`mflux-generate-flux2`, `mflux-generate-z-image-turbo`,
    `mflux-generate-fibo`, `mflux-generate-qwen`, `mflux-generate-kontext`)
    in addition to the legacy `mflux-generate` (flux1-only). We auto-detect
    the family from the model id and call the right binary, with
    per-family step / guidance defaults so users who just pick a model
    don't have to know it needs 4 steps vs 25.

  Recommended defaults (May 2026):
    - **Comfortable+ (32 GB+)**  → `Runpod/FLUX.2-klein-4B-mflux-4bit`
      via `mflux-generate-flux2`, 4 steps, guidance 1.0. Apache 2.0,
      ~4.3 GB on disk, 4 candidates per shot in 50-75 s.
    - **Compact (16-32 GB)**     → `filipstrand/Z-Image-Turbo-mflux-4bit`
      via `mflux-generate-z-image-turbo`, 9 steps, guidance 0.0.
      Apache 2.0, ~5.9 GB on disk.

See `docs/IMAGE_GEN_RESEARCH_2026-05.md` for the full landscape and
tier-aware default table.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ImageEngineConfig:
    """How the agent generates anchor stills.

    Persisted in `state/agent_image_config.json`. The HTTP `/agent/image/config`
    GET masks `bfl_api_key` (returns `has_bfl_api_key` bool only).
    """

    # `mock` was the historical default but it produces flat colored
    # rectangles — every fresh install hit this and got confused output
    # before realizing they had to pick a real engine in Settings. New
    # default is `mflux` with FLUX.2 [klein] (Apache 2.0, ~4.3 GB,
    # 4-step generation). Existing configs with kind=mock get auto-
    # promoted at panel load time IF mflux is installed (see
    # _load_agent_image_config in mlx_ltx_panel.py).
    kind: str = "mflux"                             # "mock" | "mflux" | "bfl"

    # BFL (cloud)
    bfl_api_key: str = ""
    bfl_model: str = "flux-dev"                     # flux-dev | flux-pro | flux-pro-1.1 | flux-schnell
    bfl_base_url: str = "https://api.bfl.ml/v1"

    # mflux (local, MLX-native — `pip install mflux`).
    # Default updated 2026-05 to FLUX.2 [klein] 4B 4-bit — Apache 2.0,
    # 4-step inference, ~4.3 GB on disk, ~12-18 s/image on Comfortable
    # (M-series 32 GB+). Picks superseded the previous `krea-dev`
    # default which is non-commercial and 12 GB.
    #
    # `mflux_model` accepts either a named shorthand the CLI understands
    # ("krea-dev", "dev", "schnell", "dev-fill", "flux2-klein-4b") OR an
    # HF repo id / local path (e.g. "Runpod/FLUX.2-klein-4B-mflux-4bit").
    # `mflux_family` is "auto" by default — `_infer_mflux_family` reads
    # the model string and dispatches to the right CLI:
    #   flux1            → mflux-generate           (krea-dev, dev, schnell)
    #   flux2            → mflux-generate-flux2     (klein-4B/9B, dev)
    #   z_image_turbo    → mflux-generate-z-image-turbo
    #   z_image          → mflux-generate-z-image
    #   fibo             → mflux-generate-fibo
    #   qwen             → mflux-generate-qwen
    #   kontext          → mflux-generate-kontext
    # Override only when the inference fails for a custom model name.
    mflux_model: str = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    mflux_family: str = "auto"                      # "auto" or one of the family ids above
    mflux_base_model: str = ""                      # only needed when mflux_model is a path/HF id and family inference can't tell the architecture
    mflux_steps: int = 0                            # 0 = use family default (4/9/25 per family)
    mflux_quantize: int = 4                         # 4 | 8 | 16 — 4-bit fits comfortably on 64 GB
    mflux_guidance: float | None = None             # None = use family default (1.0 / 0.0 / 4.5 / 5.0 per family)
    mflux_python_path: str = ""                     # optional override for the mflux CLI location
    # Optional Lightning / acceleration LoRAs. With qwen_edit + a 4-step
    # Lightning LoRA, generation drops from ~5 min to ~10-15 sec per
    # image. Passed straight through to mflux's `--lora-paths` and
    # `--lora-scales` flags (lengths must match if both set; mflux
    # rejects the mismatch). Each path is a HuggingFace repo id, a
    # collection-format string (`repo:filename.safetensors`), or a
    # local file path.
    mflux_lora_paths: list[str] = field(default_factory=list)
    mflux_lora_scales: list[float] = field(default_factory=list)

    def to_public_dict(self) -> dict:
        d = asdict(self)
        if d.get("bfl_api_key"):
            d["bfl_api_key"] = ""
            d["has_bfl_api_key"] = True
        else:
            d["has_bfl_api_key"] = False
        return d


# Aspect ratios → (width, height). Picks favor larger dimensions because
# Qwen-Image-Edit-2509 is documented to be best at 1024² and 1280×720;
# the previous 1024×576 default for 16:9 was a Flux-era artifact and
# visibly degrades qwen_edit output. Flux2 / Z-Image fine here too — they
# scale gracefully past 1024.
ASPECT_DIMS = {
    "16:9":  (1280, 720),
    "4:3":   (1024, 768),
    "1:1":   (1024, 1024),
    "9:16":  (720, 1280),
    "3:4":   (768, 1024),
    "21:9":  (1280, 544),
}


def generate(*, prompt: str, n: int, output_dir: Path,
             aspect: str = "16:9",
             base_seed: int | None = None,
             refs: list[str] | None = None,
             config: ImageEngineConfig,
             on_log: "callable | None" = None) -> list[dict]:
    """Generate `n` candidate images for one shot. Saves under `output_dir`.

    Args:
        prompt: text prompt
        n: number of candidates
        output_dir: where PNGs land
        aspect: aspect-ratio key from ASPECT_DIMS
        base_seed: when given, candidate i uses base_seed + i (reproducible)
        refs: optional list of reference image paths (1-3 supported by
              Qwen-Image-Edit-2509 via mflux-generate-qwen-edit). Refs are
              the way to lock character + place / character + product
              composition without training a LoRA. The agent's path:
              user uploads or library-picks 1-3 reference images, the
              engine composes them per the prompt, the resulting still
              becomes an LTX keyframe. Backends without multi-ref support
              (mock, plain qwen, flux1/2, z_image, fibo, BFL) currently
              ignore refs — they fall back to text-only generation. The
              caller is responsible for picking a config.kind/family
              that respects refs (today: qwen_edit). If refs are passed
              to a non-supporting family, a warning shows in the result
              dict but the call still succeeds with text-only output.
        config: backend selection + per-backend params
        on_log: optional callback `(line: str) -> None` invoked once per
            stdout/stderr line emitted by the underlying engine. Lets the
            panel surface live mflux progress (e.g., tqdm `[12/30]` step
            lines) instead of waiting silently. None = discard log lines.

    Returns a list of `{png_path, seed, engine, width, height, ...}`
    dicts in submission order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = ASPECT_DIMS.get(aspect, ASPECT_DIMS["16:9"])

    refs = list(refs or [])
    # Validate ref paths up-front so we fail fast — Qwen-Edit-2509 supports
    # 1-3 input images; more than 3 is unsupported by the model.
    if refs:
        if len(refs) > 3:
            raise ValueError(
                f"refs: Qwen-Image-Edit-2509 supports 1-3 input images, got {len(refs)}"
            )
        for r in refs:
            if not Path(r).is_file():
                raise FileNotFoundError(f"ref image not found: {r}")

    if config.kind == "mock":
        return _generate_mock(prompt, n, width, height, output_dir, base_seed, refs=refs)
    if config.kind == "mflux":
        return _generate_mflux(prompt, n, width, height, output_dir, base_seed, config, refs=refs, on_log=on_log)
    if config.kind == "bfl":
        return _generate_bfl(prompt, n, width, height, output_dir, base_seed, config)
    raise ValueError(f"unknown image engine kind: {config.kind!r}")


def health_check(config: ImageEngineConfig) -> tuple[bool, str]:
    """Light readiness probe for the configured backend."""
    if config.kind == "mock":
        try:
            from PIL import Image  # noqa: F401
            return True, "mock engine ready (PIL available)"
        except ImportError:
            return False, "PIL not installed — pip install Pillow"
    if config.kind == "mflux":
        bin_path = _resolve_mflux_bin(config)
        if not bin_path:
            return False, ("mflux not installed. Run: "
                           "ltx-2-mlx/env/bin/pip install mflux")
        return True, f"mflux ready at {bin_path}; model={config.mflux_model}"
    if config.kind == "bfl":
        if not config.bfl_api_key:
            return False, "BFL API key not configured"
        return True, f"BFL configured for {config.bfl_model}"
    return False, f"unknown engine kind: {config.kind!r}"


# Per-family CLI command. mflux 0.17.x splits FLUX.1, FLUX.2, Z-Image,
# FIBO, Qwen, and Kontext into separate `mflux-generate-*` binaries —
# each one knows its architecture, expected step count, guidance scale,
# etc. We dispatch by family. The legacy `mflux-generate` remains for
# FLUX.1 (krea-dev / dev / schnell / kontext) backward-compat.
MFLUX_FAMILY_BIN = {
    "flux1":          "mflux-generate",
    "flux2":          "mflux-generate-flux2",
    "z_image":        "mflux-generate-z-image",
    "z_image_turbo":  "mflux-generate-z-image-turbo",
    "fibo":           "mflux-generate-fibo",
    "qwen":           "mflux-generate-qwen",
    # Qwen-Image-Edit-2509 (multi-reference, 1-3 input images via --image-paths).
    # Apache 2.0. Default model: Qwen/Qwen-Image-Edit-2509.
    # Trained for "person + person", "person + product", "person + scene"
    # combinations — exactly the place + character composition the agent
    # needs for keyframe stills. mflux 0.11.1+.
    "qwen_edit":      "mflux-generate-qwen-edit",
    "kontext":        "mflux-generate-kontext",
}

# Sensible per-family defaults so a user who picks a model from the
# dropdown gets the right step count + guidance without having to
# read the model card. `steps` and `guidance` apply to `_generate_mflux`
# when `config.mflux_steps == 0` or `mflux_guidance` is unset (the
# server only overrides when the user explicitly picks a custom value).
MFLUX_FAMILY_DEFAULTS = {
    "flux1":         {"steps": 25, "guidance": 4.5,  "base_model": "dev"},
    "flux2":         {"steps": 4,  "guidance": 1.0,  "base_model": "flux2-klein-4b"},
    "z_image":       {"steps": 25, "guidance": 5.0,  "base_model": ""},
    "z_image_turbo": {"steps": 9,  "guidance": 0.0,  "base_model": ""},
    "fibo":          {"steps": 30, "guidance": 5.0,  "base_model": ""},
    "qwen":          {"steps": 30, "guidance": 5.0,  "base_model": ""},
    # qwen_edit default: 8 steps. The Qwen card recommends 30-40 for
    # final-quality, but the agent + Image Studio are iteration tools —
    # ~1 min/image at Q4-8steps, then bump to 30 steps once the user
    # picks a composition they like. Pair with a Lightning 4-step LoRA
    # (mflux_lora_paths) to drop further to ~10-15 s. Guidance 4.0 from
    # the model card.
    "qwen_edit":     {"steps": 8,  "guidance": 4.0,  "base_model": ""},
    "kontext":       {"steps": 30, "guidance": 4.5,  "base_model": ""},
}


def _infer_mflux_family(model: str) -> str:
    """Map a model id / shorthand to its mflux family.

    Examples:
      "Runpod/FLUX.2-klein-4B-mflux-4bit"             → "flux2"
      "filipstrand/Z-Image-Turbo-mflux-4bit"          → "z_image_turbo"
      "Tongyi-MAI/Z-Image"                            → "z_image"
      "filipstrand/FLUX.1-Krea-dev-mflux-4bit"        → "flux1"
      "krea-dev" / "dev" / "schnell"                  → "flux1"
      "flux2-klein-4b" / "flux2-klein-9b"             → "flux2"
      "briaai/FIBO" / "fibo"                          → "fibo"
      "filipstrand/Qwen-Image-mflux-6bit" / "qwen-*"  → "qwen"
      "*kontext*"                                     → "kontext"

    Falls back to "flux1" so legacy configs keep working.
    """
    s = (model or "").lower()
    if "z-image-turbo" in s or "z_image_turbo" in s or "zimage-turbo" in s:
        return "z_image_turbo"
    if "z-image" in s or "z_image" in s or "tongyi-mai/z-image" in s:
        return "z_image"
    if "flux2" in s or "flux.2" in s or "flux-2" in s:
        return "flux2"
    if "kontext" in s:
        return "kontext"
    if "fibo" in s or "briaai/fibo" in s:
        return "fibo"
    # qwen_edit must be matched BEFORE plain qwen — "qwen-image-edit-2509"
    # contains "qwen" + "image" so the plain qwen branch would steal it.
    if "qwen" in s and ("image-edit" in s or "image_edit" in s or "qwen-edit" in s):
        return "qwen_edit"
    if "qwen" in s and "image" in s:
        return "qwen"
    # Default: flux1 (krea-dev / dev / schnell, plus filipstrand/FLUX.1-*)
    return "flux1"


def _resolve_mflux_family(config: ImageEngineConfig) -> str:
    """Resolve config.mflux_family, falling back to inference."""
    fam = (config.mflux_family or "auto").strip().lower()
    if fam == "auto" or fam not in MFLUX_FAMILY_BIN:
        return _infer_mflux_family(config.mflux_model)
    return fam


def _resolve_mflux_bin(config: ImageEngineConfig) -> str | None:
    """Find the `mflux-generate-*` executable for the configured family.

    Order:
      1. config.mflux_python_path (explicit override — useful when mflux
         is installed in a venv outside the panel's standard location).
         The override is taken as-is; user is responsible for pointing
         at the right per-family binary.
      2. The Phosphene panel's bundled venv (ltx-2-mlx/env/bin/<bin>).
      3. The sibling image-gen venv (image-gen/env/bin/<bin>) — reserved
         for a future split where mflux's transformers/mlx pins differ
         from LTX's. Doesn't exist today; safe to probe.
      4. shutil.which(<bin>) — system PATH fallback.

    Returns the absolute path to the binary, or None if not installed.
    """
    import os
    import shutil

    fam = _resolve_mflux_family(config)
    bin_name = MFLUX_FAMILY_BIN.get(fam, "mflux-generate")

    if config.mflux_python_path:
        cand = Path(config.mflux_python_path)
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "ltx-2-mlx" / "env" / "bin" / bin_name,
        repo_root / "image-gen" / "env" / "bin" / bin_name,
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    return shutil.which(bin_name)


# ---- Backends ---------------------------------------------------------------
def _generate_mock(prompt: str, n: int, width: int, height: int,
                   output_dir: Path, base_seed: int | None,
                   refs: list[str] | None = None) -> list[dict]:
    """Draw `n` distinguishable colored PNGs locally. Each carries a label
    with the candidate index + first ~60 chars of the prompt so the user
    can verify the UX flow without confusing identical images.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError(
            "PIL not available — install Pillow to use the mock engine, "
            "or configure a real backend in Settings → Image generation."
        ) from e

    palette = [
        (220, 100, 80),  (80, 150, 220),  (220, 200, 80),  (130, 220, 130),
        (180, 100, 200), (240, 160, 100), (100, 200, 200), (180, 180, 180),
    ]
    font = None
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font = ImageFont.truetype(path, max(18, height // 28))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    results = []
    for i in range(n):
        color = palette[i % len(palette)]
        img = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(img)

        # Subtle gradient stripe on top so candidates aren't perfectly flat
        for y in range(0, height // 4):
            tint = 1 - (y / (height // 4)) * 0.4
            stripe = tuple(min(255, int(c * tint + 20)) for c in color)
            draw.line([(0, y), (width, y)], fill=stripe)

        title = f"Mock candidate {i + 1}"
        body = (prompt or "")[:120]
        draw.text((24, 24), title, fill=(255, 255, 255), font=font)
        # Word-wrap the body
        words = body.split()
        line = ""
        y = 24 + (font.size if hasattr(font, "size") else 18) + 12
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > 36:
                draw.text((24, y), line, fill=(255, 255, 255), font=font)
                y += (font.size if hasattr(font, "size") else 18) + 6
                line = w
            else:
                line = test
        if line:
            draw.text((24, y), line, fill=(255, 255, 255), font=font)

        seed = (base_seed if base_seed is not None else 1000) + i
        out_path = output_dir / f"cand_{i:02d}_mock.png"
        img.save(out_path, "PNG")
        results.append({
            "png_path": str(out_path), "seed": seed, "engine": "mock",
            "width": width, "height": height,
            # Mock doesn't compose refs into the output. Surface this
            # explicitly so callers (panel /image/generate, agent
            # generate_shot_images) can warn the user that picking a
            # ref didn't influence the result.
            "refs_ignored": bool(refs),
        })
    return results


def _generate_mflux(prompt: str, n: int, width: int, height: int,
                    output_dir: Path, base_seed: int | None,
                    config: ImageEngineConfig,
                    refs: list[str] | None = None,
                    on_log: "callable | None" = None) -> list[dict]:
    """Subprocess the right `mflux-generate-*` binary ONCE for all `n` seeds.

    Pre-2026-05-08 this spawned one subprocess per candidate, paying the
    ~30-60 s model-load cost N times. mflux's per-family CLIs already
    loop internally over `args.seed`, loading the model exactly once
    and looping over `for seed in args.seed: image = qwen.generate_image(...)`.
    By passing every seed in a single invocation and using mflux's
    `{seed}` template in `--output`, we drop wall-time on n=4 batches
    from "4 × (load + gen)" to "1 × load + 4 × gen", which is a
    3-4× speedup on Lightning configs and a 1.3-1.6× speedup on
    raw 8-step Q4. Bigger when generation itself is fast (load dominates).

    Family is inferred from `config.mflux_model` (or taken from
    `config.mflux_family` when set). FLUX.1 / FLUX.2 / Z-Image /
    Z-Image-Turbo / FIBO / Qwen / Qwen-Edit / Kontext each have their
    dedicated CLI from mflux 0.17.x.

    `config.mflux_lora_paths` + `config.mflux_lora_scales` plumb through
    to `--lora-paths` and `--lora-scales`. The intended use is a
    Lightning LoRA on qwen_edit:
      mflux_lora_paths = ["lightx2v/Qwen-Image-Edit-2511-Lightning"]
      mflux_lora_scales = [1.0]
      mflux_steps = 4
    which lands at ~10-15 s/image vs ~5 min for raw 30-step gen.

    Recommended defaults (May 2026):
      - **Qwen-Image-Edit-2509** — Apache 2.0, multi-ref. Iteration:
        Q4 + 8 steps, ~1 min/image. Final: Q8 + 30 steps, ~5 min.
      - **FLUX.2 [klein] 4B 4-bit** — Apache 2.0, 4 steps, ~4.3 GB.
      - **Z-Image-Turbo 4-bit** — Apache 2.0, 9 steps, ~5.9 GB.

    First run downloads weights (4-34 GB depending on family) to
    ~/.cache/huggingface. Subsequent calls use the cached copy.
    """
    import os
    import random
    import subprocess

    bin_path = _resolve_mflux_bin(config)
    fam = _resolve_mflux_family(config)
    bin_name = MFLUX_FAMILY_BIN.get(fam, "mflux-generate")
    fam_defaults = MFLUX_FAMILY_DEFAULTS.get(fam, MFLUX_FAMILY_DEFAULTS["flux1"])

    if not bin_path:
        raise RuntimeError(
            f"{bin_name} not found (family: {fam}). Install or upgrade "
            f"mflux into the panel's venv: "
            f"`ltx-2-mlx/env/bin/pip install -U mflux>=0.17` "
            f"(see docs/AGENTIC_FLOWS.md § Image generation backends). "
            f"First-run model download is 4-34 GB to ~/.cache/huggingface "
            f"depending on family."
        )

    # qwen_edit refuses to start without --image-paths (mflux argparse
    # marks it required for the qwen-edit CLI). Catch this here with a
    # clear validation error instead of letting the user wait minutes
    # for a confusing argparse failure.
    if fam == "qwen_edit" and not refs:
        raise ValueError(
            "qwen_edit requires at least 1 reference image. Either "
            "switch the engine to 'qwen' (text-to-image, same model "
            "family) or pass refs=[<path>] to compose against."
        )

    # Effective steps + guidance: prefer user-set values if non-zero,
    # otherwise fall back to the family default.
    eff_steps = config.mflux_steps if config.mflux_steps > 0 else fam_defaults["steps"]
    eff_guidance = (config.mflux_guidance
                    if config.mflux_guidance is not None
                    else fam_defaults["guidance"])

    # Build the seed list. mflux's CLI loops `for seed in args.seed`
    # AFTER loading the model exactly once, so passing all seeds at
    # once amortizes the cold-start cost across N images.
    if base_seed is not None:
        seeds = [base_seed + i for i in range(n)]
    else:
        # Equivalent to mflux's --auto-seeds N but explicit — gives us
        # the seed values up-front so we can map them to output paths.
        seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]

    # Output path uses mflux's `{seed}` template — `--output cand_{seed}_mflux.png`
    # writes one file per seed value (see qwen_image_edit_generate.py:61
    # `Path(args.output.format(seed=seed))`). Including the seed in the
    # filename also guarantees no collisions if the user resubmits the
    # same shot label without `append: true`.
    output_template = str(output_dir / "cand_{seed}_mflux.png")

    cmd = [
        bin_path,
        "--model", config.mflux_model,
        "--prompt", prompt,
        "--output", output_template,
        "--steps", str(eff_steps),
        "--width", str(width),
        "--height", str(height),
        "-q", str(config.mflux_quantize),
        "--guidance", str(eff_guidance),
        "--seed", *[str(s) for s in seeds],
    ]
    # Multi-reference input — only the qwen_edit family consumes
    # --image-paths today (mflux v0.11.1+). For other families we
    # silently drop refs and tag refs_ignored=True on each result.
    refs_used: list[str] = []
    if refs and fam == "qwen_edit":
        refs_used = [str(Path(r).resolve()) for r in refs]
        cmd.extend(["--image-paths", *refs_used])
    # Optional Lightning / acceleration LoRAs
    if config.mflux_lora_paths:
        cmd.append("--lora-paths")
        cmd.extend(config.mflux_lora_paths)
        if config.mflux_lora_scales:
            cmd.append("--lora-scales")
            cmd.extend(str(s) for s in config.mflux_lora_scales)
    # When `--model` is a HuggingFace id or local path (contains a
    # slash or starts with `~`), mflux needs `--base-model` to know
    # which architecture to instantiate. Fall through to the per-family
    # default if the user hasn't set one.
    looks_like_path = ("/" in config.mflux_model
                       or config.mflux_model.startswith("~"))
    if looks_like_path:
        base = (config.mflux_base_model
                or fam_defaults.get("base_model") or "")
        if base:
            cmd.extend(["--base-model", base])

    # Inherit env so HF_HOME / HF_TOKEN are honored for the one-time
    # weight download. Family-aware total timeout — first run can pull
    # 22-34 GB for qwen_edit which makes the cold-start much longer
    # than steady-state. Per-family steady-state per-image is roughly:
    #   flux2 / z_image_turbo: ~15 s @ 4-9 steps Q4
    #   qwen_edit:             ~30 s @ 8 steps Q4 / ~5 min @ 30 steps Q8
    # Plus the one-time load, plus first-run download.
    env = os.environ.copy()
    per_image_budget = 60 if fam in ("flux2", "z_image_turbo") else 360
    cold_start_budget = 240 if fam in ("flux2", "z_image_turbo") else 1800
    timeout_s = cold_start_budget + per_image_budget * n

    # Switched to Popen + line-streaming so the panel can surface mflux's
    # tqdm progress bars (`[12/30]`-style step lines) and weight-download
    # status to the user while the subprocess runs. Capturing both
    # stdout and stderr to one stream so the line ordering matches what
    # mflux actually printed; tail of the stream is kept for inclusion
    # in any error message.
    import collections
    last_lines: collections.deque[str] = collections.deque(maxlen=64)
    stderr_tail = ""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    try:
        deadline = time.time() + timeout_s
        if proc.stdout is not None:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n")
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                last_lines.append(line)
                if on_log is not None:
                    try:
                        on_log(line)
                    except Exception:                # noqa: BLE001
                        # A buggy logger callback must not break the gen.
                        pass
                if time.time() > deadline:
                    proc.kill()
                    raise RuntimeError(
                        f"{bin_name} timed out after {timeout_s}s on a batch of "
                        f"{n} seeds. First run downloads weights — qwen_edit Q8 "
                        f"weights are ~34 GB; give it longer or pre-pull."
                    )
        rc = proc.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired as e:
        try:
            proc.kill()
        except OSError:
            pass
        raise RuntimeError(
            f"{bin_name} timed out after {timeout_s}s on a batch of {n} seeds. "
            f"First run downloads weights; give it longer next time."
        ) from e
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

    stderr_tail = "\n".join(last_lines)
    if rc != 0:
        raise RuntimeError(
            f"{bin_name} failed (exit {rc}) on batch of {n} seeds. "
            f"Tail of stdout/stderr:\n{stderr_tail[:1200]}"
        )

    # Build results from whatever files actually landed. If the
    # subprocess errored partway through (e.g. one seed's generation
    # raised), some seeds may not have produced an output — return the
    # ones that did rather than failing the whole batch. Order by seed
    # so candidate numbering is deterministic.
    results: list[dict] = []
    for i, seed in enumerate(seeds):
        path = Path(output_template.format(seed=seed))
        if not path.is_file():
            # Surface an explicit error only if zero candidates landed —
            # partial success returns whatever we got.
            continue
        results.append({
            "png_path": str(path),
            "seed": seed,
            "engine": "mflux",
            "family": fam,
            "model": config.mflux_model,
            "width": width, "height": height,
            "refs": refs_used,
            "refs_ignored": (bool(refs) and fam != "qwen_edit"),
            "lora_paths": list(config.mflux_lora_paths or []),
        })

    if not results:
        raise RuntimeError(
            f"{bin_name} exited 0 but produced no files for any of {n} seeds "
            f"({seeds!r}). Check the panel log for mflux warnings."
        )
    return results


def _generate_bfl(prompt: str, n: int, width: int, height: int,
                  output_dir: Path, base_seed: int | None,
                  config: ImageEngineConfig) -> list[dict]:
    """Submit + poll Black Forest Labs API for each candidate.

    BFL's API is async-poll. POST `/{model}` returns `{id}`; GET
    `/get_result?id=...` returns `{status, result}` where status flips
    to "Ready" with `result.sample` (a download URL).
    """
    if not config.bfl_api_key:
        raise RuntimeError(
            "BFL API key not configured. Add one under Settings → Image "
            "generation, or pick the mock engine to test the workflow."
        )
    headers = {"Content-Type": "application/json", "X-Key": config.bfl_api_key}
    poll_headers = {"X-Key": config.bfl_api_key}
    submit_url = f"{config.bfl_base_url.rstrip('/')}/{config.bfl_model}"
    poll_url_base = f"{config.bfl_base_url.rstrip('/')}/get_result"

    results = []
    for i in range(n):
        body = {
            "prompt": prompt,
            "width": width, "height": height,
            "prompt_upsampling": False,
            "safety_tolerance": 2,
        }
        if base_seed is not None:
            body["seed"] = base_seed + i

        # Submit
        req = urllib.request.Request(
            submit_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:                       # noqa: BLE001
                pass
            raise RuntimeError(
                f"BFL submit failed (HTTP {e.code} {e.reason}): {detail[:400]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"BFL unreachable: {e.reason}") from e

        task_id = resp.get("id") or resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"BFL submit returned no task id: {resp!r}")

        # Poll
        deadline = time.time() + 180
        sample_url = None
        seed_used = base_seed + i if base_seed is not None else 0
        while time.time() < deadline:
            time.sleep(1.5)
            poll_url = f"{poll_url_base}?id={task_id}"
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(poll_url, headers=poll_headers),
                    timeout=20,
                ) as r:
                    rs = json.loads(r.read())
            except urllib.error.URLError:
                continue
            status = rs.get("status", "")
            if status == "Ready":
                inner = rs.get("result") or {}
                sample_url = inner.get("sample")
                if "seed" in inner:
                    seed_used = inner["seed"]
                break
            if status in ("Error", "Failed", "Content Moderated"):
                raise RuntimeError(f"BFL task {task_id} {status}: {rs!r}")
        if sample_url is None:
            raise RuntimeError(f"BFL task {task_id} timed out after 180s")

        # Download
        try:
            with urllib.request.urlopen(sample_url, timeout=60) as r:
                data = r.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"BFL sample download failed: {e.reason}") from e

        out_path = output_dir / f"cand_{i:02d}_bfl.png"
        out_path.write_bytes(data)
        results.append({
            "png_path": str(out_path), "seed": seed_used, "engine": "bfl",
            "width": width, "height": height,
        })
    return results
