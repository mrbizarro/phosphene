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

This module is the dispatch layer. Two backends ship in v1:

  - **mock** — flat-colored PNGs drawn with PIL. Zero deps beyond what
    LTX already needs. Used for testing the UX without spending API
    credits and as a fallback when no API key is set.
  - **bfl**  — Black Forest Labs (api.bfl.ml). Async submit + poll.
    Requires `bfl_api_key`. Models: flux-dev (cheap, 25 steps),
    flux-pro (better, slower), flux-schnell (4 steps, fastest).

Future backends (v2): mflux for fully-local Mac generation; Replicate;
fal.ai. Slot in by adding a `_generate_<kind>()` function and a clause
in `generate()`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ImageEngineConfig:
    """How the agent generates anchor stills.

    Persisted in `state/agent_image_config.json`. The HTTP `/agent/image/config`
    GET masks `bfl_api_key` (returns `has_bfl_api_key` bool only).
    """

    kind: str = "mock"                              # "mock" | "bfl"
    bfl_api_key: str = ""
    bfl_model: str = "flux-dev"                     # flux-dev | flux-pro | flux-schnell
    bfl_base_url: str = "https://api.bfl.ml/v1"

    def to_public_dict(self) -> dict:
        d = asdict(self)
        if d.get("bfl_api_key"):
            d["bfl_api_key"] = ""
            d["has_bfl_api_key"] = True
        else:
            d["has_bfl_api_key"] = False
        return d


# Aspect ratios → (width, height). Flux is most stable on these.
ASPECT_DIMS = {
    "16:9":  (1024, 576),
    "4:3":   (1024, 768),
    "1:1":   (768, 768),
    "9:16":  (576, 1024),
    "3:4":   (768, 1024),
    "21:9":  (1280, 544),
}


def generate(*, prompt: str, n: int, output_dir: Path,
             aspect: str = "16:9",
             base_seed: int | None = None,
             config: ImageEngineConfig) -> list[dict]:
    """Generate `n` candidate images for one shot. Saves under `output_dir`.

    Returns a list of `{png_path, seed, engine, width, height}` dicts in
    submission order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = ASPECT_DIMS.get(aspect, ASPECT_DIMS["16:9"])

    if config.kind == "mock":
        return _generate_mock(prompt, n, width, height, output_dir, base_seed)
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
    if config.kind == "bfl":
        if not config.bfl_api_key:
            return False, "BFL API key not configured"
        return True, f"BFL configured for {config.bfl_model}"
    return False, f"unknown engine kind: {config.kind!r}"


# ---- Backends ---------------------------------------------------------------
def _generate_mock(prompt: str, n: int, width: int, height: int,
                   output_dir: Path, base_seed: int | None) -> list[dict]:
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
        })
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
