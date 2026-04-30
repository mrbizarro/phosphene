#!/usr/bin/env python3
"""Parametric SVG generator for the Phosphene logo system.

Builds the radial-dash mark (matching the AI-generated reference at
1024×1024) along with simplified variants for tiny applications:

  phosphene_full.svg          full color · 9 rings · 36 dashes/ring
  phosphene_full_<size>.png   PNG renders at 16/32/64/128/256/512/1024

  phosphene_simple.svg        favicon-grade · 4 rings · 16 dashes/ring
  phosphene_simple_<size>.png  PNG renders at 16/32/64

  phosphene_mono.svg          monochrome white-on-void
  phosphene_mono_<size>.png   PNG renders at 32/64

The parametric build is the "vectorization" — instead of tracing a
raster (which produces noisy paths), we rebuild the design as math.
Same visual, cleaner SVG, infinitely scalable, easy to tweak.

Run: ./build_phosphene_logo.py
Outputs land alongside this script in assets/.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Brand palette — locked colors from the prompt spec
# ---------------------------------------------------------------------------
COLORS = {
    "void":     "#00061a",   # background
    "yellow":   "#FFE13B",   # center spark
    "magenta":  "#FF2E9F",   # inner rings (warm core)
    "violet":   "#B14AFF",   # middle rings (primary brand color)
    "cyan":     "#5EEAFF",   # outer rings (cool periphery)
    "white":    "#FFFFFF",   # mono variant
}


# ---------------------------------------------------------------------------
# Core mark generator
# ---------------------------------------------------------------------------
def build_mark(
    *,
    canvas: int = 1024,
    n_rings: int = 9,
    # Dashes-per-ring is the dial that controls "rays vs concentric rings."
    # At 36 you get a starburst; at 64+ the radial dashes pack tightly enough
    # that the eye reads them as rings (matches the AI-reference look).
    dashes_per_ring: int = 64,
    inner_radius_frac: float = 0.085,
    outer_radius_frac: float = 0.46,
    # Shorter, slightly thicker dashes = stockier "ring tick" feel rather
    # than the long-pointy "ray" feel of the previous tuning.
    dash_length_frac: float = 0.018,
    dash_width_frac: float = 0.0058,
    center_dot_frac: float = 0.012,
    phase_delta_deg: float = 1.6,       # subtle ring-to-ring rotation
    ring_colors: list[str] | None = None,
    bg: str = COLORS["void"],
    center_color: str = COLORS["yellow"],
    transparent_bg: bool = False,
) -> str:
    """Return SVG markup for a radial-dash phosphene mark.

    Geometry: `n_rings` concentric rings, evenly spaced from inner to outer
    radius. Each ring has `dashes_per_ring` dashes oriented radially. Each
    ring is rotated by `phase_delta_deg` * ring_index degrees so dashes
    don't align across rings — this is what creates the moiré-shimmer look
    that defines a phosphene.
    """
    cx = cy = canvas / 2

    if ring_colors is None:
        # Default 9-ring gradient: 3 magenta (warm core) → 3 violet → 3 cyan
        ring_colors = (
            [COLORS["magenta"]] * 3
            + [COLORS["violet"]]  * 3
            + [COLORS["cyan"]]    * 3
        )
    # If caller passes a shorter list, repeat the last color for outer rings
    if len(ring_colors) < n_rings:
        ring_colors = ring_colors + [ring_colors[-1]] * (n_rings - len(ring_colors))

    # Ring radii: linear interpolation from inner to outer
    inner_r = canvas * inner_radius_frac
    outer_r = canvas * outer_radius_frac
    if n_rings == 1:
        radii = [inner_r]
    else:
        step = (outer_r - inner_r) / (n_rings - 1)
        radii = [inner_r + i * step for i in range(n_rings)]

    dash_len = canvas * dash_length_frac
    dash_w = canvas * dash_width_frac
    dot_r = canvas * center_dot_frac

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas} {canvas}" '
        f'width="{canvas}" height="{canvas}">'
    )
    if not transparent_bg:
        parts.append(f'<rect width="{canvas}" height="{canvas}" fill="{bg}"/>')

    # One <g> per ring so the SVG is human-readable and any ring is easy
    # to retheme later in code or by hand-editing.
    for i, (radius, color) in enumerate(zip(radii, ring_colors)):
        phase_rad = math.radians(i * phase_delta_deg)
        parts.append(f'<g data-ring="{i}" stroke="{color}" stroke-width="{dash_w:.3f}" '
                     f'stroke-linecap="round">')
        for d in range(dashes_per_ring):
            theta = phase_rad + d * (2 * math.pi / dashes_per_ring)
            r1 = radius - dash_len / 2
            r2 = radius + dash_len / 2
            x1 = cx + r1 * math.cos(theta)
            y1 = cy + r1 * math.sin(theta)
            x2 = cx + r2 * math.cos(theta)
            y2 = cy + r2 * math.sin(theta)
            parts.append(
                f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'
            )
        parts.append('</g>')

    # Center dot (the spark)
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{dot_r:.2f}" fill="{center_color}"/>'
    )
    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
def variant_full(canvas: int = 1024) -> str:
    """The hero version — matches the AI reference."""
    return build_mark(canvas=canvas)


def variant_simple(canvas: int = 1024) -> str:
    """Favicon-grade: 4 rings, fewer dashes, thicker strokes. Survives at 32 px.

    Color story preserved (yellow → magenta → violet → cyan) but compressed
    to the four anchor colors.
    """
    return build_mark(
        canvas=canvas,
        n_rings=4,
        dashes_per_ring=16,
        inner_radius_frac=0.10,
        outer_radius_frac=0.43,
        dash_length_frac=0.045,    # 2× thicker dashes for legibility
        dash_width_frac=0.012,
        center_dot_frac=0.025,
        phase_delta_deg=4.0,        # more pronounced phase shift at 16 dashes
        ring_colors=[
            COLORS["magenta"],
            COLORS["violet"],
            COLORS["violet"],
            COLORS["cyan"],
        ],
    )


def variant_mono(canvas: int = 1024) -> str:
    """Monochrome white-on-void — for contexts where the gradient muddies."""
    return build_mark(
        canvas=canvas,
        ring_colors=[COLORS["white"]] * 9,
        center_color=COLORS["white"],
    )


def variant_wordmark(
    canvas_height: int = 256,
    aspect: float = 4.0,
    word: str = "phosphene",
    text_color: str = COLORS["violet"],
) -> str:
    """Horizontal lockup: mark on left, "phosphene" wordmark on right.

    Built as a pure SVG composite — guaranteed correct spelling (vs the
    AI prompt path where the model sometimes produces 'phospnene' or
    similar). Falls back through a font stack so it renders cleanly on
    systems without a specific typeface; replace with your locked
    typeface (e.g. 'Inter Tight' or 'General Sans') after the brand
    typography is locked.

    The mark itself is rendered at the canvas height; the wordmark sits
    to the right of it with breathing room. Total width = canvas_height
    × aspect (default 4:1 = 1024×256, suitable for Pinokio Discover and
    GitHub social cards).
    """
    h = canvas_height
    w = int(h * aspect)
    mark_size = h
    cx_mark = mark_size / 2
    cy_mark = h / 2

    # Reuse the full-mark generator but inline its <g> content into our
    # composite SVG (so we don't double-wrap an <svg>).
    mark_svg_full = build_mark(canvas=mark_size, transparent_bg=True)
    # Strip the <svg> outer + extract <g>/<circle>
    inner_start = mark_svg_full.index('>') + 1
    inner_end = mark_svg_full.rindex('</svg>')
    mark_inner = mark_svg_full[inner_start:inner_end]

    # Wordmark x position — right edge of mark + breathing room
    text_x = mark_size + h * 0.12
    text_y = h * 0.62      # baseline tuned for sentence-case sans
    font_size = h * 0.42
    letterspacing = font_size * 0.02

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <rect width="{w}" height="{h}" fill="{COLORS['void']}"/>
  <g transform="translate(0, 0)">{mark_inner}</g>
  <text x="{text_x}" y="{text_y}"
        font-family="'Inter Tight','General Sans','Söhne','SF Pro Display',-apple-system,system-ui,sans-serif"
        font-size="{font_size:.1f}"
        font-weight="500"
        letter-spacing="{letterspacing:.2f}"
        fill="{text_color}">{word}</text>
</svg>'''


def variant_favicon_16(canvas: int = 256) -> str:
    """Ultra-simplified for browser-tab favicon at 16×16. Renders at 256
    so we have headroom; rsvg/cairosvg downscaling does the heavy lifting.

    Just a violet circle with a yellow center dot. The rings disappear at
    16 px anyway — this is the silhouette that survives.
    """
    cx = cy = canvas / 2
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas} {canvas}" width="{canvas}" height="{canvas}">
  <rect width="{canvas}" height="{canvas}" fill="{COLORS['void']}"/>
  <circle cx="{cx}" cy="{cy}" r="{canvas * 0.40}" fill="none"
          stroke="{COLORS['violet']}" stroke-width="{canvas * 0.06}"/>
  <circle cx="{cx}" cy="{cy}" r="{canvas * 0.10}" fill="{COLORS['yellow']}"/>
</svg>'''


# ---------------------------------------------------------------------------
# Render PNG previews via cairosvg
# ---------------------------------------------------------------------------
def render_pngs(svg_text: str, base_path: Path, sizes: list[int]) -> None:
    """Use cairosvg to rasterize the SVG at each requested size.

    `sizes` are interpreted as the longer-edge length; the renderer keeps
    the SVG's native aspect ratio (so a 4:1 wordmark stays 4:1, not
    forced to square)."""
    import re
    import cairosvg

    # Pull viewBox to know the source aspect ratio.
    m = re.search(r'viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"', svg_text)
    if m:
        src_w, src_h = float(m.group(1)), float(m.group(2))
    else:
        src_w, src_h = 1.0, 1.0
    aspect = src_w / src_h

    for size in sizes:
        # Treat `size` as the longer edge.
        if aspect >= 1:
            out_w, out_h = size, max(1, int(round(size / aspect)))
        else:
            out_w, out_h = max(1, int(round(size * aspect))), size
        out = base_path.parent / f"{base_path.stem}_{size}.png"
        cairosvg.svg2png(
            bytestring=svg_text.encode(),
            write_to=str(out),
            output_width=out_w,
            output_height=out_h,
        )
        print(f"  rendered {out.name}  ({out_w}×{out_h})")


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = Path(__file__).parent
    print(f"writing to: {out_dir}")

    builds = [
        ("phosphene_full",     variant_full(),    [16, 32, 64, 128, 256, 512, 1024]),
        ("phosphene_simple",   variant_simple(),  [16, 32, 64, 128, 256]),
        ("phosphene_mono",     variant_mono(),    [32, 64, 128, 256]),
        ("phosphene_favicon",  variant_favicon_16(), [16, 32, 64]),
        # Wordmark lockups — different aspect ratios for different homes
        ("phosphene_wordmark", variant_wordmark(canvas_height=256, aspect=4.0),
         [256, 512, 1024]),       # 4:1 — Pinokio tile / GitHub social card
        ("phosphene_wordmark_compact", variant_wordmark(canvas_height=256, aspect=3.0),
         [256, 512]),              # 3:1 — header banner, narrower contexts
    ]

    for name, svg_text, sizes in builds:
        svg_path = out_dir / f"{name}.svg"
        svg_path.write_text(svg_text)
        print(f"\n{name}.svg ({len(svg_text):,} bytes)")
        render_pngs(svg_text, svg_path, sizes)


if __name__ == "__main__":
    main()
