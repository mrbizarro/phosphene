#!/usr/bin/env python3
"""Auto-editor — pick the good part of each clip, and cut on the beat.

WHY THIS EXISTS
---------------
`_sb_export()` in the panel assembles a film by concatenating WHOLE clips in
`n` order. That is the right default (it is exactly what the folder contains,
and it never surprises anyone), but it is not an edit — it is a playlist. Two
things are wrong with a playlist made of generated video:

1. **A generated clip is not uniformly good.** The first frames are often soft
   while the sampler settles, the tail commonly degrades — softening, smearing,
   sometimes washing to white — and in between there are usually two or three
   seconds that are clearly the shot. Playing all of it spends the viewer's
   attention on the parts you would have cut by hand.
2. **The cuts ignore the music.** A cut lands wherever the clip happened to
   end, so a 5.125 s render cuts at 5.125 s, then 10.250 s, then 15.375 s —
   a grid that has nothing to do with the track under it. The eye notices;
   that is most of why an assembled film reads as "clips glued end to end"
   while the same shots cut on the downbeat read as a film.

This module fixes both, as ANALYSIS ONLY. It opens no sockets, imports nothing
from `mlx_ltx_panel`, loads no model, and writes no video. It answers two
questions — "which stretch of this clip is the good one" and "where are the
beats" — and turns the answers into a cut plan that the panel's existing
assembler can execute in the filtergraph pass it was already running.

    storyboard_edit.py      which seconds, and where the cuts go   (this file)
    mlx_ltx_panel.py        copies, filtergraph, encode, gallery

DEPENDENCIES: numpy and the standard library. Nothing else. Not librosa, not
opencv, not torch, not decord. Everything below is FFT, autocorrelation and
array slicing, decoded through the ffmpeg the panel already ships with. That
is a deliberate constraint: this feature must not turn a 20 GB install into a
20 GB install plus a scientific Python stack.

HONESTY ABOUT WHAT THE MEASUREMENTS ARE
---------------------------------------
None of these are perceptual quality models. They are cheap proxies with known
failure modes, and every one of them reports its raw numbers so a human can
disagree:

  * *Sharpness* = variance of the Laplacian on a downscaled grey frame. It goes
    down for blur AND up for noise/grain. Within ONE clip (same scene, same
    grain, same encoder) that confound is mostly constant, which is why the
    score is normalised against the clip's own distribution and never compared
    across clips.
  * *Stability* = mean absolute frame-to-frame difference. Near zero means the
    picture is frozen (dead footage, or the sampler stopped moving); very high
    means thrash — strobing, a scene break inside the clip, or the artefacting
    that generated video does when it loses the plot. Both ends are bad, so the
    score is a band, not a maximum — and it is scored on the window's quietest
    quarter as well as its average, because a window that is mostly frozen and
    briefly busy averages into the healthy band and plays as a still frame.
  * *Luma sanity* = mean brightness plus the fraction of crushed / blown
    pixels. This one exists because of a real bug class: renders that go white
    (or black) in the last second. A window that contains that is rejected
    outright, not merely marked down.
  * *Head / tail guard* = a positional penalty, configurable, because "the tail
    softens" is a property of the GENERATOR, not something the pixels always
    reveal. It is a prior, and it is applied as one.

  * *Beat tracking* is a constant-tempo model: one BPM, one phase, 4/4. It is
    right on the electronic/orchestral material this was built for and it will
    be wrong on rubato, on a live drummer, and on anything that changes tempo
    mid-track. `beat_map()` therefore returns a `confidence` and the raw
    evidence behind it, and `plan_cut()` will happily produce an unsnapped
    plan when the caller decides the confidence is too low. A confident wrong
    answer is the one outcome this module is built to avoid.

DETERMINISM
-----------
Every function here is deterministic: fixed sample grids, no randomness, no
dict-ordering dependence, ties broken by the earlier candidate. The same
inputs give byte-identical plans, which is what makes a plan reviewable and a
test possible.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

__all__ = [
    "best_window", "beat_map", "snap", "plan_cut",
    "probe_media", "format_plan", "plan_total", "audio_span_for",
    "song_map", "director_pacing", "SECTION_LABELS",
]


# ---------------------------------------------------------------------------
# ffmpeg — resolved the same way the panel resolves it, but WITHOUT importing
# the panel. This module has to stay importable from a test, a CLI, or a
# future tool that has no business booting a web server.
# ---------------------------------------------------------------------------
def _resolve_ffmpeg() -> Path:
    candidates = [
        os.environ.get("LTX_FFMPEG"),
        shutil.which("ffmpeg"),
        str(Path.home() / "pinokio/bin/ffmpeg-env/bin/ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path("/usr/local/bin/ffmpeg")


FFMPEG = _resolve_ffmpeg()
FFPROBE = FFMPEG.parent / "ffprobe"


# ---------------------------------------------------------------------------
# Analysis constants. All of them are arguments somewhere below; these are the
# defaults, gathered here so a person recalibrating has one place to look.
# ---------------------------------------------------------------------------
ANALYSIS_FPS = 4.0          # frames per second SAMPLED, not the clip's rate.
ANALYSIS_WIDTH = 320        # grey analysis frames are 320 px wide.

# Motion, in mean absolute 8-bit grey levels between samples 1/ANALYSIS_FPS
# apart. Calibrated on real LTX-2 / H3 renders: a locked-off shot with only
# hair and breath moving sits at 1-3, ordinary camera movement at 4-12, a whip
# pan or a hard cut inside the clip at 25+.
MOTION_DEAD = 0.30          # below this the picture is not moving at all
MOTION_LOW = 1.20           # bottom of the "alive" band
MOTION_HIGH = 18.0          # top of the "alive" band
MOTION_THRASH = 38.0        # above this it is strobing / breaking up

# Luma, 0-255 on the grey analysis frame.
LUMA_BLACK = 8.0            # window mean below this = the render went dark
LUMA_WHITE = 243.0          # window mean above this = the render went white
LUMA_GOOD_LO = 18.0         # soft band; outside it the score ramps down
LUMA_GOOD_HI = 232.0
CRUSH_LEVEL = 8             # a pixel at or below this is crushed
BLOWN_LEVEL = 247           # a pixel at or above this is blown
# Veto thresholds, deliberately high. A night exterior on this corpus runs 72 %
# crushed and is perfectly good footage; only a frame that is essentially ALL
# black or ALL white is a render failure rather than a look.
CRUSH_VETO = 0.96
BLOWN_VETO = 0.96

# Score weights. The positives sum to 1.0 so a raw score reads as a fraction.
W_SHARP = 0.45
W_STABLE = 0.35
W_LUMA = 0.20
# Positional priors. The tail is weighted heavier than the head because the
# tail failure (softening, drift, wash-out) is both more common and more
# visible than the head one (a soft first frame or two).
HEAD_GUARD_S = 0.30
TAIL_GUARD_S = 0.75
HEAD_WEIGHT = 0.10
TAIL_WEIGHT = 0.25

# Beat tracking.
AUDIO_SR = 22050
N_FFT = 1024
HOP = 256
BPM_MIN, BPM_MAX = 60.0, 200.0
BPM_PRIOR_CENTER = 120.0    # log-normal tempo prior, in BPM
BPM_PRIOR_OCTAVES = 0.9     # sigma, in octaves
LOW_BAND_HZ = 250.0         # "kick band", used to choose the downbeat phase
DEFAULT_METER = 4


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


# ===========================================================================
# Probing
# ===========================================================================
def probe_media(path) -> dict | None:
    """Geometry + duration + audio facts, or None when ffprobe can't read it.

    Deliberately its own function rather than a call into the panel: the panel
    imports mlx, and an editor that needs a 20 GB model loaded to measure a
    Laplacian is not an editor. The shape of the return matches the panel's
    `_sb_probe_clip()` so a caller can hand either one to `best_window()`.
    """
    p = Path(str(path))
    try:
        out = subprocess.run(
            [str(FFPROBE), "-v", "error", "-show_entries",
             "stream=codec_type,width,height,sample_rate,duration,avg_frame_rate"
             ":format=duration", "-of", "json", str(p)],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout or "{}")
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    w = h = 0
    v_dur = 0.0
    fps = 0.0
    has_audio = False
    rate = 0
    for st in (data.get("streams") or []):
        if not isinstance(st, dict):
            continue
        kind = str(st.get("codec_type") or "")
        if kind == "video" and not w:
            try:
                w, h = int(st.get("width") or 0), int(st.get("height") or 0)
            except (TypeError, ValueError):
                w = h = 0
            v_dur = _f(st.get("duration"))
            raw = str(st.get("avg_frame_rate") or "0/1")
            if "/" in raw:
                num, _, den = raw.partition("/")
                fps = _f(num) / _f(den, 1.0) if _f(den, 1.0) else 0.0
            else:
                fps = _f(raw)
        elif kind == "audio":
            has_audio = True
            try:
                rate = max(rate, int(st.get("sample_rate") or 0))
            except (TypeError, ValueError):
                pass
    duration = v_dur or _f((data.get("format") or {}).get("duration"))
    if duration <= 0:
        return None
    if w <= 0 or h <= 0:
        # Audio-only file (a soundtrack). Still a valid probe.
        return {"w": 0, "h": 0, "duration": round(duration, 6), "fps": 0.0,
                "has_audio": has_audio, "sample_rate": rate}
    return {"w": w, "h": h, "duration": round(duration, 6),
            "fps": round(fps, 6), "has_audio": has_audio, "sample_rate": rate}


# ===========================================================================
# PART A — which seconds of this clip are the good ones
# ===========================================================================
def _decode_grey(path, width: int, height: int, fps: float) -> np.ndarray:
    """Decode the clip to a stack of small grey frames, `fps` samples/second.

    Sampling rather than decoding every frame is the whole performance story:
    a 5 s 1920x1080 clip is 124 frames, and 20 of them at 320 px answer the
    question just as well for a fiftieth of the work. `format=gray` means one
    byte per pixel, so the raw stream is parsed with a single reshape and no
    colour conversion of our own.
    """
    cmd = [str(FFMPEG), "-v", "error", "-nostdin", "-i", str(path),
           "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"fps={fps:g},scale={width}:{height}:flags=bicubic,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ffmpeg could not decode {Path(path).name}: {exc}")
    if out.returncode != 0:
        err = (out.stderr or b"").decode("utf-8", "replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg could not decode {Path(path).name}: {err}")
    stride = width * height
    n = len(out.stdout) // stride
    if n < 1:
        raise RuntimeError(f"{Path(path).name} decoded to no frames")
    buf = np.frombuffer(out.stdout[:n * stride], dtype=np.uint8)
    return buf.reshape(n, height, width)


def _frame_metrics(frames: np.ndarray) -> dict:
    """Per-sampled-frame numbers. Everything downstream is a mean of these."""
    f = frames.astype(np.float32)
    # 4-neighbour Laplacian, done with slices rather than a convolution import.
    lap = (4.0 * f[:, 1:-1, 1:-1]
           - f[:, :-2, 1:-1] - f[:, 2:, 1:-1]
           - f[:, 1:-1, :-2] - f[:, 1:-1, 2:])
    n = f.shape[0]
    sharp = lap.reshape(n, -1).var(axis=1)
    luma = f.reshape(n, -1).mean(axis=1)
    crushed = (frames <= CRUSH_LEVEL).reshape(n, -1).mean(axis=1)
    blown = (frames >= BLOWN_LEVEL).reshape(n, -1).mean(axis=1)
    if n > 1:
        motion = np.abs(np.diff(f, axis=0)).reshape(n - 1, -1).mean(axis=1)
        # motion[i] describes the step INTO frame i, so index 0 has none;
        # repeating the first real value keeps the arrays the same length
        # without inventing movement that was never measured.
        motion = np.concatenate([motion[:1], motion])
    else:
        motion = np.zeros(1, dtype=np.float32)
    return {"sharp": sharp.astype(np.float64), "luma": luma.astype(np.float64),
            "motion": motion.astype(np.float64),
            "crushed": crushed.astype(np.float64),
            "blown": blown.astype(np.float64)}


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below `lo`, 1 above `hi`, linear between. `hi` may be below `lo`."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    return float(min(1.0, max(0.0, t)))


def _motion_score(m: float) -> float:
    """A band, not a maximum: frozen is dead, thrashing is artefacting."""
    if m <= MOTION_DEAD or m >= MOTION_THRASH:
        return 0.0
    if m < MOTION_LOW:
        return _ramp(m, MOTION_DEAD, MOTION_LOW)
    if m > MOTION_HIGH:
        return 1.0 - _ramp(m, MOTION_HIGH, MOTION_THRASH)
    return 1.0


def _luma_score(mean_luma: float) -> float:
    if mean_luma <= LUMA_BLACK or mean_luma >= LUMA_WHITE:
        return 0.0
    if mean_luma < LUMA_GOOD_LO:
        return _ramp(mean_luma, LUMA_BLACK, LUMA_GOOD_LO)
    if mean_luma > LUMA_GOOD_HI:
        return 1.0 - _ramp(mean_luma, LUMA_GOOD_HI, LUMA_WHITE)
    return 1.0


def best_window(video_path, want_seconds: float, *, probe: dict | None = None,
                analysis_fps: float = ANALYSIS_FPS,
                head_guard: float = HEAD_GUARD_S,
                tail_guard: float = TAIL_GUARD_S,
                head_weight: float = HEAD_WEIGHT,
                tail_weight: float = TAIL_WEIGHT,
                weights: tuple[float, float, float] | None = None) -> dict:
    """Pick the best `want_seconds` inside one clip.

    Returns::

        {"start", "end", "score", "reason", "per_second", "components",
         "runner_up", "duration", "usable", "candidates"}

    `start`/`end` are seconds into the SOURCE clip and `end - start` is exactly
    `want_seconds` unless the clip is shorter than that, in which case the whole
    clip is returned and `reason` says so.

    The score is the weighted sum of three normalised terms (sharpness,
    stability, luma) minus the head/tail positional penalty. Sharpness is
    normalised against THIS clip's own 5th-95th percentile, so the number
    answers "which part of this clip" and must never be compared between
    clips. Stability and luma are absolute, because "frozen", "thrashing",
    "black" and "blown" mean the same thing everywhere.

    A window containing a near-black or blown-out stretch is vetoed rather
    than merely marked down; if EVERY window is vetoed the least-bad one is
    returned with `usable=False` so the caller can drop the shot instead of
    discovering the problem in the finished film.
    """
    w_sharp, w_stable, w_luma = weights or (W_SHARP, W_STABLE, W_LUMA)
    path = Path(str(video_path))
    info = probe or probe_media(path)
    if not info or _f(info.get("duration")) <= 0:
        raise RuntimeError(f"could not probe {path.name}")
    duration = float(info["duration"])
    src_w = int(info.get("w") or 0) or ANALYSIS_WIDTH
    src_h = int(info.get("h") or 0) or int(ANALYSIS_WIDTH * 9 / 16)

    aw = min(ANALYSIS_WIDTH, src_w - src_w % 2)
    ah = max(2, int(round(aw * src_h / src_w)))
    ah -= ah % 2
    frames = _decode_grey(path, aw, ah, analysis_fps)
    met = _frame_metrics(frames)
    n = frames.shape[0]
    step = 1.0 / analysis_fps
    times = np.arange(n) * step

    # Sharpness is normalised inside the clip. p05/p95 rather than min/max so
    # one glitched frame cannot define the scale.
    lo = float(np.percentile(met["sharp"], 5))
    hi = float(np.percentile(met["sharp"], 95))
    flat = (hi - lo) < 1e-6
    if flat:
        sharp_norm = np.full(n, 0.5)
    else:
        sharp_norm = np.clip((met["sharp"] - lo) / (hi - lo), 0.0, 1.0)

    per_second: list[dict] = []
    for sec in range(int(math.ceil(duration - 1e-9)) or 1):
        sel = (times >= sec) & (times < sec + 1)
        if not sel.any():
            continue
        per_second.append({
            "t": float(sec),
            "sharpness": round(float(met["sharp"][sel].mean()), 2),
            "sharpness_norm": round(float(sharp_norm[sel].mean()), 4),
            "motion": round(float(met["motion"][sel].mean()), 3),
            "luma": round(float(met["luma"][sel].mean()), 2),
            "crushed": round(float(met["crushed"][sel].mean()), 4),
            "blown": round(float(met["blown"][sel].mean()), 4),
        })

    # Asking for at least the whole clip is legal and common — the plan is
    # capacity-bound. Clamp rather than refuse, and let the normal path run so
    # the single candidate still gets a real score and real diagnostics; only
    # `forced` differs, and the reason line says so.
    want = float(want_seconds)
    if want <= 0 or want > duration:
        want = duration

    win = max(2, int(round(want * analysis_fps)))
    win = min(win, n)
    last_start = max(0, n - win)

    cands: list[dict] = []
    for i in range(last_start + 1):
        start = min(i * step, duration - want)
        start = max(0.0, round(start, 6))
        end = round(start + want, 6)
        sl = slice(i, i + win)
        m_sharp = float(sharp_norm[sl].mean())
        # motion[i] is the step INTO frame i, so the movement WITHIN the
        # window is measured from its second sample onwards.
        msl = slice(i + 1, i + win) if win > 1 else sl
        m_motion = float(met["motion"][msl].mean())
        peak_motion = float(met["motion"][msl].max())
        # The QUIETEST quarter of the window, not just its average. A window
        # that is two thirds frozen and one third moving averages into the
        # healthy band and looks fine to a mean — and plays as a still frame
        # that eventually twitches. Asking "was it alive throughout?" as well
        # as "was it alive on average?" is what separates those two.
        low_motion = float(np.percentile(met["motion"][msl], 25))
        m_luma = float(met["luma"][sl].mean())
        crushed = float(met["crushed"][sl].max())
        blown = float(met["blown"][sl].max())

        s_sharp = m_sharp
        s_stable = 0.5 * _motion_score(m_motion) + 0.5 * _motion_score(low_motion)
        # A single violent sample inside an otherwise calm window is a cut, a
        # flash or a glitch — worse than steady fast motion, so it is charged
        # separately rather than being averaged away.
        if peak_motion >= MOTION_THRASH and m_motion < MOTION_HIGH:
            s_stable *= 0.35
        s_luma = _luma_score(m_luma)

        head_over = max(0.0, min(end, head_guard) - start)
        tail_over = max(0.0, end - max(start, duration - tail_guard))
        pen = head_weight * (head_over / want) + tail_weight * (tail_over / want)

        vetoed = (m_luma <= LUMA_BLACK or m_luma >= LUMA_WHITE
                  or crushed >= CRUSH_VETO or blown >= BLOWN_VETO)
        score = w_sharp * s_sharp + w_stable * s_stable + w_luma * s_luma - pen
        cands.append({
            "start": start, "end": end,
            "score": round(score, 6), "vetoed": bool(vetoed),
            "components": {
                "sharpness": round(s_sharp, 4),
                "stability": round(s_stable, 4),
                "luma": round(s_luma, 4),
                "penalty": round(pen, 4),
                "motion_raw": round(m_motion, 3),
                "motion_low": round(low_motion, 3),
                "motion_peak": round(peak_motion, 3),
                "luma_raw": round(m_luma, 2),
                "crushed_max": round(crushed, 4),
                "blown_max": round(blown, 4),
            },
        })

    # Deterministic order: not vetoed first, then score, then the EARLIER
    # start. Never a dict iteration order, never a stable-sort accident.
    ranked = sorted(cands, key=lambda c: (c["vetoed"], -c["score"], c["start"]))
    best = ranked[0]
    runner = next((c for c in ranked[1:]
                   if abs(c["start"] - best["start"]) >= step - 1e-9), None)

    comp = best["components"]
    bits = [f"sharpness {comp['sharpness']:.2f}",
            f"stability {comp['stability']:.2f} ({comp['motion_raw']:.1f}/frame)",
            f"luma {comp['luma_raw']:.0f}"]
    if comp["penalty"] > 0.001:
        bits.append(f"head/tail cost {comp['penalty']:.3f}")
    reason = (f"{best['start']:.2f}-{best['end']:.2f}s scored "
              f"{best['score']:.3f} — " + ", ".join(bits))
    if last_start == 0:
        reason += (f"; the only window there is — {want:.2f}s asked of a "
                   f"{duration:.2f}s clip leaves nothing to choose")
    if flat:
        reason += "; sharpness was flat across the clip, so it did not decide"
    if runner:
        reason += (f"; beat {runner['start']:.2f}s "
                   f"({runner['score']:.3f}) by "
                   f"{best['score'] - runner['score']:.3f}")
    if best["vetoed"]:
        reason += ("; EVERY window failed the luma check (near-black or "
                   "blown) — this is the least bad one")

    return {
        "path": str(path),
        "start": best["start"], "end": best["end"],
        "score": best["score"], "usable": not best["vetoed"],
        "reason": reason, "components": comp,
        "runner_up": ({"start": runner["start"], "score": runner["score"]}
                      if runner else None),
        "per_second": per_second,
        "duration": round(duration, 6),
        "forced": last_start == 0,
        "candidates": len(cands),
    }


# ===========================================================================
# PART B — where the beats are
# ===========================================================================
def _decode_pcm(path, sr: int = AUDIO_SR,
                span: tuple[float, float] | None = None) -> np.ndarray:
    """Mono float32 in [-1, 1]. One ffmpeg call, one reshape, no resampler.

    `span` is (start, end) in seconds and is applied by ffmpeg, so analysing
    the first 90 s of an eight-minute track costs 90 s of decode, not 479.
    """
    cmd = [str(FFMPEG), "-v", "error", "-nostdin"]
    if span:
        cmd += ["-ss", f"{float(span[0]):.6f}"]
    cmd += ["-i", str(path)]
    if span:
        cmd += ["-t", f"{max(0.0, float(span[1]) - float(span[0])):.6f}"]
    cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ffmpeg could not decode audio: {exc}")
    if out.returncode != 0:
        err = (out.stderr or b"").decode("utf-8", "replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg could not decode audio: {err}")
    pcm = np.frombuffer(out.stdout[:len(out.stdout) // 2 * 2], dtype="<i2")
    if pcm.size == 0:
        raise RuntimeError(f"{Path(path).name} carries no audio")
    return (pcm.astype(np.float32) / 32768.0)


def _onset_envelopes(x: np.ndarray, sr: int, n_fft: int = N_FFT,
                     hop: int = HOP) -> tuple[np.ndarray, np.ndarray, float]:
    """Spectral flux, full band and kick band, at `sr / hop` frames a second.

    Chunked so the magnitude spectrogram of an eight-minute track never exists
    in memory all at once — only a 1-D envelope survives each chunk.
    """
    if x.size < n_fft * 2:
        raise RuntimeError("audio is too short to analyse")
    view = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    n_frames = view.shape[0]
    win = np.hanning(n_fft).astype(np.float32)
    low_bin = max(2, int(round(LOW_BAND_HZ / (sr / n_fft))) + 1)

    flux = np.zeros(n_frames, dtype=np.float32)
    flux_low = np.zeros(n_frames, dtype=np.float32)
    prev = None
    chunk = 2048
    for c0 in range(0, n_frames, chunk):
        blk = np.asarray(view[c0:c0 + chunk], dtype=np.float32) * win
        mag = np.abs(np.fft.rfft(blk, axis=1)).astype(np.float32)
        spec = np.log1p(10.0 * mag)
        if prev is None:
            d = np.diff(spec, axis=0, prepend=spec[:1])
        else:
            d = np.diff(spec, axis=0, prepend=prev[None, :])
        pos = np.maximum(d, 0.0)
        flux[c0:c0 + spec.shape[0]] = pos.sum(axis=1)
        flux_low[c0:c0 + spec.shape[0]] = pos[:, :low_bin].sum(axis=1)
        prev = spec[-1]
    frame_rate = sr / float(hop)

    def _rectify(env: np.ndarray) -> np.ndarray:
        k = max(1, int(round(0.5 * frame_rate)))
        local = np.convolve(env, np.ones(k, dtype=np.float32) / k, mode="same")
        out = np.maximum(env - local, 0.0)
        s = float(out.std())
        return (out / s) if s > 1e-9 else out

    return _rectify(flux), _rectify(flux_low), frame_rate


def _autocorr(env: np.ndarray) -> np.ndarray:
    e = env - env.mean()
    n = e.size
    size = 1 << int(math.ceil(math.log2(max(4, 2 * n))))
    spec = np.fft.rfft(e, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    ac /= np.arange(n, 0, -1)              # unbiased-ish
    peak = ac[0]
    return ac / peak if peak > 1e-12 else ac


def _grid_score(env: np.ndarray, frame_rate: float, period: float,
                phase: float) -> float:
    """Mean onset strength ON the grid, relative to the track's own mean.

    >1 means the grid sits on the onsets. This is what separates a real tempo
    from an autocorrelation artefact, and it is also what makes an octave
    error visible: a grid at double tempo lands half its beats on nothing.
    """
    if period <= 0:
        return 0.0
    n = env.size
    k = np.arange(int((n / frame_rate - phase) / period) + 1)
    idx = np.rint((phase + k * period) * frame_rate).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    if idx.size < 8:
        return 0.0
    base = float(env.mean())
    return float(env[idx].mean() / base) if base > 1e-9 else 0.0


def _phase_profile(env: np.ndarray, frame_rate: float, period: float,
                   nbins: int = 256) -> np.ndarray:
    """Mean onset strength as a function of phase, for ALL phases at once.

    Folding every sample into a phase bin is O(n) and answers the same
    question the naive sweep answers in O(phases x beats). It is what makes
    a two-hundred-step tempo refinement cost a tenth of a second instead of a
    minute — and the refinement is not optional: one autocorrelation lag is
    11.6 ms at this frame rate, which is 2.4 % of a 126 BPM beat, and 2.4 %
    of tempo error is a whole beat of drift after 40 seconds of film.
    """
    if period <= 0 or env.size == 0:
        return np.zeros(nbins)
    t = np.arange(env.size, dtype=np.float64) / frame_rate
    b = np.floor((np.mod(t, period) / period) * nbins).astype(np.int64) % nbins
    total = np.bincount(b, weights=env.astype(np.float64), minlength=nbins)
    count = np.bincount(b, minlength=nbins).astype(np.float64)
    count[count == 0] = 1.0
    return total / count


def _best_phase(env: np.ndarray, frame_rate: float, period: float,
                nbins: int = 256) -> tuple[float, float]:
    """Phase with the strongest onsets on it, then an exact local refinement."""
    prof = _phase_profile(env, frame_rate, period, nbins)
    ph = float(np.argmax(prof)) * period / nbins
    best, best_phase = _grid_score(env, frame_rate, period, ph), ph
    span = period / nbins
    for i in range(-12, 13):
        cand = (ph + span * i / 6.0) % period
        sc = _grid_score(env, frame_rate, period, cand)
        if sc > best + 1e-12:
            best, best_phase = sc, cand
    return float(best_phase % period), float(best)


def _refine_period(env: np.ndarray, frame_rate: float, period: float,
                   rel: float = 0.045, steps: int = 180) -> tuple[float, float, float]:
    """Local tempo sweep by grid score → (period, phase, score).

    Two passes: `rel` wide at `steps` resolution, then a tenth as wide. The
    grid score is extremely sensitive to tempo over a long span (drift
    accumulates), which is exactly what makes it a better final estimator than
    the autocorrelation peak that seeded it.
    """
    best = (-1.0, period, 0.0)
    for width, n in ((rel, steps), (rel / 10.0, steps // 2)):
        centre = best[1] if best[0] >= 0 else period
        for i in range(n + 1):
            p = centre * (1.0 - width) + centre * 2.0 * width * i / n
            if p <= 0:
                continue
            ph, sc = _best_phase(env, frame_rate, p)
            if sc > best[0] + 1e-12:
                best = (sc, p, ph)
    return best[1], best[2], best[0]


def _strong_onsets(env: np.ndarray, frame_rate: float,
                   count: int = 400) -> np.ndarray:
    idx = [i for i in range(1, env.size - 1)
           if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] > 0]
    idx.sort(key=lambda i: (-float(env[i]), i))
    return np.asarray(sorted(idx[:count]), dtype=np.float64) / frame_rate


def _phase_coherence(onsets: np.ndarray, period: float, phase: float) -> float:
    """Rayleigh vector of onset PEAK TIMES against the grid, 0…1.

    Reported, never scored. On dense percussive material it is close to zero
    even when the grid is perfect, because sixteenth-note peaks sit at four
    equally spaced phases and their unit vectors cancel exactly. It is kept
    because it is the right statistic for sparse material and because a
    reader who knows the difference should be able to see the number.
    `grid_lock_ms` below is the statistic to trust here.
    """
    if onsets.size < 8 or period <= 0:
        return 0.0
    ang = 2.0 * math.pi * ((onsets - phase) / period)
    return float(abs(np.mean(np.exp(1j * ang))))


def _grid_lock_ms(env: np.ndarray, frame_rate: float, period: float,
                  phase: float, nbins: int = 256) -> float:
    """Half-width at half-maximum of the onset-energy peak, in milliseconds.

    THE honest accuracy number. Fold the onset envelope by the period and you
    get energy as a function of phase; if the grid is real that curve has a
    spike at the beat, and its half-width says how tightly the music is
    actually locked to the grid we are about to cut on. A width approaching
    an eighth of a beat means there is no spike and the "grid" is decoration.
    """
    prof = _phase_profile(env, frame_rate, period, nbins)
    if prof.size == 0 or float(prof.max()) <= 0:
        return float("nan")
    top = int(np.argmax(prof))
    prof = np.roll(prof, -top)
    half = (float(prof[0]) + float(np.median(prof))) / 2.0
    right = 0
    while right < nbins // 2 and prof[right] > half:
        right += 1
    left = 0
    while left < nbins // 2 and prof[(-left) % nbins] > half:
        left += 1
    return float((right + left) / 2.0 * period / nbins * 1000.0)


def _tempo_prior(bpm: float) -> float:
    return float(math.exp(-0.5 * (math.log2(bpm / BPM_PRIOR_CENTER)
                                  / BPM_PRIOR_OCTAVES) ** 2))


def beat_map(audio_path, *, meter: int = DEFAULT_METER, sr: int = AUDIO_SR,
             span: tuple[float, float] | None = None) -> dict:
    """Tempo, beat grid and downbeats for a soundtrack.

    Returns::

        {"bpm", "beats": [t…], "downbeats": [t…], "confidence",
         "period", "phase", "meter", "span", "duration", "diagnostics"}

    `span` is (start, end) in seconds and limits BOTH the analysis and the
    beats returned. That is not a performance shortcut, it is a correctness
    one: the model here is ONE constant tempo, and real tracks — generated
    ones especially — drift. Fitting a single BPM across eight minutes buys a
    grid that is wrong at both ends; fitting it across the ninety seconds you
    are actually cutting buys one that is right where it is used. Beats are
    never extrapolated past the fitted span, so a caller that asks for a cut
    beyond it gets "no beat here" rather than a confident wrong answer.

    `confidence` is built from four independent pieces of evidence, all of
    which are in `diagnostics`:

      * `pulse_ratio` — onset strength on the grid over onset strength
        everywhere. 1.0 means the grid is no better than chance.
      * `acf_salience` — how far the winning autocorrelation lag stands above
        the rest of the searched range, in standard deviations.
      * `octave_margin` — how far the winning tempo beat its own half, double
        and two-thirds. A small margin is the classic 63-vs-126 ambiguity and
        it is reported, not hidden.
      * `grid_lock_ms` — the half-width of the onset-energy spike at the
        beat. Ten milliseconds means the music really is locked to this grid;
        a width approaching an eighth of a beat means it is decoration.

    And `tempo_drift_bpm` compares the first half of the span with the second.
    A non-zero drift is the constant-tempo model admitting its own limit.
    """
    t0 = float(span[0]) if span else 0.0
    x = _decode_pcm(audio_path, sr, span)
    duration = x.size / float(sr)
    env, env_low, frame_rate = _onset_envelopes(x, sr)

    ac = _autocorr(env)
    lag_lo = max(2, int(math.floor(60.0 / BPM_MAX * frame_rate)))
    lag_hi = min(ac.size - 2, int(math.ceil(60.0 / BPM_MIN * frame_rate)))
    if lag_hi <= lag_lo:
        raise RuntimeError("audio is too short for a tempo estimate")
    band = ac[lag_lo:lag_hi + 1]

    peaks = [i for i in range(1, band.size - 1)
             if band[i] >= band[i - 1] and band[i] > band[i + 1]]
    if not peaks:
        peaks = [int(np.argmax(band))]
    peaks.sort(key=lambda i: (-float(band[i]), i))

    # Hypotheses: the strongest autocorrelation peaks, plus each one's octave
    # and triplet relatives. The relatives matter because a 3:2 confusion
    # (126 heard as 84) is the failure this corpus actually produces, and the
    # autocorrelation often carries only one of the pair.
    hypotheses: list[float] = []
    for i in peaks[:6]:
        base = (lag_lo + i) / frame_rate
        for mult in (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0):
            p = base * mult
            bpm = 60.0 / p
            if BPM_MIN <= bpm <= BPM_MAX and all(abs(p - q) > 1e-3
                                                 for q in hypotheses):
                hypotheses.append(p)

    onsets = _strong_onsets(env, frame_rate)
    scored = []
    for p0 in hypotheses:
        period, phase, grid = _refine_period(env, frame_rate, p0)
        bpm = 60.0 / period
        if not (BPM_MIN <= bpm <= BPM_MAX):
            continue
        prior = _tempo_prior(bpm)
        # Onset energy ON the grid, times a log-normal tempo prior. The prior
        # is what stops the half-tempo grid from winning by explaining fewer
        # onsets more strongly; the refinement above is what makes the grid
        # score trustworthy enough to be compared at all.
        scored.append({"period": period, "bpm": bpm, "phase": phase,
                       "grid": grid, "prior": prior,
                       "coherence": _phase_coherence(onsets, period, phase),
                       "final": grid * prior})
    if not scored:
        raise RuntimeError("no usable tempo hypothesis")
    scored.sort(key=lambda s: (-s["final"], -s["bpm"]))
    win = scored[0]
    period, phase = win["period"], win["phase"]

    # Evidence.
    li = int(round(period * frame_rate))
    acf_at = float(ac[li]) if lag_lo <= li <= lag_hi else float(band.max())
    salience = (acf_at - float(band.mean())) / (float(band.std()) + 1e-9)
    rivals = [s["final"] for s in scored[1:]
              if abs(math.log2(s["bpm"] / win["bpm"])) > 0.25]
    octave_margin = (1.0 - max(rivals) / win["final"]
                     if rivals and win["final"] > 1e-9 else 1.0)

    # Drift: the constant-tempo model checking itself. Each half is refined
    # around the winner, narrowly, so this measures drift and not a re-run of
    # the octave argument.
    half = env.size // 2
    drift = None
    if half > int(20 * frame_rate):
        p_a, _, _ = _refine_period(env[:half], frame_rate, period,
                                   rel=0.02, steps=80)
        p_b, _, _ = _refine_period(env[half:], frame_rate, period,
                                   rel=0.02, steps=80)
        drift = round(60.0 / p_b - 60.0 / p_a, 3)

    # Frame f of the envelope describes samples [f*hop, f*hop + n_fft), and is
    # timestamped at f/frame_rate — the window's START. A click at time T
    # therefore raises the flux at roughly T - n_fft/sr, so the grid computed
    # above sits systematically EARLY by about half a window once the Hann
    # taper is accounted for. Everything internal is consistent in that frame,
    # so the correction belongs here, once, on the times that leave: shift the
    # phase to the window's centre. Measured residual on a synthetic click
    # track after this correction: about 10 ms, a quarter of a video frame at
    # 24 fps.
    lead = N_FFT / (2.0 * sr)
    n_beats = int((duration - phase) / period) + 1
    local = [phase + k * period for k in range(max(0, n_beats))]
    beats = [round(t0 + b + lead, 6) for b in local
             if b + lead <= duration + 1e-9]

    # Downbeat phase: which of the `meter` beat classes carries the kick.
    downbeat_offset, best_low = 0, -1.0
    idx_all = np.rint(np.asarray(local[:len(beats)]) * frame_rate).astype(np.int64)
    idx_all = idx_all[(idx_all >= 0) & (idx_all < env_low.size)]
    for off in range(meter):
        sel = idx_all[off::meter]
        if sel.size < 4:
            continue
        v = float(env_low[sel].mean())
        if v > best_low + 1e-12:
            best_low, downbeat_offset = v, off
    downbeats = beats[downbeat_offset::meter]

    lock_ms = _grid_lock_ms(env, frame_rate, period, phase)
    pulse_ratio = float(win["grid"])
    c_pulse = min(1.0, max(0.0, (pulse_ratio - 1.0) / 2.0))
    c_sal = 1.0 - math.exp(-max(0.0, salience) / 3.0)
    c_oct = min(1.0, max(0.0, octave_margin))
    # An eighth of a beat is the width at which "on the grid" stops meaning
    # anything, so that is the scale the lock is measured against.
    c_lock = (1.0 - min(1.0, lock_ms / (period * 1000.0 / 8.0))
              if lock_ms == lock_ms else 0.0)
    confidence = round(0.3 * c_pulse + 0.25 * c_sal + 0.2 * c_oct
                       + 0.25 * c_lock, 4)

    return {
        "bpm": round(60.0 / period, 4),
        "period": round(period, 8),
        "phase": round(t0 + phase + lead, 6),
        "meter": meter,
        "beats": beats,
        "downbeats": downbeats,
        "confidence": confidence,
        "span": [round(t0, 4), round(t0 + duration, 4)],
        "duration": round(duration, 4),
        "diagnostics": {
            "pulse_ratio": round(pulse_ratio, 4),
            "acf_salience": round(float(salience), 4),
            "octave_margin": round(float(octave_margin), 4),
            "grid_lock_ms": round(lock_ms, 2) if lock_ms == lock_ms else None,
            "coherence": round(float(win["coherence"]), 4),
            "tempo_drift_bpm": drift,
            "beat_ms": round(period * 1000.0, 2),
            "n_beats": len(beats),
            "n_downbeats": len(downbeats),
            "downbeat_offset": downbeat_offset,
            "frame_rate": round(frame_rate, 4),
            "n_onsets": int(onsets.size),
            "runners_up": [{"bpm": round(s["bpm"], 3),
                            "grid": round(s["grid"], 4),
                            "coherence": round(s["coherence"], 4),
                            "final": round(s["final"], 4)}
                           for s in scored[1:4]],
        },
    }


# ---------------------------------------------------------------------------
# THE SONG MAP — sections and energy, ON THE BAR GRID
# ---------------------------------------------------------------------------
# A beat grid says WHERE to cut; it says nothing about how OFTEN. A music
# video that cuts every two bars through the intro and every two bars through
# the chorus reads as a metronome, and the Director had no way to know which
# was which. This is the second axis: for every bar of the fitted grid, how
# loud and how bright the track is, and from those a list of SECTIONS with a
# label (intro / verse / chorus / bridge / outro) and an energy in 0..1.
#
# ALIGNED TO DOWNBEATS, deliberately. A section boundary that lands between
# two bars is a boundary no cut can sit on; snapping the analysis to the bar
# grid means every section edge IS a downbeat and the planner's slots can
# begin exactly there. Labels are a heuristic — position plus relative energy
# and brightness — and are reported as such: `energy` is the number the
# pacing is derived from, the label is the word the planner is told.
SECTION_LABELS = ("intro", "verse", "chorus", "bridge", "outro")
SECTION_MIN_BARS = 4           # shorter than a phrase is a fill, not a section
SECTION_STEP = 0.15            # the jump in combined level that opens one


def _bar_features(x: np.ndarray, sr: int, downbeats: list[float],
                  t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar RMS and spectral centroid, both normalised to 0..1."""
    edges = [max(0.0, float(b) - t0) for b in downbeats if t0 <= float(b) < t0 + duration]
    if not edges:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    edges.append(min(duration, edges[-1] + (edges[-1] - edges[-2] if len(edges) > 1 else duration)))
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    rms, cent = [], []
    for a, b in zip(edges, edges[1:]):
        i0, i1 = int(a * sr), max(int(a * sr) + n_fft, int(b * sr))
        seg = x[i0:i1]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        rms.append(float(np.sqrt(np.mean(seg.astype(np.float32) ** 2))))
        # A handful of windows across the bar is all a centroid needs; a full
        # spectrogram per bar would be a megabyte to learn one number.
        steps = max(1, min(8, seg.size // n_fft))
        cs = []
        for k in range(steps):
            j = k * (seg.size - n_fft) // max(1, steps - 1) if steps > 1 else 0
            mag = np.abs(np.fft.rfft(seg[j:j + n_fft] * np.hanning(n_fft)))
            tot = float(mag.sum())
            cs.append(float((mag * freqs).sum() / tot) if tot > 1e-9 else 0.0)
        cent.append(float(np.mean(cs)))
    rms = np.asarray(rms, dtype=np.float32)
    cent = np.asarray(cent, dtype=np.float32)
    rms = rms / (float(rms.max()) + 1e-8)
    cent = cent / (float(cent.max()) + 1e-8)
    return rms, cent


def _label_section(energy: float, brightness: float, mean_energy: float,
                   idx: int, total: int) -> str:
    first, last = idx == 0, idx == total - 1
    high = energy > mean_energy * 1.2
    low = energy < mean_energy * 0.6
    if first and low:
        return "intro"
    if last and low:
        return "outro"
    if high and brightness > 0.5:
        return "chorus"
    if low:
        return "bridge"
    return "verse"


def song_map(audio_path, beats: dict | None = None, *, sr: int = AUDIO_SR,
             span: tuple[float, float] | None = None,
             min_bars: int = SECTION_MIN_BARS, step: float = SECTION_STEP) -> dict:
    """Sections and energy for a soundtrack, on its own bar grid.

    Returns::

        {"bpm", "bars": [{"n", "start", "end", "energy", "brightness"}],
         "sections": [{"start", "end", "label", "energy", "bars": [i, j)}],
         "mean_energy", "duration"}

    `beats` is a `beat_map()` result to reuse (the Director already has one);
    absent, it is computed here. A track with fewer than `2 * min_bars` bars
    is ONE section labelled by its own energy, never an error — a jingle is
    still cuttable.
    """
    b = beats if isinstance(beats, dict) and beats.get("downbeats") else beat_map(audio_path, sr=sr, span=span)
    t0 = float((b.get("span") or [0.0])[0])
    duration = float(b.get("duration") or 0.0)
    x = _decode_pcm(audio_path, sr=sr, span=(t0, t0 + duration) if duration > 0 else span)
    if duration <= 0:
        duration = x.size / float(sr)
    downs = [float(t) for t in (b.get("downbeats") or [])]
    rms, cent = _bar_features(x, sr, downs, t0, duration)
    n = int(rms.size)
    bars = []
    for i in range(n):
        start = float(downs[i])
        end = float(downs[i + 1]) if i + 1 < len(downs) else round(t0 + duration, 6)
        bars.append({"n": i + 1, "start": round(start, 6), "end": round(end, 6),
                     "energy": round(float(rms[i]), 4),
                     "brightness": round(float(cent[i]), 4)})
    out = {"bpm": b.get("bpm"), "bars": bars, "sections": [],
           "mean_energy": round(float(rms.mean()), 4) if n else 0.0,
           "duration": round(duration, 4)}
    if n == 0:
        return out
    combined = 0.7 * rms + 0.3 * cent
    bounds = [0]
    if n >= 2 * min_bars:
        # THE JUMP BETWEEN THE LAST `min_bars` AND THE NEXT `min_bars`, at
        # every bar, and a boundary only where that jump PEAKS: a window that
        # straddles a real edge crosses the threshold for several bars in a
        # row, and taking the first crossing put the boundary a phrase early
        # (and a second one a phrase late).
        diff = np.zeros(n + 1, dtype=np.float32)
        for i in range(min_bars, n - min_bars + 1):
            left = float(combined[i - min_bars:i].mean())
            right = float(combined[i:i + min_bars].mean())
            diff[i] = abs(right - left)
        for i in range(min_bars, n - min_bars + 1):
            if diff[i] <= step or i - bounds[-1] < min_bars:
                continue
            lo, hi = max(0, i - min_bars + 1), min(n + 1, i + min_bars)
            if diff[i] >= float(diff[lo:hi].max()) - 1e-9:
                bounds.append(i)
    bounds.append(n)
    mean_e = float(rms.mean())
    total = len(bounds) - 1
    for k in range(total):
        i, j = bounds[k], bounds[k + 1]
        e = float(rms[i:j].mean())
        br = float(cent[i:j].mean())
        out["sections"].append({
            "start": bars[i]["start"], "end": bars[j - 1]["end"],
            "label": _label_section(e, br, mean_e, k, total),
            "energy": round(e, 4), "brightness": round(br, 4),
            "bars": [i, j],
        })
    return out


# HOW OFTEN TO CUT, PER SECTION. `base` is the user's bars-per-shot; the
# chorus cuts twice as often, the intro and the outro half as often, and a
# bridge holds the base. The energy number breaks ties for an unlabelled
# section: hotter than the track's mean cuts faster. Never below one bar.
def director_pacing(label: str, energy: float, mean_energy: float,
                    base: int = 2) -> int:
    base = max(1, int(base))
    lab = str(label or "").lower()
    if lab == "chorus":
        return max(1, base // 2)
    if lab in ("intro", "outro"):
        return base * 2
    if lab == "bridge":
        return base
    if mean_energy > 0 and energy > mean_energy * 1.2:
        return max(1, base // 2)
    return base


def audio_span_for(target_seconds: float | None) -> tuple[float, float] | None:
    """The stretch of music a film of `target_seconds` should be fitted to.

    One definition, used by `plan_cut()` and by the CLI, so a plan built two
    different ways is built against the same grid. The 1.5x + 10 s margin
    covers a cut that overruns its target and keeps the grid from ending
    mid-film; it is not tuned, it is simply generous.
    """
    if not target_seconds or target_seconds <= 0:
        return None
    return (0.0, float(target_seconds) * 1.5 + 10.0)


def snap(t: float, beats, max_shift: float) -> float:
    """Nearest beat within `max_shift`, else `t` unchanged.

    Bisection, not a scan — `plan_cut()` calls this once per candidate cut
    against a thousand-beat grid, and a linear search there is the difference
    between instant and noticeable.
    """
    t = float(t)
    if not beats or max_shift is None or max_shift <= 0:
        return t
    arr = list(beats)
    i = bisect.bisect_left(arr, t)
    best, dist = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(arr):
            d = abs(arr[j] - t)
            # `<` keeps the EARLIER beat when a cut falls exactly between two.
            if dist is None or d < dist - 1e-12:
                best, dist = float(arr[j]), d
    # Inclusive of its own edge, with a float-comparison margin: a tolerance
    # of exactly half a beat has to accept a cut exactly half a beat away, or
    # the caller's arithmetic and this function's disagree at the boundary.
    if best is None or dist > float(max_shift) + 1e-9:
        return t
    return best


# ===========================================================================
# PART C — the cut plan
# ===========================================================================
def _snap_in_range(t: float, grid, tol: float,
                   lo: float, hi: float) -> float | None:
    """Closest grid point to `t` that is ALSO inside [lo, hi]. None if there
    is none.

    This is `snap()` with the planner's constraints folded in, and it is a
    different function on purpose. `snap()` looks at the nearest beat and
    stops — which is the right public contract, and exactly wrong here: the
    nearest downbeat to a wanted cut is very often one that would make the
    shot longer than the clip, and answering "no beat then" would leave a cut
    unsnapped while a perfectly good beat sat 200 ms away. So this walks
    outward from `t` and returns the first candidate that satisfies both the
    tolerance and the shot-length limits. Ties break EARLIER, so the plan is
    the same every time.
    """
    if grid is None or tol is None or tol <= 0 or hi < lo:
        return None
    arr = list(grid)
    if not arr:
        return None
    left = max(lo, t - float(tol))
    right = min(hi, t + float(tol))
    if right < left:
        return None
    i = bisect.bisect_left(arr, left - 1e-9)
    best, dist = None, None
    while i < len(arr) and arr[i] <= right + 1e-9:
        d = abs(arr[i] - t)
        if dist is None or d < dist - 1e-12:
            best, dist = float(arr[i]), d
        i += 1
    return best


def _balance(avail: list[float], target: float, min_shot: float) -> list[float]:
    """Split `target` over the clips without asking any clip for more than it
    has, and without producing a shot below `min_shot`.

    Water-filling: scale everything to the target, clamp to
    [min_shot, available], hand the deficit or surplus back to the clips that
    still have room, repeat. Bounded to eight passes so it always terminates.
    """
    n = len(avail)
    if n == 0:
        return []
    want = [min(a, target / n) for a in avail]
    for _ in range(8):
        total = sum(want)
        if abs(total - target) < 1e-6:
            break
        room = [(avail[i] - want[i]) if total < target
                else (want[i] - min_shot) for i in range(n)]
        room = [max(0.0, r) for r in room]
        pool = sum(room)
        if pool <= 1e-9:
            break
        delta = target - total
        for i in range(n):
            want[i] += delta * (room[i] / pool)
        want = [min(avail[i], max(min(min_shot, avail[i]), want[i]))
                for i in range(n)]
    return want


def plan_cut(clips, *, audio=None, target_seconds: float | None = None,
             min_shot: float = 1.5, max_shot: float | None = None,
             meter: int = DEFAULT_METER,
             downbeat_tolerance: float | None = None,
             beat_tolerance: float | None = None,
             min_confidence: float = 0.0,
             audio_span: tuple[float, float] | None = None,
             window_kwargs: dict | None = None,
             probes: dict | None = None) -> list[dict]:
    """Turn a list of clips into a cut a human can read and ffmpeg can execute.

    `clips` is paths (str/Path) or dicts carrying at least `path`. `audio` is a
    soundtrack path, an already-computed `beat_map()` result, or None. The
    returned list is one entry per clip, IN THE ORDER GIVEN — this plans the
    edit, it does not reorder the film:

        {"n", "path", "start", "end", "duration",        # what to play
         "film_start", "film_end",                       # where it lands
         "snap": {"kind", "wanted", "landed", "shift_ms"},
         "window": {"score", "reason", "components", "runner_up", "usable"},
         "notes": [...]}

    Boundaries prefer downbeats, fall back to beats, and fall back again to the
    unsnapped time — and every one of those says which it was and by how many
    milliseconds it moved, because "beat-aligned" is a claim, and a claim needs
    a number next to it.

    `min_shot` is never violated by the snapping: a candidate cut that would
    make the shot too short is not a candidate. The one exception is a source
    clip shorter than `min_shot`, which is used whole and flagged in `notes`.

    When `audio` is a PATH and `target_seconds` is known, the beat grid is
    fitted to the stretch of music the film will actually cover (plus a
    margin), not to the whole track — see `beat_map`'s `span`. A single
    constant tempo fitted across eight minutes of a generated track is a
    compromise at both ends; fitted across the ninety seconds being cut it is
    right where it is used. Pass `audio_span` to override, or hand in a
    pre-computed beat map to take full control.
    """
    items: list[dict] = []
    for c in clips:
        if isinstance(c, dict):
            items.append({"path": str(c.get("path")), "probe": c.get("probe")})
        else:
            items.append({"path": str(c), "probe": None})
    if not items:
        return []

    for it in items:
        if not it["probe"] and probes:
            it["probe"] = probes.get(it["path"])
        if not it["probe"]:
            it["probe"] = probe_media(it["path"])
        if not it["probe"]:
            raise RuntimeError(f"could not probe {Path(it['path']).name}")

    # ---- the music ------------------------------------------------------
    bm: dict | None = None
    music_note: list[str] = []
    if audio is not None:
        if isinstance(audio, dict):
            bm = audio
        else:
            span = audio_span or audio_span_for(target_seconds)
            bm = beat_map(audio, meter=meter, span=span)
        if bm.get("confidence", 0.0) < min_confidence:
            music_note.append(
                f"beat confidence {bm.get('confidence'):.2f} is below the "
                f"{min_confidence:.2f} the caller asked for — cutting "
                f"unsnapped rather than cutting to a grid we do not trust")
            bm = None
    period = float(bm["period"]) if bm else 0.0
    bar = period * meter if bm else 0.0
    if bm:
        if downbeat_tolerance is None:
            downbeat_tolerance = bar / 2.0
        if beat_tolerance is None:
            # A WHOLE beat, not half of one. Half a beat is the tolerance you
            # want when a cut can land anywhere; here it cannot — the shot may
            # not outrun its clip — and the nearest beat under that ceiling is
            # frequently just past half a beat away. With the tight tolerance
            # those cuts fell through to "unsnapped", and a film that lands
            # every other cut on the grid reads worse than one that lands all
            # of them a beat early. One beat is also the worst case for shot
            # length: 474 ms off a five-second shot.
            beat_tolerance = period

    # ---- how long each shot wants to be ---------------------------------
    durations = [float(it["probe"]["duration"]) for it in items]
    avail = [min(d, max_shot) if max_shot else d for d in durations]
    if target_seconds and target_seconds > 0:
        want = _balance(avail, float(target_seconds), min_shot)
    else:
        want = list(avail)

    # ---- walk the timeline, placing every cut ---------------------------
    plan: list[dict] = []
    cursor = 0.0
    for i, it in enumerate(items):
        notes: list[str] = list(music_note) if i == 0 else []
        src_dur = durations[i]
        lo = min_shot
        hi = avail[i]
        if hi < lo:
            notes.append(f"source is only {src_dur:.2f}s — shorter than the "
                         f"{min_shot:.2f}s minimum, so it is used whole")
            lo = hi
        wanted_end = cursor + min(hi, max(lo, want[i]))

        kind, landed = "none", wanted_end
        if bm:
            cand = _snap_in_range(wanted_end, bm["downbeats"],
                                  downbeat_tolerance,
                                  cursor + lo, cursor + hi)
            if cand is not None:
                kind, landed = "downbeat", cand
            else:
                cand = _snap_in_range(wanted_end, bm["beats"], beat_tolerance,
                                      cursor + lo, cursor + hi)
                if cand is not None:
                    kind, landed = "beat", cand
                else:
                    notes.append("no beat inside tolerance that also respects "
                                 "the shot-length limits — left unsnapped")
        shot = max(lo, min(hi, landed - cursor))
        landed = cursor + shot
        shot = round(shot, 6)

        win = best_window(it["path"], shot, probe=it["probe"],
                          **(window_kwargs or {}))
        # best_window returns the whole clip when there is nothing to choose;
        # honour the plan's length rather than the clip's in that case.
        start = float(win["start"])
        if win.get("forced") and (win["end"] - win["start"]) > shot:
            start = min(start, max(0.0, src_dur - shot))
        end = round(start + shot, 6)
        if end > src_dur + 1e-6:
            start = max(0.0, round(src_dur - shot, 6))
            end = round(start + shot, 6)
        if not win.get("usable", True):
            notes.append("every window in this clip failed the luma check")

        plan.append({
            "n": i + 1,
            "path": it["path"],
            # Free — the probe that read this clip's geometry read its streams
            # too. The timeline's audio lane needs it to tell a clip with no
            # sound from one whose sound is simply linked to its picture.
            "has_audio": bool((it.get("probe") or {}).get("has_audio")),
            "start": round(start, 6),
            "end": round(end, 6),
            "duration": shot,
            "film_start": round(cursor, 6),
            "film_end": round(landed, 6),
            "snap": {"kind": kind,
                     "wanted": round(wanted_end, 6),
                     "landed": round(landed, 6),
                     "shift_ms": round((landed - wanted_end) * 1000.0, 2)},
            "window": {"score": win["score"], "reason": win["reason"],
                       "components": win["components"],
                       "runner_up": win["runner_up"],
                       "usable": win.get("usable", True),
                       "source_duration": round(src_dur, 6),
                       "slack": round(max(0.0, src_dur - shot), 6)},
            "per_second": win.get("per_second", []),
            "notes": notes,
        })
        cursor = landed
    return plan


def plan_total(plan: list[dict]) -> float:
    return round(sum(float(p["duration"]) for p in plan), 6)


def format_plan(plan: list[dict], beats: dict | None = None) -> str:
    """The plan as text, for a manifest or a terminal. No colour, no cleverness."""
    out: list[str] = []
    if beats:
        d = beats.get("diagnostics", {})
        out.append(f"Music: {beats['bpm']:.2f} BPM, {beats['meter']}/4, "
                   f"beat {d.get('beat_ms')} ms, confidence "
                   f"{beats['confidence']:.2f} "
                   f"(pulse {d.get('pulse_ratio')}, salience "
                   f"{d.get('acf_salience')}, octave margin "
                   f"{d.get('octave_margin')}, onset energy locked to "
                   f"+/-{d.get('grid_lock_ms')} ms)")
    out.append(f"{len(plan)} shots, {plan_total(plan):.3f} s total")
    out.append("")
    out.append(f"{'#':>2}  {'shot':<34} {'window':<16} {'len':>6} "
               f"{'at':>7}  {'snap':<9} {'shift':>8}  score")
    for p in plan:
        name = Path(p["path"]).name
        if len(name) > 34:
            name = name[:31] + "..."
        out.append(f"{p['n']:>2}  {name:<34} "
                   f"{p['start']:>6.2f}-{p['end']:<6.2f}  "
                   f"{p['duration']:>6.3f} {p['film_start']:>7.3f}  "
                   f"{p['snap']['kind']:<9} "
                   f"{p['snap']['shift_ms']:>7.0f}ms  "
                   f"{p['window']['score']:.3f}")
    out.append("")
    for p in plan:
        out.append(f"S{p['n']:02d} {Path(p['path']).name}")
        out.append(f"     {p['window']['reason']}")
        for note in p["notes"]:
            out.append(f"     NOTE: {note}")
    return "\n".join(out)


# ===========================================================================
# CLI — the same three questions, from a terminal
# ===========================================================================
def _main(argv: list[str]) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("window", help="best window inside one clip")
    w.add_argument("clip")
    w.add_argument("seconds", type=float)
    w.add_argument("--json", action="store_true")

    b = sub.add_parser("beats", help="tempo + beat grid for a soundtrack")
    b.add_argument("audio")
    b.add_argument("--span", type=float,
                   help="analyse only the first N seconds (see beat_map)")
    b.add_argument("--json", action="store_true")

    p = sub.add_parser("plan", help="a cut plan for a list of clips")
    p.add_argument("clips", nargs="+")
    p.add_argument("--audio")
    p.add_argument("--span", type=float,
                   help="fit the beat grid to the first N seconds only")
    p.add_argument("--target", type=float)
    p.add_argument("--min-shot", type=float, default=1.5)
    p.add_argument("--max-shot", type=float)
    p.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "window":
        res = best_window(a.clip, a.seconds)
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"{Path(a.clip).name}: {res['start']:.3f}-{res['end']:.3f}s")
            print(f"  {res['reason']}")
        return 0
    if a.cmd == "beats":
        res = beat_map(a.audio, span=(0.0, a.span) if a.span else None)
        if a.json:
            slim = dict(res)
            slim["beats"] = res["beats"][:16]
            slim["downbeats"] = res["downbeats"][:8]
            print(json.dumps(slim, indent=2))
        else:
            d = res["diagnostics"]
            print(f"{res['bpm']:.2f} BPM  confidence {res['confidence']:.2f}  "
                  f"{d['n_beats']} beats, {d['n_downbeats']} downbeats")
            print(f"  pulse {d['pulse_ratio']}  salience {d['acf_salience']}  "
                  f"octave margin {d['octave_margin']}  "
                  f"lock +/-{d['grid_lock_ms']} ms")
            print(f"  onset energy peaks within {d['grid_lock_ms']} ms of the grid"
                  + (f", tempo drifts {d['tempo_drift_bpm']:+.2f} BPM across "
                     f"the analysed span" if d.get("tempo_drift_bpm") else ""))
        return 0
    bm = (beat_map(a.audio, span=((0.0, a.span) if a.span
                                  else audio_span_for(a.target)))
          if a.audio else None)
    plan = plan_cut(a.clips, audio=bm, target_seconds=a.target,
                    min_shot=a.min_shot, max_shot=a.max_shot)
    if a.json:
        print(json.dumps(plan, indent=2))
    else:
        print(format_plan(plan, bm))
    return 0


if __name__ == "__main__":
    import sys  # noqa: PLC0415
    raise SystemExit(_main(sys.argv[1:]))
