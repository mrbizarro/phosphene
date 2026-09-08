#!/usr/bin/env python3
"""Who is talking, per beat, for a two-shot with one character on each side.

Face motion = mean dense optical flow magnitude inside the upper-middle band
of the LEFT half and of the RIGHT half of the frame (where the two heads sit
in a seated two-shot); speech = audio RMS. Per 5-second beat: the two face
motions during the frames where speech is present, their ratio, and a
verdict — "left", "right", or "both" (ratio under 1.4 = the two faces move
alike while someone speaks: unison, or the wrong mouth). Crude, but it turns
"sometimes they both talk" into a table.

usage: speaker_check.py <mp4> [beat_seconds=5]

Calibration (2026-09-07): on the first aliens take it reproduced what the
owner heard (beat 3 "right" when the script said left; beats 4-6 "both").
It is a heuristic: it measures the whole head, not the mouth, so a nod or
the push-in on the larger head reads as speech. Use it to flag a beat for
listening, not to grade a clip on its own.
"""
import sys, subprocess, json
import numpy as np, cv2

def _ffmpeg() -> str:
    """The same ladder the panel climbs: LTX_FFMPEG, then PATH, then the Pinokio
    build, then Homebrew. Never a hardcoded home directory — this script ships."""
    import os, shutil
    from pathlib import Path
    for c in (os.environ.get("LTX_FFMPEG"), shutil.which("ffmpeg"),
              str(Path.home() / "pinokio/bin/ffmpeg-env/bin/ffmpeg"),
              "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if c and Path(c).exists():
            return c
    return "ffmpeg"


FF = _ffmpeg()
path = sys.argv[1]; beat = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
prev = None; left = []; right = []
while True:
    ok, fr = cap.read()
    if not ok: break
    h, w = fr.shape[:2]; sc = 320.0 / w
    g = cv2.cvtColor(cv2.resize(fr, (320, int(h * sc))), cv2.COLOR_BGR2GRAY)
    if prev is not None:
        fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.hypot(fl[..., 0], fl[..., 1]); H, W = mag.shape
        band = mag[int(H * 0.15):int(H * 0.65)]                  # heads, not the table
        left.append(float(band[:, :W // 2].mean())); right.append(float(band[:, W // 2:].mean()))
    else:
        left.append(0.0); right.append(0.0)
    prev = g
left = np.array(left); right = np.array(right); n = len(left); t = np.arange(n) / fps
# speech energy: mono 16 kHz PCM, RMS per video frame
pcm = subprocess.run([FF, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
spf = int(16000 / fps); rms = np.array([np.sqrt(np.mean(a[i * spf:(i + 1) * spf] ** 2) + 1e-12) for i in range(n)])
thr = max(0.02, np.percentile(rms, 60))                          # "speech present" = louder than the quiet 60 %
print(f"{path.split('/')[-1]}: {n} frames, speech threshold rms {thr:.3f}")
print(f"{'beat':>4} {'span':>11} {'speech%':>7} {'L-face':>7} {'R-face':>7} {'ratio':>6}  verdict")
for k in range(int(np.ceil(n / fps / beat))):
    sel = (t >= k * beat) & (t < (k + 1) * beat); sp = sel & (rms > thr)
    if sp.sum() < 3:
        print(f"{k + 1:>4} {k * beat:5.1f}-{(k + 1) * beat:5.1f} {'-':>7}  (no speech)"); continue
    L, R = left[sp].mean(), right[sp].mean(); ratio = max(L, R) / max(min(L, R), 1e-6)
    who = "both" if ratio < 1.4 else ("left" if L > R else "right")
    print(f"{k + 1:>4} {k * beat:5.1f}-{(k + 1) * beat:5.1f} {100 * sp.sum() / sel.sum():6.0f}% {L:7.3f} {R:7.3f} {ratio:6.2f}  {who}")
