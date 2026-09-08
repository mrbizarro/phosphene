#!/usr/bin/env python3
"""Camera motion across the joins of a one shot.

Dense optical flow (Farneback) on a 320-px-wide grayscale copy, averaged over
the frame = a global motion vector per frame (camera move + subject motion;
on a two-shot at a table the camera dominates). For each join time t we
compare the mean vector over [t-1.0, t-0.1] s with [t+0.1, t+1.0] s:
direction change in degrees and speed ratio after/before. A clean join is a
small angle and a ratio near 1; the seams the owner sees are a big angle
(direction change) or a ratio far from 1 (stop / restart). Also prints the
motion magnitude per half-second so a stall anywhere shows.

usage: seam_motion.py <mp4> <join_sec> [<join_sec> ...]
"""
import sys, math
import cv2, numpy as np

path = sys.argv[1]; joins = [float(x) for x in sys.argv[2:]]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev = None; vecs = []            # per-frame (dx, dy) in px/frame at 320-px width
while True:
    ok, fr = cap.read()
    if not ok: break
    h, w = fr.shape[:2]; sc = 320.0 / w
    g = cv2.cvtColor(cv2.resize(fr, (320, int(h * sc))), cv2.COLOR_BGR2GRAY)
    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vecs.append((float(flow[..., 0].mean()), float(flow[..., 1].mean())))
    else:
        vecs.append((0.0, 0.0))
    prev = g
vecs = np.array(vecs); t = np.arange(len(vecs)) / fps
mag = np.hypot(vecs[:, 0], vecs[:, 1])
print(f"{path.split('/')[-1]}: {len(vecs)} frames @ {fps:.0f} fps, {len(vecs)/fps:.1f} s")
print("motion magnitude per 0.5 s (px/frame at 320 px wide):")
row = []
for k in range(int(len(vecs) / fps * 2)):
    m = mag[(t >= k / 2) & (t < (k + 1) / 2)]
    row.append(f"{m.mean():.2f}" if len(m) else "-")
for i in range(0, len(row), 20):
    print(f"  {i/2:5.1f}s  " + " ".join(row[i:i + 20]))
def win(a, b):
    sel = (t >= a) & (t < b); v = vecs[sel].mean(axis=0) if sel.any() else np.zeros(2)
    return v, float(np.hypot(*v))
for j in joins:
    vb, mb = win(j - 1.0, j - 0.1); va, ma = win(j + 0.1, j + 1.0)
    ang = 0.0
    if mb > 0.02 and ma > 0.02:
        ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(vb, va)) / (mb * ma)))))
    ratio = (ma / mb) if mb > 1e-6 else float("inf")
    STATIC = 0.05                       # px/frame at 320 px wide: below this the camera is standing still
    if mb < STATIC and ma < STATIC:
        verdict = "clean (static both sides)"
    elif mb < STATIC or ma < STATIC:
        verdict = "stop/restart"        # moving on one side, still on the other
    else:
        verdict = "clean" if (ang < 35 and 0.6 <= ratio <= 1.7) else ("direction change" if ang >= 35 else "stop/restart")
    print(f"join @ {j:5.1f}s: before ({vb[0]:+.2f},{vb[1]:+.2f}) |{mb:.2f}|  after ({va[0]:+.2f},{va[1]:+.2f}) |{ma:.2f}|  "
          f"angle {ang:5.1f}°  speed x{ratio:.2f}  -> {verdict}")
