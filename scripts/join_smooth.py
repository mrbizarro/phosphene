#!/usr/bin/env python3
"""Join one-shot parts with the two seam fixes from the H3 extender (pmhaidn,
GPLv3 — re-implemented, not copied): a colour match at the boundary that
cosine-fades back to the part's own colours over 24 frames, and a one-frame
optical-flow warp that removes the first-frame hesitation and balances the
velocity across the cut. Frames are never dropped; audio is concatenated.

usage: join_smooth.py [--hold-exposure] <out.mp4> <part1.mp4> <part2.mp4> [...]
"""
import subprocess, sys, json, os, tempfile
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
FP = os.path.join(os.path.dirname(FF), "ffprobe") if os.path.dirname(FF) else "ffprobe"

def probe(p):
    j = json.loads(subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate", "-of", "json", p],
                                  capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = j["r_frame_rate"].split("/"); return int(j["width"]), int(j["height"]), float(num) / float(den)

def frames(p, w, h):
    raw = subprocess.run([FF, "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()

def color_seam_fade(b, a_last, span=24):
    """Match b's first frame to a's last frame (per-channel mean/std), fade the correction out over `span` frames."""
    bf = b[:span].astype(np.float32) / 255.0; s = a_last.astype(np.float32) / 255.0; t = bf[0]
    # Exposure/colour OFFSET only (per-channel mean), not a std rescale: on a high-key frame
    # (a white salt flat) rescaling the spread clips the highlights and the mean never lands.
    corr = bf.copy()
    for c in range(3):
        corr[..., c] = bf[..., c] + (s[..., c].mean() - t[..., c].mean())
    corr = np.clip(corr, 0, 1)
    n = bf.shape[0]; ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n)))
    out = ramp[:, None, None, None] * corr + (1 - ramp[:, None, None, None]) * bf
    b[:span] = (out * 255.0 + 0.5).astype(np.uint8); return b

def gray(x): return (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).astype(np.uint8)

def seam_warp(a, b):
    """The extender's one-frame fix: advance b[0] 20% toward b[1] (kills the first-frame hesitation),
    then warp a[-1] and b[0] 15% toward each other (balances velocity across the cut)."""
    h, w = a.shape[1:3]; x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    b0, b1 = b[0].astype(np.float32), b[1].astype(np.float32)
    fl = cv2.calcOpticalFlowFarneback(gray(b[0]), gray(b[1]), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    adv = 0.20
    w0 = cv2.remap(b0, x - fl[..., 0] * adv, y - fl[..., 1] * adv, cv2.INTER_LINEAR)
    w1 = cv2.remap(b1, x + fl[..., 0] * (1 - adv), y + fl[..., 1] * (1 - adv), cv2.INTER_LINEAR)
    b0 = (1 - adv) * w0 + adv * w1
    al = a[-1].astype(np.float32)
    fs = cv2.calcOpticalFlowFarneback(gray(a[-1]), gray(b0.astype(np.uint8)), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    k = 0.15
    wa = cv2.remap(al, x - fs[..., 0] * k, y - fs[..., 1] * k, cv2.INTER_LINEAR)
    wb = cv2.remap(b0, x + fs[..., 0] * k, y + fs[..., 1] * k, cv2.INTER_LINEAR)
    a[-1] = np.clip(0.85 * al + 0.15 * wb, 0, 255).astype(np.uint8)
    b[0] = np.clip(0.85 * b0 + 0.15 * wa, 0, 255).astype(np.uint8)
    return a, b

def hold_exposure(seqs, smooth=12):
    """Pin the mean luma of every frame to the level of the take's first second.

    Measured on the Bizarro one shot: part 2 brightened 137 -> 178 across its ten
    seconds and part 3 re-levelled to 157 one frame after its anchored first frame,
    so the join read as an exposure pop and the part as a slow overexposure. The
    scene's light is constant by contract (the light lock), so a per-frame offset
    toward one target, smoothed over `smooth` frames, is the honest correction.
    """
    means = np.concatenate([q.reshape(q.shape[0], -1).mean(axis=1) for q in seqs]).astype(np.float32)
    target = float(np.median(means[:24]))
    delta = target - means
    k = np.ones(smooth, np.float32) / smooth
    delta = np.convolve(np.pad(delta, (smooth // 2, smooth - smooth // 2 - 1), mode="edge"), k, mode="valid")
    i = 0
    for q in seqs:
        n = q.shape[0]
        d = delta[i:i + n].reshape(n, 1, 1, 1)
        q[:] = np.clip(q.astype(np.float32) + d, 0, 255).astype(np.uint8)
        i += n
    return seqs


def main():
    hold = "--hold-exposure" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--hold-exposure"]
    out, parts = args[0], args[1:]
    w, h, fps = probe(parts[0])
    seqs = [frames(p, w, h) for p in parts]
    if hold:
        seqs = hold_exposure(seqs)
    for i in range(1, len(seqs)):
        seqs[i] = color_seam_fade(seqs[i], seqs[i - 1][-1])
        seqs[i - 1], seqs[i] = seam_warp(seqs[i - 1], seqs[i])
    with tempfile.TemporaryDirectory() as td:
        # audio: a butt-join with a 15 ms fade out / fade in at every seam. NOT a crossfade:
        # acrossfade overlaps the parts and shortens the sound by its length at every seam
        # while every video frame is kept, so the voice ran 0.12 s earlier per join —
        # 0.6 s ahead of the mouth by the sixth part of a take (owner, 2026-09-07:
        # "at some point the lip sync doesn't work anymore"). The fades kill the click
        # and cost no time; the sound stays on the frame it was rendered with.
        aud = os.path.join(td, "a.m4a"); ins = []; fc = ""
        for i, q in enumerate(parts): ins += ["-i", q]
        if len(parts) == 1:
            subprocess.run([FF, "-v", "error", "-y", "-i", parts[0], "-vn", "-c:a", "aac", "-b:a", "192k", aud], check=True)
        else:
            durs = [float(subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", q],
                                         capture_output=True, text=True).stdout.strip()) for q in parts]
            for i, d in enumerate(durs):
                fin = f"afade=t=in:st=0:d=0.015," if i > 0 else ""
                fout = f"afade=t=out:st={max(0.0, d - 0.015):.3f}:d=0.015" if i < len(parts) - 1 else "anull"
                fc += f"[{i}:a]{fin}{fout}[f{i}];"
            fc += "".join(f"[f{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[aout]"
            subprocess.run([FF, "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[aout]",
                            "-c:a", "aac", "-b:a", "192k", aud], check=True)
        enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                                "-i", aud, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "-shortest", out],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for s in seqs: enc.stdin.write(s.tobytes())
            enc.stdin.close()
        except BrokenPipeError:
            pass
        err = enc.stderr.read().decode(errors="ignore"); enc.wait()
        if enc.returncode != 0:
            raise SystemExit(f"encoder failed ({enc.returncode}): {err[-800:]}")
    print("joined", len(parts), "parts ->", out, "| seams smoothed:", len(parts) - 1)

if __name__ == "__main__": main()
