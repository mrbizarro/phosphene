#!/usr/bin/env python3
"""lipsync_check.py <mp4> [...] — lip-sync by mouth OPENING, not motion.
Face found once per second with OpenCV's frontal cascade (full resolution); inside
the lower third of the face box, the fraction of pixels darker than the box's own
median minus 40 = how open the mouth is. That series, smoothed over 3 frames, is
correlated with the audio RMS over the frames where speech is present, at lags
-6..+6. A voice that the mouth follows: positive r near lag 0.

Calibration (2026-09-07, the owner's ear as ground truth, Bizarro one shots):
  parts judged fine       r = +0.29, +0.39, +0.41
  parts judged voiceover  r = +0.17, -0.13, +0.19
  anchor on a mid-word frame +0.47 · anchor on a closed mouth -0.15
Gate: r >= 0.25 passes; below 0.20 fails; between, listen. The optical-flow
variant that preceded this scored a fine part and a voiceover part the same."""
import sys, subprocess, numpy as np, cv2
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
casc=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
def score(p):
    cap=cv2.VideoCapture(p); fps=cap.get(cv2.CAP_PROP_FPS); opens=[]; box=None; i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        if box is None or i%int(fps)==0:
            f=casc.detectMultiScale(g,1.1,5,minSize=(80,80))
            if len(f): box=max(f,key=lambda b:b[2]*b[3])
        if box is None: opens.append(np.nan); i+=1; continue
        x,y,w,h=box; roi=g[y+int(h*0.62):y+h, x+int(w*0.25):x+int(w*0.75)]
        thr=np.median(roi)-40; opens.append(float((roi<thr).mean())); i+=1
    n=len(opens); o=np.array(opens); o=np.where(np.isnan(o),np.nanmean(o),o)
    pcm=subprocess.run([FF,"-v","error","-i",p,"-vn","-ac","1","-ar","16000","-f","s16le","-"],capture_output=True,check=True).stdout
    a=np.frombuffer(pcm,np.int16).astype(np.float32)/32768; spf=int(16000/fps)
    rms=np.array([np.sqrt(np.mean(a[j*spf:(j+1)*spf]**2)+1e-12) for j in range(n)])
    k=np.ones(3)/3; m=np.convolve(o,k,"same"); r=np.convolve(rms,k,"same"); sp=r>max(0.02,np.percentile(rms,55))
    if sp.sum()<24: return None
    best=(0,-9)
    for lag in range(-6,7):
        x=m[lag:][sp[:n-lag]] if lag>=0 else m[:n+lag][sp[-lag:]]; y=r[:n-lag][sp[:n-lag]] if lag>=0 else r[-lag:][sp[-lag:]]
        if len(x)>=24 and x.std()>1e-6: c=float(np.corrcoef(x,y)[0,1]); best=max(best,(lag,c),key=lambda t:t[1])
    return best, float(o[sp].mean()/max(o[~sp].mean(),1e-6))
for p in sys.argv[1:]:
    s=score(p)
    v = "no face/speech" if s is None else f"best r={s[0][1]:+.2f} @ {s[0][0]:+d}   open(speech)/open(quiet)={s[1]:.2f}  -> " + ("SYNC" if s[0][1] >= 0.25 else ("VOICEOVER" if s[0][1] < 0.20 else "listen"))
    print(f"{p.split('/')[-1]:45s} {v}")
