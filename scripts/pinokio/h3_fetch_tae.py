#!/usr/bin/env python3
"""Fetch the H3 draft decoder (madebyollin's TAE for H3) to a target path.

The Hugging Face repo this came from (madebyollin/taeh3) was deleted in
September 2026 and every H3 install/update died at this step with
"Repository Not Found" (Pinokio report by @macstephen, who also found the
file in the author's GitHub repo). The same bytes live at a PINNED commit
of github.com/madebyollin/taehv (MIT); we fetch from there first, verify
the sha256 against the copy that shipped for months, and only then place
it. HF stays as a fallback in case GitHub is unreachable.

Usage: h3_fetch_tae.py <target-path>
Exit 0 = the file is in place and verified. Non-zero = nothing placed.
"""
import hashlib
import os
import sys
import tempfile
import urllib.request

EXPECTED_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
EXPECTED_BYTES = 22709752
SOURCES = [
    ("github.com/madebyollin/taehv @ 62f7591 (MIT)",
     "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors"),
    ("huggingface.co/madebyollin/taeh3 (original, may be gone)",
     "https://huggingface.co/madebyollin/taeh3/resolve/main/taeh3.safetensors"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: h3_fetch_tae.py <target-path>", file=sys.stderr)
        return 2
    target = sys.argv[1]
    if os.path.isfile(target):
        with open(target, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == EXPECTED_SHA256:
                print("TAE draft decoder already in place (verified)")
                return 0
        print("existing TAE file does not match the known hash - refetching")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    for label, url in SOURCES:
        print(f"fetching TAE draft decoder from {label} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  failed: {exc}")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != EXPECTED_BYTES or digest != EXPECTED_SHA256:
            print(f"  rejected: {len(data)} bytes, sha256 {digest[:16]}... (expected "
                  f"{EXPECTED_BYTES} bytes, {EXPECTED_SHA256[:16]}...)")
            continue
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".", suffix=".part")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        print("TAE draft decoder ready (verified sha256)")
        return 0
    print("ERROR: could not fetch a verified TAE draft decoder from any source; "
          "drafts will fall back to the full decoder. Re-run 'Install Hailuo H3' "
          "later.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
