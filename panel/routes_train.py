"""/train family routes — moved out of the chain (slice 4).

Bodies are verbatim from mlx_ltx_panel.py's do_GET/do_POST chains except
the two mechanical renames the move forces: `self` -> `h`, and panel
globals -> `P.<name>`. See panel/routes_stats.py for the pattern and
panel/__init__.py for why P is assigned rather than imported.
"""
from __future__ import annotations

from urllib.parse import parse_qs

from panel.routes import get, post

P = None  # the running mlx_ltx_panel module; assigned at wiring time


# ====== Train Character — completed LoRA list =====================
@get("/train/list")
def get_train_list(h, parsed) -> None:
    h._json({
        "ok": True,
        "loras": P._train_list_completed(),
        "loras_dir": str(P._safe_loras_dir()),
    })


@get("/train/auto-caption/status")
def get_train_auto_caption_status(h, parsed) -> None:
    # Snapshot of CAPTION_STATE for the Train tab to poll while a
    # caption run is in flight. Cheap — read under LOCK, return.
    # The full log still goes through /status; this just surfaces
    # the structured progress fields (i / n / current_file) so the
    # UI can render a clean progress bar without parsing log text.
    with P.LOCK:
        snap = dict(P.CAPTION_STATE)
    h._json({"ok": True, **snap})


# ====== Train Character — server-side suggestion for a fresh
# trigger token. The JS already has its own generator (so the
# button is instant), but this endpoint exists for agent callers
# and CLI users who want a hint without running JS.
@get("/train/suggest-trigger")
def get_train_suggest_trigger(h, parsed) -> None:
    h._json({"ok": True, "trigger": P._suggest_trigger_token()})


# ====== Train Character — preflight model check.
# required_files.json's q4 entry ships transformer-distilled only.
# Training needs the FULL-PRECISION transformer-dev from the Q8 repo
# (extra ~21 GB), which isn't in the default install. Surface what's
# missing so the UI can offer a one-click download via the existing
# hf flow. (The Q4 repo's transformer-dev is quantized — the trainer
# refuses it, #35 — so the preflight neither offers nor greenlights it.)
@get("/train/preflight")
def get_train_preflight(h, parsed) -> None:
    req = P._train_required_models()
    h._json({"ok": True, "required": req,
                "all_ready": all(r["ready"] for r in req)})


# ====== Train Character — listing of the in-progress dataset
# (uploaded but not yet trained). Used by the UI to restore the
# thumbnail grid after a page reload.
@get("/train/dataset")
def get_train_dataset(h, parsed) -> None:
    qs = P.parse_qs(parsed.query)
    requested = (qs.get("job_id", [""])[0] or "").strip()
    if not requested:
        h._json({"ok": True, "job_id": None, "images": []}); return
    try:
        job_id = P._safe_job_id(requested)
    except ValueError as e:
        h._json({"error": str(e)}, 400); return
    ds = P._train_dataset_dir(job_id)
    images_dir = ds / "images"
    captions_dir = ds / "captions"
    cmap = P._train_read_caption_map(ds)
    # Reverse map (saved_stem → original_stem) so the UI can show
    # the user's filename in tooltips.
    reverse_cmap = {v: k for k, v in cmap.items()}
    images: list[dict] = []
    captioned_count = 0
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in P.TRAIN_IMAGE_EXTS:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            cap_path = captions_dir / f"{p.stem}.txt"
            has_caption = cap_path.is_file()
            word_count = None
            if has_caption:
                try:
                    word_count = P._caption_word_count(
                        cap_path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    word_count = None
                captioned_count += 1
            images.append({
                "filename": p.name,
                "path": str(p),
                "size_bytes": st.st_size,
                "captioned": has_caption,
                "caption_words": word_count,
                "original_stem": reverse_cmap.get(p.stem, p.stem),
            })
    # Parked captions — uploaded .txt without a matching image yet.
    parked: list[str] = []
    if captions_dir.is_dir():
        image_stems = {P.Path(im["filename"]).stem for im in images}
        for c in sorted(captions_dir.iterdir()):
            if not c.is_file() or c.suffix.lower() != ".txt":
                continue
            if c.stem not in image_stems:
                parked.append(c.stem)
    h._json({
        "ok": True,
        "job_id": job_id,
        "count": len(images),
        "captioned_count": captioned_count,
        "parked_captions": parked,
        "min": P.TRAIN_MIN_IMAGES,
        "max": P.TRAIN_MAX_IMAGES,
        "images": images,
    })


# ====== Train Character — serve a dataset thumbnail (the raw
# uploaded image). Mirrors /file but scoped to TRAIN_DIR so the
# UI can render <img src="/train/file?job_id=...&filename=...">
# without exposing the rest of state/.
@get("/train/file")
def get_train_file(h, parsed) -> None:
    qs = P.parse_qs(parsed.query)
    requested = (qs.get("job_id", [""])[0] or "").strip()
    filename = (qs.get("filename", [""])[0] or "").strip()
    if not requested or not filename:
        h.send_error(400, "job_id + filename required"); return
    try:
        job_id = P._safe_job_id(requested)
    except ValueError:
        h.send_error(400); return
    if "/" in filename or ".." in filename:
        h.send_error(400); return
    ds_root = P.TRAIN_DIR.resolve()
    target = (P.TRAIN_DIR / job_id / "images" / filename).resolve()
    if not target.is_relative_to(ds_root) or not target.is_file():
        h.send_error(404); return
    ext = target.suffix.lower()
    ctype = {
        ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    try:
        data = target.read_bytes()
    except OSError:
        h.send_error(500); return
    h.send_response(200)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(data)))
    h.send_header("Cache-Control", "no-cache")
    h.end_headers()
    h.wfile.write(data)


# ====== Train Character — start training (enqueue a mode='train' job)
@post("/train/start")
def post_train_start(h, path, qs, ctype) -> None:
    P._analytics_feature("train_start")
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    train_job_id = (form.get("train_job_id", [""])[0] or "").strip()
    if not train_job_id:
        h._json({"error": "train_job_id required"}, 400); return
    if float(P.SYSTEM_RAM_GB or 0) and float(P.SYSTEM_RAM_GB) < P.TRAIN_MIN_RAM_GB:
        h._json({"error": f"Training needs at least {P.TRAIN_MIN_RAM_GB} GB of memory; this Mac has "
                          f"{float(P.SYSTEM_RAM_GB):.0f} GB. On this machine the trainer runs out of memory "
                          f"before it finishes."}, 409); return
    try:
        train_job_id = P._safe_job_id(train_job_id)
    except ValueError as e:
        h._json({"error": str(e)}, 400); return
    ds = P._train_dataset_dir(train_job_id)
    images_dir = ds / "images"
    if not images_dir.is_dir():
        h._json({"error": f"dataset not found: {images_dir}"}, 404); return
    image_files = [x for x in images_dir.iterdir()
                   if x.is_file() and x.suffix.lower() in P.TRAIN_IMAGE_EXTS]
    if len(image_files) < P.TRAIN_MIN_IMAGES:
        h._json({
            "error": f"not enough images: {len(image_files)} (need >= {P.TRAIN_MIN_IMAGES})"
        }, 400)
        return
    # Train type — "character" (default, voice-paired) or "style"
    # (no voice). Validate up front so an unrecognized value 400s
    # rather than silently routing to character.
    train_type = (form.get("train_type", ["character"])[0]
                  or "character").lower()
    if train_type not in P.TRAIN_TYPES:
        h._json({
            "error": f"train_type must be one of {list(P.TRAIN_TYPES)} "
                     f"(got {train_type!r})"
        }, 400)
        return
    # If the user opted into voice training, a clip must exist on
    # disk before we queue the job — defensive guard so the worker
    # doesn't have to fail mid-flight. The Train tab also disables
    # the toggle when no clip is uploaded, but the API is the
    # source of truth. Styles never train audio — silently coerce
    # train_audio to false for style runs so a stale form value
    # doesn't get the user into the voice-required branch.
    if train_type == "style":
        form["train_audio"] = ["false"]
        wants_audio = False
    else:
        train_audio_raw = (form.get("train_audio", ["false"])[0]
                           or "false").lower()
        wants_audio = train_audio_raw in ("1", "true", "yes", "on")
        if wants_audio and P._existing_voice_file(ds) is None:
            h._json({
                "error": "voice clip required when train_audio is "
                         "true — upload one via /train/upload-voice "
                         "first"
            }, 400)
            return
    # Make sure the form has the canonical image_count so make_job's
    # estimate and step cap use the dataset on disk, not stale JS state.
    form["mode"] = ["train"]
    form["train_job_id"] = [train_job_id]
    form["train_type"] = [train_type]
    form["image_count"] = [str(len(image_files))]
    job = P.make_job(form)
    with P.QUEUE_COND:
        P.STATE["queue"].append(job)
        P.QUEUE_COND.notify_all()
    P.persist_queue()
    h._json({"ok": True, "queued_id": job["id"],
                "train_job_id": train_job_id,
                "params": job["params"]})


# Remove the (single) voice clip from a pending dataset. Used by
# the Train tab "remove" button. Safe to call even if no clip
# exists — returns ok in that case so the UI can stay idempotent.
@post("/train/remove-voice")
def post_train_remove_voice(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    train_job_id = (form.get("train_job_id", [""])[0] or "").strip()
    if not train_job_id:
        h._json({"error": "train_job_id required"}, 400); return
    try:
        train_job_id = P._safe_job_id(train_job_id)
    except ValueError as e:
        h._json({"error": str(e)}, 400); return
    ds = P._train_dataset_dir(train_job_id)
    removed: list[str] = []
    for ext in P.TRAIN_VOICE_EXTS:
        p = ds / f"voice{ext}"
        if p.is_file():
            try:
                p.unlink()
                removed.append(p.name)
            except OSError as e:
                h._json({"error": f"could not delete: {e}"}, 500)
                return
    h._json({"ok": True, "job_id": train_job_id,
                "removed": removed})


# Remove a single image from a pending dataset (before training has
# started). Re-numbers the remaining files so the lab side still
# sees char_001…NN. Captions track image stems exactly so we move
# them in lockstep, and we update caption_map.json so the
# original_stem → saved_stem mapping survives the renumber.
@post("/train/remove-image")
def post_train_remove_image(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    train_job_id = (form.get("train_job_id", [""])[0] or "").strip()
    filename = (form.get("filename", [""])[0] or "").strip()
    if not train_job_id or not filename:
        h._json({"error": "train_job_id + filename required"}, 400); return
    try:
        train_job_id = P._safe_job_id(train_job_id)
    except ValueError as e:
        h._json({"error": str(e)}, 400); return
    # Filename containment — must be a basename, no slashes.
    if "/" in filename or ".." in filename:
        h._json({"error": "bad filename"}, 400); return
    ds = P._train_dataset_dir(train_job_id)
    images_dir = ds / "images"
    captions_dir = ds / "captions"
    target = images_dir / filename
    if not target.is_file() or not target.resolve().is_relative_to(images_dir.resolve()):
        h._json({"error": "image not found"}, 404); return
    removed_stem = target.stem
    try:
        target.unlink()
    except OSError as e:
        h._json({"error": f"could not delete: {e}"}, 500); return
    # Drop the matching caption.
    removed_caption = captions_dir / f"{removed_stem}.txt"
    if removed_caption.is_file():
        try:
            removed_caption.unlink()
        except OSError:
            pass
    # Drop the caption_map entry pointing at the removed saved_stem.
    cmap = P._train_read_caption_map(ds)
    cmap = {k: v for k, v in cmap.items() if v != removed_stem}

    # Renumber to keep the char_NNN sequence dense. Build the
    # mapping first so we can update captions + caption_map in
    # one consistent pass after the renames land.
    remaining = sorted([x for x in images_dir.iterdir()
                        if x.is_file() and x.suffix.lower() in P.TRAIN_IMAGE_EXTS])
    stem_renames: dict[str, str] = {}
    for idx, p in enumerate(remaining, start=1):
        old_stem = p.stem
        new_stem = f"char_{idx:03d}"
        new_name = f"{new_stem}{p.suffix.lower()}"
        if p.name != new_name:
            try:
                p.rename(p.with_name(new_name))
                stem_renames[old_stem] = new_stem
            except OSError:
                pass
    # Apply the same renames to captions (best-effort — if a
    # caption is missing we just skip it; the renumber must not
    # fail the request).
    if captions_dir.is_dir():
        for old_stem, new_stem in stem_renames.items():
            old_cap = captions_dir / f"{old_stem}.txt"
            new_cap = captions_dir / f"{new_stem}.txt"
            if old_cap.is_file() and not new_cap.is_file():
                try:
                    old_cap.rename(new_cap)
                except OSError:
                    pass
    # Update caption_map values to the new saved stems.
    if stem_renames:
        cmap = {k: stem_renames.get(v, v) for k, v in cmap.items()}
    P._train_write_caption_map(ds, cmap)
    h._json({"ok": True, "count": len(remaining),
                "job_id": train_job_id})


# Delete a trained LoRA — removes the .safetensors + sidecar from
# mlx_models/loras/. The Trained-LoRAs list in the form refreshes
# off /train/list afterwards.
@post("/train/delete")
def post_train_delete(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    target_path = (form.get("path", [""])[0] or "").strip()
    if not target_path:
        h._json({"error": "path required"}, 400); return
    try:
        resolved = P.Path(target_path).resolve()
    except OSError as e:
        h._json({"error": f"unresolvable: {e}"}, 400); return
    loras_root = P._safe_loras_dir().resolve()
    if not resolved.is_relative_to(loras_root):
        h._json({"error": "path not under mlx_models/loras/"}, 400); return
    if not resolved.is_file() or resolved.suffix.lower() != ".safetensors":
        h._json({"error": "not a .safetensors file"}, 404); return
    # Style LoRAs sit on disk as `<trigger>.style.safetensors` with a
    # sibling `<trigger>.style.json` (NOT `<trigger>.style.safetensors`
    # → `.json` via with_suffix — that would strip only ".safetensors"
    # giving "<trigger>.style.json" anyway, but explicit is safer).
    if resolved.name.endswith(".style.safetensors"):
        sidecar = resolved.with_name(
            resolved.name[: -len(".safetensors")] + ".json"
        )
    else:
        sidecar = resolved.with_suffix(".json")
    try:
        # Only allow delete on entries flagged as Phosphene-trained.
        # Accepts both kinds (train_character + train_style) — never
        # blow away an unrelated LoRA the user downloaded from CivitAI.
        if sidecar.is_file():
            try:
                meta = P.json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            if meta.get("kind") not in ("train_character", "train_style"):
                h._json({
                    "error": "refusing to delete: not a Phosphene-trained LoRA"
                }, 403)
                return
            sidecar.unlink()
        resolved.unlink()
    except OSError as e:
        h._json({"error": f"delete failed: {e}"}, 500); return
    h._json({"ok": True, "deleted": str(resolved)})


# ====== Train Character — install missing model on demand.
# Today only `ltx_dev_transformer` is downloadable here; future keys
# can be added by extending _train_install_dev_transformer / the
# preflight list. Returns 202 (Accepted) — caller polls /status's
# `download` block for progress (existing infra).
@post("/train/install")
def post_train_install(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    key = (form.get("key", [""])[0] or "").strip()
    # The 2.3 base pack goes through the ORDINARY download lane — the
    # same singleton, the same progress tail, the same Resume and the
    # same deep-verify every other pack uses. A second downloader for
    # the trainer's benefit is exactly the improvisation this campaign
    # removes.
    if key == "ltx23_base":
        repo = next((r for r in P._repos() if r.get("key") == "q4"), None)
        if not repo:
            h._json({"error": "LTX-2.3 is not a registered pack"}, 400); return
        with P.DOWNLOAD_LOCK:
            if P.DOWNLOAD["active"]:
                h._json({"error": f"another download is in progress: "
                                     f"{P.DOWNLOAD['repo_id']}. Wait for it to "
                                     f"finish (or click Cancel)."}, 409); return
            P.DOWNLOAD["active"] = True
            P.DOWNLOAD["key"] = "q4"
            P.DOWNLOAD["repo_id"] = repo["repo_id"]
            P.DOWNLOAD["started_ts"] = P.time.time()
            P.DOWNLOAD["last_line"] = "starting…"
        P.threading.Thread(target=P._download_thread, args=(repo,), daemon=True).start()
        h._json({"ok": True, "key": "q4", "repo_id": repo["repo_id"]}, 202)
        return
    if key != "ltx_dev_transformer":
        h._json({"error": f"unknown install key {key!r}"}, 400); return
    result = P._train_install_dev_transformer(P.push)
    if not result.get("ok"):
        h._json(result, 409 if "active" in result.get("error", "") else 500)
        return
    h._json(result, 202)


@post("/train/auto-caption")
def post_train_auto_caption(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    # Auto-caption every image in a Train Character dataset using the
    # local Gemma 3 12B multimodal model. Same weights the panel
    # already downloads for prompt enhancement — no extra model fetch.
    # Spawns caption_with_gemma.py as a subprocess so Gemma's ~6 GB
    # RSS releases the moment captioning finishes; stdout streams
    # back as JSON-line events the worker thread pushes into the
    # log + into CAPTION_STATE for UI polling.
    train_job_id = (form.get("train_job_id", [""])[0] or "").strip()
    if not train_job_id:
        h._json({"error": "train_job_id required"}, 400); return
    try:
        train_job_id = P._safe_job_id(train_job_id)
    except ValueError as e:
        h._json({"error": str(e)}, 400); return
    ds = P._train_dataset_dir(train_job_id)
    images_dir = ds / "images"
    if not images_dir.is_dir():
        h._json({"error": f"dataset not found: {images_dir}"}, 404)
        return
    image_files = [x for x in images_dir.iterdir()
                   if x.is_file() and x.suffix.lower() in P.TRAIN_IMAGE_EXTS]
    if not image_files:
        h._json({"error": "no images to caption"}, 400); return
    # Trigger token defaults to the job_id (which is how new jobs
    # are minted via _new_train_job_id() — they ARE the trigger).
    trigger = (form.get("trigger", [""])[0] or train_job_id).strip()
    if not trigger:
        h._json({"error": "trigger required"}, 400); return

    # Refuse if another captioning is in flight — two Gemmas in
    # RAM at the same time would push us into swap.
    with P.LOCK:
        if P.CAPTION_STATE.get("running"):
            h._json({
                "error": "auto-caption already running for "
                         f"{P.CAPTION_STATE.get('train_job_id')}",
            }, 409)
            return
        # Refuse while ANY GPU job is in flight. This originally only
        # blocked training, which allowed Gemma auto-captioning to run
        # beside a long video render. On 48 GB Macs that pushes MLX
        # into compressor/swap, and the active render usually never
        # recovers speed mid-job even after Gemma exits.
        cur = P.STATE.get("current")
        if cur or P._GPU_LOCK.locked():
            cur_mode = ((cur or {}).get("params") or {}).get("mode") or "render"
            h._json({
                "error": f"GPU is busy with {cur_mode} — wait for the "
                         "current job to finish before auto-captioning",
            }, 409)
            return
        # Reset state for this run.
        P.CAPTION_STATE.update({
            "running": True,
            "train_job_id": train_job_id,
            "trigger": trigger,
            "i": 0,
            "n": len(image_files),
            "current_file": None,
            "last_caption": None,
            "started_at": P.time.time(),
            "elapsed_sec": 0.0,
            "error": None,
        })

    script_path = P.ROOT / "caption_with_gemma.py"
    cmd = [
        str(P.HELPER_PYTHON),
        str(script_path),
        "--dataset", str(ds),
        "--trigger", trigger,
    ]
    P.push(f"[caption] $ {' '.join(P.shlex.quote(c) for c in cmd)}")

    def _runner() -> None:
        t_start = P.time.time()
        try:
            proc = P.subprocess.Popen(
                cmd,
                stdout=P.subprocess.PIPE,
                stderr=P.subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Own process group so /stop's killpg can take down
                # the captioner + any child it spawned.
                start_new_session=True,
            )
        except Exception as exc:
            P.push(f"[caption] failed to spawn: "
                 f"{type(exc).__name__}: {exc}")
            with P.LOCK:
                P.CAPTION_STATE["running"] = False
                P.CAPTION_STATE["error"] = str(exc)
            return
        with P.LOCK:
            P.CAPTION_PROC = proc
        # Stream stdout. Each line is either JSON (a structured
        # event from caption_with_gemma.py) or plain text (fallback
        # — surface as-is to the log).
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            payload = None
            if line.lstrip().startswith("{"):
                try:
                    payload = P.json.loads(line)
                except P.json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                evt = payload.get("event")
                if evt == "loading":
                    P.push("[caption] loading Gemma 3 12B (~3s on warm)…")
                elif evt == "loaded":
                    P.push(f"[caption] Gemma loaded in "
                         f"{payload.get('elapsed_sec','?')}s — "
                         f"captioning {P.CAPTION_STATE['n']} images.")
                elif evt == "progress":
                    i = int(payload.get("i") or 0)
                    n = int(payload.get("n") or P.CAPTION_STATE["n"])
                    fname = payload.get("file") or ""
                    caption = payload.get("caption") or ""
                    with P.LOCK:
                        P.CAPTION_STATE["i"] = i
                        P.CAPTION_STATE["n"] = n
                        P.CAPTION_STATE["current_file"] = fname
                        P.CAPTION_STATE["last_caption"] = caption
                        P.CAPTION_STATE["elapsed_sec"] = round(
                            P.time.time() - t_start, 1)
                    # One log line per image so users see real
                    # progress, truncated to keep the pane tidy.
                    preview = caption[:90] + ("…" if len(caption) > 90 else "")
                    P.push(f"[caption] {i}/{n} {fname}: {preview}")
                elif evt == "done":
                    P.push(f"[caption] done — {payload.get('count')} "
                         f"captions in "
                         f"{payload.get('elapsed_sec','?')}s")
                elif evt == "error":
                    P.push(f"[caption] error: {payload.get('message')}")
                    with P.LOCK:
                        P.CAPTION_STATE["error"] = payload.get("message")
                else:
                    # Unknown event — log the raw line.
                    P.push(f"[caption] {line}")
            else:
                P.push(f"[caption] {line}")
        rc = proc.wait()
        with P.LOCK:
            P.CAPTION_STATE["running"] = False
            P.CAPTION_STATE["elapsed_sec"] = round(
                P.time.time() - t_start, 1)
            if rc != 0 and not P.CAPTION_STATE.get("error"):
                P.CAPTION_STATE["error"] = (
                    f"subprocess exited with code {rc}")
        if rc == 0:
            P.push(f"[caption] subprocess exit 0 — refresh dataset "
                 "view to see updated captions.")
        else:
            P.push(f"[caption] subprocess exit {rc}")

    P.threading.Thread(target=_runner, daemon=True,
                     name="auto-caption").start()
    h._json({
        "ok": True,
        "train_job_id": train_job_id,
        "trigger": trigger,
        "image_count": len(image_files),
    })


# ====== Train Character — multipart dataset upload ===============
# POSTs one file per request (image OR caption), accumulating under
# state/train_character/<job_id>/{images,captions}/. Captions are
# paired by filename stem via a caption_map.json (original_stem →
# saved_stem) the image upload records — if no matching image is
# present yet, captions are parked under their original stem and
# reconciled when the image arrives (or on training start, as a
# safety net).
@post("/train/upload")
def post_train_upload(h, path, qs, ctype) -> None:
    # The chain arm carried this condition; its failure fell
    # through to the chain end, which answers 404.
    if not (ctype.startswith("multipart/form-data")):
        h.send_error(404)
        return
    MAX_UPLOAD_BYTES = P.TRAIN_MAX_BYTES_PER_IMAGE
    try:
        clen = int(h.headers.get("Content-Length") or "0")
    except ValueError:
        clen = 0
    if clen <= 0:
        h._json({"error": "Content-Length required"}, 411); return
    if clen > MAX_UPLOAD_BYTES:
        # The cap is on the REQUEST, and this route takes ONE image per
        # request. Saying "per file" sent a caller hunting for an
        # oversized image that did not exist — their largest was 2.4 MB
        # and the 73 MB batch was the thing being refused.
        h._json({
            "error": f"upload too large: this request is {clen} bytes and the "
                     f"limit is {MAX_UPLOAD_BYTES} per request. This route takes "
                     f"ONE image per request (field 'file') — send them one at a "
                     f"time, passing the returned job_id on each subsequent call."
        }, 413)
        return
    try:
        form = P._parse_multipart_form(h.rfile, ctype, clen)
        # job_id is optional — if absent, mint a fresh one for the
        # first upload of a session. The JS client echoes the
        # returned id back on every subsequent upload to keep the
        # whole batch landing in one directory.
        requested_id = (form.getvalue("job_id") or "").strip() if "job_id" in form else ""
        if requested_id:
            try:
                train_job_id = P._safe_job_id(requested_id)
            except ValueError as e:
                h._json({"error": str(e)}, 400); return
        else:
            train_job_id = P._new_train_job_id()

        if "file" not in form:
            h._json({"error": "no field 'file'"}, 400); return
        fld = form["file"]
        if not getattr(fld, "filename", None):
            h._json({"error": "no filename"}, 400); return
        ext = P.Path(fld.filename).suffix.lower()
        original_stem = P.Path(fld.filename).stem
        if ext not in P.TRAIN_IMAGE_EXTS and ext not in P.TRAIN_CAPTION_EXTS:
            h._json({
                "error": f"unsupported file type {ext!r}; want one of "
                         f"{sorted(P.TRAIN_IMAGE_EXTS | P.TRAIN_CAPTION_EXTS)}"
            }, 400)
            return

        ds = P._train_dataset_dir(train_job_id)
        ds.mkdir(parents=True, exist_ok=True)

        # ----- CAPTION branch (.txt / .json) -----
        if ext in P.TRAIN_CAPTION_EXTS:
            data = fld.file.read()
            if len(data) > P.TRAIN_CAPTION_MAX_BYTES:
                h._json({
                    "error": f"caption too large (max "
                             f"{P.TRAIN_CAPTION_MAX_BYTES} bytes)"
                }, 413)
                return
            captions_dir = ds / "captions"
            captions_dir.mkdir(parents=True, exist_ok=True)
            cmap = P._train_read_caption_map(ds)
            saved_stem = cmap.get(original_stem, original_stem)
            raw_text = P._decode_caption_bytes(data)
            normalised = P._normalise_caption_text(raw_text)
            dest = captions_dir / f"{saved_stem}.txt"
            dest.write_text(normalised, encoding="utf-8")
            paired = original_stem in cmap
            h._json({
                "ok": True,
                "job_id": train_job_id,
                "kind": "caption",
                "filename": dest.name,
                "original_stem": original_stem,
                "saved_stem": saved_stem,
                "paired": paired,
                "word_count": P._caption_word_count(normalised),
                "path": str(dest),
            })
            return

        # ----- IMAGE branch (.png/.jpg/.jpeg/.webp) -----
        images_dir = ds / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        existing = [x for x in images_dir.iterdir()
                    if x.is_file() and x.suffix.lower() in P.TRAIN_IMAGE_EXTS]
        if len(existing) >= P.TRAIN_MAX_IMAGES:
            h._json({
                "error": f"already at {P.TRAIN_MAX_IMAGES} images "
                         "(max). Remove some before uploading more.",
                "job_id": train_job_id,
                "count": len(existing),
            }, 409)
            return

        # Numeric ordering: char_001.jpg, char_002.jpg, ...
        # Lets the lab side load them in a deterministic order.
        next_idx = len(existing) + 1
        saved_name = f"char_{next_idx:03d}{ext}"
        saved_stem = f"char_{next_idx:03d}"
        dest = images_dir / saved_name
        dest.write_bytes(fld.file.read())
        # Record original_stem → saved_stem so a later caption with
        # the original filename pairs on arrival.
        P._train_set_caption_map_entry(ds, original_stem, saved_stem)
        # Reconcile any caption already parked under the original
        # stem (when .txt was uploaded BEFORE its matching image).
        captioned = P._train_reconcile_one(ds, original_stem, saved_stem)
        count = len(existing) + 1
        h._json({
            "ok": True,
            "job_id": train_job_id,
            "kind": "image",
            "filename": saved_name,
            "original_stem": original_stem,
            "saved_stem": saved_stem,
            "captioned": captioned,
            "path": str(dest),
            "count": count,
            "min": P.TRAIN_MIN_IMAGES,
            "max": P.TRAIN_MAX_IMAGES,
        })
    except Exception as exc:
        h._json({"error": f"train upload failed: {exc}"}, 500)


# ====== Train Character — bulk ZIP dataset upload =================
# Mr Bizarro's "Cinematron" workflow: drop a ZIP containing paired
# image_NNN.png + image_NNN.txt files, get a fully-staged dataset
# back in one shot. Validates every entry against
# TRAIN_BUNDLE_NAME_RE (basename only, no subdirs, no traversal),
# dedupes by stem, and emits a paired_count + unpaired_warnings
# payload so the UI can decide whether to start training or nudge
# the user to upload more pairs.
@post("/train/upload-bundle")
def post_train_upload_bundle(h, path, qs, ctype) -> None:
    # The chain arm carried this condition; its failure fell
    # through to the chain end, which answers 404.
    if not (ctype.startswith("multipart/form-data")):
        h.send_error(404)
        return
    try:
        clen = int(h.headers.get("Content-Length") or "0")
    except ValueError:
        clen = 0
    if clen <= 0:
        h._json({"error": "Content-Length required"}, 411); return
    if clen > P.TRAIN_BUNDLE_MAX_BYTES:
        h._json({
            "error": f"bundle too large (max "
                     f"{P.TRAIN_BUNDLE_MAX_BYTES // (1024*1024)} MB)"
        }, 413)
        return
    try:
        form = P._parse_multipart_form(h.rfile, ctype, clen)
        requested_id = (form.getvalue("job_id") or "").strip() if "job_id" in form else ""
        if requested_id:
            try:
                train_job_id = P._safe_job_id(requested_id)
            except ValueError as e:
                h._json({"error": str(e)}, 400); return
        else:
            train_job_id = P._new_train_job_id()

        if "file" not in form:
            h._json({"error": "no field 'file'"}, 400); return
        fld = form["file"]
        if not getattr(fld, "filename", None):
            h._json({"error": "no filename"}, 400); return
        if P.Path(fld.filename).suffix.lower() != ".zip":
            h._json({"error": "expected a .zip bundle"}, 400)
            return

        zip_bytes = fld.file.read()
        try:
            zf = P.zipfile.ZipFile(P.io.BytesIO(zip_bytes), mode="r")
        except P.zipfile.BadZipFile:
            h._json({"error": "invalid ZIP file"}, 400); return

        # Filter entries server-side first so we can fast-fail on
        # path traversal / unsupported types before doing any disk
        # writes. The regex enforces basename + safe extension.
        entries = [e for e in zf.infolist() if not e.is_dir()]
        if len(entries) > P.TRAIN_BUNDLE_MAX_ENTRIES:
            h._json({
                "error": f"too many entries in ZIP (max "
                         f"{P.TRAIN_BUNDLE_MAX_ENTRIES})"
            }, 413)
            return

        validated: list[tuple[P.zipfile.ZipInfo, str, str]] = []  # (info, stem, ext)
        rejected: list[str] = []
        for info in entries:
            # Strip any leading path components — accept basename only.
            name = P.Path(info.filename).name
            if not name or not P.TRAIN_BUNDLE_NAME_RE.fullmatch(name):
                rejected.append(info.filename)
                continue
            stem = P.Path(name).stem
            ext = P.Path(name).suffix.lower()
            validated.append((info, stem, ext))

        # Partition into images + captions. Stems with the same
        # extension are deduped (last wins — the user uploaded a
        # newer copy on purpose).
        images_by_stem: dict[str, tuple[P.zipfile.ZipInfo, str]] = {}
        captions_by_stem: dict[str, P.zipfile.ZipInfo] = {}
        for info, stem, ext in validated:
            if ext == ".txt":
                captions_by_stem[stem] = info
            else:
                images_by_stem[stem] = (info, ext)

        # Apply the count cap. If the bundle has more images than
        # TRAIN_MAX_IMAGES, accept the first N by sorted stem so
        # the choice is deterministic.
        ds = P._train_dataset_dir(train_job_id)
        ds.mkdir(parents=True, exist_ok=True)
        images_dir = ds / "images"
        captions_dir = ds / "captions"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        existing = [x for x in images_dir.iterdir()
                    if x.is_file() and x.suffix.lower() in P.TRAIN_IMAGE_EXTS]
        next_idx = len(existing) + 1
        slots_left = P.TRAIN_MAX_IMAGES - len(existing)
        if slots_left <= 0:
            h._json({
                "error": f"already at {P.TRAIN_MAX_IMAGES} images; "
                         "remove some before adding more.",
                "job_id": train_job_id,
            }, 409)
            return

        ordered_stems = sorted(images_by_stem.keys())
        accepted_stems = ordered_stems[:slots_left]
        truncated = len(ordered_stems) > slots_left

        cmap = P._train_read_caption_map(ds)
        image_count = 0
        caption_count = 0
        paired_count = 0
        unpaired_warnings: list[str] = []

        # Pass 1 — images. Save renumbered, update caption_map.
        stem_to_saved: dict[str, str] = {}
        for stem in accepted_stems:
            info, ext = images_by_stem[stem]
            try:
                data = zf.read(info)
            except (KeyError, RuntimeError) as e:
                unpaired_warnings.append(f"read {info.filename}: {e}")
                continue
            saved_name = f"char_{next_idx:03d}{ext}"
            saved_stem = f"char_{next_idx:03d}"
            (images_dir / saved_name).write_bytes(data)
            cmap[stem] = saved_stem
            stem_to_saved[stem] = saved_stem
            image_count += 1
            next_idx += 1
        P._train_write_caption_map(ds, cmap)

        # Pass 2 — captions. Write under saved stem when paired,
        # otherwise under original stem (parked).
        for stem, info in captions_by_stem.items():
            try:
                data = zf.read(info)
            except (KeyError, RuntimeError) as e:
                unpaired_warnings.append(f"read {info.filename}: {e}")
                continue
            if len(data) > P.TRAIN_CAPTION_MAX_BYTES:
                unpaired_warnings.append(
                    f"{stem}.txt exceeds {P.TRAIN_CAPTION_MAX_BYTES} bytes — skipped")
                continue
            saved_stem = stem_to_saved.get(stem) or cmap.get(stem) or stem
            raw_text = P._decode_caption_bytes(data)
            normalised = P._normalise_caption_text(raw_text)
            (captions_dir / f"{saved_stem}.txt").write_text(
                normalised, encoding="utf-8")
            caption_count += 1
            if stem in stem_to_saved or stem in cmap:
                paired_count += 1
            else:
                unpaired_warnings.append(
                    f"{stem}.txt has no matching image yet (parked)")

        # Captions that pair via an image already on disk (not from
        # this bundle) — count those too via reconciliation.
        reconciled, total_imgs, recon_warnings = P._train_reconcile_captions(ds)
        unpaired_warnings.extend(recon_warnings)

        # Images uploaded WITHOUT a caption — surface as a warning.
        missing_caption_stems: list[str] = []
        for stem in accepted_stems:
            saved_stem = stem_to_saved[stem]
            if not (captions_dir / f"{saved_stem}.txt").is_file():
                missing_caption_stems.append(stem)
        if missing_caption_stems:
            unpaired_warnings.append(
                f"{len(missing_caption_stems)} image(s) without captions: "
                + ", ".join(missing_caption_stems[:5])
                + ("…" if len(missing_caption_stems) > 5 else ""))

        if rejected:
            unpaired_warnings.append(
                f"{len(rejected)} entries rejected (bad name/extension): "
                + ", ".join(rejected[:3])
                + ("…" if len(rejected) > 3 else ""))
        if truncated:
            unpaired_warnings.append(
                f"bundle had more than {P.TRAIN_MAX_IMAGES - len(existing)} "
                "image slots available; extras were ignored.")

        h._json({
            "ok": True,
            "job_id": train_job_id,
            "image_count": image_count,
            "caption_count": caption_count,
            "paired_count": reconciled,
            "total_images": total_imgs,
            "unpaired_warnings": unpaired_warnings,
            "min": P.TRAIN_MIN_IMAGES,
            "max": P.TRAIN_MAX_IMAGES,
        })
    except Exception as exc:
        h._json({"error": f"bundle upload failed: {exc}"}, 500)


# ====== Train Character — voice clip upload (optional, single file)
# Mirrors /train/upload but for the audio clip used by the optional
# voice-LoRA phase. Stores as `state/train_character/<job_id>/voice.<ext>`
# — single file per dataset, re-upload overwrites. Server-side
# duration validation is skipped on purpose (no ffprobe dependency);
# the UI's <audio controls> preview is the user's verification.
@post("/train/upload-voice")
def post_train_upload_voice(h, path, qs, ctype) -> None:
    # The chain arm carried this condition; its failure fell
    # through to the chain end, which answers 404.
    if not (ctype.startswith("multipart/form-data")):
        h.send_error(404)
        return
    try:
        clen = int(h.headers.get("Content-Length") or "0")
    except ValueError:
        clen = 0
    if clen <= 0:
        h._json({"error": "Content-Length required"}, 411); return
    if clen > P.TRAIN_VOICE_MAX_BYTES:
        h._json({
            "error": f"voice upload too large (max "
                     f"{P.TRAIN_VOICE_MAX_BYTES // (1024*1024)} MB)"
        }, 413)
        return
    try:
        form = P._parse_multipart_form(h.rfile, ctype, clen)
        # job_id is required: the voice clip must attach to an
        # existing dataset. The Train UI always uploads the first
        # image before exposing the voice section, so a job_id is
        # always available client-side.
        requested_id = (form.getvalue("job_id") or "").strip() if "job_id" in form else ""
        if not requested_id:
            h._json({
                "error": "job_id required — upload at least one "
                         "image first to create a dataset"
            }, 400)
            return
        try:
            train_job_id = P._safe_job_id(requested_id)
        except ValueError as e:
            h._json({"error": str(e)}, 400); return
        ds = P._train_dataset_dir(train_job_id)
        if not ds.is_dir():
            h._json({
                "error": f"dataset not found for job_id {train_job_id!r}"
            }, 404)
            return

        if "file" not in form:
            h._json({"error": "no field 'file'"}, 400); return
        fld = form["file"]
        if not getattr(fld, "filename", None):
            h._json({"error": "no filename"}, 400); return
        ext = P.Path(fld.filename).suffix.lower()
        if ext not in P.TRAIN_VOICE_EXTS:
            h._json({
                "error": f"unsupported audio type {ext!r}; want one of "
                         f"{sorted(P.TRAIN_VOICE_EXTS)}"
            }, 400)
            return

        # Single file per dataset — remove any prior `voice.*` so we
        # don't accumulate dead files when the user re-uploads with
        # a different extension.
        for prior_ext in P.TRAIN_VOICE_EXTS:
            prior = ds / f"voice{prior_ext}"
            if prior.is_file():
                try:
                    prior.unlink()
                except OSError:
                    pass

        dest = ds / f"voice{ext}"
        data = fld.file.read()
        if len(data) > P.TRAIN_VOICE_MAX_BYTES:
            h._json({
                "error": f"voice upload too large (max "
                         f"{P.TRAIN_VOICE_MAX_BYTES // (1024*1024)} MB)"
            }, 413)
            return
        dest.write_bytes(data)

        # Probe duration via ffprobe so we can surface the real
        # length to the UI (e.g. "14 s") AND reject obviously
        # broken clips before the user spends 15 minutes on a
        # training run that's going to fail at preprocess. The
        # trainer's audio decoder is ffmpeg-based and handles
        # MP3/M4A/FLAC natively, so we never need to convert to
        # WAV on the way in — only measure.
        duration_s: float | None = None
        try:
            if P.FFPROBE.is_file():
                out = P.subprocess.run(
                    [str(P.FFPROBE), "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     str(dest)],
                    capture_output=True, text=True, timeout=10,
                )
                if out.returncode == 0 and out.stdout.strip():
                    duration_s = float(out.stdout.strip())
        except (P.subprocess.TimeoutExpired, ValueError, OSError):
            duration_s = None

        if duration_s is not None and duration_s < P.TRAIN_VOICE_MIN_SECONDS:
            # Reject (and delete) — running the audio LoRA on a
            # 1-second clip produces nothing usable, and the
            # user gets a friendlier error here than from the
            # trainer 5 min into preprocess.
            try:
                dest.unlink()
            except OSError:
                pass
            h._json({
                "error": (
                    f"voice clip too short — measured "
                    f"{duration_s:.1f} s, need at least "
                    f"{P.TRAIN_VOICE_MIN_SECONDS} s (10–25 s recommended)"
                ),
            }, 400)
            return

        h._json({
            "ok": True,
            "job_id": train_job_id,
            "filename": dest.name,
            "path": str(dest),
            "size": dest.stat().st_size,
            "duration_seconds": duration_s,
            "min_seconds": P.TRAIN_VOICE_MIN_SECONDS,
            "max_seconds": P.TRAIN_VOICE_MAX_SECONDS,
        })
    except Exception as exc:
        h._json({"error": f"voice upload failed: {exc}"}, 500)
