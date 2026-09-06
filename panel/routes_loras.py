"""/loras family routes — moved out of the chain (slice 4).

Bodies are verbatim from mlx_ltx_panel.py's do_GET/do_POST chains except
the two mechanical renames the move forces: `self` -> `h`, and panel
globals -> `P.<name>`. See panel/routes_stats.py for the pattern and
panel/__init__.py for why P is assigned rather than imported.
"""
from __future__ import annotations

from urllib.parse import parse_qs

from panel.routes import get, post

P = None  # the running mlx_ltx_panel module; assigned at wiring time


@get("/civitai/test")
def get_civitai_test(h, parsed) -> None:
    # Sanity-check the saved CivitAI key by hitting an
    # auth-required endpoint and reporting back the upstream
    # status. Lets the Settings UI render a green/red dot
    # without users having to risk a 300 MB download just to
    # discover the key is malformed.
    key = P._active_civitai_key()
    if not key:
        h._json({"ok": False, "error": "No CivitAI key configured."}, 400)
        return
    try:
        # /api/v1/me requires auth; success returns the user
        # profile, failure returns 401. We never echo the
        # username back — just enough info to tell the user
        # the key works.
        P._civitai_request("/me", timeout=10)
        h._json({"ok": True, "message": "CivitAI auth works."})
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg:
            h._json({
                "ok": False,
                "error": "Key rejected by CivitAI (401/403). "
                         "Re-paste the key and try again, or generate a new one.",
            }, 401)
        else:
            h._json({
                "ok": False,
                "error": f"Network error reaching CivitAI: {msg[:200]}",
            }, 502)


@get("/hf/test")
def get_hf_test(h, parsed) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    # Same idea for Hugging Face — call /api/whoami-v2 which
    # is auth-required.
    token = P._active_hf_token()
    if not token:
        h._json({"ok": False, "error": "No Hugging Face token configured."}, 400)
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}",
                     "User-Agent": P.CIVITAI_USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
        # Don't echo the username — just confirm.
        h._json({"ok": True, "message": "Hugging Face auth works."})
    except urllib.request.HTTPError as he:
        if he.code in (401, 403):
            h._json({
                "ok": False,
                "error": "Token rejected by Hugging Face (401/403). "
                         "Re-paste the token, or generate a new one with read access.",
            }, 401)
        else:
            h._json({
                "ok": False,
                "error": f"HTTP {he.code} reaching Hugging Face.",
            }, 502)
    except Exception as exc:
        h._json({
            "ok": False,
            "error": f"Network error reaching Hugging Face: {str(exc)[:200]}",
        }, 502)


@get("/loras")
def get_loras(h, parsed) -> None:
    # Returns: { user: [user-installed], curated: [Lightricks
    # officials minus hdr_toggle entries], loras_dir: <abs path>,
    # civitai_auth: bool, mode_filter: <echoed>,
    # exclude_characters: <echoed> }.
    # The HDR-toggle special-case is filtered out of `curated`
    # because the UI exposes it as a plain checkbox elsewhere —
    # showing it in the picker would just confuse users.
    #
    # Optional `?mode=<tag>` filter (added with the unified
    # picker — see _classify_lora_modes). Filters the user list
    # down to LoRAs whose `compatible_modes` includes the tag
    # OR includes "unknown" (so unclassified LoRAs show up in
    # every filter — better than hiding them).
    #
    # Optional `?exclude_characters=1` filter (added 2026-05-17
    # with the Manual-tab Characters picker). When on, drops every
    # entry that `_is_character_lora()` flags as part of a trained
    # character bundle (face / audio / voice or kind=train_character
    # sidecar). The Manual-tab style picker passes this so the
    # Characters picker is the only surface where face+audio LoRAs
    # live — no confusing duplicates in two pickers at once.
    #
    # Two LIBRARIES are returned as one list (2026-08-09): the panel's
    # own mlx_models/loras/ tree and the Hailuo pack's loras/ dir, each
    # entry tagged `lane`. The browser fetches this endpoint ONCE with
    # no filter and re-filters client-side on every mode/engine flip,
    # so both lanes have to be present in that one payload or an engine
    # switch would need a round-trip to show anything.
    qs = P.parse_qs(parsed.query)
    mode_filter = (qs.get("mode", [""])[0] or "").strip().lower()
    exclude_characters_raw = (qs.get("exclude_characters", [""])[0] or "").strip().lower()
    exclude_characters = exclude_characters_raw in ("1", "true", "yes", "on")
    try:
        h3_user_loras = P.list_h3_user_loras()
    except Exception:
        h3_user_loras = []
    user_loras = P.list_user_loras() + h3_user_loras
    if mode_filter:
        # The lane is checked BEFORE compatible_modes, and the
        # "unknown" wildcard is deliberately NOT honoured across lanes.
        # An untagged .safetensors in mlx_models/loras/ classifies as
        # ["unknown"], which is permissive by design — but permissive
        # must not mean "offer it to an engine that cannot load it".
        # The directory a file was found in is the authority here.
        if mode_filter == "video:h3":
            user_loras = [l for l in user_loras if l.get("lane") == "h3"]
        else:
            user_loras = [
                l for l in user_loras
                if l.get("lane") != "h3"
                and l.get("ltx_compatible") is not False
                and (mode_filter in (l.get("compatible_modes") or [])
                     or "unknown" in (l.get("compatible_modes") or []))
            ]
    if exclude_characters:
        blocked_names = P._character_lora_basenames()
        user_loras = [
            l for l in user_loras
            if not P._is_character_lora(l, blocked_names)
        ]
    curated = [c for c in P.list_curated_loras()
               if not c.get("is_hdr_toggle")
               and not c.get("is_restore_lora")
               and not c.get("is_ingredients_lora")
               and not c.get("is_control_lora")]
    h._json({
        "user": user_loras,
        "curated": curated,
        "loras_dir": str(P._safe_loras_dir()),
        # The H3 lane's directory + its gate. `h3_lora_supported` is
        # whether the INSTALLED runner takes `--lora` at all — an older
        # H3 pack renders every tier fine and can't take an adapter, so
        # the picker must not be offered on that install. Reported here
        # (not just in /status) so the CivitAI modal can name the right
        # install path without a second round-trip.
        "h3_loras_dir": str(P._h3_loras_dir()),
        "h3_lora_supported": P.h3_supports_lora(),
        "h3_lora_max_stack": P.H3_LORA_MAX_STACK,
        # Echo back so the JS picker can verify the server filtered
        # by the mode it asked for (vs back-compat unfiltered).
        "mode_filter": mode_filter or None,
        "exclude_characters": bool(exclude_characters),
        # True iff a CivitAI key is configured. Source of truth:
        # the saved panel settings first, falling back to the
        # env var if a power user prefers shell-level config.
        "civitai_auth": bool(P._active_civitai_key()),
        # Same pattern for HF — used for gated repo downloads
        # (HDR LoRA, etc.).
        "hf_auth": bool(P._active_hf_token()),
    })


@get("/loras/download")
def get_loras_download(h, parsed) -> None:
    # Stream a user-installed LoRA back to the browser with
    # Content-Disposition: attachment so the browser pops the save
    # dialog instead of trying to render the safetensors bytes.
    # Use case (Mr Bizarro 2026-05-17): users want to back up NSFW or
    # personal LoRAs before deleting them from the panel, or move
    # them to another machine. Path must resolve under the loras
    # dir and end in .safetensors — same bounds the /loras/delete
    # POST enforces.
    qs = P.parse_qs(parsed.query)
    try:
        lp = P.Path(qs.get("path", [""])[0]).resolve()
    except Exception:
        h.send_error(400); return
    try:
        base = P._safe_loras_dir().resolve()
    except OSError:
        h.send_error(500); return
    if not lp.is_relative_to(base) or not lp.is_file():
        h.send_error(404); return
    if lp.suffix.lower() != ".safetensors":
        h.send_error(400); return
    try:
        size = lp.stat().st_size
    except OSError:
        h.send_error(404); return
    # Filename for the Content-Disposition header — keep it ASCII-safe
    # since the header doesn't survive non-ASCII without RFC 5987
    # encoding, which most browsers handle but adds noise. The on-disk
    # name is already URL-safe by virtue of having been written by
    # us or by CivitAI's slugified filenames.
    safe_name = lp.name.replace('"', '')
    h.send_response(200)
    h.send_header("Content-Type", "application/octet-stream")
    h.send_header(
        "Content-Disposition",
        f'attachment; filename="{safe_name}"',
    )
    h.send_header("Content-Length", str(size))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    # Stream in 1 MiB chunks so we don't blow RAM on a multi-GB LoRA.
    CHUNK = 1024 * 1024
    with lp.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            try:
                h.wfile.write(buf)
            except (BrokenPipeError, ConnectionResetError):
                # Client closed mid-download — fine, nothing to clean up.
                return


@get("/civitai/search")
def get_civitai_search(h, parsed) -> None:
    # Proxy CivitAI's API. Filtering down to LTX-Video LoRAs by
    # baseModel ("LTXV 2.3" is the canonical string used on
    # civitai.com for LTX-2.3 LoRAs as of 2026-05). Returns the
    # subset of fields the panel cares about, plus a flat
    # download_url that points at the .safetensors directly.
    qs = P.parse_qs(P.urlparse(h.path).query)
    query = qs.get("query", [""])[0]
    nsfw = (qs.get("nsfw", ["false"])[0] or "false").lower() == "true"
    cursor = qs.get("cursor", [""])[0]
    limit = max(1, min(50, int(qs.get("limit", ["20"])[0] or "20")))
    # context="image" filters the API to Qwen + HiDream base
    # models for the Images workflow; default "video" keeps the
    # LTX-2.3 result set for the Video workflow.
    context = (qs.get("context", ["video"])[0] or "video").lower()
    if context not in P._CIVITAI_BASE_MODELS_BY_CONTEXT:
        context = "video"
    # Optional `family` narrows a context to one engine family —
    # "ltx" / "h3" on video, "qwen" / "hidream" on image. Empty /
    # "all" leaves the whole context list active. An unknown value
    # is ignored by _civitai_search rather than erroring, so a stale
    # tab degrades to "All" instead of to a 400.
    family = (qs.get("family", [""])[0] or "").lower()
    try:
        results = P._civitai_search(query=query, nsfw=nsfw,
                                 cursor=cursor, limit=limit,
                                 context=context, family=family)
        results["context"] = context
        results["family"] = family or "all"
        # Echo the family list so the client can render pills
        # without hardcoding the catalog. Every context that HAS
        # families gets them — video grew a pair (ltx / h3) when the
        # H3 engine learned to take LoRAs, and the client renders
        # both surfaces from this one echo.
        _fams = P._CIVITAI_FAMILIES_BY_CONTEXT.get(context) or {}
        if _fams:
            results["available_families"] = list(_fams.keys())
        h._json(results)
    except Exception as exc:
        h._json({"error": f"civitai search failed: {exc}",
                    "items": []}, 502)


@get("/loras/updates")
def get_loras_updates(h, parsed) -> None:
    """Which installed CivitAI LoRAs have a newer version on CivitAI.

    One request per LoRA that carries a `civitai_id` in its sidecar; the
    newest `modelVersions[0]` is compared with the sidecar's
    `civitai_version_id`. Each item carries what /civitai/download needs
    (download_url + meta), so the update is the same one-click install the
    browser tab does — the new file lands beside the old one."""
    rows = []
    try:
        rows = [r for r in P.list_user_loras() if r.get("civitai_id")]
    except Exception:                                              # noqa: BLE001
        rows = []
    items, errors = [], []
    for r in rows:
        mid = r.get("civitai_id")
        try:
            m = P._civitai_request(f"/models/{mid}", timeout=12.0)
        except Exception as exc:                                   # noqa: BLE001
            errors.append({"path": r.get("path"), "error": str(exc)[:160]})
            continue
        versions = [v for v in (m.get("modelVersions") or []) if isinstance(v, dict)]
        if not versions:
            continue
        latest = versions[0]
        have = r.get("civitai_version_id")
        try:
            newer = int(latest.get("id") or 0) != int(have or 0)
        except (TypeError, ValueError):
            newer = str(latest.get("id")) != str(have)
        if not newer:
            continue
        files = latest.get("files") or []
        primary = (next((f for f in files if f.get("primary") and
                         str(f.get("name", "")).endswith(".safetensors")), None)
                   or next((f for f in files if str(f.get("name", "")).endswith(".safetensors")), None))
        if not primary or not primary.get("downloadUrl"):
            continue
        images = latest.get("images") or []
        preview = next((i.get("url") for i in images if i.get("url")), None)
        items.append({
            "path": r.get("path"), "name": r.get("name"),
            "current_version_id": have, "latest_version_id": latest.get("id"),
            "latest_version_name": latest.get("name"),
            "published_at": latest.get("publishedAt") or latest.get("createdAt"),
            "base_model": latest.get("baseModel"),
            "download_url": primary.get("downloadUrl"),
            "meta": {
                "id": m.get("id"), "version_id": latest.get("id"),
                "name": m.get("name") or r.get("name"),
                "description": P.re.sub(r"<[^>]+>", " ", str(m.get("description") or ""))[:600].strip(),
                "preview_url": preview,
                "filename": primary.get("name"),
                "download_url": primary.get("downloadUrl"),
                "trigger_words": list(latest.get("trainedWords") or []),
                "base_model": latest.get("baseModel"),
                "civitai_url": f"https://civitai.com/models/{m.get('id')}",
            },
        })
    h._json({"ok": True, "checked": len(rows), "items": items, "errors": errors})


@post("/loras/guide")
def post_loras_guide(h, path, qs, ctype) -> None:
    """Write a one-paragraph guide for an installed LoRA and keep it in the
    sidecar under `guide`. The planner model writes it from what the sidecar
    already knows (name, description, trigger words, base model, strength) —
    nothing is fetched. Refused while a render runs: the planner is a 12B
    model and the GPU is taken."""
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    lp = form.get("path", [""])[0] if isinstance(form.get("path"), list) else (form.get("path") or "")
    row = next((r for r in P.list_user_loras() if r.get("path") == lp), None)
    if not row:
        h._json({"ok": False, "error": "that LoRA is not in the list"}, 404)
        return
    if P.STATE.get("current"):
        h._json({"ok": False, "error": "a render is running — write the guide when it finishes"}, 409)
        return
    trig = ", ".join(row.get("trigger_words") or []) or "none"
    system = ("You write short, practical guides for video-model LoRA adapters. Plain English, "
              "second person, no marketing. One paragraph of 3 to 5 sentences: what the LoRA "
              "does to the picture, how to prompt it (name the trigger words if there are any, "
              "and say when to leave them out), what strength to start from and what happens "
              "above and below it, and one thing it is bad at or fights with. No headings, no lists.")
    user = (f"LoRA name: {row.get('name')}\n"
            f"Base model: {row.get('base_model') or 'unknown'}\n"
            f"Trigger words: {trig}\n"
            f"Recommended strength: {row.get('recommended_strength', 1.0)}\n"
            f"Author's description: {(row.get('description') or '').strip()[:1200] or 'none given'}\n"
            "Write the guide.")
    session = None
    try:
        session = P.storyboard_planner.PlannerSession()
        resp = session.generate(system, user, max_tokens=320, temperature=0.4)
        text = str((resp or {}).get("text") or "").strip()
    except Exception as exc:                                       # noqa: BLE001
        h._json({"ok": False, "error": f"the planner could not write it: {exc}"}, 502)
        return
    finally:
        for m in ("release", "close", "stop", "shutdown"):
            if session is not None and hasattr(session, m):
                try:
                    getattr(session, m)()
                except Exception:                                  # noqa: BLE001
                    pass
                break
    text = P.re.sub(r"\s+", " ", text).strip().strip('"')
    if len(text) < 40:
        h._json({"ok": False, "error": "the planner returned nothing usable"}, 502)
        return
    sidecar = P.Path(lp).with_suffix(".json")
    try:
        raw = P.json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, ValueError):
        raw = {}
    raw["guide"] = text
    raw.setdefault("name", row.get("name"))
    try:
        sidecar.write_text(P.json.dumps(raw, indent=2, ensure_ascii=False))
    except OSError as exc:
        h._json({"ok": False, "error": f"could not save the guide: {exc}"}, 500)
        return
    h._json({"ok": True, "guide": text})


@post("/loras/refresh")
def post_loras_refresh(h, path, qs, ctype) -> None:
    # Rescan mlx_models/loras/. The result is whatever
    # list_user_loras returns — filesystem is the source of
    # truth, no caching layer to invalidate.
    h._json({
        "ok": True,
        "user": P.list_user_loras(),
        "loras_dir": str(P._safe_loras_dir()),
    })


@post("/loras/delete")
def post_loras_delete(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    # Remove a user-installed LoRA (the .safetensors file plus
    # its sidecar JSON if present). Path must be inside the
    # loras dir — we resolve and bound-check to prevent
    # path-traversal mischief from a hostile form payload.
    target = form.get("path", [""])[0] or form.get("path", "")
    if isinstance(target, list): target = target[0] if target else ""
    try:
        p = P.Path(target).resolve()
        base = P._safe_loras_dir().resolve()
        if not p.is_relative_to(base) or not p.is_file():
            raise RuntimeError("path not inside loras dir")
        if p.suffix.lower() != ".safetensors":
            raise RuntimeError("not a safetensors file")
        p.unlink()
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            sidecar.unlink()
        h._json({"ok": True, "removed": str(p)})
    except Exception as exc:
        h._json({"ok": False, "error": str(exc)}, 400)


@post("/loras/rename")
def post_loras_rename(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    # Update the display name of a user-installed LoRA. We only
    # touch the sidecar JSON's `name` field — the .safetensors
    # filename is left alone so any saved job that references the
    # old path keeps working. Creates a minimal sidecar if one
    # doesn't exist yet (e.g. for hand-dropped .safetensors files
    # that have no metadata).
    target = form.get("path", [""])[0] or form.get("path", "")
    if isinstance(target, list): target = target[0] if target else ""
    new_name = form.get("name", [""])[0] or form.get("name", "")
    if isinstance(new_name, list): new_name = new_name[0] if new_name else ""
    new_name = str(new_name).strip()
    if not new_name:
        h._json({"ok": False, "error": "name is required"}, 400)
        return
    # Cap length — UI shows ~30 chars before truncating, 120 leaves
    # plenty of room for descriptive renames without inviting abuse.
    if len(new_name) > 120:
        h._json({"ok": False, "error": "name too long (max 120 chars)"}, 400)
        return
    try:
        p = P.Path(target).resolve()
        base = P._safe_loras_dir().resolve()
        if not p.is_relative_to(base) or not p.is_file():
            raise RuntimeError("path not inside loras dir")
        if p.suffix.lower() != ".safetensors":
            raise RuntimeError("not a safetensors file")
        sidecar = p.with_suffix(".json")
        # Read existing sidecar (merge-update) or start fresh.
        payload: dict = {}
        if sidecar.exists():
            try:
                payload = P.json.loads(sidecar.read_text()) or {}
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                # Bad JSON on disk — overwrite rather than refuse
                # the rename. The user clearly wants this LoRA to
                # have a clean name; a corrupt sidecar shouldn't
                # block that.
                payload = {}
        payload["name"] = new_name
        P.atomic_write_text(sidecar, P.json.dumps(payload, indent=2))
        h._json({"ok": True, "name": new_name, "path": str(p)})
    except Exception as exc:
        h._json({"ok": False, "error": str(exc)}, 400)


@get("/hf/loras")
def get_hf_loras(h, parsed) -> None:
    """A Hugging Face org's LoRAs for one lane, in the CivitAI grid's shape.
    `?source=playtime&lane=h3|ltx&q=<name filter>&refresh=1`."""
    qs = P.parse_qs(parsed.query)
    source = (qs.get("source", ["playtime"])[0] or "playtime").strip().lower()
    lane = (qs.get("lane", ["h3"])[0] or "h3").strip().lower()
    q = (qs.get("q", [""])[0] or "").strip().lower()
    if source not in P.HF_LORA_SOURCES or lane not in ("h3", "ltx"):
        h._json({"ok": False, "error": "unknown source or lane", "items": []}, 400)
        return
    try:
        items = P.hf_lora_catalog(source, lane, force=qs.get("refresh", ["0"])[0] == "1")
    except Exception as exc:                                       # noqa: BLE001
        h._json({"ok": False, "error": f"Hugging Face could not be reached: {exc}", "items": []}, 502)
        return
    if q:
        items = [i for i in items if q in i["name"].lower() or q in i["id"].lower()]
    src = P.HF_LORA_SOURCES[source]
    h._json({"ok": True, "source": source, "label": src["label"], "url": src["url"],
                "lane": lane, "items": items, "has_more": False})


@post("/hf/loras/download")
def post_hf_loras_download(h, path, qs, ctype) -> None:
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    def f(k):
        v = form.get(k, "")
        return (v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")) or ""
    repo, filename = f("repo").strip(), f("filename").strip()
    try:
        meta = P.json.loads(f("meta") or "{}")
    except P.json.JSONDecodeError:
        meta = {}
    if not repo or "/" not in repo or not filename:
        h._json({"ok": False, "error": "repo and filename are required"}, 400)
        return
    try:
        h._json(P._hf_lora_download(repo, filename, meta if isinstance(meta, dict) else {}))
    except Exception as exc:                                       # noqa: BLE001
        h._json({"ok": False, "error": str(exc)}, 400)


@post("/civitai/download")
def post_civitai_download(h, path, qs, ctype) -> None:
    P._analytics_feature("civitai_download")
    _rb = h._read_form_body()
    if _rb is None:
        return
    body, form = _rb
    # Triggers a download of a CivitAI LoRA into mlx_models/loras/.
    # Streams progress through STATE['log'] like the model
    # downloads do. Validates the requested URL points at
    # civitai.com to prevent the endpoint being weaponized as
    # a generic HTTP fetcher.
    url = form.get("download_url", [""])[0] or form.get("download_url", "")
    if isinstance(url, list): url = url[0] if url else ""
    try:
        meta_raw = form.get("meta", [""])[0] or form.get("meta", "")
        if isinstance(meta_raw, list): meta_raw = meta_raw[0] if meta_raw else ""
        meta = P.json.loads(meta_raw) if meta_raw else {}
    except P.json.JSONDecodeError:
        meta = {}
    try:
        result = P._civitai_download(url, meta)
        h._json({"ok": True, **result})
    except Exception as exc:
        h._json({"ok": False, "error": str(exc)}, 400)


# H3 LoRA import. This is deliberately a separate endpoint from
# /upload: reference media can be any image/audio, whereas an H3
# adapter must pass its model-layout gate BEFORE it lands in the picker.
@post("/h3/loras/import")
def post_h3_loras_import(h, path, qs, ctype) -> None:
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
        h._json({"ok": False, "error": "Content-Length required"}, 411)
        return
    if clen > P.H3_LORA_UPLOAD_MAX_BYTES:
        h._json({"ok": False,
                    "error": f"upload too large (max "
                             f"{P.H3_LORA_UPLOAD_MAX_BYTES // (1024 * 1024)} MB)"}, 413)
        return
    # The response is composed FIRST and written LAST, because the
    # request body has to leave the socket before the reply goes onto
    # it. Answering a refusal early and returning would strand the
    # remaining megabytes unread — harmless under the HTTP/1.0 close
    # this handler happens to run today, the head of the next response
    # the day anyone turns keep-alive on.
    drain = None
    status, payload = 500, {"ok": False, "error": "import failed"}
    try:
        # STREAMED, not buffered: `_parse_multipart_form` would put the
        # whole adapter in memory several times over (see
        # H3_LORA_UPLOAD_MAX_BYTES). This walks to the file part and
        # hands back a drain that writes straight into the staging
        # file, so a 2 GB import costs one 1 MiB chunk of RSS.
        filename, drain, _ = P._stream_multipart_file_part(
            h.rfile, ctype, clen, "file",
            chunk=P.H3_LORA_STREAM_CHUNK)
        payload = {"ok": True, **P.import_h3_lora_staged(
            filename,
            lambda fh: drain(fh, P.H3_LORA_UPLOAD_MAX_BYTES),
            size_hint=clen)}
        status = 200
    except P.MultipartTooLarge as exc:
        status, payload = 413, {"ok": False, "error": str(exc)}
    except (ValueError, FileExistsError, RuntimeError) as exc:
        status, payload = 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        status = 500
        payload = {"ok": False, "error": f"import failed: {exc}"}
    try:
        if drain is None:
            # The framing itself was unreadable, so there is no safe
            # resynchronisation point. Close rather than guess.
            raise RuntimeError("no drain")
        drain.discard()
    except Exception:
        h.close_connection = True
    h._json(payload, status)
