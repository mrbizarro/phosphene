"""Anchor stills + long windows on the Storyboard, and the closed-tab push
alert. Pins: the plan switches persist on the board; a shot with a still
becomes an anchored i2v job; a long shot with the switch becomes a windows
chain with the board's style as invariant; the still prompt drops camera
moves; the still job form is an ordinary image job; the reconcile step folds
a finished still back; VAPID keys persist and push with nobody listening
sends nothing."""
import json
import os
import tempfile
from pathlib import Path

import pytest

import storyboard as sb
import mlx_ltx_panel as p


def test_shot_wants_still_only_for_text_and_character():
    assert sb.shot_wants_still({"mode": "text"})
    assert sb.shot_wants_still({"mode": "character"})
    assert not sb.shot_wants_still({"mode": "keyframe"})
    assert not sb.shot_wants_still({"mode": "extend"})
    assert not sb.shot_wants_still("nope")


def test_still_prompt_drops_camera_moves_keeps_the_scene():
    out = sb.still_prompt("A woman at a kitchen table, the camera slowly pushes in on her face, warm light.")
    assert "pushes" not in out and "camera" not in out
    assert "kitchen table" in out and "warm light" in out
    assert out.endswith("exactly as the shot opens.")


def test_shot_with_still_becomes_anchored_i2v():
    j = sb.shot_to_job({"n": 1, "mode": "text", "prompt": "x", "still": "/tmp/s.png", "duration_s": 4},
                       {"quality": "balanced", "width": 1024, "height": 576}, h3_available=False)
    assert j["mode"] == "i2v" and j["image"] == "/tmp/s.png"
    assert j["i2v_reference_mode"] == "anchor"


def test_shot_without_still_stays_t2v():
    j = sb.shot_to_job({"n": 1, "mode": "text", "prompt": "x", "duration_s": 4},
                       {"quality": "balanced"}, h3_available=False)
    assert j["mode"] == "t2v" and "image" not in j


def test_long_shot_with_switch_is_a_windows_chain():
    j = sb.shot_to_job({"n": 2, "mode": "text", "prompt": "x", "duration_s": 9, "location": "harbor"},
                       {"quality": "balanced"}, h3_available=False, long_windows=True, style="35mm, warm")
    assert j["frames"] > 121
    assert j["temporal_mode"] == "windows"
    assert j["window_invariants"] == "35mm, warm; harbor"


def test_long_shot_without_switch_or_short_shot_is_native():
    j = sb.shot_to_job({"n": 2, "mode": "text", "prompt": "x", "duration_s": 9},
                       {"quality": "balanced"}, h3_available=False)
    assert "temporal_mode" not in j
    j2 = sb.shot_to_job({"n": 2, "mode": "text", "prompt": "x", "duration_s": 3},
                        {"quality": "balanced"}, h3_available=False, long_windows=True)
    assert "temporal_mode" not in j2


def test_still_job_form_is_an_ordinary_image_job(monkeypatch):
    monkeypatch.setattr(p, "_character_sheet_png", lambda t: Path("/tmp/sheet.png") if t == "ana" else None)
    board = {"id": "b1", "title": "Test", "locations": {}, "wardrobe": {}}
    shot = {"n": 3, "mode": "character", "character_id": "ana", "prompt": "Ana smiles", "seed": 7}
    f = p._sb_still_job_form(shot, board, {"width": 720, "height": 1280})
    assert f["mode"] == "image" and f["n"] == "1" and f["seed"] == "7"
    assert f["aspect"] == "9:16"
    assert f["engine_override"] == "qwen_edit_inline"
    assert json.loads(f["refs"]) == ["/tmp/sheet.png"]
    assert f["session_tag"] == "sb:b1#3:still"
    # every key must be one make_job reads — build the job for real
    job = p.make_job({k: str(v) for k, v in f.items()})
    assert job["params"]["mode"] == "image"
    assert job["params"]["refs"] == ["/tmp/sheet.png"]
    assert job["params"]["engine_override"] == "qwen_edit_inline"


def test_still_job_form_without_sheet_uses_default_engine(monkeypatch):
    monkeypatch.setattr(p, "_character_sheet_png", lambda t: None)
    f = p._sb_still_job_form({"n": 1, "mode": "text", "prompt": "x"}, {"id": "b", "title": ""}, {"width": 1280, "height": 704})
    assert f["engine_override"] == "auto" and json.loads(f["refs"]) == [] and f["aspect"] == "16:9"


def test_still_aspect_is_the_nearest_engine_aspect(monkeypatch):
    monkeypatch.setattr(p, "_character_sheet_png", lambda t: None)
    def asp(w, h):
        return p._sb_still_job_form({"n": 1, "mode": "text", "prompt": "x"}, {"id": "b"}, {"width": w, "height": h})["aspect"]
    assert asp(640, 448) == "4:3"          # the draft canvas — not 16:9
    assert asp(1024, 576) == "16:9"
    assert asp(1280, 704) == "16:9"
    assert asp(768, 1024) == "3:4"
    assert asp(1024, 1024) == "1:1"
    assert asp(1280, 544) == "21:9"


def test_reconcile_folds_a_finished_still(monkeypatch):
    monkeypatch.setattr(p, "_sb_job_index", lambda: {
        "j1": {"status": "done", "output_path": "/out/still.png"},
        "j2": {"status": "failed", "error": "boom"},
    })
    board = {"shots": [{"n": 1, "still_job_id": "j1"}, {"n": 2, "still_job_id": "j2"}]}
    assert p._sb_reconcile(board) is True
    assert board["shots"][0]["still"] == "/out/still.png"
    assert board["shots"][1].get("still") is None
    assert board["shots"][1]["still_error"] == "boom"


def test_vapid_keys_persist_and_push_without_listeners_sends_nothing(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(p, "STATE_DIR", Path(d))
        k1 = p._vapid_keys()
        assert k1 and k1["public"] and k1["private"]
        assert p._vapid_keys() == k1
        assert (Path(d) / "vapid.json").is_file()
        assert p._push_subs() == []
        assert p.push_notify("t", "b") == 0
        p._push_save_subs([{"endpoint": "https://x/1", "keys": {"p256dh": "a", "auth": "b"}}])
        assert len(p._push_subs()) == 1


def test_push_routes_are_registered():
    from panel import routes_meta  # noqa: F401
    from panel.routes import GET_ROUTES, POST_ROUTES
    assert "/sw.js" in GET_ROUTES and "/push/key" in GET_ROUTES
    for r in ("/push/subscribe", "/push/unsubscribe", "/push/test"):
        assert r in POST_ROUTES
    assert (p.ROOT / "webapp" / "sw.js").is_file()


def test_loras_update_and_guide_routes_are_registered():
    from panel import routes_loras  # noqa: F401
    from panel.routes import GET_ROUTES, POST_ROUTES
    assert "/loras/updates" in GET_ROUTES
    assert "/loras/guide" in POST_ROUTES


def test_sidecar_guide_survives_the_reader(tmp_path):
    lp = tmp_path / "x.safetensors"
    lp.write_bytes(b"0" * 2048)
    (tmp_path / "x.json").write_text(json.dumps({"name": "X", "guide": "Use it at 0.8 for a painterly look."}))
    meta = p._read_lora_sidecar(lp)
    assert meta["guide"] == "Use it at 0.8 for a painterly look."
    assert p._read_lora_sidecar(tmp_path / "missing.safetensors")["guide"] == ""


def test_clear_still_forgets_the_still_and_the_clip_made_from_it():
    board = {"shots": [{"n": 1, "still": "/a.png", "still_job_id": "j", "draft_job_id": "d",
                        "draft_output": "/o.mp4", "status": "done"},
                       {"n": 2, "still": "/b.png"}]}
    shot = p._sb_clear_still(board, 1)
    assert shot is board["shots"][0]
    for k in ("still", "still_job_id", "still_error", "draft_job_id", "draft_output"):
        assert k not in shot
    assert shot["status"] == "pending"
    assert board["shots"][1]["still"] == "/b.png"          # only shot 1
    assert p._sb_clear_still(board, 9) is None
