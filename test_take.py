"""ONE TAKE: the plan arithmetic, the beats, the make_job mapping on both
engines, the estimate route, and the H3 runner with a stubbed engine."""
import json
import subprocess
from pathlib import Path

import pytest

import mlx_ltx_panel as p


def test_plan_h3_parts_and_ltx_frames():
    h = p.take_plan(60, "h3")
    assert h["beats"] == 12 and [len(x) for x in h["parts"]] == [3, 3, 3, 3]
    assert h["frames"] == 4 * 362
    assert p.take_plan(45, "h3")["parts"][-1] == [6, 7, 8]
    l = p.take_plan(30, "ltx")
    assert l["beats"] == 6 and len(l["parts"]) == 1 and l["frames"] == 721
    for s in p.TAKE_SECONDS:                       # every length is 8k+1 on LTX
        assert (p.take_plan(s, "ltx")["frames"] - 1) % 8 == 0
    assert p.take_plan(50, "h3") is None and p.take_plan("x", "h3") is None


def test_beats_normalise_from_json_or_lines():
    assert p.take_beats(json.dumps(["a", "b"]), 4) == ["a", "b", "", ""]
    assert p.take_beats("a\nb\n\nc", 3) == ["a", "b", ""]
    assert p.take_beats("", 2) == ["", ""]
    assert p.take_beats(["x"] * 5, 3) == ["x", "x", "x"]


def test_make_job_h3_take_forces_15s_parts_and_first_chain():
    j = p.make_job({"mode": "t2v", "engine": "h3", "prompt": "a hen skates",
                    "take_seconds": "60", "beats": json.dumps([f"b{i}" for i in range(12)]),
                    "h3_quality": "high", "h3_length": "3s"})
    q = j["params"]
    assert q["h3_length"] == "15s" and [c.split(" Continuity")[0] for c in q["h3_chain_prompts"]] == ["b0", "b1", "b2"]
    assert q["take"]["seconds"] == 60 and q["take"]["engine"] == "h3"
    assert q["take"]["beat_prompts"][11].startswith("b11 Continuity")


def test_make_job_ltx_take_is_the_windows_chain():
    j = p.make_job({"mode": "t2v", "engine": "ltx", "prompt": "x",
                    "take_seconds": "30", "beats": "one\ntwo\n"})
    q = j["params"]
    assert q["long_mode"] == "windows" and q["frames"] == 721
    assert [w.split(" Continuity")[0] for w in q["window_prompts"][:3]] == ["one", "two", ""]
    assert q["take"]["engine"] == "ltx"


def test_make_job_without_take_is_unchanged():
    q = p.make_job({"mode": "t2v", "engine": "ltx", "prompt": "x"})["params"]
    assert q["take"] is None and q["long_mode"] == "native"
    q2 = p.make_job({"mode": "t2v", "engine": "ltx", "prompt": "x", "take_seconds": "0"})["params"]
    assert q2["take"] is None


def test_estimate_route_registered_and_prices_h3():
    from panel import routes_meta  # noqa: F401
    from panel.routes import GET_ROUTES
    assert "/take/estimate" in GET_ROUTES
    m = p.take_estimate_minutes("h3", "high", 60)
    assert m is None or m > 30                      # four 15 s parts, never minutes
    assert p.take_estimate_minutes("ltx", "balanced", 60) is None


def _tiny_clip(path: Path, colour: str) -> None:
    subprocess.run([str(p.FFMPEG), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=64x64:d=0.5:r=24", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-t", "0.5", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def test_take_runner_chains_parts_and_joins(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"; out_dir.mkdir()
    monkeypatch.setattr(p, "OUTPUT", out_dir)
    monkeypatch.setattr(p, "STATE_DIR", tmp_path / "state")
    hidden = []
    monkeypatch.setattr(p, "set_hidden", lambda path, flag: hidden.append((path, flag)))
    seen = []

    def fake_h3(child):
        seen.append(dict(child["params"]))
        out = out_dir / f"{child['id']}.mp4"
        _tiny_clip(out, "red" if len(seen) == 1 else "blue")
        (out_dir / f"{child['id']}.mp4.json").write_text(json.dumps({"width": 64, "height": 64, "seed": 5}))
        child["output_path"] = str(out)
    monkeypatch.setattr(p, "run_h3_job_inner", fake_h3)
    monkeypatch.setattr(p, "take_drift", lambda *a, **k: {"ok": True, "delta": 0.0, "drifted": False})
    j = p.make_job({"mode": "t2v", "engine": "h3", "prompt": "a hen skates", "take_seconds": "30",
                    "beats": json.dumps(["b1", "b2", "b3", "b4", "b5", "b6"]), "preset_label": "ride"})
    p.run_take_job_inner(j)
    assert len(seen) == 2
    assert seen[0]["mode"] == "t2v" and [c.split(" Continuity")[0] for c in seen[0]["h3_chain_prompts"]] == ["b1", "b2", "b3"]
    assert seen[1]["mode"] == "i2v" and [c.split(" Continuity")[0] for c in seen[1]["h3_chain_prompts"]] == ["b4", "b5", "b6"]
    assert Path(seen[1]["image"]).is_file()          # the last frame of part 1
    assert seen[1]["take"] is None                   # a part is not a take
    final = Path(j["output_path"])
    assert final.is_file() and final.name.endswith("_take30s.mp4")
    side = json.loads(final.with_suffix(final.suffix + ".json").read_text())
    assert side["take"]["seconds"] == 30 and side["take"]["beats"][5].startswith("b6")
    assert len(side["take"]["parts"]) == 2 and side["width"] == 64
    assert [f for f, flag in hidden if flag] == side["take"]["parts"]


def test_take_runner_stops_between_parts(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"; out_dir.mkdir()
    monkeypatch.setattr(p, "OUTPUT", out_dir)
    monkeypatch.setattr(p, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(p, "set_hidden", lambda *a: None)

    def fake_h3(child):
        out = out_dir / f"{child['id']}.mp4"; _tiny_clip(out, "red"); child["output_path"] = str(out)
        job["cancel_requested"] = True                # Stop pressed during part 1
    monkeypatch.setattr(p, "run_h3_job_inner", fake_h3)
    job = p.make_job({"mode": "t2v", "engine": "h3", "prompt": "x", "take_seconds": "30"})
    with pytest.raises(RuntimeError, match="stopped"):
        p.run_take_job_inner(job)


# ---- the Storyboard door -------------------------------------------------------
import storyboard as sb


def _beat_shots(n):
    return [{"n": i + 1, "mode": "text", "engine": "h3", "prompt": f"beat {i + 1}", "duration_s": 5.0}
            for i in range(n)]


def test_collapse_take_keeps_one_shot_with_the_beats():
    t = sb.collapse_take(_beat_shots(6), 30)
    assert len(t) == 1
    s = t[0]
    assert s["n"] == 1 and s["take_seconds"] == 30 and s["duration_s"] == 30.0
    assert s["beats"] == [f"beat {i}" for i in range(1, 7)] and s["prompt"] == "beat 1"
    assert s["frames"] == 721
    short = sb.collapse_take(_beat_shots(4), 60)[0]          # planner wrote fewer than asked
    assert len(short["beats"]) == 12 and short["beats"][4] == ""
    assert sb.collapse_take([], 30) == []


def test_take_shot_becomes_take_fields_on_both_engines():
    t = sb.collapse_take(_beat_shots(6), 30)[0]
    j = sb.shot_to_job(t, {"quality": "balanced"}, h3_available=True, engine_mode="h3")
    assert j["engine"] == "h3" and j["take_seconds"] == "30"
    assert json.loads(j["beats"]) == t["beats"]
    j2 = sb.shot_to_job({**t, "engine": "ltx"}, {"quality": "balanced"}, h3_available=False)
    assert j2["engine"] == "ltx" and j2["take_seconds"] == "30" and j2["frames"] == 721
    # the panel's make_job then does what the Video tab would
    q = p.make_job({k: str(v) for k, v in j.items()})["params"]
    assert q["take"]["seconds"] == 30 and q["h3_length"] == "15s"


def test_take_shot_is_priced_by_parts_and_validates():
    t = sb.collapse_take(_beat_shots(12), 60)[0]
    secs = sb.shot_render_secs(t, {"quality": "balanced"}, h3_cost=lambda q, l: 3400.0 if l == "15s" else 1.0)
    assert secs == 4 * 3400.0
    b = sb.new_storyboard("sb_t", "t")
    b["shots"] = [t]
    errs = sb.validate_storyboard_detail(b)
    assert not [e for e in errs if e["code"] == "bad_duration"], errs
    # without the take flag the same 90 s shot is over the single-clip cap
    long = sb.collapse_take(_beat_shots(18), 90)[0]
    b["shots"] = [long]
    assert not [e for e in sb.validate_storyboard_detail(b) if e["code"] == "bad_duration"]
    b["shots"] = [{**long, "take_seconds": None}]
    assert [e for e in sb.validate_storyboard_detail(b) if e["code"] == "bad_duration"]


def test_take_concept_asks_for_beats_that_never_cut():
    c = p._sb_take_concept("a hen skateboards through the city", 60)
    assert "12 beats" in c and "never cuts" in c and "a hen skateboards" in c


# ---- continuity: the light lock and the drift retake ---------------------------
def test_light_lock_reads_time_and_weather_from_the_prompt():
    assert "still night" in p.take_light_lock("A hen skates through the city at night in the rain") 
    assert "still rain" in p.take_light_lock("A hen skates through the city at night in the rain")
    assert "exactly as before" in p.take_light_lock("A hen skates")
    q = p.make_job({"mode": "t2v", "engine": "h3", "prompt": "Broadway at night", "take_seconds": "30",
                    "beats": json.dumps(["a", "", "c", "d", "e", "f"])})["params"]
    assert q["take"]["beat_prompts"][0].endswith("season.") and "still night" in q["take"]["beat_prompts"][0]
    assert q["take"]["beat_prompts"][1] == ""                      # a blank beat still holds
    assert q["h3_chain_prompts"][0].startswith("a Continuity:")
    assert q["h3_chain_prompts"][1].startswith(p.TAKE_HOLD)          # ...spelled out for H3's runner


def _lit_clip(path: Path, first: str, last: str) -> None:
    # two-second clip whose colour goes from `first` to `last` (a light drift)
    subprocess.run([str(p.FFMPEG), "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={first}:s=64x64:d=1:r=24",
                    "-f", "lavfi", "-i", f"color=c={last}:s=64x64:d=1:r=24", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-t", "2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def test_drift_metric_sees_a_night_to_day_jump(tmp_path):
    _lit_clip(tmp_path / "steady.mp4", "0x202020", "0x242424")
    _lit_clip(tmp_path / "jump.mp4", "0x202020", "0xd0d0d0")
    assert p.take_drift(tmp_path / "steady.mp4")["drifted"] is False
    d = p.take_drift(tmp_path / "jump.mp4")
    assert d["drifted"] is True and d["delta"] > p.TAKE_DRIFT_MAX


def test_take_runner_retakes_a_drifting_part_once_and_keeps_the_steadier(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"; out_dir.mkdir()
    monkeypatch.setattr(p, "OUTPUT", out_dir)
    monkeypatch.setattr(p, "STATE_DIR", tmp_path / "state")
    hidden = []
    monkeypatch.setattr(p, "set_hidden", lambda path, flag: hidden.append(path))
    calls = []

    def fake_h3(child):
        calls.append(child["id"])
        out = out_dir / f"{child['id']}.mp4"
        if child["id"].endswith("-p1"):
            _lit_clip(out, "0x202020", "0xd0d0d0")                 # part 1 drifts
        else:
            _lit_clip(out, "0x202020", "0x262626")                 # the retake and part 2 hold
        (out_dir / f"{child['id']}.mp4.json").write_text("{}")
        child["output_path"] = str(out)
    monkeypatch.setattr(p, "run_h3_job_inner", fake_h3)
    j = p.make_job({"mode": "t2v", "engine": "h3", "prompt": "Broadway at night", "take_seconds": "30",
                    "beats": json.dumps(["b1", "b2", "b3", "b4", "b5", "b6"])})
    p.run_take_job_inner(j)
    assert calls == [f"{j['id']}-p1", f"{j['id']}-p1r", f"{j['id']}-p2"]
    side = json.loads(Path(j["output_path"]).with_suffix(".mp4.json").read_text())
    assert side["take"]["parts"][0].endswith("-p1r.mp4")            # the steadier retake is part 1
    assert str(out_dir / f"{j['id']}-p1.mp4") in hidden               # the drifting one is hidden
    assert j["take_drift"][0]["drifted"] is True
