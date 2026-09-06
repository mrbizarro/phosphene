#!/usr/bin/env python3
"""ONE SHOT inside a film.

The planner may choose a take — `take_seconds` (30 / 45 / 60 / 90 / 120) plus `beats`,
one per five seconds — for ONE scene of an otherwise ordinary multi-shot film, as a
cinematic tool (a walk-and-talk, a chase, a confession). Before this a take only reached
the board when the whole film was one (`collapse_take`). These tests drive the planner's
coercion, the board's validation / pricing / job mapping and the system prompt text with
canned model JSON: no model, no panel, no panel JavaScript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import storyboard as sb  # noqa: E402
import storyboard_planner as P  # noqa: E402


KW = dict(concept="a nurse's night shift", n_shots=2, storyboard_mod=sb)

BEATS = [
    "Night, flat fluorescent light. She walks straight toward the lens down the empty "
    "corridor, soles squeaking on the vinyl.",
    "A wheeled trolley rolls into frame from the left and she steps around it, its wheels "
    "rattling as it passes.",
    "The closed pharmacy hatch slides past on her right, the fluorescent hum steady.",
    "A man on a bench rises as she reaches him and falls into step beside her, a call bell "
    "ringing behind them.",
    "The corridor bends left and the double doors of the ward come into view; her pace slows.",
    "She pushes the doors open and stops in the doorway, the doors swinging behind her.",
]


def _normal(n: int, **kw) -> dict:
    d = {
        "n": n, "title": "The key", "character_id": None, "duration_s": 5,
        "camera": "static", "face": "none",
        "description": "Live-action, cinematic, a close-up of a brass key on a wet stone "
                       "step. Rain beads on the metal and one drop runs off the bow.",
        "settle": "the key lies still and wet",
        "soundscape": "Steady rain on stone, a distant gutter running.",
        "music": "N/A",
    }
    d.update(kw)
    return d


def _one_shot(n: int, **kw) -> dict:
    d = {
        "n": n, "title": "Down the corridor", "character_id": None, "duration_s": 30,
        "camera": "tracking", "face": "medium",
        "description": "Live-action, cinematic, a medium shot moving backwards ahead of a "
                       "nurse in pale blue scrubs as she walks a hospital corridor at night "
                       "to the double doors of a ward and pushes them open.",
        "settle": "she stands in the ward doorway, still",
        "soundscape": "Soft soles on vinyl flooring, the fluorescent hum, a distant call bell.",
        "music": "N/A",
        "take_seconds": 30,
        "beats": list(BEATS),
    }
    d.update(kw)
    return d


def _plan(*shots, **kw):
    args = dict(KW, n_shots=len(shots))
    args.update(kw)
    return P.coerce_spec({"title": "Night shift", "shots": list(shots)}, **args)


# ---- (a) a planner response with one normal shot and one One Shot -----------------

def test_one_shot_among_normal_shots_validates_and_keeps_its_take():
    spec, warns = _plan(_normal(1), _one_shot(2))
    assert len(spec["shots"]) == 2
    a, b = spec["shots"]
    # the normal shot is untouched by the feature
    assert "take_seconds" not in a and "beats" not in a and "beat_lines" not in a
    assert a["duration_s"] == 5.0
    # the One Shot carries its take, is as long as its take, and is ONE shot
    assert b["take_seconds"] == 30 and b["duration_s"] == 30.0 and b["frames"] == 721
    assert b["n"] == 2 and b["engine"] == "h3"
    assert len(b["beats"]) == 6 and all(b["beats"])
    assert b["beat_lines"] == BEATS
    # the prompt the panel reads first is the first beat
    assert b["prompt"] == b["beats"][0]
    # the board accepts it: no bad_duration, no bad_take_seconds, nothing at all
    assert sb.validate_storyboard(spec) == []
    assert not [w for w in warns if "take" in w.lower()]


def test_beats_are_assembled_in_the_register_with_the_settle_on_the_last_only():
    spec, _ = _plan(_normal(1), _one_shot(2))
    beats = spec["shots"][1]["beats"]
    # every beat can render a part on its own: the camera law and the soundscape ride on each
    for x in beats:
        assert x.startswith("integrated_multimodal_description: [Shot 1] ")
        assert "The camera tracks" in x and "\n\noverall_soundscape: " in x
    # ...but the end state only on the last one — an early settle would stop the take
    assert "completely finished before the shot ends" in beats[-1]
    assert not any("completely finished before the shot ends" in x for x in beats[:-1])
    # the LTX register, when the film is on LTX: prose, no H3 field labels
    spec2, _ = _plan(_normal(1), _one_shot(2), engine="ltx")
    for x in spec2["shots"][1]["beats"]:
        assert "integrated_multimodal_description" not in x and x.endswith(".")
    assert spec2["shots"][1]["duration_s"] == 30.0


def test_take_seconds_snaps_to_the_nearest_take_and_is_never_clamped():
    for raw, want in ((32, 30), ("60s", 60), (100, 90), (500, 120), (20, 30), ("45", 45)):
        for engine in ("h3", "ltx"):
            spec, warns = _plan(_normal(1), _one_shot(2, take_seconds=raw,
                                                      beats=["b"] * (want // 5)),
                                engine=engine)
            s = spec["shots"][1]
            assert s["take_seconds"] == want, (raw, engine)
            # not clamped to 60, not snapped onto H3's 3/5/10/15 grid
            assert s["duration_s"] == float(want), (raw, engine, s["duration_s"])
            assert s["engine"] == engine
    assert P._coerce_take_seconds(None) is None
    assert P._coerce_take_seconds(0) is None
    assert P._coerce_take_seconds("off") is None
    assert P._coerce_take_seconds(False) is None
    assert P._coerce_take_seconds(30) == 30


# ---- (b) beats padding / trimming ----------------------------------------------------

def test_beats_are_padded_and_trimmed_never_rejected():
    # too many: extras dropped
    spec, warns = _plan(_normal(1), _one_shot(2, beats=BEATS + ["one beat too many"]))
    s = spec["shots"][1]
    assert len(s["beats"]) == 6 and s["beat_lines"] == BEATS
    assert any("7 beats" in w and "extras dropped" in w for w in warns)
    # too few: the missing ones are blank, and stay blank once assembled
    spec, warns = _plan(_normal(1), _one_shot(2, beats=BEATS[:4]))
    s = spec["shots"][1]
    assert s["beat_lines"] == BEATS[:4] + ["", ""]
    assert s["beats"][4] == "" and s["beats"][5] == "" and all(s["beats"][:4])
    assert any("4 beats" in w and "previous moment" in w for w in warns)
    # the settle moved to the last WRITTEN beat
    assert "completely finished before the shot ends" in s["beats"][3]
    assert sb.validate_storyboard(spec) == []
    # shapes a small model actually produces
    assert P._coerce_beats("a\nb\n\nc", 3) == ["a", "b", ""]
    assert P._coerce_beats(["1. a", "Beat 2: b", "3) c"], 4) == ["a", "b", "c", ""]
    assert P._coerce_beats([{"text": "a"}, {"beat": "b"}, None], 3) == ["a", "b", ""]
    assert P._coerce_beats({"1": "a", "2": "b"}, 2) == ["a", "b"]
    assert P._coerce_beats(None, 2) == ["", ""]
    # the board's own shaping, for a hand-edited shot
    assert sb.take_beats_for({"take_seconds": 30, "beats": "a\nb"}) == ["a", "b", "", "", "", ""]
    assert sb.take_beats_for({"take_seconds": 30, "beats": ["x"] * 9}) == ["x"] * 6
    assert sb.take_beats_for({"take_seconds": 5, "beats": ["x"]}) == []
    assert sb.take_beats_for(_normal(1)) == []


def test_a_take_with_no_written_beat_is_an_ordinary_shot():
    spec, warns = _plan(_normal(1), _one_shot(2, beats=[]))
    s = spec["shots"][1]
    assert "take_seconds" not in s and "beats" not in s
    assert s["duration_s"] == 15.0            # H3's grid, as for any 30 s ordinary shot
    assert any("wrote no beats" in w for w in warns)
    spec, _ = _plan(_normal(1), _one_shot(2, beats=["", "", ""]))
    assert "take_seconds" not in spec["shots"][1]


# ---- (c) shot_to_job: the take fields go out for the One Shot and nothing else -------

def test_shot_to_job_posts_take_fields_for_the_one_shot_only():
    spec, _ = _plan(_normal(1), _one_shot(2))
    policy = {"quality": "balanced"}
    jobs = [sb.shot_to_job(s, policy, h3_available=True, board_id="sb1", board_title="Night")
            for s in sb.shooting_order(spec["shots"])]
    assert len(jobs) == 2                      # one job per shot: the take is not split here
    normal = next(j for j in jobs if j["preset_label"].startswith("S01"))
    take = next(j for j in jobs if j["preset_label"].startswith("S02"))
    for k in ("take_seconds", "beats"):
        assert k not in normal
    assert normal["h3_length"] == "5s" and normal["frames"] == 124
    assert take["take_seconds"] == "30"
    beats = json.loads(take["beats"])
    assert len(beats) == 6 and all(beats)
    assert beats[0].startswith("integrated_multimodal_description: [Shot 1] Night")
    assert take["engine"] == "h3" and take["h3_length"] == "15s"
    assert take["session_tag"] == "sb:sb1#2"


def test_shot_to_job_on_ltx_is_the_whole_take_and_never_the_windows_chain():
    spec, _ = _plan(_normal(1), _one_shot(2), engine="ltx")
    s = spec["shots"][1]
    j = sb.shot_to_job(s, {"quality": "balanced"}, h3_available=False, long_windows=True)
    assert j["engine"] == "ltx" and j["take_seconds"] == "30" and j["frames"] == 721
    assert len(json.loads(j["beats"])) == 6
    assert "temporal_mode" not in j and "window_invariants" not in j
    # the ordinary long shot next to it still gets the chain
    long_shot = dict(spec["shots"][0], duration_s=20.0)
    j2 = sb.shot_to_job(long_shot, {"quality": "balanced"}, h3_available=False, long_windows=True)
    assert j2.get("temporal_mode") == "windows"


def test_beats_are_padded_at_job_time_and_composed_with_the_place():
    shot = {"n": 2, "mode": "text", "engine": "h3", "prompt": "a", "duration_s": 30.0,
            "take_seconds": 30, "beats": ["a", "b"], "location_id": "ward"}
    locs = {"ward": sb.new_location("ward", "The ward", "a dim ward with one lit bed")}
    j = sb.shot_to_job(shot, {"quality": "quick"}, locations=locs)
    beats = json.loads(j["beats"])
    assert len(beats) == 6 and beats[2:] == ["", "", "", ""]
    # every written beat carries the location, because a later part renders from ITS beats
    assert all("a dim ward with one lit bed" in b for b in beats[:2])
    assert beats[0].startswith("a, ")
    # a One Shot with a cast character carries the trigger on every written beat
    cast = P._normalise_cast(["bizarrotrn"])
    spec, _ = _plan(_normal(1), _one_shot(2, character_id="bizarrotrn"), cast=cast)
    s = spec["shots"][1]
    assert s["engine"] == "ltx" and s["character_id"] == "bizarrotrn"
    assert all("bizarrotrn" in b for b in s["beats"])
    j3 = sb.shot_to_job(s, {"quality": "balanced"})
    assert all("bizarrotrn" in b for b in json.loads(j3["beats"]))
    assert j3["character_id"] == "bizarrotrn" and j3["take_seconds"] == "30"


# ---- pricing and parts on both engines -----------------------------------------------

def test_one_shot_is_priced_by_parts_on_both_engines():
    spec, _ = _plan(_normal(1), _one_shot(2))
    s = spec["shots"][1]
    assert sb.take_parts(s) == 2                        # 6 beats in threes on H3
    assert sb.shot_render_secs(s, {"quality": "balanced"},
                               h3_cost=lambda q, l: 1000.0 if l == "15s" else 1.0) == 2000.0
    ltx = dict(s, engine="ltx")
    assert sb.take_parts(ltx) == 3                      # 6 beats in twos on LTX
    assert sb.shot_render_secs(ltx, {"quality": "balanced"}) == 30 * 60.0
    # 45 s on LTX ends on a one-beat part; the parts still add up to the take
    ltx45 = dict(ltx, take_seconds=45, duration_s=45.0)
    assert sb.take_parts(ltx45) == 5
    assert sb.shot_render_secs(ltx45, {"quality": "quick"}) == 45 * 24.0
    # a stale duration_s does not change what the take costs
    assert sb.shot_render_secs(dict(ltx, duration_s=5.0), {"quality": "balanced"}) == 30 * 60.0
    assert sb.take_parts(_normal(1)) == 0
    # the film estimate counts the take once, in its engine's bucket, with the runtime it has
    board = sb.new_storyboard("sb_os", "Night shift", shots=spec["shots"])
    est = sb.estimate(board, pass_name="draft")
    assert est["shots"] == 2 and est["runtime_secs"] == 35


def test_validator_flags_an_illegal_take_length_but_never_the_beat_count():
    b = sb.new_storyboard("sb_v", "v")
    good = {"n": 1, "mode": "text", "engine": "h3", "prompt": "p", "duration_s": 30.0,
            "take_seconds": 30, "beats": ["a", "b"]}
    b["shots"] = [good]
    assert sb.validate_storyboard_detail(b) == []
    b["shots"] = [dict(good, take_seconds=40, duration_s=40.0)]
    codes = [e["code"] for e in sb.validate_storyboard_detail(b)]
    assert codes == ["bad_take_seconds"]
    for off in (None, 0, "0", "off", ""):
        b["shots"] = [dict(good, take_seconds=off, duration_s=5.0)]
        assert sb.validate_storyboard_detail(b) == [], off
    assert sb.take_seconds_of(good) == 30
    assert sb.take_seconds_of(dict(good, take_seconds="60")) == 60
    assert sb.take_seconds_of(dict(good, take_seconds=True)) is None
    assert sb.take_seconds_of(_normal(1)) is None


def test_collapse_take_keeps_a_nested_one_shots_beats_in_order():
    shots = [
        {"n": 1, "mode": "text", "engine": "h3", "prompt": "p1", "duration_s": 5.0},
        {"n": 2, "mode": "text", "engine": "h3", "prompt": "a", "duration_s": 30.0,
         "take_seconds": 30, "beats": ["a", "b", "c"]},
        {"n": 3, "mode": "text", "engine": "h3", "prompt": "p3", "duration_s": 5.0},
    ]
    t = sb.collapse_take(shots, 30)[0]
    assert t["beats"] == ["p1", "a", "b", "c", "p3", ""]
    assert t["take_seconds"] == 30 and t["duration_s"] == 30.0 and t["n"] == 1


# ---- (d) the system prompt carries the rule and the example --------------------------

def test_system_prompt_carries_the_one_shot_rule_and_the_example():
    for hint, chars in (("auto", False), ("auto", True), ("h3", False), ("ltx", True)):
        sp = P._build_system_prompt(hint, chars)
        assert "ONE SHOT - a take that never cuts, used as a cinematic tool" in sp, hint
        # what it is, when, when not, how many
        assert "single unbroken take of 30 to 120 seconds" in sp
        assert "walk-and-talk, a chase or a POV ride, a monologue or a" in sp
        assert "WRONG tool for a montage, for cross-cutting" in sp
        assert "Use it at most once" in sp and "never for the whole film" in sp
        # how to write it
        assert '"take_seconds"  one of 30, 45, 60, 90, 120' in sp
        assert "EXACTLY take_seconds / 5 strings" in sp
        assert "state the time of day and the weather ONCE" in sp
        # the contract admits the two extra keys
        assert "a One Shot adds two more" in sp
        # the example shot object, in the film's own JSON dialect
        assert '"take_seconds": 30,' in sp and '"beats": [' in sp
        assert '"title": "Down the corridor"' in sp
        assert sp.count("Night, flat fluorescent light.") == 1
        head = '"n": 3,\n  "title": "Down the corridor"'
        ex = sp[sp.index(head):]
        ex = ex[:ex.index("\n}\n") + 3]
        obj = json.loads("{" + ex)
        assert obj["n"] == 3
        assert obj["take_seconds"] == 30 and len(obj["beats"]) == 6
        assert obj["duration_s"] == obj["take_seconds"]
        for key in ("n", "title", "character_id", "duration_s", "camera", "face",
                    "description", "settle", "soundscape", "music"):
            assert key in obj, key
        # in the H3 register the example opens with the style token; on an
        # LTX-only film that word is the one the LTX rules forbid, so it is gone
        if hint == "ltx":
            assert "cinematic" not in obj["description"]
            assert obj["description"].startswith("A medium shot")
        else:
            assert obj["description"].startswith("Live-action, cinematic, a medium shot")


def test_model_view_shows_the_take_so_a_replan_keeps_it():
    spec, _ = _plan(_normal(1), _one_shot(2))
    a, b = spec["shots"]
    va, vb = P._shot_to_model_view(a), P._shot_to_model_view(b)
    assert "take_seconds" not in va and "beats" not in va
    assert vb["take_seconds"] == 30 and vb["beats"] == BEATS      # as written, not assembled
    # a re-plan that returns the take again keeps it, one that drops it loses it
    again, _ = P._coerce_for_mode(
        {"title": "Night shift", "shots": [_one_shot(2)]}, "shot", 2, spec,
        concept=KW["concept"], n_shots=2, style="", cast=[], board_id=spec["id"],
        engine="auto", tier="draft", duration_s=5.0, seed_base=1, max_dim=None, sb=sb)
    assert again["shots"][1]["take_seconds"] == 30 and len(again["shots"][1]["beats"]) == 6
    assert "take_seconds" not in again["shots"][0]


def test_face_law_applies_to_every_beat():
    bad = list(BEATS)
    bad[3] = "She turns and walks on, seen from behind, her back to the camera."
    spec, warns = _plan(_normal(1), _one_shot(2, beats=bad))
    s = spec["shots"][1]
    assert "seen from behind" not in s["beat_lines"][3]
    assert "back to the camera" not in s["beats"][3]
    assert any("beat 4" in w and "face-hiding" in w for w in warns)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
