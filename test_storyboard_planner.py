#!/usr/bin/env python3
"""Tests for storyboard_planner.py.

Two layers, deliberately separated:

  * The STUB layer (default, no GPU, no weights, runs anywhere in <1 s) drives the whole
    extraction / coercion / repair / validation pipeline through a fake PlannerSession that
    replays canned model output. Every ugly thing a 4-bit 12B has actually been observed to
    do — fenced JSON, prose around the object, a whole three-field prompt pasted into one
    key, a bad character name, an out-of-range duration, truncation mid-object — is a test
    case here rather than a surprise in front of a user.

  * The LIVE layer (opt-in) loads the real planner model, plans a fixed 6-shot concept,
    asserts it validates, and prints the measured plan-phase RSS. It is skipped unless
    PLANNER_LIVE=1 so `python3 -m unittest test_storyboard_planner` stays instant on a
    machine with no weights.

    PLANNER_LIVE=1 python3.11 -m unittest test_storyboard_planner -v
"""

from __future__ import annotations

import copy
import re
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import storyboard  # noqa: E402
import storyboard_planner as P  # noqa: E402


# --------------------------------------------------------------------------------------
# Stub model
# --------------------------------------------------------------------------------------

class StubSession(object):
    """Stands in for PlannerSession. Replays canned replies, records what it was asked."""

    def __init__(self, replies, screenplay="", geography=""):
        self.replies = list(replies)
        # The screenplay pass answers from its OWN slot and never consumes a
        # queued reply. `replies` is the shot-pass script in every test in this
        # file, and threading a screenplay through all of them would have made
        # 27 tests about pass ordering instead of about what they each test.
        # Default "" == the model gave nothing usable, which is the documented
        # fallback and makes the shot brief byte-identical to before this pass.
        self.screenplay = screenplay
        # The geography pass gets its own slot for the same reason, and needs
        # one more than the screenplay did: it only runs when the brief carries
        # a location, so the default "" leaves every test that names no place
        # measuring exactly what it measured before it existed.
        self.geography = geography
        self.calls = []
        self.screenplay_calls = []
        self.geography_calls = []
        self.model_path = Path("stub-model")
        self.stats = {"model_path": "stub-model", "load_s": 0.0, "calls": 0,
                      "gen_s_total": 0.0, "prompt_tokens": 0, "output_tokens": 0,
                      "peak_rss_bytes": 0, "mx_peak_bytes": 0, "released": False}
        self.released = False

    def generate(self, system, user, **kw):
        self.stats["calls"] += 1
        if system == P._SCREENPLAY_SYSTEM:
            # Recorded SEPARATELY. Every assertion in this file reads `calls` as
            # "calls in the shot pass" — indexing into it for the repair prompt,
            # and counting it to prove a clean plan cost no re-plan. Appending
            # the screenplay call to the same list would silently redefine what
            # eleven of those tests measure. `stats["calls"]` still counts it,
            # so the real cost is not hidden anywhere.
            self.screenplay_calls.append({"system": system, "user": user, "kw": kw})
            return {"text": self.screenplay, "load_s": 0.0, "gen_s": 0.0,
                    "prompt_tokens": 10, "output_tokens": 10,
                    "peak_rss_bytes": 0, "mx_peak_bytes": 0}
        if system == P._GEOGRAPHY_SYSTEM:
            self.geography_calls.append({"system": system, "user": user, "kw": kw})
            return {"text": self.geography, "load_s": 0.0, "gen_s": 0.0,
                    "prompt_tokens": 10, "output_tokens": 10,
                    "peak_rss_bytes": 0, "mx_peak_bytes": 0}
        self.calls.append({"system": system, "user": user, "kw": kw})
        if not self.replies:
            raise AssertionError("stub ran out of replies after %d calls" % len(self.calls))
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return {"text": nxt, "load_s": 0.0, "gen_s": 0.0,
                "prompt_tokens": 10, "output_tokens": 10,
                "peak_rss_bytes": 0, "mx_peak_bytes": 0}

    def release(self):
        self.released = True
        self.stats["released"] = True
        return self.stats

    unload = release


def _shot(n, desc=None, **kw):
    """One shot in the shape the model is asked for: description WITHOUT the camera
    sentence and WITHOUT the ending — those are the `camera` and `settle` keys."""
    d = {
        "n": n,
        "title": "Beat %d" % n,
        "character_id": None,
        "duration_s": 5,
        "camera": "static",
        "description": desc or (
            "Live-action, cinematic, a close-up of a brass key lying on a wet stone step. "
            "Rain beads on the metal and one drop runs off the bow of the key."),
        "settle": "the key lies still and wet",
        "soundscape": "Steady rain on stone, a distant gutter running, no voices.",
        "music": "N/A",
    }
    d.update(kw)
    return d


def _plan_json(n=6, **kw):
    return json.dumps({"title": "The Key", "shots": [_shot(i + 1, **kw) for i in range(n)]})


class _StubFactory(object):
    """Replaces PlannerSession so plan_film() OWNS the stub — which means the release
    path under test is the same one production takes."""

    def __init__(self, stub):
        self.stub = stub

    def __call__(self, **kw):
        return self.stub


def _plan(replies, **kw):
    """plan_film() against a stub, with the arguments the panel would pass.

    The stub is installed as the PlannerSession class rather than handed in as
    `session=`, so plan_film() owns it and its `finally:` release is exercised.
    """
    stub = StubSession(replies, screenplay=kw.pop("screenplay_text", ""),
                       geography=kw.pop("geography_text", ""))
    kw.setdefault("concept", "a lost brass key finds its door")
    kw.setdefault("n_shots", 6)
    concept = kw.pop("concept")
    real = P.PlannerSession
    P.PlannerSession = _StubFactory(stub)
    try:
        out = P.plan_film(concept, **kw)
    finally:
        P.PlannerSession = real
    return out, stub


# --------------------------------------------------------------------------------------

class TestJSONExtraction(unittest.TestCase):
    def test_clean_object(self):
        self.assertEqual(P.extract_json_object('{"title":"x","shots":[]}')["title"], "x")

    def test_fenced(self):
        raw = "Sure!\n```json\n{\"title\":\"x\",\"shots\":[]}\n```\nLet me know."
        self.assertEqual(P.extract_json_object(raw)["title"], "x")

    def test_bare_fence_and_trailing_prose(self):
        raw = "```\n{\"title\":\"y\",\"shots\":[]}\n```\n\nHope that helps!"
        self.assertEqual(P.extract_json_object(raw)["title"], "y")

    def test_prose_before_and_after_unfenced(self):
        raw = 'Here is the plan:\n{"title":"z","shots":[{"n":1}]}\nWant changes?'
        self.assertEqual(P.extract_json_object(raw)["title"], "z")

    def test_think_block_is_ignored(self):
        raw = '<think>{"title":"WRONG","shots":[]}</think>{"title":"right","shots":[]}'
        self.assertEqual(P.extract_json_object(raw)["title"], "right")

    def test_trailing_commas_and_smart_quotes(self):
        raw = '{“title”: “q”, “shots”: [1,2,],}'
        self.assertEqual(P.extract_json_object(raw)["title"], "q")

    def test_object_wrapped_in_list(self):
        self.assertEqual(P.extract_json_object('[{"title":"L","shots":[]}]')["title"], "L")

    def test_truncated_output_is_rescued(self):
        raw = '{"title":"T","shots":[{"n":1,"description":"a key on a step'
        got = P.extract_json_object(raw)
        self.assertIsNotNone(got)
        self.assertEqual(got["title"], "T")

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        raw = '{"title":"a { brace } inside","shots":[]}'
        self.assertEqual(P.extract_json_object(raw)["title"], "a { brace } inside")

    def test_no_json_at_all(self):
        self.assertIsNone(P.extract_json_object("I'm sorry, I can't help with that."))


class TestThreeFieldSplit(unittest.TestCase):
    def test_pasted_assembled_prompt_is_taken_apart(self):
        blob = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a key.\n\n"
                "overall_soundscape: Rain on stone.\n\n"
                "non_diegetic_music: N/A")
        d, s, m = P._split_three_fields(blob)
        self.assertTrue(d.startswith("[Shot 1] Live-action"))
        self.assertEqual(s, "Rain on stone.")
        self.assertEqual(m, "N/A")
        self.assertNotIn("overall_soundscape", d)
        self.assertNotIn("non_diegetic_music", d)

    def test_plain_description_passes_through(self):
        d, s, m = P._split_three_fields("Live-action, cinematic, a key.")
        self.assertEqual(d, "Live-action, cinematic, a key.")
        self.assertEqual((s, m), ("", ""))


class TestCoercion(unittest.TestCase):
    def setUp(self):
        self.kw = dict(concept="a lost brass key", n_shots=3, storyboard_mod=storyboard)

    def test_assembles_the_h3_three_field_dialect(self):
        spec, _ = P.coerce_spec(json.loads(_plan_json(3)), **self.kw)
        p = spec["shots"][0]["prompt"]
        self.assertTrue(p.startswith("integrated_multimodal_description: [Shot 1] "))
        self.assertIn("\n\noverall_soundscape: ", p)
        self.assertIn("\n\nnon_diegetic_music: ", p)
        self.assertEqual(spec["shots"][0]["engine"], "h3")
        self.assertEqual(spec["shots"][0]["tier"], "draft")

    def test_shot_marker_is_never_doubled(self):
        raw = json.loads(_plan_json(1))
        raw["shots"][0]["description"] = "[Shot 1] Live-action, cinematic, a key."
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        self.assertEqual(spec["shots"][0]["prompt"].count("[Shot 1]"), 1)

    def test_renumbers_and_deduplicates_shot_numbers(self):
        raw = {"title": "t", "shots": [_shot(4), _shot(4), _shot(9)]}
        spec, _ = P.coerce_spec(raw, **self.kw)
        self.assertEqual([s["n"] for s in spec["shots"]], [1, 2, 3])

    def test_durations_are_clamped_and_snapped(self):
        raw = {"title": "t", "shots": [_shot(1, duration_s=999), _shot(2, duration_s=-4),
                                       _shot(3, duration_s="8")]}
        spec, _ = P.coerce_spec(raw, **self.kw)
        got = [s["duration_s"] for s in spec["shots"]]
        self.assertEqual(got, [15.0, 5.0, 10.0])
        for d in got:
            self.assertTrue(0 < d <= 60)

    def test_seeds_are_assigned_and_deterministic(self):
        a, _ = P.coerce_spec(json.loads(_plan_json(3)), **self.kw)
        b, _ = P.coerce_spec(json.loads(_plan_json(3)), **self.kw)
        self.assertEqual([s["seed"] for s in a["shots"]], [s["seed"] for s in b["shots"]])
        self.assertEqual(len({s["seed"] for s in a["shots"]}), 3)

    def test_character_shot_gets_ltx_engine_mode_and_trigger(self):
        raw = {"title": "t", "shots": [_shot(1, character_id="bizarrotrn")]}
        spec, _ = P.coerce_spec(raw, cast=P._normalise_cast(["bizarrotrn"]),
                                **dict(self.kw, n_shots=1))
        s = spec["shots"][0]
        self.assertEqual(s["engine"], "ltx")
        self.assertEqual(s["mode"], "character")
        self.assertEqual(s["character_id"], "bizarrotrn")
        self.assertIn("bizarrotrn", s["prompt"])
        self.assertNotIn("integrated_multimodal_description", s["prompt"])
        self.assertEqual(spec["cast"][0]["id"], "bizarrotrn")

    def test_unknown_character_is_dropped_not_passed_to_the_validator(self):
        raw = {"title": "t", "shots": [_shot(1, character_id="someone_who_does_not_exist")]}
        spec, warns = P.coerce_spec(raw, cast=P._normalise_cast(["bizarrotrn"]),
                                    **dict(self.kw, n_shots=1))
        self.assertIsNone(spec["shots"][0].get("character_id"))
        self.assertEqual(spec["shots"][0]["mode"], "text")
        self.assertTrue(any("unknown character" in w for w in warns))

    def test_ltx_register_strips_h3_markup(self):
        raw = {"title": "t", "shots": [_shot(
            1, character_id="bizarrotrn",
            description="[Shot 1] A man on a step. He (S1) says: <d>[English] Found it.</d>")]}
        spec, _ = P.coerce_spec(raw, cast=P._normalise_cast(["bizarrotrn"]),
                                style="documentary realism, no letterbox",
                                **dict(self.kw, n_shots=1))
        p = spec["shots"][0]["prompt"]
        for junk in ("<d>", "</d>", "[Shot 1]", "(S1)"):
            self.assertNotIn(junk, p)
        self.assertIn("'Found it.'", p)
        self.assertIn("documentary realism, no letterbox", p)
        self.assertIn("Audio:", p)

    def test_camera_choice_becomes_a_canonical_camera_sentence(self):
        raw = {"title": "t", "shots": [_shot(1, camera="push_in", description="A key."),
                                       _shot(2, camera="nonsense", description="A door."),
                                       _shot(3, camera="dolly_in", description="A step.")]}
        spec, _ = P.coerce_spec(raw, **self.kw)
        self.assertIn("pushes in with small amplitude at slow speed", spec["shots"][0]["prompt"])
        self.assertIn("holds a static shot", spec["shots"][1]["prompt"])   # unknown -> static
        self.assertIn("pushes in with small amplitude", spec["shots"][2]["prompt"])  # alias

    def test_stored_camera_is_the_key_that_actually_rendered(self):
        # Observed: the model answered the `face` enum in the `camera` slot, and the shot
        # card then read `cam=medium` while the prompt said "holds a static shot".
        raw = {"title": "t", "shots": [_shot(1, camera="medium"), _shot(2, camera="dolly_in")]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=2))
        self.assertEqual([s["camera"] for s in spec["shots"]], ["static", "push_in"])
        self.assertIn("holds a static shot", spec["shots"][0]["prompt"])
        self.assertTrue(any("not one of" in w for w in warns), warns)
        for s in spec["shots"]:
            self.assertIn(s["camera"], P.CAMERA_KEYS)

    def test_a_camera_sentence_the_model_wrote_is_not_duplicated(self):
        raw = {"title": "t", "shots": [_shot(
            1, camera="push_in",
            description="A key. The camera holds a static shot and never moves.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        self.assertEqual(spec["shots"][0]["prompt"].lower().count("the camera"), 1)

    def test_settle_phrase_becomes_the_end_state_law(self):
        raw = {"title": "t", "shots": [_shot(1, description="A key.",
                                             settle="the key lies still and wet")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        p = spec["shots"][0]["prompt"]
        self.assertIn("completely finished before the shot ends", p)
        self.assertIn("for the last two seconds the key lies still and wet", p)

    def test_face_law_is_added_only_when_a_person_is_in_the_shot(self):
        raw = {"title": "t", "shots": [
            _shot(1, description="Live-action, cinematic, a woman on a step."),
            _shot(2, description="Live-action, cinematic, an empty steel dumpster.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=2))
        self.assertIn("holds the exact angle to the lens", spec["shots"][0]["prompt"])
        self.assertNotIn("holds the exact angle to the lens", spec["shots"][1]["prompt"])

    # --- the face law -------------------------------------------------------------
    # Faces are the quality metric. A live plan wrote "his face obscured by the angle" and
    # the render put the head half out of frame; a sweep of 56 shots found 5 face-hiding
    # phrases, mostly "silhouetted against the ..." on the final shot.

    def test_face_level_renders_the_right_law(self):
        raw = {"title": "t", "shots": [
            _shot(1, face="close", description="A close-up of a boxer's face."),
            _shot(2, face="medium", description="A woman at a workbench."),
            _shot(3, face="none", description="An empty steel dumpster.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=3))
        self.assertIn("The face fills much of the frame", spec["shots"][0]["prompt"])
        self.assertIn("holds the exact angle to the lens", spec["shots"][1]["prompt"])
        self.assertNotIn("fills much of the frame", spec["shots"][1]["prompt"])
        self.assertNotIn("holds the exact angle", spec["shots"][2]["prompt"])
        self.assertEqual([s["face"] for s in spec["shots"]], ["close", "medium", "none"])

    def test_face_level_defaults_from_whether_a_person_is_on_screen(self):
        raw = {"title": "t", "shots": [
            _shot(1, face=None, description="Live-action, cinematic, a woman on a step."),
            _shot(2, face=None, description="Live-action, cinematic, an empty dumpster.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=2))
        self.assertEqual([s["face"] for s in spec["shots"]], ["medium", "none"])

    def test_face_none_cannot_switch_off_the_law_when_a_person_is_present(self):
        """The escape that shipped a silhouette: the model labelled a wide shot containing a
        woman `face: "none"`, which disabled the scrub."""
        raw = {"title": "t", "shots": [
            _shot(1, face="none",
                  description=("Live-action, cinematic, a wide shot from the rooftop showing "
                               "the woman standing beside the neon sign, silhouetted against "
                               "the vibrant lights of the market below."),
                  settle="the market glows below and she stands silhouetted against the lights"),
            _shot(2, face="none", description="Live-action, cinematic, an empty rooftop.")]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=2))
        self.assertEqual(spec["shots"][0]["face"], "medium")
        self.assertEqual(spec["shots"][1]["face"], "none")   # genuinely no person: honoured
        body = spec["shots"][0]["prompt"].replace(P._FACE_LAW_MEDIUM, " ")
        self.assertNotIn("silhouetted", body)
        self.assertTrue(any("said no face" in w for w in warns), warns)

    def test_person_silhouette_regex_catches_the_forms_that_shipped(self):
        for blocking in ("silhouetted against the vibrant lights of the market below",
                         "and she stands silhouetted against the lights",
                         "He's silhouetted against the bright light of the lens",
                         "her silhouette framed against the backdrop of the city lights",
                         "they stood silhouetted in the doorway"):
            self.assertTrue(P._PERSON_SILHOUETTE_RE.search(blocking), blocking)
        for fine in ("the dune line behind him is a clean dark silhouette against a pale sky",
                     "the lighthouse stands silhouetted against the night sky",
                     "the crane is a hard silhouette on the skyline"):
            self.assertFalse(P._PERSON_SILHOUETTE_RE.search(fine), fine)

    def test_hidden_faces_are_refused_unless_the_brief_asked(self):
        raw = {"title": "t", "shots": [_shot(1, face="hidden",
                                             description="A woman on a rooftop.")]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        self.assertEqual(spec["shots"][0]["face"], "medium")
        self.assertIn("holds the exact angle", spec["shots"][0]["prompt"])
        self.assertTrue(any("kept visible" in w for w in warns), warns)

    def test_hidden_faces_are_allowed_when_the_brief_asked(self):
        raw = {"title": "t", "shots": [_shot(1, face="hidden",
                                             description="A woman on a rooftop.")]}
        spec, warns = P.coerce_spec(raw, allow_hidden_faces=True, **dict(self.kw, n_shots=1))
        self.assertEqual(spec["shots"][0]["face"], "hidden")
        self.assertNotIn("holds the exact angle", spec["shots"][0]["prompt"])
        self.assertEqual([w for w in warns if "kept visible" in w], [])

    def test_face_blocking_prose_is_scrubbed_out(self):
        # Every one of these is verbatim from a real plan, or the reported defect.
        raw = {"title": "t", "shots": [
            _shot(1, description=("Live-action, cinematic, a close-up of a boxer on a stool, "
                                  "his face obscured by the angle, sweat on his shoulders. "
                                  "He breathes out slowly.")),
            _shot(2, description=("Live-action, cinematic, a wide shot of the keeper at the "
                                  "window. He stands very still."),
                  settle=("he is standing at the window, silhouetted against the setting "
                          "sun, watching the light sweep out")),
            _shot(3, description=("Live-action, cinematic, a medium shot of a woman on a "
                                  "rooftop, her silhouette framed against the city lights. "
                                  "She sets down her tools.")),
            _shot(4, description=("Live-action, cinematic, a violinist seen from behind, "
                                  "her bow arm rising. She draws one long note."))]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=4))
        joined = " ".join(s["prompt"] for s in spec["shots"])
        for banned in ("obscured", "silhouetted against", "her silhouette", "seen from behind"):
            self.assertNotIn(banned, joined, "%r survived the scrub" % banned)
        # The rest of each direction survives — this is a clause scrub, not a shot delete.
        self.assertEqual(len(spec["shots"]), 4)
        self.assertIn("sweat on his shoulders", spec["shots"][0]["prompt"])
        self.assertIn("watching the light sweep out", spec["shots"][1]["prompt"])
        self.assertIn("She sets down her tools", spec["shots"][2]["prompt"])
        self.assertIn("her bow arm rising", spec["shots"][3]["prompt"])
        self.assertEqual(len([w for w in warns if "face-hiding framing" in w]), 4, warns)

    def test_scrub_leaves_legitimate_non_person_silhouettes_alone(self):
        # From the C1 exemplar: a landscape silhouette is good cinematography, not a
        # hidden face. Same for a face half in shadow.
        desc = ("Live-action, cinematic, a medium close-up of a man on a dune ridge. Hard "
                "low sun rakes from camera left, carving one bright edge down his cheekbone "
                "while the other side of his face falls into open shadow; the dune line "
                "behind him is a clean dark silhouette against a pale sky.")
        out, removed = P._scrub_face_blocking(desc)
        self.assertEqual(removed, [])
        self.assertIn("clean dark silhouette against a pale sky", out)
        self.assertIn("falls into open shadow", out)

    def test_scrub_does_not_fire_on_ordinary_behind(self):
        desc = "A daughter stands behind her mother, the crowd close behind him."
        out, removed = P._scrub_face_blocking(desc)
        self.assertEqual(removed, [])
        self.assertEqual(out, desc)

    def test_brief_detection_for_hidden_faces(self):
        for yes in ("a film told entirely in silhouette",
                    "we only ever see her from behind",
                    "a faceless narrator",
                    "shot without showing his face"):
            self.assertTrue(P._WANTS_HIDDEN_RE.search(yes), yes)
        for no in ("a boxer between rounds, close on the face",
                   "a lighthouse keeper on his last night",
                   "the dune line behind him is a dark shape"):
            self.assertFalse(P._WANTS_HIDDEN_RE.search(no), no)

    def test_curly_punctuation_is_normalised_out_of_the_prompt(self):
        raw = {"title": "t", "shots": [_shot(
            1, description="A key — he says “you’ve got it”…",
            soundscape="Rain — steady.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        p = spec["shots"][0]["prompt"]
        for ch in ("’", "“", "”", "—", "…"):
            self.assertNotIn(ch, p)

    def test_no_text_clause_is_omitted_when_typography_was_requested(self):
        raw = {"title": "t", "shots": [
            _shot(1, description='A wall where the word "PHOSPHENE" burns in.'),
            _shot(2, description="A plain wall."),
            # Observed failure: the model used single quotes and the refusal landed on a
            # title sequence whose whole point was the lettering.
            _shot(3, description="Mercury coalescing into the letter 'P' on black glass."),
            _shot(4, description="The mercury's surface stays perfectly smooth.")]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=4))
        self.assertNotIn("No text appears", spec["shots"][0]["prompt"])
        self.assertIn("No text appears", spec["shots"][1]["prompt"])
        self.assertNotIn("No text appears", spec["shots"][2]["prompt"])
        # An apostrophe is not a quoted string.
        self.assertIn("No text appears", spec["shots"][3]["prompt"])
        self.assertTrue(any("double quotes" in w for w in warns), warns)

    def test_typography_detector_does_not_fire_on_dialogue_or_props(self):
        # Both were real false positives: single-quoted LTX dialogue, and "neon sign" —
        # where the refusal is exactly what the shot needs.
        dialogue = ("A man at a workbench. He says, 'Everything has a story, you know. "
                    "And a purpose. Even if it's broken.'")
        neon = "A woman repairing a large crimson neon sign on a rooftop."
        self.assertEqual(P._typography_strings(dialogue), [])
        self.assertEqual(P._typography_strings(neon), [])
        self.assertEqual(P._typography_strings("the word \"PHOSPHENE\" burns in"),
                         ["PHOSPHENE"])
        self.assertEqual(P._typography_strings("the letter 'P' forms"), ["P"])
        spec, warns = P.coerce_spec(
            {"title": "t", "shots": [_shot(1, description=dialogue), _shot(2, description=neon)]},
            **dict(self.kw, n_shots=2))
        for s in spec["shots"]:
            self.assertIn("No text appears", s["prompt"])
        self.assertEqual([w for w in warns if "double quotes" in w], [])

    def test_camera_talk_is_stripped_out_of_the_settle_clause(self):
        raw = {"title": "t", "shots": [
            _shot(1, settle="the camera stops orbiting, the lens keeps turning silently"),
            _shot(2, settle="the camera holds on the scene")]}
        spec, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=2))
        p1, p2 = spec["shots"][0]["prompt"], spec["shots"][1]["prompt"]
        self.assertIn("for the last two seconds the lens keeps turning silently", p1)
        self.assertNotIn("the camera stops orbiting", p1)
        # Nothing but camera talk -> no settle clause at all, rather than a contradiction.
        self.assertNotIn("completely finished before the shot ends", p2)
        self.assertTrue(any("instead of an end state" in w for w in warns), warns)

    def test_camera_monoculture_is_reported_as_a_warning(self):
        raw = {"title": "t", "shots": [_shot(i + 1, camera="static", settle="it is still")
                                       for i in range(4)]}
        _, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=4))
        self.assertTrue(any("same camera behaviour" in w for w in warns), warns)

    def test_missing_end_states_are_reported_as_a_warning(self):
        raw = {"title": "t", "shots": [_shot(i + 1, settle="") for i in range(4)]}
        _, warns = P.coerce_spec(raw, **dict(self.kw, n_shots=4))
        self.assertTrue(any("no shot named an end state" in w for w in warns), warns)

    def test_unbalanced_dialogue_tag_is_closed(self):
        raw = {"title": "t", "shots": [_shot(1, description="A key. <d>[English] Hello.")]}
        spec, _ = P.coerce_spec(raw, **dict(self.kw, n_shots=1))
        p = spec["shots"][0]["prompt"]
        self.assertEqual(p.count("<d>"), p.count("</d>"))

    def test_empty_description_shot_is_dropped(self):
        raw = {"title": "t", "shots": [_shot(1), _shot(2, description=""), _shot(3)]}
        spec, warns = P.coerce_spec(raw, **self.kw)
        self.assertEqual(len(spec["shots"]), 2)
        self.assertTrue(any("empty description" in w for w in warns))

    def test_policy_is_clamped_to_the_machine_cap(self):
        spec, _ = P.coerce_spec(json.loads(_plan_json(3)), max_dim=768, **self.kw)
        for key in ("draft", "final"):
            p = spec["policy"][key]
            self.assertLessEqual(max(p["width"], p["height"]), 768)

    def test_garbage_input_still_produces_a_legal_envelope(self):
        spec, warns = P.coerce_spec(None, **self.kw)
        self.assertEqual(spec["schema"], storyboard.SCHEMA_VERSION)
        self.assertTrue(spec["id"])
        self.assertEqual(spec["shots"], [])
        self.assertTrue(warns)


class TestValidatorContract(unittest.TestCase):
    """The point of the module: what comes out passes the REAL validator."""

    def test_stub_plan_validates_clean(self):
        out, stub = _plan([_plan_json(6)])
        self.assertFalse(P.is_plan_error(out), out.get("error"))
        self.assertEqual(storyboard.validate_storyboard(out), [])
        self.assertEqual(len(out["shots"]), 6)
        self.assertTrue(out["_planner"]["first_try_clean"])
        self.assertEqual(out["_planner"]["attempts"], 1)
        self.assertTrue(stub.released)

    def test_plan_survives_a_roundtrip_through_save_and_load(self):
        out, _ = _plan([_plan_json(6)])
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            storyboard.save_storyboard(Path(td), out)
            back = storyboard.load_storyboard(Path(td), out["id"])
        self.assertEqual(storyboard.validate_storyboard(back), [])
        self.assertEqual(back["shots"][0]["prompt"], out["shots"][0]["prompt"])

    def test_plan_translates_to_panel_jobs(self):
        out, _ = _plan([_plan_json(6)])
        shot = out["shots"][0]
        job = storyboard.shot_to_job(shot, out["policy"]["draft"],
                                     board_id=out["id"], board_title=out["title"])
        self.assertEqual(job["enhance"], "off")
        self.assertTrue(job["prompt"])
        # The PANEL's mode vocabulary, not the storyboard schema's: mlx_ltx_panel has one
        # backend video mode for both of v1's shot types, and "text"/"character" are not it.
        self.assertIn(job["mode"], ("t2v", "i2v", "keyframe", "extend", "a2v"))
        # The engine the planner assigned must survive the translation — this is the seam
        # that used to send every H3 shot to LTX.
        self.assertEqual(job["engine"], shot["engine"])
        self.assertEqual(job["session_tag"], "sb:%s#%d" % (out["id"], shot["n"]))

    def test_character_plan_validates_with_known_ids(self):
        raw = json.dumps({"title": "t", "shots": [
            _shot(1, character_id="bizarrotrn"), _shot(2), _shot(3)]})
        out, _ = _plan([raw], n_shots=3, characters=["bizarrotrn"])
        self.assertFalse(P.is_plan_error(out), out.get("error"))
        self.assertEqual(storyboard.validate_storyboard(
            out, known_character_ids=["bizarrotrn"]), [])
        self.assertEqual(out["_planner"]["engine_mix"], {"ltx": 1, "h3": 2})


class TestRepairRoundTrip(unittest.TestCase):
    def test_wrong_shot_count_triggers_exactly_one_repair(self):
        out, stub = _plan([_plan_json(4), _plan_json(6)])
        self.assertFalse(P.is_plan_error(out))
        self.assertEqual(len(out["shots"]), 6)
        self.assertEqual(len(stub.calls), 2)
        self.assertEqual(out["_planner"]["attempts"], 2)
        self.assertIn("exactly 6 were requested", stub.calls[1]["user"])

    def test_unparseable_first_reply_is_repaired(self):
        out, stub = _plan(["I cannot do that.", _plan_json(6)])
        self.assertFalse(P.is_plan_error(out))
        self.assertEqual(len(out["shots"]), 6)
        self.assertIn("did not contain a JSON object", stub.calls[1]["user"])

    def test_two_bad_replies_return_a_structured_error_not_a_traceback(self):
        out, stub = _plan(["nope", "still nope"])
        self.assertTrue(P.is_plan_error(out))
        err = out["error"]
        self.assertEqual(err["kind"], "invalid_plan")
        self.assertTrue(err["message"])
        self.assertTrue(err["problems"])
        self.assertNotIn("Traceback", json.dumps(out))
        self.assertEqual(len(stub.calls), 2)   # never more than one repair
        self.assertTrue(stub.released)

    def test_a_worse_repair_does_not_replace_a_better_first_draft(self):
        # First draft: 5 shots (count wrong, otherwise valid). Repair: unusable.
        out, _ = _plan([_plan_json(5), "{}"])
        self.assertFalse(P.is_plan_error(out))
        self.assertEqual(len(out["shots"]), 5)
        self.assertFalse(out["_planner"]["shot_count_ok"])

    def test_model_unavailable_is_a_structured_error(self):
        out, _ = _plan([P.PlannerError("planner model not found at /nowhere")])
        self.assertTrue(P.is_plan_error(out))
        self.assertEqual(out["error"]["kind"], "model_unavailable")
        self.assertIn("mlx-lm", out["error"]["hint"])


class TestFeedback(unittest.TestCase):
    def setUp(self):
        self.base, _ = _plan([_plan_json(6)])

    def test_feedback_string_is_parsed_into_a_shot_reroll(self):
        self.assertEqual(P._parse_feedback("shot 4: he should not turn his head"),
                         ("shot", 4, "he should not turn his head"))
        self.assertEqual(P._parse_feedback({"shot": 2, "note": "colder"}),
                         ("shot", 2, "colder"))
        self.assertEqual(P._parse_feedback("make it colder")[0], "film")
        self.assertEqual(P._parse_feedback(None)[0], "none")

    def test_shot_reroll_changes_only_that_shot_and_leaves_the_rest_byte_stable(self):
        before = copy.deepcopy(self.base)
        reply = json.dumps({"title": "The Key", "shots": [_shot(
            4, description=(
                "Live-action, cinematic, a wide shot of a green door in a brick wall. "
                "The camera holds a static shot, the frame never moves. Rain runs down the "
                "paint and the letterbox flap swings once and stops. The movement is "
                "completely finished before the shot ends, and for the last two seconds the "
                "door is simply shut and streaming, with no new movement of any kind."))]})
        out, _ = _plan([reply], feedback="shot 4: make it a door, not a key",
                       previous=before, n_shots=6)
        self.assertFalse(P.is_plan_error(out), out.get("error"))
        self.assertEqual(storyboard.validate_storyboard(out), [])
        self.assertEqual(len(out["shots"]), 6)
        self.assertIn("green door", out["shots"][3]["prompt"])
        self.assertEqual(out["shots"][3]["n"], 4)
        for i in (0, 1, 2, 4, 5):
            self.assertEqual(json.dumps(out["shots"][i], sort_keys=True),
                             json.dumps(before["shots"][i], sort_keys=True),
                             "shot %d drifted during a per-shot re-roll" % (i + 1))

    def test_shot_reroll_prompt_carries_the_neighbours_but_asks_for_one_shot(self):
        _, stub = _plan([json.dumps({"title": "T", "shots": [_shot(4)]})],
                        feedback={"shot": 4, "note": "colder"},
                        previous=copy.deepcopy(self.base), n_shots=6)
        user = stub.calls[0]["user"]
        self.assertIn("re-rolling ONE shot", user)
        self.assertIn("n=4", user)
        self.assertIn("One shot only", user)

    def test_film_feedback_replans_everything(self):
        _, stub = _plan([_plan_json(6)], feedback="make it colder and drop the voiceover",
                        previous=copy.deepcopy(self.base), n_shots=6)
        user = stub.calls[0]["user"]
        self.assertIn("DIRECTOR'S NOTES", user)
        self.assertIn("make it colder", user)

    def test_feedback_without_previous_is_a_programmer_error(self):
        with self.assertRaises(P.PlannerError):
            _plan([_plan_json(6)], feedback="shot 2: colder")


class TestPromptContent(unittest.TestCase):
    """The prompt IS the product. If a law silently falls out, a test should notice."""

    def test_system_prompt_carries_the_laws_and_the_dialect(self):
        sys_p = P._build_system_prompt("auto", False)
        for needle in ("integrated_multimodal_description", "overall_soundscape",
                       "non_diegetic_music", "<d>[English]", "holds a static shot",
                       "completely finished before the shot ends", "no negative prompt",
                       "Heads stay square to the lens", "VARIETY IS A HARD REQUIREMENT"):
            self.assertIn(needle, sys_p, "missing from the system prompt: %r" % needle)

    def test_contract_lists_every_camera_key_the_coercer_accepts(self):
        sys_p = P._build_system_prompt("auto", False)
        for key in P.CAMERA_KEYS:
            self.assertIn(key, sys_p, "camera %r is legal but never offered to the model" % key)

    def test_face_law_is_in_the_prompt_and_hidden_is_not_offered_by_default(self):
        sys_p = P._build_system_prompt("auto", False)
        self.assertIn("L11 THE FACE IS THE WHOLE POINT", sys_p)
        self.assertIn('"face"         ONE of: close, medium, none.', sys_p)
        self.assertIn("There is no fourth option", sys_p)
        self.assertNotIn('you may use "hidden"', sys_p)
        relaxed = P._build_system_prompt("auto", False, allow_hidden=True)
        self.assertIn('you may use "hidden"', relaxed)

    def test_no_exemplar_teaches_the_model_to_hide_a_face(self):
        """The audit that caught it: the box exemplar used to say 'he tips his face up to
        the sky' and hold it there in the settle — a face aimed away from the lens, taught
        by example. The LTX exemplar sent the eyes off-camera on a talking head."""
        sys_p = P._build_system_prompt("auto", True)
        # Strip L11 itself, which necessarily quotes the phrases it forbids.
        body = sys_p.split("L11 THE FACE IS THE WHOLE POINT")[0]
        for phrase in ("tips his face up", "tipped up", "off-camera", "obscur",
                       "from behind", "seen from behind", "back to the camera"):
            self.assertNotIn(phrase, body, "exemplars still teach %r" % phrase)
        self.assertEqual(P._FACE_BLOCK_RE.findall(body), [])
        # Every exemplar declares a face level.
        self.assertEqual(re.findall(r'"face": "(\w+)"', sys_p),
                         ["close", "medium", "close", "none", "close"])

    def test_ltx_example_appears_only_when_there_is_a_cast(self):
        self.assertNotIn("letterbox", P._build_system_prompt("auto", False))
        self.assertIn("letterbox", P._build_system_prompt("auto", True))

    def test_forcing_ltx_drops_the_h3_dialect(self):
        sys_p = P._build_system_prompt("ltx", True)
        self.assertNotIn("integrated_multimodal_description:", sys_p.split("EVERY shot")[-1])
        self.assertIn("LTX register", sys_p)

    def test_user_prompt_lists_cast_and_must_include(self):
        u = P._build_user_prompt("a key", 6, "documentary", P._normalise_cast(["bizarrotrn"]),
                                 ["a green door"])
        self.assertIn("bizarrotrn", u)
        self.assertIn("a green door", u)
        self.assertIn("exactly 6 shots", u)


class TestMemoryDiscipline(unittest.TestCase):
    def test_release_is_idempotent_and_safe_before_spawn(self):
        s = P.PlannerSession(model_path="/definitely/not/here")
        self.assertTrue(s.release()["released"])
        self.assertTrue(s.release()["released"])

    def test_missing_model_raises_a_planner_error_not_an_oserror(self):
        s = P.PlannerSession(model_path="/definitely/not/here")
        with self.assertRaises(P.PlannerError):
            s.generate("sys", "user")
        s.release()

    def test_context_manager_releases(self):
        s = P.PlannerSession(model_path="/definitely/not/here")
        with s:
            pass
        self.assertTrue(s.stats["released"])

    def test_a_borrowed_session_is_the_callers_to_release(self):
        stub = StubSession([_plan_json(3)])
        out = P.plan_film("a key", n_shots=3, session=stub)
        self.assertFalse(P.is_plan_error(out))
        self.assertFalse(stub.released, "plan_film released a session it does not own")
        stub.release()

    def test_an_owned_session_is_released_even_when_the_model_explodes(self):
        stub = StubSession([RuntimeError("boom")])
        real = P.PlannerSession
        P.PlannerSession = _StubFactory(stub)
        try:
            with self.assertRaises(RuntimeError):
                P.plan_film("a key", n_shots=3)
        finally:
            P.PlannerSession = real
        self.assertTrue(stub.released, "the model survived an exception inside plan_film")

    def test_missing_model_reports_release_in_the_metadata(self):
        out = P.plan_film("a key", n_shots=3, model_path="/definitely/not/here")
        self.assertTrue(P.is_plan_error(out))
        self.assertEqual(out["error"]["kind"], "model_unavailable")
        self.assertTrue(out["_planner"]["model_released"])


# --------------------------------------------------------------------------------------
# LIVE — real weights. Opt-in.
# --------------------------------------------------------------------------------------

LIVE_CONCEPT = ("A lighthouse keeper on his last night before the light is automated. "
                "He says goodbye to the machine that kept him company for thirty years.")


@unittest.skipUnless(os.environ.get("PLANNER_LIVE") == "1",
                     "set PLANNER_LIVE=1 to run against the real planner model")
class TestLivePlanner(unittest.TestCase):
    def test_six_shot_plan_validates_and_the_model_is_released(self):
        if not P.DEFAULT_MODEL_PATH.exists():
            self.skipTest("planner model not on disk: %s" % P.DEFAULT_MODEL_PATH)
        out = P.plan_film(LIVE_CONCEPT, n_shots=6,
                          style="Live-action, cinematic, photoreal, heavy 35mm film grain")
        self.assertFalse(P.is_plan_error(out),
                         "planner failed: %s" % json.dumps(out.get("error"), indent=2))
        errs = storyboard.validate_storyboard(out)
        self.assertEqual(errs, [], "validator complained: %s" % errs)
        self.assertEqual(len(out["shots"]), 6)

        meta = out["_planner"]
        for s in out["shots"]:
            self.assertTrue(s["prompt"].startswith("integrated_multimodal_description: [Shot 1] "))
            self.assertIn("\n\noverall_soundscape: ", s["prompt"])
            self.assertIn("\n\nnon_diegetic_music: ", s["prompt"])
            self.assertEqual(s["prompt"].count("<d>"), s["prompt"].count("</d>"))
            self.assertEqual(s["tier"], "draft")
            self.assertEqual(s["engine"], "h3")

        # The whole point of the subprocess design.
        self.assertTrue(meta["model_released"])
        self.assertGreater(meta["peak_rss_gb"], 0.5)
        self.assertLess(meta["peak_rss_gb"], 16.0,
                        "plan-phase RSS blew past the budget: %s GB" % meta["peak_rss_gb"])

        sys.stderr.write(
            "\n  LIVE: model=%s  attempts=%d  load=%.1fs  gen=%.1fs  total=%.1fs\n"
            "        peak RSS=%.2f GB  mlx peak=%.2f GB  first-try-clean=%s\n"
            % (meta["model"], meta["attempts"], meta["load_s"] or 0.0, meta["gen_s"] or 0.0,
               meta["elapsed_s"], meta["peak_rss_gb"], meta["mx_peak_gb"],
               meta["first_try_clean"]))


class TestAppearanceLaw(unittest.TestCase):
    """L12. Every string here came from, or was written against, the live defect:

    "ww2 scene but main characters are humanoid animals, bizarrotrn is the boss of the
    team. they are with the allies. the bad guys are natzi animals as well."

    which planned "bizarrotrn Bizarro, a grizzled badger with a military uniform" — a
    SPECIES assigned to the one face in the film that already has one.
    """

    def _v(self, text):
        return P._appearance_violations(text, "bizarrotrn", "Bizarro")

    def test_the_live_defect_is_caught(self):
        self.assertTrue(self._v(
            "bizarrotrn Bizarro, a grizzled badger with a military uniform, leans over "
            "the map table."))

    def test_species_age_and_build_are_all_caught(self):
        for bad in (
            "bizarrotrn Bizarro, an anthropomorphic badger in an allied coat, salutes.",
            "bizarrotrn, a tall broad-shouldered man, kicks the door open.",
            "bizarrotrn Bizarro, a scarred old wolf, moves into the treeline.",
            "bizarrotrn is a humanoid fox in a flight jacket.",
            "bizarrotrn Bizarro, a grey-furred stoat, checks his rifle.",
        ):
            self.assertTrue(self._v(bad), "missed: %s" % bad)

    def test_wardrobe_rank_and_action_are_never_violations(self):
        # These are what we WANT the planner to write. A law that eats them is worse than
        # no law: it would strip the only concrete direction a cast shot carries.
        for good in (
            "bizarrotrn Bizarro, in a muddy allied officer's uniform, leans over the map.",
            "bizarrotrn Bizarro, the boss of the team, points at the ridge.",
            "bizarrotrn Bizarro, a sergeant in the allied unit, waves the squad forward.",
            "bizarrotrn Bizarro, carrying a field radio, crouches behind the sandbags.",
            "bizarrotrn Bizarro stands at the map table and traces the river with a finger.",
            "bizarrotrn Bizarro, a commander who has not slept, rubs his eyes.",
        ):
            self.assertFalse(self._v(good), "false positive: %s" % good)

    def test_every_other_character_may_still_be_an_animal(self):
        # The premise is the film's to keep. Only the CAST character is undescribable.
        for good in (
            "A grizzled badger sergeant in a nazi greatcoat sneers across the table.",
            "Two humanoid wolves in enemy uniforms drag a crate through the mud.",
            "bizarrotrn Bizarro faces a hulking boar officer across the bunker.",
        ):
            self.assertFalse(self._v(good), "the premise was suppressed: %s" % good)

    def test_the_fallback_substitutes_canon_and_keeps_the_action(self):
        out, cuts = P._neutralise_appearance(
            "bizarrotrn Bizarro, a grizzled badger with a military uniform, leans over "
            "the map table.", "bizarrotrn", "Bizarro", "man")
        self.assertEqual(out, "bizarrotrn Bizarro, a man with a military uniform, leans "
                              "over the map table.")
        self.assertTrue(cuts)

    def test_the_fallback_never_eats_the_verb(self):
        # The failure mode of the two designs that were rejected before token-scrub.
        for text in (
            "bizarrotrn Bizarro, a scarred old wolf, moves into the treeline.",
            "bizarrotrn, a tall, broad-shouldered man, kicks the door open.",
            "bizarrotrn Bizarro, a grey-furred stoat, checks his rifle.",
        ):
            out, _ = P._neutralise_appearance(text, "bizarrotrn", "Bizarro", "man")
            self.assertFalse(P._appearance_violations(out, "bizarrotrn", "Bizarro"))
            self.assertTrue(out.rstrip().endswith("."), out)
            self.assertIn(text.rsplit(",", 1)[-1].strip().rstrip("."), out)

    def test_the_bundles_subject_noun_reaches_the_planner(self):
        # lora-lab writes subject_noun into every bundle; it was being dropped on the floor.
        cast = P._normalise_cast([{"id": "bizarrotrn", "name": "Bizarro",
                                   "subject_noun": "man", "pronoun": "he"}])
        self.assertEqual(cast[0]["subject_noun"], "man")
        prompt = P._build_user_prompt("a ww2 film", 6, "", cast, [])
        self.assertIn("a man", prompt)
        self.assertIn("never a species", prompt.lower().replace("-", " ")
                      if "never a species" in prompt.lower() else prompt.lower())


class TestSpeechLaw(unittest.TestCase):
    """L13. From the same film: shot 1 read "He explains the mission, his voice low and
    authoritative" with no line written anywhere, and the soundscape carried "the quiet
    murmur of other animals". The voice gate was correctly OFF — there were no words — so
    the model was told a man was speaking and given nothing to say. It babbled."""

    def test_the_live_defect_is_caught(self):
        v = P._speech_violations(
            "He explains the mission, his voice low and authoritative.",
            "Boots on wet planks, the quiet murmur of other animals, rain on canvas.")
        self.assertEqual(len(v), 3, v)          # verb, voice description, audio cue

    def test_a_written_line_is_the_whole_point_and_passes(self):
        self.assertFalse(P._speech_violations(
            "bizarrotrn Bizarro leans over the map. <d>[English] Move out, and keep to the "
            "treeline.</d> His jaw ceases speaking motion and his mouth settles closed.",
            "Rain on canvas and boots on wet planks."))

    def test_an_empty_dialogue_tag_is_not_a_line(self):
        self.assertTrue(P._speech_violations("He explains the plan. <d>[English] </d>", ""))

    def test_a_genuinely_silent_shot_passes(self):
        self.assertFalse(P._speech_violations(
            "bizarrotrn Bizarro traces the river on the map with one finger and taps twice.",
            "Rain on canvas, boots on wet planks, a radio hissing static."))

    def test_negated_cues_are_how_silence_is_written(self):
        # THE first false positive this law produced: the planner's own exemplars end
        # soundscapes with "no voices", which is a declaration of silence, not speech.
        for quiet in ("Steady rain on stone, a distant gutter running, no voices.",
                      "Wind over the ridge. Nobody speaks and no voice is heard at any point.",
                      "Rain on canvas, without chatter of any kind."):
            self.assertFalse(P._speech_violations("He taps the map twice.", quiet), quiet)

    def test_the_fallback_keeps_the_good_ambience(self):
        # Dropping the whole sentence for one bad clause threw away the boots and the rain,
        # which are exactly what should carry a silent shot.
        _, sound, notes = P._neutralise_speech(
            "He explains the mission.",
            "Boots on wet planks, the quiet murmur of other animals, rain on canvas.")
        self.assertIn("Boots on wet planks", sound)
        self.assertIn("rain on canvas", sound)
        self.assertNotIn("murmur", sound)
        self.assertIn("Nobody speaks", sound)
        self.assertTrue(notes)

    def test_a_briefing_room_is_a_room(self):
        # The first false positive the LIVE run produced, three times in one film:
        # `brief(?:s|ing)?` matched the noun in "a dimly lit briefing room".
        self.assertFalse(P._speech_violations("A dimly lit briefing room, full of maps.", ""))
        self.assertFalse(P._speech_violations("He exits the briefing room.", ""))
        # ...but the verb still has to be caught.
        self.assertTrue(P._speech_violations("He is briefing the team on the crossing.", ""))
        self.assertTrue(P._speech_violations("He briefs the team.", ""))

    def test_a_prose_quoted_line_is_rewrapped_not_deleted(self):
        # The LIVE run's second finding: 4 of 12 shots carried REAL lines written as
        # ordinary quotes. No <d> tag means the voice gate never flips — but the words are
        # the director's and deleting them is worse than the bug.
        d = ("bizarrotrn Bizarro nods curtly. He says, 'Gentlemen, we have a situation.'")
        v = P._speech_violations(d, "Chairs scrape.")
        self.assertEqual(len(v), 1)
        self.assertIn("prose quotes", v[0])
        out, sound, notes = P._neutralise_speech(d, "Chairs scrape.")
        self.assertIn("<d>[English] Gentlemen, we have a situation.</d>", out)
        self.assertTrue(P._has_dialogue(out), "the voice gate must now flip on")
        self.assertFalse(P._speech_violations(out, sound))
        self.assertIn("ceases speaking motion", out, "L6 wants the mouth stopped")
        self.assertNotIn("</d>.", out, "double punctuation after the tag")
        self.assertTrue(any("rewrapped" in n for n in notes), notes)
        self.assertEqual(sound, "Chairs scrape.", "a spoken shot keeps its soundscape")

    def test_the_fallback_silences_the_verb_and_the_voice(self):
        desc, sound, _ = P._neutralise_speech(
            "bizarrotrn Bizarro explains the mission, his voice low and authoritative.",
            "Rain on canvas.")
        self.assertFalse(P._speech_violations(desc, sound), desc)
        self.assertNotIn("explains", desc)
        self.assertNotIn("voice", desc)


class TestLawEnforcement(unittest.TestCase):
    """The laws are enforced by the plan loop, not merely reported by it."""

    CAST = [{"id": "bizarrotrn", "trigger": "bizarrotrn", "name": "Bizarro",
             "subject_noun": "man", "pronoun": "he"}]
    BAD = ("bizarrotrn Bizarro, a grizzled badger with a military uniform, leans over the "
           "map table and sets down a brass compass beside the river marker.")
    GOOD = ("bizarrotrn Bizarro, the unit's commander in a muddy field uniform, leans over "
            "the map table and sets down a brass compass beside the river marker.")

    def _film(self, desc, n=2):
        return json.dumps({"title": "Night Crossing", "shots": [
            _shot(i + 1, description=desc if i == 0 else
                  "Live-action, cinematic, rain beads on a brass compass on a folded map.",
                  character_id="bizarrotrn" if i == 0 else None)
            for i in range(n)]})

    def test_a_violation_is_re_planned_and_the_fix_is_kept(self):
        one = json.dumps({"title": "Night Crossing",
                          "shots": [_shot(1, description=self.GOOD,
                                          character_id="bizarrotrn")]})
        spec, stub = _plan([self._film(self.BAD), one], n_shots=2,
                           characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertFalse(P._appearance_violations(spec["shots"][0]["description"],
                                                  "bizarrotrn", "Bizarro"))
        self.assertIn("badger", stub.calls[1]["user"],
                      "the re-plan must quote the offending clause back")
        self.assertIn("APPEARANCE (L12)", stub.calls[1]["user"])

    def test_a_second_violation_is_neutralised_mechanically(self):
        # The model refuses twice. The law still holds — that is the difference between
        # enforcement and hope.
        spec, _ = _plan([self._film(self.BAD), self._film(self.BAD)], n_shots=2,
                        characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        desc = spec["shots"][0]["description"]
        self.assertFalse(P._appearance_violations(desc, "bizarrotrn", "Bizarro"), desc)
        self.assertNotIn("badger", desc.lower())
        self.assertIn("a man", desc)
        self.assertIn("compass", desc, "the action was destroyed by the fallback")
        self.assertNotIn("badger", spec["shots"][0]["prompt"].lower(),
                         "the assembled prompt was not rebuilt after the fix")

    def test_a_lawful_plan_costs_no_extra_model_calls(self):
        spec, stub = _plan([self._film(self.GOOD)], n_shots=2,
                           characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.calls), 1, "a clean plan must not trigger a re-plan")

    def test_fixes_for_several_shots_all_survive(self):
        """The bug a live 4-shot film found and the single-offender tests could not.

        `replan` closed over the caller's spec, so every re-roll spliced its fix into the
        ORIGINAL plan: shot 1 fixed, shot 2 fixed against the unfixed plan, shot 1's fix
        gone. The film reported "re-planned and now obeys" three times and then failed the
        final scan on all three — three model calls spent to change nothing.
        """
        bad = json.dumps({"title": "Night Crossing", "shots": [
            _shot(1, description=self.BAD, character_id="bizarrotrn"),
            _shot(2, description=self.BAD.replace("brass compass", "field radio"),
                  character_id="bizarrotrn"),
            _shot(3, description="Live-action, cinematic, rain beads on a folded map."),
        ]})
        def fixed(n, extra):
            return json.dumps({"title": "Night Crossing", "shots": [
                _shot(n, description=self.GOOD.replace("brass compass", extra),
                      character_id="bizarrotrn")]})
        spec, stub = _plan([bad, fixed(1, "brass compass"), fixed(2, "field radio")],
                           n_shots=3, characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.calls), 3, "one plan + one re-plan per offending shot")
        for s in spec["shots"]:
            self.assertFalse(
                P._appearance_violations(s.get("description", ""), "bizarrotrn", "Bizarro"),
                "shot %s kept its violation: %r" % (s.get("n"), s.get("description", "")[:80]))
        # and each fix is the one that shot was given, not the last one to arrive
        self.assertIn("brass compass", spec["shots"][0]["description"])
        self.assertIn("field radio", spec["shots"][1]["description"])

    def test_the_law_must_not_eat_the_films_premise(self):
        """The over-correction, measured live on the owner's film.

        Told never to give the CAST character a species, the planner generalised it to
        never mentioning species at all: twelve shots, zero animal words, every soldier on
        both sides quietly an ordinary human. Each shot was individually lawful and the
        film was wrong. The law is scoped to one character; the premise belongs to
        everyone else.
        """
        concept = ("ww2 scene but main characters are humanoid animals, bizarrotrn is the "
                   "boss of the team. the bad guys are natzi animals as well.")
        self.assertTrue(P._premise_species(concept), "the brief plainly names creatures")
        plain = json.dumps({"title": "Operation Wildfire", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, two soldiers in muddy allied "
                                 "uniforms drag a crate through the rain."),
        ]})
        withanimals = json.dumps({"title": "Operation Wildfire", "shots": [
            _shot(2, description="Live-action, cinematic, two humanoid wolves in muddy "
                                 "allied uniforms drag a crate through the rain.")]})
        spec, stub = _plan([plain, withanimals], n_shots=2, concept=concept,
                           characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        # the uncast shot carries the premise again...
        self.assertTrue(P._SPECIES_RE.search(spec["shots"][1]["description"]),
                        spec["shots"][1]["description"])
        # ...while the cast character is still undescribed
        self.assertFalse(P._appearance_violations(spec["shots"][0]["description"],
                                                  "bizarrotrn", "Bizarro"))
        self.assertIn("premise", stub.calls[-1]["user"].lower())

    def test_the_premise_needs_its_OWN_terms_not_any_species(self):
        """A robot is not a fox.

        `_premise_lost()` accepted ANY species word, so a brief asking for a fox
        was considered preserved by a shot containing a robot — the check passed
        while the film had quietly become a different film. Found by the external
        review, which also noted the old test codified the same weak condition by
        asserting only that some _SPECIES_RE matched.
        """
        premise = P._premise_species("a lone fox mechanic repairs a lunar rover")
        self.assertEqual(premise, ["fox"])
        robot = {"shots": [{"n": 1, "character_id": None,
                            "description": "A hulking robot welds a strut in the bay."}]}
        foxes = {"shots": [{"n": 1, "character_id": None,
                            "description": "Two foxes in overalls weld a strut."}]}
        self.assertTrue(P._premise_lost(robot, [], premise),
                        "a robot must not satisfy a brief that asked for a fox")
        self.assertFalse(P._premise_lost(foxes, [], premise))

    def test_irregular_plurals_still_match(self):
        premise = P._premise_species("the soldiers are humanoid wolves")
        self.assertIn("wolves", premise)
        one = {"shots": [{"n": 1, "character_id": None,
                          "description": "A wolf sharpens his bayonet."}]}
        self.assertFalse(P._premise_lost(one, [], premise),
                         "'wolf' must satisfy a brief that said 'wolves'")

    def test_a_premise_repair_that_breaks_a_law_is_rejected(self):
        """The review's probe: a returned shot containing 'explains the mission'
        with no dialogue tags, accepted, under a warning claiming the premise was
        back. The premise is the lesser law — a film that keeps its animals but
        babbles is worse than one that lost them."""
        concept = "ww2 scene but the soldiers are humanoid wolves"
        plain = json.dumps({"title": "Op", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, two soldiers drag a crate."),
        ]})
        # The premise comes back, but so does an unwritten speech act.
        unlawful = json.dumps({"title": "Op", "shots": [
            _shot(2, description="Live-action, cinematic, two humanoid wolves drag a "
                                 "crate. The sergeant explains the mission to them.")]})
        spec, _ = _plan([plain, unlawful], n_shots=2, concept=concept,
                        characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        shot2 = spec["shots"][1]
        # Whatever survived, it must be LAWFUL — either neutralised or rejected.
        self.assertFalse(P._speech_violations(shot2.get("description", ""),
                                              shot2.get("soundscape", "")),
                         "an unlawful repair was shipped: %r" % shot2.get("description"))
        # And the warning must record what actually happened. "the premise is
        # back" is honest ONLY once the shot is lawful — which is the whole
        # point: the claim and the state have to match.
        warns = " ".join((spec.get("_planner") or {}).get("warnings") or [])
        self.assertTrue(
            ("needed silencing to stay lawful" in warns) or ("REJECTED" in warns),
            "the unlawful repair was accepted silently: %s" % warns)

    def test_the_final_scan_runs_after_the_last_mutation(self):
        # The structural hole: each pass validated only its own condition, so the
        # LAST repair could undo an earlier guarantee with nothing left to look.
        self.assertTrue(hasattr(P, "_assert_final_invariants"))
        spec = {"shots": [{"n": 1, "character_id": "bizarrotrn",
                           "engine": "ltx", "camera": "static", "face": "medium",
                           "settle": "he is still", "music": "N/A",
                           "soundscape": "Rain on canvas.",
                           "description": "bizarrotrn Bizarro, a grizzled badger, "
                                          "explains the mission to the unit."}]}
        warnings = []
        out, degraded = P._assert_final_invariants(
            spec, self.CAST, warnings, style="", sb=None)
        s = out["shots"][0]
        self.assertFalse(P._appearance_violations(s["description"], "bizarrotrn", "Bizarro"))
        self.assertFalse(P._speech_violations(s["description"], s["soundscape"]))
        self.assertTrue(any("final invariant" in w for w in warnings), warnings)

    def test_an_unrepairable_plan_says_so_out_loud(self):
        spec = {"shots": [{"n": 1, "character_id": None, "engine": "ltx",
                           "camera": "static", "face": "medium", "settle": "still",
                           "music": "N/A", "soundscape": "Wind.",
                           "description": "A radio voice reads the coordinates aloud."}]}
        warnings = []
        _out, _deg = P._assert_final_invariants(
            spec, self.CAST, warnings, style="", sb=None)
        joined = " ".join(warnings)
        if P._speech_violations(spec["shots"][0]["description"],
                                spec["shots"][0]["soundscape"]):
            self.assertIn("UNREPAIRED", joined,
                          "a plan shipped with a known violation must say so")

    def test_multi_term_premise_needs_ALL_its_terms(self):
        """Codex's probe: "a fox and badger premise is considered preserved when
        only one appears". Half the premise silently dropped, check said fine."""
        premise = P._premise_species("a heist film where the crew are a fox and a badger")
        self.assertEqual(premise, ["badger", "fox"])
        two_shots = lambda a, b: {"shots": [
            {"n": 1, "character_id": None, "description": a},
            {"n": 2, "character_id": None, "description": b}]}
        only_fox = two_shots("A fox in a tuxedo cracks the safe.", "Rain on the car.")
        self.assertTrue(P._premise_lost(only_fox, [], premise),
                        "half a premise is not a preserved premise")
        self.assertEqual(P._premise_missing_terms(only_fox, [], premise), ["badger"])
        both = two_shots("A fox cracks the safe.", "A badger keeps watch.")
        self.assertFalse(P._premise_lost(both, [], premise))

    def test_the_presence_budget_does_not_demand_the_impossible(self):
        """A brief naming five species cannot show five in a one-shot film, and
        the format's own composition limits forbid crowding a shot to satisfy a
        checker. The requirement is capped by the number of uncast shots."""
        premise = P._premise_species("the crew are a fox and a badger")
        one_shot = {"shots": [{"n": 1, "character_id": None,
                               "description": "A fox cracks the safe."}]}
        self.assertFalse(P._premise_lost(one_shot, [], premise),
                         "with room for one creature, one is enough")

    def test_the_final_pass_checks_the_premise_too(self):
        # It checked L12/L13 and stopped, so a failed premise repair was the last
        # word on the plan and nothing downstream ever re-asked.
        premise = ["wolves"]
        spec = {"shots": [{"n": 1, "character_id": None, "engine": "ltx",
                           "camera": "static", "face": "medium", "settle": "still",
                           "music": "N/A", "soundscape": "Wind.",
                           "description": "Two soldiers drag a crate through mud."}]}
        warnings = []
        _out, degraded = P._assert_final_invariants(
            spec, self.CAST, warnings, style="", sb=None, premise=premise)
        self.assertTrue(degraded, "a lost premise must degrade the plan")
        self.assertIn("LOST PART OF ITS PREMISE", " ".join(degraded))

    def test_a_degraded_plan_does_not_claim_ok(self):
        """A failed premise repair must not stamp _planner.ok=True."""
        concept = "ww2 scene but the soldiers are humanoid wolves"
        # Both the plan and its repair omit the premise entirely.
        plain = json.dumps({"title": "Op", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, two soldiers drag a crate."),
        ]})
        still_plain = json.dumps({"title": "Op", "shots": [
            _shot(2, description="Live-action, cinematic, two soldiers lift a crate.")]})
        spec, _ = _plan([plain, still_plain], n_shots=2, concept=concept,
                        characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), "still a usable storyboard")
        blk = spec["_planner"]
        self.assertFalse(blk["ok"], "a degraded plan must not report ok=True")
        self.assertTrue(blk["degraded"])
        self.assertTrue(blk["degraded_reasons"])
        self.assertIn("PREMISE", " ".join(blk["degraded_reasons"]).upper())

    def test_a_clean_plan_still_reports_ok(self):
        ok = json.dumps({"title": "Op", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, a humanoid wolf lifts a crate."),
        ]})
        spec, _ = _plan([ok], n_shots=2, concept="soldiers are humanoid wolves",
                        characters=self.CAST, engine="ltx")
        blk = spec["_planner"]
        self.assertTrue(blk["ok"])
        self.assertFalse(blk["degraded"])
        self.assertEqual(blk["degraded_reasons"], [])

    def test_a_shot_reroll_is_scanned_like_any_other_plan(self):
        """The enforcement block is skipped for fb_mode == "shot" — correctly,
        because re-planning a re-plan from inside itself turns a 25 s call into
        four minutes. But it skipped the final SCAN too, so a re-rolled shot
        carrying an L13 violation came back ok: true, degraded: false, empty
        reasons. Codex's probe, as a test."""
        prev = {"schema": 1, "id": "sb-x", "title": "Op", "created_at": 0, "cast": [],
                "policy": P.default_policy(),
                "shots": [{"n": 1, "title": "a", "mode": "text", "engine": "ltx",
                           "tier": "draft", "prompt": "x", "duration_s": 5.0, "seed": 1,
                           "refs": [], "status": "pending",
                           "description": "A quiet room.", "camera": "static",
                           "face": "medium", "settle": "still",
                           "soundscape": "Wind.", "music": "N/A"}]}
        reroll = json.dumps({"title": "Op", "shots": [_shot(
            1, description="Live-action, cinematic, he explains the mission to the unit.",
            soundscape="The low murmur of voices.")]})
        spec, _ = _plan([reroll], n_shots=1, feedback={"shot": 1, "note": "redo"},
                        previous=prev)
        self.assertFalse(P.is_plan_error(spec), spec)
        s = spec["shots"][0]
        # The violation must not survive into the plan the user gets...
        self.assertFalse(P._speech_violations(s["description"], s["soundscape"]),
                         "a re-rolled shot shipped an L13 violation: %r" % s["description"])
        self.assertNotIn("explains", s["description"])
        self.assertNotIn("murmur", s["soundscape"])
        # ...and the assembled prompt must be rebuilt, not left stale.
        self.assertNotIn("explains", s["prompt"])
        self.assertTrue(any("final invariant" in w
                            for w in (spec["_planner"].get("warnings") or [])),
                        spec["_planner"].get("warnings"))

    def test_a_clean_shot_reroll_stays_clean_and_quiet(self):
        prev = {"schema": 1, "id": "sb-y", "title": "Op", "created_at": 0, "cast": [],
                "policy": P.default_policy(),
                "shots": [{"n": 1, "title": "a", "mode": "text", "engine": "ltx",
                           "tier": "draft", "prompt": "x", "duration_s": 5.0, "seed": 1,
                           "refs": [], "status": "pending",
                           "description": "A quiet room.", "camera": "static",
                           "face": "medium", "settle": "still",
                           "soundscape": "Wind.", "music": "N/A"}]}
        reroll = json.dumps({"title": "Op", "shots": [_shot(
            1, description="Live-action, cinematic, he traces the river with one finger.")]})
        spec, _ = _plan([reroll], n_shots=1, feedback={"shot": 1, "note": "redo"},
                        previous=prev)
        blk = spec["_planner"]
        self.assertTrue(blk["ok"])
        self.assertFalse(blk["degraded"])
        self.assertFalse(any("final invariant" in w for w in (blk.get("warnings") or [])))

    def test_a_film_that_kept_its_premise_costs_no_extra_call(self):
        concept = "ww2 scene but the soldiers are humanoid animals"
        ok = json.dumps({"title": "Operation Wildfire", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, a humanoid badger sergeant in a "
                                 "nazi greatcoat sneers across the table."),
        ]})
        spec, stub = _plan([ok], n_shots=2, concept=concept, characters=self.CAST,
                           engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.calls), 1)

    def test_a_film_with_no_creature_premise_is_never_premise_checked(self):
        ok = json.dumps({"title": "The Key", "shots": [
            _shot(1, description=self.GOOD, character_id="bizarrotrn"),
            _shot(2, description="Live-action, cinematic, rain beads on a brass key."),
        ]})
        spec, stub = _plan([ok], n_shots=2, concept="a lost brass key finds its door",
                           characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.calls), 1, "an ordinary film must not pay for this check")

    def test_one_shot_breaking_both_laws_is_re_planned_once(self):
        both = (self.BAD.rstrip(".") + ", and he explains the mission, his voice low.")
        spec, stub = _plan([self._film(both), self._film(both)], n_shots=2,
                           characters=self.CAST, engine="ltx")
        self.assertFalse(P.is_plan_error(spec), spec)
        # Exactly one re-plan call for the one offending shot, carrying BOTH complaints.
        self.assertEqual(len(stub.calls), 2, [c["user"][:40] for c in stub.calls])
        self.assertIn("APPEARANCE (L12)", stub.calls[1]["user"])
        self.assertIn("SPEECH (L13)", stub.calls[1]["user"])


class TestScreenplayPass(unittest.TestCase):
    """Pass one writes the SCENE; pass two shoots it.

    Owner, on the films this planner produced before it existed: "the prompt
    writing should be like a movie director... The action is well planned. You
    understand what I mean? It is not working properly. It's just a succession
    of shots." A film is written screenplay -> shot breakdown -> per-shot
    polish; this had no screenplay step at all, so structure and coverage were
    being invented in the same breath.
    """
    SCENE = "\n".join([
        "BEAT - He throws both arms wide in front of the soapy car.",
        'BIZARRO: "Ladies and gentlemen."',
        "BEAT - She keeps scrubbing and does not look up.",
        'ARIA: "Update the app."',
    ])

    def _film(self, desc="A man stands in the rain."):
        return json.dumps({"title": "T", "shots": [
            {"n": i + 1, "title": "B%d" % (i + 1), "character_id": None,
             "duration_s": 5, "camera": "static", "description": desc,
             "settle": "he is still", "soundscape": "Rain, no voices.",
             "music": "N/A"} for i in range(2)]})

    def test_the_scene_is_written_before_the_shots(self):
        spec, stub = _plan([self._film()], n_shots=2, screenplay_text=self.SCENE)
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.screenplay_calls), 1)
        self.assertIn("You are a screenwriter", stub.screenplay_calls[0]["system"])

    def test_the_scene_reaches_the_shot_brief_verbatim(self):
        spec, stub = _plan([self._film()], n_shots=2, screenplay_text=self.SCENE)
        brief = stub.calls[0]["user"]
        self.assertIn("THE SCENE", brief)
        self.assertIn('BIZARRO: "Ladies and gentlemen."', brief)
        self.assertIn("keep every spoken line", brief)

    def test_a_model_that_answers_with_chatter_is_ignored(self):
        # Small models like to open with "Sure! Here is the scene:" and a
        # paragraph that is not a beat. Handing that down as if it were the
        # scene is worse than having no scene.
        _, stub = _plan([self._film()], n_shots=2,
                        screenplay_text="Sure! Here is a great scene for you.\nEnjoy!")
        self.assertNotIn("THE SCENE", stub.calls[0]["user"])

    def test_no_screenplay_leaves_the_brief_exactly_as_it_was(self):
        _, stub = _plan([self._film()], n_shots=2, screenplay_text="")
        self.assertNotIn("THE SCENE", stub.calls[0]["user"])

    def test_it_can_be_turned_off(self):
        _, stub = _plan([self._film()], n_shots=2, screenplay=False,
                        screenplay_text=self.SCENE)
        self.assertEqual(stub.screenplay_calls, [])

    def test_a_reroll_does_not_rewrite_the_scene(self):
        # A per-shot re-roll is fixing ONE shot. Regenerating the screenplay
        # under it would move the ground the other shots are standing on.
        base, _ = _plan([_plan_json(6)])
        _, stub = _plan([_plan_json(1)], n_shots=6, previous=base,
                        feedback={"shot": 1, "note": "again"},
                        screenplay_text=self.SCENE)
        self.assertEqual(stub.screenplay_calls, [])


# --------------------------------------------------------------------------------------
# The geography pass — the screenplay is TIME, this is SPACE
# --------------------------------------------------------------------------------------

CARWASH = {"id": "carwash", "name": "The car wash",
           "description": "a suburban driveway on a bright afternoon"}

# What the blocking model is asked for, on the day's own scene. Everything the
# feature promises is in these two views: the reverse does not hold the car and
# SAYS so, and the sun that rakes in from camera left rakes in from camera right
# once the camera has turned around.
GEO_JSON = json.dumps({
    "floor_plan":
        "The driveway runs left to right in front of the garage. BIZARRO stands at the "
        "near end of a soapy blue sedan facing the street; behind him is the garage door. "
        "ARIA crouches at the front wheel with a sponge. Across the street, behind "
        "BIZARRO's eyeline, is a row of low houses and a hand-painted sign on the verge. "
        "The low afternoon sun comes over the houses.",
    "views": [
        {"location": "The car wash", "id": "establishing",
         "name": "Establishing - facing the driveway", "light": "camera left",
         "description": "the soapy blue sedan on the driveway with a woman crouched at "
                        "the front wheel, the garage door behind them"},
        {"location": "The car wash", "id": "reverse",
         "name": "Reverse - facing the street", "light": "camera right",
         "description": "the row of low houses across the street and a hand-painted sign "
                        "on the far verge, no car in frame"},
    ]})


def _geo_film(shots):
    return json.dumps({"title": "The Car Wash", "shots": shots})


class TestGeographyPass(unittest.TestCase):
    """The planner learns SPACE.

    Owner, after a full day of manual continuity work: "Do you notice all the
    work I had to do to make this scene happen in the same place and have
    proper angles? ... You need to first make a concept of the whole situation.
    For instance, a man or woman in a bar — behind him there is this, behind her
    there is that. When they sit together, this is what you see."

    The whole day was the prototype: carwash/carwash_reverse, the flipped sun,
    the off-frame eyelines, the no-car reverse background — all hand-built, all
    derivable from one paragraph of floor plan.
    """

    def _film(self, n=2, **kw):
        rows = []
        for i in range(n):
            row = {"n": i + 1, "title": "B%d" % (i + 1), "character_id": None,
                   "duration_s": 5, "camera": "static",
                   "description": "A man throws both arms wide.",
                   "settle": "he is still", "soundscape": "Hose water, no voices.",
                   "music": "N/A", "location": "The car wash"}
            row.update(kw)
            rows.append(row)
        return _geo_film(rows)

    def _plan_carwash(self, replies=None, **kw):
        kw.setdefault("n_shots", 2)
        kw.setdefault("locations", [dict(CARWASH)])
        kw.setdefault("geography_text", GEO_JSON)
        return _plan(replies or [self._film()], **kw)

    # ---- the pass itself ----------------------------------------------------
    def test_the_floor_plan_is_written_before_the_shots(self):
        spec, stub = self._plan_carwash()
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual(len(stub.geography_calls), 1)
        self.assertIn("You are a director blocking a scene",
                      stub.geography_calls[0]["system"])
        self.assertIn("The car wash", stub.geography_calls[0]["user"])

    def test_it_reads_the_scene_so_it_knows_who_turns_to_whom(self):
        scene = "\n".join(['BEAT - He turns to her.',
                           'BIZARRO: "Ladies and gentlemen."',
                           'BEAT - She keeps scrubbing.',
                           'ARIA: "Update the app."'])
        _, stub = self._plan_carwash(screenplay_text=scene)
        self.assertIn("THE SCENE THAT HAPPENS HERE", stub.geography_calls[0]["user"])
        self.assertIn("Ladies and gentlemen", stub.geography_calls[0]["user"])

    def test_the_views_are_merged_into_the_boards_locations(self):
        spec, _ = self._plan_carwash()
        views = spec["locations"][0]["views"]
        self.assertEqual([v["id"] for v in views], ["establishing", "reverse"])
        self.assertIn("sedan", views[0]["description"])
        self.assertIn("houses", views[1]["description"])
        self.assertNotIn("sedan", views[1]["description"])

    def test_the_light_flips_at_180_degrees(self):
        # Measured on the real cut, and the one thing a second hand-authored
        # location got right only because a human remembered to flip it.
        spec, _ = self._plan_carwash()
        front, back = spec["locations"][0]["views"]
        self.assertIn("camera left", front["description"])
        self.assertNotIn("camera right", front["description"])
        self.assertIn("camera right", back["description"])
        self.assertNotIn("camera left", back["description"])

    def test_the_views_and_the_laws_reach_the_shot_brief(self):
        _, stub = self._plan_carwash()
        brief = stub.calls[0]["user"]
        self.assertIn("VIEWS of The car wash", brief)
        self.assertIn("establishing", brief)
        self.assertIn("reverse", brief)
        self.assertIn("THE FLOOR PLAN", brief)
        self.assertIn("behind him is the garage door", brief)
        self.assertIn("THE 180-DEGREE RULE", brief)
        self.assertIn("Never put a character's own body in the view behind them", brief)

    def test_views_without_a_floor_plan_still_reach_the_model(self):
        # THE COERCION BUG. `_geography_plan` derives the floor plan from the
        # model's own optional key, so a reply carrying views and no plan
        # legitimately returns ("", {...views...}) — and the shot prompt was
        # only rebuilt `if scene_text or floor_plan`. With the screenplay also
        # empty (routine: `_screenplay_text` returns "" whenever fewer than
        # three lines match the form), `locs` had been replaced by the merged
        # list while the brief was still the pre-geography one: no view ids,
        # no laws. The model named no view, `_apply_geography` found views on
        # the location and none on any shot, and stamped views[0] on every
        # shot — the whole film pinned to the establishing angle, which is the
        # behaviour this pass exists to end.
        geo = json.dumps({"views": json.loads(GEO_JSON)["views"]})
        film = self._film(view="reverse")
        spec, stub = self._plan_carwash([film], geography_text=geo,
                                        screenplay_text="")
        self.assertFalse(P.is_plan_error(spec), spec)
        brief = stub.calls[0]["user"]
        self.assertIn("VIEWS of The car wash", brief)
        self.assertIn("establishing", brief)
        self.assertIn("reverse", brief)
        self.assertIn("THE 180-DEGREE RULE", brief)
        # ...and a model that WAS told keeps the view it named, rather than
        # being coerced onto the establishing angle it never saw.
        self.assertEqual(spec["shots"][0]["view"], "reverse")
        self.assertFalse([w for w in spec["_planner"]["warnings"]
                          if "the establishing view" in w],
                         spec["_planner"]["warnings"])

    def test_a_film_with_no_views_reads_exactly_as_it_did_before(self):
        _, stub = _plan([self._film()], n_shots=2, locations=[dict(CARWASH)],
                        geography=False)
        brief = stub.calls[0]["user"]
        self.assertIn("LOCATIONS", brief)
        self.assertNotIn("VIEWS of", brief)
        self.assertNotIn("THE 180-DEGREE RULE", brief)
        self.assertNotIn("THE FLOOR PLAN", brief)

    def test_chatter_is_discarded_and_the_plan_proceeds(self):
        # The screenplay pass's rule, one pass over: a model that answers a
        # blocking brief with enthusiasm must leave the plan behaving exactly
        # as it did before this pass existed.
        spec, stub = self._plan_carwash(
            geography_text="Sure! Here is a great layout for your scene. Enjoy!")
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertNotIn("views", spec["locations"][0])
        self.assertNotIn("VIEWS of", stub.calls[0]["user"])
        self.assertTrue(any("no usable floor plan" in w
                            for w in spec["_planner"]["warnings"]),
                        spec["_planner"]["warnings"])

    def test_it_can_be_turned_off(self):
        _, stub = self._plan_carwash(geography=False)
        self.assertEqual(stub.geography_calls, [])

    def test_a_film_with_nowhere_to_be_does_not_pay_for_a_floor_plan(self):
        # There is nowhere to hang a view and no board field to carry it.
        _, stub = _plan([_plan_json(2)], n_shots=2, geography_text=GEO_JSON)
        self.assertEqual(stub.geography_calls, [])

    def test_a_reroll_does_not_reblock_the_scene(self):
        # The other shots are standing on this geography.
        base, _ = self._plan_carwash()
        _, stub = self._plan_carwash([self._film(1)], previous=base,
                                     feedback={"shot": 1, "note": "again"})
        self.assertEqual(stub.geography_calls, [])

    # ---- what lands on the shots -------------------------------------------
    def test_a_shot_carries_its_view_and_its_eyeline(self):
        spec, _ = self._plan_carwash([self._film(view="reverse", eyeline="frame-left")])
        self.assertEqual(spec["shots"][0]["view"], "reverse")
        self.assertEqual(spec["shots"][0]["eyeline"], "left")
        self.assertEqual(P._load_validator()[0](spec), [])

    def test_a_shot_that_names_no_view_is_put_on_the_establishing_one(self):
        # Unspecified coverage is the master. Leaving it blank composes the
        # location's own neutral sentence — the pre-views behaviour this pass
        # exists to end.
        spec, _ = self._plan_carwash()
        self.assertEqual(spec["shots"][0]["view"], "establishing")
        self.assertTrue(any("no view of The car wash" in w
                            for w in spec["_planner"]["warnings"]),
                        spec["_planner"]["warnings"])

    def test_a_view_that_is_not_on_the_floor_plan_never_reaches_the_board(self):
        # `unknown_view` is a hard validator error; writing the model's
        # invention onto the shot would fail the plan outright.
        spec, _ = self._plan_carwash([self._film(view="from_the_roof")])
        self.assertEqual(spec["shots"][0]["view"], "establishing")
        self.assertEqual(P._load_validator()[0](spec), [])

    def test_an_eyeline_outside_the_vocabulary_is_dropped_not_written(self):
        spec, _ = self._plan_carwash([self._film(eyeline="over the shoulder")])
        self.assertNotIn("eyeline", spec["shots"][0])
        self.assertEqual(P._load_validator()[0](spec), [])
        self.assertTrue(any("not left, right or lens" in w
                            for w in spec["_planner"]["warnings"]))

    def test_every_way_a_model_says_a_side_lands_in_the_vocabulary(self):
        for raw, want in (("frame-right", "right"), ("screen left", "left"),
                          ("Camera Right", "right"), ("to camera", "lens"),
                          ("LENS", "lens"), ("left.", "left")):
            self.assertEqual(P._eyeline_key(raw), want, raw)
        for junk in ("over the shoulder", "up", "", None, "north"):
            self.assertEqual(P._eyeline_key(junk), "")

    def test_a_film_level_replan_is_shown_the_geography_it_must_keep(self):
        # Without this the re-plan came back with every shot re-coerced onto
        # the establishing view: nothing in the plan the model was shown said
        # which way any of them had been pointing.
        base, _ = self._plan_carwash([self._film(view="reverse", eyeline="right")])
        _, stub = self._plan_carwash([self._film(view="reverse", eyeline="right")],
                                     previous=base, feedback="make it colder")
        brief = stub.calls[0]["user"]
        self.assertIn('"view": "reverse"', brief)
        self.assertIn('"eyeline": "right"', brief)
        self.assertIn('"location": "carwash"', brief)

    def test_the_plan_carries_the_places_its_shots_name(self):
        # THE REGRESSION: shots have been stamped with `location_id` since
        # locations existed and the spec never carried `locations`, so the
        # validator returned `unknown_location` for EVERY shot — a brief with a
        # location in it spent its one repair round-trip on a fault no model
        # could fix and came back `invalid_plan`.
        spec, _ = _plan([self._film()], n_shots=2, locations=[dict(CARWASH)])
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual([l["id"] for l in spec["locations"]], ["carwash"])
        self.assertEqual(spec["shots"][0]["location_id"], "carwash")
        self.assertEqual(P._load_validator()[0](spec), [])


class TestGeographyLaws(unittest.TestCase):
    """Warning level, both of them, and neither can fail a plan.

    A film whose eyelines are a little loose is still a film; a film that will
    not render is not.
    """
    LOCS = [dict(CARWASH, views=[
        {"id": "establishing", "name": "Establishing",
         "description": "the soapy blue sedan on the driveway, camera left light"},
        {"id": "reverse", "name": "Reverse",
         "description": "the row of low houses across the street, no car in frame, "
                        "camera right light"},
    ])]

    def _spec(self, *shots):
        return {"shots": [dict({"n": i + 1, "location_id": "carwash"}, **s)
                          for i, s in enumerate(shots)]}

    # ---- the 180-degree line ------------------------------------------------
    def test_two_people_cannot_both_look_the_same_way(self):
        warn = []
        spec = P._enforce_eyelines(
            self._spec({"character_id": "bizarrotrn", "eyeline": "right"},
                       {"character_id": "ariatrn", "eyeline": "right"}), warn)
        self.assertEqual(spec["shots"][1]["eyeline"], "left")
        self.assertIn("180-degree line", warn[0])

    def test_a_complementary_pair_is_left_alone(self):
        warn = []
        spec = P._enforce_eyelines(
            self._spec({"character_id": "bizarrotrn", "eyeline": "right"},
                       {"character_id": "ariatrn", "eyeline": "left"}), warn)
        self.assertEqual([s["eyeline"] for s in spec["shots"]], ["right", "left"])
        self.assertEqual(warn, [])

    def test_the_same_person_twice_is_not_a_reverse(self):
        # Two shots of one man looking the same way is continuity, not a break.
        warn = []
        P._enforce_eyelines(
            self._spec({"character_id": "bizarrotrn", "eyeline": "right"},
                       {"character_id": "bizarrotrn", "eyeline": "right"}), warn)
        self.assertEqual(warn, [])

    def test_a_cut_to_another_place_does_not_cross_the_line(self):
        warn = []
        spec = self._spec({"character_id": "bizarrotrn", "eyeline": "right"},
                          {"character_id": "ariatrn", "eyeline": "right"})
        spec["shots"][1]["location_id"] = "kitchen"
        P._enforce_eyelines(spec, warn)
        self.assertEqual(spec["shots"][1]["eyeline"], "right")
        self.assertEqual(warn, [])

    def test_the_lens_and_the_unstated_are_never_flipped(self):
        # "lens" is a claim; nothing is not. Neither is a side of the line.
        warn = []
        spec = P._enforce_eyelines(
            self._spec({"character_id": "a", "eyeline": "lens"},
                       {"character_id": "b", "eyeline": "lens"},
                       {"character_id": "a"}, {"character_id": "b"}), warn)
        self.assertEqual([s.get("eyeline") for s in spec["shots"]],
                         ["lens", "lens", None, None])
        self.assertEqual(warn, [])

    # ---- what the reverse angle exists to keep out --------------------------
    def test_the_car_is_not_allowed_back_into_the_reverse(self):
        warn = []
        spec = self._spec({"view": "reverse",
                           "description": "He turns from the car and grins at her."})
        P._enforce_geography(spec, self.LOCS, warn)
        self.assertEqual(len(warn), 1, warn)
        self.assertIn("shot 1", warn[0])
        self.assertIn("'reverse'", warn[0])
        self.assertIn("car", warn[0])

    def test_the_same_object_on_the_view_that_holds_it_is_fine(self):
        warn = []
        P._enforce_geography(
            self._spec({"view": "establishing",
                        "description": "He turns from the car and grins at her."}),
            self.LOCS, warn)
        self.assertEqual(warn, [])

    def test_the_other_person_in_a_reverse_is_never_a_violation(self):
        # An earlier draft inferred absence by diffing the content words of the
        # other views, and flagged exactly the thing a reverse angle is FOR:
        # her, her sponge, her wheel, all legitimately in frame.
        warn = []
        P._enforce_geography(
            self._spec({"view": "reverse",
                        "description": "The woman looks up from the front wheel, sponge "
                                       "in hand, and answers him."}),
            self.LOCS, warn)
        self.assertEqual(warn, [])

    # ---- the wiring ---------------------------------------------------------
    def test_both_laws_run_inside_plan_film(self):
        # A law that is only unit-tested is a law that can be left unwired.
        cast = [{"id": "bizarrotrn", "trigger": "bizarrotrn", "name": "Bizarro"},
                {"id": "ariatrn", "trigger": "ariatrn", "name": "Aria"}]
        rows = [
            {"n": 1, "title": "A", "character_id": "bizarrotrn", "duration_s": 5,
             "camera": "static", "description": "bizarrotrn throws both arms wide and grins.",
             "settle": "he is still", "soundscape": "Hose water, no voices.",
             "music": "N/A", "location": "The car wash", "view": "establishing",
             "eyeline": "right"},
            {"n": 2, "title": "B", "character_id": "ariatrn", "duration_s": 5,
             "camera": "static",
             "description": "ariatrn lifts the sponge from the car and shrugs.",
             "settle": "she is still", "soundscape": "Hose water, no voices.",
             "music": "N/A", "location": "The car wash", "view": "reverse",
             "eyeline": "right"},
        ]
        spec, _ = _plan([json.dumps({"title": "The Car Wash", "shots": rows})],
                        n_shots=2, characters=cast, engine="ltx",
                        locations=[dict(CARWASH)], geography_text=GEO_JSON)
        self.assertFalse(P.is_plan_error(spec), spec)
        self.assertEqual([s["eyeline"] for s in spec["shots"]], ["right", "left"])
        self.assertEqual(P._load_validator()[0](
            spec, known_character_ids=["bizarrotrn", "ariatrn"]), [])
        warns = spec["_planner"]["warnings"]
        self.assertTrue(any("180-degree line" in w for w in warns), warns)
        self.assertTrue(any("puts the car back in frame" in w for w in warns), warns)
        # WARNING level: a loose eyeline is not a reason to call a film broken.
        self.assertTrue(spec["_planner"]["ok"])
        self.assertFalse(spec["_planner"]["degraded"])

    def test_absence_is_read_from_the_views_own_words(self):
        self.assertEqual(P._absent_terms("houses across the street, no car in frame"),
                         ["car"])
        self.assertEqual(P._absent_terms("the verge, without the hand-painted sign"),
                         ["hand-painted", "sign"])
        # A view that negates nothing constrains nothing.
        self.assertEqual(P._absent_terms("the row of low houses across the street"), [])
        self.assertEqual(P._absent_terms("the sign is not visible from here"), [])


class TestGeographyParser(unittest.TestCase):
    """`_geography_plan`: everything between the model's JSON and the board."""

    def _plan_from(self, obj, locs=(CARWASH,)):
        return P._geography_plan({"text": json.dumps(obj)}, [dict(l) for l in locs])

    def test_the_carwash_reply_parses_into_two_views(self):
        floor, views = P._geography_plan({"text": GEO_JSON}, [dict(CARWASH)])
        self.assertIn("garage door", floor)
        self.assertEqual([v["id"] for v in views["carwash"]],
                         ["establishing", "reverse"])

    def test_the_light_side_is_folded_into_the_sentence_that_renders(self):
        # The side is the most load-bearing word in a reverse angle, and the
        # model likes to answer it in `light` and forget it in the prose.
        _, views = self._plan_from({"floor_plan": "x", "views": [
            {"location": "The car wash", "id": "reverse",
             "light": "camera right",
             "description": "the row of low houses across the street"}]})
        self.assertIn("the light rakes in from camera right",
                      views["carwash"][0]["description"])

    def test_a_sentence_that_already_names_a_side_is_left_alone(self):
        _, views = self._plan_from({"floor_plan": "x", "views": [
            {"location": "The car wash", "id": "reverse", "light": "camera right",
             "description": "the houses, the sun raking in from camera right"}]})
        self.assertEqual(views["carwash"][0]["description"].count("camera right"), 1)

    def test_ids_are_slugged_deduped_and_capped(self):
        _, views = self._plan_from({"floor_plan": "x", "views": [
            {"location": "The car wash", "name": "Establishing — facing the drive",
             "description": "the soapy sedan on the driveway, camera left"},
            {"location": "The car wash", "id": "Reverse Angle!",
             "description": "the houses across the street, no car, camera right"},
            {"location": "The car wash", "id": "reverse_angle",
             "description": "the houses again, a second time, camera right"},
            {"location": "The car wash", "id": "d",
             "description": "over her shoulder at the garage door, camera left"},
            {"location": "The car wash", "id": "e",
             "description": "the fifth angle nobody asked for, camera left"},
        ]})
        ids = [v["id"] for v in views["carwash"]]
        self.assertEqual(len(ids), P.MAX_VIEWS_PER_LOCATION)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(P._VIEW_ID_RE.match(i) for i in ids), ids)
        self.assertEqual(ids[0], "establishing_facing_the_drive")

    def test_a_view_with_no_description_is_not_a_view(self):
        # An id with nothing under it composes the location's own text and
        # pretends coverage exists.
        floor, views = self._plan_from({"floor_plan": "x", "views": [
            {"location": "The car wash", "id": "reverse", "description": "too short"}]})
        self.assertEqual((floor, views), ("", {}))

    def test_a_single_location_takes_the_views_that_name_nothing(self):
        _, views = self._plan_from({"floor_plan": "x", "views": [
            {"id": "reverse", "description": "the houses across the street, camera right"}]})
        self.assertEqual(list(views), ["carwash"])

    def test_a_view_for_a_place_that_is_not_in_the_film_is_dropped(self):
        floor, views = self._plan_from(
            {"floor_plan": "x", "views": [
                {"location": "The bar", "id": "wide",
                 "description": "the bar shelves behind him, camera left"}]},
            locs=(CARWASH, {"id": "kitchen", "name": "The kitchen",
                            "description": "a kitchen"}))
        self.assertEqual((floor, views), ("", {}))

    def test_chatter_and_broken_json_yield_nothing(self):
        for text in ("Sure! Here is the layout.", "", "{not json",
                     json.dumps({"floor_plan": "x"}),
                     json.dumps({"floor_plan": "x", "views": "wide"})):
            self.assertEqual(P._geography_plan({"text": text}, [dict(CARWASH)]),
                             ("", {}), text)

    def test_the_users_own_locations_are_never_rewritten_in_place(self):
        # On the panel `locations` IS the user's Locations box, parsed.
        mine = [dict(CARWASH)]
        merged = P._merge_views(mine, {"carwash": [{"id": "a", "name": "A",
                                                    "description": "d"}]})
        self.assertNotIn("views", mine[0])
        self.assertIn("views", merged[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
