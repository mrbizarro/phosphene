#!/usr/bin/env python3
"""The Director: a track on the storyboard brief turns the plan into a music
video cut to the beat.

What is locked: the grid is the same downbeat fit the Editor's Prepare runs;
the shot count and every duration come from the track and not from the
chips; the brief the planner reads carries the grid and the two laws that
have already cost renders (lead with the movement; no dialogue under a
replaced track); a bad path is refused before the planner runs; and a board
planned this way opens in the Editor already cut under its track."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlx_ltx_panel as panel                                        # noqa: E402
import storyboard_edit as sedt                                       # noqa: E402


def _beats(bpm=128.0, bars=40, meter=4):
    bar = 60.0 / bpm * meter
    downs = [round(i * bar, 6) for i in range(bars)]
    return {"bpm": bpm, "downbeats": downs, "beats": downs,
            "confidence": 0.9, "duration": round(bars * bar, 6)}


def _flat_map(*a, **k):
    """No sections: the base stride throughout."""
    return {"bpm": 128.0, "bars": [], "sections": [], "mean_energy": 0.5}


class TheGrid(unittest.TestCase):
    def test_slots_are_downbeats_every_n_bars(self):
        with mock.patch.object(sedt, "beat_map", return_value=_beats()), \
                mock.patch.object(sedt, "song_map", side_effect=_flat_map), \
                tempfile_file() as track:
            g = panel._sb_director_grid({"path": track, "bars_per_shot": 2})
        self.assertNotIn("error", g)
        self.assertEqual(g["bpm"], 128.0)
        self.assertEqual(g["bars_per_shot"], 2)
        self.assertEqual(g["count"], 20)
        self.assertAlmostEqual(g["slot_sec"], 3.75, places=2)
        self.assertAlmostEqual(g["slots"][1]["start"], 3.75, places=3)
        self.assertAlmostEqual(g["slots"][-1]["end"], 75.0, places=3)
        self.assertTrue(any("MUSIC VIDEO" in m for m in g["must"]))
        self.assertTrue(any("begin each description with the movement" in m
                            for m in g["must"]))
        self.assertTrue(any("no shot contains dialogue" in m for m in g["must"]))

    def test_a_long_track_is_capped_and_says_so(self):
        with mock.patch.object(sedt, "beat_map", return_value=_beats(bars=200)), \
                mock.patch.object(sedt, "song_map", side_effect=_flat_map), \
                tempfile_file() as track:
            g = panel._sb_director_grid({"path": track, "bars_per_shot": 1},
                                        max_shots=48)
        self.assertEqual(g["count"], 48)
        self.assertIn("first", g["note"])
        self.assertTrue(any("more bars than 48 shots" in m for m in g["must"]))

    def test_a_missing_file_and_a_beatless_track_are_sentences(self):
        self.assertIn("no audio file", panel._sb_director_grid({"path": "/nope.wav"})["error"])
        with mock.patch.object(sedt, "beat_map",
                               return_value={"bpm": 0, "downbeats": [], "duration": 0}), \
                mock.patch.object(sedt, "song_map", side_effect=_flat_map), \
                tempfile_file() as track:
            g = panel._sb_director_grid({"path": track})
        self.assertIn("no beat", g["error"])

    def test_sections_change_the_stride_and_the_brief_names_the_arc(self):
        beats = _beats(bars=24)                     # 24 bars, 1.875 s each
        secs = [{"start": 0.0, "end": 8 * 1.875, "label": "intro", "energy": 0.2,
                 "brightness": 0.3, "bars": [0, 8]},
                {"start": 8 * 1.875, "end": 16 * 1.875, "label": "chorus", "energy": 0.9,
                 "brightness": 0.8, "bars": [8, 16]},
                {"start": 16 * 1.875, "end": 24 * 1.875, "label": "outro", "energy": 0.2,
                 "brightness": 0.3, "bars": [16, 24]}]
        with mock.patch.object(sedt, "beat_map", return_value=beats), \
                mock.patch.object(sedt, "song_map",
                                  return_value={"bpm": 128.0, "sections": secs,
                                                "mean_energy": 0.43, "bars": []}), \
                tempfile_file() as track:
            g = panel._sb_director_grid({"path": track, "bars_per_shot": 2})
        # intro: 8 bars at 4 = 2 slots; chorus: 8 bars at 1 = 8; outro: 2
        self.assertEqual([s["section"] for s in g["slots"]],
                         ["intro"] * 2 + ["chorus"] * 8 + ["outro"] * 2)
        self.assertEqual([s["bars"] for s in g["slots"]][:3], [4, 4, 1])
        self.assertAlmostEqual(g["slots"][2]["end"] - g["slots"][2]["start"], 1.875, places=3)
        self.assertTrue(any("shots 3–10 are the chorus" in m for m in g["must"]))
        self.assertTrue(any("shots 1–2 are the intro" in m for m in g["must"]))
        self.assertEqual(len(g["sections"]), 3)

    def test_a_song_map_failure_falls_back_to_the_base_stride(self):
        with mock.patch.object(sedt, "beat_map", return_value=_beats()), \
                mock.patch.object(sedt, "song_map", side_effect=RuntimeError("boom")), \
                tempfile_file() as track:
            g = panel._sb_director_grid({"path": track, "bars_per_shot": 2})
        self.assertEqual(g["count"], 20)

    def test_the_brief_carries_the_grid(self):
        c = panel._sb_director_concept("A boxer at dawn.",
                                       {"bpm": 128.0, "total_sec": 75.0, "count": 20,
                                        "slot_sec": 3.75, "bars_per_shot": 2})
        self.assertTrue(c.startswith("A boxer at dawn."))
        self.assertIn("128 bpm", c)
        self.assertIn("20 shots of about 3.8s", c)
        self.assertIn("ARC", c)


class TheFirstCutIsOnTheTrack(unittest.TestCase):
    def test_the_boards_soundtrack_is_the_auto_edits_default(self):
        board = {"id": "sb_dir", "title": "d", "shots": [],
                 "soundtrack": {"path": "/x/track.wav"}}
        with mock.patch.object(panel, "_sbe_board_clips", return_value=[]), \
                mock.patch.object(panel, "_sbe_prepare_cache", return_value={}), \
                mock.patch.object(panel, "_sbe_board_dir", return_value=Path("/tmp/x")), \
                mock.patch("storyboard_editor.edit_from_plan") as efp:
            panel._sbe_auto_edit(board)
        # no clips yet: the empty edit is what comes back, and nothing raised
        efp.assert_called_once()

    def test_plan_refuses_a_bad_track_before_the_planner_runs(self):
        from test_storyboard_editor_api import FakeHandler
        h = FakeHandler()
        with mock.patch.object(panel, "_sb_claim_planner", return_value=None), \
                mock.patch.object(panel, "_sb_release_planner"), \
                mock.patch.dict(panel.STATE, {"running": None, "queue": []}), \
                mock.patch.object(panel.threading, "Thread") as th:
            panel.Handler._storyboard_post(
                h, "plan", "", {"concept": "x", "soundtrack": "/nope/track.wav"})
        self.assertEqual(h.status, 400)
        self.assertIn("no audio file", h.payload["error"])
        th.assert_not_called()


class tempfile_file:
    """A real, empty file at a real path — `is_file()` is all the grid asks."""
    def __enter__(self):
        import tempfile
        self._t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._t.close()
        return self._t.name

    def __exit__(self, *a):
        Path(self._t.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()


class TheRulerGetsTheArc(unittest.TestCase):
    def test_the_payload_carries_the_boards_sections(self):
        board = {"id": "sb_dir", "title": "d", "shots": [],
                 "soundtrack": {"path": "/x/t.wav",
                                "sections": [{"start": 0, "end": 30, "label": "intro",
                                              "energy": 0.3}]}}
        edit = {"version": 2, "clips": [], "audio": None}
        with mock.patch.object(panel, "_sbe_board_clips", return_value=[]), \
                mock.patch.object(panel, "_sbe_board_dir", return_value=Path("/tmp/x")), \
                mock.patch.object(panel, "_sbe_proxy_map", return_value={}), \
                mock.patch.object(panel, "_sbe_relinks", return_value=[]), \
                mock.patch.object(panel, "_sbe_job_state", return_value={}), \
                mock.patch("storyboard_editor.list_drafts", return_value=[]), \
                mock.patch("storyboard_editor.load_draft_index", return_value={"active": ""}), \
                mock.patch("storyboard_editor.pending_backup", return_value=None), \
                mock.patch("storyboard_editor.current_session", return_value={}):
            p = panel._sbe_payload(board, edit)
        self.assertEqual(p["sections"][0]["label"], "intro")


class TheAutoPipeline(unittest.TestCase):
    def test_the_brief_stores_the_switch(self):
        from test_storyboard_editor_api import FakeHandler
        seen = {}
        def fake_save(state_dir, board):
            seen["auto"] = board.get("auto")
        with mock.patch.object(panel, "_sb_claim_planner", return_value=None), \
                mock.patch.object(panel, "_sb_release_planner"), \
                mock.patch.dict(panel.STATE, {"running": None, "queue": []}), \
                mock.patch.object(panel.storyboard, "save_storyboard", side_effect=fake_save), \
                mock.patch.object(panel.threading, "Thread") as th:
            th.return_value.start = lambda: None
            h = FakeHandler()
            panel.Handler._storyboard_post(h, "plan", "", {"concept": "x", "auto": "1"})
        self.assertEqual(h.status, 202, h.payload)
        self.assertTrue(seen["auto"])

    def test_the_film_waits_for_a_shot_that_did_not_render(self):
        board = {"id": "b", "title": "t", "auto": True,
                 "shots": [{"n": 1, "draft_output": "/x/a.mp4"}, {"n": 2}]}
        with mock.patch.object(panel, "_sbe_auto_edit") as ae, \
                mock.patch.object(panel, "push") as p:
            self.assertIsNone(panel._sb_auto_film(board))
        ae.assert_not_called()
        self.assertIn("1 shot(s) did not render", p.call_args[0][0])

    def test_the_film_is_cut_and_assembled_when_every_shot_landed(self):
        board = {"id": "b", "title": "t", "auto": True,
                 "shots": [{"n": 1, "draft_output": "/x/a.mp4"},
                           {"n": 2, "status": "skipped"},
                           {"n": 3, "final_output": "/x/c.mp4"}]}
        edit = {"version": 2, "clips": [{"id": "a"}, {"id": "c"}], "beats": {"bpm": 120}}
        with mock.patch.object(panel, "_sbe_auto_edit", return_value=edit) as ae, \
                mock.patch.object(panel, "_sbe_board_dir", return_value=Path("/tmp/x")), \
                mock.patch("storyboard_editor.save_edit") as save, \
                mock.patch("storyboard_editor.load_edit", return_value=edit), \
                mock.patch.object(panel, "_sbe_render_edit",
                                  return_value={"ok": True, "path": "/x/film.mp4"}) as rend, \
                mock.patch.object(panel.storyboard, "save_storyboard") as sb_save, \
                mock.patch.object(panel, "push"):
            film = panel._sb_auto_film(board)
        ae.assert_called_once_with(board)
        save.assert_called_once()
        rend.assert_called_once()
        self.assertEqual(film["path"], "/x/film.mp4")
        self.assertEqual(board["auto_film"], "/x/film.mp4")
        sb_save.assert_called_once()
