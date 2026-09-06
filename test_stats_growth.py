"""The stats page answers "how many renders, and are we growing".

Total renders and total installs are ALL-TIME tiles, not windows; the growth
block carries cumulative installs by first-boot day, renders per day (30d)
and weekly active installs, in one shape for the fleet path and the local
path so the page has one renderer.
"""
import json, time, unittest
from unittest import mock

import mlx_ltx_panel as p


def _rec(ev, ts, **props):
    return {"event": ev, "ts": ts, "props": props}


class LocalGrowth(unittest.TestCase):
    def test_totals_and_series_from_the_local_log(self):
        now = time.time()
        recs = [
            _rec("app_boot", now - 40 * 86400, version="4.9.0"),
            _rec("render_completed", now - 40 * 86400, engine="ltx"),      # all-time, outside 30d
            _rec("render_completed", now - 2 * 86400, engine="ltx"),
            _rec("render_completed", now - 2 * 86400, engine="h3"),
            _rec("render_failed", now - 1 * 86400, engine="ltx", error_signature="x"),
            _rec("app_boot", now - 1 * 86400, version="4.10.5"),
        ]
        with mock.patch.object(p, "_usage_log_read", lambda: recs):
            u = p._usage_local_report()
        self.assertEqual(u["tiles"]["total_renders"], 3)          # completed only, all time
        self.assertEqual(u["tiles"]["total_installs"], 1)
        g = u["growth"]
        self.assertEqual(len(g["installs_by_day"]), 1)
        self.assertEqual(g["installs_by_day"][0]["cumulative"], 1)
        self.assertEqual(sum(r["count"] for r in g["renders_by_day"]), 2)   # 30-day window
        self.assertEqual(g["active_by_week"], [])

    def test_growth_block_accumulates_installs(self):
        g = p._usage_growth_block([("2026-08-01", 3), ("2026-08-02", 2)],
                                  [("2026-08-02", 7)], [("2026-07-27", 40)])
        self.assertEqual([r["cumulative"] for r in g["installs_by_day"]], [3, 5])
        self.assertEqual(g["renders_by_day"], [{"date": "2026-08-02", "count": 7}])
        self.assertEqual(g["active_by_week"], [{"week": "2026-07-27", "installs": 40}])


class FleetQueriesAndPage(unittest.TestCase):
    def test_fleet_queries_have_no_window_on_the_totals(self):
        q = p._USAGE_FLEET_QUERIES
        for name in ("total_renders", "total_installs", "installs_by_day", "renders_by_day", "active_by_week"):
            self.assertIn(name, q)
        self.assertNotIn("INTERVAL", q["total_renders"])
        self.assertNotIn("INTERVAL", q["total_installs"])
        self.assertNotIn("INTERVAL", q["installs_by_day"])
        self.assertIn("render_completed", q["total_renders"])
        # the running week is excluded, or the caption reads as a collapse
        self.assertIn("toStartOfWeek(timestamp) < toStartOfWeek(now())", q["active_by_week"])

    def test_page_has_the_tiles_and_the_chart(self):
        html = p.STATS_HTML_FILE.read_text(encoding="utf-8")
        self.assertIn("Total renders", html)
        self.assertIn("t.total_installs", html)
        self.assertIn('id="chart-usage-growth"', html)
        self.assertIn("function renderUsageGrowth", html)
        self.assertIn("renderUsageGrowth(u.growth", html)


if __name__ == "__main__":
    unittest.main()
