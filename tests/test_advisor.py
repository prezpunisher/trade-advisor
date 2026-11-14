import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import advisor as V
import advisor_config as A


def path(*rets):
    """A price path: returns at each point, then the exit-time return last."""
    return np.array(rets, dtype=float), np.array([False] * len(rets))


class ExitRules(unittest.TestCase):
    def test_target_first(self):
        r, g = path(0.01, 0.04, -0.25, -0.30)
        self.assertEqual(V.exit_return(r, g, 0.03, 0.20), 0.03)

    def test_stop_first(self):
        r, g = path(-0.21, 0.10, 0.05)
        self.assertEqual(V.exit_return(r, g, 0.03, 0.20), -0.20)

    def test_gap_through_stop_takes_the_worse_price(self):
        r, g = path(0.0, -0.35, 0.10)
        g[1] = True
        self.assertAlmostEqual(V.exit_return(r, g, 0.03, 0.20), -0.35)

    def test_neither_hit_sells_at_exit_time(self):
        r, g = path(0.01, -0.05, 0.02)
        self.assertAlmostEqual(V.exit_return(r, g, 0.10, 0.20), 0.02)
        self.assertAlmostEqual(V.exit_return(r, g, None, None), 0.02)

    def test_exit_time_price_never_counts_as_a_hit(self):
        r, g = path(0.0, 0.50)                      # only the 2:50 price is above the target
        self.assertAlmostEqual(V.exit_return(r, g, 0.10, 0.20), 0.50)


def row(day, tags, pnl):
    return {"ticker": "X", "date": day, "tags": tags, "pnl": pnl}


class PlanSearch(unittest.TestCase):
    def setUp(self):
        days = [f"2026-10-{d:02d}" for d in range(1, 13)]
        self.rows, pnl = [], []
        for d in days:
            for i in range(10):
                good = i < 5
                self.rows.append(row(d, {"Direction": "Puts" if good else "Calls"}, None))
                pnl.append([40.0 if good else -40.0, 10.0])
        self.m = np.array(pnl)
        self.exits = [(0.10, 0.20), A.CURRENT]

    def test_finds_the_condition_that_works_and_checks_it_on_unseen_days(self):
        out = V.search(self.rows, self.m, self.exits, 10, 5)
        best = out["plans"][0]
        self.assertEqual([f["label"] for f in best["filters"]], ["Puts"])
        self.assertEqual((best["target"], best["stop"]), (0.10, 0.20))
        self.assertTrue(best["proven"])
        self.assertLess(out["tuned_days"][1], out["check_days"][0])     # tuning and checking never share a day

    def test_tiny_samples_are_not_picked(self):
        out = V.search(self.rows, self.m, self.exits, 10_000, 5)
        self.assertEqual(out["plans"], [])

    def test_a_plan_that_fails_on_new_days_is_not_proven(self):
        m = self.m.copy()
        late = np.array([r["date"] >= "2026-10-09" for r in self.rows])
        m[late, 0] = -m[late, 0]
        out = V.search(self.rows, m, self.exits, 10, 5)
        self.assertFalse(any(p["proven"] for p in out["plans"] if p["filters"] and p["filters"][0]["label"] == "Puts"))


class Report(unittest.TestCase):
    def test_same_plan_skips_conditions_the_other_evidence_lacks(self):
        plan = {"filters": [{"group": "15m check", "label": "All three 15m agree"}], "target": 0.10, "stop": 0.20}
        rows = [row("2026-10-01", {"Direction": "Puts"}, None)]
        self.assertIsNone(V.same_plan(plan, rows, np.zeros((1, 1)), [(0.10, 0.20)]))

    def test_verdicts(self):
        good, bad, few = {"n": 20, "avg": 30}, {"n": 20, "avg": -40}, {"n": 2, "avg": 99}
        self.assertEqual(V.verdict(good, {"n": 0}), "Trade")
        self.assertEqual(V.verdict(good, {"n": 6, "avg": -20}), "Watch")    # real prices disagree
        self.assertEqual(V.verdict(bad, {"n": 0}), "Skip")
        self.assertEqual(V.verdict(few, {"n": 0}), "Watch")

    def test_no_trade_list_is_left_out(self):
        research = {"watchlist_check": [
            {"ticker": t, "date": "2026-11-0%d" % d, "dir": "P", "alert": "8:50", "all3_ok": True,
             "exits": {"3": {"pnl": 20}, "5": {"pnl": 30}, "10": {"pnl": 60}}}
            for t in ("AAA", "NVDA") for d in range(2, 9)], "shadow": []}
        out = V.build([], research, {"NVDA"}, set())
        self.assertEqual({c["ticker"] for c in out["tickers"]}, {"AAA"})
        self.assertEqual(out["real"]["picks"], 7)
        json.dumps(out, default=str)


class ReadOnly(unittest.TestCase):
    def test_never_writes_into_daybook(self):
        src = (Path(__file__).resolve().parents[1] / "advisor.py").read_text()
        self.assertNotIn("write_text(", src.replace("A.OUT.write_text(", ""))
        self.assertNotIn("ntfy", src.lower())


if __name__ == "__main__":
    unittest.main()
