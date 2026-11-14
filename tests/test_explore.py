import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import explore as X


def trade(*points, start=9 * 60):
    """Option returns at successive points (5 minutes apart), then the 2:50 price last."""
    r = np.array(points, float)
    mins = np.array([start + 5 * i for i in range(len(r))])
    return r, np.zeros(len(r), bool), mins


def ret(r, g, mins, **spec):
    spec = {"target": None, "stop": None, "cut": X.EXIT, "lock": None, **spec}
    return X.outcome(r, g, mins, [spec])[0]


class Exits(unittest.TestCase):
    def test_target_and_stop(self):
        t = trade(0.02, 0.12, -0.3, 0.0)
        self.assertEqual(ret(*t, target=0.10, stop=0.20), 0.10)
        self.assertEqual(ret(*t, target=0.15, stop=0.20), -0.20)
        self.assertEqual(ret(*t), 0.0)                                   # no target, no stop: the 2:50 price

    def test_stop_on_a_gap_takes_the_worse_price(self):
        r, g, mins = trade(0.0, -0.4, 0.1)
        g[1] = True
        self.assertAlmostEqual(ret(r, g, mins, stop=0.20), -0.4)

    def test_time_cut_sells_at_that_time(self):
        t = trade(0.01, 0.02, 0.04, 0.06, 0.5, start=10 * 60 + 50)      # 10:50, 10:55, 11:00, 11:05 ...
        self.assertAlmostEqual(ret(*t, cut="11:00"), 0.04)

    def test_lock_in_moves_the_stop_up(self):
        t = trade(0.05, 0.12, 0.03, -0.01, -0.3, 0.2)
        self.assertEqual(ret(*t, stop=0.20, lock=(0.10, 0.0)), 0.0)      # up 12%, then back to even: out flat
        self.assertEqual(ret(*t, stop=0.20), -0.20)

    def test_half_and_half(self):
        t = trade(0.12, 0.35, 0.0)
        specs = [{"target": 0.10, "target2": 0.30, "stop": 0.20, "cut": X.EXIT, "lock": None}]
        self.assertAlmostEqual(X.outcome(*t, specs)[0], 0.20)

    def test_every_spec_is_scored(self):
        specs = X.exit_specs()
        out = X.outcome(*trade(0.01, -0.05, 0.2, 0.1), specs)
        self.assertEqual(len(out), len(specs))
        self.assertTrue(np.isfinite(out).all())


class Steadiness(unittest.TestCase):
    def test_weeks_up_and_worst_drop(self):
        s = X.steady(["2026-11-02", "2026-11-03", "2026-11-09", "2026-11-10"], [100, -50, -80, 20])
        self.assertEqual((s["weeks"], s["weeks_up"]), (2, 1))
        self.assertEqual(s["worst_drop"], 130)


if __name__ == "__main__":
    unittest.main()
