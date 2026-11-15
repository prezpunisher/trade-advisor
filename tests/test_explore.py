import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

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


class RealExit(unittest.TestCase):
    def bars(self, rows, start="2026-11-09 08:45"):
        import pandas as pd
        idx = pd.date_range(start, periods=len(rows), freq="1min", tz="America/Chicago")
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)

    def test_target_stop_and_hold(self):
        import tracker as T
        b = self.bars([(2.0, 2.1, 1.9, 2.0), (2.0, 3.2, 2.0, 3.1), (3.1, 3.1, 0.5, 0.6)])
        at = b.index[0]
        win = T.real_exit(b, at, {"target": 0.50, "stop": None, "cut": "14:50", "lock": None})
        self.assertEqual((win["entry"], win["exit"], win["why"]), (2.0, 3.0, "Target +50%"))
        held = T.real_exit(b, at, {"target": None, "stop": None, "cut": "14:50", "lock": None})
        self.assertEqual((held["exit"], held["why"]), (0.6, "Held to 2:50"))
        stop = T.real_exit(b, at, {"target": None, "stop": 0.50, "cut": "14:50", "lock": None})
        self.assertEqual(stop["exit"], 1.0)

    def test_same_minute_counts_as_the_stop(self):
        import tracker as T
        b = self.bars([(2.0, 2.0, 2.0, 2.0), (2.0, 3.5, 0.9, 2.0)])
        r = T.real_exit(b, b.index[0], {"target": 0.50, "stop": 0.50, "cut": "14:50", "lock": None})
        self.assertTrue(r["why"].startswith("Stop"))

    def test_results_count_per_1000(self):
        import tracker as T
        rows = [{"plan": "1", "date": "2026-11-09", "ticker": "AAA", "dir": "CALLS", "ret": 0.5, "est_ret": 0.4, "pnl": 1500}]
        self.assertEqual(T.summary(rows)["1"]["avg"], 500.0)

    def test_no_trade_stocks_are_left_out_of_results(self):
        import tracker as T
        rows = [{"plan": "1", "date": "2026-11-09", "ticker": t, "dir": "CALLS", "ret": r, "est_ret": None}
                for t, r in (("AAA", 0.5), ("BBB", -0.3))]
        s = T.summary(rows, {"BBB"})["1"]
        self.assertEqual((s["n"], s["avg"]), (1, 500.0))


class BiggestTrades(unittest.TestCase):
    def test_biggest_win_and_loss_name_the_trade(self):
        who = [{"ticker": t, "date": "2026-11-0%d" % i, "time": "08:45", "setup": "Gap and go", "tags": {"Direction": "Calls"}}
               for i, t in enumerate(("AAA", "BBB", "CCC"), 2)]
        s = X.steady([w["date"] for w in who], [50, -400, 900], X.obj(who))
        self.assertEqual((s["max_win_trade"]["ticker"], s["max_loss_trade"]["ticker"]), ("CCC", "BBB"))
        self.assertEqual((s["max_win"], s["max_loss"]), (900, -400))


class HistoryContract(unittest.TestCase):
    def test_nearest_strike_that_traded_on_the_weekly(self):
        import history as H
        day, exp = date(2025, 3, 10), date(2025, 3, 14)
        listed = [{"ticker": f"O:AAA250314C{k * 1000:08d}", "contract_type": "call", "expiration_date": "2025-03-14", "strike_price": k}
                  for k in (98, 100, 102)] + \
                 [{"ticker": "O:AAA250313C00100000", "contract_type": "call", "expiration_date": "2025-03-13", "strike_price": 100},
                  {"ticker": "O:AAA250314P00100000", "contract_type": "put", "expiration_date": "2025-03-14", "strike_price": 100}]
        bars = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [5]},
                            index=pd.DatetimeIndex([pd.Timestamp("2025-03-10 08:45", tz="America/Chicago")]))
        asked = []
        old = H.contracts, H.option_bars
        H.contracts = lambda *a, **k: listed
        H.option_bars = lambda sym, d, f=True: asked.append(sym) or (bars.iloc[:0] if sym.endswith("00100000") else bars)
        try:
            s = {"ticker": "AAA", "exp": exp, "day": day, "call": True, "spot": 100.4}
            sym, b = H.priced(s, {("AAA", exp): (100.4, 100.4)})
        finally:
            H.contracts, H.option_bars = old
        self.assertEqual(asked[0], "AAA250314C00100000")      # the Friday weekly, nearest strike first
        self.assertEqual(sym, "AAA250314C00102000")           # 100 didn't trade that day: next nearest


class SmallAccount(unittest.TestCase):
    def test_one_morning_pick_and_the_first_later_alert(self):
        import tracker as T
        row = lambda plan, t, sig, entry, ret: {"plan": plan, "date": "2026-11-16", "ticker": t, "dir": "CALLS",
                                                "signal": sig, "entry": entry, "ret": ret}
        rows = [row("4", "AAA", "8:45", 3.0, 0.1), row("4", "BBB", "8:45", 1.0, 0.2), row("4", "CCC", "8:45", 5.0, -0.3),
                row("2v", "CCC", "8:45", 5.0, -0.3),                      # #2v's trade wins over the cheaper one
                row("L", "CCC", "9:40", 2.0, 0.5),                        # same stock as the 8:45 pick: skipped
                row("L", "DDD", "1:05", 2.0, 0.4), row("L", "EEE", "10:15", 2.0, -0.1)]
        s = T.small_account(rows)
        self.assertEqual([(p["slot"], p["ticker"]) for p in s["picks"][::-1]], [("8:45", "CCC"), ("later", "EEE")])
