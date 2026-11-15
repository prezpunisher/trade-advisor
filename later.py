#!/usr/bin/env python3
"""The later trade: if you hold one gap-and-go from 8:45, which alert later in the day is worth a second $1,000?

For each setup explore.py knows (other than the 8:45 gap trades), on every day since history.TEST_FROM it takes the
FIRST alert across the stocks at or after LATER_FROM (one trade a day, so it fits a small account), buys Daybook's
contract at the first real 1-minute price and holds to 2:50 with a -30% stop, the same exit as tracked plan #4.
Real option prices from Massive, cached by history.py. Suggestions only.

  python later.py            # writes docs/later.json
"""
import json
import sys
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np

import advisor_config as A
import explore as X
import history as H
import tracker as T

sys.path.insert(0, str(A.DAYBOOK))

OUT = A.HERE / "docs" / "later.json"
LATER_FROM = "09:30"
SETUPS = ["15m EMA+MACD+CCI", "Opening range breakout", "VWAP cross", "5m EMA 9/21 cross", "RSI bounce", "First-hour trend"]
EXITS = {"Hold to 2:50 pm · Stop −30%": {"target": None, "stop": 0.30, "cut": "14:50", "lock": None},
         "Sell +50% or 2:50 pm · Stop −30%": {"target": 0.50, "stop": 0.30, "cut": "14:50", "lock": None}}


def first_alerts(data):
    """setup -> one alert a day: the earliest at or after LATER_FROM across the stocks (ties: by ticker)."""
    import scanner as S
    spy = X.day_frame(data["SPY"][0])
    vix = H.prior_vix()
    cand = {s: {} for s in SETUPS}
    skip = H.no_trade()
    for t in H.TICKERS:
        if t in skip:
            continue
        b5, d_adj, d_raw = data[t]
        x, x15, ctx = X.day_frame(b5), X.fifteen(b5), X.daily_context(d_adj)
        raw_open = {d.date(): float(v) for d, v in d_raw["Open"].items()}
        prev_close = dict(zip([d.date() for d in d_raw.index][1:], d_raw["Close"].values[:-1]))
        for day in sorted(set(x.index.date)):
            if not H.TEST_FROM <= day <= H.TEST_TO or day not in ctx.index or day not in raw_open or day not in prev_close:
                continue
            gap = raw_open[day] / float(prev_close[day]) - 1
            if abs(gap) > H.SPLIT_GAP:
                continue
            for setup, call, at in X.entries(t, x, x15, day, gap):
                if setup not in cand or at.strftime("%H:%M") < LATER_FROM:
                    continue
                best = cand[setup].get(day)
                if best is None or (at, t) < (best["at"], best["ticker"]):
                    prev_bar = at - timedelta(minutes=5)
                    if prev_bar not in x.index:
                        continue
                    row = x.loc[prev_bar]
                    sp = spy.loc[prev_bar] if prev_bar in spy.index else None
                    done15 = x15[x15.index + timedelta(minutes=15) <= at]
                    tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, ctx.loc[day], sp, gap,
                                  vix.get(day, 19.0), at)
                    cand[setup][day] = {"ticker": t, "day": day, "call": call, "at": at, "spot": float(row["Close"]),
                                        "gap": gap, "exp": day + timedelta(days=S.weekly_dte(day)), "tags": tags}
    return {s: sorted(v.values(), key=lambda a: a["day"]) for s, v in cand.items()}


def main():
    data = H.load_all(False)
    alerts = first_alerts(data)
    out = {"from": LATER_FROM, "setups": []}
    for setup, al in alerts.items():
        rng = H.week_ranges(al)
        rows = []
        for a in al:
            sym, bars = H.priced(a, rng, True)
            if bars is None:
                continue
            for name, ex in EXITS.items():
                real = T.real_exit(bars, a["at"], ex)
                if real:
                    rows.append({"exit": name, "date": a["day"].isoformat(), "ticker": a["ticker"],
                                 "dir": "CALLS" if a["call"] else "PUTS", "signal": T.X_hm(a["at"]), "symbol": sym,
                                 "vix_high": a["tags"]["VIX"] == "VIX over 22",
                                 "daily_with": a["tags"]["Daily trend"] == "With daily trend",
                                 **{k: real[k] for k in ("entry", "entry_time", "exit", "exit_time", "why", "ret")}})
        res = {}
        for name in EXITS:
            r = [x for x in rows if x["exit"] == name]
            pnl = np.array([x["ret"] for x in r]) * A.TRADE_DOLLARS
            res[name] = {**X.steady([x["date"] for x in r], pnl),
                         "by_year": {y: round(float(np.mean([x["ret"] for x in r if x["date"].startswith(y)])) * A.TRADE_DOLLARS, 1)
                                     for y in ("2025", "2026") if any(x["date"].startswith(y) for x in r)},
                         "with_daily_trend": round(float(np.mean([x["ret"] for x in r if x["daily_with"]] or [np.nan])) * A.TRADE_DOLLARS, 1),
                         "median_time": sorted(x["signal"] for x in r)[len(r) // 2] if r else None}
        out["setups"].append({"setup": setup, "days": len(al), "results": res})
        print(setup, len(al), {k: (v["n"], v.get("avg"), v.get("win"), v.get("by_year"), v.get("with_daily_trend"), v.get("median_time")) for k, v in res.items()}, flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
