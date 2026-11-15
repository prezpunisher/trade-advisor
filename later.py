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
import pandas as pd

import advisor_config as A
import explore as X
import history as H
import tracker as T

sys.path.insert(0, str(A.DAYBOOK))

OUT = A.HERE / "docs" / "later.json"
LATER_FROM = "09:30"
SETUPS = ["15m EMA+MACD+CCI", "Opening range breakout", "VWAP cross", "5m EMA 9/21 cross", "RSI bounce", "First-hour trend"]
WITH_TREND = " · with the daily trend"      # the same setups, taking the day's first alert that goes with the daily trend
HOLD = "Hold to 2:50 pm · Stop −30%"
LATER_PICK = "VWAP cross" + WITH_TREND       # the small account's later trade (best on 2 years of real prices, both years)
KEEP = ("date", "ticker", "dir", "symbol", "signal", "entry", "entry_time", "exit", "exit_time", "why", "ret")
EXITS = {"Hold to 2:50 pm · Stop −30%": {"target": None, "stop": 0.30, "cut": "14:50", "lock": None},
         "Sell +50% or 2:50 pm · Stop −30%": {"target": 0.50, "stop": 0.30, "cut": "14:50", "lock": None}}


def first_alerts(data):
    """setup -> one alert a day: the earliest at or after LATER_FROM across the stocks (ties: by ticker)."""
    import scanner as S
    spy = X.day_frame(data["SPY"][0])
    vix = H.prior_vix()
    cand = {s + w: {} for s in SETUPS for w in ("", WITH_TREND)}
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
                if setup not in SETUPS or at.strftime("%H:%M") < LATER_FROM:
                    continue
                prev_bar = at - timedelta(minutes=5)
                if prev_bar not in x.index:
                    continue
                row = x.loc[prev_bar]
                sp = spy.loc[prev_bar] if prev_bar in spy.index else None
                done15 = x15[x15.index + timedelta(minutes=15) <= at]
                tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, ctx.loc[day], sp, gap,
                              vix.get(day, 19.0), at)
                for key in (setup, setup + WITH_TREND):
                    if key.endswith(WITH_TREND) and tags["Daily trend"] != "With daily trend":
                        continue
                    best = cand[key].get(day)
                    if best is None or (at, t) < (best["at"], best["ticker"]):
                        cand[key][day] = {"ticker": t, "day": day, "call": call, "at": at, "spot": float(row["Close"]),
                                          "gap": gap, "exp": day + timedelta(days=S.weekly_dte(day)), "tags": tags}
    return {s: sorted(v.values(), key=lambda a: a["day"]) for s, v in cand.items()}


def small_account(later_rows):
    """The plan for a small account: one gap-and-go at 8:45 (plan #2v's if there is one, else the cheapest option) and
    the day's first LATER_PICK alert, $1,000 each, both held to 2:50 with a -30% stop. Never the same stock twice."""
    h = json.loads(H.OUT.read_text())
    real = [r for r in h["rows"] if "ret" in r]
    v2 = {(r["date"], r["ticker"], r["dir"]) for r in real if r["plan"] == "2v"}
    morning = {}
    for r in real:
        if r["plan"] == "4":
            morning.setdefault(r["date"], []).append(r)
    late = {r["date"]: r for r in later_rows}
    trades = []
    for d in sorted(set(morning) | set(late)):
        first = None
        if d in morning:
            first = min(morning[d], key=lambda r: ((r["date"], r["ticker"], r["dir"]) not in v2, r["entry"]))
            trades.append({"slot": "8:45", "picked": "plan #2v" if (d, first["ticker"], first["dir"]) in v2 else "cheapest option",
                           **{k: first[k] for k in KEEP}})
        if d in late and not (first and late[d]["ticker"] == first["ticker"]):
            trades.append({"slot": "later", "picked": "first VWAP cross with the trend", **{k: late[d][k] for k in KEEP}})
    pnl = np.array([t["ret"] for t in trades]) * A.TRADE_DOLLARS
    dates = [t["date"] for t in trades]
    per_day = pd.Series(pnl).groupby(np.array(dates)).sum()
    months = per_day.groupby(pd.to_datetime(per_day.index).to_period("M")).sum()
    part = lambda slot: X.steady([t["date"] for t in trades if t["slot"] == slot],
                                 np.array([t["ret"] for t in trades if t["slot"] == slot]) * A.TRADE_DOLLARS)
    who = lambda t: {k: t[k] for k in ("ticker", "dir", "date", "slot")}
    return {"all": {**X.steady(dates, pnl), "max_win_trade": who(trades[int(pnl.argmax())]),
                    "max_loss_trade": who(trades[int(pnl.argmin())])},
            "morning": part("8:45"), "later": part("later"),
            "days": int(len(per_day)), "worst_day": round(float(per_day.min())), "best_day": round(float(per_day.max())),
            "per_day": round(float(per_day.mean()), 1),
            "months": [{"m": str(k), "total": round(float(v))} for k, v in months.items()],
            "months_up": int((months > 0).sum()),
            "by_year": {y: round(float(pnl[[d.startswith(y) for d in dates]].mean()), 1) for y in ("2025", "2026")},
            "no_trade": h.get("no_trade", []), "trades": trades[::-1]}


def main():
    data = H.load_all(False)
    alerts = first_alerts(data)
    out = {"from": LATER_FROM, "setups": []}
    picked = []
    for setup, al in alerts.items():
        rng = H.week_ranges(al)
        rows, keep = [], {}
        for a in al:
            sym, bars = H.priced(a, rng, True)
            if bars is None:
                continue
            if setup == LATER_PICK:
                keep.setdefault(a["day"].isoformat(), {})[sym] = bars
            for name, ex in EXITS.items():
                real = T.real_exit(bars, a["at"], ex)
                if real:
                    rows.append({"rule": name, "date": a["day"].isoformat(), "ticker": a["ticker"],
                                 "dir": "CALLS" if a["call"] else "PUTS", "signal": T.X_hm(a["at"]), "symbol": sym,
                                 "vix_high": a["tags"]["VIX"] == "VIX over 22",
                                 "daily_with": a["tags"]["Daily trend"] == "With daily trend",
                                 **{k: real[k] for k in ("entry", "entry_time", "exit", "exit_time", "why", "ret")}})
        res = {}
        for name in EXITS:
            r = [x for x in rows if x["rule"] == name]
            pnl = np.array([x["ret"] for x in r]) * A.TRADE_DOLLARS
            res[name] = {**X.steady([x["date"] for x in r], pnl),
                         "by_year": {y: round(float(np.mean([x["ret"] for x in r if x["date"].startswith(y)])) * A.TRADE_DOLLARS, 1)
                                     for y in ("2025", "2026") if any(x["date"].startswith(y) for x in r)},
                         "with_daily_trend": {**{k: v for k, v in X.steady([x["date"] for x in r if x["daily_with"]],
                                                                          np.array([x["ret"] for x in r if x["daily_with"]]) * A.TRADE_DOLLARS).items()
                                                 if k in ("n", "avg", "win", "weeks", "weeks_up", "avg_without_best_day")},
                                              "by_year": {y: round(float(np.mean([x["ret"] for x in r if x["daily_with"] and x["date"].startswith(y)] or [np.nan])) * A.TRADE_DOLLARS, 1)
                                                          for y in ("2025", "2026")}},
                         "median_time": sorted(x["signal"] for x in r)[len(r) // 2] if r else None}
        out["setups"].append({"setup": setup, "days": len(al), "results": res})
        if setup == LATER_PICK:
            picked = [x for x in rows if x["rule"] == HOLD]
            H.save_hbars(keep)                      # charts for the page
        print(setup, len(al), {k: (v["n"], v.get("avg"), v.get("win"), v.get("by_year"), v.get("with_daily_trend")) for k, v in res.items() if k.startswith("Hold")}, flush=True)
    out["small"] = small_account(picked)
    s = out["small"]
    print("small account:", s["all"]["n"], s["all"]["avg"], s["all"]["win"], "per day", s["per_day"], "worst day", s["worst_day"],
          "months up", s["months_up"], len(s["months"]), s["by_year"], "8:45", s["morning"]["avg"], "later", s["later"]["avg"],
          "without best day", s["all"]["avg_without_best_day"])
    OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
