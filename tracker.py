#!/usr/bin/env python3
"""Tracker: follows the plans picked out by explore.py on REAL option prices, day by day. Suggestions only: nothing is
pushed or traded.

Each weekday after the close, for every stock explore.py studies, it finds the day's trades for each tracked plan,
picks the contract Daybook would (this week's at-the-money weekly), buys at the first real 1-minute price at entry
time and sells by the plan's exit rules on real 1-minute prices. Next to each trade it keeps what the estimate
(explore.py's model) said, so you can see how far off the estimate is. Days already booked are never rewritten.

  python tracker.py          # books any days not yet booked; writes docs/tracking.json
"""
import json
import sys
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import advisor_config as A
import explore as X

sys.path.insert(0, str(A.DAYBOOK))

OUT = A.HERE / "docs" / "tracking.json"
BACKFILL_DAYS = 7          # Yahoo keeps 1-minute option prices for about a week
RETRY_DAYS = 3             # a day with missing option prices is tried again for this many days

# frozen when tracking started (2026-11-14), so the scoreboard is about the same rules every day
PLANS = [
    {"id": "1", "setup": "Gap and go", "filters": {"Volume": "Volume surge", "Weekly trend": "Against weekly trend"},
     "exit": {"target": 0.50, "stop": None, "cut": "14:50", "lock": None}},
    {"id": "2", "setup": "Gap and go", "filters": {"Daily trend": "With daily trend", "SPY": "SPY moving with it"},
     "exit": {"target": None, "stop": 0.50, "cut": "14:50", "lock": None}},
    {"id": "3", "setup": "Gap and go", "filters": {"Daily trend": "No daily trend", "SPY": "SPY moving against"},
     "exit": {"target": None, "stop": None, "cut": "12:30", "lock": None}},
]


def real_exit(bars, at, ex):
    """Buy at the first 1-minute open from `at`; sell by the plan's rules on real 1-minute prices.
    A target and stop in the same minute counts as the stop. Returns a dict or None (no prices)."""
    b = bars[(bars.index >= at) & (bars.index.strftime("%H:%M") < ex["cut"])]
    if b.empty or b["Open"].iloc[0] <= 0:
        return None
    entry = float(b["Open"].iloc[0])
    qty = max(int(A.TRADE_DOLLARS // (entry * 100)), 1)

    def leg(target):
        for ts, r in b.iterrows():
            if ex["stop"] is not None and r["Low"] <= entry * (1 - ex["stop"]):
                px = min(float(r["Open"]), entry * (1 - ex["stop"]))
                return ts, px, f"Stop −{round(ex['stop'] * 100)}%"
            if target is not None and r["High"] >= entry * (1 + target):
                return ts, max(float(r["Open"]), entry * (1 + target)), f"Target +{round(target * 100)}%"
        ts = b.index[-1]
        return ts, float(b["Close"].iloc[-1]), "Held to 2:50" if ex["cut"] == "14:50" else f"Out by {X_hm(pd.Timestamp('2000-01-01 ' + ex['cut']))}"

    legs = [leg(ex["target"])] + ([leg(ex["target2"])] if "target2" in ex else [])
    px = sum(l[1] for l in legs) / len(legs)
    return {"entry": round(entry, 2), "entry_time": X_hm(b.index[0]), "qty": qty,
            "exit": round(px, 2), "exit_time": X_hm(legs[-1][0]), "why": legs[0][2],
            "high": round(float(b["High"].max()), 2), "low": round(float(b["Low"].min()), 2),
            "cost": round(entry * qty * 100), "ret": round(px / entry - 1, 4), "pnl": round((px - entry) * qty * 100)}


BARS = A.HERE / "docs" / "bars"       # the real 1-minute prices behind each trade, for the page's charts


def save_bars(day, sym, h):
    f = BARS / f"{day.isoformat()}.json"
    try:
        saved = json.loads(f.read_text())
    except FileNotFoundError:
        saved = {}
    if sym in saved:
        return
    saved[sym] = {"t": list(h.index.strftime("%H:%M")), **{k: [round(float(v), 2) for v in h[c]] for k, c in
                  (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"))}, "v": [int(v) for v in h["Volume"]]}
    BARS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(saved, separators=(",", ":")))


def backfill_bars(rows):
    """Booked trades whose price chart isn't saved yet: fetch while Yahoo still has the contract."""
    import scanner as S
    import yfinance as yf
    for r in rows:
        if "ret" not in r or not r.get("symbol"):
            continue
        f = BARS / f"{r['date']}.json"
        if f.exists() and r["symbol"] in json.loads(f.read_text()):
            continue
        d = date.fromisoformat(r["date"])
        try:
            h = yf.Ticker(r["symbol"]).history(start=d, end=d + timedelta(days=1), interval="1m")
        except Exception:
            continue
        if not h.empty:
            h.index = h.index.tz_convert(S.CT)
            save_bars(d, r["symbol"], h)


def X_hm(ts):
    return ts.strftime("%-I:%M")


def day_trades(day, data, spy, vixc):
    """Every tracked plan's trades on `day`, scored on real option prices."""
    import scanner as S
    import strategy_15m as M
    b5, daily = data
    out = []
    for t in sorted(b5):
        if t not in daily:
            continue
        x, x15, ctx = X.day_frame(b5[t]), X.fifteen(b5[t]), X.daily_context(daily[t])
        if day not in ctx.index or not (x.index.date == day).any():
            continue
        c = ctx.loc[day]
        if any(pd.isna(c[k]) for k in ("rv", "prev_close", "atr")):
            continue
        opens = {d.date(): float(o) for d, o in daily[t]["Open"].items()}
        if day not in opens:
            continue
        gap = opens[day] / float(c["prev_close"]) - 1
        v = next((vixc[d] for d in sorted(vixc, reverse=True) if d < day), 19.0)
        for setup, call, at in X.entries(t, x, x15, day, gap):
            prev_bar = at - timedelta(minutes=5)
            if prev_bar not in x.index:
                continue
            row = x.loc[prev_bar]
            done15 = x15[x15.index + timedelta(minutes=15) <= at]
            sp = spy.loc[prev_bar] if spy is not None and prev_bar in spy.index else None
            tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, c, sp, gap, v, at)
            plans = [p for p in PLANS if p["setup"] == setup and all(tags.get(g) == l for g, l in p["filters"].items())]
            if not plans:
                continue
            spot = float(row["Close"])
            sym, vol = M.pick_contract(t, call, day, spot)
            bars = None
            if sym:
                try:
                    import yfinance as yf
                    h = yf.Ticker(sym).history(start=day, end=day + timedelta(days=1), interval="1m")
                    if not h.empty:
                        h.index = h.index.tz_convert(S.CT)
                        bars = h
                        save_bars(day, sym, h)
                except Exception:
                    bars = None
            est_path = X.path(call, spot, max(float(c["rv"]) * 1.1, 0.25), S.weekly_dte(day), at,
                              x[(x.index >= at) & (x.index.date == day)])
            for p in plans:
                real = real_exit(bars, at, p["exit"]) if bars is not None else None
                est = None
                if est_path is not None:
                    est = round(float(X.outcome(est_path[0], est_path[1], est_path[2], [p["exit"]])[0]), 4)
                out.append({"plan": p["id"], "date": day.isoformat(), "ticker": t, "dir": "CALLS" if call else "PUTS",
                            "gap": round(gap * 100, 2), "stock": round(spot, 2), "symbol": sym, "option_volume": vol,
                            "signal": X_hm(at), **(real or {"no_data": True}), "est_ret": est})
    return out


def summary(rows):
    """Per plan, on real prices and on the estimate for the same trades. Results are per $1,000 traded, so a
    contract that costs more than $1,000 doesn't count bigger."""
    s = {}
    for p in PLANS:
        r = [x for x in rows if x["plan"] == p["id"] and "ret" in x]
        real = np.array([x["ret"] for x in r], float) * A.TRADE_DOLLARS
        est = np.array([x["est_ret"] for x in r if x.get("est_ret") is not None], float) * A.TRADE_DOLLARS
        s[p["id"]] = {"n": len(r), "days": len({x["date"] for x in r}),
                      "win": round(float((real > 0).mean()), 3) if len(r) else None,
                      "avg": round(float(real.mean()), 1) if len(r) else None,
                      "total": round(float(real.sum())) if len(r) else 0,
                      "est_avg": round(float(est.mean()), 1) if len(est) else None,
                      "missing": sum(1 for x in rows if x["plan"] == p["id"] and x.get("no_data"))}
    return s


def trading_days(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main():
    import scanner as S
    import strategy_15m as M
    try:
        book = json.loads(OUT.read_text())
    except FileNotFoundError:
        book = {"started": "2026-11-14", "rows": [], "days": {}}
    now = datetime.now(S.CT)
    last = now.date() if now.strftime("%H:%M") >= "15:15" else now.date() - timedelta(days=1)
    todo = []
    for d in trading_days(last - timedelta(days=BACKFILL_DAYS + 3), last):
        st = book["days"].get(d.isoformat())
        if st == "booked":
            continue
        if st and st.startswith("retry") and (last - d).days > RETRY_DAYS:
            continue
        todo.append(d)
    tickers, skip = X.universe()
    if todo:
        b5 = M.candles(tickers + ["SPY"], "60d", "5m")
        daily = M.candles(tickers, "2y", "1d")
        vix = M.candles(["^VIX"], "1y", "1d").get("^VIX")
        vixc = {d.date(): float(v) for d, v in vix["Close"].items()} if vix is not None else {}
        spy = X.day_frame(b5["SPY"]) if "SPY" in b5 else None
        for d in todo:
            if not any((df.index.date == d).any() for df in b5.values()):
                continue                                         # market closed
            rows = [r for r in day_trades(d, (b5, daily), spy, vixc) if r["ticker"] not in skip]
            book["rows"] = [r for r in book["rows"] if r["date"] != d.isoformat()] + rows
            book["days"][d.isoformat()] = "retry" if any(r.get("no_data") for r in rows) else "booked"
    book["rows"].sort(key=lambda r: (r["date"], r["signal"], r["ticker"], r["plan"]))
    book["plans"] = PLANS
    backfill_bars(book["rows"])
    book["summary"] = summary(book["rows"])
    book["updated"] = datetime.now().astimezone().isoformat(timespec="minutes")
    OUT.write_text(json.dumps(book, indent=1, default=str))
    print("tracker:", {k: (v["n"], v["avg"], v["est_avg"]) for k, v in book["summary"].items()},
          "days:", {k: v for k, v in sorted(book["days"].items())[-8:]})


if __name__ == "__main__":
    main()
