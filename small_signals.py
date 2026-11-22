#!/usr/bin/env python3
"""Today's 8:45 CT trade candidates for the small Robinhood account, in priority order. Read-only: places nothing.

Uses the same code that produced the cheap-contract test (trade-advisor's explore.py / tracker.py rules and Daybook's
indicators), on Yahoo 5-minute candles, so live signals match the test.

  #2v  Gap and go + with daily trend + SPY moving with it + VIX 22 or under  -> hold to 2:50, stop -30%
  #5   Gap and go + wrong side of 15m EMA50 + SPY moving against            -> hold to 2:50, stop -30%
  #A3  Today's Daybook watchlist pick with all three 15m checks agreeing    -> +10% target, -20% stop, out 2:50

  DAYBOOK_DIR=... python small_signals.py                              # prints JSON
  ... python small_signals.py --day 2026-11-20                         # replay a past day (as of 8:46 CT)
"""
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAYBOOK = Path(os.environ.get("DAYBOOK_DIR", HERE.parent / "options-scanner")).resolve()
ADVISOR = HERE
os.environ["DAYBOOK_DIR"] = str(DAYBOOK)
sys.path[:0] = [str(HERE), str(DAYBOOK)]

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import config as C
import explore as X
import scanner as S
import strategy_15m as M
import tracker as T

TICKERS = ["AAPL", "AMD", "AVGO", "CRWV", "HOOD", "INTC", "IWM", "JPM", "META", "MRVL", "MSFT", "MSTR", "NBIS", "NFLX",
           "NOW", "ORCL", "PLTR", "QQQ", "SPY", "TSM", "WMT"]          # the stocks the test used (history.py)
EXITS = {"2v": {"target": None, "stop": 0.30}, "5": {"target": None, "stop": 0.30}, "A3": {"target": 0.10, "stop": 0.20}}
ORDER = ["2v", "5", "A3"]


def gap_signals(day, cutoff, skip):
    tickers = [t for t in TICKERS if t not in skip and t not in C.EXCLUDE]
    b5 = M.candles(tickers + ["SPY"], "60d", "5m")
    daily = M.candles(tickers, "2y", "1d")
    vix = M.candles(["^VIX"], "1y", "1d").get("^VIX")
    vixc = {d.date(): float(v) for d, v in vix["Close"].items()} if vix is not None else {}
    b5 = {t: df[df.index < cutoff] for t, df in b5.items()}                 # only what was known at the cutoff
    daily = {t: df[df.index.date <= day] for t, df in daily.items()}
    spy = X.day_frame(b5["SPY"]) if "SPY" in b5 else None
    out = []
    for t in tickers:
        if t not in b5 or t not in daily:
            continue
        x, x15, ctx = X.day_frame(b5[t]), X.fifteen(b5[t]), X.daily_context(daily[t])
        if day not in ctx.index or not (x.index.date == day).any():
            continue
        c = ctx.loc[day]
        if any(pd.isna(c[k]) for k in ("prev_close", "atr")):
            continue
        opens = {d.date(): float(o) for d, o in daily[t]["Open"].items()}
        if day not in opens:
            continue
        gap = opens[day] / float(c["prev_close"]) - 1
        if abs(gap) > 0.20:
            continue
        v = next((vixc[d] for d in sorted(vixc, reverse=True) if d < day), 19.0)
        # explore.entries' gap-and-go rule (it needs a full hour of candles for its other setups, so it's inlined):
        # gap of 0.4%+ and the first 15 minutes (8:30-8:45) moved the same way; entry at 8:45
        xd = x[x.index.date == day]
        if len(xd) < 3 or abs(gap) < 0.004:
            continue
        went_up = xd["Close"].iloc[2] > xd["Open"].iloc[0]
        if (gap > 0) != went_up:
            continue
        setup, call, at = "Gap and go", bool(went_up), xd.index[2] + timedelta(minutes=5)
        for _ in [0]:
            prev_bar = at - timedelta(minutes=5)
            if prev_bar not in x.index:
                continue
            row = x.loc[prev_bar]
            done15 = x15[x15.index + timedelta(minutes=15) <= at]
            sp = spy.loc[prev_bar] if spy is not None and prev_bar in spy.index else None
            tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, c, sp, gap, v, at)
            for p in T.PLANS:
                if p["id"] in ("2v", "5") and T.matches(p, setup, tags, at):
                    out.append({"plan": p["id"], "ticker": t, "direction": "CALLS" if call else "PUTS",
                                "stock": round(float(row["Close"]), 2), "gap_pct": round(gap * 100, 2),
                                "vix_prior": round(v, 1), "signal_time": at.strftime("%H:%M")})
    return out


def all3_signals(day, cutoff, skip):
    status = json.loads((DAYBOOK / "docs" / "status.json").read_text())
    runs = sorted((r for r in status["runs"] if r.get("kind") == "watchlist" and r["time"][:10] == day.isoformat()),
                  key=lambda r: r["time"])
    picks = runs[-1].get("picks") or [] if runs else []
    picks = [p for p in picks if p["ticker"] not in skip and p["ticker"] not in C.EXCLUDE]
    if not picks:
        return [], bool(runs)
    b5 = M.candles([p["ticker"] for p in picks], "60d", "5m")
    out = []
    for p in picks:
        df = b5.get(p["ticker"])
        if df is None:
            continue
        x15 = X.fifteen(df[df.index < cutoff])
        done = x15[x15.index + timedelta(minutes=15) <= cutoff]
        if done.empty:
            continue
        a, sgn = done.iloc[-1], (1 if p["direction"] == "CALLS" else -1)
        ok = (np.sign(a["close"] - a["ema"]) == sgn and np.sign(a["macd"] - a["sig"]) == sgn and np.sign(a["cci"]) == sgn)
        if ok:
            out.append({"plan": "A3", "ticker": p["ticker"], "direction": p["direction"], "stock": p.get("price"),
                        "signal_time": cutoff.strftime("%H:%M")})
    return out, True


def candidates(day, live=False):
    """The day's candidates as of 8:46 CT (live: as of now, if that's later)."""
    cutoff = pd.Timestamp(datetime(day.year, day.month, day.day, 8, 46), tz=S.CT)
    if live:
        cutoff = max(cutoff, pd.Timestamp.now(tz=S.CT))
    try:
        skip = set(json.loads((DAYBOOK / "docs" / "no_trade.json").read_text()).get("tickers", []))
    except FileNotFoundError:
        skip = set()
    gaps = gap_signals(day, cutoff, skip)
    a3, watchlist_found = all3_signals(day, cutoff, skip)
    cands = sorted(gaps + a3, key=lambda c: ORDER.index(c["plan"]))
    seen, uniq = set(), []
    for c in cands:                                       # one row per ticker+direction (highest-priority plan)
        k = (c["ticker"], c["direction"])
        if k not in seen:
            seen.add(k)
            uniq.append({**c, "exit": EXITS[c["plan"]]})
    exp = day + timedelta(days=S.weekly_dte(day))
    return {"day": day.isoformat(), "as_of": cutoff.strftime("%H:%M CT"), "weekly_expiry": exp.isoformat(),
            "daybook_watchlist_found": watchlist_found, "candidates": uniq}


def main():
    args = sys.argv[1:]
    if "--day" in args:
        print(json.dumps(candidates(date.fromisoformat(args[args.index("--day") + 1])), indent=1))
    else:
        print(json.dumps(candidates(datetime.now(S.CT).date(), live=True), indent=1))


if __name__ == "__main__":
    main()
