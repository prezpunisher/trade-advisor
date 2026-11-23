#!/usr/bin/env python3
"""Today's 8:45 CT trade candidates for the small Robinhood account, in priority order. Read-only: places nothing.

Uses the same code that produced the cheap-contract test (trade-advisor's explore.py / tracker.py rules and Daybook's
indicators), on Yahoo 5-minute candles, so live signals match the test.

  #2v  Gap and go + with daily trend + SPY moving with it + VIX 22 or under  -> hold to 2:50, stop -30%
  #5   Gap and go + wrong side of 15m EMA50 + SPY moving against            -> hold to 2:50, stop -30%
  #A3  Today's Daybook watchlist pick with all three 15m checks agreeing    -> hold to 2:50, stop -30%

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
EXITS = {"2v": {"target": None, "stop": 0.30}, "5": {"target": None, "stop": 0.30}, "A3": {"target": None, "stop": 0.30}}
ORDER = ["2v", "5", "A3"]
WARN = []


# ---------------------------------------------------------------- market data: Massive first, Yahoo as backup

MASSIVE = "https://api.massive.com"
SOURCES = {}            # ticker -> "massive" | "yahoo", reported in the output


def massive_key():
    k = os.environ.get("MASSIVE_API_KEY")
    f = ADVISOR / "massive_api_key.env"
    if not k and f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith("MASSIVE_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
    return k


def massive_get(path, params):
    import urllib.parse, urllib.request
    url = f"{MASSIVE}{path}?{urllib.parse.urlencode(params)}"
    out = []
    while url:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {massive_key()}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        if d.get("status") == "NOT_AUTHORIZED":
            raise RuntimeError("not entitled")
        out += d.get("results") or []
        url = d.get("next_url")
    return out


def massive_frame(results, tz):
    d = pd.DataFrame(results)
    idx = pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert(tz)
    return pd.DataFrame({"Open": d["o"].values, "High": d["h"].values, "Low": d["l"].values, "Close": d["c"].values,
                         "Volume": d["v"].values}, index=pd.DatetimeIndex(idx))


def massive_candles(t, day):
    """(5m regular-hours candles in CT, daily candles with today's row built from the 5m ones), like history.py."""
    r5 = massive_get(f"/v2/aggs/ticker/{t}/range/5/minute/{day - timedelta(days=90)}/{day}",
                     {"adjusted": "false", "sort": "asc", "limit": 50000})
    rd = massive_get(f"/v2/aggs/ticker/{t}/range/1/day/{day - timedelta(days=730)}/{day - timedelta(days=1)}",
                     {"adjusted": "true", "sort": "asc", "limit": 50000})
    if not r5 or not rd:
        return None, None
    b5 = massive_frame(r5, "America/Chicago")
    hm = b5.index.strftime("%H:%M")
    b5 = b5[(hm >= "08:30") & (hm < "15:00")]
    d = massive_frame(rd, "America/New_York")
    d.index = pd.DatetimeIndex(d.index.tz_localize(None).normalize())
    today = b5[b5.index.date == day]
    if len(today):
        d.loc[pd.Timestamp(day)] = [float(today["Open"].iloc[0]), float(today["High"].max()), float(today["Low"].min()),
                                    float(today["Close"].iloc[-1]), float(today["Volume"].sum())]
    return b5, d


def load(tickers, day, cutoff, need_daily=True):
    """5m (and daily) candles per ticker: Massive when it has the candles up to the cutoff, else Yahoo."""
    b5, daily, missing = {}, {}, []
    last_needed = cutoff - timedelta(minutes=10)            # the 8:35 candle must exist by 8:46
    for t in tickers:
        got = None
        if massive_key():
            try:
                got = massive_candles(t, day)
            except Exception:
                got = None
        if got and got[0] is not None and len(got[0]) and got[0].index[-1] >= last_needed:
            b5[t], daily[t] = got
            SOURCES[t] = "massive"
        else:
            missing.append(t)
    if missing:
        yb5 = M.candles(missing, "60d", "5m")
        yd = M.candles(missing, "2y", "1d") if need_daily else {}
        for t in missing:
            if t in yb5:
                b5[t] = yb5[t]
                SOURCES[t] = "yahoo"
            if t in yd:
                daily[t] = yd[t]
    return b5, daily


def vix_closes():
    """date -> VIX close, from Yahoo (Massive's plan doesn't include indices). {} when unreachable."""
    try:
        vix = M.candles(["^VIX"], "1y", "1d").get("^VIX")
        return {d.date(): float(v) for d, v in vix["Close"].items()} if vix is not None and len(vix) else {}
    except Exception:
        return {}


def gap_signals(day, cutoff, skip):
    tickers = [t for t in TICKERS if t not in skip and t not in C.EXCLUDE]
    b5, daily = load(tickers + (["SPY"] if "SPY" not in tickers else []), day, cutoff)
    vixc = vix_closes()
    WARN.extend([] if any(d < day for d in vixc) else ["VIX unavailable: #2v skipped today"])
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
        v = next((vixc[d] for d in sorted(vixc, reverse=True) if d < day), None)
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
            tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, c, sp, gap, v if v is not None else 99.0, at)
            for p in T.PLANS:
                if p["id"] == "2v" and v is None:
                    continue                              # never guess the VIX
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
    b5, _ = load([p["ticker"] for p in picks], day, cutoff, need_daily=False)
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
            "daybook_watchlist_found": watchlist_found, "candidates": uniq,
            "data": {"massive": sorted(t for t, v in SOURCES.items() if v == "massive"),
                     "yahoo": sorted(t for t, v in SOURCES.items() if v == "yahoo")}, "warnings": WARN}


def main():
    args = sys.argv[1:]
    if "--day" in args:
        print(json.dumps(candidates(date.fromisoformat(args[args.index("--day") + 1])), indent=1))
    else:
        print(json.dumps(candidates(datetime.now(S.CT).date(), live=True), indent=1))


if __name__ == "__main__":
    main()
