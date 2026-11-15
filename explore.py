#!/usr/bin/env python3
"""Explore: looks for better ways to trade the stocks Daybook presents. Suggestions only.

For every stock Daybook has shown (minus the no-trade list) it replays the last ~60 days of 5-minute candles and tries:
  setups   8 ways to enter (Daybook's 15m signal, opening-range breakout, VWAP cross, 5m EMA 9/21 cross, RSI bounce,
           gap-and-go, gap fade, first-hour trend)
  filters  15 conditions at the entry (daily and weekly trend, SPY, VWAP side, volume surge, RSI room, how far the
           stock already moved, gap, time of day, VIX, weekday, 15m MACD/CCI/EMA50)
  exits    every target x stop, a time cut, locking in a gain, and selling half at one target and half at another
A plan = a setup (or any setup) + up to two filters + an exit. Plans are picked on the older ~2/3 of days and checked on
the newest ~1/3 they never saw; only plans that also made money there are kept, ranked by how steady they were.
Option prices are estimated from the stock's moves (Yahoo keeps no old option prices), with no trading costs.
Every result is per $1,000 traded, so $10 = 1%.

  python explore.py          # writes docs/explore.json
"""
import json
import math
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import advisor_config as A

sys.path.insert(0, str(A.DAYBOOK))

OUT = A.HERE / "docs" / "explore.json"
FIRST, LAST_ENTRY, EXIT = "08:45", "13:30", "14:50"
TARGETS = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, None]
STOPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, None]
CUTS = ["11:00", "12:30", EXIT]
LOCKS = [None, (0.10, 0.0), (0.20, 0.10)]             # after +A%, the stop moves up to +B%
HALVES = [(0.05, 0.20), (0.10, 0.30), (0.10, None), (0.15, 0.50), (0.20, None)]
MIN_TRAIN, MIN_CHECK = 30, 10
SETUPS = ["15m EMA+MACD+CCI", "Opening range breakout", "VWAP cross", "5m EMA 9/21 cross", "RSI bounce",
          "Gap and go", "Gap fade", "First-hour trend"]


def exit_specs():
    specs = [{"target": t, "stop": s, "cut": c, "lock": k} for t in TARGETS for s in STOPS for c in CUTS for k in LOCKS]
    specs += [{"target": a, "target2": b, "stop": s, "cut": EXIT, "lock": None} for a, b in HALVES for s in STOPS]
    return specs


# ---------------------------------------------------------------- data

def load(tickers):
    import strategy_15m as M
    b5 = M.candles(tickers + ["SPY"], "60d", "5m")
    daily = M.candles(tickers, "2y", "1d")
    vix = M.candles(["^VIX"], "1y", "1d").get("^VIX")
    return b5, daily, vix


def day_frame(df):
    """5m candles with VWAP, EMA 9/21, RSI 14 and a 20-candle volume average."""
    import scanner as S
    x = df.copy()
    d = x.index.date
    tp = (x["High"] + x["Low"] + x["Close"]) / 3
    x["vwap"] = (tp * x["Volume"]).groupby(d).cumsum() / x["Volume"].groupby(d).cumsum().replace(0, np.nan)
    x["e9"], x["e21"] = S.ema(x["Close"], 9), S.ema(x["Close"], 21)
    x["rsi"] = S.rsi(x["Close"], 14)
    x["vavg"] = x["Volume"].rolling(20).mean().shift(1)
    x["dopen"] = x["Open"].groupby(d).transform("first")
    return x


def fifteen(df):
    import strategy_15m as M
    agg = df.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    return M.ind15(agg)


def daily_context(df):
    """Per date, known before the open: daily trend (Daybook's), weekly trend, ATR, volatility, prior close."""
    import scanner as S
    prior = S.daily_bias(df)
    c = df["Close"]
    e20 = S.ema(c, 20)
    wk = np.where((c > e20) & (e20 > e20.shift(5)), 1, np.where((c < e20) & (e20 < e20.shift(5)), -1, 0))
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    extra = pd.DataFrame({"week": pd.Series(wk, index=df.index).shift(1), "atr": tr.rolling(14).mean().shift(1)})
    extra.index = extra.index.date
    return prior.join(extra)


# ---------------------------------------------------------------- entries

def entries(t, x, x15, day, gap):
    """(setup, call, entry time) for one stock-day: the first trigger of each setup in each direction."""
    import strategy_15m as M
    out, seen = [], set()

    def add(name, call, at):
        if (name, call) not in seen and FIRST <= at.strftime("%H:%M") <= LAST_ENTRY:
            seen.add((name, call))
            out.append((name, call, at))

    for s in M.signals(x15, day):
        add(SETUPS[0], s["direction"] == "CALLS", s["at"])
    xd = x[x.index.date == day]
    if len(xd) < 12:
        return out
    end = xd.index + timedelta(minutes=5)
    hi, lo = xd["High"].iloc[:3].max(), xd["Low"].iloc[:3].min()        # opening range: 8:30-8:45
    c, v, prev = xd["Close"], xd["vwap"], xd["Close"].shift()
    for i in range(3, len(xd)):
        at, ci = end[i], c.iloc[i]
        if at.strftime("%H:%M") <= "11:00":
            if ci > hi:
                add(SETUPS[1], True, at)
            if ci < lo:
                add(SETUPS[1], False, at)
        pv, pc = v.iloc[i - 1], prev.iloc[i]
        if pc <= pv < ci:
            add(SETUPS[2], True, at)
        if pc >= pv > ci:
            add(SETUPS[2], False, at)
        up, up0 = xd["e9"].iloc[i] > xd["e21"].iloc[i], xd["e9"].iloc[i - 1] > xd["e21"].iloc[i - 1]
        if up and not up0:
            add(SETUPS[3], True, at)
        if up0 and not up:
            add(SETUPS[3], False, at)
        r, r0 = xd["rsi"].iloc[i], xd["rsi"].iloc[i - 1]
        if r0 <= 30 < r:
            add(SETUPS[4], True, at)
        if r0 >= 70 > r:
            add(SETUPS[4], False, at)
    first = xd.iloc[:3]
    went_up = first["Close"].iloc[-1] > first["Open"].iloc[0]
    if abs(gap) >= 0.004:
        add(SETUPS[5] if (gap > 0) == went_up else SETUPS[6], went_up, end[2])
    k = int(np.searchsorted(end, end[0] + timedelta(minutes=55)))          # the 9:25-9:30 candle
    if k < len(xd):
        ck = c.iloc[k]
        if ck > v.iloc[k] and ck > xd["Open"].iloc[0] and ck > hi:
            add(SETUPS[7], True, end[k])
        if ck < v.iloc[k] and ck < xd["Open"].iloc[0] and ck < lo:
            add(SETUPS[7], False, end[k])
    return out


def tags(call, row, r15, ctx, spy, gap, vix, at):
    side = 1 if call else -1
    hm = at.strftime("%H:%M")
    rsi = row["rsi"] if call else 100 - row["rsi"]
    moved = side * (row["Close"] - row["dopen"]) / ctx["atr"] if ctx["atr"] else 0
    sp = 1 if spy is not None and spy["Close"] > spy["vwap"] and spy["Close"] > spy["dopen"] else \
        (-1 if spy is not None and spy["Close"] < spy["vwap"] and spy["Close"] < spy["dopen"] else 0)
    trend = lambda v, name: f"With {name}" if v == side else (f"Against {name}" if v == -side else f"No {name}")
    return {
        "Direction": "Calls" if call else "Puts",
        "Daily trend": trend(ctx["bias"], "daily trend"),
        "Weekly trend": trend(ctx["week"], "weekly trend"),
        "SPY": "SPY moving with it" if sp == side else ("SPY moving against" if sp == -side else "SPY mixed"),
        "VWAP": "Right side of VWAP" if side * (row["Close"] - row["vwap"]) > 0 else "Wrong side of VWAP",
        "Volume": "Volume surge" if row["vavg"] and row["Volume"] >= 1.5 * row["vavg"] else "Normal volume",
        "RSI": "RSI has room" if rsi < 50 else ("RSI stretched" if rsi > 70 else "RSI 50–70"),
        "Moved": "Moved against today" if moved < 0 else ("Big move already" if moved > 0.5 else "Small move so far"),
        "Gap": "Gapped its way" if side * gap > 0.002 else ("Gapped against" if side * gap < -0.002 else "Flat open"),
        "Time": "8:45–9:30" if hm <= "09:30" else ("9:30–10:30" if hm <= "10:30" else ("10:30–12:00" if hm <= "12:00" else "After 12:00")),
        "VIX": "VIX under 16" if vix < 16 else ("VIX over 22" if vix > 22 else "VIX 16–22"),
        "Weekday": at.strftime("%a"),
        "15m MACD": "15m MACD agrees" if r15 is not None and side * (r15["macd"] - r15["sig"]) > 0 else "15m MACD disagrees",
        "15m CCI": "15m CCI past ±100" if r15 is not None and side * r15["cci"] > 100 else "15m CCI inside ±100",
        "15m EMA50": "Right side of 15m EMA50" if r15 is not None and side * (r15["close"] - r15["ema"]) > 0 else "Wrong side of 15m EMA50",
    }


# ---------------------------------------------------------------- prices

def path(call, spot, vol, dte, t0, bars):
    """Estimated option price (return vs the buy) at each 5m point until 2:50 CT, no trading costs.
    Returns (returns, gap flags, minute-of-day of each point)."""
    import scanner as S
    years = lambda ts: max(dte - (ts - t0).total_seconds() / 86400, 1e-4) / 365
    buy = S.bs_price(spot, spot, years(t0), vol, call)
    if buy <= 0.05:
        return None
    r, g, mins = [], [], []
    for ts, b in bars.iterrows():
        pts = [b["Open"], b["Low"], b["High"]] if b["Close"] >= b["Open"] else [b["Open"], b["High"], b["Low"]]
        for i, px in enumerate(pts):
            r.append(S.bs_price(px, spot, years(ts), vol, call) / buy - 1)
            g.append(i == 0)
            mins.append(ts.hour * 60 + ts.minute)
        if (ts + timedelta(minutes=5)).strftime("%H:%M") >= EXIT:
            break
    r.append(S.bs_price(b["Close"], spot, years(ts), vol, call) / buy - 1)
    g.append(False)
    mins.append(ts.hour * 60 + ts.minute + 5)
    return np.array(r), np.array(g), np.array(mins), buy


def outcome(r, g, mins, specs):
    """Return of one trade under every exit spec."""
    body, n = r[:-1], len(r) - 1
    up = np.maximum.accumulate(body)
    down = -np.minimum.accumulate(body)
    hit_t = lambda t: n if t is None else int(np.searchsorted(up, t, "left"))
    hit_s = lambda s: n if s is None else int(np.searchsorted(down, s, "left"))
    cut_i = {c: int(np.searchsorted(mins[:-1], int(c[:2]) * 60 + int(c[3:]), "left")) for c in CUTS[:-1]}
    lock_i = {}
    for k in LOCKS[1:]:
        a = hit_t(k[0])
        later = np.flatnonzero(body[a + 1:] <= k[1]) if a < n else []
        lock_i[k] = a + 1 + int(later[0]) if len(later) else n

    def one(t, s, cut, lock):
        it, il = hit_t(t), hit_s(s)
        ik = lock_i[lock] if lock else n
        ic = cut_i.get(cut, n)
        i = min(it, il, ik, ic)
        if i == n:
            return float(r[-1])
        if i == il:
            return float(r[i]) if g[i] else -s
        if i == ik:
            return float(r[i]) if g[i] else lock[1]
        if i == it:
            return t
        return float(r[i])                                              # sold at the time cut

    out = np.empty(len(specs))
    for j, sp in enumerate(specs):
        v = one(sp["target"], sp["stop"], sp["cut"], sp["lock"])
        if "target2" in sp:
            v = (v + one(sp["target2"], sp["stop"], sp["cut"], None)) / 2
        out[j] = v
    return out


def trades(tickers):
    import scanner as S
    b5, daily, vix = load(tickers)
    vixc = {d.date(): float(v) for d, v in vix["Close"].items()} if vix is not None else {}
    spy = day_frame(b5["SPY"]) if "SPY" in b5 else None
    today = datetime.now(S.CT).date()
    specs = exit_specs()
    rows = []
    for t in tickers:
        if t not in b5 or t not in daily:
            continue
        x, x15, ctx = day_frame(b5[t]), fifteen(b5[t]), daily_context(daily[t])
        opens = {d.date(): float(o) for d, o in daily[t]["Open"].items()}
        for day in sorted(set(x.index.date)):
            if day >= today or day not in ctx.index or day not in opens:
                continue
            c = ctx.loc[day]
            if any(pd.isna(c[k]) for k in ("rv", "prev_close", "atr")):
                continue
            vol = max(float(c["rv"]) * 1.1, 0.25)
            gap = opens[day] / float(c["prev_close"]) - 1
            v = next((vixc[d] for d in sorted(vixc, reverse=True) if d < day), 19.0)
            for setup, call, at in entries(t, x, x15, day, gap):
                bars = x[(x.index >= at) & (x.index.date == day)]
                prev_bar = at - timedelta(minutes=5)
                if bars.empty or prev_bar not in x.index:
                    continue
                row = x.loc[prev_bar]
                done15 = x15[x15.index + timedelta(minutes=15) <= at]
                r15 = done15.iloc[-1] if len(done15) else None
                sp = spy.loc[prev_bar] if spy is not None and prev_bar in spy.index else None
                p = path(call, float(row["Close"]), vol, S.weekly_dte(day), at, bars)
                if p is None:
                    continue
                ret = outcome(p[0], p[1], p[2], specs)
                rows.append({"ticker": t, "date": day.isoformat(), "time": at.strftime("%H:%M"), "setup": setup,
                             "pnl": ret * A.TRADE_DOLLARS,          # per $1,000 traded
                             "tags": tags(call, row, r15, c, sp, gap, v, at)})
    rows.sort(key=lambda r: (r["date"], r["time"], r["ticker"]))
    return rows, specs


# ---------------------------------------------------------------- search

def steady(dates, pnl):
    """Stats that show how steady a plan was: win rate, average, weeks up, worst drop."""
    pnl = np.asarray(pnl, float)
    if not len(pnl):
        return {"n": 0}
    wk = pd.Series(pnl, index=pd.to_datetime(dates)).groupby(pd.to_datetime(dates).to_period("W")).sum()
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max())
    return {"n": int(len(pnl)), "win": round(float((pnl > 0).mean()), 3), "avg": round(float(pnl.mean()), 1),
            "total": round(float(pnl.sum())), "weeks": int(len(wk)), "weeks_up": int((wk > 0).sum()),
            "avg_win": round(float(pnl[pnl > 0].mean()), 1) if (pnl > 0).any() else None,
            "avg_loss": round(float(pnl[pnl <= 0].mean()), 1) if (pnl <= 0).any() else None,
            "worst_drop": round(dd)}


def search(rows, m, specs):
    days = sorted({r["date"] for r in rows})
    cut = days[int(len(days) * (1 - A.TEST_SHARE))]
    train = np.array([r["date"] < cut for r in rows])
    groups = sorted({g for r in rows for g in r["tags"]} - {"Weekday"})       # weekday is shown, not planned on
    labels = sorted({(g, r["tags"][g]) for r in rows for g in groups})
    single = {lab: np.array([r["tags"][lab[0]] == lab[1] for r in rows]) for lab in labels}
    keys = [()] + [(a,) for a in labels] + [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:] if a[0] != b[0]]
    setup_of = np.array([r["setup"] for r in rows])
    found = []
    for setup in [None] + SETUPS:
        base = train & (setup_of == setup if setup else True)
        if base.sum() < MIN_TRAIN:
            continue
        idx = np.flatnonzero(base)
        mt = m[idx].astype(np.float32)
        mask = np.array([np.all([single[k][idx] for k in key], axis=0) if key else np.ones(len(idx), bool) for key in keys],
                        dtype=np.float32)
        n = mask.sum(axis=1)
        ok = n >= MIN_TRAIN
        sums = mask[ok] @ mt
        avg = sums / n[ok][:, None]
        score = avg * (n[ok] / (n[ok] + 30))[:, None]                    # small samples shrink toward 0
        best_j = score.argmax(axis=1)
        for kk, j, sc in zip(np.flatnonzero(ok), best_j, score[np.arange(len(best_j)), best_j]):
            if sc > 0:
                found.append((float(sc), setup, keys[kk], int(j)))
    found.sort(key=lambda f: -f[0])

    dates = np.array([r["date"] for r in rows])

    def plan(setup, key, j):
        sel = (setup_of == setup if setup else np.ones(len(rows), bool))
        for k in key:
            sel = sel & single[k]
        tr, ck = sel & train, sel & ~train
        tick = {}
        for i in np.flatnonzero(sel):
            tick.setdefault(rows[i]["ticker"], []).append(m[i, j])
        best = sorted(((t, steady([0] * len(v), v)) for t, v in tick.items() if len(v) >= 4), key=lambda x: -x[1]["avg"])
        return {"setup": setup or "Any setup", "filters": [{"group": g, "label": v} for g, v in key],
                "exit": specs[j], "tuned": steady(dates[tr], m[tr, j]), "check": steady(dates[ck], m[ck, j]),
                "all": steady(dates[sel], m[sel, j]),
                "stocks": [{"ticker": t, **s} for t, s in best[:5]],
                "avoid": [{"ticker": t, **s} for t, s in best[::-1][:3] if s["avg"] < 0]}

    plans, seen = [], set()
    for sc, setup, key, j in found[:400]:
        p = plan(setup, key, j)
        c = p["check"]
        if c["n"] < MIN_CHECK or c["avg"] <= 0:
            continue
        sig = (p["setup"], tuple(sorted(f["label"] for f in p["filters"])))
        if sig in seen:
            continue
        seen.add(sig)
        a = p["all"]
        p["steadiness"] = round(a["weeks_up"] / max(a["weeks"], 1), 2)
        p["score"] = round(min(p["tuned"]["avg"], c["avg"]) * p["steadiness"], 1)
        plans.append(p)
    plans.sort(key=lambda p: -p["score"])
    return {"tuned_days": [days[0], max(d for d in days if d < cut)], "check_days": [cut, days[-1]],
            "tried": len(found), "plans": plans[:6]}


def exit_table(rows, m, specs, setup=None):
    """Plain target x stop comparison (hold until 2:50, no lock) for one setup or all of them."""
    sel = np.array([setup is None or r["setup"] == setup for r in rows])
    dates = np.array([r["date"] for r in rows])[sel]
    out = []
    for j, sp in enumerate(specs):
        if sp["cut"] == EXIT and sp["lock"] is None and "target2" not in sp:
            out.append({"target": sp["target"], "stop": sp["stop"], **steady(dates, m[sel, j])})
    return out


def named(rows, m, specs, setup, items):
    """The exits asked about by name, for one setup."""
    sel = np.array([setup is None or r["setup"] == setup for r in rows])
    dates = np.array([r["date"] for r in rows])[sel]
    out = []
    for label, want in items:
        j = next(i for i, sp in enumerate(specs)
                 if all(sp.get(k) == v for k, v in want.items()) and ("target2" in sp) == ("target2" in want))
        out.append({"label": label, "exit": specs[j], **steady(dates, m[sel, j])})
    return out


ASKED = [("Sell +3% · stop −20% (Daybook now)", {"target": 0.03, "stop": 0.20, "cut": EXIT, "lock": None}),
         ("Sell +10% · stop −20%", {"target": 0.10, "stop": 0.20, "cut": EXIT, "lock": None}),
         ("Sell +15% · stop −20%", {"target": 0.15, "stop": 0.20, "cut": EXIT, "lock": None}),
         ("Sell +10% · stop −10%", {"target": 0.10, "stop": 0.10, "cut": EXIT, "lock": None}),
         ("Sell +15% · stop −10%", {"target": 0.15, "stop": 0.10, "cut": EXIT, "lock": None}),
         ("Sell +20% · stop −10%", {"target": 0.20, "stop": 0.10, "cut": EXIT, "lock": None}),
         ("Half +10%, half +30% · stop −20%", {"target": 0.10, "target2": 0.30, "stop": 0.20})]


def build(rows, specs, skip):
    rows = [r for r in rows if r["ticker"] not in skip]
    m = np.array([r["pnl"] for r in rows])
    by_setup = []
    j_cur = next(i for i, sp in enumerate(specs) if sp == {"target": 0.03, "stop": 0.20, "cut": EXIT, "lock": None})
    dates = np.array([r["date"] for r in rows])
    for s in SETUPS:
        sel = np.array([r["setup"] == s for r in rows])
        if not sel.any():
            continue
        avg = np.nanmean(m[sel], axis=0)
        j = int(np.argmax(avg))
        by_setup.append({"setup": s, "n": int(sel.sum()), "now": steady(dates[sel], m[sel, j_cur]),
                         "best_exit": specs[j], "best": steady(dates[sel], m[sel, j])})
    days = sorted({r["date"] for r in rows})
    return {
        "updated": datetime.now().astimezone().isoformat(timespec="minutes"),
        "days": [days[0], days[-1]], "trading_days": len(days), "trades": len(rows),
        "stocks": sorted({r["ticker"] for r in rows}), "no_trade": sorted(skip), "exits_tried": len(specs),
        "search": search(rows, m, specs),
        "asked": {"Daybook's 15m signal": named(rows, m, specs, SETUPS[0], ASKED),
                  "Every setup": named(rows, m, specs, None, ASKED)},
        "grid": {"targets": TARGETS, "stops": STOPS, "cells": exit_table(rows, m, specs)},
        "setups": sorted(by_setup, key=lambda s: -s["best"]["avg"]),
    }


def universe():
    import config as C
    import research_config as R
    status = json.loads((A.DAYBOOK / "docs" / "status.json").read_text())
    shown = {p["ticker"] for r in status["runs"] if r.get("kind") == "watchlist" for p in r.get("picks") or []}
    try:
        skip = set(json.loads((A.DAYBOOK / "docs" / "no_trade.json").read_text()).get("tickers", []))
    except FileNotFoundError:
        skip = set()
    return sorted((shown | set(R.TICKERS)) - skip - set(C.EXCLUDE)), skip


def main():
    tickers, skip = universe()
    cache = os.environ.get("EXPLORE_CACHE")            # local tuning only: reuse downloaded trades
    if cache and os.path.exists(cache):
        rows, specs = pickle.load(open(cache, "rb"))
    else:
        rows, specs = trades(tickers)
        if cache:
            pickle.dump((rows, specs), open(cache, "wb"))
    out = build(rows, specs, skip)
    OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))
    s = out["search"]
    print(f"explore: {out['trades']} trades, {len(out['stocks'])} stocks, {out['trading_days']} days, "
          f"{s['tried']} plans made money on the tuning days, {len(s['plans'])} also on the check days")


if __name__ == "__main__":
    main()
