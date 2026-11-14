#!/usr/bin/env python3
"""Advisor: studies everything Daybook has collected and suggests what could work. Suggestions only.

Two kinds of evidence:
  estimated  60 days of 15m EMA50 + MACD + CCI signals on ~50 liquid names, with option prices estimated from the
             stock's 5-minute moves (Yahoo keeps no old option prices). Lots of trades, approximate prices.
  real       Daybook's own picks and 15m signals, scored on real 1-minute option prices. Few trades, exact prices.

  python advisor.py          # writes docs/advisor.json
"""
import itertools
import json
import math
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np

import advisor_config as A

sys.path.insert(0, str(A.DAYBOOK))      # Daybook's indicator and pricing code, imported read-only


# ---------------------------------------------------------------- one trade's price path

def option_path(call, spot, vol, dte, t0, bars, bs, exit_ct):
    """Estimated option sell price (as a return on the buy) at every 5m candle point until the exit time.
    Returns (returns, gap flags, buy price) or None when the option is too cheap to price."""
    years = lambda ts: max(dte - (ts - t0).total_seconds() / 86400, 1e-4) / 365
    half = A.SPREAD_PCT / 2 + A.SLIPPAGE_PCT
    fee = A.FEE_PER_CONTRACT / 100
    mid0 = bs(spot, spot, years(t0), vol, call)
    if mid0 <= 0.05:
        return None
    buy = mid0 * (1 + half) + fee
    rets, gaps = [], []
    for ts, b in bars.iterrows():
        pts = [b["Open"], b["Low"], b["High"]] if b["Close"] >= b["Open"] else [b["Open"], b["High"], b["Low"]]
        for i, px in enumerate(pts):
            rets.append((bs(px, spot, years(ts), vol, call) * (1 - half) - fee) / buy - 1)
            gaps.append(i == 0)
        if (ts + timedelta(minutes=5)).strftime("%H:%M") >= exit_ct:
            break
    rets.append((bs(b["Close"], spot, years(ts), vol, call) * (1 - half) - fee) / buy - 1)   # sold at the exit time
    gaps.append(False)
    return np.array(rets), np.array(gaps), buy


def exit_return(rets, gaps, target, stop):
    """First of: target hit (sells at the target), stop hit (at the stop, or worse on a gap), else the exit-time price."""
    body = rets[:-1]
    it = np.flatnonzero(body >= target)[:1] if target is not None else []
    il = np.flatnonzero(body <= -stop)[:1] if stop is not None else []
    it = it[0] if len(it) else None
    il = il[0] if len(il) else None
    if il is not None and (it is None or il < it):
        return float(rets[il]) if gaps[il] else -stop
    if it is not None:
        return target
    return float(rets[-1])


# ---------------------------------------------------------------- estimated trades

def tags_for(call, s, bias, gap, first, hm):
    side = 1 if call else -1
    g = gap * side
    return {
        "Direction": "Calls" if call else "Puts",
        "Daily trend": "With daily trend" if bias == side else ("Against daily trend" if bias == -side else "No daily trend"),
        "CCI": "CCI past ±100" if s["cci_strong"] else "CCI under ±100",
        "Time": "Before 10:00" if hm < "10:00" else ("10:00–12:00" if hm < "12:00" else "After 12:00"),
        "Open": "Gapped its way" if g > 0.002 else ("Gapped against" if g < -0.002 else "Flat open"),
        "Signal": "First signal of day" if first else "Later signal",
        "EMA": "Near EMA50" if abs(s["close"] - s["ema"]) / s["ema"] < 0.003 else "Stretched from EMA50",
    }


def estimated_trades(tickers):
    import research_config as R
    import scanner as S
    import strategy_15m as M
    x15 = {t: M.ind15(df) for t, df in M.candles(tickers, f"{A.DAYS}d", "15m").items()}
    b5 = M.candles(tickers, f"{A.DAYS}d", "5m")
    daily = M.candles(tickers, "1y", "1d")
    today = datetime.now(S.CT).date()
    out = []
    for t, x in x15.items():
        if t not in b5 or t not in daily:
            continue
        prior = S.daily_bias(daily[t])
        opens = {d.date(): float(o) for d, o in daily[t]["Open"].items()}
        for day in sorted(set(x.index.date)):
            if day >= today or day not in prior.index or day not in opens:
                continue
            p = prior.loc[day]
            vol = max(float(p["rv"]) * 1.1 if not math.isnan(p["rv"]) else R.MIN_VOL, R.MIN_VOL)
            gap = opens[day] / float(p["prev_close"]) - 1 if p["prev_close"] else 0.0
            bars_day = b5[t][b5[t].index.date == day]
            for n, s in enumerate(M.signals(x, day)):
                call = s["direction"] == "CALLS"
                after = bars_day[bars_day.index >= s["at"]]
                if after.empty:
                    continue
                path = option_path(call, s["close"], vol, S.weekly_dte(day), s["at"], after, S.bs_price, R.EXIT_CT)
                if path is None:
                    continue
                rets, gaps, buy = path
                hm = s["at"].strftime("%H:%M")
                out.append({"ticker": t, "date": day.isoformat(), "time": hm, "call": call,
                            "qty": max(int(A.TRADE_DOLLARS // (buy * 100)), 1), "buy": buy, "rets": rets, "gaps": gaps,
                            "tags": tags_for(call, s, p["bias"], gap, n == 0, hm)})
    out.sort(key=lambda r: (r["date"], r["time"], r["ticker"]))
    return out


def pnl_matrix(trades, exits):
    """$ result of every trade under every exit: rows = trades, columns = exits."""
    m = np.zeros((len(trades), len(exits)))
    for i, tr in enumerate(trades):
        cost = tr["qty"] * tr["buy"] * 100
        for j, (tg, st) in enumerate(exits):
            m[i, j] = exit_return(tr["rets"], tr["gaps"], tg, st) * cost
    return m


def stats(pnl):
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if not len(pnl):
        return {"n": 0}
    return {"n": int(len(pnl)), "win": round(float((pnl > 0).mean()), 3), "avg": round(float(pnl.mean()), 1),
            "total": round(float(pnl.sum()))}


def tag_masks(rows):
    """Every single condition and every pair of conditions from different groups."""
    labels = sorted({(g, v) for r in rows for g, v in r["tags"].items()})
    masks = {(): np.ones(len(rows), bool)}
    for g, v in labels:
        masks[((g, v),)] = np.array([r["tags"].get(g) == v for r in rows])
    for a, b in itertools.combinations(labels, 2):
        if a[0] != b[0]:
            masks[(a, b)] = masks[(a,)] & masks[(b,)]
    return masks


def search(rows, m, exits, min_train, min_check):
    """Best entry conditions (none, one or two) + exit, picked on the older days and checked on the newest days,
    which the pick never saw. m: $ result per row per exit (NaN = no data for that exit)."""
    days = sorted({r["date"] for r in rows})
    cut = days[int(len(days) * (1 - A.TEST_SHARE))]
    train = np.array([r["date"] < cut for r in rows])
    masks = tag_masks(rows)
    ranked = []
    for key, mask in masks.items():
        sub = m[mask & train]
        n = (~np.isnan(sub)).sum(axis=0)
        if not n.max(initial=0):
            continue
        avg = np.nanmean(sub, axis=0)
        score = np.where(n >= min_train, np.nan_to_num(avg) * n / (n + 20), -np.inf)   # small samples shrink to 0
        j = int(np.argmax(score))
        if np.isfinite(score[j]):
            ranked.append((score[j], key, j))
    ranked.sort(key=lambda r: -r[0])

    def describe(key, j):
        mask = masks[key]
        return {"filters": [{"group": g, "label": v} for g, v in key], "target": exits[j][0], "stop": exits[j][1],
                "tuned": stats(m[mask & train, j]), "check": stats(m[mask & ~train, j]), "all": stats(m[mask, j])}

    plans = []
    for _, key, j in ranked[:3]:
        p = describe(key, j)
        p["proven"] = p["check"]["n"] >= min_check and p["check"]["avg"] > 0
        plans.append(p)
    return {"tuned_days": [days[0], max(d for d in days if d < cut)], "check_days": [cut, days[-1]],
            "plans": plans, "current": describe((), exits.index(A.CURRENT)) if A.CURRENT in exits else None}


# ---------------------------------------------------------------- real Daybook trades

REAL_EXITS = [(t, s) for s in (0.20, 0.15) for t in (0.03, 0.05, 0.10)]


def real_rows(research, skip):
    """Daybook's watchlist picks and 15m signals, with their real-price results at +3/+5/+10 and -20/-15."""
    rows = []
    for src, items in (("Watchlist pick", research.get("watchlist_check", [])), ("15m signal", research.get("shadow", []))):
        for r in items:
            if r["ticker"] in skip or not r.get("exits"):
                continue
            call = (r.get("dir") or r.get("direction", "")[:1]) == "C"
            hm = r.get("alert") or r.get("time") or "12:00"
            h, mm = (int(v) for v in hm.split(":"))
            hm24 = f"{h + 12 if h < 7 else h:02d}:{mm:02d}"         # stored as "1:05" for 1:05 pm
            tags = {"Source": src, "Direction": "Calls" if call else "Puts",
                    "Time": "Before 10:00" if hm24 < "10:00" else ("10:00–12:00" if hm24 < "12:00" else "After 12:00"),
                    "CCI": "CCI past ±100" if r.get("cci_strong") else "CCI under ±100"}
            if src == "Watchlist pick":
                tags["15m check"] = "All three 15m agree" if r.get("all3_ok") else "15m not all agree"
                tags["MACD"] = "MACD crossed against" if r.get("cross_against_1h") else "No MACD cross against"
            else:
                tags["Daily trend"] = "With daily trend" if r.get("daily_trend_agrees") else "Not with daily trend"
            pnl = [np.nan] * len(REAL_EXITS)
            for k, (tg, st) in enumerate(REAL_EXITS):
                ex = r["exits"] if st == 0.20 else (r.get("exits15") or {})
                if str(int(tg * 100)) in ex:
                    pnl[k] = ex[str(int(tg * 100))]["pnl"]
            rows.append({"ticker": r["ticker"], "date": r["date"], "tags": tags, "pnl": pnl})
    return rows


# ---------------------------------------------------------------- report

def patterns(rows, pnl, least):
    out = []
    for g in sorted({g for r in rows for g in r["tags"]}):
        for v in sorted({r["tags"].get(g) for r in rows} - {None}):
            s = stats([p for r, p in zip(rows, pnl) if r["tags"].get(g) == v])
            if s["n"] >= least:
                out.append({"group": g, "label": v, **s})
    return sorted(out, key=lambda r: -r["avg"])


def same_plan(plan, rows, m, exits):
    """How a plan found in one kind of evidence did in the other (None when that evidence lacks a condition)."""
    if not rows or (plan["target"], plan["stop"]) not in exits:
        return None
    groups = {g for r in rows for g in r["tags"]}
    if any(f["group"] not in groups for f in plan["filters"]):
        return None
    mask = np.array([all(r["tags"].get(f["group"]) == f["label"] for f in plan["filters"]) for r in rows])
    return stats(m[mask, exits.index((plan["target"], plan["stop"]))])


def verdict(est, real):
    good = est.get("n", 0) >= A.MIN_TICKER and est["avg"] > 0
    bad = est.get("n", 0) >= A.MIN_TICKER and est["avg"] < -10
    known = real.get("n", 0) >= A.MIN_TICKER_REAL
    if good and (not known or real["avg"] >= 0):
        return "Trade"
    if (bad and (not known or real["avg"] < 0)) or (known and real["avg"] < 0 and not good):
        return "Skip"
    return "Watch"


def build(trades, research, skip, mine):
    trades = [t for t in trades if t["ticker"] not in skip]
    exits = [(tg, st) for st in A.STOPS for tg in A.TARGETS]
    m = pnl_matrix(trades, exits)
    real = real_rows(research, skip)
    rm = np.array([r["pnl"] for r in real], dtype=float).reshape(len(real), len(REAL_EXITS))

    est_found = search(trades, m, exits, A.MIN_TRAIN, A.MIN_TEST) if trades else {"plans": []}
    real_found = search(real, rm, REAL_EXITS, 12, 6) if len({r["date"] for r in real}) >= 6 else {"plans": []}
    for p in est_found["plans"]:
        p["other"] = same_plan(p, real, rm, REAL_EXITS)
    for p in real_found["plans"]:
        p["other"] = same_plan(p, trades, m, exits)

    # the headline: the best real-price plan that held up on unseen days, else the best estimated one that did
    hero = next((dict(p, source="real") for p in real_found["plans"] if p["proven"]), None) \
        or next((dict(p, source="est") for p in est_found["plans"] if p["proven"]), None)
    ex = (hero["target"], hero["stop"]) if hero else A.CURRENT
    je = exits.index(ex) if ex in exits else exits.index(A.CURRENT)
    jr = REAL_EXITS.index(ex) if ex in REAL_EXITS else REAL_EXITS.index(A.CURRENT)

    cards = []
    for t in sorted({t["ticker"] for t in trades} | {r["ticker"] for r in real}):
        e = stats(m[[i for i, x in enumerate(trades) if x["ticker"] == t], je])
        rl = stats(rm[[i for i, x in enumerate(real) if x["ticker"] == t], jr])
        cards.append({"ticker": t, "group": "mine" if t in mine or rl["n"] else "new",
                      "verdict": verdict(e, rl), "est": e, "real": rl})

    est_days = sorted({t["date"] for t in trades})
    real_days = sorted({r["date"] for r in real})
    return {
        "updated": datetime.now().astimezone().isoformat(timespec="minutes"),
        "no_trade": sorted(skip),
        "current": {"target": A.CURRENT[0], "stop": A.CURRENT[1]},
        "hero": hero,
        "hero_per_week": round(hero["all"]["n"] / max(len(real_days if hero["source"] == "real" else est_days) / 5, 1), 1) if hero else None,
        "exit": {"target": ex[0], "stop": ex[1]},
        "real": {"picks": len(real), "days": [real_days[0], real_days[-1]] if real else None, "trading_days": len(real_days),
                 "search": real_found,
                 "grid": [{"target": tg, "stop": st, **stats(rm[:, k])} for k, (tg, st) in enumerate(REAL_EXITS)],
                 "patterns": patterns(real, rm[:, jr], 6)},
        "est": {"signals": len(trades), "names": len({t["ticker"] for t in trades}),
                "days": [est_days[0], est_days[-1]] if trades else None, "trading_days": len(est_days),
                "search": est_found,
                "grid": {"targets": A.TARGETS, "stops": A.STOPS,
                         "cells": [{"target": tg, "stop": st, **stats(m[:, k])} for k, (tg, st) in enumerate(exits)]},
                "patterns": patterns(trades, m[:, je], A.MIN_PATTERN)},
        "tickers": sorted(cards, key=lambda c: ("Trade Watch Skip".split().index(c["verdict"]),
                                                -((c["real"].get("total") or 0) + (c["est"].get("total") or 0)))),
    }


def main():
    import research_config as R
    research = json.loads((A.DAYBOOK / "docs" / "research.json").read_text())
    try:
        skip = set(json.loads((A.DAYBOOK / "docs" / "no_trade.json").read_text()).get("tickers", []))
    except FileNotFoundError:
        skip = set()
    universe = [t for t in dict.fromkeys(R.TICKERS + A.EXTRA_TICKERS) if t not in skip]
    cache = os.environ.get("ADVISOR_CACHE")          # local tuning only: reuse downloaded trades
    if cache and os.path.exists(cache):
        trades = pickle.load(open(cache, "rb"))
    else:
        trades = estimated_trades(universe)
        if cache:
            pickle.dump(trades, open(cache, "wb"))
    out = build(trades, research, skip, set(R.TICKERS))
    A.OUT.parent.mkdir(exist_ok=True)
    A.OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))
    h = out["hero"]
    print(f"advisor: {out['est']['signals']} estimated signals on {out['est']['names']} tickers, {out['real']['picks']} real picks; "
          f"headline: {h and ([f['label'] for f in h['filters']], h['target'], h['stop'], h['source'], h['check'])}")


if __name__ == "__main__":
    main()
