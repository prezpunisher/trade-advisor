#!/usr/bin/env python3
"""Cheap-contract test: the small Robinhood account buys 1 contract at $0.50 or less. Suggestions only.

Replays three plans on REAL 1-minute option prices from Massive, but instead of the contract nearest the stock price
it buys the nearest-to-the-money contract of the same weekly expiry and side whose first price at the entry time is
$0.50 or less (and at least $0.05). One contract per trade, so results are in real dollars.

  Contracts: $0.25 or less (since 2026-11-22; was $0.50). The bot: up to 4 trades a day, one per ticker, $100 a day.

  #2v  Gap and go + with daily trend + SPY moving with it + VIX 22 or under -> hold to 2:50, stop -30%   (history.py signals)
  #5   Gap and go + wrong side of 15m EMA50 + SPY moving against           -> hold to 2:50, stop -30%   (history.py signals)
  #A3  Daybook watchlist pick where all three 15m checks agree            -> hold to 2:50, stop -30%
       (Daybook research.json watchlist_check, since 2026-10-26)

SLIP = what we assume a cheap option costs in spread each way (buy a bit higher, sell a bit lower).

  python cheap_test.py test     # the 2-year test (needs history.py's cached stock candles); writes docs/cheap.json
  python cheap_test.py track    # books each finished day since TRACK_FROM (Yahoo candles + Massive option prices)
"""
import json
import sys
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import advisor_config as A
import history as H
import tracker as T

import strategy as ST

CFG = ST.load()                  # strategy.json (STRATEGY_FILE=... for a draft)
MAX_PRICE, MIN_PRICE = CFG["account"]["max_price"], CFG["account"]["min_price"]
BOT_PER_DAY, BOT_BUDGET = CFG["account"]["trades_per_day"], CFG["account"]["daily_budget"]
SLIP = CFG["test"]["slip"]
LATE_MIN = 5                     # the contract must trade within 5 minutes of the signal
OUT = Path(__import__("os").environ.get("CHEAP_OUT", A.HERE / "docs" / "cheap.json"))
PLANS = {pid: {"name": p["name"], "exit": p["exit"]} for pid, p in ST.plans(CFG).items()}

def first_price(sym, day, at):
    b = H.option_bars(sym, day)
    if b is None or b.empty:
        return None, None
    after = b[b.index >= at]
    if after.empty or (after.index[0] - at) > timedelta(minutes=LATE_MIN):
        return None, b
    return float(after["Open"].iloc[0]), b


def cheap_contract(t, day, exp, call, spot, at):
    """Nearest-to-the-money strike (at or out of the money) priced MIN..MAX at entry. Prices fall as strikes move
    out of the money, so a binary search finds it in a few requests."""
    listed = H.contracts(t, exp, spot, spot)
    side = "call" if call else "put"
    mine = [c for c in listed or [] if c.get("contract_type") == side]
    if not mine:
        return None
    e = max(c["expiration_date"] for c in mine)
    mine = [c for c in mine if c["expiration_date"] == e and ((c["strike_price"] >= spot) if call else (c["strike_price"] <= spot))]
    mine.sort(key=lambda c: abs(c["strike_price"] - spot))           # nearest the money first
    cache = {}

    def px(i):
        if i not in cache:
            sym = mine[i]["ticker"].removeprefix("O:")
            p, b = first_price(sym, day, at)
            cache[i] = (p, sym, b)
        return cache[i]

    lo, hi = 0, len(mine) - 1
    if hi < 0:
        return None
    # find the first index whose price is <= MAX (ignoring strikes that didn't trade: step outward past them)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        j = mid
        while j <= hi and px(j)[0] is None:
            j += 1
        if j > hi:
            hi = mid - 1
            continue
        if px(j)[0] <= MAX_PRICE:
            best, hi = j, mid - 1
        else:
            lo = j + 1
    if best is None:
        return None
    p, sym, b = px(best)
    if p < MIN_PRICE:
        return None
    return sym, p, b


def run_exit(b, at, ex):
    r = T.real_exit(b, at, ex)
    if not r:
        return None
    entry, exitp = r["entry"] + SLIP, max(r["exit"] - SLIP, 0)
    return {**r, "qty": 1, "cost": round(entry * 100), "pnl": round((exitp - entry) * 100, 2), "ret": round(exitp / entry - 1, 4)}


def gap_signals():
    """history.py's gap-and-go entries, matched against strategy.json's gap plans (not tracker.PLANS)."""
    data = H.load_all(fetch=False)
    gap = ST.gap_plans(CFG)
    saved, T.PLANS = T.PLANS, gap                 # history.signals() matches entries against T.PLANS
    try:
        sigs = H.signals(data)
    finally:
        T.PLANS = saved
    out = []
    for s in sigs:
        for p in gap:
            if p["id"] in s["plans"]:
                out.append({**s, "plan": p["id"]})
    return out


def all3_signals():
    sys.path.insert(0, str(A.DAYBOOK))
    r = json.loads((A.DAYBOOK / "docs" / "research.json").read_text())
    out = []
    for w in r["watchlist_check"]:
        if not w.get("all3_ok") or not w.get("symbol"):
            continue
        sym = w["symbol"]                                   # e.g. NVDA261127P00140000
        k = len(sym) - 15
        exp = datetime.strptime(sym[k:k + 6], "%y%m%d").date()
        strike = int(sym[k + 7:]) / 1000
        h, m = (int(v) for v in w["alert"].split(":"))
        day = date.fromisoformat(w["date"])
        at = pd.Timestamp(datetime(day.year, day.month, day.day, h if h >= 7 else h + 12, m), tz="America/Chicago")
        out.append({"ticker": w["ticker"], "day": day, "call": w["dir"] == "C", "at": at, "spot": strike, "exp": exp,
                    "plan": "A3", "gap": None, "vix": None})
    return out


TRACK_FROM = date(2026, 11, 23)       # paper tracking starts the bot's first day
ORDER = list(PLANS)


def stats(r):
    pnl = np.array([x["pnl"] for x in r], float)
    if not len(pnl):
        return {"n": 0}
    eq = np.cumsum(pnl)
    months = pd.Series(pnl, index=pd.to_datetime([x["date"] for x in r]).to_period("M")).groupby(level=0).sum()
    streak = run = 0
    for v in pnl:
        run = run + 1 if v <= 0 else 0
        streak = max(streak, run)
    return {"n": len(r), "win": round(float((pnl > 0).mean()), 3), "avg": round(float(pnl.mean()), 2),
            "total": round(float(pnl.sum()), 2), "avg_cost": round(float(np.mean([x["cost"] for x in r])), 1),
            "worst": round(float(pnl.min()), 2), "best": round(float(pnl.max()), 2),
            "max_drawdown": round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 2),
            "loss_streak": streak, "months_up": int((months > 0).sum()), "months": int(len(months)),
            "monthly": [{"m": str(k), "total": round(float(v), 2)} for k, v in months.items()],
            "first": r[0]["date"], "last": r[-1]["date"]}


def bot_rows(rows):
    """The trades the bot takes: in priority order (#2v, #5, A3, then entry time, then ticker), up to BOT_PER_DAY a
    day, one per ticker, while the day's total cost stays within BOT_BUDGET. Also flags them with "bot": True."""
    out, days = [], {}
    for x in rows:
        x.pop("bot", None)
    for x in sorted(rows, key=lambda x: (x["date"], ORDER.index(x["plan"]), x["entry_time"], x["ticker"])):
        d = days.setdefault(x["date"], {"n": 0, "cost": 0, "tickers": set()})
        if d["n"] >= BOT_PER_DAY or x["ticker"] in d["tickers"] or d["cost"] + x["cost"] > BOT_BUDGET:
            continue
        d["n"] += 1
        d["cost"] += x["cost"]
        d["tickers"].add(x["ticker"])
        x["bot"] = True
        out.append(x)
    return out


def summarize(rows):
    out = [{"id": pid, "name": p["name"], "exit": p["exit"], **stats(sorted((x for x in rows if x["plan"] == pid),
            key=lambda x: (x["date"], x["entry_time"])))} for pid, p in PLANS.items()]
    out.append({"id": "bot", "name": f"The bot: up to {BOT_PER_DAY} trades a day, ${BOT_BUDGET} at most, #2v then #5 then all-3", **stats(bot_rows(rows))})
    return out


def book(s, rows, skipped):
    got = cheap_contract(s["ticker"], s["day"], s["exp"], s["call"], s["spot"], s["at"])
    r = run_exit(got[2], s["at"], PLANS[s["plan"]]["exit"]) if got else None
    if not r:
        skipped[s["plan"]] = skipped.get(s["plan"], 0) + 1
        return
    rows.append({"plan": s["plan"], "date": s["day"].isoformat(), "ticker": s["ticker"],
                 "dir": "CALLS" if s["call"] else "PUTS", "symbol": got[0], "stock": round(s["spot"], 2), **r})


def test():
    skip = H.no_trade()                                     # Daybook's no-trade list applies to every plan
    sigs = [s for s in gap_signals() + (all3_signals() if "A3" in PLANS else []) if s["ticker"] not in skip]
    rows, skipped = [], {}
    for i, s in enumerate(sigs):
        book(s, rows, skipped)
        if i % 50 == 0:
            print(f"{i}/{len(sigs)}", file=sys.stderr, flush=True)
    return {"rows": rows, "skipped": skipped, "summary": summarize(rows)}


def track(saved):
    """Paper-book each finished trading day since TRACK_FROM with today's live rules (small_signals.py)."""
    import small_signals as SS
    import scanner as S
    live = saved.get("live") or {"started": TRACK_FROM.isoformat(), "rows": [], "days": [], "skipped": {}}
    done = set(live["days"])
    now = datetime.now(S.CT)
    last = now.date() if now.strftime("%H:%M") >= "15:10" else now.date() - timedelta(days=1)
    d = TRACK_FROM
    while d <= last:
        if d.weekday() < 5 and d.isoformat() not in done:
            c = SS.candidates(d)
            for k in c["candidates"]:
                h, m = (int(v) for v in k["signal_time"].split(":"))
                at = pd.Timestamp(datetime(d.year, d.month, d.day, h, m), tz=S.CT)
                book({"ticker": k["ticker"], "day": d, "call": k["direction"] == "CALLS", "spot": float(k["stock"]),
                      "at": at, "exp": date.fromisoformat(c["weekly_expiry"]), "plan": k["plan"]},
                     live["rows"], live["skipped"])
            live["days"].append(d.isoformat())
            print(d, len(c["candidates"]), "candidates", file=sys.stderr)
        d += timedelta(days=1)
    live["summary"] = summarize(live["rows"])
    return live


def main():
    saved = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "test" in sys.argv[1:] or "test" not in saved:
        saved["test"] = test()
    if "track" in sys.argv[1:]:
        saved["live"] = track(saved)
    saved.update({"updated": datetime.now().isoformat(timespec="minutes"), "max_price": MAX_PRICE, "slip": SLIP,
                  "bot_per_day": BOT_PER_DAY, "bot_budget": BOT_BUDGET,
                  "plans": [{"id": k, "name": v["name"], "exit": v["exit"]} for k, v in PLANS.items()]})
    for k in ("summary", "rows"):
        saved.pop(k, None)                                   # old layout
    OUT.write_text(json.dumps(saved, separators=(",", ":")))
    for s in saved["test"]["summary"]:
        print("test", {k: s.get(k) for k in ("id", "n", "win", "avg", "total", "max_drawdown", "loss_streak")})
    for s in (saved.get("live") or {}).get("summary", []):
        print("live", {k: s.get(k) for k in ("id", "n", "win", "avg", "total")})


if __name__ == "__main__":
    main()
