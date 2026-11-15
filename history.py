#!/usr/bin/env python3
"""2-year real-price test: replays the tracked plans (tracker.PLANS) on REAL option prices from Massive. Suggestions
only: nothing is pushed or traded.

Yahoo deletes weekly options days after they expire, so explore.py had to estimate option prices. Massive keeps them.
For every stock the tracker follows, on every trading day since TEST_FROM, it finds the gap-and-go entries exactly
like the tracker (same 5m candles rules, same filters), picks the contract Daybook would (this week's weekly nearest
the stock price), buys at the first real 1-minute price at the entry time and sells by each plan's exit rules on real
1-minute prices. Next to each trade it keeps what the estimate said. Results are per $1,000 traded.

Everything downloaded is kept in massive_cache/ (never committed), so it can stop and pick up where it left off.
The API key is read from massive_api_key.env (git-ignored) or the MASSIVE_API_KEY variable and is never printed.

  python history.py fetch     # downloads what's missing, newest days first (5 requests a minute); publishes as it goes
  python history.py build     # scores every downloaded trade; writes docs/history.json and docs/hbars/
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import advisor_config as A
import explore as X
import tracker as T

sys.path.insert(0, str(A.DAYBOOK))

API = "https://api.massive.com"
KEY_FILE = A.HERE / "massive_api_key.env"
CACHE = A.HERE / "massive_cache"
OUT = A.HERE / "docs" / "history.json"
HBARS = A.HERE / "docs" / "hbars"           # the real 1-minute prices behind each trade, for the page's charts
# frozen 2026-11-14: the stocks explore.py and the tracker study
TICKERS = ["AAPL", "AMD", "AVGO", "CRWV", "HOOD", "INTC", "IWM", "JPM", "META", "MRVL", "MSFT", "MSTR", "NBIS", "NFLX",
           "NOW", "ORCL", "PLTR", "QQQ", "SPY", "TSM", "WMT"]
DATA_FROM = date(2024, 11, 18)             # the plan's history starts about 2 years back
TEST_FROM = date(2025, 2, 3)               # the first ~2.5 months only warm up the daily and weekly trends
TEST_TO = date(2026, 11, 13)
PACE = 12.5                                # seconds between requests: 5 a minute
SPLIT_GAP = 0.20                           # a bigger "gap" is a stock split, not a trade
STRIKES_TRIED = 4                          # like Daybook: the nearest strikes, first one that traded that day
PUBLISH_EVERY = 3 * 3600


# ---------------------------------------------------------------- Massive

def api_key():
    k = os.environ.get("MASSIVE_API_KEY")
    if not k and KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            if line.strip().startswith("MASSIVE_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
    if not k:
        sys.exit("No Massive API key: put MASSIVE_API_KEY=... in massive_api_key.env")
    return k


_last = [0.0]


def call(url):
    for attempt in range(8):
        time.sleep(max(0.0, _last[0] + PACE - time.time()))
        _last[0] = time.time()
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key()}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60)
                continue
            if e.code in (401, 403):
                return {"results": [], "denied": True}
            if e.code >= 500:
                time.sleep(30)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(30)
    raise RuntimeError("Massive kept failing")


def get(path, params, fetch=True):
    """Cached GET: every page of results, or None when not downloaded yet and fetch is False."""
    q = urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / (hashlib.sha1(f"{path}?{q}".encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    if not fetch:
        return None
    url, results, denied = f"{API}{path}?{q}", [], False
    while url:
        d = call(url)
        denied |= bool(d.get("denied")) or d.get("status") == "NOT_AUTHORIZED"
        results += d.get("results") or []
        url = d.get("next_url")
    out = {"results": results, "denied": denied}
    CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps(out, separators=(",", ":")))
    return out


def frame(results, tz):
    if not results:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    d = pd.DataFrame(results)
    idx = pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert(tz)
    return pd.DataFrame({"Open": d["o"].values, "High": d["h"].values, "Low": d["l"].values, "Close": d["c"].values,
                         "Volume": d["v"].values}, index=pd.DatetimeIndex(idx))


def candles(t, fetch=True):
    """(5m regular-hours candles in CT as traded, daily split-adjusted, daily as traded) or None."""
    parts, a = [], DATA_FROM
    while a <= TEST_TO:
        b = min(a + timedelta(days=179), TEST_TO)
        r = get(f"/v2/aggs/ticker/{t}/range/5/minute/{a}/{b}", {"adjusted": "false", "sort": "asc", "limit": 50000}, fetch)
        if r is None:
            return None
        parts.append(frame(r["results"], "America/Chicago"))
        a = b + timedelta(days=1)
    b5 = pd.concat(parts)
    hm = b5.index.strftime("%H:%M")
    b5 = b5[(hm >= "08:30") & (hm < "15:00")]
    b5 = b5[~b5.index.duplicated()].sort_index()
    daily = []
    for adj in ("true", "false"):
        r = get(f"/v2/aggs/ticker/{t}/range/1/day/{DATA_FROM}/{TEST_TO}", {"adjusted": adj, "sort": "asc", "limit": 50000}, fetch)
        if r is None:
            return None
        d = frame(r["results"], "America/New_York")
        d.index = pd.DatetimeIndex(d.index.tz_localize(None).normalize())
        daily.append(d)
    return b5, daily[0], daily[1]


def contracts(t, exp, lo, hi, fetch=True):
    """Listed contracts for one ticker's weekly expiry (or the day before, for a Friday holiday)."""
    r = get("/v3/reference/options/contracts", {
        "underlying_ticker": t, "expiration_date.gte": str(exp - timedelta(days=1)), "expiration_date.lte": str(exp),
        "strike_price.gte": round(lo * 0.9, 2), "strike_price.lte": round(hi * 1.1, 2),
        "expired": "true" if exp < date.today() else "false", "limit": 1000}, fetch)
    return None if r is None else r["results"]


def option_bars(sym, day, fetch=True):
    r = get(f"/v2/aggs/ticker/O:{sym}/range/1/minute/{day}/{day}", {"adjusted": "false", "sort": "asc", "limit": 50000}, fetch)
    if r is None:
        return None
    b = frame(r["results"], "America/Chicago")
    hm = b.index.strftime("%H:%M")
    return b[(hm >= "08:30") & (hm < "15:00")]


# ---------------------------------------------------------------- the trades

def signals(data):
    """Every gap-and-go entry since TEST_FROM, with its tags and the tracked plans it belongs to (from stock data only)."""
    import scanner as S
    spy = X.day_frame(data["SPY"][0])
    out = []
    for t in TICKERS:
        if t not in data:
            continue
        b5, d_adj, d_raw = data[t]
        x, x15, ctx = X.day_frame(b5), X.fifteen(b5), X.daily_context(d_adj)
        raw_open = {d.date(): float(v) for d, v in d_raw["Open"].items()}
        raw_close = pd.Series(d_raw["Close"].values, index=[d.date() for d in d_raw.index])
        prev_close = raw_close.shift(1)
        for day in sorted(set(x.index.date)):
            if not TEST_FROM <= day <= TEST_TO or day not in ctx.index or day not in raw_open or pd.isna(prev_close.get(day)):
                continue
            c = ctx.loc[day]
            if any(pd.isna(c[k]) for k in ("rv", "atr")):
                continue
            gap = raw_open[day] / float(prev_close[day]) - 1
            if abs(gap) > SPLIT_GAP:
                continue
            for setup, call, at in X.entries(t, x, x15, day, gap):
                if setup != "Gap and go":
                    continue
                prev_bar = at - timedelta(minutes=5)
                if prev_bar not in x.index:
                    continue
                row = x.loc[prev_bar]
                done15 = x15[x15.index + timedelta(minutes=15) <= at]
                sp = spy.loc[prev_bar] if prev_bar in spy.index else None
                tags = X.tags(call, row, done15.iloc[-1] if len(done15) else None, c, sp, gap, 19.0, at)
                plans = [p["id"] for p in T.PLANS if p["setup"] == setup and all(tags.get(g) == l for g, l in p["filters"].items())]
                if not plans:
                    continue
                spot = float(row["Close"])
                est = X.path(call, spot, max(float(c["rv"]) * 1.1, 0.25), S.weekly_dte(day), at,
                             x[(x.index >= at) & (x.index.date == day)])
                out.append({"ticker": t, "day": day, "call": call, "at": at, "spot": spot, "gap": gap, "plans": plans,
                            "exp": day + timedelta(days=S.weekly_dte(day)), "est": est})
    # trades of the filtered plans (#1-#3) first, since they're few, then every other gap-and-go; newest first in each
    out.sort(key=lambda s: (s["plans"] != ["4"], s["day"], s["at"], s["ticker"]), reverse=True)
    return out


def week_ranges(sigs):
    """(ticker, expiry) -> (lowest, highest) entry price, so each week's contract list is one request."""
    rng = {}
    for s in sigs:
        k = (s["ticker"], s["exp"])
        lo, hi = rng.get(k, (s["spot"], s["spot"]))
        rng[k] = (min(lo, s["spot"]), max(hi, s["spot"]))
    return rng


def priced(s, rng, fetch=True):
    """(symbol, 1-minute bars) for one entry: Daybook's contract, the nearest strike that traded that day.
    None when not downloaded yet; (None, None) when there's no such contract."""
    listed = contracts(s["ticker"], s["exp"], *rng[(s["ticker"], s["exp"])], fetch)
    if listed is None:
        return None
    side = "call" if s["call"] else "put"
    mine = [c for c in listed if c.get("contract_type") == side]
    if not mine:
        return None, None
    exp = max(c["expiration_date"] for c in mine)
    mine = sorted((c for c in mine if c["expiration_date"] == exp), key=lambda c: abs(c["strike_price"] - s["spot"]))
    for c in mine[:STRIKES_TRIED]:
        sym = c["ticker"].removeprefix("O:")
        b = option_bars(sym, s["day"], fetch)
        if b is None:
            return None
        if not b.empty:
            return sym, b
    return None, None


# ---------------------------------------------------------------- results

def load_all(fetch):
    data = {}
    for t in TICKERS:
        c = candles(t, fetch)
        if c is not None:
            data[t] = c
    return data


def score(sigs, rng, fetch=False):
    rows, pending, bars = [], 0, {}
    for s in sigs:
        got = priced(s, rng, fetch)
        if got is None:
            pending += 1
            continue
        sym, b = got
        base = {"date": s["day"].isoformat(), "ticker": s["ticker"], "dir": "CALLS" if s["call"] else "PUTS",
                "gap": round(s["gap"] * 100, 2), "stock": round(s["spot"], 2), "symbol": sym, "signal": T.X_hm(s["at"])}
        if b is not None:
            bars.setdefault(base["date"], {})[sym] = b
        for pid in s["plans"]:
            p = next(q for q in T.PLANS if q["id"] == pid)
            real = T.real_exit(b, s["at"], p["exit"]) if b is not None else None
            est = None
            if s["est"] is not None:
                est = round(float(X.outcome(s["est"][0], s["est"][1], s["est"][2], [p["exit"]])[0]), 4)
            rows.append({**base, "plan": pid, **(real or {"no_data": True}), "est_ret": est})
    return rows, pending, bars


def plan_stats(rows):
    out = []
    for p in T.PLANS:
        r = sorted((x for x in rows if x["plan"] == p["id"] and "ret" in x), key=lambda x: (x["date"], x["signal"]))
        pnl = np.array([x["ret"] for x in r], float) * A.TRADE_DOLLARS
        st = X.steady([x["date"] for x in r], pnl)
        months = []
        if len(r):
            m = pd.Series(pnl, index=pd.to_datetime([x["date"] for x in r]).to_period("M"))
            for k, v in m.groupby(level=0):
                months.append({"m": str(k), "n": int(len(v)), "avg": round(float(v.mean()), 1),
                               "win": round(float((v > 0).mean()), 3), "total": round(float(v.sum()))})
        est = [x["est_ret"] for x in r if x.get("est_ret") is not None]
        late = sum(1 for x in r if x["entry_time"] != x["signal"])
        out.append({"id": p["id"], "setup": p["setup"], "filters": p["filters"], "exit": p["exit"], **st,
                    "est_avg": round(float(np.mean(est)) * A.TRADE_DOLLARS, 1) if est and len(est) == len(r) else None,
                    "missing": sum(1 for x in rows if x["plan"] == p["id"] and x.get("no_data")),
                    "late_entries": late, "months": months,
                    "months_up": sum(1 for m in months if m["total"] > 0),
                    **({"best": T.pick(max(r, key=lambda x: x["ret"])), "worst": T.pick(min(r, key=lambda x: x["ret"]))} if r else {})})
    return out


def save_hbars(bars):
    HBARS.mkdir(parents=True, exist_ok=True)
    for day, syms in bars.items():
        f = HBARS / f"{day}.json"
        try:
            saved = json.loads(f.read_text())
        except FileNotFoundError:
            saved = {}
        new = {s: b for s, b in syms.items() if s not in saved}
        if not new:
            continue
        for s, b in new.items():
            saved[s] = {"t": list(b.index.strftime("%H:%M")), **{k: [round(float(v), 2) for v in b[c]] for k, c in
                        (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"))}, "v": [int(v) for v in b["Volume"]]}
        f.write_text(json.dumps(saved, separators=(",", ":")))


def build(sigs, rng):
    rows, pending, bars = score(sigs, rng)
    save_hbars(bars)
    done = sorted({s["day"] for s in sigs} - {s["day"] for s in sigs if priced(s, rng, False) is None})
    all_days = sorted({s["day"] for s in sigs})
    rows.sort(key=lambda r: (r["date"], r["signal"], r["ticker"], r["plan"]), reverse=True)
    out = {"updated": datetime.now().astimezone().isoformat(timespec="minutes"), "source": "Massive",
           "tickers": TICKERS, "test_from": TEST_FROM.isoformat(), "test_to": TEST_TO.isoformat(),
           "days_done": len(done), "days_total": len(all_days), "covered": [done[0].isoformat(), done[-1].isoformat()] if done else None,
           "entries": len(sigs), "pending": pending, "plans": plan_stats(rows),
           "rows": [{k: v for k, v in r.items() if k not in ("high", "low", "pnl")} for r in rows]}
    OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))
    return out


def publish(msg):
    """Commits only the history files and asks the Advisor workflow to redeploy the page."""
    git = lambda *a: subprocess.run(["git", *a], cwd=A.HERE, capture_output=True, text=True)
    git("add", "docs/history.json", "docs/hbars")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return
    git("commit", "-q", "-m", f"history: {msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    for _ in range(3):
        if git("pull", "-q", "--rebase", "-X", "theirs").returncode == 0 and git("push", "-q").returncode == 0:
            subprocess.run(["gh", "workflow", "run", "advisor.yml"], cwd=A.HERE, capture_output=True)
            return
        time.sleep(20)


def log(*a):
    print(datetime.now().strftime("%m-%d %H:%M"), *a, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    fetch = mode == "fetch"
    if fetch:
        api_key()
    data = load_all(fetch)
    if "SPY" not in data:
        sys.exit("No SPY candles yet: run `python history.py fetch`")
    log("stock candles:", len(data), "stocks")
    sigs = signals(data)
    rng = week_ranges(sigs)
    log("gap-and-go entries on tracked plans:", len(sigs))
    if fetch:
        last_pub, day = time.time(), None
        for i, s in enumerate(sigs):
            if s["day"] != day:
                day = s["day"]
                log(f"{day} ({i}/{len(sigs)} entries done)")
            if i and s["plans"] == ["4"] and sigs[i - 1]["plans"] != ["4"]:      # filtered plans done: show them now
                publish(f"plans 1-3: {build(sigs, rng)['days_done']} days")
                last_pub = time.time()
            priced(s, rng, True)
            if time.time() - last_pub > PUBLISH_EVERY:
                out = build(sigs, rng)
                publish(f"{out['days_done']} of {out['days_total']} days")
                last_pub = time.time()
    out = build(sigs, rng)
    log("history:", out["days_done"], "of", out["days_total"], "days;", out["pending"], "entries not downloaded;",
        {p["id"]: (p["n"], p.get("avg"), p.get("win")) for p in out["plans"]})
    if fetch:
        publish(f"{out['days_done']} of {out['days_total']} days")


if __name__ == "__main__":
    main()
