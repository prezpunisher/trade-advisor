#!/usr/bin/env python3
"""Archive: saves each day's real 1-minute option prices near the money for every stock explore.py studies, so later
backtests can use real option prices instead of estimates. Yahoo drops a weekly contract's prices within days of
expiry, so this has to be saved as it happens.

Per stock and day: this week's expiry (next week's on Thu/Fri, like Daybook), calls and puts, the 3 strikes closest
to the stock's open. Saved gzipped in archive/<YYYY-MM-DD>.json.gz.

  python archive.py              # saves any recent trading day not saved yet
"""
import gzip
import json
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import advisor_config as A
import explore as X

sys.path.insert(0, str(A.DAYBOOK))

DIR = A.HERE / "archive"
STRIKES = 3
LOOKBACK = 12                     # calendar days looked back; Yahoo keeps a contract until about a week past expiry


def day_contracts(t, day, spot):
    import scanner as S
    import strategy_15m as M
    import yfinance as yf
    exp = day + timedelta(days=S.weekly_dte(day))
    try:
        ch = yf.Ticker(t).option_chain(exp.isoformat())
        listed = {True: list(ch.calls["strike"]), False: list(ch.puts["strike"])}
    except Exception:
        guess = sorted({round(spot / s) * s for s in (0.5, 1, 2.5, 5, 10)})
        listed = {True: guess, False: guess}
    return [M.occ(t, exp, call, k) for call in (True, False)
            for k in sorted(listed[call], key=lambda k: abs(k - spot))[:STRIKES]]


def save_day(day, tickers, opens):
    import scanner as S
    import yfinance as yf
    out = {}
    for t in tickers:
        if t not in opens:
            continue
        for sym in day_contracts(t, day, opens[t]):
            try:
                h = yf.Ticker(sym).history(start=day, end=day + timedelta(days=1), interval="1m")
            except Exception:
                continue
            if h.empty:
                continue
            h.index = h.index.tz_convert(S.CT)
            out[sym] = {"t": list(h.index.strftime("%H:%M")), **{k: [round(float(v), 2) for v in h[c]] for k, c in
                        (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"))}, "v": [int(v) for v in h["Volume"]]}
    DIR.mkdir(exist_ok=True)                   # an empty file marks a day with nothing to save, so it isn't retried
    if True:
        with gzip.open(DIR / f"{day.isoformat()}.json.gz", "wt") as f:
            json.dump(out, f, separators=(",", ":"))
    return len(out)


def main():
    import scanner as S
    import strategy_15m as M
    tickers, _ = X.universe()                 # the no-trade list is archived too: cheap, and it may change
    tickers = sorted(set(tickers) | set(json.loads((A.DAYBOOK / "docs" / "no_trade.json").read_text()).get("tickers", [])))
    daily = M.candles(tickers, "1mo", "1d")
    now = datetime.now(S.CT)
    last = now.date() if now.strftime("%H:%M") >= "15:15" else now.date() - timedelta(days=1)
    for back in range(LOOKBACK):
        day = last - timedelta(days=back)
        if day.weekday() > 4 or (DIR / f"{day.isoformat()}.json.gz").exists():
            continue
        opens = {t: float(df["Open"][df.index.date == day].iloc[0]) for t, df in daily.items() if (df.index.date == day).any()}
        if not opens:
            continue                           # market closed
        print(f"archive {day}: {save_day(day, tickers, opens)} contracts")


if __name__ == "__main__":
    main()
