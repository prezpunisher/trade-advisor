#!/usr/bin/env python3
"""Jobs for the Automated Config page (Daybook config.html). Run by .github/workflows/config.yml from an issue the repo
owner opens through the page's buttons. Never trades; only changes strategy.json on "Make live".

  Backtest config   issue body holds a ```json``` strategy draft -> validated, replayed with cheap_test.py on 2 years of
                    real option prices -> docs/backtests/<issue>.json + docs/backtests/index.json
  Make live         issue title "Make live: backtest #N" -> that backtest's config becomes strategy.json (only if it fits
                    the live limits), then the main test in docs/cheap.json is rerun with it

  ISSUE_NUMBER=12 ISSUE_TITLE='Backtest config: tighter stop' ISSUE_BODY="$(cat body.md)" python config_jobs.py
Writes the issue reply to comment.md. Exits 0 even when it rejects a config (the reply says why).
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "docs" / "backtests"
OPTIONS = json.loads((HERE / "config_options.json").read_text())
COMMENT = HERE / "comment.md"
PLAN_ID = re.compile(r"^[A-Za-z0-9_-]{1,12}$")
TIME = re.compile(r"^(0[89]|1[0-4]):[0-5]\d$")


def reply(text):
    COMMENT.write_text(text)
    print(text)


def number(v, lo, hi, name, errors, allow_none=False):
    if v is None and allow_none:
        return
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
        errors.append(f"`{name}` must be a number from {lo} to {hi} (got {v!r})")


def validate(cfg, live):
    """List of problems; empty = fine. live=True also checks the bot's limits (guard.py)."""
    e = []
    if not isinstance(cfg, dict):
        return ["the config must be a JSON object"]
    lim = OPTIONS["live_limits"] if live else OPTIONS["test_limits"]
    acc = cfg.get("account") or {}
    number(acc.get("max_price"), 0.05, lim["max_price"], "account.max_price", e)
    number(acc.get("min_price"), 0.01, 1.0, "account.min_price", e)
    number(acc.get("trades_per_day"), 1, lim["trades_per_day"], "account.trades_per_day", e)
    number(acc.get("daily_budget"), 10, lim["daily_budget"], "account.daily_budget", e)
    if acc.get("contracts") != 1:
        e.append("`account.contracts` must be 1 (the guard allows 1 contract per order)")
    number((cfg.get("test") or {}).get("slip"), 0, 0.10, "test.slip", e)
    plans = cfg.get("plans")
    if not isinstance(plans, list) or not plans:
        return e + ["`plans` must be a non-empty list"]
    ids = []
    for p in plans:
        pid = str(p.get("id"))
        if not PLAN_ID.match(pid) or pid in ids:
            e.append(f"plan id {pid!r} must be 1-12 letters, digits, - or _, and unique")
        ids.append(pid)
        if p.get("setup") not in OPTIONS["setups"]:
            e.append(f"plan {pid}: setup must be one of {OPTIONS['setups']}")
        if p.get("setup") == "Daybook pick, all three 15m agree" and pid != "A3":
            e.append(f"plan {pid}: the Daybook-pick setup must keep the id A3")
        for g, v in (p.get("filters") or {}).items():
            allowed = OPTIONS["filters"].get(g)
            vals = v if isinstance(v, list) else [v]
            if allowed is None or not vals or any(x not in allowed for x in vals):
                e.append(f"plan {pid}: filter {g!r} = {v!r} isn't one of the page's choices")
        ex = p.get("exit") or {}
        number(ex.get("stop"), 0.05, 0.9, f"plan {pid} exit.stop", e, allow_none=not live)
        number(ex.get("target"), 0.02, 5.0, f"plan {pid} exit.target", e, allow_none=True)
        if not TIME.match(str(ex.get("cut"))):
            e.append(f"plan {pid}: exit.cut must be a time from 08:00 to 14:59, like 14:50")
        if live and ex.get("target") is not None:
            e.append(f"plan {pid}: a profit target can be backtested but not made live: Robinhood can't hold a target "
                     "and a stop on the same option (no OCO), and the bot keeps the stop")
        if live and ex.get("cut") != lim["cut"]:
            e.append(f"plan {pid}: exit.cut must be {lim['cut']} to go live (the bot's close routine runs then)")
    order = cfg.get("order")
    if not isinstance(order, list) or sorted(map(str, order)) != sorted(ids):
        e.append("`order` must list every plan id exactly once")
    if not any(p.get("on", True) for p in plans):
        e.append("at least one plan must be on")
    tickers = cfg.get("tickers")
    if not isinstance(tickers, list) or not tickers or not all(isinstance(t, str) and re.match(r"^[A-Z.]{1,6}$", t) for t in tickers):
        e.append("`tickers` must be a list of stock symbols like SPY")
    elif live:
        known = set(json.loads((HERE / "strategy.json").read_text())["tickers"])
        new = [t for t in tickers if t not in known]
        if new:
            e.append(f"new tickers {new} can be backtested, but going live with them waits for history.py to cache "
                     "them (ask Claude to add them)")
    return e


def config_from_body(body):
    m = re.search(r"```json\s*(\{.*?\})\s*```", body or "", re.S)
    if not m:
        raise ValueError("no ```json block found in the issue")
    return json.loads(m.group(1))


def run_test(cfg_path, out_path):
    env = {**os.environ, "STRATEGY_FILE": str(cfg_path), "CHEAP_OUT": str(out_path), "CHEAP_FETCH": "1"}
    subprocess.run([sys.executable, "cheap_test.py", "test"], cwd=HERE, env=env, check=True)


def row_line(s):
    if not s.get("n"):
        return f"| {s['id']} | 0 | | | | |"
    return (f"| {s['id']} | {s['n']} | {round(s['win'] * 100)}% | {s['avg']:+.2f} | {s['total']:+.0f} | "
            f"-{s['max_drawdown']:.0f} |")


def table(summary):
    return "\n".join(["| Plan | Trades | Won | Avg / trade | Total | Worst drop |", "|---|---|---|---|---|---|"]
                     + [row_line(s) for s in summary])


def index_add(entry):
    f = OUT / "index.json"
    idx = json.loads(f.read_text()) if f.exists() else {"runs": []}
    idx["runs"] = [r for r in idx["runs"] if r["id"] != entry["id"]] + [entry]
    idx["runs"].sort(key=lambda r: r["id"], reverse=True)
    f.write_text(json.dumps(idx, indent=1))


def backtest(n, title, body):
    try:
        cfg = config_from_body(body)
    except (ValueError, json.JSONDecodeError) as ex:
        return reply(f"Couldn't read the config: {ex}. Use the page's **Run backtest** button.")
    errors = validate(cfg, live=False)
    if errors:
        return reply("Not run. Fix these and try again:\n\n" + "\n".join(f"- {x}" for x in errors))
    OUT.mkdir(parents=True, exist_ok=True)
    draft = HERE / f"draft-{n}.json"
    draft.write_text(json.dumps(cfg, indent=1))
    result = HERE / f"result-{n}.json"
    try:
        run_test(draft, result)
    finally:
        draft.unlink(missing_ok=True)
    res = json.loads(result.read_text())
    result.unlink(missing_ok=True)
    test = res["test"]
    name = title.split(":", 1)[1].strip() if ":" in title else f"Backtest #{n}"
    live_problems = validate(cfg, live=True)
    rec = {"id": n, "name": name, "created": datetime.now().isoformat(timespec="minutes"), "config": cfg,
           "summary": test["summary"], "skipped": test["skipped"], "rows": test["rows"],
           "can_go_live": not live_problems, "live_problems": live_problems}
    (OUT / f"{n}.json").write_text(json.dumps(rec, separators=(",", ":")))
    bot = next(s for s in test["summary"] if s["id"] == "bot")
    index_add({k: rec[k] for k in ("id", "name", "created", "can_go_live")} | {"bot": {k: bot.get(k) for k in (
        "n", "win", "avg", "total", "max_drawdown", "loss_streak")}})
    msg = [f"Backtest #{n} done ({name}): 2 years of real option prices.", "", table(test["summary"]), ""]
    msg.append("It can go live: use **Make live** on the Automated Config page." if not live_problems else
               "It can't go live as it is:\n\n" + "\n".join(f"- {x}" for x in live_problems))
    reply("\n".join(msg))


def make_live(n):
    f = OUT / f"{n}.json"
    if not f.exists():
        return reply(f"There's no backtest #{n}. Run a backtest first.")
    rec = json.loads(f.read_text())
    cfg = rec["config"]
    errors = validate(cfg, live=True)
    if errors:
        return reply("Not made live:\n\n" + "\n".join(f"- {x}" for x in errors))
    cur = json.loads((HERE / "strategy.json").read_text())
    cfg.update({"version": int(cur.get("version", 0)) + 1, "status": "live",
                "updated": datetime.now().strftime("%Y-%m-%d"), "from_backtest": n})
    (HERE / "strategy.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    run_test(HERE / "strategy.json", HERE / "docs" / "cheap.json")       # the Small account page follows the new rules
    rec["live_since"] = cfg["updated"]
    f.write_text(json.dumps(rec, separators=(",", ":")))
    reply(f"Backtest #{n} ({rec['name']}) is now live as strategy version {cfg['version']}. The bot trades it from the "
          "next 8:46 run; the Small account page's test now uses it.")


def main():
    n = int(os.environ["ISSUE_NUMBER"])
    title = os.environ.get("ISSUE_TITLE", "")
    if title.startswith("Backtest config"):
        backtest(n, title, os.environ.get("ISSUE_BODY", ""))
    elif m := re.match(r"Make live: backtest #(\d+)$", title):
        make_live(int(m.group(1)))
    else:
        reply("This issue title isn't a config job.")


if __name__ == "__main__":
    main()
