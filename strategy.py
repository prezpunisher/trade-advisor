"""Loads strategy.json: the small account's settings, shared by small_signals.py (the live bot's signals),
cheap_test.py (2-year test and paper tracking) and the Daybook config page. STRATEGY_FILE overrides the path
(the backtest workflow points it at a draft)."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = Path(os.environ.get("STRATEGY_FILE", HERE / "strategy.json"))


def load(path=None):
    s = json.loads(Path(path or PATH).read_text())
    for p in s["plans"]:
        p.setdefault("filters", {})
        p["exit"].setdefault("lock", None)
    return s


def plans(s, on_only=True):
    """{id: plan} in priority order."""
    by = {p["id"]: p for p in s["plans"] if p.get("on", True) or not on_only}
    return {pid: by[pid] for pid in s["order"] if pid in by}


def gap_plans(s):
    return [p for p in plans(s).values() if p["setup"] == "Gap and go"]
