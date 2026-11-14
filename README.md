# Trade Advisor

Studies everything [Daybook](https://github.com/prezpunisher/daybook) has collected and suggests what could work:
the best entry conditions and exit, what helps and hurts, and which stocks to trade, watch or skip.
`explore.py` looks for better plans: 8 ways in, 15 conditions and 539 exits on 60 days of the stocks Daybook
shows, picked on older days and checked on newer ones (`docs/explore.json`). `tracker.py` follows the top plans on real 1-minute option prices each weekday (`docs/tracking.json`). Suggestions only: it reads Daybook's public data and code, never changes them, sends no pushes and places no trades.

Page: https://prezpunisher.github.io/trade-advisor/

- `advisor.py` writes `docs/advisor.json`; `docs/index.html` shows it.
- Runs after each weekday close (`.github/workflows/advisor.yml`), by hand, or from the page's Refresh button.
- Local run: `DAYBOOK_DIR=../options-scanner python advisor.py`; tests: `python -m unittest discover -s tests`.
