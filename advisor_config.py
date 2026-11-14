"""Advisor settings. The Advisor only reads Daybook's code and data; it never changes them, sends pushes or trades."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAYBOOK = Path(os.environ.get("DAYBOOK_DIR", HERE.parent / "options-scanner")).resolve()   # read-only checkout
OUT = HERE / "docs" / "advisor.json"

DAYS = 60                         # Yahoo keeps about 60 days of 15m and 5m candles
TRADE_DOLLARS = 1000              # same $1,000 per trade as Daybook
SPREAD_PCT = 0.04                 # estimated option prices: full bid/ask spread, half paid each side
SLIPPAGE_PCT = 0.005              # per side
FEE_PER_CONTRACT = 0.05           # Robinhood: $0 commission, ~4-5c of regulatory fees per contract per side

# Daybook's research tickers are always tested; these are extra liquid weekly-option names to look for new ideas
EXTRA_TICKERS = ["COIN", "MSTR", "UBER", "BA", "DIS", "CRM", "ADBE", "SHOP", "SMCI", "MU", "ARM", "SOFI", "BAC",
                 "XOM", "COST", "LLY", "UNH", "V", "MA", "PYPL", "SNOW", "ABNB", "DKNG", "RIVN", "NKE", "SBUX",
                 "TGT", "CVX", "GS", "C"]

# exits compared (None = no target / no stop: hold to 2:50 CT)
TARGETS = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, None]
STOPS = [0.10, 0.15, 0.20, 0.30, 0.50, None]
CURRENT = (0.03, 0.20)            # Daybook's plan: +3% / -20%

# plan search: tuned on the older days, checked on the newest days it never saw
TEST_SHARE = 1 / 3
MIN_TRAIN = 25                    # trades a plan needs on the tuning days
MIN_TEST = 8                      # trades it needs on the check days to count as proven
MIN_PATTERN = 20                  # trades a pattern needs before it's shown
MIN_TICKER = 8                    # estimated trades a ticker needs for a Trade/Skip call
MIN_TICKER_REAL = 4               # real picks that can overrule the estimate
