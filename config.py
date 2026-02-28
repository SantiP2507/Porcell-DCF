"""
config.py — Global settings and default assumptions.

Single source of truth for all configuration. Change here → propagates everywhere.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CACHE_DIR       = BASE_DIR / ".cache"
CHART_OUTPUT_DIR = BASE_DIR / "charts_output"

CACHE_DIR.mkdir(exist_ok=True)
CHART_OUTPUT_DIR.mkdir(exist_ok=True)

# ── Supabase ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Alpha Vantage (fallback data source) ───────────────────────────────────
ALPHA_VANTAGE_KEY      = os.getenv("ALPHA_VANTAGE_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# ── Data ───────────────────────────────────────────────────────────────────
HISTORY_YEARS     = 5
CACHE_TTL_SECONDS = 60 * 60 * 4  # 4 hours

# ── DCF Defaults ───────────────────────────────────────────────────────────
DCF_PROJECTION_YEARS = 5
MAX_TERMINAL_VALUE_PCT = 0.80  # Warn if TV > 80% of EV — model becomes unreliable

# Default scenarios: full parameter sets used by the DCF engine.
# fcf_growth_rate:      Annual FCF growth during the projection period.
# discount_rate:        Required return / WACC.
# terminal_growth_rate: Perpetual growth rate after projection. MUST be < discount_rate.
# fcf_haircut:          Fraction to reduce TTM FCF before projecting (bear case prudence).
DCF_SCENARIOS = {
    "bear": {
        "fcf_growth_rate":     0.03,   # Minimal growth — margin pressure, competition
        "discount_rate":       0.12,   # Higher uncertainty premium
        "terminal_growth_rate": 0.02,
        "fcf_haircut":         0.10,   # 10% haircut on trailing FCF for cyclical risk
    },
    "base": {
        "fcf_growth_rate":     0.07,   # Moderate growth — trend continuation
        "discount_rate":       0.10,   # Standard WACC proxy
        "terminal_growth_rate": 0.025,
        "fcf_haircut":         0.00,
    },
    "bull": {
        "fcf_growth_rate":     0.12,   # Strong growth — market share gains, margins expand
        "discount_rate":       0.09,   # Lower risk premium in a strong execution environment
        "terminal_growth_rate": 0.03,
        "fcf_haircut":         0.00,
    },
}

# ── Sensitivity Grid ───────────────────────────────────────────────────────
# These are absolute values (not multipliers).
SENSITIVITY_GROWTH_RATES   = [-0.05, 0.00, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
SENSITIVITY_DISCOUNT_RATES = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]

# ── Screener Universe ──────────────────────────────────────────────────────
DEFAULT_SCREEN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "BRK-B", "JPM", "V", "MA",
    "UNH", "LLY", "JNJ", "PG", "KO",
    "HD", "COST", "ABBV", "MRK", "CVX",
    "WMT", "BAC", "XOM", "PFE", "AVGO",
]

SCREENER_TOP_N      = 10   # Number of candidates to surface per daily screen
MAX_LEVERAGE_RATIO  = 8.0  # Net Debt / TTM FCF — above this is too risky to screen in
MIN_FCF_YIELD       = 0.01 # Minimum 1% FCF yield (FCF / market cap) to qualify

# Minimum margin of safety: price must be < base fair value × this
MIN_MARGIN_OF_SAFETY = 0.85

# Rule-based scoring weights (sum to 1.0)
SCORING_WEIGHTS = {
    "valuation_gap":      0.35,
    "fcf_stability":      0.25,
    "leverage":           0.20,
    "valuation_coverage": 0.20,
}

# ── Chart Settings ─────────────────────────────────────────────────────────
CHART_STYLE        = "seaborn-v0_8-whitegrid"
CHART_DPI          = 150
CHART_FIGSIZE_WIDE = (14, 7)
CHART_FIGSIZE_SQUARE = (10, 8)
CHART_ACCENT_COLOR = "#2563EB"   # Blue
CHART_BASE_COLOR   = "#16A34A"   # Green
CHART_BEAR_COLOR   = "#DC2626"   # Red
CHART_BULL_COLOR   = "#9333EA"   # Purple
