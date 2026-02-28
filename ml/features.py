"""
ml/features.py — Extract ML feature vectors from ValuationSummary objects.

This is the bridge between the DCF engine and the ML models.
Every ValuationSummary that comes out of the DCF engine can be converted
into a fixed-length numeric feature vector suitable for sklearn.

Keeping feature extraction in one place means:
  - All three models use identical features
  - Adding a new feature only requires changing this file
  - Features are interpretable (no black-box transformations)
"""

import statistics
import numpy as np
from typing import Optional, Dict

from data.models import ValuationSummary
from ml.bootstrap import FEATURE_NAMES, generate_stock_features, features_to_array


def extract_features(summary: ValuationSummary) -> Optional[np.ndarray]:
    """
    Convert a ValuationSummary into a feature vector for ML models.

    Returns None if the summary is missing critical data (e.g. no base case).
    """
    if summary.base is None or not summary.base.is_valid:
        return None
    if summary.financials is None or summary.market_data is None:
        return None

    f = summary.financials
    m = summary.market_data

    # ── Valuation gap metrics ─────────────────────────────────────────────
    valuation_gap = summary.upside_base or 0.0
    bear_upside   = summary.upside_bear or 0.0
    bull_upside   = summary.upside_bull or 0.0

    # ── FCF stability (coefficient of variation, inverted) ────────────────
    hist = f.historical_fcf
    if len(hist) >= 2 and statistics.mean(hist) > 0:
        cv = statistics.stdev(hist) / abs(statistics.mean(hist))
        fcf_stability = max(0.0, 1.0 - cv)
    else:
        fcf_stability = 0.5  # unknown — neutral

    # ── Leverage ──────────────────────────────────────────────────────────
    leverage_ratio = (
        f.net_debt / f.trailing_fcf
        if f.trailing_fcf > 0
        else 0.0
    )

    # ── FCF yield ─────────────────────────────────────────────────────────
    fcf_yield = (
        f.trailing_fcf / m.market_cap
        if m.market_cap > 0
        else 0.0
    )

    # ── Terminal value % (base case) ──────────────────────────────────────
    terminal_value_pct = summary.base.terminal_value_pct

    # ── 3-year FCF CAGR ───────────────────────────────────────────────────
    if len(hist) >= 4:
        start = hist[-4]
        end   = hist[-1]
        if start > 0 and end > 0:
            fcf_growth_3y = (end / start) ** (1 / 3) - 1
        else:
            fcf_growth_3y = 0.0
    else:
        fcf_growth_3y = 0.0

    # ── Bear/bull fair values for spread calculation ───────────────────────
    bear_fv = summary.bear.fair_value_per_share if summary.bear and summary.bear.is_valid else m.price * (1 + bear_upside)
    bull_fv = summary.bull.fair_value_per_share if summary.bull and summary.bull.is_valid else m.price * (1 + bull_upside)

    feats = generate_stock_features(
        valuation_gap   = valuation_gap,
        bear_upside     = bear_upside,
        bull_upside     = bull_upside,
        fcf_stability   = fcf_stability,
        leverage_ratio  = leverage_ratio,
        fcf_yield       = fcf_yield,
        terminal_value_pct = terminal_value_pct,
        fcf_growth_3y   = fcf_growth_3y,
        market_price    = m.price,
        bear_fv         = bear_fv,
        bull_fv         = bull_fv,
    )

    return features_to_array(feats)


def extract_features_dict(summary: ValuationSummary) -> Optional[Dict[str, float]]:
    """Return features as a named dict (for logging/debugging)."""
    arr = extract_features(summary)
    if arr is None:
        return None
    return dict(zip(FEATURE_NAMES, arr))
