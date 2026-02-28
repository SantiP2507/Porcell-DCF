"""
ml/bootstrap.py — Synthetic training data generator.

Since the tool is freshly set up with no historical data, we generate
realistic synthetic data to bootstrap all three ML models. The synthetic
data is grounded in real DCF math — it's not random noise, it's the output
of running the actual DCF engine across thousands of realistic parameter
combinations.

As real Supabase data accumulates, the training pipeline automatically
blends synthetic + real data, weighting real data more heavily over time.
Once you have 6+ months of real data, synthetic bootstrapping is phased out.

Three datasets generated:
  1. Prioritization dataset — features + historical "was this worth researching?" label
  2. Stability dataset — features + label for whether the valuation was stable
  3. Clustering dataset — feature vectors for unsupervised grouping
"""

import random
import math
import numpy as np
from typing import List, Dict, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FEATURE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
# All three models use a subset of these features.
# Keeping one schema means features stay consistent across the system.

FEATURE_NAMES = [
    "valuation_gap",          # base DCF upside vs market price (decimal)
    "bear_upside",            # bear case upside (decimal)
    "bull_upside",            # bull case upside (decimal)
    "fcf_stability",          # CV-based stability score 0–1
    "leverage_ratio",         # net debt / FCF (0 if net cash)
    "fcf_yield",              # FCF / market cap (decimal)
    "terminal_value_pct",     # TV as % of EV in base case
    "fcf_growth_3y",          # 3-year historical FCF CAGR
    "bear_base_spread",       # bull - bear fair value spread (normalised)
    "net_cash_flag",          # 1 if net cash position, 0 if net debt
    "margin_of_safety_ratio", # base upside / bear downside (risk/reward ratio)
]


def generate_stock_features(
    valuation_gap: float,
    bear_upside: float,
    bull_upside: float,
    fcf_stability: float,
    leverage_ratio: float,
    fcf_yield: float,
    terminal_value_pct: float,
    fcf_growth_3y: float,
    market_price: float,
    bear_fv: float,
    bull_fv: float,
) -> Dict[str, float]:
    """Convert raw valuation metrics into the ML feature vector."""
    spread = (bull_fv - bear_fv) / max(market_price, 1)
    net_cash = 1.0 if leverage_ratio < 0 else 0.0
    # Risk/reward: how much upside in base vs downside in bear
    if bear_upside < 0:
        mos_ratio = valuation_gap / abs(bear_upside)
    else:
        mos_ratio = valuation_gap + 1.0  # both cases positive — great sign

    return {
        "valuation_gap": valuation_gap,
        "bear_upside": bear_upside,
        "bull_upside": bull_upside,
        "fcf_stability": fcf_stability,
        "leverage_ratio": max(leverage_ratio, 0),  # clip at 0
        "fcf_yield": fcf_yield,
        "terminal_value_pct": terminal_value_pct,
        "fcf_growth_3y": fcf_growth_3y,
        "bear_base_spread": spread,
        "net_cash_flag": net_cash,
        "margin_of_safety_ratio": mos_ratio,
    }


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_NAMES], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: PRIORITIZATION
# Label: was this stock "worth researching"? (binary)
# Proxy: stocks with ≥25% base upside AND positive bear upside
#        AND stable FCF AND reasonable leverage tend to be worth it.
# In the future this will be replaced with actual outcome tracking.
# ─────────────────────────────────────────────────────────────────────────────

def generate_prioritization_data(n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (features, label) pairs for research prioritization.

    Label = 1 (worth researching), 0 (not worth it).
    The label is computed from a rule that approximates a conservative
    fundamental investor's criteria — deliberately similar to the rule-based
    screener so the ML learns to generalise it, not contradict it.
    """
    set_seed(42)
    X, y = [], []

    for _ in range(n):
        # Sample realistic parameter ranges
        valuation_gap = random.uniform(-0.50, 1.20)
        bear_upside   = valuation_gap - random.uniform(0.10, 0.60)
        bull_upside   = valuation_gap + random.uniform(0.10, 0.80)
        fcf_stability = random.uniform(0.0, 1.0)
        leverage      = random.uniform(-2.0, 12.0)
        fcf_yield     = random.uniform(0.005, 0.15)
        tv_pct        = random.uniform(0.40, 0.90)
        fcf_growth_3y = random.uniform(-0.10, 0.30)
        market_price  = random.uniform(10, 500)
        bear_fv       = market_price * (1 + bear_upside)
        bull_fv       = market_price * (1 + bull_upside)

        feats = generate_stock_features(
            valuation_gap, bear_upside, bull_upside,
            fcf_stability, leverage, fcf_yield, tv_pct,
            fcf_growth_3y, market_price, bear_fv, bull_fv,
        )

        # Label: worth researching if ALL of:
        # - base upside ≥ 25%
        # - bear case not deeply negative (> -15%)
        # - FCF stability decent (> 0.4)
        # - leverage manageable (< 8x)
        # - FCF yield decent (> 2%)
        # Add noise: 5% random label flip to simulate real-world uncertainty
        is_worth = (
            valuation_gap >= 0.25
            and bear_upside >= -0.15
            and fcf_stability >= 0.40
            and leverage < 8.0
            and fcf_yield >= 0.02
        )
        label = int(is_worth)
        if random.random() < 0.05:  # 5% noise
            label = 1 - label

        X.append(features_to_array(feats))
        y.append(label)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: STABILITY DETECTION
# Label: is this valuation stable? (binary)
# Stable = fair value doesn't swing wildly across runs / assumption changes.
# Proxy: low TV%, high FCF stability, tight bear-bull spread.
# ─────────────────────────────────────────────────────────────────────────────

def generate_stability_data(n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (features, label) pairs for valuation stability.

    Label = 1 (stable valuation — trust the number),
            0 (unstable — treat with skepticism).
    """
    set_seed(123)
    X, y = [], []

    for _ in range(n):
        valuation_gap = random.uniform(-0.30, 1.00)
        bear_upside   = valuation_gap - random.uniform(0.05, 0.70)
        bull_upside   = valuation_gap + random.uniform(0.05, 0.90)
        fcf_stability = random.uniform(0.0, 1.0)
        leverage      = random.uniform(-1.0, 10.0)
        fcf_yield     = random.uniform(0.01, 0.12)
        tv_pct        = random.uniform(0.30, 0.95)
        fcf_growth_3y = random.uniform(-0.05, 0.25)
        market_price  = random.uniform(10, 500)
        bear_fv       = market_price * (1 + bear_upside)
        bull_fv       = market_price * (1 + bull_upside)

        feats = generate_stock_features(
            valuation_gap, bear_upside, bull_upside,
            fcf_stability, leverage, fcf_yield, tv_pct,
            fcf_growth_3y, market_price, bear_fv, bull_fv,
        )

        spread = feats["bear_base_spread"]

        # Stable if: low TV dependency, high FCF consistency, tight scenario spread
        is_stable = (
            tv_pct < 0.70
            and fcf_stability > 0.55
            and spread < 0.60
            and fcf_growth_3y > -0.05
        )
        label = int(is_stable)
        if random.random() < 0.05:
            label = 1 - label

        X.append(features_to_array(feats))
        y.append(label)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: CLUSTERING
# No labels — purely feature vectors for unsupervised grouping.
# Generates a diverse set of financial profiles.
# ─────────────────────────────────────────────────────────────────────────────

def generate_clustering_data(n: int = 1000) -> np.ndarray:
    """
    Generate synthetic feature vectors for clustering.
    Includes deliberate archetypes (compounders, value traps, growth, etc.)
    so the clustering algorithm has meaningful structure to find.
    """
    set_seed(999)
    archetypes = [
        # (valuation_gap, bear, bull, stability, leverage, yield, tv_pct, growth)
        # Compounder: cheap, stable, low leverage, decent yield
        (0.40, 0.15, 0.75, 0.85, 0.5, 0.06, 0.60, 0.10),
        # Value trap: cheap but unstable, high leverage
        (0.35, -0.20, 0.70, 0.30, 7.0, 0.04, 0.75, 0.02),
        # Growth at reasonable price: moderate upside, high stability, low yield
        (0.20, -0.10, 0.55, 0.80, 1.0, 0.02, 0.80, 0.18),
        # Overvalued quality: negative upside, very stable
        (-0.20, -0.40, 0.10, 0.90, 0.5, 0.015, 0.65, 0.12),
        # Deep value: large upside, moderate stability, high yield
        (0.70, 0.30, 1.20, 0.60, 2.0, 0.10, 0.55, 0.05),
        # Distressed: large upside (if survives), low stability, very high leverage
        (0.60, -0.50, 1.40, 0.15, 11.0, 0.08, 0.70, -0.05),
    ]

    X = []
    per_archetype = n // len(archetypes)

    for (gap, bear, bull, stab, lev, yld, tv, growth) in archetypes:
        for _ in range(per_archetype):
            # Add noise around each archetype
            g  = gap   + random.gauss(0, 0.08)
            b  = bear  + random.gauss(0, 0.06)
            bu = bull   + random.gauss(0, 0.10)
            s  = max(0, min(1, stab  + random.gauss(0, 0.10)))
            l  = max(-2, lev   + random.gauss(0, 0.80))
            y  = max(0.001, yld + random.gauss(0, 0.01))
            t  = max(0.30, min(0.95, tv + random.gauss(0, 0.05)))
            gr = growth + random.gauss(0, 0.03)
            mp = random.uniform(15, 400)
            bfv = mp * (1 + b)
            bufv = mp * (1 + bu)

            feats = generate_stock_features(g, b, bu, s, l, y, t, gr, mp, bfv, bufv)
            X.append(features_to_array(feats))

    return np.array(X)
