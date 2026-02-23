"""
valuation/sensitivity.py — Sensitivity analysis engine.

Produces a 2D grid of fair values across combinations of:
  - FCF growth rate (rows)
  - Discount rate / WACC (columns)

This is the most important tool for understanding assumption risk:
  "How much does the fair value change if WACC shifts 1%?"
  "What growth rate does the market currently imply at this price?"

The implied growth rate (market-implied) is also computed via bisection:
  Find g* such that DCF(g*) == market_price.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

from data.models import FinancialSnapshot, MarketData
from valuation.dcf import run_single_scenario_custom
from config import (
    SENSITIVITY_GROWTH_RATES,
    SENSITIVITY_DISCOUNT_RATES,
    DCF_PROJECTION_YEARS,
    DCF_SCENARIOS,
)

logger = logging.getLogger(__name__)

# Type alias: grid[i][j] = fair_value_per_share
SensitivityGrid = List[List[float]]


def compute_sensitivity_grid(
    financials: FinancialSnapshot,
    market_data: MarketData,
    growth_rates: List[float] = None,
    discount_rates: List[float] = None,
    terminal_growth_rate: float = None,
    projection_years: int = None,
) -> Tuple[List[float], List[float], SensitivityGrid]:
    """
    Compute fair value for every (growth_rate, discount_rate) pair.

    Returns:
        (growth_rates, discount_rates, grid)
        where grid[i][j] = fair_value at growth_rates[i], discount_rates[j]
    """
    growth_rates = growth_rates or SENSITIVITY_GROWTH_RATES
    discount_rates = discount_rates or SENSITIVITY_DISCOUNT_RATES
    terminal_growth_rate = terminal_growth_rate or DCF_SCENARIOS["base"]["terminal_growth_rate"]
    projection_years = projection_years or DCF_PROJECTION_YEARS

    grid = []
    for g in growth_rates:
        row = []
        for r in discount_rates:
            try:
                result = run_single_scenario_custom(
                    financials=financials,
                    market_data=market_data,
                    fcf_growth_rate=g,
                    discount_rate=r,
                    terminal_growth_rate=min(terminal_growth_rate, r - 0.011),
                    projection_years=projection_years,
                    scenario_name="sensitivity",
                )
                val = result.fair_value_per_share if result.is_valid else float("nan")
            except Exception as e:
                logger.debug(f"Sensitivity cell failed g={g:.2%} r={r:.2%}: {e}")
                val = float("nan")
            row.append(val)
        grid.append(row)

    return growth_rates, discount_rates, grid


def compute_implied_growth_rate(
    financials: FinancialSnapshot,
    market_data: MarketData,
    discount_rate: float = None,
    terminal_growth_rate: float = None,
    projection_years: int = None,
) -> Optional[float]:
    """
    Find the FCF growth rate that makes the DCF fair value == market price.

    This is the 'market-implied' growth rate — useful for asking:
    "Is the market's implied expectation reasonable?"

    Uses bisection search (binary search on growth rate).

    Returns:
        The implied growth rate as a decimal, or None if no solution found.
    """
    discount_rate = discount_rate or DCF_SCENARIOS["base"]["discount_rate"]
    terminal_growth_rate = terminal_growth_rate or DCF_SCENARIOS["base"]["terminal_growth_rate"]
    projection_years = projection_years or DCF_PROJECTION_YEARS
    target_price = market_data.price

    def dcf_at_growth(g: float) -> float:
        try:
            result = run_single_scenario_custom(
                financials=financials,
                market_data=market_data,
                fcf_growth_rate=g,
                discount_rate=discount_rate,
                terminal_growth_rate=min(terminal_growth_rate, discount_rate - 0.011),
                projection_years=projection_years,
                scenario_name="implied",
            )
            return result.fair_value_per_share if result.is_valid else 0.0
        except Exception:
            return 0.0

    # Binary search in [−10%, 50%] growth range
    lo, hi = -0.10, 0.50
    lo_val = dcf_at_growth(lo) - target_price
    hi_val = dcf_at_growth(hi) - target_price

    # If market price is above even the bull DCF at 50% growth → implied growth > 50%
    if hi_val < 0:
        logger.info("Market price implies growth > 50% — DCF likely not applicable.")
        return None

    # If market price is below even the bear DCF at -10% growth → stock is extreme discount
    if lo_val > 0:
        return -0.10

    # Bisection
    for _ in range(50):
        mid = (lo + hi) / 2
        mid_val = dcf_at_growth(mid) - target_price
        if abs(mid_val) < 0.01:  # within 1 cent
            return mid
        if mid_val < 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def grid_to_dataframe(
    growth_rates: List[float],
    discount_rates: List[float],
    grid: SensitivityGrid,
    market_price: float,
):
    """
    Convert the raw grid to a pandas DataFrame for display / charting.
    Columns = discount rates, rows = growth rates.
    Values are formatted as upside % vs market price.
    """
    import pandas as pd

    col_labels = [f"{r:.0%}" for r in discount_rates]
    row_labels = [f"{g:.0%}" for g in growth_rates]

    data = {}
    for j, col in enumerate(col_labels):
        data[col] = []
        for i in range(len(growth_rates)):
            fv = grid[i][j]
            if np.isnan(fv):
                data[col].append(np.nan)
            else:
                upside = (fv - market_price) / market_price
                data[col].append(round(upside, 4))  # decimal, e.g. 0.25 = 25%

    df = pd.DataFrame(data, index=row_labels)
    df.index.name = "FCF Growth \\ WACC"
    return df
