"""
valuation/scenarios.py — Bear/base/bull scenario orchestration.

Thin wrapper around dcf.py that:
  1. Accepts optional per-scenario overrides (e.g. custom growth rates).
  2. Merges overrides with config defaults.
  3. Runs all three scenarios and returns the summary.

Useful for interactive analysis where you want to ask:
  "What if base growth is 10% instead of 7%?"
"""

import copy
import logging
from typing import Optional, Dict, Any

from data.models import FinancialSnapshot, MarketData, ValuationSummary
from valuation.dcf import run_full_valuation
from config import DCF_SCENARIOS, DCF_PROJECTION_YEARS

logger = logging.getLogger(__name__)


def build_scenarios(
    financials: FinancialSnapshot,
    market_data: MarketData,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    projection_years: int = DCF_PROJECTION_YEARS,
) -> ValuationSummary:
    """
    Build bear/base/bull ValuationSummary with optional per-scenario overrides.

    Args:
        financials:      FinancialSnapshot
        market_data:     MarketData
        overrides:       Dict like {"base": {"fcf_growth_rate": 0.10}}
                         — merged on top of config defaults.
        projection_years: Override default horizon.

    Returns:
        ValuationSummary

    Example:
        summary = build_scenarios(
            financials, market_data,
            overrides={"base": {"fcf_growth_rate": 0.10, "discount_rate": 0.09}}
        )
    """
    # Deep copy to avoid mutating the global config
    scenarios = copy.deepcopy(DCF_SCENARIOS)

    if overrides:
        for scenario_name, params in overrides.items():
            if scenario_name not in scenarios:
                logger.warning(
                    f"Unknown scenario '{scenario_name}' in overrides — skipping."
                )
                continue
            scenarios[scenario_name].update(params)
            logger.info(
                f"Applied overrides to '{scenario_name}' scenario: {params}"
            )

    return run_full_valuation(
        financials=financials,
        market_data=market_data,
        scenarios=scenarios,
        projection_years=projection_years,
    )


def summarize_scenarios(summary: ValuationSummary) -> str:
    """
    Return a compact text summary of all three scenarios vs market price.
    """
    price = summary.market_price
    lines = [
        f"\n{'═'*60}",
        f"  VALUATION SUMMARY: {summary.ticker}",
        f"  Market Price: ${price:.2f}",
        f"{'═'*60}",
        f"  {'Scenario':<10} {'Fair Value':>12} {'Upside':>10} {'TV%':>8}",
        f"  {'─'*44}",
    ]

    for label, result in [
        ("Bear", summary.bear),
        ("Base", summary.base),
        ("Bull", summary.bull),
    ]:
        if result and result.is_valid:
            upside = (result.fair_value_per_share - price) / price
            tv_pct = result.terminal_value_pct
            upside_str = f"{upside:+.1%}"
            lines.append(
                f"  {label:<10} ${result.fair_value_per_share:>10.2f} "
                f"{upside_str:>10} {tv_pct:>7.0%}"
            )
        else:
            reason = result.warning if result else "Not computed"
            lines.append(f"  {label:<10} {'N/A':>12} {'—':>10}  [{reason[:30]}]")

    lines.append(f"{'═'*60}\n")
    return "\n".join(lines)
