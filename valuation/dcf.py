"""
valuation/dcf.py — Core Discounted Cash Flow engine.

This module implements the mathematical heart of the system:
  1. Apply FCF haircut to get base FCF for year 1.
  2. Project FCF forward N years using a constant growth rate.
  3. Discount each projected FCF back to present value.
  4. Compute terminal value using the Gordon Growth Model.
  5. Subtract net debt to get equity value.
  6. Divide by shares outstanding to get fair value per share.

All assumptions are explicit — no hidden defaults.

Gordon Growth Model for terminal value:
    TV = FCF_N+1 / (discount_rate - terminal_growth_rate)
    where FCF_N+1 = FCF_N * (1 + terminal_growth_rate)

This model breaks down if terminal_growth_rate >= discount_rate (TV → ∞ or negative).
We enforce terminal_growth_rate < discount_rate − 0.01 as a hard constraint.
"""

import logging
from typing import Dict

from data.models import (
    FinancialSnapshot,
    MarketData,
    DCFResult,
    ValuationSummary,
)
from config import DCF_SCENARIOS, DCF_PROJECTION_YEARS, MAX_TERMINAL_VALUE_PCT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_full_valuation(
    financials: FinancialSnapshot,
    market_data: MarketData,
    scenarios: Dict = None,
    projection_years: int = None,
) -> ValuationSummary:
    """
    Run bear, base, and bull DCF scenarios and return a ValuationSummary.

    Args:
        financials:       FinancialSnapshot with FCF, debt, cash data.
        market_data:      Current price and shares outstanding.
        scenarios:        Optional custom scenario dict. Defaults to config.DCF_SCENARIOS.
        projection_years: Optional override for projection horizon.

    Returns:
        ValuationSummary with all three scenarios populated.
    """
    scenarios = scenarios or DCF_SCENARIOS
    projection_years = projection_years or DCF_PROJECTION_YEARS

    results = {}
    for scenario_name, params in scenarios.items():
        result = _run_single_scenario(
            ticker=financials.ticker,
            scenario_name=scenario_name,
            trailing_fcf=financials.trailing_fcf,
            net_debt=financials.net_debt,
            shares_outstanding=market_data.shares_outstanding,
            fcf_growth_rate=params["fcf_growth_rate"],
            discount_rate=params["discount_rate"],
            terminal_growth_rate=params["terminal_growth_rate"],
            fcf_haircut=params.get("fcf_haircut", 0.0),
            projection_years=projection_years,
        )
        results[scenario_name] = result

    return ValuationSummary(
        ticker=financials.ticker,
        market_price=market_data.price,
        shares_outstanding=market_data.shares_outstanding,
        bear=results.get("bear"),
        base=results.get("base"),
        bull=results.get("bull"),
        financials=financials,
        market_data=market_data,
    )


def run_single_scenario_custom(
    financials: FinancialSnapshot,
    market_data: MarketData,
    fcf_growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    fcf_haircut: float = 0.0,
    projection_years: int = None,
    scenario_name: str = "custom",
) -> DCFResult:
    """
    Run a single DCF with fully custom parameters.
    Used by the sensitivity analysis engine.
    """
    return _run_single_scenario(
        ticker=financials.ticker,
        scenario_name=scenario_name,
        trailing_fcf=financials.trailing_fcf,
        net_debt=financials.net_debt,
        shares_outstanding=market_data.shares_outstanding,
        fcf_growth_rate=fcf_growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        fcf_haircut=fcf_haircut,
        projection_years=projection_years or DCF_PROJECTION_YEARS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE DCF MATH
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_scenario(
    ticker: str,
    scenario_name: str,
    trailing_fcf: float,       # USD millions, TTM
    net_debt: float,           # USD millions (negative = net cash)
    shares_outstanding: float, # millions
    fcf_growth_rate: float,    # annual FCF growth during projection period
    discount_rate: float,      # WACC / required return
    terminal_growth_rate: float,  # perpetual growth after projection
    fcf_haircut: float,        # fraction to reduce year-1 FCF (bear case prudence)
    projection_years: int,
) -> DCFResult:
    """Internal: run a single DCF scenario and return a DCFResult."""

    # ── Safety checks ────────────────────────────────────────────────────
    warning = None

    # Hard constraint: terminal growth must be materially below discount rate.
    # If not, the Gordon Growth Model produces unreliable or infinite results.
    min_spread = 0.01  # 1% minimum spread
    if terminal_growth_rate >= discount_rate - min_spread:
        warning = (
            f"Terminal growth rate ({terminal_growth_rate:.1%}) is too close to "
            f"or exceeds discount rate ({discount_rate:.1%}). "
            f"Capping terminal growth at {discount_rate - min_spread - 0.001:.1%}."
        )
        logger.warning(f"{ticker} [{scenario_name}]: {warning}")
        terminal_growth_rate = discount_rate - min_spread - 0.001

    # Negative FCF: model will technically run but results are meaningless
    if trailing_fcf <= 0:
        warning = (
            f"Trailing FCF is negative (${trailing_fcf:.1f}M). "
            "DCF fair value estimate is not meaningful for cash-burning businesses."
        )
        logger.warning(f"{ticker} [{scenario_name}]: {warning}")

    # ── Step 1: Base FCF (year 1 starting point) ─────────────────────────
    # Apply haircut in bear case to account for cyclicality or one-time items
    base_fcf = trailing_fcf * (1 - fcf_haircut)

    # ── Step 2: Project FCF for N years ──────────────────────────────────
    projected_fcfs = []
    for year in range(1, projection_years + 1):
        fcf_year = base_fcf * ((1 + fcf_growth_rate) ** year)
        projected_fcfs.append(fcf_year)

    # ── Step 3: Discount each FCF to present value ────────────────────────
    # PV(FCF_t) = FCF_t / (1 + r)^t
    pv_of_fcfs = []
    for year, fcf in enumerate(projected_fcfs, start=1):
        pv = fcf / ((1 + discount_rate) ** year)
        pv_of_fcfs.append(pv)

    pv_fcf_sum = sum(pv_of_fcfs)

    # ── Step 4: Terminal Value (Gordon Growth Model) ──────────────────────
    # FCF in the first year beyond the projection horizon
    fcf_terminal = projected_fcfs[-1] * (1 + terminal_growth_rate)

    # Terminal value in perpetuity at the end of year N
    terminal_value = fcf_terminal / (discount_rate - terminal_growth_rate)

    # Discount terminal value back to today (end of projection horizon)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    # ── Step 5: Enterprise Value ──────────────────────────────────────────
    enterprise_value = pv_fcf_sum + pv_terminal_value

    # ── Step 6: Equity Value = EV − Net Debt ─────────────────────────────
    # Net debt can be negative (net cash position), which adds to equity value
    equity_value = enterprise_value - net_debt

    # ── Step 7: Fair Value per Share ──────────────────────────────────────
    if shares_outstanding <= 0:
        fair_value_per_share = 0.0
        warning = (warning or "") + " | Zero shares outstanding — cannot compute per-share value."
        is_valid = False
    else:
        fair_value_per_share = (equity_value * 1_000_000) / (shares_outstanding * 1_000_000)
        # equity_value is in millions, shares_outstanding in millions → ratio is correct
        fair_value_per_share = equity_value / shares_outstanding
        is_valid = fair_value_per_share > 0

    # ── Step 8: Diagnostic — terminal value % ────────────────────────────
    terminal_value_pct = (
        pv_terminal_value / enterprise_value if enterprise_value > 0 else 0.0
    )
    if terminal_value_pct > MAX_TERMINAL_VALUE_PCT:
        tv_warn = (
            f"Terminal value is {terminal_value_pct:.0%} of EV. "
            f"Model is highly sensitive to long-term growth assumptions. "
            f"Treat this estimate with extra caution."
        )
        warning = f"{warning} | {tv_warn}" if warning else tv_warn
        logger.warning(f"{ticker} [{scenario_name}]: {tv_warn}")

    return DCFResult(
        ticker=ticker,
        scenario=scenario_name,
        fcf_growth_rate=fcf_growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        projection_years=projection_years,
        base_fcf=base_fcf,
        projected_fcfs=projected_fcfs,
        pv_of_fcfs=pv_of_fcfs,
        pv_fcf_sum=pv_fcf_sum,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
        terminal_value_pct=terminal_value_pct,
        is_valid=is_valid,
        warning=warning,
    )


def explain_result(result: DCFResult, market_price: float) -> str:
    """
    Return a human-readable explanation of a DCF result, including
    all assumptions used. This is the 'transparency' output.
    """
    upside = (result.fair_value_per_share - market_price) / market_price if market_price > 0 else 0

    lines = [
        f"\n{'─'*60}",
        f"  DCF Result: {result.ticker} [{result.scenario.upper()}]",
        f"{'─'*60}",
        f"  Assumptions:",
        f"    Base FCF:              ${result.base_fcf:>10.1f}M",
        f"    FCF Growth Rate:       {result.fcf_growth_rate:>10.1%}",
        f"    Discount Rate (WACC):  {result.discount_rate:>10.1%}",
        f"    Terminal Growth Rate:  {result.terminal_growth_rate:>10.1%}",
        f"    Projection Years:      {result.projection_years:>10d}",
        f"",
        f"  Valuation Components:",
        f"    PV of FCFs (Yrs 1-{result.projection_years}): ${result.pv_fcf_sum:>8.1f}M",
        f"    PV of Terminal Value:  ${result.pv_terminal_value:>8.1f}M  ({result.terminal_value_pct:.0%} of EV)",
        f"    Enterprise Value:      ${result.enterprise_value:>8.1f}M",
        f"    Equity Value:          ${result.equity_value:>8.1f}M",
        f"",
        f"  Result:",
        f"    Fair Value / Share:   ${result.fair_value_per_share:>9.2f}",
        f"    Market Price:         ${market_price:>9.2f}",
        f"    Implied Upside:       {upside:>9.1%}",
    ]

    if result.warning:
        lines.append(f"\n  ⚠️  Warning: {result.warning}")

    lines.append(f"{'─'*60}\n")
    return "\n".join(lines)
