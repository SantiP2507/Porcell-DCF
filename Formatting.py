"""
utils/formatting.py — Console output formatting.

Provides consistent, readable output for:
  - Financial numbers (millions, billions, per-share)
  - Percentage display
  - Full valuation report printout
  - FCF history table
"""

from typing import Optional
from data.models import ValuationSummary, FinancialSnapshot, MarketData


def fmt_millions(value: Optional[float], decimals: int = 1) -> str:
    """Format a number in millions with M suffix. None → 'N/A'."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000:
        return f"${value/1_000:,.{decimals}f}B"
    return f"${value:,.{decimals}f}M"


def fmt_billions(value: Optional[float], decimals: int = 2) -> str:
    """Format a number (in millions) as billions."""
    if value is None:
        return "N/A"
    return f"${value/1_000:,.{decimals}f}B"


def fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: Optional[float], show_sign: bool = False) -> str:
    if value is None:
        return "N/A"
    spec = "+.1%" if show_sign else ".1%"
    return format(value, spec)


def print_financial_snapshot(f: FinancialSnapshot, m: MarketData) -> None:
    """Print a clean summary of the financial inputs used."""
    print(f"\n{'─'*55}")
    print(f"  FINANCIAL SNAPSHOT: {f.ticker}")
    print(f"{'─'*55}")
    print(f"  Market Price:        {fmt_price(m.price)}")
    print(f"  Market Cap:          {fmt_billions(m.market_cap)}")
    print(f"  Shares Outstanding:  {m.shares_outstanding:,.1f}M")
    print()
    print(f"  Trailing FCF (TTM):  {fmt_millions(f.trailing_fcf)}")
    print(f"  Revenue (TTM):       {fmt_millions(f.revenue_ttm)}")
    print(f"  Op. Income (TTM):    {fmt_millions(f.operating_income_ttm)}")
    print()
    print(f"  Cash & Equivalents:  {fmt_millions(f.cash_and_equivalents)}")
    print(f"  Total Debt:          {fmt_millions(f.total_debt)}")
    net_label = "Net Cash" if f.net_debt < 0 else "Net Debt"
    print(f"  {net_label}:           {fmt_millions(abs(f.net_debt))}")

    if f.trailing_fcf > 0 and f.net_debt > 0:
        print(f"  Net Debt / FCF:      {f.net_debt / f.trailing_fcf:.1f}x")

    if m.market_cap > 0 and f.trailing_fcf > 0:
        fcf_yield = f.trailing_fcf / m.market_cap
        print(f"  FCF Yield:           {fmt_pct(fcf_yield)}")

    print(f"\n  FCF History (oldest → newest):")
    if f.historical_fcf:
        for i, val in enumerate(f.historical_fcf):
            bar = "█" * max(0, int(val / max(abs(v) for v in f.historical_fcf) * 20))
            print(f"    Year -{len(f.historical_fcf) - i - 1}:  {fmt_millions(val):>10}  {bar}")
    else:
        print("    (no history available)")
    print(f"{'─'*55}\n")


def print_valuation_report(summary: ValuationSummary) -> None:
    """Print the full valuation report for a ticker."""
    print(f"\n{'═'*60}")
    print(f"  DCF VALUATION REPORT: {summary.ticker}")
    print(f"  Market Price: {fmt_price(summary.market_price)}")
    print(f"{'═'*60}")

    print(f"\n  {'Scenario':<10} {'Fair Value':>11} {'Upside':>9} {'EV':>12} {'TV%':>7}")
    print(f"  {'─'*52}")

    for label, result in [
        ("Bear ▼", summary.bear),
        ("Base  ●", summary.base),
        ("Bull ▲", summary.bull),
    ]:
        if result and result.is_valid:
            upside = (result.fair_value_per_share - summary.market_price) / summary.market_price
            print(
                f"  {label:<10} {fmt_price(result.fair_value_per_share):>11} "
                f"{fmt_pct(upside, show_sign=True):>9} "
                f"{fmt_millions(result.enterprise_value):>12} "
                f"{result.terminal_value_pct:>6.0%}"
            )
            if result.warning:
                print(f"  {'':10} ⚠️  {result.warning[:65]}")
        else:
            reason = result.warning if result else "Not computed"
            print(f"  {label:<10} {'N/A':>11}  [{reason[:40]}]")

    # Implied growth rate if available
    print(f"\n  Assumptions (Base Case):")
    if summary.base:
        b = summary.base
        print(f"    FCF Growth Rate:    {fmt_pct(b.fcf_growth_rate)}")
        print(f"    Discount Rate:      {fmt_pct(b.discount_rate)}")
        print(f"    Terminal Growth:    {fmt_pct(b.terminal_growth_rate)}")
        print(f"    Projection Years:   {b.projection_years}")
        print(f"    Base FCF:           {fmt_millions(b.base_fcf)}")

    print(f"{'═'*60}\n")


def print_projected_fcfs(summary: ValuationSummary) -> None:
    """Print year-by-year FCF projections for all scenarios."""
    if not summary.base:
        return

    years = list(range(1, summary.base.projection_years + 1))

    print(f"\n  FCF PROJECTIONS: {summary.ticker}")
    print(f"  {'Year':<6} {'Bear':>10} {'Base':>10} {'Bull':>10}")
    print(f"  {'─'*38}")

    for i, year in enumerate(years):
        bear_fcf = summary.bear.projected_fcfs[i] if summary.bear and i < len(summary.bear.projected_fcfs) else None
        base_fcf = summary.base.projected_fcfs[i] if i < len(summary.base.projected_fcfs) else None
        bull_fcf = summary.bull.projected_fcfs[i] if summary.bull and i < len(summary.bull.projected_fcfs) else None

        print(
            f"  {year:<6} "
            f"{fmt_millions(bear_fcf, 0):>10} "
            f"{fmt_millions(base_fcf, 0):>10} "
            f"{fmt_millions(bull_fcf, 0):>10}"
        )
    print()
