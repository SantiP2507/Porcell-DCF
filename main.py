"""
main.py — Entry point for the DCF Valuation & Research Engine.

Usage:
  # Full analysis on a single ticker
  python main.py --ticker AAPL

  # Full analysis with sensitivity heatmap
  python main.py --ticker AAPL --sensitivity

  # Daily screener across a watchlist
  python main.py --screen --tickers AAPL MSFT GOOG AMZN META NVDA

  # Screener with sensitivity for each candidate
  python main.py --screen --tickers AAPL MSFT GOOG --sensitivity

  # Custom scenario overrides
  python main.py --ticker AAPL --base-growth 0.10 --base-wacc 0.09

  # Verbose logging (show DEBUG output)
  python main.py --ticker AAPL --verbose

  # Show charts interactively (opens matplotlib window)
  python main.py --ticker AAPL --show-charts
"""

import argparse
import logging
import sys
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SETUP (must happen before other imports that use logging)
# ─────────────────────────────────────────────────────────────────────────────

def _setup(verbose: bool):
    """Configure logging before anything else."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="  %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy libraries
    for name in ["urllib3", "yfinance", "peewee", "requests", "matplotlib"]:
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ticker(
    ticker: str,
    run_sensitivity: bool = False,
    show_charts: bool = False,
    scenario_overrides: Optional[dict] = None,
) -> Optional[object]:  # returns ValuationSummary
    """
    Full end-to-end analysis for one ticker:
      1. Fetch data
      2. Run DCF (3 scenarios)
      3. Print reports
      4. Compute sensitivity grid (optional)
      5. Generate all charts
      6. Persist to Supabase (if configured)

    Returns the ValuationSummary, or None on failure.
    """
    from data.fetcher import fetch_all, validate_financials
    from valuation.scenarios import build_scenarios, summarize_scenarios
    from valuation.sensitivity import compute_sensitivity_grid
    from charts.plotter import plot_all
    from supabase.client import save_valuation, save_market_snapshot
    from utils.formatting import (
        print_financial_snapshot,
        print_valuation_report,
        print_projected_fcfs,
    )

    print(f"\n{'─'*60}")
    print(f"  Analyzing: {ticker.upper()}")
    print(f"{'─'*60}")

    # ── 1. Fetch Data ──────────────────────────────────────────────────────
    try:
        market_data, financials = fetch_all(ticker)
    except ValueError as e:
        logger.error(f"Data fetch failed for {ticker}: {e}")
        return None

    # Print data quality warnings
    warnings = validate_financials(financials)
    for w in warnings:
        print(f"\n  ⚠️  {w}")

    # Print financial snapshot
    print_financial_snapshot(financials, market_data)

    # ── 2. Run DCF ─────────────────────────────────────────────────────────
    try:
        summary = build_scenarios(
            financials=financials,
            market_data=market_data,
            overrides=scenario_overrides,
        )
    except Exception as e:
        logger.error(f"DCF engine failed for {ticker}: {e}")
        return None

    # ── 3. Print Reports ───────────────────────────────────────────────────
    print_valuation_report(summary)
    print_projected_fcfs(summary)

    # DCF explanation for each scenario
    from valuation.dcf import explain_result
    for result in [summary.bear, summary.base, summary.bull]:
        if result and result.is_valid:
            print(explain_result(result, market_data.price))

    # ── 4. Sensitivity Analysis ────────────────────────────────────────────
    sensitivity_data = None
    if run_sensitivity:
        print(f"  Computing sensitivity grid for {ticker}...")
        try:
            g_rates, d_rates, grid = compute_sensitivity_grid(financials, market_data)

            from valuation.sensitivity import grid_to_dataframe, compute_implied_growth_rate
            df = grid_to_dataframe(g_rates, d_rates, grid, market_data.price)

            print(f"\n  SENSITIVITY ANALYSIS: {ticker}")
            print(f"  Upside % vs Market Price (${market_data.price:.2f})")
            print(f"  Rows = FCF Growth Rate | Cols = WACC\n")

            # Format for console: multiply by 100 for display
            display_df = df.applymap(lambda x: f"{x*100:+.0f}%" if x == x else "N/A")
            try:
                from tabulate import tabulate
                print(tabulate(display_df, headers="keys", tablefmt="rounded_outline"))
            except ImportError:
                print(display_df.to_string())

            # Implied growth rate
            implied_g = compute_implied_growth_rate(financials, market_data)
            if implied_g is not None:
                print(
                    f"\n  Market-Implied FCF Growth Rate (at base WACC): "
                    f"{implied_g:.1%}"
                )

            sensitivity_data = (g_rates, d_rates, grid)
            print()

        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")

    # ── 5. Charts ──────────────────────────────────────────────────────────
    try:
        chart_paths = plot_all(summary, sensitivity_data=sensitivity_data, show=show_charts)
        if chart_paths:
            print(f"  Charts saved:")
            for p in chart_paths:
                print(f"    → {p}")
        print()
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")

    # ── 6. Persist ─────────────────────────────────────────────────────────
    save_valuation(summary)
    save_market_snapshot(summary)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    tickers: List[str],
    run_sensitivity: bool = False,
    show_charts: bool = False,
) -> None:
    """
    Run DCF on all tickers, screen candidates, display ranked results,
    generate screener chart, and persist to Supabase.
    """
    from research.screener import screen_candidates, print_screener_results
    from charts.plotter import plot_screener_ranking
    from supabase.client import save_research_candidates

    print(f"\n{'═'*60}")
    print(f"  📊  DCF SCREENER — {len(tickers)} tickers")
    print(f"{'═'*60}\n")

    summaries = []
    for ticker in tickers:
        summary = analyze_ticker(
            ticker,
            run_sensitivity=run_sensitivity,
            show_charts=show_charts,
        )
        if summary:
            summaries.append(summary)

    if not summaries:
        print("  No tickers could be analyzed.")
        return

    # Screen and rank
    candidates = screen_candidates(summaries)

    # Print results table
    print_screener_results(candidates)

    # Screener chart
    if candidates:
        try:
            p = plot_screener_ranking(candidates, show=show_charts)
            if p:
                print(f"  Screener chart saved: {p}\n")
        except Exception as e:
            logger.error(f"Screener chart failed: {e}")

    # Persist
    save_research_candidates(candidates)

    print(
        f"  Screened {len(summaries)} tickers | "
        f"{len(candidates)} candidates | "
        f"Charts saved to {__import__('config').CHART_OUTPUT_DIR}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="DCF Valuation & Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker", "-t",
        metavar="TICKER",
        help="Analyze a single ticker (e.g. AAPL)",
    )
    group.add_argument(
        "--screen", "-s",
        action="store_true",
        help="Run daily screener (requires --tickers)",
    )

    # Ticker list for screener
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="List of tickers for screener mode",
    )

    # Analysis options
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Compute and display sensitivity heatmap",
    )
    parser.add_argument(
        "--show-charts",
        action="store_true",
        help="Open charts interactively (requires display)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug logging",
    )

    # Custom scenario overrides
    parser.add_argument(
        "--base-growth",
        type=float,
        metavar="RATE",
        help="Override base FCF growth rate (e.g. 0.10 = 10%%)",
    )
    parser.add_argument(
        "--base-wacc",
        type=float,
        metavar="RATE",
        help="Override base discount rate / WACC (e.g. 0.09 = 9%%)",
    )
    parser.add_argument(
        "--base-terminal",
        type=float,
        metavar="RATE",
        help="Override base terminal growth rate (e.g. 0.03 = 3%%)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        metavar="N",
        help="Override projection horizon (default: 5 years)",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    _setup(args.verbose)

    # Build scenario overrides from CLI flags
    scenario_overrides = {}
    base_override = {}
    if args.base_growth is not None:
        base_override["fcf_growth_rate"] = args.base_growth
    if args.base_wacc is not None:
        base_override["discount_rate"] = args.base_wacc
    if args.base_terminal is not None:
        base_override["terminal_growth_rate"] = args.base_terminal
    if base_override:
        scenario_overrides["base"] = base_override

    # ── Single ticker mode ─────────────────────────────────────────────────
    if args.ticker:
        result = analyze_ticker(
            ticker=args.ticker,
            run_sensitivity=args.sensitivity,
            show_charts=args.show_charts,
            scenario_overrides=scenario_overrides or None,
        )
        sys.exit(0 if result else 1)

    # ── Screener mode ──────────────────────────────────────────────────────
    if args.screen:
        if not args.tickers:
            print("  Error: --screen requires --tickers TICK1 TICK2 ...")
            sys.exit(1)
        run_screener(
            tickers=[t.upper() for t in args.tickers],
            run_sensitivity=args.sensitivity,
            show_charts=args.show_charts,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
