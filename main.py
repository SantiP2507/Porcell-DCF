"""
main.py — CLI entry point for the DCF Valuation & Research Engine.

Usage:
  python main.py --ticker AAPL
  python main.py --ticker MSFT --sensitivity
  python main.py --ticker AAPL --base-growth 0.10 --base-wacc 0.09
  python main.py --screen
  python main.py --screen --tickers AAPL MSFT NVDA
  python main.py --ticker AAPL --save            # persist to Supabase
  python main.py --clear-cache
  python main.py --ticker AAPL --show-charts     # interactive display
"""

import argparse
import logging
import sys
from datetime import date
from typing import Optional, List


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="  %(levelname)-8s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy third-party loggers
    for lib in ["urllib3", "yfinance", "requests", "matplotlib", "PIL"]:
        logging.getLogger(lib).setLevel(logging.ERROR)


log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ticker(
    ticker: str,
    run_sensitivity: bool = False,
    show_charts: bool = False,
    make_charts: bool = True,
    scenario_overrides: Optional[dict] = None,
    save: bool = False,
    years: Optional[int] = None,
) -> Optional[object]:
    """
    Full end-to-end analysis pipeline for one ticker.
    Returns ValuationSummary on success, None on failure.
    """
    from data.fetcher import fetch_all, validate_financials
    from valuation.scenarios import build_scenarios, summarize_scenarios
    from valuation.dcf import explain_result
    from config import DCF_PROJECTION_YEARS

    ticker = ticker.upper().strip()
    print(f"\n{'─'*62}")
    print(f"  Analyzing: {ticker}")
    print(f"{'─'*62}")

    # ── Step 1: Fetch ──────────────────────────────────────────────────────
    print("  Fetching data...", end=" ", flush=True)
    try:
        market_data, financials = fetch_all(ticker)
        print("✓")
    except Exception as e:
        print(f"✗\n  Error: {e}")
        return None

    # ── Step 2: Validate ───────────────────────────────────────────────────
    warnings = validate_financials(financials)
    for w in warnings:
        print(f"  ⚠️  {w}")

    # Summary line
    net_debt_label = f"Net Debt: ${financials.net_debt:.0f}M" if financials.net_debt >= 0 else f"Net Cash: ${-financials.net_debt:.0f}M"
    print(f"\n  Price:   ${market_data.price:.2f}  |  Mkt Cap: ${market_data.market_cap/1000:.1f}B")
    print(f"  TTM FCF: ${financials.trailing_fcf:.0f}M  |  {net_debt_label}")
    if financials.historical_fcf:
        hist_str = "  ".join(f"${v:.0f}M" for v in financials.historical_fcf[-5:])
        print(f"  FCF History ({len(financials.historical_fcf)}yr): {hist_str}")

    # ── Step 3: Build Scenarios ────────────────────────────────────────────
    projection_years = years or DCF_PROJECTION_YEARS
    print(f"\n  Running DCF ({projection_years}-year horizon)...", end=" ", flush=True)
    try:
        summary = build_scenarios(
            financials=financials,
            market_data=market_data,
            overrides=scenario_overrides,
            projection_years=projection_years,
        )
        print("✓")
    except Exception as e:
        print(f"✗\n  DCF error: {e}")
        return None

    # ── Step 4: Print Results ─────────────────────────────────────────────
    print(summarize_scenarios(summary))

    for result in [summary.bear, summary.base, summary.bull]:
        if result and result.is_valid:
            print(explain_result(result, market_data.price))

    # ── Step 5: Sensitivity Analysis ──────────────────────────────────────
    sensitivity_data = None
    if run_sensitivity:
        from valuation.sensitivity import (
            compute_sensitivity_grid, grid_to_dataframe,
            compute_implied_growth_rate,
        )
        print("  Computing sensitivity grid...", end=" ", flush=True)
        try:
            g_rates, d_rates, grid = compute_sensitivity_grid(financials, market_data)
            sensitivity_data = (g_rates, d_rates, grid)
            print("✓")

            df = grid_to_dataframe(g_rates, d_rates, grid, market_data.price)
            print(f"\n  Sensitivity: {ticker} | Price: ${market_data.price:.2f} | Values = Upside %")
            print(f"  {'Growth \\ WACC':<15}", end="")
            for col in df.columns:
                print(f"  {col:>8}", end="")
            print()
            print(f"  {'─'*65}")
            for idx, row in df.iterrows():
                print(f"  {idx:<15}", end="")
                for val in row:
                    display = f"{val*100:+.0f}%" if val == val else "N/A"
                    print(f"  {display:>8}", end="")
                print()

            implied = compute_implied_growth_rate(financials, market_data)
            if implied is not None:
                print(f"\n  Market-implied FCF growth rate: {implied:.1%}")
            else:
                print(f"\n  Market price implies growth > 50% — consider growth decomposition.")
        except Exception as e:
            print(f"✗\n  Sensitivity error: {e}")

    # ── Step 6: Charts ────────────────────────────────────────────────────
    if make_charts:
        print("\n  Generating charts...", end=" ", flush=True)
        try:
            from charts.plotter import plot_all
            paths = plot_all(summary=summary, sensitivity_data=sensitivity_data, show=show_charts)
            print(f"✓  ({len(paths)} charts)")
            for p in paths:
                print(f"    → {p.name}")
        except Exception as e:
            print(f"✗\n  Charts error: {e}")

    # ── Step 7: Persist ────────────────────────────────────────────────────
    if save:
        _persist_valuation(summary, financials)

    return summary


def _persist_valuation(summary, financials):
    """Save valuation results to Supabase."""
    from db.supabase_client import (
        save_valuation, save_market_snapshot, is_configured
    )
    if not is_configured():
        print("\n  ⚠️  Supabase not configured (SUPABASE_URL / SUPABASE_KEY not set)")
        return

    print("\n  Saving to Supabase...", end=" ", flush=True)
    assumptions = {
        "trailing_fcf_m": financials.trailing_fcf,
        "net_debt_m":     financials.net_debt,
        "bear":  _scenario_assumptions(summary.bear),
        "base":  _scenario_assumptions(summary.base),
        "bull":  _scenario_assumptions(summary.bull),
    }
    ok = save_valuation(
        ticker=summary.ticker,
        valuation_date=date.today(),
        bear_value=summary.bear.fair_value_per_share if summary.bear else None,
        base_value=summary.base.fair_value_per_share if summary.base else None,
        bull_value=summary.bull.fair_value_per_share if summary.bull else None,
        market_price=summary.market_price,
        terminal_value_pct=summary.base.terminal_value_pct if summary.base else None,
        assumptions=assumptions,
    )
    save_market_snapshot(
        ticker=summary.ticker,
        snapshot_date=date.today(),
        price=summary.market_price,
        fcf=financials.trailing_fcf,
        valuation_gap=summary.upside_base,
    )
    print("✓" if ok else "partial (market snapshot saved)")


def _scenario_assumptions(result) -> Optional[dict]:
    if result is None:
        return None
    return {
        "fcf_growth_rate":     result.fcf_growth_rate,
        "discount_rate":       result.discount_rate,
        "terminal_growth_rate": result.terminal_growth_rate,
        "projection_years":    result.projection_years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    tickers: List[str],
    top_n: int = 10,
    show_charts: bool = False,
    make_charts: bool = True,
    save: bool = False,
) -> None:
    """Run DCF on all tickers, screen, rank, display, and optionally persist."""
    from data.fetcher import fetch_all, validate_financials
    from valuation.scenarios import build_scenarios
    from research.screener import screen_candidates, print_screener_results
    from config import DEFAULT_SCREEN_TICKERS

    universe = tickers or DEFAULT_SCREEN_TICKERS
    print(f"\n{'═'*62}")
    print(f"  DCF DAILY SCREENER — {len(universe)} tickers")
    print(f"{'═'*62}")

    summaries = []
    failed = []

    for i, ticker in enumerate(universe, 1):
        print(f"  [{i:2d}/{len(universe)}] {ticker:<8}", end=" ", flush=True)
        try:
            market_data, financials = fetch_all(ticker)
            summary = build_scenarios(financials=financials, market_data=market_data)
            summaries.append(summary)
            upside = summary.upside_base
            tag = f"Base: {upside:+.0%}" if upside is not None else "no base"
            print(f"✓  {tag}")
        except Exception as e:
            failed.append(ticker)
            print(f"✗  {str(e)[:50]}")

    print(f"\n  Done: {len(summaries)} ok | {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    if not summaries:
        print("  No valid valuations — nothing to screen.")
        return

    candidates = screen_candidates(summaries, top_n=top_n)
    print_screener_results(candidates)

    if make_charts and candidates:
        try:
            from charts.plotter import plot_screener_ranking
            p = plot_screener_ranking(candidates, show=show_charts)
            if p:
                print(f"  Screener chart: {p}")
        except Exception as e:
            log.warning(f"Screener chart failed: {e}")

    if save and candidates:
        from db.supabase_client import save_research_candidates_batch, is_configured
        if is_configured():
            n = save_research_candidates_batch(candidates)
            print(f"\n  Saved {n}/{len(candidates)} candidates to Supabase")
        else:
            print("  ⚠️  Supabase not configured")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="DCF Valuation & Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ticker", "-t", metavar="SYMBOL",
                      help="Analyze a single ticker (e.g. AAPL)")
    mode.add_argument("--screen", "-s", action="store_true",
                      help="Run the daily research screener")
    mode.add_argument("--clear-cache", action="store_true",
                      help="Clear the local data cache")

    p.add_argument("--tickers", nargs="+", metavar="SYM",
                   help="Custom ticker list for --screen (default: universe in config)")
    p.add_argument("--top-n", type=int, default=10,
                   help="Max candidates to return from screener (default: 10)")

    p.add_argument("--sensitivity", action="store_true",
                   help="Run sensitivity grid (FCF growth × WACC)")
    p.add_argument("--years", type=int, default=None,
                   help="Projection horizon override (default: 5)")
    p.add_argument("--base-growth", type=float, metavar="RATE",
                   help="Override base FCF growth rate, e.g. 0.10")
    p.add_argument("--base-wacc", type=float, metavar="RATE",
                   help="Override base discount rate (WACC), e.g. 0.09")
    p.add_argument("--base-terminal", type=float, metavar="RATE",
                   help="Override base terminal growth rate, e.g. 0.025")

    p.add_argument("--save", action="store_true",
                   help="Persist results to Supabase (requires credentials)")
    p.add_argument("--show-charts", action="store_true",
                   help="Display charts interactively (requires GUI)")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip chart generation")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging")

    return p.parse_args()




def _ensure_ml_models():
    """Train ML models on first run, retrain weekly thereafter."""
    try:
        from ml.trainer import train_all, should_retrain, mark_retrained
        if should_retrain():
            print("  Initializing ML models (first run or weekly retrain)...")
            train_all(force=False)
            mark_retrained()
            print("  ML models ready.")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"ML model setup failed (non-fatal): {e}")

def main():
    args = _parse_args()
    _setup_logging(args.verbose)
    _ensure_ml_models()

    if args.clear_cache:
        from data.cache import cache_clear
        cache_clear()
        print("  ✓  Cache cleared.")
        return

    if args.ticker:
        overrides = {}
        base_ov = {}
        if args.base_growth is not None:
            base_ov["fcf_growth_rate"] = args.base_growth
        if args.base_wacc is not None:
            base_ov["discount_rate"] = args.base_wacc
        if args.base_terminal is not None:
            base_ov["terminal_growth_rate"] = args.base_terminal
        if base_ov:
            overrides["base"] = base_ov

        result = analyze_ticker(
            ticker=args.ticker,
            run_sensitivity=args.sensitivity,
            show_charts=args.show_charts,
            make_charts=not args.no_charts,
            scenario_overrides=overrides or None,
            save=args.save,
            years=args.years,
        )
        sys.exit(0 if result else 1)

    if args.screen:
        run_screener(
            tickers=[t.upper() for t in args.tickers] if args.tickers else None,
            top_n=args.top_n,
            show_charts=args.show_charts,
            make_charts=not args.no_charts,
            save=args.save,
        )


if __name__ == "__main__":
    main()
