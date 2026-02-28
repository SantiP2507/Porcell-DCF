"""
charts/plotter.py — All visualization logic for the DCF Research Engine.

Charts generated on each run:
  1. Scenario waterfall / bar chart — bear/base/bull vs market price
  2. FCF history + projection fan chart — historical trend + 3 scenario projections
  3. Sensitivity heatmap — fair value upside across growth × discount rate grid
  4. Terminal value composition — pie/bar showing PV(FCFs) vs PV(TV)
  5. Research screener ranking — horizontal bar chart of priority scores

Design principles:
  - Every chart saves to disk AND can display interactively (show=True).
  - Charts update on every run — they reflect the current market price.
  - No web server, no external dependencies beyond matplotlib/seaborn.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np

from config import (
    CHART_OUTPUT_DIR,
    CHART_STYLE,
    CHART_DPI,
    CHART_FIGSIZE_WIDE,
    CHART_FIGSIZE_SQUARE,
    CHART_ACCENT_COLOR,
    CHART_BEAR_COLOR,
    CHART_BULL_COLOR,
    CHART_BASE_COLOR,
)
from data.models import ValuationSummary, ResearchCandidate

logger = logging.getLogger(__name__)

# Suppress matplotlib warnings in non-interactive mode
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for scripts & servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import matplotlib.patches as mpatches
    import seaborn as sns
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    logger.warning("matplotlib/seaborn not installed. Charts disabled.")


def _check_available():
    if not CHARTS_AVAILABLE:
        raise RuntimeError("matplotlib is required for charting. Run: pip install matplotlib seaborn")


def _apply_style():
    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")  # fallback


def _save_and_show(fig, filename: str, show: bool = False) -> Path:
    """Save figure to CHART_OUTPUT_DIR and optionally display it."""
    path = CHART_OUTPUT_DIR / filename
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    logger.info(f"Chart saved: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Scenario Valuation Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_scenario_valuation(
    summary: ValuationSummary,
    show: bool = False,
) -> Optional[Path]:
    """
    Horizontal bar chart showing bear/base/bull fair values vs market price.
    Bars are colored red/grey/green; a vertical dashed line marks market price.
    """
    _check_available()
    _apply_style()

    scenarios = []
    values = []
    colors = []
    for label, result, color in [
        ("Bear", summary.bear, CHART_BEAR_COLOR),
        ("Base", summary.base, CHART_BASE_COLOR),
        ("Bull", summary.bull, CHART_BULL_COLOR),
    ]:
        if result and result.is_valid:
            scenarios.append(label)
            values.append(result.fair_value_per_share)
            colors.append(color)

    if not scenarios:
        logger.warning("No valid scenarios to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(scenarios, values, color=colors, alpha=0.85, height=0.5)

    # Market price line
    ax.axvline(
        x=summary.market_price,
        color=CHART_ACCENT_COLOR,
        linewidth=2,
        linestyle="--",
        label=f"Market Price: ${summary.market_price:.2f}",
    )

    # Value labels on bars
    for bar, val in zip(bars, values):
        upside = (val - summary.market_price) / summary.market_price
        label = f"${val:.2f}  ({upside:+.1%})"
        ax.text(
            bar.get_width() + (max(values) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        f"{summary.ticker} — DCF Fair Value by Scenario",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Fair Value per Share (USD)", fontsize=11)
    ax.legend(fontsize=10)

    # Extend xlim to make room for labels
    ax.set_xlim(0, max(values) * 1.30)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))

    plt.tight_layout()
    return _save_and_show(fig, f"{summary.ticker}_scenarios.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: FCF History + Scenario Projection Fan
# ─────────────────────────────────────────────────────────────────────────────

def plot_fcf_projection(
    summary: ValuationSummary,
    show: bool = False,
) -> Optional[Path]:
    """
    Line chart with:
      - Historical FCF (solid, dark) — actual past data
      - Bear/base/bull projections as separate lines from year 0 onward
      - Shaded region between bear and bull projections

    Year 0 = TTM FCF (the anchor point). Years 1-N = projections.
    """
    _check_available()
    _apply_style()

    hist = summary.financials.historical_fcf if summary.financials else []
    base = summary.base
    if not base:
        logger.warning("No base scenario — skipping FCF projection chart.")
        return None

    n_hist = len(hist)
    hist_x = list(range(-n_hist + 1, 1))  # e.g. [-4, -3, -2, -1, 0]
    proj_x = list(range(0, base.projection_years + 1))  # [0, 1, 2, ..., N]

    # Anchor year 0 = base_fcf (normalized TTM, after any haircut)
    def proj_series(result):
        if result is None:
            return None
        return [result.base_fcf] + result.projected_fcfs

    bear_proj = proj_series(summary.bear)
    base_proj = proj_series(summary.base)
    bull_proj = proj_series(summary.bull)

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)

    # Historical (solid, accent)
    if hist:
        ax.plot(hist_x, hist, color=CHART_ACCENT_COLOR, linewidth=2.5,
                marker="o", markersize=6, label="Historical FCF", zorder=5)

    # Shaded uncertainty band between bear and bull
    if bear_proj and bull_proj:
        ax.fill_between(
            proj_x, bear_proj, bull_proj,
            alpha=0.12, color=CHART_BASE_COLOR, label="Bear–Bull Range"
        )

    # Projection lines
    for label, proj, color, ls in [
        ("Bear Projection", bear_proj, CHART_BEAR_COLOR, "--"),
        ("Base Projection", base_proj, CHART_BASE_COLOR, "-"),
        ("Bull Projection", bull_proj, CHART_BULL_COLOR, "--"),
    ]:
        if proj:
            ax.plot(proj_x, proj, color=color, linewidth=2, linestyle=ls,
                    marker="s", markersize=5, label=label, zorder=4)

    # Separator line at year 0
    ax.axvline(x=0, color="black", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(0.15, ax.get_ylim()[1] * 0.95, "← History  |  Projection →",
            fontsize=9, color="gray", va="top")

    ax.set_title(
        f"{summary.ticker} — Free Cash Flow: History & Scenario Projections",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Year (0 = TTM)", fontsize=11)
    ax.set_ylabel("FCF (USD Millions)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}M"))
    ax.legend(fontsize=9)
    ax.set_xticks(hist_x + proj_x[1:])

    plt.tight_layout()
    return _save_and_show(fig, f"{summary.ticker}_fcf_projection.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Sensitivity Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_heatmap(
    ticker: str,
    growth_rates: List[float],
    discount_rates: List[float],
    grid,   # SensitivityGrid (list of lists) or DataFrame of upside %
    market_price: float,
    show: bool = False,
) -> Optional[Path]:
    """
    Heatmap of DCF upside % across the growth × discount rate grid.
    Green = upside, red = downside. Market price implied values are highlighted.

    Expects grid values as decimal upside fractions (e.g. 0.25 = 25% upside).
    """
    _check_available()

    import pandas as pd

    # Accept either raw grid (list[list]) or pre-built DataFrame
    if isinstance(grid, list):
        from valuation.sensitivity import grid_to_dataframe
        df = grid_to_dataframe(growth_rates, discount_rates, grid, market_price)
    else:
        df = grid  # already a DataFrame of upside %

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_SQUARE)

    # Center the colormap at 0 (no upside / no downside)
    data_arr = df.values.astype(float)
    vmax = min(abs(np.nanmax(data_arr)), 2.0)  # cap at 200% for readability
    vmin = -vmax

    # Convert to % for display
    display_df = df * 100

    sns.heatmap(
        display_df,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=0,
        vmin=vmin * 100,
        vmax=vmax * 100,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Upside vs Market Price (%)"},
        annot_kws={"size": 9},
    )

    ax.set_title(
        f"{ticker} — DCF Sensitivity: Upside % vs Market (${market_price:.2f})\n"
        f"Rows = FCF Growth Rate, Columns = WACC",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Discount Rate (WACC)", fontsize=11)
    ax.set_ylabel("FCF Growth Rate", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    return _save_and_show(fig, f"{ticker}_sensitivity.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: EV Composition — PV(FCFs) vs PV(Terminal Value)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ev_composition(
    summary: ValuationSummary,
    show: bool = False,
) -> Optional[Path]:
    """
    Stacked bar chart showing how enterprise value is split between:
      - PV of explicit FCF projections (years 1–N)
      - PV of terminal value

    High terminal value % = valuation depends heavily on long-term assumptions.
    """
    _check_available()
    _apply_style()

    scenarios_to_plot = []
    for label, result in [
        ("Bear", summary.bear),
        ("Base", summary.base),
        ("Bull", summary.bull),
    ]:
        if result and result.is_valid:
            scenarios_to_plot.append((label, result))

    if not scenarios_to_plot:
        return None

    labels = [s[0] for s in scenarios_to_plot]
    pv_fcfs = [s[1].pv_fcf_sum for s in scenarios_to_plot]
    pv_tvs = [s[1].pv_terminal_value for s in scenarios_to_plot]
    tv_pcts = [s[1].terminal_value_pct for s in scenarios_to_plot]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(labels))
    w = 0.5

    bars1 = ax.bar(x, pv_fcfs, w, label="PV of FCFs (Yrs 1–N)",
                   color=CHART_ACCENT_COLOR, alpha=0.85)
    bars2 = ax.bar(x, pv_tvs, w, bottom=pv_fcfs, label="PV of Terminal Value",
                   color="#F59E0B", alpha=0.85)

    # Terminal value % annotations
    for i, (pv_fcf, pv_tv, tv_pct) in enumerate(zip(pv_fcfs, pv_tvs, tv_pcts)):
        total = pv_fcf + pv_tv
        ax.text(i, total + total * 0.02, f"TV: {tv_pct:.0%}",
                ha="center", fontsize=10, fontweight="bold",
                color=CHART_BEAR_COLOR if tv_pct > 0.75 else "black")

    ax.set_title(
        f"{summary.ticker} — Enterprise Value Composition by Scenario",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_ylabel("Value (USD Millions)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}M"))
    ax.legend(fontsize=10)

    # Warn if any scenario has TV% > 75%
    if any(t > 0.75 for t in tv_pcts):
        ax.text(
            0.5, -0.14, "WARNING: Terminal value > 75% of EV — results highly sensitive to long-term assumptions.",
            ha="center", fontsize=9, color=CHART_BEAR_COLOR,
            transform=ax.transAxes
        )

    plt.tight_layout()
    return _save_and_show(fig, f"{summary.ticker}_ev_composition.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: Research Screener Priority Ranking
# ─────────────────────────────────────────────────────────────────────────────

def plot_screener_ranking(
    candidates: List[ResearchCandidate],
    show: bool = False,
) -> Optional[Path]:
    """
    Horizontal bar chart of research candidate priority scores.
    Shows top N candidates ranked by composite priority score.
    Bars are color-coded by valuation gap magnitude.
    """
    _check_available()
    _apply_style()

    if not candidates:
        logger.warning("No candidates to plot.")
        return None

    tickers = [c.ticker for c in candidates]
    scores = [c.priority_score for c in candidates]
    gaps = [c.valuation_gap for c in candidates]

    # Color by valuation gap — bigger gap = greener
    norm_gaps = [(g - min(gaps)) / (max(gaps) - min(gaps) + 1e-9) for g in gaps]
    colors = [plt.cm.RdYlGn(0.3 + 0.7 * n) for n in norm_gaps]  # type: ignore

    fig, ax = plt.subplots(figsize=(10, max(4, len(candidates) * 0.55)))

    bars = ax.barh(tickers, scores, color=colors, alpha=0.88, height=0.65)

    # Annotate each bar with gap %
    for bar, gap, score in zip(bars, gaps, scores):
        ax.text(
            score + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"Gap: {gap:+.0%}",
            va="center", fontsize=9, color="black"
        )

    ax.set_title(
        "Daily Research Candidates — Priority Score\n"
        "(Score 0–100, higher = more worth investigating)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Priority Score", fontsize=11)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()  # highest priority at top

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(min(gaps) * 100, max(gaps) * 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("Valuation Gap (Base Upside %)", fontsize=9)

    plt.tight_layout()
    return _save_and_show(fig, "screener_ranking.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Plot all charts for a single ticker
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    summary: ValuationSummary,
    sensitivity_data=None,
    show: bool = False,
) -> List[Path]:
    """
    Generate all charts for a ticker. Returns list of file paths.

    Args:
        summary:          ValuationSummary from DCF engine
        sensitivity_data: Optional tuple (growth_rates, discount_rates, grid)
        show:             If True, display charts interactively
    """
    paths = []

    try:
        p = plot_scenario_valuation(summary, show=show)
        if p:
            paths.append(p)
    except Exception as e:
        logger.error(f"Scenario chart failed: {e}")

    try:
        p = plot_fcf_projection(summary, show=show)
        if p:
            paths.append(p)
    except Exception as e:
        logger.error(f"FCF projection chart failed: {e}")

    try:
        p = plot_ev_composition(summary, show=show)
        if p:
            paths.append(p)
    except Exception as e:
        logger.error(f"EV composition chart failed: {e}")

    if sensitivity_data is not None:
        try:
            g_rates, d_rates, grid = sensitivity_data
            p = plot_sensitivity_heatmap(
                summary.ticker, g_rates, d_rates, grid,
                summary.market_price, show=show
            )
            if p:
                paths.append(p)
        except Exception as e:
            logger.error(f"Sensitivity chart failed: {e}")

    return paths
