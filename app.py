"""
app.py — Flask web server for the DCF Valuation Engine UI.

Run with:
  python app.py

Then open: http://localhost:5000

All DCF logic comes from the existing engine modules — this file
just exposes them as a web API and serves the frontend.

Endpoints:
  GET  /                        → serve the UI (index.html)
  POST /api/analyze             → run full DCF on a ticker
  POST /api/screen              → run screener on a watchlist
  GET  /api/history/<ticker>    → load Supabase valuation history
  GET  /api/candidates          → recent screener candidates from Supabase
"""

import sys
import io
import base64
import logging
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, request, send_from_directory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
for lib in ["urllib3", "yfinance", "requests", "matplotlib", "werkzeug"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

app = Flask(__name__, static_folder="ui", template_folder="ui")
logger = logging.getLogger(__name__)

# ── Ensure ML models exist on startup ────────────────────────────────────────
def _boot_ml():
    try:
        from ml.trainer import train_all, should_retrain, mark_retrained
        if should_retrain():
            logger.info("Training ML models...")
            train_all(force=False)
            mark_retrained()
            logger.info("ML models ready.")
    except Exception as e:
        logger.warning(f"ML boot failed (non-fatal): {e}")

_boot_ml()


# ─────────────────────────────────────────────────────────────────────────────
# SERVE UI
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# API: ANALYZE TICKER
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run full DCF analysis on a ticker.

    Request JSON:
      { "ticker": "AAPL", "sensitivity": true,
        "base_growth": 0.10, "base_wacc": 0.09 }

    Response JSON:
      { ticker, price, scenarios, financials, projections,
        sensitivity, charts, ml, warnings }
    """
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    run_sensitivity = data.get("sensitivity", False)

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        from data.fetcher import fetch_all, validate_financials
        from valuation.scenarios import build_scenarios
        from valuation.sensitivity import compute_sensitivity_grid, compute_implied_growth_rate, grid_to_dataframe

        # Build optional overrides
        overrides = {}
        base_override = {}
        if data.get("base_growth"):
            base_override["fcf_growth_rate"] = float(data["base_growth"])
        if data.get("base_wacc"):
            base_override["discount_rate"] = float(data["base_wacc"])
        if data.get("base_terminal"):
            base_override["terminal_growth_rate"] = float(data["base_terminal"])
        if base_override:
            overrides["base"] = base_override

        # Fetch + DCF
        market_data, financials = fetch_all(ticker)
        warnings = validate_financials(financials)
        summary = build_scenarios(financials, market_data, overrides or None)

        # Persist
        try:
            from supabase.client import save_valuation, save_market_snapshot
            save_valuation(summary)
            save_market_snapshot(summary)
        except Exception:
            pass

        # ML assessment
        ml_data = _run_ml(summary)

        # Sensitivity
        sensitivity_data = None
        implied_growth = None
        if run_sensitivity:
            g_rates, d_rates, grid = compute_sensitivity_grid(financials, market_data)
            df = grid_to_dataframe(g_rates, d_rates, grid, market_data.price)
            sensitivity_data = {
                "growth_rates": [f"{g:.0%}" for g in g_rates],
                "discount_rates": [f"{d:.0%}" for d in d_rates],
                "grid": [[round(v * 100, 1) if v == v else None for v in row] for row in grid],
            }
            implied_growth = compute_implied_growth_rate(financials, market_data)

        # Charts as base64
        charts = _generate_charts(summary, g_rates if run_sensitivity else None,
                                   d_rates if run_sensitivity else None,
                                   grid if run_sensitivity else None)

        return jsonify({
            "ticker": ticker,
            "price": market_data.price,
            "market_cap": market_data.market_cap,
            "shares": market_data.shares_outstanding,
            "warnings": warnings,
            "financials": {
                "trailing_fcf": financials.trailing_fcf,
                "historical_fcf": financials.historical_fcf,
                "cash": financials.cash_and_equivalents,
                "debt": financials.total_debt,
                "net_debt": financials.net_debt,
                "revenue": financials.revenue_ttm,
                "fcf_yield": financials.trailing_fcf / market_data.market_cap if market_data.market_cap > 0 else 0,
                "leverage": financials.net_debt / financials.trailing_fcf if financials.trailing_fcf > 0 else 0,
            },
            "scenarios": _serialize_scenarios(summary),
            "projections": _serialize_projections(summary),
            "sensitivity": sensitivity_data,
            "implied_growth": implied_growth,
            "ml": ml_data,
            "charts": charts,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# API: SCREENER
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/screen", methods=["POST"])
def screen():
    """
    Run DCF screener on a list of tickers.

    Request JSON:
      { "tickers": ["AAPL", "MSFT", "GOOG"] }
    """
    data = request.get_json()
    tickers = [t.upper().strip() for t in data.get("tickers", [])]

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    if len(tickers) > 30:
        return jsonify({"error": "Max 30 tickers per screen"}), 400

    results = []
    errors = []

    for ticker in tickers:
        try:
            from data.fetcher import fetch_all
            from valuation.scenarios import build_scenarios
            market_data, financials = fetch_all(ticker)
            summary = build_scenarios(financials, market_data)
            results.append(summary)
            try:
                from supabase.client import save_valuation, save_market_snapshot
                save_valuation(summary)
                save_market_snapshot(summary)
            except Exception:
                pass
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    # Screen candidates
    from research.screener import screen_candidates
    candidates = screen_candidates(results)

    # Screener chart
    screener_chart = None
    if candidates:
        try:
            screener_chart = _screener_chart_b64(candidates)
        except Exception:
            pass

    # Persist candidates
    try:
        from supabase.client import save_research_candidates
        save_research_candidates(candidates)
    except Exception:
        pass

    return jsonify({
        "total_screened": len(results),
        "candidates": [_serialize_candidate(c) for c in candidates],
        "errors": errors,
        "screener_chart": screener_chart,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API: HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/history/<ticker>")
def history(ticker):
    try:
        from supabase.client import load_valuation_history
        rows = load_valuation_history(ticker.upper(), limit=90)
        return jsonify({"ticker": ticker.upper(), "history": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/candidates")
def candidates():
    try:
        from supabase.client import load_recent_candidates
        rows = load_recent_candidates(days=7)
        return jsonify({"candidates": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_scenarios(summary):
    out = {}
    for name, result in [("bear", summary.bear), ("base", summary.base), ("bull", summary.bull)]:
        if result and result.is_valid:
            upside = (result.fair_value_per_share - summary.market_price) / summary.market_price
            out[name] = {
                "fair_value": round(result.fair_value_per_share, 2),
                "upside": round(upside, 4),
                "ev": round(result.enterprise_value, 1),
                "equity_value": round(result.equity_value, 1),
                "pv_fcfs": round(result.pv_fcf_sum, 1),
                "pv_tv": round(result.pv_terminal_value, 1),
                "tv_pct": round(result.terminal_value_pct, 4),
                "fcf_growth": result.fcf_growth_rate,
                "wacc": result.discount_rate,
                "terminal_growth": result.terminal_growth_rate,
                "warning": result.warning,
            }
        else:
            out[name] = None
    return out


def _serialize_projections(summary):
    if not summary.base:
        return []
    years = list(range(1, summary.base.projection_years + 1))
    rows = []
    for i, year in enumerate(years):
        rows.append({
            "year": year,
            "bear": round(summary.bear.projected_fcfs[i], 1) if summary.bear else None,
            "base": round(summary.base.projected_fcfs[i], 1),
            "bull": round(summary.bull.projected_fcfs[i], 1) if summary.bull else None,
        })
    return rows


def _serialize_candidate(c):
    return {
        "ticker": c.ticker,
        "score": c.priority_score,
        "bear_upside": round(c.bear_upside, 4),
        "base_upside": round(c.base_upside, 4),
        "bull_upside": round(c.bull_upside, 4),
        "valuation_gap": round(c.valuation_gap, 4),
        "reason": c.reason,
        "date": c.date.isoformat(),
    }


def _run_ml(summary):
    try:
        from ml.features import extract_features
        from ml.prioritizer import load_or_train
        from ml.stability import load_or_train as load_stab
        from ml.clustering import load_or_train as load_clust

        fv = extract_features(summary)
        if fv is None:
            return None

        p_score = load_or_train().score(fv)
        stab = load_or_train_stab().predict(fv) if False else load_stab().predict(fv)
        cluster = load_clust().predict(fv)

        return {
            "priority_score": p_score,
            "stability": {
                "is_stable": stab["is_stable"],
                "score": stab["stability_score"],
                "warnings": stab["warnings"],
            },
            "cluster": {
                "id": cluster["cluster_id"],
                "archetype": cluster["archetype"],
                "description": cluster["description"],
            },
        }
    except Exception as e:
        logger.debug(f"ML assessment failed: {e}")
        return None


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _generate_charts(summary, g_rates, d_rates, grid):
    charts = {}
    import warnings
    warnings.filterwarnings("ignore")

    BG = "#0f1117"
    CARD = "#1a1d27"
    ACCENT = "#3b82f6"
    RED = "#ef4444"
    GREEN = "#22c55e"
    GREY = "#6b7280"
    TEXT = "#e2e8f0"
    MUTED = "#94a3b8"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": CARD,
        "axes.edgecolor": "#2d3748",
        "axes.labelcolor": MUTED,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#2d3748",
        "grid.alpha": 0.6,
        "font.family": "monospace",
    })

    price = summary.market_price

    # ── Chart 1: Scenario bars ────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
        ax.set_facecolor(CARD)

        scenarios = []
        values = []
        colors = []
        for label, result, color in [
            ("Bear", summary.bear, RED),
            ("Base", summary.base, GREY),
            ("Bull", summary.bull, GREEN),
        ]:
            if result and result.is_valid:
                scenarios.append(label)
                values.append(result.fair_value_per_share)
                colors.append(color)

        bars = ax.barh(scenarios, values, color=colors, alpha=0.85, height=0.45)
        ax.axvline(x=price, color=ACCENT, linewidth=1.5, linestyle="--", label=f"Market: ${price:.2f}")

        for bar, val in zip(bars, values):
            upside = (val - price) / price
            sign = "+" if upside >= 0 else ""
            ax.text(max(values) * 1.02, bar.get_y() + bar.get_height() / 2,
                    f"${val:.2f}  ({sign}{upside:.0%})",
                    va="center", fontsize=9, color=TEXT)

        ax.set_xlim(0, max(values) * 1.35)
        ax.set_title("DCF Fair Value by Scenario", color=TEXT, fontsize=11, pad=10)
        ax.legend(fontsize=9, facecolor=CARD, edgecolor="#2d3748", labelcolor=TEXT)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.spines[:].set_visible(False)
        fig.tight_layout(pad=1.5)
        charts["scenarios"] = _fig_to_b64(fig)
    except Exception as e:
        logger.debug(f"Scenario chart failed: {e}")

    # ── Chart 2: FCF history + projections ───────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
        ax.set_facecolor(CARD)

        hist = summary.financials.historical_fcf if summary.financials else []
        n = len(hist)
        hist_x = list(range(-n + 1, 1))
        proj_x = list(range(0, summary.base.projection_years + 1)) if summary.base else []

        if hist:
            ax.plot(hist_x, hist, color=ACCENT, linewidth=2, marker="o",
                    markersize=5, label="Historical FCF", zorder=5)

        if summary.base:
            base_proj = [summary.base.base_fcf] + summary.base.projected_fcfs
            bear_proj = [summary.bear.base_fcf] + summary.bear.projected_fcfs if summary.bear else None
            bull_proj = [summary.bull.base_fcf] + summary.bull.projected_fcfs if summary.bull else None

            if bear_proj and bull_proj:
                ax.fill_between(proj_x, bear_proj, bull_proj, alpha=0.1, color=GREY)
            if bear_proj:
                ax.plot(proj_x, bear_proj, color=RED, linewidth=1.5, linestyle="--",
                        marker="s", markersize=4, label="Bear")
            ax.plot(proj_x, base_proj, color=GREY, linewidth=2, marker="s",
                    markersize=4, label="Base")
            if bull_proj:
                ax.plot(proj_x, bull_proj, color=GREEN, linewidth=1.5, linestyle="--",
                        marker="s", markersize=4, label="Bull")

        ax.axvline(x=0, color="#2d3748", linewidth=1, linestyle=":")
        ax.set_title("Free Cash Flow: History & Projections", color=TEXT, fontsize=11, pad=10)
        ax.set_xlabel("Year (0 = TTM)", color=MUTED)
        ax.set_ylabel("FCF ($M)", color=MUTED)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
        ax.legend(fontsize=9, facecolor=CARD, edgecolor="#2d3748", labelcolor=TEXT)
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=1.5)
        charts["fcf"] = _fig_to_b64(fig)
    except Exception as e:
        logger.debug(f"FCF chart failed: {e}")

    # ── Chart 3: EV composition ───────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
        ax.set_facecolor(CARD)

        import numpy as np
        labels, pv_fcfs, pv_tvs, tv_pcts = [], [], [], []
        for lbl, result in [("Bear", summary.bear), ("Base", summary.base), ("Bull", summary.bull)]:
            if result and result.is_valid:
                labels.append(lbl)
                pv_fcfs.append(result.pv_fcf_sum)
                pv_tvs.append(result.pv_terminal_value)
                tv_pcts.append(result.terminal_value_pct)

        x = np.arange(len(labels))
        ax.bar(x, pv_fcfs, 0.5, label="PV of FCFs", color=ACCENT, alpha=0.85)
        ax.bar(x, pv_tvs, 0.5, bottom=pv_fcfs, label="PV of Terminal Value", color="#f59e0b", alpha=0.85)

        for i, (pf, pt, tv) in enumerate(zip(pv_fcfs, pv_tvs, tv_pcts)):
            color = RED if tv > 0.75 else TEXT
            ax.text(i, pf + pt + (max(pv_fcfs) + max(pv_tvs)) * 0.02,
                    f"TV:{tv:.0%}", ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_title("Enterprise Value Composition", color=TEXT, fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
        ax.legend(fontsize=9, facecolor=CARD, edgecolor="#2d3748", labelcolor=TEXT)
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(pad=1.5)
        charts["ev"] = _fig_to_b64(fig)
    except Exception as e:
        logger.debug(f"EV chart failed: {e}")

    # ── Chart 4: Sensitivity heatmap ──────────────────────────────────────
    if g_rates and d_rates and grid:
        try:
            import numpy as np
            import seaborn as sns

            from valuation.sensitivity import grid_to_dataframe
            df = grid_to_dataframe(g_rates, d_rates, grid, price)
            display = df * 100

            fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
            ax.set_facecolor(CARD)

            vals = display.values.astype(float)
            vmax = min(abs(float(np.nanmax(vals))), 200)

            sns.heatmap(
                display, annot=True, fmt=".0f", cmap="RdYlGn",
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor="#0f1117",
                ax=ax, cbar_kws={"label": "Upside %", "shrink": 0.8},
                annot_kws={"size": 8, "color": "white"},
            )
            ax.set_title("Sensitivity: Upside % vs Market Price", color=TEXT, fontsize=11, pad=10)
            ax.set_xlabel("WACC", color=MUTED)
            ax.set_ylabel("FCF Growth Rate", color=MUTED)
            ax.tick_params(colors=MUTED)
            fig.tight_layout(pad=1.5)
            charts["sensitivity"] = _fig_to_b64(fig)
        except Exception as e:
            logger.debug(f"Sensitivity chart failed: {e}")

    return charts


def _screener_chart_b64(candidates):
    BG = "#0f1117"
    CARD = "#1a1d27"
    GREEN = "#22c55e"
    MUTED = "#94a3b8"
    TEXT = "#e2e8f0"

    import numpy as np
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "font.family": "monospace",
    })

    tickers = [c.ticker for c in candidates]
    scores  = [c.priority_score for c in candidates]
    gaps    = [c.valuation_gap for c in candidates]

    norm = [(g - min(gaps)) / (max(gaps) - min(gaps) + 1e-9) for g in gaps]
    colors = [plt.cm.RdYlGn(0.3 + 0.7 * n) for n in norm]

    fig, ax = plt.subplots(figsize=(8, max(3, len(candidates) * 0.55)), facecolor=BG)
    ax.set_facecolor(CARD)
    bars = ax.barh(tickers, scores, color=colors, alpha=0.88, height=0.6)

    for bar, gap, score in zip(bars, gaps, scores):
        ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                f"Gap: {gap:+.0%}", va="center", fontsize=9, color=TEXT)

    ax.set_title("Research Candidates — Priority Score", color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel("Priority Score (0-100)", color=MUTED)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    fig.tight_layout(pad=1.5)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Porcell DCF Engine — Web UI")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
