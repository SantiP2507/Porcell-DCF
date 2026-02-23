"""
research/screener.py — Daily stock research recommendation engine.

This is NOT a buy/sell signal generator. It ranks stocks worth researching
based on four conservative, fundamental criteria:

  1. Valuation gap (base DCF upside vs market price) — biggest driver
  2. FCF stability (how consistent historical FCF has been)
  3. Leverage score (lower debt = safer, higher score)
  4. Valuation robustness (does even the bear case show upside?)

Scoring is rule-based, transparent, and reproducible. No ML yet.

Design philosophy:
  - A stock makes the list because it MIGHT be worth investigating — not because
    it's a guaranteed winner. The analyst (you) does the real work after this.
  - Conservative thresholds filter out obvious garbage: money-losers, highly
    leveraged companies, and those where even the bull case offers no upside.
  - The reason string explains exactly why each candidate scored as it did.
"""

import logging
import statistics
from datetime import date
from typing import List, Optional

from data.models import (
    FinancialSnapshot,
    MarketData,
    ValuationSummary,
    ResearchCandidate,
)
from config import (
    MIN_MARGIN_OF_SAFETY,
    MAX_LEVERAGE_RATIO,
    MIN_FCF_YIELD,
    SCORING_WEIGHTS,
    SCREENER_TOP_N,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def screen_candidates(
    valuations: List[ValuationSummary],
    top_n: int = SCREENER_TOP_N,
) -> List[ResearchCandidate]:
    """
    Take a list of pre-computed ValuationSummary objects and return
    ranked research candidates.

    Args:
        valuations: List of ValuationSummary from the DCF engine.
        top_n:      Return only the top N candidates.

    Returns:
        Sorted list of ResearchCandidate (highest priority first).
    """
    candidates = []

    for summary in valuations:
        candidate = _evaluate_single(summary)
        if candidate is not None:
            candidates.append(candidate)

    # Sort by priority score (descending)
    candidates.sort(key=lambda c: c.priority_score, reverse=True)

    logger.info(
        f"Screened {len(valuations)} tickers → {len(candidates)} candidates. "
        f"Returning top {min(top_n, len(candidates))}."
    )

    return candidates[:top_n]


def _evaluate_single(summary: ValuationSummary) -> Optional[ResearchCandidate]:
    """
    Evaluate a single ticker against all screening criteria.
    Returns a ResearchCandidate if it passes, None if it's filtered out.
    """
    ticker = summary.ticker
    price = summary.market_price
    financials = summary.financials
    market_data = summary.market_data
    base = summary.base

    # ── Hard Disqualification Filters ─────────────────────────────────────
    # These eliminate stocks that are simply un-modelable or too risky
    # to spend research time on without deeper qualitative work.

    if base is None or not base.is_valid:
        logger.debug(f"{ticker}: No valid base DCF — skipping.")
        return None

    if financials is None:
        logger.debug(f"{ticker}: No financial data — skipping.")
        return None

    # Filter: FCF must be positive (DCF doesn't apply to cash burners)
    if financials.trailing_fcf <= 0:
        logger.debug(f"{ticker}: Negative FCF ({financials.trailing_fcf:.1f}M) — skipping.")
        return None

    # Filter: Minimum FCF yield (FCF / market cap)
    if market_data and market_data.market_cap > 0:
        fcf_yield = financials.trailing_fcf / market_data.market_cap
        if fcf_yield < MIN_FCF_YIELD:
            logger.debug(f"{ticker}: FCF yield {fcf_yield:.2%} below threshold — skipping.")
            return None

    # Filter: Minimum base-case upside
    upside_base = summary.upside_base
    if upside_base is None or upside_base < MIN_MARGIN_OF_SAFETY:
        logger.debug(
            f"{ticker}: Base upside {upside_base:.1%} below minimum {MIN_MARGIN_OF_SAFETY:.0%} — skipping."
        )
        return None

    # Filter: Excessive leverage
    if financials.net_debt > 0 and financials.trailing_fcf > 0:
        leverage = financials.net_debt / financials.trailing_fcf
        if leverage > MAX_LEVERAGE_RATIO:
            logger.debug(f"{ticker}: Leverage {leverage:.1f}x exceeds maximum — skipping.")
            return None

    # ── Score Components ───────────────────────────────────────────────────

    valuation_gap_score = _score_valuation_gap(upside_base)
    fcf_stability_score = _score_fcf_stability(financials.historical_fcf)
    leverage_score = _score_leverage(financials)
    robustness_score = _score_valuation_robustness(summary)

    # ── Composite Score (0–100) ────────────────────────────────────────────
    weights = SCORING_WEIGHTS
    composite = (
        weights["valuation_gap"] * valuation_gap_score
        + weights["fcf_stability"] * fcf_stability_score
        + weights["leverage"] * leverage_score
        + weights["valuation_coverage"] * robustness_score
    ) * 100  # scale to 0–100

    # ── Reason String ──────────────────────────────────────────────────────
    reason = _build_reason(
        ticker, upside_base, summary, financials,
        valuation_gap_score, fcf_stability_score, leverage_score, robustness_score
    )

    return ResearchCandidate(
        ticker=ticker,
        date=date.today(),
        priority_score=round(composite, 1),
        valuation_gap=upside_base,
        reason=reason,
        bear_upside=summary.upside_bear or 0.0,
        base_upside=upside_base,
        bull_upside=summary.upside_bull or 0.0,
        fcf_stability_score=fcf_stability_score,
        leverage_score=leverage_score,
        valuation_summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS (each returns 0.0 – 1.0)
# ─────────────────────────────────────────────────────────────────────────────

def _score_valuation_gap(upside: float) -> float:
    """
    Score based on base-case DCF upside vs market price.
    0.25 upside (MIN_MARGIN_OF_SAFETY) → score 0.0 (just qualified)
    1.00 upside (100%) → score 1.0 (extremely cheap)

    Uses a linear scale from MIN_MOS to 100% upside.
    """
    if upside <= MIN_MARGIN_OF_SAFETY:
        return 0.0
    # Cap at 100% upside for scoring (prevent outliers from dominating)
    capped = min(upside, 1.0)
    return (capped - MIN_MARGIN_OF_SAFETY) / (1.0 - MIN_MARGIN_OF_SAFETY)


def _score_fcf_stability(historical_fcf: List[float]) -> float:
    """
    Score based on consistency of historical FCF.

    Metric: Coefficient of Variation (CV) = std / mean
    Lower CV = more stable = higher score.

    Also penalizes if any year had negative FCF (inconsistent generation).

    Returns 0.0 (erratic) to 1.0 (perfectly consistent).
    """
    if len(historical_fcf) < 2:
        # Not enough history — give benefit of the doubt, moderate score
        return 0.5

    mean_fcf = statistics.mean(historical_fcf)
    if mean_fcf <= 0:
        return 0.0  # Negative or zero mean FCF is bad

    stdev = statistics.stdev(historical_fcf)
    cv = stdev / abs(mean_fcf)  # higher CV = less stable

    # Penalize if any year was negative
    negative_years = sum(1 for f in historical_fcf if f < 0)
    negative_penalty = negative_years / len(historical_fcf) * 0.3

    # CV of 0 = perfectly stable, CV of 1.0+ = very volatile
    # Map: CV 0 → score 1.0, CV >= 1.0 → score 0.0
    stability = max(0.0, 1.0 - cv)
    return max(0.0, stability - negative_penalty)


def _score_leverage(financials: FinancialSnapshot) -> float:
    """
    Score based on leverage (Net Debt / FCF).
    Net cash position (negative net debt) → score 1.0
    No debt → score 0.9
    Leverage 8x (max) → score 0.0
    """
    if financials.net_debt <= 0:
        # Net cash position: safer, higher score
        # Extra credit for large net cash relative to FCF
        cash_to_fcf = abs(financials.net_debt) / max(financials.trailing_fcf, 1)
        return min(1.0, 0.9 + cash_to_fcf * 0.1)

    if financials.trailing_fcf <= 0:
        return 0.0  # Can't assess leverage without positive FCF

    leverage = financials.net_debt / financials.trailing_fcf
    # Linear scale: 0x leverage = 0.9 score, 8x = 0.0 score
    score = max(0.0, 0.9 - (leverage / MAX_LEVERAGE_RATIO) * 0.9)
    return score


def _score_valuation_robustness(summary: ValuationSummary) -> float:
    """
    Score based on whether the bear case also shows upside.
    If even the pessimistic scenario is above market price, that's robust.

    bear_upside > 0 → full credit
    bear_upside between -MOS and 0 → partial credit
    bear_upside < -MOS → zero credit (too risky)
    """
    bear_upside = summary.upside_bear
    if bear_upside is None:
        return 0.3  # No bear case — uncertain, give partial credit

    if bear_upside >= 0.10:
        return 1.0   # Bear case is solidly positive
    elif bear_upside >= 0:
        return 0.6   # Bear case marginally positive
    elif bear_upside >= -0.20:
        return 0.3   # Bear case modestly negative — acceptable margin
    else:
        return 0.0   # Bear case is deeply negative — too much risk


def _build_reason(
    ticker: str,
    upside_base: float,
    summary: ValuationSummary,
    financials: FinancialSnapshot,
    val_score: float,
    stability_score: float,
    lev_score: float,
    robust_score: float,
) -> str:
    """Build a human-readable reason string explaining the candidate's score."""
    parts = []

    # Valuation
    parts.append(f"Base DCF upside: {upside_base:+.0%}")

    # Bear case
    bear_up = summary.upside_bear
    if bear_up is not None:
        parts.append(
            f"bear case {'also positive' if bear_up > 0 else f'shows {bear_up:.0%}'}: "
            f"{bear_up:+.0%}"
        )

    # FCF quality
    if stability_score >= 0.7:
        parts.append("consistent FCF history")
    elif stability_score >= 0.4:
        parts.append("moderate FCF consistency")
    else:
        parts.append("volatile FCF — verify quality")

    # Leverage
    if financials.net_debt <= 0:
        parts.append(f"net cash position (${abs(financials.net_debt):.0f}M)")
    elif financials.trailing_fcf > 0:
        lev = financials.net_debt / financials.trailing_fcf
        if lev < 2:
            parts.append(f"low leverage ({lev:.1f}x)")
        elif lev < 5:
            parts.append(f"moderate leverage ({lev:.1f}x)")
        else:
            parts.append(f"elevated leverage ({lev:.1f}x) — monitor closely")

    return "; ".join(parts).capitalize()


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_screener_results(candidates: List[ResearchCandidate]) -> None:
    """Print a formatted table of research candidates to stdout."""
    if not candidates:
        print("\n  No candidates met screening criteria today.\n")
        return

    today = date.today().strftime("%Y-%m-%d")
    print(f"\n{'═'*78}")
    print(f"  📋  DAILY RESEARCH CANDIDATES — {today}")
    print(f"{'═'*78}")
    print(
        f"  {'#':<3} {'Ticker':<8} {'Score':>6} {'Bear':>7} {'Base':>7} {'Bull':>7}  Reason"
    )
    print(f"  {'─'*72}")

    for rank, c in enumerate(candidates, start=1):
        print(
            f"  {rank:<3} {c.ticker:<8} {c.priority_score:>6.1f} "
            f"{c.bear_upside:>+6.0%} {c.base_upside:>+6.0%} {c.bull_upside:>+6.0%}  "
            f"{c.reason[:50]}"
        )

    print(f"{'═'*78}")
    print(
        f"\n  Criteria: ≥{MIN_MARGIN_OF_SAFETY:.0%} base upside | "
        f"FCF+ | Leverage ≤{MAX_LEVERAGE_RATIO:.0f}x | FCF yield ≥{MIN_FCF_YIELD:.0%}"
    )
    print(f"  ⚠️  These are research candidates, not buy signals.\n")
