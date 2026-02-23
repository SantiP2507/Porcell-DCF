"""
research/screener.py — Daily stock research recommendation engine.

ML-enhanced version. Hard filters are always rule-based (transparent).
Priority scoring uses ML prioritizer when available, rule-based as fallback.
Each candidate also gets a stability assessment and cluster label.
"""

import logging
import statistics
from datetime import date
from typing import List, Optional

from data.models import ValuationSummary, ResearchCandidate, FinancialSnapshot
from config import (
    MIN_MARGIN_OF_SAFETY, MAX_LEVERAGE_RATIO, MIN_FCF_YIELD,
    SCORING_WEIGHTS, SCREENER_TOP_N,
)

logger = logging.getLogger(__name__)


def screen_candidates(
    valuations: List[ValuationSummary],
    top_n: int = SCREENER_TOP_N,
    use_ml: bool = True,
) -> List[ResearchCandidate]:
    """Screen and rank research candidates. Returns top_n sorted by priority."""
    prioritizer = stability_model = clustering_model = None

    if use_ml:
        try:
            from ml.prioritizer import load_or_train
            from ml.stability import load_or_train as load_stability
            from ml.clustering import load_or_train as load_clustering
            prioritizer      = load_or_train()
            stability_model  = load_stability()
            clustering_model = load_clustering()
            logger.info("ML models loaded.")
        except Exception as e:
            logger.warning(f"ML unavailable, using rule-based: {e}")

    candidates = []
    for summary in valuations:
        c = _evaluate_single(summary, prioritizer, stability_model, clustering_model)
        if c is not None:
            candidates.append(c)

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    mode = "ML" if prioritizer else "rule-based"
    logger.info(f"Screened {len(valuations)} tickers ({mode}) -> {len(candidates)} candidates.")
    return candidates[:top_n]


def _evaluate_single(summary, prioritizer, stability_model, clustering_model):
    ticker     = summary.ticker
    financials = summary.financials
    base       = summary.base

    # Hard filters
    if base is None or not base.is_valid:
        return None
    if financials is None:
        return None
    if financials.trailing_fcf <= 0:
        logger.debug(f"{ticker}: negative FCF")
        return None

    md = summary.market_data
    if md and md.market_cap > 0:
        if financials.trailing_fcf / md.market_cap < MIN_FCF_YIELD:
            logger.debug(f"{ticker}: FCF yield too low")
            return None

    upside_base = summary.upside_base
    if upside_base is None or upside_base < MIN_MARGIN_OF_SAFETY:
        logger.debug(f"{ticker}: insufficient upside {upside_base}")
        return None

    if financials.net_debt > 0 and financials.trailing_fcf > 0:
        if financials.net_debt / financials.trailing_fcf > MAX_LEVERAGE_RATIO:
            logger.debug(f"{ticker}: leverage too high")
            return None

    # Feature vector for ML
    feature_vector = None
    try:
        from ml.features import extract_features
        feature_vector = extract_features(summary)
    except Exception:
        pass

    # Priority score
    if prioritizer and feature_vector is not None:
        priority_score = prioritizer.score(feature_vector)
        scoring_mode = "ml"
    else:
        priority_score = _rule_based_score(summary, financials) * 100
        scoring_mode = "rule"

    # Stability
    stability_result = {"is_stable": True, "confidence": 0.5, "warnings": [], "stability_score": 50}
    if stability_model and feature_vector is not None:
        try:
            stability_result = stability_model.predict(feature_vector)
        except Exception:
            pass

    # Cluster
    cluster_result = {"cluster_id": -1, "archetype": "Unknown", "description": ""}
    if clustering_model and feature_vector is not None:
        try:
            cluster_result = clustering_model.predict(feature_vector)
        except Exception:
            pass

    reason = _build_reason(ticker, upside_base, summary, financials,
                           stability_result, cluster_result, scoring_mode)

    return ResearchCandidate(
        ticker=ticker,
        date=date.today(),
        priority_score=round(priority_score, 1),
        valuation_gap=upside_base,
        reason=reason,
        bear_upside=summary.upside_bear or 0.0,
        base_upside=upside_base,
        bull_upside=summary.upside_bull or 0.0,
        fcf_stability_score=_score_fcf_stability(financials.historical_fcf),
        leverage_score=_score_leverage(financials),
        valuation_summary=summary,
    )


def _rule_based_score(summary, financials):
    w = SCORING_WEIGHTS
    return (
        w["valuation_gap"]          * _score_valuation_gap(summary.upside_base or 0)
        + w["fcf_stability"]        * _score_fcf_stability(financials.historical_fcf)
        + w["leverage_score"]       * _score_leverage(financials)
        + w["valuation_robustness"] * _score_robustness(summary)
    )


def _score_valuation_gap(upside):
    if upside <= MIN_MARGIN_OF_SAFETY:
        return 0.0
    return min((upside - MIN_MARGIN_OF_SAFETY) / (1.0 - MIN_MARGIN_OF_SAFETY), 1.0)


def _score_fcf_stability(historical_fcf):
    if len(historical_fcf) < 2:
        return 0.5
    mean_fcf = statistics.mean(historical_fcf)
    if mean_fcf <= 0:
        return 0.0
    cv = statistics.stdev(historical_fcf) / abs(mean_fcf)
    neg_penalty = sum(1 for f in historical_fcf if f < 0) / len(historical_fcf) * 0.3
    return max(0.0, max(0.0, 1.0 - cv) - neg_penalty)


def _score_leverage(financials):
    if financials.net_debt <= 0:
        return min(1.0, 0.9 + abs(financials.net_debt) / max(financials.trailing_fcf, 1) * 0.1)
    if financials.trailing_fcf <= 0:
        return 0.0
    return max(0.0, 0.9 - (financials.net_debt / financials.trailing_fcf / MAX_LEVERAGE_RATIO) * 0.9)


def _score_robustness(summary):
    bear = summary.upside_bear
    if bear is None:
        return 0.3
    if bear >= 0.10:
        return 1.0
    elif bear >= 0:
        return 0.6
    elif bear >= -0.20:
        return 0.3
    return 0.0


def _build_reason(ticker, upside_base, summary, financials,
                  stability_result, cluster_result, scoring_mode):
    parts = [f"Base upside: {upside_base:+.0%}"]

    bear = summary.upside_bear
    if bear is not None:
        parts.append(f"bear: {bear:+.0%}")

    archetype = cluster_result.get("archetype", "")
    if archetype and archetype != "Unknown":
        parts.append(f"archetype: {archetype}")

    if not stability_result.get("is_stable", True):
        parts.append("valuation unstable")
    elif stability_result.get("stability_score", 50) > 70:
        parts.append("stable valuation")

    hist = financials.historical_fcf
    if len(hist) >= 3:
        if _score_fcf_stability(hist) >= 0.70:
            parts.append("consistent FCF")
        elif _score_fcf_stability(hist) < 0.40:
            parts.append("volatile FCF")

    if financials.net_debt <= 0:
        parts.append(f"net cash ${abs(financials.net_debt):.0f}M")
    elif financials.trailing_fcf > 0:
        lev = financials.net_debt / financials.trailing_fcf
        if lev < 2:
            parts.append(f"low leverage {lev:.1f}x")
        elif lev >= 5:
            parts.append(f"high leverage {lev:.1f}x")

    if scoring_mode == "ml":
        parts.append("ML-scored")

    return "; ".join(parts).capitalize()


def print_screener_results(candidates):
    if not candidates:
        print("\n  No candidates met screening criteria today.\n")
        return

    today = date.today().strftime("%Y-%m-%d")
    print(f"\n{'='*80}")
    print(f"  DAILY RESEARCH CANDIDATES — {today}")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Ticker':<8} {'Score':>6} {'Bear':>7} {'Base':>7} {'Bull':>7}  Reason")
    print(f"  {'-'*74}")

    for rank, c in enumerate(candidates, start=1):
        print(
            f"  {rank:<3} {c.ticker:<8} {c.priority_score:>6.1f} "
            f"{c.bear_upside:>+6.0%} {c.base_upside:>+6.0%} {c.bull_upside:>+6.0%}  "
            f"{c.reason[:52]}"
        )

    print(f"{'='*80}")
    print(f"\n  Filters: >{MIN_MARGIN_OF_SAFETY:.0%} upside | FCF+ | Leverage <={MAX_LEVERAGE_RATIO:.0f}x")
    print(f"  These are research candidates, not buy signals.\n")
