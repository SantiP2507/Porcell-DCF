"""
supabase/client.py — Supabase persistence layer.

Real implementation using supabase-python.
Install: pip install supabase

Set in .env:
  SUPABASE_URL=https://yourproject.supabase.co
  SUPABASE_KEY=your_anon_or_service_key

If credentials are missing, all operations silently no-op so the
rest of the system keeps working without persistence.

Tables (create these in Supabase — see README.md for SQL):
  - valuations
  - research_candidates
  - market_snapshots
  - ml_labels (optional — for future outcome tracking)
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from config import SUPABASE_URL, SUPABASE_KEY
from data.models import ValuationSummary, ResearchCandidate

logger = logging.getLogger(__name__)

_client = None


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """Return Supabase client, initializing on first call. None if unconfigured."""
    global _client

    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.debug("Supabase credentials not set — persistence disabled.")
        return None

    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client connected.")
        return _client
    except ImportError:
        logger.error(
            "supabase-python not installed. Run: pip install supabase\n"
            "Persistence is disabled until installed."
        )
        return None
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: VALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_valuation(summary: ValuationSummary) -> bool:
    """
    Save bear/base/bull DCF results to the valuations table.
    One row per scenario. Uses upsert to avoid duplicates on the same day.
    """
    client = _get_client()
    if client is None:
        return False

    rows = []
    for scenario_name, result in [
        ("bear", summary.bear),
        ("base", summary.base),
        ("bull", summary.bull),
    ]:
        if result is None or not result.is_valid:
            continue

        rows.append({
            "ticker":             summary.ticker,
            "date":               date.today().isoformat(),
            "scenario":           scenario_name,
            "bear_value":         summary.bear.fair_value_per_share if summary.bear else None,
            "base_value":         summary.base.fair_value_per_share if summary.base else None,
            "bull_value":         summary.bull.fair_value_per_share if summary.bull else None,
            "market_price":       summary.market_price,
            "terminal_value_pct": result.terminal_value_pct,
            "assumptions":        json.dumps({
                "fcf_growth_rate":      result.fcf_growth_rate,
                "discount_rate":        result.discount_rate,
                "terminal_growth_rate": result.terminal_growth_rate,
                "projection_years":     result.projection_years,
                "base_fcf_m":           result.base_fcf,
            }),
            "created_at":         datetime.utcnow().isoformat(),
        })

    if not rows:
        return False

    try:
        client.table("valuations").upsert(rows, on_conflict="ticker,date,scenario").execute()
        logger.info(f"Saved {len(rows)} valuation rows for {summary.ticker}.")
        return True
    except Exception as e:
        logger.error(f"Failed to save valuation for {summary.ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: RESEARCH CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def save_research_candidates(candidates: List[ResearchCandidate]) -> bool:
    """Save daily screener results to research_candidates."""
    client = _get_client()
    if client is None:
        return False

    if not candidates:
        return True

    rows = [
        {
            "ticker":        c.ticker,
            "date":          c.date.isoformat(),
            "priority":      round(c.priority_score),
            "reason":        c.reason,
            "valuation_gap": c.valuation_gap,
            "bear_upside":   c.bear_upside,
            "base_upside":   c.base_upside,
            "bull_upside":   c.bull_upside,
            "created_at":    datetime.utcnow().isoformat(),
        }
        for c in candidates
    ]

    try:
        client.table("research_candidates").insert(rows).execute()
        logger.info(f"Saved {len(rows)} research candidates.")
        return True
    except Exception as e:
        logger.error(f"Failed to save research candidates: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: MARKET SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

def save_market_snapshot(summary: ValuationSummary) -> bool:
    """Save lightweight price + gap snapshot to market_snapshots."""
    client = _get_client()
    if client is None:
        return False

    row = {
        "ticker":        summary.ticker,
        "date":          date.today().isoformat(),
        "price":         summary.market_price,
        "fcf":           summary.financials.trailing_fcf if summary.financials else None,
        "valuation_gap": summary.upside_base,
        "created_at":    datetime.utcnow().isoformat(),
    }

    try:
        client.table("market_snapshots").upsert(row, on_conflict="ticker,date").execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save market snapshot for {summary.ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: ML LABELS
# ─────────────────────────────────────────────────────────────────────────────

def save_ml_label(
    ticker: str,
    run_date: date,
    label_type: str,
    label_value: int,
    notes: str = "",
) -> bool:
    """
    Save a training label for ML model improvement.

    label_type:  "prioritization" or "stability"
    label_value: 1 (positive outcome) or 0 (negative outcome)

    Call this manually after observing whether a flagged stock was
    actually worth investigating — this is how the ML improves over time.
    """
    client = _get_client()
    if client is None:
        return False

    row = {
        "ticker":      ticker,
        "run_date":    run_date.isoformat(),
        "label_type":  label_type,
        "label_value": label_value,
        "notes":       notes,
        "created_at":  datetime.utcnow().isoformat(),
    }

    try:
        client.table("ml_labels").insert(row).execute()
        logger.info(f"Saved ML label for {ticker} ({label_type}={label_value}).")
        return True
    except Exception as e:
        logger.error(f"Failed to save ML label: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# READ: HISTORICAL VALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_valuation_history(ticker: str, limit: int = 90) -> List[Dict[str, Any]]:
    """Load historical base-case valuations for a ticker."""
    client = _get_client()
    if client is None:
        return []

    try:
        response = (
            client.table("valuations")
            .select("date, base_value, bear_value, bull_value, market_price, terminal_value_pct, assumptions")
            .eq("ticker", ticker)
            .eq("scenario", "base")
            .order("date", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to load valuation history for {ticker}: {e}")
        return []


def load_all_valuations_for_training(limit: int = 5000) -> List[Dict[str, Any]]:
    """Load all base-case valuations for ML training. Called by ml/trainer.py."""
    client = _get_client()
    if client is None:
        return []

    try:
        response = (
            client.table("valuations")
            .select("*")
            .eq("scenario", "base")
            .order("date", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to load valuations for training: {e}")
        return []


def load_recent_candidates(days: int = 7) -> List[Dict[str, Any]]:
    """Load research candidates from the last N days."""
    client = _get_client()
    if client is None:
        return []

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        response = (
            client.table("research_candidates")
            .select("*")
            .gte("date", cutoff)
            .order("date", desc=True)
            .order("priority", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to load recent candidates: {e}")
        return []


def load_market_history(ticker: str, days: int = 180) -> List[Dict[str, Any]]:
    """Load price + gap history for a ticker."""
    client = _get_client()
    if client is None:
        return []

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        response = (
            client.table("market_snapshots")
            .select("date, price, fcf, valuation_gap")
            .eq("ticker", ticker)
            .gte("date", cutoff)
            .order("date", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to load market history for {ticker}: {e}")
        return []


def check_connection() -> bool:
    """Verify Supabase connection is working. Returns True if OK."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("valuations").select("id").limit(1).execute()
        logger.info("Supabase connection verified.")
        return True
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        return False
