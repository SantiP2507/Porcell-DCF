"""
supabase/client.py — Supabase persistence layer (stubs).

All functions here are fully implemented EXCEPT the actual Supabase calls,
which are marked with clear TODO comments. Once you create your tables and
set SUPABASE_URL + SUPABASE_KEY in .env, uncomment the supabase-python calls.

Tables expected (create yourself — see README.md):
  - valuations
  - research_candidates
  - market_snapshots (optional)

This module silently no-ops if Supabase credentials are not configured,
so the rest of the system works without persistence.
"""

import json
import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any

from config import SUPABASE_URL, SUPABASE_KEY
from data.models import ValuationSummary, ResearchCandidate, DCFResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

_client = None  # Lazy-initialized on first use


def _get_client():
    """
    Return a Supabase client, initializing it on first call.
    Returns None if credentials are not configured.
    """
    global _client

    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.debug("Supabase credentials not set — persistence disabled.")
        return None

    try:
        # TODO: Uncomment when supabase-python is installed:
        # from supabase import create_client, Client
        # _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # logger.info("Supabase client initialized.")
        # return _client

        # Placeholder: return a mock that logs all calls
        _client = _MockSupabaseClient(SUPABASE_URL)
        logger.info(f"Supabase mock client active (URL: {SUPABASE_URL[:30]}...).")
        return _client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: VALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_valuation(summary: ValuationSummary) -> bool:
    """
    Persist a full DCF ValuationSummary to the `valuations` table.

    Saves one row per scenario (bear, base, bull) with all assumptions in JSON.
    Returns True on success, False on failure or if persistence is disabled.
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

        row = {
            "ticker": summary.ticker,
            "date": date.today().isoformat(),
            "scenario": scenario_name,
            "bear_value": summary.bear.fair_value_per_share if summary.bear else None,
            "base_value": summary.base.fair_value_per_share if summary.base else None,
            "bull_value": summary.bull.fair_value_per_share if summary.bull else None,
            "market_price": summary.market_price,
            "terminal_value_pct": result.terminal_value_pct,
            "assumptions": json.dumps({
                "fcf_growth_rate": result.fcf_growth_rate,
                "discount_rate": result.discount_rate,
                "terminal_growth_rate": result.terminal_growth_rate,
                "projection_years": result.projection_years,
                "base_fcf_m": result.base_fcf,
            }),
            "created_at": datetime.utcnow().isoformat(),
        }
        rows.append(row)

    if not rows:
        return False

    try:
        # TODO: Replace mock call with real Supabase upsert:
        # response = client.table("valuations").upsert(rows).execute()
        # return len(response.data) > 0

        client.table("valuations").upsert(rows).execute()  # mock
        logger.info(f"Saved {len(rows)} valuation rows for {summary.ticker}.")
        return True

    except Exception as e:
        logger.error(f"Failed to save valuation for {summary.ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: RESEARCH CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def save_research_candidates(candidates: List[ResearchCandidate]) -> bool:
    """
    Persist research candidates to the `research_candidates` table.
    Called once per daily screener run.
    """
    client = _get_client()
    if client is None:
        return False

    if not candidates:
        return True

    rows = [
        {
            "ticker": c.ticker,
            "date": c.date.isoformat(),
            "priority": round(c.priority_score),
            "reason": c.reason,
            "valuation_gap": c.valuation_gap,
            "bear_upside": c.bear_upside,
            "base_upside": c.base_upside,
            "bull_upside": c.bull_upside,
            "created_at": datetime.utcnow().isoformat(),
        }
        for c in candidates
    ]

    try:
        # TODO: Replace with:
        # response = client.table("research_candidates").insert(rows).execute()
        # return len(response.data) > 0

        client.table("research_candidates").insert(rows).execute()  # mock
        logger.info(f"Saved {len(rows)} research candidates.")
        return True

    except Exception as e:
        logger.error(f"Failed to save research candidates: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: MARKET SNAPSHOTS (optional)
# ─────────────────────────────────────────────────────────────────────────────

def save_market_snapshot(summary: ValuationSummary) -> bool:
    """
    Save a lightweight market snapshot (price, FCF, gap) to `market_snapshots`.
    Optional — useful for tracking how the gap evolves over time.
    """
    client = _get_client()
    if client is None:
        return False

    fcf = summary.financials.trailing_fcf if summary.financials else None
    gap = summary.upside_base

    row = {
        "ticker": summary.ticker,
        "date": date.today().isoformat(),
        "price": summary.market_price,
        "fcf": fcf,
        "valuation_gap": gap,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        # TODO: Replace with:
        # client.table("market_snapshots").insert(row).execute()

        client.table("market_snapshots").insert(row).execute()  # mock
        return True

    except Exception as e:
        logger.error(f"Failed to save market snapshot for {summary.ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# READ: HISTORICAL VALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_valuation_history(ticker: str, limit: int = 90) -> List[Dict[str, Any]]:
    """
    Load historical valuation records for a ticker (last N rows).
    Useful for tracking how your DCF estimate has changed over time.

    Returns:
        List of dicts from the `valuations` table, or empty list if unavailable.
    """
    client = _get_client()
    if client is None:
        return []

    try:
        # TODO: Replace with:
        # response = (
        #     client.table("valuations")
        #     .select("*")
        #     .eq("ticker", ticker)
        #     .eq("scenario", "base")
        #     .order("date", desc=True)
        #     .limit(limit)
        #     .execute()
        # )
        # return response.data or []

        return client.table("valuations").select("*").eq("ticker", ticker).execute()  # mock

    except Exception as e:
        logger.error(f"Failed to load valuation history for {ticker}: {e}")
        return []


def load_recent_candidates(days: int = 7) -> List[Dict[str, Any]]:
    """
    Load research candidates from the last N days.
    Useful for reviewing what the screener has been flagging lately.
    """
    client = _get_client()
    if client is None:
        return []

    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        # TODO: Replace with:
        # response = (
        #     client.table("research_candidates")
        #     .select("*")
        #     .gte("date", cutoff)
        #     .order("date", desc=True)
        #     .order("priority", desc=True)
        #     .execute()
        # )
        # return response.data or []

        return client.table("research_candidates").select("*").gte("date", cutoff).execute()  # mock

    except Exception as e:
        logger.error(f"Failed to load recent candidates: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLIENT (used until real Supabase credentials are plugged in)
# ─────────────────────────────────────────────────────────────────────────────

class _MockSupabaseClient:
    """
    Lightweight mock that mimics the supabase-python chainable API.
    Logs all operations instead of actually writing to a database.
    Replace with real supabase.create_client() when ready.
    """

    def __init__(self, url: str):
        self._url = url
        self._table_name = None
        self._operation = None

    def table(self, name: str) -> "_MockSupabaseClient":
        self._table_name = name
        return self

    def upsert(self, data) -> "_MockSupabaseClient":
        self._operation = ("upsert", data)
        return self

    def insert(self, data) -> "_MockSupabaseClient":
        self._operation = ("insert", data)
        return self

    def select(self, *args) -> "_MockSupabaseClient":
        self._operation = ("select", args)
        return self

    def eq(self, col, val) -> "_MockSupabaseClient":
        return self

    def gte(self, col, val) -> "_MockSupabaseClient":
        return self

    def order(self, col, **kwargs) -> "_MockSupabaseClient":
        return self

    def limit(self, n) -> "_MockSupabaseClient":
        return self

    def execute(self):
        """Log the operation and return empty data (mock)."""
        op, payload = self._operation or ("unknown", None)
        n = len(payload) if isinstance(payload, list) else 1
        logger.debug(
            f"[MOCK Supabase] {op.upper()} → table='{self._table_name}' "
            f"rows={n}. (No real write — configure SUPABASE_URL/KEY to persist.)"
        )
        return []  # Real client returns response.data
