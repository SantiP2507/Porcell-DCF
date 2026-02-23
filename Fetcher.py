"""
data/fetcher.py — Market and financial data ingestion.

Primary source: yfinance (free, no API key).
Fallback: Alpha Vantage (requires free API key in .env).

Design notes:
- All monetary values are normalized to USD millions before returning.
- Missing / None values from yfinance are handled explicitly — never silently.
- The fetcher returns typed dataclasses (FinancialSnapshot, MarketData),
  so the rest of the system never touches raw yfinance dicts.
"""

import logging
from datetime import date
from typing import Optional, Tuple
import requests

from data.models import MarketData, FinancialSnapshot
from config import ALPHA_VANTAGE_KEY, ALPHA_VANTAGE_BASE_URL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all(ticker: str) -> Tuple[MarketData, FinancialSnapshot]:
    """
    Fetch both market and financial data for a ticker.
    Tries yfinance first; falls back to Alpha Vantage if configured.

    Returns:
        (MarketData, FinancialSnapshot) tuple.
    Raises:
        ValueError if data is insufficient to run a DCF.
    """
    ticker = ticker.upper().strip()
    logger.info(f"Fetching data for {ticker}...")

    try:
        market_data = _fetch_market_data_yf(ticker)
        financials = _fetch_financials_yf(ticker)
        logger.info(f"{ticker}: yfinance fetch successful.")
        return market_data, financials

    except Exception as e:
        logger.warning(f"{ticker}: yfinance failed ({e}). Trying Alpha Vantage fallback...")

        if not ALPHA_VANTAGE_KEY:
            raise ValueError(
                f"yfinance failed for {ticker} and no ALPHA_VANTAGE_KEY is set. "
                "Cannot fetch data."
            ) from e

        try:
            market_data = _fetch_market_data_av(ticker)
            financials = _fetch_financials_av(ticker)
            logger.info(f"{ticker}: Alpha Vantage fetch successful.")
            return market_data, financials
        except Exception as e2:
            raise ValueError(
                f"Both data sources failed for {ticker}. "
                f"yfinance: {e} | Alpha Vantage: {e2}"
            ) from e2


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_market_data_yf(ticker: str) -> MarketData:
    """Fetch current price, shares outstanding, and market cap via yfinance."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info

    # yfinance key names vary slightly by ticker — try multiple candidates
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    shares = info.get("sharesOutstanding")  # raw count
    market_cap_raw = info.get("marketCap")  # raw USD

    if price is None:
        raise ValueError(f"Could not get price for {ticker} from yfinance.")
    if shares is None or shares == 0:
        raise ValueError(f"Could not get shares outstanding for {ticker}.")

    shares_millions = shares / 1_000_000
    market_cap_millions = (market_cap_raw or price * shares) / 1_000_000

    return MarketData(
        ticker=ticker,
        price=float(price),
        shares_outstanding=float(shares_millions),
        market_cap=float(market_cap_millions),
        as_of_date=date.today(),
    )


def _fetch_financials_yf(ticker: str) -> FinancialSnapshot:
    """
    Fetch income statement, balance sheet, and cash flow statement via yfinance.
    Extracts trailing FCF, historical annual FCFs, cash, and debt.
    """
    import yfinance as yf
    import pandas as pd

    t = yf.Ticker(ticker)

    # ── Cash Flow Statement ───────────────────────────────────────────────
    # yfinance returns annual cashflow as a DataFrame: rows = line items, cols = years
    cf = t.cashflow          # annual (most recent 4 years)
    cf_q = t.quarterly_cashflow  # quarterly (recent 4 quarters)

    if cf is None or cf.empty:
        raise ValueError(f"No cash flow data for {ticker}.")

    def _get_row(df, *candidates):
        """Try multiple row name candidates (yfinance naming is inconsistent)."""
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return None

    # Operating cash flow
    ocf_row = _get_row(cf,
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash From Operating Activities",
    )
    # Capital expenditures (usually negative in yfinance)
    capex_row = _get_row(cf,
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of Plant And Equipment",
    )

    if ocf_row is None:
        raise ValueError(f"Cannot find operating cash flow for {ticker}.")
    if capex_row is None:
        raise ValueError(f"Cannot find capex for {ticker}.")

    # Historical annual FCF (sorted oldest → newest)
    # Columns are datetime index, newest first — so we reverse
    years = list(reversed(cf.columns))
    historical_fcf = []
    for col in years:
        ocf = _safe_float(ocf_row.get(col))
        capex = _safe_float(capex_row.get(col))
        if ocf is not None and capex is not None:
            # capex is stored as negative in yfinance — FCF = OCF + capex
            historical_fcf.append((ocf + capex) / 1_000_000)  # → millions

    if not historical_fcf:
        raise ValueError(f"Could not compute historical FCF for {ticker}.")

    # TTM FCF: sum last 4 quarters if available, else use most recent annual
    ttm_fcf = _compute_ttm_fcf(cf_q) or historical_fcf[-1]

    # ── Balance Sheet ─────────────────────────────────────────────────────
    bs = t.balance_sheet
    cash = 0.0
    total_debt = 0.0

    if bs is not None and not bs.empty:
        cash_row = _get_row(bs,
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        )
        debt_row = _get_row(bs,
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt",
        )
        if cash_row is not None:
            # Use most recent column (first)
            cash = (_safe_float(cash_row.iloc[0]) or 0.0) / 1_000_000
        if debt_row is not None:
            total_debt = (_safe_float(debt_row.iloc[0]) or 0.0) / 1_000_000

        # Also add short-term debt if available separately
        st_debt_row = _get_row(bs, "Current Debt", "Short Term Debt", "Current Portion Of Long Term Debt")
        if st_debt_row is not None:
            total_debt += (_safe_float(st_debt_row.iloc[0]) or 0.0) / 1_000_000

    # ── Income Statement ──────────────────────────────────────────────────
    inc = t.income_stmt
    revenue_ttm = 0.0
    op_income_ttm = 0.0

    if inc is not None and not inc.empty:
        rev_row = _get_row(inc, "Total Revenue", "Revenue")
        op_row = _get_row(inc, "Operating Income", "Ebit")
        if rev_row is not None:
            revenue_ttm = (_safe_float(rev_row.iloc[0]) or 0.0) / 1_000_000
        if op_row is not None:
            op_income_ttm = (_safe_float(op_row.iloc[0]) or 0.0) / 1_000_000

    # ── Metadata ──────────────────────────────────────────────────────────
    info = t.info
    fiscal_year_end = info.get("fiscalYearEnd") or info.get("lastFiscalYearEnd")

    return FinancialSnapshot(
        ticker=ticker,
        trailing_fcf=ttm_fcf,
        historical_fcf=historical_fcf,
        cash_and_equivalents=cash,
        total_debt=total_debt,
        revenue_ttm=revenue_ttm,
        operating_income_ttm=op_income_ttm,
        fiscal_year_end=str(fiscal_year_end) if fiscal_year_end else None,
        currency="USD",
        as_of_date=date.today(),
    )


def _compute_ttm_fcf(quarterly_cf) -> Optional[float]:
    """
    Attempt to compute TTM FCF from last 4 quarterly filings.
    Returns None if quarterly data is unavailable or incomplete.
    """
    if quarterly_cf is None or quarterly_cf.empty:
        return None

    try:
        ocf_row = None
        capex_row = None
        for name in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
            if name in quarterly_cf.index:
                ocf_row = quarterly_cf.loc[name]
                break
        for name in ["Capital Expenditure", "Capital Expenditures"]:
            if name in quarterly_cf.index:
                capex_row = quarterly_cf.loc[name]
                break

        if ocf_row is None or capex_row is None:
            return None

        # Sum last 4 quarters (columns are newest-first)
        cols = quarterly_cf.columns[:4]
        total_ocf = sum(_safe_float(ocf_row.get(c)) or 0 for c in cols)
        total_capex = sum(_safe_float(capex_row.get(c)) or 0 for c in cols)
        ttm = (total_ocf + total_capex) / 1_000_000  # → millions

        return ttm if ttm != 0 else None
    except Exception as e:
        logger.debug(f"TTM FCF computation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA VANTAGE FALLBACK IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_market_data_av(ticker: str) -> MarketData:
    """Fetch price and basic market data from Alpha Vantage GLOBAL_QUOTE."""
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY,
    }
    r = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("Global Quote", {})

    price = float(data.get("05. price") or 0)
    if price == 0:
        raise ValueError(f"Alpha Vantage returned no price for {ticker}.")

    # Alpha Vantage doesn't return shares outstanding in GLOBAL_QUOTE.
    # We attempt to get it from the OVERVIEW endpoint.
    params2 = {"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHA_VANTAGE_KEY}
    r2 = requests.get(ALPHA_VANTAGE_BASE_URL, params=params2, timeout=10)
    overview = r2.json() if r2.ok else {}

    shares_raw = float(overview.get("SharesOutstanding") or 0)
    market_cap_raw = float(overview.get("MarketCapitalization") or 0)

    if shares_raw == 0:
        raise ValueError(f"Alpha Vantage returned no shares outstanding for {ticker}.")

    return MarketData(
        ticker=ticker,
        price=price,
        shares_outstanding=shares_raw / 1_000_000,
        market_cap=market_cap_raw / 1_000_000,
        as_of_date=date.today(),
    )


def _fetch_financials_av(ticker: str) -> FinancialSnapshot:
    """
    Fetch cash flow, balance sheet, and income data from Alpha Vantage.
    Note: Alpha Vantage free tier has rate limits (5 req/min, 500/day).
    """
    def _av_get(function: str) -> dict:
        r = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params={"function": function, "symbol": ticker, "apikey": ALPHA_VANTAGE_KEY},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    cf_data = _av_get("CASH_FLOW")
    bs_data = _av_get("BALANCE_SHEET")
    inc_data = _av_get("INCOME_STATEMENT")

    annual_reports_cf = cf_data.get("annualReports", [])
    if not annual_reports_cf:
        raise ValueError(f"No Alpha Vantage cash flow data for {ticker}.")

    historical_fcf = []
    for report in reversed(annual_reports_cf):  # oldest → newest
        ocf = _safe_float(report.get("operatingCashflow"))
        capex = _safe_float(report.get("capitalExpenditures"))
        if ocf is not None and capex is not None:
            historical_fcf.append((ocf - capex) / 1_000_000)  # AV capex is positive

    ttm_fcf = historical_fcf[-1] if historical_fcf else 0.0

    # Balance sheet — most recent annual
    bs_latest = (bs_data.get("annualReports") or [{}])[0]
    cash = (_safe_float(bs_latest.get("cashAndCashEquivalentsAtCarryingValue")) or 0) / 1_000_000
    st_inv = (_safe_float(bs_latest.get("shortTermInvestments")) or 0) / 1_000_000
    lt_debt = (_safe_float(bs_latest.get("longTermDebt")) or 0) / 1_000_000
    st_debt = (_safe_float(bs_latest.get("shortLongTermDebtTotal")) or 0) / 1_000_000
    total_debt = max(lt_debt, st_debt)  # avoid double-counting

    # Income statement — most recent annual
    inc_latest = (inc_data.get("annualReports") or [{}])[0]
    revenue = (_safe_float(inc_latest.get("totalRevenue")) or 0) / 1_000_000
    op_income = (_safe_float(inc_latest.get("operatingIncome")) or 0) / 1_000_000

    return FinancialSnapshot(
        ticker=ticker,
        trailing_fcf=ttm_fcf,
        historical_fcf=historical_fcf,
        cash_and_equivalents=cash + st_inv,
        total_debt=total_debt,
        revenue_ttm=revenue,
        operating_income_ttm=op_income,
        as_of_date=date.today(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None if not numeric."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if not (f != f) else None  # guard against NaN
    except (TypeError, ValueError):
        return None


def validate_financials(f: FinancialSnapshot) -> list[str]:
    """
    Return a list of warnings about data quality issues.
    Empty list = data is clean enough to model.
    """
    warnings = []

    if f.trailing_fcf <= 0:
        warnings.append(
            f"Trailing FCF is negative (${f.trailing_fcf:.1f}M). "
            "DCF is unreliable for FCF-negative businesses — interpret with extreme caution."
        )

    if len(f.historical_fcf) < 3:
        warnings.append(
            f"Only {len(f.historical_fcf)} years of FCF history. "
            "Need at least 3 for meaningful stability assessment."
        )

    if f.total_debt > 0 and f.trailing_fcf > 0:
        leverage = f.net_debt / f.trailing_fcf
        if leverage > 8:
            warnings.append(
                f"High leverage: Net Debt / FCF = {leverage:.1f}x. "
                "Equity value is highly sensitive to debt assumptions."
            )

    return warnings
