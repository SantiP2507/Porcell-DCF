"""
data/models.py — Typed data structures for all financial inputs.

Using dataclasses keeps the model explicit: you can see exactly what data
the DCF engine needs, and where it comes from.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import date


@dataclass
class MarketData:
    """Current market-derived inputs for a ticker."""
    ticker: str
    price: float                    # Current share price (USD)
    shares_outstanding: float       # Shares outstanding (in millions)
    market_cap: float               # Market cap (USD millions)
    as_of_date: date = field(default_factory=date.today)

    @property
    def market_cap_usd(self) -> float:
        """Market cap in full dollars (not millions)."""
        return self.market_cap * 1_000_000


@dataclass
class FinancialSnapshot:
    """
    Key financial metrics extracted from income statement, balance sheet,
    and cash flow statement. All monetary values are in USD millions
    unless otherwise noted.
    """
    ticker: str

    # ── Cash Flow ──────────────────────────────────────────────────────────
    # Trailing twelve months (TTM) free cash flow.
    # FCF = Operating Cash Flow − Capital Expenditures
    # This is the most important input to the DCF. If negative or erratic,
    # the model will flag the stock as un-modelable via DCF.
    trailing_fcf: float             # USD millions, TTM

    # Historical annual FCF — used to assess stability/trend
    # Ordered oldest → newest. Minimum 3 years for meaningful analysis.
    historical_fcf: List[float] = field(default_factory=list)  # USD millions

    # ── Balance Sheet ──────────────────────────────────────────────────────
    cash_and_equivalents: float = 0.0       # USD millions
    total_debt: float = 0.0                 # Short + long term debt, USD millions

    @property
    def net_debt(self) -> float:
        """Net debt = total debt − cash. Negative means net cash position."""
        return self.total_debt - self.cash_and_equivalents

    # ── Income Statement ───────────────────────────────────────────────────
    revenue_ttm: float = 0.0               # USD millions, TTM
    operating_income_ttm: float = 0.0      # USD millions, TTM

    # ── Metadata ──────────────────────────────────────────────────────────
    fiscal_year_end: Optional[str] = None  # e.g. "December"
    currency: str = "USD"
    as_of_date: date = field(default_factory=date.today)


@dataclass
class DCFResult:
    """
    Output of a single DCF scenario run.
    Contains both per-share valuation and intermediate components.
    """
    ticker: str
    scenario: str                   # "bear", "base", or "bull"

    # ── Assumptions used ──────────────────────────────────────────────────
    fcf_growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    projection_years: int
    base_fcf: float                 # FCF used in year 1 (after any haircut), USD millions

    # ── Projected FCF ─────────────────────────────────────────────────────
    projected_fcfs: List[float] = field(default_factory=list)   # USD millions per year
    pv_of_fcfs: List[float] = field(default_factory=list)       # PV of each year's FCF

    # ── Valuation components ──────────────────────────────────────────────
    pv_fcf_sum: float = 0.0         # Sum of PV of projected FCFs, USD millions
    terminal_value: float = 0.0     # Terminal value (undiscounted), USD millions
    pv_terminal_value: float = 0.0  # PV of terminal value, USD millions
    enterprise_value: float = 0.0   # EV = PV(FCFs) + PV(TV), USD millions
    equity_value: float = 0.0       # EV − net debt, USD millions
    fair_value_per_share: float = 0.0  # Equity value / shares outstanding

    # ── Diagnostics ───────────────────────────────────────────────────────
    terminal_value_pct: float = 0.0    # TV as % of EV — if >75%, flag as unreliable
    is_valid: bool = True
    warning: Optional[str] = None


@dataclass
class ValuationSummary:
    """Aggregated output across all three scenarios for one ticker."""
    ticker: str
    market_price: float
    shares_outstanding: float       # millions

    bear: Optional[DCFResult] = None
    base: Optional[DCFResult] = None
    bull: Optional[DCFResult] = None

    financials: Optional[FinancialSnapshot] = None
    market_data: Optional[MarketData] = None

    @property
    def upside_base(self) -> Optional[float]:
        """Base case upside vs current price (as a decimal, e.g. 0.30 = 30%)."""
        if self.base and self.base.fair_value_per_share > 0:
            return (self.base.fair_value_per_share - self.market_price) / self.market_price
        return None

    @property
    def upside_bear(self) -> Optional[float]:
        if self.bear and self.bear.fair_value_per_share > 0:
            return (self.bear.fair_value_per_share - self.market_price) / self.market_price
        return None

    @property
    def upside_bull(self) -> Optional[float]:
        if self.bull and self.bull.fair_value_per_share > 0:
            return (self.bull.fair_value_per_share - self.market_price) / self.market_price
        return None


@dataclass
class ResearchCandidate:
    """A stock flagged as worth researching by the screener."""
    ticker: str
    date: date
    priority_score: float           # 0–100, higher = more worth researching
    valuation_gap: float            # Base-case upside vs market price (decimal)
    reason: str                     # Human-readable explanation
    bear_upside: float = 0.0
    base_upside: float = 0.0
    bull_upside: float = 0.0
    fcf_stability_score: float = 0.0
    leverage_score: float = 0.0
    valuation_summary: Optional[ValuationSummary] = None
