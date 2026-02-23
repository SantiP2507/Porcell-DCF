# DCF Valuation & Research Engine

A production-quality Python equity research system built around discounted cash flow valuation.

## Philosophy
- Transparency over precision
- Conservative assumptions by default
- Ranges (bear/base/bull), not single point estimates
- Research signals, not trading signals

## Setup

```bash
pip install yfinance pandas numpy matplotlib scipy requests python-dotenv supabase
```

Create a `.env` file in the project root:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

## Usage

```bash
# Run full valuation on a ticker
python main.py --ticker AAPL

# Run daily research recommendation screen
python main.py --screen

# Run valuation + save to Supabase
python main.py --ticker MSFT --save

# Run sensitivity analysis only
python main.py --ticker GOOGL --sensitivity
```

## Module Structure

```
dcf_engine/
├── main.py                  # CLI entry point
├── config.py                # Global settings and defaults
├── data/
│   ├── fetcher.py           # Pulls live data from yfinance
│   ├── financials.py        # Parses and normalizes financial statements
│   └── cache.py             # Simple disk cache to avoid repeated API calls
├── valuation/
│   ├── dcf.py               # Core DCF engine
│   ├── scenarios.py         # Bear/base/bull scenario builder
│   └── sensitivity.py       # Sensitivity analysis (growth vs. WACC grid)
├── research/
│   ├── screener.py          # Daily candidate generation
│   └── scorer.py            # Rule-based ranking
├── charts/
│   └── plotter.py           # All matplotlib charts
├── db/
│   └── supabase_client.py   # Supabase read/write placeholders
└── utils/
    └── helpers.py           # Shared utilities
```

## Supabase Tables (create these yourself)

### valuations
```sql
create table valuations (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  date date not null,
  bear_value numeric,
  base_value numeric,
  bull_value numeric,
  market_price numeric,
  terminal_value_pct numeric,
  assumptions jsonb,
  created_at timestamptz default now()
);
```

### research_candidates
```sql
create table research_candidates (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  date date not null,
  priority integer,
  reason text,
  valuation_gap numeric,
  created_at timestamptz default now()
);
```

### market_snapshots (optional)
```sql
create table market_snapshots (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  date date not null,
  price numeric,
  fcf numeric,
  valuation_gap numeric,
  created_at timestamptz default now()
);
```
