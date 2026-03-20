# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Streamlit dashboard and PDF report generator for Dubai real estate apartment transactions. Uses DLD (Dubai Land Department) transaction CSV data to track prices, volumes, and buyer opportunities across neighbourhoods and market tiers.

## Commands

```bash
# Run the dashboard (streamlit not on PATH on this machine)
python -m streamlit run dubai_dashboard.py

# Generate the PDF market report
python market_report.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### dubai_dashboard.py (~950 lines)

Single-file Streamlit app with this flow:

1. **Constants** — `NEIGHBORHOODS` (20 tracked areas), `TIER_MAP`/`TIER_AREAS` (5 market segments), `COLOR_MAP`, date range
2. **Data loaders** (all `@st.cache_data`) — Read CSV, filter to `PROP_SB_TYPE_EN == "Flat"`, aggregate with Polars:
   - `generate_dubai_data()` → daily by neighbourhood + bedroom type
   - `generate_dubai_wide_data()` → daily Dubai-wide median/mean
   - `generate_weekly_data()` → weekly with pct_change momentum
   - `generate_area_weekly_change()` → per-area first-vs-last-week change
   - `generate_tier_data()` → daily median per market tier with 7d rolling avg
3. **Chart builders** (return `go.Figure`) — `line_chart`, `bar_chart`, `price_vs_time_scatter`, `dubai_wide_transactions_chart`, `dubai_wide_median_price_chart`, `weekly_pct_change_chart`, `area_pct_change_chart`, `tier_price_chart`
4. **Streamlit UI** — Sidebar filters → KPI cards → Dubai-wide charts → tier chart → filtered neighbourhood charts → raw data table

Shared chart styling via `_layout_defaults()` (dark theme: `#0e1117` background, `#2a2e35` gridlines, `#fafafa` text).

### market_report.py (~280 lines)

Standalone PDF generator using fpdf2. Hardcoded analysis findings from the current dataset. Custom `Report(FPDF)` class with helpers: `section()`, `kpi_row()`, `table()`, `bullet()`. Output: `report_YYYY-MM-DD.pdf`.

### Data

CSV files in `data/` from DLD open data. The dashboard uses `data/transactions-2026-03-20 unit.csv` (apartment-level transactions). Key columns: `INSTANCE_DATE`, `AREA_EN`, `ROOMS_EN`, `PROP_SB_TYPE_EN`, `TRANS_VALUE`, `ACTUAL_AREA`.

## Key Patterns

- **Polars only** for data processing (not pandas). Schema overrides needed for `TRANS_VALUE` and `ACTUAL_AREA` (some rows have decimals that break int inference).
- CSV has a UTF-8 BOM — use `encoding="utf8-lossy"` with `pl.read_csv`.
- `AREA_EN` has casing variants (e.g. "BUSINESS BAY" and "Business Bay") — both must be handled.
- `ROOMS_EN` format is `"1 B/R"`, `"2 B/R"`, `"Studio"` etc. — mapped to `"1BR"`, `"2BR"` via `.str.replace(" B/R", "BR")`.
- Bedroom filter uses `"All"` (not `"Both"`) for the unfiltered option.
- `_layout_defaults()` must NOT include `margin` — each chart sets its own to avoid duplicate keyword errors.
- Market tiers in `TIER_AREAS` should cover ~90% of transactions for meaningful tier charts.
