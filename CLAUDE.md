# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Streamlit dashboard for Dubai real estate apartment transactions. The app loads normalized DLD transaction snapshots from Google Cloud Storage Parquet first, with the production Dubai Data API kept as a fallback/source for refreshing the snapshot. The snapshot holds 12 months of history.

## Commands

```bash
# Run the dashboard from the project virtual environment
.\.venv\Scripts\python.exe -m streamlit run dubai_dashboard.py

# Test Dubai Data API connectivity and column mapping
.\.venv\Scripts\python.exe smoke_test_dda_api.py --limit 5

# Test whether the API contains a specific date range
.\.venv\Scripts\python.exe smoke_test_dda_api.py --limit 5 --start-date 2026-01-01 --end-date 2026-04-30 --require-records

# Test Google Cloud Storage Parquet write/read/delete
.\.venv\Scripts\python.exe smoke_test_gcs.py

# Incrementally update the normalized DLD transactions snapshot in GCS and verify it
.\.venv\Scripts\python.exe store_dld_transactions_gcs.py

# Force a full snapshot rebuild for the default 12-month window
.\.venv\Scripts\python.exe store_dld_transactions_gcs.py --full-refresh --last-months 12

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### dubai_dashboard.py

Single-file Streamlit app with this flow:

1. **Constants** - `AREA_DISPLAY` (official DLD district name -> friendly community name), tracked `NEIGHBORHOODS` (display names), market tier definitions, chart colors, and schema assumptions.
2. **Data loading** - cached GCS Parquet snapshot handling, falling back to the production DDA API when needed.
3. **Data normalization** - maps DDA API columns into the dashboard schema.
4. **Aggregation helpers** - daily, weekly, Dubai-wide, tier, and area-level metrics using Polars.
5. **Chart builders** - Plotly figures for price trends, volume, momentum, tiers, and scatter views.
6. **Streamlit UI** - sidebar filters and API controls, KPI cards, charts, and raw data table.

### dda_api.py

Dubai Data API helper module. It handles:

- loading config from environment variables or `.streamlit/secrets.toml`
- requesting an OAuth bearer token for the Dubai Data API
- fetching paginated dataset records
- building DLD transaction filters
- normalizing DLD transaction records to the dashboard column names
- validating required dashboard columns

`DEFAULT_LOOKBACK_MONTHS = 12` and `DEFAULT_MAX_RECORDS = 500_000` size the default fetch window.

Default production endpoint:

```text
https://apis.data.dubai/secure/ddads/openapi/1.0.0/dld/dld_transactions-open-api
```

### smoke_test_dda_api.py

Small CLI diagnostic for the DDA connector. It verifies credentials, endpoint access, raw columns, normalized column mapping, and optional date coverage. Use this before blaming dashboard logic for missing API records.

### smoke_test_gcs.py

Small CLI diagnostic for Google Cloud Storage. It writes a tiny Polars Parquet file, reads it back, verifies the round trip, and deletes the test object unless `--keep-object` is supplied.

### store_dld_transactions_gcs.py

Incrementally updates the GCS Parquet snapshot. It reads the existing snapshot,
fetches from the current max `INSTANCE_DATE` through today, normalizes, merges,
deduplicates, rewrites the compact Parquet object, and verifies row count,
columns, duplicate removal, and date coverage. Use `--full-refresh` to replace
the snapshot from a requested API window.

### Google Cloud Storage

Current GCS resources:

```text
Project: realstateproject
Bucket: dubai-real-estate-dashboard-jef
Service account: dubai-dashboard-gcs@realstateproject-500813.iam.gserviceaccount.com
Bucket role: Cloud Storage -> Storage Object Admin
```

Expected local/Streamlit secrets:

```toml
GCS_BUCKET = "dubai-real-estate-dashboard-jef"
GCS_TEST_PREFIX = "dubai-dashboard-smoke-tests"
GCS_TRANSACTIONS_OBJECT = "dld_transactions/dld_transactions_latest.parquet"

GCP_SERVICE_ACCOUNT_JSON = '''
{
  "paste": "the full service account JSON here"
}
'''
```

## Key Patterns

- Use Polars for data processing. Do not introduce pandas unless there is a strong reason.
- Keep the dashboard schema stable: `INSTANCE_DATE`, `GROUP_EN`, `IS_OFFPLAN_EN`, `AREA_EN`, `PROP_SB_TYPE_EN`, `TRANS_VALUE`, `ACTUAL_AREA`, and `ROOMS_EN` are required.
- `AREA_EN` holds official DLD district names (e.g. `MARSA DUBAI` is Dubai Marina, `AL BARSHA SOUTH FOURTH` is JVC, `AL JADAF` is Al Jadaf). Always map through `AREA_DISPLAY` / `_area_display_expr()` before showing areas to users; unmapped districts pass through unchanged.
- `ACTUAL_AREA` is in square metres. Convert with `SQM_TO_SQFT` (10.7639) before displaying any sqft figure — `meter_sale_price` in the raw data confirms per-square-metre units.
- `ROOMS_EN` values such as `"1 B/R"` and `"2 B/R"` are normalized to `"1BR"` and `"2BR"`.
- The bedroom filter uses `"All"` as the unfiltered option.
- `_layout_defaults()` should not include `margin`; individual charts set margins themselves.
- Keep API credentials out of git. `.streamlit/secrets.toml` is intentionally ignored.
- Keep GCS service account JSON out of git. Use Streamlit Cloud secrets for deployment.

## API Notes

- Test/STG and production credentials are different.
- The DDA API may be restricted by region/network policy.
- Never print or commit `client_secret`, `client_id`, or `x-DDA-SecurityApplicationIdentifier`.
