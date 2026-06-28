# CHATGPT.md

This file provides guidance to ChatGPT/Codex when working with code in this repository.

## Project Overview

Interactive Streamlit dashboard and PDF report generator for Dubai real estate apartment transactions. The app loads normalized DLD transaction snapshots from Google Cloud Storage Parquet first, with the production Dubai Data API kept as a fallback/source for refreshing the snapshot.

## Commands

```bash
# Run the dashboard from the project virtual environment
.\.venv\Scripts\python.exe -m streamlit run dubai_dashboard.py

# Generate the PDF market report
.\.venv\Scripts\python.exe market_report.py

# Test Dubai Data API connectivity and column mapping
.\.venv\Scripts\python.exe smoke_test_dda_api.py --limit 5

# Test whether the API contains a specific date range
.\.venv\Scripts\python.exe smoke_test_dda_api.py --limit 5 --start-date 2026-01-01 --end-date 2026-04-30 --require-records

# Test Google Cloud Storage Parquet write/read/delete
.\.venv\Scripts\python.exe smoke_test_gcs.py

# Store the latest normalized DLD transactions snapshot in GCS and verify it
.\.venv\Scripts\python.exe store_dld_transactions_gcs.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### dubai_dashboard.py

Single-file Streamlit app with this flow:

1. **Constants** - tracked neighbourhoods, market tier definitions, chart colors, default data source, and schema assumptions.
2. **Data loading** - cached GCS Parquet snapshot handling, falling back to the production DDA API when needed.
3. **Data normalization** - maps DDA API columns and CSV columns into the dashboard schema.
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

Default production endpoint:

```text
https://apis.data.dubai/secure/ddads/openapi/1.0.0/dld/dld_transactions-open-api
```

### smoke_test_dda_api.py

Small CLI diagnostic for the DDA connector. It verifies credentials, endpoint access, raw columns, normalized column mapping, and optional date coverage. Use this before blaming dashboard logic for missing API records.

### smoke_test_gcs.py

Small CLI diagnostic for Google Cloud Storage. It writes a tiny Polars Parquet file, reads it back, verifies the round trip, and deletes the test object unless `--keep-object` is supplied.

### store_dld_transactions_gcs.py

Fetches normalized DLD transaction records with the existing `dda_api.py` helpers, writes them to GCS as Parquet, reads the object back, and verifies row count, columns, and date coverage.

### market_report.py

Standalone PDF generator using fpdf2. It builds `report_YYYY-MM-DD.pdf` from the current local analysis assumptions.

### Data

CSV files live in `data/` for offline reference:

```text
data/transactions-2026-03-20 unit.csv
```

Important columns include `INSTANCE_DATE`, `AREA_EN`, `ROOMS_EN`, `PROP_SB_TYPE_EN`, `TRANS_VALUE`, and `ACTUAL_AREA`.

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
- Use `encoding="utf8-lossy"` for DLD CSV files.
- Keep the dashboard schema stable: `INSTANCE_DATE`, `GROUP_EN`, `IS_OFFPLAN_EN`, `AREA_EN`, `PROP_SB_TYPE_EN`, `TRANS_VALUE`, `ACTUAL_AREA`, and `ROOMS_EN` are required.
- `ROOMS_EN` values such as `"1 B/R"` and `"2 B/R"` are normalized to `"1BR"` and `"2BR"`.
- `AREA_EN` can have casing variants, so compare areas case-insensitively where possible.
- The bedroom filter uses `"All"` as the unfiltered option.
- `_layout_defaults()` should not include `margin`; individual charts set margins themselves.
- Keep API credentials out of git. `.streamlit/secrets.toml` is intentionally ignored.
- Keep GCS service account JSON out of git. Use Streamlit Cloud secrets for deployment.

## API Notes

- Test/STG and production credentials are different.
- The DDA API may be restricted by region/network policy.
- Never print or commit `client_secret`, `client_id`, or `x-DDA-SecurityApplicationIdentifier`.
