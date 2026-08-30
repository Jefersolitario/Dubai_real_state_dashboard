# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Streamlit dashboard for Dubai real estate apartment transactions. The app loads normalized DLD transaction snapshots from Google Cloud Storage Parquet first, with the production Dubai Data API kept as a fallback/source for refreshing the snapshot. The snapshot holds 24 months of history.

## Commands

```bash
# Run the dashboard from the project virtual environment
.\.venv\Scripts\python.exe -m streamlit run dubai_dashboard.py

# Test Dubai Data API connectivity and column mapping
.\.venv\Scripts\python.exe -m tests.smoke_test_dda_api --limit 5

# Test whether the API contains a specific date range
.\.venv\Scripts\python.exe -m tests.smoke_test_dda_api --limit 5 --start-date 2026-01-01 --end-date 2026-04-30 --require-records

# Test Google Cloud Storage Parquet write/read/delete
.\.venv\Scripts\python.exe -m tests.smoke_test_gcs

# Incrementally update the normalized DLD transactions snapshot in GCS and verify it
.\.venv\Scripts\python.exe -m ingestion.store_dld_transactions_gcs

# Force a full snapshot rebuild for the default 24-month window
.\.venv\Scripts\python.exe -m ingestion.store_dld_transactions_gcs --full-refresh --last-months 24

# Pull DLD reference datasets into GCS dld_reference/ (projects, buildings,
# service charges, units registry, Ejari rent grid + Rent Scanner artifacts;
# --only to pick subsets)
.\.venv\Scripts\python.exe -m ingestion.store_reference_data_gcs --only projects buildings units
.\.venv\Scripts\python.exe -m ingestion.store_reference_data_gcs --probe-rents  # print raw Ejari columns (minutes)
.\.venv\Scripts\python.exe -m ingestion.store_reference_data_gcs --only rents   # long pull (~2-4h, chunked; publishes rent_index + both Rent Scanner artifacts)

# Audit the Ejari feed + rent artifacts after a rents refresh (duplicates,
# types, outliers, gaps, pagination sentinel; writes reports/rent_data_quality_report.md)
.\.venv\Scripts\python.exe -m tests.audit_rent_data_quality --check-pagination

# Offline fair-value model checks and optimization
.\.venv\Scripts\python.exe -m tests.smoke_test_fair_value
.\.venv\Scripts\python.exe -m model.optimize_fair_value

# Campaign runner for unit-signal features (rents, renovation permits) with
# reference frames + tail-veto gate. --rungs filters to named candidates; the
# run's own table goes to reports/campaign_last_run.md, while
# reports/rent_campaign_report.md is the CURATED cumulative record (edit by
# hand — a filtered run must not erase earlier campaigns).
# 2026-08 verdict across campaigns 3-4: every candidate rejected — district
# rents, project-linked rents (`rent_project`) and renovation permits
# (`renovation_permits`) all landed at +0.00..+0.02pp vs the 0.05pp gate. The
# groups stay in fair_value_model.py, off by default; the residual ~4% error
# is unit-level (view, floor, condition) and not reachable from public
# aggregates.
.\.venv\Scripts\python.exe -m model.optimize_rent_features
.\.venv\Scripts\python.exe -m model.optimize_rent_features --rungs renovation

# Pull DM building permits and publish dld_reference/modification_permits.parquet
# (adjustment/addition permits linked to projects via parcel_id -> dld_buildings;
# yearly chunks so the gateway token cannot expire mid-pull). Feeds the Fair
# Value tab's "Building works" column.
.\.venv\Scripts\python.exe -m ingestion.store_reference_data_gcs --only permits

# Train the fair-value model offline and publish the inference bundle to GCS
# (run after each snapshot refresh — weekly cadence; the app never trains)
.\.venv\Scripts\python.exe -m model.train_fair_value

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### dubai_dashboard.py

Single-file Streamlit app with this flow:

1. **Constants** - schema assumptions and chart colors. `AREA_DISPLAY`, `NEIGHBORHOODS`, tier definitions, `SQM_TO_SQFT`, and shared expressions/`layout_defaults` live in `dashboard_constants.py` (import from there).
2. **Data loading** - cached GCS Parquet snapshot handling, falling back to the production DDA API when needed.
3. **Data normalization** - maps DDA API columns into the dashboard schema.
4. **Aggregation helpers** - daily, weekly, Dubai-wide, tier, and area-level metrics using Polars.
5. **Chart builders** - Plotly figures for price trends, volume, momentum, tiers, and scatter views.
6. **Streamlit UI** - four pages via `st.segmented_control` (not `st.tabs`, which executes every tab body on each rerun): "Market Overview" (`_render_market_overview`: headline KPIs, Dubai Market Pulse, tier trends), "Buyer Opportunity Scanner" (`_render_zone_analysis`: zone picker + deal-threshold slider driving `_render_zone_deal_finder` — zone KPIs, diverging below-median scanner (`zone_psf_chart`), ranked deals table + CSV (transfers >60% below the median are excluded as non-market token prices), per-zone deal counts (`deals_by_zone_chart`) — plus the raw-data table; only rendered when selected), "Rent Scanner" (`render_rent_scanner` from `rent_scanner_tab.py`, lazy — Ejari contracts vs the zone's 14-day median rent: dot view + deals table for windows ≤92 days inside the contract artifact's coverage, weekly percentile box plots otherwise; renewals excluded by default when the reg-type flag exists; shares `zone_select` with the sales scanner), and "Fair Value Model" (`render_fair_value_tab` from `fair_value_tab.py`, lazy — model bundle load and scoring only run when this page is selected); shared sidebar filters.

### dashboard_constants.py (root)

Shared constants and pure Polars/layout helpers (`AREA_DISPLAY`, `NEIGHBORHOODS`, `TIER_*`, `DISTRICT_TIER`, `SQM_TO_SQFT`, `area_display_expr`, `bedroom_type_expr`, `layout_defaults`). No Streamlit/Plotly imports so pure modules can use it.

### model/fair_value_model.py

Pure Polars + scikit-learn fair-value model: `feature_engineering` (Sales-only apartments, METER_SALE_PRICE area-mismatch guard, non-market-procedure exclusion, no PSF trim — apply `trim_psf` before TRAINING only; strictly past-only derived features via `closed="left"` rolling windows and `shift(1)`; optional `reference` frames for the units-registry/projects/rent-grid joins), 10-fold date-ordered `TimeSeriesSplit` `cross_validate` (fold metrics include MedAPE plus the tail pair `p90_ape` and `flag_prop` — the share of ordinary sales pushed below −15% spread by model error; campaign acceptance requires a MedAPE win AND no tail worsening), `train_fair_value_model` (HistGradientBoostingRegressor on log AED/sqft, out-of-sample permutation importances), `score_transactions` (`spread_pct = actual/fair value − 1`), and `flag_distress` (distressed = below threshold AND ≥1 residual-independent signal: forced-sale procedure, illiquid project, multiple sellers — a deep discount alone never qualifies; also emits `signal_strength` = −spread ÷ segment expected error, established ~4% vs cold-start ~6.5%). The shipped configuration (`fair_value_config.json`, via `load_shipping_config()`) uses floor/balcony features from the units registry (`unit_floor` + `project_meta` + `rel_floor`) and per-unit-type comps (`comps_rooms`) on top of repeat-sale, price-history, and relative-size groups — CV 4.08%, holdout 4.15%. The units join needs `PROJECT_NUMBER` populated in the snapshot (a normalized snapshot cannot back-fill it; rebuild from a raw pull if coverage drops).

### model/data_cleaning.py

Pure-Polars data-quality module: `clean_transactions(df, reference=None) ->
(df, CleaningReport)` labels every row `clean` / `repaired` / `review_only` /
`quarantine` (`dq_rule` + `dq_action` columns) instead of silently dropping.
Digit-shift typos (missing/extra zeros) are repaired in place when the
corrected price lands near the project median sale price AND the recorded area is credible
for the layout (project×rooms median area, optionally the units registry);
bulk-deal allocations (≥3 same-project same-day identical prices ≥25% below
the project median — at-market developer launch batches stay clean), suspected
related-party/token prices (<40% of the project median), and partial-ownership shares are
routed `review_only`: excluded from training and the flag list, shown in the
tab's "Excluded suspicious records" expander. Key feed facts:
METER_SALE_PRICE is mechanically derived from TRANS_VALUE/ACTUAL_AREA
(agrees to ~1e-7), so project median sale prices — not MSP — are the repair instrument;
PROCEDURE_AREA currently equals ACTUAL_AREA on every row, so the
partial-transfer rule is future-proofing. Enabled via the `"data_cleaning"`
feature-config flag (see `fair_value_config.json` for the shipped state;
measured before enabling per `reports/data_cleaning_report.md`).

### fair_value_tab.py (root)

Streamlit UI for the Fair Value tab. Caching contract: `get_features` (one untrimmed feature pass per data version, full history — trailing comps need the past; loads GCS reference frames when the shipped config requires them), `get_model` (loads the pre-trained GCS bundle), `get_scored(data_version, score_start, score_end, _result)` (threshold-independent predictions for the selected scoring window only; default "Last month" keeps the tab fast); the threshold slider only re-runs `flag_distress`. The flagged table ranks distressed-first then by `signal_strength`; `FEATURE_LABELS` maps model feature names to plain language for the importance chart.

### rent_scanner_tab.py (root)

Streamlit UI for the Rent Opportunity Scanner (tenant view of Ejari rents,
all figures annual AED/sqft/yr). Caching contract: `load_rent_artifacts`
(resource cache, 6h TTL) reads `rent_weekly_stats.parquet` +
`rent_recent_contracts.parquet` from GCS and derives `rent_version` from
`blob.updated` — the sales snapshot's `data_version` never keys rent caches;
`generate_rent_psf_timeseries` (data cache, `max_entries=4`) slices the
contract frame per filter selection. `use_dot_view` picks the view: raw dots
+ ranked below-median deals table for windows ≤`RENT_DOT_MAX_DAYS` (92) that
the contract artifact covers, weekly precomputed box plots (q1–q3 box,
p10–p90 whiskers) otherwise. Renewals (RERA-capped) are excluded by default
via an anchored `^new` match on `reg_type` when the flag exists; token rents
beyond −60% vs the median are excluded like the sales scanner. Ejari has no
building key, so deal rows are zone × rooms × size only. The page shows an
actionable error until `--only rents` has published the artifacts.

### model/train_fair_value.py

Offline training CLI: loads the snapshot, trains the shipping configuration,
and publishes a pickled inference bundle to
`gs://<bucket>/fair_value_model/fair_value_model_latest.pkl`
(`GCS_MODEL_OBJECT`). The Streamlit tab only loads this bundle (6h cache TTL)
and predicts — training is too heavy for Streamlit Cloud (the in-app CV was
profiled at ~2-4 minutes and caused watchdog kills). Ops flow: refresh
snapshot (`python -m ingestion.store_dld_transactions_gcs`) → refresh
reference data when stale (`python -m ingestion.store_reference_data_gcs`,
monthly is plenty) → `python -m model.train_fair_value` → the deployed app
picks the new bundle up automatically.

### ingestion/store_reference_data_gcs.py

Pulls DLD reference datasets to GCS under `dld_reference/`: `projects.parquet`
(+developer names), `project_buildings_agg.parquet` (floors/flats per
project), `service_charges.parquet` (latest budget year), `units_slim.parquet`
(flats registry: floor, exact area, balcony — chunked by rooms), and
`rent_index.parquet` (weekly AREA_EN × rooms-band trailing-180d median rent
PSF, strictly past, built from a monthly-chunked Ejari pull with
sanitization), plus the Rent Scanner artifacts from the same pull:
`rent_weekly_stats.parquet` (weekly AREA_EN × rooms-band percentile stats
n/median/q1/q3/p10/p90, including an "All" band — quantiles don't compose
across bands — and a `segment` all/new column) and
`rent_recent_contracts.parquet` (contract-level slim rows for the 20 scanner
districts, trailing 183 days, deduped), and `rent_project_index.parquet`
(weekly per-project trailing-180d rent PSF + contract count, strictly past —
contracts resolved to their project via the Ejari `project_name_en` join plus
the layout fingerprint: district × exact area @2dp sqm × rooms band against
the units registry, unique-key-only; validated 62% unique / 97% accurate on
labeled sales. Needs the projects + units artifacts published first; the
linkage prints coverage and route-agreement diagnostics). The pull pages by `contract_id`
(stable pagination — date ordering duplicated AND skipped ~25% of rows),
dedupes on (contract_id, line_number) plus one row per single-property
contract, and excludes bulk `no_of_prop>1` contracts whose annual_amount is
a contract total, not a unit rent (IAAO allocated-price rule). `--probe-rents`
prints the raw Ejari columns in minutes — run it before the long pull to
confirm the optional `contract_reg_type_en` (new-vs-renewal) field name. `sale_index.parquet` is also pulled but the gateway dataset is
frozen at 2024-05, so no model feature uses it. Long pulls are chunked so a
gateway blip costs one chunk (HTTP 408 is retryable in `dda_api`).

### -m model.optimize_fair_value

Improvement loop: ladder of feature/model proposals measured by mean CV MedAPE, accepts a change only if it improves ≥0.05pp, stops after <0.2pp gains for 2 consecutive iterations or 10 iterations. Writes `reports/fair_value_optimization_report.md`, optional `--progress-json`, and `fair_value_config.json` (the shipping config).

### -m tests.smoke_test_fair_value

Synthetic end-to-end test of the fair-value pipeline (planted underpriced rows must dominate the flags). Run before touching model logic.

### ingestion/dda_api.py

Dubai Data API helper module. It handles:

- loading config from environment variables or `.streamlit/secrets.toml`
- requesting an OAuth bearer token for the Dubai Data API
- fetching paginated dataset records
- building DLD transaction filters
- normalizing DLD transaction records to the dashboard column names
- validating required dashboard columns

`DEFAULT_LOOKBACK_MONTHS = 24` and `DEFAULT_MAX_RECORDS = 1_000_000` size the default fetch window. `fetch_dataset_records` retries each page with exponential backoff (the gateway intermittently returns 502).

Pagination MUST order by a unique key (`transaction_id` for sales, `contract_id` for rents): ordering by date shuffles page boundaries between requests — measured 11% (sales) and ~25% (rents) of rows both duplicated and silently skipped per window. The rent audit's `--check-pagination` is the regression sentinel.

Default production endpoint:

```text
https://apis.data.dubai/secure/ddads/openapi/1.0.0/dld/dld_transactions-open-api
```

### tests/smoke_test_dda_api.py

Small CLI diagnostic for the DDA connector. It verifies credentials, endpoint access, raw columns, normalized column mapping, and optional date coverage. Use this before blaming dashboard logic for missing API records.

### -m tests.smoke_test_gcs

Small CLI diagnostic for Google Cloud Storage. It writes a tiny Polars Parquet file, reads it back, verifies the round trip, and deletes the test object unless `--keep-object` is supplied.

### -m tests.audit_rent_data_quality

Rigorous audit of the Ejari feed and the three rent artifacts, following
`reports/data_cleaning_research_report.md` (IAAO reason codes, local-comp
outlier bands, explicit duplicate definitions D1–D4, sensitivity tests
against the scanner's weekly medians). Fetches a 2-month all-columns raw
sample (stably ordered), checks types/nulls/bands, outliers vs zone×band
comps, week/month/day gaps, cross-artifact integrity, and (with
`--check-pagination`) re-measures the date-ordered paging defect as a
regression sentinel. Writes `reports/rent_data_quality_report.md`; exits
non-zero when open ACTION findings exist. Run after every rents refresh;
alarm thresholds from the research: repairs > 0.2%, review-routed > 2%.

### -m ingestion.store_dld_transactions_gcs

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
GCS_MODEL_OBJECT = "fair_value_model/fair_value_model_latest.pkl"

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
