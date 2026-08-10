"""
Dubai Real Estate Apartment Dashboard  –  Streamlit + Polars
=============================================================
Visualises apartment prices across Dubai neighbourhoods using the
production Dubai Data DLD transactions API.

Run
---
    pip install streamlit polars plotly numpy
    streamlit run dubai_dashboard.py

Production data source
----------------------
Configure DDA credentials through Streamlit secrets or environment variables.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import polars as pl
from polars.exceptions import PanicException
import plotly.graph_objects as go
import streamlit as st

from ingestion.dda_api import (
    DDAApiError,
    DDAConfig,
    DEFAULT_MAX_RECORDS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_LOOKBACK_MONTHS,
    build_dld_transactions_params,
    fetch_dataset_records,
    infer_column_mapping,
    last_months_date_range,
    months_before,
    load_dda_config,
    normalize_dld_transactions,
    records_to_dataframe,
    validate_normalized_columns,
)
from ingestion.gcs_storage import (
    configured_snapshot,
    dataframe_to_parquet_bytes,
    gcs_client,
    read_parquet_object,
)
from ingestion.store_dld_transactions_gcs import (
    count_new_rows,
    dedupe_snapshot,
    stable_sort,
)

from dashboard_constants import (
    NEIGHBORHOODS,
    SQM_TO_SQFT,
    TIER_COLORS,
    TIER_MAP,
    TIER_ORDER,
    area_display_expr as _area_display_expr,
    layout_defaults as _layout_defaults,
    y_cap_range as _y_cap_range,
)
from fair_value_tab import render_fair_value_tab
from rent_scanner_tab import render_rent_scanner

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_DEFAULT_LOOKBACK_MONTHS = DEFAULT_LOOKBACK_MONTHS

TRANSACTION_SCHEMA = {
    "INSTANCE_DATE": pl.Utf8,
    "GROUP_EN": pl.Utf8,
    "IS_OFFPLAN_EN": pl.Utf8,
    "AREA_EN": pl.Utf8,
    "PROP_SB_TYPE_EN": pl.Utf8,
    "PROCEDURE_EN": pl.Utf8,
    "TRANS_VALUE": pl.Float64,
    "ACTUAL_AREA": pl.Float64,
    "ROOMS_EN": pl.Utf8,
}

# Zone deal-finder colors — same semantics as the Fair Value tab's
# COLOR_NORMAL / COLOR_BELOW so "red = below fair price" reads the same
# across pages. Pair validated for colorblind separation on the dark surface.
ZONE_COLOR_CONTEXT = "#636efa"   # at/above the zone median (context)
ZONE_COLOR_BELOW = "#ef553b"     # below the zone median / flagged deal

# A spread beyond -60% means the price was under 40% of the zone median —
# the data-cleaning module's token-price threshold for gifts/related-party
# transfers. Such records are registered transfers, not buyable prices, so
# the deal list and deal counts exclude them.
NON_MARKET_SPREAD = 0.60


# ---------------------------------------------------------------------------
# Data loading (Polars) - production DDA API
# ---------------------------------------------------------------------------

def _source_raw_transactions() -> pl.DataFrame:
    """The session's loaded transactions (empty typed frame before load)."""
    api_df = st.session_state.get("api_raw_df")
    if isinstance(api_df, pl.DataFrame) and not api_df.is_empty():
        return api_df
    return pl.DataFrame(schema=TRANSACTION_SCHEMA)


def _flat_transactions(trans_type: str) -> pl.DataFrame:
    """Apartment (flat) rows only, filtered to the transaction type."""
    return _trans_type_filter(
        _source_raw_transactions().filter(
            pl.col("PROP_SB_TYPE_EN")
            .cast(pl.Utf8)
            .str.to_lowercase()
            == "flat"
        ),
        trans_type,
    )


@st.cache_data(show_spinner="Loading transaction data...")
def generate_dubai_data(
    trans_type: str = "All",
    data_version: int = 0,
) -> pl.DataFrame:
    """Aggregate real DLD apartment transactions from the production API.

    Aggregates individual transactions to daily averages per
    neighbourhood and bedroom type, matching the dashboard schema.
    """
    raw = _flat_transactions(trans_type)
    return (
        raw
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            _area_display_expr().alias("neighborhood"),
            pl.col("ROOMS_EN").str.replace(" B/R", "BR").alias("bedroom_type"),
            (pl.col("TRANS_VALUE") / (pl.col("ACTUAL_AREA") * SQM_TO_SQFT))
                .alias("price_per_sqft"),
        ])
        .group_by(["date", "neighborhood", "bedroom_type"])
        .agg([
            pl.col("TRANS_VALUE").mean().round(0).alias("avg_price_aed"),
            pl.col("price_per_sqft").mean().round(1).alias("price_per_sqft_aed"),
            (pl.col("ACTUAL_AREA") * SQM_TO_SQFT).mean().round(1).alias("avg_size_sqft"),
            pl.col("TRANS_VALUE").count().alias("transaction_count"),
        ])
    )


@st.cache_data(show_spinner="Computing Dubai-wide aggregates...")
def generate_dubai_wide_data(
    trans_type: str = "All",
    data_version: int = 0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Daily Dubai-wide transaction count and median price (all flats)."""
    raw = (
        _flat_transactions(trans_type)
        .with_columns(
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
        )
    )
    raw = _filter_raw_date_window(raw, start_date, end_date)
    return (
        raw
        .group_by("date")
        .agg([
            pl.col("TRANS_VALUE").count().alias("transaction_count"),
            pl.col("TRANS_VALUE").median().round(0).alias("median_price_aed"),
            pl.col("TRANS_VALUE").mean().round(0).alias("avg_price_aed"),
        ])
        .sort("date")
    )


@st.cache_data(show_spinner="Computing weekly stats...")
def generate_weekly_data(
    trans_type: str = "All",
    data_version: int = 0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Weekly Dubai-wide aggregates with % change."""
    raw = (
        _flat_transactions(trans_type)
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
        ])
    )
    raw = _filter_raw_date_window(raw, start_date, end_date)
    raw = raw.with_columns(
        pl.col("date")
          .dt.truncate("1w")
          .alias("week"),
    )
    if start_date is not None:
        raw = raw.with_columns(
            pl.when(pl.col("week") < start_date)
              .then(pl.lit(start_date))
              .otherwise(pl.col("week"))
              .alias("week")
        )

    return (
        raw
        .group_by("week")
        .agg([
            pl.col("TRANS_VALUE").count().alias("txns"),
            pl.col("TRANS_VALUE").median().round(0).alias("median"),
            pl.col("TRANS_VALUE").mean().round(0).alias("mean"),
        ])
        .sort("week")
        .with_columns([
            pl.col("median").pct_change().mul(100).round(1).alias("median_pct_chg"),
            pl.col("mean").pct_change().mul(100).round(1).alias("mean_pct_chg"),
            pl.col("txns").pct_change().mul(100).round(1).alias("txn_pct_chg"),
            (pl.col("mean") / pl.col("median")).round(2).alias("mean_median_ratio"),
        ])
    )


@st.cache_data(show_spinner="Computing area-level trends...")
def generate_area_weekly_change(
    trans_type: str = "All",
    data_version: int = 0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Per-area first-to-last-week median price % change (min 50 txns)."""
    base = (
        _flat_transactions(trans_type)
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .dt.truncate("1w")
              .alias("week"),
            _area_display_expr().alias("area"),
        ])
    )
    base = _filter_raw_date_window(base, start_date, end_date)
    raw = (
        base
        .group_by(["area", "week"])
        .agg([
            pl.col("TRANS_VALUE").count().alias("txns"),
            pl.col("TRANS_VALUE").median().round(0).alias("median"),
        ])
        .sort(["area", "week"])
    )
    # Keep areas with >= 50 total transactions
    area_totals = raw.group_by("area").agg(
        pl.col("txns").sum().alias("total_txns")
    ).filter(pl.col("total_txns") >= 50)

    first = raw.group_by("area").first().select(["area", pl.col("median").alias("first_median")])
    last = raw.group_by("area").last().select(["area", pl.col("median").alias("last_median")])

    return (
        area_totals
        .join(first, on="area")
        .join(last, on="area")
        .with_columns(
            ((pl.col("last_median") - pl.col("first_median")) / pl.col("first_median") * 100)
            .round(1)
            .alias("pct_change")
        )
        .sort("pct_change")
    )


@st.cache_data(show_spinner="Computing tier aggregates...")
def generate_tier_data(
    trans_type: str = "All",
    data_version: int = 0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Daily median price per market tier."""
    raw = (
        _flat_transactions(trans_type)
        .with_columns(_area_display_expr().alias("display_area"))
        .filter(pl.col("display_area").is_in(list(TIER_MAP.keys())))
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            pl.col("display_area").replace_strict(TIER_MAP).alias("tier"),
        ])
    )
    raw = _filter_raw_date_window(raw, start_date, end_date)
    return (
        raw
        .group_by(["date", "tier"])
        .agg([
            pl.col("TRANS_VALUE").median().round(0).alias("median_price"),
            pl.col("TRANS_VALUE").count().alias("txns"),
        ])
        .sort(["tier", "date"])
        .with_columns(
            pl.col("median_price")
              .rolling_mean(7)
              .over("tier")
              .alias("median_7d")
        )
    )


@st.cache_data(show_spinner="Computing area price/sqft time series...")
def generate_area_psf_timeseries(
    trans_type: str = "All",
    bedroom: str = "All",
    data_version: int = 0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Transaction-level price/sqft and rolling area median — buyer opportunity scanner.

    Rolling window: 14 days (not 7).
    Rationale: the citywide charts use a 7-day window over ~11k transactions.
    Individual areas average ~550 transactions each across the full period,
    or ~12/day — roughly 1/20th the citywide density. Doubling the window
    to 14 days keeps a comparable effective sample size (~168 observations)
    while still being reactive enough to show meaningful trend shifts over
    the loaded dataset.  min_periods=3 so the line appears from day 3 onward
    rather than staying blank for the first 13 days.

    Returns
    -------
    txns_df   : one row per transaction — [date, area, price_per_sqft,
                bedroom_type, building, size_sqft, price_aed, offplan]
    rolling_df: one row per (date, area) — [date, area, daily_median_psf,
                rolling_median_psf]
    """
    raw = (
        _flat_transactions(trans_type)
        .filter(pl.col("ACTUAL_AREA") > 0)   # guard against divide-by-zero
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            _area_display_expr().alias("area"),
            pl.col("ROOMS_EN").str.replace(" B/R", "BR").alias("bedroom_type"),
            (pl.col("TRANS_VALUE") / (pl.col("ACTUAL_AREA") * SQM_TO_SQFT))
                .alias("price_per_sqft"),
        ])
    )

    raw = _filter_raw_date_window(raw, start_date, end_date)

    if bedroom != "All":
        raw = raw.filter(pl.col("bedroom_type") == bedroom)

    # Individual transactions for scatter dots and the deals table.
    # BUILDING_NAME_EN is an optional normalized column — older snapshots
    # (and the empty fallback frame) may not carry it.
    building_col = (
        pl.col("BUILDING_NAME_EN")
        if "BUILDING_NAME_EN" in raw.columns
        else pl.lit(None, dtype=pl.Utf8)
    )
    txns_df = (
        raw
        .select([
            pl.col("date"),
            pl.col("area"),
            pl.col("price_per_sqft"),
            pl.col("bedroom_type"),
            building_col.fill_null("—").alias("building"),
            (pl.col("ACTUAL_AREA") * SQM_TO_SQFT).round(0).alias("size_sqft"),
            pl.col("TRANS_VALUE").alias("price_aed"),
            pl.col("IS_OFFPLAN_EN").alias("offplan"),
        ])
        .sort(["area", "date"])
    )

    # Daily area median, then 14-day rolling mean of those medians.
    # We roll over the DAILY median (not raw transactions) so that
    # high-volume days don't dominate the smoothed line.
    rolling_df = (
        raw
        .group_by(["date", "area"])
        .agg(pl.col("price_per_sqft").median().round(0).alias("daily_median_psf"))
        .sort(["area", "date"])
        .with_columns(
            pl.col("daily_median_psf")
              .rolling_mean(14)
              .over("area")
              .alias("rolling_median_psf")
        )
    )

    # rechunk: hand st.cache_data compact single-chunk buffers — chunked
    # string-view arrays from filtered frames have hit arrow binview panics
    # in downstream kernels (polars 1.43).
    return txns_df.rechunk(), rolling_df.rechunk()


# ---------------------------------------------------------------------------
# Legacy connector helpers kept for offline maintenance scripts.
# ---------------------------------------------------------------------------

def _get_dld_token(
    api_key: str,
    api_secret: str,
    security_application_identifier: str = "",
) -> str:
    """Obtain an OAuth2 bearer token from the Dubai Data API."""
    import requests

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if security_application_identifier:
        headers["x-DDA-SecurityApplicationIdentifier"] = (
            security_application_identifier
        )

    resp = requests.post(
        "https://apis.data.dubai/oauth/client_credential/accesstoken",
        params={"grant_type": "client_credentials"},
        data={"client_id": api_key, "client_secret": api_secret},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# Column mapping: API field names → CSV field names used by the dashboard
_API_TO_CSV = {
    "transaction_id":       "TRANSACTION_NUMBER",
    "instance_date":        "INSTANCE_DATE",
    "trans_group_en":       "GROUP_EN",
    "procedure_name_en":    "PROCEDURE_EN",
    "reg_type_en":          "IS_OFFPLAN_EN",
    "property_usage_en":    "USAGE_EN",
    "area_name_en":         "AREA_EN",
    "property_type_en":     "PROP_TYPE_EN",
    "property_sub_type_en": "PROP_SB_TYPE_EN",
    "actual_worth":         "TRANS_VALUE",
    "procedure_area":       "ACTUAL_AREA",
    "rooms_en":             "ROOMS_EN",
    "has_parking":          "PARKING",
    "nearest_metro_en":     "NEAREST_METRO_EN",
    "nearest_mall_en":      "NEAREST_MALL_EN",
    "nearest_landmark_en":  "NEAREST_LANDMARK_EN",
    "no_of_parties_role_1": "TOTAL_SELLER",
    "no_of_parties_role_2": "TOTAL_BUYER",
    "master_project_en":    "MASTER_PROJECT_EN",
    "project_name_en":      "PROJECT_EN",
    "meter_sale_price":     "METER_SALE_PRICE",
}






# Production secrets are read server-side only; no credentials are exposed in
# the Streamlit interface.
def _streamlit_secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _load_api_transactions(
    config: DDAConfig,
    start: date,
    end: date,
    page_size: int,
    max_records: int,
) -> tuple[pl.DataFrame, dict]:
    """Fetch a window of transactions straight from the DDA API."""
    params = build_dld_transactions_params(start, end, order_desc=True)
    records = fetch_dataset_records(
        config,
        params=params,
        page_size=page_size,
        max_records=max_records,
    )
    raw_df = records_to_dataframe(records)
    normalized = normalize_dld_transactions(raw_df)
    validation = validate_normalized_columns(normalized)
    return normalized, {
        "raw_columns": raw_df.columns,
        "mapping": infer_column_mapping(raw_df.columns),
        "validation": validation,
        "params": params,
        "date_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    }


def _load_gcs_transactions(
    secrets: dict,
    bucket_name: str,
    object_name: str,
) -> tuple[pl.DataFrame, dict]:
    """Load + normalize the GCS snapshot; returns (frame, metadata) or None."""
    raw_df, blob = read_parquet_object(secrets, bucket_name, object_name)
    normalized = normalize_dld_transactions(raw_df)
    validation = validate_normalized_columns(normalized)
    if normalized.is_empty():
        raise DDAApiError("GCS transaction snapshot is empty.")
    if validation["missing_required"]:
        raise DDAApiError("GCS transaction snapshot schema is incomplete.")

    # The snapshot retains raw API columns alongside the normalized ones
    # (~70 total); keep only the dashboard schema — cuts every in-memory
    # copy by ~70% (267 MB -> 77 MB measured).
    raw_columns = raw_df.columns
    mapping = infer_column_mapping(raw_columns)
    del raw_df

    bounds = _date_bounds_from_transactions(normalized)
    object_metadata = blob.metadata or {}
    date_start = object_metadata.get("date_start")
    date_end = object_metadata.get("date_end")
    if bounds:
        date_start = date_start or bounds[0].isoformat()
        date_end = date_end or bounds[1].isoformat()

    return normalized, {
        "source": "gcs_parquet",
        "raw_columns": raw_columns,
        "mapping": mapping,
        "validation": validation,
        "params": {
            "gcs_bucket": bucket_name,
            "gcs_object": object_name,
        },
        "date_window": {
            "start": date_start,
            "end": date_end,
        },
        "loaded_at": (
            object_metadata.get("stored_at_utc")
            or (blob.updated.isoformat() if blob.updated else None)
            or datetime.now().isoformat(timespec="seconds")
        ),
        "date_bounds": {
            "start": date_start,
            "end": date_end,
        },
        "gcs": {
            "bucket": bucket_name,
            "object": object_name,
            "size": blob.size,
            "updated": blob.updated.isoformat() if blob.updated else None,
            "metadata": object_metadata,
        },
    }




def _date_bounds_from_transactions(df: pl.DataFrame) -> tuple[date, date] | None:
    """(min, max) INSTANCE_DATE in the frame, or None when empty."""
    if df.is_empty() or "INSTANCE_DATE" not in df.columns:
        return None

    dates = (
        df
        .select(
            pl.col("INSTANCE_DATE")
            .cast(pl.Utf8)
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("date")
        )
        .drop_nulls()
    )
    if dates.is_empty():
        return None
    return dates["date"].min(), dates["date"].max()


def _missing_date_windows(
    df: pl.DataFrame,
    requested_start: date,
    requested_end: date,
) -> list[tuple[date, date]]:
    """Date spans the loaded snapshot does not cover for the request."""
    bounds = _date_bounds_from_transactions(df)
    if not bounds:
        return [(requested_start, requested_end)]

    loaded_start, loaded_end = bounds
    windows = []
    if requested_start < loaded_start:
        windows.append((requested_start, loaded_start))
    if requested_end > loaded_end:
        windows.append((loaded_end, requested_end))
    return [(start, end) for start, end in windows if start <= end]


def _write_gcs_snapshot(
    secrets: dict,
    bucket_name: str,
    object_name: str,
    df: pl.DataFrame,
    stats: dict[str, int | str],
) -> tuple[pl.DataFrame, dict]:
    """Upload the merged snapshot and verify the readback matches."""
    bounds = _date_bounds_from_transactions(df)
    blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)
    blob.metadata = {
        "row_count": str(df.height),
        "date_start": bounds[0].isoformat() if bounds else "",
        "date_end": bounds[1].isoformat() if bounds else "",
        "stored_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **{key: str(value) for key, value in stats.items()},
    }
    blob.upload_from_string(
        dataframe_to_parquet_bytes(df),
        content_type="application/vnd.apache.parquet",
    )

    refreshed_df, refreshed_meta = _load_gcs_transactions(
        secrets,
        bucket_name,
        object_name,
    )
    if refreshed_df.height != df.height:
        raise DDAApiError(
            f"GCS write verification failed: expected {df.height:,} rows, "
            f"read back {refreshed_df.height:,}."
        )
    if dedupe_snapshot(refreshed_df).height != refreshed_df.height:
        raise DDAApiError("GCS write verification failed: duplicate rows remain.")
    return refreshed_df, refreshed_meta


def _fetch_missing_windows(
    config: DDAConfig, windows: list[tuple[date, date]]
) -> tuple[list[pl.DataFrame], int]:
    """Fetch each uncovered date window from the API; returns (frames, row count).

    Raises DDAApiError when a response is missing required schema columns.
    """
    incoming_frames: list[pl.DataFrame] = []
    fetched_rows = 0
    for start, end in windows:
        incoming, incoming_meta = _load_api_transactions(
            config,
            start,
            end,
            DEFAULT_PAGE_SIZE,
            DEFAULT_MAX_RECORDS,
        )
        validation = incoming_meta["validation"]
        if validation["missing_required"]:
            raise DDAApiError(
                "Incremental DDA response schema is incomplete: "
                + ", ".join(validation["missing_required"])
            )
        if not incoming.is_empty():
            incoming_frames.append(incoming)
            fetched_rows += incoming.height
    return incoming_frames, fetched_rows


def _merge_incoming_rows(
    df: pl.DataFrame, incoming_frames: list[pl.DataFrame]
) -> tuple[pl.DataFrame, dict]:
    """Dedupe and combine existing + fetched rows; returns (final frame, stats).

    Both merge sides already share the canonical schema:
    normalize_dld_transactions projects every frame at the source.
    """
    incoming_df = pl.concat(incoming_frames, how="diagonal_relaxed")
    existing_deduped = dedupe_snapshot(df)
    incoming_deduped = dedupe_snapshot(incoming_df)
    combined = pl.concat([existing_deduped, incoming_deduped], how="diagonal_relaxed")
    final_df = stable_sort(dedupe_snapshot(combined))
    stats = {
        "mode": "incremental_dashboard",
        "existing_rows": df.height,
        "added_rows": count_new_rows(existing_deduped, final_df),
        "existing_duplicate_rows_removed": df.height - existing_deduped.height,
        "incoming_duplicate_rows_removed": incoming_df.height - incoming_deduped.height,
        "overlap_duplicate_rows": (
            existing_deduped.height + incoming_deduped.height - final_df.height
        ),
    }
    return final_df, stats


def _ensure_transactions_cover_range(
    df: pl.DataFrame,
    meta: dict,
    requested_start: date,
    requested_end: date,
) -> tuple[pl.DataFrame, dict, dict | None]:
    """Top up the snapshot from the API when the picker exceeds coverage."""
    windows = _missing_date_windows(df, requested_start, requested_end)
    if not windows:
        return df, meta, None

    refresh_key = "|".join(f"{start}:{end}" for start, end in windows)
    if st.session_state.get("gcs_refresh_checked") == refresh_key:
        return df, meta, None

    secrets = _streamlit_secrets()
    config = load_dda_config(secrets)
    missing = config.missing_fields()
    if missing:
        raise DDAApiError(
            "Cannot refresh missing date range; missing DDA configuration: "
            + ", ".join(missing)
        )

    # Mark the attempt BEFORE fetching: when the API is unreachable (the
    # normal state on Streamlit Cloud) the fetch raises, and setting the key
    # afterwards would retry the doomed call on every widget interaction —
    # freezing each rerun for the full network timeout.
    st.session_state["gcs_refresh_checked"] = refresh_key
    incoming_frames, fetched_rows = _fetch_missing_windows(config, windows)
    if not incoming_frames:
        return df, meta, {
            "mode": "incremental",
            "fetched_rows": 0,
            "added_rows": 0,
            "duplicate_rows_removed": 0,
            "windows": windows,
            "written": False,
        }

    final_df, stats = _merge_incoming_rows(df, incoming_frames)
    stats = {**stats, "fetched_rows": fetched_rows, "windows": windows}
    if stats["added_rows"] == 0 and stats["existing_duplicate_rows_removed"] == 0:
        return df, meta, {**stats, "written": False}

    bucket_name, object_name = configured_snapshot(secrets)
    if not bucket_name:
        return final_df, {
            **meta,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
            "source": "dda_incremental_memory",
        }, {**stats, "written": False}

    refreshed_df, refreshed_meta = _write_gcs_snapshot(
        secrets,
        bucket_name,
        object_name,
        final_df,
        {key: value for key, value in stats.items() if key != "windows"},
    )
    try:
        load_production_transactions.clear()
    except Exception:
        pass
    return refreshed_df, refreshed_meta, {**stats, "written": True}


# cache_resource (not cache_data): avoids keeping a full pickle copy of the
# ~300k-row frame resident plus a deserialize copy on every rerun.
@st.cache_resource(
    show_spinner="Loading latest DLD transactions...",
    ttl=60 * 60,
)
def load_production_transactions() -> tuple[pl.DataFrame, dict]:
    """Session data loader: GCS snapshot first, DDA API fallback (cached)."""
    secrets = _streamlit_secrets()
    bucket_name, object_name = configured_snapshot(secrets)
    gcs_error: Exception | None = None
    if bucket_name:
        try:
            return _load_gcs_transactions(secrets, bucket_name, object_name)
        except Exception as exc:
            gcs_error = exc
            LOGGER.exception("GCS transaction snapshot load failed: %s", exc)

    config = load_dda_config(secrets)
    missing = config.missing_fields()
    if missing:
        fallback_note = ""
        if gcs_error:
            fallback_note = f" GCS cache read failed: {gcs_error}"
        elif not bucket_name:
            fallback_note = " GCS cache is not configured."
        raise DDAApiError(
            "Missing production data source configuration: "
            + ", ".join(missing)
            + fallback_note
        )

    start, end = last_months_date_range(API_DEFAULT_LOOKBACK_MONTHS)
    df, meta = _load_api_transactions(
        config,
        start,
        end,
        DEFAULT_PAGE_SIZE,
        DEFAULT_MAX_RECORDS,
    )
    validation = meta["validation"]
    if df.is_empty():
        raise DDAApiError("Production data source returned no records.")
    if validation["missing_required"]:
        raise DDAApiError("Production data source schema is incomplete.")

    bounds = _date_bounds_from_transactions(df)
    return df, {
        **meta,
        "source": "dda_api",
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
        "date_bounds": {
            "start": bounds[0].isoformat() if bounds else start.isoformat(),
            "end": bounds[1].isoformat() if bounds else end.isoformat(),
        },
        "gcs_fallback_error": str(gcs_error) if gcs_error else None,
    }


def _loaded_date_bounds() -> tuple[date, date]:
    """Date coverage of the currently loaded transactions."""
    api_df = st.session_state.get("api_raw_df")
    if isinstance(api_df, pl.DataFrame):
        bounds = _date_bounds_from_transactions(api_df)
        if bounds:
            return bounds
    return last_months_date_range(API_DEFAULT_LOOKBACK_MONTHS)


def _date_picker_bounds() -> tuple[date, date, date]:
    """(min, max, default_end) limits for the sidebar date picker."""
    loaded_start, loaded_end = _loaded_date_bounds()
    default_start, today = last_months_date_range(API_DEFAULT_LOOKBACK_MONTHS)
    return min(loaded_start, default_start), today, today


# ---------------------------------------------------------------------------
# Filter helper (Polars)
# ---------------------------------------------------------------------------

def _trans_type_filter(df: pl.DataFrame, trans_type: str) -> pl.DataFrame:
    """Filter rows to Sale / Mortgage / Off-Plan; 'All' passes through."""
    if trans_type == "Sale":
        return df.filter(pl.col("GROUP_EN").cast(pl.Utf8).str.to_lowercase() == "sales")
    if trans_type == "Mortgage":
        return df.filter(pl.col("GROUP_EN").cast(pl.Utf8).str.to_lowercase() == "mortgage")
    if trans_type == "Off-Plan":
        status_mask = pl.lit(False)
        if "IS_OFFPLAN_EN" in df.columns:
            status_mask = (
                pl.col("IS_OFFPLAN_EN")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains("off")
                .fill_null(False)
            )

        procedure_mask = pl.lit(False)
        if "PROCEDURE_EN" in df.columns:
            procedure_mask = (
                pl.col("PROCEDURE_EN")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains("development|payment plan")
                .fill_null(False)
            )

        return df.filter(status_mask | procedure_mask)
    return df  # "All"


def _filter_raw_date_window(
    df: pl.DataFrame,
    start: date | None,
    end: date | None,
    column: str = "date",
) -> pl.DataFrame:
    """Limit a raw transaction frame to the selected date window."""
    if df.is_empty() or column not in df.columns:
        return df

    mask = pl.lit(True)
    if start is not None:
        mask = mask & (pl.col(column) >= start)
    if end is not None:
        mask = mask & (pl.col(column) <= end)
    return df.filter(mask)


def apply_filters(
    df: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    """Narrow the aggregated frame to the sidebar selections."""
    mask = (
        pl.col("neighborhood").is_in(neighborhoods)
        & (pl.col("date") >= start)
        & (pl.col("date") <= end)
    )
    if bedroom != "All":
        mask = mask & (pl.col("bedroom_type") == bedroom)
    return df.filter(mask)


# ---------------------------------------------------------------------------
# Chart builders (Plotly)
# ---------------------------------------------------------------------------
# _y_cap_range lives in dashboard_constants (shared with rent_scanner_tab).




def dubai_wide_transactions_chart(dw: pl.DataFrame) -> go.Figure:
    """Daily transaction volume bar chart for all of Dubai."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dw["date"].to_list(),
        y=dw["transaction_count"].to_list(),
        marker_color="#636efa",
        opacity=0.85,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Transactions: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_layout_defaults("Daily Transaction Volume — All Dubai Apartments"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="Transactions", gridcolor="#2a2e35"),
        showlegend=False,
        margin=dict(l=60, r=20, t=48, b=40),
    )
    return fig


def dubai_wide_median_price_chart(dw: pl.DataFrame) -> go.Figure:
    """Daily median/mean price with 7-day rolling averages for all of Dubai."""
    dw = dw.with_columns([
        pl.col("median_price_aed").rolling_mean(7).alias("median_7d"),
        pl.col("avg_price_aed").rolling_mean(7).alias("mean_7d"),
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dw["date"].to_list(),
        y=dw["median_price_aed"].to_list(),
        mode="markers",
        name="Median (daily)",
        marker=dict(color="#00cc96", size=5, opacity=0.35),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Median: AED %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dw["date"].to_list(),
        y=dw["median_7d"].to_list(),
        mode="lines",
        name="Median (7d avg)",
        line=dict(color="#00cc96", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Median 7d avg: AED %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dw["date"].to_list(),
        y=dw["avg_price_aed"].to_list(),
        mode="markers",
        name="Mean (daily)",
        marker=dict(color="#ab63fa", size=5, opacity=0.35),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Mean: AED %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dw["date"].to_list(),
        y=dw["mean_7d"].to_list(),
        mode="lines",
        name="Mean (7d avg)",
        line=dict(color="#ab63fa", width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Mean 7d avg: AED %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_layout_defaults("Daily Median & Mean Price (with 7-day rolling avg) — All Dubai Apartments"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=48, b=40),
    )
    return fig


def weekly_pct_change_chart(wk: pl.DataFrame) -> go.Figure:
    """Weekly % change in median and mean price."""
    fig = go.Figure()
    weeks = wk["week"].to_list()
    fig.add_trace(go.Bar(
        x=weeks,
        y=wk["median_pct_chg"].to_list(),
        name="Median % chg",
        marker_color=["#ef553b" if v is not None and v < 0 else "#00cc96" for v in wk["median_pct_chg"].to_list()],
        hovertemplate="<b>%{x|%d %b}</b><br>Median: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weeks,
        y=wk["mean_pct_chg"].to_list(),
        mode="lines+markers",
        name="Mean % chg",
        line=dict(color="#ab63fa", width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%d %b}</b><br>Mean: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#555555", line_width=1)
    fig.update_layout(
        **_layout_defaults("Weekly Median Price % Change — All Dubai Apartments"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="% Change", gridcolor="#2a2e35", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=48, b=40),
    )
    return fig


def area_pct_change_chart(area_df: pl.DataFrame) -> go.Figure:
    """Horizontal bar chart: per-area median price % change (first to last week)."""
    fig = go.Figure()
    areas = area_df["area"].to_list()
    pcts = area_df["pct_change"].to_list()
    colors = ["#ef553b" if v < 0 else "#00cc96" for v in pcts]
    fig.add_trace(go.Bar(
        y=areas,
        x=pcts,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Change: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#555555", line_width=1)
    fig.update_layout(
        **_layout_defaults("Area-Level Median Price Change (first vs last week, min 50 txns)"),
        xaxis=dict(title="% Change", gridcolor="#2a2e35", zeroline=False),
        yaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
        margin=dict(l=200, r=20, t=48, b=40),
        height=max(400, len(areas) * 22),
    )
    return fig


def tier_price_chart(tier_df: pl.DataFrame) -> go.Figure:
    """Daily median price by market tier with 7-day rolling average."""
    fig = go.Figure()
    for tier in TIER_ORDER:
        sub = tier_df.filter(pl.col("tier") == tier).sort("date")
        if sub.is_empty():
            continue
        color = TIER_COLORS[tier]
        # Faded daily dots
        fig.add_trace(go.Scatter(
            x=sub["date"].to_list(),
            y=sub["median_price"].to_list(),
            mode="markers",
            name=f"{tier} (daily)",
            legendgroup=tier,
            showlegend=False,
            marker=dict(color=color, size=4, opacity=0.3),
            hovertemplate=f"<b>{tier}</b><br>%{{x|%d %b}}<br>Median: AED %{{y:,.0f}}<extra></extra>",
        ))
        # 7-day rolling average line
        fig.add_trace(go.Scatter(
            x=sub["date"].to_list(),
            y=sub["median_7d"].to_list(),
            mode="lines",
            name=tier,
            legendgroup=tier,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{tier}</b><br>%{{x|%d %b}}<br>7d avg: AED %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        **_layout_defaults("Median Price by Market Tier (7-day rolling avg)"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=48, b=40),
    )
    return fig




def zone_psf_chart(
    zone_scored: pl.DataFrame,
    zone_rolling: pl.DataFrame,
    zone: str,
    threshold_pct: int,
) -> go.Figure:
    """One zone's transactions against its 14-day rolling median.

    Diverging encoding around the median: strong red = cleared the deal
    threshold, soft red = below the median, muted blue = at/above (kept for
    context). Same red/blue semantics as the Fair Value tab.
    """
    thr = threshold_pct / 100.0
    # Classification and hover strings are assembled in plain Python on
    # purpose: polars string kernels (pl.format / fill_null) panicked on
    # these cache-round-tripped frames in production (arrow binview
    # assertion, polars 1.43), and a Rust panic kills the Streamlit script
    # thread with the page frozen on RUNNING. Float/date .to_list() reads
    # are plain sequential accesses and have never tripped it.
    rows = zip(
        zone_scored["date"].to_list(),
        zone_scored["price_per_sqft"].to_list(),
        zone_scored["pct_vs_median"].to_list(),
        zone_scored["building"].to_list(),
        zone_scored["bedroom_type"].to_list(),
        zone_scored["size_sqft"].to_list(),
        zone_scored["price_aed"].to_list(),
    )
    points: dict[str, tuple[list, list, list]] = {
        key: ([], [], []) for key in ("non_market", "context", "below", "deal")
    }
    for day, psf, pct, building, rooms, size, price in rows:
        if pct is None:
            key, pct_label = "context", "—"
        elif pct <= -NON_MARKET_SPREAD:
            key, pct_label = "non_market", f"{pct * 100:+.1f}%"
        elif pct <= -thr:
            key, pct_label = "deal", f"{pct * 100:+.1f}%"
        elif pct < 0:
            key, pct_label = "below", f"{pct * 100:+.1f}%"
        else:
            key, pct_label = "context", f"{pct * 100:+.1f}%"
        xs, ys, custom = points[key]
        xs.append(day)
        ys.append(psf)
        custom.append((pct_label, building or "—", rooms or "—", size, price))

    hover = (
        "%{x|%d %b %Y}<br>"
        "AED %{y:,.0f}/sqft · %{customdata[0]} vs median<br>"
        "%{customdata[1]} · %{customdata[2]} · %{customdata[3]:,.0f} sqft · "
        "AED %{customdata[4]:,.0f}"
        "<extra></extra>"
    )
    layers = (
        ("non_market", f"Non-market (> {NON_MARKET_SPREAD:.0%} below)",
         dict(color="#8b949e", size=4, opacity=0.25)),
        ("context", "At/above median",
         dict(color=ZONE_COLOR_CONTEXT, size=5, opacity=0.30)),
        ("below", "Below median",
         dict(color=ZONE_COLOR_BELOW, size=5, opacity=0.45)),
        ("deal", f"Deal — ≥ {threshold_pct}% below",
         dict(color=ZONE_COLOR_BELOW, size=7, opacity=0.95,
              line=dict(width=1, color="#0e1117"))),
    )
    fig = go.Figure()
    for key, name, marker in layers:
        xs, ys, custom = points[key]
        if not xs:
            continue
        # Scattergl: WebGL markers stay responsive at thousands of dots,
        # where SVG scatter rendering freezes the browser tab.
        fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode="markers",
            name=name,
            marker=marker,
            customdata=custom,
            hovertemplate=hover,
        ))

    if not zone_rolling.is_empty():
        rx = zone_rolling["date"].to_list()
        rmed = zone_rolling["rolling_median_psf"]
        fig.add_trace(go.Scatter(
            x=rx,
            y=rmed.to_list(),
            mode="lines",
            name="14-day median",
            line=dict(color="#fafafa", width=2),
            hovertemplate="%{x|%d %b %Y}<br>14-day median: AED %{y:,.0f}/sqft<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=rx,
            y=(rmed * (1 - thr)).to_list(),
            mode="lines",
            name=f"−{threshold_pct}% threshold",
            line=dict(color=ZONE_COLOR_BELOW, width=1, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Threshold: AED %{y:,.0f}/sqft<extra></extra>",
        ))

    fig.update_layout(
        **_layout_defaults(f"{zone} — Price / sqft vs 14-day Rolling Median"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            title="AED / sqft", tickformat=",.0f", gridcolor="#2a2e35",
            range=_y_cap_range(zone_scored["price_per_sqft"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="closest",
        margin=dict(l=60, r=20, t=54, b=40),
    )
    return fig


def deals_by_zone_chart(deal_counts: pl.DataFrame, threshold_pct: int) -> go.Figure:
    """Horizontal bar of below-threshold deal counts per selected zone."""
    d = deal_counts.sort("deals")
    fig = go.Figure(go.Bar(
        x=d["deals"].to_list(),
        y=d["area"].to_list(),
        orientation="h",
        marker=dict(color=ZONE_COLOR_BELOW),
        text=d["deals"].to_list(),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} deals<extra></extra>",
    ))
    fig.update_layout(
        **_layout_defaults(
            f"Deals by zone — ≥ {threshold_pct}% below the zone's own median"
        ),
        xaxis=dict(title="Transactions", tickformat=",d",
                   gridcolor="#2a2e35", zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        height=max(240, 80 + 22 * d.height),
        showlegend=False,
        margin=dict(l=10, r=50, t=48, b=40),
    )
    return fig


def _render_headline_kpis(
    filtered: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    trans_type: str,
    start_date: date,
    end_date: date,
) -> date:
    """Page title, filter caption, and latest-month KPI cards; returns latest date."""
    # ── KPI cards ────────────────────────────────────────────────────────────────
    latest_date = filtered["date"].max()
    latest_df   = filtered.filter(pl.col("date") == latest_date)
    prev_date   = date(latest_date.year - 1, latest_date.month, 1)
    prev_df     = filtered.filter(pl.col("date") == prev_date)

    avg_price   = latest_df["avg_price_aed"].mean()
    avg_sqft    = latest_df["price_per_sqft_aed"].mean()
    avg_txn     = int(latest_df["transaction_count"].sum())

    if not prev_df.is_empty():
        prev_price = prev_df["avg_price_aed"].mean()
        yoy_delta  = (avg_price - prev_price) / prev_price * 100
        yoy_str    = f"{yoy_delta:+.1f}%"
    else:
        yoy_str    = "N/A"
        yoy_delta  = None

    st.markdown("## Dubai Apartment Prices")
    st.caption(
        f"Showing **{len(neighborhoods)}** neighbourhood(s) · "
        f"**{bedroom}** · "
        f"**{trans_type}** · "
        f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Price",     f"AED {avg_price:,.0f}",   help="Mean transaction price in the latest selected month")
    with c2:
        st.metric("Price / sqft",  f"AED {avg_sqft:,.0f}",   help="Mean price per square foot")
    with c3:
        delta_color = "normal" if yoy_delta is None else ("normal" if yoy_delta >= 0 else "inverse")
        st.metric("YoY Change",    yoy_str,                    delta=yoy_str if yoy_delta is not None else None, delta_color=delta_color)
    with c4:
        st.metric("Transactions",  f"{avg_txn:,}",            help="Total monthly transactions across selected neighbourhoods")

    st.divider()
    return latest_date


def _render_market_pulse(
    trans_type: str, data_version: str, start_date: date, end_date: date
) -> None:
    """Dubai-wide momentum KPIs and the four unfiltered market charts."""
    # ── Dubai-wide charts (unfiltered) ────────────────────────────────────────────
    DUBAI_WIDE = generate_dubai_wide_data(trans_type, data_version, start_date, end_date)
    WEEKLY = generate_weekly_data(trans_type, data_version, start_date, end_date)
    AREA_CHANGES = generate_area_weekly_change(trans_type, data_version, start_date, end_date)

    # KPI cards for Dubai-wide price momentum
    first_wk = WEEKLY.row(0, named=True)
    last_wk = WEEKLY.row(-1, named=True)
    total_med_chg = (last_wk["median"] - first_wk["median"]) / first_wk["median"] * 100
    peak_med = WEEKLY["median"].max()
    from_peak = (last_wk["median"] - peak_med) / peak_med * 100

    st.markdown("### Dubai Market Pulse")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Median Price (latest wk)",
            f"AED {last_wk['median']:,.0f}",
            delta=f"{last_wk['median_pct_chg']:+.1f}% vs prev wk",
            delta_color="inverse",
        )
    with m2:
        st.metric(
            "Period Change",
            f"{total_med_chg:+.1f}%",
            delta=f"AED {last_wk['median'] - first_wk['median']:+,.0f}",
            delta_color="inverse",
        )
    with m3:
        st.metric(
            "From Peak",
            f"{from_peak:+.1f}%",
            delta=f"AED {last_wk['median'] - peak_med:+,.0f}",
            delta_color="inverse",
            help=f"Peak median was AED {peak_med:,.0f}",
        )
    with m4:
        st.metric(
            "Mean/Median Ratio",
            f"{last_wk['mean_median_ratio']:.2f}x",
            help="Values above 1.5x suggest a skewed market with high-end outliers pulling up the mean. "
                 "Rising ratio = premium segment decoupling from the broader market.",
        )

    st.plotly_chart(dubai_wide_median_price_chart(DUBAI_WIDE), use_container_width=True)

    st.plotly_chart(weekly_pct_change_chart(WEEKLY), use_container_width=True)

    st.plotly_chart(dubai_wide_transactions_chart(DUBAI_WIDE), use_container_width=True)

    st.plotly_chart(area_pct_change_chart(AREA_CHANGES), use_container_width=True)

    st.divider()


def _render_tier_section(
    trans_type: str, data_version: str, start_date: date, end_date: date
) -> None:
    """Prime/mid/affordable tier price chart."""
    # ── Tier chart ────────────────────────────────────────────────────────────────
    TIER_DATA = generate_tier_data(trans_type, data_version, start_date, end_date)
    st.plotly_chart(tier_price_chart(TIER_DATA), use_container_width=True)

    st.divider()


def _render_zone_deal_finder(
    psf_txns: pl.DataFrame,
    psf_rolling: pl.DataFrame,
    zone: str,
    neighborhoods: list[str],
    threshold_pct: int,
) -> None:
    """Below-median deal finder: KPIs, diverging scanner, ranked deals table."""
    thr = threshold_pct / 100.0

    # Score every selected zone against its own median so the per-zone deal
    # counts at the bottom stay comparable across zones.
    scored = (
        psf_txns
        .filter(pl.col("area").is_in(neighborhoods))
        .join(
            psf_rolling.select(["date", "area", "rolling_median_psf"]),
            on=["date", "area"],
            how="left",
        )
        .with_columns(
            (pl.col("price_per_sqft") / pl.col("rolling_median_psf") - 1)
            .alias("pct_vs_median")
        )
    )
    zone_scored = scored.filter(pl.col("area") == zone)
    if zone_scored.is_empty():
        st.warning(f"No transactions for {zone} in the selected window.")
        return
    zone_rolling = psf_rolling.filter(
        (pl.col("area") == zone) & pl.col("rolling_median_psf").is_not_null()
    )
    deals = zone_scored.filter(
        (pl.col("pct_vs_median") <= -thr)
        & (pl.col("pct_vs_median") > -NON_MARKET_SPREAD)
    ).sort("pct_vs_median")

    st.caption(
        "**Dots** = individual DLD transactions in the zone. "
        "**Solid line** = its 14-day rolling median AED/sqft; "
        "**dashed line** = the deal threshold below it. "
        "**Red dots** closed under the median — the strong red ones cleared "
        "the threshold and are listed in the table underneath. Grey dots more "
        f"than {NON_MARKET_SPREAD:.0%} below are gifts/token-price transfers, "
        "not buyable prices, and are excluded from the deal list."
    )

    latest_median = (
        zone_rolling["rolling_median_psf"].tail(1).item()
        if zone_rolling.height
        else None
    )
    deepest = deals["pct_vs_median"].min() if deals.height else None
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "14d median (latest)",
            f"AED {latest_median:,.0f}/sqft" if latest_median is not None else "N/A",
            help="Latest 14-day rolling median price per sqft in this zone",
        )
    with k2:
        st.metric(
            "Transactions",
            f"{zone_scored.height:,}",
            help="Transactions in this zone in the selected date range",
        )
    with k3:
        st.metric(
            f"Deals ≥ {threshold_pct}% below",
            f"{deals.height:,}",
            help="Transactions that closed at least the threshold below the rolling median",
        )
    with k4:
        st.metric(
            "Deepest discount",
            f"{abs(deepest) * 100:.0f}% below" if deepest is not None else "—",
            help="The furthest any transaction closed below the rolling median",
        )

    try:
        zone_fig = zone_psf_chart(zone_scored, zone_rolling, zone, threshold_pct)
    except PanicException:
        # A polars Rust panic derives from BaseException, escapes Streamlit's
        # `except Exception` handler, and kills the script thread with the
        # page frozen on RUNNING. Degrade to an error box instead and keep
        # the deals table below alive.
        LOGGER.exception("zone_psf_chart hit a polars panic")
        zone_fig = None
    if zone_fig is None:
        st.error(
            "The scanner chart failed for this filter combination — the deals "
            "table below still works. Try a different zone/bedroom selection."
        )
    else:
        st.plotly_chart(zone_fig, use_container_width=True)

    st.markdown(f"### Below-median deals — {zone}")
    if deals.is_empty():
        st.info(
            f"No transactions closed ≥ {threshold_pct}% below the 14-day median "
            "in this window. Lower the threshold or widen the date range."
        )
    else:
        deals_display = deals.select([
            pl.col("date").cast(pl.Utf8).alias("Date"),
            pl.col("building").alias("Building"),
            pl.col("bedroom_type").alias("Rooms"),
            pl.col("size_sqft").alias("Size (sqft)"),
            pl.col("price_aed").round(0).alias("Price (AED)"),
            pl.col("price_per_sqft").round(0).alias("AED/sqft"),
            pl.col("rolling_median_psf").round(0).alias("Zone median AED/sqft"),
            (pl.col("pct_vs_median") * 100).round(1).alias("% vs median"),
            pl.col("offplan").alias("Registration"),
        ])
        st.dataframe(
            deals_display,
            use_container_width=True,
            height=min(400, 38 + 35 * deals_display.height),
        )
        zone_slug = "".join(ch if ch.isalnum() else "_" for ch in zone.lower())
        st.download_button(
            "Download deals CSV",
            data=deals_display.write_csv().encode(),
            file_name=f"deals_{zone_slug}.csv",
            mime="text/csv",
        )
        st.caption(
            "Below-median ≠ automatic bargain: small, low-floor, and older units "
            "sit below the median naturally, and the deepest discounts are often "
            "bulk allocations or related-party transfers. Cross-check candidates "
            "on the Fair Value page, which prices each unit's own characteristics."
        )

    if len(neighborhoods) > 1:
        deal_counts = (
            scored
            .filter(
                (pl.col("pct_vs_median") <= -thr)
                & (pl.col("pct_vs_median") > -NON_MARKET_SPREAD)
            )
            .group_by("area")
            .agg(pl.len().alias("deals"))
        )
        if not deal_counts.is_empty():
            st.markdown("### Where the deals are")
            st.caption(
                f"Transactions ≥ {threshold_pct}% below their own zone's 14-day "
                "median, across the zones selected in the sidebar. Zones without "
                "deals are not shown."
            )
            st.plotly_chart(
                deals_by_zone_chart(deal_counts, threshold_pct),
                use_container_width=True,
            )


def _render_zone_analysis(
    filtered: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    trans_type: str,
    data_version: str,
    start_date: date,
    end_date: date,
    date_min: date,
    date_max: date,
) -> None:
    """Render the Buyer Opportunity Scanner page: below-median deals per zone."""
    if filtered.is_empty():
        st.warning(
            "No data for the selected filters/date range. "
            f"The loaded data covers {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}."
        )
        return

    st.markdown("## Buyer Opportunity Scanner")
    st.caption(
        "Sidebar filters apply to everything on this page — "
        f"**{len(neighborhoods)}** neighbourhood(s) · "
        f"Bedrooms: **{bedroom}** · "
        f"Type: **{trans_type}** · "
        f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}"
    )

    zone_options = [n for n in NEIGHBORHOODS if n in neighborhoods]
    if st.session_state.get("zone_select") not in zone_options:
        st.session_state.pop("zone_select", None)
    zone_col, threshold_col = st.columns([2, 3])
    with zone_col:
        zone = st.selectbox(
            "Zone",
            zone_options,
            key="zone_select",
            help="Every chart below focuses on this zone",
        )
    with threshold_col:
        threshold_pct = st.slider(
            "Deal threshold — % below the zone's 14-day median",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="zone_threshold",
            help="A transaction at least this far under the rolling median counts as a deal",
        )

    PSF_TXNS, PSF_ROLLING = generate_area_psf_timeseries(
        trans_type, bedroom, data_version, start_date, end_date,
    )
    _render_zone_deal_finder(PSF_TXNS, PSF_ROLLING, zone, neighborhoods, threshold_pct)

    # ── Raw data table ────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Data Table", expanded=False):
        display_df = (
            filtered
            .sort(["date", "neighborhood", "bedroom_type"], descending=[True, False, False])
            .with_columns(pl.col("date").cast(pl.Utf8))
        )
        st.dataframe(display_df, use_container_width=True, height=300)
        csv_bytes = "\n".join(
            [",".join(display_df.columns)] +
            [",".join(str(v) for v in row) for row in display_df.iter_rows()]
        ).encode()
        st.download_button("Download CSV", data=csv_bytes, file_name="dubai_re_data.csv", mime="text/csv")


def _render_market_overview(
    filtered: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    trans_type: str,
    data_version: str,
    start_date: date,
    end_date: date,
    date_min: date,
    date_max: date,
) -> None:
    """Render the Market Overview page by sequencing its sections."""
    if filtered.is_empty():
        st.warning(
            "No data for the selected filters/date range. "
            f"The loaded data covers {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}."
        )
        return

    _render_headline_kpis(
        filtered, neighborhoods, bedroom, trans_type, start_date, end_date
    )
    _render_market_pulse(trans_type, data_version, start_date, end_date)
    _render_tier_section(trans_type, data_version, start_date, end_date)

# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dubai Real Estate Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS
st.markdown(
    """
    <style>
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        [data-testid="stMetricLabel"] { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }
        /* Must clear the ~3.75rem sticky stAppHeader or the first element
           (the page selector) renders underneath it, clipped and unclickable. */
        .block-container { padding-top: 4.5rem; }
        h1 { font-size: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    api_snapshot, api_meta = load_production_transactions()
except Exception as exc:
    LOGGER.exception("DDA production data source startup failed: %s", exc)
    st.error(
        "Production data is temporarily unavailable. Please try again shortly."
    )
    st.stop()

st.session_state["api_raw_df"] = api_snapshot
st.session_state["api_meta"] = api_meta
data_version = api_meta.get("loaded_at", "production")
active_api_rows = api_snapshot.height

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏙️ Dubai RE Dashboard")
    st.caption("Apartments · Latest DLD Transactions")
    st.divider()

    neighborhoods = st.multiselect(
        "Neighbourhood",
        options=NEIGHBORHOODS,
        default=NEIGHBORHOODS,
        help="Select one or more Dubai neighbourhoods",
    )

    trans_type = st.radio(
        "Transaction Type",
        options=["All", "Sale", "Mortgage", "Off-Plan"],
        index=0,
        horizontal=True,
        help="Sale = ready secondary market · Mortgage = financed purchases · Off-Plan = under-construction units",
    )

    bedroom = st.radio(
        "Bedroom Type",
        options=["Studio", "1BR", "2BR", "3BR", "All"],
        index=4,
        horizontal=True,
    )

    date_min, date_max, default_end = _date_picker_bounds()
    # Show the last 12 months by default; the snapshot keeps 24 months for
    # model training, and the picker still allows selecting the full range.
    default_start = max(date_min, months_before(default_end, 12))

    date_range = st.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=date_min,
        max_value=date_max,
        format="YYYY/MM/DD",
        key="date_range",
    )
    # Safely unpack; user may still be selecting end date
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
    requested_start_date = start_date
    requested_end_date = end_date
    start_date = max(start_date, date_min)
    end_date = min(end_date, date_max)
    if start_date > end_date:
        start_date, end_date = date_min, date_max
    refresh_info = None
    try:
        with st.spinner("Refreshing missing DLD transactions..."):
            api_snapshot, api_meta, refresh_info = _ensure_transactions_cover_range(
                api_snapshot,
                api_meta,
                start_date,
                end_date,
            )
    except Exception as exc:
        LOGGER.exception("Incremental transaction refresh failed: %s", exc)
        st.warning(
            "Could not refresh the missing date range from DDA. "
            "Showing the available GCS snapshot."
        )

    st.session_state["api_raw_df"] = api_snapshot
    st.session_state["api_meta"] = api_meta
    data_version = api_meta.get("loaded_at", data_version)
    active_api_rows = api_snapshot.height
    api_bounds = _date_bounds_from_transactions(api_snapshot)
    if api_bounds:
        date_min = min(date_min, api_bounds[0])
        date_max = max(date_max, api_bounds[1])
    if refresh_info:
        if refresh_info.get("written"):
            added = refresh_info.get("added_rows", 0)
            if added:
                st.caption(f"Updated GCS cache: +{added:,} rows.")
            else:
                st.caption("Cleaned and verified the GCS cache.")
        elif refresh_info.get("added_rows", 0) == 0:
            st.caption("No newer DLD records were available for the requested range.")

    if (start_date, end_date) != (requested_start_date, requested_end_date):
        st.caption(
            "Adjusted to loaded data range: "
            f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}."
        )
    st.divider()
    date_summary = f"{date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}"
    if api_bounds:
        date_summary = f"{api_bounds[0]:%Y-%m-%d} to {api_bounds[1]:%Y-%m-%d}"
    st.caption(f"Data covers {date_summary} ({active_api_rows:,} transactions).")

# ── Load & filter data ───────────────────────────────────────────────────────
DAILY_DATA = generate_dubai_data(trans_type, data_version)

if not neighborhoods:
    st.warning("Select at least one neighbourhood in the sidebar.")
    st.stop()

# Segmented control instead of st.tabs: st.tabs executes every tab body on
# each rerun, which would load the model bundle and score transactions even
# when the user never opens the Fair Value view. Only the active page runs,
# which likewise keeps the Buyer Opportunity Scanner charts and CSV build
# off the other pages' rerun path.
_PAGE_OPTIONS = [
    "📊 Market Overview",
    "🔎 Buyer Opportunity Scanner",
    "🔑 Rent Scanner",
    "🎯 Fair Value Model",
]
# A stored label from before a page rename (or a deselected None) would no
# longer match the options and crash the widget — drop it first.
if st.session_state.get("active_page") not in _PAGE_OPTIONS:
    st.session_state.pop("active_page", None)
active_page = st.segmented_control(
    "Page",
    _PAGE_OPTIONS,
    default="📊 Market Overview",
    key="active_page",
    label_visibility="collapsed",
)
# Clicking the highlighted segment deselects it (returns None); treat that
# as the default page rather than rendering an unselected state.
if active_page is None:
    active_page = "📊 Market Overview"

# Streamlit drops the state of widgets that are not rendered in a run; the
# Fair Value and Buyer Opportunity Scanner widgets disappear while another page is shown,
# so re-assign their keys every run to keep the user's selections alive.
for _page_key in (
    "fv_window",
    "fv_threshold",
    "zone_select",
    "zone_threshold",
    "rent_threshold",
    "rent_incl_renewals",
):
    if _page_key in st.session_state:
        st.session_state[_page_key] = st.session_state[_page_key]

if active_page == "🎯 Fair Value Model":
    render_fair_value_tab(neighborhoods, bedroom, start_date, end_date, data_version)
elif active_page == "🔑 Rent Scanner":
    # Rent data lives in its own GCS artifacts — no DAILY_DATA/apply_filters
    # here, and all rent reads happen inside the page module (lazy). The
    # sales PSF generator is injected for the implied-gross-yield KPI.
    render_rent_scanner(
        neighborhoods,
        bedroom,
        start_date,
        end_date,
        generate_area_psf_timeseries,
        data_version,
    )
elif active_page == "🔎 Buyer Opportunity Scanner":
    # Fair Value never consumes the filtered frame — compute it only on the
    # pages that do, so Fair Value interactions don't pay for an unused
    # aggregation.
    filtered = apply_filters(DAILY_DATA, neighborhoods, bedroom, start_date, end_date)
    _render_zone_analysis(
        filtered,
        neighborhoods,
        bedroom,
        trans_type,
        data_version,
        start_date,
        end_date,
        date_min,
        date_max,
    )
else:
    filtered = apply_filters(DAILY_DATA, neighborhoods, bedroom, start_date, end_date)
    _render_market_overview(
        filtered,
        neighborhoods,
        bedroom,
        trans_type,
        data_version,
        start_date,
        end_date,
        date_min,
        date_max,
    )
