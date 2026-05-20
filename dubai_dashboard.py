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
from datetime import date, datetime

import polars as pl
import plotly.colors
import plotly.graph_objects as go
import streamlit as st

from dda_api import (
    DDAApiError,
    DEFAULT_MAX_RECORDS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_LOOKBACK_MONTHS,
    build_dld_transactions_params,
    fetch_dataset_records,
    infer_column_mapping,
    last_months_date_range,
    load_dda_config,
    normalize_dld_transactions,
    records_to_dataframe,
    validate_normalized_columns,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_API = "Dubai Data API - PROD"
DEFAULT_SOURCE = SOURCE_API
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

NEIGHBORHOODS: list[str] = [
    # High-volume
    "JUMEIRAH VILLAGE CIRCLE",
    "BUSINESS BAY",
    "MAJAN",
    "DUBAI MARINA",
    "BURJ KHALIFA",
    "JUMEIRAH LAKES TOWERS",
    "DUBAI CREEK HARBOUR",
    "ARJAN",
    "DUBAI SPORTS CITY",
    "SILICON OASIS",
    # Mid-volume
    "INTERNATIONAL CITY PH 1",
    "MEYDAN ONE",
    "DISCOVERY GARDENS",
    "AL FURJAN",
    "SOBHA HEARTLAND",
    "PALM JUMEIRAH",
    "DUBAI HILLS",
    "THE GREENS",
    "DUBAI PRODUCTION CITY",
    "MOTOR CITY",
]

DATE_START = date(2026, 2, 2)
DATE_END   = date(2026, 3, 19)

TIER_MAP: dict[str, str] = {}
TIER_AREAS: dict[str, list[str]] = {
    "Ultra-premium": [
        "BLUEWATERS", "DUBAI HARBOUR", "DUBAI WATER CANAL",
    ],
    "Premium": [
        "PALM JUMEIRAH", "BURJ KHALIFA", "DUBAI CREEK HARBOUR", "DUBAI HILLS",
        "MEYDAN ONE", "JUMEIRAH BEACH RESIDENCE", "AL BARARI",
    ],
    "Mid-market": [
        "DUBAI MARINA", "BUSINESS BAY", "JUMEIRAH LAKES TOWERS", "SOBHA HEARTLAND",
        "Business Bay", "THE GREENS", "AL FURJAN", "DUBAI HEALTHCARE CITY - PHASE 2",
        "DAMAC HILLS", "JUMEIRAH VILLAGE TRIANGLE", "TOWN SQUARE", "Al Yelayiss 2",
    ],
    "Value": [
        "JUMEIRAH VILLAGE CIRCLE", "ARJAN", "MOTOR CITY", "DISCOVERY GARDENS",
        "SILICON OASIS", "DUBAI PRODUCTION CITY", "DUBAI SOUTH",
    ],
    "Budget": [
        "INTERNATIONAL CITY PH 1", "DUBAI LAND RESIDENCE COMPLEX", "LIWAN",
        "DUBAI SPORTS CITY", "MAJAN", "DUBAI INVESTMENT PARK SECOND",
    ],
}
for tier, areas in TIER_AREAS.items():
    for a in areas:
        TIER_MAP[a.upper()] = tier
TIER_ORDER = ["Ultra-premium", "Premium", "Mid-market", "Value", "Budget"]
TIER_COLORS = {
    "Ultra-premium": "#e377c2",
    "Premium":       "#ff7f0e",
    "Mid-market":    "#636efa",
    "Value":         "#00cc96",
    "Budget":        "#ffa15a",
}

# Alphabet palette has 26 entries – enough for 20 neighbourhoods
COLORS = plotly.colors.qualitative.Alphabet
COLOR_MAP = {n: COLORS[i % len(COLORS)] for i, n in enumerate(NEIGHBORHOODS)}

# ---------------------------------------------------------------------------
# Data loading (Polars) - production DDA API
# ---------------------------------------------------------------------------

def _source_raw_transactions(data_source: str) -> pl.DataFrame:
    api_df = st.session_state.get("api_raw_df")
    if isinstance(api_df, pl.DataFrame) and not api_df.is_empty():
        return api_df
    return pl.DataFrame(schema=TRANSACTION_SCHEMA)


def _flat_transactions(data_source: str, trans_type: str) -> pl.DataFrame:
    return _trans_type_filter(
        _source_raw_transactions(data_source).filter(
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
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
) -> pl.DataFrame:
    """Aggregate real DLD apartment transactions from the production API.

    Aggregates individual transactions to daily averages per
    neighbourhood and bedroom type, matching the dashboard schema.
    """
    raw = _flat_transactions(data_source, trans_type)
    return (
        raw
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            pl.col("AREA_EN").alias("neighborhood"),
            pl.col("ROOMS_EN").str.replace(" B/R", "BR").alias("bedroom_type"),
            (pl.col("TRANS_VALUE") / pl.col("ACTUAL_AREA")).alias("price_per_sqft"),
        ])
        .group_by(["date", "neighborhood", "bedroom_type"])
        .agg([
            pl.col("TRANS_VALUE").mean().round(0).alias("avg_price_aed"),
            pl.col("price_per_sqft").mean().round(1).alias("price_per_sqft_aed"),
            pl.col("ACTUAL_AREA").mean().round(1).alias("avg_size_sqft"),
            pl.col("TRANS_VALUE").count().alias("transaction_count"),
        ])
    )


@st.cache_data(show_spinner="Computing Dubai-wide aggregates...")
def generate_dubai_wide_data(
    trans_type: str = "All",
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
) -> pl.DataFrame:
    """Daily Dubai-wide transaction count and median price (all flats)."""
    return (
        _flat_transactions(data_source, trans_type)
        .with_columns(
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
        )
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
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
) -> pl.DataFrame:
    """Weekly Dubai-wide aggregates with % change."""
    return (
        _flat_transactions(data_source, trans_type)
        .with_columns(
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .dt.truncate("1w")
              .alias("week"),
        )
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
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
) -> pl.DataFrame:
    """Per-area first-to-last-week median price % change (min 50 txns)."""
    raw = (
        _flat_transactions(data_source, trans_type)
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .dt.truncate("1w")
              .alias("week"),
            pl.col("AREA_EN").alias("area"),
        ])
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
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
) -> pl.DataFrame:
    """Daily median price per market tier."""
    return (
        _flat_transactions(data_source, trans_type)
        .filter(pl.col("AREA_EN").is_in(list(TIER_MAP.keys())))
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            pl.col("AREA_EN").replace_strict(TIER_MAP).alias("tier"),
        ])
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
    data_source: str = DEFAULT_SOURCE,
    data_version: int = 0,
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
    txns_df   : one row per transaction — [date, area, price_per_sqft]
    rolling_df: one row per (date, area) — [date, area, daily_median_psf,
                rolling_median_psf]
    """
    raw = (
        _flat_transactions(data_source, trans_type)
        .filter(pl.col("ACTUAL_AREA") > 0)   # guard against divide-by-zero
        .with_columns([
            pl.col("INSTANCE_DATE").str.slice(0, 10)
              .str.to_date("%Y-%m-%d")
              .alias("date"),
            pl.col("AREA_EN").str.to_uppercase().alias("area"),
            pl.col("ROOMS_EN").str.replace(" B/R", "BR").alias("bedroom_type"),
            (pl.col("TRANS_VALUE") / pl.col("ACTUAL_AREA")).alias("price_per_sqft"),
        ])
    )

    if bedroom != "All":
        raw = raw.filter(pl.col("bedroom_type") == bedroom)

    # Individual transactions for scatter dots
    txns_df = (
        raw
        .select(["date", "area", "price_per_sqft"])
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

    return txns_df, rolling_df


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


def fetch_dld_live_data(
    api_key: str,
    api_secret: str,
    start: str = "2026-01-17",
    end: str   = "2026-05-17",
    security_application_identifier: str = "",
) -> pl.DataFrame:
    """Fetch real apartment transactions from the Dubai Data / DLD API.

    Registration
    ------------
    1. Go to https://data.dubai/en/
    2. Create a free account
    3. Find the DLD Transactions dataset and click "Request API Access Key"
    4. You will receive an API Key and API Secret in two separate emails
    5. Free tier: limited monthly calls

    Parameters
    ----------
    api_key    : client_id from Dubai Data email
    api_secret : client_secret from Dubai Data email
    start / end: ISO date strings for the transaction date range
    """
    import requests

    token = _get_dld_token(api_key, api_secret, security_application_identifier)

    API_URL = "https://apis.data.dubai/open/dld/dld_transactions-open-api"
    all_records: list[dict] = []
    offset = 0
    page_size = 500

    while True:
        resp = requests.get(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "filter": (
                    f"property_sub_type_en='Flat' AND trans_group_en='Sales'"
                    f" AND instance_date>='{start}' AND instance_date<='{end}'"
                ),
                "order_by": "instance_date",
                "limit":  page_size,
                "offset": offset,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        records = data if isinstance(data, list) else data.get("records", data.get("result", []))
        if not records:
            break
        all_records.extend(records)
        if len(records) < page_size:
            break
        offset += page_size

    if not all_records:
        return pl.DataFrame()

    # Rename API columns to match the CSV column names the dashboard expects
    df = pl.DataFrame(all_records)
    rename_map = {k: v for k, v in _API_TO_CSV.items() if k in df.columns}
    return df.rename(rename_map)


def fetch_bayut_transactions(
    api_key: str,
    neighborhood: str = "Dubai Marina",
    bedrooms: int = 1,
) -> pl.DataFrame:
    """Fetch listings/transactions from the Bayut API (750 free calls/month).

    Registration
    ------------
    1. Go to https://bayutapi.com/ → sign up (no credit card required)
    2. API key sent by email
    3. Docs: https://docs.bayutapi.com/
    """
    import requests

    resp = requests.get(
        "https://api.bayutapi.com/v1/transactions",
        headers={"X-API-Key": api_key},
        params={
            "location": neighborhood,
            "bedrooms": bedrooms,
            "purpose":  "for-sale",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return pl.DataFrame(resp.json().get("transactions", []))


# Production secrets are read server-side only; no credentials are exposed in
# the Streamlit interface.
def _streamlit_secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _load_api_transactions(
    config,
    start: date,
    end: date,
    page_size: int,
    max_records: int,
) -> tuple[pl.DataFrame, dict]:
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


def _probe_api_columns(config, limit: int = 10) -> tuple[pl.DataFrame, dict]:
    records = fetch_dataset_records(
        config,
        params={"order_by": "instance_date", "order_dir": "desc"},
        page_size=limit,
        max_records=limit,
    )
    raw_df = records_to_dataframe(records)
    normalized = normalize_dld_transactions(raw_df)
    return normalized, {
        "raw_columns": raw_df.columns,
        "mapping": infer_column_mapping(raw_df.columns),
        "validation": validate_normalized_columns(normalized),
    }


def _date_bounds_from_transactions(df: pl.DataFrame) -> tuple[date, date] | None:
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


@st.cache_data(
    show_spinner="Loading latest DLD transactions...",
    ttl=60 * 60,
)
def load_production_transactions() -> tuple[pl.DataFrame, dict]:
    config = load_dda_config(_streamlit_secrets())
    missing = config.missing_fields()
    if missing:
        raise DDAApiError(
            "Missing production data source configuration: "
            + ", ".join(missing)
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
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
        "date_bounds": {
            "start": bounds[0].isoformat() if bounds else start.isoformat(),
            "end": bounds[1].isoformat() if bounds else end.isoformat(),
        },
    }


def _loaded_date_bounds_for_source(data_source: str) -> tuple[date, date]:
    api_df = st.session_state.get("api_raw_df")
    if isinstance(api_df, pl.DataFrame):
        bounds = _date_bounds_from_transactions(api_df)
        if bounds:
            return bounds
    return last_months_date_range(API_DEFAULT_LOOKBACK_MONTHS)


def _date_picker_bounds(data_source: str) -> tuple[date, date, date]:
    loaded_start, loaded_end = _loaded_date_bounds_for_source(data_source)
    return loaded_start, loaded_end, loaded_end


# ---------------------------------------------------------------------------
# Filter helper (Polars)
# ---------------------------------------------------------------------------

def _trans_type_filter(df: pl.DataFrame, trans_type: str) -> pl.DataFrame:
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


def apply_filters(
    df: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    start: date,
    end: date,
) -> pl.DataFrame:
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

def _layout_defaults(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13), x=0.01),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#fafafa"),
    )


def line_chart(df: pl.DataFrame, bedroom: str) -> go.Figure:
    fig   = go.Figure()
    dash_ = {"Studio": "dot", "1BR": "solid", "2BR": "dash", "3BR": "dashdot"}
    nbhds = df["neighborhood"].unique().to_list()
    br_types = df["bedroom_type"].unique().sort().to_list()

    for nbhd in NEIGHBORHOODS:          # stable iteration order
        if nbhd not in nbhds:
            continue
        color = COLOR_MAP[nbhd]
        for br in br_types:
            sub = (
                df.filter((pl.col("neighborhood") == nbhd) & (pl.col("bedroom_type") == br))
                .sort("date")
            )
            if sub.is_empty():
                continue
            show_leg = br == br_types[0]
            fig.add_trace(go.Scatter(
                x=sub["date"].to_list(),
                y=sub["avg_price_aed"].to_list(),
                mode="lines",
                name=nbhd,
                legendgroup=nbhd,
                showlegend=show_leg,
                line=dict(color=color, width=2, dash=dash_.get(br, "solid")),
                hovertemplate=(
                    f"<b>{nbhd} – {br}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Avg Price: AED %{y:,.0f}<br><extra></extra>"
                ),
            ))

    fig.update_layout(
        **_layout_defaults("Average Apartment Price Over Time"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=48, b=40),
    )
    if len(br_types) > 1:
        legend_parts = [f"{dash_.get(br, 'solid').capitalize()} = {br}" for br in br_types if br in dash_]
        if legend_parts:
            fig.add_annotation(
                text=" · ".join(legend_parts),
                xref="paper", yref="paper", x=0.01, y=0.97,
                showarrow=False, font=dict(size=10, color="#8b949e"),
            )
    return fig


def bar_chart(df: pl.DataFrame, latest_date: date) -> go.Figure:
    fig     = go.Figure()
    latest  = df.filter(pl.col("date") == latest_date)
    br_types = latest["bedroom_type"].unique().sort().to_list()

    # Sort neighbourhoods by descending avg price
    order = (
        latest.group_by("neighborhood")
        .agg(pl.col("avg_price_aed").mean())
        .sort("avg_price_aed", descending=True)["neighborhood"]
        .to_list()
    )

    opacity_map = {"Studio": 0.50, "1BR": 1.0, "2BR": 0.80, "3BR": 0.65}
    for br in br_types:
        sub = latest.filter(pl.col("bedroom_type") == br)
        prices = {r["neighborhood"]: r["avg_price_aed"] for r in sub.iter_rows(named=True)}
        fig.add_trace(go.Bar(
            x=order,
            y=[prices.get(n, None) for n in order],
            name=br,
            marker_color=[COLOR_MAP[n] for n in order],
            opacity=opacity_map.get(br, 1.0),
            hovertemplate="<b>%{x}</b><br>Avg Price: AED %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_layout_defaults(f"Price Comparison · {latest_date.strftime('%b %Y')}"),
        barmode="group",
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=48, b=95),
    )
    return fig


def price_vs_time_scatter(df: pl.DataFrame) -> go.Figure:
    """Scatter of avg_price_aed vs date for every neighbourhood in the filtered set.

    One trace per neighbourhood; each monthly data point is a dot, coloured by
    neighbourhood.  Bedroom type and transaction count appear in the tooltip.
    """
    fig   = go.Figure()
    nbhds = set(df["neighborhood"].to_list())

    for nbhd in NEIGHBORHOODS:
        if nbhd not in nbhds:
            continue
        sub = df.filter(pl.col("neighborhood") == nbhd).sort("date")
        cd  = sub.select(["bedroom_type", "transaction_count"]).to_numpy()
        fig.add_trace(go.Scatter(
            x=sub["date"].to_list(),
            y=sub["avg_price_aed"].to_list(),
            mode="markers",
            name=nbhd,
            legendgroup=nbhd,
            marker=dict(
                color=COLOR_MAP[nbhd],
                size=5,
                opacity=0.70,
                line=dict(width=0),
            ),
            customdata=cd,
            hovertemplate=(
                f"<b>{nbhd}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Avg Price: AED %{y:,.0f}<br>"
                "Type: %{customdata[0]}<br>"
                "Transactions: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_layout_defaults("Avg Price vs Time — All Neighbourhoods"),
        xaxis=dict(showgrid=False, zeroline=False, title="Date"),
        yaxis=dict(title="Avg Price (AED)", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            font=dict(size=9),
            tracegroupgap=2,
        ),
        hovermode="closest",
        margin=dict(l=60, r=140, t=48, b=40),
    )
    return fig


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


def area_psf_chart(
    txns_df: pl.DataFrame,
    rolling_df: pl.DataFrame,
    neighborhoods: list[str],
) -> go.Figure:
    """Scatter + line chart: individual transactions (dots) and 14-day rolling
    area median (solid line) for AED/sqft over time.

    Dots below the line = that deal was cheaper than the area's recent median
    — the clearest transaction-level buy signal available in the data.
    """
    fig = go.Figure()

    for nbhd in NEIGHBORHOODS:           # stable ordering
        if nbhd not in neighborhoods:
            continue
        color = COLOR_MAP[nbhd]

        t = txns_df.filter(pl.col("area") == nbhd).sort("date")
        r = rolling_df.filter(pl.col("area") == nbhd).sort("date")

        if t.is_empty():
            continue

        # ── Scatter: individual transactions ──────────────────────────────
        fig.add_trace(go.Scatter(
            x=t["date"].to_list(),
            y=t["price_per_sqft"].to_list(),
            mode="markers",
            name=nbhd,
            legendgroup=nbhd,
            showlegend=True,
            marker=dict(color=color, size=5, opacity=0.35),
            hovertemplate=(
                f"<b>{nbhd}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Transaction: AED %{y:,.0f}/sqft"
                "<extra></extra>"
            ),
        ))

        # ── Line: 14-day rolling median of daily area median ──────────────
        if not r.is_empty():
            fig.add_trace(go.Scatter(
                x=r["date"].to_list(),
                y=r["rolling_median_psf"].to_list(),
                mode="lines",
                name=f"{nbhd} 14d median",
                legendgroup=nbhd,
                showlegend=False,            # share legend entry with dots
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{nbhd}</b><br>"
                    "%{x|%d %b %Y}<br>"
                    "14-day median: AED %{y:,.0f}/sqft"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        **_layout_defaults(
            "Price / sqft — Individual Transactions (dots) & 14-day Rolling Area Median (line)"
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="AED / sqft", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=9),
        ),
        hovermode="closest",
        margin=dict(l=60, r=20, t=54, b=40),
    )
    return fig


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
        .block-container { padding-top: 1.2rem; }
        h1 { font-size: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

data_source = SOURCE_API
try:
    api_snapshot, api_meta = load_production_transactions()
except DDAApiError as exc:
    LOGGER.exception("DDA production data source startup failed: %s", exc)
    st.error(
        "Production data is unavailable. Ask the deployment owner to configure "
        "the DDA secrets in the hosting environment. Check the Streamlit app "
        "logs for the missing key names."
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

    date_min, date_max, default_end = _date_picker_bounds(data_source)
    default_start = date_min

    date_range = st.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=date_min,
        max_value=date_max,
        format="YYYY/MM/DD",
        key=f"date_range_{data_source}_{data_version}_{date_min}_{date_max}",
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
    if (start_date, end_date) != (requested_start_date, requested_end_date):
        st.caption(
            "Adjusted to loaded data range: "
            f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}."
        )
    st.divider()
    api_bounds = _date_bounds_from_transactions(api_snapshot)
    date_summary = f"{date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}"
    if api_bounds:
        date_summary = f"{api_bounds[0]:%Y-%m-%d} to {api_bounds[1]:%Y-%m-%d}"
    st.caption(f"Data covers {date_summary} ({active_api_rows:,} transactions).")

# ── Load & filter data ───────────────────────────────────────────────────────
DF = generate_dubai_data(trans_type, data_source, data_version)

if not neighborhoods:
    st.warning("Select at least one neighbourhood in the sidebar.")
    st.stop()

filtered = apply_filters(DF, neighborhoods, bedroom, start_date, end_date)

if filtered.is_empty():
    st.warning(
        "No data for the selected filters/date range. "
        f"The loaded data covers {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}."
    )
    st.stop()

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

# ── Dubai-wide charts (unfiltered) ────────────────────────────────────────────
DW = generate_dubai_wide_data(trans_type, data_source, data_version)
WK = generate_weekly_data(trans_type, data_source, data_version)
AREA_CHG = generate_area_weekly_change(trans_type, data_source, data_version)

# KPI cards for Dubai-wide price momentum
first_wk = WK.row(0, named=True)
last_wk = WK.row(-1, named=True)
total_med_chg = (last_wk["median"] - first_wk["median"]) / first_wk["median"] * 100
peak_med = WK["median"].max()
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

st.plotly_chart(dubai_wide_median_price_chart(DW), use_container_width=True)

st.plotly_chart(weekly_pct_change_chart(WK), use_container_width=True)

st.plotly_chart(dubai_wide_transactions_chart(DW), use_container_width=True)

st.plotly_chart(area_pct_change_chart(AREA_CHG), use_container_width=True)

st.divider()

# ── Tier chart ────────────────────────────────────────────────────────────────
TIER_DF = generate_tier_data(trans_type, data_source, data_version)
st.plotly_chart(tier_price_chart(TIER_DF), use_container_width=True)

st.divider()

# ── Buyer Opportunity Scanner (Metric #3) ────────────────────────────────────
st.markdown("### Buyer Opportunity Scanner — Price / sqft vs Area Median")
st.caption(
    "**Dots** = individual DLD transactions. "
    "**Solid line** = 14-day rolling median of each area's own daily median AED/sqft. "
    "A dot **below** the line means that specific deal closed cheaper than the area's "
    "recent median — the clearest transaction-level buy signal in the data (Metric #3)."
)
PSF_TXNS, PSF_ROLLING = generate_area_psf_timeseries(
    trans_type,
    bedroom,
    data_source,
    data_version,
)

# KPI row: cheapest current 14-day rolling median among selected areas
if not PSF_ROLLING.is_empty():
    latest_rolling = (
        PSF_ROLLING
        .filter(pl.col("area").is_in(neighborhoods))
        .filter(pl.col("rolling_median_psf").is_not_null())
        .group_by("area")
        .agg(pl.col("rolling_median_psf").last().alias("latest_median_psf"))
        .sort("latest_median_psf")
        .head(3)
    )
    if not latest_rolling.is_empty():
        cols = st.columns(len(latest_rolling))
        for i, row in enumerate(latest_rolling.iter_rows(named=True)):
            with cols[i]:
                st.metric(
                    label=f"{row['area'].title()} — lowest AED/sqft",
                    value=f"AED {row['latest_median_psf']:,.0f}/sqft",
                    help="Latest 14-day rolling median for this area",
                )

PSF_TXNS_FILTERED = PSF_TXNS.filter(pl.col("area").is_in(neighborhoods))
PSF_ROLLING_FILTERED = PSF_ROLLING.filter(pl.col("area").is_in(neighborhoods))
if PSF_TXNS_FILTERED.is_empty():
    st.warning("No transaction data for the selected filters.")
else:
    st.plotly_chart(
        area_psf_chart(PSF_TXNS_FILTERED, PSF_ROLLING_FILTERED, neighborhoods),
        use_container_width=True,
    )

st.divider()

# ── Filtered charts ───────────────────────────────────────────────────────────
st.plotly_chart(line_chart(filtered, bedroom), use_container_width=True)

st.plotly_chart(price_vs_time_scatter(filtered), use_container_width=True)

st.plotly_chart(bar_chart(filtered, latest_date), use_container_width=True)

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
