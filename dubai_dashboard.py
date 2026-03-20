"""
Dubai Real Estate Apartment Dashboard  –  Streamlit + Polars
=============================================================
Visualises 1BR / 2BR apartment prices across 20 Dubai neighbourhoods
for the period Jan 2020 – Mar 2026.

Run
---
    pip install streamlit polars plotly numpy
    streamlit run dubai_dashboard.py

Live API access (see sidebar in the app for full instructions)
--------------------------------------------------------------
Option 1 – DLD Open Data (FREE, no key needed)
    CSV download: https://dubailand.gov.ae/en/open-data/real-estate-data/

Option 2 – Dubai Pulse API (FREE tier, requires registration)
    Sign up : https://www.dubaipulse.gov.ae/
    Dataset : https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open
    Free quota available; paid tiers for higher volume (AED per 100 calls).

Option 3 – Bayut API (FREE, 750 calls/month, no credit card)
    Sign up : https://bayutapi.com/
    Docs    : https://docs.bayutapi.com/
"""

from __future__ import annotations

from datetime import date

import polars as pl
import plotly.colors
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH = "data/transactions-2026-03-20 unit.csv"

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
DATE_END   = date(2026, 3, 18)

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
        TIER_MAP[a] = tier
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
# Data loading (Polars) – real DLD transaction CSV
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading transaction data…")
def generate_dubai_data() -> pl.DataFrame:
    """Load and aggregate real DLD apartment transactions from CSV.

    Aggregates individual transactions to daily averages per
    neighbourhood and bedroom type, matching the dashboard schema.
    """
    df = (
        pl.read_csv(
            CSV_PATH,
            encoding="utf8-lossy",
            schema_overrides={"TRANS_VALUE": pl.Float64, "ACTUAL_AREA": pl.Float64},
        )
        .filter(pl.col("PROP_SB_TYPE_EN") == "Flat")
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
    return df


@st.cache_data(show_spinner="Computing Dubai-wide aggregates…")
def generate_dubai_wide_data() -> pl.DataFrame:
    """Daily Dubai-wide transaction count and median price (all flats)."""
    return (
        pl.read_csv(
            CSV_PATH,
            encoding="utf8-lossy",
            schema_overrides={"TRANS_VALUE": pl.Float64, "ACTUAL_AREA": pl.Float64},
        )
        .filter(pl.col("PROP_SB_TYPE_EN") == "Flat")
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


@st.cache_data(show_spinner="Computing weekly stats…")
def generate_weekly_data() -> pl.DataFrame:
    """Weekly Dubai-wide aggregates with % change."""
    return (
        pl.read_csv(
            CSV_PATH,
            encoding="utf8-lossy",
            schema_overrides={"TRANS_VALUE": pl.Float64, "ACTUAL_AREA": pl.Float64},
        )
        .filter(pl.col("PROP_SB_TYPE_EN") == "Flat")
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


@st.cache_data(show_spinner="Computing area-level trends…")
def generate_area_weekly_change() -> pl.DataFrame:
    """Per-area first-to-last-week median price % change (min 50 txns)."""
    raw = (
        pl.read_csv(
            CSV_PATH,
            encoding="utf8-lossy",
            schema_overrides={"TRANS_VALUE": pl.Float64, "ACTUAL_AREA": pl.Float64},
        )
        .filter(pl.col("PROP_SB_TYPE_EN") == "Flat")
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


@st.cache_data(show_spinner="Computing tier aggregates…")
def generate_tier_data() -> pl.DataFrame:
    """Daily median price per market tier."""
    return (
        pl.read_csv(
            CSV_PATH,
            encoding="utf8-lossy",
            schema_overrides={"TRANS_VALUE": pl.Float64, "ACTUAL_AREA": pl.Float64},
        )
        .filter(pl.col("PROP_SB_TYPE_EN") == "Flat")
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


# ---------------------------------------------------------------------------
# Live API connector (reference implementation – swap generate_dubai_data)
# ---------------------------------------------------------------------------

def fetch_dld_live_data(
    api_key: str,
    api_secret: str,
    start: str = "2024-01-01",
    end: str   = "2026-03-01",
) -> pl.DataFrame:
    """Fetch real transaction data from Dubai Pulse / DLD API.

    Registration
    ------------
    1. Go to https://www.dubaipulse.gov.ae/
    2. Create a free account → request access to the DLD Transactions dataset
    3. You will receive an API Key and API Secret by email
    4. Free tier: limited monthly calls; paid tiers at AED per 100 calls

    Parameters
    ----------
    api_key    : client_id from Dubai Pulse dashboard
    api_secret : client_secret from Dubai Pulse dashboard
    start / end: ISO date strings for the transaction date range
    """
    import requests  # already in requirements.txt

    # Step 1 – OAuth2 client-credentials token
    token_resp = requests.post(
        "https://api.dubaipulse.gov.ae/oauth/token",
        data={
            "grant_type":    "client_credentials",
            "client_id":     api_key,
            "client_secret": api_secret,
        },
        timeout=30,
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]

    # Step 2 – Query DLD transactions (apartments, 1-2 bedrooms)
    resp = requests.get(
        "https://api.dubaipulse.gov.ae/dld-transactions/v1/transactions",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "property_type":         "apartment",
            "transaction_date_from": start,
            "transaction_date_to":   end,
            "bedrooms":              "1,2",
            "page_size":             500,
        },
        timeout=60,
    )
    resp.raise_for_status()
    records = resp.json().get("records", [])

    # Map to the schema expected by this dashboard and return as Polars DataFrame
    return pl.DataFrame(records)


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


# ---------------------------------------------------------------------------
# Filter helper (Polars)
# ---------------------------------------------------------------------------

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
                    "Date: %{x|%b %Y}<br>"
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
                "Date: %{x|%b %Y}<br>"
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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏙️ Dubai RE Dashboard")
    st.caption("Apartments · Feb 2026 – Mar 2026 · DLD Transactions")
    st.divider()

    neighborhoods = st.multiselect(
        "Neighbourhood",
        options=NEIGHBORHOODS,
        default=NEIGHBORHOODS,
        help="Select one or more Dubai neighbourhoods",
    )

    bedroom = st.radio(
        "Bedroom Type",
        options=["Studio", "1BR", "2BR", "3BR", "All"],
        index=4,
        horizontal=True,
    )

    date_range = st.date_input(
        "Date Range",
        value=(DATE_START, DATE_END),
        min_value=DATE_START,
        max_value=DATE_END,
        format="YYYY/MM/DD",
    )
    # Safely unpack; user may still be selecting end date
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

    st.divider()

    # ── Live API access info ────────────────────────────────────────────────
    with st.expander("📡 Live API Access", expanded=False):
        st.markdown(
            """
**Option 1 – DLD Open Data (FREE)**
- No registration needed
- Download CSV files directly:
  [dubailand.gov.ae/en/open-data](https://dubailand.gov.ae/en/open-data/real-estate-data/)

---

**Option 2 – Dubai Pulse API (FREE tier)**
- Free quota each month (paid tiers: AED per 100 calls above quota)
- Register at [dubaipulse.gov.ae](https://www.dubaipulse.gov.ae/)
- You get an API Key + Secret by email
- Dataset: DLD Transactions (1.5 M records, 2004–2026)
- See `fetch_dld_live_data()` in this file for the connector

---

**Option 3 – Bayut API (FREE, 750 calls/month)**
- No credit card required
- Sign up at [bayutapi.com](https://bayutapi.com/)
- See `fetch_bayut_transactions()` in this file for the connector
            """
        )

    with st.expander("🔑 Connect to Dubai Pulse API", expanded=False):
        st.caption("Enter credentials to fetch real data (optional)")
        dp_key    = st.text_input("API Key",    type="password", placeholder="Your Dubai Pulse API Key")
        dp_secret = st.text_input("API Secret", type="password", placeholder="Your Dubai Pulse API Secret")
        if st.button("Fetch Live Data", disabled=not (dp_key and dp_secret)):
            with st.spinner("Fetching from Dubai Pulse…"):
                try:
                    live_df = fetch_dld_live_data(dp_key, dp_secret, str(start_date), str(end_date))
                    st.success(f"Loaded {len(live_df):,} live records.")
                    st.session_state["live_df"] = live_df
                except Exception as exc:
                    st.error(f"API error: {exc}")

    st.divider()
    st.caption(
        "Data: DLD transactions CSV (apartments only)."
    )

# ── Load & filter data ───────────────────────────────────────────────────────
DF = generate_dubai_data()

if not neighborhoods:
    st.warning("Select at least one neighbourhood in the sidebar.")
    st.stop()

filtered = apply_filters(DF, neighborhoods, bedroom, start_date, end_date)

if filtered.is_empty():
    st.warning("No data for the selected filters.")
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
DW = generate_dubai_wide_data()
WK = generate_weekly_data()
AREA_CHG = generate_area_weekly_change()

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
TIER_DF = generate_tier_data()
st.plotly_chart(tier_price_chart(TIER_DF), use_container_width=True)

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
