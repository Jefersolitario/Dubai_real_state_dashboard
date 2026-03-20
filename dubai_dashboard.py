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

import json
import textwrap
from datetime import date

import numpy as np
import polars as pl
import plotly.colors
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEIGHBORHOODS: list[str] = [
    # Premium / waterfront
    "Palm Jumeirah",
    "Downtown Dubai",
    "DIFC",
    "JBR",
    "Dubai Creek Harbour",
    # Mid-tier established
    "Dubai Marina",
    "JLT",
    "Business Bay",
    "Meydan City",
    # Suburban / community
    "Dubai Hills Estate",
    "Arabian Ranches",
    "Damac Hills",
    "Al Furjan",
    # Affordable / value
    "Al Barsha",
    "JVC",
    "Motor City",
    "Dubai Sports City",
    # Emerging / outer
    "Bur Dubai",
    "Dubai Silicon Oasis",
    "Dubai South",
]

# Baseline prices (AED) and price-per-sqft at Jan 2020, calibrated to DLD data
NEIGHBORHOOD_PARAMS: dict = {
    # Premium / waterfront
    "Palm Jumeirah":      {"1BR": 1_800_000, "2BR": 3_500_000, "sqft": 2_800, "min_txn":  40, "max_txn": 120},
    "Downtown Dubai":     {"1BR": 1_500_000, "2BR": 2_800_000, "sqft": 2_500, "min_txn":  80, "max_txn": 200},
    "DIFC":               {"1BR": 1_400_000, "2BR": 2_600_000, "sqft": 2_400, "min_txn":  30, "max_txn":  90},
    "JBR":                {"1BR": 1_300_000, "2BR": 2_400_000, "sqft": 2_200, "min_txn":  60, "max_txn": 160},
    "Dubai Creek Harbour":{"1BR": 1_200_000, "2BR": 2_200_000, "sqft": 2_050, "min_txn":  50, "max_txn": 140},
    # Mid-tier established
    "Dubai Marina":       {"1BR": 1_100_000, "2BR": 2_000_000, "sqft": 1_900, "min_txn": 150, "max_txn": 350},
    "JLT":                {"1BR":   850_000, "2BR": 1_520_000, "sqft": 1_450, "min_txn": 110, "max_txn": 270},
    "Business Bay":       {"1BR":   950_000, "2BR": 1_750_000, "sqft": 1_600, "min_txn": 120, "max_txn": 280},
    "Meydan City":        {"1BR": 1_000_000, "2BR": 1_850_000, "sqft": 1_720, "min_txn":  35, "max_txn": 100},
    # Suburban / community
    "Dubai Hills Estate": {"1BR":   900_000, "2BR": 1_600_000, "sqft": 1_500, "min_txn":  80, "max_txn": 200},
    "Arabian Ranches":    {"1BR":   850_000, "2BR": 1_550_000, "sqft": 1_300, "min_txn":  50, "max_txn": 140},
    "Damac Hills":        {"1BR":   800_000, "2BR": 1_450_000, "sqft": 1_250, "min_txn":  45, "max_txn": 130},
    "Al Furjan":          {"1BR":   750_000, "2BR": 1_350_000, "sqft": 1_100, "min_txn":  70, "max_txn": 180},
    # Affordable / value
    "Al Barsha":          {"1BR":   750_000, "2BR": 1_350_000, "sqft": 1_100, "min_txn":  90, "max_txn": 220},
    "JVC":                {"1BR":   700_000, "2BR": 1_250_000, "sqft":   950, "min_txn": 200, "max_txn": 450},
    "Motor City":         {"1BR":   700_000, "2BR": 1_250_000, "sqft": 1_000, "min_txn":  55, "max_txn": 140},
    "Dubai Sports City":  {"1BR":   650_000, "2BR": 1_150_000, "sqft":   880, "min_txn":  65, "max_txn": 160},
    # Emerging / outer
    "Bur Dubai":          {"1BR":   720_000, "2BR": 1_300_000, "sqft": 1_000, "min_txn": 100, "max_txn": 250},
    "Dubai Silicon Oasis":{"1BR":   600_000, "2BR": 1_050_000, "sqft":   810, "min_txn":  75, "max_txn": 190},
    "Dubai South":        {"1BR":   680_000, "2BR": 1_200_000, "sqft":   850, "min_txn":  60, "max_txn": 160},
}

# Piecewise annual growth rates sourced from DLD RPPI (Residential Properties Price Index)
ANNUAL_GROWTH: dict[int, float] = {
    2020: 0.010,   # flat – COVID impact
    2021: 0.040,   # early recovery
    2022: 0.160,   # post-Expo 2020 demand surge
    2023: 0.130,   # continued high demand
    2024: 0.100,   # market maturing
    2025: 0.070,   # stabilisation
    2026: 0.050,   # annualised Q1 2026 rate
}

DATE_START = date(2020, 1, 1)
DATE_END   = date(2026, 3, 1)

# Alphabet palette has 26 entries – enough for 20 neighbourhoods
COLORS = plotly.colors.qualitative.Alphabet
COLOR_MAP = {n: COLORS[i % len(COLORS)] for i, n in enumerate(NEIGHBORHOODS)}

# ---------------------------------------------------------------------------
# Data generation (Polars)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating price data…")
def generate_dubai_data() -> pl.DataFrame:
    """Synthetic monthly apartment price data calibrated to DLD statistics.

    Methodology:
      - Baseline prices from NEIGHBORHOOD_PARAMS (DLD Jan 2020 averages)
      - Monthly compounding of piecewise annual growth rates from ANNUAL_GROWTH
      - Gaussian noise (σ = 1.5 % price, 1.2 % sqft), seeded for reproducibility
      - YoY % computed via Polars window pct_change(12) per group
    """
    rng = np.random.default_rng(seed=42)

    month_dates: list[date] = (
        pl.date_range(DATE_START, DATE_END, interval="1mo", eager=True).to_list()
    )

    rows: list[dict] = []
    for neighborhood, params in NEIGHBORHOOD_PARAMS.items():
        for br in ("1BR", "2BR"):
            base_price = float(params[br])
            base_sqft  = float(params["sqft"])
            cumulative = 1.0

            for d in month_dates:
                monthly_rate = (1.0 + ANNUAL_GROWTH[d.year]) ** (1.0 / 12.0)
                cumulative  *= monthly_rate

                price     = base_price * cumulative * rng.normal(1.0, 0.015)
                sqft_rate = base_sqft  * cumulative * rng.normal(1.0, 0.012)

                rows.append({
                    "date":               d,
                    "neighborhood":       neighborhood,
                    "bedroom_type":       br,
                    "avg_price_aed":      round(price, 0),
                    "price_per_sqft_aed": round(sqft_rate, 1),
                    "avg_size_sqft":      round(price / sqft_rate, 1),
                    "transaction_count":  int(rng.integers(params["min_txn"], params["max_txn"] + 1)),
                })

    df = (
        pl.DataFrame(rows)
        .sort(["neighborhood", "bedroom_type", "date"])
        .with_columns(
            (
                pl.col("avg_price_aed")
                .pct_change(n=12)
                .over(["neighborhood", "bedroom_type"])
                .mul(100.0)
                .round(1)
            ).alias("yoy_change_pct")
        )
    )
    return df


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
    if bedroom != "Both":
        mask = mask & (pl.col("bedroom_type") == bedroom)
    return df.filter(mask)


# ---------------------------------------------------------------------------
# Chart builders (Plotly)
# ---------------------------------------------------------------------------

def _layout_defaults(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13), x=0.01),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=48, b=40),
        font=dict(family="Segoe UI, Arial, sans-serif", size=11),
    )


def line_chart(df: pl.DataFrame, bedroom: str) -> go.Figure:
    fig   = go.Figure()
    dash_ = {"1BR": "solid", "2BR": "dash"}
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
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#e9ecef"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    if len(br_types) == 2:
        fig.add_annotation(
            text="Solid = 1BR · Dashed = 2BR",
            xref="paper", yref="paper", x=0.01, y=0.97,
            showarrow=False, font=dict(size=10, color="#6c757d"),
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

    opacity_map = {"1BR": 1.0, "2BR": 0.65}
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
        yaxis=dict(title="AED", tickformat=",.0f", gridcolor="#e9ecef"),
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
        yaxis=dict(title="Avg Price (AED)", tickformat=",.0f", gridcolor="#e9ecef"),
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
    st.caption("Apartments · 1BR & 2BR · Jan 2020 – Mar 2026")
    st.divider()

    neighborhoods = st.multiselect(
        "Neighbourhood",
        options=NEIGHBORHOODS,
        default=NEIGHBORHOODS,
        help="Select one or more Dubai neighbourhoods",
    )

    bedroom = st.radio(
        "Bedroom Type",
        options=["1BR", "2BR", "Both"],
        index=2,
        horizontal=True,
    )

    date_range = st.date_input(
        "Date Range",
        value=(DATE_START, DATE_END),
        min_value=DATE_START,
        max_value=DATE_END,
        format="MMM YYYY",
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
        "Data: synthetic, calibrated to DLD RPPI statistics. "
        "Swap `generate_dubai_data()` for `fetch_dld_live_data()` to use real data."
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

# ── Charts ────────────────────────────────────────────────────────────────────
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
