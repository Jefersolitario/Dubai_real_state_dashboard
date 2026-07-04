"""Streamlit "Fair Value Model" tab.

Trains the fair-value model (cached per data version), scores every
apartment sale against its predicted fair value, and renders model
quality metrics, prediction charts, feature importances, and a sortable
table of below-fair-value / distressed-asset candidates.

The model always trains on ALL apartment Sales rows in the loaded
snapshot; the sidebar filters only narrow what is displayed.
"""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from dashboard_constants import AREA_DISPLAY, SQM_TO_SQFT
from fair_value_model import (
    FairValueResult,
    feature_engineering,
    flag_distress,
    score_transactions,
    train_fair_value_model,
)

MIN_TRAINING_ROWS = 5_000
SCATTER_MAX_POINTS = 15_000

COLOR_NORMAL = "#636efa"
COLOR_BELOW = "#ef553b"
COLOR_DISTRESSED = "#e377c2"

DATA_SOURCES_TABLE = """
| # | Feature | Column / Source | Availability |
|---|---------|-----------------|--------------|
| 1 | Unit size | `ACTUAL_AREA` (sqm) — DLD transactions | Now |
| 2 | Rooms | `ROOMS_EN` → ordinal | Now |
| 3 | District | `AREA_EN` | Now |
| 4 | Project / building proxy | `PROJECT_EN`, `MASTER_PROJECT_EN` | Now |
| 5 | Off-plan vs ready | `IS_OFFPLAN_EN` | Now |
| 6 | Parking | `PARKING` | Now |
| 7 | Amenity proximity | `NEAREST_METRO_EN` / `NEAREST_MALL_EN` / `NEAREST_LANDMARK_EN` | Now |
| 8 | Market tier | dashboard tier constant | Now |
| 9 | Time / market trend | `INSTANCE_DATE` → trend + month | Now |
| 10 | Deal structure | `TOTAL_BUYER`, `TOTAL_SELLER` | Now |
| 11 | Distress procedure signal | `PROCEDURE_EN`, `GROUP_EN` | Now |
| 12 | Trailing area/project comparables | derived in-dataset (past-only) | Optional (loop-tested) |
| 13 | Floor, view, developer, building age | raw DDA fields / Dubai Pulse Buildings datasets | **Phase 2** |
| 14 | Rental yield | Dubai Pulse Rent Contracts (Ejari); DLD Smart Rental Index | **Phase 2 — planned** |
| 15 | Live listing asking prices | Bayut / Property Finder | **Phase 2 — planned** |
| 16 | Official sale price index | Dubai Pulse `dld_residential_sale_index` | **Phase 2** |
"""


def _layout_defaults(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13), x=0.01),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#fafafa"),
    )


def _raw_transactions() -> pl.DataFrame:
    df = st.session_state.get("api_raw_df")
    if isinstance(df, pl.DataFrame):
        return df
    return pl.DataFrame()


@st.cache_resource(show_spinner="Training fair-value model (cached per data refresh)...")
def get_model(data_version: str) -> FairValueResult:
    feats = feature_engineering(_raw_transactions())
    return train_fair_value_model(feats)


@st.cache_data(show_spinner="Scoring transactions against fair value...")
def get_scored(data_version: str, spread_threshold: float) -> pl.DataFrame:
    result = get_model(data_version)
    feats = feature_engineering(_raw_transactions())
    scored = score_transactions(result, feats)
    return flag_distress(scored, spread_threshold=spread_threshold)


def pred_vs_actual_chart(scored: pl.DataFrame) -> go.Figure:
    normal = scored.filter(~pl.col("below_fair_value"))
    if normal.height > SCATTER_MAX_POINTS:
        normal = normal.sample(SCATTER_MAX_POINTS, seed=42)
    below = scored.filter(pl.col("below_fair_value") & ~pl.col("distressed"))
    distressed = scored.filter(pl.col("distressed"))

    fig = go.Figure()
    for frame, name, color, size in (
        (normal, "Near fair value", COLOR_NORMAL, 4),
        (below, "Below fair value", COLOR_BELOW, 6),
        (distressed, "Distressed candidate", COLOR_DISTRESSED, 7),
    ):
        if frame.is_empty():
            continue
        fig.add_trace(
            go.Scattergl(
                x=frame["pred_psf"].to_list(),
                y=frame["psf"].to_list(),
                mode="markers",
                name=name,
                marker=dict(color=color, size=size, opacity=0.55),
                hovertemplate=(
                    "Fair value: AED %{x:,.0f}/sqft<br>"
                    "Actual: AED %{y:,.0f}/sqft<extra></extra>"
                ),
            )
        )
    lo = min(scored["pred_psf"].min(), scored["psf"].min())
    hi = max(scored["pred_psf"].max(), scored["psf"].max())
    fig.add_trace(
        go.Scattergl(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Actual = fair value",
            line=dict(color="#fafafa", width=1, dash="dash"),
        )
    )
    fig.update_layout(
        **_layout_defaults("Actual vs Predicted Fair Value (AED/sqft)"),
        xaxis_title="Predicted fair value (AED/sqft)",
        yaxis_title="Actual transaction price (AED/sqft)",
        height=460,
        margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    return fig


def spread_histogram(scored: pl.DataFrame, threshold: float) -> go.Figure:
    spreads = (scored["spread_pct"] * 100).to_list()
    fig = go.Figure(
        go.Histogram(
            x=spreads,
            nbinsx=80,
            marker_color=COLOR_NORMAL,
            hovertemplate="Spread %{x:.0f}%: %{y} deals<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold * 100,
        line_color=COLOR_BELOW,
        line_dash="dash",
        annotation_text=f"threshold {threshold:.0%}",
        annotation_font_color=COLOR_BELOW,
    )
    fig.add_vline(x=0, line_color="#fafafa", line_width=1)
    fig.update_layout(
        **_layout_defaults("Spread Distribution — Actual Price vs Fair Value"),
        xaxis_title="Spread: actual / fair value − 1 (%)",
        yaxis_title="Transactions",
        height=340,
        margin=dict(l=40, r=20, t=45, b=40),
        showlegend=False,
    )
    return fig


def importance_chart(importances: pl.DataFrame) -> go.Figure:
    top = importances.head(15).sort("importance_mean")
    fig = go.Figure(
        go.Bar(
            x=top["importance_mean"].to_list(),
            y=top["feature"].to_list(),
            orientation="h",
            error_x=dict(array=top["importance_std"].to_list(), color="#fafafa"),
            marker_color=COLOR_NORMAL,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout_defaults("Feature Importance (permutation, recent 10% of data)"),
        xaxis_title="Importance (increase in prediction error when shuffled)",
        height=420,
        margin=dict(l=40, r=20, t=45, b=40),
    )
    return fig


def _display_filter(
    scored: pl.DataFrame,
    neighborhoods: list[str],
    bedroom: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    df = scored.with_columns(
        pl.col("AREA_EN").replace(AREA_DISPLAY).alias("area_display"),
        pl.col("ROOMS_EN").cast(pl.Utf8).str.replace(" B/R", "BR").alias("bedroom_type"),
    )
    mask = pl.col("date").is_between(start_date, end_date)
    if neighborhoods:
        mask = mask & pl.col("area_display").is_in(neighborhoods)
    if bedroom != "All":
        mask = mask & (pl.col("bedroom_type") == bedroom)
    return df.filter(mask)


def render_fair_value_tab(
    neighborhoods: list[str],
    bedroom: str,
    start_date: date,
    end_date: date,
    data_version: str,
) -> None:
    raw = _raw_transactions()
    if raw.is_empty():
        st.info("No transaction data loaded.")
        return

    feats_rows = feature_engineering(raw).height
    if feats_rows < MIN_TRAINING_ROWS:
        st.info(
            f"Not enough apartment sales to train a reliable model "
            f"({feats_rows:,} rows; need at least {MIN_TRAINING_ROWS:,}). "
            "Widen the loaded data range."
        )
        return

    result = get_model(data_version)

    st.markdown("### Fair Value Model — Below-Market & Distressed-Asset Scanner")
    st.caption(
        "The model predicts each apartment sale's **fair value** (AED/sqft) from size, "
        "location, project, rooms, off-plan status, amenities, and market trend, then "
        "measures the **spread** between the actual closed price and that fair value. "
        f"It trains on **all {result.trained_rows:,} apartment Sales** in the loaded "
        "snapshot — the sidebar filters only narrow what is shown below."
    )

    threshold_pct = st.slider(
        "Below-fair-value threshold",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        format="-%d%%",
        help=(
            "A sale is flagged 'below fair value' when its price is at least this far "
            "under the model's predicted fair value. 'Distressed candidate' additionally "
            "requires a corroborating signal (forced-sale procedure, deep discount, "
            "illiquid project, or multiple sellers)."
        ),
    )
    scored = get_scored(data_version, -threshold_pct / 100)
    view = _display_filter(scored, neighborhoods, bedroom, start_date, end_date)

    metrics = result.metrics
    n_below = view.filter(pl.col("below_fair_value")).height
    n_distressed = view.filter(pl.col("distressed")).height
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Model error (MedAPE)",
            f"{metrics['medape_mean']:.1%}",
            help=(
                "Median absolute % gap between actual price and predicted fair value on "
                "out-of-time validation folds (10-fold TimeSeriesSplit: always trained "
                f"on the past, tested on the future). ± {metrics['medape_std']:.1%} across folds."
            ),
        )
    with c2:
        st.metric(
            "Model R² (log price/sqft)",
            f"{metrics['r2_mean']:.2f}",
            help="Share of price variation the model explains on out-of-time validation folds.",
        )
    with c3:
        st.metric(
            "Below fair value",
            f"{n_below:,}",
            help=f"Sales in the current view priced ≥{threshold_pct}% under predicted fair value.",
        )
    with c4:
        st.metric(
            "Distressed candidates",
            f"{n_distressed:,}",
            help="Below fair value AND at least one corroborating distress signal.",
        )

    if view.is_empty():
        st.warning("No scored transactions for the selected filters/date range.")
        return

    st.plotly_chart(pred_vs_actual_chart(view), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(spread_histogram(view, -threshold_pct / 100), use_container_width=True)
    with col_b:
        st.plotly_chart(importance_chart(result.importances), use_container_width=True)

    st.markdown("#### Flagged transactions")
    st.caption(
        "Sorted distressed-first, deepest discount on top. **Spread** = actual price vs "
        "predicted fair value; negative means the deal closed under fair value."
    )
    building_cols = (
        [pl.col("BUILDING_NAME_EN").alias("Building")]
        if "BUILDING_NAME_EN" in view.columns
        else []
    )
    table = (
        view.filter(pl.col("below_fair_value"))
        .sort(["distressed", "spread_pct"], descending=[True, False])
        .select(
            pl.col("date").cast(pl.Utf8).alias("Date"),
            pl.col("area_display").alias("Area"),
            pl.col("PROJECT_EN").alias("Project"),
            *building_cols,
            pl.col("bedroom_type").alias("Rooms"),
            (pl.col("ACTUAL_AREA") * SQM_TO_SQFT).round(0).alias("Size (sqft)"),
            pl.col("TRANS_VALUE").round(0).alias("Actual price (AED)"),
            pl.col("fair_value_aed").round(0).alias("Fair value (AED)"),
            (pl.col("spread_pct") * 100).round(1).alias("Spread (%)"),
            pl.col("distressed").alias("Distressed"),
            pl.col("signals").alias("Signals"),
        )
    )
    st.dataframe(table, use_container_width=True, height=420)
    st.download_button(
        "Download flagged transactions CSV",
        data=table.write_csv().encode(),
        file_name="fair_value_flagged_transactions.csv",
        mime="text/csv",
    )

    with st.expander("📚 Data & methodology", expanded=False):
        st.markdown(
            "**Model**: gradient-boosted trees (`HistGradientBoostingRegressor`) on "
            "log(AED/sqft), validated with a 10-fold date-ordered `TimeSeriesSplit` "
            "(train on the past, test on the future), then refit on all rows for scoring. "
            "**Spread** = actual price / predicted fair value − 1.\n\n"
            "**Distressed candidate** = spread at/below the threshold **and** at least one "
            "corroborating signal: forced-sale procedure keyword in `PROCEDURE_EN`, "
            "deep discount (≤ −25%), illiquid project (fewer than 8 sales in the window), "
            "or multiple sellers on the deal.\n"
        )
        st.markdown("**Data needed & sources**")
        st.markdown(DATA_SOURCES_TABLE)
        st.markdown("**Procedure types in the loaded data** (for tuning distress keywords)")
        proc_counts = (
            feature_engineering(raw)
            .group_by("PROCEDURE_EN")
            .len()
            .sort("len", descending=True)
        )
        st.dataframe(proc_counts, use_container_width=True, height=200)
