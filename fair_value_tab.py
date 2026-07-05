"""Streamlit "Fair Value Model" tab.

Trains the fair-value model (cached per data version), scores every
apartment sale against its predicted fair value, and renders model
quality metrics, prediction charts, feature importances, and a sortable
table of below-fair-value / distressed-asset candidates.

The model always trains on ALL apartment Sales rows in the loaded
snapshot; the sidebar filters only narrow what is displayed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from dashboard_constants import (
    SQM_TO_SQFT,
    area_display_expr,
    bedroom_type_expr,
    layout_defaults as _layout_defaults,
)
from fair_value_model import (
    FairValueResult,
    feature_engineering,
    flag_distress,
    load_bundle,
    load_shipping_config,
    reference_needed,
    score_transactions,
    train_fair_value_model,
    trim_psf,
)
from gcs_storage import read_model_bundle_bytes, read_reference_frames

MODEL_STALE_DAYS = 7

MIN_TRAINING_ROWS = 5_000
SCATTER_MAX_POINTS = 15_000

COLOR_NORMAL = "#636efa"
COLOR_BELOW = "#ef553b"
COLOR_DISTRESSED = "#e377c2"

# Plain-language names for model features, used in the importance chart and
# methodology notes. Unmapped features fall back to their raw column name.
FEATURE_LABELS = {
    "log_sqft": "Unit size",
    "days_since_start": "Market trend (time)",
    "month": "Month of year (seasonality)",
    "rooms_ord": "Bedrooms",
    "AREA_EN": "District",
    "IS_OFFPLAN_EN": "Off-plan vs ready",
    "tier": "Market tier (prime/mid/affordable)",
    "PROJECT_EN": "Project",
    "MASTER_PROJECT_EN": "Master development",
    "BUILDING_NAME_EN": "Building",
    "parking_count": "Parking spaces",
    "total_buyer": "Buyers on the deal",
    "total_seller": "Sellers on the deal",
    "NEAREST_METRO_EN": "Nearest metro station",
    "NEAREST_MALL_EN": "Nearest mall",
    "NEAREST_LANDMARK_EN": "Nearest landmark",
    "area_comp_psf": "District price/sqft (last 30 days)",
    "project_comp_psf": "Project price/sqft (last 60 days)",
    "project_comp_psf_30": "Project price/sqft (last 30 days)",
    "project_comp_psf_90": "Project price/sqft (last 90 days)",
    "building_comp_psf": "Building price/sqft (last 90 days)",
    "project_hist_psf": "Project long-run price level",
    "building_hist_psf": "Building long-run price level",
    "project_txn_90d": "Project sales activity (90 days)",
    "area_txn_30d": "District sales activity (30 days)",
    "area_momentum": "District price momentum",
    "rel_log_sqft": "Unit size vs project's typical unit",
    "project_comp_std": "Project price dispersion",
    "prior_unit_psf": "Same unit's previous sale price",
    "days_since_prior_sale": "Time since unit last sold",
    "prior_unit_psf_adj": "Previous sale price (market-adjusted)",
    "building_age_years": "Building age",
    "project_units": "Project size (units)",
    "project_max_floors": "Project height (floors)",
    "developer_name": "Developer",
    "area_rent_psf_180d": "Area rent level (Ejari, 180 days)",
    "implied_gross_yield": "Implied gross rental yield",
    "service_cost": "Project service charge",
    "unit_floor": "Unit's floor (exact registry match)",
    "layout_floor_mean": "Typical floor of this layout",
    "layout_units": "Identical units in the project",
    "unit_balcony_sqm": "Balcony size",
    "rel_floor_pct": "Floor position within the tower",
    "project_rooms_comp_psf": "Project price/sqft, same unit type (90 days)",
    "area_rooms_comp_psf": "District price/sqft, same unit type (30 days)",
    "project_rooms_comp_std": "Price dispersion, same unit type",
    "project_rooms_txn_90d": "Sales activity, same unit type (90 days)",
    "rent_contracts_180d": "Area rental-market activity (180 days)",
}


def feature_label(name: str) -> str:
    """Plain-language display name for a model feature (falls back to raw)."""
    return FEATURE_LABELS.get(name, name)


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
| 12 | Trailing area/project comparables | derived in-dataset (past-only) | Now (in model) |
| 13 | Same unit's previous sale (repeat-sale) | derived in-dataset (past-only) | Now (in model) |
| 14 | Floor, balcony, layout (units registry) | DLD Units via project + exact-area match | Now (in model) |
| 15 | Developer, building age, project size & height | DLD Projects + Buildings (project-level, in GCS) | Now (in model) |
| 16 | Rental yield | DLD Rent Contracts (Ejari) → weekly area × rooms rent index | Tested — no prediction gain; kept for display |
| 17 | Building reviews & comfort sentiment | Google Places / DLD building stars | **Phase 3 — planned** |
| 18 | Live listing asking prices | Bayut / Property Finder | **Phase 3 — planned** |
| 19 | Official sale price index | Dubai Pulse `dld_residential_sale_index` | Dead end — dataset frozen at 2024-05 |

Transactions carry no unit key, so units-registry data joins by project +
exact registered area: the true floor is known where a layout's area is
unique in its project (~5% of sales); elsewhere the layout's floor
distribution and balcony size are used (~76% of sales matched).
"""


def _raw_transactions() -> pl.DataFrame:
    """The session's loaded transactions frame (empty frame before load)."""
    df = st.session_state.get("api_raw_df")
    if isinstance(df, pl.DataFrame):
        return df
    return pl.DataFrame()


@st.cache_resource(show_spinner="Preparing model features...")
def get_features(data_version: str) -> pl.DataFrame:
    """Untrimmed feature frame — one pass per data refresh, used everywhere.

    Untrimmed on purpose: scoring must cover the deep-discount tail; the
    training path applies trim_psf separately.
    """
    feature_config, _ = load_shipping_config()
    reference = None
    ref_names = reference_needed(feature_config)
    if ref_names:
        reference = read_reference_frames(st.secrets, ref_names)
    return feature_engineering(_raw_transactions(), feature_config, reference=reference)


@st.cache_resource(
    ttl=6 * 3600, show_spinner="Loading pre-trained fair-value model..."
)
def get_model(data_version: str) -> tuple[FairValueResult, dict]:
    """Load the pre-trained inference bundle from GCS — the app never trains.

    Training happens offline via ``python train_fair_value.py`` (weekly
    cadence); the TTL picks up a fresh bundle without an app restart.
    """
    data, _ = read_model_bundle_bytes(st.secrets)
    return load_bundle(data)


@st.cache_resource(show_spinner="Training fair-value model (heavy, local use)...")
def train_local_model(data_version: str) -> tuple[FairValueResult, dict]:
    """Manual fallback when no bundle exists — only ever runs on button click."""
    feature_config, model_params = load_shipping_config()
    result = train_fair_value_model(
        trim_psf(get_features(data_version)), feature_config, model_params, run_cv=False
    )
    return result, {"trained_at": None, "source": "trained in this session"}


@st.cache_resource(show_spinner="Scoring transactions against fair value...", max_entries=4)
def get_scored(
    data_version: str,
    score_start: date,
    score_end: date,
    trained_at: str,
    _result: FairValueResult,
) -> pl.DataFrame:
    """Fair-value predictions for scoreable rows in the window (threshold-independent).

    Features are built over the full history (trailing comparables need the
    past), but the predict pass only runs on rows inside the scoring window —
    that is what keeps the default "last month" view fast. The threshold
    slider only re-runs the cheap flag_distress expressions, never this
    predict pass. ``_result`` is underscore-prefixed (unhashable);
    ``trained_at`` stands in for it in the cache key so a refreshed bundle
    invalidates stale predictions. ``max_entries`` bounds the cache — in
    sidebar-range mode every distinct date pair is a new key, and unbounded
    scored frames are exactly the memory profile that kills the cloud host.
    """
    feats = get_features(data_version).filter(
        pl.col("date").is_between(score_start, score_end)
    )
    if feats.is_empty():
        # model.predict rejects 0-row input; return the empty frame with the
        # scoring columns so downstream flagging/filtering degrade gracefully.
        return feats.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("pred_psf", "fair_value_aed", "spread_pct")
        )
    return score_transactions(_result, feats)


def _features_or_error(data_version: str) -> pl.DataFrame | None:
    """get_features with the missing-reference case rendered, not raised."""
    try:
        return get_features(data_version)
    except FileNotFoundError as exc:
        st.error(
            f"A reference dataset the model needs is missing: {exc}. "
            "Publish it with `python store_reference_data_gcs.py` (offline)."
        )
        return None


@st.cache_resource
def get_feature_bounds(data_version: str) -> tuple[date, date]:
    """(min, max) feature dates — cached so reruns skip the full-column scan."""
    feats = get_features(data_version)
    return feats["date"].min(), feats["date"].max()


# Scoring-window choices: one mapping drives both the widget and the window
# math, so an option can't silently fall through to the all-history path.
# Values: days back from the newest sale; "sidebar" and "all" are sentinels.
SCORING_WINDOWS: dict[str, int | str] = {
    "Last month": 30,
    "Last 3 months": 90,
    "Sidebar date range": "sidebar",
    "All history": "all",
}


def pred_vs_actual_chart(scored: pl.DataFrame) -> go.Figure:
    """Scatter of actual vs predicted AED/sqft, coloured by flag status."""
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
                x=frame["pred_psf"].to_numpy(),
                y=frame["psf"].to_numpy(),
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
    """Distribution of actual-vs-fair-value spreads with the threshold line."""
    spreads = (scored["spread_pct"] * 100).to_numpy()
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
    """Top-15 permutation importances with plain-language feature names."""
    top = importances.head(15).sort("importance_mean")
    labels = [feature_label(f) for f in top["feature"].to_list()]
    fig = go.Figure(
        go.Bar(
            x=top["importance_mean"].to_numpy(),
            y=labels,
            orientation="h",
            error_x=dict(array=top["importance_std"].to_numpy(), color="#fafafa"),
            marker_color=COLOR_NORMAL,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout_defaults("What Drives the Fair-Value Estimate"),
        xaxis_title="Importance (how much the prediction worsens without this input)",
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
    """Apply the sidebar neighbourhood/bedroom/date filters to scored rows."""
    df = scored.with_columns(
        area_display_expr().alias("area_display"),
        bedroom_type_expr().alias("bedroom_type"),
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
    """Render the Fair Value page: model load, scoring, flags, charts."""
    raw = _raw_transactions()
    if raw.is_empty():
        st.info("No transaction data loaded.")
        return

    try:
        result, meta = get_model(data_version)
    except FileNotFoundError:
        st.info(
            "No pre-trained model bundle found in GCS. Publish one with "
            "`python train_fair_value.py` (run offline — training is too heavy "
            "for this deployment)."
        )
        feats = _features_or_error(data_version)
        if feats is None or feats.height < MIN_TRAINING_ROWS:
            if feats is not None:
                st.info(
                    f"Local training also unavailable: only {feats.height:,} apartment "
                    f"sales loaded (need at least {MIN_TRAINING_ROWS:,})."
                )
            return
        if not st.button("Train in this session (heavy — local use only)"):
            return
        result, meta = train_local_model(data_version)
    except Exception as exc:  # unreadable bundle, GCS outage, version break
        st.error(f"Could not load the fair-value model bundle: {exc}")
        return

    feats = _features_or_error(data_version)
    if feats is None:
        return
    if feats.is_empty():
        st.warning("No scoreable apartment sales in the loaded data.")
        return
    data_min, data_max = get_feature_bounds(data_version)

    st.markdown("### Fair Value Model — Below-Market & Distressed-Asset Scanner")
    st.caption(
        "The model predicts each apartment sale's **fair value** (AED/sqft) from size, "
        "location, project, rooms, off-plan status, amenities, and market trend, then "
        "measures the **spread** between the actual closed price and that fair value. "
        f"It was trained offline on **{result.trained_rows:,} apartment Sales**. "
        "The neighbourhood, bedroom, and date filters narrow what is shown below; "
        "the Transaction Type filter does not apply here (this tab always analyses "
        "Sales — mortgage rows record loan amounts, not market prices)."
    )

    trained_at = meta.get("trained_at")
    if trained_at:
        trained_dt = datetime.fromisoformat(trained_at)
        age_days = (datetime.now(timezone.utc) - trained_dt).days
        st.caption(
            f"Model trained **{trained_dt:%Y-%m-%d}** "
            f"({meta.get('data_min_date')} – {meta.get('data_max_date')} data)."
        )
        if age_days > MODEL_STALE_DAYS:
            st.warning(
                f"The model is {age_days} days old. Refresh it by running "
                "`python store_dld_transactions_gcs.py` then "
                "`python train_fair_value.py` (offline)."
            )

    # Seed the widget state once, then pass key= only: with the lazy page
    # dispatch these widgets unmount while Market Overview is shown, and the
    # keep-alive re-assignment in dubai_dashboard.py preserves the values.
    st.session_state.setdefault("fv_window", next(iter(SCORING_WINDOWS)))
    st.session_state.setdefault("fv_threshold", 15)
    col_window, col_threshold = st.columns([1, 2])
    with col_window:
        window = st.selectbox(
            "Scoring window",
            list(SCORING_WINDOWS),
            key="fv_window",
            help=(
                "Only sales in this window are scored against fair value — the "
                "default keeps the tab fast. Features still use the full history, "
                "so predictions are identical to scoring everything."
            ),
        )
    with col_threshold:
        threshold_pct = st.slider(
            "Below-fair-value threshold",
            min_value=5,
            max_value=30,
            step=1,
            format="-%d%%",
            key="fv_threshold",
            help=(
                "A sale is flagged 'below fair value' when its price is at least this far "
                "under the model's predicted fair value. 'Distressed candidate' additionally "
                "requires a signal independent of the model residual: a forced-sale "
                "procedure, an illiquid project, or multiple sellers on the deal."
            ),
        )

    window_spec = SCORING_WINDOWS[window]
    if window_spec == "sidebar":
        score_start, score_end = start_date, end_date
    elif window_spec == "all":
        score_start, score_end = data_min, data_max
    else:
        score_start, score_end = data_max - timedelta(days=window_spec), data_max

    scored = flag_distress(
        get_scored(data_version, score_start, score_end, str(trained_at), result),
        spread_threshold=-threshold_pct / 100,
    )
    if scored.is_empty():
        st.warning(f"No apartment sales between {score_start} and {score_end}.")
        return
    st.caption(
        f"Scoring **{scored.height:,} sales** from **{score_start}** to "
        f"**{score_end}**; the sidebar filters narrow this further."
    )
    view = _display_filter(scored, neighborhoods, bedroom, start_date, end_date)

    metrics = result.metrics or {}
    flagged = view.filter(pl.col("below_fair_value"))
    n_below = flagged.height
    n_distressed = view.filter(pl.col("distressed")).height
    medape = metrics.get("medape_mean")
    medape_std = metrics.get("medape_std")
    r2 = metrics.get("r2_mean")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Model error (MedAPE)",
            f"{medape:.1%}" if medape is not None else "—",
            help=(
                "Median absolute % gap between actual price and predicted fair value on "
                "out-of-time validation folds (10-fold TimeSeriesSplit: always trained "
                "on the past, tested on the future)."
                + (f" ± {medape_std:.1%} across folds." if medape_std is not None else "")
            ),
        )
    with c2:
        st.metric(
            "Model R² (log price/sqft)",
            f"{r2:.2f}" if r2 is not None else "—",
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
            help=(
                "Below fair value AND at least one residual-independent signal "
                "(forced-sale procedure, illiquid project, multiple sellers)."
            ),
        )

    if view.is_empty():
        if score_end < start_date or score_start > end_date:
            st.warning(
                f"The scoring window ({score_start} to {score_end}) does not "
                f"overlap the sidebar date range ({start_date} to {end_date}). "
                "Pick 'Sidebar date range' as the scoring window, or widen "
                "the sidebar dates."
            )
        else:
            st.warning("No scored transactions for the selected filters/date range.")
        return

    st.markdown("#### Flagged transactions")
    st.caption(
        "Sorted distressed-first, strongest signal on top. **Spread** = actual price vs "
        "predicted fair value; negative means the deal closed under fair value. "
        "**Signal (×)** = the discount divided by the model's typical error for that "
        "kind of sale — a −15% spread is ~3.6× the typical error where a project has "
        "recent comparable sales, but under 1.5× for a cold-start sale (first sales in "
        "a project in a while), where the same discount is weak evidence. Prefer "
        "high-× deals."
    )
    building_cols = (
        [pl.col("BUILDING_NAME_EN").alias("Building")]
        if "BUILDING_NAME_EN" in view.columns
        else []
    )
    table = (
        flagged
        .sort(["distressed", "signal_strength"], descending=[True, True])
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
            pl.col("signal_strength").round(1).alias("Signal (×)"),
            pl.col("cold_start").alias("Cold start"),
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

    st.plotly_chart(pred_vs_actual_chart(view), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(spread_histogram(view, -threshold_pct / 100), use_container_width=True)
    with col_b:
        st.plotly_chart(importance_chart(result.importances), use_container_width=True)

    with st.expander("📚 Data & methodology", expanded=False):
        st.markdown(
            "**How it works.** The model estimates what each apartment *should* have "
            "sold for (its **fair value**, in AED/sqft) from the property's "
            "characteristics and the market around it at the time of sale, then "
            "compares that estimate with the actual closed price. "
            "**Spread** = actual price / predicted fair value − 1; a spread of −20% "
            "means the deal closed 20% under the model's fair value.\n\n"
            "**What the model looks at** (see the importance chart for what matters "
            "most): the unit's own previous sale price and how long ago it sold; the "
            "project's and building's recent and long-run price levels; the unit's "
            "size and how it compares with the project's typical unit; bedrooms, "
            "district, project, off-plan status; and the overall market trend at the "
            "date of sale. All history-based features use only sales that closed "
            "**before** the transaction being valued — the model never peeks at the "
            "future or at the deal itself.\n\n"
            "**Model & validation.** Gradient-boosted trees "
            "(`HistGradientBoostingRegressor`) on log(AED/sqft), validated with a "
            "10-fold date-ordered `TimeSeriesSplit` (always trained on the past, "
            "tested on the future). The model is trained **offline** on the full "
            "24-month history and published as a bundle; this page only loads the "
            "bundle and predicts, and the scoring-window selector controls which "
            "sales are scored (features always use the full history, so results are "
            "identical either way).\n\n"
            "**Accuracy.** Cross-validated median error ≈ 4.2%, confirmed at 4.3% on "
            "an untouched two-month holdout that was never used for any modelling "
            "decision. Caveat: sales in projects with no recent comparable sales "
            "('cold starts', a few % of rows) carry roughly double the error — treat "
            "flags on a project's first sales in a while with extra care.\n\n"
            "**Signal strength (×)** standardizes the discount by the model's typical "
            "error for that segment (established projects vs cold starts), so the list "
            "ranks by *how unusual* a price is, not just how low. The same −15% can be "
            "strong evidence in a liquid project and noise in a cold start.\n\n"
            "**Distressed candidate** = spread at/below the threshold **and** at least "
            "one corroborating signal that is independent of the model residual: "
            "forced-sale procedure keyword in `PROCEDURE_EN`, illiquid project (fewer "
            "than 8 sales in the window), or multiple sellers on the deal. A 'deep "
            "discount' (≤ −25%) is annotated in the Signals column for context but "
            "never counts as corroboration on its own — that would let a single model "
            "mistake label a normal sale as distressed.\n\n"
            "**Data hygiene.** Deep-discount outliers are excluded from **training** "
            "(0.5%/99.5% PSF trim) but always **scored**, so genuine fire-sales stay "
            "visible. Rows whose recorded area contradicts the official AED/sqm price "
            "by more than 10% are dropped as data errors, and non-market procedures "
            "(developer transfers, lease-to-own, payment plans) are excluded.\n"
        )
        st.markdown("**Data needed & sources**")
        st.markdown(DATA_SOURCES_TABLE)
        st.markdown("**Procedure types in the loaded data** (for tuning distress keywords)")
        proc_counts = (
            scored.group_by("PROCEDURE_EN").len().sort("len", descending=True)
        )
        st.dataframe(proc_counts, use_container_width=True, height=200)
