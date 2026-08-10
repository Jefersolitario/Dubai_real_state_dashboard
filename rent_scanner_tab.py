"""Rent Opportunity Scanner page (Ejari data).

Tenant-view twin of the sales Buyer Opportunity Scanner: individual rent
contracts against each zone's 14-day rolling median rent PSF. Ejari volume
is far beyond the app's memory budget, so the page adapts to the selected
date range: windows of RENT_DOT_MAX_DAYS or less that the contract-level
artifact covers render raw dots plus a ranked deals table; anything longer
renders weekly box plots from precomputed percentile stats. Both artifacts
are published offline by
`python -m ingestion.store_reference_data_gcs --only rents`.

Caching contract: load_rent_artifacts (resource cache, 6h TTL) fetches both
GCS frames plus `rent_version` (the blob's updated timestamp — the sales
snapshot's data_version never keys rent caches); generate_rent_psf_timeseries
(data cache, max_entries=4) slices the contract frame per filter selection.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl
from polars.exceptions import PanicException
import plotly.graph_objects as go
import streamlit as st

from dashboard_constants import (
    NEIGHBORHOODS,
    area_display_expr,
    layout_defaults,
    y_cap_range,
)
from ingestion.gcs_storage import REFERENCE_OBJECTS, read_parquet_object, setting

LOGGER = logging.getLogger(__name__)

# Dot view allowed up to any 3 calendar months (92 days). The worst-case zone
# at "All" bedrooms is ~15-25k dots — comfortable for Scattergl and a few MB
# of hover payload; longer windows switch to the weekly box summary.
RENT_DOT_MAX_DAYS = 92

# Same semantics and hexes as dubai_dashboard's ZONE_COLOR_* (importing from
# there would be circular); validated for colorblind separation on the dark
# chart surface.
RENT_COLOR_CONTEXT = "#636efa"   # at/above the zone median (context)
RENT_COLOR_BELOW = "#ef553b"     # below the zone median / flagged deal

# Mirror of dubai_dashboard's NON_MARKET_SPREAD: a rent more than 60% under
# the zone median is a token/related-party arrangement, not a market rent.
RENT_NON_MARKET_SPREAD = 0.60


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=6 * 3600, show_spinner="Loading Ejari rent data...")
def load_rent_artifacts() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(weekly_stats, recent_contracts, meta) from GCS.

    meta carries coverage bounds, capability flags, and rent_version — the
    blob's updated timestamp, which changes exactly when the offline rents
    pull republishes the artifacts.
    """
    bucket = setting(st.secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    frames: dict[str, pl.DataFrame] = {}
    versions: list[str] = []
    for name in ("rent_weekly_stats", "rent_recent_contracts"):
        df, blob = read_parquet_object(st.secrets, bucket, REFERENCE_OBJECTS[name])
        frames[name] = df.rechunk()
        versions.append(blob.updated.isoformat() if blob.updated else "")
    weekly = frames["rent_weekly_stats"]
    recent = frames["rent_recent_contracts"]
    meta = {
        "rent_version": max(versions),
        "recent_min": recent["start"].min(),
        "recent_max": recent["start"].max(),
        "weekly_min": weekly["week"].min(),
        "weekly_max": weekly["week"].max(),
        "has_reg_type": "reg_type" in recent.columns,
        "has_new_segment": "segment" in weekly.columns
        and "new" in weekly["segment"].unique().to_list(),
    }
    return weekly, recent, meta


def _rent_artifacts_or_error() -> tuple[pl.DataFrame, pl.DataFrame, dict] | None:
    """Load the rent artifacts, or render an actionable error and return None."""
    try:
        return load_rent_artifacts()
    except FileNotFoundError as exc:
        st.error(
            f"Rent data is not published yet ({exc}). Run "
            "`python -m ingestion.store_reference_data_gcs --only rents` "
            "offline (~2-4h) to publish it, then reload this page."
        )
        return None


def use_dot_view(
    start: date,
    end: date,
    recent_min: date | None,
    max_days: int = RENT_DOT_MAX_DAYS,
) -> bool:
    """Raw contract dots only for short windows the contract artifact covers.

    Gate on the artifact's real floor, never "today - 6 months": the pull is
    manual, so a stale artifact silently falls back to the box view instead
    of rendering a hole.
    """
    if recent_min is None:
        return False
    return (end - start).days + 1 <= max_days and start >= recent_min


@st.cache_data(show_spinner="Slicing rent contracts...", max_entries=4)
def generate_rent_psf_timeseries(
    bedroom: str,
    rent_version: str,
    start_date: date,
    end_date: date,
    include_renewals: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-contract rent PSF and 14-day rolling zone median for the window.

    Mirrors the sales generate_area_psf_timeseries shapes: txns_df has one
    row per contract [date, area, rent_psf, rooms_band, size_sqft,
    annual_amount(, reg_type)]; rolling_df has one row per (date, area) with
    the 14-day rolling mean of the daily median rent PSF. rent_version keys
    the cache to the GCS artifact, not the sales snapshot.
    """
    _, recent, meta = load_rent_artifacts()
    txns = recent.filter(
        (pl.col("start") >= start_date) & (pl.col("start") <= end_date)
    )
    if bedroom != "All":
        txns = txns.filter(pl.col("rooms_band") == bedroom)
    if meta["has_reg_type"] and not include_renewals:
        # Anchored: "Renew"/"Renewal" also contain "new"; only a leading
        # "New" marks a genuinely new contract.
        txns = txns.filter(
            pl.col("reg_type").str.contains(r"(?i)^new").fill_null(False)
        )

    keep = ["start", "AREA_EN", "rooms_band", "size_sqft", "annual_amount", "rent_psf"]
    if meta["has_reg_type"]:
        keep.append("reg_type")
    txns_df = (
        txns.select(keep)
        .with_columns(
            area_display_expr().alias("area"),
            pl.col("rooms_band").fill_null("—"),
        )
        .drop("AREA_EN")
        .rename({"start": "date"})
        .sort(["area", "date"])
    )
    # Rent PSF runs ~60-180 AED/sqft/yr, so daily medians keep one decimal.
    rolling_df = (
        txns_df.group_by(["date", "area"])
        .agg(pl.col("rent_psf").median().round(1).alias("daily_median_psf"))
        .sort(["area", "date"])
        .with_columns(
            pl.col("daily_median_psf")
            .rolling_mean(14)
            .over("area")
            .alias("rolling_median_psf")
        )
    )
    return txns_df.rechunk(), rolling_df.rechunk()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def rent_zone_chart(
    zone_scored: pl.DataFrame,
    zone_rolling: pl.DataFrame,
    zone: str,
    threshold_pct: int,
) -> go.Figure:
    """One zone's rent contracts against its 14-day rolling median.

    Same diverging encoding as the sales scanner: strong red = cleared the
    deal threshold, soft red = below the median, muted blue = at/above.
    Classification and hover strings are assembled in plain Python (polars
    string kernels on cache-round-tripped frames hit binview panics).
    """
    thr = threshold_pct / 100.0
    has_reg = "reg_type" in zone_scored.columns
    reg_values = (
        zone_scored["reg_type"].to_list() if has_reg else [None] * zone_scored.height
    )
    rows = zip(
        zone_scored["date"].to_list(),
        zone_scored["rent_psf"].to_list(),
        zone_scored["pct_vs_median"].to_list(),
        zone_scored["rooms_band"].to_list(),
        zone_scored["size_sqft"].to_list(),
        zone_scored["annual_amount"].to_list(),
        reg_values,
    )
    points: dict[str, tuple[list, list, list]] = {
        key: ([], [], []) for key in ("non_market", "context", "below", "deal")
    }
    for day, psf, pct, rooms, size, annual, reg in rows:
        if pct is None:
            key, pct_label = "context", "—"
        elif pct <= -RENT_NON_MARKET_SPREAD:
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
        custom.append((pct_label, rooms or "—", size, annual, reg or ""))

    hover = (
        "%{x|%d %b %Y}<br>"
        "AED %{y:,.1f}/sqft/yr · %{customdata[0]} vs median<br>"
        "%{customdata[1]} · %{customdata[2]:,.0f} sqft · "
        "AED %{customdata[3]:,.0f}/yr %{customdata[4]}"
        "<extra></extra>"
    )
    layers = (
        ("non_market", f"Non-market (> {RENT_NON_MARKET_SPREAD:.0%} below)",
         dict(color="#8b949e", size=4, opacity=0.25)),
        ("context", "At/above median",
         dict(color=RENT_COLOR_CONTEXT, size=5, opacity=0.30)),
        ("below", "Below median",
         dict(color=RENT_COLOR_BELOW, size=5, opacity=0.45)),
        ("deal", f"Deal — ≥ {threshold_pct}% below",
         dict(color=RENT_COLOR_BELOW, size=7, opacity=0.95,
              line=dict(width=1, color="#0e1117"))),
    )
    fig = go.Figure()
    for key, name, marker in layers:
        xs, ys, custom = points[key]
        if not xs:
            continue
        # Scattergl: WebGL markers stay responsive at thousands of dots.
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
            hovertemplate="%{x|%d %b %Y}<br>14-day median: AED %{y:,.1f}/sqft/yr<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=rx,
            y=(rmed * (1 - thr)).to_list(),
            mode="lines",
            name=f"−{threshold_pct}% threshold",
            line=dict(color=RENT_COLOR_BELOW, width=1, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Threshold: AED %{y:,.1f}/sqft/yr<extra></extra>",
        ))

    fig.update_layout(
        **layout_defaults(f"{zone} — Rent / sqft / yr vs 14-day Rolling Median"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            title="AED / sqft / yr", tickformat=",.0f", gridcolor="#2a2e35",
            range=y_cap_range(zone_scored["rent_psf"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="closest",
        margin=dict(l=60, r=20, t=54, b=40),
    )
    return fig


def rent_box_chart(weekly_zone: pl.DataFrame, zone: str) -> go.Figure:
    """Weekly precomputed box plots: q1-q3 box, p10-p90 whiskers, median line.

    Built entirely from the aggregated stats artifact — no contract-level
    data is shipped to the browser for long ranges.
    """
    weeks = weekly_zone["week"].to_list()
    med = weekly_zone["median"].to_list()
    q1 = weekly_zone["q1"].to_list()
    q3 = weekly_zone["q3"].to_list()
    p10 = weekly_zone["p10"].to_list()
    p90 = weekly_zone["p90"].to_list()
    counts = weekly_zone["n"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=weeks,
        median=med,
        q1=q1,
        q3=q3,
        lowerfence=p10,
        upperfence=p90,
        name="Weekly distribution",
        marker_color=RENT_COLOR_CONTEXT,
        fillcolor="rgba(99, 110, 250, 0.25)",
        line=dict(width=1),
        width=7 * 86_400_000 * 0.7,  # ms on a date axis: ~70% of a week
        hoverinfo="skip",
    ))
    custom = [
        (f"{a:,.0f}–{b:,.0f}", f"{lo:,.0f}–{hi:,.0f}", n)
        for a, b, lo, hi, n in zip(q1, q3, p10, p90, counts)
    ]
    fig.add_trace(go.Scatter(
        x=weeks,
        y=med,
        mode="lines",
        name="Weekly median",
        line=dict(color="#fafafa", width=2),
        customdata=custom,
        hovertemplate=(
            "%{x|%d %b %Y}<br>"
            "Median: AED %{y:,.1f}/sqft/yr<br>"
            "Middle half (q1–q3): %{customdata[0]}<br>"
            "p10–p90: %{customdata[1]} · %{customdata[2]} contracts"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **layout_defaults(f"{zone} — Weekly Rent / sqft / yr Distribution"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(title="AED / sqft / yr", tickformat=",.0f", gridcolor="#2a2e35"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=20, t=54, b=40),
    )
    return fig


def rent_deals_by_zone_chart(deal_counts: pl.DataFrame, threshold_pct: int) -> go.Figure:
    """Horizontal bar of below-threshold rent-deal counts per selected zone."""
    d = deal_counts.sort("deals")
    fig = go.Figure(go.Bar(
        x=d["deals"].to_list(),
        y=d["area"].to_list(),
        orientation="h",
        marker=dict(color=RENT_COLOR_BELOW),
        text=d["deals"].to_list(),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} rent deals<extra></extra>",
    ))
    fig.update_layout(
        **layout_defaults(
            f"Rent deals by zone — ≥ {threshold_pct}% below the zone's own median"
        ),
        xaxis=dict(title="Contracts", tickformat=",d",
                   gridcolor="#2a2e35", zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        height=max(240, 80 + 22 * d.height),
        showlegend=False,
        margin=dict(l=10, r=50, t=48, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------

def _implied_gross_yield(
    zone: str,
    latest_rent_psf: float | None,
    sale_psf_fn,
    bedroom: str,
    data_version: str,
    start_date: date,
    end_date: date,
) -> float | None:
    """Zone median annual rent PSF ÷ zone median sale PSF; None when unknown.

    Best-effort bonus metric — any failure (empty sales slice, cache error)
    must never break the rent page.
    """
    if latest_rent_psf is None or sale_psf_fn is None:
        return None
    try:
        _, sale_rolling = sale_psf_fn("Sale", bedroom, data_version, start_date, end_date)
        zone_sales = sale_rolling.filter(
            (pl.col("area") == zone) & pl.col("rolling_median_psf").is_not_null()
        )
        if not zone_sales.height:
            return None
        sale_psf = zone_sales["rolling_median_psf"].tail(1).item()
        if not sale_psf:
            return None
        return latest_rent_psf / sale_psf
    except Exception:
        LOGGER.exception("implied gross yield unavailable")
        return None


def _render_rent_dot_view(
    zone: str,
    zone_options: list[str],
    bedroom: str,
    start_date: date,
    end_date: date,
    meta: dict,
    threshold_pct: int,
    include_renewals: bool,
    sale_psf_fn,
    data_version: str,
) -> None:
    """Short-window view: contract dots, KPIs, ranked deals table."""
    thr = threshold_pct / 100.0
    txns, rolling = generate_rent_psf_timeseries(
        bedroom, meta["rent_version"], start_date, end_date, include_renewals
    )
    scored = (
        txns.filter(pl.col("area").is_in(zone_options))
        .join(
            rolling.select(["date", "area", "rolling_median_psf"]),
            on=["date", "area"],
            how="left",
        )
        .with_columns(
            (pl.col("rent_psf") / pl.col("rolling_median_psf") - 1)
            .alias("pct_vs_median")
        )
    )
    zone_scored = scored.filter(pl.col("area") == zone)
    if zone_scored.is_empty():
        st.warning(f"No rent contracts for {zone} in the selected window.")
        return
    zone_rolling = rolling.filter(
        (pl.col("area") == zone) & pl.col("rolling_median_psf").is_not_null()
    )
    deals = zone_scored.filter(
        (pl.col("pct_vs_median") <= -thr)
        & (pl.col("pct_vs_median") > -RENT_NON_MARKET_SPREAD)
    ).sort("pct_vs_median")

    st.caption(
        "**Dots** = individual Ejari contracts in the zone. **Solid line** = "
        "its 14-day rolling median rent; **dashed line** = the deal threshold. "
        "**Red dots** signed under the median — the strong red ones cleared "
        "the threshold and are listed in the table underneath. Grey dots more "
        f"than {RENT_NON_MARKET_SPREAD:.0%} below are token/related-party "
        "rents and are excluded from the deal list."
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
            "14d median rent",
            f"AED {latest_median:,.0f}/sqft/yr" if latest_median is not None else "N/A",
            help="Latest 14-day rolling median annual rent per sqft in this zone",
        )
    with k2:
        st.metric(
            "Contracts",
            f"{zone_scored.height:,}",
            help="Ejari contracts in this zone in the selected window",
        )
    with k3:
        st.metric(
            f"Deals ≥ {threshold_pct}% below",
            f"{deals.height:,}",
            help="Contracts signed at least the threshold below the rolling median",
        )
    with k4:
        st.metric(
            "Deepest discount",
            f"{abs(deepest) * 100:.0f}% below" if deepest is not None else "—",
            help="The furthest any contract signed below the rolling median",
        )
    implied_yield = _implied_gross_yield(
        zone, latest_median, sale_psf_fn, bedroom, data_version, start_date, end_date
    )
    if implied_yield is not None:
        st.caption(
            f"Implied gross yield in {zone}: **{implied_yield:.1%}** "
            "(zone median annual rent per sqft ÷ zone median sale price per sqft)."
        )

    try:
        fig = rent_zone_chart(zone_scored, zone_rolling, zone, threshold_pct)
    except PanicException:
        # A polars Rust panic escapes `except Exception` and would kill the
        # script thread with the page frozen on RUNNING; degrade instead.
        LOGGER.exception("rent_zone_chart hit a polars panic")
        fig = None
    if fig is None:
        st.error(
            "The rent chart failed for this filter combination — the deals "
            "table below still works. Try a different zone/bedroom selection."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Below-median rents — {zone}")
    if deals.is_empty():
        st.info(
            f"No contracts signed ≥ {threshold_pct}% below the 14-day median "
            "in this window. Lower the threshold or widen the date range."
        )
    else:
        cols = [
            pl.col("date").cast(pl.Utf8).alias("Date"),
            pl.col("rooms_band").alias("Rooms"),
            pl.col("size_sqft").alias("Size (sqft)"),
            pl.col("annual_amount").round(0).alias("Annual rent (AED)"),
            pl.col("rent_psf").round(1).alias("AED/sqft/yr"),
            pl.col("rolling_median_psf").round(1).alias("Zone median AED/sqft/yr"),
            (pl.col("pct_vs_median") * 100).round(1).alias("% vs median"),
        ]
        if "reg_type" in deals.columns:
            cols.append(pl.col("reg_type").alias("Contract"))
        deals_display = deals.select(cols)
        st.dataframe(
            deals_display,
            use_container_width=True,
            height=min(400, 38 + 35 * deals_display.height),
        )
        zone_slug = "".join(ch if ch.isalnum() else "_" for ch in zone.lower())
        st.download_button(
            "Download rent deals CSV",
            data=deals_display.write_csv().encode(),
            file_name=f"rent_deals_{zone_slug}.csv",
            mime="text/csv",
        )
        st.caption(
            "A below-median rent is often an older or unrenovated unit, and "
            "Ejari carries no building attribution — treat these rows as "
            "negotiation benchmarks for the zone, not listings."
        )

    if len(zone_options) > 1:
        deal_counts = (
            scored.filter(
                (pl.col("pct_vs_median") <= -thr)
                & (pl.col("pct_vs_median") > -RENT_NON_MARKET_SPREAD)
            )
            .group_by("area")
            .agg(pl.len().alias("deals"))
        )
        if not deal_counts.is_empty():
            st.markdown("### Where the rent deals are")
            st.caption(
                f"Contracts ≥ {threshold_pct}% below their own zone's 14-day "
                "median, across the zones selected in the sidebar. Zones "
                "without deals are not shown."
            )
            st.plotly_chart(
                rent_deals_by_zone_chart(deal_counts, threshold_pct),
                use_container_width=True,
            )


def _render_rent_box_view(
    weekly: pl.DataFrame,
    zone: str,
    bedroom: str,
    start_date: date,
    end_date: date,
    meta: dict,
    include_renewals: bool,
) -> None:
    """Long-window view: weekly percentile box plots from the stats artifact."""
    band = bedroom if bedroom != "All" else "All"
    zone_weekly = (
        weekly.with_columns(area_display_expr().alias("area"))
        .filter(
            (pl.col("area") == zone)
            & (pl.col("rooms_band") == band)
            & (pl.col("week") >= start_date)
            & (pl.col("week") <= end_date)
        )
    )
    if "segment" in weekly.columns:
        segment = "new" if (meta["has_new_segment"] and not include_renewals) else "all"
        zone_weekly = zone_weekly.filter(pl.col("segment") == segment)
    zone_weekly = zone_weekly.sort("week")
    if zone_weekly.is_empty():
        st.warning(
            f"No weekly rent stats for {zone} ({band}) in the selected window."
        )
        return

    st.caption(
        "Weekly distribution of Ejari rents in the zone: box = middle half "
        "(q1–q3), whiskers = p10–p90 (not min/max), line = weekly median. "
        f"Pick a window of {RENT_DOT_MAX_DAYS} days or less within the "
        "contract artifact's coverage to see individual contracts and the "
        "deals table."
    )
    latest = zone_weekly.tail(1)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Latest weekly median",
            f"AED {latest['median'].item():,.0f}/sqft/yr",
            help="Median annual rent per sqft in the latest week shown",
        )
    with k2:
        st.metric(
            "Contracts in window",
            f"{int(zone_weekly['n'].sum()):,}",
            help="Ejari contracts behind the boxes shown",
        )
    with k3:
        st.metric("Weeks shown", f"{zone_weekly.height:,}")

    try:
        fig = rent_box_chart(zone_weekly, zone)
    except PanicException:
        LOGGER.exception("rent_box_chart hit a polars panic")
        fig = None
    if fig is None:
        st.error("The weekly distribution chart failed for this selection.")
    else:
        st.plotly_chart(fig, use_container_width=True)


def render_rent_scanner(
    neighborhoods: list[str],
    bedroom: str,
    start_date: date,
    end_date: date,
    sale_psf_fn,
    data_version: str,
) -> None:
    """Render the Rent Opportunity Scanner page."""
    loaded = _rent_artifacts_or_error()
    if loaded is None:
        return
    weekly, _recent, meta = loaded

    st.markdown("## Rent Opportunity Scanner")
    version_note = (
        f" · Ejari data updated {meta['rent_version'][:10]}"
        if meta["rent_version"]
        else ""
    )
    st.caption(
        "Ejari rental contracts (flats). Sidebar filters apply — Transaction "
        "Type is ignored here. All figures are annual AED per sqft "
        f"(**AED/sqft/yr**).{version_note}"
    )

    coverage_min = meta["weekly_min"]
    coverage_max = meta["weekly_max"]
    if meta["recent_max"] is not None and (
        coverage_max is None or meta["recent_max"] > coverage_max
    ):
        coverage_max = meta["recent_max"]
    if coverage_min is None or coverage_max is None:
        st.warning("The rent artifacts are empty; re-run the rents pull.")
        return
    requested = (start_date, end_date)
    start = max(start_date, coverage_min)
    end = min(end_date, coverage_max)
    if start > end:
        st.warning(
            "No rent data in the selected date range. Rent data covers "
            f"{coverage_min:%Y-%m-%d} to {coverage_max:%Y-%m-%d}."
        )
        return
    if (start, end) != requested:
        st.caption(f"Adjusted to rent data coverage: {start:%Y-%m-%d} to {end:%Y-%m-%d}.")

    zone_options = [n for n in NEIGHBORHOODS if n in neighborhoods]
    if st.session_state.get("zone_select") not in zone_options:
        st.session_state.pop("zone_select", None)
    dot_view = use_dot_view(start, end, meta["recent_min"])

    zone_col, control_col = st.columns([2, 3])
    with zone_col:
        zone = st.selectbox(
            "Zone",
            zone_options,
            key="zone_select",
            help="Every chart below focuses on this zone",
        )
    threshold_pct = 10
    if dot_view:
        with control_col:
            threshold_pct = st.slider(
                "Deal threshold — % below the zone's 14-day median rent",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key="rent_threshold",
                help="A contract at least this far under the rolling median counts as a deal",
            )

    include_renewals = False
    if (dot_view and meta["has_reg_type"]) or (not dot_view and meta["has_new_segment"]):
        include_renewals = st.checkbox(
            "Include renewals",
            value=False,
            key="rent_incl_renewals",
            help="Renewal rents are RERA-capped below market, so they are excluded by default",
        )
    elif dot_view:
        st.caption(
            "⚠️ The Ejari feed exposes no new-vs-renewal flag: renewals "
            "(RERA-capped below market) are mixed in, so discounts vs the "
            "median may be overstated."
        )

    if dot_view:
        _render_rent_dot_view(
            zone,
            zone_options,
            bedroom,
            start,
            end,
            meta,
            threshold_pct,
            include_renewals,
            sale_psf_fn,
            data_version,
        )
    else:
        _render_rent_box_view(
            weekly, zone, bedroom, start, end, meta, include_renewals
        )
