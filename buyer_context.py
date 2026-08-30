"""Buyer-facing context for flagged sales: rental income and building works.

Deliberately NOT model features. Campaigns 3 and 4 measured rent (and
project-linked rent) as fair-value inputs and rejected them — knowing the
neighbours' rent does not help predict a sale price the comps already
explain. It does, however, help a *buyer* decide whether a flat is worth
owning: what it would rent for, what that yields on the asking price, and
whether the building has seen recent works.

Pure Polars, no Streamlit — the caller supplies the reference frames
(published by ingestion/store_reference_data_gcs.py) and each one is
optional, so the columns appear as the artifacts become available.
"""

from __future__ import annotations

from datetime import date

import polars as pl

# Ejari rent bands; the dashboard's bedroom labels beyond 3BR fold into 4BR+.
_RENT_BAND = {"Studio": "Studio", "1BR": "1BR", "2BR": "2BR", "3BR": "3BR"}

# Window for "recent building works", and the label cut-offs.
WORKS_WINDOW_DAYS = 1826  # 5 years


def rooms_band_expr(column: str = "bedroom_type") -> pl.Expr:
    """Dashboard bedroom label -> Ejari rooms band (4BR and up -> 4BR+)."""
    label = pl.col(column).cast(pl.Utf8)
    return (
        pl.when(label.is_in(list(_RENT_BAND)))
        .then(label.replace(_RENT_BAND))
        .when(label.str.extract(r"(\d+)").cast(pl.Int32, strict=False) >= 4)
        .then(pl.lit("4BR+"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def latest_project_rent(rent_project_index: pl.DataFrame) -> pl.DataFrame:
    """Most recent trailing rent PSF per project (annual AED/sqft/yr)."""
    return (
        rent_project_index.sort("week")
        .group_by("project_number")
        .agg(pl.col("project_rent_psf_180d").last().alias("_project_rent_psf"))
        .with_columns(pl.col("project_number").cast(pl.Int64, strict=False))
        .drop_nulls()
    )


def latest_area_rent(rent_index: pl.DataFrame) -> pl.DataFrame:
    """Most recent trailing rent PSF per district x rooms band."""
    return (
        rent_index.sort("week")
        .group_by("AREA_EN", "rooms_band")
        .agg(pl.col("area_rent_psf_180d").last().alias("_area_rent_psf"))
        .drop_nulls()
    )


def building_works(permits: pl.DataFrame, today: date | None = None) -> pl.DataFrame:
    """Per project: adjustment-permit count in the window and recency.

    Computed against ``today`` rather than each sale's date: this is context
    about the building's current state for someone buying now, not a
    point-in-time model feature.
    """
    today = today or date.today()
    cutoff = pl.lit(today) - pl.duration(days=WORKS_WINDOW_DAYS)
    return (
        permits.with_columns(pl.col("permit_date").cast(pl.Date))
        .filter(pl.col("permit_date") <= pl.lit(today))
        .group_by("project_number")
        .agg(
            pl.col("permit_date").filter(pl.col("permit_date") >= cutoff)
            .len().alias("_works_count"),
            pl.col("permit_date").max().alias("_works_last"),
        )
        .with_columns(pl.col("project_number").cast(pl.Int64, strict=False))
        .drop_nulls("project_number")
    )


def _works_label(today: date) -> pl.Expr:
    """'2 · last 8mo' / 'none in 5 years' / '—' when the building is unknown."""
    months = ((pl.lit(today) - pl.col("_works_last")).dt.total_days() / 30.44).round(0)
    recency = (
        pl.when(months < 1).then(pl.lit("this month"))
        .when(months < 24).then(pl.concat_str([months.cast(pl.Int32).cast(pl.Utf8), pl.lit("mo")]))
        .otherwise(pl.concat_str([(months / 12).round(0).cast(pl.Int32).cast(pl.Utf8), pl.lit("yr")]))
    )
    return (
        pl.when(pl.col("_works_count").is_null())
        .then(pl.lit("—"))
        .when(pl.col("_works_count") == 0)
        .then(pl.lit("none in 5 years"))
        .otherwise(
            pl.concat_str([
                pl.col("_works_count").cast(pl.Utf8),
                pl.lit(" · last "),
                recency,
            ])
        )
    )


def attach_buyer_context(
    df: pl.DataFrame,
    rent_project_index: pl.DataFrame | None = None,
    rent_index: pl.DataFrame | None = None,
    permits: pl.DataFrame | None = None,
    today: date | None = None,
) -> pl.DataFrame:
    """Add rent, gross yield, and building-works columns to scored sales.

    Rent comes from the sale's own project where the Ejari linkage covers it
    (same building stock, the honest comparison) and falls back to the
    district x rooms-band grid otherwise; ``rent_basis`` records which.
    Gross yield is annual rent over the actual price — before service
    charges, which are material in Dubai.
    """
    today = today or date.today()
    out = df

    has_project_key = "PROJECT_NUMBER" in df.columns
    if (
        has_project_key
        and rent_project_index is not None
        and not rent_project_index.is_empty()
    ):
        out = out.with_columns(
            pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("project_number")
        ).join(latest_project_rent(rent_project_index), on="project_number", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("_project_rent_psf"))

    if rent_index is not None and not rent_index.is_empty():
        out = out.with_columns(rooms_band_expr().alias("_rooms_band")).join(
            latest_area_rent(rent_index),
            left_on=["AREA_EN", "_rooms_band"],
            right_on=["AREA_EN", "rooms_band"],
            how="left",
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("_area_rent_psf"))

    out = out.with_columns(
        pl.coalesce([pl.col("_project_rent_psf"), pl.col("_area_rent_psf")]).alias("rent_psf_yr"),
        pl.when(pl.col("_project_rent_psf").is_not_null())
        .then(pl.lit("project"))
        .when(pl.col("_area_rent_psf").is_not_null())
        .then(pl.lit("area"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("rent_basis"),
    ).with_columns(
        (pl.col("rent_psf_yr") * pl.col("size_sqft")).alias("est_annual_rent")
    ).with_columns(
        pl.when(pl.col("TRANS_VALUE") > 0)
        .then(pl.col("est_annual_rent") / pl.col("TRANS_VALUE"))
        .alias("gross_yield")
    )

    if has_project_key and permits is not None and not permits.is_empty():
        if "project_number" not in out.columns:
            out = out.with_columns(
                pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("project_number")
            )
        out = out.join(building_works(permits, today), on="project_number", how="left")
        out = out.with_columns(_works_label(today).alias("works_label"))

    drop = [c for c in ("_project_rent_psf", "_area_rent_psf", "_rooms_band",
                        "_works_count", "_works_last") if c in out.columns]
    return out.drop(drop)
