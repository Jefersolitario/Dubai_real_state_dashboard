"""Fair-value model for Dubai apartment transactions.

Predicts the fair value of each closed DLD apartment sale from hedonic
features (size, location, project, rooms, off-plan status, time, ...) and
scores the spread between the actual transaction price and the predicted
fair value. Large negative spreads, corroborated by distress signals
(court/forced-sale procedures, deep discounts, illiquid projects, multiple
sellers), flag potential distressed assets.

Pure Polars + scikit-learn: no Streamlit imports, so the module can be
trained, cross-validated, and smoke-tested offline.

Target: log(price per sqft). Log residuals are multiplicative, so
``spread_pct = actual / fair_value - 1`` is comparable across ticket sizes.
Validation: date-ordered ``TimeSeriesSplit(n_splits=10)`` — every fold
trains on the past and validates on the next time block.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from dashboard_constants import DISTRICT_TIER, SQM_TO_SQFT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Feature groups the optimization loop can toggle independently.
# Every derived group below is strictly past-only by construction
# (rolling windows closed="left" or shift over date-sorted frames), so no
# fold-refitting is needed and no lookahead is possible.
DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
    "project": True,        # PROJECT_EN / MASTER_PROJECT_EN categoricals
    "building": False,      # BUILDING_NAME_EN categorical (needs 24-month snapshot)
    "amenity": True,        # nearest metro/mall/landmark, parking, buyer/seller counts
    "comps_area": False,    # trailing 30-day area median PSF (strictly past-only)
    "comps_project": False, # trailing 60-day project median PSF (strictly past-only)
    # --- campaign feature groups (all strictly past-only) ---
    "comps_project_windows": False,  # 30d + 90d project median PSF variants
    "comps_building": False,         # trailing 90-day building median PSF
    "te_hist": False,                # expanding all-past project & building median PSF
    "liquidity": False,              # trailing txn counts (project 90d, area 30d)
    "momentum": False,               # area short/long comp ratio (30d vs 180d)
    "rel_size": False,               # log_sqft minus past project median log_sqft
    "comp_dispersion": False,        # trailing 60d project PSF std (uncertainty)
    "repeat_sale": False,            # prior sale PSF of the same pseudo-unit
    "repeat_sale_adj": False,        # prior unit PSF indexed by area market movement since
    # --- reference-data feature groups (require the `reference` frames) ---
    "project_meta": False,           # developer, building age, project size/height
    "rent_yield": False,             # trailing 180d area x rooms rent PSF + implied yield
    "service_charge": False,         # latest per-project service cost
    "unit_floor": False,             # floor via project+exact-area layout match (units table)
    "sale_index": False,             # official DLD monthly flat price index, 40d avail lag
    "rel_floor": False,              # layout floor mean / project max floors (needs unit_floor+project_meta)
    "rent_density": False,           # trailing 180d Ejari contract count for area x rooms band
}

DEFAULT_MODEL_PARAMS: dict = {
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

CORE_NUMERIC = ["log_sqft", "days_since_start", "month", "rooms_ord"]
AMENITY_NUMERIC = ["parking_count", "total_buyer", "total_seller"]
CORE_CATEGORICAL = ["AREA_EN", "IS_OFFPLAN_EN", "tier"]
PROJECT_CATEGORICAL = ["PROJECT_EN", "MASTER_PROJECT_EN"]
AMENITY_CATEGORICAL = ["NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN"]

MAX_CATEGORIES = 200  # top-N per categorical; the rest map to OTHER
UNSEEN_CODE = 0       # shared code for OTHER / UNKNOWN / unseen values

# PSF rows outside these quantiles are excluded from TRAINING only (target
# robustness); scoring covers every row so deep discounts are never hidden.
PSF_TRIM_QUANTILES = (0.005, 0.995)

# Drop rows where TRANS_VALUE/ACTUAL_AREA disagrees with the DLD-reported
# METER_SALE_PRICE by more than this relative tolerance (wrong-area guard).
AREA_MISMATCH_TOLERANCE = 0.10

# Optimizer output consumed at train time so the app ships the winning
# configuration instead of silently reverting to hard-coded defaults.
SHIPPING_CONFIG_PATH = Path(__file__).with_name("fair_value_config.json")

# Case-insensitive PROCEDURE_EN patterns suggesting a forced/distressed sale.
# The live DLD vocabulary should be checked via the value-counts view in the UI.
DISTRESS_PROCEDURE_PATTERN = r"(?i)court|forc|foreclos|auction|bankrupt|liquidat|execution"

# Procedures inside GROUP_EN="Sales" that are NOT arm's-length market deals:
# developer registrations and financing structures price at 0.42-0.75x market
# (data_quality_report.md) — they poison training and surface as fake bargains.
NON_MARKET_PROCEDURE_PATTERN = r"(?i)development|lease to own|payment plan"

# Columns carried through feature_engineering untouched, for scoring/display.
PASSTHROUGH_COLUMNS = [
    "TRANSACTION_NUMBER",
    "INSTANCE_DATE",
    "GROUP_EN",
    "PROCEDURE_EN",
    "AREA_EN",
    "PROJECT_EN",
    "MASTER_PROJECT_EN",
    "BUILDING_NAME_EN",
    "ROOMS_EN",
    "IS_OFFPLAN_EN",
    "TRANS_VALUE",
    "ACTUAL_AREA",
]


def feature_columns(feature_config: dict[str, bool] | None = None) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature column names for a config."""
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    numeric = list(CORE_NUMERIC)
    categorical = list(CORE_CATEGORICAL)
    if cfg["project"]:
        categorical += PROJECT_CATEGORICAL
    if cfg["building"]:
        categorical.append("BUILDING_NAME_EN")
    if cfg["amenity"]:
        numeric += AMENITY_NUMERIC
        categorical += AMENITY_CATEGORICAL
    if cfg["comps_area"]:
        numeric.append("area_comp_psf")
    if cfg["comps_project"]:
        numeric.append("project_comp_psf")
    if cfg["comps_project_windows"]:
        numeric += ["project_comp_psf_30", "project_comp_psf_90"]
    if cfg["comps_building"]:
        numeric.append("building_comp_psf")
    if cfg["te_hist"]:
        numeric += ["project_hist_psf", "building_hist_psf"]
    if cfg["liquidity"]:
        numeric += ["project_txn_90d", "area_txn_30d"]
    if cfg["momentum"]:
        numeric.append("area_momentum")
    if cfg["rel_size"]:
        numeric.append("rel_log_sqft")
    if cfg["comp_dispersion"]:
        numeric.append("project_comp_std")
    if cfg["repeat_sale"]:
        numeric += ["prior_unit_psf", "days_since_prior_sale"]
    if cfg["repeat_sale_adj"]:
        numeric.append("prior_unit_psf_adj")
    if cfg["project_meta"]:
        numeric += ["building_age_years", "project_units", "project_max_floors"]
        categorical.append("developer_name")
    if cfg["rent_yield"]:
        numeric += ["area_rent_psf_180d", "implied_gross_yield"]
    if cfg["service_charge"]:
        numeric.append("service_cost")
    if cfg["unit_floor"]:
        numeric += ["unit_floor", "layout_floor_mean", "layout_units", "unit_balcony_sqm"]
    if cfg["sale_index"]:
        numeric.append("mkt_index")
    if cfg["rel_floor"]:
        numeric.append("rel_floor_pct")
    if cfg["rent_density"]:
        numeric.append("rent_contracts_180d")
    return numeric, categorical


# reference frame names required per feature group (see store_reference_data_gcs.py)
REFERENCE_REQUIREMENTS = {
    "project_meta": ("projects", "buildings_agg"),
    "rent_yield": ("rent_index",),
    "service_charge": ("projects", "service_charges"),
    "unit_floor": ("projects", "units"),
    "sale_index": ("sale_index",),
    "rel_floor": ("projects", "units", "buildings_agg"),
    "rent_density": ("rent_index",),
}


def reference_needed(feature_config: dict | None = None) -> list[str]:
    """Reference frame names a feature config requires."""
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    names: list[str] = []
    for group, frames in REFERENCE_REQUIREMENTS.items():
        if cfg.get(group):
            names.extend(f for f in frames if f not in names)
    return names


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering(
    raw: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    date_origin: date | None = None,
    reference: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Filter to scoreable sales and derive model features + target.

    Keeps only apartment Sales rows with positive price and area, drops rows
    whose ACTUAL_AREA contradicts the DLD-reported METER_SALE_PRICE (plot
    area recorded instead of unit area), and returns one row per transaction
    with the ``log_psf`` target, model features, and passthrough columns.

    Deliberately does NOT trim PSF outliers: the deepest discounts are the
    distressed assets this model exists to surface. Apply :func:`trim_psf`
    to the result before TRAINING so the target stays robust.

    ``date_origin`` anchors ``days_since_start``; pass the training origin
    when scoring new data so the trend feature stays aligned.
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}

    # Frames that bypassed normalize_dld_transactions may lack optional
    # columns; backfill them as nulls so derivations below never crash.
    optional_sources = [
        "PARKING", "TOTAL_BUYER", "TOTAL_SELLER", "METER_SALE_PRICE",
        "PROCEDURE_EN", "PROJECT_EN", "MASTER_PROJECT_EN", "BUILDING_NAME_EN",
        "TRANSACTION_NUMBER", "USAGE_EN", "PROP_TYPE_EN",
        "NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN",
    ]
    missing_sources = [c for c in optional_sources if c not in raw.columns]
    if missing_sources:
        raw = raw.with_columns(pl.lit(None).alias(c) for c in missing_sources)

    df = raw.filter(
        (pl.col("GROUP_EN").cast(pl.Utf8).str.to_lowercase().str.contains("sale"))
        & (pl.col("PROP_SB_TYPE_EN").cast(pl.Utf8).str.to_lowercase() == "flat")
        & (pl.col("TRANS_VALUE").cast(pl.Float64, strict=False) > 0)
        & (pl.col("ACTUAL_AREA").cast(pl.Float64, strict=False) > 0)
        & ~pl.col("PROCEDURE_EN")
        .cast(pl.Utf8)
        .str.contains(NON_MARKET_PROCEDURE_PATTERN)
        .fill_null(False)
    )
    if df.is_empty():
        return df

    # Merge case variants (e.g. "Imperial Residence" vs "IMPERIAL RESIDENCE")
    # so the encoder sees one category per real-world name.
    df = df.with_columns(
        pl.col(c).cast(pl.Utf8).str.to_uppercase().alias(c)
        for c in ("PROJECT_EN", "MASTER_PROJECT_EN", "BUILDING_NAME_EN")
        if c in df.columns
    )

    df = df.with_columns(
        pl.col("INSTANCE_DATE")
        .cast(pl.Utf8)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias("date"),
        (pl.col("TRANS_VALUE") / (pl.col("ACTUAL_AREA") * SQM_TO_SQFT)).alias("psf"),
    ).drop_nulls(["date", "psf"])

    # Units guard: DLD publishes METER_SALE_PRICE (AED per sqm). When present,
    # TRANS_VALUE / ACTUAL_AREA must agree with it — a large mismatch means
    # ACTUAL_AREA holds a plot/common area, which would fabricate an extreme
    # fake discount. Rows without METER_SALE_PRICE are kept.
    msp = pl.col("METER_SALE_PRICE").cast(pl.Float64, strict=False)
    psf_sqm = pl.col("TRANS_VALUE") / pl.col("ACTUAL_AREA")
    df = df.filter(
        msp.is_null()
        | (msp <= 0)
        | (((psf_sqm - msp) / msp).abs() <= AREA_MISMATCH_TOLERANCE)
    )
    if df.is_empty():
        return df

    origin = date_origin or df.select(pl.col("date").min()).item()

    rooms = pl.col("ROOMS_EN").cast(pl.Utf8)
    df = df.with_columns(
        pl.col("psf").log().alias("log_psf"),
        (pl.col("ACTUAL_AREA") * SQM_TO_SQFT).log().alias("log_sqft"),
        (pl.col("date") - pl.lit(origin)).dt.total_days().cast(pl.Float64).alias("days_since_start"),
        pl.col("date").dt.month().cast(pl.Float64).alias("month"),
        pl.when(rooms.str.to_lowercase().str.contains("studio"))
        .then(pl.lit(0.0))
        .otherwise(rooms.str.extract(r"(\d+)").cast(pl.Float64, strict=False))
        .alias("rooms_ord"),
        pl.col("AREA_EN").cast(pl.Utf8).replace_strict(DISTRICT_TIER, default="UNKNOWN").alias("tier"),
    )

    if cfg["amenity"]:
        df = df.with_columns(
            pl.col("PARKING").cast(pl.Utf8).str.extract(r"(\d+)").cast(pl.Float64, strict=False).alias("parking_count"),
            pl.col("TOTAL_BUYER").cast(pl.Float64, strict=False).alias("total_buyer"),
            pl.col("TOTAL_SELLER").cast(pl.Float64, strict=False).alias("total_seller"),
        )

    # Trailing statistics: strictly past-only (closed="left" excludes the
    # current day; shift(1) excludes the current row) so they never leak the
    # transaction's own price or any future information. Rows without a
    # project/building are masked to null rather than pooled into one
    # implicit null group (HGB treats NaN natively).
    derived_groups = (
        "comps_area", "comps_project", "comps_project_windows",
        "comps_building", "te_hist", "liquidity", "momentum",
        "rel_size", "comp_dispersion", "repeat_sale", "repeat_sale_adj",
    )
    if any(cfg[g] for g in derived_groups):
        df = df.sort("date")

        def past_stat(value: str, window: str, group: str, stat: str = "median") -> pl.Expr:
            rolled = getattr(pl.col(value), f"rolling_{stat}_by")(
                "date", window_size=window, closed="left"
            ).over(group)
            return pl.when(pl.col(group).is_not_null()).then(rolled)

        comps = []
        if cfg["comps_area"]:
            comps.append(past_stat("psf", "30d", "AREA_EN").alias("area_comp_psf"))
        if cfg["comps_project"]:
            comps.append(past_stat("psf", "60d", "PROJECT_EN").alias("project_comp_psf"))
        if cfg["comps_project_windows"]:
            comps.append(past_stat("psf", "30d", "PROJECT_EN").alias("project_comp_psf_30"))
            comps.append(past_stat("psf", "90d", "PROJECT_EN").alias("project_comp_psf_90"))
        if cfg["comps_building"]:
            comps.append(past_stat("psf", "90d", "BUILDING_NAME_EN").alias("building_comp_psf"))
        if cfg["te_hist"]:
            # Expanding all-past medians (window far longer than the data span).
            comps.append(past_stat("psf", "3650d", "PROJECT_EN").alias("project_hist_psf"))
            comps.append(past_stat("psf", "3650d", "BUILDING_NAME_EN").alias("building_hist_psf"))
        if cfg["liquidity"]:
            df = df.with_columns(pl.lit(1.0).alias("_one"))
            comps.append(past_stat("_one", "90d", "PROJECT_EN", "sum").alias("project_txn_90d"))
            comps.append(past_stat("_one", "30d", "AREA_EN", "sum").alias("area_txn_30d"))
        if cfg["momentum"]:
            comps.append(
                (
                    past_stat("psf", "30d", "AREA_EN")
                    / past_stat("psf", "180d", "AREA_EN")
                ).alias("area_momentum")
            )
        if cfg["rel_size"]:
            comps.append(
                (pl.col("ACTUAL_AREA").log() - past_stat("ACTUAL_AREA", "3650d", "PROJECT_EN").log())
                .alias("rel_log_sqft")
            )
        if cfg["comp_dispersion"]:
            comps.append(past_stat("psf", "60d", "PROJECT_EN", "std").alias("project_comp_std"))
        if comps:
            df = df.with_columns(comps)
        if "_one" in df.columns:
            df = df.drop("_one")

        if cfg["repeat_sale"] or cfg["repeat_sale_adj"]:
            # Pseudo-unit: same building + same room label + same area to
            # 0.1 sqm. shift(1) over the date-sorted frame = the unit's most
            # recent PRIOR sale only.
            unit_key = pl.concat_str(
                pl.col("BUILDING_NAME_EN").fill_null("?"),
                pl.col("ROOMS_EN").cast(pl.Utf8).fill_null("?"),
                (pl.col("ACTUAL_AREA") * 10).round(0).cast(pl.Int64).cast(pl.Utf8),
                separator="|",
            )
            df = df.with_columns(unit_key.alias("_unit"))
            df = df.with_columns(
                pl.when(pl.col("BUILDING_NAME_EN").is_not_null())
                .then(pl.col("psf").shift(1).over("_unit"))
                .alias("prior_unit_psf"),
                pl.when(pl.col("BUILDING_NAME_EN").is_not_null())
                .then(
                    (pl.col("date") - pl.col("date").shift(1).over("_unit"))
                    .dt.total_days()
                    .cast(pl.Float64)
                )
                .alias("days_since_prior_sale"),
            )
            if cfg["repeat_sale_adj"]:
                # Index the prior price by area-market movement since the
                # prior sale: both comp values are strictly past their rows.
                df = df.with_columns(
                    past_stat("psf", "30d", "AREA_EN").alias("_area_now")
                ).with_columns(
                    pl.col("_area_now").shift(1).over("_unit").alias("_area_then")
                ).with_columns(
                    (
                        pl.col("prior_unit_psf")
                        * pl.col("_area_now")
                        / pl.col("_area_then")
                    ).alias("prior_unit_psf_adj")
                ).drop("_area_now", "_area_then")
            df = df.drop("_unit")

    # Reference-data joins (projects / rents / service charges). All values
    # are static facts or strictly-past aggregates; the rent grid's week-w
    # value covers only contracts starting before w, and a backward as-of
    # join attaches week <= transaction date, so it is strictly past.
    ref_needed = reference_needed(cfg)
    if ref_needed:
        missing_ref = [n for n in ref_needed if reference is None or n not in reference]
        if missing_ref:
            raise ValueError(
                f"feature_config requires reference frames {missing_ref}; "
                "load them via gcs_storage.read_reference_frames "
                "(published by store_reference_data_gcs.py)"
            )

    if cfg["project_meta"] or cfg["service_charge"]:
        lookup = (
            reference["projects"]
            .select(
                pl.col("project_number").cast(pl.Int64, strict=False),
                pl.col("project_id").cast(pl.Int64, strict=False),
                pl.col("developer_name").cast(pl.Utf8),
                pl.coalesce(pl.col("completion_date"), pl.col("project_end_date"))
                .cast(pl.Utf8)
                .str.slice(0, 10)
                .str.to_date("%Y-%m-%d", strict=False)
                .alias("_completion"),
                pl.col("no_of_units").cast(pl.Float64, strict=False).alias("project_units"),
            )
            .drop_nulls("project_number")
            .unique("project_number", keep="first")
        )
        if cfg["project_meta"]:
            bagg = reference["buildings_agg"].select(
                pl.col("project_id").cast(pl.Int64, strict=False),
                pl.col("project_max_floors").cast(pl.Float64, strict=False),
            ).unique("project_id")
            lookup = lookup.join(bagg, on="project_id", how="left")
        if cfg["service_charge"]:
            sc = reference["service_charges"].select(
                pl.col("project_id").cast(pl.Int64, strict=False),
                pl.col("service_cost").cast(pl.Float64, strict=False),
            ).unique("project_id")
            lookup = lookup.join(sc, on="project_id", how="left")
        df = (
            df.with_columns(pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_pn"))
            .join(lookup.rename({"project_number": "_pn"}).drop("project_id"), on="_pn", how="left")
            .drop("_pn")
        )
        if cfg["project_meta"]:
            df = df.with_columns(
                ((pl.col("date") - pl.col("_completion")).dt.total_days() / 365.25)
                .alias("building_age_years")
            )
        df = df.drop("_completion")

    if cfg["rent_yield"] or cfg["rent_density"]:
        rooms = pl.col("ROOMS_EN").cast(pl.Utf8)
        band = (
            pl.when(rooms.str.to_lowercase().str.contains("studio"))
            .then(pl.lit("Studio"))
            .when(rooms.str.extract(r"(\d+)").cast(pl.Int32, strict=False) == 1)
            .then(pl.lit("1BR"))
            .when(rooms.str.extract(r"(\d+)").cast(pl.Int32, strict=False) == 2)
            .then(pl.lit("2BR"))
            .when(rooms.str.extract(r"(\d+)").cast(pl.Int32, strict=False) == 3)
            .then(pl.lit("3BR"))
            .when(rooms.str.extract(r"(\d+)").cast(pl.Int32, strict=False) >= 4)
            .then(pl.lit("4BR+"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )
        grid_cols = [
            pl.col("AREA_EN").cast(pl.Utf8),
            pl.col("rooms_band").cast(pl.Utf8).alias("_band"),
            pl.col("week").cast(pl.Date),
            pl.col("area_rent_psf_180d").cast(pl.Float64, strict=False),
        ]
        if cfg["rent_density"]:
            grid_cols.append(
                pl.col("rent_contracts_180d").cast(pl.Float64, strict=False)
            )
        grid = (
            reference["rent_index"]
            .select(grid_cols)
            .drop_nulls(["AREA_EN", "_band", "week"])
            .sort("week")
        )
        df = df.with_columns(band.alias("_band")).sort("date")
        df = df.join_asof(
            grid,
            left_on="date",
            right_on="week",
            by=["AREA_EN", "_band"],
            strategy="backward",
        ).drop("_band", "week")
        if cfg["rent_yield"]:
            denom_cols = [c for c in ("project_comp_psf", "area_comp_psf") if c in df.columns]
            if denom_cols:
                df = df.with_columns(
                    (pl.col("area_rent_psf_180d") / pl.coalesce([pl.col(c) for c in denom_cols]))
                    .alias("implied_gross_yield")
                )
            else:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("implied_gross_yield"))

    if cfg["unit_floor"]:
        # Static unit facts from the DLD units registry. Transactions carry no
        # unit key, so we match project (via project_number -> project_id) +
        # exact registered area: where that layout is unique in the project,
        # unit_floor is the true floor; where layouts stack, only the layout's
        # floor distribution and balcony area are attached.
        bridge = (
            reference["projects"]
            .select(
                pl.col("project_number").cast(pl.Int64, strict=False).alias("_pn"),
                pl.col("project_id").cast(pl.Int64, strict=False).alias("_pid"),
            )
            .drop_nulls()
            .unique("_pn")
        )
        units = (
            reference["units"]
            .select(
                pl.col("project_id").cast(pl.Int64, strict=False).alias("_pid"),
                pl.col("floor_num").cast(pl.Float64, strict=False),
                pl.col("actual_area").cast(pl.Float64, strict=False)
                .round(2).alias("_akey"),
                pl.col("unit_balcony_area").cast(pl.Float64, strict=False),
            )
            .drop_nulls(["_pid", "_akey"])
        )
        layouts = (
            units.group_by("_pid", "_akey")
            .agg(
                pl.len().cast(pl.Float64).alias("layout_units"),
                pl.col("floor_num").mean().alias("layout_floor_mean"),
                pl.col("floor_num").first().alias("_floor_single"),
                pl.col("unit_balcony_area").mean().alias("unit_balcony_sqm"),
            )
            .with_columns(
                pl.when(pl.col("layout_units") == 1)
                .then(pl.col("_floor_single"))
                .otherwise(None)
                .alias("unit_floor")
            )
            .drop("_floor_single")
        )
        df = (
            df.with_columns(
                pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_pn"),
                pl.col("ACTUAL_AREA").cast(pl.Float64).round(2).alias("_akey"),
            )
            .join(bridge, on="_pn", how="left")
            .join(layouts, on=["_pid", "_akey"], how="left")
            .drop("_pn", "_pid", "_akey")
        )

    if cfg["rel_floor"]:
        # Where the layout sits in the tower: floor mean over the building's
        # max floors. Both inputs are static registry facts.
        if not (cfg["unit_floor"] and cfg["project_meta"]):
            raise ValueError("rel_floor requires unit_floor and project_meta enabled")
        df = df.with_columns(
            pl.when(pl.col("project_max_floors") > 0)
            .then(pl.col("layout_floor_mean") / pl.col("project_max_floors"))
            .otherwise(None)
            .alias("rel_floor_pct")
        )

    if cfg["sale_index"]:
        # Official monthly flat index; the value for month m (dated the 1st)
        # is treated as available 40 days later (~10th of the next month), so
        # the as-of join can never use an index covering the sale's own month.
        idx = (
            reference["sale_index"]
            .select(
                pl.col("month").cast(pl.Date),
                pl.coalesce(pl.col("flat_price_index"), pl.col("flat_index"))
                .cast(pl.Float64, strict=False)
                .alias("mkt_index"),
            )
            .drop_nulls()
            .sort("month")
            .with_columns(pl.col("month").dt.offset_by("40d").alias("_avail"))
        )
        df = df.sort("date").join_asof(
            idx.select("_avail", "mkt_index"),
            left_on="date",
            right_on="_avail",
            strategy="backward",
        ).drop("_avail")

    _, categorical = feature_columns(cfg)
    missing = [c for c in categorical if c not in df.columns]
    if missing:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing)
    df = df.with_columns(
        pl.col(c).cast(pl.Utf8).fill_null("UNKNOWN").alias(c) for c in categorical
    )
    keep = ["date", "psf", "log_psf"]
    numeric, _ = feature_columns(cfg)
    keep += numeric + categorical
    keep += [c for c in PASSTHROUGH_COLUMNS if c in df.columns and c not in keep]
    return df.select(keep).sort("date")


def trim_psf(df: pl.DataFrame) -> pl.DataFrame:
    """Drop PSF outliers for TRAINING (data-entry noise robustness).

    Never apply this to the frame being scored — the trimmed tail is where
    the deepest distressed discounts live.
    """
    if df.is_empty():
        return df
    lo, hi = df.select(
        pl.col("psf").quantile(PSF_TRIM_QUANTILES[0]).alias("lo"),
        pl.col("psf").quantile(PSF_TRIM_QUANTILES[1]).alias("hi"),
    ).row(0)
    return df.filter(pl.col("psf").is_between(lo, hi))


def load_shipping_config() -> tuple[dict[str, bool], dict]:
    """(feature_config, model_params) the app should train with.

    Reads the optimizer-written ``fair_value_config.json`` when present so
    the dashboard ships the winning configuration; falls back to the module
    defaults otherwise.
    """
    feature_config = dict(DEFAULT_FEATURE_CONFIG)
    model_params = dict(DEFAULT_MODEL_PARAMS)
    try:
        payload = json.loads(SHIPPING_CONFIG_PATH.read_text())
        feature_config.update(payload.get("feature_config", {}))
        model_params.update(payload.get("model_params", {}))
    except (FileNotFoundError, ValueError):
        pass
    return feature_config, model_params


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def fit_encoders(
    df: pl.DataFrame, feature_config: dict[str, bool] | None = None
) -> dict[str, dict[str, int]]:
    """Per-categorical value->code maps from training data.

    Code 0 is reserved for OTHER/UNKNOWN/unseen; the most frequent
    ``MAX_CATEGORIES - 1`` values get codes 1..N.
    """
    _, categorical = feature_columns(feature_config)
    max_categories = int((feature_config or {}).get("max_categories", MAX_CATEGORIES))
    encoders: dict[str, dict[str, int]] = {}
    for col in categorical:
        top = (
            df.group_by(col)
            .len()
            .sort(["len", col], descending=[True, False])
            .head(max_categories - 1)
            .get_column(col)
            .to_list()
        )
        encoders[col] = {value: code for code, value in enumerate(top, start=1)}
    return encoders


def to_matrix(
    df: pl.DataFrame,
    encoders: dict[str, dict[str, int]],
    feature_config: dict[str, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[int]]:
    """(X, y, feature_names, categorical_idx) for scikit-learn."""
    numeric, categorical = feature_columns(feature_config)
    frame = df.with_columns(
        pl.col(col)
        .replace_strict(encoders[col], default=UNSEEN_CODE)
        .cast(pl.Float64)
        .alias(f"{col}__code")
        for col in categorical
    )
    feature_names = numeric + categorical
    columns = numeric + [f"{c}__code" for c in categorical]
    X = frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()
    y = frame.get_column("log_psf").to_numpy() if "log_psf" in frame.columns else None
    categorical_idx = list(range(len(numeric), len(columns)))
    return X, y, feature_names, categorical_idx


# ---------------------------------------------------------------------------
# Validation + training
# ---------------------------------------------------------------------------

def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true - y_pred  # log(actual) - log(fair value)
    spread = np.expm1(resid)  # actual / fair value - 1
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "medape": float(np.median(np.abs(spread))),
        "mae_log": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def _make_model(model_params: dict | None, categorical_idx: list[int], random_state: int):
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    return HistGradientBoostingRegressor(
        categorical_features=categorical_idx,
        random_state=random_state,
        **params,
    )


def cross_validate(
    df: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    model_params: dict | None = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """Out-of-time CV metrics via date-ordered TimeSeriesSplit.

    Encoders are refit on each fold's training slice so validation rows
    with unseen categories are handled exactly as they would be live.
    """
    df = df.sort("date")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(np.arange(df.height)):
        train_df = df[train_idx.min() : train_idx.max() + 1]
        val_df = df[val_idx.min() : val_idx.max() + 1]
        encoders = fit_encoders(train_df, feature_config)
        X_tr, y_tr, _, cat_idx = to_matrix(train_df, encoders, feature_config)
        X_va, y_va, _, _ = to_matrix(val_df, encoders, feature_config)
        model = _make_model(model_params, cat_idx, random_state)
        model.fit(X_tr, y_tr)
        folds.append(_fold_metrics(y_va, model.predict(X_va)))

    return summarize_folds(folds, n_rows=df.height, n_splits=n_splits)


def summarize_folds(folds: list[dict[str, float]], n_rows: int, n_splits: int) -> dict:
    """Mean ± std summary of per-fold metrics (shared with the optimizer)."""
    summary = {
        "n_splits": n_splits,
        "n_rows": n_rows,
        "folds": folds,
    }
    for key in ("medape", "mae_log", "r2"):
        values = [f[key] for f in folds]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


@dataclass
class FairValueResult:
    model: HistGradientBoostingRegressor
    encoders: dict[str, dict[str, int]]
    feature_names: list[str]
    categorical_idx: list[int]
    feature_config: dict[str, bool]
    metrics: dict
    importances: pl.DataFrame
    date_origin: date
    trained_rows: int = 0


def train_fair_value_model(
    df: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    model_params: dict | None = None,
    n_splits: int = 10,
    random_state: int = 42,
    importance_sample: int = 20_000,
    run_cv: bool = True,
) -> FairValueResult:
    """CV-validate, then refit on all rows for scoring.

    Reported metrics are honest out-of-time numbers from the 10-fold
    TimeSeriesSplit; the returned model is refit on the full frame
    (standard AVM anomaly-scoring pattern). Permutation importance is
    OUT-OF-SAMPLE: a helper model fit on all rows except the most recent
    ~10% is evaluated on that held-out tail, so high-cardinality features
    don't get inflated by in-sample memorization. Pass ``run_cv=False``
    when the CV numbers are already known (e.g. from the optimization loop).
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    df = df.sort("date")
    metrics = (
        cross_validate(df, cfg, model_params, n_splits=n_splits, random_state=random_state)
        if run_cv
        else {}
    )

    encoders = fit_encoders(df, cfg)
    X, y, feature_names, cat_idx = to_matrix(df, encoders, cfg)
    model = _make_model(model_params, cat_idx, random_state)
    model.fit(X, y)

    tail = max(1, df.height // 10)
    rng = np.random.default_rng(random_state)
    tail_idx = np.arange(df.height - tail, df.height)
    if tail > importance_sample:
        tail_idx = rng.choice(tail_idx, size=importance_sample, replace=False)
    head_end = df.height - tail
    if head_end >= 100:
        pi_model = _make_model(model_params, cat_idx, random_state)
        pi_model.fit(X[:head_end], y[:head_end])
    else:  # tiny frames: fall back to the full model
        pi_model = model
    perm = permutation_importance(
        pi_model, X[tail_idx], y[tail_idx], n_repeats=5, random_state=random_state
    )
    importances = pl.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort("importance_mean", descending=True)

    # Recover the origin days_since_start was actually derived from
    # (feature_engineering's untrimmed min date): for any row,
    # origin = date - days_since_start. Using the trimmed frame's min date
    # here would offset the trend axis when re-featurizing fresh data with
    # date_origin=result.date_origin.
    first_date, first_days = df.select("date", "days_since_start").row(0)
    origin = first_date - timedelta(days=int(first_days))

    return FairValueResult(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        categorical_idx=cat_idx,
        feature_config=cfg,
        metrics=metrics,
        importances=importances,
        date_origin=origin,
        trained_rows=df.height,
    )


# ---------------------------------------------------------------------------
# Model bundle (offline training -> light inference artifact)
# ---------------------------------------------------------------------------

BUNDLE_VERSION = 1


def export_bundle(result: FairValueResult, extra: dict | None = None) -> bytes:
    """Serialize a trained model + everything inference needs.

    The bundle is what the Streamlit app loads instead of training —
    training happens offline (train_fair_value.py) where CPU/RAM exist.
    """
    import sklearn

    payload = {
        "bundle_version": BUNDLE_VERSION,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": result.model,
        "encoders": result.encoders,
        "feature_names": result.feature_names,
        "categorical_idx": result.categorical_idx,
        "feature_config": result.feature_config,
        "metrics": result.metrics,
        "importances": result.importances.to_dicts(),
        "date_origin": result.date_origin,
        "trained_rows": result.trained_rows,
        **(extra or {}),
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(data: bytes) -> tuple[FairValueResult, dict]:
    """(FairValueResult, metadata) from export_bundle bytes.

    Warns (does not fail) on sklearn version mismatch — HGB pickles are
    generally forward-compatible within minor releases, and the weekly
    retrain refreshes the bundle anyway.
    """
    import sklearn

    payload = pickle.loads(data)
    if payload.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            "Model bundle was trained with scikit-learn "
            f"{payload.get('sklearn_version')} but {sklearn.__version__} is "
            "installed; retrain with train_fair_value.py if predictions look off.",
            stacklevel=2,
        )
    result = FairValueResult(
        model=payload["model"],
        encoders=payload["encoders"],
        feature_names=payload["feature_names"],
        categorical_idx=payload["categorical_idx"],
        feature_config=payload["feature_config"],
        metrics=payload["metrics"],
        importances=pl.DataFrame(payload["importances"]),
        date_origin=payload["date_origin"],
        trained_rows=payload["trained_rows"],
    )
    meta_keys = ("bundle_version", "sklearn_version", "trained_at",
                 "data_min_date", "data_max_date", "source")
    metadata = {key: payload.get(key) for key in meta_keys}
    return result, metadata


# ---------------------------------------------------------------------------
# Scoring + distress flagging
# ---------------------------------------------------------------------------

def score_transactions(result: FairValueResult, df: pl.DataFrame) -> pl.DataFrame:
    """Add fair-value predictions and the price spread to a feature frame.

    ``df`` must come from :func:`feature_engineering` with the same
    feature_config (pass ``date_origin=result.date_origin`` when scoring
    data outside the training frame).
    """
    X, _, _, _ = to_matrix(df, result.encoders, result.feature_config)
    pred_log_psf = result.model.predict(X)
    return df.with_columns(
        pl.Series("pred_psf", np.exp(pred_log_psf)),
    ).with_columns(
        (pl.col("pred_psf") * pl.col("ACTUAL_AREA") * SQM_TO_SQFT).alias("fair_value_aed"),
    ).with_columns(
        (pl.col("TRANS_VALUE") / pl.col("fair_value_aed") - 1).alias("spread_pct"),
    )


def flag_distress(
    scored: pl.DataFrame,
    spread_threshold: float = -0.15,
    deep_discount: float = -0.25,
    min_project_txns: int = 8,
) -> pl.DataFrame:
    """Flag below-fair-value rows and corroborated distressed-asset candidates.

    ``distressed`` requires the spread to be at/below ``spread_threshold``
    AND at least one corroborating signal that is INDEPENDENT of the model
    residual (forced-sale procedure, illiquid project, multiple sellers) —
    a lone model miss, however large, is never enough. ``sig_deep_discount``
    is annotated for display but deliberately does not count, since it is
    derived from the same residual as ``below_fair_value``.
    """
    procedure = (
        pl.col("PROCEDURE_EN").cast(pl.Utf8).str.contains(DISTRESS_PROCEDURE_PATTERN).fill_null(False)
        if "PROCEDURE_EN" in scored.columns
        else pl.lit(False)
    )
    multi_seller = (
        (pl.col("total_seller") > 1).fill_null(False)
        if "total_seller" in scored.columns
        else pl.lit(False)
    )
    df = scored.with_columns(
        (pl.col("spread_pct") <= spread_threshold).alias("below_fair_value"),
        procedure.alias("sig_procedure"),
        (pl.col("spread_pct") <= deep_discount).alias("sig_deep_discount"),
        (pl.len().over("PROJECT_EN") < min_project_txns).alias("sig_illiquid_project"),
        multi_seller.alias("sig_multi_seller"),
    )
    # Only residual-independent signals corroborate distress.
    corroborating = ["sig_procedure", "sig_illiquid_project", "sig_multi_seller"]
    signal_labels = {
        "sig_procedure": "forced-sale procedure",
        "sig_illiquid_project": "illiquid project",
        "sig_multi_seller": "multiple sellers",
        "sig_deep_discount": "deep discount",
    }
    df = df.with_columns(
        sum(pl.col(c).cast(pl.Int32) for c in corroborating).alias("distress_score"),
    ).with_columns(
        (pl.col("below_fair_value") & (pl.col("distress_score") >= 1)).alias("distressed"),
        pl.concat_str(
            [
                pl.when(pl.col(c)).then(pl.lit(label)).otherwise(pl.lit(None))
                for c, label in signal_labels.items()
            ],
            separator="; ",
            ignore_nulls=True,
        ).alias("signals"),
    )
    return df
