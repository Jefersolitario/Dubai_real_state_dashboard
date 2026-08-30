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

from model import data_cleaning
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
    "price_history": False,          # expanding all-past project & building median PSF
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
    "rel_floor": False,              # layout floor mean / project max floors (needs unit_floor+project_meta)
    "rent_density": False,           # trailing 180d Ejari contract count for area x rooms band
    "comps_rooms": False,            # trailing comps per project x rooms and area x rooms (strictly past)
    "rooms_dynamics": False,         # dispersion + liquidity per project x rooms (strictly past)
    # --- rent campaign feature groups (campaign 3; all strictly past) ---
    "rent_level": False,             # area x rooms trailing 180d rent PSF, level only
    "rent_momentum": False,          # rent grid now vs ~13 weeks ago (area x rooms)
    "yield_matched": False,          # rent PSF / area x rooms sale comp (needs comps_rooms)
    "yield_spread": False,           # matched yield minus same-band city yield (needs comps_rooms)
    "rent_pooling": False,           # rent PSF shrunk toward the city band (precision-weighted)
    "rent_anchor": False,            # income-approach price anchor: rent / city band yield
    "rent_divergence": False,        # rent momentum minus area sale-price momentum
    "rent_new_segment": False,       # new-contract (non-renewal) rents: level + premium vs stock
    "rent_project": False,           # project-linked rents: trailing PSF, count, same-project yield
    "renovation_permits": False,     # DM adjustment-permit history on the project (campaign 4)
    # --- data quality (see data_cleaning.py) ---
    "data_cleaning": False,          # repair digit-shift typos; exclude review_only/quarantine rows
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
    "PROJECT_NUMBER",  # key for project-linked reference data (rents, permits)
    "MASTER_PROJECT_EN",
    "BUILDING_NAME_EN",
    "ROOMS_EN",
    "IS_OFFPLAN_EN",
    "TRANS_VALUE",
    "ACTUAL_AREA",
    # present only when the data_cleaning feature group is enabled
    "dq_rule",
    "dq_action",
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
    if cfg["price_history"]:
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
    if cfg["rel_floor"]:
        numeric.append("rel_floor_pct")
    if cfg["rent_density"]:
        numeric.append("rent_contracts_180d")
    if cfg["comps_rooms"]:
        numeric += ["project_rooms_comp_psf", "area_rooms_comp_psf"]
    if cfg["rooms_dynamics"]:
        numeric += ["project_rooms_comp_std", "project_rooms_txn_90d"]
    if cfg["rent_level"]:
        numeric.append("rent_psf_180d")
    if cfg["rent_momentum"]:
        numeric.append("rent_mom_91d")
    if cfg["yield_matched"]:
        numeric.append("matched_gross_yield")
    if cfg["yield_spread"]:
        numeric.append("yield_spread")
    if cfg["rent_pooling"]:
        numeric.append("rent_psf_shrunk")
    if cfg["rent_anchor"]:
        numeric.append("rent_implied_psf")
    if cfg["rent_divergence"]:
        numeric.append("rent_price_div")
    if cfg["rent_new_segment"]:
        numeric += ["rent_new_psf_13w", "rent_new_premium"]
    if cfg["rent_project"]:
        numeric += [
            "project_rent_psf_180d",
            "project_rent_contracts_180d",
            "project_gross_yield",
        ]
    if cfg["renovation_permits"]:
        numeric += ["permits_window", "days_since_permit"]
    return numeric, categorical


# reference frame names required per feature group (see store_reference_data_gcs.py)
REFERENCE_REQUIREMENTS = {
    "project_meta": ("projects", "buildings_agg"),
    "rent_yield": ("rent_index",),
    "service_charge": ("projects", "service_charges"),
    "unit_floor": ("projects", "units"),
    "rel_floor": ("projects", "units", "buildings_agg"),
    "rent_density": ("rent_index",),
    "rent_level": ("rent_index",),
    "rent_momentum": ("rent_index",),
    "yield_matched": ("rent_index",),
    "yield_spread": ("rent_index",),
    "rent_pooling": ("rent_index",),
    "rent_anchor": ("rent_index",),
    "rent_divergence": ("rent_index",),
    "rent_new_segment": ("rent_index", "rent_weekly_stats"),
    "rent_project": ("rent_project_index",),
    "renovation_permits": ("modification_permits",),
}

# feature groups resolved inside _join_rent_grid
RENT_GRID_GROUPS = (
    "rent_yield", "rent_density", "rent_level", "rent_momentum",
    "yield_matched", "yield_spread", "rent_pooling", "rent_anchor",
    "rent_divergence", "rent_new_segment",
)


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

# Feature groups whose derivations need the date-sorted frame (all built on
# strictly-past rolling windows or shifts).
DERIVED_FEATURE_GROUPS = (
    "comps_area", "comps_project", "comps_project_windows",
    "comps_building", "price_history", "liquidity", "momentum",
    "rel_size", "comp_dispersion", "repeat_sale", "repeat_sale_adj",
    "comps_rooms", "rooms_dynamics",
)


def _past_stat(value: str, window: str, group: str, stat: str = "median") -> pl.Expr:
    """Strictly-past rolling ``stat`` of ``value`` per ``group`` (closed left).

    ``closed="left"`` excludes the current day and, together with the date
    sort, the row itself — the expression can never see its own price or
    any future information. Rows whose ``group`` is null are masked to null
    rather than pooled into one implicit null group.
    """
    rolled = getattr(pl.col(value), f"rolling_{stat}_by")(
        "date", window_size=window, closed="left"
    ).over(group)
    return pl.when(pl.col(group).is_not_null()).then(rolled)


def _backfill_optional_columns(raw: pl.DataFrame) -> pl.DataFrame:
    """Null-backfill optional columns for frames that bypassed normalization."""
    optional_sources = [
        "PARKING", "TOTAL_BUYER", "TOTAL_SELLER", "METER_SALE_PRICE",
        "PROCEDURE_EN", "PROJECT_EN", "MASTER_PROJECT_EN", "BUILDING_NAME_EN",
        "TRANSACTION_NUMBER", "USAGE_EN", "PROP_TYPE_EN", "PROJECT_NUMBER",
        "NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN",
    ]
    missing_sources = [c for c in optional_sources if c not in raw.columns]
    if missing_sources:
        return raw.with_columns(pl.lit(None).alias(c) for c in missing_sources)
    return raw


def _filter_scoreable_sales(raw: pl.DataFrame) -> pl.DataFrame:
    """Apartment Sales rows with positive price/area, market procedures only.

    Also merges case variants of project/building names (e.g. "Imperial
    Residence" vs "IMPERIAL RESIDENCE") so the encoder sees one category
    per real-world name.
    """
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
    return df.with_columns(
        pl.col(c).cast(pl.Utf8).str.to_uppercase().alias(c)
        for c in ("PROJECT_EN", "MASTER_PROJECT_EN", "BUILDING_NAME_EN")
        if c in df.columns
    )


def _parse_date_and_price(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``date`` and AED/sqft ``psf``; drop rows where either is unparseable."""
    return df.with_columns(
        pl.col("INSTANCE_DATE")
        .cast(pl.Utf8)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias("date"),
        (pl.col("TRANS_VALUE") / (pl.col("ACTUAL_AREA") * SQM_TO_SQFT)).alias("psf"),
    ).drop_nulls(["date", "psf"])


def _drop_area_mismatch_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Units guard: keep rows whose price/area agrees with METER_SALE_PRICE.

    DLD publishes METER_SALE_PRICE (AED per sqm). When present,
    TRANS_VALUE / ACTUAL_AREA must agree with it — a large mismatch means
    ACTUAL_AREA holds a plot/common area, which would fabricate an extreme
    fake discount. Rows without METER_SALE_PRICE are kept.
    """
    msp = pl.col("METER_SALE_PRICE").cast(pl.Float64, strict=False)
    psf_sqm = pl.col("TRANS_VALUE") / pl.col("ACTUAL_AREA")
    return df.filter(
        msp.is_null()
        | (msp <= 0)
        | (((psf_sqm - msp) / msp).abs() <= AREA_MISMATCH_TOLERANCE)
    )


def _add_core_columns(df: pl.DataFrame, origin: date, cfg: dict) -> pl.DataFrame:
    """Target, size/time/rooms/tier core features, and amenity columns."""
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
    return df


def _add_trailing_comps(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Strictly-past trailing comparables, liquidity, momentum, dispersion.

    Expects the frame date-sorted. Temporary key/counter columns are dropped
    before returning.
    """
    comps: list[pl.Expr] = []
    if cfg["comps_area"]:
        comps.append(_past_stat("psf", "30d", "AREA_EN").alias("area_comp_psf"))
    if cfg["comps_project"]:
        comps.append(_past_stat("psf", "60d", "PROJECT_EN").alias("project_comp_psf"))
    if cfg["comps_project_windows"]:
        comps.append(_past_stat("psf", "30d", "PROJECT_EN").alias("project_comp_psf_30"))
        comps.append(_past_stat("psf", "90d", "PROJECT_EN").alias("project_comp_psf_90"))
    if cfg["comps_building"]:
        comps.append(_past_stat("psf", "90d", "BUILDING_NAME_EN").alias("building_comp_psf"))
    if cfg["price_history"]:
        # Expanding all-past medians (window far longer than the data span).
        comps.append(_past_stat("psf", "3650d", "PROJECT_EN").alias("project_hist_psf"))
        comps.append(_past_stat("psf", "3650d", "BUILDING_NAME_EN").alias("building_hist_psf"))
    if cfg["liquidity"] or cfg["rooms_dynamics"]:
        df = df.with_columns(pl.lit(1.0).alias("_one"))
    if cfg["liquidity"]:
        comps.append(_past_stat("_one", "90d", "PROJECT_EN", "sum").alias("project_txn_90d"))
        comps.append(_past_stat("_one", "30d", "AREA_EN", "sum").alias("area_txn_30d"))
    if cfg["momentum"]:
        comps.append(
            (
                _past_stat("psf", "30d", "AREA_EN")
                / _past_stat("psf", "180d", "AREA_EN")
            ).alias("area_momentum")
        )
    if cfg["rel_size"]:
        comps.append(
            (pl.col("ACTUAL_AREA").log() - _past_stat("ACTUAL_AREA", "3650d", "PROJECT_EN").log())
            .alias("rel_log_sqft")
        )
    if cfg["comp_dispersion"]:
        comps.append(_past_stat("psf", "60d", "PROJECT_EN", "std").alias("project_comp_std"))
    if cfg["comps_rooms"] or cfg["rooms_dynamics"]:
        # Comps within the same unit type, not pooled across the project/
        # area: what 2BRs in this project sold for, not "units". Combined
        # keys go null if either part is null (concat_str propagates null),
        # so _past_stat's mask applies as usual.
        df = df.with_columns(
            pl.concat_str(
                [pl.col("PROJECT_EN"), pl.col("ROOMS_EN").cast(pl.Utf8)],
                separator="|",
            ).alias("_proj_rooms"),
            pl.concat_str(
                [pl.col("AREA_EN"), pl.col("ROOMS_EN").cast(pl.Utf8)],
                separator="|",
            ).alias("_area_rooms"),
        )
        if cfg["comps_rooms"]:
            pr_win = str(cfg.get("proj_rooms_window", "90d"))
            ar_win = str(cfg.get("area_rooms_window", "30d"))
            comps.append(_past_stat("psf", pr_win, "_proj_rooms").alias("project_rooms_comp_psf"))
            comps.append(_past_stat("psf", ar_win, "_area_rooms").alias("area_rooms_comp_psf"))
        if cfg["rooms_dynamics"]:
            comps.append(_past_stat("psf", "90d", "_proj_rooms", "std").alias("project_rooms_comp_std"))
            comps.append(_past_stat("_one", "90d", "_proj_rooms", "sum").alias("project_rooms_txn_90d"))
    if comps:
        df = df.with_columns(comps)
    for tmp in ("_one", "_proj_rooms", "_area_rooms"):
        if tmp in df.columns:
            df = df.drop(tmp)
    return df


def _add_repeat_sale_features(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Prior sale of the same pseudo-unit (building + rooms + area to 0.1 sqm).

    ``shift(1)`` over the date-sorted frame yields the unit's most recent
    PRIOR sale only — never its own row, never the future.
    """
    if not (cfg["repeat_sale"] or cfg["repeat_sale_adj"]):
        return df
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
            _past_stat("psf", "30d", "AREA_EN").alias("_area_now")
        ).with_columns(
            pl.col("_area_now").shift(1).over("_unit").alias("_area_then")
        ).with_columns(
            (
                pl.col("prior_unit_psf")
                * pl.col("_area_now")
                / pl.col("_area_then")
            ).alias("prior_unit_psf_adj")
        ).drop("_area_now", "_area_then")
    return df.drop("_unit")


def _require_reference_frames(cfg: dict, reference: dict[str, pl.DataFrame] | None) -> None:
    """Fail fast with a pointer when a needed reference frame was not passed."""
    needed = reference_needed(cfg)
    if not needed:
        return
    missing_ref = [n for n in needed if reference is None or n not in reference]
    if missing_ref:
        raise ValueError(
            f"feature_config requires reference frames {missing_ref}; "
            "load them via gcs_storage.read_reference_frames "
            "(published by store_reference_data_gcs.py)"
        )


def _join_project_reference(
    df: pl.DataFrame, cfg: dict, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """Join static project facts: developer, age, size, height, service cost."""
    if not (cfg["project_meta"] or cfg["service_charge"]):
        return df
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
        buildings_per_project = reference["buildings_agg"].select(
            pl.col("project_id").cast(pl.Int64, strict=False),
            pl.col("project_max_floors").cast(pl.Float64, strict=False),
        ).unique("project_id")
        lookup = lookup.join(buildings_per_project, on="project_id", how="left")
    if cfg["service_charge"]:
        service_charge_lookup = reference["service_charges"].select(
            pl.col("project_id").cast(pl.Int64, strict=False),
            pl.col("service_cost").cast(pl.Float64, strict=False),
        ).unique("project_id")
        lookup = lookup.join(service_charge_lookup, on="project_id", how="left")
    df = (
        df.with_columns(pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_project_number"))
        .join(lookup.rename({"project_number": "_project_number"}).drop("project_id"), on="_project_number", how="left")
        .drop("_project_number")
    )
    if cfg["project_meta"]:
        df = df.with_columns(
            ((pl.col("date") - pl.col("_completion")).dt.total_days() / 365.25)
            .alias("building_age_years")
        )
    return df.drop("_completion")


def _join_rent_grid(
    df: pl.DataFrame, cfg: dict, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """As-of join the weekly Ejari rent grid + derived rent features.

    Strictly past by construction: the grid's week-w value covers only
    contracts starting before w (rolling ``closed="left"`` upstream in
    store_reference_data_gcs), every grid-level lag or pool below looks
    further back only, and the backward as-of join attaches week <=
    transaction date. Ejari has no building key, so everything here is
    area x rooms-band granularity.
    """
    if not any(cfg[g] for g in RENT_GRID_GROUPS):
        return df
    if (cfg["yield_matched"] or cfg["yield_spread"]) and not cfg["comps_rooms"]:
        raise ValueError("yield_matched/yield_spread require comps_rooms enabled")
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
    need_city = cfg["yield_spread"] or cfg["rent_pooling"] or cfg["rent_anchor"]
    grid = (
        reference["rent_index"]
        .select(
            pl.col("AREA_EN").cast(pl.Utf8),
            pl.col("rooms_band").cast(pl.Utf8).alias("_rooms_band"),
            pl.col("week").cast(pl.Date),
            pl.col("area_rent_psf_180d").cast(pl.Float64, strict=False),
            pl.col("rent_contracts_180d").cast(pl.Float64, strict=False),
        )
        .drop_nulls(["AREA_EN", "_rooms_band", "week"])
        .sort("AREA_EN", "_rooms_band", "week")
    )
    if cfg["rent_momentum"] or cfg["rent_divergence"]:
        # 13 grid steps ~ 91 days; a sparse cell that skips weeks lags
        # further into the past, never forward.
        prev = pl.col("area_rent_psf_180d").shift(13).over("AREA_EN", "_rooms_band")
        grid = grid.with_columns(
            pl.when(prev > 0)
            .then(pl.col("area_rent_psf_180d") / prev)
            .alias("rent_mom_91d")
        )
    if need_city:
        # Contracts-weighted city rent per band-week: quantiles don't
        # compose across districts, but a weighted mean of medians is a
        # stable city-level location for pooling and the yield spread.
        city = grid.group_by("_rooms_band", "week").agg(
            (
                (pl.col("area_rent_psf_180d") * pl.col("rent_contracts_180d")).sum()
                / pl.col("rent_contracts_180d").sum()
            ).alias("_city_band_rent")
        )
        grid = grid.join(city, on=["_rooms_band", "week"], how="left")
        if cfg["rent_pooling"] or cfg["rent_anchor"]:
            k = float(cfg.get("rent_pool_k", 25))
            n = pl.col("rent_contracts_180d")
            grid = grid.with_columns(
                ((n * pl.col("area_rent_psf_180d") + k * pl.col("_city_band_rent")) / (n + k))
                .alias("rent_psf_shrunk")
            )
    df = df.with_columns(band.alias("_rooms_band")).sort("date")
    df = df.join_asof(
        grid.sort("week"),
        left_on="date",
        right_on="week",
        by=["AREA_EN", "_rooms_band"],
        strategy="backward",
    ).drop("week")

    if cfg["rent_new_segment"]:
        # Renewals are RERA-capped and lag the market; the "new" segment of
        # the weekly stats is the leading series. Weekly medians are noisy,
        # so smooth with a strictly-past 91d rolling median per cell.
        new_grid = (
            reference["rent_weekly_stats"]
            .filter((pl.col("segment") == "new") & (pl.col("rooms_band") != "All"))
            .select(
                pl.col("AREA_EN").cast(pl.Utf8),
                pl.col("rooms_band").cast(pl.Utf8).alias("_rooms_band"),
                pl.col("week").cast(pl.Date),
                pl.col("median").cast(pl.Float64, strict=False).alias("_new_wk"),
            )
            .drop_nulls(["AREA_EN", "_rooms_band", "week", "_new_wk"])
            .sort("week")
            .with_columns(
                pl.col("_new_wk")
                .rolling_median_by("week", window_size="91d", closed="left")
                .over("AREA_EN", "_rooms_band")
                .alias("rent_new_psf_13w")
            )
            .drop("_new_wk")
        )
        df = df.join_asof(
            new_grid,
            left_on="date",
            right_on="week",
            by=["AREA_EN", "_rooms_band"],
            strategy="backward",
        ).drop("week")

    if cfg["yield_spread"] or cfg["rent_anchor"]:
        # Trailing city-wide sale PSF per rooms band, strictly past — the
        # denominator that turns the city band rent into a cap-rate level.
        df = df.with_columns(_past_stat("psf", "60d", "_rooms_band").alias("_band_price"))

    exprs: list[pl.Expr] = []
    matched = pl.when(pl.col("area_rooms_comp_psf") > 0).then(
        pl.col("area_rent_psf_180d") / pl.col("area_rooms_comp_psf")
    )
    city_yield = pl.when(pl.col("_band_price") > 0).then(
        pl.col("_city_band_rent") / pl.col("_band_price")
    ) if (cfg["yield_spread"] or cfg["rent_anchor"]) else None
    if cfg["rent_level"]:
        exprs.append(pl.col("area_rent_psf_180d").alias("rent_psf_180d"))
    if cfg["yield_matched"]:
        exprs.append(matched.alias("matched_gross_yield"))
    if cfg["yield_spread"]:
        exprs.append((matched - city_yield).alias("yield_spread"))
    if cfg["rent_anchor"]:
        src = "rent_psf_shrunk" if cfg["rent_pooling"] else "area_rent_psf_180d"
        exprs.append(
            pl.when(city_yield > 0).then(pl.col(src) / city_yield).alias("rent_implied_psf")
        )
    if cfg["rent_divergence"]:
        area_mom = _past_stat("psf", "30d", "AREA_EN") / _past_stat("psf", "180d", "AREA_EN")
        exprs.append((pl.col("rent_mom_91d") - area_mom).alias("rent_price_div"))
    if cfg["rent_new_segment"]:
        exprs.append(
            pl.when(pl.col("area_rent_psf_180d") > 0)
            .then(pl.col("rent_new_psf_13w") / pl.col("area_rent_psf_180d"))
            .alias("rent_new_premium")
        )
    if cfg["rent_yield"]:
        denom_cols = [c for c in ("project_comp_psf", "area_comp_psf") if c in df.columns]
        if denom_cols:
            exprs.append(
                (pl.col("area_rent_psf_180d") / pl.coalesce([pl.col(c) for c in denom_cols]))
                .alias("implied_gross_yield")
            )
        else:
            exprs.append(pl.lit(None, dtype=pl.Float64).alias("implied_gross_yield"))
    if exprs:
        df = df.with_columns(exprs)
    return df.drop(
        c for c in ("_rooms_band", "_city_band_rent", "_band_price") if c in df.columns
    )


def _join_project_rent(
    df: pl.DataFrame, cfg: dict, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """As-of join the project-linked rent index (strictly past by construction).

    The index is built from Ejari contracts resolved to their project via the
    name join + layout fingerprint (store_reference_data_gcs). Joining on
    PROJECT_NUMBER puts rents and sale comps on the SAME stock, so
    ``project_gross_yield`` is the quality-matched ratio the district-level
    campaign features could not build.
    """
    if not cfg["rent_project"]:
        return df
    grid = (
        reference["rent_project_index"]
        .select(
            pl.col("project_number").cast(pl.Int64, strict=False).alias("_pn"),
            pl.col("week").cast(pl.Date),
            pl.col("project_rent_psf_180d").cast(pl.Float64, strict=False),
            pl.col("project_rent_contracts_180d").cast(pl.Float64, strict=False),
        )
        .drop_nulls(["_pn", "week"])
        .sort("week")
    )
    df = df.with_columns(
        pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_pn")
    ).sort("date")
    df = df.join_asof(
        grid, left_on="date", right_on="week", by=["_pn"], strategy="backward"
    ).drop("_pn", "week")
    if "project_comp_psf" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("project_comp_psf") > 0)
            .then(pl.col("project_rent_psf_180d") / pl.col("project_comp_psf"))
            .alias("project_gross_yield")
        )
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("project_gross_yield")
        )
    return df


def _join_permits(
    df: pl.DataFrame, cfg: dict, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """Renovation-permit history on the sale's project, strictly past.

    Dubai Municipality adjustment/addition permits are public events dated
    before the sale; two backward as-of joins against the cumulative event
    count give the trailing permit count (window from
    ``permit_window_days``, default 5 years) and the recency of the last
    permit — a building-investment proxy for unit condition. A project
    absent from the permits data means no adjustment permits: count 0, no
    recency.
    """
    if not cfg["renovation_permits"]:
        return df
    events = (
        reference["modification_permits"]
        .select(
            pl.col("project_number").cast(pl.Int64, strict=False).alias("_pn"),
            pl.col("permit_date").cast(pl.Date),
        )
        .drop_nulls()
        .unique()
        .sort("permit_date")
        .with_columns(pl.col("permit_date").cum_count().over("_pn").alias("_cum"))
    )
    df = df.with_columns(
        pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_pn")
    ).sort("date")
    df = df.join_asof(
        events, left_on="date", right_on="permit_date", by=["_pn"], strategy="backward"
    ).rename({"_cum": "_cum_now"})
    window_days = int(cfg.get("permit_window_days", 1826))
    df = df.with_columns(
        (pl.col("date") - pl.col("permit_date")).dt.total_days()
        .cast(pl.Float64).alias("days_since_permit"),
        (pl.col("date") - pl.duration(days=window_days)).alias("_dw"),
    ).drop("permit_date")
    df = df.join_asof(
        events, left_on="_dw", right_on="permit_date", by=["_pn"], strategy="backward"
    ).rename({"_cum": "_cum_window_ago"})
    return df.with_columns(
        (pl.col("_cum_now").fill_null(0) - pl.col("_cum_window_ago").fill_null(0))
        .cast(pl.Float64).alias("permits_window")
    ).drop("_pn", "_dw", "_cum_now", "_cum_window_ago", "permit_date")


def _join_units_registry(
    df: pl.DataFrame, cfg: dict, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """Match the DLD units registry for floor/balcony facts, plus rel_floor.

    Transactions carry no unit key, so we match project (via
    project_number -> project_id) + exact registered area: where that layout
    is unique in the project, unit_floor is the true floor; where layouts
    stack, only the layout's floor distribution and balcony area attach.
    """
    if cfg["unit_floor"]:
        bridge = (
            reference["projects"]
            .select(
                pl.col("project_number").cast(pl.Int64, strict=False).alias("_project_number"),
                pl.col("project_id").cast(pl.Int64, strict=False).alias("_project_id"),
            )
            .drop_nulls()
            .unique("_project_number")
        )
        # Area-key precision is configurable: 2dp = strict (fewer, surer
        # matches), 1dp/0dp = looser (more matches, blurrier layouts).
        akey_round = int(cfg.get("unit_floor_round", 2))
        units = (
            reference["units"]
            .select(
                pl.col("project_id").cast(pl.Int64, strict=False).alias("_project_id"),
                pl.col("floor_num").cast(pl.Float64, strict=False),
                pl.col("actual_area").cast(pl.Float64, strict=False)
                .round(akey_round).alias("_area_key"),
                pl.col("unit_balcony_area").cast(pl.Float64, strict=False),
            )
            .drop_nulls(["_project_id", "_area_key"])
        )
        layouts = (
            units.group_by("_project_id", "_area_key")
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
                pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_project_number"),
                pl.col("ACTUAL_AREA").cast(pl.Float64).round(akey_round).alias("_area_key"),
            )
            .join(bridge, on="_project_number", how="left")
            .join(layouts, on=["_project_id", "_area_key"], how="left")
            .drop("_project_number", "_project_id", "_area_key")
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
    return df


def _select_output_columns(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Fill categoricals, add project_txn_total, project to the output columns."""
    _, categorical = feature_columns(cfg)
    missing = [c for c in categorical if c not in df.columns]
    if missing:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing)
    df = df.with_columns(
        pl.col(c).cast(pl.Utf8).fill_null("UNKNOWN").alias(c) for c in categorical
    )
    # Full-history project size, carried through for flag_distress: the
    # illiquidity distress signal must count a project's sales over the
    # whole snapshot, not over whatever scoring window is later selected.
    df = df.with_columns(pl.len().over("PROJECT_EN").alias("project_txn_total"))

    keep = ["date", "psf", "log_psf", "project_txn_total"]
    numeric, _ = feature_columns(cfg)
    keep += numeric + categorical
    keep += [c for c in PASSTHROUGH_COLUMNS if c in df.columns and c not in keep]
    return df.select(keep).sort("date")


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

    With ``"data_cleaning": True`` in the config, :mod:`data_cleaning`
    replaces the METER_SALE_PRICE guard: digit-shift typos are repaired in
    place, and non-market rows (bulk allocations, suspected token
    transfers, partial-ownership shares) are excluded here and surfaced
    separately in the UI. ``dq_rule``/``dq_action`` ride along as
    passthrough columns.

    Deliberately does NOT trim PSF outliers: the deepest discounts are the
    distressed assets this model exists to surface. Apply :func:`trim_psf`
    to the result before TRAINING so the target stays robust.

    ``date_origin`` anchors ``days_since_start``; pass the training origin
    when scoring new data so the trend feature stays aligned.

    Each stage lives in its own helper; this function only sequences them.
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}

    raw = _backfill_optional_columns(raw)
    df = _filter_scoreable_sales(raw)
    if df.is_empty():
        return df
    if cfg.get("data_cleaning"):
        # Repair digit-shift typos and keep only market-price rows. The
        # METER_SALE_PRICE guard below must be skipped: repairs re-derive
        # that field, and the cleaner already quarantines basis mismatches.
        cleaned, _ = data_cleaning.clean_transactions(df, reference=reference)
        df = data_cleaning.kept_rows(cleaned)
        if df.is_empty():
            return df
        df = _parse_date_and_price(df)
    else:
        df = _parse_date_and_price(df)
        df = _drop_area_mismatch_rows(df)
    if df.is_empty():
        return df

    origin = date_origin or df.select(pl.col("date").min()).item()
    df = _add_core_columns(df, origin, cfg)

    if any(cfg[g] for g in DERIVED_FEATURE_GROUPS):
        df = df.sort("date")
        df = _add_trailing_comps(df, cfg)
        df = _add_repeat_sale_features(df, cfg)

    _require_reference_frames(cfg, reference)
    df = _join_project_reference(df, cfg, reference)
    df = _join_rent_grid(df, cfg, reference)
    df = _join_project_rent(df, cfg, reference)
    df = _join_permits(df, cfg, reference)
    df = _join_units_registry(df, cfg, reference)
    return _select_output_columns(df, cfg)


def excluded_suspicious_sales(
    raw: pl.DataFrame, reference: dict[str, pl.DataFrame] | None = None
) -> pl.DataFrame:
    """Sales the cleaning step routes to human review instead of the model.

    Runs the same scoreable-sales filter + :mod:`data_cleaning` pass as
    ``feature_engineering`` with ``"data_cleaning": True`` and returns the
    ``review_only`` rows (bulk allocations, suspected token transfers,
    partial-ownership shares) with their ``dq_rule`` labels — real
    transfers whose registered price is not a standalone market price.
    """
    df = _filter_scoreable_sales(_backfill_optional_columns(raw))
    if df.is_empty():
        return df
    cleaned, _ = data_cleaning.clean_transactions(df, reference=reference)
    return data_cleaning.review_rows(cleaned)


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

FLAG_SPREAD_THRESHOLD = -0.15  # default dashboard flag threshold


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-fold accuracy + tail metrics from log-price truths and predictions."""
    resid = y_true - y_pred  # log(actual) - log(fair value)
    spread = np.expm1(resid)  # actual / fair value - 1
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "medape": float(np.median(np.abs(spread))),
        "mae_log": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # Tail metrics: flags live in the error tail, which the median cannot
        # see. p90_ape = worst-decile error; flag_prop = share of ordinary
        # validation sales the model's error alone pushes below the flag
        # threshold (one-sided: overestimated fair value = false bargain).
        "p90_ape": float(np.percentile(np.abs(spread), 90)),
        "flag_prop": float(np.mean(spread <= FLAG_SPREAD_THRESHOLD)),
    }


def _make_model(
    model_params: dict | None, categorical_idx: list[int], random_state: int
) -> HistGradientBoostingRegressor:
    """Configured HGB regressor (defaults merged under ``model_params``)."""
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
    for key in ("medape", "mae_log", "r2", "p90_ape", "flag_prop"):
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

BUNDLE_VERSION = 2  # v2: feature-config key "te_hist" renamed to "price_history"

# Old bundles store feature configs under retired key names. Migrate on load
# so a rollback or stale cached bundle keeps working instead of silently
# dropping a feature group (which changes the matrix width and crashes
# predict — or worse, mispredicts if the column count happens to match).
CONFIG_KEY_MIGRATIONS = {"te_hist": "price_history"}


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
    feature_config = {
        CONFIG_KEY_MIGRATIONS.get(key, key): value
        for key, value in (payload["feature_config"] or {}).items()
    }
    result = FairValueResult(
        model=payload["model"],
        encoders=payload["encoders"],
        feature_names=payload["feature_names"],
        categorical_idx=payload["categorical_idx"],
        feature_config=feature_config,
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
    X, _, built_names, _ = to_matrix(df, result.encoders, result.feature_config)
    if result.feature_names and built_names != result.feature_names:
        raise ValueError(
            "Feature mismatch between this code and the model bundle: built "
            f"{built_names} but the bundle was trained on {result.feature_names}. "
            "Re-publish the bundle with train_fair_value.py (or update "
            "CONFIG_KEY_MIGRATIONS if a feature-group key was renamed)."
        )
    pred_log_psf = result.model.predict(X)
    return df.with_columns(
        pl.Series("pred_psf", np.exp(pred_log_psf)),
    ).with_columns(
        (pl.col("pred_psf") * pl.col("ACTUAL_AREA") * SQM_TO_SQFT).alias("fair_value_aed"),
    ).with_columns(
        (pl.col("TRANS_VALUE") / pl.col("fair_value_aed") - 1).alias("spread_pct"),
    )


# Expected model error by comparable-data segment, used to standardize the
# spread into a signal strength. Measured on the floor-champion's last four
# out-of-time CV folds (2026-07-05): established MedAPE 3.5%, cold-start
# (no trailing project comp) 6.4% with a 2.2x false-flag rate. Slightly
# rounded up for conservatism.
EXPECTED_ERR_ESTABLISHED = 0.04
EXPECTED_ERR_COLD_START = 0.065


def flag_distress(
    scored: pl.DataFrame,
    spread_threshold: float = FLAG_SPREAD_THRESHOLD,
    deep_discount: float = -0.25,
    min_project_txns: int = 8,
    expected_err_established: float = EXPECTED_ERR_ESTABLISHED,
    expected_err_cold: float = EXPECTED_ERR_COLD_START,
) -> pl.DataFrame:
    """Flag below-fair-value rows and corroborated distressed-asset candidates.

    ``distressed`` requires the spread to be at/below ``spread_threshold``
    AND at least one corroborating signal that is INDEPENDENT of the model
    residual (forced-sale procedure, illiquid project, multiple sellers) —
    a lone model miss, however large, is never enough. ``sig_deep_discount``
    is annotated for display but deliberately does not count, since it is
    derived from the same residual as ``below_fair_value``.

    ``signal_strength`` standardizes the discount by the model's expected
    error for the row's segment (established vs cold-start): a -15% spread
    is ~3.6x the typical error where comps are rich, but barely 1.4x for a
    cold-start sale — the same discount, very different evidence.
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
    cold = (
        pl.col("project_comp_psf").is_null()
        if "project_comp_psf" in scored.columns
        else pl.lit(False)
    )
    expected_err = (
        pl.when(cold)
        .then(pl.lit(expected_err_cold))
        .otherwise(pl.lit(expected_err_established))
    )
    df = scored.with_columns(
        (pl.col("spread_pct") <= spread_threshold).alias("below_fair_value"),
        procedure.alias("sig_procedure"),
        (pl.col("spread_pct") <= deep_discount).alias("sig_deep_discount"),
        # Illiquidity must be judged over the FULL history: feature_engineering
        # carries project_txn_total for exactly this. Counting rows of `scored`
        # would make the signal depend on the user's scoring window (a liquid
        # project looks illiquid in a 30-day slice). The fallback only exists
        # for frames that predate the passthrough column.
        (
            (pl.col("project_txn_total") < min_project_txns)
            if "project_txn_total" in scored.columns
            else (pl.len().over("PROJECT_EN") < min_project_txns)
        ).alias("sig_illiquid_project"),
        multi_seller.alias("sig_multi_seller"),
        (-pl.col("spread_pct") / expected_err).alias("signal_strength"),
        cold.alias("cold_start"),
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
