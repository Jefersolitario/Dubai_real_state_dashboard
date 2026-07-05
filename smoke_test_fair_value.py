"""Offline end-to-end check of the fair-value model on synthetic data.

Builds a synthetic transactions frame that mimics the normalized DDA
schema, plants a known share of deliberately underpriced rows, and checks
that feature engineering, TimeSeriesSplit cross-validation, scoring, and
distress flagging behave: metrics are finite, the spread math round-trips,
planted bargains dominate the flagged set, and unseen categories score
without errors. No Streamlit, no network.
"""

import sys
from datetime import date, timedelta

import numpy as np
import polars as pl

from dashboard_constants import SQM_TO_SQFT
from fair_value_model import (
    cross_validate,
    export_bundle,
    feature_engineering,
    flag_distress,
    load_bundle,
    score_transactions,
    train_fair_value_model,
    trim_psf,
)

N_ROWS = 6000
PLANT_SHARE = 0.05
PLANT_DISCOUNT = 0.25
SEED = 7

AREAS = {
    "MARSA DUBAI": 2200.0,
    "BUSINESS BAY": 1900.0,
    "AL JADAF": 1400.0,
    "AL BARSHA SOUTH FOURTH": 1100.0,
    "NADD HESSA": 800.0,
}
PROJECT_FACTORS = {"P1": 0.9, "P2": 1.0, "P3": 1.1, "P4": 1.25}
BUILDING_FACTORS = {"B1": 0.96, "B2": 1.0, "B3": 1.05}
ROOM_FACTORS = {"Studio": 1.08, "1 B/R": 1.0, "2 B/R": 0.95, "3 B/R": 0.92}


def synthetic_frame() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(SEED)
    start = date(2024, 7, 1)

    areas = rng.choice(list(AREAS), size=N_ROWS)
    projects = rng.choice(list(PROJECT_FACTORS), size=N_ROWS)
    buildings = rng.choice(list(BUILDING_FACTORS), size=N_ROWS)
    rooms = rng.choice(list(ROOM_FACTORS), size=N_ROWS)
    day_offsets = rng.integers(0, 730, size=N_ROWS)
    sqm = rng.uniform(35.0, 220.0, size=N_ROWS)
    offplan = rng.choice(["Off-Plan", "Ready"], size=N_ROWS)

    base = np.array([AREAS[a] for a in areas])
    proj = np.array([PROJECT_FACTORS[p] for p in projects])
    bldg = np.array([BUILDING_FACTORS[b] for b in buildings])
    room = np.array([ROOM_FACTORS[r] for r in rooms])
    trend = 1.0 + 0.15 * day_offsets / 730.0  # gentle market appreciation
    noise = np.exp(rng.normal(0.0, 0.05, size=N_ROWS))
    psf = base * proj * bldg * room * trend * noise

    planted = rng.random(N_ROWS) < PLANT_SHARE
    psf = np.where(planted, psf * (1.0 - PLANT_DISCOUNT), psf)

    procedures = np.where(
        planted & (rng.random(N_ROWS) < 0.5), "Sell By Court Order", "Sell"
    )

    frame = pl.DataFrame(
        {
            "TRANSACTION_NUMBER": [f"TX-{i:06d}" for i in range(N_ROWS)],
            "INSTANCE_DATE": [
                (start + timedelta(days=int(d))).isoformat() for d in day_offsets
            ],
            "GROUP_EN": ["Sales"] * N_ROWS,
            "PROCEDURE_EN": procedures,
            "IS_OFFPLAN_EN": offplan,
            "USAGE_EN": ["Residential"] * N_ROWS,
            "AREA_EN": areas,
            "PROP_TYPE_EN": ["Unit"] * N_ROWS,
            "PROP_SB_TYPE_EN": ["Flat"] * N_ROWS,
            "TRANS_VALUE": psf * sqm * SQM_TO_SQFT,
            "ACTUAL_AREA": sqm,
            "PROCEDURE_AREA": sqm,
            "ROOMS_EN": rooms,
            "PARKING": rng.choice(["1", "2", None], size=N_ROWS).tolist(),
            "NEAREST_METRO_EN": ["Metro A"] * N_ROWS,
            "NEAREST_MALL_EN": ["Mall A"] * N_ROWS,
            "NEAREST_LANDMARK_EN": ["Landmark A"] * N_ROWS,
            "TOTAL_BUYER": rng.integers(1, 3, size=N_ROWS).astype(float),
            "TOTAL_SELLER": rng.integers(1, 3, size=N_ROWS).astype(float),
            "PROJECT_EN": projects,
            "MASTER_PROJECT_EN": projects,
            "BUILDING_NAME_EN": [f"{p}-{b}" for p, b in zip(projects, buildings)],
            "METER_SALE_PRICE": psf * SQM_TO_SQFT,
            "is_planted": planted,
        }
    )

    # A few mortgage rows that must be excluded from training/scoring.
    mortgage = frame.head(50).with_columns(pl.lit("Mortgages").alias("GROUP_EN"))
    return pl.concat([frame, mortgage]), planted


def check(condition: bool, label: str) -> bool:
    print(f"{'[ok]  ' if condition else '[FAIL]'} {label}")
    return condition


def main() -> int:
    raw, planted = synthetic_frame()
    ok = True

    feats = feature_engineering(raw)
    ok &= check(feats.height > 0, f"feature_engineering produced {feats.height} rows")
    ok &= check(
        feats.height <= N_ROWS,
        "mortgage rows excluded from the model frame",
    )

    train_feats = trim_psf(feats)
    cv = cross_validate(train_feats, n_splits=10)
    ok &= check(
        all(np.isfinite(cv[k]) for k in ("medape_mean", "mae_log_mean", "r2_mean")),
        f"CV metrics finite (MedAPE {cv['medape_mean']:.3%} ± {cv['medape_std']:.3%}, "
        f"R² {cv['r2_mean']:.3f}, {cv['n_splits']} folds)",
    )
    ok &= check(cv["medape_mean"] < 0.15, "CV MedAPE under 15% on synthetic data")

    result = train_fair_value_model(train_feats)
    scored = score_transactions(result, feats)  # scoring covers the untrimmed tail

    roundtrip = scored.select(
        ((pl.col("TRANS_VALUE") / pl.col("fair_value_aed") - 1) - pl.col("spread_pct"))
        .abs()
        .max()
    ).item()
    ok &= check(roundtrip < 1e-9, f"spread math round-trips (max err {roundtrip:.2e})")

    flagged = flag_distress(scored, spread_threshold=-0.15)
    plants = raw.select("TRANSACTION_NUMBER", "is_planted").unique("TRANSACTION_NUMBER")
    flagged = flagged.join(plants, on="TRANSACTION_NUMBER", how="left").with_columns(
        pl.col("is_planted").fill_null(False)
    )
    below = flagged.filter(pl.col("below_fair_value"))
    n_planted_total = flagged.filter(pl.col("is_planted")).height
    precision = below.filter(pl.col("is_planted")).height / max(below.height, 1)
    recall = below.filter(pl.col("is_planted")).height / max(n_planted_total, 1)
    ok &= check(
        precision > 0.6,
        f"planted bargains dominate the flagged set (precision {precision:.0%}, n={below.height})",
    )
    ok &= check(recall > 0.5, f"most planted bargains are caught (recall {recall:.0%})")

    court = flagged.filter(
        pl.col("is_planted") & (pl.col("PROCEDURE_EN") == "Sell By Court Order")
    )
    distressed_share = court.filter(pl.col("distressed")).height / max(court.height, 1)
    ok &= check(
        distressed_share > 0.5,
        f"court-order bargains labelled distressed ({distressed_share:.0%} of {court.height})",
    )

    unseen = raw.head(200).with_columns(
        pl.lit("BRAND NEW DISTRICT").alias("AREA_EN"),
        pl.lit("Unseen Project").alias("PROJECT_EN"),
    )
    unseen_feats = feature_engineering(unseen, date_origin=result.date_origin)
    unseen_scored = score_transactions(result, unseen_feats)
    finite = unseen_scored.select(pl.col("pred_psf").is_finite().all()).item()
    ok &= check(bool(finite), f"unseen categories score finitely ({unseen_feats.height} rows)")

    bundle = export_bundle(result, extra={"source": "smoke"})
    loaded, _meta = load_bundle(bundle)
    rescored = score_transactions(loaded, feats)
    identical = np.allclose(
        scored.get_column("pred_psf").to_numpy(),
        rescored.get_column("pred_psf").to_numpy(),
    )
    ok &= check(
        identical,
        f"model bundle round-trips with identical predictions ({len(bundle) / 1e6:.1f} MB)",
    )

    top = result.importances.head(3).get_column("feature").to_list()
    print(f"       top importances: {top}")

    ok &= check_reference_groups(raw)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def check_reference_groups(raw: pl.DataFrame) -> bool:
    """unit_floor join against tiny synthetic reference frames."""
    projects = pl.DataFrame({"project_number": [101, 102], "project_id": [9001, 9002]})
    units = pl.DataFrame({
        "project_id": [9001, 9001, 9001, 9002, 9002],
        "building_number": ["1"] * 5,
        "floor_num": [7.0, 3.0, 12.0, 5.0, 9.0],
        "actual_area": [88.55, 120.00, 120.00, 60.25, 60.25],
        "rooms_en": ["1 B/R", "2 B/R", "2 B/R", "Studio", "Studio"],
        "unit_balcony_area": [10.0, 20.0, 22.0, 5.0, 6.0],
    })
    sub = raw.head(2000).with_columns(
        pl.Series("PROJECT_NUMBER", [101, 102] * 1000),
        pl.Series("ACTUAL_AREA", [88.55, 60.25] * 1000),
    )
    feats = feature_engineering(
        sub,
        {"project": True, "amenity": True, "unit_floor": True},
        reference={"projects": projects, "units": units},
    )
    uniq = feats.filter(pl.col("layout_units") == 1)
    stacked = feats.filter(pl.col("layout_units") == 2)
    good = (
        uniq.height > 0
        and stacked.height > 0
        and uniq["unit_floor"].drop_nulls().unique().to_list() == [7.0]
        and stacked["unit_floor"].is_null().all()
        and abs(stacked["layout_floor_mean"][0] - 7.0) < 1e-9
    )
    return check(good, "unit_floor reference join behaves as designed")


if __name__ == "__main__":
    sys.exit(main())
