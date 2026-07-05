"""Outcome backtest for the fair-value flags (distress-claim validation).

Hypothesis: if a flagged below-fair-value sale is a genuine bargain, the buyer
acquired real equity — when the same unit resells, its realized appreciation
should beat the area market's over the same period.

Method (see distress_validation_report.md):
- WALK-FORWARD scoring (hindsight-bias guard): each quarterly block of entries
  is flagged by a model trained only on data strictly before that block, so no
  entry decision uses a model that saw the entry or its resale in training.
- Pair each sale with the SAME pseudo-unit's next sale (building + rooms +
  area to 0.1 sqm — the repeat-sale key the model itself uses).
- Excess return = (resale PSF / entry PSF) − (area market PSF at resale /
  area market PSF at entry), the area market being the trailing 30-day median
  of the unit's own district.
- Compare flagged entries vs near-fair controls, raw and matched within
  (area × rooms band × entry month) cells; bootstrap CIs; signal-strength
  decile calibration; forced-sale lift over the full history.

Usage:
    python backtest_flags.py            # full run, writes charts + JSON
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from datetime import date

from dda_api import normalize_dld_transactions
from fair_value_model import (
    DISTRESS_PROCEDURE_PATTERN,
    feature_engineering,
    flag_distress,
    load_shipping_config,
    reference_needed,
    score_transactions,
    train_fair_value_model,
    trim_psf,
)
from gcs_storage import load_local_secrets, read_reference_frames
from store_dld_transactions_gcs import dedupe_snapshot

SCRATCH = Path(
    "/tmp/claude-0/-home-user-Dubai-real-state-dashboard/"
    "21fc9bb3-8870-5396-9c58-82e3bfe62b79/scratchpad"
)
SNAPSHOT_PARQUET = SCRATCH / "dld_24m.parquet"

FLAG_THRESHOLD = -0.15          # entry cohort: spread at/below this
CONTROL_BAND = 0.05             # controls: |spread| <= this (near fair value)
MIN_HOLDING_DAYS = 30           # resales sooner are likely assignments/artifacts
MIN_RUNWAY_DAYS = 180           # entries need >= this much future data
WINSOR_QUANTILES = (0.01, 0.99)
N_BOOTSTRAP = 2000
HORIZON_BUCKETS = [(30, 183), (30, 365), (30, 550)]


# Walk-forward cutoffs: each model trains on data strictly BEFORE the cutoff
# and flags only the following quarter, so no entry decision ever uses a model
# that saw the entry (or its resale) in training — the hindsight-bias guard.
WALK_FORWARD_BLOCKS = [
    (date(2025, 1, 1), date(2025, 4, 1)),
    (date(2025, 4, 1), date(2025, 7, 1)),
    (date(2025, 7, 1), date(2025, 10, 1)),
    (date(2025, 10, 1), date(2026, 1, 1)),
]
# Regime diagnostic: one model frozen at Feb 2026 scores the falling market
# Feb-Jul 2026 fully out-of-sample.
REGIME_CUTOFF = date(2026, 2, 1)


def build_feature_history() -> pl.DataFrame:
    """Full-history untrimmed feature frame via the exact app path."""
    secrets = load_local_secrets()
    feature_config, _model_params = load_shipping_config()
    snapshot = dedupe_snapshot(
        normalize_dld_transactions(pl.read_parquet(SNAPSHOT_PARQUET))
    )
    reference = read_reference_frames(secrets, reference_needed(feature_config))
    return feature_engineering(snapshot, feature_config, reference=reference)


def walk_forward_scored(features: pl.DataFrame) -> pl.DataFrame:
    """Out-of-sample spreads: per block, train < cutoff, score [cutoff, end)."""
    feature_config, model_params = load_shipping_config()
    blocks: list[pl.DataFrame] = []
    for cutoff, block_end in WALK_FORWARD_BLOCKS:
        train_frame = trim_psf(features.filter(pl.col("date") < cutoff))
        model = train_fair_value_model(
            train_frame, feature_config, model_params, run_cv=False
        )
        block = features.filter(
            (pl.col("date") >= cutoff) & (pl.col("date") < block_end)
        )
        scored_block = score_transactions(model, block)
        blocks.append(scored_block)
        print(f"walk-forward {cutoff}: trained on {train_frame.height:,}, "
              f"scored {block.height:,}", flush=True)
    return flag_distress(pl.concat(blocks, how="vertical"))


def regime_drift_diagnostic(features: pl.DataFrame) -> list[dict]:
    """Median spread + flag rate by month, model frozen before the downturn.

    A model trained only through REGIME_CUTOFF scores Feb-Jul 2026 fully
    out-of-sample; if fair values lag a falling market, the monthly median
    spread drifts negative and the flag rate inflates.
    """
    feature_config, model_params = load_shipping_config()
    train_frame = trim_psf(features.filter(pl.col("date") < REGIME_CUTOFF))
    model = train_fair_value_model(
        train_frame, feature_config, model_params, run_cv=False
    )
    scored = score_transactions(
        model, features.filter(pl.col("date") >= REGIME_CUTOFF)
    )
    monthly = (
        scored.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("n"),
            pl.col("spread_pct").median().alias("median_spread"),
            (pl.col("spread_pct") <= FLAG_THRESHOLD).mean().alias("flag_rate"),
        )
        .sort("month")
    )
    return monthly.to_dicts()


def build_resale_pairs_from(
    features: pl.DataFrame, walk_forward: pl.DataFrame
) -> pl.DataFrame:
    """One row per (entry sale -> same pseudo-unit's next sale) pair.

    Resale outcomes and the district index come from the FULL history
    (realized facts); the entry's spread/flag columns join in from the
    walk-forward frame, so cohort membership is strictly out-of-sample.
    Adds the entry/resale area-market levels (trailing 30-day district
    median — measurement, not modelling, so the current day is included)
    and the unit's excess return over its own district.
    """
    scored = features.join(
        walk_forward.select(
            "TRANSACTION_NUMBER", "spread_pct", "signal_strength", "cold_start"
        ).unique("TRANSACTION_NUMBER"),
        on="TRANSACTION_NUMBER",
        how="left",
    )
    pseudo_unit = pl.concat_str(
        pl.col("BUILDING_NAME_EN").fill_null("?"),
        pl.col("ROOMS_EN").cast(pl.Utf8).fill_null("?"),
        (pl.col("ACTUAL_AREA") * 10).round(0).cast(pl.Int64).cast(pl.Utf8),
        separator="|",
    )
    df = (
        scored.sort("date")
        .with_columns(
            pseudo_unit.alias("unit_key"),
            pl.col("psf")
            .rolling_median_by("date", window_size="30d", closed="both")
            .over("AREA_EN")
            .alias("area_market_psf"),
        )
        .sort(["unit_key", "date"])
        .with_columns(
            pl.col("date").shift(-1).over("unit_key").alias("resale_date"),
            pl.col("psf").shift(-1).over("unit_key").alias("resale_psf"),
            pl.col("area_market_psf").shift(-1).over("unit_key").alias("resale_area_market_psf"),
        )
    )
    pairs = df.filter(
        pl.col("BUILDING_NAME_EN").is_not_null()
        & pl.col("spread_pct").is_not_null()
        & pl.col("resale_date").is_not_null()
        & ((pl.col("resale_date") - pl.col("date")).dt.total_days() >= MIN_HOLDING_DAYS)
    ).with_columns(
        (pl.col("resale_date") - pl.col("date")).dt.total_days().alias("holding_days"),
        (
            (pl.col("resale_psf") / pl.col("psf"))
            - (pl.col("resale_area_market_psf") / pl.col("area_market_psf"))
        ).alias("excess_return"),
        pl.col("date").dt.strftime("%Y-%m").alias("entry_month"),
        pl.col("ROOMS_EN").cast(pl.Utf8).fill_null("?").alias("rooms_band"),
    )
    lo, hi = pairs.select(
        pl.col("excess_return").quantile(WINSOR_QUANTILES[0]).alias("lo"),
        pl.col("excess_return").quantile(WINSOR_QUANTILES[1]).alias("hi"),
    ).row(0)
    return pairs.with_columns(pl.col("excess_return").clip(lo, hi))


def bootstrap_median_ci(values: np.ndarray, n_iter: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """(median, ci_low, ci_high) via nonparametric bootstrap of the median."""
    rng = np.random.default_rng(42)
    medians = np.median(
        rng.choice(values, size=(n_iter, values.size), replace=True), axis=1
    )
    return float(np.median(values)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def matched_cell_difference(pairs: pl.DataFrame) -> tuple[float, float, float, int]:
    """Flagged-minus-control median excess return within matched cells.

    Cells are (AREA_EN, rooms_band, entry_month); only cells containing both
    cohorts count. Returns (weighted mean difference, ci_low, ci_high, n_cells)
    with the CI bootstrapped over cells.
    """
    cells = (
        pairs.with_columns(
            pl.when(pl.col("spread_pct") <= FLAG_THRESHOLD).then(pl.lit("flagged"))
            .when(pl.col("spread_pct").abs() <= CONTROL_BAND).then(pl.lit("control"))
            .otherwise(pl.lit(None)).alias("cohort")
        )
        .drop_nulls("cohort")
        .group_by("AREA_EN", "rooms_band", "entry_month", "cohort")
        .agg(pl.col("excess_return").median().alias("median_excess"), pl.len().alias("n"))
        .pivot(on="cohort", index=["AREA_EN", "rooms_band", "entry_month"],
               values=["median_excess", "n"])
        .drop_nulls(["median_excess_flagged", "median_excess_control"])
        .with_columns(
            (pl.col("median_excess_flagged") - pl.col("median_excess_control")).alias("diff"),
        )
    )
    if cells.is_empty():
        return float("nan"), float("nan"), float("nan"), 0
    diffs = cells["diff"].to_numpy()
    weights = cells["n_flagged"].to_numpy().astype(float)
    point = float(np.average(diffs, weights=weights))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots.append(np.average(diffs[idx], weights=weights[idx]))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), cells.height


def forced_sale_lift(scored: pl.DataFrame) -> dict:
    """Concentration of forced-sale procedures in the deep-spread tail."""
    labelled = scored.with_columns(
        pl.col("PROCEDURE_EN").cast(pl.Utf8)
        .str.contains(DISTRESS_PROCEDURE_PATTERN).fill_null(False).alias("forced")
    )
    n_total = labelled.height
    n_forced = int(labelled["forced"].sum())
    out = {"n_total": n_total, "n_forced": n_forced,
           "base_rate": n_forced / n_total if n_total else 0.0, "tails": {}}
    for tail_share in (0.05, 0.10):
        k = int(tail_share * n_total)
        tail = labelled.sort("spread_pct").head(k)
        hits = int(tail["forced"].sum())
        rate = hits / k if k else 0.0
        out["tails"][f"bottom_{int(tail_share*100)}pct"] = {
            "n": k, "forced_hits": hits, "rate": rate,
            "lift": (rate / out["base_rate"]) if out["base_rate"] else None,
        }
    return out


def main() -> int:
    """Run the walk-forward backtest; print findings, save charts + JSON."""
    features = build_feature_history()
    data_max = features["date"].max()
    print(f"feature history: {features.height:,} sales through {data_max}")

    scored = walk_forward_scored(features)
    print(f"walk-forward scored entries: {scored.height:,}")

    pairs_all = build_resale_pairs_from(features, scored)
    entry_cutoff = data_max - timedelta(days=MIN_RUNWAY_DAYS)
    pairs = pairs_all.filter(pl.col("date") <= entry_cutoff)
    print(f"resale pairs (holding >= {MIN_HOLDING_DAYS}d, entry <= {entry_cutoff}): {pairs.height:,}")

    summary: dict = {"n_pairs": pairs.height, "horizons": {}}
    for lo_d, hi_d in HORIZON_BUCKETS:
        bucket = pairs.filter(pl.col("holding_days").is_between(lo_d, hi_d))
        flagged = bucket.filter(pl.col("spread_pct") <= FLAG_THRESHOLD)
        control = bucket.filter(pl.col("spread_pct").abs() <= CONTROL_BAND)
        if flagged.height < 20 or control.height < 20:
            continue
        f_med, f_lo, f_hi = bootstrap_median_ci(flagged["excess_return"].to_numpy())
        c_med, c_lo, c_hi = bootstrap_median_ci(control["excess_return"].to_numpy())
        m_diff, m_lo, m_hi, n_cells = matched_cell_difference(bucket)
        summary["horizons"][f"{lo_d}-{hi_d}d"] = {
            "flagged": {"n": flagged.height, "median": f_med, "ci": [f_lo, f_hi]},
            "control": {"n": control.height, "median": c_med, "ci": [c_lo, c_hi]},
            "matched_diff": {"value": m_diff, "ci": [m_lo, m_hi], "n_cells": n_cells},
        }
        print(
            f"holding {lo_d}-{hi_d}d: flagged n={flagged.height} median excess "
            f"{f_med:+.2%} [{f_lo:+.2%},{f_hi:+.2%}] | control n={control.height} "
            f"{c_med:+.2%} [{c_lo:+.2%},{c_hi:+.2%}] | matched diff {m_diff:+.2%} "
            f"[{m_lo:+.2%},{m_hi:+.2%}] over {n_cells} cells"
        )

    # Resale rates (survivorship): do flagged entries resell more often?
    eligible = scored.filter(pl.col("date") <= entry_cutoff).with_columns(
        pl.when(pl.col("spread_pct") <= FLAG_THRESHOLD).then(pl.lit("flagged"))
        .when(pl.col("spread_pct").abs() <= CONTROL_BAND).then(pl.lit("control"))
        .otherwise(pl.lit(None)).alias("cohort")
    ).drop_nulls("cohort")
    pair_keys = pairs.select("TRANSACTION_NUMBER").unique()
    resold = eligible.join(pair_keys, on="TRANSACTION_NUMBER", how="semi")
    rates = (
        eligible.group_by("cohort").len().rename({"len": "n_eligible"})
        .join(resold.group_by("cohort").len().rename({"len": "n_resold"}), on="cohort")
        .with_columns((pl.col("n_resold") / pl.col("n_eligible")).alias("resale_rate"))
    )
    summary["resale_rates"] = rates.to_dicts()
    print("resale rates:", rates.to_dicts())

    # Signal-strength decile calibration (12-month horizon, flag-eligible pairs)
    bucket = pairs.filter(pl.col("holding_days").is_between(30, 365))
    neg = bucket.filter(pl.col("spread_pct") < 0)
    deciles = (
        neg.with_columns(
            (pl.col("signal_strength").rank("ordinal") * 10 / neg.height)
            .ceil().clip(1, 10).cast(pl.Int32).alias("decile")
        )
        .group_by("decile")
        .agg(pl.col("excess_return").median().alias("median_excess"), pl.len().alias("n"))
        .sort("decile")
    )
    summary["signal_strength_deciles"] = deciles.to_dicts()

    summary["forced_sale_lift"] = forced_sale_lift(scored)
    print("forced-sale lift:", json.dumps(summary["forced_sale_lift"], indent=1))

    summary["regime_drift"] = regime_drift_diagnostic(features)
    print("regime drift (model frozen at Feb 2026):")
    for row in summary["regime_drift"]:
        print(f"  {row['month']}: median spread {row['median_spread']:+.2%}, "
              f"flag rate {row['flag_rate']:.2%} (n={row['n']:,})")

    # Charts ---------------------------------------------------------------
    bucket = pairs.filter(pl.col("holding_days").is_between(30, 365))
    flagged = bucket.filter(pl.col("spread_pct") <= FLAG_THRESHOLD)["excess_return"].to_numpy()
    control = bucket.filter(pl.col("spread_pct").abs() <= CONTROL_BAND)["excess_return"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    axes[0].hist(control * 100, bins=60, alpha=0.6, label=f"near fair value (n={control.size})",
                 color="#636efa", density=True)
    axes[0].hist(flagged * 100, bins=60, alpha=0.6, label=f"flagged ≤ −15% (n={flagged.size})",
                 color="#ef553b", density=True)
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("Excess return vs own district, entry → resale (%)")
    axes[0].set_title("Do flagged deals out-appreciate their district? (30–365d holds)")
    axes[0].legend()
    dec = deciles.to_dicts()
    axes[1].bar([d["decile"] for d in dec], [d["median_excess"] * 100 for d in dec],
                color="#636efa")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("Signal-strength decile (10 = strongest)")
    axes[1].set_ylabel("Median excess return (%)")
    axes[1].set_title("Calibration: stronger signal → bigger realized edge?")
    fig.tight_layout()
    fig.savefig(SCRATCH / "backtest_flags.png")
    plt.close(fig)

    (SCRATCH / "backtest_summary.json").write_text(json.dumps(summary, indent=1))
    print("charts + summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
