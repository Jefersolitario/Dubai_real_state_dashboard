"""Outcome backtest for the fair-value flags (distress-claim validation), v2.

Hypothesis: if a flagged below-fair-value sale is a genuine bargain, the buyer
acquired real equity — when the same unit resells, its realized appreciation
should beat the area market's over the same period.

v2 incorporates the findings of two independent audits of the first run:
- WALK-FORWARD scoring (hindsight-bias guard): each quarterly block of entries
  is flagged by a model trained only on data strictly before that block.
- STRATIFICATION by unit uniqueness: the pseudo-unit key (building + rooms +
  area to 0.1 sqm) collides across identical stacked layouts, so pairs are
  reported by `layout_units` — the units-registry count of identical-area
  units in the project. `layout_units == 1` pairs are provably the same
  physical unit and form the headline.
- RESALE-SPREAD REVERSION check: flagged entries' resales are scored with the
  walk-forward models; resale spread near 0 means the gap was priced back
  (genuine below-market entry), resale spread near -15% means the unit is
  persistently cheap (not a bargain).
- Symmetric overpriced cohort, disjoint holding buckets with per-bucket
  runway, building-level cluster bootstrap, off-plan-aware matched cells,
  deep-tail (< -35%) exclusion from the headline cohort.

Usage:
    python backtest_flags.py            # full run, writes charts + JSON
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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
DEEP_TAIL_THRESHOLD = -0.35     # below this = likely data noise, reported apart
CONTROL_BAND = 0.05             # controls: |spread| <= this (near fair value)
OVERPRICED_THRESHOLD = 0.15     # symmetric cohort: spread at/above this
MIN_HOLDING_DAYS = 30           # resales sooner are likely assignments/artifacts
WINSOR_QUANTILES = (0.01, 0.99)
N_BOOTSTRAP = 2000
# Disjoint holding buckets; each requires entry <= data_max - hi_days runway.
HOLDING_BUCKETS = [(30, 183), (184, 365), (366, 550)]

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
        blocks.append(score_transactions(model, block))
        print(f"walk-forward {cutoff}: trained on {train_frame.height:,}, "
              f"scored {block.height:,}", flush=True)
    return flag_distress(pl.concat(blocks, how="vertical"))


def regime_drift_diagnostic(features: pl.DataFrame) -> list[dict]:
    """Median spread + flag rate by month, model frozen before the downturn.

    A model trained only through REGIME_CUTOFF scores Feb-Jul 2026 fully
    out-of-sample; if fair values lag a falling market, the monthly median
    spread drifts negative and the flag rate inflates. Months with < 15 days
    of data are marked partial.
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
            pl.col("date").dt.day().n_unique().alias("days_observed"),
            pl.col("spread_pct").median().alias("median_spread"),
            (pl.col("spread_pct") <= FLAG_THRESHOLD).mean().alias("flag_rate"),
        )
        .sort("month")
        .with_columns((pl.col("days_observed") < 15).alias("partial_month"))
    )
    return monthly.to_dicts()


def build_resale_pairs_from(
    features: pl.DataFrame, walk_forward: pl.DataFrame
) -> pl.DataFrame:
    """One row per (entry sale -> next same-pseudo-unit sale) pair.

    Resale outcomes and the district index come from the FULL history
    (realized facts); the entry's spread columns join in from the
    walk-forward frame, so cohort membership is strictly out-of-sample.
    The resale's own walk-forward spread joins in where the resale falls in
    a scored block (the reversion check). `layout_units` (units-registry
    count of identical-area units in the project) stratifies pairs by how
    provably "same physical unit" they are.
    """
    wf_spreads = walk_forward.select(
        "TRANSACTION_NUMBER", "spread_pct", "signal_strength", "cold_start"
    ).unique("TRANSACTION_NUMBER")
    scored = features.join(wf_spreads, on="TRANSACTION_NUMBER", how="left")
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
            pl.col("spread_pct").shift(-1).over("unit_key").alias("resale_spread"),
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
        pl.when(pl.col("layout_units") == 1).then(pl.lit("unique_unit"))
        .when(pl.col("layout_units") > 1).then(pl.lit("stacked_layout"))
        .otherwise(pl.lit("unmatched_registry"))
        .alias("uniqueness"),
    )
    lo, hi = pairs.select(
        pl.col("excess_return").quantile(WINSOR_QUANTILES[0]).alias("lo"),
        pl.col("excess_return").quantile(WINSOR_QUANTILES[1]).alias("hi"),
    ).row(0)
    return pairs.with_columns(pl.col("excess_return").clip(lo, hi))


def entry_cohort_expr() -> pl.Expr:
    """Cohort label per entry: flagged / deep_tail / control / overpriced."""
    spread = pl.col("spread_pct")
    return (
        pl.when(spread < DEEP_TAIL_THRESHOLD).then(pl.lit("deep_tail"))
        .when(spread <= FLAG_THRESHOLD).then(pl.lit("flagged"))
        .when(spread.abs() <= CONTROL_BAND).then(pl.lit("control"))
        .when(spread >= OVERPRICED_THRESHOLD).then(pl.lit("overpriced"))
        .otherwise(pl.lit(None))
    )


def cluster_bootstrap_median_ci(
    frame: pl.DataFrame, value_col: str, cluster_col: str = "BUILDING_NAME_EN"
) -> tuple[float, float, float]:
    """(median, ci_low, ci_high) bootstrapping CLUSTERS (buildings), not rows.

    Pairs within a building share price chains and local shocks; iid row
    resampling understates the CI width.
    """
    groups = frame.partition_by(cluster_col, as_dict=False)
    arrays = [g[value_col].to_numpy() for g in groups]
    rng = np.random.default_rng(42)
    medians = np.empty(N_BOOTSTRAP)
    n = len(arrays)
    for i in range(N_BOOTSTRAP):
        picked = rng.integers(0, n, n)
        medians[i] = np.median(np.concatenate([arrays[j] for j in picked]))
    point = float(frame[value_col].median())
    return point, float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def matched_cell_difference(pairs: pl.DataFrame) -> tuple[float, float, float, int]:
    """Flagged-minus-control median excess within matched cells.

    Cells are (AREA_EN, rooms band, entry month, off-plan status); a cell
    counts only when it holds >= 3 pairs of EACH cohort, weighted by the
    balanced n_f*n_c/(n_f+n_c). CI bootstrapped over cells.
    """
    cells = (
        pairs.with_columns(entry_cohort_expr().alias("cohort"))
        .filter(pl.col("cohort").is_in(["flagged", "control"]))
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("entry_month"))
        .group_by("AREA_EN", "ROOMS_EN", "entry_month", "IS_OFFPLAN_EN", "cohort")
        .agg(pl.col("excess_return").median().alias("median_excess"), pl.len().alias("n"))
        .pivot(on="cohort", index=["AREA_EN", "ROOMS_EN", "entry_month", "IS_OFFPLAN_EN"],
               values=["median_excess", "n"])
    )
    required = {"median_excess_flagged", "median_excess_control", "n_flagged", "n_control"}
    if not required.issubset(cells.columns):
        # a sparse bucket can lack one cohort entirely
        return float("nan"), float("nan"), float("nan"), 0
    cells = (
        cells.drop_nulls(["median_excess_flagged", "median_excess_control"])
        .filter((pl.col("n_flagged") >= 3) & (pl.col("n_control") >= 3))
        .with_columns(
            (pl.col("median_excess_flagged") - pl.col("median_excess_control")).alias("diff"),
            (pl.col("n_flagged") * pl.col("n_control")
             / (pl.col("n_flagged") + pl.col("n_control"))).alias("weight"),
        )
    )
    if cells.is_empty():
        return float("nan"), float("nan"), float("nan"), 0
    diffs = cells["diff"].to_numpy()
    weights = cells["weight"].to_numpy()
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


def cohort_stats(frame: pl.DataFrame, label: str) -> dict | None:
    """Cluster-bootstrapped median excess for one cohort slice."""
    if frame.height < 20:
        return None
    med, lo, hi = cluster_bootstrap_median_ci(frame, "excess_return")
    return {"label": label, "n": frame.height, "median": med, "ci": [lo, hi]}


def main() -> int:
    """Run the corrected backtest; print findings, save charts + JSON."""
    features = build_feature_history()
    data_max = features["date"].max()
    print(f"feature history: {features.height:,} sales through {data_max}")

    scored = walk_forward_scored(features)
    print(f"walk-forward scored entries: {scored.height:,}")

    pairs = build_resale_pairs_from(features, scored).with_columns(
        entry_cohort_expr().alias("cohort")
    )
    print(f"resale pairs (holding >= {MIN_HOLDING_DAYS}d): {pairs.height:,}")

    summary: dict = {"version": 2, "n_pairs": pairs.height, "buckets": {}}

    # Headline: disjoint holding buckets x uniqueness strata ------------------
    for lo_d, hi_d in HOLDING_BUCKETS:
        runway_cutoff = data_max - timedelta(days=hi_d)
        bucket = pairs.filter(
            pl.col("holding_days").is_between(lo_d, hi_d)
            & (pl.col("date") <= runway_cutoff)
        )
        rows: list[dict] = []
        for cohort in ("flagged", "control", "overpriced", "deep_tail"):
            sub = bucket.filter(pl.col("cohort") == cohort)
            stats = cohort_stats(sub, cohort)
            if stats:
                rows.append(stats)
            if cohort in ("flagged", "control"):
                for stratum in ("unique_unit", "stacked_layout"):
                    s = cohort_stats(
                        sub.filter(pl.col("uniqueness") == stratum),
                        f"{cohort}/{stratum}",
                    )
                    if s:
                        rows.append(s)
        m_diff, m_lo, m_hi, n_cells = matched_cell_difference(bucket)
        summary["buckets"][f"{lo_d}-{hi_d}d"] = {
            "cohorts": rows,
            "matched_diff": {"value": m_diff, "ci": [m_lo, m_hi], "n_cells": n_cells},
        }
        print(f"\nholding {lo_d}-{hi_d}d (entries <= {runway_cutoff}):")
        for r in rows:
            print(f"  {r['label']:>26}: n={r['n']:>5} median {r['median']:+.2%} "
                  f"[{r['ci'][0]:+.2%}, {r['ci'][1]:+.2%}]")
        print(f"  matched flagged-control diff: {m_diff:+.2%} [{m_lo:+.2%}, {m_hi:+.2%}] "
              f"({n_cells} cells)")

    # Reversion check: do flagged entries' resales come back to fair value? --
    reversion = (
        pairs.drop_nulls("resale_spread")
        .group_by("cohort")
        .agg(
            pl.len().alias("n"),
            pl.col("spread_pct").median().alias("entry_spread_median"),
            pl.col("resale_spread").median().alias("resale_spread_median"),
        )
        .sort("cohort")
    )
    summary["reversion"] = reversion.to_dicts()
    print("\nresale-spread reversion (resales scored by walk-forward models):")
    for r in summary["reversion"]:
        print(f"  {r['cohort'] or 'other':>10}: n={r['n']:>5} entry spread "
              f"{r['entry_spread_median']:+.2%} -> resale spread {r['resale_spread_median']:+.2%}")

    # Resale rates with clean denominators, overall and unique-unit stratum --
    eligible = (
        scored.filter(
            pl.col("BUILDING_NAME_EN").is_not_null()
            & pl.col("TRANSACTION_NUMBER").is_not_null()
        )
        .with_columns(entry_cohort_expr().alias("cohort"))
        .drop_nulls("cohort")
    )
    resold_keys = pairs.select("TRANSACTION_NUMBER").unique()
    rates = (
        eligible.group_by("cohort").len().rename({"len": "n_eligible"})
        .join(
            eligible.join(resold_keys, on="TRANSACTION_NUMBER", how="semi")
            .group_by("cohort").len().rename({"len": "n_resold"}),
            on="cohort", how="left",
        )
        .with_columns(
            (pl.col("n_resold").fill_null(0) / pl.col("n_eligible")).alias("resale_rate")
        )
        .sort("cohort")
    )
    summary["resale_rates"] = rates.to_dicts()
    print("\nresale rates (building+txn non-null denominators):", rates.to_dicts())

    # Signal-strength deciles within the flag-eligible range, unique units --
    neg = pairs.filter(
        (pl.col("spread_pct") < 0)
        & (pl.col("spread_pct") >= DEEP_TAIL_THRESHOLD)
        & pl.col("holding_days").is_between(30, 365)
    )
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
    summary["regime_drift"] = regime_drift_diagnostic(features)
    print("\nregime drift (model frozen at Feb 2026):")
    for row in summary["regime_drift"]:
        marker = " (PARTIAL)" if row["partial_month"] else ""
        print(f"  {row['month']}{marker}: median spread {row['median_spread']:+.2%}, "
              f"flag rate {row['flag_rate']:.2%} (n={row['n']:,})")

    # Charts -----------------------------------------------------------------
    bucket = pairs.filter(pl.col("holding_days").is_between(30, 365))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for cohort, color in (("control", "#636efa"), ("flagged", "#ef553b")):
        for stratum, alpha in (("unique_unit", 0.85), ("stacked_layout", 0.35)):
            vals = bucket.filter(
                (pl.col("cohort") == cohort) & (pl.col("uniqueness") == stratum)
            )["excess_return"].to_numpy()
            if vals.size >= 20:
                axes[0].hist(vals * 100, bins=40, alpha=alpha, density=True,
                             label=f"{cohort} / {stratum} (n={vals.size})",
                             color=color)
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("Excess return vs own district (%)")
    axes[0].set_title("Flagged vs control, by unit-uniqueness stratum (30–365d)")
    axes[0].legend(fontsize=8)
    dec = deciles.to_dicts()
    axes[1].bar([d["decile"] for d in dec], [d["median_excess"] * 100 for d in dec],
                color="#636efa")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("Signal-strength decile (10 = strongest)")
    axes[1].set_ylabel("Median excess return (%)")
    axes[1].set_title("Signal-strength calibration (deep tail excluded)")
    fig.tight_layout()
    fig.savefig(SCRATCH / "backtest_flags_v2.png")
    plt.close(fig)

    (SCRATCH / "backtest_summary_v2.json").write_text(json.dumps(summary, indent=1))
    print("\ncharts + summary written (v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
