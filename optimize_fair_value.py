"""Fair-value model improvement loop.

Iteratively proposes feature and modeling changes, measures each against
the same out-of-time protocol (date-ordered ``TimeSeriesSplit(n_splits=10)``,
headline metric = mean CV MedAPE: the median absolute % gap between actual
price and predicted fair value on unseen future folds), keeps a change only
if it improves, and stops at diminishing returns (< 0.2 percentage points
MedAPE improvement for 2 consecutive iterations) or after 10 iterations.

Writes a markdown report of every iteration to
``fair_value_optimization_report.md``.

Usage:
    python optimize_fair_value.py                 # data from the GCS snapshot
    python optimize_fair_value.py --parquet f.pq  # data from a local parquet
    python optimize_fair_value.py --synthetic     # smoke run on synthetic data
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from fair_value_model import (
    DEFAULT_MODEL_PARAMS,
    _fold_metrics,
    _make_model,
    cross_validate,
    feature_columns,
    feature_engineering,
    fit_encoders,
    to_matrix,
    train_fair_value_model,
)

REPORT_PATH = "fair_value_optimization_report.md"
STOP_MIN_IMPROVEMENT = 0.002  # 0.2 percentage points of MedAPE
STOP_PATIENCE = 2
MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshot_from_gcs() -> pl.DataFrame:
    from gcs_storage import (
        configured_snapshot,
        gcs_client,
        load_local_secrets,
    )

    secrets = load_local_secrets()
    bucket_name, object_name = configured_snapshot(secrets)
    blob = gcs_client(secrets).bucket(bucket_name).get_blob(object_name)
    if blob is None:
        raise SystemExit(f"Snapshot gs://{bucket_name}/{object_name} not found.")
    print(f"Loading gs://{bucket_name}/{object_name} ...")
    return pl.read_parquet(io.BytesIO(blob.download_as_bytes()))


def load_synthetic() -> pl.DataFrame:
    from smoke_test_fair_value import synthetic_frame

    raw, _ = synthetic_frame()
    return raw


# ---------------------------------------------------------------------------
# Evaluation protocols (all share the same time-ordered folds)
# ---------------------------------------------------------------------------

def _summarize(folds: list[dict], n_rows: int, n_splits: int) -> dict:
    summary = {"n_splits": n_splits, "n_rows": n_rows, "folds": folds}
    for key in ("medape", "mae_log", "r2"):
        values = [f[key] for f in folds]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def _fold_slices(df: pl.DataFrame, n_splits: int):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(np.arange(df.height)):
        yield (
            df[train_idx.min() : train_idx.max() + 1],
            df[val_idx.min() : val_idx.max() + 1],
        )


def eval_area_median_baseline(feats: pl.DataFrame, n_splits: int) -> dict:
    """No-ML floor: predict each area's trailing median log PSF."""
    df = feats.sort("date")
    folds = []
    for train_df, val_df in _fold_slices(df, n_splits):
        medians = train_df.group_by("AREA_EN").agg(pl.col("log_psf").median().alias("m"))
        global_median = train_df.get_column("log_psf").median()
        pred = (
            val_df.join(medians, on="AREA_EN", how="left")
            .get_column("m")
            .fill_null(global_median)
            .to_numpy()
        )
        folds.append(_fold_metrics(val_df.get_column("log_psf").to_numpy(), pred))
    return _summarize(folds, df.height, n_splits)


def eval_ridge(feats: pl.DataFrame, feature_config: dict, n_splits: int) -> dict:
    """Linear hedonic benchmark: Ridge on one-hot categoricals + scaled numerics."""
    numeric, categorical = feature_columns(feature_config)
    df = feats.sort("date")
    folds = []
    for train_df, val_df in _fold_slices(df, n_splits):
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), list(range(len(numeric)))),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", max_categories=200, sparse_output=True),
                    list(range(len(numeric), len(numeric) + len(categorical))),
                ),
            ]
        )
        model = make_pipeline(pre, Ridge(alpha=1.0))
        encoders = fit_encoders(train_df, feature_config)
        X_tr, y_tr, _, _ = to_matrix(train_df, encoders, feature_config)
        X_va, y_va, _, _ = to_matrix(val_df, encoders, feature_config)
        X_tr, X_va = np.nan_to_num(X_tr), np.nan_to_num(X_va)
        model.fit(X_tr, y_tr)
        folds.append(_fold_metrics(y_va, model.predict(X_va)))
    return _summarize(folds, df.height, n_splits)


def eval_blend(
    feats: pl.DataFrame, feature_config: dict, model_params: dict, n_splits: int
) -> dict:
    """50/50 blend of the boosted trees and the Ridge benchmark."""
    numeric, categorical = feature_columns(feature_config)
    df = feats.sort("date")
    folds = []
    for train_df, val_df in _fold_slices(df, n_splits):
        encoders = fit_encoders(train_df, feature_config)
        X_tr, y_tr, _, cat_idx = to_matrix(train_df, encoders, feature_config)
        X_va, y_va, _, _ = to_matrix(val_df, encoders, feature_config)

        hgb = _make_model(model_params, cat_idx, 42)
        hgb.fit(X_tr, y_tr)

        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), list(range(len(numeric)))),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", max_categories=200, sparse_output=True),
                    list(range(len(numeric), len(numeric) + len(categorical))),
                ),
            ]
        )
        ridge = make_pipeline(pre, Ridge(alpha=1.0))
        ridge.fit(np.nan_to_num(X_tr), y_tr)

        pred = 0.5 * hgb.predict(X_va) + 0.5 * ridge.predict(np.nan_to_num(X_va))
        folds.append(_fold_metrics(y_va, pred))
    return _summarize(folds, df.height, n_splits)


# ---------------------------------------------------------------------------
# Improvement loop
# ---------------------------------------------------------------------------

# Proposal ladder. `toggle` flips feature groups on top of the current best
# config; `params` overrides hyperparameters on top of the current best.
CANDIDATE_LADDER: list[dict] = [
    {
        "name": "HGB core features",
        "detail": "size, area, rooms, off-plan, tier, time trend",
        "kind": "hgb",
        "reset": True,
    },
    {
        "name": "+ project categoricals",
        "detail": "PROJECT_EN / MASTER_PROJECT_EN (top 200 + OTHER)",
        "kind": "hgb",
        "toggle": {"project": True},
    },
    {
        "name": "+ building categorical",
        "detail": "BUILDING_NAME_EN (top 200 + OTHER)",
        "kind": "hgb",
        "toggle": {"building": True},
    },
    {
        "name": "+ amenity & deal features",
        "detail": "nearest metro/mall/landmark, parking, buyer/seller counts",
        "kind": "hgb",
        "toggle": {"amenity": True},
    },
    {
        "name": "+ trailing area comps",
        "detail": "30-day area median PSF, strictly past-only",
        "kind": "hgb",
        "toggle": {"comps_area": True},
    },
    {
        "name": "+ trailing project comps",
        "detail": "60-day project median PSF, strictly past-only",
        "kind": "hgb",
        "toggle": {"comps_project": True},
    },
    {
        "name": "hyperparams: slower learning",
        "detail": "learning_rate 0.04, max_iter 800",
        "kind": "hgb",
        "params": {"learning_rate": 0.04, "max_iter": 800},
    },
    {
        "name": "hyperparams: deeper trees",
        "detail": "max_leaf_nodes 127",
        "kind": "hgb",
        "params": {"max_leaf_nodes": 127},
    },
    {
        "name": "Ridge linear hedonic benchmark",
        "detail": "one-hot categoricals + scaled numerics",
        "kind": "ridge",
    },
    {
        "name": "HGB + Ridge 50/50 blend",
        "detail": "average of tree and linear predictions",
        "kind": "blend",
    },
]

BASE_FEATURE_CONFIG = {
    "project": False,
    "building": False,
    "amenity": False,
    "comps_area": False,
    "comps_project": False,
}


def resolve_candidate(spec: dict, best: dict) -> dict:
    """Concrete config for a ladder step, applied on top of the current best."""
    if spec.get("reset"):
        cfg = dict(BASE_FEATURE_CONFIG)
        params = dict(DEFAULT_MODEL_PARAMS)
    else:
        cfg = dict(best["feature_config"])
        params = dict(best["model_params"])
        cfg.update(spec.get("toggle", {}))
        params.update(spec.get("params", {}))
    return {**spec, "feature_config": cfg, "model_params": params}


def evaluate(candidate: dict, feats_by_cfg: dict, raw: pl.DataFrame, n_splits: int) -> dict:
    key = tuple(sorted(candidate["feature_config"].items()))
    if key not in feats_by_cfg:
        feats_by_cfg[key] = feature_engineering(raw, candidate["feature_config"])
    feats = feats_by_cfg[key]
    if candidate["kind"] == "hgb":
        return cross_validate(
            feats, candidate["feature_config"], candidate["model_params"], n_splits=n_splits
        )
    if candidate["kind"] == "ridge":
        return eval_ridge(feats, candidate["feature_config"], n_splits)
    if candidate["kind"] == "blend":
        return eval_blend(feats, candidate["feature_config"], candidate["model_params"], n_splits)
    raise ValueError(candidate["kind"])


def dump_progress(
    path: str | None,
    source: str,
    rows: list[dict],
    best: dict,
    status: str,
    importances: list[dict] | None = None,
) -> None:
    """Rewrite the live-progress JSON consumed by external trackers."""
    if not path:
        return
    payload = {
        "status": status,
        "source": source,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "iterations": [
            {
                "iteration": row["iteration"],
                "name": row["name"],
                "detail": row["detail"],
                "medape_mean": row["metrics"]["medape_mean"],
                "medape_std": row["metrics"]["medape_std"],
                "mae_log_mean": row["metrics"]["mae_log_mean"],
                "r2_mean": row["metrics"]["r2_mean"],
                "decision": row["decision"],
                "seconds": row["seconds"],
            }
            for row in rows
        ],
        "best": {
            "name": best["name"],
            "kind": best["kind"],
            "medape_mean": best["metrics"]["medape_mean"],
            "medape_std": best["metrics"]["medape_std"],
            "r2_mean": best["metrics"]["r2_mean"],
            "feature_config": best["feature_config"],
            "model_params": {
                k: v for k, v in best["model_params"].items() if not callable(v)
            },
            "n_rows": best["metrics"]["n_rows"],
        },
        "importances": importances or [],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fair-value model improvement loop.")
    parser.add_argument("--parquet", help="Local parquet file instead of the GCS snapshot.")
    parser.add_argument("--synthetic", action="store_true", help="Smoke run on synthetic data.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--report", default=REPORT_PATH)
    parser.add_argument(
        "--progress-json",
        help="Rewrite this JSON file after every iteration (for live progress tracking).",
    )
    args = parser.parse_args()

    if args.synthetic:
        raw = load_synthetic()
        source = "synthetic smoke data"
    elif args.parquet:
        raw = pl.read_parquet(args.parquet)
        source = args.parquet
    else:
        raw = load_snapshot_from_gcs()
        source = "GCS snapshot"

    feats_by_cfg: dict = {}
    rows: list[dict] = []

    # Iteration 0 — the no-ML floor every model iteration must beat.
    base_key = tuple(sorted(BASE_FEATURE_CONFIG.items()))
    feats_by_cfg[base_key] = feature_engineering(raw, BASE_FEATURE_CONFIG)
    baseline_feats = feats_by_cfg[base_key]
    print(f"Data: {source} — {baseline_feats.height:,} scoreable apartment sales")
    t0 = time.time()
    baseline = eval_area_median_baseline(baseline_feats, args.n_splits)
    rows.append(
        {
            "iteration": 0,
            "name": "Baseline: area median PSF (no ML)",
            "detail": "per-area trailing median",
            "metrics": baseline,
            "decision": "baseline",
            "seconds": time.time() - t0,
        }
    )
    print(
        f"[0] baseline area-median: MedAPE {baseline['medape_mean']:.2%} "
        f"± {baseline['medape_std']:.2%} ({rows[-1]['seconds']:.0f}s)"
    )

    best = {
        "name": rows[0]["name"],
        "feature_config": dict(BASE_FEATURE_CONFIG),
        "model_params": dict(DEFAULT_MODEL_PARAMS),
        "metrics": baseline,
        "kind": "baseline",
    }
    dump_progress(args.progress_json, source, rows, best, "running")

    no_gain_streak = 0
    iteration = 0
    for spec in CANDIDATE_LADDER:
        if iteration >= args.max_iterations:
            break
        iteration += 1
        candidate = resolve_candidate(spec, best)

        t0 = time.time()
        metrics = evaluate(candidate, feats_by_cfg, raw, args.n_splits)
        elapsed = time.time() - t0
        improvement = best["metrics"]["medape_mean"] - metrics["medape_mean"]
        accepted = improvement > 0
        if accepted:
            best = {
                "name": candidate["name"],
                "feature_config": candidate["feature_config"],
                "model_params": candidate["model_params"],
                "metrics": metrics,
                "kind": candidate["kind"],
            }
        meaningful = improvement >= STOP_MIN_IMPROVEMENT
        no_gain_streak = 0 if meaningful else no_gain_streak + 1
        rows.append(
            {
                "iteration": iteration,
                "name": candidate["name"],
                "detail": candidate["detail"],
                "metrics": metrics,
                "decision": (
                    f"accepted (+{improvement:.2%})" if accepted else f"rejected ({improvement:+.2%})"
                ),
                "seconds": elapsed,
            }
        )
        print(
            f"[{iteration}] {candidate['name']}: MedAPE {metrics['medape_mean']:.2%} "
            f"± {metrics['medape_std']:.2%} -> {rows[-1]['decision']} ({elapsed:.0f}s)"
        )
        dump_progress(args.progress_json, source, rows, best, "running")
        if no_gain_streak >= STOP_PATIENCE:
            print(
                f"Stopping: less than {STOP_MIN_IMPROVEMENT:.1%} MedAPE improvement "
                f"for {STOP_PATIENCE} consecutive iterations (diminishing returns)."
            )
            break

    # Feature importances for the shipping (HGB) model on the winning config.
    print("Computing permutation feature importances for the winning configuration...")
    best_key = tuple(sorted(best["feature_config"].items()))
    if best_key not in feats_by_cfg:
        feats_by_cfg[best_key] = feature_engineering(raw, best["feature_config"])
    final_result = train_fair_value_model(
        feats_by_cfg[best_key],
        best["feature_config"],
        best["model_params"],
        run_cv=False,
    )
    importances = final_result.importances.to_dicts()

    dump_progress(args.progress_json, source, rows, best, "done", importances)
    write_report(args.report, source, rows, best, args.n_splits, importances)
    print(f"\nBest: {best['name']} — MedAPE {best['metrics']['medape_mean']:.2%}")
    print(f"Feature config: {best['feature_config']}")
    print(f"Model params: {best['model_params']}")
    print(f"Report written to {args.report}")
    return 0


def write_report(
    path: str,
    source: str,
    rows: list[dict],
    best: dict,
    n_splits: int,
    importances: list[dict] | None = None,
) -> None:
    lines = [
        "# Fair-Value Model Optimization Report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} · data: {source} · "
        f"protocol: date-ordered TimeSeriesSplit({n_splits} folds), headline metric = "
        "mean CV MedAPE (median |actual − fair value| / fair value on future folds).",
        "",
        "Stop rule: < 0.2 pp MedAPE improvement for 2 consecutive iterations, or 10 iterations.",
        "",
        "## Iterations",
        "",
        "| # | Proposal | Detail | MedAPE (mean ± std) | MAE(log) | R² | Decision | Time |",
        "|---|----------|--------|--------------------:|---------:|----:|----------|-----:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['iteration']} | {row['name']} | {row['detail']} | "
            f"{m['medape_mean']:.2%} ± {m['medape_std']:.2%} | {m['mae_log_mean']:.4f} | "
            f"{m['r2_mean']:.3f} | {row['decision']} | {row['seconds']:.0f}s |"
        )
    lines += [
        "",
        "## Winning configuration",
        "",
        f"- **Model**: {best['kind']} — {best['name']}",
        f"- **MedAPE**: {best['metrics']['medape_mean']:.2%} ± {best['metrics']['medape_std']:.2%}",
        f"- **R²**: {best['metrics']['r2_mean']:.3f}",
        f"- **Feature config**: `{best['feature_config']}`",
        f"- **Model params**: `{best['model_params']}`",
        f"- **Rows**: {best['metrics']['n_rows']:,}",
        "",
    ]
    if importances:
        lines += [
            "## Feature importances (permutation, winning model)",
            "",
            "| Feature | Importance | ± std |",
            "|---------|-----------:|------:|",
        ]
        lines += [
            f"| {imp['feature']} | {imp['importance_mean']:.4f} | {imp['importance_std']:.4f} |"
            for imp in importances[:15]
        ]
        lines.append("")
    lines += [
        "## Phase 2 data candidates (not yet integrated)",
        "",
        "- Live listing asking prices (Bayut / Property Finder) — score offers, not just closed sales.",
        "- Ejari rent contracts (Dubai Pulse `dld_rent_contracts`) — project rental yield feature and distress corroboration.",
        "- Buildings/units metadata (floor, building age, developer) — strongest missing hedonic features.",
        "- Official residential sale price index — drift monitoring.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
