"""Campaign 3: rent-informed fair-value features.

Measures rent-derived feature and model proposals (from the rent-features
research report) against the shipped configuration, under the campaign 2
protocol: date-ordered ``TimeSeriesSplit`` CV on a selection window, with
acceptance requiring BOTH a MedAPE win (>= 0.05 pp) AND no tail worsening
(p90 APE +0.1 pp / false-flag propensity +0.05 pp tolerances), plus a
sequestered one-shot holdout for the final winner.

Unlike ``optimize_fair_value.py``'s ladder, this runner loads the GCS
reference frames, so reference-backed groups (rent_index, units, projects)
evaluate instead of silently degrading.

The ladder is multi-pass: after any acceptance changes the champion, every
rejected candidate is retried on top of the new best (interaction effects);
the campaign converges when a full pass accepts nothing — no further
improvements possible under the gate.

Usage:
    python -m model.optimize_rent_features --baseline-only   # measure the shipped config
    python -m model.optimize_rent_features                   # full campaign
    python -m model.optimize_rent_features --resume          # reuse completed evals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime

import numpy as np
import polars as pl

from model.fair_value_model import (
    SHIPPING_CONFIG_PATH,
    _fold_metrics,
    _make_model,
    cross_validate,
    feature_engineering,
    fit_encoders,
    load_shipping_config,
    reference_needed,
    to_matrix,
    train_fair_value_model,
    trim_psf,
)

REPORT_PATH = "reports/rent_campaign_report.md"
DEFAULT_PROGRESS = "reports/rent_campaign_progress.json"

# Campaign 2 acceptance protocol (see fair_value_optimization_report.md):
# a change ships only when MedAPE improves >= 0.05 pp AND neither tail
# metric worsens beyond fold noise. Seed-noise floor measured +/-0.01-0.02 pp.
ACCEPT_MIN_IMPROVEMENT = 0.0005  # 0.05 pp MedAPE
P90_TOLERANCE = 0.001            # +0.1 pp worst-decile APE
FLAG_PROP_TOLERANCE = 0.0005     # +0.05 pp false-flag propensity
MAX_EVALS = 40
FEATURE_CACHE_MAX = 6            # engineered full-frame cache entries kept

# CV/selection window vs sequestered one-shot holdout. Campaign 2 used
# < 2026-05-01; campaign 3 moves the fence to the latest full month.
SELECTION_END = date(2026, 8, 1)

# Derived from the rent-features research (see rent_campaign_report.md):
# each spec toggles feature groups and/or overrides model params ON TOP of
# the current champion. Ordered by expected value from the evidence.
CANDIDATE_LADDER: list[dict] = [
    {
        "name": "+ new-contract rents",
        "detail": "non-renewal Ejari rents, 13w smoothed + premium vs stock "
                  "(new-tenant indexes lead mixed ones ~3 quarters; RERA caps renewals)",
        "toggle": {"rent_new_segment": True},
    },
    {
        "name": "+ yield spread vs city band",
        "detail": "matched yield minus same-band city yield "
                  "(cap-rate level is ambiguous; the cross-sectional spread is the signal)",
        "toggle": {"yield_spread": True},
    },
    {
        "name": "+ matched gross yield",
        "detail": "district x rooms rent over the same-granularity sale comp "
                  "(cross-stock ratios carry ~18% quality-mix bias — match the stocks)",
        "toggle": {"yield_matched": True},
    },
    {
        "name": "+ rent momentum",
        "detail": "rent grid now vs ~13 weeks ago (the momentum half of the "
                  "disentangled cap rate)",
        "toggle": {"rent_momentum": True},
    },
    {
        "name": "+ rent-price divergence",
        "detail": "rent momentum minus area price momentum "
                  "(rent-price error correction predicts price growth)",
        "toggle": {"rent_divergence": True},
    },
    {
        "name": "+ income-approach anchor",
        "detail": "district rent over city band yield: the price this rent implies "
                  "(two-stage imputation, granularity-limited)",
        "toggle": {"rent_anchor": True},
    },
    {
        "name": "+ hierarchical rent pooling",
        "detail": "cell rent shrunk toward the city band, contracts-weighted "
                  "(multilevel pooling for sparse districts)",
        "toggle": {"rent_pooling": True},
    },
    {
        "name": "+ rent level (control)",
        "detail": "raw 180d district x rooms rent PSF — expected to fail again "
                  "(rents are sticky; the level re-encodes inverse price)",
        "toggle": {"rent_level": True},
    },
    {
        "name": "+ rent density (retry)",
        "detail": "trailing 180d Ejari contract count — campaign 2 reject, retried "
                  "under the tail-veto protocol",
        "toggle": {"rent_density": True},
    },
    {
        "name": "pooling prior k=10",
        "detail": "weaker shrinkage toward the city band",
        "toggle": {"rent_pooling": True, "rent_pool_k": 10},
    },
    {
        "name": "pooling prior k=50",
        "detail": "stronger shrinkage toward the city band",
        "toggle": {"rent_pooling": True, "rent_pool_k": 50},
    },
]


def load_snapshot() -> pl.DataFrame:
    from ingestion.dda_api import normalize_dld_transactions
    from ingestion.gcs_storage import (
        configured_snapshot,
        load_local_secrets,
        read_parquet_object,
    )
    from ingestion.store_dld_transactions_gcs import dedupe_snapshot

    secrets = load_local_secrets()
    bucket_name, object_name = configured_snapshot(secrets)
    print(f"Loading gs://{bucket_name}/{object_name} ...")
    raw, _ = read_parquet_object(secrets, bucket_name, object_name)
    return dedupe_snapshot(normalize_dld_transactions(raw))


def load_references(configs: list[dict]) -> dict[str, pl.DataFrame]:
    """Union of reference frames any evaluated config can require."""
    from ingestion.gcs_storage import load_local_secrets, read_reference_frames

    names: list[str] = []
    for cfg in configs:
        names += [n for n in reference_needed(cfg) if n not in names]
    if not names:
        return {}
    frames = read_reference_frames(load_local_secrets(), names)
    print(f"Reference frames loaded: {names}")
    return frames


def config_key(cfg: dict, params: dict) -> str:
    payload = json.dumps({"cfg": cfg, "params": params}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def resolve_candidate(spec: dict, best: dict) -> dict:
    cfg = dict(best["feature_config"])
    params = dict(best["model_params"])
    cfg.update(spec.get("toggle", {}))
    params.update(spec.get("params", {}))
    return {**spec, "feature_config": cfg, "model_params": params}


class FeatureCache:
    """Full-frame engineered features per config, bounded LRU."""

    def __init__(self, raw: pl.DataFrame, reference: dict[str, pl.DataFrame]):
        self.raw = raw
        self.reference = reference or None
        self._store: dict[tuple, pl.DataFrame] = {}

    def get(self, cfg: dict) -> pl.DataFrame:
        key = tuple(sorted((k, v) for k, v in cfg.items()))
        if key not in self._store:
            if len(self._store) >= FEATURE_CACHE_MAX:
                self._store.pop(next(iter(self._store)))
            self._store[key] = feature_engineering(
                self.raw, cfg, reference=self.reference
            )
        return self._store[key]


def evaluate_config(
    cache: FeatureCache, cfg: dict, params: dict, n_splits: int
) -> dict:
    feats = cache.get(cfg)
    selection = trim_psf(feats.filter(pl.col("date") < SELECTION_END))
    return cross_validate(selection, cfg, params, n_splits=n_splits)


def passes_gate(best_metrics: dict, metrics: dict) -> tuple[bool, str]:
    """(accepted, reason) under the MedAPE-win + tail-no-worsening gate."""
    gain = best_metrics["medape_mean"] - metrics["medape_mean"]
    if gain < ACCEPT_MIN_IMPROVEMENT:
        return False, f"rejected ({gain:+.2%} MedAPE)"
    if metrics["p90_ape_mean"] > best_metrics["p90_ape_mean"] + P90_TOLERANCE:
        return False, f"vetoed (p90 {metrics['p90_ape_mean']:.2%} worsens tail)"
    if metrics["flag_prop_mean"] > best_metrics["flag_prop_mean"] + FLAG_PROP_TOLERANCE:
        return False, f"vetoed (flag_prop {metrics['flag_prop_mean']:.2%} worsens tail)"
    return True, f"accepted (+{gain:.2%})"


def holdout_eval(cache: FeatureCache, cfg: dict, params: dict) -> dict:
    """Train on the trimmed selection window, score the untrimmed holdout."""
    feats = cache.get(cfg)
    selection = trim_psf(feats.filter(pl.col("date") < SELECTION_END))
    holdout = feats.filter(pl.col("date") >= SELECTION_END)
    encoders = fit_encoders(selection, cfg)
    X_tr, y_tr, _, cat_idx = to_matrix(selection, encoders, cfg)
    X_ho, y_ho, _, _ = to_matrix(holdout, encoders, cfg)
    model = _make_model(params, cat_idx, 42)
    model.fit(X_tr, y_tr)
    metrics = _fold_metrics(y_ho, model.predict(X_ho))
    metrics["n_rows"] = holdout.height
    return metrics


def metric_row(metrics: dict) -> dict:
    return {
        "medape_mean": metrics["medape_mean"],
        "medape_std": metrics["medape_std"],
        "p90_ape_mean": metrics["p90_ape_mean"],
        "flag_prop_mean": metrics["flag_prop_mean"],
        "r2_mean": metrics["r2_mean"],
        "n_rows": metrics["n_rows"],
    }


def dump_progress(path: str, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(state, f, indent=1, default=str)


def write_report(path: str, state: dict) -> None:
    b = state["baseline"]
    lines = [
        "# Rent-Feature Campaign Report (campaign 3)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} · {state['protocol']['description']}",
        "",
        f"Baseline (shipped config): MedAPE {b['metrics']['medape_mean']:.2%} "
        f"± {b['metrics']['medape_std']:.2%}, p90 {b['metrics']['p90_ape_mean']:.2%}, "
        f"flag_prop {b['metrics']['flag_prop_mean']:.2%}",
        "",
        "## Iterations",
        "",
        "| # | Pass | Proposal | MedAPE (mean ± std) | p90 APE | flag_prop | R² | Decision | Time |",
        "|---|------|----------|--------------------:|--------:|----------:|---:|----------|-----:|",
    ]
    for row in state["iterations"]:
        m = row["metrics"]
        lines.append(
            f"| {row['iteration']} | {row['pass']} | {row['name']} | "
            f"{m['medape_mean']:.2%} ± {m['medape_std']:.2%} | {m['p90_ape_mean']:.2%} | "
            f"{m['flag_prop_mean']:.2%} | {m['r2_mean']:.3f} | {row['decision']} | "
            f"{row['seconds']:.0f}s |"
        )
    best = state["best"]
    lines += [
        "",
        "## Champion",
        "",
        f"- **{best['name']}** — MedAPE {best['metrics']['medape_mean']:.2%} "
        f"± {best['metrics']['medape_std']:.2%}, p90 {best['metrics']['p90_ape_mean']:.2%}, "
        f"flag_prop {best['metrics']['flag_prop_mean']:.2%}",
        f"- Feature config: `{best['feature_config']}`",
        f"- Model params: `{best['model_params']}`",
        "",
    ]
    if state.get("holdout"):
        h = state["holdout"]
        lines += [
            "## One-shot holdout (" + state["protocol"]["holdout_window"] + ")",
            "",
            "| Config | MedAPE | p90 APE | flag_prop |",
            "|--------|-------:|--------:|----------:|",
            f"| shipped baseline | {h['baseline']['medape']:.2%} | "
            f"{h['baseline']['p90_ape']:.2%} | {h['baseline']['flag_prop']:.2%} |",
            f"| champion | {h['champion']['medape']:.2%} | "
            f"{h['champion']['p90_ape']:.2%} | {h['champion']['flag_prop']:.2%} |",
            "",
            f"Ship decision: {h['ship_decision']}",
            "",
        ]
    if state.get("importances"):
        lines += [
            "## Feature importances (permutation, champion)",
            "",
            "| Feature | Importance | ± std |",
            "|---------|-----------:|------:|",
        ]
        lines += [
            f"| {imp['feature']} | {imp['importance_mean']:.4f} | {imp['importance_std']:.4f} |"
            for imp in state["importances"][:20]
        ]
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rent-feature campaign runner.")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed evals from the progress JSON.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=MAX_EVALS)
    parser.add_argument("--progress-json", default=DEFAULT_PROGRESS)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()

    shipped_cfg, shipped_params = load_shipping_config()
    ladder_configs = [shipped_cfg] + [
        {**shipped_cfg, **spec.get("toggle", {})} for spec in CANDIDATE_LADDER
    ]

    raw = load_snapshot()
    reference = load_references(ladder_configs)
    cache = FeatureCache(raw, reference)
    print(f"Snapshot: {raw.height:,} rows; selection window < {SELECTION_END}")

    prior: dict = {}
    if args.resume:
        try:
            with open(args.progress_json) as f:
                prior = json.load(f)
        except (FileNotFoundError, ValueError):
            prior = {}
    completed: dict[str, dict] = {
        row["config_key"]: row for row in prior.get("iterations", [])
    }

    state: dict = {
        "campaign": "rent-informed features (campaign 3)",
        "status": "running",
        "started_at": prior.get("started_at")
        or datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "description": (
                f"date-ordered TimeSeriesSplit({args.n_splits}) on selection "
                f"< {SELECTION_END}; accept iff MedAPE -{ACCEPT_MIN_IMPROVEMENT:.2%} "
                f"AND p90 +{P90_TOLERANCE:.2%} / flag_prop +{FLAG_PROP_TOLERANCE:.2%} "
                "tail veto; one-shot holdout for the winner"
            ),
            "selection_end": str(SELECTION_END),
            "holdout_window": f"{SELECTION_END} onward",
        },
        "baseline": None,
        "iterations": [],
        "best": None,
        "holdout": None,
        "importances": [],
    }

    # Iteration 0 — the shipped configuration re-measured on current data.
    base_key = config_key(shipped_cfg, shipped_params)
    if prior.get("baseline") and prior["baseline"].get("config_key") == base_key:
        state["baseline"] = prior["baseline"]
        print(
            f"[0] baseline (resumed): MedAPE "
            f"{state['baseline']['metrics']['medape_mean']:.2%}"
        )
    else:
        t0 = time.time()
        metrics = evaluate_config(cache, shipped_cfg, shipped_params, args.n_splits)
        state["baseline"] = {
            "name": "shipped config (campaign 2 winner)",
            "config_key": base_key,
            "metrics": metric_row(metrics),
            "seconds": time.time() - t0,
        }
        print(
            f"[0] baseline: MedAPE {metrics['medape_mean']:.2%} "
            f"± {metrics['medape_std']:.2%}, p90 {metrics['p90_ape_mean']:.2%}, "
            f"flag_prop {metrics['flag_prop_mean']:.2%} "
            f"({state['baseline']['seconds']:.0f}s)"
        )
    best = {
        "name": state["baseline"]["name"],
        "feature_config": shipped_cfg,
        "model_params": shipped_params,
        "metrics": state["baseline"]["metrics"],
    }
    state["best"] = best
    dump_progress(args.progress_json, state)

    if args.baseline_only:
        state["status"] = "baseline-measured"
        dump_progress(args.progress_json, state)
        return 0

    evaluated: dict[str, dict] = {base_key: state["baseline"]["metrics"]}
    iteration = 0
    pass_num = 0
    pending = list(CANDIDATE_LADDER)
    while pending and iteration < args.max_evals:
        pass_num += 1
        accepted_this_pass = False
        still_pending: list[dict] = []
        for spec in pending:
            if iteration >= args.max_evals:
                still_pending.append(spec)
                continue
            candidate = resolve_candidate(spec, best)
            key = config_key(candidate["feature_config"], candidate["model_params"])
            if key == config_key(best["feature_config"], best["model_params"]):
                continue  # toggle is a no-op on the current champion
            if key in evaluated:
                metrics_row = evaluated[key]
                accepted, decision = passes_gate(best["metrics"], metrics_row)
                seconds = 0.0
            else:
                resumed = completed.get(key)
                if resumed:
                    metrics_row, seconds = resumed["metrics"], resumed["seconds"]
                else:
                    t0 = time.time()
                    metrics = evaluate_config(
                        cache,
                        candidate["feature_config"],
                        candidate["model_params"],
                        args.n_splits,
                    )
                    metrics_row = metric_row(metrics)
                    seconds = time.time() - t0
                evaluated[key] = metrics_row
                accepted, decision = passes_gate(best["metrics"], metrics_row)
            iteration += 1
            if accepted:
                best = {
                    "name": candidate["name"],
                    "feature_config": candidate["feature_config"],
                    "model_params": candidate["model_params"],
                    "metrics": metrics_row,
                }
                state["best"] = best
                accepted_this_pass = True
            else:
                still_pending.append(spec)
            state["iterations"].append(
                {
                    "iteration": iteration,
                    "pass": pass_num,
                    "name": candidate["name"],
                    "detail": candidate.get("detail", ""),
                    "config_key": key,
                    "metrics": metrics_row,
                    "decision": decision,
                    "seconds": seconds,
                }
            )
            print(
                f"[{iteration}] (pass {pass_num}) {candidate['name']}: "
                f"MedAPE {metrics_row['medape_mean']:.2%}, "
                f"p90 {metrics_row['p90_ape_mean']:.2%}, "
                f"flag {metrics_row['flag_prop_mean']:.2%} -> {decision} ({seconds:.0f}s)"
            )
            dump_progress(args.progress_json, state)
        if not accepted_this_pass:
            print("Converged: a full pass accepted nothing under the gate.")
            break
        pending = still_pending

    state["status"] = "converged"
    dump_progress(args.progress_json, state)

    # One-shot holdout: champion vs shipped baseline on the sequestered window.
    champion_wins_cv = (
        best["metrics"]["medape_mean"]
        <= state["baseline"]["metrics"]["medape_mean"] - ACCEPT_MIN_IMPROVEMENT
    )
    if champion_wins_cv:
        print("Holdout: scoring shipped baseline and champion on the sequestered window...")
        ho_base = holdout_eval(cache, shipped_cfg, shipped_params)
        ho_champ = holdout_eval(cache, best["feature_config"], best["model_params"])
        ship = (
            ho_champ["medape"] <= ho_base["medape"]
            and ho_champ["p90_ape"] <= ho_base["p90_ape"] + P90_TOLERANCE
            and ho_champ["flag_prop"] <= ho_base["flag_prop"] + FLAG_PROP_TOLERANCE
        )
        state["holdout"] = {
            "baseline": ho_base,
            "champion": ho_champ,
            "ship_decision": (
                "SHIP — champion wins CV and does not lose the holdout"
                if ship
                else "DO NOT SHIP — holdout does not confirm the CV win"
            ),
        }
        print(
            f"Holdout MedAPE: baseline {ho_base['medape']:.2%} vs "
            f"champion {ho_champ['medape']:.2%} -> {state['holdout']['ship_decision']}"
        )
        if ship:
            selection = trim_psf(
                cache.get(best["feature_config"]).filter(pl.col("date") < SELECTION_END)
            )
            result = train_fair_value_model(
                selection, best["feature_config"], best["model_params"], run_cv=False
            )
            state["importances"] = result.importances.to_dicts()
            SHIPPING_CONFIG_PATH.write_text(
                json.dumps(
                    {
                        "source": (
                            "campaign 3 (rent features): selection < "
                            f"{SELECTION_END}, MedAPE gate + tail no-worsening "
                            "veto, one-shot sequestered holdout"
                        ),
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "winner": best["name"],
                        "cv_medape_mean": best["metrics"]["medape_mean"],
                        "cv_medape_std": best["metrics"]["medape_std"],
                        "cv_r2_mean": best["metrics"]["r2_mean"],
                        "cv_p90_ape": best["metrics"]["p90_ape_mean"],
                        "cv_flag_prop": best["metrics"]["flag_prop_mean"],
                        "holdout_medape": ho_champ["medape"],
                        "holdout_note": (
                            f"one-shot holdout {SELECTION_END}+ "
                            f"({ho_champ['n_rows']:,} sales): MedAPE "
                            f"{ho_champ['medape']:.2%} vs {ho_base['medape']:.2%} "
                            f"shipped, p90 {ho_champ['p90_ape']:.2%} vs "
                            f"{ho_base['p90_ape']:.2%}, false-flag "
                            f"{ho_champ['flag_prop']:.2%} vs {ho_base['flag_prop']:.2%}"
                        ),
                        "feature_config": best["feature_config"],
                        "model_params": best["model_params"],
                    },
                    indent=1,
                )
            )
            print(f"Shipping config written to {SHIPPING_CONFIG_PATH}")
    else:
        state["holdout"] = None
        print("Champion does not beat the baseline on CV; holdout not spent.")

    state["status"] = "done"
    dump_progress(args.progress_json, state)
    write_report(args.report, state)
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
