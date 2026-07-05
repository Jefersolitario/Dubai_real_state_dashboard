"""Train the fair-value model offline and publish the inference bundle.

The Streamlit app never trains: it loads the bundle this script uploads to
GCS and only runs predictions. Run this after refreshing the snapshot
(store_dld_transactions_gcs.py) — weekly is a good cadence — and after any
optimize_fair_value.py run that changes fair_value_config.json.

Usage:
    python train_fair_value.py                 # snapshot from GCS, metrics from config
    python train_fair_value.py --cv            # also recompute 10-fold CV metrics
    python train_fair_value.py --parquet f.pq  # train from a local parquet
    python train_fair_value.py --out bundle.pkl --no-upload  # local file only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from dda_api import (
    OPTIONAL_DASHBOARD_COLUMNS,
    REQUIRED_DASHBOARD_COLUMNS,
    normalize_dld_transactions,
)
from fair_value_model import (
    SHIPPING_CONFIG_PATH,
    export_bundle,
    feature_engineering,
    load_bundle,
    load_shipping_config,
    reference_needed,
    score_transactions,
    train_fair_value_model,
    trim_psf,
)
from store_dld_transactions_gcs import dedupe_snapshot


def load_snapshot(args) -> tuple[pl.DataFrame, str]:
    if args.parquet:
        raw = pl.read_parquet(args.parquet)
        source = args.parquet
    else:
        from gcs_storage import configured_snapshot, load_local_secrets, read_parquet_object

        secrets = load_local_secrets()
        bucket_name, object_name = configured_snapshot(secrets)
        print(f"Loading gs://{bucket_name}/{object_name} ...")
        raw, _ = read_parquet_object(secrets, bucket_name, object_name)
        source = f"gs://{bucket_name}/{object_name}"

    raw = dedupe_snapshot(normalize_dld_transactions(raw))
    needed = [
        c for c in REQUIRED_DASHBOARD_COLUMNS + OPTIONAL_DASHBOARD_COLUMNS
        if c in raw.columns
    ]
    return raw.select(needed), source


def config_metrics() -> dict:
    """CV metrics recorded by the optimizer, if available."""
    try:
        cfg = json.loads(SHIPPING_CONFIG_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return {}
    metrics = {}
    if "cv_medape_mean" in cfg:
        metrics["medape_mean"] = cfg["cv_medape_mean"]
    if "cv_medape_std" in cfg:
        metrics["medape_std"] = cfg["cv_medape_std"]
    if "cv_r2_mean" in cfg:
        metrics["r2_mean"] = cfg["cv_r2_mean"]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the fair-value model and publish the inference bundle."
    )
    parser.add_argument("--parquet", help="Local parquet snapshot instead of GCS.")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Recompute 10-fold CV metrics (slow); default reuses fair_value_config.json values.",
    )
    parser.add_argument("--out", help="Also write the bundle to this local path.")
    parser.add_argument("--no-upload", action="store_true", help="Skip the GCS upload.")
    args = parser.parse_args()

    raw, source = load_snapshot(args)
    feature_config, model_params = load_shipping_config()
    print(f"Snapshot: {raw.height:,} rows from {source}")
    print(f"Config: {feature_config} | {model_params}")

    reference = None
    ref_names = reference_needed(feature_config)
    if ref_names:
        from gcs_storage import load_local_secrets, read_reference_frames

        reference = read_reference_frames(load_local_secrets(), ref_names)
        print(f"Reference frames loaded: {ref_names}")

    feats = feature_engineering(raw, feature_config, reference=reference)
    train_feats = trim_psf(feats)
    print(f"Training on {train_feats.height:,} apartment sales (of {feats.height:,} scoreable)")

    t0 = time.time()
    result = train_fair_value_model(
        train_feats, feature_config, model_params, run_cv=args.cv
    )
    if not args.cv:
        result.metrics = config_metrics()
    print(f"Trained in {time.time() - t0:.0f}s")

    data_min, data_max = feats.select(
        pl.col("date").min().alias("lo"), pl.col("date").max().alias("hi")
    ).row(0)
    bundle = export_bundle(
        result,
        extra={
            "source": source,
            "data_min_date": str(data_min),
            "data_max_date": str(data_max),
        },
    )
    print(f"Bundle size: {len(bundle) / 1e6:.1f} MB")

    # Round-trip verification: loaded bundle must reproduce predictions.
    loaded, meta = load_bundle(bundle)
    sample = feats.head(100)
    expected = score_transactions(result, sample).get_column("pred_psf").to_numpy()
    actual = score_transactions(loaded, sample).get_column("pred_psf").to_numpy()
    if not np.allclose(expected, actual):
        print("Bundle round-trip verification FAILED.")
        return 1
    print(f"Round-trip verified (100-row predictions identical; trained_at {meta['trained_at']})")

    if args.out:
        Path(args.out).write_bytes(bundle)
        print(f"Wrote {args.out}")

    if not args.no_upload:
        from gcs_storage import load_local_secrets, write_model_bundle_bytes

        secrets = load_local_secrets()
        uri = write_model_bundle_bytes(
            secrets,
            bundle,
            metadata={
                "trained_at": meta["trained_at"],
                "trained_rows": result.trained_rows,
                "data_min_date": str(data_min),
                "data_max_date": str(data_max),
            },
        )
        print(f"Uploaded {uri}")

        # Read back and verify once more against live GCS bytes.
        from gcs_storage import read_model_bundle_bytes

        readback, _ = read_model_bundle_bytes(secrets)
        loaded_rb, _ = load_bundle(readback)
        actual_rb = score_transactions(loaded_rb, sample).get_column("pred_psf").to_numpy()
        if not np.allclose(expected, actual_rb):
            print("GCS readback verification FAILED.")
            return 1
        print("GCS readback verified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
