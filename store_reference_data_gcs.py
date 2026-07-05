"""Pull DLD reference datasets and publish them to GCS under dld_reference/.

Reference data feeds the fair-value model's project_meta / rent_yield /
service_charge feature groups:

- projects.parquet            full projects table (developer, completion dates)
- project_buildings_agg.parquet   buildings aggregated per project_id
- service_charges.parquet     latest budget-year service cost per project
- rent_index.parquet          weekly area x rooms-band grid of trailing-180d
                              median rent PSF (strictly past per week)

Usage:
    python store_reference_data_gcs.py --only projects buildings service
    python store_reference_data_gcs.py --only rents     # long pull (~2-4h)
    python store_reference_data_gcs.py                  # everything
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import replace
from datetime import date

import polars as pl

from dashboard_constants import SQM_TO_SQFT
from dda_api import fetch_dataset_records, load_dda_config
from gcs_storage import dataframe_to_parquet_bytes, gcs_client, load_local_secrets, setting

REFERENCE_PREFIX = "dld_reference"
RENTS_START = date(2024, 1, 1)

# Ejari sanitization bounds (probe found corrupt years 2034/2109/2204 and
# garbage tiny areas).
RENT_YEAR_MIN, RENT_YEAR_MAX = 2020, 2027
RENT_ANNUAL_MIN, RENT_ANNUAL_MAX = 5_000, 5_000_000
RENT_AREA_MIN, RENT_AREA_MAX = 15.0, 1_000.0  # sqm


def fetch_dataset(config, dataset: str, params: dict | None = None,
                  max_records: int = 5_000_000) -> pl.DataFrame:
    cfg = replace(config, dataset=dataset)
    t0 = time.time()
    records = fetch_dataset_records(cfg, params=params or {}, max_records=max_records)
    print(f"{dataset}: {len(records):,} records in {time.time() - t0:.0f}s", flush=True)
    # infer_schema_length=None scans every row: gateway pages mix numeric and
    # string representations for the same field (seen in dld_buildings), and a
    # partial scan crashes the build after an hours-long fetch.
    return pl.DataFrame(records, infer_schema_length=None)


def upload(secrets, name: str, df: pl.DataFrame) -> None:
    bucket_name = setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    object_name = f"{REFERENCE_PREFIX}/{name}"
    blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)
    blob.metadata = {"row_count": str(df.height)}
    blob.upload_from_string(
        dataframe_to_parquet_bytes(df),
        content_type="application/vnd.apache.parquet",
    )
    readback = pl.read_parquet(io.BytesIO(blob.download_as_bytes()))
    assert readback.height == df.height, f"readback mismatch for {object_name}"
    print(f"uploaded gs://{bucket_name}/{object_name} ({df.height:,} rows, verified)", flush=True)


def pull_projects(config, secrets) -> None:
    proj = fetch_dataset(config, "dld_projects-open-api")
    dev = fetch_dataset(config, "dld_developers-open-api")
    dev_names = dev.select(
        pl.col("developer_id").cast(pl.Int64, strict=False),
        pl.col("developer_name_en").alias("developer_name_lkp"),
    ).unique("developer_id")
    proj = proj.with_columns(
        pl.col("developer_id").cast(pl.Int64, strict=False)
    ).join(dev_names, on="developer_id", how="left").with_columns(
        pl.coalesce(pl.col("developer_name"), pl.col("developer_name_lkp")).alias("developer_name")
    ).drop("developer_name_lkp")
    upload(secrets, "projects.parquet", proj)


def pull_buildings(config, secrets) -> None:
    b = fetch_dataset(config, "dld_buildings-open-api", max_records=1_000_000)
    agg = (
        b.with_columns(
            pl.col("floors").cast(pl.Float64, strict=False),
            pl.col("flats").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("project_id").is_not_null())
        .group_by("project_id")
        .agg(
            pl.col("floors").max().alias("project_max_floors"),
            pl.col("floors").mean().alias("project_mean_floors"),
            pl.col("flats").sum().alias("project_flats"),
            pl.len().alias("project_buildings"),
        )
    )
    upload(secrets, "project_buildings_agg.parquet", agg)


def pull_service_charges(config, secrets) -> None:
    s = fetch_dataset(config, "dld_oa_service_charges-open-api", max_records=1_000_000)
    latest = (
        s.with_columns(
            pl.col("budget_year").cast(pl.Int64, strict=False),
            pl.col("service_cost").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("project_id").is_not_null() & pl.col("service_cost").is_not_null())
        .group_by("project_id", "budget_year")
        .agg(pl.col("service_cost").sum().alias("service_cost_total"))
        .sort("budget_year")
        .group_by("project_id")
        .agg(
            pl.col("service_cost_total").last().alias("service_cost"),
            pl.col("budget_year").last().alias("service_cost_year"),
        )
    )
    upload(secrets, "service_charges.parquet", latest)


ROOMS_BAND_MAP = {
    "studio": "Studio",
    "1 bed": "1BR", "one bed": "1BR",
    "2 bed": "2BR", "two bed": "2BR",
    "3 bed": "3BR", "three bed": "3BR",
    "4 bed": "4BR+", "five": "4BR+", "5 bed": "4BR+", "penthouse": "4BR+",
}


def rooms_band_expr() -> pl.Expr:
    sub = pl.col("ejari_property_sub_type_en").cast(pl.Utf8).str.to_lowercase()
    expr = pl.lit(None, dtype=pl.Utf8)
    for needle, band in reversed(list(ROOMS_BAND_MAP.items())):
        expr = pl.when(sub.str.contains(needle)).then(pl.lit(band)).otherwise(expr)
    return expr


def pull_rents(config, secrets) -> None:
    params = {
        "filter": (
            "property_usage_en='Residential' AND "
            f"contract_start_date>='{RENTS_START.isoformat()}'"
        ),
        "order_by": "contract_start_date",
        "order_dir": "asc",
    }
    raw = fetch_dataset(config, "dld_rent_contracts-open-api", params=params,
                        max_records=6_000_000)

    n0 = raw.height
    rents = raw.select(
        pl.col("contract_start_date").cast(pl.Utf8).str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False).alias("start"),
        pl.col("annual_amount").cast(pl.Float64, strict=False),
        pl.col("actual_area").cast(pl.Float64, strict=False),
        pl.col("area_name_en").cast(pl.Utf8).str.to_uppercase().alias("AREA_EN"),
        rooms_band_expr().alias("rooms_band"),
        pl.col("ejari_property_type_en").cast(pl.Utf8).alias("ptype"),
    ).drop_nulls(["start", "annual_amount", "AREA_EN"])
    rents = rents.filter(
        (pl.col("start").dt.year() >= RENT_YEAR_MIN)
        & (pl.col("start").dt.year() <= RENT_YEAR_MAX)
        & (pl.col("annual_amount") >= RENT_ANNUAL_MIN)
        & (pl.col("annual_amount") <= RENT_ANNUAL_MAX)
        & (pl.col("actual_area") >= RENT_AREA_MIN)
        & (pl.col("actual_area") <= RENT_AREA_MAX)
        & pl.col("ptype").str.to_lowercase().str.contains("flat")
    ).with_columns(
        (pl.col("annual_amount") / (pl.col("actual_area") * SQM_TO_SQFT)).alias("rent_psf"),
        pl.col("start").dt.truncate("1w").alias("week"),
    )
    print(f"Ejari sanitization: {n0:,} -> {rents.height:,} usable flat contracts", flush=True)

    # Weekly grid: value at week w = median rent PSF over contracts starting in
    # the 180 days BEFORE w (strictly past: window ends at w, closed left).
    weekly = (
        rents.group_by("AREA_EN", "rooms_band", "week")
        .agg(pl.col("rent_psf").median().alias("wk_med"), pl.len().alias("wk_n"))
        .sort("week")
    )
    grid = (
        weekly.with_columns(
            pl.col("wk_med")
            .rolling_mean_by("week", window_size="180d", closed="left")
            .over("AREA_EN", "rooms_band")
            .alias("area_rent_psf_180d"),
            pl.col("wk_n")
            .rolling_sum_by("week", window_size="180d", closed="left")
            .over("AREA_EN", "rooms_band")
            .alias("rent_contracts_180d"),
        )
        .drop_nulls(["area_rent_psf_180d"])
        .select("AREA_EN", "rooms_band", "week", "area_rent_psf_180d", "rent_contracts_180d")
    )
    upload(secrets, "rent_index.parquet", grid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=["projects", "buildings", "service", "rents"],
                        choices=["projects", "buildings", "service", "rents"])
    args = parser.parse_args()

    secrets = load_local_secrets()
    config = load_dda_config(secrets)
    if config.missing_fields():
        print("Missing DDA configuration: " + ", ".join(config.missing_fields()))
        return 2

    if "projects" in args.only:
        pull_projects(config, secrets)
    if "buildings" in args.only:
        pull_buildings(config, secrets)
    if "service" in args.only:
        pull_service_charges(config, secrets)
    if "rents" in args.only:
        pull_rents(config, secrets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
