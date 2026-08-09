"""Pull DLD reference datasets and publish them to GCS under dld_reference/.

Reference data feeds the fair-value model's project_meta / rent_yield /
service_charge feature groups:

- projects.parquet            full projects table (developer, completion dates)
- project_buildings_agg.parquet   buildings aggregated per project_id
- service_charges.parquet     latest budget-year service cost per project
- rent_index.parquet          weekly area x rooms-band grid of trailing-180d
                              median rent PSF (strictly past per week)

Usage:
    python -m ingestion.store_reference_data_gcs --only projects buildings service
    python -m ingestion.store_reference_data_gcs --only rents     # long pull (~2-4h)
    python -m ingestion.store_reference_data_gcs                  # everything
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import replace
from typing import Callable
from datetime import date

import polars as pl

from dashboard_constants import SQM_TO_SQFT
from ingestion.dda_api import DDAConfig, fetch_dataset_records, load_dda_config
from ingestion.gcs_storage import dataframe_to_parquet_bytes, gcs_client, load_local_secrets, setting

REFERENCE_PREFIX = "dld_reference"
RENTS_START = date(2024, 1, 1)

# Ejari sanitization bounds (probe found corrupt years 2034/2109/2204 and
# garbage tiny areas).
RENT_YEAR_MIN, RENT_YEAR_MAX = 2020, 2027
RENT_ANNUAL_MIN, RENT_ANNUAL_MAX = 5_000, 5_000_000
RENT_AREA_MIN, RENT_AREA_MAX = 15.0, 1_000.0  # sqm


def fetch_dataset(config: "DDAConfig", dataset: str, params: dict | None = None,
                  max_records: int = 5_000_000) -> pl.DataFrame:
    """Fetch one gateway dataset into a frame (schema inferred over all rows)."""
    cfg = replace(config, dataset=dataset)
    t0 = time.time()
    records = fetch_dataset_records(cfg, params=params or {}, max_records=max_records)
    print(f"{dataset}: {len(records):,} records in {time.time() - t0:.0f}s", flush=True)
    # infer_schema_length=None scans every row: gateway pages mix numeric and
    # string representations for the same field (seen in dld_buildings), and a
    # partial scan crashes the build after an hours-long fetch.
    return pl.DataFrame(records, infer_schema_length=None)


def upload(secrets: dict, name: str, df: pl.DataFrame) -> None:
    """Write a reference frame to gs://<bucket>/dld_reference/<name> and verify readback."""
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


def pull_projects(config: "DDAConfig", secrets: dict) -> None:
    """Publish the full projects table, with developer names joined in."""
    projects_df = fetch_dataset(config, "dld_projects-open-api")
    developers_df = fetch_dataset(config, "dld_developers-open-api")
    developer_names = developers_df.select(
        pl.col("developer_id").cast(pl.Int64, strict=False),
        pl.col("developer_name_en").alias("developer_name_lkp"),
    ).unique("developer_id")
    projects_df = projects_df.with_columns(
        pl.col("developer_id").cast(pl.Int64, strict=False)
    ).join(developer_names, on="developer_id", how="left").with_columns(
        pl.coalesce(pl.col("developer_name"), pl.col("developer_name_lkp")).alias("developer_name")
    ).drop("developer_name_lkp")
    upload(secrets, "projects.parquet", projects_df)


def pull_buildings(config: "DDAConfig", secrets: dict) -> None:
    """Publish per-project building aggregates (max/mean floors, flats, count)."""
    buildings_df = fetch_dataset(config, "dld_buildings-open-api", max_records=1_000_000)
    agg = (
        buildings_df.with_columns(
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


def pull_service_charges(config: "DDAConfig", secrets: dict) -> None:
    """Publish the latest budget-year total service cost per project."""
    charges_df = fetch_dataset(config, "dld_oa_service_charges-open-api", max_records=1_000_000)
    latest = (
        charges_df.with_columns(
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


# Flats-only units pull, chunked by rooms so a gateway failure costs one
# chunk. Units without a rooms label are skipped (they cannot be matched to
# transactions meaningfully anyway).
UNITS_ROOMS_CHUNKS = [
    "Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R", "6 B/R", "PENTHOUSE",
]


def fetch_chunked(config: "DDAConfig", dataset: str,
                  param_chunks: list[tuple[str, dict]], max_records: int,
                  transform: "Callable[[pl.DataFrame], pl.DataFrame] | None" = None) -> pl.DataFrame:
    """Fetch a long pull as labelled chunks with a chunk-level retry.

    ``transform`` (chunk DataFrame -> DataFrame) is applied per chunk before
    accumulation so hours-long pulls hold slim frames in memory instead of
    every raw column. Empty pulls return an empty frame instead of crashing
    at the final concat.
    """
    frames: list[pl.DataFrame] = []
    for label, params in param_chunks:
        for attempt in (1, 2):
            try:
                chunk = fetch_dataset(config, dataset, params=params,
                                      max_records=max_records)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"{dataset} chunk {label}: retrying after "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(30)
        if chunk.height:
            frames.append(transform(chunk) if transform else chunk)
    if not frames:
        print(f"{dataset}: no records in any chunk", flush=True)
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _floor_number_expr() -> pl.Expr:
    """Numeric floor from DLD labels: '12'→12, 'B2'→-2 (basement), 'G'→0.

    Anything else (mezzanine, podium, penthouse labels) becomes null rather
    than a wrong positive number — 'B2' must not look like the 2nd floor.
    """
    floor = pl.col("floor").cast(pl.Utf8).str.strip_chars()
    return (
        pl.when(floor.str.contains(r"^\d+$"))
        .then(floor.cast(pl.Float64, strict=False))
        .when(floor.str.contains(r"(?i)^B\s*\d+$"))
        .then(-floor.str.extract(r"(\d+)").cast(pl.Float64, strict=False))
        .when(floor.str.contains(r"(?i)^GF?$|^G\s*F$"))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _slim_units(chunk: pl.DataFrame) -> pl.DataFrame:
    """Reduce a raw units chunk to the 6 columns the model join needs."""
    return chunk.select(
        pl.col("project_id").cast(pl.Int64, strict=False),
        pl.col("building_number").cast(pl.Utf8),
        _floor_number_expr().alias("floor_num"),
        pl.col("actual_area").cast(pl.Float64, strict=False),
        pl.col("rooms_en").cast(pl.Utf8),
        pl.col("unit_balcony_area").cast(pl.Float64, strict=False),
    ).drop_nulls(["project_id", "actual_area"])


def pull_units(config: "DDAConfig", secrets: dict) -> None:
    """Publish the slim flats registry (floor, exact area, balcony), chunked by rooms."""
    chunks = [
        (rooms, {"filter": f"property_sub_type_en='Flat' AND rooms_en='{rooms}'"})
        for rooms in UNITS_ROOMS_CHUNKS
    ]
    slim = fetch_chunked(config, "dld_units-open-api", chunks,
                         max_records=2_000_000, transform=_slim_units)
    if slim.is_empty():
        raise RuntimeError("units pull returned no records; not overwriting GCS")
    upload(secrets, "units_slim.parquet", slim)


def pull_sale_index(config: "DDAConfig", secrets: dict) -> None:
    """Publish the official monthly flat sale index (frozen at 2024-05 upstream)."""
    index_df = fetch_dataset(config, "dld_residential_sale_index-open-api", max_records=10_000)
    monthly = (
        index_df.select(
            pl.col("first_date_of_month").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False).alias("month"),
            pl.col("flat_monthly_price_index").cast(pl.Float64, strict=False)
            .alias("flat_price_index"),
            pl.col("flat_monthly_index").cast(pl.Float64, strict=False)
            .alias("flat_index"),
        )
        .drop_nulls("month")
        .unique("month")
        .sort("month")
    )
    upload(secrets, "sale_index.parquet", monthly)


ROOMS_BAND_MAP = {
    "studio": "Studio",
    "1 bed": "1BR", "one bed": "1BR",
    "2 bed": "2BR", "two bed": "2BR",
    "3 bed": "3BR", "three bed": "3BR",
    "4 bed": "4BR+", "five": "4BR+", "5 bed": "4BR+", "penthouse": "4BR+",
}


def rooms_band_expr() -> pl.Expr:
    """Map Ejari sub-type text to the dashboard rooms bands (Studio..4BR+)."""
    sub = pl.col("ejari_property_sub_type_en").cast(pl.Utf8).str.to_lowercase()
    expr = pl.lit(None, dtype=pl.Utf8)
    for needle, band in reversed(list(ROOMS_BAND_MAP.items())):
        expr = pl.when(sub.str.contains(needle)).then(pl.lit(band)).otherwise(expr)
    return expr


def _month_starts(start: date, end: date) -> list[date]:
    """Ascending first-of-month bounds covering [start, end], plus one past end."""
    from calendar import monthrange
    from datetime import timedelta

    from ingestion.dda_api import months_before

    anchor = end.replace(day=1)
    n_back = (anchor.year - start.year) * 12 + (anchor.month - start.month)
    months = [months_before(anchor, k) for k in range(n_back, -1, -1)]
    # first of the month after `end`: add the anchor month's exact length
    return months + [anchor + timedelta(days=monthrange(anchor.year, anchor.month)[1])]


def _slim_rents(chunk: pl.DataFrame) -> pl.DataFrame:
    """Reduce a raw Ejari chunk to the columns the weekly rent grid needs."""
    return chunk.select(
        pl.col("contract_start_date").cast(pl.Utf8).str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False).alias("start"),
        pl.col("annual_amount").cast(pl.Float64, strict=False),
        pl.col("actual_area").cast(pl.Float64, strict=False),
        pl.col("area_name_en").cast(pl.Utf8).str.to_uppercase().alias("AREA_EN"),
        rooms_band_expr().alias("rooms_band"),
        pl.col("ejari_property_type_en").cast(pl.Utf8).alias("ptype"),
        pl.lit(1).alias("_raw_row"),
    )


def pull_rents(config: "DDAConfig", secrets: dict) -> None:
    """Publish the weekly area x rooms-band trailing-180d rent-PSF grid."""
    # Monthly chunks: a ~11M-row pull runs for hours, and one unrecoverable
    # gateway error mid-pagination used to discard everything fetched so far.
    # Per-chunk the page retry still applies; a failed chunk is retried once
    # before giving up, and each chunk is slimmed to 7 columns before it is
    # kept in memory.
    bounds = _month_starts(RENTS_START, date.today())
    chunks = [
        (
            lo.isoformat(),
            {
                "filter": (
                    "property_usage_en='Residential' AND "
                    f"contract_start_date>='{lo.isoformat()}' AND "
                    f"contract_start_date<'{hi.isoformat()}'"
                ),
                "order_by": "contract_start_date",
                "order_dir": "asc",
            },
        )
        for lo, hi in zip(bounds, bounds[1:])
    ]
    raw = fetch_chunked(config, "dld_rent_contracts-open-api", chunks,
                        max_records=1_000_000, transform=_slim_rents)
    if raw.is_empty():
        raise RuntimeError("rents pull returned no records; not overwriting GCS")

    n0 = raw.height
    rents = raw.drop("_raw_row").drop_nulls(["start", "annual_amount", "AREA_EN"])
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
    """CLI entry point: pull the datasets selected via --only."""
    parser = argparse.ArgumentParser(description=__doc__)
    # saleindex is opt-in only: the gateway dataset is frozen at 2024-05 and
    # no model feature consumes it — no point spending a pull on it by default.
    parser.add_argument(
        "--only", nargs="*",
        default=["projects", "buildings", "service", "rents", "units"],
        choices=["projects", "buildings", "service", "rents", "units", "saleindex"],
    )
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
    if "units" in args.only:
        pull_units(config, secrets)
    if "saleindex" in args.only:
        pull_sale_index(config, secrets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
