"""Pull DLD reference datasets and publish them to GCS under dld_reference/.

Reference data feeds the fair-value model's project_meta / rent_yield /
service_charge feature groups:

- projects.parquet            full projects table (developer, completion dates)
- project_buildings_agg.parquet   buildings aggregated per project_id
- service_charges.parquet     latest budget-year service cost per project
- rent_index.parquet          weekly area x rooms-band grid of trailing-180d
                              median rent PSF (strictly past per week)
- rent_weekly_stats.parquet   weekly area x rooms-band percentile stats of
                              rent PSF (n/median/q1/q3/p10/p90, incl. an
                              "All" band and a segment column) for the Rent
                              Opportunity Scanner's box view
- rent_recent_contracts.parquet  contract-level slim rows for the 20 scanner
                              districts, trailing RENT_RECENT_DAYS, for the
                              scanner's dot view and deals table

Usage:
    python -m ingestion.store_reference_data_gcs --only projects buildings service
    python -m ingestion.store_reference_data_gcs --probe-rents    # print raw Ejari columns (minutes)
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
from datetime import date, timedelta

import polars as pl

from dashboard_constants import AREA_DISPLAY, NEIGHBORHOODS, SQM_TO_SQFT
from ingestion.dda_api import DDAConfig, fetch_dataset_records, load_dda_config
from ingestion.gcs_storage import dataframe_to_parquet_bytes, gcs_client, load_local_secrets, setting

REFERENCE_PREFIX = "dld_reference"
RENTS_START = date(2024, 1, 1)

# Ejari sanitization bounds (probe found corrupt years 2034/2109/2204 and
# garbage tiny areas).
RENT_YEAR_MIN, RENT_YEAR_MAX = 2020, 2027
RENT_ANNUAL_MIN, RENT_ANNUAL_MAX = 5_000, 5_000_000
RENT_AREA_MIN, RENT_AREA_MAX = 15.0, 1_000.0  # sqm

# Contract-level rent artifact: trailing window and the 20 scanner districts
# (official DLD names whose display name is in the sidebar NEIGHBORHOODS
# list) — restricted so the dashboard's resident memory stays bounded.
RENT_RECENT_DAYS = 183
RENT_DISTRICTS = sorted(
    district
    for district, display in AREA_DISPLAY.items()
    if display in set(NEIGHBORHOODS)
)

# Raw Ejari fields --probe-rents highlights; names unverified until probed.
RENT_PROBE_CANDIDATES = (
    "contract_reg_type_en",
    "contract_end_date",
    "contract_amount",
    "ejari_contract_number",
    "no_of_prop",
    "version",
    "project_name_en",
    "master_project_en",
)


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


# The feed writes both spaced and unspaced variants ("2 bed rooms+hall" and
# "1bed room+Hall") — the unspaced 1BR label alone was 41% of flats in the
# 2026-07 audit sample, so every count needs both needles.
ROOMS_BAND_MAP = {
    "studio": "Studio",
    "1 bed": "1BR", "one bed": "1BR", "1bed": "1BR",
    "2 bed": "2BR", "two bed": "2BR", "2bed": "2BR",
    "3 bed": "3BR", "three bed": "3BR", "3bed": "3BR",
    "4 bed": "4BR+", "five": "4BR+", "5 bed": "4BR+", "penthouse": "4BR+",
    "4bed": "4BR+", "5bed": "4BR+", "6 bed": "4BR+", "7 bed": "4BR+",
    "8 bed": "4BR+",
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
    """Reduce a raw Ejari chunk to the columns the rent artifacts need."""
    cols = [
        pl.col("contract_start_date").cast(pl.Utf8).str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False).alias("start"),
        pl.col("annual_amount").cast(pl.Float64, strict=False),
        pl.col("actual_area").cast(pl.Float64, strict=False),
        # strip_chars matches the sales path (dda_api normalization) so the
        # rent and sales AREA_EN vocabularies join cleanly.
        pl.col("area_name_en").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
        .alias("AREA_EN"),
        rooms_band_expr().alias("rooms_band"),
        pl.col("ejari_property_type_en").cast(pl.Utf8).alias("ptype"),
        # Row identity for pagination dedupe, plus the bulk-contract flag
        # (IAAO: allocated bulk amounts are not unit prices).
        pl.col("contract_id").cast(pl.Utf8),
        pl.col("line_number").cast(pl.Int64, strict=False),
        pl.col("no_of_prop").cast(pl.Int64, strict=False),
        pl.lit(1).alias("_raw_row"),
    ]
    # New-vs-renewal flag: RERA caps renewal rents below market, so the rent
    # scanner needs to separate them. Presence-guarded — confirm the raw
    # field name with --probe-rents; fetch_chunked's diagonal_relaxed concat
    # tolerates chunks with and without the column.
    if "contract_reg_type_en" in chunk.columns:
        cols.append(
            pl.col("contract_reg_type_en").cast(pl.Utf8).str.strip_chars()
            .alias("reg_type")
        )
    # Project identity (~26% populated per the data-quality audit): the
    # direct half of the rent-to-project linkage behind rent_project_index.
    # The raw feed carries a first-class project_number (probe 2026-08) —
    # the same key as the sales snapshot — plus the name as fallback.
    if "project_number" in chunk.columns:
        cols.append(
            pl.col("project_number").cast(pl.Int64, strict=False)
            .alias("ejari_project_number")
        )
    if "project_name_en" in chunk.columns:
        cols.append(
            pl.col("project_name_en").cast(pl.Utf8).str.to_uppercase()
            .str.strip_chars().alias("project_name")
        )
    return chunk.select(cols)


def _weekly_stats(rents: pl.DataFrame) -> pl.DataFrame:
    """Per AREA_EN x rooms_band x week percentile stats, incl. an "All" band.

    Quantiles are not composable across bands, so the "All" rows (which also
    absorb null-band contracts) must be computed here, not in the app. The
    segment column separates "all" contracts from "new" (non-renewal) ones
    when the reg_type flag is available.
    """
    def stats(df: pl.DataFrame, band: pl.Expr, segment: str) -> pl.DataFrame:
        return (
            df.group_by("AREA_EN", band.alias("rooms_band"), "week")
            .agg(
                pl.len().alias("n"),
                pl.col("rent_psf").median().round(1).alias("median"),
                pl.col("rent_psf").quantile(0.25).round(1).alias("q1"),
                pl.col("rent_psf").quantile(0.75).round(1).alias("q3"),
                pl.col("rent_psf").quantile(0.10).round(1).alias("p10"),
                pl.col("rent_psf").quantile(0.90).round(1).alias("p90"),
            )
            .with_columns(pl.lit(segment).alias("segment"))
        )

    frames = [
        stats(rents.filter(pl.col("rooms_band").is_not_null()), pl.col("rooms_band"), "all"),
        stats(rents, pl.lit("All"), "all"),
    ]
    if "reg_type" in rents.columns:
        # Anchored match: "Renew"/"Renewal" also *contain* "new", so only a
        # leading "New" identifies a genuinely new contract.
        new = rents.filter(
            pl.col("reg_type").str.contains(r"(?i)^new").fill_null(False)
        )
        frames.append(
            stats(new.filter(pl.col("rooms_band").is_not_null()), pl.col("rooms_band"), "new")
        )
        frames.append(stats(new, pl.lit("All"), "new"))
    return pl.concat(frames).sort("AREA_EN", "rooms_band", "week")


def _recent_contracts(rents: pl.DataFrame) -> pl.DataFrame:
    """Contract-level slim rows: 20 scanner districts, trailing RENT_RECENT_DAYS."""
    cutoff = date.today() - timedelta(days=RENT_RECENT_DAYS)
    # contract_id keeps row identity honest: .unique() below then removes
    # only true repeats, never two genuinely identical units let the same day.
    cols = ["start", "AREA_EN", "rooms_band", "size_sqft", "annual_amount",
            "rent_psf", "contract_id"]
    if "reg_type" in rents.columns:
        cols.append("reg_type")
    return (
        rents.filter(
            pl.col("AREA_EN").is_in(RENT_DISTRICTS) & (pl.col("start") >= cutoff)
        )
        .with_columns((pl.col("actual_area") * SQM_TO_SQFT).round(0).alias("size_sqft"))
        .select(cols)
        .unique()  # versioned/repeated gateway rows: exact-duplicate dedupe
        .sort("AREA_EN", "start")
    )


def _slim_permits(chunk: pl.DataFrame) -> pl.DataFrame:
    """Reduce a raw permits chunk to delivered adjustment/addition events."""
    return (
        chunk.select(
            pl.col("parcel_id").cast(pl.Int64, strict=False),
            pl.col("permit_date").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False).alias("permit_date"),
            pl.col("application_type_english").cast(pl.Utf8).alias("permit_type"),
            pl.col("building_type").cast(pl.Utf8),
        )
        .drop_nulls(["parcel_id", "permit_date"])
        .filter(
            pl.col("permit_type").str.contains(r"(?i)adjust|addition")
            & ~pl.col("permit_type").str.contains(r"(?i)cancel")
        )
        .unique()
    )


def pull_permits(config: "DDAConfig", secrets: dict) -> None:
    """Publish per-project renovation-permit events (dm_building_permits).

    Dubai Municipality publishes building permits with ``parcel_id`` and
    permit type; ``dld_buildings`` carries the same ``parcel_id`` next to
    ``project_id``, and the projects table bridges to ``project_number`` —
    the key the sales snapshot speaks. Only adjustment/addition permits with
    a delivered ``permit_date`` are kept: they signal renovation investment
    in the building, a candidate unit-condition proxy for the fair-value
    model (campaign 4).
    """
    from ingestion.gcs_storage import read_reference_frames

    # Cheap stages first: the parcel map and the projects bridge take
    # minutes, the permits pull takes an hour. Fetching them up front means
    # a failure here costs minutes instead of discarding a completed permits
    # pull (measured 2026-08, when a truncated buildings page did exactly that).
    projects = read_reference_frames(secrets, ["projects"])["projects"].select(
        pl.col("project_id").cast(pl.Int64, strict=False),
        pl.col("project_number").cast(pl.Int64, strict=False),
    ).drop_nulls().unique("project_id")
    buildings = fetch_dataset(
        config,
        "dld_buildings-open-api",
        params={"order_by": "property_id", "order_dir": "asc"},
        max_records=2_000_000,
    )
    parcel_map = (
        buildings.select(
            pl.col("parcel_id").cast(pl.Int64, strict=False),
            pl.col("project_id").cast(pl.Int64, strict=False),
        )
        .drop_nulls()
        .unique()
    )
    print(
        f"parcel map: {parcel_map.height:,} parcel-project pairs covering "
        f"{parcel_map['project_id'].n_unique():,} projects",
        flush=True,
    )

    # Yearly chunks: one token per chunk (a single multi-hour fetch expires
    # the gateway bearer token mid-pagination — measured 2026-08), visible
    # progress, and a chunk-level retry. 2010 onward is ample margin for the
    # trailing-5y feature window; stable unique-key ordering within chunks.
    this_year = date.today().year
    chunks = [
        (
            str(year),
            {
                "filter": (
                    f"application_submission_date>='{year}-01-01' AND "
                    f"application_submission_date<'{year + 1}-01-01'"
                ),
                "order_by": "application_id",
                "order_dir": "asc",
            },
        )
        for year in range(2010, this_year + 1)
    ]
    permits = fetch_chunked(
        replace(config, entity="dm"),
        "dm_building_permits-open-api",
        chunks,
        max_records=400_000,
        transform=_slim_permits,
    )
    if permits.is_empty():
        raise RuntimeError("permits pull returned no records; not overwriting GCS")
    permits = permits.unique()
    print(f"adjustment/addition permits with a permit_date: {permits.height:,}", flush=True)

    events = (
        permits.join(parcel_map, on="parcel_id", how="inner")
        .join(projects, on="project_id", how="inner")
        .select("project_number", "permit_date", "permit_type", "building_type")
        .unique()
        .sort("project_number", "permit_date")
    )
    print(
        f"permit-project linkage: {events.height:,} events on "
        f"{events['project_number'].n_unique():,} projects "
        f"(parcel join matched {permits.join(parcel_map, on='parcel_id', how='inner').height:,} "
        f"of {permits.height:,} permits)",
        flush=True,
    )
    upload(secrets, "modification_permits.parquet", events)


def _link_rent_projects(rents: pl.DataFrame, secrets: dict) -> pl.DataFrame | None:
    """Attach ``project_number`` to rent contracts via two measured routes.

    Route A (direct): the feed's own ``project_number`` (registry-validated),
    falling back to ``project_name_en`` joined to the projects table name
    (~26% of contracts carry project identity). Route B (fingerprint): Dubai layouts
    have near-unique exact registered areas, so (district, area to 2dp sqm,
    rooms band) resolved against the units registry identifies the project
    for ~62% of units at 97% accuracy (validated on 315k labeled sales,
    2026-08). B fills contracts A cannot; keys matching multiple projects
    stay unlinked rather than guessed.

    Needs the projects + units artifacts in GCS (run ``--only projects
    units`` first); returns None with a warning when they are missing.
    """
    from ingestion.gcs_storage import read_reference_frames

    try:
        ref = read_reference_frames(secrets, ["projects", "units"])
    except FileNotFoundError as exc:
        print(f"rent-project linkage skipped: {exc}", flush=True)
        return None

    projects = (
        ref["projects"]
        .select(
            pl.col("project_id").cast(pl.Int64, strict=False),
            pl.col("project_number").cast(pl.Int64, strict=False),
            pl.col("project_name").cast(pl.Utf8).str.to_uppercase()
            .str.strip_chars().alias("project_name"),
            pl.col("area_name_en").cast(pl.Utf8).str.to_uppercase()
            .str.strip_chars().alias("AREA_EN"),
        )
        .drop_nulls(["project_id", "project_number"])
        .unique("project_id")
    )

    # Route A0: the feed's own project_number, kept only when the registry
    # knows it (guards against placeholder/garbage ids).
    if "ejari_project_number" in rents.columns:
        valid = projects.select(
            pl.col("project_number").alias("ejari_project_number"),
            pl.col("project_number").alias("_pn_raw"),
        ).unique("ejari_project_number")
        rents = rents.join(valid, on="ejari_project_number", how="left")
    else:
        rents = rents.with_columns(pl.lit(None, dtype=pl.Int64).alias("_pn_raw"))

    # Route A: direct name join (project names are unique in the registry).
    name_lut = (
        projects.drop_nulls("project_name")
        .group_by("project_name")
        .agg(pl.col("project_number").n_unique().alias("_n"),
             pl.col("project_number").first().alias("_pn_name"))
        .filter(pl.col("_n") == 1)
        .drop("_n")
    )
    if "project_name" in rents.columns:
        rents = rents.join(name_lut, on="project_name", how="left")
    else:
        rents = rents.with_columns(pl.lit(None, dtype=pl.Int64).alias("_pn_name"))

    # Route B: (district, exact area @2dp, rooms band) -> unique project.
    fingerprint = (
        ref["units"]
        .select(
            pl.col("project_id").cast(pl.Int64, strict=False),
            ((pl.col("actual_area").cast(pl.Float64, strict=False)) * 100)
            .round(0).cast(pl.Int64).alias("_akey"),
            rooms_band_from_units().alias("rooms_band"),
        )
        .join(projects.select("project_id", "project_number", "AREA_EN"),
              on="project_id", how="inner")
        .group_by("AREA_EN", "_akey", "rooms_band")
        .agg(pl.col("project_number").n_unique().alias("_n"),
             pl.col("project_number").first().alias("_pn_fp"))
        .filter(pl.col("_n") == 1)
        .drop("_n")
    )
    rents = rents.with_columns(
        (pl.col("actual_area") * 100).round(0).cast(pl.Int64).alias("_akey")
    ).join(fingerprint, on=["AREA_EN", "_akey", "rooms_band"], how="left")

    n = rents.height
    direct = pl.coalesce([pl.col("_pn_raw"), pl.col("_pn_name")])
    both = rents.filter(direct.is_not_null() & pl.col("_pn_fp").is_not_null())
    agree = both.filter(direct == pl.col("_pn_fp"))
    for route, col in (("raw number", "_pn_raw"), ("name", "_pn_name"),
                       ("fingerprint", "_pn_fp")):
        print(f"  route {route}: {rents.get_column(col).drop_nulls().len() / n:.1%}",
              flush=True)
    rents = rents.with_columns(direct.fill_null(pl.col("_pn_fp")).alias("project_number"))
    rents = rents.drop("_akey", "_pn_raw", "_pn_name", "_pn_fp")
    linked = rents.filter(pl.col("project_number").is_not_null())
    print(
        f"rent-project linkage: {linked.height:,}/{n:,} contracts linked "
        f"({linked.height / n:.1%}); direct routes agree with the fingerprint on "
        f"{(agree.height / both.height if both.height else 0):.1%} of "
        f"{both.height:,} doubly-linked rows",
        flush=True,
    )
    return rents


def rooms_band_from_units() -> pl.Expr:
    """Units-registry rooms_en (e.g. '1 B/R') mapped to the dashboard bands."""
    rooms = pl.col("rooms_en").cast(pl.Utf8)
    digit = rooms.str.extract(r"(\d+)").cast(pl.Int32, strict=False)
    return (
        pl.when(rooms.str.to_lowercase().str.contains("studio")).then(pl.lit("Studio"))
        .when(digit == 1).then(pl.lit("1BR"))
        .when(digit == 2).then(pl.lit("2BR"))
        .when(digit == 3).then(pl.lit("3BR"))
        .when(digit >= 4).then(pl.lit("4BR+"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _project_rent_index(linked: pl.DataFrame) -> pl.DataFrame:
    """Weekly per-project trailing-180d rent stats, strictly past.

    Mirrors the district grid construction: the value at week w covers only
    contracts starting before w (rolling closed="left" over weekly medians),
    so the model's backward as-of join stays lookahead-free. PSF pools rooms
    bands within the project — per-sqft normalizes the size mix.
    """
    weekly = (
        linked.filter(pl.col("project_number").is_not_null())
        .group_by("project_number", "week")
        .agg(pl.col("rent_psf").median().alias("wk_med"), pl.len().alias("wk_n"))
        .sort("week")
    )
    return (
        weekly.with_columns(
            pl.col("wk_med")
            .rolling_mean_by("week", window_size="180d", closed="left")
            .over("project_number")
            .alias("project_rent_psf_180d"),
            pl.col("wk_n")
            .rolling_sum_by("week", window_size="180d", closed="left")
            .over("project_number")
            .alias("project_rent_contracts_180d"),
        )
        .drop_nulls(["project_rent_psf_180d"])
        .select("project_number", "week",
                "project_rent_psf_180d", "project_rent_contracts_180d")
    )


def probe_rents(config: "DDAConfig") -> int:
    """Print the raw Ejari column names from a 5-record sample.

    Run before the long rents pull to confirm the optional field names
    (new-vs-renewal flag, contract end date) that _slim_rents keeps when
    present. Needs DDA credentials only, no GCS.
    """
    cfg = replace(config, dataset="dld_rent_contracts-open-api")
    records = fetch_dataset_records(
        cfg,
        params={
            "filter": "property_usage_en='Residential'",
            "order_by": "contract_start_date",
            "order_dir": "desc",
        },
        page_size=5,
        max_records=5,
    )
    if not records:
        print("No records returned; cannot probe columns.")
        return 1
    names = sorted({key for record in records for key in record})
    print(f"Raw Ejari columns ({len(names)}):")
    for name in names:
        print(f"  {name}")
    matched = [c for c in RENT_PROBE_CANDIDATES if c in names]
    print("Candidate fields matched: " + (", ".join(matched) if matched else "none"))
    print("Sample record (values truncated to 60 chars):")
    for name in names:
        print(f"  {name} = {str(records[0].get(name, ''))[:60]}")
    return 0


def pull_rents(config: "DDAConfig", secrets: dict) -> None:
    """Publish the rent grid plus the Rent Opportunity Scanner artifacts.

    One fetch, three uploads: rent_index.parquet (model feature grid,
    unchanged schema), rent_weekly_stats.parquet (box-view percentiles),
    rent_recent_contracts.parquet (dot-view contract rows).
    """
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
                # contract_id ordering is STABLE across page requests.
                # Ordering by contract_start_date (thousands of ties per day)
                # shuffled page boundaries between requests: measured on
                # 2026-07 it duplicated ~25% of rows AND silently skipped a
                # different ~25% of contracts (rent_data_quality_report.md).
                "order_by": "contract_id",
                "order_dir": "asc",
            },
        )
        for lo, hi in zip(bounds, bounds[1:])
    ]
    raw = fetch_chunked(config, "dld_rent_contracts-open-api", chunks,
                        max_records=1_000_000, transform=_slim_rents)
    if raw.is_empty():
        raise RuntimeError("rents pull returned no records; not overwriting GCS")

    # Belt-and-braces after the stable ordering: only boundary repeats remain.
    n_fetched = raw.height
    raw = raw.unique(subset=["contract_id", "line_number"], keep="first")
    if raw.height != n_fetched:
        print(f"pagination dedupe: {n_fetched:,} -> {raw.height:,} rows", flush=True)

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
        # IAAO: allocated bulk amounts are disqualified — no_of_prop>1 rows
        # stamp the contract TOTAL against a single unit's area (measured
        # median PSF 17x the single-unit median in the audit sample).
        & (pl.col("no_of_prop") <= 1)
    ).with_columns(
        (pl.col("annual_amount") / (pl.col("actual_area") * SQM_TO_SQFT)).alias("rent_psf"),
        pl.col("start").dt.truncate("1w").alias("week"),
    )
    # Feed quirk: ~0.1% of single-property contracts carry duplicate lines
    # (same lease registered twice with line_number 1..n despite no_of_prop=1).
    # One contract = one lease observation, so keep one row per contract_id —
    # this also keeps rent_weekly_stats consistent with the deduped
    # rent_recent_contracts artifact.
    rents = rents.unique(subset=["contract_id"], keep="first")
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
    # rent_index first: it feeds the fair-value model, so a failure in the
    # newer scanner artifacts below still leaves the model's grid refreshed.
    upload(secrets, "rent_index.parquet", grid)
    upload(secrets, "rent_weekly_stats.parquet", _weekly_stats(rents))
    upload(secrets, "rent_recent_contracts.parquet", _recent_contracts(rents))
    # Project-linked rent index (rent_project model feature). Last so a
    # linkage failure never costs the three artifacts above.
    linked = _link_rent_projects(rents, secrets)
    if linked is not None:
        upload(secrets, "rent_project_index.parquet", _project_rent_index(linked))


def main() -> int:
    """CLI entry point: pull the datasets selected via --only."""
    parser = argparse.ArgumentParser(description=__doc__)
    # saleindex is opt-in only: the gateway dataset is frozen at 2024-05 and
    # no model feature consumes it — no point spending a pull on it by default.
    parser.add_argument(
        "--only", nargs="*",
        default=["projects", "buildings", "service", "rents", "units"],
        choices=["projects", "buildings", "service", "rents", "units",
                 "saleindex", "permits"],
    )
    parser.add_argument(
        "--probe-rents", action="store_true",
        help="Fetch 5 raw Ejari records, print the available columns, and exit",
    )
    args = parser.parse_args()

    secrets = load_local_secrets()
    config = load_dda_config(secrets)
    if config.missing_fields():
        print("Missing DDA configuration: " + ", ".join(config.missing_fields()))
        return 2

    if args.probe_rents:
        return probe_rents(config)

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
    if "permits" in args.only:
        pull_permits(config, secrets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
