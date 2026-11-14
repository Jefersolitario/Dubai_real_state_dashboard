import argparse
import sys
from datetime import UTC, date, datetime

import polars as pl

from ingestion.dda_api import (
    DDAConfig,
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_MAX_RECORDS,
    DEFAULT_PAGE_SIZE,
    DDAApiError,
    build_dld_transactions_params,
    fetch_dataset_records,
    last_months_date_range,
    load_dda_config,
    normalize_dld_transactions,
    validate_normalized_columns,
)
from ingestion.gcs_storage import (
    DEFAULT_TRANSACTIONS_OBJECT,
    configured_snapshot,
    dataframe_to_parquet_bytes,
    gcs_client,
    load_local_secrets,
    read_parquet_object,
)


IGNORED_DEDUPE_COLUMNS = {"load_timestamp"}
# DLD's ``transaction_id``: unique per row of the feed, so it identifies a sale
# across fetches even after DLD re-dates or amends it.
TRANSACTION_KEY = "TRANSACTION_NUMBER"


def main() -> int:
    """CLI entry point: incremental (default) or --full-refresh snapshot update."""
    parser = argparse.ArgumentParser(
        description="Incrementally update the DLD transactions GCS Parquet snapshot."
    )
    parser.add_argument("--bucket", help="GCS bucket name. Defaults to GCS_BUCKET.")
    parser.add_argument("--object", help="GCS object name for the Parquet snapshot.")
    parser.add_argument("--limit", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--last-months", type=int, default=DEFAULT_LOOKBACK_MONTHS)
    parser.add_argument("--start-date", type=parse_date)
    parser.add_argument("--end-date", type=parse_date)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Replace the snapshot from the requested API window instead of merging with GCS.",
    )
    args = parser.parse_args()

    secrets = load_local_secrets()
    default_bucket, default_object = configured_snapshot(secrets)
    bucket_name = args.bucket or default_bucket
    object_name = args.object or default_object or DEFAULT_TRANSACTIONS_OBJECT
    if not bucket_name:
        print("Missing bucket name. Pass --bucket or set GCS_BUCKET.")
        return 2

    config = load_dda_config(secrets)
    missing = config.missing_fields()
    if missing:
        print("Missing DDA configuration values: " + ", ".join(missing))
        return 2

    try:
        existing = None if args.full_refresh else load_existing_snapshot(
            secrets,
            bucket_name,
            object_name,
        )
        start_date, end_date = api_window(args, existing)
        incoming = fetch_api_snapshot(config, start_date, end_date, args.limit)

        final, stats = merge_snapshots(existing, incoming, args.full_refresh)
        if final.is_empty():
            print("No records available to write.")
            return 1
        if not stats.get("write_required"):
            print(
                "GCS snapshot already current: "
                f"{final.height:,} rows, "
                f"{date_bound(final, 'min')} to {date_bound(final, 'max')}."
            )
            return 0

        write_snapshot(secrets, bucket_name, object_name, final, stats)
    except (DDAApiError, ValueError) as exc:
        print(f"Update failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Update failed: {type(exc).__name__}: {exc}")
        return 1

    return 0


def load_existing_snapshot(secrets: dict, bucket_name: str, object_name: str) -> pl.DataFrame | None:
    """Current GCS snapshot as a normalized frame, or None when absent."""
    try:
        df, blob = read_parquet_object(secrets, bucket_name, object_name)
    except FileNotFoundError:
        print(f"No existing snapshot at gs://{bucket_name}/{object_name}; creating one.")
        return None

    df = validate_snapshot(
        "Existing GCS snapshot",
        normalize_dld_transactions(df),
    )
    print(
        "Existing GCS snapshot: "
        f"{df.height:,} rows, "
        f"{date_bound(df, 'min')} to {date_bound(df, 'max')}, "
        f"updated {blob.updated.isoformat() if blob.updated else 'unknown'}"
    )
    return df


def api_window(args: argparse.Namespace, existing: pl.DataFrame | None) -> tuple[date | None, date | None]:
    """(start, end) dates to fetch from the API for this run."""
    if args.start_date or args.end_date:
        return args.start_date, args.end_date or date.today()

    if existing is not None and not existing.is_empty():
        return parse_date(date_bound(existing, "max")), date.today()

    return last_months_date_range(args.last_months)


def fetch_api_snapshot(
    config: DDAConfig,
    start_date: date | None,
    end_date: date | None,
    limit: int,
) -> pl.DataFrame:
    """Fetch and normalize DLD transactions for the requested window."""
    if start_date and end_date and start_date > end_date:
        print(f"Skipping API fetch; start date {start_date} is after end date {end_date}.")
        return pl.DataFrame()

    print(f"Fetching DLD records for {start_date} to {end_date}")
    max_records = max(limit, 1)
    records = fetch_dataset_records(
        config,
        params=build_dld_transactions_params(start_date, end_date, order_desc=True),
        page_size=min(DEFAULT_PAGE_SIZE, max_records),
        max_records=max_records,
    )
    df = normalize_dld_transactions(records)
    if df.is_empty():
        print("API returned no records for the incremental window.")
        return df
    return validate_snapshot("Fetched API snapshot", df)


def merge_snapshots(
    existing: pl.DataFrame | None,
    incoming: pl.DataFrame,
    full_refresh: bool,
) -> tuple[pl.DataFrame, dict[str, int | str]]:
    """Combine existing + fetched rows into one row per transaction, with stats.

    Fetched rows go after the stored ones, so a sale DLD has re-dated or
    amended since it was stored is replaced by its fetched version
    (``updated_rows``) rather than kept alongside it.
    """
    if full_refresh or existing is None:
        final = dedupe_snapshot(incoming)
        return stable_sort(final), {
            "mode": "full_refresh" if full_refresh else "initial_load",
            "write_required": True,
            "existing_rows": 0,
            "fetched_rows": incoming.height,
            "added_rows": final.height,
            "updated_rows": 0,
            "existing_duplicate_rows_removed": 0,
            "incoming_duplicate_rows_removed": incoming.height - final.height,
            "overlap_duplicate_rows": 0,
        }

    existing_deduped = dedupe_snapshot(existing)
    incoming_deduped = dedupe_snapshot(incoming)
    combined = pl.concat([existing_deduped, incoming_deduped], how="diagonal_relaxed")
    final = stable_sort(dedupe_snapshot(combined))
    added_rows, updated_rows = count_changes(existing_deduped, final)
    existing_duplicate_rows_removed = existing.height - existing_deduped.height
    incoming_duplicate_rows_removed = incoming.height - incoming_deduped.height
    overlap_duplicate_rows = existing_deduped.height + incoming_deduped.height - final.height

    return final, {
        "mode": "incremental",
        "write_required": (
            added_rows > 0 or updated_rows > 0 or existing_duplicate_rows_removed > 0
        ),
        "existing_rows": existing.height,
        "fetched_rows": incoming.height,
        "added_rows": added_rows,
        "updated_rows": updated_rows,
        "existing_duplicate_rows_removed": existing_duplicate_rows_removed,
        "incoming_duplicate_rows_removed": incoming_duplicate_rows_removed,
        "overlap_duplicate_rows": overlap_duplicate_rows,
    }


def write_snapshot(secrets: dict, bucket_name: str, object_name: str, df: pl.DataFrame, stats: dict) -> None:
    validation = validate_normalized_columns(df)
    if validation["missing_required"]:
        raise ValueError("Missing required columns: " + ", ".join(validation["missing_required"]))

    blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)
    blob.metadata = {
        "row_count": str(df.height),
        "date_start": date_bound(df, "min"),
        "date_end": date_bound(df, "max"),
        "stored_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **{key: str(value) for key, value in stats.items()},
    }

    print(
        f"Writing {df.height:,} rows to gs://{bucket_name}/{object_name} "
        f"({stats['mode']}, added {stats['added_rows']:,}, "
        f"updated {stats['updated_rows']:,}, "
        f"cleaned {stats['existing_duplicate_rows_removed']:,} existing duplicates)"
    )
    blob.upload_from_string(
        dataframe_to_parquet_bytes(df),
        content_type="application/vnd.apache.parquet",
    )

    print("Reading object back for verification")
    readback, _ = read_parquet_object(secrets, bucket_name, object_name)
    readback = normalize_dld_transactions(readback)
    verify_readback(df, readback)
    print(
        "Verified GCS snapshot: "
        f"{readback.height:,} rows, "
        f"{len(readback.columns)} columns, "
        f"{date_bound(readback, 'min')} to {date_bound(readback, 'max')}"
    )


def validate_snapshot(label: str, df: pl.DataFrame) -> pl.DataFrame:
    """Fail fast when required dashboard columns are missing; returns df."""
    validation = validate_normalized_columns(df)
    if df.is_empty():
        raise ValueError(f"{label} is empty.")
    if validation["missing_required"]:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(validation["missing_required"])
        )
    return df


def parse_date(value: str) -> date:
    """YYYY-MM-DD string to date (argparse converter)."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def stable_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Deterministic ordering (date, then keys) so rewrites diff cleanly."""
    columns = ["INSTANCE_DATE"]
    if "TRANSACTION_NUMBER" in df.columns:
        columns.append("TRANSACTION_NUMBER")
    return df.sort(columns, nulls_last=True)


def dedupe_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    """One row per transaction; of several versions the last in frame order wins.

    DLD re-dates and amends sales after publishing them, so one transaction
    number can arrive more than once with different content. Callers put the
    most recently fetched rows last. A stored snapshot is date-sorted and DLD
    only moves dates forward (341 of 341 measured against the feed,
    2026-11), so there too the current version is the last one. Rows without
    a transaction number only lose exact duplicates.
    """
    if df.is_empty():
        return df
    df = df.unique(subset=dedupe_key_columns(df), keep="last", maintain_order=True)
    if TRANSACTION_KEY not in df.columns:
        return df
    number = pl.col(TRANSACTION_KEY)
    return df.filter(number.is_null() | number.is_last_distinct())


def dedupe_key_columns(df: pl.DataFrame) -> list[str]:
    """Columns compared when looking for exact duplicates (all present ones)."""
    return [
        column
        for column in df.columns
        if column not in IGNORED_DEDUPE_COLUMNS
    ]


def count_changes(existing: pl.DataFrame, final: pl.DataFrame) -> tuple[int, int]:
    """(added, updated) rows of ``final`` relative to the deduplicated ``existing``.

    A row whose transaction number ``existing`` already holds is an update —
    DLD re-dated or amended that sale; any other row ``existing`` lacks is new.
    """
    if existing.is_empty():
        return final.height, 0

    shared = [column for column in dedupe_key_columns(final) if column in existing.columns]
    # Row hashes depend on dtype (Int64 1 != Float64 1.0), and the merge may
    # have widened a column, so compare both frames under final's schema.
    schema = {column: final.schema[column] for column in shared}
    known_rows = pl.DataFrame(
        {"_row": existing.select(shared).cast(schema).hash_rows()}
    ).unique()
    changed = final.with_columns(
        final.select(shared).hash_rows().alias("_row")
    ).join(known_rows, on="_row", how="anti")
    if TRANSACTION_KEY not in shared:
        return changed.height, 0

    known_numbers = existing.select(TRANSACTION_KEY).drop_nulls().unique()
    updated = changed.join(known_numbers, on=TRANSACTION_KEY, how="semi").height
    return changed.height - updated, updated


def date_bound(df: pl.DataFrame, bound: str) -> str:
    """Min/max INSTANCE_DATE (YYYY-MM-DD) for metadata and log lines."""
    if "INSTANCE_DATE" not in df.columns or df.is_empty():
        return ""
    expression = pl.col("INSTANCE_DATE").min() if bound == "min" else pl.col("INSTANCE_DATE").max()
    value = df.select(expression).item()
    return str(value) if value is not None else ""


def verify_readback(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    """Raise unless the re-downloaded snapshot matches what was written."""
    if actual.height != expected.height:
        raise ValueError(f"Row count mismatch: expected {expected.height}, got {actual.height}")
    if actual.columns != expected.columns:
        raise ValueError("Column mismatch after GCS readback.")
    if actual.height != dedupe_snapshot(actual).height:
        raise ValueError("Readback snapshot holds a transaction on more than one row.")
    if date_bound(actual, "min") != date_bound(expected, "min"):
        raise ValueError("Minimum INSTANCE_DATE mismatch after GCS readback.")
    if date_bound(actual, "max") != date_bound(expected, "max"):
        raise ValueError("Maximum INSTANCE_DATE mismatch after GCS readback.")


if __name__ == "__main__":
    sys.exit(main())
