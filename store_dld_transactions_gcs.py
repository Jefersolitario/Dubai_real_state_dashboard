import argparse
import sys
from datetime import UTC, date, datetime

import polars as pl

from dda_api import (
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
from gcs_storage import (
    DEFAULT_TRANSACTIONS_OBJECT,
    configured_snapshot,
    dataframe_to_parquet_bytes,
    gcs_client,
    load_local_secrets,
    read_parquet_object,
)


IGNORED_DEDUPE_COLUMNS = {"load_timestamp"}


def main() -> int:
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


def load_existing_snapshot(secrets, bucket_name: str, object_name: str) -> pl.DataFrame | None:
    try:
        df, blob = read_parquet_object(secrets, bucket_name, object_name)
    except FileNotFoundError:
        print(f"No existing snapshot at gs://{bucket_name}/{object_name}; creating one.")
        return None

    df = validate_snapshot("Existing GCS snapshot", normalize_dld_transactions(df))
    print(
        "Existing GCS snapshot: "
        f"{df.height:,} rows, "
        f"{date_bound(df, 'min')} to {date_bound(df, 'max')}, "
        f"updated {blob.updated.isoformat() if blob.updated else 'unknown'}"
    )
    return df


def api_window(args, existing: pl.DataFrame | None) -> tuple[date | None, date | None]:
    if args.start_date or args.end_date:
        return args.start_date, args.end_date or date.today()

    if existing is not None and not existing.is_empty():
        return parse_date(date_bound(existing, "max")), date.today()

    return last_months_date_range(args.last_months)


def fetch_api_snapshot(
    config,
    start_date: date | None,
    end_date: date | None,
    limit: int,
) -> pl.DataFrame:
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
    if full_refresh or existing is None:
        final = dedupe_snapshot(incoming)
        return stable_sort(final), {
            "mode": "full_refresh" if full_refresh else "initial_load",
            "write_required": True,
            "existing_rows": 0,
            "fetched_rows": incoming.height,
            "added_rows": final.height,
            "existing_duplicate_rows_removed": 0,
            "incoming_duplicate_rows_removed": incoming.height - final.height,
            "overlap_duplicate_rows": 0,
        }

    existing_deduped = dedupe_snapshot(existing)
    incoming_deduped = dedupe_snapshot(incoming)
    combined = pl.concat([existing_deduped, incoming_deduped], how="diagonal_relaxed")
    final = stable_sort(dedupe_snapshot(combined))
    added_rows = count_new_rows(existing_deduped, final)
    existing_duplicate_rows_removed = existing.height - existing_deduped.height
    incoming_duplicate_rows_removed = incoming.height - incoming_deduped.height
    overlap_duplicate_rows = existing_deduped.height + incoming_deduped.height - final.height

    return final, {
        "mode": "incremental",
        "write_required": added_rows > 0 or existing_duplicate_rows_removed > 0,
        "existing_rows": existing.height,
        "fetched_rows": incoming.height,
        "added_rows": added_rows,
        "existing_duplicate_rows_removed": existing_duplicate_rows_removed,
        "incoming_duplicate_rows_removed": incoming_duplicate_rows_removed,
        "overlap_duplicate_rows": overlap_duplicate_rows,
    }


def write_snapshot(secrets, bucket_name: str, object_name: str, df: pl.DataFrame, stats: dict) -> None:
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
    validation = validate_normalized_columns(df)
    if df.is_empty():
        raise ValueError(f"{label} is empty.")
    if validation["missing_required"]:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(validation["missing_required"])
        )
    return df


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def stable_sort(df: pl.DataFrame) -> pl.DataFrame:
    columns = ["INSTANCE_DATE"]
    if "TRANSACTION_NUMBER" in df.columns:
        columns.append("TRANSACTION_NUMBER")
    return df.sort(columns, nulls_last=True)


def dedupe_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.unique(subset=dedupe_key_columns(df), keep="last", maintain_order=True)


def dedupe_key_columns(df: pl.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in IGNORED_DEDUPE_COLUMNS
    ]


def count_new_rows(existing: pl.DataFrame, final: pl.DataFrame) -> int:
    if existing.is_empty():
        return final.height

    key_columns = dedupe_key_columns(final)
    existing_hashes = key_hashes(existing, key_columns)
    final_hashes = key_hashes(final, key_columns)
    return final_hashes.join(existing_hashes, on="_key", how="anti").height


def key_hashes(df: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    aligned_columns = [column for column in key_columns if column in df.columns]
    return pl.DataFrame({"_key": df.select(aligned_columns).hash_rows()}).unique()


def date_bound(df: pl.DataFrame, bound: str) -> str:
    if "INSTANCE_DATE" not in df.columns or df.is_empty():
        return ""
    expression = pl.col("INSTANCE_DATE").min() if bound == "min" else pl.col("INSTANCE_DATE").max()
    value = df.select(expression).item()
    return str(value) if value is not None else ""


def verify_readback(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    if actual.height != expected.height:
        raise ValueError(f"Row count mismatch: expected {expected.height}, got {actual.height}")
    if actual.columns != expected.columns:
        raise ValueError("Column mismatch after GCS readback.")
    if actual.height != dedupe_snapshot(actual).height:
        raise ValueError("Readback snapshot still contains duplicate rows.")
    if date_bound(actual, "min") != date_bound(expected, "min"):
        raise ValueError("Minimum INSTANCE_DATE mismatch after GCS readback.")
    if date_bound(actual, "max") != date_bound(expected, "max"):
        raise ValueError("Maximum INSTANCE_DATE mismatch after GCS readback.")


if __name__ == "__main__":
    sys.exit(main())
