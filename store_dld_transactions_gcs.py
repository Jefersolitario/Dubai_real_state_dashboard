import argparse
import sys
from datetime import UTC, datetime

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
    dataframe_to_parquet_bytes,
    gcs_client,
    load_local_secrets,
    setting,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store normalized DLD transactions in GCS as Parquet and verify the readback."
    )
    parser.add_argument("--bucket", help="GCS bucket name. Defaults to GCS_BUCKET.")
    parser.add_argument("--object", help="GCS object name for the Parquet snapshot.")
    parser.add_argument("--limit", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--last-months", type=int, default=DEFAULT_LOOKBACK_MONTHS)
    parser.add_argument("--start-date", type=parse_date)
    parser.add_argument("--end-date", type=parse_date)
    args = parser.parse_args()

    secrets = load_local_secrets()
    bucket_name = args.bucket or setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    object_name = (
        args.object
        or setting(secrets, "GCS_TRANSACTIONS_OBJECT")
        or DEFAULT_TRANSACTIONS_OBJECT
    )
    if not bucket_name:
        print("Missing bucket name. Pass --bucket or set GCS_BUCKET.")
        return 2

    config = load_dda_config(secrets)
    missing = config.missing_fields()
    if missing:
        print("Missing DDA configuration values: " + ", ".join(missing))
        return 2

    start_date = args.start_date
    end_date = args.end_date
    if not start_date and not end_date:
        start_date, end_date = last_months_date_range(args.last_months)

    params = build_dld_transactions_params(start_date, end_date, order_desc=True)
    max_records = max(args.limit, 1)

    try:
        print(f"Fetching DLD records for {start_date} to {end_date}")
        records = fetch_dataset_records(
            config,
            params=params,
            page_size=min(DEFAULT_PAGE_SIZE, max_records),
            max_records=max_records,
        )
        df = normalize_dld_transactions(records)
        validation = validate_normalized_columns(df)
        if df.is_empty():
            print("No DLD records returned.")
            return 1
        if validation["missing_required"]:
            print("Missing required columns: " + ", ".join(validation["missing_required"]))
            return 1

        df = stable_sort(df)
        parquet_bytes = dataframe_to_parquet_bytes(df)

        blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)
        blob.metadata = {
            "row_count": str(df.height),
            "date_start": date_bound(df, "min"),
            "date_end": date_bound(df, "max"),
            "stored_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

        print(f"Writing {df.height:,} rows to gs://{bucket_name}/{object_name}")
        blob.upload_from_string(parquet_bytes, content_type="application/vnd.apache.parquet")

        print("Reading object back for verification")
        import io

        readback = pl.read_parquet(io.BytesIO(blob.download_as_bytes()))
        verify_readback(df, readback)

        print(
            "Verified GCS snapshot: "
            f"{readback.height:,} rows, "
            f"{len(readback.columns)} columns, "
            f"{date_bound(readback, 'min')} to {date_bound(readback, 'max')}"
        )
    except (DDAApiError, ValueError) as exc:
        print(f"Store failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Store failed: {type(exc).__name__}: {exc}")
        return 1

    return 0


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def stable_sort(df: pl.DataFrame) -> pl.DataFrame:
    columns = ["INSTANCE_DATE"]
    if "TRANSACTION_NUMBER" in df.columns:
        columns.append("TRANSACTION_NUMBER")
    return df.sort(columns, nulls_last=True)


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
    if date_bound(actual, "min") != date_bound(expected, "min"):
        raise ValueError("Minimum INSTANCE_DATE mismatch after GCS readback.")
    if date_bound(actual, "max") != date_bound(expected, "max"):
        raise ValueError("Maximum INSTANCE_DATE mismatch after GCS readback.")


if __name__ == "__main__":
    sys.exit(main())
