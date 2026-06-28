import argparse
import sys
import uuid

import polars as pl

from gcs_storage import dataframe_to_parquet_bytes, gcs_client, load_local_secrets, setting


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write, read, verify, and optionally delete a tiny GCS Parquet file."
    )
    parser.add_argument("--bucket", help="GCS bucket name. Defaults to GCS_BUCKET.")
    parser.add_argument("--prefix", help="Object prefix. Defaults to GCS_TEST_PREFIX or smoke-tests.")
    parser.add_argument("--keep-object", action="store_true")
    args = parser.parse_args()

    secrets = load_local_secrets()
    bucket_name = args.bucket or setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    if not bucket_name:
        print("Missing bucket name. Pass --bucket or set GCS_BUCKET.")
        return 2

    prefix = (args.prefix or setting(secrets, "GCS_TEST_PREFIX") or "smoke-tests").strip("/")
    expected = pl.DataFrame({"id": [1, 2], "label": ["write", "read"]})
    parquet_bytes = dataframe_to_parquet_bytes(expected)
    object_name = f"{prefix}/gcs_smoke_{uuid.uuid4().hex}.parquet"

    try:
        blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)

        print(f"Writing gs://{bucket_name}/{object_name}")
        blob.upload_from_string(parquet_bytes, content_type="application/vnd.apache.parquet")

        print("Reading object back")
        import io

        actual = pl.read_parquet(io.BytesIO(blob.download_as_bytes()))

        if actual.to_dict(as_series=False) != expected.to_dict(as_series=False):
            print("Round-trip verification failed.")
            return 1

        print(f"Round-trip verified: {actual.height} rows, {len(parquet_bytes):,} bytes")

        if args.keep_object:
            print("Kept test object.")
        else:
            blob.delete()
            print("Deleted test object.")
    except Exception as exc:
        print(f"GCS smoke test failed: {type(exc).__name__}: {exc}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
