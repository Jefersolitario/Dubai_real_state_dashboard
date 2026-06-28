import io
import json
import os
from pathlib import Path

import polars as pl
from google.cloud import storage
from google.oauth2 import service_account


DEFAULT_TRANSACTIONS_OBJECT = "dld_transactions/dld_transactions_latest.parquet"


def gcs_client(secrets):
    service_account_json = setting(
        secrets,
        "GCP_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    )
    if not service_account_json:
        return storage.Client()

    info = json.loads(service_account_json)
    if "\\n" in info.get("private_key", ""):
        info["private_key"] = info["private_key"].replace("\\n", "\n")

    credentials = service_account.Credentials.from_service_account_info(info)
    return storage.Client(credentials=credentials, project=info.get("project_id"))


def setting(secrets, *names):
    for name in names:
        value = os.getenv(name) or secrets.get(name)
        if value:
            return str(value)
    return ""


def load_local_secrets():
    path = Path(".streamlit/secrets.toml")
    if not path.exists():
        return {}
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def configured_snapshot(secrets):
    bucket_name = setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    object_name = setting(secrets, "GCS_TRANSACTIONS_OBJECT") or DEFAULT_TRANSACTIONS_OBJECT
    return bucket_name, object_name


def read_parquet_object(secrets, bucket_name, object_name):
    blob = gcs_client(secrets).bucket(bucket_name).get_blob(object_name)
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket_name}/{object_name}")
    return pl.read_parquet(io.BytesIO(blob.download_as_bytes())), blob


def dataframe_to_parquet_bytes(df):
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()
