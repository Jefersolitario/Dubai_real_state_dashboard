import io
import json
import os
from pathlib import Path

import polars as pl
from google.cloud import storage
from google.oauth2 import service_account


DEFAULT_TRANSACTIONS_OBJECT = "dld_transactions/dld_transactions_latest.parquet"
DEFAULT_MODEL_OBJECT = "fair_value_model/fair_value_model_latest.pkl"


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


def configured_model_object(secrets):
    bucket_name = setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    object_name = setting(secrets, "GCS_MODEL_OBJECT") or DEFAULT_MODEL_OBJECT
    return bucket_name, object_name


def read_model_bundle_bytes(secrets):
    """Raw bytes of the pre-trained fair-value model bundle, plus its blob."""
    bucket_name, object_name = configured_model_object(secrets)
    blob = gcs_client(secrets).bucket(bucket_name).get_blob(object_name)
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket_name}/{object_name}")
    return blob.download_as_bytes(), blob


def write_model_bundle_bytes(secrets, data, metadata=None):
    """Upload the fair-value model bundle; returns the gs:// URI."""
    bucket_name, object_name = configured_model_object(secrets)
    blob = gcs_client(secrets).bucket(bucket_name).blob(object_name)
    if metadata:
        blob.metadata = {key: str(value) for key, value in metadata.items()}
    blob.upload_from_string(data, content_type="application/octet-stream")
    return f"gs://{bucket_name}/{object_name}"


REFERENCE_OBJECTS = {
    "projects": "dld_reference/projects.parquet",
    "buildings_agg": "dld_reference/project_buildings_agg.parquet",
    "service_charges": "dld_reference/service_charges.parquet",
    "rent_index": "dld_reference/rent_index.parquet",
    "units": "dld_reference/units_slim.parquet",
    "sale_index": "dld_reference/sale_index.parquet",
}


def read_reference_frames(secrets, names):
    """{name: DataFrame} for the requested reference datasets.

    Published by store_reference_data_gcs.py; raises FileNotFoundError with
    that pointer when an object is missing.
    """
    bucket_name = setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
    frames = {}
    for name in names:
        object_name = REFERENCE_OBJECTS[name]
        blob = gcs_client(secrets).bucket(bucket_name).get_blob(object_name)
        if blob is None:
            raise FileNotFoundError(
                f"gs://{bucket_name}/{object_name} — publish it with "
                "store_reference_data_gcs.py"
            )
        frames[name] = pl.read_parquet(io.BytesIO(blob.download_as_bytes()))
    return frames


def read_parquet_object(secrets, bucket_name, object_name):
    blob = gcs_client(secrets).bucket(bucket_name).get_blob(object_name)
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket_name}/{object_name}")
    return pl.read_parquet(io.BytesIO(blob.download_as_bytes())), blob


def dataframe_to_parquet_bytes(df):
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()
