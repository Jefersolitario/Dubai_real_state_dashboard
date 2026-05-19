from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import os
import re

import polars as pl
import requests


DEFAULT_BASE_URL = "https://apis.data.dubai"
STAGING_BASE_URL = "https://stg-apis.data.dubai"
DEFAULT_ENTITY = "dld"
DEFAULT_DATASET = "dld_transactions-open-api"
TOKEN_PATH = "/secure/sdg/ssis/gatewayoauthtoken/1.0.0/getAccessToken"
OAUTH_TOKEN_PATH = "/oauth/client_credential/accesstoken"
OPENAPI_PATH = "/secure/ddads/openapi/1.0.0"

DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_RECORDS = 100_000
DEFAULT_LOOKBACK_MONTHS = 4

REQUIRED_DASHBOARD_COLUMNS = [
    "INSTANCE_DATE",
    "GROUP_EN",
    "IS_OFFPLAN_EN",
    "AREA_EN",
    "PROP_SB_TYPE_EN",
    "TRANS_VALUE",
    "ACTUAL_AREA",
    "ROOMS_EN",
]

OPTIONAL_DASHBOARD_COLUMNS = [
    "TRANSACTION_NUMBER",
    "PROCEDURE_EN",
    "USAGE_EN",
    "PROP_TYPE_EN",
    "PROCEDURE_AREA",
    "PARKING",
    "NEAREST_METRO_EN",
    "NEAREST_MALL_EN",
    "NEAREST_LANDMARK_EN",
    "TOTAL_BUYER",
    "TOTAL_SELLER",
    "MASTER_PROJECT_EN",
    "PROJECT_EN",
    "METER_SALE_PRICE",
]

NUMERIC_COLUMNS = {
    "TRANS_VALUE",
    "ACTUAL_AREA",
    "PROCEDURE_AREA",
    "METER_SALE_PRICE",
    "TOTAL_BUYER",
    "TOTAL_SELLER",
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "TRANSACTION_NUMBER": [
        "TRANSACTION_NUMBER",
        "transaction_id",
        "transaction_number",
        "transaction_no",
    ],
    "INSTANCE_DATE": [
        "INSTANCE_DATE",
        "instance_date",
        "transaction_date",
        "procedure_date",
    ],
    "GROUP_EN": [
        "GROUP_EN",
        "trans_group_en",
        "transaction_group_en",
    ],
    "PROCEDURE_EN": [
        "PROCEDURE_EN",
        "procedure_name_en",
        "procedure_en",
    ],
    "IS_OFFPLAN_EN": [
        "IS_OFFPLAN_EN",
        "reg_type_en",
        "is_offplan_en",
        "property_status_en",
    ],
    "USAGE_EN": [
        "USAGE_EN",
        "property_usage_en",
        "usage_en",
    ],
    "AREA_EN": [
        "AREA_EN",
        "area_name_en",
        "area_en",
        "area_name",
    ],
    "PROP_TYPE_EN": [
        "PROP_TYPE_EN",
        "property_type_en",
        "prop_type_en",
    ],
    "PROP_SB_TYPE_EN": [
        "PROP_SB_TYPE_EN",
        "property_sub_type_en",
        "property_subtype_en",
        "prop_sb_type_en",
    ],
    "TRANS_VALUE": [
        "TRANS_VALUE",
        "actual_worth",
        "transaction_value",
        "trans_value",
        "procedure_value",
    ],
    "PROCEDURE_AREA": [
        "PROCEDURE_AREA",
        "procedure_area",
    ],
    "ACTUAL_AREA": [
        "ACTUAL_AREA",
        "actual_area",
        "procedure_area",
        "property_area",
        "area",
    ],
    "ROOMS_EN": [
        "ROOMS_EN",
        "rooms_en",
        "rooms",
        "rooms_name_en",
    ],
    "PARKING": [
        "PARKING",
        "has_parking",
        "parking",
    ],
    "NEAREST_METRO_EN": [
        "NEAREST_METRO_EN",
        "nearest_metro_en",
    ],
    "NEAREST_MALL_EN": [
        "NEAREST_MALL_EN",
        "nearest_mall_en",
    ],
    "NEAREST_LANDMARK_EN": [
        "NEAREST_LANDMARK_EN",
        "nearest_landmark_en",
    ],
    "TOTAL_BUYER": [
        "TOTAL_BUYER",
        "no_of_parties_role_2",
        "total_buyer",
    ],
    "TOTAL_SELLER": [
        "TOTAL_SELLER",
        "no_of_parties_role_1",
        "total_seller",
    ],
    "MASTER_PROJECT_EN": [
        "MASTER_PROJECT_EN",
        "master_project_en",
    ],
    "PROJECT_EN": [
        "PROJECT_EN",
        "project_name_en",
        "project_en",
    ],
    "METER_SALE_PRICE": [
        "METER_SALE_PRICE",
        "meter_sale_price",
    ],
}


class DDAApiError(RuntimeError):
    """Raised when the DDA API call succeeds at HTTP level but is unusable."""


@dataclass(frozen=True)
class DDAConfig:
    base_url: str = DEFAULT_BASE_URL
    security_application_identifier: str = field(default="", repr=False)
    client_id: str = field(default="", repr=False)
    client_secret: str = field(default="", repr=False)
    entity: str = DEFAULT_ENTITY
    dataset: str = DEFAULT_DATASET
    verify_ssl: bool = True

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.base_url:
            missing.append("DDA_BASE_URL")
        if not self.security_application_identifier:
            missing.append("DDA_SECURITY_APPLICATION_IDENTIFIER")
        if not self.client_id:
            missing.append("DDA_CLIENT_ID")
        if not self.client_secret:
            missing.append("DDA_CLIENT_SECRET")
        if not self.entity:
            missing.append("DDA_ENTITY")
        if not self.dataset:
            missing.append("DDA_DATASET")
        return missing

    @property
    def token_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{TOKEN_PATH}"

    @property
    def oauth_token_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{OAUTH_TOKEN_PATH}"

    @property
    def dataset_url(self) -> str:
        base = f"{self.base_url.rstrip('/')}{OPENAPI_PATH}"
        return f"{base}/{self.entity.strip('/')}/{self.dataset.strip('/')}"


def load_dda_config(
    secrets: Mapping[str, Any] | None = None,
    secrets_path: str | Path | None = None,
) -> DDAConfig:
    file_values = _load_toml_values(secrets_path)

    def value(field_name: str, env_name: str, default: str = "") -> str:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
        for source in (secrets, file_values):
            found = _lookup_secret(source, env_name, field_name)
            if found:
                return found
        return default

    return DDAConfig(
        base_url=value("base_url", "DDA_BASE_URL", DEFAULT_BASE_URL),
        security_application_identifier=value(
            "security_application_identifier",
            "DDA_SECURITY_APPLICATION_IDENTIFIER",
        ),
        client_id=value("client_id", "DDA_CLIENT_ID"),
        client_secret=value("client_secret", "DDA_CLIENT_SECRET"),
        entity=value("entity", "DDA_ENTITY", DEFAULT_ENTITY),
        dataset=value("dataset", "DDA_DATASET", DEFAULT_DATASET),
        verify_ssl=_as_bool(value("verify_ssl", "DDA_VERIFY_SSL", "true")),
    )


def request_access_token(config: DDAConfig) -> str:
    _ensure_config(config)
    payload = _request_secure_access_token(config)
    if not payload.get("access_token"):
        payload = _request_oauth_access_token(config)
    access_token = payload.get("access_token")
    if not access_token:
        raise DDAApiError("Access token response did not include access_token.")
    return access_token


def _request_secure_access_token(config: DDAConfig) -> dict[str, Any]:
    response = requests.post(
        config.token_url,
        headers={
            "Content-Type": "application/json",
            "x-DDA-SecurityApplicationIdentifier": (
                config.security_application_identifier
            ),
        },
        json={
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        },
        timeout=30,
        verify=config.verify_ssl,
    )
    if response.status_code in {401, 403, 404, 405, 500}:
        return {}
    _raise_for_status(response, "Secure access token request")
    return response.json()


def _request_oauth_access_token(config: DDAConfig) -> dict[str, Any]:
    response = requests.post(
        config.oauth_token_url,
        headers={
            "x-DDA-SecurityApplicationIdentifier": (
                config.security_application_identifier
            ),
        },
        params={"grant_type": "client_credentials"},
        data={
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        },
        timeout=30,
        verify=config.verify_ssl,
    )
    _raise_for_status(response, "OAuth access token request")
    return response.json()


def fetch_dataset_records(
    config: DDAConfig,
    params: Mapping[str, Any] | None = None,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_records: int | None = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    token = request_access_token(config)
    records: list[dict[str, Any]] = []
    offset = int(params.get("offset", 0)) if params else 0
    base_params = dict(params or {})
    base_params.pop("offset", None)
    base_params.pop("limit", None)

    while True:
        query = {
            **base_params,
            "limit": page_size,
            "offset": offset,
        }
        response = requests.get(
            config.dataset_url,
            headers={
                "Authorization": f"Bearer {token}",
                "x-DDA-SecurityApplicationIdentifier": (
                    config.security_application_identifier
                ),
            },
            params=query,
            timeout=30,
            verify=config.verify_ssl,
        )
        _raise_for_status(response, "Dataset request")
        page_records = _extract_records(response.json())
        if not page_records:
            break

        records.extend(page_records)
        if max_records is not None and len(records) >= max_records:
            return records[:max_records]
        if len(page_records) < page_size:
            break
        offset += len(page_records)

    return records


def months_before(end: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be non-negative")

    month_index = end.year * 12 + end.month - 1 - months
    year = month_index // 12
    month = month_index % 12 + 1
    day = min(end.day, monthrange(year, month)[1])
    return date(year, month, day)


def last_months_date_range(months: int = DEFAULT_LOOKBACK_MONTHS) -> tuple[date, date]:
    end = date.today()
    return months_before(end, months), end


def build_dld_transactions_params(
    start: date | None = None,
    end: date | None = None,
    *,
    flat_only: bool = True,
    order_desc: bool = False,
) -> dict[str, str]:
    filters = []
    if flat_only:
        filters.append("property_sub_type_en='Flat'")
    if start:
        filters.append(f"instance_date>='{start.strftime('%Y-%m-%d')}'")
    if end:
        filters.append(f"instance_date<='{end.strftime('%Y-%m-%d')}'")

    params = {
        "order_by": "instance_date",
        "order_dir": "desc" if order_desc else "asc",
    }
    if filters:
        params["filter"] = " AND ".join(filters)
    return params


def records_to_dataframe(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a Polars dataframe from JSON records with stable inferred types."""
    if not records:
        return pl.DataFrame()
    return pl.from_dicts(records, infer_schema_length=None, strict=False)


def normalize_dld_transactions(data: list[dict[str, Any]] | pl.DataFrame) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        df = data.clone()
    elif data:
        df = records_to_dataframe(data)
    else:
        return pl.DataFrame()

    mapping = infer_column_mapping(df.columns)
    expressions = []
    for target, source_columns in mapping.items():
        if source_columns:
            expressions.append(
                pl.coalesce([pl.col(col) for col in source_columns]).alias(target)
            )
    if expressions:
        df = df.with_columns(expressions)

    df = _add_missing_optional_columns(df)
    df = _normalize_types(df)
    return df


def infer_column_mapping(columns: list[str]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for column in columns:
        lookup.setdefault(_canonical_column(column), []).append(column)

    mapping: dict[str, list[str]] = {}
    for target, aliases in COLUMN_ALIASES.items():
        found: list[str] = []
        for alias in aliases:
            for column in lookup.get(_canonical_column(alias), []):
                if column not in found:
                    found.append(column)
        mapping[target] = found
    return mapping


def validate_normalized_columns(df: pl.DataFrame) -> dict[str, Any]:
    statuses: dict[str, dict[str, Any]] = {}
    for column in REQUIRED_DASHBOARD_COLUMNS:
        present = column in df.columns
        non_null = 0
        if present and not df.is_empty():
            non_null = int(df.select(pl.col(column).is_not_null().sum()).item())
        statuses[column] = {
            "present": present,
            "non_null": non_null,
        }

    missing = [
        column
        for column, status in statuses.items()
        if not status["present"] or status["non_null"] == 0
    ]
    return {
        "row_count": df.height,
        "missing_required": missing,
        "columns": statuses,
    }


def _load_toml_values(secrets_path: str | Path | None) -> Mapping[str, Any]:
    path = Path(secrets_path) if secrets_path else Path(".streamlit/secrets.toml")
    if not path.exists():
        return {}

    try:
        import tomllib
    except ModuleNotFoundError:
        return {}

    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def _lookup_secret(
    source: Mapping[str, Any] | None,
    env_name: str,
    field_name: str,
) -> str:
    if not source:
        return ""

    keys = {
        env_name,
        env_name.lower(),
        field_name,
        field_name.upper(),
    }
    for key in keys:
        try:
            value = source.get(key)
        except AttributeError:
            value = None
        if value is not None and value != "":
            return str(value)

    try:
        dda_section = source.get("dda")
    except AttributeError:
        dda_section = None
    if isinstance(dda_section, Mapping):
        for key in keys:
            value = dda_section.get(key)
            if value is not None and value != "":
                return str(value)
    return ""


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _ensure_config(config: DDAConfig) -> None:
    missing = config.missing_fields()
    if missing:
        raise DDAApiError(
            "Missing DDA configuration values: " + ", ".join(missing)
        )


def _raise_for_status(response: requests.Response, context: str) -> None:
    if response.ok:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:500]
    raise DDAApiError(f"{context} failed with HTTP {response.status_code}: {detail}")


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("results", "records", "result", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
        if isinstance(value, dict):
            nested = _extract_records(value)
            if nested:
                return nested

    return []


def _add_missing_optional_columns(df: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for column in OPTIONAL_DASHBOARD_COLUMNS:
        if column not in df.columns:
            expressions.append(pl.lit(None).alias(column))
    if expressions:
        return df.with_columns(expressions)
    return df


def _normalize_types(df: pl.DataFrame) -> pl.DataFrame:
    expressions = []

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.Utf8)
                .str.replace_all(",", "")
                .str.replace_all("AED", "")
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias(column)
            )

    if "INSTANCE_DATE" in df.columns:
        value = pl.col("INSTANCE_DATE").cast(pl.Utf8).str.strip_chars()
        parsed_date = pl.coalesce(
            value.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.date(),
            value.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False).dt.date(),
            value.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
            value.str.slice(0, 10).str.to_date("%d-%m-%Y", strict=False),
            value.str.slice(0, 10).str.to_date("%d/%m/%Y", strict=False),
        )
        expressions.append(parsed_date.dt.strftime("%Y-%m-%d").alias("INSTANCE_DATE"))

    if "ROOMS_EN" in df.columns:
        rooms = pl.col("ROOMS_EN").cast(pl.Utf8).str.strip_chars()
        expressions.append(
            pl.when(rooms.str.to_lowercase() == "studio")
            .then(pl.lit("Studio"))
            .when(rooms.str.contains(r"^\d+$"))
            .then(rooms + pl.lit(" B/R"))
            .otherwise(rooms)
            .alias("ROOMS_EN")
        )

    string_columns = [
        "GROUP_EN",
        "IS_OFFPLAN_EN",
        "AREA_EN",
        "PROP_SB_TYPE_EN",
        "PROCEDURE_EN",
        "USAGE_EN",
        "PROP_TYPE_EN",
    ]
    for column in string_columns:
        if column in df.columns:
            value = pl.col(column).cast(pl.Utf8).str.strip_chars()
            if column == "AREA_EN":
                value = value.str.to_uppercase()
            expressions.append(value.alias(column))

    if expressions:
        return df.with_columns(expressions)
    return df


def _canonical_column(column: str) -> str:
    return re.sub(r"[^a-z0-9]", "", column.lower())
