from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, datetime

from dda_api import (
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_PAGE_SIZE,
    DDAApiError,
    build_dld_transactions_params,
    fetch_dataset_records,
    infer_column_mapping,
    last_months_date_range,
    load_dda_config,
    normalize_dld_transactions,
    validate_normalized_columns,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch DDA production API records and validate dashboard column mapping."
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--entity")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        help="Optional instance_date lower bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        help="Optional instance_date upper bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--last-months",
        type=int,
        default=DEFAULT_LOOKBACK_MONTHS,
        help=(
            "When no explicit dates are supplied, query this many recent "
            "months ending today. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--include-all-property-types",
        action="store_true",
        help="Do not filter to property_sub_type_en='Flat' when date filters are used.",
    )
    parser.add_argument(
        "--require-records",
        action="store_true",
        help="Return exit code 1 when the requested query returns no records.",
    )
    args = parser.parse_args()

    config = load_dda_config()
    if args.entity:
        config = replace(config, entity=args.entity)
    if args.dataset:
        config = replace(config, dataset=args.dataset)

    missing = config.missing_fields()
    if missing:
        print("Missing DDA configuration values: " + ", ".join(missing))
        print("Create .streamlit/secrets.toml from .streamlit/secrets.example.toml.")
        return 2

    max_records = max(args.limit, 1)
    page_size = min(max_records, DEFAULT_PAGE_SIZE)
    start_date = args.start_date
    end_date = args.end_date
    if not start_date and not end_date:
        start_date, end_date = last_months_date_range(args.last_months)

    params = build_dld_transactions_params(
        start_date,
        end_date,
        flat_only=not args.include_all_property_types,
        order_desc=True,
    )

    try:
        records = fetch_dataset_records(
            config,
            params=params,
            page_size=page_size,
            max_records=max_records,
        )
    except DDAApiError as exc:
        print(f"API error: {exc}")
        return 1

    raw_columns = list(records[0].keys()) if records else []
    normalized = normalize_dld_transactions(records)
    validation = validate_normalized_columns(normalized)
    mapping = infer_column_mapping(raw_columns)

    print(f"Endpoint entity/dataset: {config.entity}/{config.dataset}")
    print(f"Date window: {start_date} to {end_date}")
    print(f"Query params: {params}")
    print(f"Fetched records: {len(records)}")
    print(f"Date coverage: {_date_coverage(records)}")
    if len(records) == max_records:
        print("Reached requested --limit; coverage may be truncated by max_records.")

    if args.require_records and not records:
        print(
            "No records returned for the requested query. "
            "If connectivity succeeds for latest rows, this usually means the "
            "dataset does not include that date range."
        )
        return 1

    print("Raw columns:")
    for column in raw_columns:
        print(f"  - {column}")

    print("Dashboard column mapping:")
    for target, sources in mapping.items():
        source_text = ", ".join(sources) if sources else "MISSING"
        print(f"  - {target} <= {source_text}")

    if validation["missing_required"]:
        print(
            "Missing or empty required normalized columns: "
            + ", ".join(validation["missing_required"])
        )
        return 1

    print("Required normalized columns: OK")
    return 0


def _parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid YYYY-MM-DD date."
        ) from exc


def _date_coverage(records: list[dict]) -> str:
    dates = sorted(
        str(record.get("instance_date", ""))[:10]
        for record in records
        if record.get("instance_date")
    )
    if not dates:
        return "none"
    return f"{dates[0]} to {dates[-1]}"


if __name__ == "__main__":
    sys.exit(main())
