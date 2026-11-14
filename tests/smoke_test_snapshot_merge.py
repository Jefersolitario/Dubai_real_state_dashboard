"""Offline check of the snapshot merge: one row per DLD transaction.

DLD re-dates and amends sales after publishing them. These checks drive the
real merge (`merge_snapshots`, shared by the CLI refresh and the dashboard's
background top-up) with small normalized frames: a re-dated or amended sale
replaces its stored version, a stored pair collapses to the later date, an
identical re-fetch changes nothing, and rows without a transaction number
keep the exact-duplicate rule. No Streamlit, no network.
"""

import sys

import polars as pl

from ingestion.dda_api import normalize_dld_transactions
from ingestion.store_dld_transactions_gcs import (
    TRANSACTION_KEY,
    dedupe_snapshot,
    merge_snapshots,
)


def check(condition: bool, label: str) -> bool:
    print(f"{'[ok]  ' if condition else '[FAIL]'} {label}")
    return condition


def sale(number: str | None, day: str, value: float = 1_000_000.0, **extra) -> dict:
    return {
        "TRANSACTION_NUMBER": number,
        "INSTANCE_DATE": day,
        "GROUP_EN": "Sales",
        "PROCEDURE_EN": "Sell - Pre registration",
        "IS_OFFPLAN_EN": "Off-Plan Properties",
        "AREA_EN": "MARSA DUBAI",
        "PROP_SB_TYPE_EN": "Flat",
        "TRANS_VALUE": value,
        "ACTUAL_AREA": 70.5,
        "ROOMS_EN": "1 B/R",
        "PROJECT_EN": "Sample Tower",
        "BUILDING_NAME_EN": "Sample Tower A",
        **extra,
    }


def frame(*rows: dict) -> pl.DataFrame:
    return normalize_dld_transactions(list(rows))


def versions(df: pl.DataFrame, number: str) -> list[tuple]:
    return (
        df.filter(pl.col(TRANSACTION_KEY) == number)
        .select("INSTANCE_DATE", "TRANS_VALUE", "BUILDING_NAME_EN")
        .rows()
    )


def check_redated_sale_is_replaced() -> bool:
    stored = frame(sale("1-102-2026-1", "2026-11-06"), sale("1-102-2026-2", "2026-11-06"))
    fetched = frame(sale("1-102-2026-1", "2026-11-11"))
    final, stats = merge_snapshots(stored, fetched, full_refresh=False)
    return check(
        final.height == 2
        and versions(final, "1-102-2026-1") == [("2026-11-11", 1_000_000.0, "Sample Tower A")]
        and (stats["added_rows"], stats["updated_rows"]) == (0, 1)
        and stats["write_required"],
        "a re-dated sale replaces its stored version and is stored as an update",
    )


def check_amended_sale_is_replaced() -> bool:
    stored = frame(sale("1-102-2026-1", "2026-11-06"))
    fetched = frame(
        sale("1-102-2026-1", "2026-11-06", value=1_250_000.0, BUILDING_NAME_EN="Sample Tower B")
    )
    final, stats = merge_snapshots(stored, fetched, full_refresh=False)
    return check(
        versions(final, "1-102-2026-1") == [("2026-11-06", 1_250_000.0, "Sample Tower B")]
        and (stats["added_rows"], stats["updated_rows"]) == (0, 1),
        "an amended sale (same day, new price and building) replaces its stored version",
    )


def check_identical_refetch_changes_nothing() -> bool:
    rows = [sale("1-102-2026-1", "2026-11-06"), sale("1-102-2026-2", "2026-11-07", value=2e6)]
    stored = frame(*rows)
    final, stats = merge_snapshots(stored, frame(*rows), full_refresh=False)
    return check(
        final.equals(stored)
        and (stats["added_rows"], stats["updated_rows"]) == (0, 0)
        and stats["overlap_duplicate_rows"] == 2
        and not stats["write_required"],
        "fetching the same rows again changes nothing and asks for no write",
    )


def check_new_and_revised_are_counted_apart() -> bool:
    stored = frame(sale("1-102-2026-1", "2026-11-06"), sale("1-102-2026-2", "2026-11-06"))
    fetched = frame(
        sale("1-102-2026-2", "2026-11-12"),               # re-dated
        sale("1-102-2026-3", "2026-11-12", value=3e6),    # new
        sale("1-102-2026-1", "2026-11-06"),               # unchanged
    )
    final, stats = merge_snapshots(stored, fetched, full_refresh=False)
    in_order = final["INSTANCE_DATE"].to_list() == sorted(final["INSTANCE_DATE"].to_list())
    return check(
        final.height == 3 and final[TRANSACTION_KEY].n_unique() == 3 and in_order
        and (stats["added_rows"], stats["updated_rows"]) == (1, 1),
        "new sales count as added, revised ones as updated, and the result is date-ordered",
    )


def check_stored_pair_collapses_to_the_later_date() -> bool:
    # What production holds today: both versions, in date order.
    stored = frame(
        sale("1-102-2026-1", "2026-11-06"),
        sale("1-102-2026-2", "2026-11-07"),
        sale("1-102-2026-1", "2026-11-11"),
    )
    alone = dedupe_snapshot(stored)
    final, stats = merge_snapshots(stored, frame(sale("1-102-2026-2", "2026-11-07")), full_refresh=False)
    return check(
        versions(alone, "1-102-2026-1") == [("2026-11-11", 1_000_000.0, "Sample Tower A")]
        and final.height == 2
        and stats["existing_duplicate_rows_removed"] == 1
        and (stats["added_rows"], stats["updated_rows"]) == (0, 0)
        and stats["write_required"],
        "a pair already stored keeps its later date, and removing it alone requires a write",
    )


def check_fetched_version_wins_whatever_its_date() -> bool:
    stored = frame(sale("1-102-2026-1", "2026-11-11"))
    fetched = frame(sale("1-102-2026-1", "2026-11-06"))
    final, _ = merge_snapshots(stored, fetched, full_refresh=False)
    return check(
        versions(final, "1-102-2026-1") == [("2026-11-06", 1_000_000.0, "Sample Tower A")],
        "in a merge the fetched version wins even when its date is the earlier one",
    )


def check_rows_without_a_number() -> bool:
    stored = frame(
        sale(None, "2026-11-06"),
        sale(None, "2026-11-06"),                 # exact duplicate
        sale(None, "2026-11-06", value=2e6),      # another sale that also lacks a number
        sale("1-102-2026-1", "2026-11-06"),
    )
    alone = dedupe_snapshot(stored)
    fetched = frame(sale(None, "2026-11-08", value=3e6), sale("1-102-2026-9", "2026-11-08"))
    final, stats = merge_snapshots(stored, fetched, full_refresh=False)
    return check(
        alone.height == 3 and alone[TRANSACTION_KEY].null_count() == 2
        and final.height == 5 and final[TRANSACTION_KEY].null_count() == 3
        and (stats["added_rows"], stats["updated_rows"]) == (2, 0)
        and stats["existing_duplicate_rows_removed"] == 1,
        "rows without a transaction number only lose exact duplicates",
    )


def check_full_refresh() -> bool:
    fetched = frame(
        sale("1-102-2026-1", "2026-11-06"),
        sale("1-102-2026-1", "2026-11-11"),
        sale("1-102-2026-2", "2026-11-07"),
    )
    final, stats = merge_snapshots(None, fetched, full_refresh=True)
    return check(
        final.height == 2 and final[TRANSACTION_KEY].n_unique() == 2
        and stats["mode"] == "full_refresh" and stats["added_rows"] == 2
        and stats["updated_rows"] == 0 and stats["incoming_duplicate_rows_removed"] == 1,
        "a full refresh also ends with one row per transaction",
    )


def check_frame_without_the_number_column() -> bool:
    rows = frame(sale("x", "2026-11-06"), sale("x", "2026-11-06"), sale("x", "2026-11-07")).drop(
        TRANSACTION_KEY
    )
    return check(
        dedupe_snapshot(rows).height == 2,
        "a frame without the number column falls back to exact duplicates",
    )


def check_widened_column_is_not_a_change() -> bool:
    # An all-null stored column has no dtype of its own; the fetch gives it one.
    stored = frame(sale("1-102-2026-1", "2026-11-06"), sale("1-102-2026-2", "2026-11-06"))
    fetched = frame(sale("1-102-2026-3", "2026-11-08", PARKING="1"))
    widened = stored.schema["PARKING"] != fetched.schema["PARKING"]
    _, stats = merge_snapshots(stored, fetched, full_refresh=False)
    return check(
        widened and (stats["added_rows"], stats["updated_rows"]) == (1, 0),
        "a column the fetch gives a type to does not turn stored rows into updates",
    )


def main() -> int:
    ok = check_redated_sale_is_replaced()
    ok &= check_amended_sale_is_replaced()
    ok &= check_identical_refetch_changes_nothing()
    ok &= check_new_and_revised_are_counted_apart()
    ok &= check_stored_pair_collapses_to_the_later_date()
    ok &= check_fetched_version_wins_whatever_its_date()
    ok &= check_rows_without_a_number()
    ok &= check_full_refresh()
    ok &= check_frame_without_the_number_column()
    ok &= check_widened_column_is_not_a_change()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
