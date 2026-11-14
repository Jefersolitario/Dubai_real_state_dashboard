"""Offline check of the Rent Scanner pipeline on synthetic Ejari-shaped data.

Exercises the real ingestion helpers (`_slim_rents`, `_weekly_stats`,
`_recent_contracts`) and the real chart builders from `rent_scanner_tab`
(Streamlit runs in bare mode): district round-trip, column slimming and
rooms banding, weekly percentile stats with the "All" band and the
new-contract segment, the recent-contract window and dedupe, the dot/box
view switch, and that every chart builds and serializes. No network, no
credentials.
"""

import sys
from datetime import date, timedelta

import polars as pl

from dashboard_constants import AREA_DISPLAY, NEIGHBORHOODS
from ingestion.store_reference_data_gcs import (
    RENT_DISTRICTS,
    _recent_contracts,
    _slim_rents,
    _weekly_stats,
)
from rent_scanner_tab import (
    rent_box_chart,
    rent_deals_by_zone_chart,
    rent_zone_chart,
    use_dot_view,
)

SQM_TO_SQFT = 10.7639
BASE_RECORD = {
    "contract_start_date": "2026-07-15T00:00:00",
    "annual_amount": "84000",
    "actual_area": "70.5",
    "area_name_en": " Business Bay ",
    "ejari_property_sub_type_en": "2 bed room flat",
    "ejari_property_type_en": "Flat",
    "contract_id": "CNTX1",
    "line_number": 1,
    "no_of_prop": 1,
    "some_other_raw_col": "x",
}


def check(condition: bool, label: str) -> bool:
    print(f"{'[ok]  ' if condition else '[FAIL]'} {label}")
    return condition


def check_districts() -> bool:
    good = (
        len(RENT_DISTRICTS) == len(NEIGHBORHOODS)
        and {AREA_DISPLAY[d] for d in RENT_DISTRICTS} == set(NEIGHBORHOODS)
    )
    return check(good, f"RENT_DISTRICTS round-trips to the {len(NEIGHBORHOODS)} picker neighbourhoods")


def check_slim_rents() -> bool:
    without_reg = _slim_rents(pl.DataFrame([BASE_RECORD] * 3, infer_schema_length=None))
    with_reg = _slim_rents(pl.DataFrame(
        [{**BASE_RECORD, "contract_reg_type_en": "New "},
         {**BASE_RECORD, "contract_reg_type_en": "Renew"}],
        infer_schema_length=None,
    ))
    unspaced = _slim_rents(pl.DataFrame(
        [{**BASE_RECORD, "ejari_property_sub_type_en": "1bed room+Hall"}],
        infer_schema_length=None,
    ))
    mixed = pl.concat([without_reg, with_reg], how="diagonal_relaxed")
    good = (
        "reg_type" not in without_reg.columns and "reg_type" in with_reg.columns
        and all(c in without_reg.columns for c in ("contract_id", "line_number", "no_of_prop"))
        and without_reg["AREA_EN"][0] == "BUSINESS BAY"
        and with_reg["reg_type"][0] == "New"
        and without_reg["rooms_band"][0] == "2BR"
        and without_reg["start"][0] == date(2026, 7, 15)
        and unspaced["rooms_band"][0] == "1BR"
        and mixed.height == 5 and mixed["reg_type"].null_count() == 3
    )
    return check(good, "_slim_rents strips/uppercases areas, bands rooms, guards reg_type")


def _contract(today: date, counter: list[int], days_ago: int, area: str, band: str | None,
              psf: float, reg: str = "New", cid: str | None = None) -> dict:
    start = today - timedelta(days=days_ago)
    counter[0] += 1
    return {
        "start": start,
        "annual_amount": psf * 70.0 * SQM_TO_SQFT,
        "actual_area": 70.0,
        "AREA_EN": area,
        "rooms_band": band,
        "ptype": "Flat",
        "reg_type": reg,
        "contract_id": cid or f"CNT{counter[0]:05d}",
        "line_number": 1,
        "no_of_prop": 1,
        "rent_psf": float(psf),
        "week": start - timedelta(days=start.weekday()),
    }


def check_weekly_and_recent() -> bool:
    today = date.today()
    counter = [0]
    anchor = 30 + (today - timedelta(days=30)).weekday()
    rows = [_contract(today, counter, anchor - i % 3, "BUSINESS BAY", "2BR", psf)
            for i, psf in enumerate([100, 110, 120, 130, 140])]
    rows.append(_contract(today, counter, anchor, "BUSINESS BAY", None, 50))
    rows.append(_contract(today, counter, anchor - 1, "BUSINESS BAY", None, 60))
    rows.append(_contract(today, counter, anchor - 2, "BUSINESS BAY", "2BR", 80, reg="Renew"))
    rents = pl.DataFrame(rows)
    rents = rents.with_columns(pl.lit(rents["week"].min()).alias("week"))

    stats = _weekly_stats(rents)

    def cell(band: str, segment: str) -> pl.DataFrame:
        return stats.filter((pl.col("rooms_band") == band) & (pl.col("segment") == segment))

    expected_q1 = rents.filter(pl.col("rooms_band") == "2BR")["rent_psf"].quantile(0.25)
    ok = check(
        set(stats.columns) >= {"AREA_EN", "rooms_band", "week", "n", "median", "q1", "q3", "p10", "p90", "segment"}
        and cell("2BR", "all")["n"].item() == 6
        and cell("All", "all")["n"].item() == 8
        and cell("2BR", "new")["n"].item() == 5
        and cell("2BR", "new")["median"].item() == 120.0
        and abs(cell("2BR", "all")["q1"].item() - round(expected_q1, 1)) < 1e-9,
        "_weekly_stats: 'All' band absorbs unbanded rows, 'new' segment drops renewals",
    )

    extra = pl.DataFrame([
        _contract(today, counter, 10, "BUSINESS BAY", "1BR", 90, cid="CNTDUP"),
        _contract(today, counter, 10, "BUSINESS BAY", "1BR", 90, cid="CNTDUP"),
        _contract(today, counter, 10, "BUSINESS BAY", "1BR", 90, cid="CNTTWIN"),
        _contract(today, counter, 10, "NOT A SCANNER DISTRICT", "1BR", 90),
        _contract(today, counter, 400, "BUSINESS BAY", "1BR", 90),
    ])
    recent = _recent_contracts(pl.concat([rents, extra], how="diagonal_relaxed"))
    ok &= check(
        {"size_sqft", "reg_type", "contract_id"} <= set(recent.columns)
        and recent.filter(pl.col("AREA_EN") == "NOT A SCANNER DISTRICT").height == 0
        and recent.filter(pl.col("rooms_band") == "1BR").height == 2
        and abs(recent["size_sqft"][0] - round(70.0 * SQM_TO_SQFT, 0)) < 1e-9
        and "reg_type" not in _recent_contracts(rents.drop("reg_type")).columns,
        "_recent_contracts: district and window filters, exact-duplicate collapse",
    )
    return ok


def check_view_switch() -> bool:
    floor = date(2026, 3, 1)
    good = (
        use_dot_view(date(2026, 5, 1), date(2026, 7, 31), floor) is True
        and use_dot_view(date(2026, 5, 1), date(2026, 8, 1), floor) is False
        and use_dot_view(date(2026, 2, 1), date(2026, 3, 15), floor) is False
        and use_dot_view(date(2026, 5, 1), date(2026, 7, 31), None) is False
    )
    return check(good, "use_dot_view: 92-day limit and artifact-coverage gate")


def check_charts() -> bool:
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(60)]
    rows = []
    for i, day in enumerate(days):
        for j in range(3):
            psf = 100.0 + (i * 7 + j * 11) % 40
            if i == 30 and j == 0:
                psf = 60.0
            if i == 40 and j == 0:
                psf = 10.0
            rows.append({
                "date": day, "area": "Business Bay", "rent_psf": psf,
                "rooms_band": "2BR", "size_sqft": 750.0,
                "annual_amount": psf * 750.0, "reg_type": "New",
            })
    txns = pl.DataFrame(rows)
    rolling = (
        txns.group_by(["date", "area"])
        .agg(pl.col("rent_psf").median().round(1).alias("daily_median_psf"))
        .sort(["area", "date"])
        .with_columns(pl.col("daily_median_psf").rolling_mean(14).over("area").alias("rolling_median_psf"))
        .filter(pl.col("rolling_median_psf").is_not_null())
    )
    scored = txns.join(
        rolling.select(["date", "area", "rolling_median_psf"]), on=["date", "area"], how="left"
    ).with_columns((pl.col("rent_psf") / pl.col("rolling_median_psf") - 1).alias("pct_vs_median"))

    dots = rent_zone_chart(scored, rolling, "Business Bay", 10)
    names = [trace.name for trace in dots.data]
    weekly = pl.DataFrame({
        "week": [date(2026, 1, 5) + timedelta(weeks=i) for i in range(20)],
        "median": [110.0 + i for i in range(20)],
        "q1": [100.0 + i for i in range(20)],
        "q3": [120.0 + i for i in range(20)],
        "p10": [90.0 + i for i in range(20)],
        "p90": [130.0 + i for i in range(20)],
        "n": [200 + i for i in range(20)],
    })
    boxes = rent_box_chart(weekly, "Business Bay")
    bars = rent_deals_by_zone_chart(pl.DataFrame({"area": ["A", "B"], "deals": [3, 9]}), 10)
    good = (
        any("Deal" in name for name in names) and any("Non-market" in name for name in names)
        and len(dots.to_json()) > 1000
        and len(boxes.data) == 2 and len(boxes.to_json()) > 500
        and len(bars.to_json()) > 300
    )
    return check(good, "dot, box and deals-by-zone charts build and serialize")


def main() -> int:
    ok = check_districts()
    ok &= check_slim_rents()
    ok &= check_weekly_and_recent()
    ok &= check_view_switch()
    ok &= check_charts()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
