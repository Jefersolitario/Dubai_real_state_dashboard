"""Rigorous data-quality audit of the Ejari rent feed and the GCS rent artifacts.

Methodology follows the repo's data-cleaning best-practice research
(reports/data_cleaning_research_report.md): every exclusion carries a reason
code (IAAO "valid until documented otherwise"), outliers are judged against
LOCAL comparables (zone x rooms band, per Land Registry price bands / AVM
practice — never global thresholds), duplicates get an explicit definition,
and every proposed cleaning rule is sensitivity-tested against the statistic
it protects (the scanner's weekly zone medians). Findings are written to
reports/rent_data_quality_report.md in the sales audit's format.

Inputs:
- a raw 2-month all-columns sample of dld_rent_contracts-open-api (fetched
  live, or loaded from --raw-parquet with its --type-census json)
- the three published GCS artifacts (rent_index, rent_weekly_stats,
  rent_recent_contracts) unless --raw-only
- optionally the rents pull log (--pull-log) for month-level gap detection

Usage:
    python -m tests.audit_rent_data_quality
    python -m tests.audit_rent_data_quality --raw-only --raw-parquet sample.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, timedelta

import polars as pl

from dashboard_constants import AREA_DISPLAY
from ingestion.dda_api import fetch_dataset_records, load_dda_config
from ingestion.gcs_storage import (
    REFERENCE_OBJECTS,
    load_local_secrets,
    read_parquet_object,
    setting,
)
from ingestion.store_reference_data_gcs import (
    RENT_ANNUAL_MAX,
    RENT_ANNUAL_MIN,
    RENT_AREA_MAX,
    RENT_AREA_MIN,
    RENT_DISTRICTS,
    RENT_YEAR_MAX,
    RENT_YEAR_MIN,
    rooms_band_expr,
)

SQM_TO_SQFT = 10.7639
DEFAULT_MONTHS = (("2026-07-01", "2026-08-01"), ("2024-06-01", "2024-07-01"))

# Verdicts
OK, WARN, ACTION = "OK", "WARN", "ACTION"

findings: list[dict] = []


def add(section: str, check: str, verdict: str, size: str, evidence: str, action: str) -> None:
    findings.append({
        "section": section, "check": check, "verdict": verdict,
        "size": size, "evidence": evidence, "action": action,
    })
    mark = {"OK": "ok", "WARN": "WARN", "ACTION": "ACTION"}[verdict]
    print(f"[{mark:6}] {section} / {check}: {size} — {evidence}")


# ---------------------------------------------------------------------------
# Raw sample acquisition
# ---------------------------------------------------------------------------

def fetch_raw_sample(months) -> tuple[pl.DataFrame, dict]:
    """Fetch raw months with ALL wire columns + a pre-DataFrame type census."""
    config = load_dda_config(load_local_secrets())
    cfg = replace(config, dataset="dld_rent_contracts-open-api")
    records: list[dict] = []
    census: dict[str, dict[str, int]] = {}
    for lo, hi in months:
        month_records = fetch_dataset_records(
            cfg,
            params={
                "filter": (
                    "property_usage_en='Residential' AND "
                    f"contract_start_date>='{lo}' AND contract_start_date<'{hi}'"
                ),
                "order_by": "contract_start_date",
                "order_dir": "asc",
            },
            max_records=200_000,
        )
        print(f"raw sample {lo[:7]}: {len(month_records):,} records", flush=True)
        for r in month_records:
            for k, v in r.items():
                bucket = census.setdefault(k, {})
                bucket[type(v).__name__] = bucket.get(type(v).__name__, 0) + 1
            r["_sample_month"] = lo[:7]
        records.extend(month_records)
    return pl.DataFrame(records, infer_schema_length=None), census


# ---------------------------------------------------------------------------
# Section 1 — duplicates
# ---------------------------------------------------------------------------

def audit_duplicates(raw: pl.DataFrame) -> pl.DataFrame:
    """Returns the frame of D2/D4 surplus rows (used later for sensitivity)."""
    n = raw.height
    work = raw.drop("_sample_month")

    # D1: exact full-row duplicates (gateway/pagination repeats)
    d1 = n - work.height + work.height - work.unique().height
    d1b = work.drop("load_timestamp").height - work.drop("load_timestamp").unique().height \
        if "load_timestamp" in work.columns else d1
    add("1 Duplicates", "D1 exact full-row duplicates",
        OK if d1b == 0 else ACTION, f"{d1b:,} rows ({d1b / n:.3%})",
        f"all-column dupes {d1:,}; excluding load_timestamp {d1b:,}",
        "none needed" if d1b == 0 else "dedupe at slim time")

    # D2: canonical duplicate — same (contract_id, line_number) more than once
    key = ["contract_id", "line_number"]
    grp = work.group_by(key).agg(pl.len().alias("copies"))
    d2_groups = grp.filter(pl.col("copies") > 1)
    d2_surplus = int((d2_groups["copies"] - 1).sum()) if d2_groups.height else 0
    conflicting = 0
    if d2_groups.height:
        dupes = work.join(d2_groups.select(key), on=key, how="inner")
        conflicting = (
            dupes.group_by(key)
            .agg(
                pl.col("annual_amount").n_unique().alias("na"),
                pl.col("actual_area").n_unique().alias("nb"),
            )
            .filter((pl.col("na") > 1) | (pl.col("nb") > 1))
            .height
        )
    add("1 Duplicates", "D2 key duplicates (contract_id, line_number)",
        OK if d2_surplus == 0 else ACTION,
        f"{d2_surplus:,} surplus rows ({d2_surplus / n:.3%})",
        f"{d2_groups.height:,} duplicated keys, {conflicting:,} with conflicting amount/area",
        "none needed" if d2_surplus == 0 else "dedupe on (contract_id, line_number) keeping latest load_timestamp")

    # D3: multi-line contracts (NOT duplicates — multi-property contracts)
    lines = work.group_by("contract_id").agg(
        pl.len().alias("lines"), pl.col("no_of_prop").max().alias("props")
    )
    multi = lines.filter(pl.col("lines") > 1)
    consistent = multi.filter(pl.col("lines") == pl.col("props")).height
    add("1 Duplicates", "D3 multi-line contracts (multi-property, legit)",
        OK, f"{multi.height:,} contracts ({multi.height / max(lines.height, 1):.2%})",
        f"lines==no_of_prop for {consistent:,}/{multi.height:,} of them",
        "document semantics; not duplicates")

    # D4: re-registration suspects — same everything, different contract_id
    d4_key = ["area_id", "ejari_property_sub_type_id", "actual_area",
              "contract_start_date", "contract_end_date", "annual_amount"]
    d4 = (
        work.drop_nulls(d4_key)
        .group_by(d4_key)
        .agg(pl.col("contract_id").n_unique().alias("ids"), pl.len().alias("rows"))
        .filter(pl.col("ids") > 1)
    )
    d4_surplus = int((d4["rows"] - 1).sum()) if d4.height else 0
    add("1 Duplicates", "D4 re-registration suspects (same attrs, different id)",
        OK if d4_surplus / n < 0.005 else WARN,
        f"{d4_surplus:,} surplus rows ({d4_surplus / n:.3%})",
        f"{d4.height:,} attribute groups with >1 contract_id",
        "monitor; likely same-building identical units + genuine re-registrations")

    surplus = work.join(d2_groups.select(key), on=key, how="inner") if d2_groups.height else work.head(0)
    return surplus


# ---------------------------------------------------------------------------
# Section 2 — missing data & type errors
# ---------------------------------------------------------------------------

def audit_types_and_missing(raw: pl.DataFrame, census: dict) -> None:
    n = raw.height

    # Wire-type census: numeric fields containing strings
    for field in ("annual_amount", "contract_amount", "actual_area", "no_of_prop", "line_number"):
        types = census.get(field, {})
        n_str = types.get("str", 0)
        add("2 Types & missing", f"wire type of {field}",
            OK if n_str == 0 else WARN,
            f"{n_str:,} string values ({n_str / n:.3%})",
            f"wire types: {types}",
            "none needed" if n_str == 0 else "strict-cast check below quantifies parse failures")

    # Silent-null measurement (what production strict=False casting would null)
    for field in ("annual_amount", "actual_area"):
        col = raw[field]
        if col.dtype == pl.Utf8:
            bad = raw.filter(
                pl.col(field).is_not_null()
                & pl.col(field).cast(pl.Float64, strict=False).is_null()
            ).height
        else:
            bad = 0
        add("2 Types & missing", f"unparsable numerics in {field}",
            OK if bad == 0 else ACTION, f"{bad:,} rows",
            "values that production casting silently nulls",
            "none needed" if bad == 0 else "inspect raw values; extend parsing")

    for field in ("contract_start_date", "contract_end_date"):
        parsed = raw.select(
            pl.col(field).cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False).alias("d"),
            pl.col(field).is_not_null().alias("has"),
        )
        bad = parsed.filter(pl.col("has") & pl.col("d").is_null()).height
        add("2 Types & missing", f"unparsable dates in {field}",
            OK if bad == 0 else ACTION, f"{bad:,} rows", "post slice-10 ISO parse",
            "none needed" if bad == 0 else "extend date parsing")

    # Null / empty-string rates on load-bearing columns
    key_cols = ["contract_id", "contract_start_date", "contract_end_date",
                "annual_amount", "contract_amount", "actual_area", "area_name_en",
                "ejari_property_sub_type_en", "ejari_property_type_en",
                "contract_reg_type_en", "no_of_prop", "project_name_en"]
    rates = []
    for c in key_cols:
        if c not in raw.columns:
            continue
        nulls = raw[c].null_count()
        empties = raw.filter(pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars() == "").height \
            if raw[c].dtype == pl.Utf8 else 0
        rates.append((c, nulls / n, empties))
    worst = sorted(rates, key=lambda r: -r[1])[:4]
    add("2 Types & missing", "null/empty rates (key columns)",
        OK if all(r[1] < 0.5 for r in rates if r[0] not in ("project_name_en",)) else WARN,
        "; ".join(f"{c}={rate:.1%}" for c, rate, _ in worst),
        f"empty strings: {sum(r[2] for r in rates):,} total",
        "project/master fields expected sparse (Ejari has no building key)")

    # Area-name hygiene: variants differing only by strip/upper
    names = raw.select(pl.col("area_name_en").cast(pl.Utf8)).drop_nulls()
    raw_distinct = names.n_unique()
    norm_distinct = names.select(
        pl.col("area_name_en").str.to_uppercase().str.strip_chars()
    ).n_unique()
    add("2 Types & missing", "area_name_en case/whitespace variants",
        OK if raw_distinct == norm_distinct else WARN,
        f"{raw_distinct - norm_distinct} collapsing variants",
        f"{raw_distinct} raw vs {norm_distinct} normalized names",
        "strip+upper already applied at slim time (parity with sales path)")

    # Rooms-band mapping coverage among flats
    flats = raw.filter(
        pl.col("ejari_property_type_en").cast(pl.Utf8).str.to_lowercase().str.contains("flat")
    ).with_columns(
        pl.col("ejari_property_sub_type_en").cast(pl.Utf8).alias("subtype")
    )
    banded = flats.select(
        pl.col("subtype"),
        rooms_band_expr_on("subtype").alias("band"),
    )
    unmapped = banded.filter(pl.col("band").is_null() & pl.col("subtype").is_not_null())
    top_unmapped = (
        unmapped.group_by("subtype").agg(pl.len().alias("n"))
        .sort("n", descending=True).head(8)
    )
    share = unmapped.height / max(flats.height, 1)
    add("2 Types & missing", "rooms-band mapping coverage (flats)",
        OK if share < 0.05 else WARN,
        f"{unmapped.height:,} unmapped ({share:.2%})",
        "top unmapped: " + "; ".join(f"{r['subtype']}={r['n']}" for r in top_unmapped.iter_rows(named=True)),
        "extend ROOMS_BAND_MAP for high-volume labels" if share >= 0.05 else "acceptable (folds into the All band)")

    # reg_type vocabulary (validates the anchored ^new matcher)
    if "contract_reg_type_en" in raw.columns:
        vocab = (
            raw.group_by(pl.col("contract_reg_type_en").cast(pl.Utf8).alias("v"))
            .agg(pl.len().alias("n")).sort("n", descending=True)
        )
        labels = {r["v"] for r in vocab.iter_rows(named=True)}
        risky = [v for v in labels if v and not v.lower().startswith("new") and "new" in v.lower()]
        add("2 Types & missing", "contract_reg_type_en vocabulary",
            OK, "; ".join(f"{r['v']}={r['n']:,}" for r in vocab.iter_rows(named=True)),
            f"labels containing 'new' without starting with it: {risky or 'none'}",
            "anchored ^new matcher validated" if not risky else "revisit the matcher")


def rooms_band_expr_on(col: str) -> pl.Expr:
    """rooms_band_expr but applied to an arbitrary column name."""
    # rooms_band_expr targets ejari_property_sub_type_en; rebuild on `col`.
    from ingestion.store_reference_data_gcs import ROOMS_BAND_MAP
    expr = pl.lit(None, dtype=pl.Utf8)
    lowered = pl.col(col).str.to_lowercase()
    for needle, band in reversed(list(ROOMS_BAND_MAP.items())):
        expr = pl.when(lowered.str.contains(needle, literal=True)).then(pl.lit(band)).otherwise(expr)
    return expr


# ---------------------------------------------------------------------------
# Section 3 — outliers vs local comparables
# ---------------------------------------------------------------------------

def audit_outliers(raw: pl.DataFrame, d2_surplus: pl.DataFrame) -> None:
    n_raw = raw.height
    flats = (
        raw.with_columns(
            pl.col("contract_start_date").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False).alias("start"),
            pl.col("contract_end_date").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False).alias("end"),
            pl.col("annual_amount").cast(pl.Float64, strict=False),
            pl.col("contract_amount").cast(pl.Float64, strict=False),
            pl.col("actual_area").cast(pl.Float64, strict=False),
            pl.col("no_of_prop").cast(pl.Int64, strict=False),
        )
        .filter(
            pl.col("ejari_property_type_en").cast(pl.Utf8).str.to_lowercase().str.contains("flat")
        )
        .with_columns(rooms_band_expr_on("ejari_property_sub_type_en").alias("band"))
    )

    # Corrupt years (what the sanitize window discards)
    bad_years = flats.filter(
        pl.col("start").is_not_null()
        & ~pl.col("start").dt.year().is_between(RENT_YEAR_MIN, RENT_YEAR_MAX)
    ).height
    add("3 Outliers", "corrupt contract years (outside sanitize window)",
        OK if bad_years / max(flats.height, 1) < 0.01 else WARN,
        f"{bad_years:,} rows ({bad_years / max(flats.height, 1):.3%})",
        f"years outside [{RENT_YEAR_MIN},{RENT_YEAR_MAX}] (probe sample itself had a 2204 row)",
        "already excluded by sanitization; monitor the rate")

    sane = flats.filter(
        pl.col("start").is_not_null()
        & pl.col("start").dt.year().is_between(RENT_YEAR_MIN, RENT_YEAR_MAX)
        & pl.col("annual_amount").is_between(RENT_ANNUAL_MIN, RENT_ANNUAL_MAX)
        & pl.col("actual_area").is_between(RENT_AREA_MIN, RENT_AREA_MAX)
    ).with_columns(
        (pl.col("annual_amount") / (pl.col("actual_area") * SQM_TO_SQFT)).alias("rent_psf")
    )
    n = sane.height

    # Local comparable: zone x band median, fallback zone, then global
    comp_zone_band = sane.group_by("area_id", "band").agg(
        pl.col("rent_psf").median().alias("comp_zb"), pl.len().alias("n_zb")
    )
    comp_zone = sane.group_by("area_id").agg(pl.col("rent_psf").median().alias("comp_z"))
    global_med = sane["rent_psf"].median()
    scored = (
        sane.join(comp_zone_band, on=["area_id", "band"], how="left")
        .join(comp_zone, on="area_id", how="left")
        .with_columns(
            pl.when(pl.col("n_zb") >= 8).then(pl.col("comp_zb"))
            .otherwise(pl.coalesce(pl.col("comp_z"), pl.lit(global_med))).alias("comp")
        )
        .with_columns((pl.col("rent_psf") / pl.col("comp")).alias("ratio"))
    )

    absurd = scored.filter((pl.col("rent_psf") > 2000) | (pl.col("rent_psf") < 10)).height
    add("3 Outliers", "absurd PSF surviving sanitize bounds",
        OK if absurd / n < 0.001 else WARN,
        f"{absurd:,} rows ({absurd / n:.3%})",
        "annual PSF > 2,000 or < 10 AED/sqft/yr",
        "candidates for a ratio-based sanitize bound")

    token = scored.filter(pl.col("ratio") < 0.40).height
    high = scored.filter(pl.col("ratio") > 4.0).height
    add("3 Outliers", "token rents (<40% of local comp — IAAO nominal price)",
        WARN if token / n > 0.005 else OK,
        f"{token:,} rows ({token / n:.3%})",
        f"plus {high:,} rows above 4x comp ({high / n:.3%})",
        "scanner already excludes beyond -60%; route as review_only in a cleaning pass")

    # Digit-shift repair candidates (Land Registry typo class, comp instrument)
    shift = scored.filter(
        ((pl.col("ratio") >= 8.0) & (pl.col("ratio") <= 12.5))
        | ((pl.col("ratio") >= 0.08) & (pl.col("ratio") <= 0.125))
    )
    band_area = sane.group_by("band").agg(pl.col("actual_area").median().alias("band_area"))
    shift_ok = (
        shift.join(band_area, on="band", how="left")
        .filter(
            (pl.col("actual_area") / pl.col("band_area")).is_between(0.65, 1.35)
        ).height
    )
    add("3 Outliers", "digit-shift repair candidates (x10 off local comp)",
        OK if shift_ok / n < 0.002 else WARN,
        f"{shift_ok:,} rows ({shift_ok / n:.3%})",
        f"{shift.height:,} at ~10x ratio, {shift_ok:,} with band-credible area",
        "repairable in a cleaning pass (within Land Registry's ~0.05% typo norm)"
        if shift_ok / n < 0.002 else "add rent digit-shift repair before trusting deep discounts")

    # Bulk contracts: no_of_prop semantics (IAAO bulk/allocated prices)
    single = scored.filter(pl.col("no_of_prop") <= 1)
    bulk = scored.filter(pl.col("no_of_prop") > 1)
    if bulk.height:
        m1 = single["rent_psf"].median()
        mb = bulk["rent_psf"].median()
        ratio_bulk = mb / m1 if m1 else float("nan")
        add("3 Outliers", "bulk contracts no_of_prop>1 (allocated-amount risk)",
            ACTION if (ratio_bulk > 1.5 or ratio_bulk < 0.67) and bulk.height / n > 0.002 else WARN
            if bulk.height / n > 0.02 else OK,
            f"{bulk.height:,} rows ({bulk.height / n:.3%})",
            f"median PSF {mb:,.0f} vs {m1:,.0f} single ({ratio_bulk:.2f}x) — "
            f"a ratio far from 1 means annual_amount is a contract TOTAL stamped per line",
            "if ratio >> 1: divide by no_of_prop or route review_only in artifacts")
    else:
        add("3 Outliers", "bulk contracts no_of_prop>1", OK, "0 rows", "none in sample", "none needed")

    # Same-day identical-amount groups >=25% below comp (bulk_allocation mirror)
    grp = (
        scored.group_by("area_id", "start", "annual_amount", "ejari_property_sub_type_id")
        .agg(pl.len().alias("k"), pl.col("ratio").median().alias("gratio"))
        .filter((pl.col("k") >= 3) & (pl.col("gratio") <= 0.75))
    )
    bulk_rows = int(grp["k"].sum()) if grp.height else 0
    add("3 Outliers", "same-day identical-amount groups ≥25% below comp",
        OK if bulk_rows / n < 0.005 else WARN,
        f"{bulk_rows:,} rows in {grp.height:,} groups ({bulk_rows / n:.3%})",
        "bulk_allocation mirror of the sales rule",
        "route review_only in a cleaning pass if material")

    # Durations + annualization proof
    dur = scored.filter(pl.col("end").is_not_null()).with_columns(
        (pl.col("end") - pl.col("start")).dt.total_days().alias("days")
    )
    neg = dur.filter(pl.col("days") <= 0).height
    annualish = dur.filter(pl.col("days").is_between(330, 400)).height
    multi = dur.filter(pl.col("days") > 400)
    mism = 0
    if multi.height:
        mism = multi.with_columns(
            (pl.col("days") / 365.25).alias("years")
        ).filter(
            pl.col("contract_amount").is_not_null()
            & (
                (pl.col("annual_amount") * pl.col("years") - pl.col("contract_amount")).abs()
                / pl.col("contract_amount")
                > 0.20
            )
        ).height
    add("3 Outliers", "durations & annualization",
        OK if neg / max(dur.height, 1) < 0.001 else WARN,
        f"{neg:,} non-positive; {annualish / max(dur.height, 1):.1%} ~12mo; {multi.height:,} multi-year",
        f"multi-year annual*years vs contract_amount mismatch >20%: {mism:,} "
        f"({mism / max(multi.height, 1):.1%} of multi-year)",
        "annual_amount confirmed annualized" if mism / max(multi.height, 1) < 0.1
        else "annual_amount NOT reliably annualized for multi-year — needs handling")

    # Cross-field impossibilities
    imposs = sane.filter(
        ((pl.col("band") == "Studio") & (pl.col("actual_area") > 150))
        | ((pl.col("band") == "4BR+") & (pl.col("actual_area") < 40))
    ).height
    add("3 Outliers", "band/area impossibilities",
        OK if imposs / n < 0.002 else WARN, f"{imposs:,} rows ({imposs / n:.3%})",
        "Studio >150sqm or 4BR+ <40sqm",
        "mislabeled sub-types; negligible unless material")

    # Sensitivity: do D2 duplicates + token rents move the weekly zone medians?
    top_zones = (
        scored.group_by("area_id").agg(pl.len().alias("n")).sort("n", descending=True).head(3)
    )
    drop_keys = set()
    if d2_surplus.height:
        drop_keys = set(zip(d2_surplus["contract_id"].to_list(), d2_surplus["line_number"].to_list()))
    impact_rows = []
    for zid in top_zones["area_id"].to_list():
        z = scored.filter(pl.col("area_id") == zid).with_columns(
            pl.col("start").dt.truncate("1w").alias("week")
        )
        with_med = z.group_by("week").agg(pl.col("rent_psf").median().alias("m"))
        clean = z.filter(pl.col("ratio") >= 0.40)
        if drop_keys:
            clean = clean.filter(
                ~pl.struct(["contract_id", "line_number"]).map_elements(
                    lambda s: (s["contract_id"], s["line_number"]) in drop_keys,
                    return_dtype=pl.Boolean,
                )
            )
        clean_med = clean.group_by("week").agg(pl.col("rent_psf").median().alias("mc"))
        j = with_med.join(clean_med, on="week", how="inner").with_columns(
            ((pl.col("mc") / pl.col("m") - 1).abs()).alias("shift")
        )
        if j.height:
            impact_rows.append(float(j["shift"].max()))
    max_shift = max(impact_rows) if impact_rows else 0.0
    add("3 Outliers", "sensitivity: cleaning impact on weekly zone medians",
        OK if max_shift < 0.02 else WARN,
        f"max weekly median shift {max_shift:.2%}",
        "top-3 zones, medians with vs without token rents + D2 duplicates",
        "medians are robust; cleaning protects the tails, not the center"
        if max_shift < 0.02 else "cleaning materially moves medians — prioritize rules")


# ---------------------------------------------------------------------------
# Section 4 — gaps
# ---------------------------------------------------------------------------

def audit_gaps(pull_log: str | None, weekly: pl.DataFrame | None, recent: pl.DataFrame | None) -> None:
    if pull_log:
        try:
            counts = []
            for line in open(pull_log):
                if "records in" in line and "dld_rent_contracts" in line:
                    counts.append(int(line.split(":")[1].split("records")[0].strip().replace(",", "")))
            if counts:
                med = sorted(counts)[len(counts) // 2]
                low = [i for i, c in enumerate(counts) if c < 0.5 * med]
                add("4 Gaps", "month-level volumes (pull log)",
                    OK if not low else ACTION,
                    f"{len(counts)} chunks, median {med:,}/month",
                    f"months below 50% of median: {[f'chunk {i}={counts[i]:,}' for i in low] or 'none'}",
                    "none needed" if not low else "re-pull the low months — likely truncated chunks")
        except OSError:
            add("4 Gaps", "month-level volumes", WARN, "log unavailable", pull_log, "pass --pull-log")

    if weekly is not None:
        wk = weekly.filter((pl.col("rooms_band") == "All") & (pl.col("segment") == "all"))
        gap_rows = []
        for district in RENT_DISTRICTS:
            d = wk.filter(pl.col("AREA_EN") == district).sort("week")
            if d.height < 2:
                gap_rows.append((district, -1, 0))
                continue
            weeks = d["week"].to_list()
            expected = set()
            cur = weeks[0]
            while cur <= weeks[-1]:
                expected.add(cur)
                cur = cur + timedelta(days=7)
            missing = sorted(expected - set(weeks))
            runs = 0
            if missing:
                runs = 1
                for a, b in zip(missing, missing[1:]):
                    if (b - a).days > 7:
                        runs += 1
            gap_rows.append((district, len(missing), runs))
        worst = sorted(gap_rows, key=lambda r: -r[1])[:4]
        total_missing = sum(max(g, 0) for _, g, _ in gap_rows)
        add("4 Gaps", "week-level continuity (20 scanner districts, All band)",
            OK if total_missing == 0 else WARN,
            f"{total_missing} missing weeks across districts",
            "worst: " + "; ".join(f"{AREA_DISPLAY.get(d, d)}={g}" for d, g, _ in worst),
            "small districts legitimately skip weeks; investigate only clustered runs")

        span = (weekly["week"].min(), weekly["week"].max())
        add("4 Gaps", "overall weekly coverage", OK,
            f"{span[0]} → {span[1]}",
            f"{weekly['week'].n_unique()} distinct weeks",
            "matches RENTS_START → now")

    if recent is not None:
        daily = recent.group_by("start").agg(pl.len().alias("n")).sort("start")
        days = daily["start"].to_list()
        zero_days = 0
        if days:
            expected = {days[0] + timedelta(days=i) for i in range((days[-1] - days[0]).days + 1)}
            zero_days = len(expected - set(days))
        add("4 Gaps", "day-level continuity (recent contracts, all districts)",
            OK if zero_days <= 2 else WARN,
            f"{zero_days} zero-contract days in {len(days)} covered",
            f"window {days[0] if days else '—'} → {days[-1] if days else '—'}",
            "isolated public holidays are normal; clusters mean a feed outage")
        if days:
            lag = (date.today() - days[-1]).days
            add("4 Gaps", "freshness", OK if lag <= 7 else WARN,
                f"{lag} days behind today", f"latest contract start {days[-1]}",
                "Ejari registration lag is normal up to ~1 week")


# ---------------------------------------------------------------------------
# Section 5 — cross-artifact integrity
# ---------------------------------------------------------------------------

def audit_integrity(weekly: pl.DataFrame, recent: pl.DataFrame, index: pl.DataFrame, blob_meta: dict) -> None:
    for name, df in (("rent_weekly_stats", weekly), ("rent_recent_contracts", recent), ("rent_index", index)):
        meta_rows = int(blob_meta.get(name, {}).get("row_count", -1))
        add("5 Integrity", f"{name} blob row_count",
            OK if meta_rows == df.height else ACTION,
            f"{df.height:,} rows", f"blob metadata says {meta_rows:,}",
            "none needed" if meta_rows == df.height else "re-upload")

    bad_q = weekly.filter(
        (pl.col("p10") > pl.col("q1")) | (pl.col("q1") > pl.col("median"))
        | (pl.col("median") > pl.col("q3")) | (pl.col("q3") > pl.col("p90")) | (pl.col("n") < 1)
    ).height
    add("5 Integrity", "weekly_stats quantile ordering", OK if bad_q == 0 else ACTION,
        f"{bad_q} violating rows", "p10<=q1<=median<=q3<=p90 and n>=1",
        "none needed" if bad_q == 0 else "aggregation bug — investigate")

    dup_idx = index.height - index.select(["AREA_EN", "rooms_band", "week"]).unique().height
    add("5 Integrity", "rent_index key uniqueness", OK if dup_idx == 0 else ACTION,
        f"{dup_idx} duplicate keys", "(AREA_EN, rooms_band, week)",
        "none needed" if dup_idx == 0 else "aggregation bug")

    # Recompute weekly stats from the contract artifact for full weeks in coverage
    floor = recent["start"].min()
    first_full_week = floor + timedelta(days=(7 - floor.weekday()) % 7)
    rec = recent.with_columns(pl.col("start").dt.truncate("1w").alias("week")).filter(
        pl.col("week") > first_full_week
    )
    mine = rec.group_by("AREA_EN", "week").agg(
        pl.col("rent_psf").median().round(1).alias("median"), pl.len().alias("n")
    )
    pub = weekly.filter(
        (pl.col("rooms_band") == "All") & (pl.col("segment") == "all")
        & pl.col("AREA_EN").is_in(RENT_DISTRICTS)
    ).select(["AREA_EN", "week", "median", "n"])
    j = mine.join(pub, on=["AREA_EN", "week"], how="inner", suffix="_pub")
    if j.height:
        med_diff = (j["median"] - j["median_pub"]).abs().max()
        n_diff = (j["n"] - j["n_pub"]).abs().max()
        n_mismatch = j.filter(pl.col("n") != pl.col("n_pub")).height
        add("5 Integrity", "recomputed weekly stats vs published",
            OK if (med_diff or 0) <= 0.1 + 1e-9 else WARN,
            f"{j.height:,} overlapping (district, week) cells",
            f"max |median diff| {med_diff}; n mismatches {n_mismatch:,} (max {n_diff}) — "
            "n gaps measure the .unique() dedupe in the contract artifact",
            "pipeline consistent" if (med_diff or 0) <= 0.1 + 1e-9 else "investigate aggregation drift")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: str, context: str) -> None:
    icon = {"OK": "✅", "WARN": "⚠️", "ACTION": "🛑"}
    lines = [
        "# Rent Data Quality Report — Ejari Feed & GCS Artifacts",
        "",
        f"Audited {date.today().isoformat()} · {context}",
        "",
        "Methodology per `reports/data_cleaning_research_report.md`: reason-coded",
        "exclusions (IAAO), outliers judged against local zone×band comparables",
        "(Land Registry price bands / AVM practice), explicit duplicate",
        "definitions, and sensitivity tests against the scanner's weekly zone",
        "medians. Verdicts: ✅ OK · ⚠️ WARN (monitor) · 🛑 ACTION (fix).",
        "",
        "## Executive summary",
        "",
    ]
    actions = [f for f in findings if f["verdict"] == ACTION]
    warns = [f for f in findings if f["verdict"] == WARN]
    if not actions:
        lines.append("**No blocking issues.** " + (
            f"{len(warns)} monitored findings below." if warns else "All checks green."))
    else:
        lines.append(f"**{len(actions)} issues need action**, {len(warns)} to monitor:")
        for f in actions:
            lines.append(f"- 🛑 **{f['check']}** — {f['size']}: {f['action']}")
    for f in warns:
        lines.append(f"- ⚠️ {f['check']} — {f['size']}")
    lines.append("")

    current = None
    for f in findings:
        if f["section"] != current:
            current = f["section"]
            lines += [f"## {current}", "", "| Check | Verdict | Size | Evidence | Action |", "|---|---|---|---|---|"]
        lines.append(
            f"| {f['check']} | {icon[f['verdict']]} {f['verdict']} | {f['size']} | "
            f"{f['evidence']} | {f['action']} |"
        )
        if f is findings[-1] or findings[findings.index(f) + 1]["section"] != current:
            lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nreport written to {path}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-parquet", help="reuse a saved raw sample parquet")
    parser.add_argument("--type-census", help="json with the wire-type census for --raw-parquet")
    parser.add_argument("--raw-only", action="store_true", help="skip GCS artifact checks")
    parser.add_argument("--pull-log", help="rents pull output file for month-level gap checks")
    parser.add_argument("--out", default="reports/rent_data_quality_report.md")
    args = parser.parse_args()

    if args.raw_parquet:
        raw = pl.read_parquet(args.raw_parquet)
        census = json.load(open(args.type_census)) if args.type_census else {}
        print(f"loaded raw sample: {raw.height:,} rows x {len(raw.columns)} cols")
    else:
        raw, census = fetch_raw_sample(DEFAULT_MONTHS)

    months = sorted(raw["_sample_month"].unique().to_list()) if "_sample_month" in raw.columns else []
    context = f"raw sample: {raw.height:,} rows ({', '.join(months)})"

    d2_surplus = audit_duplicates(raw)
    audit_types_and_missing(raw, census)
    audit_outliers(raw, d2_surplus)

    weekly = recent = index = None
    blob_meta: dict = {}
    if not args.raw_only:
        secrets = load_local_secrets()
        bucket = setting(secrets, "GCS_BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
        frames = {}
        for name in ("rent_weekly_stats", "rent_recent_contracts", "rent_index"):
            df, blob = read_parquet_object(secrets, bucket, REFERENCE_OBJECTS[name])
            frames[name] = df
            blob_meta[name] = blob.metadata or {}
        weekly, recent, index = frames["rent_weekly_stats"], frames["rent_recent_contracts"], frames["rent_index"]
        context += (
            f" · artifacts: weekly_stats {weekly.height:,}, recent_contracts "
            f"{recent.height:,}, rent_index {index.height:,} rows"
        )

    audit_gaps(args.pull_log, weekly, recent)
    if not args.raw_only:
        audit_integrity(weekly, recent, index, blob_meta)

    write_report(args.out, context)
    n_action = sum(1 for f in findings if f["verdict"] == ACTION)
    print(f"{len(findings)} checks: {n_action} ACTION, "
          f"{sum(1 for f in findings if f['verdict'] == WARN)} WARN")
    return 1 if n_action else 0


if __name__ == "__main__":
    sys.exit(main())
