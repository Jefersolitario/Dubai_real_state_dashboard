"""Transaction data cleaning: classify, repair, and route — never silently drop.

Corrupted DLD records fabricate fake discounts that poison fair-value
training and surface as bogus "distressed" flags. This module labels every
row with a disposition instead of blindly clipping or deleting:

- ``clean``       — untouched, usable for training and flagging.
- ``repaired``    — an obvious digit-shift typo (missing/extra zero) was
                    corrected in place; the row is kept.
- ``review_only`` — a real transfer whose registered price is NOT a
                    standalone market price (bulk-deal allocation, suspected
                    related-party/token transfer, partial-ownership share).
                    Excluded from training and from the distress flag list,
                    but surfaced to a human with its reason.
- ``quarantine``  — unusable record (price/area basis cannot be resolved).

Instrument note: DLD's METER_SALE_PRICE is mechanically derived from
TRANS_VALUE / ACTUAL_AREA in the open-data feed (they agree to ~1e-7 on
303k sales), so it CANNOT arbitrate which field a typo corrupted. The
working instruments are robust comparables (project median AED/sqm, with
area fallbacks) and the layout-median registered area per project x rooms
(optionally cross-checked against the DLD units registry). Likewise
PROCEDURE_AREA currently equals ACTUAL_AREA on every feed row, so the
partial-ownership rule is future-proofing: it fires only if DLD ever
populates the share area, and today such transfers are caught by the
token-transfer price signature instead.

Pure Polars, no Streamlit imports — usable offline and in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Ratio zones (row AED/sqm vs comparable) that look like a digit shift.
# key = decimal exponent k restoring the ratio: repaired ratio = ratio * 10^k.
DIGIT_SHIFT_ZONES: dict[int, tuple[float, float]] = {
    2: (0.005, 0.015),   # two missing zeros in price (or two extra in area)
    1: (0.05, 0.15),     # one missing zero in price (or one extra in area)
    -1: (6.5, 15.0),     # one extra zero in price (or one missing in area)
    -2: (65.0, 150.0),   # two extra zeros in price
}

# A repair is accepted only if the corrected AED/sqm lands within this band
# of the comparable — otherwise we would be fabricating a price.
REPAIRED_RATIO_BAND = (0.65, 1.35)

# The recorded area is "credible" when within this band of the layout-median
# registered area for the same project x rooms (or a registry layout match).
AREA_CREDIBLE_BAND = (0.65, 1.35)

# Registry areas within this relative tolerance of ACTUAL_AREA count as a
# layout match (mirrors the 2dp area key used by the units-registry join).
REGISTRY_AREA_TOLERANCE = 0.02

# METER_SALE_PRICE consistency guard (kept from the legacy wrong-area drop;
# only fires on rows the feed did NOT derive, e.g. older snapshot vintages).
MSP_MISMATCH_TOLERANCE = 0.10

# Partial-ownership share signatures: PROCEDURE_AREA / ACTUAL_AREA near a
# simple fraction, with the per-share price consistent with the comparable.
PARTIAL_FRACTIONS = (0.25, 0.5, 0.75)
PARTIAL_FRACTION_TOLERANCE = 0.02

# Bulk-deal allocation: >=3 same-project same-day transactions at an
# identical TRANS_VALUE (or identical AED/sqm to 0.1%) whose group price
# sits well below the comparable. At-market bulk groups (developer launch
# lists) are genuine primary-market prices and stay clean.
BULK_MIN_GROUP = 3
BULK_MAX_RATIO = 0.75

# Suspected related-party / token transfer: internally consistent record
# priced below this fraction of the comparable. Deliberately conservative
# (-60%): the 40-60% discount zone may contain real fire-sales and is left
# to the model + human review, not the cleaner.
TOKEN_MAX_RATIO = 0.40

# Unexplained extreme overpricing (unrepaired high-side ratio) — routed to
# review so a 10x typo cannot silently inflate comparables.
EXTREME_HIGH_RATIO = 6.5

# Minimum group sizes before a median is trusted as a comparable.
MIN_PROJECT_COMP_N = 20
MIN_AREA_ROOMS_COMP_N = 20
MIN_LAYOUT_AREA_N = 5

ACTION_CLEAN = "clean"
ACTION_REPAIRED = "repaired"
ACTION_REVIEW = "review_only"
ACTION_QUARANTINE = "quarantine"

# Rules in reporting order.
RULE_LABELS: dict[str, str] = {
    "price_digit_shift": "price typo repaired (missing/extra zeros)",
    "area_digit_shift": "area typo repaired (missing/extra zeros)",
    "area_basis_mismatch": "price/area basis mismatch vs METER_SALE_PRICE",
    "nonpositive_price_area": "non-positive price or area",
    "partial_transfer": "partial-ownership share, not a whole-unit price",
    "bulk_allocation": "bulk-deal allocation below market",
    "suspected_token_transfer": "suspected related-party/token price",
    "extreme_price_unexplained": "extreme price, cause unresolved",
}

_EXAMPLE_COLUMNS = [
    "INSTANCE_DATE", "PROJECT_EN", "AREA_EN", "ROOMS_EN",
    "dq_orig_trans_value", "TRANS_VALUE", "dq_orig_actual_area",
    "ACTUAL_AREA", "dq_comp_ratio", "dq_rule", "dq_action",
]


@dataclass
class CleaningReport:
    """Counts and examples from one :func:`clean_transactions` pass."""

    total_rows: int
    rule_counts: dict[str, int] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, pl.DataFrame] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-block summary of dispositions."""
        lines = [f"rows examined: {self.total_rows:,}"]
        for action in (ACTION_CLEAN, ACTION_REPAIRED, ACTION_REVIEW, ACTION_QUARANTINE):
            n = self.action_counts.get(action, 0)
            share = n / self.total_rows if self.total_rows else 0.0
            lines.append(f"  {action:12s} {n:>8,} ({share:.3%})")
        for rule, label in RULE_LABELS.items():
            if self.rule_counts.get(rule):
                lines.append(f"    {rule:28s} {self.rule_counts[rule]:>7,}  {label}")
        return "\n".join(lines)


def kept_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Rows usable for training and scoring (clean + repaired)."""
    return df.filter(pl.col("dq_action").is_in([ACTION_CLEAN, ACTION_REPAIRED]))


def review_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Real transfers at non-market prices — show to a human, never flag."""
    return df.filter(pl.col("dq_action") == ACTION_REVIEW)


def quarantined_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Unusable records (unresolved price/area basis)."""
    return df.filter(pl.col("dq_action") == ACTION_QUARANTINE)


def _registry_layout_match(
    df: pl.DataFrame, reference: dict[str, pl.DataFrame] | None
) -> pl.DataFrame:
    """Add ``_dq_registry_match``: ACTUAL_AREA matches a real layout.

    Uses the DLD units registry (``reference["units"]`` + the
    project_number -> project_id bridge in ``reference["projects"]``) when
    provided; otherwise the column is null and the layout-median heuristic
    decides area credibility alone.
    """
    if (
        reference is None
        or "units" not in reference
        or "projects" not in reference
        or "PROJECT_NUMBER" not in df.columns
    ):
        return df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("_dq_registry_match"))

    bridge = (
        reference["projects"]
        .select(
            pl.col("project_number").cast(pl.Int64, strict=False).alias("_pn"),
            pl.col("project_id").cast(pl.Int64, strict=False).alias("_pid"),
        )
        .drop_nulls()
        .unique("_pn")
    )
    layouts = (
        reference["units"]
        .select(
            pl.col("project_id").cast(pl.Int64, strict=False).alias("_pid"),
            pl.col("actual_area").cast(pl.Float64, strict=False).alias("_reg_area"),
        )
        .drop_nulls()
        .unique(["_pid", "_reg_area"])
    )
    probe = (
        df.select(
            pl.col("PROJECT_NUMBER").cast(pl.Int64, strict=False).alias("_pn"),
            pl.col("ACTUAL_AREA").cast(pl.Float64, strict=False).alias("_aa"),
        )
        .with_row_index("_ri")
        .drop_nulls(["_pn", "_aa"])
        .join(bridge, on="_pn", how="inner")
        .join(layouts, on="_pid", how="inner")
        .filter(
            ((pl.col("_reg_area") - pl.col("_aa")) / pl.col("_aa")).abs()
            <= REGISTRY_AREA_TOLERANCE
        )
        .select(pl.col("_ri").unique())
        .with_columns(pl.lit(True).alias("_dq_registry_match"))
    )
    return (
        df.with_row_index("_ri")
        .join(probe, on="_ri", how="left")
        .drop("_ri")
    )


def _with_comparables(df: pl.DataFrame) -> pl.DataFrame:
    """Attach robust AED/sqm comparables and layout-median areas.

    Comparable precedence: project median (>= MIN_PROJECT_COMP_N rows) ->
    area x rooms median -> area median -> global median. Medians are taken
    over the full frame including corrupt rows — medians shrug off the
    <1% corruption this module exists to catch. Null projects fall back to
    an area-scoped key so they are never pooled into one global group.
    """
    proj_key = pl.coalesce(
        pl.col("PROJECT_EN").cast(pl.Utf8).str.to_uppercase(),
        pl.format("<no-project>:{}", pl.col("AREA_EN").cast(pl.Utf8).fill_null("?")),
    )
    df = df.with_columns(
        pl.col("TRANS_VALUE").cast(pl.Float64, strict=False).alias("_dq_tv"),
        pl.col("ACTUAL_AREA").cast(pl.Float64, strict=False).alias("_dq_aa"),
        proj_key.alias("_dq_proj"),
        pl.col("AREA_EN").cast(pl.Utf8).fill_null("?").alias("_dq_area"),
        pl.col("ROOMS_EN").cast(pl.Utf8).fill_null("?").alias("_dq_rooms"),
        pl.col("INSTANCE_DATE").cast(pl.Utf8).str.slice(0, 10).alias("_dq_day"),
    )
    psm = pl.col("_dq_tv") / pl.col("_dq_aa")
    df = df.with_columns(psm.alias("_dq_psm"))
    valid = pl.col("_dq_psm").is_finite() & (pl.col("_dq_psm") > 0)
    psm_v = pl.when(valid).then(pl.col("_dq_psm"))
    aa_v = pl.when(valid).then(pl.col("_dq_aa"))
    df = df.with_columns(
        psm_v.median().over("_dq_proj").alias("_dq_proj_med"),
        psm_v.count().over("_dq_proj").alias("_dq_proj_n"),
        psm_v.median().over("_dq_area", "_dq_rooms").alias("_dq_ar_med"),
        psm_v.count().over("_dq_area", "_dq_rooms").alias("_dq_ar_n"),
        psm_v.median().over("_dq_area").alias("_dq_a_med"),
        psm_v.median().alias("_dq_g_med"),
        aa_v.median().over("_dq_proj", "_dq_rooms").alias("_dq_layout_med"),
        aa_v.count().over("_dq_proj", "_dq_rooms").alias("_dq_layout_n"),
        aa_v.median().over("_dq_area", "_dq_rooms").alias("_dq_ar_layout_med"),
    )
    comp = (
        pl.when(pl.col("_dq_proj_n") >= MIN_PROJECT_COMP_N).then(pl.col("_dq_proj_med"))
        .when(pl.col("_dq_ar_n") >= MIN_AREA_ROOMS_COMP_N).then(pl.col("_dq_ar_med"))
        .otherwise(pl.coalesce(pl.col("_dq_a_med"), pl.col("_dq_g_med")))
    )
    layout_med = (
        pl.when(pl.col("_dq_layout_n") >= MIN_LAYOUT_AREA_N).then(pl.col("_dq_layout_med"))
        .otherwise(pl.coalesce(pl.col("_dq_ar_layout_med"), pl.col("_dq_layout_med")))
    )
    return df.with_columns(
        comp.alias("_dq_comp"),
        layout_med.alias("_dq_layout"),
        (pl.col("_dq_psm") / comp).alias("dq_comp_ratio"),
    )


def _digit_shift_exprs(registry_available: bool) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """(k, price_fix, area_fix) expressions for the digit-shift zones.

    ``k`` is the decimal exponent restoring the price/comparable ratio.
    ``price_fix``: the recorded area is credible (layout median band or a
    registry layout match), so TRANS_VALUE is the corrupted field.
    ``area_fix``: correcting ACTUAL_AREA by 10^-k lands it in the credible
    band, so the area is the corrupted field. Both require the repaired
    ratio inside REPAIRED_RATIO_BAND.
    """
    ratio = pl.col("dq_comp_ratio")
    k = pl.lit(None, dtype=pl.Int32)
    for exp, (lo, hi) in DIGIT_SHIFT_ZONES.items():
        k = pl.when(ratio.is_between(lo, hi)).then(pl.lit(exp, dtype=pl.Int32)).otherwise(k)
    factor = pl.lit(10.0).pow(k.cast(pl.Float64))
    repaired_ok = (ratio * factor).is_between(*REPAIRED_RATIO_BAND)
    aa_ratio = pl.col("_dq_aa") / pl.col("_dq_layout")
    area_credible = aa_ratio.is_between(*AREA_CREDIBLE_BAND).fill_null(False)
    if registry_available:
        area_credible = area_credible | pl.col("_dq_registry_match").fill_null(False)
    price_fix = k.is_not_null() & repaired_ok & area_credible
    corrected_aa_ratio = aa_ratio * pl.lit(10.0).pow(-k.cast(pl.Float64))
    area_fix = (
        k.is_not_null()
        & repaired_ok
        & ~price_fix
        & corrected_aa_ratio.is_between(*AREA_CREDIBLE_BAND).fill_null(False)
    )
    return k, price_fix, area_fix


def clean_transactions(
    df: pl.DataFrame, reference: dict[str, pl.DataFrame] | None = None
) -> tuple[pl.DataFrame, CleaningReport]:
    """Label every transaction with a data-quality disposition; repair typos.

    Designed for frames already filtered to apartment Sales (the rules'
    priors are calibrated there), but degrades gracefully on any frame with
    ``TRANS_VALUE`` and ``ACTUAL_AREA``. Optional columns used when present:
    ``METER_SALE_PRICE``, ``PROCEDURE_AREA``, ``PROJECT_EN``, ``AREA_EN``,
    ``ROOMS_EN``, ``INSTANCE_DATE``, ``PROJECT_NUMBER``. ``reference`` may
    carry the GCS ``projects`` + ``units`` frames for registry layout checks.

    Returns the FULL frame (no rows dropped) with ``dq_rule`` /
    ``dq_action`` columns, digit-shift repairs applied in place to
    ``TRANS_VALUE`` / ``ACTUAL_AREA`` (cast to Float64, with
    ``METER_SALE_PRICE`` re-derived for repaired rows), plus a
    :class:`CleaningReport`. Route with :func:`kept_rows`,
    :func:`review_rows`, :func:`quarantined_rows`.
    """
    for required in ("TRANS_VALUE", "ACTUAL_AREA"):
        if required not in df.columns:
            raise ValueError(f"clean_transactions requires column {required!r}")
    total = df.height
    if total == 0:
        out = df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("dq_rule"),
            pl.lit(ACTION_CLEAN).alias("dq_action"),
        )
        return out, CleaningReport(total_rows=0)

    optional = ["METER_SALE_PRICE", "PROCEDURE_AREA", "PROJECT_EN", "AREA_EN",
                "ROOMS_EN", "INSTANCE_DATE", "PROJECT_NUMBER"]
    df = df.with_columns(
        pl.lit(None).alias(c) for c in optional if c not in df.columns
    )
    df = _with_comparables(df)
    df = _registry_layout_match(df, reference)
    registry_available = df["_dq_registry_match"].dtype == pl.Boolean

    tv, aa = pl.col("_dq_tv"), pl.col("_dq_aa")
    psm, ratio = pl.col("_dq_psm"), pl.col("dq_comp_ratio")
    nonpositive = tv.is_null() | aa.is_null() | (tv <= 0) | (aa <= 0)

    k, price_fix, area_fix = _digit_shift_exprs(registry_available)

    msp = pl.col("METER_SALE_PRICE").cast(pl.Float64, strict=False)
    msp_mismatch = (
        msp.is_not_null() & (msp > 0)
        & (((psm - msp) / msp).abs() > MSP_MISMATCH_TOLERANCE)
    ).fill_null(False)

    pa = pl.col("PROCEDURE_AREA").cast(pl.Float64, strict=False)
    share = pa / aa
    share_price_ok = ((tv / pa) / pl.col("_dq_comp")).is_between(*REPAIRED_RATIO_BAND)
    partial = (
        (pa > 0)
        & pl.any_horizontal(
            *((share - f).abs() <= PARTIAL_FRACTION_TOLERANCE for f in PARTIAL_FRACTIONS)
        )
        & share_price_ok
    ).fill_null(False)

    # Bulk groups: identical TRANS_VALUE, or identical AED/sqm to ~0.1%
    # (log-bin of width 1e-3), same project and day.
    psm_bin = (psm.log() * 1000).round(0)
    df = df.with_columns(
        pl.len().over("_dq_proj", "_dq_day", "_dq_tv").alias("_dq_tvgrp_n"),
        ratio.median().over("_dq_proj", "_dq_day", "_dq_tv").alias("_dq_tvgrp_ratio"),
        pl.len().over("_dq_proj", "_dq_day", psm_bin).alias("_dq_bingrp_n"),
        ratio.median().over("_dq_proj", "_dq_day", psm_bin).alias("_dq_bingrp_ratio"),
    )
    bulk = (
        (
            (pl.col("_dq_tvgrp_n") >= BULK_MIN_GROUP)
            & (pl.col("_dq_tvgrp_ratio") <= BULK_MAX_RATIO)
        )
        | (
            (pl.col("_dq_bingrp_n") >= BULK_MIN_GROUP)
            & (pl.col("_dq_bingrp_ratio") <= BULK_MAX_RATIO)
        )
    ).fill_null(False)

    token = (ratio < TOKEN_MAX_RATIO).fill_null(False)
    extreme_high = (ratio > EXTREME_HIGH_RATIO).fill_null(False)

    # First matching rule wins; order encodes precedence.
    rule = (
        pl.when(nonpositive).then(pl.lit("nonpositive_price_area"))
        .when(price_fix).then(pl.lit("price_digit_shift"))
        .when(area_fix).then(pl.lit("area_digit_shift"))
        .when(msp_mismatch).then(pl.lit("area_basis_mismatch"))
        .when(partial).then(pl.lit("partial_transfer"))
        .when(bulk).then(pl.lit("bulk_allocation"))
        .when(token).then(pl.lit("suspected_token_transfer"))
        .when(extreme_high).then(pl.lit("extreme_price_unexplained"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    action_for_rule = {
        "nonpositive_price_area": ACTION_QUARANTINE,
        "price_digit_shift": ACTION_REPAIRED,
        "area_digit_shift": ACTION_REPAIRED,
        "area_basis_mismatch": ACTION_QUARANTINE,
        "partial_transfer": ACTION_REVIEW,
        "bulk_allocation": ACTION_REVIEW,
        "suspected_token_transfer": ACTION_REVIEW,
        "extreme_price_unexplained": ACTION_REVIEW,
    }
    df = df.with_columns(rule.alias("dq_rule"))
    df = df.with_columns(
        pl.col("dq_rule")
        .replace_strict(action_for_rule, default=ACTION_CLEAN)
        .alias("dq_action"),
        pl.when(rule.is_in(["price_digit_shift", "area_digit_shift"]))
        .then(k)
        .alias("_dq_k"),
    )

    # Apply repairs in place; keep originals for the report and re-derive
    # METER_SALE_PRICE so the repaired row stays internally consistent.
    kf = pl.lit(10.0).pow(pl.col("_dq_k").cast(pl.Float64))
    df = df.with_columns(
        pl.col("TRANS_VALUE").alias("dq_orig_trans_value"),
        pl.col("ACTUAL_AREA").alias("dq_orig_actual_area"),
        pl.when(pl.col("dq_rule") == "price_digit_shift")
        .then(tv * kf).otherwise(tv).alias("TRANS_VALUE"),
        pl.when(pl.col("dq_rule") == "area_digit_shift")
        .then(aa / kf).otherwise(aa).alias("ACTUAL_AREA"),
    )
    df = df.with_columns(
        pl.when(pl.col("dq_action") == ACTION_REPAIRED)
        .then(pl.col("TRANS_VALUE") / pl.col("ACTUAL_AREA"))
        .otherwise(msp)
        .alias("METER_SALE_PRICE")
    )

    rule_counts = {
        r: n for r, n in df.group_by("dq_rule").len().iter_rows() if r is not None
    }
    action_counts = dict(df.group_by("dq_action").len().iter_rows())
    examples: dict[str, pl.DataFrame] = {}
    for r in rule_counts:
        examples[r] = (
            df.filter(pl.col("dq_rule") == r)
            .select([c for c in _EXAMPLE_COLUMNS if c in df.columns])
            .head(5)
        )
    report = CleaningReport(
        total_rows=total,
        rule_counts=rule_counts,
        action_counts=action_counts,
        examples=examples,
    )
    df = df.drop([c for c in df.columns if c.startswith("_dq_")])
    df = df.drop(["dq_orig_trans_value", "dq_orig_actual_area"])
    return df, report
