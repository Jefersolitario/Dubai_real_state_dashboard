# Data Quality Report — DLD 24-Month Snapshot

Audited 2026-07-04 · 322,828 deduplicated rows (2024-07-01 → 2026-07-02), 23 normalized columns · apartment Sales subset: 271,716 rows.

## Executive summary

The snapshot is structurally clean: **zero duplicate transaction IDs, zero unparsable dates, no empty-string pollution, and the recorded area agrees with the official AED/sqm price on 99.996% of sales** (1 mismatched row). The two material issues are (1) **non-arm's-length procedures hiding inside the Sales group** — developer registrations and financing structures that price at 0.42–0.75× market and both contaminate training and surface as fake bargains — and (2) **high missingness in the amenity proximity fields** (27–40% null). Both have concrete fixes below; the first is fixed in code with this report.

## 1. Wrong / misleading labels

| Issue | Size | Evidence | Action |
|---|---|---|---|
| **Non-market procedures inside GROUP_EN="Sales"** | ~5,500 rows (2.0%) | Median PSF vs market: Sell Development **0.49×**, Development Registration **0.53×**, Delayed Development 0.52×, Sale On Payment Plan **0.63×**, Lease to Own 0.75× — vs Sell 0.78–1.07× for genuine market procedures | **FIXED**: `feature_engineering` now excludes procedures matching development-registration / lease-to-own / payment-plan patterns from both training and scoring (they are not buyable market deals) |
| Non-residential ROOMS_EN inside Flat subtype | 7 rows ("Shop" 4, "Office" 3) | Mislabeled sub-type | Negligible; rooms_ord = null → NaN handled natively |
| "PENTHOUSE" rooms label | 134 rows | Not a bedroom count; rooms_ord = null | Acceptable (NaN); optional future `is_penthouse` flag for the luxury segment |
| Project-name case variants | 1 collision, 156 rows ("IMPERIAL RESIDENCE" vs "Imperial Residence") | Encoder treats them as two categories | **FIXED**: project/master-project/building names are upper-cased in `feature_engineering` before encoding |
| Mortgage TRANS_VALUE = loan, not price | 40,407 rows | Median mortgage PSF is **0.58×** the sales median — empirically confirms the loan-amount assumption | Already excluded from the model (Sales-only) — assumption now verified with data |
| Gifts group | 10,705 rows | Non-market transfers | Already excluded (GROUP filter) |

## 2. Missing data & imputation strategy

| Column | Null rate | Current handling | Recommendation |
|---|---|---|---|
| NEAREST_MALL_EN | 40.0% | "UNKNOWN" category | Keep — missingness is informative for trees (older districts lack tagging). Phase 2: replace with numeric distances (OSM geocoding), impute by district centroid |
| NEAREST_METRO_EN | 38.8% | "UNKNOWN" category | Same as above |
| NEAREST_LANDMARK_EN | 26.9% | "UNKNOWN" category | Same as above |
| PROJECT_EN | 9.8% | "UNKNOWN" category; project comp masked to null (never a pooled fake comp) | Keep. Phase 2: recover project from BUILDING_NAME_EN (0% null, 4,257 buildings) via a building→project mapping — likely the best imputation available |
| MASTER_PROJECT_EN | 9.0% | "UNKNOWN" category | Keep |
| project_comp_psf (derived) | ~10% incl. cold-start | native NaN (HGB splits on missingness) | Keep — do NOT impute with area averages (tested in the optimization loop: area comps add nothing) |
| ROOMS_EN | 15 rows + 156 unparsable | rooms_ord = NaN | Keep |
| TOTAL_BUYER / TOTAL_SELLER | 32 rows (+13 zero-buyer rows) | NaN / used as numeric | Keep; zero-party rows are registration artifacts, negligible |

Guiding principle (already implemented): **categorical missingness becomes an explicit "UNKNOWN" level, numeric missingness stays NaN for the tree model, and nothing derived from the target is ever imputed** — rows with unusable price/area are dropped, not filled.

## 3. Typos & vocabulary hygiene

- Exactly **one** case-collision across 4,800+ project names (fixed via upper-casing). No whitespace or spelling near-duplicates detected at the normalized level — DLD's English vocabulary is centrally controlled and clean.
- `PARKING` is a binary has-parking flag (1: 266,659 / 0: 5,057), not a space count — the feature name `parking_count` is cosmetic only.
- `USAGE_EN`, `PROP_TYPE_EN`, `PROP_SB_TYPE_EN` are constant (Residential/Unit/Flat) in this apartment snapshot — zero information, correctly excluded from features.

## 4. Numeric outliers (kept by design, guarded downstream)

| Check | Count | Handling |
|---|---|---|
| ACTUAL_AREA < 10 sqm | 77 | survive scoring; PSF trim removes from training |
| ACTUAL_AREA > 1,000 sqm | 124 | same |
| TRANS_VALUE < AED 50k | 277 | token/nominal transfers; annotated "deep discount", never labelled distressed without an independent signal |
| PSF < AED 100/sqft | 355 | same |
| PSF > AED 10k/sqft | 99 | training-trimmed; scored |
| Area vs METER_SALE_PRICE mismatch > 10% | **1 row** | dropped by the units guard — the guard is cheap insurance, and the underlying data is excellent |

## 5. Coverage & display gaps (non-model)

- Date coverage is complete: 25 months, no gaps (2026-07 is a partial month by definition).
- **39 districts lack a friendly display name**, including high-volume ones: WADI AL SAFA 5 (14.6k sales), WADI AL SAFA 3 (8.5k), AL BARSHAA SOUTH SECOND (7.1k), AL WASL, AL HEBIAH FIFTH. They pass through with raw DLD names and tier "UNKNOWN". Recommend extending `AREA_DISPLAY`/`TIER_AREAS` for the top ~10 — needs local market knowledge to name communities correctly, so left as a follow-up rather than guessed.

## Fixes applied with this report

1. `feature_engineering` excludes non-arm's-length procedures (development registrations, lease-to-own, payment plans) from training and scoring — verified on the untouched holdout (see commit).
2. Project / master-project / building names are case-normalized before encoding.
