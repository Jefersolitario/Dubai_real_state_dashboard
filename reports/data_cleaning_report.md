# Data Cleaning — Rules, Real-Data Counts, and the Enable Decision

**Date:** 2026-07-06 · **Module:** `data_cleaning.py` · **Decision:**
`"data_cleaning": true` shipped in `fair_value_config.json` after the CV A/B
below. Companion: `data_cleaning_research_report.md` (industry best-practice
review that validates this design).

## Why this module exists

Corrupted or non-market records fabricate fake 40–70% "discounts": a price
missing a zero, a bulk portfolio price stamped on each unit, a related-party
transfer at a token price, a partial-ownership share registered against the
whole unit's area. Nobody could actually buy at these prices, but they poison
training and previously surfaced as the backtest's degenerate deep tail
(spreads < −35%). The old handling was blunt (drop on METER_SALE_PRICE
mismatch, trim training tails, distrust the deep tail). The module replaces
this with **classify → repair → route**: every scoreable sale gets a
`dq_rule` + `dq_action` label; nothing is silently deleted.

| Disposition | Meaning | Downstream |
|---|---|---|
| `clean` | untouched | train + score + flag |
| `repaired` | digit-shift typo corrected in place | train + score + flag (labelled) |
| `review_only` | real transfer, not a standalone market price | excluded from training AND flags; shown in the tab's "Excluded suspicious records" expander |
| `quarantine` | unresolvable price/area basis | excluded everywhere, counted |

## What the feed actually allows (measured, not assumed)

Two of the plan's intended instruments turned out to be inert in the DDA feed:

- **METER_SALE_PRICE is mechanically derived** from TRANS_VALUE/ACTUAL_AREA
  (median relative difference 1.3×10⁻⁷; 99.999% within 0.1% on 303k sales).
  It cannot arbitrate which field a typo corrupted. The working instruments
  are the **project median sale price (AED/sqm)** (fallbacks: district×rooms, district,
  global) and the **layout-median registered area** per project×rooms,
  cross-checked against the DLD units registry where PROJECT_NUMBER resolves.
- **PROCEDURE_AREA equals ACTUAL_AREA on every row**, so partial-ownership
  transfers are currently invisible. The `partial_transfer` rule (share ≈
  0.25/0.5/0.75 of area, per-share price at the project median) is implemented and dormant;
  today such transfers surface through their price signature and land in the
  token-transfer review queue instead. This is the honest answer for this
  case: it cannot be repaired into a whole-unit price, so it must never be
  flagged as a deal — only shown to a human, labelled.

## Rules and their real-data counts (24-month snapshot, 303,122 sales)

| Rule | Action | n | Logic |
|---|---|---|---|
| `price_digit_shift` | repaired | **157** (0.05%) | price is ~10^k of the project median sale price (k=±1,±2), recorded area credible for the layout (±35% of project×rooms median or a registry layout match), repaired price within ±35% of the project median |
| `area_digit_shift` | repaired | 0 | symmetric: correcting the area by 10^-k makes it credible; none observed |
| `area_basis_mismatch` | quarantine | 2 | the legacy METER_SALE_PRICE guard, now labelled |
| `partial_transfer` | review_only | 0 | dormant (see above) |
| `bulk_allocation` | review_only | **1,650** (0.54%) | ≥3 same-project same-day identical prices (or identical AED/sqm to 0.1%) with the group ≥25% below the project median. Crucially, **13,294 off-plan developer-launch rows in identical-price batches stay clean** (their prices sit at 99.9% of the project median — those are real primary-market prices; excluding them would delete 4% of the market) |
| `suspected_token_transfer` | review_only | **740** (0.24%) | internally consistent price below 40% of the project median (−60%+); IAAO calls these "sales of convenience… retransacted at only a nominal price" |
| `extreme_price_unexplained` | review_only | 0 | unrepaired >6.5× the project median |
| **total non-clean** | | **2,549 (0.84%)** | within the expected 0.5–2% band |

Example repair: an 85-sqm Princess Tower 1BR registered at AED 81,395
(AED 950/sqm against a project median of 16,948) → repaired ×10 to 813,950.
The two conservatism gates matter: repairs that would still leave the price
>35% from the project median are **not** fabricated into existence — they fall through to
review/quarantine — and repaired rows re-derive METER_SALE_PRICE so the
record stays internally consistent.

## CV A/B — champion config, cleaning off vs on (the enable gate)

Identical data (full 24-month frame), folds, params, and seed; 10-fold
date-ordered TimeSeriesSplit; holdout = trained < 2026-03-01, scored on the
41k sales after (out-of-sample, falling market).

| Metric | OFF | ON | Δ |
|---|---|---|---|
| CV MedAPE | 3.503% | **3.452%** | −0.051pp |
| CV P90 APE (tail errors) | 14.684% | **14.402%** | −0.28pp |
| CV flag_prop (sales pushed under −15% by model error) | 4.326% | **4.155%** | −0.17pp |
| Holdout MedAPE | 3.359% | **3.324%** | −0.035pp |
| Holdout flags (< −15%) | 5.018% | **4.907%** | −0.11pp |
| Holdout deep tail (< −35%) | 181 (0.437%) | **137 (0.332%)** | **−24%** |

Every gate passed: flag_prop improved, MedAPE improved (the bar was only
"don't worsen"), and the deep tail — the noise the outcome backtest told us
to distrust — shrank by a quarter. **Enabled in the shipped config.**

Protocol note: these CV levels (≈3.5%) are not comparable to the campaign's
4.08% headline — the campaign measured on a selection window ending
2026-05-01 with different fold boundaries. Only the on-vs-off deltas in this
table are meaningful; the official accuracy claim remains the campaign's
sequestered-holdout 4.15%.

## Operational notes

- **Raw data stays raw** (owner directive): the GCS snapshot is never
  rewritten with cleaned values. Cleaning runs in-memory inside
  `feature_engineering` (train and inference) whenever the config flag is on,
  so rules can evolve and re-run over unchanged history — the same pattern HM
  Land Registry uses (corrections ship downstream; the register is not
  rewritten).
- The Fair Value tab shows review-routed rows in the **"Excluded suspicious
  records"** expander with plain-language reasons and each row's price next to the
  project median sale price (AED/ft²): the product answer to "how do we deal with these cases" is *show
  them, labelled — never flag them as deals, never hide them.*
- Monitoring rule (from the research report): alarm if monthly repairs exceed
  0.2% or review_only exceeds 2% — a jump means the feed changed, not the
  market.
- Runtime cost: one cleaning pass over 303k rows ≈ 0.8s — negligible against
  feature engineering.
- The bundle should be retrained after enabling (`python train_fair_value.py`)
  so the model itself trains on repaired/routed data; scoring stays
  backward-compatible either way (the flag changes rows, not feature names).

## Smoke coverage

`smoke_test_fair_value.py` plants one row per rule in a synthetic project
(price typo ×10, area typo ×10, half-share transfer, below-median bulk trio,
at-market launch trio, token price at 25% of the project median) and asserts each fires
exactly its own rule, the launch trio stays clean, repairs restore the true
values, and `feature_engineering` with the flag on uses repaired prices while
dropping review rows — 14/14 checks green.
