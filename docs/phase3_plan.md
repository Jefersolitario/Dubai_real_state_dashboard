# Phase 3 Plan — Building Reviews, Comfort Sentiment & External Quality Signals

**Status:** planned (future implementation) · **Written:** 2026-07-05
**Inputs:** `building_quality_research_report.md` (deep research), Campaign 2 findings,
`features_research_report.md` (Phase 2 research)

## Why Phase 3

Campaign 2 proved a pattern: features that *re-describe* price history (comps variants,
project identity, service charges, area rent levels) cannot beat the champion, while
genuinely **new physical information** (units-registry floor/balcony data) produced the
only accept (4.20% → 4.14% CV MedAPE). The remaining error is concentrated in what no
DLD dataset records: **how good the building actually is to live in**. The deep research
confirmed this is a real, measurable driver:

- Residents' sentiment ranked **~6th of all variables** in an XGBoost housing-price model
  (Nature HSSC 2025, "Reputation matters").
- Aspect-based sentiment separates review themes that map 1:1 onto Dubai's comfort
  reality: **AC/cooling quality, humidity/mould, maintenance responsiveness, noise,
  management** — the user's own hypothesis (hot climate ⇒ cooling, humidity control, and
  sun exposure are priced comfort factors).
- Comfort physics is priced: west-facing glass adds 5–8°C radiant load on summer
  afternoons; thermal-comfort effects measured at ~+2.2% on sale prices in
  spatial-hedonic studies; floor level contributes up to ~9.2% in high-rise markets.

**Core Phase 3 hypothesis:** per-building *experienced quality* — review ratings +
comfort-aspect sentiment — explains a slice of the residual that transaction-derived
features cannot, especially the gap between two same-spec towers in the same district.

## Workstream A — Google Places building reviews (primary, ToS-clean)

**Prerequisite:** a Google Maps Platform API key with Places API (New) enabled
(billing: ~US$10–20 one-time for the target coverage; owner action).

1. **Building universe:** top ~500 buildings by 24-month transaction count from the GCS
   snapshot (covers the large majority of scored volume). Key = normalized
   `BUILDING_NAME_EN` + `AREA_EN` + project.
2. **Resolution:** Text Search (`building name + area + Dubai`) → `place_id`;
   Place Details for `rating`, `userRatingCount`, review snippets, and the AI review
   summary. Manual reconciliation pass for the ~10–15% expected fuzzy matches.
3. **Comfort-aspect sentiment:** score review snippets per building on fixed aspects:
   `ac_cooling`, `humidity_mould`, `maintenance`, `noise`, `management`, `handover_quality`.
   Start with a keyword/valence baseline (robust at ~5–10 reviews/building); upgrade to a
   small transformer only if coverage justifies it.
4. **Storage:** `dld_reference/building_reviews.parquet` — one row per building:
   `building_key, place_id, google_rating, google_reviews_count, aspect_* scores,
   matched_confidence, fetched_at`. Refresh quarterly.
5. **Feature group `building_reviews`:** `google_rating` (shrunk toward the mean by
   review count — a 4.8 with 6 reviews must not beat a 4.4 with 900), `log_reviews_count`,
   and the aspect scores. Null where unmatched (HGB-native).

## Workstream B — DLD official building classification (watch list)

The Smart Rental Index assigns every residential building **1–5 stars on 60+ criteria**
(condition, finish quality, maintenance, facility management, amenities) — the ideal
label, government-issued. Currently **not open data** (gateway 404, checked 2026-07-05).

- Re-probe `dld_*` dataset names monthly (script: `store_reference_data_gcs.py --probe`,
  to be added).
- Stopgap if Phase 3 starts before release: manual lookup of the top ~200 buildings in
  DLD's rental-index calculator UI → `building_stars.parquet` (documented as
  manually-collected, dated snapshot).

## Workstream C — comfort/orientation from open geodata (free)

1. **Facade orientation:** OSM UAE building footprints (HDX export) → per-building share
   of facade azimuth facing W/SW (worst radiant load), footprint elongation. Join by
   geocoded building (reuse Workstream A geocodes).
2. **Cooling regime:** area-level district-cooling provider flag (Empower/Emicool served
   communities, manually mapped once onto official `AREA_EN` district names). Note: adds
   signal only for thin areas — AREA_EN categorical already encodes area identity — so
   test it, expect little, and keep expectations documented.
3. Optional (only if A shows quality signal is real): shadowing by taller neighbours from
   OSM heights — deprioritized, high effort.

## Workstream D — live listings (carried over from Phase 2 promise)

Bayut / Property Finder asking prices — **do not scrape**; requires a data partnership or
licensed feed. When available: per-unit floor, view, furnishing, chiller-free flag, and
ask-vs-fair-value spread for *live* deal screening (the original product goal: score deals
you can still buy, not just closed ones). Ejari rent grid (done in Phase 2) remains the
yield display source.

## Modeling & evaluation protocol (unchanged discipline, upgraded metrics)

- Same anti-overfitting protocol: selection window CV (10-fold, date-ordered), ≥0.05pp
  acceptance, sequestered holdout touched once per ship decision.
- **Metric upgrade (from the 2026-07-05 metric review):** alongside MedAPE, every
  iteration records **P90 APE** and **false-flag propensity** (share of validation sales
  below −15% spread), with a no-worsening gate; ship decisions also check **forced-sale
  lift** (retrieval quality: do court-order/auction sales concentrate in the bottom
  spread decile?). Rationale: flags live in the error tail, which MedAPE ignores.
- **Point-in-time honesty:** review ratings/aspect scores are *current* snapshots applied
  to historical sales — same class of caveat as `percent_completed` (which we excluded).
  Mitigations: (a) store `fetched_at` and only score live/recent windows with review
  features in production; (b) disclose in the methodology expander; (c) in CV, verify the
  gain persists when review features are restricted to buildings completed well before
  the selection window (reputation largely formed before the sale).
- Success criteria: CV gain ≥0.1pp with flat-or-better tail metrics, holdout confirmation,
  and (for the distress list specifically) improved forced-sale lift.

## Dashboard surfacing (independent of model gains)

Even if review features don't beat the champion, surface them as **context**: a building
quality panel on flagged deals (Google rating, review count, comfort-aspect flags like
"multiple reviews mention AC problems"), implied gross yield from the Ejari grid, and the
DLD star rating when it becomes available. A flagged discount in a building with poor
comfort sentiment is a *warning*, not a bargain — that context is worth more to a buyer
than 0.1pp of MedAPE.

## Execution order & effort

| Step | Depends on | Effort | Expected value |
|---|---|---|---|
| A. Google reviews + aspect sentiment | API key (owner) | 1–2 days | **High** (primary hypothesis) |
| Metric upgrade in campaign driver | — | hours | High (protects all future decisions) |
| C2. Cooling-provider flag | — | 1 hour | Low (test + document) |
| C1. OSM orientation | A's geocodes | 1 day | Medium |
| B. DLD stars probe/stopgap | DLD release | recurring probe | High when released |
| Dashboard quality panel | A | 0.5 day | High (product value, model-independent) |
| D. Live listings | partnership | external | High, blocked |

## Engineering debt (queued from the 2026-07-05 code review; do alongside Phase 3)

- **Table-driven feature-group registry** in `fair_value_model.py`: one
  `FEATURE_GROUPS` map (columns, reference requirements, dependencies, display
  labels) instead of 5–6 parallel edits per new group across two files.
- **Bundle-carried signal-strength calibration**: compute per-segment expected
  error at train time and store it in the bundle's `metrics`, replacing the
  hard-coded `EXPECTED_ERR_*` constants so weekly retrains can't silently
  miscalibrate the ranking.
- **Teach `optimize_fair_value.py` the reference-backed groups** (load reference
  frames, add ladder proposals); today a champion-protection guard merely stops
  it from regressing the shipping config.
- **`UNITS_ROOMS_CHUNKS` completeness**: derive the rooms partition from a
  distinct-values probe (or add a catch-all chunk) so a new upstream label
  cannot silently drop units from the registry.

**Out of scope for Phase 3** (documented dead ends, do not revisit without new data):
official sale index (frozen at 2024-05 on the gateway), rent level/yield/density as
*prediction* features (rejected in Campaign 2 Batches B/C), service charges and project
metadata as standalone features (rejected in Batch A), per-unit floor for stacked layouts
(structurally impossible from DLD transactions alone — solved only by listings data).
