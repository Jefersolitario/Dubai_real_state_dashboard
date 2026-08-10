# Rent Data Quality Report — Ejari Feed & GCS Artifacts

Audited 2026-08-10 · raw sample: 107,546 rows (2024-06, 2026-07) · artifacts: weekly_stats 132,493, recent_contracts 87,412, rent_index 55,472 rows

Methodology per `reports/data_cleaning_research_report.md`: reason-coded
exclusions (IAAO), outliers judged against local zone×band comparables
(Land Registry price bands / AVM practice), explicit duplicate
definitions, and sensitivity tests against the scanner's weekly zone
medians. Verdicts: ✅ OK · ⚠️ WARN (monitor) · 🛑 ACTION (fix).

## Executive summary

**No blocking issues.** 12 monitored findings below.
- ⚠️ D1 exact full-row duplicates — 52 rows (0.048%)
- ⚠️ D2 key duplicates (contract_id, line_number) — 52 surplus rows (0.048%)
- ⚠️ D4 re-registration suspects (same attrs, different id) — 5,468 surplus rows (5.084%)
- ⚠️ pagination stability (date-ordered fetch) — 13,660 dupes, 13,654 skipped of 54,953 (2024-06)
- ⚠️ absurd PSF surviving sanitize bounds — 1,815 rows (2.340%)
- ⚠️ token rents (<40% of local comp — IAAO nominal price) — 1,455 rows (1.876%)
- ⚠️ digit-shift repair candidates (x10 off local comp) — 269 rows (0.347%)
- ⚠️ bulk contracts no_of_prop>1 (allocated-amount risk) — 2,855 rows (3.681%)
- ⚠️ same-day identical-amount groups ≥25% below comp — 1,041 rows in 235 groups (1.342%)
- ⚠️ band/area impossibilities — 213 rows (0.275%)
- ⚠️ week-level continuity (20 scanner districts, All band) — 4 missing weeks across districts
- ⚠️ recomputed weekly stats vs published — 550 overlapping (district, week) cells

## 1 Duplicates

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| D1 exact full-row duplicates | ⚠️ WARN | 52 rows (0.048%) | all-column dupes 52; excluding load_timestamp 52 | residual boundary repeats — removed by the pull's (contract_id, line_number) dedupe |
| D2 key duplicates (contract_id, line_number) | ⚠️ WARN | 52 surplus rows (0.048%) | 52 duplicated keys, 0 with conflicting amount/area | removed by the pull's (contract_id, line_number) dedupe |
| D3 multi-line contracts (multi-property, legit) | ✅ OK | 1,635 contracts (1.81%) | lines==no_of_prop for 1,538/1,635 of them | document semantics; not duplicates |
| D4 re-registration suspects (same attrs, different id) | ⚠️ WARN | 5,468 surplus rows (5.084%) | 3,100 attribute groups with >1 contract_id | monitor; likely same-building identical units + genuine re-registrations |
| pagination stability (date-ordered fetch) | ⚠️ WARN | 13,660 dupes, 13,654 skipped of 54,953 (2024-06) | date-ordered paging vs the contract_id-ordered sample — the gateway defect that corrupted every pull before the contract_id fix | production already orders by contract_id; never revert to date ordering (this check is the regression sentinel) |

## 2 Types & missing

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| wire type of annual_amount | ✅ OK | 0 string values (0.000%) | wire types: {'float': 107546} | none needed |
| wire type of contract_amount | ✅ OK | 0 string values (0.000%) | wire types: {'float': 107546} | none needed |
| wire type of actual_area | ✅ OK | 0 string values (0.000%) | wire types: {'float': 104604, 'NoneType': 2942} | none needed |
| wire type of no_of_prop | ✅ OK | 0 string values (0.000%) | wire types: {'int': 107546} | none needed |
| wire type of line_number | ✅ OK | 0 string values (0.000%) | wire types: {'int': 107546} | none needed |
| unparsable numerics in annual_amount | ✅ OK | 0 rows | values that production casting silently nulls | none needed |
| unparsable numerics in actual_area | ✅ OK | 0 rows | values that production casting silently nulls | none needed |
| unparsable dates in contract_start_date | ✅ OK | 0 rows | post slice-10 ISO parse | none needed |
| unparsable dates in contract_end_date | ✅ OK | 0 rows | post slice-10 ISO parse | none needed |
| null/empty rates (key columns) | ✅ OK | project_name_en=73.7%; actual_area=2.7%; ejari_property_sub_type_en=0.0%; ejari_property_type_en=0.0% | empty strings: 0 total | project/master fields expected sparse (Ejari has no building key) |
| area_name_en case/whitespace variants | ✅ OK | 0 collapsing variants | 170 raw vs 170 normalized names | strip+upper already applied at slim time (parity with sales path) |
| rooms-band mapping coverage (flats) | ✅ OK | 479 unmapped (0.59%) | top unmapped: Room in labor Camp=257; Room=199; Duplex=13; Office=10 | acceptable (folds into the All band) |
| contract_reg_type_en vocabulary | ✅ OK | Renew=58,815; New=48,731 | labels containing 'new' without starting with it: ['Renew'] | revisit the matcher |

## 3 Outliers

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| corrupt contract years (outside sanitize window) | ✅ OK | 0 rows (0.000%) | years outside [2020,2027] (probe sample itself had a 2204 row) | already excluded by sanitization; monitor the rate |
| absurd PSF surviving sanitize bounds | ⚠️ WARN | 1,815 rows (2.340%) | annual PSF > 2,000 or < 10 AED/sqft/yr | candidates for a ratio-based sanitize bound |
| token rents (<40% of local comp — IAAO nominal price) | ⚠️ WARN | 1,455 rows (1.876%) | plus 2,206 rows above 4x comp (2.845%) | scanner already excludes beyond -60%; route as review_only in a cleaning pass |
| digit-shift repair candidates (x10 off local comp) | ⚠️ WARN | 269 rows (0.347%) | 903 at ~10x ratio, 269 with band-credible area | add rent digit-shift repair before trusting deep discounts |
| bulk contracts no_of_prop>1 (allocated-amount risk) | ⚠️ WARN | 2,855 rows (3.681%) | median PSF 1,186 vs 70 single (16.95x) — annual_amount is a contract TOTAL stamped per line | confirmed non-unit prices; excluded from all artifacts by the pull's no_of_prop<=1 filter (IAAO allocated-price rule) |
| same-day identical-amount groups ≥25% below comp | ⚠️ WARN | 1,041 rows in 235 groups (1.342%) | bulk_allocation mirror of the sales rule | route review_only in a cleaning pass if material |
| durations & annualization | ✅ OK | 0 non-positive; 94.7% ~12mo; 1,495 multi-year | multi-year annual*years vs contract_amount mismatch >20%: 109 (7.3% of multi-year) | annual_amount confirmed annualized |
| band/area impossibilities | ⚠️ WARN | 213 rows (0.275%) | Studio >150sqm or 4BR+ <40sqm | mislabeled sub-types; negligible unless material |
| sensitivity: cleaning impact on weekly zone medians | ✅ OK | max weekly median shift 0.69% | top-3 zones, medians with vs without token rents + D2 duplicates | medians are robust; cleaning protects the tails, not the center |

## 4 Gaps

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| month-level volumes (pull log) | ✅ OK | 32 chunks, median 65,078/month | months below 50% of median: none | none needed |
| week-level continuity (20 scanner districts, All band) | ⚠️ WARN | 4 missing weeks across districts | worst: Dubai Islands=4; Jumeirah Village Triangle (JVT)=0; Jumeirah Village Circle (JVC)=0; Arjan=0 | small districts legitimately skip weeks; investigate only clustered runs |
| overall weekly coverage | ✅ OK | 2024-01-01 → 2026-08-31 | 140 distinct weeks | matches RENTS_START → now |
| day-level continuity (recent contracts, all districts) | ✅ OK | 0 zero-contract days in 205 covered | window 2026-02-08 → 2026-08-31 | isolated public holidays are normal; clusters mean a feed outage |
| freshness | ✅ OK | includes future-dated lease starts | latest contract start 2026-08-31 | leases are registered in advance of their start date; the strictly-past rolling median is unaffected |

## 5 Integrity

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| rent_weekly_stats blob row_count | ✅ OK | 132,493 rows | blob metadata says 132,493 | none needed |
| rent_recent_contracts blob row_count | ✅ OK | 87,412 rows | blob metadata says 87,412 | none needed |
| rent_index blob row_count | ✅ OK | 55,472 rows | blob metadata says 55,472 | none needed |
| weekly_stats quantile ordering | ✅ OK | 0 violating rows | p10<=q1<=median<=q3<=p90 and n>=1 | none needed |
| rent_index key uniqueness | ✅ OK | 0 duplicate keys | (AREA_EN, rooms_band, week) | none needed |
| recomputed weekly stats vs published | ⚠️ WARN | 550 overlapping (district, week) cells | max |median diff| 0.7999999999999972; n mismatches 24 (max 11) — n gaps measure the .unique() dedupe in the contract artifact | investigate aggregation drift |

