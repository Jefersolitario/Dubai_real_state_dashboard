# Rent Data Quality Report — Ejari Feed & GCS Artifacts

Audited 2026-11-14 · raw sample: 121,788 rows (2024-06, 2026-07) · artifacts: weekly_stats 145,984, recent_contracts 244,053, rent_index 61,094 rows

Methodology per `reports/data_cleaning_research_report.md`: reason-coded
exclusions (IAAO), outliers judged against local zone×band comparables
(Land Registry price bands / AVM practice), explicit duplicate
definitions, and sensitivity tests against the scanner's weekly zone
medians. Verdicts: ✅ OK · ⚠️ WARN (monitor) · 🛑 ACTION (fix).

## Executive summary

**No blocking issues.** 11 monitored findings below.
- ⚠️ D1 exact full-row duplicates — 139 rows (0.114%)
- ⚠️ D2 key duplicates (contract_id, line_number) — 139 surplus rows (0.114%)
- ⚠️ D4 re-registration suspects (same attrs, different id) — 6,328 surplus rows (5.196%)
- ⚠️ pagination stability (date-ordered fetch) — 15,616 dupes, 15,605 skipped of 55,299 (2024-06)
- ⚠️ absurd PSF surviving sanitize bounds — 2,455 rows (2.921%)
- ⚠️ token rents (<40% of local comp — IAAO nominal price) — 2,142 rows (2.549%)
- ⚠️ digit-shift repair candidates (x10 off local comp) — 402 rows (0.478%)
- ⚠️ bulk contracts no_of_prop>1 (allocated-amount risk) — 3,579 rows (4.259%)
- ⚠️ same-day identical-amount groups ≥25% below comp — 1,361 rows in 278 groups (1.619%)
- ⚠️ band/area impossibilities — 230 rows (0.274%)
- ⚠️ week-level continuity (76 scanner districts, All band) — 259 missing weeks across districts

## 1 Duplicates

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| D1 exact full-row duplicates | ⚠️ WARN | 139 rows (0.114%) | all-column dupes 139; excluding load_timestamp 139 | residual boundary repeats — removed by the pull's (contract_id, line_number) dedupe |
| D2 key duplicates (contract_id, line_number) | ⚠️ WARN | 139 surplus rows (0.114%) | 139 duplicated keys, 0 with conflicting amount/area | removed by the pull's (contract_id, line_number) dedupe |
| D3 multi-line contracts (multi-property, legit) | ✅ OK | 2,103 contracts (2.14%) | lines==no_of_prop for 1,996/2,103 of them | document semantics; not duplicates |
| D4 re-registration suspects (same attrs, different id) | ⚠️ WARN | 6,328 surplus rows (5.196%) | 3,470 attribute groups with >1 contract_id | monitor; likely same-building identical units + genuine re-registrations |
| pagination stability (date-ordered fetch) | ⚠️ WARN | 15,616 dupes, 15,605 skipped of 55,299 (2024-06) | date-ordered paging vs the contract_id-ordered sample — the gateway defect that corrupted every pull before the contract_id fix | production already orders by contract_id; never revert to date ordering (this check is the regression sentinel) |

## 2 Types & missing

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| wire type of annual_amount | ✅ OK | 0 string values (0.000%) | wire types: {'float': 121788} | none needed |
| wire type of contract_amount | ✅ OK | 0 string values (0.000%) | wire types: {'float': 121788} | none needed |
| wire type of actual_area | ✅ OK | 0 string values (0.000%) | wire types: {'float': 116310, 'NoneType': 5478} | none needed |
| wire type of no_of_prop | ✅ OK | 0 string values (0.000%) | wire types: {'int': 121788} | none needed |
| wire type of line_number | ✅ OK | 0 string values (0.000%) | wire types: {'int': 121788} | none needed |
| unparsable numerics in annual_amount | ✅ OK | 0 rows | values that production casting silently nulls | none needed |
| unparsable numerics in actual_area | ✅ OK | 0 rows | values that production casting silently nulls | none needed |
| unparsable dates in contract_start_date | ✅ OK | 0 rows | post slice-10 ISO parse | none needed |
| unparsable dates in contract_end_date | ✅ OK | 0 rows | post slice-10 ISO parse | none needed |
| null/empty rates (key columns) | ✅ OK | project_name_en=75.1%; actual_area=4.5%; ejari_property_sub_type_en=0.0%; ejari_property_type_en=0.0% | empty strings: 0 total | project/master fields expected sparse (Ejari has no building key) |
| area_name_en case/whitespace variants | ✅ OK | 0 collapsing variants | 170 raw vs 170 normalized names | strip+upper already applied at slim time (parity with sales path) |
| rooms-band mapping coverage (flats) | ✅ OK | 599 unmapped (0.68%) | top unmapped: Room=316; Room in labor Camp=258; Duplex=14; Office=11 | acceptable (folds into the All band) |
| contract_reg_type_en vocabulary | ✅ OK | Renew=68,077; New=53,711 | labels containing 'new' without starting with it: ['Renew'] | revisit the matcher |

## 3 Outliers

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| corrupt contract years (outside sanitize window) | ✅ OK | 0 rows (0.000%) | years outside [2020,2027] (probe sample itself had a 2204 row) | already excluded by sanitization; monitor the rate |
| absurd PSF surviving sanitize bounds | ⚠️ WARN | 2,455 rows (2.921%) | annual PSF > 2,000 or < 10 AED/sqft/yr | candidates for a ratio-based sanitize bound |
| token rents (<40% of local comp — IAAO nominal price) | ⚠️ WARN | 2,142 rows (2.549%) | plus 2,622 rows above 4x comp (3.120%) | scanner already excludes beyond -60%; route as review_only in a cleaning pass |
| digit-shift repair candidates (x10 off local comp) | ⚠️ WARN | 402 rows (0.478%) | 1,078 at ~10x ratio, 402 with band-credible area | add rent digit-shift repair before trusting deep discounts |
| bulk contracts no_of_prop>1 (allocated-amount risk) | ⚠️ WARN | 3,579 rows (4.259%) | median PSF 1,680 vs 70 single (24.02x) — annual_amount is a contract TOTAL stamped per line | confirmed non-unit prices; excluded from all artifacts by the pull's no_of_prop<=1 filter (IAAO allocated-price rule) |
| same-day identical-amount groups ≥25% below comp | ⚠️ WARN | 1,361 rows in 278 groups (1.619%) | bulk_allocation mirror of the sales rule | route review_only in a cleaning pass if material |
| durations & annualization | ✅ OK | 0 non-positive; 94.3% ~12mo; 1,879 multi-year | multi-year annual*years vs contract_amount mismatch >20%: 111 (5.9% of multi-year) | annual_amount confirmed annualized |
| band/area impossibilities | ⚠️ WARN | 230 rows (0.274%) | Studio >150sqm or 4BR+ <40sqm | mislabeled sub-types; negligible unless material |
| sensitivity: cleaning impact on weekly zone medians | ✅ OK | max weekly median shift 1.24% | top-3 zones, medians with vs without token rents + D2 duplicates | medians are robust; cleaning protects the tails, not the center |

## 4 Gaps

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| month-level volumes (pull log) | ✅ OK | 35 chunks, median 66,448/month | months below 50% of median: none | none needed |
| week-level continuity (76 scanner districts, All band) | ⚠️ WARN | 259 missing weeks across districts | worst: Dubailand (Wadi Al Safa 4)=95; Ras Al Khor=78; Za'abeel 1=29; Jumeirah Bay Island (Jumeirah 2)=21 | small districts legitimately skip weeks; investigate only clustered runs |
| overall weekly coverage | ✅ OK | 2024-01-01 → 2026-11-30 | 153 distinct weeks | matches RENTS_START → now |
| day-level continuity (recent contracts, all districts) | ✅ OK | 0 zero-contract days in 200 covered | window 2026-05-15 → 2026-11-30 | isolated public holidays are normal; clusters mean a feed outage |
| freshness | ✅ OK | includes future-dated lease starts | latest contract start 2026-11-30 | leases are registered in advance of their start date; the strictly-past rolling median is unaffected |

## 5 Integrity

| Check | Verdict | Size | Evidence | Action |
|---|---|---|---|---|
| rent_weekly_stats blob row_count | ✅ OK | 145,984 rows | blob metadata says 145,984 | none needed |
| rent_recent_contracts blob row_count | ✅ OK | 244,053 rows | blob metadata says 244,053 | none needed |
| rent_index blob row_count | ✅ OK | 61,094 rows | blob metadata says 61,094 | none needed |
| weekly_stats quantile ordering | ✅ OK | 0 violating rows | p10<=q1<=median<=q3<=p90 and n>=1 | none needed |
| rent_index key uniqueness | ✅ OK | 0 duplicate keys | (AREA_EN, rooms_band, week) | none needed |
| recomputed weekly stats vs published | ✅ OK | 2,002 overlapping (district, week) cells | max |median diff| 0.0; n mismatches 0 (max 0) — n gaps measure the .unique() dedupe in the contract artifact | pipeline consistent |

