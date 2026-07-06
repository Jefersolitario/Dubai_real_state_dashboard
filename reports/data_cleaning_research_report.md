# Deep Research — Data-Cleaning Best Practices for Real-Estate Transaction Data

**Date:** 2026-07-06 · Commissioned to validate and improve `data_cleaning.py`
(classify / repair / route). Sources: IAAO standards, HM Land Registry (UK
HPI), S&P Case-Shiller methodology, the ZTRAX academic literature (Nolte et
al.), AVM vendor practice, and USPTO patents on market-based cleaning.

---

## 1. The industry consensus in one sentence

Every serious system — assessor offices, national price indices, AVM vendors,
academic hedonics — has converged on the same three-part doctrine:

1. **A sale is valid until documented otherwise**, and every exclusion gets a
   **labelled reason code** (never a silent drop).
2. **Non-market transfers are excluded from estimation but retained in the
   database** — the categories are remarkably consistent across jurisdictions.
3. **Cleaning rules are versioned, documented, and measured**, because
   cleaning choices demonstrably move results.

Our module's `dq_rule`/`dq_action` design is this doctrine; the research below
confirms it and sharpens several specifics.

## 2. IAAO Standard on Verification and Adjustment of Sales (2020)

The assessment industry's canonical text (used by US/Canadian assessors for
mass appraisal and ratio studies). Key content:

**Principles.** "All sales should be considered candidates for valid sales
unless sufficient information can be documented to show otherwise… If sales
are excluded without substantiation, the study may appear subjective. Reason
codes may be established for valid and invalid sales." Screening must be
"uniform and transparent… with guidance and documentation." No single set of
screening rules is universally applicable — rules must fit the jurisdiction's
data.

**Sales generally considered invalid** (exclude from modelling; verify before
any reinstatement): sales involving government agencies; charitable/religious
/educational institutions; **financial institutions as buyer or seller**
(foreclosure/REO); **sales between relatives or corporate affiliates**; sales
settling an estate; **forced sales resulting from judicial order**; sales of
doubtful title.

**Special conditions** (may be market sales; verify or exclude):

- **Partial interests** — "a sale involving a conveyance of less than the full
  interest should be excluded as a valid transaction"; even when all
  fractional owners sell the same day, the sum of prices "may not necessarily
  indicate the market value of the whole property."
- **Multi-parcel / bulk transfers** — "acquisitions or divestments by large
  corporations, pension funds, or REITs that involve multiple parcels
  typically should not be considered for analysis." Prices "verified to be an
  allocated price as part of a package or bulk transaction" are disqualified
  in every state validity-code system reviewed (Florida, Kansas, Indiana,
  Tennessee).
- **Sales of convenience** — title corrections, tenancy changes, and similar
  transfers "generally retransacted at only a nominal price" — the formal name
  for our token-transfer case.
- Trades, land contracts, auctions (with-reserve vs absolute matters).

**Exclude vs adjust.** Non-arm's-length categories are excluded outright;
otherwise-arm's-length sales with special terms (financing, personal property)
may be **adjusted** to market equivalence instead of discarded — supporting
the repair-over-delete philosophy for recoverable records.

## 3. HM Land Registry / UK House Price Index

The best public documentation of cleaning an **administrative registry feed**
(the closest analog to DLD open data):

- **Prevention at entry:** double keying of the price field (initial entry
  obscured) — because price typos are the dominant error class.
- **Daily exception reports:** prices outside **regional price bands**,
  missing postcodes, unknown property types. Bands are reviewed annually.
- **Model-based exclusion:** "if the modelled price is substantially different
  then the price is excluded from the final estimate" (~50 sales/month) — a
  production system using its own hedonic model as the plausibility
  instrument, exactly our comp-ratio approach.
- **Scale of errors:** 10–20 confirmed errors/month in ~70,000 transactions
  (~0.02%) after prevention — our 157 repaired + 2 quarantined in 303k
  (~0.05%) is the right order of magnitude for a feed *without* double keying.
- Corrections found late are shipped in the **next month's data**, never by
  rewriting history — supporting immutable raw + downstream cleaning.

## 4. S&P Case-Shiller (repeat-sales indices)

Excludes non-arm's-length pairs (family transfers, foreclosure transfers),
new construction, and **re-trades within 6 months** (flip/error guard). The
cautionary number: these filters discard **~42% of sales** in the 10-city
composite (Parcl 2022 analysis). Lesson: filters have a coverage cost, and a
distress product must be *more* conservative about deleting than an index —
which is why our review-routed rows stay visible in the UI.

## 5. Academic practice — the ZTRAX literature

Nolte et al. 2024 (*Land Economics*, "Data Practices for Studying the Impacts
of Environmental Amenities and Hazards with Nationwide Property Data"; PLACES
Lab): 14 research groups converged on shared cleaning code for Zillow's ZTRAX
(≈400M transaction records). Practices: regex screens on buyer/seller names
for public/institutional parties; intra-family transfer flags; document-type
filters; fair-market-value price filters; and the headline finding that
**cleaning choices meaningfully influence research findings**, so rules must
be published and sensitivity-tested. (Our CV A/B with cleaning on/off is that
sensitivity test, run before shipping.)

## 6. AVM vendor practice

The IAAO Standard on AVMs endorses outlier trimming as part of model
validation; vendor pipelines (ICE, ClearCapital, HouseCanary, ATTOM) describe
a standard three-stage flow — validate/clean → time-normalize → estimate —
with outlier filtering **relative to local comparables and trend**, not global
thresholds. USPTO patents 8,738,388 ("Market based data cleaning") and
7,765,125 formalize using market-derived expectations to detect and correct
erroneous records: the comp-as-instrument design is patented mainstream
practice, not an improvisation.

## 7. Scorecard: `data_cleaning.py` vs best practice

| Practice (source) | Our module | Verdict |
|---|---|---|
| Reason codes on every exclusion (IAAO) | `dq_rule` + `dq_action` on every row | ✅ |
| Valid-until-documented; no silent drops (IAAO) | full frame returned, callers route | ✅ |
| Partial interests excluded (IAAO 5.5.1.2) | `partial_transfer` rule (dormant: PROCEDURE_AREA==ACTUAL_AREA in feed) | ✅ future-proofed |
| Bulk/allocated prices disqualified (IAAO 5.5.2, state codes) | `bulk_allocation` — only groups ≥25% below comp; at-market launch batches stay | ✅ sharper than standard (launch exemption is Dubai-specific and correct) |
| Sales of convenience at nominal price (IAAO) | `suspected_token_transfer` (<40% of comp) | ✅ |
| Price bands / model-based plausibility (Land Registry) | comp ratio per project with area fallbacks | ✅ |
| Repair recoverable records (IAAO adjustment; patents) | digit-shift repair with comp + layout-area instrument, band-gated | ✅ |
| Relatives / corporate affiliates (IAAO 5.4.5) | not identifiable — DLD open feed has no party names/relationships | ⚠️ gap (feed limitation); price signature partially caught by token rule |
| Financial-institution / forced sales (IAAO 5.4.3–5.4.7) | PROCEDURE_EN has zero forced-sale vocabulary in this feed | ⚠️ gap (feed limitation; Emirates Auction matching is the planned fix) |
| Flip / rapid re-trade guard (Case-Shiller) | not implemented — repeat-sale features expose re-trades but nothing screens them | ➖ candidate, low priority (guards indices, not spread scoring) |
| Publish rules + sensitivity test (Nolte et al.) | `data_cleaning_report.md` + CV A/B before enabling | ✅ |
| Version rules; never rewrite raw data (Land Registry) | raw GCS snapshot immutable; cleaning runs at train/inference time | ✅ (per owner's directive) |

## 8. Recommendations

1. **Ship as designed.** The module matches or exceeds the documented practice
   of assessors, national indices, and AVM vendors; the two gaps (related-party
   identity, forced-sale labels) are feed limitations, not design flaws.
2. **Keep raw immutable, clean at use** (adopted): Land Registry's correction
   flow and the reproducibility argument both support it — rules can evolve
   and re-run over unchanged raw history, and every published number can be
   regenerated from raw + code version.
3. **Monitor rule counts monthly** like Land Registry's exception reports: a
   sudden jump in any `dq_rule` count signals a feed change, not a market
   change (alarm thresholds: repairs > 0.2%, review_only > 2%).
4. **When live listings arrive (Phase 3)**, add the Case-Shiller-style rapid
   re-trade screen to listing-to-sale matching, where flips genuinely distort.
5. **Emirates Auction matching** remains the path to true forced-sale labels
   (IAAO category 5.4.7), upgrading "distress" from proxy corroboration to
   ground truth.

## Sources

- [IAAO Standard on Verification and Adjustment of Sales (2020)](https://www.iaao.org/wp-content/uploads/Standard_on_Verification_Adjustment_of_Sales.pdf)
- [IAAO Standard on Automated Valuation Models](https://www.iaao.org/wp-content/uploads/Standard_on_Automated_Valuation_Models.pdf)
- [IAAO Standard on Ratio Studies / Mass Appraisal](https://www.iaao.org/wp-content/uploads/StandardOnMassAppraisal.pdf)
- [HM Land Registry data QA for the UK HPI](https://www.gov.uk/government/statistics/quality-assurance-of-administrative-data-in-the-uk-house-price-index/hm-land-registry-data)
- [Case-Shiller methodology overview](https://web.mnstate.edu/sahin/FINC_354_REF/CME_Repeat_Sale_Index_Methodology.pdf) and [coverage critique](https://www.inman.com/2024/03/11/case-shiller-is-a-go-to-source-for-prices-but-know-these-blind-spots/)
- [Nolte et al., Data Practices for Hedonic Analyses (PLACES Lab)](https://placeslab.org/hedonic-data-practices/) · [Land Economics 100(1)](https://le.uwpress.org/content/100/1/200)
- [Florida sale qualification codes](https://floridarevenue.com/property/Documents/salequalcodes_bef01012019.pdf) · [Tennessee sales verification manual](https://comptroller.tn.gov/content/dam/cot/pa/documents/manualsandreports/other-publications/SalesDataCollectionandVerificationManual.pdf)
- [US Patent 8,738,388 — Market based data cleaning](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8738388)
