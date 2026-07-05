# Building Quality, Comfort & Reviews — Research Report (Phase 2.5)

**Date:** 2026-07-05 · **Status:** research complete; recommendations ranked by feasibility
**Context:** the shipped fair-value model sits at 4.20% CV / 4.32% holdout MedAPE. Campaign 2
found that project metadata and service charges add nothing on top of the price-history
features. The user's hypothesis — *building quality as experienced by residents* (reviews),
and *climate comfort* (AC quality, humidity, direct sun) — targets exactly the residual the
model cannot see today: two identical-on-paper towers whose prices differ because one is
better built, better cooled, and better run.

---

## 1. Does the hypothesis hold in the literature? Yes.

- A 2025 study in *Humanities & Social Sciences Communications* (Nature),
  ["Reputation matters: residents' sentiment and housing price"](https://www.nature.com/articles/s41599-025-05758-z),
  found residents' sentiment ranked **~6th of all variables in XGBoost** (10th in Random
  Forest) for housing-price assessment — ahead of many structural attributes.
- Aspect-based sentiment work splits review text into themes that map 1:1 onto the user's
  intuition: **build quality, maintenance, safety, connectivity, amenities, pricing**
  ([overview](https://www.researchgate.net/publication/351246962_Theoretical_Overview_of_Sentiment_Analysis_in_the_Real_Estate_Market);
  [sentiment-index study](https://www.researchgate.net/publication/394558585_Development_of_an_Online_Real_Estate_Sentiment_Index_and_Analysis_of_Apartment_Price_Responses)).
- Climate/comfort is priced: a spatial-hedonic study of urban microclimate (Austin, TX)
  measured a **+2.2% mean sale-price effect** from thermal comfort
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0143622824002881));
  floor level contributes up to **9.2%** and water/green views **7–13%** in dense Asian
  markets ([Hangzhou hedonic study](https://www.sciencedirect.com/science/article/abs/pii/S0197397518310324));
  orientation carries a measurable premium
  ([south-facing premium, Shanghai](https://www.researchgate.net/publication/327820538_The_value_of_a_south-facing_orientation_A_hedonic_pricing_analysis_of_the_Shanghai_housing_market)).
- Dubai-specific comfort physics: **west-facing** rooms take the worst radiant load
  (glass-side temperatures 5–8°C above thermostat in summer afternoons); window heat gain
  is ~40% of home heat load ([Stråla](https://strala.ae/blogs/news/heat-gain-through-windows-in-dubai-how-curtains-cut-cooling-costs));
  cooling regime (district cooling vs chiller-free vs own DEWA-metered AC) shifts the
  tenant's true occupancy cost by thousands of AED/year
  ([Property Finder](https://www.propertyfinder.ae/blog/chiller-free-vs-district-cooling-dubai/)).

## 2. New data sources, ranked by joinability to our pipeline

### Tier 1 — on the DDA gateway TODAY (verified with our credentials, 2026-07-05)

| Dataset | What it adds | Probe result |
|---|---|---|
| `dld_units-open-api` | **`floor` per unit**, `unit_balcony_area`, `unit_number`, `parking_allocation_type`, unit mix per building/project | ✅ accessible; ~keys: `project_id`, `building_number`, `property_id` |
| `dld_residential_sale_index-open-api` | Official monthly/quarterly **flat price index** — cleaner market-trend feature than `days_since_start` | ✅ accessible |
| `dld_rental_index-open-api` | Official area × property-type × room **rent index** — cross-check/backstop for our Ejari-derived grid | ✅ accessible |
| `dld_smart_rental_index` / building classification | — | ❌ 404 (not published on the gateway) |

**The units caveat:** transactions still carry no unit key. But units join to projects
(`project_id` ↔ `project_number` via the projects table), and within a project we can match
`ROOMS_EN` + exact `ACTUAL_AREA` (both sides are decimal-precise sqm). Where a layout's area
is **unique in the building**, that pins the exact unit → true floor. Where layouts stack
(same area on many floors — the common case), the match still yields honest *distribution*
features: floor range of the layout, balcony area, parking type. Expectation management:
this is an experiment, not a guaranteed win — the earlier finding that per-transaction floor
is infeasible stands for the general case.

### Tier 2 — building reviews & ratings (the user's core idea)

1. **DLD's official building classification (Smart Rental Index 2025)**: every residential
   building rated **1–5 stars on 60+ criteria** (build condition, finish quality,
   maintenance, facility management, amenities, location) —
   [The National](https://www.thenationalnews.com/business/property/2025/01/02/dubais-new-rental-index-to-be-based-on-building-rating-system/),
   [DLD](https://dubailand.gov.ae/en/news-media/the-smart-rent-index-mitigates-inflation-in-dubai-and-enhances-market-transparency/).
   This is *exactly* the quality label we want, government-issued. **Not yet open data**
   (gateway 404; ratings are visible per building in DLD's rental calculator UI). Action:
   watch Dubai Pulse for release; a manual lookup of our top-200 buildings by transaction
   count is a legitimate stopgap.
2. **Property Finder Building Reviews** ([propertyfinder.ae/en/building-reviews](https://www.propertyfinder.ae/en/building-reviews)):
   UAE-specific per-building ratings/reviews from residents. No public API; scraping is
   against most listing portals' ToS — treat as *manual-enrichment* source for top buildings,
   or approach Property Finder for a data partnership.
3. **Google Places API** ([Place Details (New)](https://developers.google.com/maps/documentation/places/web-service/place-details)):
   the legitimate, ToS-clean route. Residential towers in Dubai are Places with **star
   rating + review count + review snippets** (and now
   [AI review summaries](https://developers.google.com/maps/documentation/places/web-service/review-summaries)).
   Plan: geocode our top ~500 buildings (by 24-month transaction count ≈ covers most volume),
   one Place Details call each (≈ $0.02/call ⇒ ~$10–20 one-time), store
   `google_rating`, `google_reviews_count`, and run **aspect-based sentiment** on snippets
   for the comfort themes the user named: *AC/cooling, humidity/mould, maintenance, noise,
   management*. Ratings are a **current** snapshot → point-in-time caveat (same class as
   `percent_completed`): document it, use as a static building trait, never as a
   time-varying signal.

### Tier 3 — comfort/climate proxies (free, engineering effort only)

- **Orientation / direct sun**: [OSM building footprints for the UAE](https://data.humdata.org/dataset/hotosm_are_buildings)
  (~594k footprints, ~49% coverage) → per-building **facade azimuth** (share of glazing
  facing W/SW), footprint elongation, and shadowing by taller neighbours (3D massing exists
  for Dubai). Join by geocoded building name. Literature effect: orientation premiums are
  real but small (1–3%); worth testing only after reviews.
- **Cooling regime**: no open bulk list of district-cooled buildings, but provider service
  areas are published — Empower (~1,400 buildings: JBR, JLT, Business Bay, DIFC, Palm,
  Silicon Oasis…), Emicool (Motor City, Sports City, DIP, Mirdif…)
  ([MyBayut](https://www.bayut.com/mybayut/district-cooling-areas-dubai/),
  [Emicool projects](https://www.emicool.com/en/projects)). An **area-level cooling-provider
  flag** is a 1-hour manual mapping. Unit-level "chiller-free" only exists in listings data
  (Phase 2 live-listings work).
- **Service charges** (already in GCS): our Campaign 2 test showed no CV gain, but it remains
  the best *published* running-cost proxy; keep for the app's context display.

## 3. What models are used for these problems

| Problem | State of practice |
|---|---|
| Tabular hedonic valuation | **Gradient-boosted trees** (XGBoost/LightGBM/HGB — what we use) remain the benchmark; Kaggle house-price winners are GBM ensembles + target encoding. Our HGB is the right tool. |
| Review text → features | **Aspect-based sentiment** with BERT-family models; simple keyword+VADER scoring is the robust baseline for small corpora. Sentiment features rank top-10 in RF/XGBoost price models ([Nature study](https://www.nature.com/articles/s41599-025-05758-z)). |
| Sentiment indices / demand forecasting | BERT-BiLSTM + ADL-MIDAS hybrids ([Scientific Reports](https://www.nature.com/articles/s41598-025-16153-8)) — macro, not per-deal; not our use case. |
| Images (street view / satellite) | Multi-source CNN fusion pipelines ([PLOS One 2025](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0321951)); Perceiver-IO fusion for building inspection ([arXiv](https://arxiv.org/pdf/2605.26381)); big lift, marginal gain for us now. |
| Spatial structure | Spatial hedonic / GWR, and **transformer-GNNs** over property graphs ([survey](https://arxiv.org/html/2503.22119v1)); our project/building trailing-comp features already capture most spatial signal at far lower complexity. |
| Listing text | LLM extraction of structured features from descriptions — relevant when Phase 2 live-listings data lands. |

**Conclusion:** stay on HGB; the leverage is in *new inputs* (reviews, ratings, floor,
official indices), not a new model class.

## 4. Recommended execution order

1. **Units-table experiment** (gateway, free): pull `dld_units`, build per-building unit-mix
   + floor-distribution features; exact-area floor match where unique. *Risk: medium; cost: low.*
2. **Official sale index as market-trend feature** (gateway, free): monthly flat index,
   lagged one month for point-in-time safety. *Risk: low; cost: trivial.*
3. **Google Places ratings for top-500 buildings** (~$20, ToS-clean): rating + review count
   + aspect sentiment (AC/humidity/maintenance/noise). The single most direct test of the
   user's hypothesis. *Risk: name-matching effort; cost: low.*
4. **Area-level cooling-provider flag** (manual, 1h). *Risk: low; cost: trivial.*
5. **OSM orientation features** (free, one-time geospatial job) — only if 1–3 show gains.
6. **Watch list:** DLD building classification open release (the ideal label);
   Property Finder reviews partnership; live listings (chiller-free, floor, view) in Phase 2.

All of these are **static building traits** — the anti-leakage rule is disclosure (ratings
reflect *today's* reputation) rather than as-of joins, plus the usual selection-window CV
with the sequestered holdout before anything ships.
