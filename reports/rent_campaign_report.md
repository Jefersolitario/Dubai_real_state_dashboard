# Unit-Signal Campaign Report (campaigns 3 and 4)

Curated cumulative record: campaign 3 (rent features, two phases) and
campaign 4 (renovation permits). The runner writes only its latest run to
`reports/campaign_last_run.md`; this file is maintained by hand so a
filtered `--rungs` run cannot erase earlier results.


Generated 2026-08-29 17:38 · date-ordered TimeSeriesSplit(10) on selection < 2026-08-01; accept iff MedAPE -0.05% AND p90 +0.10% / flag_prop +0.05% tail veto; one-shot holdout for the winner

Baseline (shipped config): MedAPE 3.97% ± 0.42%, p90 15.22%, flag_prop 4.64%

## Iterations

| # | Pass | Proposal | MedAPE (mean ± std) | p90 APE | flag_prop | R² | Decision | Time |
|---|------|----------|--------------------:|--------:|----------:|---:|----------|-----:|
| 1 | 1 | + project-linked rents | 3.95% ± 0.43% | 15.20% | 4.62% | 0.927 | rejected (+0.02% MedAPE) | 292s |
| 2 | 1 | + new-contract rents | 3.96% ± 0.42% | 15.23% | 4.66% | 0.927 | rejected (+0.01% MedAPE) | 297s |
| 3 | 1 | + yield spread vs city band | 3.97% ± 0.41% | 15.23% | 4.61% | 0.927 | rejected (+0.00% MedAPE) | 297s |
| 4 | 1 | + matched gross yield | 3.97% ± 0.43% | 15.22% | 4.64% | 0.927 | rejected (+0.00% MedAPE) | 289s |
| 5 | 1 | + rent momentum | 3.97% ± 0.42% | 15.20% | 4.63% | 0.927 | rejected (+0.01% MedAPE) | 285s |
| 6 | 1 | + rent-price divergence | 3.98% ± 0.42% | 15.24% | 4.64% | 0.927 | rejected (-0.00% MedAPE) | 276s |
| 7 | 1 | + income-approach anchor | 3.96% ± 0.44% | 15.23% | 4.64% | 0.927 | rejected (+0.01% MedAPE) | 286s |
| 8 | 1 | + hierarchical rent pooling | 3.97% ± 0.41% | 15.22% | 4.62% | 0.927 | rejected (+0.00% MedAPE) | 282s |
| 9 | 1 | + rent level (control) | 3.95% ± 0.43% | 15.21% | 4.62% | 0.927 | rejected (+0.02% MedAPE) | 293s |
| 10 | 1 | + rent density (retry) | 3.96% ± 0.42% | 15.22% | 4.60% | 0.927 | rejected (+0.02% MedAPE) | 287s |
| 11 | 1 | pooling prior k=10 | 3.97% ± 0.42% | 15.26% | 4.67% | 0.927 | rejected (+0.00% MedAPE) | 289s |
| 12 | 1 | pooling prior k=50 | 3.98% ± 0.42% | 15.22% | 4.66% | 0.927 | rejected (-0.00% MedAPE) | 285s |

## Conclusion — rent features are a measured null at BOTH granularities

Two phases, one verdict. Phase 1 (district × rooms features from the deep-research
ladder) and phase 2 (project-linked rents) were both rejected by the same
pre-registered gate; the sequestered holdout was never spent and
`fair_value_config.json` is unchanged.

Phase 2 first solved the linkage problem the research called binding: the raw
Ejari feed carries `project_number` (~27% of contracts, registry-validated) and
a layout fingerprint — district × exact area @2dp sqm × rooms band against the
units registry — resolves more (62% unique / 97% accurate on 315k labeled
sales; the two routes agree 98.3% on 249k doubly-linked live contracts).
The 2024-01+ pull linked 420,318 of 1,372,421 usable contracts (30.6%) and
published `rent_project_index.parquet` (110,609 project-weeks).

Even so, `+ project-linked rents` (trailing project rent PSF, contract count,
same-stock project gross yield) gained only +0.02pp MedAPE — the best rung of
the campaign, with small tail improvements, but 2.5x below the 0.05pp gate.
The economics explain it: sales concentrate in projects whose price history is
already dense, which is exactly where a rent comp is most redundant; and the
error the model has left (~4% MedAPE) sits at unit level (floor, view,
condition), which no project aggregate can see. All rent groups — district and
project — remain in `fair_value_model.py`, off by default, with the linkage
pipeline live in the rents pull for the Rent Scanner and any future use.

Evidence base: the "Rent Signal Evidence" research report (25 claims,
8 adversarially confirmed, 0 refuted). Measured local facts: matched gross
yield median 5.2%; new-contract rents +10% above the mixed 180d index;
district rent grid 142 districts; project index covers 1,544 projects
(weeks 2024-01-08 through 2026-08-31).

## Campaign 4 — renovation permits (2026-08-30)

Same baseline, same gate. Dubai Municipality alteration permits joined to
projects via `parcel_id` -> `dld_buildings` -> `project_number`.

| # | Proposal | MedAPE (mean ± std) | p90 APE | flag_prop | Decision | Time |
|---|----------|--------------------:|--------:|----------:|----------|-----:|
| 1 | + renovation permits (5y) | 3.95% ± 0.39% | 15.21% | 4.65% | rejected (+0.02%) | 289s |
| 2 | renovation permit window 2y | 3.96% ± 0.42% | 15.24% | 4.65% | rejected (+0.02%) | 286s |

**Verdict: rejected, and the data explains why.** The permits feed is real and
large — 347,890 delivered adjustment/addition permits since 2010 — but it is
overwhelmingly a *villa* record: by building type the linked events are Floor
Area Ratio (14,968), Investment Villa (6,573), Private Villa (3,667) and only
3,507 Multi Storey. Just 41,736 of 347,890 permits (12%) matched a parcel in
the DLD project registry at all, yielding 29,116 events on 833 projects, and
only **12.4% of apartment sales carry any permit in the trailing five years**.
A signal present on one row in eight, on the wrong building type, cannot move
a median. Both window lengths landed at +0.02pp, the same place every other
unit-signal candidate has landed.

The feature group stays in `fair_value_model.py`, off by default. The permits
artifact is NOT wasted: it powers the **Building works** column on the Fair
Value tab (permit count and recency per project), which is buyer context
rather than a model input.

## Standing conclusion across campaigns 3 and 4

Three independent unit-level signals — district rents, project-linked rents,
renovation permits — were each derived from published evidence, implemented
strictly past-only, and measured under a pre-registered gate. All were
rejected, and every single candidate landed in the +0.00 to +0.02pp band. That
consistency is itself the finding: the shipped model's remaining ~4% error is
not a missing-aggregate problem. It lives at the individual unit — the view,
the floor, the condition of the kitchen, the quality of the renovation — which
DLD's open data does not describe and which a buyer establishes by viewing the
flat. Further feature campaigns against public aggregates are not a promising
use of effort; a genuine step change would need unit-level data the open feeds
do not publish.

## Champion

- **shipped config (campaign 2 winner)** — MedAPE 3.97% ± 0.42%, p90 15.22%, flag_prop 4.64%
- Feature config: `{'project': True, 'building': False, 'amenity': True, 'comps_area': False, 'comps_project': True, 'comps_project_windows': False, 'comps_building': False, 'price_history': True, 'liquidity': False, 'momentum': False, 'rel_size': True, 'comp_dispersion': False, 'repeat_sale': True, 'repeat_sale_adj': False, 'project_meta': True, 'rent_yield': False, 'service_charge': False, 'unit_floor': True, 'rel_floor': True, 'rent_density': False, 'comps_rooms': True, 'rooms_dynamics': False, 'rent_level': False, 'rent_momentum': False, 'yield_matched': False, 'yield_spread': False, 'rent_pooling': False, 'rent_anchor': False, 'rent_divergence': False, 'rent_new_segment': False, 'rent_project': False, 'data_cleaning': True}`
- Model params: `{'learning_rate': 0.04, 'max_iter': 800, 'max_leaf_nodes': 127, 'early_stopping': True, 'validation_fraction': 0.1, 'loss': 'absolute_error', 'l2_regularization': 1.0}`
