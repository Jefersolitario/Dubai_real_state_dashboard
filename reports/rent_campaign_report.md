# Rent-Feature Campaign Report (campaign 3)

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

## Champion

- **shipped config (campaign 2 winner)** — MedAPE 3.97% ± 0.42%, p90 15.22%, flag_prop 4.64%
- Feature config: `{'project': True, 'building': False, 'amenity': True, 'comps_area': False, 'comps_project': True, 'comps_project_windows': False, 'comps_building': False, 'price_history': True, 'liquidity': False, 'momentum': False, 'rel_size': True, 'comp_dispersion': False, 'repeat_sale': True, 'repeat_sale_adj': False, 'project_meta': True, 'rent_yield': False, 'service_charge': False, 'unit_floor': True, 'rel_floor': True, 'rent_density': False, 'comps_rooms': True, 'rooms_dynamics': False, 'rent_level': False, 'rent_momentum': False, 'yield_matched': False, 'yield_spread': False, 'rent_pooling': False, 'rent_anchor': False, 'rent_divergence': False, 'rent_new_segment': False, 'rent_project': False, 'data_cleaning': True}`
- Model params: `{'learning_rate': 0.04, 'max_iter': 800, 'max_leaf_nodes': 127, 'early_stopping': True, 'validation_fraction': 0.1, 'loss': 'absolute_error', 'l2_regularization': 1.0}`
