# Rent-Feature Campaign Report (campaign 3)

Generated 2026-08-29 06:35 · date-ordered TimeSeriesSplit(10) on selection < 2026-08-01; accept iff MedAPE -0.05% AND p90 +0.10% / flag_prop +0.05% tail veto; one-shot holdout for the winner

Baseline (shipped config): MedAPE 3.97% ± 0.42%, p90 15.22%, flag_prop 4.64%

## Iterations

| # | Pass | Proposal | MedAPE (mean ± std) | p90 APE | flag_prop | R² | Decision | Time |
|---|------|----------|--------------------:|--------:|----------:|---:|----------|-----:|
| 1 | 1 | + new-contract rents | 3.96% ± 0.42% | 15.20% | 4.64% | 0.927 | rejected (+0.01% MedAPE) | 185s |
| 2 | 1 | + yield spread vs city band | 3.96% ± 0.41% | 15.26% | 4.64% | 0.927 | rejected (+0.01% MedAPE) | 187s |
| 3 | 1 | + matched gross yield | 3.96% ± 0.41% | 15.20% | 4.63% | 0.927 | rejected (+0.02% MedAPE) | 181s |
| 4 | 1 | + rent momentum | 3.99% ± 0.43% | 15.22% | 4.68% | 0.927 | rejected (-0.02% MedAPE) | 181s |
| 5 | 1 | + rent-price divergence | 3.99% ± 0.42% | 15.26% | 4.64% | 0.927 | rejected (-0.01% MedAPE) | 182s |
| 6 | 1 | + income-approach anchor | 3.97% ± 0.42% | 15.19% | 4.61% | 0.927 | rejected (+0.00% MedAPE) | 186s |
| 7 | 1 | + hierarchical rent pooling | 3.99% ± 0.43% | 15.20% | 4.62% | 0.927 | rejected (-0.02% MedAPE) | 179s |
| 8 | 1 | + rent level (control) | 3.97% ± 0.42% | 15.18% | 4.62% | 0.927 | rejected (+0.00% MedAPE) | 182s |
| 9 | 1 | + rent density (retry) | 3.96% ± 0.42% | 15.24% | 4.64% | 0.927 | rejected (+0.02% MedAPE) | 174s |
| 10 | 1 | pooling prior k=10 | 3.95% ± 0.41% | 15.24% | 4.65% | 0.927 | rejected (+0.02% MedAPE) | 182s |
| 11 | 1 | pooling prior k=50 | 3.96% ± 0.42% | 15.22% | 4.64% | 0.927 | rejected (+0.01% MedAPE) | 175s |

## Conclusion — rent features are a measured null at district granularity

Every research-derived candidate was rejected in a single full pass: the best
(matched gross yield, rent density, pooling k=10) gained +0.01–0.02 pp MedAPE,
an order of magnitude below the 0.05 pp acceptance gate and inside the measured
seed-noise floor; momentum and divergence variants were slightly negative. No
candidate improved the tail metrics either. The sequestered holdout was not
spent and `fair_value_config.json` is unchanged.

This outcome matches the literature the ladder was built from: rents are sticky,
so district-level rent aggregates largely re-encode price information the model
already holds at project granularity, and the effects rent data could add live
at unit/complex level (~40% of yield variance in comparable studies) — invisible
without a building key, which Ejari does not provide. The eight feature groups
remain in `fair_value_model.py`, off by default, ready to re-test if a
finer-grained rent linkage ever becomes available (e.g. a DLD contract feed
with building identifiers).

Evidence base and campaign design: the "Rent Signal Evidence" research report
and "Rent Campaign Plan" (claude.ai artifacts, session 2026-08-29); candidates
were derived from 25 claims across 21 sources (8 adversarially confirmed,
0 refuted). Notable measured local facts: matched gross yield median 5.2%;
new-contract rents price +10% above the mixed 180d Ejari index (the RERA
renewal drag), yet even that leading series added only +0.01 pp.

## Champion

- **shipped config (campaign 2 winner)** — MedAPE 3.97% ± 0.42%, p90 15.22%, flag_prop 4.64%
- Feature config: `{'project': True, 'building': False, 'amenity': True, 'comps_area': False, 'comps_project': True, 'comps_project_windows': False, 'comps_building': False, 'price_history': True, 'liquidity': False, 'momentum': False, 'rel_size': True, 'comp_dispersion': False, 'repeat_sale': True, 'repeat_sale_adj': False, 'project_meta': True, 'rent_yield': False, 'service_charge': False, 'unit_floor': True, 'rel_floor': True, 'rent_density': False, 'comps_rooms': True, 'rooms_dynamics': False, 'rent_level': False, 'rent_momentum': False, 'yield_matched': False, 'yield_spread': False, 'rent_pooling': False, 'rent_anchor': False, 'rent_divergence': False, 'rent_new_segment': False, 'data_cleaning': True}`
- Model params: `{'learning_rate': 0.04, 'max_iter': 800, 'max_leaf_nodes': 127, 'early_stopping': True, 'validation_fraction': 0.1, 'loss': 'absolute_error', 'l2_regularization': 1.0}`
