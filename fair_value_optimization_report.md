# Fair-Value Model Optimization Report

Generated 2026-07-04 16:02 · data: GCS snapshot · protocol: date-ordered TimeSeriesSplit(10 folds), headline metric = mean CV MedAPE (median |actual − fair value| / fair value on future folds).

Stop rule: < 0.2 pp MedAPE improvement for 2 consecutive iterations, or 10 iterations.

## Iterations

| # | Proposal | Detail | MedAPE (mean ± std) | MAE(log) | R² | Decision | Time |
|---|----------|--------|--------------------:|---------:|----:|----------|-----:|
| 0 | Baseline: area median PSF (no ML) | per-area trailing median | 15.47% ± 1.08% | 0.2119 | 0.469 | baseline | 0s |
| 1 | HGB core features | size, area, rooms, off-plan, tier, time trend | 9.47% ± 0.43% | 0.1355 | 0.762 | accepted (+6.00%) | 36s |
| 2 | + project categoricals | PROJECT_EN / MASTER_PROJECT_EN (top 200 + OTHER) | 7.94% ± 0.49% | 0.1185 | 0.809 | accepted (+1.53%) | 42s |
| 3 | + building categorical | BUILDING_NAME_EN (top 200 + OTHER) | 7.94% ± 0.48% | 0.1178 | 0.811 | rejected (+0.01%) | 43s |
| 4 | + amenity & deal features | nearest metro/mall/landmark, parking, buyer/seller counts | 7.69% ± 0.48% | 0.1142 | 0.823 | accepted (+0.25%) | 49s |
| 5 | + trailing area comps | 30-day area median PSF, strictly past-only | 7.68% ± 0.46% | 0.1141 | 0.825 | rejected (+0.02%) | 47s |
| 6 | + trailing project comps | 60-day project median PSF, strictly past-only | 5.91% ± 0.45% | 0.0879 | 0.891 | accepted (+1.78%) | 48s |
| 7 | hyperparams: slower learning | learning_rate 0.04, max_iter 800 | 5.85% ± 0.49% | 0.0874 | 0.892 | accepted (+0.06%) | 80s |
| 8 | hyperparams: deeper trees | max_leaf_nodes 127 | 5.86% ± 0.52% | 0.0878 | 0.891 | rejected (-0.02%) | 124s |

## Winning configuration

- **Model**: hgb — hyperparams: slower learning
- **MedAPE**: 5.85% ± 0.49%
- **R²**: 0.892
- **Feature config**: `{'project': True, 'building': False, 'amenity': True, 'comps_area': False, 'comps_project': True}`
- **Model params**: `{'learning_rate': 0.04, 'max_iter': 800, 'max_leaf_nodes': 63, 'early_stopping': True, 'validation_fraction': 0.1}`
- **Rows**: 268,997

## Feature importances (permutation, winning model)

| Feature | Importance | ± std |
|---------|-----------:|------:|
| project_comp_psf | 0.9476 | 0.0040 |
| log_sqft | 0.5086 | 0.0056 |
| rooms_ord | 0.2080 | 0.0042 |
| MASTER_PROJECT_EN | 0.0774 | 0.0014 |
| AREA_EN | 0.0600 | 0.0008 |
| PROJECT_EN | 0.0251 | 0.0004 |
| IS_OFFPLAN_EN | 0.0149 | 0.0002 |
| NEAREST_METRO_EN | 0.0058 | 0.0004 |
| NEAREST_LANDMARK_EN | 0.0028 | 0.0003 |
| parking_count | 0.0016 | 0.0003 |
| tier | 0.0007 | 0.0001 |
| NEAREST_MALL_EN | 0.0006 | 0.0001 |
| total_seller | 0.0001 | 0.0000 |
| total_buyer | 0.0001 | 0.0000 |
| days_since_start | 0.0000 | 0.0000 |

## Phase 2 data candidates (not yet integrated)

- Live listing asking prices (Bayut / Property Finder) — score offers, not just closed sales.
- Ejari rent contracts (Dubai Pulse `dld_rent_contracts`) — project rental yield feature and distress corroboration.
- Buildings/units metadata (floor, building age, developer) — strongest missing hedonic features.
- Official residential sale price index — drift monitoring.

## Independent audit addendum (2026-07-04)

A post-run audit (code-mechanics review with live Polars experiments + empirical
re-validation) confirmed the results:

- **Untouched holdout** (train < 2026-05-01, test on the final ~2 months the
  selection loop never optimized against): MedAPE **5.92%** trimmed / **6.00%**
  untrimmed, R² 0.82–0.85, vs 14.4% area-median baseline on the same period —
  the CV headline is not selection-inflated.
- **No leakage**: `project_comp_psf` is strictly past-only (verified: a
  project's first sales get null comps, never their own price; same-day sales
  never see each other). Shuffling the training target collapses the model to
  R² ≈ 0, as a clean model should.
- **Honesty notes**: the CV credited the project-comps feature +1.78pp; the
  untouched holdout shows +1.06pp (mild selection optimism from choosing the
  config on the same folds — quote the true error as ≈ 6%). The headline also
  blends ~5.9% MedAPE for the 96.8% of rows with recent project comps and
  ~12% for the 3.2% cold-start rows (a project's first sales in a 60-day
  window).
- Deepest scored "discounts" (spread ≈ −99%) are token/nominal-consideration
  transfers; they are annotated but correctly not labelled distressed because
  no residual-independent signal corroborates them.

## Campaign addendum (2026-07-04): 52 attempts, sequestered-holdout protocol

An extended campaign explored 8 new strictly-past feature groups plus
hyperparameters, objectives, ensembles, segmentation, and ablations.
**Anti-overfitting protocol**: every selection decision used only data before
2026-05-01 (10-fold date-ordered TimeSeriesSplit within that window); the
final two months were evaluated exactly once, as a ship/no-ship gate.

| Milestone | Selection CV MedAPE |
|---|---|
| Baseline (previous shipping config) | 5.69% |
| + repeat_sale (same unit's prior sale PSF) | 4.76% (−0.93pp, largest gain) |
| + te_hist (expanding project & building median PSF) | 4.65% |
| + rel_size (size vs project norm — interaction found on retest) | 4.48% |
| + absolute_error objective (train what we measure) | 4.26% |
| + leaves 127 / L2 1.0 | **4.20% ± 0.45%**, R² 0.923 |

44 of 52 attempts were rejected by the ≥0.05pp noise gate, including:
building categorical, comp window variants, liquidity counts, momentum,
comp dispersion, indexed repeat-sale, 3-seed bagging, off-plan/ready
segmentation, encoder capacity, and every ablation (confirming the kept
features all carry signal). The campaign stopped at diminishing returns —
continuing toward 100 attempts would only have inflated selection bias.

**Holdout gate (one-shot, untouched May–July 2026, n=17,953):**
campaign winner **4.32% / R² 0.874** vs prior production config
**5.91% / R² 0.833** → shipped. The near-match between selection CV (4.20%)
and holdout (4.32%) indicates minimal winner's curse.

**Winning model importances (out-of-sample permutation):** prior_unit_psf
0.48, building_hist_psf 0.31, MASTER_PROJECT_EN 0.08, log_sqft 0.05,
rooms_ord 0.05, project_comp_psf 0.04 — repeat-sales and building-level
history now dominate, exactly as the research report predicted.
