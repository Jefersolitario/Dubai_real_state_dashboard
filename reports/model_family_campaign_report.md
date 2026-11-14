# Model-Family Campaign Report (campaign 5)

Run 2026-11-14. Question: can the shipped fair-value model be improved by
retraining more often, by another learner, by stacking, or by zero-shot
time-series foundation models (TimesFM 3, Chronos)? Every number below comes
from `python -m model.optimize_model_family` (commands at the end).

Reviewed the same day in `reports/model_tuning_review_report.md`, which asked
how much these verdicts depend on settings. Retraining and the foundation-model
verdict stand, with intervals; the XGBoost figure below does not (it was its
settings), and is annotated where it appears.

## Verdict

| Lever | Result on the 69 test days | Decision |
|---|---|---|
| **Retrain at least monthly** (same design) | median error 4.37% (model as live, frozen 10 Aug) → **4.13%** | Do it: the largest gain found, no code change |
| Average of the shipped model and its anchored-target twin | 4.27% → 4.18% when both are trained once; 4.13% → **4.06%** when both are retrained weekly | Real and confirmed, below the complexity bar: not shipped, available on request |
| The same plus LightGBM | 4.27% → 4.16% | As above; LightGBM adds 0.02 pp for a new app dependency |
| LightGBM, XGBoost, CatBoost on their own | no gain | Rejected. XGBoost's −0.15 pp was its settings: set like the shipped model it equals it (review, section 6) |
| TimesFM 3 / Chronos-2 / Chronos-Bolt forecasts as features | no gain | Rejected |

The model's design stays as shipped. What it needs is to be retrained: the
bundle the app serves was trained on 10 August and is 96 days old.

One caution before retraining routinely (section 1): the app scores every sale
with the newest bundle, and a model refitted on everything has already seen
the sales it then scores, which shrinks their discounts. Retraining pays off
on sales that arrive after it.

## Protocol

- **Data**: the stored snapshot with one row per transaction and tidy names —
  325,234 scoreable sales, 2024-08-12 to 2026-11-12.
- **Selection**: sales through 2026-09-03 only (303,183 after the training
  price trim), the repo's date-ordered `TimeSeriesSplit(10)`. Each fold trains
  on the past and validates on the following two to three months; every
  feature is strictly past-only. Out-of-fold predictions are kept, so blends
  are judged on rows that neither their members nor their weights have seen.
- **Gate** (fixed before any run): campaign 2's — median error down by at
  least 0.05 pp with no tail worsening (worst-decile error +0.10 pp, false-flag
  share +0.05 pp). A blend must also beat its best member by 0.10 pp.
- **Test**: 2026-09-04 to 2026-11-12, 18,989 sales. Read for the retraining
  walk-forward (a measurement of the process, not a model choice) and for the
  baseline and the two blends cross-validation had already singled out: once
  with every model trained on sales to 3 Sep, and once more with weekly
  retraining for the two-model blend. No candidate was chosen from it.
- Baseline under this harness: **3.98% ± 0.44%**, worst decile 15.21%,
  false-flag share 4.56% — the same as campaign 4's baseline (3.97%).

## 1. Retraining cadence

The same design refitted at weekly cut-offs; a cadence uses the newest model
it allows. Features are the live past-only ones in every case, so only the age
of the fitted model varies.

| Model refresh | Median error | Worst decile | False-flag share |
|---|---:|---:|---:|
| Frozen 10 Aug (the live bundle) | 4.37% | 16.05% | 5.06% |
| Trained once, on sales to 3 Sep | 4.27% | 15.71% | 5.25% |
| Retrained every 28 days | 4.13% | 15.69% | 5.20% |
| Retrained every 14 days | 4.13% | 15.56% | 5.08% |
| Retrained every 7 days | 4.13% | 15.54% | 4.99% |

Monthly already captures the gain in the median; weekly keeps trimming the
tails. A fit takes about 40 seconds on four cores.

The selection data shows the same thing without touching the test window.
Out-of-fold error by the age of the model that made the prediction (folds
3-10 pooled):

| Days since the model's training data ended | Median error |
|---|---:|
| 0-13 | 3.52% |
| 14-27 | 3.74% |
| 28-41 | 3.85% |
| 42-55 | 3.78% |
| 56-69 | 4.12% |
| 70-83 (two folds) | 4.21% |

### What a retrain does to the flags

The gain above is for sales the model has not seen. The app, though, rescores
its whole window with the newest bundle, and `train_fair_value` refits on
every sale. The same sales scored three ways (default −15% threshold):

| Scored by | Last month, 8,656 sales: median abs. spread | below −15% | distressed | All 69 days: below −15% | distressed |
|---|---:|---:|---:|---:|---:|
| Frozen 10 Aug model (out-of-sample, stale) | 4.45% | 425 | 19 | 960 | 57 |
| The weekly model in force when each sale happened (out-of-sample) | 4.23% | 415 | 22 | 947 | 62 |
| One model refitted today on everything (in-sample) | 3.40% | 368 | 14 | 814 | 45 |

Refitting on everything removes about one flag in eight, not because the
model got better at pricing those flats but because it has seen their prices.
With weekly retraining only the newest week is out-of-sample at any time. The
walk-forward backtest that validated the flags (`distress_validation_report.md`)
used out-of-sample scores throughout, so the clean way to retrain often is to
keep each sale's out-of-sample score — publish the weekly walk-forward
predictions for the recent months alongside the bundle and let the tab prefer
them. That is not built.

## 2. Other learners

Ten folds, same rows and features for every learner.

| Learner | Median error | Gain vs shipped | Worst decile | False-flag share | Decision |
|---|---:|---:|---:|---:|---|
| HistGradientBoosting (shipped) | 3.981% | — | 15.21% | 4.56% | baseline |
| Same, anchored target | 3.961% | +0.020 | 15.29% | 4.70% | vetoed: more false flags |
| LightGBM | 3.961% | +0.020 | 15.15% | 4.52% | rejected: below the gate |
| Same, up to 2,400 iterations | 3.975% | +0.006 | 15.16% | 4.53% | rejected |
| XGBoost, at settings copied from the shipped model | 4.135% | −0.154 | 15.56% | 4.55% | rejected — but see below |
| XGBoost, tuned in the review | 3.983% | −0.002 | 15.13% | 4.49% | level with shipped: nothing to gain by switching |
| CatBoost | 4.253% | −0.272 | 15.80% | 4.40% | rejected |

LightGBM and XGBoost were allowed 3,000 trees and CatBoost 1,500 (its fits
already take seven minutes each); the later folds reached those budgets, so
the figures are for these budgets, not for an exhaustive tuning.

The review found what that sentence was hiding. Each challenger ran once.
XGBoost had been left at one sale per leaf where the other learners require
20, was given far too many trees (its out-of-time error is best near 200 and
its early stopping, which watches a random tenth of the training sales, never
fires), and loses 0.08 pp to its own handling of 200-level categories. With
those three fixed it scores 3.983%. LightGBM was already at a good setting and
CatBoost was stopped before it converged and has not been re-run. The three
gradient-boosting libraries are interchangeable here once set alike.

The anchored target predicts the deviation from the sale's most specific
trailing comparable (unit-type comp, then project, building, district) instead
of the price level, because trees cannot extrapolate a level they have not
seen. On its own it is a wash; as a second opinion it is what makes the blend
work.

## 3. Blends

Out-of-fold rows of folds 3-10 (220,496 sales); the shipped model scores
3.796% on these.

| Blend (plain average, in log space) | Median error | Gain vs shipped | Gain vs best member | Worst decile | False-flag share |
|---|---:|---:|---:|---:|---:|
| shipped + LightGBM + anchored | 3.698% | +0.098 | +0.081 | 14.62% | 4.31% |
| LightGBM + anchored | 3.707% | +0.089 | +0.072 | 14.62% | 4.33% |
| shipped + anchored | 3.733% | +0.062 | +0.046 | 14.70% | 4.37% |
| shipped + LightGBM | 3.736% | +0.060 | +0.045 | 14.72% | 4.37% |
| shipped (alone) | 3.796% | — | — | 14.84% | 4.43% |

- Fitted weights (non-negative, summing to one, fitted on earlier folds only)
  add nothing over the plain average: 3.697% against 3.698%.
- A fourth member does not help: with XGBoost the plain average drops to
  +0.077, with CatBoost to +0.083, and fitted weights only bring either back
  to where the three-model blend already was.
- No blend clears the bar of +0.10 pp over its best member, and none comes
  close to the +0.3 pp false-flag improvement campaign 2 asked of an ensemble.

One pass over the test window, every model trained once on sales to 3 Sep:

| Model | Median error (95% CI) | Worst decile | False-flag share | Off-plan | Resale |
|---|---:|---:|---:|---:|---:|
| shipped | 4.27% (4.17-4.35) | 15.71% | 5.25% | 3.54% | 6.24% |
| shipped + anchored | 4.18% (4.08-4.27) | 15.49% | 5.13% | 3.51% | 5.91% |
| shipped + LightGBM + anchored | 4.16% (4.06-4.24) | 15.45% | 5.10% | 3.48% | 5.84% |

Paired gain over the shipped model: +0.086 pp (95% CI +0.040 to +0.135) for
the two-model average, +0.107 pp (+0.061 to +0.155) for the three-model one.

The two levers add up. Retrained weekly over the same window:

| Model | Median error (95% CI) | Worst decile | False-flag share |
|---|---:|---:|---:|
| shipped, retrained weekly | 4.13% (4.05-4.22) | 15.54% | 4.99% |
| shipped + anchored, both retrained weekly | 4.06% (3.98-4.16) | 15.47% | 4.99% |

The gain is real and sits mostly in resale flats. It is also a quarter of what
retraining buys, and it costs a second model in the bundle, a new bundle
format and (for the third member) a LightGBM dependency in the app.

## 4. Time-series foundation models

These models forecast a series; the fair-value model prices one sale. The only
way in is as a feature: weekly median price per segment, and for every week a
forecast of that week's level made from earlier weeks only, joined to the
week's sales together with its gap to the trailing comp and the forecast
interval width.

Forecast accuracy first (absolute error in log points against the realised
weekly median, selection period; lower is better):

| Forecaster | District × unit type (181 series) mean / median | Project (1,228 series) mean / median | Time on 4 CPU cores |
|---|---:|---:|---:|
| Last observed week | 0.1146 / 0.0636 | — | instant |
| Mean of the last 4 observed weeks | 0.0998 / 0.0599 | — | instant |
| Exponentially weighted mean (0.3) | 0.0973 / 0.0595 | 0.0700 / 0.0480 | instant |
| Chronos-Bolt small (48M) | 0.1002 / 0.0600 | 0.0753 / 0.0506 | 30 s / 134 s |
| Chronos-Bolt base (205M) | 0.1001 / 0.0601 | — | 83 s |
| Chronos-2 (119M) | 0.0985 / 0.0578 | 0.0736 / 0.0492 | 99 s / 542 s |
| Chronos-2, a district's unit types jointly | 0.0976 / 0.0575 | — | 90 s |
| TimesFM 3 (331M) | 0.0996 / 0.0579 | not run: about an hour on CPU | 620 s |

Chronos-2 and TimesFM 3 are the best on the median by about 3% relative. On
the mean an exponentially weighted average is as good or better, and at
project level it beats them on both.

As features in the fair-value model (ten folds, shipped model 3.981%):

| Added feature | Median error | Worst decile | False-flag share |
|---|---:|---:|---:|
| District × type, weighted mean (control) | 3.996% | 15.25% | 4.55% |
| District × type, Chronos-2 | 3.996% | 15.28% | 4.60% |
| District × type, Chronos-2 joint | 3.981% | 15.19% | 4.53% |
| District × type, TimesFM 3 | 3.999% | 15.26% | 4.57% |
| Project, weighted mean (control) | 3.986% | 15.22% | 4.57% |
| Project, Chronos-2 | 3.979% | 15.18% | 4.53% |

Nothing moves. The model already holds trailing comps computed to the day
before each sale, and its remaining error sits at unit level:

| Out-of-fold segment (folds 3-10) | Share of sales | Median error |
|---|---:|---:|
| Off-plan, same layout sold before | 63.9% | 2.95% |
| Off-plan, first sale of the layout | 12.3% | 5.46% |
| Resale, same layout sold before | 15.8% | 5.67% |
| Resale, no prior sale of the layout | 7.9% | 7.35% |

A resale flat whose layout sold within the last 90 days still carries 5.2%
error. No forecast of a district's or project's price level can see the floor,
view and condition that separate two such flats.

Licences, should any of these be revisited: Chronos is Apache-2.0; the TimesFM
3 weights are under Google's non-commercial licence.

## Not run

| Candidate | Why |
|---|---|
| TimesFM 2.5 | Superseded by TimesFM 3 |
| TiRex | Ships GPU kernels; same role as the forecasters above, which added nothing |
| Toto, Moirai 2, FlowState, Sundial | GPU-oriented, rank below Chronos-2 on fev-bench; Moirai is non-commercial |
| TabPFN-TS | 109 s per 100 series on a GPU in fev-bench |
| TabPFN 2.5, TabICL as the regressor | 300k training rows exceed their context; GPU needed |
| Deep nets trained from scratch (TFT, PatchTST, DeepAR, N-BEATS; RealMLP, TabM) | GPU training; behind zero-shot models on fev-bench, and N-BEATS trailed gradient boosting in arXiv 2502.03395 |
| Fine-tuning a foundation model on Dubai series | GPU; the zero-shot versions gave no signal to build on |
| Recency weights, short training windows, hyperparameter re-tunes, segmentation | Rejected under the same gate in campaigns 1-2 |

## Reproduce

```bash
python -m model.optimize_model_family prep
python -m model.optimize_model_family cv hgb hgb_anchored lgbm xgb cat hgb_iter2400
python -m model.optimize_model_family blend hgb lgbm hgb_anchored xgb
python -m model.optimize_model_family age hgb
python -m model.optimize_model_family forecast area_rooms naive mean4 ewma bolt_small bolt_base chronos2 chronos2_mv timesfm3
python -m model.optimize_model_family forecast project ewma bolt_small chronos2
python -m model.optimize_model_family cv hgb@area_rooms:ewma hgb@area_rooms:chronos2 hgb@area_rooms:chronos2_mv hgb@area_rooms:timesfm3 hgb@project:ewma hgb@project:chronos2
python -m model.optimize_model_family cadence
python -m model.optimize_model_family holdout hgb hgb+hgb_anchored hgb+lgbm+hgb_anchored
```

Extra packages used (none is an app requirement): lightgbm 4.7.0, xgboost
3.4.1, catboost 1.2.10; torch 2.9.1 (CPU), chronos-forecasting 2.3.2, timesfm
3.0.2. Machine: 4 CPU cores, 16 GB, no GPU.
