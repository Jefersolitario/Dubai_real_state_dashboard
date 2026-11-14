# Model Tuning Review (campaign 5, second pass)

Run 2026-11-14, after `reports/model_family_campaign_report.md`. Two questions
were put to that report: how well was each model tuned, and are the
pre-trained time-series forecasters (TimesFM 3, Chronos) more sensitive to
their settings than XGBoost? The numbers below come from
`python -m model.review_model_tuning` (commands at the end): 52 forecaster
settings, 63 XGBoost forecasters, 4 fine-tunes and 40 tree-learner settings.

## Answer

**No. Used as the campaign used them — pre-trained, not fitted to our data —
TimesFM 3 and Chronos are far less sensitive to their settings than XGBoost.**
On the same forecasting task:

| Forecaster | Settings tried | Best | Typical | Worst | Spread | Beat a moving average | Worse than "same as last week" |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chronos-2, one setting changed at a time | 20 | 0.0965 | 0.0981 | 0.1062 | 10% | 5 of 20 | none |
| Chronos-2, 12 random combinations of those settings | 12 | 0.0969 | 0.0987 | 0.1030 | 6% | 2 of 12 | none |
| Chronos-Bolt | 9 | 0.0990 | 0.0999 | 0.1022 | 3% | none | none |
| TimesFM 3 | 11 | 0.0974 | 0.0984 | 0.1009 | 4% | 1 of 11 | none |
| XGBoost, 60 random settings | 60 | 0.0968 | 0.1011 | 0.1557, and 9 diverged | 58% | 13 of 60 | 13 of 60 |
| Moving average (its one setting, the weight) | 5 | 0.0977 | 0.0997 | 0.1036 | 6% | — | none |

Mean absolute error of the one-week-ahead forecast, in log points (0.10 ≈
10%), every run scored on the same 4,892 cells. "Spread" is worst minus best
over the typical value, diverged runs left out. The moving average scores
0.0977 and "same as last week" 0.1139.

What that table does and does not say:

1. **The pre-trained models have one setting that matters: how much history
   they are shown.** Eight weeks instead of all of it costs Chronos-2 8%;
   everything else — how gaps are filled, logs or raw prices, model size,
   covariates, forecasting districts jointly — moves it by under 2%. The
   campaign's defaults were within 2% of the best setting found for each
   model. Tuning them further was never going to change the outcome.
2. **XGBoost can match them, but only when it is set up correctly.** Its best
   setting equals the best Chronos-2 setting. Its library defaults are 11–14%
   worse and lose to "same as last week" on the median error. Most of that
   fragility is two modelling choices rather than the numeric knobs: asked to
   predict the *change* from a moving average, with features expressed
   relative to it, 23 random settings span only 6% and none fails; asked for
   the price level itself, 8 of 19 diverge.
3. **The pre-trained models stop being insensitive the moment you fine-tune
   them** (section 5). Four learning rates for Chronos-2: three change nothing
   (within ±1%), the fourth makes it 3% worse than not fine-tuning at all.
4. **None of this changes what the forecasts are worth to the pricing model**
   (section 7). Handing the model the *realised* price level of the sale's own
   district and week — which no forecast can beat — improves it by 0.016 pp.

The review does change the campaign report in one place, and it is about the
tree learners, not the forecasters (section 6). **XGBoost was not 0.15 pp
worse than the shipped model; its settings were.** With the leaf size the
other learners use, its categories fed as ordinary numbers, and the
incumbent's learning rate, it scores 3.983% over the ten folds against 3.981%
shipped (the campaign had it at 4.135%). The three libraries are
interchangeable on this problem once set alike, so the design still stays —
there is nothing to gain by switching — but the campaign's ranking of them was
an artefact. The same sweep found an apparent 0.05 pp improvement to the
shipped model's own settings, which did not survive a ten-fold check.

## 1. What campaign 5 actually tuned

Almost nothing except the incumbent. The shipped model's settings came out of
the earlier optimisation loop, which scored them on date-ordered folds. Every
challenger ran once, at settings copied from the incumbent or left at a
library default.

| Model | Settings in the campaign | Tuned? | What the review found wrong with that |
|---|---|---|---|
| HistGradientBoosting (shipped) | rate 0.04, at most 800 trees, 127 leaves, 20 sales per leaf, absolute loss | Yes, earlier, on out-of-time folds | Its early stopping never fires (800 of 800 trees on 8 folds of 10); the cap is what limits it |
| LightGBM | the same, at most 3,000 trees | No, one run | Never stopped early on 9 folds of 10 |
| XGBoost | the same, at most 3,000 trees, but `min_child_weight` left at the library's 1 where the others require 20 sales per leaf | No, one run | Not like for like; far too many trees; its own category handling hurts it (section 6) |
| CatBoost | rate 0.1, at most 1,500 trees, depth 8 | No, one run | Hit its cap on 8 folds of 10: stopped before it had converged. Not re-run here |
| Chronos-Bolt small, base | pre-trained, whole history, one week ahead | No | Nothing varied |
| Chronos-2 | pre-trained; each series alone, and jointly by district | Two variants | History length, gap handling, covariates, cross-learning, fine-tuning never tried |
| TimesFM 3 | pre-trained, library defaults | No | Its flags never tried; run at batch size 4, five times slower than batch 64 for identical output |
| All forecasters | one formulation: weekly median of a district × rooms band, in logs, at least 3 sales, one week ahead | — | The noise in that target was never measured, and no longer horizon was scored |

The campaign also reported its head-to-head differences without intervals.
With them (whole days resampled, since sales on one day share the market move):

| Claim in the campaign report | Difference | 95% interval |
|---|---:|---|
| Retrained weekly vs the live August bundle, 69 test days | +0.24 pp | +0.16 to +0.31 |
| Retrained every 28 days vs trained once on 3 Sep | +0.14 pp | +0.04 to +0.27 |
| Every 7 days vs every 28 days | 0.00 pp | −0.06 to +0.04 |
| Trained once on 3 Sep vs the live August bundle | +0.10 pp | −0.02 to +0.20 |
| LightGBM vs shipped, selection CV | +0.02 pp | +0.00 to +0.04 |
| XGBoost vs shipped, selection CV | −0.15 pp | −0.18 to −0.13 |
| CatBoost vs shipped, selection CV | −0.26 pp | −0.29 to −0.24 |
| Each forecast feature vs shipped, selection CV | −0.02 to +0.00 pp | every interval inside ±0.035 |

The retraining result stands: weekly against the live bundle is the only
comparison here whose whole interval lies above the 0.05 pp gate. (The live
bundle's predictions were scored outside the runner, as its 4.37% was in the
campaign report; the other rows come from the `intervals` step.)

## 2. The forecasting task, and how noisy its target is

Same panel as the campaign: 181 series (55 districts × rooms band), 118 weeks,
each value the log of that week's median AED/sqft when the week has at least 3
sales. Every forecast is made from the weeks before it only, at a rolling
origin. The sweeps use every second selection week (48 origins); all runs of a
family are compared on the cells every one of them forecast, with intervals
from resampling whole weeks. Chronos-2, Chronos-Bolt and TimesFM 3 at their
campaign settings reproduce the campaign's forecasts to the last digit.

A weekly median of a handful of sales is itself a noisy measurement. Splitting
each week's sales at random into two halves and comparing their medians puts
the expected gap between the observed median and the level behind it at
**0.054 log points** on average (cells with at least 4 sales) — against a
total error of 0.095–0.098 for every competent forecaster.

| Sales in the week | Cells | Noise in the target | "Same as last week" | Moving average | Chronos-2 | Chronos-2, cross-learning | Chronos-Bolt | TimesFM 3 | XGBoost, careful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4–9 | 1,548 | 0.079 | 0.143 | 0.121 | 0.122 | 0.120 | 0.124 | 0.124 | 0.122 |
| 10–29 | 1,990 | 0.050 | 0.108 | 0.089 | 0.089 | 0.088 | 0.090 | 0.089 | 0.088 |
| 30 or more | 1,095 | 0.026 | 0.075 | 0.072 | 0.071 | 0.070 | 0.072 | 0.071 | 0.071 |

Two readings. In thin weeks half the error is noise in what is being
predicted, so differences between forecasters are compressed there. And even
where the noise is small (30 or more sales) the weekly median still moves by 7%
in ways no method anticipates — which projects happened to sell that week —
and every method lands within about 3% of every other.

## 3. Pre-trained forecasters, one setting at a time

Mean absolute error one week ahead, and the change from the model's campaign
setting with its 95% interval. Negative is better.

**Chronos-2** (5,589 cells; campaign setting 0.0990, moving average 0.0980)

| Setting changed | Error | Change | Interval |
|---|---:|---:|---|
| History: last 8 weeks only | 0.1069 | +8.1% | +6.6% to +9.9% |
| History: last 13 weeks | 0.1036 | +4.6% | +3.5% to +6.1% |
| History: last 26 weeks | 0.0999 | +1.0% | +0.3% to +1.7% |
| History: last 52 weeks | 0.0986 | −0.4% | −0.8% to +0.1% |
| Gaps filled by interpolation (the model's own handling of missing weeks is the default) | 0.1004 | +1.4% | +0.7% to +2.3% |
| Gaps filled forward | 0.1001 | +1.1% | +0.4% to +2.0% |
| Gaps dropped | 0.0992 | +0.2% | −0.5% to +1.1% |
| Raw AED/sqft instead of logs | 0.0988 | −0.1% | −0.4% to +0.2% |
| A week counts with 1 sale (default 3) | 0.0987 | −0.2% | −1.0% to +0.5% |
| A week counts with 5 sales | 0.1008 | +1.8% | +0.4% to +3.4% |
| Jointly by district (the campaign's second variant) | 0.0979 | −1.0% | −1.7% to −0.3% |
| Jointly by rooms band | 0.0975 | −1.5% | −2.1% to −0.8% |
| All 181 series jointly | 0.0975 | −1.4% | −2.2% to −0.7% |
| Cross-learning across the batch | 0.0974 | −1.5% | −2.2% to −0.9% |
| Covariate: weekly sales count | 0.0986 | −0.4% | −0.8% to 0.0% |
| Covariate: city-wide price level | 0.0985 | −0.4% | −0.8% to −0.1% |
| Both covariates | 0.0986 | −0.4% | −0.9% to +0.1% |
| Smaller model (28M parameters) | 0.0991 | +0.1% | −0.4% to +0.6% |
| Synthetic-data-only variant | 0.0992 | +0.3% | −0.8% to +1.1% |

Twelve random *combinations* of these settings (history × gaps × transform ×
minimum sales × joint or separate × covariate) span 0.0970 to 0.1027 on their
common cells, −1.5% to +4.3% around the campaign setting, and the ordering is
again history length: the four combinations with 13 weeks of history are the
four worst.

One trap, found by those combinations: handed a series with *no* observed week
inside a short window, Chronos-2 does not fail — it returns a number near
zero. Three combinations scored 0.12 to 0.57 until the harness stopped passing
such series (15 to 126 cells out of 5,636 carried the whole excess). The
one-series path already skipped them; the joint path now does too.

**Chronos-Bolt** (5,621 cells; campaign setting, the small model, 0.1009;
moving average 0.0985)

| Setting changed | Error | Change |
|---|---:|---:|
| Tiny / mini / base model | 0.1018 / 0.1016 / 0.1012 | +0.8% / +0.7% / +0.3% |
| History 13 / 26 / 52 weeks | 0.1039 / 0.1019 / 0.1011 | +2.9% / +1.0% / +0.2% |
| Gaps filled by interpolation | 0.1023 | +1.4% |
| Raw AED/sqft | 0.1007 | −0.2% |

Every Chronos-Bolt setting is worse than the moving average.

**TimesFM 3** (5,621 cells; campaign setting 0.1002; moving average 0.0985)

| Setting changed | Error | Change | Interval |
|---|---:|---:|---|
| History 13 / 26 / 52 weeks | 0.1026 / 0.1008 / 0.0997 | +2.4% / +0.6% / −0.5% | ctx 13: +1.3% to +3.6% |
| Gaps dropped (its default interpolates) | 0.0985 | −1.7% | −2.5% to −0.9% |
| A week counts with 1 sale | 0.0990 | −1.2% | −2.4% to −0.2% |
| Jointly by district (multivariate) | 0.0999 | −0.3% | −0.8% to +0.1% |
| Covariate: weekly sales count / city-wide level | 0.1000 / 0.1003 | −0.2% / +0.1% | — |
| Raw AED/sqft | 0.1003 | +0.1% | 0.0% to +0.3% |
| Symmetric averaging flag | 0.1000 | −0.2% | −0.4% to −0.1% |
| `use_znorm` flag | identical | 0 | the model normalises its input itself: forecasts differ by 1e-7 |

**Longer horizons.** The campaign only asked for one week ahead, the easiest
case for a moving average. Further out nothing changes:

| Weeks ahead | Moving average | Chronos-2 | Chronos-2, cross-learning | Chronos-Bolt | TimesFM 3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0987 | 0.0997 | 0.0982 | 0.1013 | 0.1005 |
| 4 | 0.1045 | 0.1052 | 0.1036 | 0.1076 | 0.1090 |
| 13 | 0.1216 | 0.1242 | 0.1213 | 0.1265 | 0.1239 |

At no horizon does a pre-trained model beat the moving average by more than
its interval (best case, Chronos-2 with cross-learning at 4 weeks: −0.0009,
interval −0.0022 to +0.0005).

## 4. XGBoost on the same task

One model across the 181 series, refitted every 4 weeks on earlier weeks only,
forecasting one week ahead from 24 features (lags, rolling means, counts,
district and city momentum, season). Sixty settings drawn at random from the
usual ranges, plus the library's defaults and one careful setting written
before the search. 10,030 cells; moving average 0.0978, "same as last week"
0.1149.

| Settings | Runs | Diverged | Best | Typical | Worst | Beat the moving average |
|---|---:|---:|---:|---:|---:|---:|
| All 60 draws | 60 | 9 | 0.0963 | 0.1011 | 0.1576 | 27% |
| Target is the price level itself | 19 | 8 | 0.1009 | 0.1061 | 0.1576 | none |
| Target is the change from a moving average or the last value | 41 | 1 | 0.0963 | 0.0999 | 0.1245 | 39% |
| … and the level features are expressed relative to the moving average | 23 | 0 | 0.0963 | 0.0974 | 0.1020 | 65% |
| Learning rate 0.3, the library default | 14 | 5 | 0.1039 | 0.1075 | 0.1245 | none |
| Library defaults, level target | 1 | — | 0.1104 | | | no (median error worse than "same as last week") |
| Library defaults, change target | 1 | — | 0.1074 | | | no |
| The careful setting | 1 | — | 0.0969 | | | yes |

The nine divergences are all the pseudo-Huber loss on an uncentred target: its
curvature vanishes far from the target, and with weak regularisation the leaf
values explode. That is a known trap, and it is exactly what sensitivity to
settings means in practice — a pre-trained forecaster has no equivalent.

## 5. Fine-tuning Chronos-2

Zero-shot use has no training settings at all. Fine-tuning brings them back:
Chronos-2 was fine-tuned on the 155 series' first 60 weeks (100 steps, batch
64, four-week horizon) at four learning rates and then forecast the following
48 weeks exactly like the zero-shot model. 2,870 common cells; zero-shot
0.1031, moving average 0.1028.

| Learning rate | Error | vs zero-shot | Interval | Median error |
|---|---:|---:|---|---:|
| not fine-tuned | 0.1031 | — | — | 0.0615 |
| 1e-6 (the library default) | 0.1026 | −0.6% | −1.5% to +0.3% | 0.0620 |
| 1e-5 | 0.1025 | −0.6% | −2.0% to +0.8% | 0.0619 |
| 1e-4 | 0.1023 | −0.8% | −2.7% to +1.1% | 0.0611 |
| 1e-3 | 0.1065 | **+3.3%** | +1.3% to +5.3% | 0.0667 (+8.5%) |

Nothing gained at any rate, and a rate ten times too high makes the model
measurably worse than leaving it alone. So the honest form of the answer is:
*a pre-trained forecaster is insensitive because there is nothing to set; once
you train it, it has a learning rate like everything else, and here the
downside is real and the upside is not.* Each run took four minutes on CPU;
longer schedules and LoRA were not tried.

## 6. The tree learners on the pricing task

Here XGBoost is on home ground, so this is the fairer place to ask how
sensitive it is. Each learner was run on the last three folds of the
campaign's ten (27,562 sales each, same rows, same early-stopping split) with
one setting changed at a time, and every fit also recorded its out-of-time
error at each tree count. The shipped model scores **3.775%** on these folds;
three seeds of it give 3.771–3.777%, so seed noise is about 0.005 pp. Three
folds are not ten, though, as the last check in this section shows.

**XGBoost**

| Setting | Median error | vs shipped |
|---|---:|---:|
| Library defaults | 4.313% | +0.54 pp |
| Campaign setting (rate 0.04, 3,000 trees, 1 sale per leaf) | 3.904% | +0.13 pp |
| The same at rate 0.1 and 2,000 trees | 4.011% | +0.24 pp |
| *Base for the rows below: rate 0.1, 800 trees, 20 sales per leaf* | 3.898% | +0.12 pp |
| Squared-error loss instead of absolute | 4.269% | +0.49 pp |
| Row and column subsampling 0.8 | 3.980% | +0.21 pp |
| Depth-wise trees, depth 8 | 3.948% | +0.17 pp |
| 256 categories per split instead of 64 | 3.935% | +0.16 pp |
| 100 sales per leaf / 63 leaves | 3.919% / 3.918% | +0.14 pp |
| Category codes fed as ordinary numbers | 3.822% | +0.05 pp |
| … and the incumbent's learning rate 0.04, 2,000 trees | **3.775%** | **0.00 pp** |

**LightGBM**

| Setting | Median error | vs shipped |
|---|---:|---:|
| Library defaults | 4.160% | +0.39 pp |
| *Base for the rows below: rate 0.1, 1,600 trees* | 3.852% | +0.08 pp |
| Squared-error loss | 4.207% | +0.43 pp |
| Row and column subsampling 0.8 | 3.988% | +0.21 pp |
| 63 / 255 leaves; category codes as numbers | 3.851% / 3.863%; 3.861% | +0.08 / +0.09 pp |
| 100 sales per leaf | 3.808% | +0.03 pp |
| Stopping on the most recent tenth (below) | 3.781% | +0.01 pp |
| Campaign setting (rate 0.04, 3,000 trees) | **3.740%** | **−0.04 pp** |
| The campaign setting with 100 sales per leaf | 3.749% | −0.03 pp |
| The campaign setting at its best tree count, 1,200 | 3.728% | −0.05 pp |

**HistGradientBoosting, the shipped model**

| Setting | Median error | vs shipped |
|---|---:|---:|
| Library defaults | 4.132% | +0.36 pp |
| Squared-error loss | 4.012% | +0.24 pp |
| Category codes as ordinary numbers | 3.794% | +0.02 pp |
| 200 sales per leaf | 3.785% | +0.01 pp |
| Learning rate 0.02 / 0.08 | 3.783% / 3.775% | +0.01 / 0.00 pp |
| Shipped | 3.775% | — |
| 50 sales per leaf / 255 leaves | 3.767% / 3.763% | −0.01 pp |
| Up to 2,400 trees instead of 800 | 3.760% | −0.02 pp |
| Stopping on the most recent tenth | 3.748% | −0.03 pp |
| 100 sales per leaf | 3.745% | −0.03 pp |
| 100 sales per leaf and up to 2,400 trees | **3.725%** | **−0.05 pp** |

What this shows:

- **On this task the three libraries are about equally sensitive, and
  moderately so.** Between sensible settings each moves by 0.3–0.5 pp (8–13%
  of its best). One choice dominates in all three — the loss function,
  0.24–0.37 pp — and the library defaults cost 0.36–0.54 pp. In relative terms
  that is the same order as what the pre-trained forecasters lose from their
  one bad setting (2–8% for too little history) and nothing like XGBoost's
  fragility as a forecaster: there, with 10,000 noisy rows and a target that
  can be posed well or badly, the settings decide everything; here, with
  300,000 rows and one obvious target, they decide a tenth of the error.
- **Set alike, the three converge.** XGBoost 3.775%, shipped 3.775%, LightGBM
  3.740%. Over all ten folds the tuned XGBoost (1,600 trees, fixed) scores
  3.983% against 3.981% shipped — interval on the difference −0.02 to
  +0.03 pp, and level on the seven folds the sweep never looked at (4.074% vs
  4.069%). The campaign's "XGBoost −0.15 pp" was its settings: one sale per
  leaf, far too many trees, and its own handling of 200-level categories,
  which costs it 0.08 pp here (raising the categories-per-split limit makes it
  worse, not better; feeding the codes as numbers fixes it — and does nothing
  for LightGBM or the shipped model, whose category splits work differently).
- **Early stopping as configured never fires, for any of the three.** It
  watches a random tenth of the training sales, which share dates and projects
  with the other nine tenths, so its error keeps falling while the error on
  later sales has already turned. XGBoost with one sale per leaf is best at
  200 trees and 0.10 pp worse at 2,000, with the stopping rule silent
  throughout; LightGBM at rate 0.1 loses 0.05 pp between 600 and 1,600 trees.
  Stopping on the *most recent* tenth and refitting on everything picks
  254–784 trees for LightGBM (3.852% → 3.781%) and 1,460–1,840 for the shipped
  model (3.775% → 3.748%). The shipped model was never exposed to this: its
  800-tree cap came from the optimisation loop, which scores on later sales,
  and unlike XGBoost it does not degrade past the cap — it improves slightly.
- **The incumbent sits on a flat optimum, and the one apparent improvement
  was selection noise.** Requiring 100 sales per leaf and lifting the tree cap
  looked 0.05 pp better on these three folds, with better tails. Run over all
  ten folds it is 3.984% against 3.981%: the three late folds gain 0.05 pp,
  the seven earlier ones lose 0.03 pp. Picking the best of fifteen settings on
  three folds manufactures a gain of exactly that size. Nothing changes in
  `fair_value_config.json`.
- A caution on the campaign's gate: across three seeds of the shipped model
  the worst-decile error varies by 0.12 pp and the false-flag share by 0.10 pp
  on these three folds. The tail vetoes (+0.10 pp and +0.05 pp) are therefore
  about one seed's worth of noise at this sample size; they are meaningful on
  the full ten folds, not on a subset.

## 7. What any forecast feature could be worth

The campaign added each forecast to the pricing model as features and found
nothing (every change inside ±0.035 pp). Tuning the forecasters cannot alter
that, and the reason can be measured directly. Instead of a forecast, hand the
model the *answer*: the realised median price of the sale's own segment in the
sale's own week, computed from that week's other sales. This is deliberate
leakage and only a ceiling — no forecast made from earlier weeks can know more
about the level than the level itself.

| Added to the shipped model (10-fold selection CV) | Median error | Gain | Worst decile | False-flag share |
|---|---:|---:|---:|---:|
| Nothing (shipped) | 3.981% | — | 15.21% | 4.56% |
| Realised level of the district × rooms band that week (96% of sales covered) | 3.965% | +0.016 pp | 15.13% | 4.51% |
| Realised level of the project that week (72% covered) | 3.959% | +0.022 pp | 14.89% | 4.38% |
| Best-tuned Chronos-2 forecast (cross-learning), as the campaign's three features | 3.991% | −0.010 pp | 15.22% | 4.55% |

Intervals on the gains, pooled over the out-of-fold sales: district level
−0.004 to +0.032 pp, project level +0.005 to +0.041 pp, the tuned forecast
−0.020 to +0.007 pp.

A sale sits a median 9.2% away from its segment's realised level that week and
10.3% from the trailing comparable the model already has. Knowing the level
exactly closes a tenth of that gap; the model's own error is 4% because it
prices from the project's and the unit's own history, not from the district.
The remaining error is unit-level — the campaign's conclusion, now with a
number on it: **a perfect price-level forecast is worth 0.02 pp**, less than
half the 0.05 pp gate. (The project-level row does trim the tails; that is the
value of knowing what the same project sold for *this week* — launch batches —
which is information about concurrent sales, not something a forecast has.)

## 8. What changes in the campaign report, and what stands

| Campaign verdict | After this review |
|---|---|
| Retraining is the lever: +0.24 pp weekly against the live August bundle | **Stands.** Interval +0.16 to +0.31 pp; monthly captures it (7 vs 28 days: 0.00, −0.06 to +0.04) |
| TimesFM 3 / Chronos forecasts as features add nothing | **Stands, and is now explained.** No setting moves their accuracy by more than a few percent, none beats a moving average by more than its interval at 1, 4 or 13 weeks, the best-tuned one adds −0.01 pp, and a perfect forecast would add 0.02 pp |
| XGBoost is 0.15 pp worse than the shipped model | **Withdrawn.** Set like the incumbent it equals it: 3.983% vs 3.981% over ten folds (interval on the difference −0.02 to +0.03 pp), worst decile 15.13% vs 15.21%, false-flag share 4.49% vs 4.56% |
| LightGBM is +0.02 pp, not worth a new dependency | **Stands.** It was already at its good setting (the incumbent's learning rate); interval 0.00 to +0.04 pp |
| CatBoost is 0.27 pp worse | **Open.** Stopped before converging in the campaign; not re-run |
| The shipped design and settings stay | **Stands.** Apart from the loss function, thirteen settings of the shipped model span 3.73–3.79% on the late folds, and the best of them gains nothing over ten folds |
| Blends: real (+0.09 pp held out) but below the complexity bar | Not revisited |

Lessons for the next campaign, in the order they cost this one:

1. **Tune the challenger with the incumbent's protocol or do not rank it.** One
   run at copied settings put XGBoost 0.15 pp behind; a dozen runs put it
   level.
2. **Do not pick settings on a few folds and report the same folds.** The
   shipped model's "improvement" was +0.05 pp on three folds and −0.03 pp on
   the other seven.
3. **Early stopping on a random split is blind to drift.** It never fired for
   any learner. Either fix the tree count on out-of-time folds (what the
   shipped model does) or stop on the most recent sales and refit.
4. **Put an interval on every difference.** Three seeds of one model differ by
   0.006 pp in the median but 0.12 pp in the worst decile.
5. **A pre-trained forecaster needs its inputs checked, not its settings
   tuned.** The only way to make Chronos-2 fail here was to hand it a series
   with nothing in it.

## Not done

- **CatBoost with a larger budget.** Its campaign run stopped at its
  1,500-tree cap; a fair run is about 25 minutes per fold on this machine.
- **Fine-tuning TimesFM 3.** The package exposes inference only.
- **Fine-tuning Chronos-2 with LoRA, or for more than 100 steps** (needs the
  `peft` package, or a GPU to be practical).
- **Forecasts as features at other encodings.** Section 7's ceiling makes this
  moot for a district- or project-level price forecast.

## Reproduce

```bash
python -m model.optimize_model_family prep                       # the campaign's feature frame
python -m model.review_model_tuning forecast --list              # setting names
python -m model.review_model_tuning forecast c2 c2_ctx13 c2_cross bolt_small tfm tfm_compress
python -m model.review_model_tuning xgb-forecast 60
python -m model.review_model_tuning finetune 1e-5 100
python -m model.review_model_tuning score-forecasts
python -m model.review_model_tuning learners --list              # setting names
python -m model.review_model_tuning learners x_base x_mcw20 x_m20_ordinal x_m20_ordinal_lr04 l_base l_recent l_lr04 h_base h_msl100_long
python -m model.review_model_tuning score-learners
python -m model.review_model_tuning learners x_tuned h_msl100_long --all-folds   # the two ten-fold checks
python -m model.review_model_tuning oracle area_rooms
python -m model.review_model_tuning forecast c2_cross --every 1  # then, as features of the pricing model:
python -m model.review_model_tuning export-forecast c2_cross
python -m model.optimize_model_family cv hgb@area_rooms:c2_cross
python -m model.review_model_tuning intervals lgbm xgb cat hgb_anchored   # needs the campaign's cv and cadence steps
```

Extra packages as for the campaign runner (`lightgbm`, `xgboost`; `torch`,
`chronos-forecasting`, `timesfm` for the forecast steps); none is an app
requirement. On 4 CPU cores a Chronos-2 setting takes about a minute, a
TimesFM 3 setting about two, a fine-tune of 100 steps about four, a tree
setting one to three.
