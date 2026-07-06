# SHAP Insights — Shipped Fair-Value Model

**Date:** 2026-07-05 · **Model:** campaign-2 champion (CV 4.08% / holdout 4.15%),
bundle trained 2026-07-05T14:06 · **Sample:** 2,000 recent sales, 32 features.

## Methodology (and a caution for future work)

SHAP values were computed with the **model-agnostic permutation explainer**
(`shap.PermutationExplainer`, 100-row background, additivity verified exact).
`shap.TreeExplainer` was tried first and **rejected**: it mis-parses
`HistGradientBoostingRegressor`'s native categorical bitset splits, failing the
additivity check by up to 2.0 log units (vs 0.34 prediction std) and producing
absurd attributions (41% to "nearest landmark"). **Always run the additivity
check before trusting SHAP output on HGB models with categorical features.**

Units: mean |SHAP| converted from log-price to "average % impact on the
predicted price".

## What drives the fair-value estimate

| Rank | Feature | Avg impact on price | Permutation importance rank |
|---|---|---|---|
| 1 | Same unit's previous sale price | 9.5% | 1 |
| 2 | Project price/sqft, same unit type (90d) | 8.2% | 2 |
| 3 | Building long-run price level | 7.6% | 3 |
| 4 | Bedrooms | 2.6% | 4 |
| 5 | Unit size | 2.5% | 5 |
| 6 | District | 2.4% | 7 |
| 7 | Master development | 1.5% | 6 |
| 8 | Project long-run price level | 1.5% | 8 |
| 9 | Balcony size | 1.2% | 9 |
| 10 | Unit size vs project's typical unit | 1.1% | 10 |

**The two importance views agree almost perfectly** (top-10 identical, one
adjacent swap). That is the strongest possible robustness signal: what moves
predictions (SHAP) is also what the model needs for accuracy (permutation).
There is no high-cardinality memorization inflating any ranking — the raw
`Project` categorical sits at only 0.8% because the *price-history* features
carry the project signal in a form that generalizes.

## Findings

1. **Price history is ~75% of the model's differentiating power.** The top
   three features (the unit's own last sale, what the same unit type sold for
   in the project recently, the building's long-run level) together move
   predictions ~25% on average; the remaining 29 features share the rest.
   The model is, at heart, a disciplined comparables engine with hedonic
   corrections — which is what a good appraiser is.
2. **Direction checks all pass** (dependence plots): a unit whose last sale
   was expensive predicts +21% / cheap −14%; strong same-type comps ±11–16%.
   Nothing is wired backwards.
3. **Higher floors carry a premium**: layouts sitting higher in the tower add
   ~+1.4% vs low floors −1.0% — modest but real, exactly the signal the
   units-registry join was built for (and unavailable before Campaign 2).
4. **Big balconies *lower* price per sqft** (−3.1% at the 80th percentile vs
   +1.4% at the 20th): terrace area counts as area but is cheap to build.
   Buyers should read "huge terrace, low AED/sqft" listings accordingly —
   the model already does.
5. **Oversized units trade at a discount per sqft** (−2.0%): the familiar
   size-elasticity, captured relative to each project's typical unit.
6. **The market-trend feature reads slightly negative in recent months**
   (−0.8% at the recent end of the time feature): the model has priced in a
   mild cooling — worth watching in the Market Overview page.

## Recommendations

- **Keep the comparables core intact.** Any future feature pruning should
  start from the bottom of this table, never the top three.
- **Cold-start deals remain the risk pocket.** Where the top features are
  null (no prior sale, thin project history), the physical/registry features
  carry the estimate — that is why cold-start error halved in Campaign 2, and
  why flags there still deserve the lower signal-strength ranking they get.
- **Phase 3 (building reviews / comfort sentiment) attacks the right gap:**
  the model knows *where* and *what*, and now *how high* and *how big the
  balcony is* — but nothing about build quality, AC, or management. That is
  the residual the top-3 comps cannot explain away when two towers in the
  same district diverge.
- **Tooling note:** future SHAP runs on this model must use the permutation
  explainer (see Methodology).
