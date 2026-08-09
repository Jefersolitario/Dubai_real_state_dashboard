# Distress-Claim Validation — Walk-Forward Outcome Backtest (v2, corrected)

**Date:** 2026-07-05 · **Script:** `backtest_flags.py` (v2) · **Data:** full
24-month snapshot (266,161 scoreable sales through 2026-07-02), shipped model
configuration, walk-forward flags (141,749 out-of-sample scored entries).

> **Correction note.** The first version of this report headlined "+21% median
> excess return" from nested horizon buckets, iid bootstrap CIs, and pairs that
> mostly compare *different units of the same layout*. Two independent audits
> found those statistics overstated by mechanical artifacts (unit-key
> collisions, selection-on-price mean reversion, fake horizon replication,
> too-narrow CIs). This v2 fixes the methodology — uniqueness strata,
> resale-spread reversion check, symmetric cohort, disjoint buckets,
> building-cluster bootstrap, off-plan-aware matching, deep-tail exclusion —
> and restates the findings. The effect survives, smaller and better understood.

## The decisive test: do flagged entries' gaps close, or are the units just cheap?

Each pair's **resale** was scored by the walk-forward models. If flagged units
were "persistently cheap for an unobserved reason" (fairly priced lemons), their
resales would also sit near −15–20% spread. They do not:

| Cohort | Entry spread (median) | Resale spread (median) | n |
|---|---|---|---|
| Flagged (−35% … −15%) | **−19.8%** | **+3.2%** | 1,075 |
| Controls (±5%) | 0.0% | +0.1% | 11,893 |
| Overpriced (≥ +15%) | +21.6% | +0.4% | 1,754 |

Flagged entries' resales come back to (slightly above) fair value: **the gap is
real and it closes**. Overpriced entries also revert — premiums paid don't
persist either. Both directions behave as genuine one-time price gaps, not as
persistent unit-quality effects.

## Corrected headline: excess return vs own district (cluster-robust CIs)

Disjoint holding buckets, entries restricted to have full runway; CIs from a
building-level cluster bootstrap; deep tail (< −35%) excluded from "flagged"
and shown separately:

**Held 30–183 days** (entries ≤ 2025-12-31):

| Cohort / stratum | n | Median excess return |
|---|---|---|
| Flagged — all | 1,200 | **+18.2%** [+16.7, +20.1] |
| Flagged — **unique unit** (registry-proven same apartment) | 60 | **+14.2%** [+10.5, +23.0] |
| Flagged — stacked layout (same-spec unit trades) | 885 | +18.7% [+16.9, +20.5] |
| Control — all | 14,280 | −0.3% [−0.5, −0.0] |
| Overpriced (≥ +15%) | 1,938 | −9.0% [−10.0, −8.0] |
| Deep tail (< −35%) | 290 | +64.5% [degenerate — winsor-clipped; treat as data noise] |
| Matched flagged−control (99 off-plan-aware cells) | | **+20.6%** [+17.4, +23.7] |

**Held 184–365 days** (entries ≤ 2025-07-02): flagged +19.6% [+15.0, +23.0]
(n=228) vs controls −3.3% [−4.2, −2.3]. The 366–550d bucket has too few
walk-forward entries with runway to report (0 qualifying matched cells).

Interpretation notes:
- The **unique-unit stratum is the strongest evidence**: collisions are
  impossible there, and the +14.2% gap over controls (−0.9% in that stratum)
  stands with a CI clear of zero despite n=60.
- The stacked-layout majority measures "entry price vs the next same-spec
  trade in the building" — still a real price gap, but not literally the same
  buyer's exit; its higher +18.7% partly reflects within-layout position the
  model cannot see (floor/view), so quote the unique-unit number when claiming
  buyer-realizable value.
- The flat profile across holding buckets + the reversion table = the value is
  captured **at entry** (gap recapture), not post-purchase appreciation skill.
- Overpriced entries mirror at −9% — the metric is symmetric, so the flagged
  result must be read together with the reversion check above (which is what
  rules out "mean reversion around noisy comps" as the whole story: resales
  land AT fair value, not below it).

## Signal-strength calibration (deep tail excluded)

Monotone: decile 1 → −0.5%, decile 5 → +1.0%, decile 8 → +4.8%, decile 9 →
+7.5%, decile 10 → **+17.8%** (each n≈1,536; 30–365d holds). The UI's ranking
order matches realized value. (Caveat: monotonicity alone is also predicted by
gap-recapture mechanics; its value is confirming the *ordering* users see.)

## Regime drift (model frozen 2026-02, scoring the falling market OOS)

| Month | Median spread | Flag rate |
|---|---|---|
| 2026-02 | +0.14% | 4.6% |
| 2026-03 | −0.04% | 3.6% |
| 2026-04 | −0.53% | 4.7% |
| 2026-05 | −1.11% | 7.6% |
| 2026-06 | −1.09% | 6.7% |
| 2026-07 | −1.69% | 6.7% | *(partial month: 2 days)*

A ~4-month-stale model overstates fair values by ≈1.1% (June) and inflates the
flag rate ~2pp in this decline. No trend-extrapolation blowup (trees saturate;
no momentum features shipped); the mechanism is comp-window lag. **Ops rule:**
weekly retrains keep staleness at days; alarm if the live monthly median spread
drifts beyond ±2%.

## Survivorship and other disclosures

- Resale rates (clean denominators): flagged 27.3%, controls 21.5%, deep tail
  33.3%, overpriced 24.5%. Outcomes condition on reselling; flagged units
  resell more (flippers), deep-tail "units" most of all (consistent with
  non-standard records).
- `PROCEDURE_EN` contains **zero forced-sale vocabulary** across all 141,749
  OOS entries — the "distressed" corroboration currently rests on
  illiquidity/multi-seller proxies only. External labels (Emirates Auction
  match, ~100 records) remain the path to a true precision/recall for the
  distress *cause*; the monthly top-20 human review starts with live listings.
- Controls at longer holds drift slightly negative (−3.3%): units that resell
  within 6–12 months in a decelerating market underperform their district
  index; cohort comparisons are within-bucket so this does not bias the gap.

## Verdict (corrected)

The system's **below-fair-value flags are validated out-of-sample**: flagged
prices are genuinely below market — their resales return to fair value, and on
registry-proven same-unit pairs the buyer's realized edge over the district is
**≈ +14%** (all-pairs view ≈ +18–20%, which includes same-spec trades rather
than strict resales). The ranking is calibrated. The **distress-cause label
remains unvalidated** (no forced-sale ground truth in this dataset) — keep the
product wording as "below fair value — corroborated" until external labels
exist, and treat sub-−35% "discounts" as probable data noise, not deals.
