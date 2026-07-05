# Distress-Claim Validation — Walk-Forward Outcome Backtest

**Date:** 2026-07-05 · **Script:** `backtest_flags.py` · **Data:** full 24-month
snapshot (266,161 scoreable sales through 2026-07-02), shipped model configuration.

## Design (and why it is hindsight-safe)

Entries are flagged **walk-forward**: four quarterly models, each trained only on
data strictly before its quarter, flag only that quarter (Jan 2025 – Jan 2026,
141,749 out-of-sample scored entries). No flag decision comes from a model that
saw the entry — or its later resale — in training. Features were already
strictly past by construction; this extends the guarantee to the model weights.

Outcomes are realized facts: each entry is paired with the same pseudo-unit's
next sale (building + rooms + area to 0.1 sqm — the model's own repeat-sale
key), and the outcome metric is **excess return** = the unit's price change
minus its own district's change (trailing 30-day district median) over the same
dates. Guards: ≥30-day holding (shorter = likely assignments), ≥180-day resale
runway, excess returns winsorized at 1%/99%, non-market procedures excluded on
both legs, 2,000-draw bootstrap CIs.

## Result 1 — Flagged deals carry real, recoverable value

| Holding period | Flagged (spread ≤ −15%) | Near-fair controls | Matched-cell difference |
|---|---|---|---|
| 30–183 days | **+21.4%** [+19.1, +23.1] (n=1,490) | −0.3% [−0.5, −0.1] (n=14,280) | **+24.0%** [+21.8, +26.2] (649 cells) |
| 30–365 days | **+21.5%** [+19.9, +23.0] (n=1,886) | −0.6% [−0.7, −0.4] (n=16,703) | **+23.6%** [+21.5, +25.5] (749 cells) |
| 30–550 days | **+21.1%** [+19.6, +22.8] (n=1,943) | −0.6% [−0.8, −0.5] (n=17,048) | **+23.4%** [+21.3, +25.4] (763 cells) |

- Controls sit at ≈0 by construction — the sanity check passes.
- Flagged entries out-appreciate their own district by ≈ the full flagged
  discount: the model's "below fair value" gap is **recovered at resale**.
- Flagged units also **resell more often** (28.2% vs 21.5% of eligible entries)
  — consistent with investors buying discounts and exiting. Both cohorts'
  outcomes condition on reselling; this differential is disclosed.

## Result 2 — The signal-strength ranking is well calibrated

Median excess return by signal-strength decile (30–365d holds, negative-spread
entries): monotone from **−0.5%** (decile 1) through +1.0% (5), +4.8% (8),
+8.7% (9) to **+23.3%** (decile 10). The ranking the UI sorts by is exactly the
ordering of realized value — the "prefer high-× deals" guidance is validated.

## Result 3 — Regime drift is real, modest, and now quantified

A model frozen at 2026-02-01 scoring the falling market out-of-sample:

| Month | Median spread | Flag rate |
|---|---|---|
| 2026-02 | +0.14% | 4.6% |
| 2026-03 | −0.04% | 3.6% |
| 2026-04 | −0.53% | 4.7% |
| 2026-05 | −1.11% | 7.6% |
| 2026-06 | −1.09% | 6.7% |
| 2026-07 | −1.69% | 6.7% |

A 5-month-stale model over-states fair values by ~1.7% and inflates the flag
rate by ~2pp in this downturn. The model does **not** extrapolate the bull
trend (trees saturate; no momentum features shipped) — the failure mode is
comp-window **lag**, and weekly retraining keeps effective staleness at days,
not months. **Ops rule adopted:** track monthly median spread; if it drifts
beyond ±2%, retrain immediately rather than waiting for the weekly cadence.

## Result 4 — The forced-sale signal never fires (product finding)

`PROCEDURE_EN` in the snapshot contains **zero** forced-sale procedures (no
court/auction/liquidation vocabulary at all — sales appear only as
Sell / Sell - Pre registration / Delayed Sell etc.). The "distressed"
corroboration therefore rests entirely on illiquidity and multi-seller signals
today. DLD-ordered auctions exist but are published elsewhere (e.g. Emirates
Auction), not in this dataset's procedure field.

## Honest limitations

1. **Recoverable value ≠ purchasable deal.** The backtest proves flagged prices
   are ~20% below what the unit fetches at its next sale. It cannot prove each
   entry was available to *you* (off-market transfers, related-party pricing,
   or recording quirks can produce real-looking discounts). Distinguishing
   those requires external ground truth — the planned Emirates Auction match
   (~100 labels → precision/recall) and, once live listings land, the monthly
   top-20 human review (precision@20).
2. Outcomes condition on resale (28% vs 22% resale rates disclosed above);
   units that never resold are not measured.
3. The window covers one cycle turn; the regime table covers a mild decline.

## Verdict

The **below-fair-value claim is validated out-of-sample**: flags mark deals
whose discount is real and fully recovered at resale, and the signal-strength
ranking orders them correctly. The **"distressed" label should be reframed**
until external labels exist: what the system reliably finds is *deeply
underpriced sales*; the distress *cause* (forced sale vs off-market transfer
vs motivated seller) is currently inferred from weak proxies. Recommended copy:
"below fair value — corroborated" rather than "distressed", plus the
Emirates Auction labelling exercise before the listings launch.
