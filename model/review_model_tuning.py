"""Campaign 5 review: how much do its verdicts depend on settings?

Campaign 5 ran every alternative learner and every pre-trained forecaster at
one configuration. This module measures what that left open:

* ``forecast`` — the pre-trained forecasters (Chronos-2, Chronos-Bolt,
  TimesFM 3) asked the same question in different ways, one change at a time:
  history length, gap handling, log or raw prices, minimum sales per week,
  model size, joint forecasting, covariates.
* ``xgb-forecast`` — XGBoost as a forecaster of the same weekly panel over a
  random search of its settings, so its spread can be read against theirs.
* ``finetune`` — Chronos-2 fine-tuned on the panel's early weeks at a given
  learning rate and step count.
* ``score-forecasts`` — every forecast run scored on identical cells, with the
  sampling noise of the target itself.
* ``learners`` — the tree learners on the pricing task, one setting at a time
  on the late CV folds; each fit also reports its error at every tree count.
* ``score-learners`` — the table for those runs.
* ``oracle`` — the pricing model handed the *realised* weekly level of the
  sale's segment (leave-one-out): a ceiling for any forecast feature.
* ``export-forecast`` — a forecast run written in the campaign runner's format,
  so ``optimize_model_family cv hgb@area_rooms:<name>`` can test it as features.
* ``intervals`` — day-block bootstrap intervals for the campaign's own
  head-to-head numbers (learners against the shipped model, retraining cadences).

Working files go to ``<workdir>/review``; ``optimize_model_family prep`` must
have run. Extra packages as for the campaign runner. Findings are curated in
``reports/model_tuning_review_report.md``.

Usage:
    python -m model.review_model_tuning forecast c2 c2_ctx13 tfm tfm_district
    python -m model.review_model_tuning forecast --list
    python -m model.review_model_tuning xgb-forecast 60
    python -m model.review_model_tuning finetune 1e-5 100
    python -m model.review_model_tuning score-forecasts
    python -m model.review_model_tuning learners x_base x_mcw20 l_base h_base
    python -m model.review_model_tuning learners x_tuned h_msl100_long --all-folds
    python -m model.review_model_tuning score-learners
    python -m model.review_model_tuning oracle area_rooms
    python -m model.review_model_tuning forecast c2_cross --every 1
    python -m model.review_model_tuning export-forecast c2_cross
    python -m model.review_model_tuning intervals lgbm xgb hgb_anchored
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from model.fair_value_model import _fold_metrics, _make_model, fit_encoders, to_matrix
from model.optimize_model_family import (
    DEFAULT_WORKDIR,
    FIRST_ORIGIN_WEEK,
    MIN_CONTEXT_WEEKS,
    N_SPLITS,
    ROOM_BANDS,
    SEED,
    SELECTION_END,
    TEST_END,
    TEST_START,
    Campaign,
    _early_stop_split,
    fit_hgb,
    segment_expr,
)

HORIZONS = (1, 4, 13)            # weeks ahead stored for every pre-trained forecast
QUANTILES = [0.1, 0.5, 0.9]
LATE_FOLDS = (8, 9, 10)          # folds of TimeSeriesSplit(10) the learner sweeps use
TREE_GRID = [100, 200, 300, 400, 600, 800, 1200, 1600, 2000, 2400, 3000, 4000]
FINETUNE_FROM_WEEK = 60          # Chronos-2 is fine-tuned on weeks before this one
DENSE_SALES = 15                 # a week with this many sales has little sampling noise


# ---------------------------------------------------------------------------
# the weekly panel every forecast run shares
# ---------------------------------------------------------------------------

class Panel:
    """Weekly log median AED/sqft per district x rooms band, with counts and noise."""

    def __init__(self, campaign: Campaign):
        feats = campaign.features().select("date", "psf", "AREA_EN", "ROOMS_EN")
        sales = feats.with_columns(
            segment_expr("area_rooms").alias("seg"), pl.col("date").dt.truncate("1w").alias("week")
        ).filter(pl.col("seg").is_not_null())
        weekly = sales.group_by("seg", "week").agg(
            pl.col("psf").median().alias("med"), pl.len().alias("n"), pl.col("psf").log().alias("x")
        )
        self.weeks: list[date] = pl.date_range(
            sales["week"].min(), sales["week"].max(), interval="1w", eager=True
        ).to_list()
        enough = (
            weekly.filter(pl.col("n") >= 3).group_by("seg").agg(pl.len().alias("obs"))
            .filter(pl.col("obs") >= MIN_CONTEXT_WEEKS + 4)
        )
        self.segs: list[str] = sorted(enough["seg"].to_list())
        seg_i = {s: i for i, s in enumerate(self.segs)}
        week_i = {w: i for i, w in enumerate(self.weeks)}
        shape = (len(self.segs), len(self.weeks))
        self.med = np.full(shape, np.nan, dtype=np.float32)
        self.cnt = np.zeros(shape, dtype=np.float32)
        self.noise = np.full(shape, np.nan, dtype=np.float32)
        rng = np.random.default_rng(3)
        rows = weekly.filter(pl.col("seg").is_in(self.segs)).select("seg", "week", "med", "n", "x")
        for seg, week, med, n, x in rows.iter_rows():
            i, j = seg_i[seg], week_i[week]
            self.med[i, j], self.cnt[i, j] = np.log(med), n
            if n >= 4:
                self.noise[i, j] = _split_half_noise(np.asarray(x), rng)
        city = sales.group_by("week").agg(pl.col("psf").median().log().alias("c"))
        self.city = np.full(len(self.weeks), np.nan, dtype=np.float32)
        for week, c in city.iter_rows():
            self.city[week_i[week]] = c
        # last week that starts inside the selection period (its final days reach into the test window)
        self.last_selection_week = max(i for i, w in enumerate(self.weeks) if w <= SELECTION_END)
        self.Y3 = self.observed(3)
        self.eligible = np.zeros(shape, dtype=bool)
        for t in range(FIRST_ORIGIN_WEEK, shape[1]):
            self.eligible[np.isfinite(self.Y3[:, :t]).sum(axis=1) >= MIN_CONTEXT_WEEKS, t] = True

    def observed(self, min_sales: int) -> np.ndarray:
        """The panel with weeks of fewer than ``min_sales`` sales treated as missing."""
        return np.where(self.cnt >= min_sales, self.med, np.nan).astype(np.float32)

    def rows(self, t: int) -> np.ndarray:
        """Series forecast at origin ``t``: the campaign's rule, >= 8 observed weeks before it."""
        return np.flatnonzero(self.eligible[:, t])

    def target(self, h: int) -> np.ndarray:
        """Realised level of week t+h-1 at column t, selection weeks only."""
        out = np.full(self.med.shape, np.nan, dtype=np.float32)
        cols = np.arange(self.med.shape[1] - (h - 1))
        keep = cols[cols + h - 1 <= self.last_selection_week]
        out[:, keep] = self.Y3[:, keep + h - 1]
        return out


def _split_half_noise(x: np.ndarray, rng: np.random.Generator, splits: int = 24) -> float:
    """Expected |weekly median - true level| from the gap between two random halves.

    sd(m_a - m_b) = sd(m_all) * sqrt(n/n_a + n/n_b), so equal halves give a
    factor of 2. A lower bound on what no forecast can predict: it ignores
    which projects happened to sell that week.
    """
    gaps = []
    for _ in range(splits):
        p = rng.permutation(x.size)
        a, b = x[p[: x.size // 2]], x[p[x.size // 2:]]
        gaps.append(abs(np.median(a) - np.median(b)) / np.sqrt(x.size / a.size + x.size / b.size))
    return float(np.mean(gaps))


def moving_average(panel: Panel, kind: str, alpha: float = 0.3) -> np.ndarray:
    """Flat forecasts from the observed weeks before each origin: naive / mean4 / ewma."""
    Y = panel.Y3
    out = np.full(Y.shape, np.nan, dtype=np.float32)
    for s in range(Y.shape[0]):
        level, seen = np.nan, []
        for t in range(Y.shape[1]):
            if panel.eligible[s, t]:
                out[s, t] = {"naive": seen[-1], "mean4": np.mean(seen[-4:])}.get(kind, level)
            if np.isfinite(Y[s, t]):
                seen.append(Y[s, t])
                level = Y[s, t] if len(seen) == 1 else alpha * Y[s, t] + (1 - alpha) * level
    return out


# ---------------------------------------------------------------------------
# forecast: pre-trained forecasters, one setting at a time
# ---------------------------------------------------------------------------

_BASE = dict(ctx=None, gaps="nan", transform="log", min_sales=3, group="uni", cov=None)
_C2 = dict(_BASE, model="chronos2", repo="amazon/chronos-2")
_BOLT = dict(_BASE, model="bolt", repo="amazon/chronos-bolt-small")
_TFM = dict(_BASE, model="timesfm3", gaps="interp", sym=False)

FORECAST_SETTINGS: dict[str, dict] = {
    # Chronos-2: the first entry is the campaign's run
    "c2": _C2,
    "c2_ctx8": dict(_C2, ctx=8), "c2_ctx13": dict(_C2, ctx=13),
    "c2_ctx26": dict(_C2, ctx=26), "c2_ctx52": dict(_C2, ctx=52),
    "c2_interp": dict(_C2, gaps="interp"), "c2_ffill": dict(_C2, gaps="ffill"),
    "c2_compress": dict(_C2, gaps="compress"),
    "c2_raw": dict(_C2, transform="raw"),
    "c2_min1": dict(_C2, min_sales=1), "c2_min5": dict(_C2, min_sales=5),
    "c2_district": dict(_C2, group="district"), "c2_rooms": dict(_C2, group="rooms"),
    "c2_all": dict(_C2, group="all"), "c2_cross": dict(_C2, group="cross"),
    "c2_cov_count": dict(_C2, cov=("count",)), "c2_cov_city": dict(_C2, cov=("city",)),
    "c2_cov_both": dict(_C2, cov=("count", "city")),
    "c2_small": dict(_C2, repo="autogluon/chronos-2-small"),
    "c2_synth": dict(_C2, repo="autogluon/chronos-2-synth"),
    # Chronos-Bolt
    "bolt_small": _BOLT,
    "bolt_tiny": dict(_BOLT, repo="amazon/chronos-bolt-tiny"),
    "bolt_mini": dict(_BOLT, repo="amazon/chronos-bolt-mini"),
    "bolt_base": dict(_BOLT, repo="amazon/chronos-bolt-base"),
    "bolt_ctx13": dict(_BOLT, ctx=13), "bolt_ctx26": dict(_BOLT, ctx=26),
    "bolt_ctx52": dict(_BOLT, ctx=52),
    "bolt_interp": dict(_BOLT, gaps="interp"), "bolt_raw": dict(_BOLT, transform="raw"),
    # TimesFM 3 (its own gap handling is linear interpolation; use_znorm is inert,
    # the model normalises its input itself)
    "tfm": _TFM,
    "tfm_sym": dict(_TFM, sym=True),
    "tfm_ctx13": dict(_TFM, ctx=13), "tfm_ctx26": dict(_TFM, ctx=26), "tfm_ctx52": dict(_TFM, ctx=52),
    "tfm_compress": dict(_TFM, gaps="compress"), "tfm_raw": dict(_TFM, transform="raw"),
    "tfm_min1": dict(_TFM, min_sales=1),
    "tfm_district": dict(_TFM, group="district"),
    "tfm_cov_count": dict(_TFM, cov=("count",)), "tfm_cov_city": dict(_TFM, cov=("city",)),
}

# Twelve random combinations of the Chronos-2 settings (drawn once, seed 0), to set
# beside the random search of XGBoost: (history, gaps, transform, min sales, group, covariate).
for _i, (_ctx, _gaps, _transform, _min, _group, _cov) in enumerate([
    (52, "ffill", "log", 1, "cross", None), (13, "nan", "log", 5, "uni", ("count",)),
    (52, "ffill", "raw", 5, "cross", None), (52, "nan", "raw", 5, "district", None),
    (None, "ffill", "log", 3, "uni", ("city",)), (13, "nan", "raw", 5, "district", None),
    (13, "nan", "raw", 1, "cross", None), (13, "nan", "log", 3, "district", None),
    (26, "nan", "log", 1, "uni", None), (52, "nan", "raw", 1, "district", None),
    (None, "nan", "log", 3, "district", None), (None, "compress", "log", 5, "cross", None),
]):
    FORECAST_SETTINGS[f"c2_rnd{_i:02d}"] = dict(
        _C2, ctx=_ctx, gaps=_gaps, transform=_transform, min_sales=_min, group=_group, cov=_cov
    )


def _fill(row: np.ndarray, mode: str) -> np.ndarray | None:
    """One context under a gap rule; None when the window holds no observed week."""
    seen = np.flatnonzero(np.isfinite(row))
    if not seen.size:
        return None
    if mode == "nan":
        return row
    if mode == "compress":
        return row[seen]
    row = row[seen[0]:].copy()
    idx, ok = np.arange(row.size), np.isfinite(row)
    if mode == "interp":
        return np.interp(idx, idx[ok], row[ok]).astype(np.float32)
    return row[np.maximum.accumulate(np.where(ok, idx, 0))]          # forward fill


class SettingsRunner:
    """One pre-trained forecaster under one setting; call it per forecast origin."""

    def __init__(self, cfg: dict, panel: Panel, threads: int):
        import torch

        torch.set_num_threads(threads)
        self.torch, self.cfg, self.panel = torch, cfg, panel
        self.Y = panel.observed(cfg["min_sales"])
        if cfg["model"] == "bolt":
            from chronos import ChronosBoltPipeline
            self.pipe = ChronosBoltPipeline.from_pretrained(cfg["repo"], device_map="cpu")
        elif cfg["model"] == "chronos2":
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained(cfg["repo"], device_map="cpu")
        else:
            import timesfm
            # the default batch of 4 gives identical forecasts five times slower
            self.pipe = timesfm.TimesFM3Forecaster.from_pretrained(
                "google/timesfm-3.0-pytorch", device="cpu", per_core_batch_size=64
            )

    def _start(self, t: int) -> int:
        return 0 if self.cfg["ctx"] is None else max(0, t - self.cfg["ctx"])

    def _to_log(self, values: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(values, 1e-6)) if self.cfg["transform"] == "raw" else values

    def _from_log(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values) if self.cfg["transform"] == "raw" else values

    def _covariates(self, i: int, t: int, length: int) -> dict[str, np.ndarray]:
        """Past-only covariates aligned with a context of ``length`` trailing weeks."""
        out = {}
        for name in self.cfg["cov"] or ():
            series = np.log1p(self.panel.cnt[i, :t]) if name == "count" else self.panel.city[:t]
            out[name] = series[-length:].astype(np.float32)
        return out

    def __call__(self, t: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(median forecasts [rows, horizons], 10-90% width one week ahead), in log points."""
        cfg, H = self.cfg, max(HORIZONS)
        at = [h - 1 for h in HORIZONS]
        fc = np.full((len(rows), len(HORIZONS)), np.nan, dtype=np.float32)
        width = np.full(len(rows), np.nan, dtype=np.float32)
        if cfg["group"] in ("district", "rooms", "all"):
            key = {"district": lambda s: s.split("|")[0], "rooms": lambda s: s.split("|")[1],
                   "all": lambda s: "all"}[cfg["group"]]
            groups: dict[str, list[int]] = {}
            for pos, i in enumerate(rows):
                # A series with no observed week inside the window gets no forecast, as in the
                # one-series path: handed an all-missing series the model still returns a
                # number (near zero) instead of failing.
                if np.isfinite(self.Y[i, self._start(t):t]).any():
                    groups.setdefault(key(self.panel.segs[i]), []).append(pos)
            blocks = {k: self._from_log(self.Y[rows[v], self._start(t):t]) for k, v in groups.items()}
            if cfg["model"] == "chronos2":
                keys = list(blocks)
                q, _ = self.pipe.predict_quantiles(
                    [blocks[k] for k in keys], prediction_length=H, quantile_levels=QUANTILES
                )
                for k, tensor in zip(keys, q):
                    arr = self._to_log(tensor.numpy())               # (variates, H, quantiles)
                    fc[groups[k]], width[groups[k]] = arr[:, at, 1], arr[:, 0, 2] - arr[:, 0, 0]
            else:                                                    # TimesFM: one call per variate count
                by_size: dict[int, list[str]] = {}
                for k, block in blocks.items():
                    by_size.setdefault(block.shape[0], []).append(k)
                for keys in by_size.values():
                    outs = self.pipe.predict_batch(
                        [blocks[k] for k in keys], horizon=H, return_quantiles=True,
                        use_symmetric_averaging=cfg["sym"],
                    )
                    for k, out in zip(keys, outs):
                        med = self._to_log(np.atleast_2d(np.asarray(out.forecast)))
                        qq = self._to_log(np.asarray(out.quantiles)).reshape(len(groups[k]), H, -1)
                        fc[groups[k]], width[groups[k]] = med[:, at], qq[:, 0, -1] - qq[:, 0, 0]
            return fc, width

        contexts = [_fill(self.Y[i, self._start(t):t], cfg["gaps"]) for i in rows]
        live = [p for p, c in enumerate(contexts) if c is not None]
        if not live:
            return fc, width
        inputs = [self._from_log(contexts[p]) for p in live]
        if cfg["model"] == "chronos2":
            if cfg["cov"]:
                inputs = [
                    {"target": x, "past_covariates": self._covariates(rows[p], t, len(x))}
                    for p, x in zip(live, inputs)
                ]
            extra = dict(cross_learning=True, batch_size=100) if cfg["group"] == "cross" else {}
            q, _ = self.pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=QUANTILES, **extra)
            arr = self._to_log(np.stack([tensor.numpy()[0] for tensor in q]))
        elif cfg["model"] == "bolt":
            q, _ = self.pipe.predict_quantiles(
                [self.torch.tensor(x) for x in inputs], prediction_length=H, quantile_levels=QUANTILES
            )
            arr = self._to_log(q.numpy())
        else:
            cov = None
            if cfg["cov"]:
                cov = [np.stack(list(self._covariates(rows[p], t, len(x)).values())) for p, x in zip(live, inputs)]
            outs = list(self.pipe.predict_batch(
                inputs, horizon=H, past_only_covariates=cov, return_quantiles=True,
                use_symmetric_averaging=cfg["sym"],
            ))
            med = self._to_log(np.stack([np.asarray(o.forecast) for o in outs]))
            qq = self._to_log(np.stack([np.asarray(o.quantiles) for o in outs]))
            arr = np.stack([qq[:, :, 0], med, qq[:, :, -1]], axis=-1)
        fc[live], width[live] = arr[:, at, 1], arr[:, 0, 2] - arr[:, 0, 0]
        return fc, width


def _review_dir(campaign: Campaign) -> Path:
    path = Path(campaign.path("review"))
    path.mkdir(exist_ok=True)
    return path


def _rolling(panel: Panel, runner, first: int, last: int, every: int) -> tuple[np.ndarray, np.ndarray]:
    S, W = panel.med.shape
    fc = np.full((S, W, len(HORIZONS)), np.nan, dtype=np.float32)
    width = np.full((S, W), np.nan, dtype=np.float32)
    for t in range(first, last + 1, every):
        rows = panel.rows(t)
        if rows.size:
            fc[rows, t], width[rows, t] = runner(t, rows)
    return fc, width


def run_forecast(campaign: Campaign, names: list[str], every: int, through_test: bool, threads: int) -> None:
    panel = Panel(campaign)
    last = panel.med.shape[1] - 1 if through_test else panel.last_selection_week
    for name in names:
        started = time.time()
        fc, width = _rolling(panel, SettingsRunner(FORECAST_SETTINGS[name], panel, threads), FIRST_ORIGIN_WEEK, last, every)
        np.savez(_review_dir(campaign) / f"fc_{name}.npz", fc=fc, width=width)
        err = np.abs(fc[:, :, 0] - panel.target(1))
        err = err[np.isfinite(err)]
        print(f"== {name}: {time.time() - started:.0f}s | one week ahead, selection weeks: MAE "
              f"{err.mean():.4f} median {np.median(err):.4f} log pts (n {err.size:,})", flush=True)


def run_finetune(campaign: Campaign, learning_rate: float, steps: int, threads: int) -> None:
    """Fine-tune Chronos-2 on weeks before ``FINETUNE_FROM_WEEK`` and forecast the later ones."""
    panel = Panel(campaign)
    runner = SettingsRunner(_C2, panel, threads)
    train = []
    for row in panel.Y3[:, :FINETUNE_FROM_WEEK]:
        seen = np.flatnonzero(np.isfinite(row))
        if seen.size >= 12:
            train.append(row[seen[0]:].copy())
    scratch = _review_dir(campaign) / f"finetune_{learning_rate:g}_{steps}"
    started = time.time()
    runner.pipe = runner.pipe.fit(
        train, prediction_length=4, learning_rate=learning_rate, num_steps=steps, batch_size=64,
        output_dir=str(scratch), remove_printer_callback=True, report_to="none",
    )
    fit_seconds = time.time() - started
    fc, width = _rolling(panel, runner, FINETUNE_FROM_WEEK, panel.last_selection_week, 1)
    shutil.rmtree(scratch, ignore_errors=True)
    name = f"ft_lr{learning_rate:g}_s{steps}"
    np.savez(_review_dir(campaign) / f"fc_{name}.npz", fc=fc, width=width)
    err = np.abs(fc[:, :, 0] - panel.target(1))
    err = err[np.isfinite(err)]
    print(f"== {name}: {len(train)} series, fitted in {fit_seconds:.0f}s | MAE {err.mean():.4f} "
          f"median {np.median(err):.4f} (n {err.size:,})", flush=True)


# ---------------------------------------------------------------------------
# xgb-forecast: XGBoost on the same panel, random search
# ---------------------------------------------------------------------------

XGB_FORECAST_SPACE = {
    "target": ["level", "delta_ewma", "delta_last"],     # what the trees predict
    "feats": ["raw", "rel"],                              # level features as they are, or minus the moving average
    "weight": ["none", "count"],
    "objective": ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"],
    "n_estimators": [50, 100, 300, 1000],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "max_depth": [2, 3, 4, 6, 8],
    "min_child_weight": [1, 5, 20, 50],
    "subsample": [0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}
_XGB_LIBRARY_DEFAULT = dict(
    objective="reg:squarederror", n_estimators=100, learning_rate=0.3, max_depth=6,
    min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, weight="none",
)
XGB_FORECAST_NAMED = {
    "default_level_raw": dict(_XGB_LIBRARY_DEFAULT, target="level", feats="raw"),
    "default_delta_rel": dict(_XGB_LIBRARY_DEFAULT, target="delta_ewma", feats="rel"),
    "careful": dict(
        objective="reg:absoluteerror", n_estimators=300, learning_rate=0.03, max_depth=3,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        weight="count", target="delta_ewma", feats="rel",
    ),
}
XGB_FORECAST_REFIT_WEEKS, XGB_FORECAST_FIRST_FIT = 4, 24


def _forecast_features(panel: Panel) -> dict[int, dict]:
    """Per-origin feature blocks; block t uses weeks before t only."""
    Y, cnt, city = panel.Y3, panel.cnt, panel.city
    S, W = Y.shape
    ewma = {a: np.full((S, W), np.nan, dtype=np.float32) for a in (0.3, 0.1)}
    for a, table in ewma.items():
        level = np.full(S, np.nan, dtype=np.float32)
        for t in range(W):
            table[:, t] = level
            seen = np.isfinite(Y[:, t])
            level = np.where(seen, np.where(np.isfinite(level), a * Y[:, t] + (1 - a) * level, Y[:, t]), level)
    district = np.array([s.split("|")[0] for s in panel.segs])
    bands = ROOM_BANDS + ["4+"]
    band = np.array([bands.index(s.split("|")[1]) for s in panel.segs], dtype=np.float32)
    blocks = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                               # all-missing windows give NaN on purpose
        for t in range(FIRST_ORIGIN_WEEK, W):
            past = Y[:, :t]
            last_idx = np.where(np.isfinite(past), np.arange(t), -1).max(axis=1)
            last = np.where(last_idx >= 0, past[np.arange(S), np.maximum(last_idx, 0)], np.nan)
            mean = {k: np.nanmean(past[:, max(0, t - k):], axis=1) for k in (4, 8, 13, 26, t)}
            momentum = mean[4] - mean[13]
            season = panel.weeks[t].isocalendar()[1] / 52.0 * 2 * np.pi
            anchor = ewma[0.3][:, t]
            levels = [last, past[:, t - 1], past[:, t - 2], past[:, t - 3], past[:, t - 4],
                      mean[4], mean[8], mean[13], mean[26], mean[t], ewma[0.1][:, t]]
            other = [
                (t - 1 - last_idx).astype(np.float32), np.nanstd(past[:, max(0, t - 13):], axis=1), momentum,
                np.array([np.nanmean(momentum[district == d]) for d in district]),
                np.log1p(cnt[:, t - 1]), np.log1p(cnt[:, t - 4:t].mean(axis=1)),
                np.log1p(cnt[:, max(0, t - 13):t].mean(axis=1)),
                np.full(S, city[t - 1] - np.nanmean(city[max(0, t - 9):t - 1])), np.full(S, city[t - 1] - city[t - 5]),
                band, np.full(S, np.sin(season)), np.full(S, np.cos(season)),
            ]
            blocks[t] = {
                "raw": np.column_stack(levels + [anchor] + other).astype(np.float32),
                "rel": np.column_stack([f - anchor for f in levels] + [anchor] + other).astype(np.float32),
                "ewma": anchor, "last": last, "y": Y[:, t], "w": np.clip(cnt[:, t], 1, 50), "rows": panel.rows(t),
            }
    return blocks


def _xgb_forecast(cfg: dict, blocks: dict, shape: tuple[int, int], last: int, threads: int) -> np.ndarray:
    import xgboost

    fc = np.full(shape, np.nan, dtype=np.float32)
    params = {k: v for k, v in cfg.items() if k not in ("target", "feats", "weight")}
    if params["objective"] == "reg:pseudohubererror":
        params["huber_slope"] = 0.1

    def anchor(block: dict) -> np.ndarray:
        return {"level": np.zeros(shape[0], dtype=np.float32), "delta_ewma": block["ewma"],
                "delta_last": block["last"]}[cfg["target"]]

    model = None
    for t in range(XGB_FORECAST_FIRST_FIT, last + 1):
        if (t - XGB_FORECAST_FIRST_FIT) % XGB_FORECAST_REFIT_WEEKS == 0:
            X, y, w = [], [], []
            for u in range(FIRST_ORIGIN_WEEK, t):
                b = blocks[u]
                rows = b["rows"][np.isfinite(b["y"][b["rows"]]) & np.isfinite(anchor(b)[b["rows"]])]
                X.append(b[cfg["feats"]][rows]); y.append(b["y"][rows] - anchor(b)[rows]); w.append(b["w"][rows])
            model = xgboost.XGBRegressor(tree_method="hist", n_jobs=threads, random_state=0, **params)
            model.fit(np.concatenate(X), np.concatenate(y),
                      sample_weight=np.concatenate(w) if cfg["weight"] == "count" else None)
        b = blocks[t]
        rows = b["rows"][np.isfinite(anchor(b)[b["rows"]])]
        fc[rows, t] = model.predict(b[cfg["feats"]][rows]) + anchor(b)[rows]
    return fc


def run_xgb_forecast(campaign: Campaign, n_random: int, seed: int, threads: int) -> None:
    panel = Panel(campaign)
    blocks = _forecast_features(panel)
    rng = np.random.default_rng(seed)
    configs = dict(XGB_FORECAST_NAMED)
    for i in range(n_random):
        configs[f"random{i:02d}"] = {k: v[int(rng.integers(len(v)))] for k, v in XGB_FORECAST_SPACE.items()}
    target, runs = panel.target(1), {}
    for name, cfg in configs.items():
        runs[name] = _xgb_forecast(cfg, blocks, panel.med.shape, panel.last_selection_week, threads)
        err = np.abs(runs[name] - target)
        err = err[np.isfinite(err)]
        print(f"{name:<18} MAE {err.mean():.4f} median {np.median(err):.4f} | "
              + " ".join(f"{k}={v}" for k, v in cfg.items()), flush=True)
    np.savez(_review_dir(campaign) / "fc_xgb_search.npz", **runs)
    (_review_dir(campaign) / "xgb_search_settings.json").write_text(json.dumps(configs, indent=1))


# ---------------------------------------------------------------------------
# score-forecasts
# ---------------------------------------------------------------------------

def _load_forecasts(campaign: Campaign, panel: Panel) -> dict[str, np.ndarray]:
    """name -> forecasts [series, weeks, horizons], NaN where a run issued none."""
    runs: dict[str, np.ndarray] = {}
    wide = lambda flat: np.repeat(flat[:, :, None], len(HORIZONS), axis=2)
    for path in sorted(_review_dir(campaign).glob("fc_*.npz")):
        stored = np.load(path)
        if path.stem == "fc_xgb_search":
            for key in stored.files:
                one = np.full(panel.med.shape + (len(HORIZONS),), np.nan, dtype=np.float32)
                one[:, :, 0] = stored[key]
                runs[f"xgb:{key}"] = one
        else:
            runs[path.stem[3:]] = stored["fc"]
    runs["naive"], runs["mean4"] = wide(moving_average(panel, "naive")), wide(moving_average(panel, "mean4"))
    for alpha in (0.1, 0.2, 0.3, 0.5, 0.7):
        runs[f"ewma{alpha}"] = wide(moving_average(panel, "ewma", alpha))
    return runs


def _week_bootstrap(a: np.ndarray, b: np.ndarray, mask: np.ndarray, reps: int = 400) -> tuple[float, float, float]:
    """mean(a) - mean(b) over the masked cells, 95% interval from resampling whole weeks."""
    weeks = np.flatnonzero(mask.any(axis=0))
    per_week = [(a[mask[:, w], w], b[mask[:, w], w]) for w in weeks]
    rng = np.random.default_rng(5)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(weeks), len(weeks))
        draws.append(np.concatenate([per_week[i][0] for i in pick]).mean()
                     - np.concatenate([per_week[i][1] for i in pick]).mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(a[mask].mean() - b[mask].mean()), float(low), float(high)


def _family_table(panel: Panel, runs: dict, title: str, default: str, names: list[str], first: int) -> dict:
    refs = ["naive", "ewma0.3"]
    target = panel.target(1)
    mask = np.isfinite(target)
    mask[:, :first] = False
    for name in refs + names:
        mask &= np.isfinite(runs[name][:, :, 0])
    err = {name: np.abs(runs[name][:, :, 0] - target) for name in refs + names}
    dense = mask & (panel.cnt >= DENSE_SALES)
    print(f"\n### {title} ({int(mask.sum()):,} cells, {int(mask.any(axis=0).sum())} weeks)")
    print(f"{'run':<22}{'MAE':>8}{'median':>8}{'dense':>8}   MAE vs family default (95% CI)")
    for name in refs + names:
        d = _week_bootstrap(err[name], err[default], mask) if name != default else (0.0, 0.0, 0.0)
        print(f"{name:<22}{err[name][mask].mean():>8.4f}{np.median(err[name][mask]):>8.4f}"
              f"{err[name][dense].mean():>8.4f}   {d[0]:+.4f} ({d[1]:+.4f} .. {d[2]:+.4f})")
    mae = {name: float(err[name][mask].mean()) for name in names}
    best, worst = min(mae, key=mae.get), max(mae, key=mae.get)
    out = {"default": mae[default], "best": [best, mae[best]], "worst": [worst, mae[worst]], "n_settings": len(names),
           "median_setting": float(np.median(list(mae.values()))), "moving_average": float(err["ewma0.3"][mask].mean())}
    print(f"  default {mae[default]:.4f} | best {mae[best]:.4f} ({best}, {mae[best] / mae[default] - 1:+.1%}) | "
          f"worst {mae[worst]:.4f} ({worst}, {mae[worst] / mae[default] - 1:+.1%}) | moving average {out['moving_average']:.4f}")
    return out


def score_forecasts(campaign: Campaign) -> None:
    panel = Panel(campaign)
    runs = _load_forecasts(campaign, panel)
    have = sorted(runs)
    target = panel.target(1)
    # a forecast of a log price beyond e^20 has diverged: counted, then left out of the tables
    diverged = [n for n in have if n.startswith("xgb:") and np.nanmax(np.abs(runs[n][:, :, 0])) > 20]
    one_at_a_time = lambda prefix: [n for n in have if n.startswith(prefix) and "_rnd" not in n]
    families = {
        "Chronos-2, one setting at a time": ("c2", one_at_a_time("c2"), FIRST_ORIGIN_WEEK),
        "Chronos-2, random combinations": ("c2", ["c2"] + [n for n in have if n.startswith("c2_rnd")], FIRST_ORIGIN_WEEK),
        "Chronos-Bolt": ("bolt_small", one_at_a_time("bolt"), FIRST_ORIGIN_WEEK),
        "TimesFM 3": ("tfm", one_at_a_time("tfm"), FIRST_ORIGIN_WEEK),
        "Chronos-2 fine-tuned": ("c2", ["c2"] + [n for n in have if n.startswith("ft_")], FINETUNE_FROM_WEEK),
        "XGBoost forecaster": ("xgb:careful", [n for n in have if n.startswith("xgb:") and n not in diverged],
                               XGB_FORECAST_FIRST_FIT),
        "Moving average, smoothing weight": ("ewma0.3", [n for n in have if n.startswith("ewma")], FIRST_ORIGIN_WEEK),
    }
    out = {"diverged": diverged}
    for title, (default, names, first) in families.items():
        if default in runs and len(names) > 1:
            out[title] = _family_table(panel, runs, title, default, names, first)

    # every family on identical cells
    groups = {title: [n for n in names if n not in ("xgb:careful", "xgb:default_level_raw", "xgb:default_delta_rel")]
              for title, (default, names, _) in families.items() if title != "Chronos-2 fine-tuned" and default in runs}
    groups["Chronos-2, random combinations"] = [n for n in groups.get("Chronos-2, random combinations", []) if n != "c2"]
    everything = sorted({n for names in groups.values() for n in names} | {"naive", "ewma0.3"})
    mask = np.isfinite(target)
    mask[:, :XGB_FORECAST_FIRST_FIT] = False
    for name in everything:
        mask &= np.isfinite(runs[name][:, :, 0])
    mae = {name: float(np.abs(runs[name][:, :, 0] - target)[mask].mean()) for name in everything}
    print(f"\n### Every family on the same {int(mask.sum()):,} cells: \"same as last week\" {mae['naive']:.4f}, "
          f"moving average {mae['ewma0.3']:.4f}; {len(diverged)} XGBoost settings diverged")
    print(f"{'family':<36}{'n':>4}{'best':>9}{'typical':>9}{'worst':>9}{'spread':>8}{'beat MA':>9}{'worse than naive':>18}")
    for title, names in groups.items():
        if names:
            v = np.array([mae[n] for n in names])
            print(f"{title:<36}{len(names):>4}{v.min():>9.4f}{np.median(v):>9.4f}{v.max():>9.4f}"
                  f"{(v.max() - v.min()) / np.median(v):>8.0%}{int((v < mae['ewma0.3']).sum()):>9}"
                  f"{int((v > mae['naive']).sum()):>18}")
            out[f"same cells: {title}"] = {"n": len(names), "best": float(v.min()), "typical": float(np.median(v)),
                                            "worst": float(v.max())}

    lead = [n for n in ("naive", "ewma0.3", "c2", "c2_cross", "bolt_small", "tfm") if n in runs]
    for j, h in enumerate(HORIZONS):
        ahead = panel.target(h)
        cells = np.isfinite(ahead)
        for name in lead:
            cells &= np.isfinite(runs[name][:, :, j])
        print(f"\n### {h} week(s) ahead ({int(cells.sum()):,} cells): " + " | ".join(
            f"{name} {np.abs(runs[name][:, :, j] - ahead)[cells].mean():.4f}" for name in lead))
    lead = lead + [n for n in ("xgb:careful",) if n in runs]
    cells = np.isfinite(target) & np.isfinite(panel.noise)
    for name in lead:
        cells &= np.isfinite(runs[name][:, :, 0])
    out["noise"] = float(panel.noise[cells].mean())
    print(f"\n### Sampling noise of the weekly median itself: {out['noise']:.4f} ({int(cells.sum()):,} cells with >= 4 sales)")
    for low, high in ((4, 9), (10, 29), (30, 10 ** 6)):
        band = cells & (panel.cnt >= low) & (panel.cnt <= high)
        print(f"  weeks with {low}-{high if high < 10 ** 6 else '+'} sales (n {int(band.sum()):,}): noise "
              f"{panel.noise[band].mean():.4f} | " + " | ".join(
                  f"{name} {np.abs(runs[name][:, :, 0] - target)[band].mean():.4f}" for name in lead))
    (_review_dir(campaign) / "forecast_scores.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# learners: tree learners on the pricing task, one setting at a time
# ---------------------------------------------------------------------------

def _matrices(campaign: Campaign, train: pl.DataFrame, val: pl.DataFrame):
    encoders = fit_encoders(train, campaign.cfg)
    X_tr, y_tr, _, cat_idx = to_matrix(train, encoders, campaign.cfg)
    X_va, y_va, _, _ = to_matrix(val, encoders, campaign.cfg)
    return X_tr, y_tr, X_va, y_va, cat_idx


def _tree_curve(y_va: np.ndarray, built: int, predict_at) -> dict[str, list[float]]:
    """Validation MedAPE / P90 / flag share had training stopped at each tree count."""
    out = {}
    for k in TREE_GRID:
        if k <= built:
            m = _fold_metrics(y_va, predict_at(k))
            out[str(k)] = [m["medape"], m["p90_ape"], m["flag_prop"]]
    return out


def _stop_split(n: int, recent: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows to fit on and rows to stop on: a random tenth, or the most recent tenth.

    Training folds are date-ordered, so the last tenth is the newest sales —
    the only split that can see a model drifting out of date.
    """
    if not recent:
        return _early_stop_split(n)
    cut = int(n * 0.9)
    return np.arange(cut), np.arange(cut, n)


def _fit_xgb(campaign: Campaign, train, val, threads: int, **over) -> tuple[np.ndarray, dict]:
    import xgboost

    X_tr, y_tr, X_va, y_va, cat_idx = _matrices(campaign, train, val)
    native = over.pop("cat_mode", "native") == "native"
    recent = over.pop("stop_on", "random") == "recent"
    params = dict(objective="reg:absoluteerror", tree_method="hist", grow_policy="lossguide", max_leaves=127,
                  max_depth=0, learning_rate=0.1, n_estimators=800, reg_lambda=1.0, max_bin=256,
                  early_stopping_rounds=100, n_jobs=threads, random_state=SEED)
    if native:
        types = ["c" if i in set(cat_idx) else "q" for i in range(X_tr.shape[1])]
        params.update(enable_categorical=True, feature_types=types, max_cat_to_onehot=1)
    params.update(over)
    fit_idx, stop_idx = _stop_split(len(y_tr), recent)
    model = xgboost.XGBRegressor(**params)
    model.fit(X_tr[fit_idx], y_tr[fit_idx], eval_set=[(X_tr[stop_idx], y_tr[stop_idx])], verbose=False)
    if recent:                                   # refit on every sale with the tree count the newest tenth chose
        picked = int(model.best_iteration + 1)
        model = xgboost.XGBRegressor(**{**params, "n_estimators": picked, "early_stopping_rounds": None})
        model.fit(X_tr, y_tr, verbose=False)
        return model.predict(X_va), {
            "n_iter": picked, "curve": _tree_curve(y_va, picked, lambda k: model.predict(X_va, iteration_range=(0, k))),
        }
    built = int(model.get_booster().num_boosted_rounds())
    stopped = params.get("early_stopping_rounds") is not None
    return model.predict(X_va), {
        "n_iter": int(model.best_iteration + 1) if stopped else built,
        "curve": _tree_curve(y_va, built, lambda k: model.predict(X_va, iteration_range=(0, k))),
    }


def _fit_lgbm(campaign: Campaign, train, val, threads: int, **over) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    X_tr, y_tr, X_va, y_va, cat_idx = _matrices(campaign, train, val)
    categorical = cat_idx if over.pop("cat_mode", "native") == "native" else []
    recent = over.pop("stop_on", "random") == "recent"
    params = dict(objective="l1", learning_rate=0.1, n_estimators=1600, num_leaves=127, min_child_samples=20,
                  reg_lambda=1.0, max_bin=255, n_jobs=threads, random_state=SEED, verbose=-1)
    params.update(over)
    fit_idx, stop_idx = _stop_split(len(y_tr), recent)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr[fit_idx], y_tr[fit_idx], eval_set=[(X_tr[stop_idx], y_tr[stop_idx])], eval_metric="l1",
              categorical_feature=categorical, callbacks=[lgb.early_stopping(100, verbose=False)])
    if recent:
        picked = int(model.best_iteration_ or model.n_estimators)
        model = lgb.LGBMRegressor(**{**params, "n_estimators": picked})
        model.fit(X_tr, y_tr, categorical_feature=categorical)
        return model.predict(X_va), {
            "n_iter": picked, "curve": _tree_curve(y_va, picked, lambda k: model.predict(X_va, num_iteration=k)),
        }
    built = int(model.booster_.current_iteration())
    return model.predict(X_va), {
        "n_iter": int(model.best_iteration_ or built),
        "curve": _tree_curve(y_va, built, lambda k: model.predict(X_va, num_iteration=k)),
    }


def _fit_hgb(campaign: Campaign, train, val, threads: int, **over) -> tuple[np.ndarray, dict]:
    """The shipped learner; thread count follows OMP_NUM_THREADS."""
    X_tr, y_tr, X_va, y_va, cat_idx = _matrices(campaign, train, val)
    seed = over.pop("seed", SEED)
    if over.pop("cat_mode", "native") == "ordinal":
        cat_idx = []                              # category codes (frequency rank) as an ordinary number
    if over.pop("stop_on", "random") == "recent":
        # sklearn can only stop on a random split: grow on the older nine tenths,
        # take the best tree count on the newest tenth, refit on everything
        fit_idx, stop_idx = _stop_split(len(y_tr), True)
        probe = _make_model({**campaign.params, **over, "early_stopping": False}, cat_idx, seed)
        probe.fit(X_tr[fit_idx], y_tr[fit_idx])
        errors = [np.mean(np.abs(y_tr[stop_idx] - p)) for p in probe.staged_predict(X_tr[stop_idx])]
        over = {**over, "max_iter": int(np.argmin(errors)) + 1, "early_stopping": False}
    model = _make_model({**campaign.params, **over}, cat_idx, seed)
    model.fit(X_tr, y_tr)
    staged = {i: p for i, p in enumerate(model.staged_predict(X_va), 1) if i in TREE_GRID}
    return model.predict(X_va), {
        "n_iter": int(model.n_iter_), "curve": _tree_curve(y_va, int(model.n_iter_), lambda k: staged[k]),
    }


_M20 = dict(min_child_weight=20)
_M20_CODES = {**_M20, "cat_mode": "ordinal"}
LEARNER_SETTINGS: dict[str, tuple] = {
    # XGBoost: the campaign's settings at learning rate 0.1, then one change at a time
    "x_base": (_fit_xgb, dict(n_estimators=2000)),
    "x_mcw20": (_fit_xgb, _M20), "x_mcw100": (_fit_xgb, dict(min_child_weight=100)),
    "x_m20_leaves63": (_fit_xgb, {**_M20, "max_leaves": 63}),
    "x_m20_depth8": (_fit_xgb, {**_M20, "grow_policy": "depthwise", "max_depth": 8, "max_leaves": 0}),
    "x_m20_sub": (_fit_xgb, {**_M20, "subsample": 0.8, "colsample_bytree": 0.8}),
    "x_m20_sqerr": (_fit_xgb, {**_M20, "objective": "reg:squarederror"}),
    "x_m20_catthr256": (_fit_xgb, {**_M20, "max_cat_threshold": 256}),
    "x_m20_ordinal": (_fit_xgb, _M20_CODES),            # category codes as ordinary numbers
    "x_ordinal": (_fit_xgb, dict(cat_mode="ordinal")),
    "x_m20_ordinal_leaves255": (_fit_xgb, {**_M20_CODES, "max_leaves": 255}),
    "x_m20_ordinal_lr04": (_fit_xgb, {**_M20_CODES, "learning_rate": 0.04, "n_estimators": 2000}),
    "x_m20_recent": (_fit_xgb, {**_M20, "stop_on": "recent"}),
    # where the sweep ended: leaf size and learning rate of the incumbent, codes as numbers, fixed tree count
    "x_tuned": (_fit_xgb, {**_M20_CODES, "learning_rate": 0.04, "n_estimators": 1600, "early_stopping_rounds": None}),
    "x_default": (_fit_xgb, dict(objective="reg:squarederror", grow_policy="depthwise", max_depth=6, max_leaves=0,
                                 learning_rate=0.3, n_estimators=100, early_stopping_rounds=None)),
    # LightGBM
    "l_base": (_fit_lgbm, {}),
    "l_mcs100": (_fit_lgbm, dict(min_child_samples=100)),
    "l_leaves63": (_fit_lgbm, dict(num_leaves=63)), "l_leaves255": (_fit_lgbm, dict(num_leaves=255)),
    "l_sub": (_fit_lgbm, dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)),
    "l_ordinal": (_fit_lgbm, dict(cat_mode="ordinal")),
    "l_sqerr": (_fit_lgbm, dict(objective="l2")),
    "l_recent": (_fit_lgbm, dict(stop_on="recent")),
    "l_lr04": (_fit_lgbm, dict(learning_rate=0.04, n_estimators=3000)),
    "l_mcs100_lr04": (_fit_lgbm, dict(min_child_samples=100, learning_rate=0.04, n_estimators=3000)),
    "l_default": (_fit_lgbm, dict(objective="l2", n_estimators=100, num_leaves=31, reg_lambda=0.0)),
    # the shipped HistGradientBoosting model
    "h_base": (_fit_hgb, {}),
    "h_long": (_fit_hgb, dict(max_iter=2400)),
    "h_lr02": (_fit_hgb, dict(learning_rate=0.02)), "h_lr08": (_fit_hgb, dict(learning_rate=0.08)),
    "h_leaves255": (_fit_hgb, dict(max_leaf_nodes=255)),
    "h_msl50": (_fit_hgb, dict(min_samples_leaf=50)), "h_msl100": (_fit_hgb, dict(min_samples_leaf=100)),
    "h_msl200": (_fit_hgb, dict(min_samples_leaf=200)),
    "h_msl100_long": (_fit_hgb, dict(min_samples_leaf=100, max_iter=2400)),
    "h_sqerr": (_fit_hgb, dict(loss="squared_error")),
    "h_ordinal": (_fit_hgb, dict(cat_mode="ordinal")),
    "h_recent": (_fit_hgb, dict(stop_on="recent", max_iter=2400)),
    "h_seed1": (_fit_hgb, dict(seed=1)), "h_seed2": (_fit_hgb, dict(seed=2)),
    "h_default": (_fit_hgb, dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, early_stopping="auto",
                                 loss="squared_error", l2_regularization=0.0)),
}


def run_learners(campaign: Campaign, names: list[str], threads: int, all_folds: bool = False) -> None:
    """Late folds by default; all ten to confirm a setting the late folds singled out."""
    selection = campaign.selection()
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(selection.height)))
    log = _review_dir(campaign) / ("learner_results_all_folds.jsonl" if all_folds else "learner_results.jsonl")
    for name in names:
        fit, over = LEARNER_SETTINGS[name]
        folds, started = {}, time.time()
        for k in range(1, N_SPLITS + 1) if all_folds else LATE_FOLDS:
            train_idx, val_idx = splits[k - 1]
            train = selection[train_idx.min(): train_idx.max() + 1]
            val = selection[val_idx.min(): val_idx.max() + 1]
            prediction, info = fit(campaign, train, val, threads, **dict(over))
            folds[str(k)] = {**_fold_metrics(val["log_psf"].to_numpy(), prediction), **info}
        row = {"name": name, "folds": folds, "seconds": round(time.time() - started),
               "medape": float(np.mean([f["medape"] for f in folds.values()])),
               "p90": float(np.mean([f["p90_ape"] for f in folds.values()])),
               "flag": float(np.mean([f["flag_prop"] for f in folds.values()]))}
        with log.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"== {name}: MedAPE {row['medape']:.3%} P90 {row['p90']:.2%} flag {row['flag']:.2%} | "
              f"{row['seconds']}s | per fold " + " ".join(f"{f['medape']:.3%}" for f in folds.values()), flush=True)


def score_learners(campaign: Campaign) -> None:
    rows = {}
    for line in (_review_dir(campaign) / "learner_results.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["name"]] = row
    for title, prefix in (("XGBoost", "x_"), ("LightGBM", "l_"), ("HistGradientBoosting (shipped)", "h_")):
        family = {n: r for n, r in rows.items() if n.startswith(prefix)}
        if not family:
            continue
        print(f"\n### {title}, folds {LATE_FOLDS}")
        print(f"{'setting':<18}{'as stopped':>11}{'trees':>16}   MedAPE had training stopped at each tree count")
        for name, row in sorted(family.items(), key=lambda kv: kv[1]["medape"]):
            counts = sorted({int(k) for f in row["folds"].values() for k in f["curve"]})
            full = [k for k in counts if all(str(k) in f["curve"] for f in row["folds"].values())]
            curve = "  ".join(f"{k}:{np.mean([f['curve'][str(k)][0] for f in row['folds'].values()]):.2%}" for k in full)
            trees = "/".join(str(f["n_iter"]) for f in row["folds"].values())
            print(f"{name:<18}{row['medape']:>11.3%}{trees:>16}   {curve}")


# ---------------------------------------------------------------------------
# oracle: a ceiling for any price-level forecast used as a feature
# ---------------------------------------------------------------------------

def _leave_one_out_median(x: np.ndarray) -> np.ndarray:
    """Median of the other values, for each value."""
    order = np.argsort(x, kind="stable")
    xs, out, m = x[order], np.empty(x.size), x.size - 1
    for rank in range(x.size):
        pick = lambda k: xs[k + 1] if rank <= k else xs[k]
        out[order[rank]] = pick(m // 2) if m % 2 else 0.5 * (pick(m // 2 - 1) + pick(m // 2))
    return out


def run_oracle(campaign: Campaign, level: str) -> None:
    """Shipped model plus the realised level of the sale's own segment and week.

    Deliberate leakage (it uses the same week's other sales): no forecast made
    from earlier weeks can know more about the segment's level than this.
    """
    comp = "area_rooms_comp_psf" if level == "area_rooms" else "project_comp_psf"
    frame = campaign.selection().with_columns(
        segment_expr(level).alias("_seg"), pl.col("date").dt.truncate("1w").alias("_week")
    ).with_row_index("_row")
    oracle = np.full(frame.height, np.nan)
    groups = frame.filter(pl.col("_seg").is_not_null()).group_by("_seg", "_week").agg("_row", "log_psf")
    for rows, values in groups.select("_row", "log_psf").iter_rows():
        if len(rows) > 3:
            oracle[np.asarray(rows)] = _leave_one_out_median(np.asarray(values, dtype=float))
    frame = frame.with_columns(pl.Series("oracle", oracle).fill_nan(None)).with_columns(
        (pl.col("oracle") - pl.col(comp).log()).alias("oracle_gap")
    )
    folds = {"hgb": [], "hgb+oracle": []}
    for train_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(frame.height)):
        train, val = frame[train_idx.min(): train_idx.max() + 1], frame[val_idx.min(): val_idx.max() + 1]
        truth = val["log_psf"].to_numpy()
        folds["hgb"].append(_fold_metrics(truth, fit_hgb(campaign, train, val)[0])["medape"])
        folds["hgb+oracle"].append(
            _fold_metrics(truth, fit_hgb(campaign, train, val, extra=["oracle", "oracle_gap"])[0])["medape"]
        )
        print(f"  fold {len(folds['hgb'])}: shipped {folds['hgb'][-1]:.3%} | with the realised level "
              f"{folds['hgb+oracle'][-1]:.3%}", flush=True)
    print(f"== oracle {level} (covers {frame['oracle'].is_not_null().mean():.1%} of sales): shipped "
          f"{np.mean(folds['hgb']):.3%} -> {np.mean(folds['hgb+oracle']):.3%}")


# ---------------------------------------------------------------------------
# export-forecast, intervals
# ---------------------------------------------------------------------------

def export_forecast(campaign: Campaign, name: str) -> None:
    """Write a forecast run as ``fc_area_rooms_<name>.parquet`` for the campaign runner's ``cv``.

    Run the setting with ``--every 1`` first: a sale needs the forecast of its own week.
    """
    panel = Panel(campaign)
    stored = np.load(_review_dir(campaign) / f"fc_{name}.npz")
    fc, width = stored["fc"][:, :, 0], stored["width"]
    seg_i, week_i = np.nonzero(np.isfinite(fc))
    pl.DataFrame({
        "seg": [panel.segs[i] for i in seg_i], "week": [panel.weeks[j] for j in week_i],
        "forecast": fc[seg_i, week_i], "width": width[seg_i, week_i], "actual": panel.Y3[seg_i, week_i],
    }).write_parquet(campaign.path(f"fc_area_rooms_{name}.parquet"))
    print(f"{name}: {seg_i.size:,} forecasts over {len(set(week_i.tolist()))} weeks -> "
          f"optimize_model_family cv hgb@area_rooms:{name}")


def _day_bootstrap(a: np.ndarray, b: np.ndarray, days: np.ndarray, reps: int = 1000) -> tuple[float, float, float]:
    """median(a) - median(b) in pp, 95% interval from resampling whole days.

    Sales on one day share the market move and often the launch they belong to.
    """
    _, day_of = np.unique(days, return_inverse=True)
    order = np.argsort(day_of, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(day_of))])
    blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng = np.random.default_rng(9)
    draws = []
    for _ in range(reps):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(np.median(a[idx]) - np.median(b[idx]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return 100 * float(np.median(a) - np.median(b)), 100 * float(low), 100 * float(high)


def run_intervals(campaign: Campaign, names: list[str]) -> None:
    """Intervals for the campaign's comparisons, from the predictions its steps saved."""
    truth, fold = np.load(campaign.path("oof_y.npy")), np.load(campaign.path("oof_fold.npy"))
    days = campaign.selection()["date"].to_numpy()
    shipped = np.abs(np.expm1(truth - np.load(campaign.path("oof_hgb.npy"))))
    print("Selection CV, gain over the shipped model in pp of MedAPE (positive = better)")
    for name in names:
        tag = name.replace("@", "_at_").replace(":", "_")
        other = np.abs(np.expm1(truth - np.load(campaign.path(f"oof_{tag}.npy"))))
        ok = np.isfinite(shipped) & np.isfinite(other)
        per_fold = [100 * (np.median(shipped[ok & (fold == k)]) - np.median(other[ok & (fold == k)]))
                    for k in range(1, N_SPLITS + 1)]
        gain, low, high = _day_bootstrap(shipped[ok], other[ok], days[ok], reps=500)
        print(f"  {name:<30} mean of folds {np.mean(per_fold):+.3f} | better on {sum(g > 0 for g in per_fold)}/10 | "
              f"pooled {gain:+.3f} ({low:+.3f} .. {high:+.3f})")
    cadence = Path(campaign.path("cadence_predictions.npy"))
    if not cadence.exists():
        print("no cadence_predictions.npy: run optimize_model_family cadence for the retraining intervals")
        return
    feats = campaign.features()
    test = feats.filter(pl.col("date").is_between(TEST_START, TEST_END))
    truth, day_list = test["log_psf"].to_numpy(), test["date"].to_list()
    predictions = np.load(cadence)                           # one column per weekly cut-off
    cutoffs = [TEST_START - timedelta(days=1) + timedelta(days=7 * i) for i in range(predictions.shape[1])]

    def at_cadence(step: int | None) -> np.ndarray:
        out = np.empty(len(truth))
        for i, day in enumerate(day_list):
            usable = [j for j, c in enumerate(cutoffs) if c < day and (step is None or (c - cutoffs[0]).days % step == 0)]
            out[i] = predictions[i, 0 if step is None else usable[-1]]
        return np.abs(np.expm1(truth - out))

    print("\nRetraining cadence on the test days (positive = the second is better)")
    for label, first, second in (("trained once -> every 28 days", None, 28), ("trained once -> every 7 days", None, 7),
                                 ("every 28 days -> every 7 days", 28, 7)):
        gain, low, high = _day_bootstrap(at_cadence(first), at_cadence(second), test["date"].to_numpy())
        print(f"  {label:<32} {gain:+.3f} ({low:+.3f} .. {high:+.3f})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["forecast", "xgb-forecast", "finetune", "score-forecasts", "learners",
                                         "score-learners", "oracle", "export-forecast", "intervals"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--every", type=int, default=2, help="forecast every k-th origin week")
    parser.add_argument("--through-test", action="store_true", help="forecast the test weeks as well")
    parser.add_argument("--all-folds", action="store_true", help="learners: all ten folds, not the late three")
    parser.add_argument("--list", action="store_true", help="print the setting names of the step")
    args = parser.parse_args()
    if args.list:
        print(" ".join(FORECAST_SETTINGS if args.step == "forecast" else LEARNER_SETTINGS))
        return 0
    campaign = Campaign(args.workdir)
    if args.step == "forecast":
        run_forecast(campaign, args.names, args.every, args.through_test, args.threads)
    elif args.step == "xgb-forecast":
        run_xgb_forecast(campaign, int(args.names[0]), int(args.names[1]) if len(args.names) > 1 else 0, args.threads)
    elif args.step == "finetune":
        run_finetune(campaign, float(args.names[0]), int(args.names[1]), args.threads)
    elif args.step == "score-forecasts":
        score_forecasts(campaign)
    elif args.step == "learners":
        run_learners(campaign, args.names, args.threads, args.all_folds)
    elif args.step == "export-forecast":
        export_forecast(campaign, args.names[0])
    elif args.step == "intervals":
        run_intervals(campaign, args.names)
    elif args.step == "score-learners":
        score_learners(campaign)
    else:
        run_oracle(campaign, args.names[0] if args.names else "area_rooms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
