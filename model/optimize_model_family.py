"""Campaign 5: other learners, blends, forecast features and retraining cadence.

Asks whether the shipped HistGradientBoosting model can be beaten by another
learner (LightGBM, XGBoost, CatBoost), by an anchored target, by a blend, or by
adding zero-shot time-series forecasts (Chronos, TimesFM) as features — and
what retraining more often is worth.

Protocol. Selection uses sales through ``SELECTION_END`` only, on the same
date-ordered ``TimeSeriesSplit(10)`` as ``cross_validate``; every learner's
out-of-fold predictions are kept so blends are judged on rows neither their
members nor their weights have seen. The test window that follows
(``TEST_START``..``TEST_END``) is read by ``cadence`` and ``holdout`` only.
The acceptance gate is campaign 2's: MedAPE down >= 0.05 pp with no tail
worsening; a blend must also beat its best member by 0.10 pp.

Extra packages, none of them app requirements: ``lightgbm``, ``xgboost``,
``catboost`` for the learners; ``torch`` (CPU build), ``chronos-forecasting``
and ``timesfm`` for ``forecast``. Working files go to ``--workdir``.

Usage:
    python -m model.optimize_model_family prep
    python -m model.optimize_model_family cv hgb hgb_anchored lgbm xgb cat
    python -m model.optimize_model_family blend hgb lgbm hgb_anchored
    python -m model.optimize_model_family age hgb
    python -m model.optimize_model_family forecast area_rooms ewma chronos2 timesfm3
    python -m model.optimize_model_family cv hgb@area_rooms:chronos2
    python -m model.optimize_model_family cadence
    python -m model.optimize_model_family holdout hgb hgb+lgbm+hgb_anchored
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from model.fair_value_model import (
    _fold_metrics,
    _make_model,
    feature_columns,
    feature_engineering,
    fit_encoders,
    load_shipping_config,
    reference_needed,
    summarize_folds,
    to_matrix,
    trim_psf,
)

SELECTION_END = date(2026, 9, 3)
TEST_START, TEST_END = date(2026, 9, 4), date(2026, 11, 12)
N_SPLITS = 10
SEED = 42
DEFAULT_WORKDIR = ".tmp/campaign5"

# Trailing comparables, most specific first: the anchored target predicts the
# deviation from the first one a sale has.
ANCHORS = [
    "project_rooms_comp_psf", "project_comp_psf", "project_hist_psf",
    "building_hist_psf", "area_rooms_comp_psf",
]
ROOM_BANDS = ["Studio", "1 B/R", "2 B/R", "3 B/R"]
SIMPLE_FORECASTS = ("naive", "mean4", "ewma")
# A week with fewer sales than this is treated as unobserved in its series.
MIN_WEEK_SALES = {"area_rooms": 3, "project": 2}
MIN_CONTEXT_WEEKS = 8
FIRST_ORIGIN_WEEK = 12
BLEND_FIRST_FOLD = 3   # folds 1-2 only supply rows for fitting blend weights


class Campaign:
    """Working directory plus the shipped configuration every step shares."""

    def __init__(self, workdir: str):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.params = load_shipping_config()
        self.numeric, self.categorical = feature_columns(self.cfg)

    def path(self, name: str) -> str:
        return str(self.dir / name)

    def features(self) -> pl.DataFrame:
        return pl.read_parquet(self.path("feats.parquet")).sort(
            "date", "TRANSACTION_NUMBER", maintain_order=True
        )

    def selection(self) -> pl.DataFrame:
        return trim_psf(self.features().filter(pl.col("date") <= SELECTION_END)).sort(
            "date", "TRANSACTION_NUMBER", maintain_order=True
        )


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

def prep(campaign: Campaign) -> None:
    """Build the shipped-config feature frame once from the stored snapshot."""
    from ingestion.dda_api import normalize_dld_transactions
    from ingestion.gcs_storage import (
        configured_snapshot,
        load_local_secrets,
        read_parquet_object,
        read_reference_frames,
    )
    from ingestion.store_dld_transactions_gcs import dedupe_snapshot

    secrets = load_local_secrets()
    bucket_name, object_name = configured_snapshot(secrets)
    stored, _ = read_parquet_object(secrets, bucket_name, object_name)
    raw = dedupe_snapshot(normalize_dld_transactions(stored))
    names = reference_needed(campaign.cfg)
    reference = read_reference_frames(secrets, names) if names else None
    feats = feature_engineering(raw, campaign.cfg, reference=reference)
    feats.write_parquet(campaign.path("feats.parquet"))
    selection = campaign.selection()
    print(
        f"{stored.height:,} stored rows -> {raw.height:,} transactions -> {feats.height:,} "
        f"scoreable sales ({feats['date'].min()}..{feats['date'].max()}); selection "
        f"{selection.height:,} rows through {SELECTION_END}"
    )


# ---------------------------------------------------------------------------
# learners: (train frame, validation frame) -> validation log-PSF predictions
# ---------------------------------------------------------------------------

def _early_stop_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random 90/10 split of a training fold, as HGB's own early stopping does."""
    mask = np.random.default_rng(SEED).random(n) < 0.1
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def anchor_log(campaign: Campaign, frame: pl.DataFrame, train: pl.DataFrame) -> np.ndarray:
    """log of each row's most specific trailing comparable price.

    The few rows with no comparable at all fall back to the training median.
    """
    fallback = float(train["psf"].median())
    expr = pl.coalesce([pl.col(c) for c in ANCHORS if c in frame.columns] + [pl.lit(fallback)])
    return frame.select(expr.log()).to_series().to_numpy()


def fit_hgb(
    campaign: Campaign,
    train: pl.DataFrame,
    val: pl.DataFrame,
    params: dict | None = None,
    anchored: bool = False,
    extra: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    encoders = fit_encoders(train, campaign.cfg)
    X_tr, y_tr, _, cat_idx = to_matrix(train, encoders, campaign.cfg)
    X_va, _, _, _ = to_matrix(val, encoders, campaign.cfg)
    if extra:
        E_tr, E_va = train.select(extra).to_numpy(), val.select(extra).to_numpy()
        # sklearn's binning fails on a column with no observed value (the first
        # fold predates the forecasts): hand it a constant, which no split can use.
        empty = ~np.isfinite(E_tr).any(axis=0)
        E_tr[:, empty], E_va[:, empty] = 0.0, 0.0
        X_tr, X_va = np.column_stack([X_tr, E_tr]), np.column_stack([X_va, E_va])
    offset_tr = offset_va = 0.0
    if anchored:
        offset_tr = anchor_log(campaign, train, train)
        offset_va = anchor_log(campaign, val, train)
        X_tr = np.column_stack([X_tr, offset_tr])
        X_va = np.column_stack([X_va, offset_va])
    model = _make_model({**campaign.params, **(params or {})}, cat_idx, SEED)
    model.fit(X_tr, y_tr - offset_tr)
    return model.predict(X_va) + offset_va, {"n_iter": int(model.n_iter_)}


def fit_lgbm(campaign: Campaign, train: pl.DataFrame, val: pl.DataFrame) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    encoders = fit_encoders(train, campaign.cfg)
    X_tr, y_tr, _, cat_idx = to_matrix(train, encoders, campaign.cfg)
    X_va, _, _, _ = to_matrix(val, encoders, campaign.cfg)
    fit_idx, stop_idx = _early_stop_split(len(y_tr))
    model = lgb.LGBMRegressor(
        objective="l1", learning_rate=0.04, n_estimators=3000, num_leaves=127,
        min_child_samples=20, reg_lambda=1.0, max_bin=255, n_jobs=-1,
        random_state=SEED, verbose=-1,
    )
    model.fit(
        X_tr[fit_idx], y_tr[fit_idx],
        eval_set=[(X_tr[stop_idx], y_tr[stop_idx])], eval_metric="l1",
        categorical_feature=cat_idx, callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model.predict(X_va), {"n_iter": int(model.best_iteration_ or model.n_estimators)}


def fit_xgb(campaign: Campaign, train: pl.DataFrame, val: pl.DataFrame) -> tuple[np.ndarray, dict]:
    import xgboost

    encoders = fit_encoders(train, campaign.cfg)
    X_tr, y_tr, _, cat_idx = to_matrix(train, encoders, campaign.cfg)
    X_va, _, _, _ = to_matrix(val, encoders, campaign.cfg)
    categorical = set(cat_idx)
    fit_idx, stop_idx = _early_stop_split(len(y_tr))
    model = xgboost.XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist", grow_policy="lossguide",
        max_leaves=127, max_depth=0, learning_rate=0.04, n_estimators=3000,
        reg_lambda=1.0, max_bin=256, enable_categorical=True, max_cat_to_onehot=1,
        feature_types=["c" if i in categorical else "q" for i in range(X_tr.shape[1])],
        early_stopping_rounds=100, n_jobs=-1, random_state=SEED,
    )
    model.fit(X_tr[fit_idx], y_tr[fit_idx], eval_set=[(X_tr[stop_idx], y_tr[stop_idx])], verbose=False)
    return model.predict(X_va), {"n_iter": int(model.best_iteration + 1)}


def fit_cat(campaign: Campaign, train: pl.DataFrame, val: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """CatBoost on the raw category strings (ordered target statistics, no top-200 cap)."""
    from catboost import CatBoostRegressor, FeaturesData, Pool

    def pool(frame: pl.DataFrame, idx: np.ndarray | None = None) -> Pool:
        num = frame.select(pl.col(c).cast(pl.Float32) for c in campaign.numeric).to_numpy()
        cats = (
            frame.select(pl.col(c).cast(pl.Utf8).fill_null("?") for c in campaign.categorical)
            .to_numpy().astype(object)
        )
        target = frame["log_psf"].to_numpy()
        if idx is not None:
            num, cats, target = num[idx], cats[idx], target[idx]
        data = FeaturesData(
            num_feature_data=np.ascontiguousarray(num), cat_feature_data=cats,
            num_feature_names=campaign.numeric, cat_feature_names=campaign.categorical,
        )
        return Pool(data, label=target)

    fit_idx, stop_idx = _early_stop_split(train.height)
    model = CatBoostRegressor(
        loss_function="MAE", learning_rate=0.1, iterations=1500, depth=8, l2_leaf_reg=3.0,
        border_count=128, od_type="Iter", od_wait=100, random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )
    model.fit(pool(train, fit_idx), eval_set=pool(train, stop_idx))
    return model.predict(pool(val)), {"n_iter": int(model.get_best_iteration() or 0) + 1}


def forecast_columns(spec: str) -> list[str]:
    """Feature columns for a forecast spec ``level:model``."""
    level, model = spec.split(":")
    columns = [f"fc_{level}_{model}", f"fg_{level}_{model}"]
    return columns if model in SIMPLE_FORECASTS else columns + [f"fw_{level}_{model}"]


def learner(campaign: Campaign, name: str):
    """Resolve a learner name; ``hgb@level:model`` adds that forecast as features."""
    if "@" in name:
        base, spec = name.split("@")
        if base != "hgb":
            raise ValueError("forecast features are only wired to the shipped hgb learner")
        return lambda tr, va: fit_hgb(campaign, tr, va, extra=forecast_columns(spec))
    table = {
        "hgb": lambda tr, va: fit_hgb(campaign, tr, va),
        "hgb_iter2400": lambda tr, va: fit_hgb(campaign, tr, va, params={"max_iter": 2400}),
        "hgb_anchored": lambda tr, va: fit_hgb(campaign, tr, va, anchored=True),
        "lgbm": lambda tr, va: fit_lgbm(campaign, tr, va),
        "xgb": lambda tr, va: fit_xgb(campaign, tr, va),
        "cat": lambda tr, va: fit_cat(campaign, tr, va),
    }
    return table[name]


def segment_expr(level: str) -> pl.Expr:
    if level == "project":
        return pl.col("PROJECT_EN")
    band = pl.when(pl.col("ROOMS_EN").is_in(ROOM_BANDS)).then(pl.col("ROOMS_EN")).otherwise(pl.lit("4+"))
    return pl.concat_str([pl.col("AREA_EN"), band], separator="|")


def with_forecasts(campaign: Campaign, frame: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Join, for each ``hgb@level:model`` learner, the forecast of the sale's own week."""
    for spec in sorted({name.split("@")[1] for name in names if "@" in name}):
        level, model = spec.split(":")
        comp = "area_rooms_comp_psf" if level == "area_rooms" else "project_comp_psf"
        forecast = pl.read_parquet(campaign.path(f"fc_{level}_{model}.parquet")).select(
            pl.col("seg").alias("_seg"), pl.col("week").alias("_week"),
            pl.col("forecast").cast(pl.Float64).alias(f"fc_{level}_{model}"),
            pl.col("width").cast(pl.Float64).alias(f"fw_{level}_{model}"),
        )
        frame = (
            frame.with_columns(segment_expr(level).alias("_seg"), pl.col("date").dt.truncate("1w").alias("_week"))
            .join(forecast, on=["_seg", "_week"], how="left")
            .with_columns((pl.col(f"fc_{level}_{model}") - pl.col(comp).log()).alias(f"fg_{level}_{model}"))
            .drop("_seg", "_week")
        )
    return frame


# ---------------------------------------------------------------------------
# cv
# ---------------------------------------------------------------------------

def run_cv(campaign: Campaign, names: list[str]) -> None:
    selection = with_forecasts(campaign, campaign.selection(), names)
    np.save(campaign.path("oof_y.npy"), selection["log_psf"].to_numpy())
    for name in names:
        fit = learner(campaign, name)
        oof = np.full(selection.height, np.nan)
        fold_of = np.zeros(selection.height, dtype=np.int8)
        folds = []
        started = time.time()
        splits = TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(selection.height))
        for k, (train_idx, val_idx) in enumerate(splits, start=1):
            train = selection[train_idx.min(): train_idx.max() + 1]
            val = selection[val_idx.min(): val_idx.max() + 1]
            prediction, info = fit(train, val)
            oof[val_idx], fold_of[val_idx] = prediction, k
            folds.append(_fold_metrics(val["log_psf"].to_numpy(), prediction))
            print(
                f"  {name} fold {k:>2}: train to {train['date'].max()} | MedAPE "
                f"{folds[-1]['medape']:.3%} | {info} | {time.time() - started:.0f}s", flush=True,
            )
        summary = summarize_folds(folds, n_rows=selection.height, n_splits=N_SPLITS)
        summary.update(name=name, seconds=round(time.time() - started))
        tag = name.replace("@", "_at_").replace(":", "_")
        np.save(campaign.path(f"oof_{tag}.npy"), oof)
        np.save(campaign.path("oof_fold.npy"), fold_of)
        Path(campaign.path(f"cv_{tag}.json")).write_text(json.dumps(summary, indent=1))
        print(
            f"== {name}: MedAPE {summary['medape_mean']:.3%} ± {summary['medape_std']:.2%} | "
            f"P90 {summary['p90_ape_mean']:.2%} | flag {summary['flag_prop_mean']:.2%} | "
            f"R2 {summary['r2_mean']:.3f} | {summary['seconds']}s", flush=True,
        )


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------

def blend_weights(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise mean absolute log error."""
    from scipy.optimize import minimize

    if len(target) > 60_000:
        idx = np.random.default_rng(0).choice(len(target), 60_000, replace=False)
        predictions, target = predictions[idx], target[idx]

    def loss(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        return float(np.mean(np.abs(predictions @ (w / w.sum()) - target)))

    # Start from a unit-sized simplex: scipy's default one around equal weights
    # is so small that the search stops before it has moved.
    k = predictions.shape[1]
    simplex = np.vstack([np.zeros(k), np.eye(k)])
    best = minimize(loss, np.zeros(k), method="Nelder-Mead",
                    options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-9, "maxiter": 4000})
    w = np.exp(best.x - best.x.max())
    return w / w.sum()


def run_blend(campaign: Campaign, names: list[str]) -> None:
    """Every blend of the named learners, on out-of-fold rows only.

    Weights for fold k are fitted on the out-of-fold rows of earlier folds.
    """
    y = np.load(campaign.path("oof_y.npy"))
    fold = np.load(campaign.path("oof_fold.npy"))
    P = np.column_stack([np.load(campaign.path(f"oof_{n}.npy")) for n in names])
    folds = [k for k in sorted(set(fold)) if k >= BLEND_FIRST_FOLD]

    def mean_of(per_fold: list[dict]) -> dict:
        return {key: float(np.mean([f[key] for f in per_fold])) for key in ("medape", "p90_ape", "flag_prop", "r2")}

    rows = {name: mean_of([_fold_metrics(y[fold == k], P[fold == k, j]) for k in folds]) for j, name in enumerate(names)}
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), size):
            label = " + ".join(names[j] for j in combo)
            equal, fitted = [], []
            for k in folds:
                past, now = (fold > 0) & (fold < k), fold == k
                w = blend_weights(P[past][:, combo], y[past])
                equal.append(_fold_metrics(y[now], P[now][:, combo].mean(axis=1)))
                fitted.append(_fold_metrics(y[now], P[now][:, combo] @ w))
            rows[f"mean({label})"] = mean_of(equal)
            rows[f"fitted({label})"] = mean_of(fitted)

    base = rows[names[0]]["medape"]
    print(f"folds {folds[0]}-{folds[-1]}, {int((fold >= BLEND_FIRST_FOLD).sum()):,} out-of-fold rows; gain is versus {names[0]}")
    print(f"{'model':<54}{'MedAPE':>9}{'gain pp':>9}{'P90':>9}{'flag':>8}")
    for label, m in sorted(rows.items(), key=lambda item: item[1]["medape"]):
        print(f"{label:<54}{m['medape']:>9.3%}{(base - m['medape']) * 100:>+9.3f}{m['p90_ape']:>9.2%}{m['flag_prop']:>8.2%}")
    Path(campaign.path("blend_results.json")).write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# age
# ---------------------------------------------------------------------------

def run_age(campaign: Campaign, name: str) -> None:
    """Out-of-fold error by how long ago the scoring model's training data ended."""
    selection = campaign.selection()
    frame = selection.select("date", "log_psf").with_columns(
        pl.Series("fold", np.load(campaign.path("oof_fold.npy"))),
        pl.Series("pred", np.load(campaign.path(f"oof_{name}.npy"))),
    ).filter(pl.col("fold") >= BLEND_FIRST_FOLD).with_columns(
        (pl.col("date") - pl.col("date").min().over("fold")).dt.total_days().alias("age"),
        (pl.col("log_psf") - pl.col("pred")).exp().sub(1).abs().alias("ape"),
    )
    table = (
        frame.group_by((pl.col("age") // 14 * 14).alias("from_day"))
        .agg(pl.len().alias("n"), pl.col("fold").n_unique().alias("folds"), pl.col("ape").median().alias("medape"))
        .sort("from_day")
    )
    print(f"{name}: out-of-fold MedAPE by model age (days since its training data ended)")
    for row in table.iter_rows(named=True):
        if row["n"] >= 3000:
            print(f"  days {row['from_day']:>3}-{row['from_day'] + 13:<3} | {row['n']:>6,} sales in {row['folds']} folds | {row['medape']:.2%}")


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def weekly_panel(campaign: Campaign, level: str) -> tuple[list[str], list[date], np.ndarray]:
    """(segments, weeks, log weekly median PSF matrix with NaN where unobserved)."""
    sales = campaign.features().with_columns(
        segment_expr(level).alias("seg"), pl.col("date").dt.truncate("1w").alias("week")
    ).filter(pl.col("seg").is_not_null())
    weekly = sales.group_by("seg", "week").agg(pl.col("psf").median().alias("median"), pl.len().alias("n")).filter(
        pl.col("n") >= MIN_WEEK_SALES[level]
    )
    weeks = pl.date_range(sales["week"].min(), sales["week"].max(), interval="1w", eager=True).to_list()
    counts = weekly.group_by("seg").agg(pl.len().alias("observed"))
    segments = sorted(counts.filter(pl.col("observed") >= MIN_CONTEXT_WEEKS + 4)["seg"].to_list())
    row, column = {s: i for i, s in enumerate(segments)}, {w: i for i, w in enumerate(weeks)}
    Y = np.full((len(segments), len(weeks)), np.nan, dtype=np.float32)
    for seg, week, median in weekly.filter(pl.col("seg").is_in(segments)).select("seg", "week", "median").iter_rows():
        Y[row[seg], column[week]] = np.log(median)
    return segments, weeks, Y


def simple_forecast(model: str, context: np.ndarray) -> np.ndarray:
    out = np.full(context.shape[0], np.nan, dtype=np.float32)
    for i, series in enumerate(context):
        seen = series[~np.isnan(series)]
        if not seen.size:
            continue
        if model == "naive":
            out[i] = seen[-1]
        elif model == "mean4":
            out[i] = seen[-4:].mean()
        else:  # ewma, alpha 0.3 over the observed weeks
            level = seen[0]
            for value in seen[1:]:
                level = 0.3 * value + 0.7 * level
            out[i] = level
    return out


def _fill_gaps(series: np.ndarray) -> np.ndarray:
    """Leading gaps dropped, interior gaps linearly filled (TimesFM has no missing-value input)."""
    seen = np.flatnonzero(~np.isnan(series))
    series = series[seen[0]:]
    idx = np.arange(series.size)
    known = ~np.isnan(series)
    return np.interp(idx, idx[known], series[known]).astype(np.float32)


class PretrainedForecaster:
    """Zero-shot one-week-ahead forecasts from a pretrained model, on CPU."""

    def __init__(self, model: str, segments: list[str]):
        import torch

        self.torch, self.model, self.segments = torch, model, segments
        if model in ("bolt_small", "bolt_base"):
            from chronos import ChronosBoltPipeline

            self.pipe = ChronosBoltPipeline.from_pretrained(f"amazon/chronos-{model.replace('_', '-')}", device_map="cpu")
        elif model in ("chronos2", "chronos2_mv"):
            from chronos import Chronos2Pipeline

            self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        elif model == "timesfm3":
            import timesfm

            self.pipe = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
        else:
            raise ValueError(f"unknown forecaster {model}")

    def __call__(self, context: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(median forecast, 10-90% interval width) for the given series rows."""
        median = np.full(len(rows), np.nan, dtype=np.float32)
        width = np.full(len(rows), np.nan, dtype=np.float32)
        levels = [0.1, 0.5, 0.9]
        with self.torch.no_grad():
            if self.model.startswith("bolt"):
                for start in range(0, len(rows), 256):
                    part = slice(start, start + 256)
                    q, _ = self.pipe.predict_quantiles(self.torch.tensor(context[rows[part]]), prediction_length=1, quantile_levels=levels)
                    q = q.numpy()[:, 0, :]
                    median[part], width[part] = q[:, 1], q[:, 2] - q[:, 0]
            elif self.model == "chronos2":
                for start in range(0, len(rows), 256):
                    part = slice(start, start + 256)
                    q, _ = self.pipe.predict_quantiles([context[i] for i in rows[part]], prediction_length=1, quantile_levels=levels)
                    q = np.stack([t.numpy()[0, 0, :] for t in q])
                    median[part], width[part] = q[:, 1], q[:, 2] - q[:, 0]
            elif self.model == "chronos2_mv":
                # One multivariate task per district: its unit types are forecast jointly.
                groups: dict[str, list[int]] = {}
                for position, i in enumerate(rows):
                    groups.setdefault(self.segments[i].split("|")[0], []).append(position)
                keys = list(groups)
                for start in range(0, len(keys), 64):
                    batch = keys[start:start + 64]
                    q, _ = self.pipe.predict_quantiles([context[rows[groups[k]]] for k in batch], prediction_length=1, quantile_levels=levels)
                    for key, tensor in zip(batch, q):
                        values = tensor.numpy()[:, 0, :]
                        median[groups[key]], width[groups[key]] = values[:, 1], values[:, 2] - values[:, 0]
            else:
                outputs = self.pipe.predict_batch([_fill_gaps(context[i]) for i in rows], horizon=1, return_quantiles=True)
                for position, output in enumerate(outputs):
                    median[position] = np.asarray(output.forecast).reshape(-1)[0]
                    if output.quantiles is not None:
                        deciles = np.asarray(output.quantiles).reshape(-1)
                        if deciles.size >= 9:
                            width[position] = deciles[-1] - deciles[0]
        return median, width


def run_forecast(campaign: Campaign, level: str, models: list[str]) -> None:
    """For every week t, forecast each segment's level from weeks < t only."""
    segments, weeks, Y = weekly_panel(campaign, level)
    print(f"{level}: {len(segments)} series x {len(weeks)} weeks, {np.isfinite(Y).mean():.1%} observed", flush=True)
    for model in models:
        started = time.time()
        forecast = np.full(Y.shape, np.nan, dtype=np.float32)
        width = np.full(Y.shape, np.nan, dtype=np.float32)
        runner = None if model in SIMPLE_FORECASTS else PretrainedForecaster(model, segments)
        for t in range(FIRST_ORIGIN_WEEK, len(weeks)):
            context = Y[:, :t]
            rows = np.flatnonzero(np.isfinite(context).sum(axis=1) >= MIN_CONTEXT_WEEKS)
            if not rows.size:
                continue
            if runner is None:
                forecast[rows, t] = simple_forecast(model, context[rows])
            else:
                forecast[rows, t], width[rows, t] = runner(context, rows)
        seg_i, week_i = np.nonzero(np.isfinite(forecast))
        frame = pl.DataFrame({
            "seg": [segments[i] for i in seg_i], "week": [weeks[j] for j in week_i],
            "forecast": forecast[seg_i, week_i], "width": width[seg_i, week_i], "actual": Y[seg_i, week_i],
        })
        frame.write_parquet(campaign.path(f"fc_{level}_{model}.parquet"))
        scored = frame.filter(pl.col("actual").is_not_nan() & (pl.col("week") <= SELECTION_END))
        error = (scored["forecast"] - scored["actual"]).abs()
        print(
            f"== {level} {model}: {frame.height:,} forecasts in {time.time() - started:.0f}s | selection-period "
            f"error vs the realised weekly median: mean {error.mean():.4f}, median {error.median():.4f} log points",
            flush=True,
        )


# ---------------------------------------------------------------------------
# cadence + holdout (the only steps that read the test window)
# ---------------------------------------------------------------------------

def _weekly_cutoffs() -> list[date]:
    return [SELECTION_END + timedelta(days=7 * i) for i in range(10)]


def _test_frame(campaign: Campaign, names: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    feats = with_forecasts(campaign, campaign.features(), names)
    return feats, feats.filter(pl.col("date").is_between(TEST_START, TEST_END))


def _describe(label: str, y: np.ndarray, prediction: np.ndarray, inside: np.ndarray) -> dict:
    full, trimmed = _fold_metrics(y, prediction), _fold_metrics(y[inside], prediction[inside])
    print(
        f"{label:<46} MedAPE {full['medape']:.3%} P90 {full['p90_ape']:.2%} flag {full['flag_prop']:.2%} "
        f"R2 {full['r2']:.3f} | inside the training price range {trimmed['medape']:.3%}", flush=True,
    )
    return {"label": label, "all": full, "trimmed": trimmed}


def run_cadence(campaign: Campaign, name: str = "hgb") -> None:
    """Refit at weekly cut-offs; a cadence uses the newest model it allows."""
    feats, test = _test_frame(campaign, [name])
    y, days = test["log_psf"].to_numpy(), test["date"].to_list()
    cutoffs = _weekly_cutoffs()
    fit = learner(campaign, name)
    predictions = {}
    for cutoff in cutoffs:
        predictions[cutoff], _ = fit(trim_psf(feats.filter(pl.col("date") <= cutoff)), test)
        print(f"  fitted on sales to {cutoff}", flush=True)
    np.save(campaign.path("cadence_predictions.npy"), np.column_stack([predictions[c] for c in cutoffs]))
    first = trim_psf(feats.filter(pl.col("date") <= cutoffs[0]))
    inside = test["psf"].is_between(float(first["psf"].min()), float(first["psf"].max())).to_numpy()

    def at_cadence(step: int | None) -> np.ndarray:
        out = np.empty(len(y))
        for i, day in enumerate(days):
            usable = [c for c in cutoffs if c < day and (step is None or (c - cutoffs[0]).days % step == 0)]
            out[i] = predictions[cutoffs[0] if step is None else usable[-1]][i]
        return out

    rows = [
        _describe(label, y, at_cadence(step), inside)
        for label, step in (
            (f"{name}: trained once, on sales to {cutoffs[0]}", None),
            (f"{name}: retrained every 28 days", 28),
            (f"{name}: retrained every 14 days", 14),
            (f"{name}: retrained every 7 days", 7),
        )
    ]
    Path(campaign.path("cadence_results.json")).write_text(json.dumps(rows, indent=1))


def run_holdout(campaign: Campaign, names: list[str]) -> None:
    """One pass over the test window; ``a+b`` is the equal-weight blend of learners a and b."""
    members = sorted({member for name in names for member in name.split("+")})
    feats, test = _test_frame(campaign, members)
    y = test["log_psf"].to_numpy()
    train = trim_psf(feats.filter(pl.col("date") <= SELECTION_END))
    inside = test["psf"].is_between(float(train["psf"].min()), float(train["psf"].max())).to_numpy()
    predictions = {}
    for member in members:
        predictions[member], _ = learner(campaign, member)(train, test)
        print(f"  {member} fitted on {train.height:,} sales to {SELECTION_END}", flush=True)
    print(f"test window {TEST_START}..{TEST_END}: {len(y):,} sales")
    rows, blended = [], {}
    for name in names:
        blended[name] = np.mean([predictions[m] for m in name.split("+")], axis=0)
        rows.append(_describe(name, y, blended[name], inside))
    base_error = np.abs(np.expm1(y - blended[names[0]]))
    rng = np.random.default_rng(11)
    for name in names[1:]:
        error = np.abs(np.expm1(y - blended[name]))
        gains = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), len(y))
            gains.append(np.median(base_error[idx]) - np.median(error[idx]))
        lo, hi = np.percentile(gains, [2.5, 97.5])
        print(
            f"paired gain of {name} over {names[0]}: {(np.median(base_error) - np.median(error)) * 100:+.3f} pp "
            f"(95% CI {lo * 100:+.3f} .. {hi * 100:+.3f})"
        )
    Path(campaign.path("holdout_results.json")).write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prep", "cv", "blend", "age", "forecast", "cadence", "holdout"])
    parser.add_argument("names", nargs="*", help="learners, or for forecast: the level then the models")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    args = parser.parse_args()
    campaign = Campaign(args.workdir)
    if args.step == "prep":
        prep(campaign)
    elif args.step == "cv":
        run_cv(campaign, args.names)
    elif args.step == "blend":
        run_blend(campaign, args.names)
    elif args.step == "age":
        run_age(campaign, args.names[0] if args.names else "hgb")
    elif args.step == "forecast":
        run_forecast(campaign, args.names[0], args.names[1:])
    elif args.step == "cadence":
        run_cadence(campaign, args.names[0] if args.names else "hgb")
    else:
        run_holdout(campaign, args.names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
