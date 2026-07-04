"""Fair-value model for Dubai apartment transactions.

Predicts the fair value of each closed DLD apartment sale from hedonic
features (size, location, project, rooms, off-plan status, time, ...) and
scores the spread between the actual transaction price and the predicted
fair value. Large negative spreads, corroborated by distress signals
(court/forced-sale procedures, deep discounts, illiquid projects, multiple
sellers), flag potential distressed assets.

Pure Polars + scikit-learn: no Streamlit imports, so the module can be
trained, cross-validated, and smoke-tested offline.

Target: log(price per sqft). Log residuals are multiplicative, so
``spread_pct = actual / fair_value - 1`` is comparable across ticket sizes.
Validation: date-ordered ``TimeSeriesSplit(n_splits=10)`` — every fold
trains on the past and validates on the next time block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from dashboard_constants import DISTRICT_TIER, SQM_TO_SQFT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Feature groups the optimization loop can toggle independently.
DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
    "project": True,        # PROJECT_EN / MASTER_PROJECT_EN categoricals
    "building": False,      # BUILDING_NAME_EN categorical (needs 24-month snapshot)
    "amenity": True,        # nearest metro/mall/landmark, parking, buyer/seller counts
    "comps_area": False,    # trailing 30-day area median PSF (strictly past-only)
    "comps_project": False, # trailing 60-day project median PSF (strictly past-only)
}

DEFAULT_MODEL_PARAMS: dict = {
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

CORE_NUMERIC = ["log_sqft", "days_since_start", "month", "rooms_ord"]
AMENITY_NUMERIC = ["parking_count", "total_buyer", "total_seller"]
CORE_CATEGORICAL = ["AREA_EN", "IS_OFFPLAN_EN", "tier"]
PROJECT_CATEGORICAL = ["PROJECT_EN", "MASTER_PROJECT_EN"]
AMENITY_CATEGORICAL = ["NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN"]

MAX_CATEGORIES = 200  # top-N per categorical; the rest map to OTHER
UNSEEN_CODE = 0       # shared code for OTHER / UNKNOWN / unseen values

# PSF rows outside these quantiles are treated as data-entry noise.
PSF_TRIM_QUANTILES = (0.005, 0.995)

# Case-insensitive PROCEDURE_EN patterns suggesting a forced/distressed sale.
# The live DLD vocabulary should be checked via the value-counts view in the UI.
DISTRESS_PROCEDURE_PATTERN = r"(?i)court|forc|foreclos|auction|bankrupt|liquidat|execution"

# Columns carried through feature_engineering untouched, for scoring/display.
PASSTHROUGH_COLUMNS = [
    "TRANSACTION_NUMBER",
    "INSTANCE_DATE",
    "GROUP_EN",
    "PROCEDURE_EN",
    "AREA_EN",
    "PROJECT_EN",
    "MASTER_PROJECT_EN",
    "BUILDING_NAME_EN",
    "ROOMS_EN",
    "IS_OFFPLAN_EN",
    "TRANS_VALUE",
    "ACTUAL_AREA",
]


def feature_columns(feature_config: dict[str, bool] | None = None) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature column names for a config."""
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    numeric = list(CORE_NUMERIC)
    categorical = list(CORE_CATEGORICAL)
    if cfg["project"]:
        categorical += PROJECT_CATEGORICAL
    if cfg["building"]:
        categorical.append("BUILDING_NAME_EN")
    if cfg["amenity"]:
        numeric += AMENITY_NUMERIC
        categorical += AMENITY_CATEGORICAL
    if cfg["comps_area"]:
        numeric.append("area_comp_psf")
    if cfg["comps_project"]:
        numeric.append("project_comp_psf")
    return numeric, categorical


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering(
    raw: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    date_origin: date | None = None,
) -> pl.DataFrame:
    """Filter to scoreable sales and derive model features + target.

    Keeps only apartment Sales rows with positive price and area, trims PSF
    outliers, and returns one row per transaction with the ``log_psf``
    target, model features, and passthrough columns for display.

    ``date_origin`` anchors ``days_since_start``; pass the training origin
    when scoring new data so the trend feature stays aligned.
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}

    df = raw.filter(
        (pl.col("GROUP_EN").cast(pl.Utf8).str.to_lowercase().str.contains("sale"))
        & (pl.col("PROP_SB_TYPE_EN").cast(pl.Utf8).str.to_lowercase() == "flat")
        & (pl.col("TRANS_VALUE").cast(pl.Float64, strict=False) > 0)
        & (pl.col("ACTUAL_AREA").cast(pl.Float64, strict=False) > 0)
    )
    if df.is_empty():
        return df

    df = df.with_columns(
        pl.col("INSTANCE_DATE")
        .cast(pl.Utf8)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias("date"),
        (pl.col("TRANS_VALUE") / (pl.col("ACTUAL_AREA") * SQM_TO_SQFT)).alias("psf"),
    ).drop_nulls(["date", "psf"])

    lo, hi = df.select(
        pl.col("psf").quantile(PSF_TRIM_QUANTILES[0]).alias("lo"),
        pl.col("psf").quantile(PSF_TRIM_QUANTILES[1]).alias("hi"),
    ).row(0)
    df = df.filter(pl.col("psf").is_between(lo, hi))
    if df.is_empty():
        return df

    origin = date_origin or df.select(pl.col("date").min()).item()

    rooms = pl.col("ROOMS_EN").cast(pl.Utf8)
    df = df.with_columns(
        pl.col("psf").log().alias("log_psf"),
        (pl.col("ACTUAL_AREA") * SQM_TO_SQFT).log().alias("log_sqft"),
        (pl.col("date") - pl.lit(origin)).dt.total_days().cast(pl.Float64).alias("days_since_start"),
        pl.col("date").dt.month().cast(pl.Float64).alias("month"),
        pl.when(rooms.str.to_lowercase().str.contains("studio"))
        .then(pl.lit(0.0))
        .otherwise(rooms.str.extract(r"(\d+)").cast(pl.Float64, strict=False))
        .alias("rooms_ord"),
        pl.col("AREA_EN").cast(pl.Utf8).replace_strict(DISTRICT_TIER, default="UNKNOWN").alias("tier"),
    )

    if cfg["amenity"]:
        df = df.with_columns(
            pl.col("PARKING").cast(pl.Utf8).str.extract(r"(\d+)").cast(pl.Float64, strict=False).alias("parking_count"),
            pl.col("TOTAL_BUYER").cast(pl.Float64, strict=False).alias("total_buyer"),
            pl.col("TOTAL_SELLER").cast(pl.Float64, strict=False).alias("total_seller"),
        )

    # Trailing comparables: strictly past-only (closed="left" excludes the
    # current day) so they never leak the transaction's own price.
    if cfg["comps_area"] or cfg["comps_project"]:
        df = df.sort("date")
        comps = []
        if cfg["comps_area"]:
            comps.append(
                pl.col("psf")
                .rolling_median_by("date", window_size="30d", closed="left")
                .over("AREA_EN")
                .alias("area_comp_psf")
            )
        if cfg["comps_project"]:
            comps.append(
                pl.col("psf")
                .rolling_median_by("date", window_size="60d", closed="left")
                .over("PROJECT_EN")
                .alias("project_comp_psf")
            )
        df = df.with_columns(comps)

    _, categorical = feature_columns(cfg)
    missing = [c for c in categorical if c not in df.columns]
    if missing:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing)
    df = df.with_columns(
        pl.col(c).cast(pl.Utf8).fill_null("UNKNOWN").alias(c) for c in categorical
    )
    keep = ["date", "psf", "log_psf"]
    numeric, _ = feature_columns(cfg)
    keep += numeric + categorical
    keep += [c for c in PASSTHROUGH_COLUMNS if c in df.columns and c not in keep]
    return df.select(keep).sort("date")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def fit_encoders(
    df: pl.DataFrame, feature_config: dict[str, bool] | None = None
) -> dict[str, dict[str, int]]:
    """Per-categorical value->code maps from training data.

    Code 0 is reserved for OTHER/UNKNOWN/unseen; the most frequent
    ``MAX_CATEGORIES - 1`` values get codes 1..N.
    """
    _, categorical = feature_columns(feature_config)
    encoders: dict[str, dict[str, int]] = {}
    for col in categorical:
        top = (
            df.group_by(col)
            .len()
            .sort(["len", col], descending=[True, False])
            .head(MAX_CATEGORIES - 1)
            .get_column(col)
            .to_list()
        )
        encoders[col] = {value: code for code, value in enumerate(top, start=1)}
    return encoders


def to_matrix(
    df: pl.DataFrame,
    encoders: dict[str, dict[str, int]],
    feature_config: dict[str, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[int]]:
    """(X, y, feature_names, categorical_idx) for scikit-learn."""
    numeric, categorical = feature_columns(feature_config)
    frame = df.with_columns(
        pl.col(col)
        .replace_strict(encoders[col], default=UNSEEN_CODE)
        .cast(pl.Float64)
        .alias(f"{col}__code")
        for col in categorical
    )
    feature_names = numeric + categorical
    columns = numeric + [f"{c}__code" for c in categorical]
    X = frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()
    y = frame.get_column("log_psf").to_numpy() if "log_psf" in frame.columns else None
    categorical_idx = list(range(len(numeric), len(columns)))
    return X, y, feature_names, categorical_idx


# ---------------------------------------------------------------------------
# Validation + training
# ---------------------------------------------------------------------------

def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true - y_pred  # log(actual) - log(fair value)
    spread = np.expm1(resid)  # actual / fair value - 1
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "medape": float(np.median(np.abs(spread))),
        "mae_log": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def _make_model(model_params: dict | None, categorical_idx: list[int], random_state: int):
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    return HistGradientBoostingRegressor(
        categorical_features=categorical_idx,
        random_state=random_state,
        **params,
    )


def cross_validate(
    df: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    model_params: dict | None = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """Out-of-time CV metrics via date-ordered TimeSeriesSplit.

    Encoders are refit on each fold's training slice so validation rows
    with unseen categories are handled exactly as they would be live.
    """
    df = df.sort("date")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(np.arange(df.height)):
        train_df = df[train_idx.min() : train_idx.max() + 1]
        val_df = df[val_idx.min() : val_idx.max() + 1]
        encoders = fit_encoders(train_df, feature_config)
        X_tr, y_tr, _, cat_idx = to_matrix(train_df, encoders, feature_config)
        X_va, y_va, _, _ = to_matrix(val_df, encoders, feature_config)
        model = _make_model(model_params, cat_idx, random_state)
        model.fit(X_tr, y_tr)
        folds.append(_fold_metrics(y_va, model.predict(X_va)))

    summary = {
        "n_splits": n_splits,
        "n_rows": df.height,
        "folds": folds,
    }
    for key in ("medape", "mae_log", "r2"):
        values = [f[key] for f in folds]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


@dataclass
class FairValueResult:
    model: HistGradientBoostingRegressor
    encoders: dict[str, dict[str, int]]
    feature_names: list[str]
    categorical_idx: list[int]
    feature_config: dict[str, bool]
    metrics: dict
    importances: pl.DataFrame
    date_origin: date
    trained_rows: int = 0


def train_fair_value_model(
    df: pl.DataFrame,
    feature_config: dict[str, bool] | None = None,
    model_params: dict | None = None,
    n_splits: int = 10,
    random_state: int = 42,
    importance_sample: int = 20_000,
) -> FairValueResult:
    """CV-validate, then refit on all rows for scoring.

    Reported metrics are honest out-of-time numbers from the 10-fold
    TimeSeriesSplit; the returned model is refit on the full frame
    (standard AVM anomaly-scoring pattern). Permutation importance is
    computed on the most recent ~10% of rows.
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    df = df.sort("date")
    metrics = cross_validate(df, cfg, model_params, n_splits=n_splits, random_state=random_state)

    encoders = fit_encoders(df, cfg)
    X, y, feature_names, cat_idx = to_matrix(df, encoders, cfg)
    model = _make_model(model_params, cat_idx, random_state)
    model.fit(X, y)

    tail = max(1, df.height // 10)
    rng = np.random.default_rng(random_state)
    tail_idx = np.arange(df.height - tail, df.height)
    if tail > importance_sample:
        tail_idx = rng.choice(tail_idx, size=importance_sample, replace=False)
    perm = permutation_importance(
        model, X[tail_idx], y[tail_idx], n_repeats=5, random_state=random_state
    )
    importances = pl.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort("importance_mean", descending=True)

    return FairValueResult(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        categorical_idx=cat_idx,
        feature_config=cfg,
        metrics=metrics,
        importances=importances,
        date_origin=df.select(pl.col("date").min()).item(),
        trained_rows=df.height,
    )


# ---------------------------------------------------------------------------
# Scoring + distress flagging
# ---------------------------------------------------------------------------

def score_transactions(result: FairValueResult, df: pl.DataFrame) -> pl.DataFrame:
    """Add fair-value predictions and the price spread to a feature frame.

    ``df`` must come from :func:`feature_engineering` with the same
    feature_config (pass ``date_origin=result.date_origin`` when scoring
    data outside the training frame).
    """
    X, _, _, _ = to_matrix(df, result.encoders, result.feature_config)
    pred_log_psf = result.model.predict(X)
    return df.with_columns(
        pl.Series("pred_psf", np.exp(pred_log_psf)),
    ).with_columns(
        (pl.col("pred_psf") * pl.col("ACTUAL_AREA") * SQM_TO_SQFT).alias("fair_value_aed"),
    ).with_columns(
        (pl.col("TRANS_VALUE") / pl.col("fair_value_aed") - 1).alias("spread_pct"),
    )


def flag_distress(
    scored: pl.DataFrame,
    spread_threshold: float = -0.15,
    deep_discount: float = -0.25,
    min_project_txns: int = 8,
) -> pl.DataFrame:
    """Flag below-fair-value rows and corroborated distressed-asset candidates.

    ``distressed`` requires the spread to be at/below ``spread_threshold``
    AND at least one corroborating signal, so a single model miss is not
    enough to label a sale distressed.
    """
    df = scored.with_columns(
        (pl.col("spread_pct") <= spread_threshold).alias("below_fair_value"),
        pl.col("PROCEDURE_EN")
        .cast(pl.Utf8)
        .str.contains(DISTRESS_PROCEDURE_PATTERN)
        .fill_null(False)
        .alias("sig_procedure"),
        (pl.col("spread_pct") <= deep_discount).alias("sig_deep_discount"),
        (pl.len().over("PROJECT_EN") < min_project_txns).alias("sig_illiquid_project"),
        (pl.col("total_seller") > 1).fill_null(False).alias("sig_multi_seller")
        if "total_seller" in scored.columns
        else pl.lit(False).alias("sig_multi_seller"),
    )
    signal_labels = {
        "sig_procedure": "forced-sale procedure",
        "sig_deep_discount": "deep discount",
        "sig_illiquid_project": "illiquid project",
        "sig_multi_seller": "multiple sellers",
    }
    df = df.with_columns(
        sum(pl.col(c).cast(pl.Int32) for c in signal_labels).alias("distress_score"),
    ).with_columns(
        (pl.col("below_fair_value") & (pl.col("distress_score") >= 1)).alias("distressed"),
        pl.concat_str(
            [
                pl.when(pl.col(c)).then(pl.lit(label)).otherwise(pl.lit(None))
                for c, label in signal_labels.items()
            ],
            separator="; ",
            ignore_nulls=True,
        ).alias("signals"),
    )
    return df
