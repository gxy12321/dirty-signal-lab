from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ModelSpec:
    name: str
    estimator: object


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std, mean, std


def _walk_forward_split(n: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    split = int(n * train_frac)
    idx = np.arange(n)
    return idx < split, idx >= split


def available_models() -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {}

    # baseline linear ridge
    from sklearn.linear_model import Ridge

    models["ridge"] = ModelSpec("ridge", Ridge(alpha=1.0))

    # tree ensembles
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    models["rf"] = ModelSpec(
        "rf",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=7,
        ),
    )
    models["extratrees"] = ModelSpec(
        "extratrees",
        ExtraTreesRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=7,
        ),
    )

    # gradient boosting
    import xgboost as xgb

    models["xgb"] = ModelSpec(
        "xgb",
        xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=7,
        ),
    )

    import lightgbm as lgb

    models["lgbm"] = ModelSpec(
        "lgbm",
        lgb.LGBMRegressor(
            n_estimators=600,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=7,
        ),
    )

    # simple MLP
    from sklearn.neural_network import MLPRegressor

    models["mlp"] = ModelSpec(
        "mlp",
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=7,
        ),
    )

    return models


def _prepare_xy(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(y)
    train_mask, _ = _walk_forward_split(n, train_frac=train_frac)

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    train_mask = train_mask & valid

    if train_mask.sum() < 30:
        return None

    x_train = x[train_mask]
    y_train = y[train_mask]

    x_train_std, mean, std = _standardize(x_train)
    x_all_std = (x - mean) / std
    x_all_std = np.where(np.isfinite(x_all_std), x_all_std, 0.0)
    return x_train_std, y_train, x_all_std


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    train_frac: float = 0.7,
) -> np.ndarray:
    prepared = _prepare_xy(x, y, train_frac=train_frac)
    if prepared is None:
        return np.zeros(len(y))

    x_train_std, y_train, x_all_std = prepared

    models = available_models()
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(models)}")

    estimator = models[model_name].estimator
    estimator.fit(x_train_std, y_train)
    preds = estimator.predict(x_all_std)
    preds = np.where(np.isfinite(preds), preds, 0.0)
    return preds


def fit_predict_all(
    x: np.ndarray,
    y: np.ndarray,
    model_names: Iterable[str],
    train_frac: float = 0.7,
) -> dict[str, np.ndarray]:
    prepared = _prepare_xy(x, y, train_frac=train_frac)
    if prepared is None:
        return {name: np.zeros(len(y)) for name in model_names}

    x_train_std, y_train, x_all_std = prepared
    models = available_models()

    preds: dict[str, np.ndarray] = {}
    for name in model_names:
        if name not in models:
            raise ValueError(f"Unknown model '{name}'. Available: {sorted(models)}")
        estimator = models[name].estimator
        estimator.fit(x_train_std, y_train)
        pred = estimator.predict(x_all_std)
        preds[name] = np.where(np.isfinite(pred), pred, 0.0)

    return preds
