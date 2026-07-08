from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabulate import tabulate

from config import (CATEGORICAL_COLS, CHAMPION_MODEL_PATH, DB_PATH,
                    FULL_PROFILE, MODELS_DIR, PROFILES, REPORTS_DIR,
                    TARGET_COL, TrainingProfile)
from data_loader import get_train_test_split
from feature_selection import ShapFeatureSelector
from mlp_model import MLPRegressor


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R2 for a paired set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth fare amounts.
    y_pred : np.ndarray
        Model predictions aligned with ``y_true``.

    Returns
    -------
    dict[str, float]
        Keys: ``rmse``, ``mae``, ``r2``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _detect_cat_cols(df: pd.DataFrame) -> list[str]:
    """Return column names whose dtype is object or category."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _ordinal_pipeline(estimator: Any, cat_cols: list[str]) -> Pipeline:
    """Wrap *estimator* with ``OrdinalEncoder`` for any remaining categorical columns.

    When *cat_cols* is empty the pipeline is a transparent single-step wrapper
    so downstream code can always call ``pipeline.fit / predict`` uniformly.

    Unknown categories at inference time are encoded as ``-1`` rather than
    raising an error, which prevents breakage on unseen distance-bucket values.
    """
    if not cat_cols:
        return Pipeline([("model", estimator)])

    ct = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", ct), ("model", estimator)])


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    profile: TrainingProfile = FULL_PROFILE,
) -> dict[str, Any]:
    """Train a ``LinearRegression`` baseline.

    No hyperparameter tuning is performed — this run establishes a lower-bound
    baseline that all other models must beat.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training split.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out evaluation split.
    profile : TrainingProfile
        Accepted for interface uniformity with the other trainers; a linear
        fit is cheap enough that no profile setting applies.

    Returns
    -------
    dict[str, Any]
        Contains ``model`` (str), ``pipeline``, ``rmse``, ``mae``, ``r2``.
    """
    cat_cols = _detect_cat_cols(X_train)
    pipeline = _ordinal_pipeline(LinearRegression(fit_intercept=True), cat_cols)
    pipeline.fit(X_train, y_train)
    metrics = compute_metrics(y_test.to_numpy(), pipeline.predict(X_test))
    return {"model": "LinearRegression", "pipeline": pipeline, **metrics}


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    profile: TrainingProfile = FULL_PROFILE,
) -> dict[str, Any]:
    """Train a ``RandomForestRegressor``.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training split.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out evaluation split.
    profile : TrainingProfile
        Resource profile providing forest size and parallelism settings.

    Returns
    -------
    dict[str, Any]
        Contains ``model``, ``pipeline``, and the three eval metrics.
    """
    cat_cols = _detect_cat_cols(X_train)
    rf = RandomForestRegressor(
        n_estimators=profile.rf_n_estimators,
        max_depth=profile.rf_max_depth,
        min_samples_leaf=profile.rf_min_samples_leaf,
        n_jobs=profile.n_jobs,
        random_state=42,
    )
    pipeline = _ordinal_pipeline(rf, cat_cols)
    pipeline.fit(X_train, y_train)
    metrics = compute_metrics(y_test.to_numpy(), pipeline.predict(X_test))
    return {"model": "RandomForest", "pipeline": pipeline, **metrics}


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    profile: TrainingProfile = FULL_PROFILE,
) -> dict[str, Any]:
    """Train a ``CatBoostRegressor`` with early stopping.

    Categoricals in *X_train* (detected by object dtype) are passed to
    CatBoost via ``cat_features`` so native encoding is used instead of
    OrdinalEncoder.  The test split is used as the eval set for early stopping;
    training halts after 50 consecutive rounds without improvement.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training split.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out evaluation split (also used as CatBoost eval set).
    profile : TrainingProfile
        Resource profile providing tree count, depth, quantisation border
        count, and thread count.

    Returns
    -------
    dict[str, Any]
        Contains ``model``, ``pipeline``, and the three eval metrics.
    """
    cat_cols = _detect_cat_cols(X_train)
    model = CatBoostRegressor(
        iterations=profile.cb_iterations,
        learning_rate=0.05,
        depth=profile.cb_depth,
        border_count=profile.cb_border_count,
        thread_count=profile.n_jobs,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=42,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols or None,
        eval_set=(X_test, y_test),
    )
    metrics = compute_metrics(y_test.to_numpy(), model.predict(X_test))
    return {"model": "CatBoost", "pipeline": model, **metrics}


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    profile: TrainingProfile = FULL_PROFILE,
) -> dict[str, Any]:
    """Train a PyTorch MLP.

    Architecture (full profile): ``input → 256 → ReLU → Dropout(0.3) → 128 →
    ReLU → 1``; the lite profile shrinks the hidden layers and epoch count.
    ``StandardScaler`` is applied inside ``MLPRegressor`` before the network
    sees the data.  Any remaining categorical columns are ``OrdinalEncoded``
    by the outer pipeline before the scaler.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training split.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out evaluation split.
    profile : TrainingProfile
        Resource profile providing network size, epochs, batch size, and an
        optional CPU thread cap for torch.

    Returns
    -------
    dict[str, Any]
        Contains ``model``, ``pipeline``, and the three eval metrics.
    """
    if profile.torch_num_threads is not None:
        import torch

        torch.set_num_threads(profile.torch_num_threads)

    cat_cols = _detect_cat_cols(X_train)
    mlp = MLPRegressor(
        hidden_sizes=list(profile.mlp_hidden_sizes),
        dropout=0.3,
        lr=1e-3,
        batch_size=profile.mlp_batch_size,
        epochs=profile.mlp_epochs,
        random_state=42,
    )
    pipeline = _ordinal_pipeline(mlp, cat_cols)
    pipeline.fit(X_train, y_train)
    metrics = compute_metrics(y_test.to_numpy(), pipeline.predict(X_test))
    return {"model": "PyTorchMLP", "pipeline": pipeline, **metrics}


def run_all(
    db_path: Optional[Path] = None,
    profile: TrainingProfile = FULL_PROFILE,
) -> None:
    """Train all models under the given resource profile and save the champion.

    Parameters
    ----------
    db_path : Path, optional
        SQLite database path; defaults to ``config.DB_PATH``.
    profile : TrainingProfile
        ``FULL_PROFILE`` (default) or ``LITE_PROFILE`` for low-end client
        hardware — the lite profile samples the dataset, downcasts to
        float32, shrinks every model, and caps thread usage.
    """
    print(f"Profile: {profile.name}")
    print("Loading features from SQLite…")
    X_train, X_test, y_train, y_test = get_train_test_split(
        db_path=db_path or DB_PATH,
        max_rows=profile.max_rows,
        float32=profile.float32,
    )
    print(f"  train rows : {len(X_train):,}   features: {X_train.shape[1]}")
    print(f"  test  rows : {len(X_test):,}")

    print(
        f"\nRunning SHAP feature selection "
        f"(CatBoost surrogate, {profile.selector_iterations} iterations)…"
    )
    selector = ShapFeatureSelector(
        threshold=0.01,
        cat_features=[c for c in CATEGORICAL_COLS if c in X_train.columns],
        catboost_params={
            "iterations": profile.selector_iterations,
            "thread_count": profile.n_jobs,
        },
        random_state=42,
    )
    max_fit = profile.selector_max_fit_rows
    if max_fit is not None and len(X_train) > max_fit:
        print(f"  fitting selector on {max_fit:,}-row subsample")
        X_fit = X_train.sample(n=max_fit, random_state=42)
        selector.fit(X_fit, y_train.loc[X_fit.index])
    else:
        selector.fit(X_train, y_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shap_plot_path = REPORTS_DIR / "shap_beeswarm.png"
    selector.save_shap_plot(shap_plot_path)
    print(f"  SHAP beeswarm → {shap_plot_path}")

    n_orig = X_train.shape[1]
    n_sel = len(selector.selected_features_)
    print(f"  Selected {n_sel}/{n_orig} features")
    print(f"  {selector.selected_features_}")

    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    trainers: list[tuple[str, Any]] = [
        ("LinearRegression", train_linear_regression),
        ("RandomForest", train_random_forest),
        ("CatBoost", train_catboost),
        ("PyTorchMLP", train_mlp),
    ]

    results: list[dict[str, Any]] = []
    for label, fn in trainers:
        _section(label)
        result = fn(X_train_sel, y_train, X_test_sel, y_test, profile=profile)
        results.append(result)
        print(
            f"RMSE {result['rmse']:.4f}  "
            f"MAE {result['mae']:.4f}  "
            f"R2 {result['r2']:.4f}"
        )

    best = min(results, key=lambda r: r["rmse"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], CHAMPION_MODEL_PATH)
    print(
        f"\n  Champion: {best['model']} (RMSE {best['rmse']:.4f}) → {CHAMPION_MODEL_PATH}"
    )

    results_sorted = sorted(results, key=lambda r: r["rmse"])
    rows = [
        [
            ("* " if r is results_sorted[0] else "  ") + r["model"],
            f"{r['rmse']:.4f}",
            f"{r['mae']:.4f}",
            f"{r['r2']:.4f}",
        ]
        for r in results
    ]

    print(f"\n{'═' * 54}")
    print(
        tabulate(
            rows,
            headers=["Model", "RMSE", "MAE", "R2"],
            tablefmt="rounded_outline",
            colalign=("left", "right", "right", "right"),
        )
    )
    print(f"{'═' * 54}")
    print("  * best RMSE\n")


def _section(label: str) -> None:
    print(f"\n{"─" * 62}\n  {label}\n{"─" * 62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all fare-prediction models.")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite DB (overrides config.DB_PATH).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="full",
        help=(
            "Resource profile: 'full' for a workstation, 'lite' for low-end "
            "client hardware (samples data, shrinks models, caps threads)."
        ),
    )
    args = parser.parse_args()
    run_all(db_path=args.db, profile=PROFILES[args.profile])
