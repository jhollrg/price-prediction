from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLS,
    CHAMPION_MODEL_PATH,
    FEATURE_COLS,
)
from features import build_feature_row


def load_champion_model(path: Optional[Path] = None) -> Any:
    model_path = path or CHAMPION_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No champion model found at {model_path}. Run `make train` first."
        )
    return joblib.load(model_path)


def predict_from_features(
    X: pd.DataFrame,
    model: Optional[Any] = None,
) -> np.ndarray:
    """Run inference on a pre-built feature DataFrame.

    Accepts any model that responds to ``model.predict(df)`` — sklearn
    ``Pipeline`` objects and CatBoost models are both supported.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.  Extra columns are silently ignored; column order
        does not matter.  Must contain all columns the model was fitted on.
    model : model object, optional
        Pre-loaded model.  If ``None``, ``load_champion_model()`` is called.

    Returns
    -------
    np.ndarray of shape (n_rows,)
        Predicted fare amounts as a 1-D float64 array.
    """
    if model is None:
        model = load_champion_model()

    raw = model.predict(X)

    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0].to_numpy()
    return np.asarray(raw, dtype=float).ravel()


def predict_from_trips(
    trips: Union[list[dict], pd.DataFrame],
    model: Optional[Any] = None,
) -> np.ndarray:
    """Convert raw trip records to features and return fare predictions.

    Each trip is processed by ``features.build_feature_row``, which mirrors
    the SQL feature-engineering pipeline and fills zone-aggregate fields with
    NYC TLC medians when real zone data is unavailable.

    Parameters
    ----------
    trips : list[dict] or pd.DataFrame
        Raw trip records.  Each row / dict must contain the fields expected by
        ``build_feature_row`` (pickup/dropoff coordinates, datetime, distance).
        A DataFrame is converted to a list of row dicts automatically.
    model : model object, optional
        Pre-loaded model.  If ``None``, ``load_champion_model()`` is called.

    Returns
    -------
    np.ndarray of shape (n_trips,)
        Predicted fare amounts.
    """
    if isinstance(trips, pd.DataFrame):
        records: list[dict] = trips.to_dict(orient="records")
    else:
        records = list(trips)

    feature_rows = [build_feature_row(t) for t in records]
    X = pd.DataFrame(feature_rows)
    return predict_from_features(X, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fare prediction on a CSV of trips.")
    parser.add_argument("input_csv", type=Path, help="CSV with raw trip columns.")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Where to write predictions.  Defaults to stdout.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a joblib model file (defaults to models/champion.joblib).",
    )
    args = parser.parse_args()

    loaded_model = load_champion_model(path=args.model)
    df = pd.read_csv(args.input_csv)
    preds = predict_from_trips(df, model=loaded_model)

    result = df.copy()
    result["predicted_fare"] = preds

    if args.output_csv:
        result.to_csv(args.output_csv, index=False)
    else:
        print(result[["predicted_fare"]].to_string())
