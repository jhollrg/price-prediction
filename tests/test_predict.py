"""Smoke and bounds tests for the predict pipeline.

All tests use a real sklearn model trained on the ``sqlite_feature_db`` fixture
(a genuine SQLite file populated in ``conftest.py``).  No connection mocking -
every database access goes through ``data_loader.load_features`` on the real
file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import predict_from_features, predict_from_trips
from config import CATEGORICAL_COLS, FEATURE_COLS


@pytest.fixture(scope="module")
def trained_lr_model(sqlite_feature_db: Path):
    """LinearRegression trained on the SQLite fixture with no model mocking.

    Loads the feature table via ``data_loader.load_features`` (real I/O),
    calls ``build_feature_matrix``, then wraps a ``LinearRegression`` in an
    ``OrdinalEncoder`` pipeline for the categorical column.

    Scope is ``module`` so training runs once per file, not per test.
    """
    from data_loader import load_features
    from features import build_feature_matrix
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    df = load_features(db_path=sqlite_feature_db)
    X, y = build_feature_matrix(df)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    if cat_cols:
        ct = ColumnTransformer(
            transformers=[
                (
                    "ord",
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
        model = Pipeline([("pre", ct), ("reg", LinearRegression())])
    else:
        model = Pipeline([("reg", LinearRegression())])

    model.fit(X, y)
    return model


@pytest.fixture()
def feature_df_10rows() -> pd.DataFrame:
    """10 plausible feature rows matching FEATURE_COLS + CATEGORICAL_COLS.

    Column names and dtypes mirror the ``features`` table schema.
    ``avg_speed_zone`` and ``fare_per_mile`` use the current column names
    (not the legacy ``avg_speed_mph_by_hour``).
    """
    rng = np.random.default_rng(7)
    n = 10
    trip_distance = rng.uniform(1.0, 10.0, n)
    pickup_hour = rng.integers(8, 20, n)
    angle = 2.0 * np.pi * pickup_hour / 24.0
    return pd.DataFrame({
        "passenger_count":  rng.integers(1, 5, n).astype(int),
        "trip_distance":    trip_distance,
        "pickup_hour":      pickup_hour.astype(int),
        "pickup_dow":       rng.integers(0, 5, n).astype(int),
        "is_weekend":       np.zeros(n, dtype=int),
        "is_rush_hour":     rng.integers(0, 2, n).astype(int),
        "time_of_day_sin":  np.sin(angle),
        "time_of_day_cos":  np.cos(angle),
        "haversine_km":     trip_distance * 1.60934,
        "fare_per_mile":    rng.uniform(3.0, 7.0, n),
        "zone_avg_fare":    rng.uniform(10.0, 20.0, n),
        "zone_trip_count":  rng.integers(100, 3000, n).astype(int),
        "avg_speed_zone":   rng.uniform(8.0, 20.0, n),
        "distance_bucket":  ["medium"] * n,
    })


@pytest.fixture()
def raw_trips_10() -> list[dict]:
    """10 realistic raw trip dicts with all fields required by build_feature_row."""
    rng = np.random.default_rng(13)
    n = 10
    records = []
    for _ in range(n):
        records.append({
            "pickup_datetime":   "2023-06-15 09:15:00",
            "dropoff_datetime":  "2023-06-15 09:40:00",
            "pickup_latitude":   float(rng.uniform(40.65, 40.80)),
            "pickup_longitude":  float(rng.uniform(-74.05, -73.85)),
            "dropoff_latitude":  float(rng.uniform(40.65, 40.80)),
            "dropoff_longitude": float(rng.uniform(-74.05, -73.85)),
            "trip_distance":     float(rng.uniform(1.0, 8.0)),
            "passenger_count":   int(rng.integers(1, 5)),
        })
    return records


class TestPredictSmoke:
    """Basic shape and dtype checks — does the pipeline produce any output?"""

    def test_output_shape(
        self,
        feature_df_10rows: pd.DataFrame,
        trained_lr_model,
    ) -> None:
        """predict_from_features must return exactly one prediction per row."""
        preds = predict_from_features(feature_df_10rows, model=trained_lr_model)
        assert preds.shape == (10,)

    def test_output_is_float(
        self,
        feature_df_10rows: pd.DataFrame,
        trained_lr_model,
    ) -> None:
        preds = predict_from_features(feature_df_10rows, model=trained_lr_model)
        assert np.issubdtype(preds.dtype, np.floating)


class TestPredictBounds:
    """Predicted fares must fall within plausible NYC TLC bounds."""

    def test_fares_in_dollar_range(
        self,
        feature_df_10rows: pd.DataFrame,
        trained_lr_model,
    ) -> None:
        """Predictions for typical trips should be between $1 and $200."""
        preds = predict_from_features(feature_df_10rows, model=trained_lr_model)
        assert np.all(preds >= 1.0), (
            f"Prediction ${preds.min():.2f} is below the $1 lower bound."
        )
        assert np.all(preds <= 200.0), (
            f"Prediction ${preds.max():.2f} exceeds the $200 upper bound."
        )


class TestPredictFromTrips:
    """Verify predict_from_trips handles list-of-dicts and DataFrame inputs."""

    def test_list_of_dicts_shape(
        self,
        raw_trips_10: list[dict],
        trained_lr_model,
    ) -> None:
        preds = predict_from_trips(raw_trips_10, model=trained_lr_model)
        assert preds.shape == (10,)

    def test_dataframe_input_shape(
        self,
        raw_trips_10: list[dict],
        trained_lr_model,
    ) -> None:
        df = pd.DataFrame(raw_trips_10)
        preds = predict_from_trips(df, model=trained_lr_model)
        assert preds.shape == (10,)

    def test_single_trip_dict(self, trained_lr_model) -> None:
        """A one-element list should produce shape (1,) output."""
        trip = {
            "pickup_datetime":   "2023-06-15 09:15:00",
            "dropoff_datetime":  "2023-06-15 09:40:00",
            "pickup_latitude":   40.748817,
            "pickup_longitude":  -73.985428,
            "dropoff_latitude":  40.712776,
            "dropoff_longitude": -74.005974,
            "trip_distance":     3.2,
            "passenger_count":   1,
        }
        preds = predict_from_trips([trip], model=trained_lr_model)
        assert preds.shape == (1,)
