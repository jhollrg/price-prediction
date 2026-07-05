"""Central configuration for paths and model hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

DB_PATH = DATA_DIR / "nyc_taxi.db"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion.joblib"

TARGET_COL: str = "fare_amount"

FEATURE_COLS: list[str] = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_dow",
    "is_weekend",
    "is_rush_hour",
    "time_of_day_sin",
    "time_of_day_cos",
    "haversine_km",
    "fare_per_mile",
    "zone_avg_fare",
    "zone_trip_count",
    "avg_speed_zone",
]

CATEGORICAL_COLS: list[str] = [
    "distance_bucket",
]


@dataclass
class RFParams:
    """Hyperparameters for scikit-learn RandomForestRegressor."""

    n_estimators: int = 300
    max_depth: int = 20
    min_samples_leaf: int = 10
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class GBParams:
    """Hyperparameters for scikit-learn GradientBoostingRegressor."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    random_state: int = 42


@dataclass
class CatBoostParams:
    """Hyperparameters for CatBoostRegressor."""

    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 8
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    random_seed: int = 42
    verbose: int = 100


@dataclass
class MLPParams:
    """Hyperparameters for the PyTorch MLP wrapper."""

    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 50
    random_state: int = 42


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    rf: RFParams = field(default_factory=RFParams)
    gb: GBParams = field(default_factory=GBParams)
    catboost: CatBoostParams = field(default_factory=CatBoostParams)
    mlp: MLPParams = field(default_factory=MLPParams)


cfg = Config()
