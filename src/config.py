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


@dataclass(frozen=True)
class TrainingProfile:
    """Resource profile controlling data volume and model sizes.

    The ``full`` profile reproduces the original training setup and assumes a
    workstation with plenty of RAM.  The ``lite`` profile targets low-end
    client hardware (e.g. 4 GB RAM, 2 CPU cores, no GPU): it samples the
    dataset in SQL before anything reaches pandas, downcasts features to
    float32, shrinks every model, and caps thread usage so the machine stays
    responsive during training.

    Parameters
    ----------
    name : str
        Profile identifier, matches the key in ``PROFILES``.
    max_rows : int, optional
        Cap on rows loaded from SQLite (systematic sample). ``None`` loads all.
    float32 : bool
        Downcast float feature columns to float32 after loading, halving the
        in-memory footprint of the feature matrix.
    n_jobs : int
        Parallelism for sklearn models and CatBoost. ``-1`` uses all cores.
    selector_iterations : int
        CatBoost iterations for the SHAP feature-selection surrogate.
    selector_max_fit_rows : int, optional
        Subsample size for fitting the SHAP selector. ``None`` uses all rows.
    rf_n_estimators, rf_max_depth, rf_min_samples_leaf : int
        RandomForestRegressor size controls.
    cb_iterations, cb_depth, cb_border_count : int
        CatBoostRegressor size controls (``border_count`` lowers the memory
        used for numeric feature quantisation).
    mlp_hidden_sizes : tuple[int, ...]
        Hidden-layer widths of the PyTorch MLP.
    mlp_epochs, mlp_batch_size : int
        MLP training-loop controls.
    torch_num_threads : int, optional
        Intra-op thread cap for PyTorch on CPU. ``None`` leaves the default.
    """

    name: str
    max_rows: int | None
    float32: bool
    n_jobs: int
    selector_iterations: int
    selector_max_fit_rows: int | None
    rf_n_estimators: int
    rf_max_depth: int
    rf_min_samples_leaf: int
    cb_iterations: int
    cb_depth: int
    cb_border_count: int
    mlp_hidden_sizes: tuple[int, ...]
    mlp_epochs: int
    mlp_batch_size: int
    torch_num_threads: int | None


FULL_PROFILE = TrainingProfile(
    name="full",
    max_rows=None,
    float32=False,
    n_jobs=-1,
    selector_iterations=500,
    selector_max_fit_rows=None,
    rf_n_estimators=200,
    rf_max_depth=12,
    rf_min_samples_leaf=1,
    cb_iterations=1_000,
    cb_depth=8,
    cb_border_count=254,
    mlp_hidden_sizes=(256, 128),
    mlp_epochs=50,
    mlp_batch_size=1_024,
    torch_num_threads=None,
)

LITE_PROFILE = TrainingProfile(
    name="lite",
    max_rows=200_000,
    float32=True,
    n_jobs=2,
    selector_iterations=150,
    selector_max_fit_rows=50_000,
    rf_n_estimators=60,
    rf_max_depth=10,
    rf_min_samples_leaf=20,
    cb_iterations=300,
    cb_depth=6,
    cb_border_count=64,
    mlp_hidden_sizes=(64, 32),
    mlp_epochs=15,
    mlp_batch_size=512,
    torch_num_threads=2,
)

PROFILES: dict[str, TrainingProfile] = {
    "full": FULL_PROFILE,
    "lite": LITE_PROFILE,
}
