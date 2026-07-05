"""Load engineered features from the SQLite database into a DataFrame."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CATEGORICAL_COLS, DB_PATH, FEATURE_COLS, TARGET_COL
from features import build_feature_matrix


def load_features(
    db_path: Path = DB_PATH,
    limit: Optional[int] = None,
    where: str = "",
) -> pd.DataFrame:
    """Load the features table from SQLite and return a clean DataFrame.

    Parameters
    ----------
    db_path : Path
        Absolute path to the SQLite database file.
    limit : int, optional
        If given, appends ``LIMIT <n>`` to the query.  Useful for fast dev
        iterations and smoke tests without loading the full ~10 M-row dataset.
    where : str
        Optional SQL WHERE clause fragment (omit the ``WHERE`` keyword),
        e.g. ``"pickup_hour BETWEEN 8 AND 10"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``TARGET_COL`` + ``FEATURE_COLS`` + ``CATEGORICAL_COLS``.
        Rows with NULL in any feature column are dropped before returning.
    """
    select_cols = ", ".join([TARGET_COL] + FEATURE_COLS + CATEGORICAL_COLS)
    query = f"SELECT {select_cols} FROM features"

    if where:
        query += f" WHERE {where}"
    if limit is not None:
        query += f" LIMIT {limit}"

    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, con)
    finally:
        con.close()

    df = df.dropna(subset=FEATURE_COLS + CATEGORICAL_COLS)
    return df


def _populate_db(db_path: Path, sql_dir: Path, raw_dir: Path) -> None:
    """Create SQLite schema, load raw parquet files into trips, run feature SQL.

    Parameters
    ----------
    db_path : Path
        Destination SQLite file (created if absent).
    sql_dir : Path
        Directory containing ``create_tables.sql`` and ``feature_queries.sql``.
    raw_dir : Path
        Directory that holds ``*.parquet`` source files downloaded by
        ``scripts/download_data.py``.
    """
    import sqlite3 as _sqlite3
    import sys as _sys

    # Column mapping: TLC parquet source names → our trips schema
    _COL_MAP: dict[str, str] = {
        "VendorID":               "vendor_id",
        "vendor_id":              "vendor_id",
        "tpep_pickup_datetime":   "pickup_datetime",
        "tpep_dropoff_datetime":  "dropoff_datetime",
        "passenger_count":        "passenger_count",
        "trip_distance":          "trip_distance",
        "pickup_longitude":       "pickup_longitude",
        "pickup_latitude":        "pickup_latitude",
        "dropoff_longitude":      "dropoff_longitude",
        "dropoff_latitude":       "dropoff_latitude",
        "PULocationID":           "pickup_zone_id",
        "DOLocationID":           "dropoff_zone_id",
        "pickup_zone_id":         "pickup_zone_id",
        "dropoff_zone_id":        "dropoff_zone_id",
        "fare_amount":            "fare_amount",
        "tip_amount":             "tip_amount",
        "total_amount":           "total_amount",
        "payment_type":           "payment_type",
    }
    _TRIPS_COLS = [
        "vendor_id", "pickup_datetime", "dropoff_datetime",
        "passenger_count", "trip_distance",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "pickup_zone_id", "dropoff_zone_id",
        "fare_amount", "tip_amount", "total_amount", "payment_type",
    ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[db] database : {db_path}")
    con = _sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode = WAL")

    con.executescript((sql_dir / "create_tables.sql").read_text())
    print("[db] schema   : OK")

    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        print(f"[db] ERROR: no .parquet files in {raw_dir}. Run 'make data' first.")
        con.close()
        _sys.exit(1)

    total = 0
    for pq in parquets:
        print(f"[db] loading  : {pq.name} …", end="", flush=True)
        df = pd.read_parquet(pq)
        df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
        keep = [c for c in _TRIPS_COLS if c in df.columns]
        df = df[keep]
        for col in ("pickup_datetime", "dropoff_datetime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        # SQLite caps bound parameters per statement (SQLITE_MAX_VARIABLE_NUMBER);
        # method="multi" packs chunksize * n_cols of them into one INSERT, so the
        # chunksize must shrink as the column count grows to stay under the limit.
        safe_chunksize = max(1, 999 // len(df.columns))
        df.to_sql("trips", con, if_exists="append", index=False,
                  method="multi", chunksize=safe_chunksize)
        total += len(df)
        print(f" {len(df):,} rows")

    print(f"[db] trips    : {total:,} total rows")

    # DELETE first so re-runs stay idempotent
    con.execute("DELETE FROM features")
    con.commit()
    print("[db] features : running SQL pipeline …")
    con.executescript((sql_dir / "feature_queries.sql").read_text())

    n_feat = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    print(f"[db] features : {n_feat:,} rows")
    con.close()
    print("[db] done.")


def get_train_test_split(
    db_path: Path = DB_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test from the features table.

    Loads the full features table, extracts the model-ready feature matrix via
    ``build_feature_matrix``, then performs a random stratification-free split.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database.
    test_size : float
        Fraction of rows reserved for the held-out test set.
    random_state : int
        Seed for the split RNG, ensuring reproducibility across runs.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices for training and test splits.
    y_train, y_test : pd.Series
        Target vectors aligned with the corresponding feature matrices.
    """
    df = load_features(db_path=db_path)
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    _root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Populate SQLite DB from raw parquet files and build feature table."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite file (default: config.DB_PATH).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_root / "data" / "raw",
        help="Directory containing .parquet source files.",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=_root / "sql",
        help="Directory containing create_tables.sql and feature_queries.sql.",
    )
    _args = parser.parse_args()
    _populate_db(db_path=_args.db, sql_dir=_args.sql_dir, raw_dir=_args.raw_dir)
