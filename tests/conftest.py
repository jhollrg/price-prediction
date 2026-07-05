"""Configure sys.path and provide shared pytest fixtures."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def sqlite_feature_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real on-disk SQLite database with 500 synthetic feature rows.

    Uses ``tmp_path_factory`` (session-scoped) so the file is created once per
    test session and shared across all test modules.  No connection mocking —
    the database is a genuine SQLite file in pytest's auto-cleaned temp tree.

    Schema mirrors ``sql/create_tables.sql``.

    Returns
    -------
    Path
        Absolute path to the populated SQLite file.
    """
    db_path = tmp_path_factory.mktemp("db") / "test_features.db"

    rng = np.random.default_rng(42)
    n = 500

    trip_distance = rng.uniform(0.5, 15.0, n)
    pickup_hour = rng.integers(0, 24, n)
    pickup_dow = rng.integers(0, 7, n)
    is_weekend = (pickup_dow >= 5).astype(int)
    is_rush_arr = (
        ((pickup_hour >= 7) & (pickup_hour <= 8))
        | ((pickup_hour >= 17) & (pickup_hour <= 18))
    )
    is_rush_hour = ((pickup_dow < 5) & is_rush_arr).astype(int)

    angle = 2.0 * np.pi * pickup_hour / 24.0
    time_sin = np.sin(angle)
    time_cos = np.cos(angle)

    haversine_km_vals = trip_distance * 1.60934
    distance_bucket = np.where(
        trip_distance < 1.0,
        "short",
        np.where(
            trip_distance < 3.0,
            "medium",
            np.where(trip_distance < 10.0, "long", "very_long"),
        ),
    )

    # Fare formula matches TLC structure: $3 base + $2.50/mile + noise.
    # Clipped to [3.50, 150.00] so every row is valid training data and the
    # LinearRegression fixture produces predictions well within [1, 200].
    fare_amount = 3.0 + 2.5 * trip_distance + rng.normal(0.0, 0.5, n)
    fare_amount = np.clip(fare_amount, 3.50, 150.0)
    fare_per_mile = fare_amount / trip_distance

    zone_avg_fare = np.clip(rng.normal(13.5, 2.0, n), 5.0, 50.0)
    zone_trip_count = rng.integers(10, 5001, n)
    avg_speed_zone = np.clip(rng.normal(12.0, 3.0, n), 3.0, 30.0)
    passenger_count = rng.integers(1, 7, n)
    pickup_zone_id = rng.integers(1, 264, n)
    dropoff_zone_id = rng.integers(1, 264, n)

    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE features (
            trip_id          INTEGER PRIMARY KEY,
            fare_amount      REAL    NOT NULL,
            passenger_count  INTEGER,
            trip_distance    REAL,
            pickup_zone_id   INTEGER,
            dropoff_zone_id  INTEGER,
            pickup_hour      INTEGER,
            pickup_dow       INTEGER,
            is_weekend       INTEGER,
            is_rush_hour     INTEGER,
            time_of_day_sin  REAL,
            time_of_day_cos  REAL,
            haversine_km     REAL,
            distance_bucket  TEXT,
            fare_per_mile    REAL,
            zone_avg_fare    REAL,
            zone_trip_count  INTEGER,
            avg_speed_zone   REAL
        )
    """)
    con.executemany(
        "INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                i + 1,
                float(fare_amount[i]),
                int(passenger_count[i]),
                float(trip_distance[i]),
                int(pickup_zone_id[i]),
                int(dropoff_zone_id[i]),
                int(pickup_hour[i]),
                int(pickup_dow[i]),
                int(is_weekend[i]),
                int(is_rush_hour[i]),
                float(time_sin[i]),
                float(time_cos[i]),
                float(haversine_km_vals[i]),
                str(distance_bucket[i]),
                float(fare_per_mile[i]),
                float(zone_avg_fare[i]),
                int(zone_trip_count[i]),
                float(avg_speed_zone[i]),
            )
            for i in range(n)
        ],
    )
    con.commit()
    con.close()
    return db_path
