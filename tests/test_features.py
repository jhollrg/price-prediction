"""Unit tests for pure feature-engineering functions in src/features.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.features import (
    build_feature_matrix,
    distance_bucket,
    encode_day_of_week,
    encode_hour,
    haversine_km,
    rush_hour_flag,
)

from src.config import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL


class TestHaversineKm:
    def test_same_point_is_zero(self) -> None:
        """A trip from a point to itself should be 0 km."""
        assert haversine_km(40.75, -73.99, 40.75, -73.99) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_jfk_to_manhattan(self) -> None:
        """JFK Airport to Times Square should be ~22 km (rough check)."""
        dist = haversine_km(40.6413, -73.7781, 40.7580, -73.9855)
        assert 18.0 < dist < 26.0

    def test_vectorised_input(self) -> None:
        """haversine_km should accept numpy arrays and return an array."""
        lat1 = np.array([40.75, 40.65])
        lon1 = np.array([-73.99, -73.78])
        lat2 = np.array([40.76, 40.66])
        lon2 = np.array([-74.00, -73.79])
        result = haversine_km(lat1, lon1, lat2, lon2)
        assert result.shape == (2,)
        assert np.all(result >= 0)

    def test_symmetry(self) -> None:
        """Distance A→B should equal distance B→A."""
        d1 = haversine_km(40.75, -73.99, 40.65, -73.78)
        d2 = haversine_km(40.65, -73.78, 40.75, -73.99)
        assert d1 == pytest.approx(d2, rel=1e-6)


class TestHaversineKnownDistance:
    def test_nyc_to_lax_within_one_percent(self) -> None:
        """Manhattan (40.7128°N, 74.0060°W) → LAX (33.9416°N, 118.4085°W).

        The commonly cited great-circle distance is ~3 940 km.  We assert
        within 1 % to allow for minor Earth-radius convention differences.
        """
        dist = haversine_km(40.7128, -74.0060, 33.9416, -118.4085)
        assert dist == pytest.approx(3_940.0, rel=0.01)


class TestHaversineNanSafe:
    def test_scalar_nan_lat_returns_nan(self) -> None:
        """A NaN latitude input must propagate to NaN output, not raise."""
        result = haversine_km(float("nan"), -74.0060, 40.7128, -74.0060)
        assert np.isnan(float(result))

    def test_scalar_nan_lon_returns_nan(self) -> None:
        result = haversine_km(40.7128, float("nan"), 40.7128, -73.9857)
        assert np.isnan(float(result))

    def test_array_nan_propagates_only_to_bad_row(self) -> None:
        """NaN in one element of a vector should not corrupt neighbouring rows."""
        lat1 = np.array([40.75, float("nan"), 40.65])
        lon1 = np.array([-73.99, -73.99, -73.78])
        lat2 = np.array([40.76, 40.76, 40.66])
        lon2 = np.array([-74.00, -74.00, -73.79])
        result = haversine_km(lat1, lon1, lat2, lon2)
        assert not np.isnan(result[0]), "Valid row 0 should not be NaN"
        assert np.isnan(result[1]),     "Row with NaN input must yield NaN"
        assert not np.isnan(result[2]), "Valid row 2 should not be NaN"



class TestRushHourFlag:
    @pytest.mark.parametrize("hour,dow", [(7, 0), (8, 2), (17, 4), (17, 1), (18, 3)])
    def test_weekday_rush_hours_are_flagged(self, hour: int, dow: int) -> None:
        assert rush_hour_flag(hour, dow) == 1

    @pytest.mark.parametrize("hour,dow", [(7, 5), (8, 6), (16, 6)])
    def test_weekend_rush_windows_are_not_flagged(self, hour: int, dow: int) -> None:
        """Weekend trips during AM/PM rush should NOT be flagged."""
        assert rush_hour_flag(hour, dow) == 0

    @pytest.mark.parametrize("hour,dow", [(3, 1), (12, 3), (22, 0)])
    def test_off_peak_hours_are_not_flagged(self, hour: int, dow: int) -> None:
        assert rush_hour_flag(hour, dow) == 0

    def test_vectorised_input(self) -> None:
        hours = np.array([7, 12, 17])
        dows = np.array([0, 0, 6])  # Mon, Mon, Sun
        result = rush_hour_flag(hours, dows)
        np.testing.assert_array_equal(result, [1, 0, 0])


class TestRushHourFlagExplicit:
    """Explicit named scenarios called out in the project specification."""

    def test_8am_tuesday_is_rush(self) -> None:
        """8 am on a Tuesday (dow=1) falls in the AM rush window."""
        assert rush_hour_flag(8, 1) == 1

    def test_10am_saturday_is_not_rush(self) -> None:
        """10 am on a Saturday (dow=5) is both off-peak and a weekend."""
        assert rush_hour_flag(10, 5) == 0



class TestEncodeDayOfWeek:
    def test_output_range(self) -> None:
        for dow in range(7):
            s, c = encode_day_of_week(dow)
            assert -1.0 <= s <= 1.0
            assert -1.0 <= c <= 1.0

    def test_sin_cos_unit_circle(self) -> None:
        """sin² + cos² should equal 1 for every day."""
        for dow in range(7):
            s, c = encode_day_of_week(dow)
            assert s ** 2 + c ** 2 == pytest.approx(1.0, abs=1e-9)


class TestEncodeHour:
    def test_midnight_and_noon_differ(self) -> None:
        s0, c0 = encode_hour(0)
        s12, c12 = encode_hour(12)
        assert (s0, c0) != (s12, c12)

    def test_sin_cos_unit_circle(self) -> None:
        for h in range(24):
            s, c = encode_hour(h)
            assert s ** 2 + c ** 2 == pytest.approx(1.0, abs=1e-9)

class TestDistanceBucket:
    @pytest.mark.parametrize("dist,expected", [
        (0.5, "short"),
        (1.0, "medium"),
        (2.9, "medium"),
        (3.0, "long"),
        (9.9, "long"),
        (10.0, "very_long"),
        (50.0, "very_long"),
    ])
    def test_bucket_boundaries(self, dist: float, expected: str) -> None:
        assert distance_bucket(dist) == expected


class TestFeatureMatrixNoLeakage:
    """Ensure build_feature_matrix separates X from y without leaking the target."""
    @pytest.fixture()
    def complete_df(self) -> pd.DataFrame:
        """Minimal DataFrame containing all required feature and target columns."""
        rng = np.random.default_rng(99)
        n = 20
        trip_distance = rng.uniform(0.5, 15.0, n)
        pickup_hour = rng.integers(0, 24, n)
        angle = 2.0 * np.pi * pickup_hour / 24.0
        fare_amount = 3.0 + 2.5 * trip_distance + rng.normal(0.0, 0.5, n)
        return pd.DataFrame({
            TARGET_COL:          fare_amount,
            "passenger_count":   rng.integers(1, 7, n),
            "trip_distance":     trip_distance,
            "pickup_hour":       pickup_hour,
            "pickup_dow":        rng.integers(0, 7, n),
            "is_weekend":        rng.integers(0, 2, n),
            "is_rush_hour":      rng.integers(0, 2, n),
            "time_of_day_sin":   np.sin(angle),
            "time_of_day_cos":   np.cos(angle),
            "haversine_km":      trip_distance * 1.60934,
            "fare_per_mile":     fare_amount / trip_distance,
            "zone_avg_fare":     rng.uniform(8.0, 25.0, n),
            "zone_trip_count":   rng.integers(10, 5000, n),
            "avg_speed_zone":    rng.uniform(5.0, 25.0, n),
            "distance_bucket":   rng.choice(["short", "medium", "long", "very_long"], n),
        })

    def test_target_not_in_X(self, complete_df: pd.DataFrame) -> None:
        X, _ = build_feature_matrix(complete_df)
        assert TARGET_COL not in X.columns, (
            f"Target column '{TARGET_COL}' leaked into the feature matrix."
        )

    def test_X_has_exactly_feature_cols(self, complete_df: pd.DataFrame) -> None:
        X, _ = build_feature_matrix(complete_df)
        assert set(X.columns) == set(FEATURE_COLS + CATEGORICAL_COLS)

    def test_y_name_matches_target(self, complete_df: pd.DataFrame) -> None:
        _, y = build_feature_matrix(complete_df)
        assert y.name == TARGET_COL

    def test_X_and_y_have_same_length(self, complete_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(complete_df)
        assert len(X) == len(y)
