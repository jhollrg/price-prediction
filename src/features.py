"""Python-side feature engineering matching sql/feature_queries.sql.

Used by the predict pipeline to compute features on raw trip records without
hitting the database, and by unit tests to verify parity with the SQL layer.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from config import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL


_EARTH_RADIUS_KM: float = 6371.0

# Rush windows match feature_queries.sql: AM 07:00–08:59, PM 17:00–18:59
_RUSH_WINDOWS: list[tuple[int, int]] = [(7, 8), (17, 18)]

# NYC TLC medians (2023), used at inference when no DB lookup is available.
_ZONE_FALLBACKS: dict[str, float] = {
    "fare_per_mile": 5.00,
    "zone_avg_fare": 13.50,
    "zone_trip_count": 1_000.0,
    "avg_speed_zone": 12.0,
}


def _as_array(x: Union[int, float, np.ndarray, pd.Series]) -> np.ndarray:
    """Convert any numeric input to a float64 ndarray without copying if possible."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(x, dtype=float)


def _restore_type(
    result: np.ndarray,
    reference: Union[int, float, np.ndarray, pd.Series],
) -> Union[int, float, np.ndarray, pd.Series]:
    """Return *result* in the same container type as *reference*.

    Scalars come back as Python int/float; Series preserve their index.
    """
    if isinstance(reference, pd.Series):
        return pd.Series(result, index=reference.index)
    if np.ndim(reference) == 0:
        return result.item()
    return result


def haversine_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute great-circle distance in kilometres using the haversine formula.

    Uses the half-versine identity rather than the spherical law of cosines;
    the latter loses ~4 significant figures for distances below 1 km, which
    covers the majority of Manhattan trips.

    NaN in any coordinate propagates to NaN output without raising.  The
    ``np.clip`` before ``arcsin`` absorbs floating-point rounding that would
    otherwise push the argument fractionally above 1.0 and trigger a warning.

    Parameters
    ----------
    lat1 : float or np.ndarray
        Pickup latitude in decimal degrees.
    lon1 : float or np.ndarray
        Pickup longitude in decimal degrees.
    lat2 : float or np.ndarray
        Dropoff latitude in decimal degrees.
    lon2 : float or np.ndarray
        Dropoff longitude in decimal degrees.

    Returns
    -------
    float or np.ndarray
        Great-circle distance in kilometres.  Shape matches the broadcast
        shape of the inputs; scalar inputs yield a scalar float.
    """
    lat1_r = np.deg2rad(lat1)
    lat2_r = np.deg2rad(lat2)
    dlat   = np.deg2rad(lat2 - lat1)
    dlon   = np.deg2rad(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )

    result = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # Return Python float for scalar inputs so callers don't get 0-d arrays.
    if np.ndim(lat1) == 0 and np.ndim(lat2) == 0:
        return float(result)
    return result


# Public alias matching the name used in the user-facing API docs.
haversine_distance = haversine_km


def rush_hour_flag(
    hour: Union[int, np.ndarray, pd.Series],
    day_of_week: Union[int, np.ndarray, pd.Series],
) -> Union[int, np.ndarray, pd.Series]:
    """Return 1 if *hour* falls within a weekday rush window, else 0.

    Rush windows mirror ``feature_queries.sql``:
    AM peak 07:00–08:59, PM peak 17:00–18:59, weekdays only (Mon–Fri).

    Parameters
    ----------
    hour : int or array-like
        Hour of day, 0–23.
    day_of_week : int or array-like
        ISO weekday where 0 = Monday and 6 = Sunday (pandas convention).

    Returns
    -------
    int or np.ndarray or pd.Series
        Binary rush-hour indicator.  Return type matches *hour*.
    """
    h   = _as_array(hour)
    dow = _as_array(day_of_week)

    is_weekday = dow < 5
    in_rush    = np.zeros_like(h, dtype=bool)
    for start, end in _RUSH_WINDOWS:
        in_rush |= (h >= start) & (h <= end)

    result = (is_weekday & in_rush).astype(np.int8)
    return _restore_type(result, hour)


def encode_day_of_week(
    day_of_week: Union[int, np.ndarray, pd.Series],
) -> tuple[
    Union[float, np.ndarray, pd.Series],
    Union[float, np.ndarray, pd.Series],
]:
    """Cyclic sin/cos encoding of day-of-week (period = 7).

    Encodes ordinality so that Sunday (6) and Monday (0) are adjacent in the
    encoded space, avoiding the boundary discontinuity in linear models.

    Parameters
    ----------
    day_of_week : int or array-like
        ISO weekday, 0–6.

    Returns
    -------
    tuple of (sin_val, cos_val)
        Both in ``[-1, 1]``.  ``sin² + cos² == 1`` for all inputs.
        Return types match *day_of_week*.
    """
    arr = _as_array(day_of_week)
    angle = 2.0 * np.pi * arr / 7.0
    sin_v = np.sin(angle)
    cos_v = np.cos(angle)
    return _restore_type(sin_v, day_of_week), _restore_type(cos_v, day_of_week)


def encode_hour(
    hour: Union[int, np.ndarray, pd.Series],
) -> tuple[
    Union[float, np.ndarray, pd.Series],
    Union[float, np.ndarray, pd.Series],
]:
    """Cyclic sin/cos encoding of hour-of-day (period = 24).

    Ensures 23:00 and 00:00 are represented as neighbouring points rather
    than opposite ends of a linear scale.

    Parameters
    ----------
    hour : int or array-like
        Hour of day, 0–23.

    Returns
    -------
    tuple of (sin_val, cos_val)
        Both in ``[-1, 1]``.  ``sin² + cos² == 1`` for all inputs.
        Return types match *hour*.
    """
    arr = _as_array(hour)
    angle = 2.0 * np.pi * arr / 24.0
    sin_v = np.sin(angle)
    cos_v = np.cos(angle)
    return _restore_type(sin_v, hour), _restore_type(cos_v, hour)


def distance_bucket(
    distance_miles: Union[float, np.ndarray, pd.Series],
) -> Union[str, np.ndarray, pd.Series]:
    """Bin trip distance (miles) into four labelled buckets.

    Thresholds match ``feature_queries.sql`` and the TLC's informal tier
    boundaries: short metered, medium, highway-capable, and airport/outer-borough.

    Parameters
    ----------
    distance_miles : float or array-like
        Trip distance in miles.

    Returns
    -------
    str or np.ndarray or pd.Series
        One of ``{'short', 'medium', 'long', 'very_long'}``.
        Return type matches *distance_miles*.
    """
    scalar_input = np.ndim(distance_miles) == 0
    arr = _as_array(distance_miles)

    result = np.select(
        [arr < 1.0, arr < 3.0, arr < 10.0],
        ["short",   "medium",  "long"],
        default="very_long",
    )

    if isinstance(distance_miles, pd.Series):
        return pd.Series(result, index=distance_miles.index, dtype=object)
    if scalar_input:
        return str(result.item())
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features from ``pickup_datetime`` and append them.

    Mirrors the ``temporal`` CTE in ``feature_queries.sql``.  The input
    DataFrame is never mutated.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``pickup_datetime`` column parseable by
        ``pd.to_datetime`` (strings, Timestamps, or epoch integers all work).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added (existing columns
        are preserved and unchanged):

        - ``pickup_hour``      int, 0–23
        - ``pickup_dow``       int, 0 = Monday … 6 = Sunday (ISO)
        - ``is_weekend``       int, 1 if Saturday or Sunday
        - ``is_rush_hour``     int, 1 if weekday AM/PM peak
        - ``time_of_day_sin``  float, cyclic sin of hour
        - ``time_of_day_cos``  float, cyclic cos of hour
    """
    out = df.copy()
    dt  = pd.to_datetime(df["pickup_datetime"], utc=False)

    out["pickup_hour"] = dt.dt.hour
    out["pickup_dow"]  = dt.dt.dayofweek          # pandas: 0 = Monday

    out["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    out["is_rush_hour"] = rush_hour_flag(
        out["pickup_hour"], out["pickup_dow"]
    ).astype(int)

    sin_h, cos_h = encode_hour(out["pickup_hour"])
    out["time_of_day_sin"] = sin_h.to_numpy()
    out["time_of_day_cos"] = cos_h.to_numpy()

    return out


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute haversine distance and distance bucket from coordinate columns.

    Mirrors the ``geo`` CTE in ``feature_queries.sql``.  The input DataFrame
    is never mutated.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pickup_latitude``, ``pickup_longitude``,
        ``dropoff_latitude``, ``dropoff_longitude``, and ``trip_distance``
        (odometer miles, used for bucketing).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added:

        - ``haversine_km``      float, great-circle distance in kilometres
        - ``trip_distance_km``  float, alias of ``haversine_km``
        - ``distance_bucket``   str, one of short / medium / long / very_long
    """
    out = df.copy()

    km = haversine_km(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
    )

    out["haversine_km"]     = km
    out["trip_distance_km"] = km                      # public-facing alias
    out["distance_bucket"]  = distance_bucket(df["trip_distance"])

    return out


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the model-ready feature matrix and target vector from *df*.

    Expects *df* to have already passed through ``add_time_features`` and
    ``add_geo_features`` (or to have been loaded from the ``features`` table
    via ``data_loader.load_features``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all columns listed in ``config.FEATURE_COLS``,
        ``config.CATEGORICAL_COLS``, and ``config.TARGET_COL``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns ordered as
        ``FEATURE_COLS + CATEGORICAL_COLS``.  A copy is returned; *df* is
        not mutated.
    y : pd.Series
        Target series (``fare_amount``), aligned with *X* by index.

    Raises
    ------
    KeyError
        If any required column is absent from *df*.
    """
    all_feature_cols = FEATURE_COLS + CATEGORICAL_COLS

    missing_features = [c for c in all_feature_cols if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"DataFrame is missing required feature columns: {missing_features}"
        )
    if TARGET_COL not in df.columns:
        raise KeyError(f"DataFrame is missing target column '{TARGET_COL}'")

    X = df[all_feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_feature_row(trip: dict) -> dict:
    """Convert a single raw trip dict into a flat feature dict for inference.

    Zone-aggregate features (``zone_avg_fare``, ``zone_trip_count``,
    ``avg_speed_zone``, ``fare_per_mile``) are not computable from a single
    trip record; they are filled with NYC TLC medians from ``_ZONE_FALLBACKS``
    when not supplied in *trip*.

    Parameters
    ----------
    trip : dict
        Must contain:

        - ``pickup_datetime``   str or Timestamp
        - ``dropoff_datetime``  str or Timestamp
        - ``pickup_latitude``   float
        - ``pickup_longitude``  float
        - ``dropoff_latitude``  float
        - ``dropoff_longitude`` float
        - ``trip_distance``     float, odometer miles

        Optional:

        - ``passenger_count``   int  (default 1)
        - ``pickup_zone_id``    int  (unused at row level; kept for schema compat)
        - any key in ``_ZONE_FALLBACKS`` to override the median fallback

    Returns
    -------
    dict
        Keys match ``config.FEATURE_COLS + config.CATEGORICAL_COLS`` exactly.
    """
    pickup_dt  = pd.Timestamp(trip["pickup_datetime"])
    dropoff_dt = pd.Timestamp(trip["dropoff_datetime"])

    hour = pickup_dt.hour
    dow  = pickup_dt.dayofweek          # 0 = Monday

    h_sin, h_cos = encode_hour(hour)

    trip_dist_mi = float(trip["trip_distance"])
    dist_km      = haversine_km(
        trip["pickup_latitude"],  trip["pickup_longitude"],
        trip["dropoff_latitude"], trip["dropoff_longitude"],
    )

    duration_h = (dropoff_dt - pickup_dt).total_seconds() / 3_600.0
    if duration_h > 0 and trip_dist_mi > 0:
        speed = min(trip_dist_mi / duration_h, 80.0)
    else:
        speed = float("nan")

    return {
        "passenger_count":   int(trip.get("passenger_count", 1)),
        "trip_distance":     trip_dist_mi,
        "pickup_hour":       hour,
        "pickup_dow":        dow,
        "is_weekend":        int(dow >= 5),
        "is_rush_hour":      rush_hour_flag(hour, dow),
        "time_of_day_sin":   float(h_sin),
        "time_of_day_cos":   float(h_cos),
        "haversine_km":      dist_km,
        "fare_per_mile":     float(trip.get("fare_per_mile",    _ZONE_FALLBACKS["fare_per_mile"])),
        "zone_avg_fare":     float(trip.get("zone_avg_fare",    _ZONE_FALLBACKS["zone_avg_fare"])),
        "zone_trip_count":   float(trip.get("zone_trip_count",  _ZONE_FALLBACKS["zone_trip_count"])),
        "avg_speed_zone":    float(trip.get("avg_speed_zone",   _ZONE_FALLBACKS["avg_speed_zone"])),
        "distance_bucket":   distance_bucket(trip_dist_mi),
    }
