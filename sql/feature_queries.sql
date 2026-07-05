-- Feature engineering pipeline — NYC TLC Taxi Fare Prediction.
-- Reads raw trips, applies a quality gate, derives all model features through
-- a chain of CTEs, and inserts one row per qualified trip into `features`.
--
-- Prerequisites : create_tables.sql executed; `trips` table populated.
-- Idempotency   : DELETE FROM features; before re-running.
-- SQLite >= 3.35 required for CTE-prefixed INSERT and window functions.

WITH

-- Quality gate applied before any computation so downstream CTEs never see
-- nonsense values. Min fare = $3.00 base + $0.50 improvement surcharge;
-- $500 cap removes limo flat-rates that skew fare-per-mile distributions.
-- Zone IDs (not lat/lon) are used for the geo gate: TLC retroactively unified
-- all historical trip files, including 2015, onto the zone-ID-only schema, so
-- raw coordinates aren't present in the source parquet. Valid zone IDs are
-- 1-263 (264/265 are "Unknown"/"Outside NYC" placeholders, excluded here).
filtered AS (
    SELECT *
    FROM   trips
    WHERE  fare_amount      BETWEEN 3.50 AND 500.00
      AND  trip_distance    BETWEEN 0.10 AND 100.0
      AND  passenger_count  BETWEEN 1    AND 6
      AND  dropoff_datetime > pickup_datetime
      AND  pickup_zone_id   BETWEEN 1 AND 263
      AND  dropoff_zone_id  BETWEEN 1 AND 263
),

-- pickup_dow uses the ISO weekday convention (0 = Monday) via a modular shift
-- from SQLite's Sunday-origin %w output, kept consistent with pandas
-- dt.dayofweek in the Python feature pipeline. Rush windows: AM 07:00-08:59,
-- PM 17:00-18:59 on weekdays. Cyclic sin/cos encoding of hour avoids the
-- 23 -> 0 discontinuity showing up as a large step in linear models.
temporal AS (
    SELECT
        *,

        CAST(strftime('%H', pickup_datetime) AS INTEGER)
            AS pickup_hour,

        CAST((CAST(strftime('%w', pickup_datetime) AS INTEGER) + 6) % 7 AS INTEGER)
            AS pickup_dow,

        CASE WHEN strftime('%w', pickup_datetime) IN ('0', '6') THEN 1 ELSE 0 END
            AS is_weekend,

        CASE
            WHEN strftime('%w', pickup_datetime) NOT IN ('0', '6')
             AND (    CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 7  AND 8
                  OR  CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 17 AND 18)
            THEN 1 ELSE 0
        END
            AS is_rush_hour,

        SIN(2.0 * 3.14159265358979 * CAST(strftime('%H', pickup_datetime) AS REAL) / 24.0)
            AS time_of_day_sin,

        COS(2.0 * 3.14159265358979 * CAST(strftime('%H', pickup_datetime) AS REAL) / 24.0)
            AS time_of_day_cos

    FROM filtered
),

-- Haversine great-circle distance via the half-versine identity (chosen over
-- the spherical law of cosines, which loses ~4 decimal places of precision
-- for distances under 1 km — exactly the range that dominates city-centre
-- trips). Falls back to trip_distance (miles -> km) when coordinates are
-- NULL, which is always true for the current zone-ID-only source; this keeps
-- the column meaningful without lat/lon, and reverts to true great-circle
-- distance automatically if a future source restores coordinates.
geo AS (
    SELECT
        *,

        CASE
            WHEN pickup_latitude  IS NOT NULL AND pickup_longitude  IS NOT NULL
             AND dropoff_latitude IS NOT NULL AND dropoff_longitude IS NOT NULL
            THEN 2.0 * ASIN(SQRT(
                  SIN((dropoff_latitude  - pickup_latitude)  * 3.14159265358979 / 360.0)
                * SIN((dropoff_latitude  - pickup_latitude)  * 3.14159265358979 / 360.0)
                + COS(pickup_latitude   * 3.14159265358979 / 180.0)
                * COS(dropoff_latitude  * 3.14159265358979 / 180.0)
                * SIN((dropoff_longitude - pickup_longitude) * 3.14159265358979 / 360.0)
                * SIN((dropoff_longitude - pickup_longitude) * 3.14159265358979 / 360.0)
            )) * 6371.0
            ELSE trip_distance * 1.60934
        END
            AS haversine_km,

        CASE
            WHEN trip_distance <  1.0  THEN 'short'
            WHEN trip_distance <  3.0  THEN 'medium'
            WHEN trip_distance < 10.0  THEN 'long'
            ELSE                            'very_long'
        END
            AS distance_bucket

    FROM temporal
),

-- speed_mph is capped at 80 mph: values above that are GPS jitter artefacts
-- from sub-minute duration estimates on long trips, not real velocity.
-- NULLIF guards the degenerate duration = 0 case that slips past the
-- dropoff > pickup filter when timestamps share the same second.
trip_metrics AS (
    SELECT
        *,
        (JULIANDAY(dropoff_datetime) - JULIANDAY(pickup_datetime)) * 24.0
            AS duration_hours,

        fare_amount / NULLIF(trip_distance, 0)
            AS fare_per_mile,

        MIN(
            trip_distance
              / NULLIF(
                    (JULIANDAY(dropoff_datetime) - JULIANDAY(pickup_datetime)) * 24.0,
                    0
                ),
            80.0
        )
            AS speed_mph

    FROM geo
),

-- avg_speed_zone (AVG(speed_mph) within pickup_zone_id + pickup_hour) encodes
-- congestion: e.g. Midtown at 17:00 with avg_speed_zone = 5 mph signals
-- gridlock and predicts a longer, pricier ride than raw distance suggests.
-- zone_avg_fare / zone_trip_count give demand-density priors per zone without
-- leaking the individual trip's own target.
zone_aggregates AS (
    SELECT
        *,

        AVG(speed_mph) OVER (PARTITION BY pickup_zone_id, pickup_hour)
            AS avg_speed_zone,

        AVG(fare_amount) OVER (PARTITION BY pickup_zone_id)
            AS zone_avg_fare,

        COUNT(*)         OVER (PARTITION BY pickup_zone_id)
            AS zone_trip_count

    FROM trip_metrics
)

-- Select exactly the columns declared in `features`; no intermediate
-- computation columns pass through.
INSERT INTO features (
    trip_id,
    fare_amount,
    passenger_count,
    trip_distance,
    pickup_zone_id,
    dropoff_zone_id,
    pickup_hour,
    pickup_dow,
    is_weekend,
    is_rush_hour,
    time_of_day_sin,
    time_of_day_cos,
    haversine_km,
    distance_bucket,
    fare_per_mile,
    zone_avg_fare,
    zone_trip_count,
    avg_speed_zone
)
SELECT
    trip_id,
    fare_amount,
    passenger_count,
    trip_distance,
    pickup_zone_id,
    dropoff_zone_id,
    pickup_hour,
    pickup_dow,
    is_weekend,
    is_rush_hour,
    time_of_day_sin,
    time_of_day_cos,
    haversine_km,
    distance_bucket,
    fare_per_mile,
    zone_avg_fare,
    zone_trip_count,
    avg_speed_zone
FROM zone_aggregates;
