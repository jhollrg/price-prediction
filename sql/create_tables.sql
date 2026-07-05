-- Raw trips table: one row per TLC taxi trip
CREATE TABLE IF NOT EXISTS trips (
    trip_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_id        INTEGER,
    pickup_datetime  TEXT NOT NULL,
    dropoff_datetime TEXT NOT NULL,
    passenger_count  INTEGER,
    pickup_longitude REAL,
    pickup_latitude  REAL,
    dropoff_longitude REAL,
    dropoff_latitude  REAL,
    pickup_zone_id   INTEGER,
    dropoff_zone_id  INTEGER,
    trip_distance    REAL,
    fare_amount      REAL NOT NULL,
    tip_amount       REAL,
    total_amount     REAL,
    payment_type     INTEGER
);

-- Engineered features table: one row per trip, joined from trips
CREATE TABLE IF NOT EXISTS features (
    trip_id              INTEGER PRIMARY KEY REFERENCES trips(trip_id),

    -- Target
    fare_amount          REAL NOT NULL,

    -- Raw pass-throughs
    passenger_count      INTEGER,
    trip_distance        REAL,
    pickup_zone_id       INTEGER,
    dropoff_zone_id      INTEGER,

    -- Temporal encodings
    pickup_hour          INTEGER,   -- 0-23
    pickup_dow           INTEGER,   -- 0=Monday … 6=Sunday
    is_weekend           INTEGER,   -- 0/1
    is_rush_hour         INTEGER,   -- 0/1
    time_of_day_sin      REAL,      -- cyclic sin encoding of hour
    time_of_day_cos      REAL,      -- cyclic cos encoding of hour

    -- Distance features
    haversine_km         REAL,
    distance_bucket      TEXT,      -- short/medium/long/very_long

    -- Fare normalisation
    fare_per_mile        REAL,      -- fare_amount / trip_distance; NULL when distance = 0

    -- Zone-level aggregates (window stats over the full batch)
    zone_avg_fare        REAL,
    zone_trip_count      INTEGER,

    -- Congestion proxy: avg speed in mph within (pickup_zone, pickup_hour) cell
    avg_speed_zone       REAL
);
