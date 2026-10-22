-- Quickr — Apache Cassandra schema
-- Development only: single node, replication factor 1.

CREATE KEYSPACE IF NOT EXISTS quickr
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE quickr;

-- Dashboard query: all categories of one type
-- (traffic, weather, vehicle, city, distance band, partner age/rating band).
CREATE TABLE IF NOT EXISTS delivery_summary_by_category (
    category_type      text,
    category_value     text,
    deliveries         bigint,
    avg_duration_min   double,
    median_duration_min double,
    p90_duration_min   double,
    late_rate          double,
    sort_order         int,
    updated_at         timestamp,
    PRIMARY KEY ((category_type), category_value)
) WITH CLUSTERING ORDER BY (category_value ASC);

-- Pre-aggregated cross-filter cube for the Analytics page.
-- Partitioned by traffic level so the dashboard queries 1–4 bounded partitions,
-- then filters weather / vehicle / city client-side. Histograms are 2-minute bins.
CREATE TABLE IF NOT EXISTS delivery_segment_summary (
    traffic_level      text,
    weather            text,
    vehicle_type       text,
    city               text,
    deliveries         bigint,
    avg_duration_min   double,
    median_duration_min double,
    p90_duration_min   double,
    late_rate          double,
    histogram          map<int, int>,
    updated_at         timestamp,
    PRIMARY KEY ((traffic_level), weather, vehicle_type, city)
) WITH CLUSTERING ORDER BY (weather ASC, vehicle_type ASC, city ASC);

-- Duration distribution (dashboard chart), one row per 2-minute bucket.
CREATE TABLE IF NOT EXISTS delivery_duration_distribution (
    bucket_start_min int PRIMARY KEY,
    bucket_label     text,
    delivery_count   bigint,
    updated_at       timestamp
);

-- Exact counts for the configurable duration threshold on the scorecard.
-- One small row per integer minute; 2-minute buckets cannot answer an
-- arbitrary threshold exactly, so this table is intentionally separate.
CREATE TABLE IF NOT EXISTS delivery_duration_thresholds (
    threshold_min      int PRIMARY KEY,
    deliveries_above   bigint,
    valid_deliveries   bigint,
    above_pct          double,
    updated_at         timestamp
);

-- Small metric store: model scores, data-quality counters, dataset notes.
CREATE TABLE IF NOT EXISTS delivery_metrics (
    metric_group  text,
    metric_name   text,
    metric_value  double,
    metric_text   text,
    model_name    text,
    model_version text,
    evaluated_at  timestamp,
    PRIMARY KEY ((metric_group), metric_name)
);

-- Predictions are partitioned by day (bounded partitions) and ordered by time.
CREATE TABLE IF NOT EXISTS delivery_predictions (
    prediction_date          date,
    created_at               timestamp,
    prediction_id            uuid,
    predicted_duration_min   double,
    late_risk_probability    double,
    risk_category            text,
    model_name               text,
    model_version            text,
    traffic_level            text,
    weather                  text,
    vehicle_type             text,
    city                     text,
    distance_km              double,
    input_json               text,
    PRIMARY KEY ((prediction_date), created_at, prediction_id)
) WITH CLUSTERING ORDER BY (created_at DESC);
