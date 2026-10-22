"""Thin, query-oriented access layer for the Quickr dashboard."""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT, Session
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import dict_factory

try:
    from dotenv import load_dotenv
except ImportError:  # the Docker pipeline image does not need .env support
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "127.0.0.1")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9043"))
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "quickr")
CASSANDRA_DATACENTER = os.getenv("CASSANDRA_DATACENTER", "datacenter1")

SUMMARY_TABLE = "delivery_summary_by_category"
SEGMENT_TABLE = "delivery_segment_summary"
DISTRIBUTION_TABLE = "delivery_duration_distribution"
THRESHOLD_TABLE = "delivery_duration_thresholds"
METRICS_TABLE = "delivery_metrics"
PREDICTIONS_TABLE = "delivery_predictions"

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]
METRIC_GROUPS = (
    "data_quality",
    "missing_values",
    "missing_values_after_cleaning",
    "model_performance",
    "delay_risk_performance",
    "dataset",
    "app_meta",
)


class CassandraUnavailable(RuntimeError):
    """Raised when the local Cassandra development node cannot be reached."""


def config() -> dict[str, Any]:
    return {
        "host": CASSANDRA_HOST,
        "port": CASSANDRA_PORT,
        "keyspace": CASSANDRA_KEYSPACE,
        "datacenter": CASSANDRA_DATACENTER,
    }


def connect(timeout: float = 8.0) -> Session:
    """Connect to the local keyspace. Raises CassandraUnavailable on failure."""
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=CASSANDRA_DATACENTER),
        consistency_level=ConsistencyLevel.LOCAL_ONE,
        row_factory=dict_factory,
        request_timeout=20,
    )
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        connect_timeout=timeout,
        control_connection_timeout=timeout,
        protocol_version=4,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    )
    try:
        session = cluster.connect(CASSANDRA_KEYSPACE)
        return session
    except Exception as exc:  # driver exceptions are not all the same type
        cluster.shutdown()
        raise CassandraUnavailable(
            f"Could not connect to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT} "
            f"(keyspace '{CASSANDRA_KEYSPACE}'). Is Docker running and has "
            f"scripts/init_cassandra.py been executed? Details: {exc}"
        ) from exc


def check_connection(session: Session) -> str:
    row = session.execute("SELECT release_version FROM system.local").one()
    return str(row["release_version"]) if row else "unknown"


def row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a driver row into a JSON-friendly dict."""
    if row is None:
        return {}
    data = dict(row)
    for key, value in list(data.items()):
        if isinstance(value, uuid.UUID):
            data[key] = str(value)
        elif isinstance(value, (date, datetime)):
            data[key] = value.isoformat()
    return data


def _prepare(session: Session, cql: str) -> Any:
    """Prepare a CQL statement, cached for the lifetime of this session.

    The cache is attached to the session object rather than keyed by ``id()``:
    Python can reuse an object id after a session is shut down, which would
    otherwise return a prepared statement that belongs to a different cluster.
    """
    cache = getattr(session, "_quickr_prepared", None)
    if cache is None:
        cache = {}
        setattr(session, "_quickr_prepared", cache)
    if cql not in cache:
        cache[cql] = session.prepare(cql)
    return cache[cql]


def fetch_summary(session: Session, category_type: str) -> list[dict[str, Any]]:
    cql = (
        f"SELECT category_type, category_value, deliveries, avg_duration_min, "
        f"median_duration_min, p90_duration_min, late_rate, sort_order, updated_at "
        f"FROM {CASSANDRA_KEYSPACE}.{SUMMARY_TABLE} WHERE category_type = ?"
    )
    rows = session.execute(_prepare(session, cql), (category_type,))
    return sorted([row_to_dict(row) for row in rows], key=lambda r: r["category_value"] or "")


def fetch_all_summaries(session: Session) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for category_type in (
        "traffic",
        "weather",
        "vehicle",
        "city",
        "order_type",
        "festival",
        "distance_band",
        "age_band",
        "rating_band",
    ):
        result[category_type] = fetch_summary(session, category_type)
    return result


def fetch_duration_distribution(session: Session) -> list[dict[str, Any]]:
    cql = (
        f"SELECT bucket_start_min, bucket_label, delivery_count, updated_at "
        f"FROM {CASSANDRA_KEYSPACE}.{DISTRIBUTION_TABLE}"
    )
    rows = session.execute(cql)
    return sorted([row_to_dict(row) for row in rows], key=lambda r: r["bucket_start_min"])


def fetch_duration_thresholds(session: Session) -> list[dict[str, Any]]:
    """Exact above-threshold counts written by the notebook."""
    cql = (
        f"SELECT threshold_min, deliveries_above, valid_deliveries, above_pct, updated_at "
        f"FROM {CASSANDRA_KEYSPACE}.{THRESHOLD_TABLE}"
    )
    rows = session.execute(cql)
    return sorted([row_to_dict(row) for row in rows], key=lambda r: int(r["threshold_min"]))


def fetch_segment_summary(
    session: Session, traffic_level: str | None = None
) -> list[dict[str, Any]]:
    """Query one traffic partition, or each of the four bounded partitions.

    The segment cube has at most a few hundred rows. The dashboard needs every
    filter combination, so querying the four known partitions is intentional;
    it is not an unbounded full-table scan.
    """
    cql = (
        f"SELECT traffic_level, weather, vehicle_type, city, deliveries, "
        f"avg_duration_min, median_duration_min, p90_duration_min, late_rate, "
        f"histogram, updated_at FROM {CASSANDRA_KEYSPACE}.{SEGMENT_TABLE} "
        f"WHERE traffic_level = ?"
    )
    levels = [traffic_level] if traffic_level else TRAFFIC_LEVELS
    rows: list[dict[str, Any]] = []
    for level in levels:
        rows.extend(row_to_dict(row) for row in session.execute(_prepare(session, cql), (level,)))
    return rows


def fetch_metric_group(session: Session, metric_group: str) -> dict[str, dict[str, Any]]:
    cql = (
        f"SELECT metric_group, metric_name, metric_value, metric_text, model_name, "
        f"model_version, evaluated_at FROM {CASSANDRA_KEYSPACE}.{METRICS_TABLE} "
        f"WHERE metric_group = ?"
    )
    rows = session.execute(_prepare(session, cql), (metric_group,))
    return {row["metric_name"]: row_to_dict(row) for row in rows}


def fetch_metrics(session: Session, groups: Iterable[str] = METRIC_GROUPS) -> dict[str, dict]:
    return {group: fetch_metric_group(session, group) for group in groups}


def metric_value(group: dict[str, dict], name: str, default: float | None = None) -> float | None:
    row = group.get(name)
    if not row or row.get("metric_value") is None:
        return default
    return float(row["metric_value"])


def metric_text(group: dict[str, dict], name: str, default: str = "") -> str:
    row = group.get(name)
    return str(row.get("metric_text") or default) if row else default


def make_prediction_record(result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Build an idempotency-safe prediction row (UUID prevents accidental collapse)."""
    now = datetime.now(timezone.utc)
    return {
        "prediction_date": now.date(),
        "created_at": now,
        "prediction_id": uuid.uuid4(),
        "predicted_duration_min": float(result["predicted_duration_min"]),
        "late_risk_probability": (
            float(result["late_risk_probability"])
            if result.get("late_risk_probability") is not None
            else None
        ),
        "risk_category": result.get("risk_category"),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "traffic_level": inputs.get("Road_traffic_density"),
        "weather": inputs.get("Weatherconditions"),
        "vehicle_type": inputs.get("Type_of_vehicle"),
        "city": inputs.get("City"),
        "distance_km": float(result.get("distance_km", 0.0)),
        "input_json": json.dumps(inputs, default=str, sort_keys=True),
    }


def save_prediction(session: Session, record: dict[str, Any]) -> None:
    cql = (
        f"INSERT INTO {CASSANDRA_KEYSPACE}.{PREDICTIONS_TABLE} "
        f"(prediction_date, created_at, prediction_id, predicted_duration_min, "
        f"late_risk_probability, risk_category, model_name, model_version, "
        f"traffic_level, weather, vehicle_type, city, distance_km, input_json) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    values = (
        record["prediction_date"],
        record["created_at"],
        record["prediction_id"],
        record["predicted_duration_min"],
        record["late_risk_probability"],
        record["risk_category"],
        record["model_name"],
        record["model_version"],
        record["traffic_level"],
        record["weather"],
        record["vehicle_type"],
        record["city"],
        record["distance_km"],
        record["input_json"],
    )
    session.execute(_prepare(session, cql), values)


def fetch_recent_predictions(session: Session, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch today's prediction partition, newest first (bounded by day + limit).

    ``prediction_date`` is written from the UTC clock in ``make_prediction_record``;
    the read uses the same clock so the day boundary cannot drift on a local-time
    machine such as IST.
    """
    limit = max(1, min(int(limit), 100))
    cql = (
        f"SELECT * FROM {CASSANDRA_KEYSPACE}.{PREDICTIONS_TABLE} "
        f"WHERE prediction_date = ? LIMIT {limit}"
    )
    today_utc = datetime.now(timezone.utc).date()
    rows = session.execute(_prepare(session, cql), (today_utc,))
    return [row_to_dict(row) for row in rows]


def table_counts(session: Session) -> dict[str, int]:
    counts = {}
    for table in (
        SUMMARY_TABLE,
        SEGMENT_TABLE,
        DISTRIBUTION_TABLE,
        THRESHOLD_TABLE,
        METRICS_TABLE,
        PREDICTIONS_TABLE,
    ):
        counts[table] = int(
            session.execute(f"SELECT COUNT(*) AS n FROM {CASSANDRA_KEYSPACE}.{table}").one()["n"]
        )
    return counts


def filter_segments(
    rows: Iterable[dict[str, Any]],
    traffic: str | None = None,
    weather: str | None = None,
    vehicle: str | None = None,
    city: str | None = None,
) -> list[dict[str, Any]]:
    """Apply dashboard filters to the pre-aggregated segment cube."""
    result = []
    for row in rows:
        if traffic and traffic != "All" and row.get("traffic_level") != traffic:
            continue
        if weather and weather != "All" and row.get("weather") != weather:
            continue
        if vehicle and vehicle != "All" and row.get("vehicle_type") != vehicle:
            continue
        if city and city != "All" and row.get("city") != city:
            continue
        result.append(row)
    return result


def percentile_from_histogram(histogram: dict[int, int], quantile: float) -> float | None:
    """Estimate a percentile from 2-minute bucket counts (bucket midpoint)."""
    if not histogram:
        return None
    total = sum(int(v) for v in histogram.values())
    if total <= 0:
        return None
    target = max(1.0, min(float(quantile), 1.0) * total)
    cumulative = 0
    for start in sorted(int(k) for k in histogram):
        cumulative += int(histogram[start])
        if cumulative >= target:
            return round(start + 1.0, 1)
    return None


def aggregate_segments(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Weighted totals and histogram for the currently filtered segment cube."""
    rows = list(rows)
    total = sum(int(row.get("deliveries") or 0) for row in rows)
    if total == 0:
        return {
            "deliveries": 0,
            "avg_duration_min": None,
            "median_duration_min": None,
            "p90_duration_min": None,
            "late_rate": None,
            "histogram": {},
        }
    histogram: dict[int, int] = {}
    for row in rows:
        for bucket, count in (row.get("histogram") or {}).items():
            histogram[int(bucket)] = histogram.get(int(bucket), 0) + int(count)
    avg = sum(float(row["avg_duration_min"]) * int(row["deliveries"]) for row in rows) / total
    late = sum(float(row["late_rate"]) * int(row["deliveries"]) for row in rows) / total
    return {
        "deliveries": total,
        "avg_duration_min": round(avg, 2),
        "median_duration_min": percentile_from_histogram(histogram, 0.5),
        "p90_duration_min": percentile_from_histogram(histogram, 0.9),
        "late_rate": round(late, 4),
        "histogram": histogram,
    }
