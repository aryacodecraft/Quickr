"""Unit tests for the dashboard's Cassandra access helpers.

These tests do not require a running Cassandra node. A separate live test skips
automatically when the local Docker container is unavailable.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest

from app import cassandra_client as cca


def _segment(**overrides):
    row = {
        "traffic_level": "Low",
        "weather": "Sunny",
        "vehicle_type": "motorcycle",
        "city": "Metropolitan",
        "deliveries": 100,
        "avg_duration_min": 20.0,
        "median_duration_min": 19.0,
        "p90_duration_min": 27.0,
        "late_rate": 0.1,
        "histogram": {10: 10, 20: 90},
    }
    row.update(overrides)
    return row


def test_filter_segments_applies_all_filters_and_all_keyword() -> None:
    rows = [
        _segment(),
        _segment(weather="Fog"),
        _segment(traffic_level="Jam", city="Urban"),
    ]
    assert len(cca.filter_segments(rows)) == 3
    assert len(cca.filter_segments(rows, traffic="All", city="All")) == 3
    assert len(cca.filter_segments(rows, weather="Fog")) == 1
    assert len(cca.filter_segments(rows, traffic="Jam", city="Urban")) == 1
    assert cca.filter_segments(rows, traffic="High") == []


def test_percentile_from_histogram_is_monotone() -> None:
    histogram = {10: 10, 12: 10, 20: 30, 40: 50}
    median = cca.percentile_from_histogram(histogram, 0.5)
    p90 = cca.percentile_from_histogram(histogram, 0.9)
    assert median is not None and p90 is not None
    assert median < p90
    assert cca.percentile_from_histogram({}, 0.5) is None
    assert cca.percentile_from_histogram({10: 0}, 0.5) is None


def test_aggregate_segments_weights_by_delivery_count() -> None:
    rows = [
        _segment(deliveries=100, avg_duration_min=20.0, late_rate=0.1, histogram={10: 100}),
        _segment(deliveries=300, avg_duration_min=40.0, late_rate=0.3, histogram={40: 300}),
    ]
    result = cca.aggregate_segments(rows)
    assert result["deliveries"] == 400
    assert result["avg_duration_min"] == pytest.approx(35.0)
    assert result["late_rate"] == pytest.approx(0.25)
    assert result["histogram"] == {10: 100, 40: 300}
    assert result["median_duration_min"] == 41.0


def test_aggregate_segments_handles_an_empty_filter() -> None:
    result = cca.aggregate_segments([])
    assert result["deliveries"] == 0
    assert result["avg_duration_min"] is None
    assert result["histogram"] == {}


def test_row_to_dict_serialises_uuid_and_dates() -> None:
    identifier = uuid.uuid4()
    created = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    row = {
        "prediction_id": identifier,
        "created_at": created,
        "prediction_date": date(2026, 1, 2),
        "predicted_duration_min": 24.5,
    }

    class Row(dict):
        pass

    converted = cca.row_to_dict(Row(row))
    assert converted["prediction_id"] == str(identifier)
    assert converted["created_at"] == created.isoformat()
    assert converted["prediction_date"] == "2026-01-02"
    assert converted["predicted_duration_min"] == 24.5
    assert cca.row_to_dict(None) == {}


def test_make_prediction_record_preserves_inputs_and_types() -> None:
    result = {
        "predicted_duration_min": 24.5,
        "late_risk_probability": 0.42,
        "risk_category": "Medium risk",
        "model_name": "Random Forest",
        "model_version": "1.0.0",
        "distance_km": 3.2,
    }
    inputs = {
        "Road_traffic_density": "Low",
        "Weatherconditions": "Sunny",
        "Type_of_vehicle": "motorcycle",
        "City": "Metropolitan",
    }
    record = cca.make_prediction_record(result, inputs)
    assert isinstance(record["prediction_id"], uuid.UUID)
    assert isinstance(record["prediction_date"], date)
    assert record["prediction_date"] == datetime.now(timezone.utc).date()
    assert record["predicted_duration_min"] == 24.5
    assert record["traffic_level"] == "Low"
    assert "Sunny" in record["input_json"]


def test_metric_accessors_handle_missing_rows() -> None:
    group = {
        "final_rows": {"metric_value": 40016.0, "metric_text": None},
        "best_model": {"metric_value": 3.82, "metric_text": "Random Forest"},
    }
    assert cca.metric_value(group, "final_rows") == 40016.0
    assert cca.metric_value(group, "missing") is None
    assert cca.metric_value(group, "missing", 0.0) == 0.0
    assert cca.metric_text(group, "best_model") == "Random Forest"
    assert cca.metric_text(group, "missing", "fallback") == "fallback"


def test_live_cassandra_round_trip_when_available() -> None:
    try:
        session = cca.connect(timeout=3.0)
    except cca.CassandraUnavailable:
        pytest.skip("Local Cassandra is not running; start it with docker compose up -d")
    try:
        version = cca.check_connection(session)
        assert version
        counts = cca.table_counts(session)
        assert set(counts) == {
            "delivery_summary_by_category",
            "delivery_segment_summary",
            "delivery_duration_distribution",
            "delivery_duration_thresholds",
            "delivery_metrics",
            "delivery_predictions",
        }
        metrics = cca.fetch_metrics(session, groups=("data_quality",))
        assert "final_rows" in metrics["data_quality"]
        thresholds = cca.fetch_duration_thresholds(session)
        assert thresholds, "the notebook should publish exact duration-threshold rows"
        assert all(int(row["threshold_min"]) >= 1 for row in thresholds)
    finally:
        session.shutdown()
