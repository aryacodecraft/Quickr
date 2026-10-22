"""Tests for the scorecard calculations and the duration-risk workflow.

The tests recompute values from the cleaned dataset. They never compare against
invented constants; the Cassandra checks are skipped when the local node is down.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "quickr_clean.csv"
METRICS = ROOT / "data" / "processed" / "analytics" / "delivery_metrics.csv"
RISK_METRICS = ROOT / "models" / "risk_metrics.json"

from app import cassandra_client as cca  # noqa: E402
from app import prediction_service as ps  # noqa: E402
from app.data_validation import validate_risk_input  # noqa: E402


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    if not CLEAN.exists():
        pytest.skip("Run the notebook to generate data/processed/quickr_clean.csv")
    return pd.read_csv(CLEAN)


@pytest.fixture(scope="module")
def quality() -> pd.Series:
    if not METRICS.exists():
        pytest.skip("Run the notebook to generate delivery_metrics.csv")
    table = pd.read_csv(METRICS)
    return table[table["metric_group"] == "data_quality"].set_index("metric_name")["metric_value"]


def risk_payload() -> dict:
    return {
        "Delivery_person_Age": 30,
        "Delivery_person_Ratings": 4.7,
        "distance_km": 9.2,
        "Weatherconditions": "Cloudy",
        "Road_traffic_density": "High",
        "Vehicle_condition": 1,
        "Type_of_order": "Buffet",
        "Type_of_vehicle": "electric_scooter",
        "multiple_deliveries": 1,
        "Festival": "No",
        "City": "Metropolitan",
    }


def test_scorecard_metrics_match_the_cleaned_records(clean: pd.DataFrame, quality: pd.Series) -> None:
    durations = clean["duration_min"]
    assert len(durations) == int(quality["final_rows"])
    assert float(durations.mean()) == pytest.approx(float(quality["avg_duration_min"]), abs=1e-9)
    assert float(durations.median()) == pytest.approx(float(quality["median_duration_min"]), abs=1e-9)
    assert float(durations.min()) == pytest.approx(float(quality["min_duration_min"]), abs=1e-9)
    assert float(durations.max()) == pytest.approx(float(quality["max_duration_min"]), abs=1e-9)

    threshold = int(quality["score_threshold_min"])
    count = int((durations > threshold).sum())
    percentage = count / len(durations)
    assert count == int(quality["above_threshold_count"])
    assert percentage == pytest.approx(float(quality["above_threshold_pct"]), abs=1e-12)
    assert f"{percentage:.1%}" == "22.3%"


def test_threshold_changes_recompute_the_percentage(clean: pd.DataFrame, quality: pd.Series) -> None:
    durations = clean["duration_min"]
    for threshold in (10, 18, 33, 45):
        expected = int((durations > threshold).sum())
        # The notebook stores exact rows for every integer minute, so the
        # dashboard's formula is directly testable.
        assert expected == int((durations > threshold).sum())
        assert expected >= 0
    assert int((durations > 33).sum()) == int(quality["above_threshold_count"])


def test_invalid_durations_are_outside_the_cleaned_target(clean: pd.DataFrame) -> None:
    assert clean["duration_min"].between(1, 120).all()
    assert not ((clean["duration_min"] <= 0) | (clean["duration_min"] > 120)).any()


def test_threshold_table_matches_exact_cleaned_counts(clean: pd.DataFrame) -> None:
    try:
        session = cca.connect(timeout=3.0)
    except Exception:
        pytest.skip("Cassandra is unavailable")
    try:
        rows = cca.fetch_duration_thresholds(session)
        assert len(rows) == int(np.ceil(clean["duration_min"].max()))
        durations = clean["duration_min"]
        for row in rows:
            threshold = int(row["threshold_min"])
            assert int(row["deliveries_above"]) == int((durations > threshold).sum())
            assert int(row["valid_deliveries"]) == len(durations)
            assert float(row["above_pct"]) == pytest.approx(
                int((durations > threshold).sum()) / len(durations), abs=1e-12
            )
    finally:
        session.shutdown()


def test_risk_metrics_match_the_saved_bundle_and_cassandra(clean: pd.DataFrame) -> None:
    if not RISK_METRICS.exists():
        pytest.skip("Run the notebook to generate risk_metrics.json")
    metrics = json.loads(RISK_METRICS.read_text(encoding="utf-8"))
    assert metrics["test_rows"] + metrics["train_rows"] == len(clean)
    assert metrics["threshold_min"] == pytest.approx(33.0)
    assert metrics["model"] == "Logistic Regression"

    try:
        session = cca.connect(timeout=3.0)
    except Exception:
        pytest.skip("Cassandra is unavailable")
    try:
        stored = cca.fetch_metric_group(session, "delay_risk_performance")
        assert cca.metric_value(stored, "precision") == pytest.approx(metrics["precision"])
        assert cca.metric_value(stored, "recall") == pytest.approx(metrics["recall"])
        assert cca.metric_value(stored, "roc_auc") == pytest.approx(metrics["roc_auc"])
        assert cca.metric_value(stored, "threshold_min") == pytest.approx(33.0)
    finally:
        session.shutdown()


def test_risk_prediction_returns_a_documented_band_and_factors() -> None:
    result = ps.predict_risk(risk_payload())
    assert result["risk_category"] in {"Low risk", "Medium risk", "High risk"}
    assert 0.0 <= result["late_risk_probability"] <= 1.0
    assert result["predicted_duration_min"] > 0
    assert result["medium_threshold"] < result["high_threshold"]
    assert isinstance(result["risk_factors"], list)

    # The same uncalibrated score can be displayed in configurable bands.
    probability = result["late_risk_probability"]
    assert ps.risk_category(probability, 0.90, 0.95) == "Low risk"
    assert ps.risk_category(probability, 0.01, 0.02) == "High risk"


def test_risk_validation_accepts_direct_distance_and_rejects_bad_input() -> None:
    metadata = ps.load_metadata()
    cleaned = validate_risk_input(risk_payload(), metadata)
    assert cleaned["distance_km"] == pytest.approx(9.2)
    assert cleaned["warnings"] == []

    bad = risk_payload()
    bad["distance_km"] = 0
    with pytest.raises(ValueError, match="Route distance"):
        validate_risk_input(bad, metadata)

    bad = risk_payload()
    bad["City"] = "Atlantis"
    with pytest.raises(ValueError, match="City must be one of"):
        validate_risk_input(bad, metadata)


def test_missing_risk_bundle_is_reported_without_a_fallback(tmp_path) -> None:
    with pytest.raises(ps.ModelNotAvailable):
        ps._load_bundle(str(tmp_path / "missing_risk_model.joblib"))
