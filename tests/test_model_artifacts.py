"""Tests for the notebook's saved regression and risk artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

TIME_MODEL = MODELS / "quickr_delivery_time_model.joblib"
RISK_MODEL = MODELS / "quickr_delay_risk_model.joblib"
METADATA = MODELS / "model_metadata.json"
METRICS = MODELS / "metrics.json"
RISK_METRICS = MODELS / "risk_metrics.json"
CONFUSION = MODELS / "confusion_matrix.csv"
TEST_PREDICTIONS = MODELS / "test_predictions.csv"


@pytest.fixture(scope="module")
def metadata() -> dict:
    if not METADATA.exists():
        pytest.skip("Run the notebook to generate models/model_metadata.json")
    return json.loads(METADATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def time_bundle():
    if not TIME_MODEL.exists():
        pytest.skip("Run the notebook to generate the delivery-time model")
    return joblib.load(TIME_MODEL)


def feature_row(metadata: dict) -> dict:
    """Build one complete model row from metadata defaults."""
    defaults = metadata["default_values"]
    row = {}
    for column in metadata["feature_order"]:
        if column in defaults:
            row[column] = defaults[column]
        else:
            options = metadata["categorical_options"].get(column)
            assert options, f"no default or categorical options for {column}"
            row[column] = options[0]
    return row


@pytest.fixture(scope="module")
def risk_bundle():
    if not RISK_MODEL.exists():
        pytest.skip("Run the notebook to generate the delay-risk model")
    return joblib.load(RISK_MODEL)


def test_two_trained_models_are_published() -> None:
    if not TIME_MODEL.exists() or not RISK_MODEL.exists():
        pytest.skip("Run the notebook to generate both models")
    assert sorted(path.name for path in MODELS.glob("*.joblib")) == sorted(
        [TIME_MODEL.name, RISK_MODEL.name]
    )


def test_metadata_contract_is_complete(metadata: dict) -> None:
    for key in (
        "model_name",
        "model_version",
        "feature_order",
        "numeric_features",
        "categorical_features",
        "categorical_options",
        "numeric_ranges",
        "default_values",
        "training_rows",
        "test_rows",
        "target_unit",
        "late_threshold_min",
        "random_seed",
        "metrics",
        "split_method",
        "run_id",
    ):
        assert key in metadata, f"metadata is missing {key}"
    assert metadata["model_name"] == "Random Forest"
    assert metadata["target_unit"] == "minutes"
    assert metadata["random_seed"] == 42
    assert metadata["training_rows"] + metadata["test_rows"] > 39_000
    assert 25 <= metadata["late_threshold_min"] <= 40


def test_feature_order_matches_declared_feature_roles(metadata: dict) -> None:
    assert metadata["feature_order"] == metadata["numeric_features"] + metadata["categorical_features"]
    assert "duration_min" not in metadata["feature_order"]
    assert metadata["categorical_options"]["City"] == sorted(metadata["categorical_options"]["City"])


def test_time_model_predicts_a_plausible_duration(time_bundle, metadata) -> None:
    row = feature_row(metadata)
    frame = pd.DataFrame([row])[metadata["feature_order"]]
    prediction = float(time_bundle["pipeline"].predict(frame)[0])
    assert np.isfinite(prediction)
    assert 1.0 <= prediction <= 120.0


def test_time_model_responds_to_traffic_without_memorising(metadata) -> None:
    bundle = joblib.load(TIME_MODEL)
    pipeline = bundle["pipeline"]

    def predict(traffic: str) -> float:
        row = feature_row(metadata)
        row["Road_traffic_density"] = traffic
        return float(pipeline.predict(pd.DataFrame([row])[metadata["feature_order"]])[0])

    low, jam = predict("Low"), predict("Jam")
    assert np.isfinite(low) and np.isfinite(jam)
    assert jam > low, "the model should not predict a jam route as faster than a low-traffic route"


def test_test_predictions_match_the_holdout_size(metadata) -> None:
    if not TEST_PREDICTIONS.exists():
        pytest.skip("Run the notebook to generate test_predictions.csv")
    frame = pd.read_csv(TEST_PREDICTIONS)
    assert len(frame) == metadata["test_rows"]
    assert {"actual_min", "predicted_min"}.issubset(frame.columns)
    assert not frame[["actual_min", "predicted_min"]].isna().any().any()
    assert np.isfinite(frame[["actual_min", "predicted_min"]].to_numpy()).all()
    assert frame["predicted_min"].between(0.0, 120.0).all()
    mae = float(np.mean(np.abs(frame["actual_min"] - frame["predicted_min"])))
    assert mae == pytest.approx(metadata["metrics"]["mae"], abs=1e-6)


def test_metrics_bundle_is_consistent_when_present(metadata: dict) -> None:
    if not METRICS.exists():
        pytest.skip("Run the notebook to generate metrics.json")
    bundle = json.loads(METRICS.read_text(encoding="utf-8"))
    assert bundle["model"] == metadata["model_name"]
    for key in ("mae", "rmse", "r2", "mape"):
        assert bundle[key] == pytest.approx(metadata["metrics"][key], abs=1e-9)
    assert bundle["mae"] > 0
    assert bundle["rmse"] >= bundle["mae"]
    assert 0.0 <= bundle["r2"] <= 1.0
    assert bundle["mape"] > 0


def test_risk_bundle_contract_and_features(risk_bundle) -> None:
    metadata = risk_bundle["metadata"]
    assert metadata["model_name"] == "Logistic Regression"
    assert metadata["training_rows"] + metadata["test_rows"] > 39_000
    assert metadata["late_threshold_min"] == pytest.approx(33.0)
    assert 0.0 < metadata["medium_threshold"] < metadata["high_threshold"] < 1.0
    assert metadata["feature_order"] == metadata["numeric_features"] + metadata["categorical_features"]
    assert "duration_min" not in metadata["feature_order"]
    assert "Time_taken(min)" not in metadata["feature_order"]

    row = feature_row(metadata)
    frame = pd.DataFrame([row])[metadata["feature_order"]]
    probability = float(risk_bundle["pipeline"].predict_proba(frame)[0, 1])
    assert np.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_confusion_matrix_and_risk_metrics_match(risk_bundle) -> None:
    if not CONFUSION.exists() or not RISK_METRICS.exists():
        pytest.skip("Run the notebook to generate the risk metrics")
    cm = pd.read_csv(CONFUSION, index_col=0)
    assert cm.shape == (2, 2)
    assert int(cm.to_numpy().sum()) == risk_bundle["metadata"]["test_rows"]
    metrics = json.loads(RISK_METRICS.read_text(encoding="utf-8"))
    assert metrics["test_rows"] == risk_bundle["metadata"]["test_rows"]
    assert metrics["threshold_min"] == pytest.approx(risk_bundle["late_threshold_min"])
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0

