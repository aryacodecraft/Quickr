"""Load saved Quickr models and produce genuine dashboard predictions."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.data_validation import validate_prediction_input, validate_risk_input

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "models")))
TIME_MODEL_PATH = MODEL_DIR / "quickr_delivery_time_model.joblib"
RISK_MODEL_PATH = MODEL_DIR / "quickr_delay_risk_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TEST_PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
RISK_METRICS_PATH = MODEL_DIR / "risk_metrics.json"

# Default bands are documented in the notebook and stored in the risk bundle.
# They are model-score bands, not calibrated real-world probabilities.
DEFAULT_MEDIUM_THRESHOLD = 0.40
DEFAULT_HIGH_THRESHOLD = 0.65

NUMERIC_LABELS = {
    "Delivery_person_Age": "Partner age",
    "Delivery_person_Ratings": "Partner rating",
    "distance_km": "Route distance",
    "Vehicle_condition": "Vehicle condition",
    "multiple_deliveries": "Concurrent deliveries",
}
CATEGORICAL_LABELS = {
    "Weatherconditions": "Weather",
    "Road_traffic_density": "Traffic",
    "Type_of_order": "Order type",
    "Type_of_vehicle": "Vehicle type",
    "Festival": "Festival",
    "City": "City",
}


class ModelNotAvailable(RuntimeError):
    """Raised when a notebook artifact has not been generated yet."""


@lru_cache(maxsize=2)
def _load_bundle(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        raise ModelNotAvailable(
            f"Model artifact not found: {path.name}. Run the notebook "
            "notebooks/Quickr_BDA_Analysis.ipynb end-to-end to train and save it."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelNotAvailable(f"Unexpected model artifact format: {path.name}")
    return bundle


def load_regression_bundle() -> dict[str, Any]:
    return _load_bundle(str(TIME_MODEL_PATH))


def load_risk_bundle() -> dict[str, Any]:
    return _load_bundle(str(RISK_MODEL_PATH))


def clear_model_cache() -> None:
    """Drop cached bundles so freshly retrained notebook artifacts are picked up."""
    _load_bundle.cache_clear()


def load_metadata() -> dict[str, Any]:
    """Return dashboard metadata, or an empty dict if the notebook has not run."""
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def models_ready() -> bool:
    return TIME_MODEL_PATH.exists() and METADATA_PATH.exists()


def risk_model_ready() -> bool:
    return RISK_MODEL_PATH.exists()


def _feature_frame(cleaned: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    columns = metadata.get("feature_order", [])
    if not columns:
        raise ModelNotAvailable("Model metadata is missing feature_order.")
    missing = [column for column in columns if column not in cleaned]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}")
    return pd.DataFrame([{column: cleaned[column] for column in columns}])


def risk_category(
    probability: float,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> str:
    """Map an uncalibrated model score to the documented three risk bands."""
    if not math.isfinite(float(probability)):
        return "Not available"
    medium = float(medium_threshold)
    high = float(high_threshold)
    if not 0.0 < medium < high < 1.0:
        raise ValueError("Risk cut-offs must satisfy 0 < medium < high < 1.")
    if probability < medium:
        return "Low risk"
    if probability < high:
        return "Medium risk"
    return "High risk"


def _factor_label(raw_name: str) -> str:
    """Turn a sklearn one-hot/scaled feature name into a readable label."""
    base = raw_name.split("__", 1)[-1]
    for column, label in NUMERIC_LABELS.items():
        if base == column:
            return label
    for column, label in CATEGORICAL_LABELS.items():
        prefix = f"{column}_"
        if base.startswith(prefix):
            return f"{label}: {base[len(prefix):].replace('_', ' ')}"
    return base.replace("_", " ")


def _risk_factors(bundle: dict[str, Any], features: pd.DataFrame) -> list[str]:
    """Top positive logistic-regression contributions for this input row.

    These are associations learned by the classifier, not causal effects.
    Contributions are only used to order the labels shown in the UI.
    """
    pipeline = bundle.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return []
    preprocess = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocess is None or model is None or not hasattr(model, "coef_"):
        return []
    try:
        matrix = preprocess.transform(features)
        if hasattr(matrix, "toarray"):
            values = np.asarray(matrix.toarray())[0]
        else:
            values = np.asarray(matrix)[0]
        coefficients = np.ravel(model.coef_)
        names = preprocess.get_feature_names_out()
        contributions = values * coefficients
        order = np.argsort(contributions)[::-1]
        result = []
        for index in order:
            if contributions[index] <= 0:
                break
            result.append(_factor_label(str(names[index])))
            if len(result) == 3:
                break
        return result
    except Exception:
        # Explanation is optional; a prediction must still work if the
        # sklearn version cannot expose feature names.
        return []


def _duration_result(
    cleaned: dict[str, Any],
    metadata: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    return {
        "predicted_duration_min": round(float(duration), 1),
        "distance_km": cleaned["distance_km"],
        "model_name": metadata.get("model_name", "Delivery-time model"),
        "model_version": metadata.get("model_version", "unknown"),
        "target_unit": metadata.get("target_unit", "minutes"),
        "warnings": list(cleaned.get("warnings", [])),
        "inputs": cleaned,
        "late_risk_probability": None,
        "risk_category": "Not available",
        "probability_threshold": None,
        "delay_label_threshold_min": metadata.get("late_threshold_min"),
    }


def _add_risk_result(
    result: dict[str, Any],
    risk_bundle: dict[str, Any],
    cleaned: dict[str, Any],
    medium_threshold: float | None = None,
    high_threshold: float | None = None,
) -> None:
    risk_metadata = risk_bundle.get("metadata", {})
    risk_features = _feature_frame(cleaned, risk_metadata)
    probability = float(risk_bundle["pipeline"].predict_proba(risk_features)[0, 1])
    medium = float(
        medium_threshold
        if medium_threshold is not None
        else risk_metadata.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD)
    )
    high = float(
        high_threshold
        if high_threshold is not None
        else risk_metadata.get("high_threshold", DEFAULT_HIGH_THRESHOLD)
    )
    category = risk_category(probability, medium, high)
    result.update(
        {
            "late_risk_probability": round(probability, 4),
            "risk_category": category,
            "probability_threshold": float(
                risk_bundle.get(
                    "classification_threshold",
                    risk_metadata.get("probability_threshold", 0.5),
                )
            ),
            "medium_threshold": medium,
            "high_threshold": high,
            "risk_factors": _risk_factors(risk_bundle, risk_features),
            "risk_model_name": risk_metadata.get("model_name", "Logistic Regression"),
            "risk_model_version": risk_metadata.get("model_version", "unknown"),
            "delay_label_threshold_min": risk_bundle.get(
                "late_threshold_min",
                risk_metadata.get("late_threshold_min", result.get("delay_label_threshold_min")),
            ),
        }
    )


def predict_delivery(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate coordinates, predict duration, and estimate delay risk.

    The return value contains only values computed by the saved pipelines. No
    sample or fallback prediction is used.
    """
    bundle = load_regression_bundle()
    metadata = bundle.get("metadata", {})
    cleaned = validate_prediction_input(payload, metadata)
    features = _feature_frame(cleaned, metadata)

    prediction = float(bundle["pipeline"].predict(features)[0])
    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite duration; check the inputs.")
    prediction = max(0.1, prediction)
    result = _duration_result(cleaned, metadata, prediction)

    if RISK_MODEL_PATH.exists():
        try:
            _add_risk_result(result, load_risk_bundle(), cleaned)
        except (ModelNotAvailable, ValueError, IndexError) as exc:
            result["warnings"].append(f"Delay-risk model unavailable: {exc}")

    return result


def predict_risk(
    payload: dict[str, Any],
    medium_threshold: float | None = None,
    high_threshold: float | None = None,
) -> dict[str, Any]:
    """Assess duration risk from the compact form that supplies distance directly."""
    regression_bundle = load_regression_bundle()
    metadata = regression_bundle.get("metadata", {})
    cleaned = validate_risk_input(payload, metadata)
    features = _feature_frame(cleaned, metadata)

    duration = float(regression_bundle["pipeline"].predict(features)[0])
    if not math.isfinite(duration):
        raise ValueError("The regression model returned a non-finite duration.")
    result = _duration_result(cleaned, metadata, max(0.1, duration))

    risk_bundle = load_risk_bundle()
    _add_risk_result(result, risk_bundle, cleaned, medium_threshold, high_threshold)
    return result


def load_test_predictions() -> pd.DataFrame:
    if not TEST_PREDICTIONS_PATH.exists():
        raise ModelNotAvailable("test_predictions.csv is missing; run the notebook.")
    return pd.read_csv(TEST_PREDICTIONS_PATH)


def load_confusion_matrix() -> pd.DataFrame:
    if not CONFUSION_MATRIX_PATH.exists():
        raise ModelNotAvailable("confusion_matrix.csv is missing; run the notebook.")
    return pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_risk_metrics() -> dict[str, Any]:
    if not RISK_METRICS_PATH.exists():
        return {}
    with RISK_METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)
