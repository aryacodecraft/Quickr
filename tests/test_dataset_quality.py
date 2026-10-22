"""Integrity tests for the real raw dataset and the cleaning funnel.

These tests recompute the raw-file facts with pandas (independent of Spark) and
compare them with the numbers the notebook stores in
``data/processed/analytics/delivery_metrics.csv`` and ``models/model_metadata.json``.
They never invent data: if the raw file changes, the tests fail instead of
silently relaxing the expectations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "food_delivery.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "quickr_clean.csv"
METRICS_PATH = ROOT / "data" / "processed" / "analytics" / "delivery_metrics.csv"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"

EXPECTED_COLUMNS = [
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken(min)",
]

NULL_TOKENS = {"", "NAN", "NULL", "NONE", "NA", "N/A"}

EXPECTED_RAW_ROWS = 45_593
EXPECTED_CLEAN_ROWS = 40_016
EXPECTED_INVALID_GPS = 4_071
EXPECTED_MISSING_CORE = 1_506


def _clean_token(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.upper() in NULL_TOKENS:
        return None
    return text


def _parse_duration(value):
    text = _clean_token(value)
    if text is None:
        return None
    text = text.replace("(min)", "").strip()
    match = np.nan
    try:
        return float(text)
    except ValueError:
        import re

        found = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(found.group(0)) if found else np.nan


def _normalize_weather(value) -> str | None:
    text = _clean_token(value)
    if text is None:
        return None
    cleaned = text.replace("conditions", "", 1).strip() if text.lower().startswith("conditions") else text
    return _clean_token(cleaned)


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        pytest.fail(
            f"Raw dataset missing at {RAW_PATH}. See README.md for the download command."
        )
    return pd.read_csv(RAW_PATH)


def test_raw_file_shape_and_schema(raw: pd.DataFrame) -> None:
    assert list(raw.columns) == EXPECTED_COLUMNS
    assert len(raw) == EXPECTED_RAW_ROWS
    assert raw["Delivery_person_ID"].nunique() > 1_000


def test_target_parses_into_a_sane_minute_range(raw: pd.DataFrame) -> None:
    duration = raw["Time_taken(min)"].map(_parse_duration)
    assert duration.notna().all(), "every raw row should carry a parseable target"
    assert duration.between(1, 120).all()
    assert 10 <= duration.min() <= 15
    assert 45 <= duration.max() <= 60


def test_no_exact_duplicate_rows(raw: pd.DataFrame) -> None:
    assert raw.duplicated().sum() == 0


def test_known_gps_defects_exist_in_the_raw_file(raw: pd.DataFrame) -> None:
    coords = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    numeric = raw[coords].apply(pd.to_numeric, errors="coerce")
    zero_rows = (numeric == 0).any(axis=1)
    assert zero_rows.sum() >= 3_000, "the (0, 0) GPS defect block should still be present"


def test_cleaning_funnel_reaches_the_documented_row_count(raw: pd.DataFrame) -> None:
    """Recompute the sequential pandas cleaning funnel used by the notebook."""
    frame = raw.copy()
    frame["duration_min"] = frame["Time_taken(min)"].map(_parse_duration)

    frame = frame.drop_duplicates()
    frame = frame[frame["duration_min"].between(1.0, 120.0)]

    coords = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for column in coords:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    gps_ok = frame[coords].notna().all(axis=1)
    for column in coords:
        gps_ok &= frame[column] != 0
    gps_ok &= frame["Restaurant_latitude"].between(6.0, 37.0)
    gps_ok &= frame["Restaurant_longitude"].between(68.0, 98.0)
    gps_ok &= frame["Delivery_location_latitude"].between(6.0, 37.0)
    gps_ok &= frame["Delivery_location_longitude"].between(68.0, 98.0)
    invalid_gps = int((~gps_ok).sum())
    frame = frame[gps_ok].copy()

    distance = _haversine_km(
        frame["Restaurant_latitude"],
        frame["Restaurant_longitude"],
        frame["Delivery_location_latitude"],
        frame["Delivery_location_longitude"],
    ).round(3)
    frame["distance_km"] = distance
    distance_ok = frame["distance_km"].between(0.1, 100.0)
    distance_outliers = int((~distance_ok).sum())
    frame = frame[distance_ok]

    frame["weather"] = frame["Weatherconditions"].map(_normalize_weather)
    frame["traffic"] = frame["Road_traffic_density"].map(lambda value: _clean_token(value))
    frame["city"] = frame["City"].map(lambda value: _clean_token(value))
    core_ok = frame["weather"].notna() & frame["traffic"].notna() & frame["city"].notna()
    missing_core = int((~core_ok).sum())
    frame = frame[core_ok]

    age = pd.to_numeric(frame["Delivery_person_Age"], errors="coerce")
    rating = pd.to_numeric(frame["Delivery_person_Ratings"], errors="coerce")
    vehicle_condition = pd.to_numeric(frame["Vehicle_condition"], errors="coerce")
    concurrent = pd.to_numeric(frame["multiple_deliveries"], errors="coerce")
    numeric_ok = (
        (age.isna() | age.between(18, 60))
        & (rating.isna() | rating.between(1, 5))
        & vehicle_condition.between(0, 3)
        & (concurrent.isna() | concurrent.between(0, 3))
    )
    invalid_numeric = int((~numeric_ok).sum())
    clean = frame[numeric_ok]

    assert invalid_gps == EXPECTED_INVALID_GPS
    assert distance_outliers == 0
    assert missing_core == EXPECTED_MISSING_CORE
    assert invalid_numeric == 0
    assert len(clean) == EXPECTED_CLEAN_ROWS
    assert clean["duration_min"].between(1, 120).all()
    assert clean["distance_km"].between(0.1, 100).all()


def test_notebook_metrics_match_the_recomputed_funnel() -> None:
    if not METRICS_PATH.exists() or not MODEL_METADATA_PATH.exists():
        pytest.skip("Run the notebook to generate delivery_metrics.csv and model_metadata.json")
    table = pd.read_csv(METRICS_PATH)
    quality = table[table["metric_group"] == "data_quality"].set_index("metric_name")["metric_value"]
    assert int(quality["raw_rows"]) == EXPECTED_RAW_ROWS
    assert int(quality["duplicate_rows_removed"]) == 0
    assert int(quality["invalid_target_rows"]) == 0
    assert int(quality["invalid_gps_rows"]) == EXPECTED_INVALID_GPS
    assert int(quality["distance_outlier_rows"]) == 0
    assert int(quality["missing_core_category_rows"]) == EXPECTED_MISSING_CORE
    assert int(quality["invalid_numeric_rows"]) == 0
    assert int(quality["final_rows"]) == EXPECTED_CLEAN_ROWS
    assert int(quality["train_rows"]) + int(quality["test_rows"]) == EXPECTED_CLEAN_ROWS
    assert 25.0 <= float(quality["late_threshold_min"]) <= 40.0

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["training_rows"] == int(quality["train_rows"])
    assert metadata["test_rows"] == int(quality["test_rows"])
    assert metadata["late_threshold_min"] == pytest.approx(float(quality["late_threshold_min"]))


def test_clean_csv_matches_the_saved_metrics() -> None:
    if not CLEAN_PATH.exists() or not METRICS_PATH.exists():
        pytest.skip("Run the notebook to generate quickr_clean.csv and delivery_metrics.csv")
    table = pd.read_csv(METRICS_PATH)
    quality = table[table["metric_group"] == "data_quality"].set_index("metric_name")["metric_value"]
    frame = pd.read_csv(CLEAN_PATH, usecols=["duration_min", "distance_km", "is_late"])
    assert len(frame) == int(quality["final_rows"])
    assert frame["duration_min"].between(1, 120).all()
    assert frame["distance_km"].between(0.1, 100).all()
    assert frame["is_late"].isin([0, 1]).all()
    assert frame["is_late"].mean() == pytest.approx(float(quality["late_rate"]), abs=1e-6)
