"""Unit tests for the shared validation rules used by the dashboard."""

from __future__ import annotations

import pytest

from app.data_validation import (
    clean_token,
    haversine_km,
    is_valid_duration,
    normalize_city,
    normalize_weather,
    parse_duration,
    validate_prediction_input,
    validate_route,
)

METADATA = {
    "categorical_options": {
        "Weatherconditions": ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"],
        "Road_traffic_density": ["High", "Jam", "Low", "Medium"],
        "Type_of_order": ["Buffet", "Drinks", "Meal", "Snack"],
        "Type_of_vehicle": ["electric_scooter", "motorcycle", "scooter"],
        "Festival": ["No", "Yes"],
        "City": ["Metropolitan", "Semi-Urban", "Urban"],
    },
    "numeric_ranges": {
        "Delivery_person_Age": {"min": 18, "max": 50},
        "Delivery_person_Ratings": {"min": 1.0, "max": 5.0},
        "Vehicle_condition": {"min": 0, "max": 3},
        "multiple_deliveries": {"min": 0, "max": 3},
    },
}

BASE_PAYLOAD = {
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": 22.745,
    "Restaurant_longitude": 75.892,
    "Delivery_location_latitude": 22.765,
    "Delivery_location_longitude": 75.912,
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 2,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1,
    "Festival": "No",
    "City": "Metropolitan",
}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("(min) 24", 24.0),
        ("  (min) 10 ", 10.0),
        ("33", 33.0),
        (None, None),
        ("NaN", None),
        ("", None),
        ("not-a-number", None),
    ],
)
def test_parse_duration(value, expected) -> None:
    assert parse_duration(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("conditions Cloudy", "Cloudy"),
        ("  conditions  Stormy", "Stormy"),
        ("conditions NaN", None),
        ("NaN", None),
        ("Sunny", "Sunny"),
    ],
)
def test_normalize_weather(value, expected) -> None:
    assert normalize_weather(value) == expected


def test_normalize_city_fixes_source_typo() -> None:
    assert normalize_city("Metropolitian") == "Metropolitan"
    assert normalize_city("Urban") == "Urban"
    assert normalize_city("NaN") is None


def test_clean_token_handles_common_missing_markers() -> None:
    assert clean_token("  value ") == "value"
    for token in ("", " ", "nan", "NULL", "None", "NA", "N/A"):
        assert clean_token(token) is None


def test_haversine_known_distance() -> None:
    # Delhi (28.6139, 77.2090) to Mumbai (19.0760, 72.8777) is ~1,150 km.
    distance = haversine_km(28.6139, 77.2090, 19.0760, 72.8777)
    assert 1_120 < distance < 1_180
    assert haversine_km(22.0, 75.0, 22.0, 75.0) == pytest.approx(0.0)


def test_haversine_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        haversine_km(float("nan"), 75.0, 22.0, 75.0)


def test_validate_route_accepts_indian_route_and_rejects_gps_scale_error() -> None:
    distance, warnings = validate_route(22.745, 75.892, 22.765, 75.912)
    assert distance > 0
    assert warnings == []

    # Anything over 100 km is treated as a GPS error, not merely warned about.
    with pytest.raises(ValueError, match="100"):
        validate_route(28.6139, 77.2090, 19.0760, 72.8777)


def test_validate_route_rejects_rows_outside_india() -> None:
    with pytest.raises(ValueError, match="Indian routes"):
        validate_route(51.5074, -0.1278, 51.5074, -0.1276)


def test_validate_prediction_input_derives_distance_and_keeps_warnings() -> None:
    cleaned = validate_prediction_input(dict(BASE_PAYLOAD), METADATA)
    assert cleaned["distance_km"] == pytest.approx(3.03, abs=0.2)
    assert cleaned["warnings"] == []
    assert cleaned["City"] == "Metropolitan"


def test_validate_prediction_input_flags_extrapolation() -> None:
    payload = dict(BASE_PAYLOAD)
    payload["Delivery_person_Age"] = 58  # above the observed training max (50)
    cleaned = validate_prediction_input(payload, METADATA)
    assert any("outside the observed training range" in warning for warning in cleaned["warnings"])


def test_validate_prediction_input_rejects_unknown_category_and_missing_fields() -> None:
    payload = dict(BASE_PAYLOAD)
    payload["City"] = "Atlantis"
    with pytest.raises(ValueError, match="City must be one of"):
        validate_prediction_input(payload, METADATA)

    incomplete = dict(BASE_PAYLOAD)
    del incomplete["Weatherconditions"]
    with pytest.raises(ValueError, match="Missing required inputs"):
        validate_prediction_input(incomplete, METADATA)


def test_is_valid_duration_bounds() -> None:
    assert is_valid_duration(1.0)
    assert is_valid_duration(120.0)
    assert not is_valid_duration(0.9)
    assert not is_valid_duration(120.1)
    assert not is_valid_duration(None)
