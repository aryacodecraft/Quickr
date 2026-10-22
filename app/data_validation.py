"""Shared validation rules for Quickr.

These rules mirror the cleaning rules documented in the notebook. Keeping them
in one small module lets the dashboard validate user input and lets the test
suite exercise the same boundary conditions without duplicating the PySpark
pipeline.
"""

from __future__ import annotations

import math
import re
from typing import Any

# The cleaned dataset only contains routes inside India (see notebook Section C).
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0
MIN_DURATION_MIN, MAX_DURATION_MIN = 1.0, 120.0
MIN_ROUTE_KM, MAX_ROUTE_KM = 0.1, 100.0

NULL_TOKENS = {"", "nan", "null", "none", "na", "n/a"}


def clean_token(value: Any) -> str | None:
    """Trim a text value and convert common missing-value tokens to None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return None
    return text


def parse_duration(value: Any) -> float | None:
    """Parse values such as ``(min) 24`` or ``24`` into minutes.

    Returns None when the value is missing or not numeric. The raw Kaggle file
    stores the target as a string with a ``(min)`` prefix.
    """
    text = clean_token(value)
    if text is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace("(min)", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_valid_duration(value: float | None) -> bool:
    return value is not None and MIN_DURATION_MIN <= value <= MAX_DURATION_MIN


def normalize_weather(value: Any) -> str | None:
    """Remove the ``conditions`` prefix and normalise missing markers."""
    text = clean_token(value)
    if text is None:
        return None
    text = re.sub(r"^conditions\s*", "", text, flags=re.IGNORECASE).strip()
    return clean_token(text)


def normalize_city(value: Any) -> str | None:
    """Fix the spelling typo present in the source data."""
    text = clean_token(value)
    if text == "Metropolitian":
        return "Metropolitan"
    return text


def normalize_traffic(value: Any) -> str | None:
    return clean_token(value)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two GPS points."""
    values = [lat1, lon1, lat2, lon2]
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise ValueError("Coordinates must be finite numbers.")
    lat1, lon1, lat2, lon2 = map(float, values)
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def validate_route(
    restaurant_lat: float,
    restaurant_lon: float,
    delivery_lat: float,
    delivery_lon: float,
) -> tuple[float, list[str]]:
    """Validate a route and return (distance_km, warnings).

    Coordinates outside the India bounding box are rejected because the model
    was trained only on Indian routes. A route longer than 100 km is also
    rejected; such records were treated as GPS errors during cleaning.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        lat1, lon1, lat2, lon2 = map(
            float, [restaurant_lat, restaurant_lon, delivery_lat, delivery_lon]
        )
    except (TypeError, ValueError):
        raise ValueError("All four coordinates must be numeric.") from None

    for name, value, low, high in [
        ("Restaurant latitude", lat1, -90, 90),
        ("Restaurant longitude", lon1, -180, 180),
        ("Delivery latitude", lat2, -90, 90),
        ("Delivery longitude", lon2, -180, 180),
    ]:
        if not math.isfinite(value) or not low <= value <= high:
            errors.append(f"{name} is outside the valid world range.")

    if not errors:
        in_india = (
            INDIA_LAT_MIN <= lat1 <= INDIA_LAT_MAX
            and INDIA_LON_MIN <= lon1 <= INDIA_LON_MAX
            and INDIA_LAT_MIN <= lat2 <= INDIA_LAT_MAX
            and INDIA_LON_MIN <= lon2 <= INDIA_LON_MAX
        )
        if not in_india:
            errors.append(
                "Quickr's model only supports Indian routes "
                f"(latitude {INDIA_LAT_MIN}–{INDIA_LAT_MAX}, "
                f"longitude {INDIA_LON_MIN}–{INDIA_LON_MAX})."
            )

    distance = haversine_km(lat1, lon1, lat2, lon2) if not errors else 0.0
    if not errors and not (MIN_ROUTE_KM <= distance <= MAX_ROUTE_KM):
        errors.append(
            f"Route distance must be between {MIN_ROUTE_KM} and {MAX_ROUTE_KM} km "
            f"(calculated {distance:.1f} km)."
        )
    if not errors and distance > 25:
        warnings.append(
            "This route is longer than most training routes; the prediction is "
            "an extrapolation."
        )

    if errors:
        raise ValueError(" ".join(errors))
    return distance, warnings


def _as_float(name: str, value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")
    return number


def validate_prediction_input(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate one dashboard payload and add the derived route distance.

    Operational ranges are intentionally a little wider than the observed
    training ranges, but still physically sensible. Values outside the training
    range are flagged with a warning instead of being silently accepted.
    """
    required = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
    ]
    missing = [key for key in required if key not in payload or payload[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    cleaned: dict[str, Any] = dict(payload)
    cleaned["Delivery_person_Age"] = _as_float("Partner age", payload["Delivery_person_Age"], 18, 60)
    cleaned["Delivery_person_Ratings"] = _as_float(
        "Partner rating", payload["Delivery_person_Ratings"], 1, 5
    )
    cleaned["Vehicle_condition"] = _as_float(
        "Vehicle condition", payload["Vehicle_condition"], 0, 3
    )
    cleaned["multiple_deliveries"] = _as_float(
        "Multiple deliveries", payload["multiple_deliveries"], 0, 3
    )

    distance, warnings = validate_route(
        payload["Restaurant_latitude"],
        payload["Restaurant_longitude"],
        payload["Delivery_location_latitude"],
        payload["Delivery_location_longitude"],
    )
    cleaned["distance_km"] = round(distance, 3)

    categorical = {
        "Weatherconditions": "Weather",
        "Road_traffic_density": "Traffic level",
        "Type_of_order": "Order type",
        "Type_of_vehicle": "Vehicle type",
        "Festival": "Festival",
        "City": "City",
    }
    options = metadata.get("categorical_options", {})
    for column, label in categorical.items():
        value = str(payload[column]).strip()
        allowed = options.get(column)
        if allowed and value not in allowed:
            raise ValueError(f"{label} must be one of: {', '.join(allowed)}.")
        cleaned[column] = value

    # Flag extrapolation rather than rejecting it outright; the dashboard shows
    # these as warnings next to the prediction.
    observed = metadata.get("numeric_ranges", {})
    for column, label in [
        ("Delivery_person_Age", "Partner age"),
        ("Delivery_person_Ratings", "Partner rating"),
        ("Vehicle_condition", "Vehicle condition"),
        ("multiple_deliveries", "Multiple deliveries"),
    ]:
        bounds = observed.get(column)
        if bounds:
            value = cleaned[column]
            if value < float(bounds["min"]) or value > float(bounds["max"]):
                warnings.append(
                    f"{label} ({value:g}) is outside the observed training range "
                    f"({bounds['min']:g}–{bounds['max']:g})."
                )

    cleaned["warnings"] = warnings
    return cleaned


def validate_risk_input(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the compact Delivery Risk form, which supplies distance directly.

    The risk form intentionally avoids four coordinate widgets. It still uses the
    same straight-line distance feature as the trained pipeline, so the user must
    enter a route distance in the observed 0.1–100 km range.
    """
    required = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
    ]
    missing = [key for key in required if key not in payload or payload[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    warnings: list[str] = []
    cleaned: dict[str, Any] = dict(payload)
    cleaned["Delivery_person_Age"] = _as_float("Partner age", payload["Delivery_person_Age"], 18, 60)
    cleaned["Delivery_person_Ratings"] = _as_float(
        "Partner rating", payload["Delivery_person_Ratings"], 1, 5
    )
    cleaned["Vehicle_condition"] = _as_float(
        "Vehicle condition", payload["Vehicle_condition"], 0, 3
    )
    cleaned["multiple_deliveries"] = _as_float(
        "Multiple deliveries", payload["multiple_deliveries"], 0, 3
    )
    distance = _as_float("Route distance", payload["distance_km"], MIN_ROUTE_KM, MAX_ROUTE_KM)
    if distance > 25:
        warnings.append(
            "This route is longer than most training routes; the prediction is an extrapolation."
        )
    cleaned["distance_km"] = round(distance, 3)

    categorical = {
        "Weatherconditions": "Weather",
        "Road_traffic_density": "Traffic level",
        "Type_of_order": "Order type",
        "Type_of_vehicle": "Vehicle type",
        "Festival": "Festival",
        "City": "City",
    }
    options = metadata.get("categorical_options", {})
    for column, label in categorical.items():
        value = str(payload[column]).strip()
        allowed = options.get(column)
        if allowed and value not in allowed:
            raise ValueError(f"{label} must be one of: {', '.join(allowed)}.")
        cleaned[column] = value

    observed = metadata.get("numeric_ranges", {})
    for column, label in [
        ("Delivery_person_Age", "Partner age"),
        ("Delivery_person_Ratings", "Partner rating"),
        ("distance_km", "Route distance"),
        ("Vehicle_condition", "Vehicle condition"),
        ("multiple_deliveries", "Multiple deliveries"),
    ]:
        bounds = observed.get(column)
        if bounds:
            value = cleaned[column]
            if value < float(bounds["min"]) or value > float(bounds["max"]):
                warnings.append(
                    f"{label} ({value:g}) is outside the observed training range "
                    f"({bounds['min']:g}–{bounds['max']:g})."
                )

    cleaned["warnings"] = warnings
    return cleaned
