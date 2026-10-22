"""Quickr — Quick-Commerce Delivery Intelligence dashboard.

Run from the project root:
    .venv\\Scripts\\streamlit run app/app.py

All historical numbers come from Apache Cassandra. Predictions come from the
scikit-learn pipelines saved by the notebook. There is no mock-data fallback.
"""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Streamlit puts the script's own folder (D:\Quickr\app) at the front of
# sys.path. That makes ``import app`` resolve to app.py instead of the package
# and produces a circular import. Always put the project root first.
ROOT_TEXT = str(ROOT)
while ROOT_TEXT in sys.path:
    sys.path.remove(ROOT_TEXT)
sys.path.insert(0, ROOT_TEXT)

from app import cassandra_client as cca  # noqa: E402
from app import prediction_service as ps  # noqa: E402
from app.data_validation import haversine_km  # noqa: E402
from app.styles import (  # noqa: E402
    COLORS,
    PLOTLY_CONFIG,
    badge,
    brand,
    callout,
    inject_css,
    insight,
    kpi_card,
    page_header,
    section_heading,
    side_label,
    status_row,
    style_fig,
)

PAGES = [
    "Overview",
    "Delivery Performance",
    "Delivery Analytics",
    "Prediction",
    "Delivery Risk",
    "Model Performance",
    "Data Quality",
]
MODEL_ORDER = ["Median baseline", "Linear Regression", "Random Forest"]

st.set_page_config(
    page_title="Quickr — Delivery Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ---------------------------------------------------------------------------
# Data access and formatting helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner="Loading analytics from Cassandra ...")
def load_core() -> dict:
    """One cached read for the dashboard's pre-aggregated analytics."""
    session = cca.connect()
    try:
        try:
            thresholds = cca.fetch_duration_thresholds(session)
        except Exception:
            # The exact-threshold table may not exist until the updated
            # notebook or scripts/init_cassandra.py has been run. Other pages
            # should still render, and the scorecard will show a setup note.
            thresholds = []
        return {
            "metrics": cca.fetch_metrics(session),
            "summaries": cca.fetch_all_summaries(session),
            "distribution": cca.fetch_duration_distribution(session),
            "segments": cca.fetch_segment_summary(session),
            "thresholds": thresholds,
        }
    finally:
        session.shutdown()


def fetch_recent_predictions(limit: int = 8) -> list[dict]:
    session = cca.connect()
    try:
        return cca.fetch_recent_predictions(session, limit=limit)
    finally:
        session.shutdown()


def save_prediction(result: dict) -> None:
    record = cca.make_prediction_record(result, result["inputs"])
    session = cca.connect()
    try:
        cca.save_prediction(session, record)
    finally:
        session.shutdown()


def fmt_int(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_min(value) -> str:
    try:
        return f"{float(value):.1f} min"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_score(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def metric(group: dict, name: str, default=None):
    return cca.metric_value(group, name, default=default)


def text_metric(group: dict, name: str, default: str = "") -> str:
    return cca.metric_text(group, name, default=default)


def summary_frame(core: dict, category: str) -> pd.DataFrame:
    rows = core["summaries"].get(category, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in ("deliveries", "avg_duration_min", "median_duration_min", "p90_duration_min", "late_rate", "sort_order"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def performance_frame(perf: dict) -> pd.DataFrame:
    if not perf:
        return pd.DataFrame()
    records = []
    for key, row in perf.items():
        if "." not in key:
            continue
        metric_name, metric_key = key.split(".", 1)
        if metric_key not in {"mae", "rmse", "r2", "mape"}:
            continue
        records.append(
            {
                "model": row.get("model_name") or metric_name,
                "metric": metric_key,
                "value": row.get("metric_value"),
                "model_version": row.get("model_version"),
            }
        )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    pivot = df.pivot_table(index="model", columns="metric", values="value", aggfunc="first").reset_index()
    pivot["_order"] = pivot["model"].apply(lambda name: MODEL_ORDER.index(name) if name in MODEL_ORDER else 99)
    return pivot.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def selected_model_row(perf: dict) -> dict | None:
    """Lowest genuine test RMSE; returns None when no valid metrics exist."""
    comparisons = performance_frame(perf)
    if comparisons.empty or "rmse" not in comparisons:
        return None
    valid = comparisons.dropna(subset=["rmse"])
    if valid.empty:
        return None
    return valid.loc[valid["rmse"].idxmin()].to_dict()


def best_category_label(core: dict) -> tuple[str, str, str]:
    """Lowest average duration among supported categories with >= 100 records."""
    label_map = {"traffic": "Traffic", "weather": "Weather", "vehicle": "Vehicle"}
    best = None
    for category, label in label_map.items():
        df = summary_frame(core, category)
        if df.empty or "deliveries" not in df:
            continue
        candidates = df[df["deliveries"] >= 100].dropna(subset=["avg_duration_min"])
        if candidates.empty:
            continue
        row = candidates.loc[candidates["avg_duration_min"].idxmin()]
        if best is None or row["avg_duration_min"] < best[0]:
            best = (float(row["avg_duration_min"]), str(row["category_value"]), label)
    if best is None:
        return "—", "—", "No category"
    return f"{best[0]:.1f} min", best[1], best[2]


def plot_theme_bar(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    x_title: str,
    value_col: str = "avg_duration_min",
    height: int = 300,
) -> go.Figure:
    if df.empty:
        return style_fig(go.Figure(), height=height)
    df = df.dropna(subset=[value_col]).copy()
    df["_order"] = df.get("sort_order", 0)
    df = df.sort_values(["_order", value_col], ascending=[True, True])
    minimum = df[value_col].min()
    colors = [COLORS["yellow"] if value == minimum else COLORS["green"] for value in df[value_col]]
    fig = go.Figure(
        go.Bar(
            x=df[category_col],
            y=df[value_col],
            marker_color=colors,
            customdata=df[["deliveries"]].values if "deliveries" in df else None,
            hovertemplate=(
                f"<b>%{{x}}</b><br>{x_title}: %{{y:.1f}} min"
                + ("<br>Deliveries: %{customdata[0]:,}<extra></extra>" if "deliveries" in df else "<extra></extra>")
            ),
        )
    )
    fig.update_traces(text=df[value_col].map(lambda v: f"{v:.1f}"), textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, yaxis_title="Minutes", xaxis_title="")
    return style_fig(fig, height=height)


def weighted_group(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[by, "deliveries", "avg_duration_min"])
    work = df.copy()
    work["_weighted"] = work["avg_duration_min"] * work["deliveries"]
    grouped = (
        work.groupby(by, as_index=False)
        .agg(deliveries=("deliveries", "sum"), _weighted=("_weighted", "sum"))
    )
    grouped["avg_duration_min"] = grouped["_weighted"] / grouped["deliveries"].clip(lower=1)
    return grouped.drop(columns="_weighted")


def filtered_histogram_frame(core: dict, histogram: dict[int, int]) -> pd.DataFrame:
    labels = {
        int(row["bucket_start_min"]): row["bucket_label"]
        for row in core["distribution"]
    }
    starts = sorted(labels)
    return pd.DataFrame(
        {
            "bucket_start": starts,
            "label": [labels[start] for start in starts],
            "count": [int(histogram.get(start, 0)) for start in starts],
        }
    )


def render_prediction_result(result: dict) -> None:
    prob = result.get("late_risk_probability")
    risk_text = "—"
    if prob is not None:
        risk_text = f"{result.get('risk_category', '—')} · {prob * 100:.1f}% model score"
        medium = result.get("medium_threshold")
        high = result.get("high_threshold")
        if medium is not None and high is not None:
            risk_text += f" (bands {medium:.0%}/{high:.0%})"
        else:
            threshold = result.get("probability_threshold")
            if threshold is not None:
                risk_text += f" (cut-off {threshold:.0%})"
    st.markdown(
        f"""
        <div class="qr-result">
            <div class="qr-result-label">Estimated delivery duration</div>
            <div class="qr-result-value">{result['predicted_duration_min']:.1f}</div>
            <div class="qr-result-unit">minutes · straight-line route {result['distance_km']:.2f} km</div>
            <div class="qr-result-row"><span>Risk band</span><strong>{html.escape(risk_text)}</strong></div>
            <div class="qr-result-row"><span>Model</span><strong>{html.escape(str(result.get('model_name', '—')))} · {html.escape(str(result.get('model_version', '—')))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for warning in result.get("warnings", []):
        st.warning(warning)
    st.caption(
        "This is a statistical estimate from historical delivery records, not a guaranteed "
        "arrival time. It uses no live traffic, weather, or order-level data."
    )


def render_risk_result(result: dict) -> None:
    probability = result.get("late_risk_probability")
    category = result.get("risk_category", "Not available")
    duration = result.get("predicted_duration_min")
    threshold = result.get("delay_label_threshold_min")
    medium = result.get("medium_threshold")
    high = result.get("high_threshold")
    score_text = f"{probability * 100:.1f}% model score" if probability is not None else "—"
    threshold_text = f"{float(threshold):.0f} min" if threshold is not None else "not available"
    band_text = (
        f"Low &lt; {medium:.0%} · Medium {medium:.0%}–{high:.0%} · High ≥ {high:.0%}"
        if medium is not None and high is not None
        else "Bands not available"
    )
    st.markdown(
        f"""
        <div class="qr-result">
            <div class="qr-result-label">Delivery risk assessment</div>
            <div class="qr-result-value">{html.escape(category)}</div>
            <div class="qr-result-unit">{html.escape(score_text)} · uncalibrated model score</div>
            <div class="qr-result-row"><span>Predicted duration</span><strong>{f"{duration:.1f} min" if duration is not None else "—"}</strong></div>
            <div class="qr-result-row"><span>Risk definition</span><strong>duration &gt; {html.escape(threshold_text)}</strong></div>
            <div class="qr-result-row"><span>Score bands</span><strong>{band_text}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    factors = result.get("risk_factors") or []
    if factors:
        st.markdown(
            "**Factors associated with a higher model score for this input:** "
            + ", ".join(html.escape(str(item)) for item in factors)
            + "."
        )
    else:
        st.caption("No single input had a strong positive association for this row.")
    st.caption(
        "These are associations learned from historical records, not causal claims. "
        "The score has not been calibrated, so read it as a relative risk band rather than "
        "a guaranteed probability. This estimate is not a promise of on-time delivery."
    )


def store_and_flash(result: dict) -> None:
    st.session_state["last_prediction"] = result
    try:
        save_prediction(result)
        st.success("Prediction saved to Cassandra.")
    except Exception as exc:
        st.error(f"The prediction was generated but could not be saved to Cassandra: {exc}")


# ---------------------------------------------------------------------------
# Reusable input form
# ---------------------------------------------------------------------------
def render_inputs(metadata: dict, prefix: str) -> dict:
    defaults = metadata.get("default_values", {})
    options = metadata.get("categorical_options", {})

    def opt(column: str, fallback: list[str]) -> list[str]:
        values = options.get(column) or fallback
        return [str(value) for value in values]

    def coord(column: str, fallback: float) -> float:
        # Keep the displayed value and the step on the same precision.
        # A six-decimal default with a four-decimal step is invalid HTML and can
        # make the browser reject/snap the field while the user is typing.
        return round(float(defaults.get(column, fallback)), 4)

    st.markdown("**Route**")
    st.caption(
        "Paste or type coordinates inside India (lat 6–37, lon 68–98). "
        "Use a dot (.) as the decimal separator."
    )
    col1, col2 = st.columns(2)
    restaurant_lat = col1.number_input(
        "Restaurant latitude",
        min_value=6.0,
        max_value=37.0,
        value=coord("Restaurant_latitude", 22.745),
        format="%.4f",
        step=0.0001,
        key=f"{prefix}_rlat",
        help="Restaurant pickup latitude, e.g. 19.0658.",
    )
    restaurant_lon = col2.number_input(
        "Restaurant longitude",
        min_value=68.0,
        max_value=98.0,
        value=coord("Restaurant_longitude", 75.892),
        format="%.4f",
        step=0.0001,
        key=f"{prefix}_rlon",
        help="Restaurant pickup longitude, e.g. 76.6182.",
    )
    col3, col4 = st.columns(2)
    delivery_lat = col3.number_input(
        "Delivery latitude",
        min_value=6.0,
        max_value=37.0,
        value=coord("Delivery_location_latitude", 22.765),
        format="%.4f",
        step=0.0001,
        key=f"{prefix}_dlat",
        help="Customer drop-off latitude.",
    )
    delivery_lon = col4.number_input(
        "Delivery longitude",
        min_value=68.0,
        max_value=98.0,
        value=coord("Delivery_location_longitude", 75.912),
        format="%.4f",
        step=0.0001,
        key=f"{prefix}_dlon",
        help="Customer drop-off longitude.",
    )
    try:
        route_km = haversine_km(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)
        st.caption(
            f"Straight-line route: **{route_km:.2f} km**. The dataset has no route geometry or "
            "traffic-aware distance, so Haversine distance is the model feature."
        )
    except ValueError:
        route_km = 0.0
        st.caption("Enter valid coordinates to calculate the route distance.")

    st.markdown("**Conditions**")
    col5, col6 = st.columns(2)
    weather = col5.selectbox(
        "Weather",
        opt("Weatherconditions", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"]),
        key=f"{prefix}_weather",
    )
    traffic = col6.selectbox(
        "Road traffic density",
        opt("Road_traffic_density", ["Low", "Medium", "High", "Jam"]),
        key=f"{prefix}_traffic",
    )
    col7, col8 = st.columns(2)
    city = col7.selectbox("City type", opt("City", ["Metropolitan", "Urban", "Semi-Urban"]), key=f"{prefix}_city")
    festival = col8.selectbox("Festival day", opt("Festival", ["No", "Yes"]), key=f"{prefix}_festival")

    st.markdown("**Partner and order**")
    col9, col10 = st.columns(2)
    age = col9.number_input(
        "Partner age (years)",
        min_value=18.0,
        max_value=60.0,
        value=float(defaults.get("Delivery_person_Age", 30)),
        step=1.0,
        key=f"{prefix}_age",
    )
    rating = col10.number_input(
        "Partner rating (1–5)",
        min_value=1.0,
        max_value=5.0,
        value=float(defaults.get("Delivery_person_Ratings", 4.7)),
        step=0.1,
        format="%.1f",
        key=f"{prefix}_rating",
    )
    col11, col12 = st.columns(2)
    vehicle_type = col11.selectbox(
        "Vehicle type",
        opt("Type_of_vehicle", ["motorcycle", "scooter", "electric_scooter"]),
        key=f"{prefix}_vehicle_type",
    )
    order_type = col12.selectbox(
        "Order type",
        opt("Type_of_order", ["Buffet", "Drinks", "Meal", "Snack"]),
        key=f"{prefix}_order_type",
    )
    col13, col14 = st.columns(2)
    vehicle_condition = col13.slider(
        "Vehicle condition (0–3)",
        min_value=0,
        max_value=3,
        value=int(defaults.get("Vehicle_condition", 2)),
        key=f"{prefix}_condition",
    )
    multiple = col14.slider(
        "Concurrent deliveries",
        min_value=0,
        max_value=3,
        value=int(defaults.get("multiple_deliveries", 1)),
        key=f"{prefix}_multiple",
    )

    return {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": restaurant_lat,
        "Restaurant_longitude": restaurant_lon,
        "Delivery_location_latitude": delivery_lat,
        "Delivery_location_longitude": delivery_lon,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "multiple_deliveries": multiple,
        "Festival": festival,
        "City": city,
    }


def render_risk_inputs(metadata: dict, prefix: str) -> dict:
    """Compact risk form: one direct distance field instead of four coordinates."""
    defaults = metadata.get("default_values", {})
    options = metadata.get("categorical_options", {})

    def opt(column: str, fallback: list[str]) -> list[str]:
        values = options.get(column) or fallback
        return [str(value) for value in values]

    distance = st.number_input(
        "Straight-line route distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=min(100.0, max(0.1, round(float(defaults.get("distance_km", 9.2)), 2))),
        step=0.1,
        format="%.2f",
        key=f"{prefix}_distance",
        help="Use the same straight-line route distance that the model was trained on.",
    )

    c1, c2 = st.columns(2)
    traffic = c1.selectbox(
        "Road traffic density",
        opt("Road_traffic_density", ["Low", "Medium", "High", "Jam"]),
        key=f"{prefix}_traffic",
    )
    weather = c2.selectbox(
        "Weather",
        opt("Weatherconditions", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"]),
        key=f"{prefix}_weather",
    )
    c3, c4 = st.columns(2)
    city = c3.selectbox("City type", opt("City", ["Metropolitan", "Urban", "Semi-Urban"]), key=f"{prefix}_city")
    festival = c4.selectbox("Festival day", opt("Festival", ["No", "Yes"]), key=f"{prefix}_festival")
    c5, c6 = st.columns(2)
    vehicle_type = c5.selectbox(
        "Vehicle type",
        opt("Type_of_vehicle", ["motorcycle", "scooter", "electric_scooter"]),
        key=f"{prefix}_vehicle_type",
    )
    order_type = c6.selectbox(
        "Order type",
        opt("Type_of_order", ["Buffet", "Drinks", "Meal", "Snack"]),
        key=f"{prefix}_order_type",
    )
    c7, c8 = st.columns(2)
    age = c7.number_input(
        "Partner age (years)",
        min_value=18.0,
        max_value=60.0,
        value=float(defaults.get("Delivery_person_Age", 30)),
        step=1.0,
        key=f"{prefix}_age",
    )
    rating = c8.number_input(
        "Partner rating (1–5)",
        min_value=1.0,
        max_value=5.0,
        value=float(defaults.get("Delivery_person_Ratings", 4.7)),
        step=0.1,
        format="%.1f",
        key=f"{prefix}_rating",
    )
    c9, c10 = st.columns(2)
    vehicle_condition = c9.slider(
        "Vehicle condition (0–3)",
        min_value=0,
        max_value=3,
        value=int(defaults.get("Vehicle_condition", 1)),
        key=f"{prefix}_condition",
    )
    multiple = c10.slider(
        "Concurrent deliveries",
        min_value=0,
        max_value=3,
        value=int(defaults.get("multiple_deliveries", 1)),
        key=f"{prefix}_multiple",
    )
    return {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "distance_km": distance,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "multiple_deliveries": multiple,
        "Festival": festival,
        "City": city,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(core: dict, metadata: dict) -> None:
    quality = core["metrics"].get("data_quality", {})
    perf = core["metrics"].get("model_performance", {})
    meta = core["metrics"].get("app_meta", {})
    total = metric(quality, "final_rows", 0)
    avg_duration = metric(quality, "avg_duration_min")
    median_duration = metric(quality, "median_duration_min")
    p90 = metric(quality, "p90_duration_min")
    late_threshold = metric(quality, "late_threshold_min")
    late_rate = metric(quality, "late_rate")
    best_value, best_name, best_group = best_category_label(core)
    decisions = performance_frame(perf)
    best_model = meta.get("best_model", {}).get("metric_text") or "—"
    mae_text = "—"
    if not decisions.empty and "rmse" in decisions and decisions["rmse"].notna().any():
        best_row = decisions.loc[decisions["rmse"].idxmin()]
        mae = best_row.get("mae")
        if mae is not None and pd.notna(mae):
            mae_text = fmt_min(mae)

    page_header(
        "Operations / Overview",
        "Quick commerce, measured in minutes.",
        "A live view of the cleaned delivery dataset stored in Cassandra. "
        "Every KPI below is calculated from observed records, not a sample or placeholder.",
        [f"{fmt_int(total)} valid orders", "Local Spark + Cassandra", "Model-backed"],
    )

    columns = st.columns(5)
    with columns[0]:
        kpi_card("Valid deliveries", fmt_int(total), "after cleaning & GPS validation", "green")
    with columns[1]:
        kpi_card("Average duration", fmt_min(avg_duration), f"p90 {fmt_min(p90)}", "yellow")
    with columns[2]:
        kpi_card("Median duration", fmt_min(median_duration), "robust central tendency", "green")
    with columns[3]:
        kpi_card(
            "Best-performing segment",
            f"{best_name}",
            f"{best_value} · {best_group} · n≥100",
            "yellow",
        )
    with columns[4]:
        kpi_card("Selected model MAE", mae_text, f"{best_model} on held-out test data", "green")

    section_heading("Operational comparisons", "Average delivery duration by the categorical fields that actually exist in the dataset.")
    traffic = summary_frame(core, "traffic")
    vehicle = summary_frame(core, "vehicle")
    weather = summary_frame(core, "weather")
    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(
            plot_theme_bar(traffic, "category_value", "Traffic level", "Traffic"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with cols[1]:
        st.plotly_chart(
            plot_theme_bar(vehicle, "category_value", "Vehicle type", "Vehicle"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with cols[2]:
        st.plotly_chart(
            plot_theme_bar(weather, "category_value", "Weather", "Weather"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_heading("What the data says", "Generated from the aggregated tables; no narrative numbers are hardcoded.")
    left, right = st.columns(2)
    with left:
        if not traffic.empty:
            low = traffic.loc[traffic["category_value"].str.lower() == "low"]
            jam = traffic.loc[traffic["category_value"].str.lower() == "jam"]
            if not low.empty and not jam.empty:
                delta = float(jam.iloc[0]["avg_duration_min"] - low.iloc[0]["avg_duration_min"])
                insight(
                    "Traffic is the clearest operational lever",
                    f"Jam deliveries average {float(jam.iloc[0]['avg_duration_min']):.1f} min, "
                    f"{delta:+.1f} min versus low traffic. Jam late-rate is "
                    f"{fmt_pct(jam.iloc[0]['late_rate'])}.",
                )
        if not weather.empty:
            fastest = weather.loc[weather["avg_duration_min"].idxmin()]
            slowest = weather.loc[weather["avg_duration_min"].idxmax()]
            insight(
                "Weather shifts the baseline",
                f"{fastest['category_value']} is the fastest condition at "
                f"{float(fastest['avg_duration_min']):.1f} min; {slowest['category_value']} is "
                f"the slowest at {float(slowest['avg_duration_min']):.1f} min.",
            )
    with right:
        if not vehicle.empty:
            top_vehicle = vehicle.loc[vehicle["deliveries"].idxmax()]
            share = float(top_vehicle["deliveries"]) / max(float(vehicle["deliveries"].sum()), 1)
            insight(
                "Fleet mix matters",
                f"{top_vehicle['category_value']} handles {share * 100:.1f}% of valid deliveries "
                f"and averages {float(top_vehicle['avg_duration_min']):.1f} min. "
                f"Small segments exist as well; treat them cautiously.",
            )
        if late_threshold is not None:
            insight(
                "Analytical late-delivery signal",
                f"{fmt_pct(late_rate)} of clean deliveries exceed the clean-dataset P75 of "
                f"{late_threshold:.0f} min. This is an analytical threshold, not a customer SLA.",
            )
    st.caption(
        f"Dataset window in the raw file: {text_metric(core['metrics'].get('dataset', {}), 'date_range', 'not available')}. "
        "No hourly demand or geographic hotspot claims are made: the timestamp/location fields are incomplete or unreliable."
    )


def page_delivery_performance(core: dict, metadata: dict) -> None:
    page_header(
        "Operations / Scorecard",
        "Five numbers for the delivery operation.",
        "A metrics-only summary of the cleaned delivery records in Cassandra. "
        "The duration-threshold card is an analytical cutoff, not an official late-delivery rate.",
        ["Actual durations", "Held-out model error", "No SLA claims"],
    )
    quality = core["metrics"].get("data_quality", {})
    perf = core["metrics"].get("model_performance", {})
    valid = metric(quality, "final_rows")
    average = metric(quality, "avg_duration_min")
    median = metric(quality, "median_duration_min")
    fastest = metric(quality, "min_duration_min")
    slowest = metric(quality, "max_duration_min")
    documented_threshold = metric(quality, "late_threshold_min", 33)

    threshold_rows = {
        int(row["threshold_min"]): row for row in core.get("thresholds", [])
    }
    if threshold_rows:
        low = min(threshold_rows)
        high = max(threshold_rows)
        default = int(round(documented_threshold if documented_threshold is not None else 33))
        default = min(high, max(low, default))
        threshold = st.slider(
            "Duration threshold (minutes)",
            min_value=low,
            max_value=high,
            value=default,
            step=1,
            help="Counts and percentages use exact valid delivery durations from Cassandra.",
        )
        row = threshold_rows.get(int(threshold), {})
        above_count = row.get("deliveries_above")
        above_pct = row.get("above_pct")
        exact_note = "exact count from the cleaned records"
    else:
        threshold = int(round(documented_threshold if documented_threshold is not None else 33))
        above_count = metric(quality, "above_threshold_count")
        above_pct = metric(quality, "above_threshold_pct")
        exact_note = "documented notebook threshold"
        st.warning(
            "The exact threshold table is not available. Run the updated notebook or "
            "`scripts/init_cassandra.py`, then reload. The card below uses only the documented P75 count."
        )

    selected = selected_model_row(perf)
    mae = selected.get("mae") if selected else None
    model_name = str(selected.get("model")) if selected else "Model metrics unavailable"

    cards = st.columns(5)
    with cards[0]:
        kpi_card("Average delivery time", fmt_min(average), "mean of valid actual durations", "green")
    with cards[1]:
        kpi_card("Median delivery time", fmt_min(median), "robust central value", "yellow")
    with cards[2]:
        kpi_card(
            "Fastest / slowest",
            f"{fmt_min(fastest)} / {fmt_min(slowest)}",
            "minimum and maximum valid duration",
            "green",
        )
    with cards[3]:
        kpi_card(
            "Above duration threshold",
            f"{fmt_int(above_count)} · {fmt_pct(above_pct)}",
            f"duration > {threshold} min · {exact_note}",
            "yellow",
        )
    with cards[4]:
        kpi_card(
            "Prediction error (MAE)",
            fmt_min(mae) if mae is not None else "—",
            f"{model_name} · 20% held-out test set" if selected else "saved metrics not found",
            "green",
        )

    st.caption(
        f"{fmt_int(valid)} valid delivery records were analyzed after the notebook's documented "
        "cleaning rules (duration 1–120 minutes, valid Indian GPS, complete core categories)."
    )
    callout(
        "<strong>Threshold, not SLA.</strong> The source dataset has no promised delivery deadline, "
        "so the fourth card is a configurable duration-threshold metric. It measures how many "
        "observed deliveries took longer than the selected number of minutes; it is not an "
        "official late-delivery rate."
    )


def page_analytics(core: dict, metadata: dict) -> None:
    page_header(
        "Operations / Analytics",
        "Explore the delivery mix.",
        "Filter the pre-aggregated segment cube by real categorical fields. The distribution and "
        "comparison charts recalculate in the dashboard; no Spark job is rerun on refresh.",
        ["Interactive filters", "Cassandra segment cube"],
    )
    traffic = summary_frame(core, "traffic")
    weather = summary_frame(core, "weather")
    vehicle = summary_frame(core, "vehicle")
    city = summary_frame(core, "city")

    def values(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["All"]
        return ["All"] + sorted(df["category_value"].dropna().astype(str).tolist())

    st.markdown('<div class="qr-section-title">Filter deliveries</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    traffic_filter = f1.selectbox("Traffic", values(traffic), key="f_traffic")
    weather_filter = f2.selectbox("Weather", values(weather), key="f_weather")
    vehicle_filter = f3.selectbox("Vehicle", values(vehicle), key="f_vehicle")
    city_filter = f4.selectbox("City", values(city), key="f_city")

    filtered_rows = cca.filter_segments(
        core["segments"],
        traffic=traffic_filter,
        weather=weather_filter,
        vehicle=vehicle_filter,
        city=city_filter,
    )
    agg = cca.aggregate_segments(filtered_rows)
    if agg["deliveries"] == 0:
        st.warning("No pre-aggregated segment matches these filters. Try a broader combination.")
        return

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Matching deliveries", fmt_int(agg["deliveries"]), "weighted from segment cube", "green")
    with k2:
        kpi_card("Average duration", fmt_min(agg["avg_duration_min"]), "weighted by segment size", "yellow")
    with k3:
        kpi_card("Median (estimated)", fmt_min(agg["median_duration_min"]), "from 2-minute histogram bins", "green")
    with k4:
        kpi_card("Above-threshold share", fmt_pct(agg["late_rate"]), "clean-dataset P75 duration proxy", "yellow")

    left, right = st.columns([1.45, 1])
    with left:
        hist = filtered_histogram_frame(core, agg["histogram"])
        fig = go.Figure(
            go.Bar(
                x=hist["label"],
                y=hist["count"],
                marker_color=COLORS["yellow"],
                hovertemplate="<b>%{x}</b><br>Deliveries: %{y:,}<extra></extra>",
            )
        )
        fig.update_layout(title="Delivery-duration distribution", xaxis_title="Duration bucket", yaxis_title="Deliveries")
        st.plotly_chart(style_fig(fig, height=335), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        section_heading("Filter read-out")
        callout(
            f"<strong>{fmt_int(agg['deliveries'])}</strong> matching deliveries across "
            f"<strong>{len(filtered_rows)}</strong> stored traffic/weather/vehicle/city segments.<br><br>"
            f"Median and p90 are estimated from the stored 2-minute histogram buckets, "
            f"not recomputed from each order."
        )
        st.caption(
            "Filters are applied to the segment cube. The distance, age, and rating charts below use "
            "separate pre-aggregates over the full cleaned dataset."
        )

    section_heading("Comparison under the current filters")
    segment_df = pd.DataFrame(filtered_rows)
    compare_cols = st.columns(3)
    for container, column, title in [
        (compare_cols[0], "traffic_level", "Traffic"),
        (compare_cols[1], "weather", "Weather"),
        (compare_cols[2], "vehicle_type", "Vehicle"),
    ]:
        grouped = weighted_group(segment_df, column)
        grouped = grouped.rename(columns={column: "category_value"})
        grouped["sort_order"] = 99
        with container:
            st.plotly_chart(
                plot_theme_bar(grouped, "category_value", title, title, height=290),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    section_heading("Distance and partner signals", "Pre-aggregated bands from the full clean dataset.")
    d1, d2 = st.columns(2)
    with d1:
        distance = summary_frame(core, "distance_band")
        if not distance.empty:
            st.plotly_chart(
                plot_theme_bar(distance, "category_value", "Duration by route distance", "Distance band"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("Distance-band aggregates are not available.")
    with d2:
        age = summary_frame(core, "age_band")
        if not age.empty:
            st.plotly_chart(
                plot_theme_bar(age, "category_value", "Duration by partner age (proxy)", "Age band"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("Age-band aggregates are not available.")
    rating = summary_frame(core, "rating_band")
    if not rating.empty:
        section_heading("Partner rating and delivery time", "Age and rating are the only partner signals in this dataset; years of experience is not available.")
        st.plotly_chart(
            plot_theme_bar(rating, "category_value", "Duration by partner rating band", "Rating band"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


def reset_inputs(prefix: str) -> None:
    """Drop this form's widget state so the metadata defaults are shown again."""
    for key in list(st.session_state):
        if key.startswith(f"{prefix}_"):
            del st.session_state[key]


def page_prediction(metadata: dict) -> None:
    page_header(
        "Models / Prediction",
        "Estimate the next delivery.",
        "Enter the fields available before a delivery completes. The saved Random Forest pipeline "
        "returns an estimated duration in minutes; the saved classifier also adds a risk band.",
        ["No live traffic", "Minutes", "Estimate ≠ guarantee"],
    )
    if not metadata:
        st.error(
            "No trained model was found. Run `notebooks/Quickr_BDA_Analysis.ipynb` "
            "end-to-end, then reload this page."
        )
        st.code(
            "jupyter nbconvert --to notebook --execute --inplace notebooks/Quickr_BDA_Analysis.ipynb",
            language="powershell",
        )
        return

    left, right = st.columns([1.55, 1])
    with left:
        with st.form("prediction_form"):
            st.markdown('<div class="qr-section-title" style="margin-top:0;">Delivery inputs</div>', unsafe_allow_html=True)
            payload = render_inputs(metadata, prefix="pred")
            submitted = st.form_submit_button("Estimate delivery time")
        if submitted:
            try:
                result = ps.predict_delivery(payload)
            except ValueError as exc:
                st.error(str(exc))
            except ps.ModelNotAvailable as exc:
                st.error(str(exc))
            else:
                store_and_flash(result)
        if st.button("Reset inputs to dataset defaults", key="pred_reset", use_container_width=True):
            reset_inputs("pred")
            st.rerun()

    with right:
        st.markdown(
            '<div class="qr-callout"><strong>How to read this</strong><br><br>'
            "The Random Forest estimates duration from historic order attributes. Route distance is "
            "calculated with the Haversine formula, not a road-network API. A separate logistic "
            "regression scores duration-based risk using the same pre-delivery features. Its "
            "uncalibrated score is shown as a band, never as a guaranteed probability."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="qr-callout" style="margin-top:12px;"><strong>Model in use</strong><br><br>'
            f"{html.escape(str(metadata.get('model_name', '—')))}<br>"
            f"<span class='qr-source'>Version {html.escape(str(metadata.get('model_version', '—')))} · "
            f"trained on {fmt_int(metadata.get('training_rows', 0))} records · target in "
            f"{html.escape(str(metadata.get('target_unit', 'minutes')))}</span></div>",
            unsafe_allow_html=True,
        )

    if "last_prediction" in st.session_state:
        section_heading("Latest estimate", "Generated by the saved pipelines; also written to Cassandra.")
        render_prediction_result(st.session_state["last_prediction"])


def page_delay_risk(core: dict, metadata: dict) -> None:
    page_header(
        "Models / Delivery risk",
        "A three-band duration-risk estimate.",
        "The saved logistic-regression classifier scores the same pre-delivery features as the "
        "duration model. The bands are model-score groups, not calibrated real-world probabilities.",
        ["Training-only label", "No target leakage", "Estimate ≠ guarantee"],
    )
    risk_metrics = core["metrics"].get("delay_risk_performance", {})
    if not risk_metrics or not ps.risk_model_ready():
        st.info(
            "The risk classifier is not available yet. Run the updated notebook end-to-end to "
            "train and save `models/quickr_delay_risk_model.joblib` and its Cassandra metrics. "
            "No risk probability is invented when the classifier is missing."
        )
        st.code(
            "jupyter nbconvert --to notebook --execute --inplace notebooks\\\\Quickr_BDA_Analysis.ipynb",
            language="powershell",
        )
        return

    precision = metric(risk_metrics, "precision")
    recall = metric(risk_metrics, "recall")
    f1 = metric(risk_metrics, "f1")
    roc_auc = metric(risk_metrics, "roc_auc")
    threshold_label = metric(
        risk_metrics,
        "threshold_min",
        metric(risk_metrics, "training_threshold_min"),
    )
    threshold_text = f"{threshold_label:.0f}" if threshold_label is not None else "n/a"
    evaluation_cut = metric(risk_metrics, "probability_threshold", 0.5)
    medium_default = metric(risk_metrics, "medium_threshold", ps.DEFAULT_MEDIUM_THRESHOLD)
    high_default = metric(risk_metrics, "high_threshold", ps.DEFAULT_HIGH_THRESHOLD)
    medium_default = min(0.90, max(0.05, float(medium_default if medium_default is not None else 0.40)))
    high_default = min(0.99, max(medium_default + 0.05, float(high_default if high_default is not None else 0.65)))

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Precision", fmt_score(precision), "of flagged test deliveries, actually above threshold", "green")
    with cols[1]:
        kpi_card("Recall", fmt_score(recall), "above-threshold deliveries correctly caught", "yellow")
    with cols[2]:
        kpi_card("F1 score", fmt_score(f1), "harmonic mean at the 0.50 test cut-off", "green")
    with cols[3]:
        kpi_card("ROC-AUC", fmt_score(roc_auc), "ranking quality on held-out data", "yellow")

    st.markdown(
        f'<div class="qr-callout"><strong>What “risk” means here</strong><br><br>'
        f"The source dataset has no promised delivery deadline or official late/on-time label. "
        f"The classifier's positive class is <code>actual duration &gt; {threshold_text} minutes</code>, "
        f"where {threshold_text} minutes is the 75th percentile of the <em>training split only</em>. "
        f"This is an analytical duration-risk proxy, not a customer SLA and not a claim that any "
        f"delivery was contractually late.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Risk band cut-offs", expanded=False):
        c1, c2 = st.columns(2)
        medium_cut = c1.number_input(
            "Medium band starts at score",
            min_value=0.05,
            max_value=0.90,
            value=round(medium_default, 2),
            step=0.05,
            format="%.2f",
            key="risk_medium_cut",
            help="Scores below this value are shown as Low risk.",
        )
        high_cut = c2.number_input(
            "High band starts at score",
            min_value=0.10,
            max_value=0.99,
            value=round(high_default, 2),
            step=0.05,
            format="%.2f",
            key="risk_high_cut",
            help="Scores at or above this value are shown as High risk.",
        )
        st.caption(
            "Defaults are 0.40 and 0.65, chosen after checking the held-out score distribution. "
            "Changing these cut-offs changes only the band labels; the score itself is unchanged "
            "and remains uncalibrated."
        )
    cutoffs_valid = 0.0 < float(medium_cut) < float(high_cut) < 1.0
    if not cutoffs_valid:
        st.error("Cut-offs must satisfy 0 < medium < high < 1.")

    left, right = st.columns([1.1, 1])
    with left:
        if not metadata:
            st.error("The saved regression metadata is missing. Run the notebook to generate the artifacts.")
        else:
            with st.form("risk_assessment_form"):
                st.markdown(
                    '<div class="qr-section-title" style="margin-top:0;">Score a delivery</div>',
                    unsafe_allow_html=True,
                )
                risk_payload = render_risk_inputs(metadata, prefix="risk")
                submitted = st.form_submit_button("Assess delivery risk")
            if submitted:
                if not cutoffs_valid:
                    st.error("Fix the risk band cut-offs before scoring.")
                else:
                    try:
                        result = ps.predict_risk(
                            risk_payload,
                            medium_threshold=float(medium_cut),
                            high_threshold=float(high_cut),
                        )
                    except (ValueError, ps.ModelNotAvailable) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["last_risk"] = result
                        store_and_flash(result)

    with right:
        section_heading("Evaluation and label definition")
        st.markdown(
            f"""
            - Positive label: duration **> {threshold_text} min**.
            - Training positive rate: **{fmt_pct(metric(risk_metrics, "train_positive_rate"))}**
              ({fmt_int(metric(risk_metrics, "train_positives"))} of {fmt_int(metric(risk_metrics, "train_rows"))}).
            - Test positive rate: **{fmt_pct(metric(risk_metrics, "test_positive_rate"))}**
              ({fmt_int(metric(risk_metrics, "test_positives"))} of {fmt_int(metric(risk_metrics, "test_rows"))}).
            - Evaluation cut-off for precision/recall: **{evaluation_cut:.0%}**.
            - Displayed bands: **{medium_cut:.0%} / {high_cut:.0%}**.
            - Split: reproducible 80/20 holdout; the threshold is fitted on training durations only.
            - The split is unstratified because the label is defined after the training split; both class rates are reported above.
            - Inputs exclude actual duration, IDs, timestamps, and pickup/completion fields.
            """
        )
        st.caption(
            "The classifier is Logistic Regression with median imputation, one-hot encoding, "
            "and standard scaling fitted inside the training split."
        )
        base = metric(risk_metrics, "test_positive_rate")
        if base is not None and precision is not None and recall is not None:
            insight(
                "Compare with the base rate",
                f"About {base * 100:.1f}% of held-out deliveries exceed the threshold. "
                f"Precision {precision:.3f} and recall {recall:.3f} describe the model at the "
                "evaluation cut-off; they do not make the score a calibrated probability.",
            )

    if "last_risk" in st.session_state:
        section_heading("Latest risk assessment", "Generated by the saved pipelines and written to Cassandra.")
        render_risk_result(st.session_state["last_risk"])
    else:
        st.info("No risk assessment has been generated in this session. Use the form above.")

    try:
        stored = fetch_recent_predictions(5)
    except Exception:
        stored = []
    stored = [
        row for row in stored
        if row.get("risk_category") in {"Low risk", "Medium risk", "High risk"}
    ]
    if stored:
        with st.expander("Recent predictions stored in Cassandra"):
            frame = pd.DataFrame(
                [
                    {
                        "Created (UTC)": row.get("created_at"),
                        "Duration (min)": row.get("predicted_duration_min"),
                        "Risk score": row.get("late_risk_probability"),
                        "Risk band": row.get("risk_category"),
                        "Traffic": row.get("traffic_level"),
                    }
                    for row in stored
                ]
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.caption("Newest first from today's UTC prediction partition.")



def page_model_performance(core: dict, metadata: dict) -> None:
    page_header(
        "Models / Performance",
        "Metrics only. No charts.",
        "Every value below is read from the notebook's saved held-out evaluation and stored in "
        "Cassandra. The page is intentionally graph-free so the numbers can be checked quickly.",
        ["Held-out data", "One selected model", "No hardcoded scores"],
    )
    perf = core["metrics"].get("model_performance", {})
    comparisons = performance_frame(perf)
    selected = selected_model_row(perf)
    if comparisons.empty or selected is None:
        st.info(
            "No valid model metrics were found. Run the notebook end-to-end to train the model "
            "and publish model_performance metrics to Cassandra. No score is shown without a "
            "saved held-out evaluation."
        )
        return

    quality = core["metrics"].get("data_quality", {})
    app_meta = core["metrics"].get("app_meta", {})
    test_rows = metadata.get("test_rows") or metric(quality, "test_rows")
    train_rows = metadata.get("training_rows") or metric(quality, "train_rows")
    mae = selected.get("mae")
    rmse = selected.get("rmse")
    r2 = selected.get("r2")
    mape = selected.get("mape")

    cols = st.columns(4)
    with cols[0]:
        kpi_card("MAE", fmt_min(mae), "mean absolute error · held-out test set", "green")
    with cols[1]:
        kpi_card("RMSE", fmt_min(rmse), "root mean squared error · held-out test set", "yellow")
    with cols[2]:
        kpi_card("R²", fmt_score(r2), "fraction of variance explained", "green")
    with cols[3]:
        kpi_card("Held-out test records", fmt_int(test_rows), "evaluated once, not training predictions", "yellow")

    section_heading(
        "Evaluation table",
        "Genuine test-set values saved by the notebook. MAPE is shown only because all valid durations are positive.",
    )
    table = comparisons.copy()
    columns = [column for column in ("model", "mae", "rmse", "r2", "mape") if column in table]
    table = table[columns].rename(
        columns={"model": "Model", "mae": "MAE (min)", "rmse": "RMSE (min)", "r2": "R²", "mape": "MAPE"}
    )
    for column in ("MAE (min)", "RMSE (min)", "R²"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    if "MAPE" in table and pd.to_numeric(table["MAPE"], errors="coerce").notna().any():
        table["MAPE"] = pd.to_numeric(table["MAPE"], errors="coerce").map(
            lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "—"
        )
    else:
        table = table.drop(columns="MAPE", errors="ignore")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Rows show every model with saved held-out metrics. The KPI cards above use the lowest-RMSE model. "
        "Lower MAE and RMSE are better. R² = 1 is perfect; 0 is no better than predicting the mean."
    )

    run_id = (
        metadata.get("run_id")
        or text_metric(app_meta, "run_id", "")
        or text_metric(app_meta, "generated_at", "not recorded")
    )
    split_method = metadata.get(
        "split_method",
        "Reproducible random 80/20 holdout, random_state=42; no test rows used for training.",
    )
    section_heading("Model information")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
            - **Selected model:** {html.escape(str(selected.get("model", "—")))}
            - **Version:** {html.escape(str(metadata.get("model_version", "—")))}
            - **Evaluation run:** {html.escape(str(run_id))}
            - **Split:** {html.escape(str(split_method))}
            """
        )
    with right:
        st.markdown(
            f"""
            - **Training rows:** {fmt_int(train_rows)}
            - **Held-out test rows:** {fmt_int(test_rows)}
            - **Results source:** held-out test predictions, not training data
            - **Target:** delivery duration in {html.escape(str(metadata.get("target_unit", "minutes")))}
            - **Artifacts:** saved pipeline and test predictions under `models/`
            """
        )

    section_heading("Interpretation")
    if pd.notna(mae) and pd.notna(r2):
        insight(
            "What the saved numbers mean",
            f"The selected model's MAE is {float(mae):.2f} minutes on the held-out set, so a typical "
            f"absolute error is about {float(mae):.0f} minutes. R² is {float(r2):.3f}, meaning the "
            "model explains that share of the test-set duration variance. These are useful signals, "
            "but one holdout and one dataset do not make the model production-ready.",
        )
    st.caption(
        "Source: Cassandra table `delivery_metrics`, group `model_performance`, written by the "
        "notebook's evaluation cell. No value on this page is calculated from training predictions."
    )


def page_data_quality(core: dict, metadata: dict) -> None:
    page_header(
        "Trust / Data quality",
        "Know what the numbers leave out.",
        "Cleaning decisions are recorded as metrics and stored in Cassandra. The raw CSV is never "
        "modified; the clean CSV and model artifacts are reproducible from the notebook.",
        ["Raw preserved", "Exclusions documented"],
    )
    quality = core["metrics"].get("data_quality", {})
    missing = core["metrics"].get("missing_values", {})
    remaining = core["metrics"].get("missing_values_after_cleaning", {})
    dataset = core["metrics"].get("dataset", {})
    raw_rows = metric(quality, "raw_rows")
    final_rows = metric(quality, "final_rows")
    excluded = metric(quality, "excluded_total")
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Raw records", fmt_int(raw_rows), text_metric(dataset, "raw_filename", "data/raw/food_delivery.csv"), "green")
    with cols[1]:
        kpi_card("Excluded", fmt_int(excluded), "invalid GPS / missing core fields", "yellow")
    with cols[2]:
        kpi_card("Clean records", fmt_int(final_rows), "used for analytics and modelling", "green")
    with cols[3]:
        train_rows = metric(quality, "train_rows")
        test_rows = metric(quality, "test_rows")
        kpi_card("Train / test", f"{fmt_int(train_rows)} / {fmt_int(test_rows)}", "reproducible 80/20 split", "yellow")

    left, right = st.columns([1.1, 1])
    with left:
        section_heading("Rows removed by the cleaning pipeline")
        exclusions = pd.DataFrame(
            [
                ("Invalid target duration", metric(quality, "invalid_target_rows")),
                ("Duplicate records", metric(quality, "duplicate_rows_removed")),
                ("Invalid GPS / outside India", metric(quality, "invalid_gps_rows")),
                ("Route > 100 km", metric(quality, "distance_outlier_rows")),
                ("Missing traffic/weather/city", metric(quality, "missing_core_category_rows")),
                ("Invalid numeric ranges", metric(quality, "invalid_numeric_rows")),
                ("Total excluded", excluded),
                ("Final clean rows", final_rows),
            ],
            columns=["Check", "Records"],
        )
        exclusions["Records"] = exclusions["Records"].map(lambda value: f"{int(value):,}" if pd.notna(value) else "—")
        st.dataframe(exclusions, use_container_width=True, hide_index=True)
        st.caption(
            "Exclusions are sequential and mutually exclusive in the notebook. GPS errors include "
            "3,640 records with (0,0) coordinates; missing categories include literal `NaN` strings."
        )
    with right:
        section_heading("Missing values in the raw file")
        if missing:
            mdf = pd.DataFrame(
                [{"column": key, "missing": float(row.get("metric_value") or 0)} for key, row in missing.items()]
            ).sort_values("missing", ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=mdf["missing"],
                    y=mdf["column"],
                    orientation="h",
                    marker_color=[COLORS["yellow"] if v == mdf["missing"].max() else COLORS["sage"] for v in mdf["missing"]],
                    hovertemplate="<b>%{y}</b><br>%{x:,.0f} missing<extra></extra>",
                )
            )
            fig.update_layout(title="Raw missing-value counts", xaxis_title="Records", yaxis_title="")
            st.plotly_chart(style_fig(fig, height=330), use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Missing-value metrics are not available. Run notebook Section B/C.")

    section_heading("Imputation after cleaning", "Numeric gaps are retained in the clean dataset and filled inside the ML pipeline, fitted on training data only.")
    imp_cols = st.columns(3)
    labels = {
        "Delivery_person_Age": "Partner age",
        "Delivery_person_Ratings": "Partner rating",
        "multiple_deliveries": "Concurrent deliveries",
    }
    for container, column in zip(imp_cols, labels):
        value = metric(remaining, column, 0)
        with container:
            kpi_card(labels[column], fmt_int(value), "missing, median-imputed in pipeline", "yellow")

    section_heading("Dataset, license, and limits")
    source_url = text_metric(dataset, "source_url", "—")
    license_text = text_metric(dataset, "license", "No explicit license found in the source repository.")
    limitations = text_metric(dataset, "limitations", "—")
    left2, right2 = st.columns([1, 1])
    with left2:
        st.markdown(
            f'<div class="qr-callout"><strong>Source</strong><br>'
            f'<span class="qr-source">{html.escape(source_url)}<br>{html.escape(license_text)}</span>'
            f'<br><br><strong>Target</strong><br>{html.escape(text_metric(dataset, "target", "delivery duration in minutes"))}'
            f'<br><br><strong>Raw file</strong><br>{html.escape(text_metric(dataset, "raw_filename", "—"))} · '
            f'{fmt_int(raw_rows)} rows · {fmt_int(metric(quality, "raw_columns"))} columns</div>',
            unsafe_allow_html=True,
        )
        st.caption("The original CSV is kept in data/raw and is never overwritten.")
    with right2:
        st.markdown("**Limitations carried into the dashboard**")
        for item in [part.strip() for part in limitations.replace(" | ", "\n").split("\n") if part.strip()]:
            st.markdown(f"- {item}")

    with st.expander("Raw schema and provenance"):
        schema_text = text_metric(dataset, "schema", "Schema not recorded.")
        st.code(schema_text, language="text")
        st.markdown(
            "This project does **not** claim hourly demand peaks or geographic hotspots. The raw "
            "`Order_Date` range is partial and the source is known to contain synthetic/messy fields; "
            "using those fields to invent trends would be misleading."
        )


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
with st.sidebar:
    brand()
    side_label("Workspace")
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.markdown("---")
    side_label("System status")

    cass_ok = True
    cass_version = ""
    cass_error = ""
    try:
        probe = cca.connect()
        cass_version = cca.check_connection(probe)
        probe.shutdown()
    except Exception as exc:
        cass_ok = False
        cass_error = str(exc)

    status_row("Cassandra", f"v{cass_version.split('.')[0]}" if cass_version else "offline", cass_ok)
    core = None
    if cass_ok:
        try:
            core = load_core()
        except Exception as exc:
            cass_ok = False
            cass_error = str(exc)
            status_row("Analytics", "unavailable", False)
        else:
            quality = core["metrics"].get("data_quality", {})
            status_row("Dataset", f"{fmt_int(metric(quality, 'final_rows', 0))} valid", True)
            status_row("Keyspace", cca.CASSANDRA_KEYSPACE, True)
    else:
        status_row("Analytics", "offline", False)

    meta = ps.load_metadata()
    if meta:
        status_row("Model", str(meta.get("model_version", "ready")), True)
        status_row("Risk model", "ready" if ps.risk_model_ready() else "not trained", ps.risk_model_ready())
    else:
        status_row("Model", "not trained", False)
        status_row("Risk model", "not trained", False)

    st.markdown("---")
    if st.button("Reload data & models", use_container_width=True):
        load_core.clear()
        ps.clear_model_cache()
        st.rerun()
    st.markdown(
        '<div class="qr-side-note">Local BDA project · Spark 3.5.3 · Cassandra 4.1<br>'
        "No live traffic or customer data. Historical analytics are read from Cassandra; "
        "the saved model is loaded from disk.</div>",
        unsafe_allow_html=True,
    )

if not cass_ok or core is None:
    page_header(
        "Setup required",
        "Cassandra is not available.",
        "The dashboard deliberately shows no data fallback. Start the database, create the schema, "
        "and run the notebook once so the analytics tables are populated.",
        ["No mock data"],
    )
    st.error(cass_error or "Could not read analytics from Cassandra.")
    st.markdown("**Run these commands from the project root:**")
    st.code(
        "docker compose up -d\n"
        ".venv\\Scripts\\python.exe scripts\\init_cassandra.py\n"
        "jupyter nbconvert --to notebook --execute --inplace notebooks/Quickr_BDA_Analysis.ipynb\n"
        ".venv\\Scripts\\streamlit run app/app.py",
        language="powershell",
    )
    st.markdown(
        "If the container is starting, wait for it to become healthy: "
        "`docker ps --filter name=quickr-cassandra`."
    )
    st.stop()

if not core["metrics"].get("data_quality"):
    page_header(
        "Setup required",
        "Cassandra is running, but it has no Quickr analytics yet.",
        "The keyspace exists, but the notebook has not published its summaries. This is not an "
        "empty-dashboard fallback situation: run the Spark notebook once.",
        ["No synthetic data"],
    )
    st.code(
        "jupyter nbconvert --to notebook --execute --inplace notebooks/Quickr_BDA_Analysis.ipynb",
        language="powershell",
    )
    st.stop()

if page == "Overview":
    page_overview(core, meta)
elif page == "Delivery Performance":
    page_delivery_performance(core, meta)
elif page == "Delivery Analytics":
    page_analytics(core, meta)
elif page == "Prediction":
    page_prediction(meta)
elif page == "Delivery Risk":
    page_delay_risk(core, meta)
elif page == "Model Performance":
    page_model_performance(core, meta)
elif page == "Data Quality":
    page_data_quality(core, meta)
