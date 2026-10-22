"""Headless smoke test for all seven dashboard pages.

The test is skipped when Cassandra or the notebook artifacts are unavailable;
it never substitutes mock data. It runs the real app, including the real
Cassandra reads and saved model metadata.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"
PAGES = [
    "Overview",
    "Delivery Performance",
    "Delivery Analytics",
    "Prediction",
    "Delivery Risk",
    "Model Performance",
    "Data Quality",
]

os.environ.setdefault("CASSANDRA_HOST", "127.0.0.1")
os.environ.setdefault("CASSANDRA_PORT", "9043")

from app import cassandra_client as cca  # noqa: E402


def _cassandra_ready() -> bool:
    try:
        session = cca.connect(timeout=3.0)
    except Exception:
        return False
    try:
        groups = cca.fetch_metrics(session, groups=("data_quality", "app_meta"))
        return bool(groups["data_quality"]) and bool(groups["app_meta"])
    except Exception:
        return False
    finally:
        session.shutdown()


@pytest.mark.skipif(
    not (ROOT / "models" / "quickr_delivery_time_model.joblib").exists()
    or not (ROOT / "models" / "model_metadata.json").exists(),
    reason="Run the notebook to generate the model artifacts first",
)
def test_all_dashboard_pages_render_without_exceptions() -> None:
    if not _cassandra_ready():
        pytest.skip("Quickr Cassandra analytics are unavailable; start Cassandra and run the notebook")

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    app.run()
    assert not app.exception, f"Overview raised: {app.exception}"

    navigation = app.radio[0]
    for page in PAGES[1:]:
        app = navigation.set_value(page).run()
        assert not app.exception, f"{page} raised: {app.exception}"
        navigation = app.radio[0]

    # Recalculate the scorecard at a different threshold and verify the card
    # against the cleaned dataset, not a hard-coded expectation.
    app = navigation.set_value("Delivery Performance").run()
    duration_frame = pd.read_csv(ROOT / "data" / "processed" / "quickr_clean.csv")
    threshold = 20
    expected_count = int((duration_frame["duration_min"] > threshold).sum())
    expected_pct = expected_count / len(duration_frame)
    app = app.slider[0].set_value(threshold).run()
    rendered = " ".join(markdown.value for markdown in app.markdown)
    assert f"{expected_count:,}" in rendered
    assert f"{expected_pct:.1%}" in rendered

    navigation = app.radio[0]
    app = navigation.set_value("Model Performance").run()
    assert not app.exception, f"Model Performance raised: {app.exception}"
    assert len(app.get("plotly_chart")) == 0, "Model Performance must remain graph-free"

    # Use a fresh AppTest for the form submission so the check does not depend
    # on a long chain of widget states.
    risk_app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    risk_app.run()
    risk_app = risk_app.radio[0].set_value("Delivery Risk").run()
    submit = next(button for button in risk_app.button if "Assess" in button.label)
    risk_app = submit.click().run()
    assert not risk_app.exception, f"Delivery Risk submission raised: {risk_app.exception}"
    assert not risk_app.error, f"Delivery Risk returned an error: {risk_app.error}"
    assert risk_app.success, "the risk prediction should be saved to Cassandra"
