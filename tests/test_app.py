import re
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
MODEL_NAMES = {"Logistic Regression", "Decision Tree", "Random Forest"}


def test_dashboard_and_predictions_for_each_model():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception
    model_metrics = {
        metric.label: metric
        for metric in app.metric
        if metric.label in MODEL_NAMES
    }
    assert set(model_metrics) == MODEL_NAMES
    for metric in model_metrics.values():
        accuracy = float(metric.value.rstrip("%"))
        assert 0 <= accuracy <= 100

    for model_name in sorted(MODEL_NAMES):
        app.selectbox[2].select(model_name)
        app.button[0].click().run()

        assert not app.exception

        probability_metric = next(
            metric
            for metric in app.metric
            if "Hiring Probability" in metric.label
        )
        probability_match = re.fullmatch(r"(\d+(?:\.\d+)?)%", probability_metric.value)
        assert probability_match is not None
        probability = float(probability_match.group(1))
        assert 0 <= probability <= 100

        result_messages = [element.value for element in app.success]
        result_messages.extend(element.value for element in app.error)
        placed = any("Xếp loại **Placed**!" in message for message in result_messages)
        not_placed = any(
            "Xếp loại **Not Placed**." in message for message in result_messages
        )

        assert placed != not_placed
        assert placed == (probability > 50)
