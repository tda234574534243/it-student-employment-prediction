import re
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
MODEL_NAMES = {"Logistic Regression", "Decision Tree", "Random Forest"}


def test_dashboard_and_predictions_for_each_model():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception
    assert not any("Nguồn dữ liệu" in widget.label for widget in app.selectbox)
    assert not any("HCMUE" in element.value for element in app.info)
    assert any("Holdout phân tầng 80/20" in element.value for element in app.header)
    assert (
        next(metric for metric in app.metric if metric.label == "Hồ sơ hợp lệ").value
        == "54,817"
    )
    model_metrics = {
        metric.label: metric
        for metric in app.metric
        if metric.label in MODEL_NAMES
    }
    assert set(model_metrics) == MODEL_NAMES
    for metric in model_metrics.values():
        score = float(metric.value.rstrip("%"))
        assert 0 <= score <= 100

    model_selector = next(
        widget for widget in app.selectbox if "Mô hình dự đoán" in widget.label
    )
    prediction_button = next(
        widget for widget in app.button if "Phân tích hồ sơ" in widget.label
    )

    for model_name in sorted(MODEL_NAMES):
        model_selector.select(model_name)
        prediction_button.click().run()
        assert not app.exception
        assert not any(
            "aptitude" in widget.label.lower() for widget in app.slider
        )

        probability_metric = next(
            metric
            for metric in app.metric
            if "Xác suất Placed trong mô hình" in metric.label
        )
        probability_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)%", probability_metric.value
        )
        assert probability_match is not None
        probability = float(probability_match.group(1))
        assert 0 <= probability <= 100

        messages = [element.value for element in app.success]
        messages.extend(element.value for element in app.warning)
        placed = any("Mô hình dự đoán: Placed." in message for message in messages)
        not_placed = any(
            "Mô hình dự đoán: Not Placed." in message for message in messages
        )
        assert placed != not_placed
        assert placed == (probability > 50)

    dataset_selector = next(widget for widget in app.selectbox if "Dataset" in widget.label)
    dataset_selector.select("Kaggle Student Placement · train.csv + test.csv có sẵn").run()
    assert not app.exception
    assert (
        next(metric for metric in app.metric if metric.label == "Hồ sơ hợp lệ").value
        == "17,937"
    )
    assert any("Test set Kaggle giữ riêng" in element.value for element in app.header)
    assert any("test giữ riêng gồm 2,039 hồ sơ" in element.value for element in app.info)
