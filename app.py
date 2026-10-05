from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.placement_data import load_kaggle_train_test


st.set_page_config(
    page_title="Dự đoán cơ hội việc làm | CNTT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "it_placement_prediction_2026.csv"
)
PLACEMENT_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "kaggle_student_placement"
KAGGLE_TRAIN_FILE = PLACEMENT_DATASET_DIRECTORY / "train.csv"
KAGGLE_TEST_FILE = PLACEMENT_DATASET_DIRECTORY / "test.csv"
DATASET_CHOICES = [
    "Kaggle 2026 · CSE/IT (tách 80/20 trong app)",
    "Kaggle Student Placement · train.csv + test.csv có sẵn",
]
TARGET = "Placement_Status"
CATEGORICAL_FEATURES = ["Branch"]
NUMERIC_FEATURES = [
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Certifications",
    "Backlogs",
]
FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
FEATURE_LABELS = {
    "CGPA": "Điểm tích lũy (CGPA)",
    "Internships": "Số kỳ thực tập",
    "Projects": "Số đồ án",
    "Coding_Skills": "Kỹ năng lập trình",
    "Certifications": "Số chứng chỉ",
    "Backlogs": "Số môn nợ",
}
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
COLORS = {"Placed": "#159879", "Not Placed": "#e57870"}
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#142b3b; --muted:#617783; --line:#e3ebef; --mint:#159879; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:#f4f7f9; color:var(--ink); }
    [data-testid="stSidebar"] { background:#102c3b; }
    [data-testid="stSidebar"] * { color:#eef7f7; }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background:#f8fbfc; border-color:#dce8eb;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] * {
        color:#123746 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
        background:#193e4e !important; color:#eef7f7 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] * {
        color:#eef7f7 !important;
    }
    .hero {
        padding:2rem 2.2rem; border-radius:22px; color:white;
        background:linear-gradient(118deg,#102c3b 0%,#165b63 62%,#159879 100%);
        margin:.4rem 0 1.3rem; box-shadow:0 14px 35px rgba(16,44,59,.14);
    }
    .hero-kicker { color:#b9ecdc; text-transform:uppercase;
        letter-spacing:.14em; font-weight:700; font-size:.75rem; }
    .hero h1 { font-family:'Manrope',sans-serif;
        font-size:clamp(1.8rem,3vw,2.7rem); line-height:1.18;
        margin:.55rem 0; color:white; }
    .hero p { color:#e2eeee; margin:0; max-width:780px; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line);
        border-radius:16px; padding:16px 18px;
        box-shadow:0 5px 18px rgba(24,54,70,.04); }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stTabs"] button { font-weight:700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#10896c; }
    div[data-testid="stPlotlyChart"] { background:white;
        border:1px solid var(--line); border-radius:16px; padding:8px; }
    div[data-testid="stButton"] > button {
        border:0; border-radius:12px; color:white;
        background:linear-gradient(110deg,#118b70,#19b58a);
        font-weight:700; padding:.65rem 1rem;
    }
    div[data-testid="stButton"] > button:hover {
        color:white; border:0; background:linear-gradient(110deg,#0d765f,#139c77);
    }
    .section-note { color:#617783; font-size:.9rem; margin-top:-.55rem; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(FEATURES + [TARGET]).difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu các cột dữ liệu bắt buộc: {', '.join(missing)}")

    frame = frame[FEATURES + [TARGET]].copy()
    frame["Branch"] = frame["Branch"].astype("string").str.strip()
    frame[TARGET] = frame[TARGET].astype("string").str.strip()
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FEATURES + [TARGET])
    frame = frame[frame[TARGET].isin(["Placed", "Not Placed"])]
    frame = frame.reset_index(drop=True)
    if frame.empty or frame[TARGET].nunique() != 2:
        raise ValueError("Dataset phải có đủ hai nhãn Placed và Not Placed.")
    return frame


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "branch",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=3,
            min_samples_leaf=4,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=160,
            min_samples_leaf=3,
            random_state=42,
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def probability_for_placed(model: Pipeline, samples: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index("Placed")
    return model.predict_proba(samples)[:, class_index]


@st.cache_data(show_spinner="Đang đánh giá mô hình trên tập kiểm thử...")
def evaluate_models(data: pd.DataFrame, test_data: pd.DataFrame | None = None):
    X = data[FEATURES]
    y = data[TARGET]
    if test_data is None and y.value_counts().min() < 2:
        raise ValueError(
            "Mỗi nhãn cần tối thiểu 2 hồ sơ để tạo tập train/test phân tầng."
        )

    if test_data is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    else:
        X_train, y_train = X, y
        X_test = test_data[FEATURES]
        y_test = test_data[TARGET]
        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                "Cả tập train và test đều phải có hai nhãn placement."
            )

    models = make_models()
    metrics = {}
    holdout_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = probability_for_placed(model, X_test)
        metrics[name] = {
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
            "precision_macro": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test == "Placed", probabilities),
        }
        holdout_results[name] = {
            "probability": probabilities,
            "prediction": predictions,
        }
        if test_data is None:
            model.fit(X, y)
    return models, metrics, holdout_results, y_test


st.sidebar.markdown("## 🎓 Dự đoán việc làm CNTT")
st.sidebar.caption("Hồ sơ sinh viên CNTT · dự đoán placement")
dataset_choice = st.sidebar.selectbox("📚 Dataset:", DATASET_CHOICES)
test_df = None
if dataset_choice == DATASET_CHOICES[1]:
    try:
        df, test_df = load_kaggle_train_test(KAGGLE_TRAIN_FILE, KAGGLE_TEST_FILE)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Không thể nạp Kaggle train/test: {exc}")
        st.stop()
    source_description = (
        "Nguồn Kaggle Student Placement Dataset, giấy phép CC0; tác giả ghi rõ "
        "50.000 dòng là dữ liệu tổng hợp. Chỉ giữ nhánh CSE/IT. Mô hình fit trên "
        "train.csv; test.csv giữ riêng để đánh giá."
    )
    evaluation_description = (
        "Kaggle train.csv và test.csv có sẵn · test.csv không tham gia huấn luyện"
    )
    evaluation_header = "📊 Test set Kaggle giữ riêng"
else:
    if not DATA_FILE.exists():
        st.error(
            f"Chưa có dataset đã chuẩn hóa: `{DATA_FILE}`. "
            "Chạy `python scripts/prepare_external_dataset.py` sau khi đặt CSV "
            "nguồn trong `data/raw/kaggle_placement_prediction_2026/`."
        )
        st.stop()
    try:
        df = load_data(str(DATA_FILE))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Không thể nạp dataset: {exc}")
        st.stop()
    source_description = (
        "Nguồn Kaggle Student Placement Prediction Dataset 2026, giấy phép CC0; "
        "tác giả ghi rõ đây là dữ liệu tổng hợp. Chỉ giữ nhánh CSE/IT."
    )
    evaluation_description = "Holdout phân tầng 80/20 từ dataset đang chọn"
    evaluation_header = "📊 Holdout phân tầng 80/20"
    evaluation_help = (
        "Balanced accuracy trên holdout 20%. Mô hình dự đoán được fit lại "
        "trên toàn bộ dataset sau khi chấm điểm."
    )
if test_df is not None:
    evaluation_help = (
        "Balanced accuracy trên test.csv đã được Kaggle tách sẵn. "
        "Mô hình dự đoán chỉ được fit bằng các dòng CSE/IT của train.csv."
    )

try:
    models, evaluation_metrics, holdout_results, y_test = evaluate_models(df, test_df)
except ValueError as exc:
    st.error(f"Không thể đánh giá mô hình: {exc}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header(evaluation_header)
for model_name in MODEL_NAMES:
    st.sidebar.metric(
        model_name,
        f"{evaluation_metrics[model_name]['balanced_accuracy'] * 100:.1f}%",
        help=evaluation_help,
    )
st.sidebar.markdown("---")
chosen_model = st.sidebar.selectbox("🤖 Mô hình dự đoán:", MODEL_NAMES)
st.sidebar.caption(
    "Điểm số chỉ áp dụng cho tập kiểm thử của nguồn đang chọn; "
    "không phải cam kết tuyển dụng hoặc hiệu quả thực tế."
)

placed_count = int((df[TARGET] == "Placed").sum())
not_placed_count = int((df[TARGET] == "Not Placed").sum())
placement_rate = placed_count / len(df) * 100

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">CSE/IT · phân tích placement</div>
      <h1>Dự đoán cơ hội việc làm<br>của sinh viên CNTT mới ra trường</h1>
      <p>Khám phá dữ liệu placement, đánh giá mô hình trên tập kiểm thử và thử dự đoán cho một hồ sơ mới.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Hồ sơ hợp lệ", f"{len(df):,}")
kpi2.metric("Placed", f"{placed_count:,}")
kpi3.metric("Tỷ lệ Placed", f"{placement_rate:.1f}%")
kpi4.metric("Số nhãn", "2", "Placed · Not Placed")
source_notice = (
    f"{source_description} Dataset train đang dùng gồm {len(df):,} hồ sơ"
    + (
        f", test giữ riêng gồm {len(test_df):,} hồ sơ. "
        if test_df is not None
        else ". "
    )
    + "Điểm aptitude không được dùng làm đặc trưng. "
    + "Nhãn Placed/Not Placed là kết quả placement trong dữ liệu, chỉ được dùng "
    + "làm biến đại diện cho cơ hội việc làm — không chứng minh tình trạng có việc "
    + "sau tốt nghiệp hay khả năng áp dụng ra ngoài mẫu."
)
st.info(source_notice)

tab_predict, tab_overview, tab_models, tab_data = st.tabs(
    [
        "🔮  Dự đoán hồ sơ",
        "🌍  Toàn cảnh dữ liệu",
        "🧠  Đánh giá mô hình",
        "🗂️  Khám phá dữ liệu",
    ]
)

with tab_predict:
    st.subheader("Thử một hồ sơ CSE/IT")
    st.markdown(
        '<p class="section-note">Giá trị đầu vào được giới hạn theo khoảng quan sát trong dataset.</p>',
        unsafe_allow_html=True,
    )
    input_col, result_col = st.columns([1, 1.15], gap="large")
    with input_col:
        with st.container(border=True):
            st.markdown("#### 📝 Hồ sơ đầu vào")
            branch = st.selectbox("Chuyên ngành:", sorted(df["Branch"].unique()))
            left, right = st.columns(2)
            defaults = df[NUMERIC_FEATURES].median()
            bounds = df[NUMERIC_FEATURES].agg(["min", "max"])
            input_values = {}
            for index, feature in enumerate(NUMERIC_FEATURES):
                column = left if index % 2 == 0 else right
                minimum = float(bounds.loc["min", feature])
                maximum = float(bounds.loc["max", feature])
                default = float(defaults[feature])
                label = FEATURE_LABELS[feature]
                with column:
                    if feature in {
                        "Internships",
                        "Projects",
                        "Coding_Skills",
                        "Certifications",
                        "Backlogs",
                    }:
                        input_values[feature] = st.slider(
                            label,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(default)),
                        )
                    else:
                        input_values[feature] = st.slider(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=0.01 if feature == "CGPA" else 1.0,
                        )
    sample = pd.DataFrame(
        [{"Branch": branch, **input_values}],
        columns=FEATURES,
    )

    with result_col:
        with st.container(border=True):
            st.markdown("#### ✨ Kết quả dự đoán")
            st.caption(f"Mô hình: **{chosen_model}**")
            if st.button(
                "🚀  Phân tích hồ sơ",
                type="primary",
                use_container_width=True,
            ):
                model = models[chosen_model]
                probability = float(probability_for_placed(model, sample)[0])
                prediction = model.predict(sample)[0]
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={"suffix": "%", "font": {"size": 38}},
                        title={"text": "Xác suất dự đoán Placed"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#159879"},
                            "bgcolor": "#edf3f4",
                            "steps": [
                                {"range": [0, 50], "color": "#fff0ef"},
                                {"range": [50, 100], "color": "#e9f8f2"},
                            ],
                            "threshold": {
                                "line": {"color": "#163847", "width": 3},
                                "thickness": 0.8,
                                "value": 50,
                            },
                        },
                    )
                )
                gauge.update_layout(
                    height=260,
                    margin={"l": 25, "r": 25, "t": 65, "b": 5},
                    paper_bgcolor="white",
                    font={"color": "#142b3b"},
                )
                st.plotly_chart(gauge, width="stretch")
                st.metric("Xác suất Placed trong mô hình", f"{probability * 100:.1f}%")
                if prediction == "Placed":
                    st.success(
                        "**Mô hình dự đoán: Placed.** Đây là ước lượng thống kê "
                        "trên dữ liệu dự án, không đảm bảo kết quả tuyển dụng."
                    )
                else:
                    st.warning(
                        "**Mô hình dự đoán: Not Placed.** Đây là ước lượng thống kê "
                        "trên dữ liệu dự án, không đảm bảo kết quả tuyển dụng."
                    )
            else:
                st.info(
                    "Nhập hồ sơ bên trái rồi nhấn **Phân tích hồ sơ** để xem dự đoán."
                )
                st.caption(
                    "Các xác suất chưa được hiệu chỉnh/đánh giá calibration; "
                    "hãy xem đây là điểm số tương đối của mô hình."
                )

with tab_overview:
    st.subheader("Toàn cảnh hồ sơ trong dataset")
    st.markdown(
        '<p class="section-note">Mô tả dữ liệu đã lọc; không suy rộng thành tỷ lệ tuyển dụng của toàn thị trường.</p>',
        unsafe_allow_html=True,
    )
    status_col, branch_col = st.columns(2, gap="large")
    with status_col:
        status_counts = (
            df[TARGET].value_counts().rename_axis("Kết quả").reset_index(name="Hồ sơ")
        )
        chart = px.pie(
            status_counts,
            names="Kết quả",
            values="Hồ sơ",
            hole=0.62,
            color="Kết quả",
            color_discrete_map=COLORS,
            title="Tỷ trọng nhãn trong mẫu",
        )
        chart.update_layout(height=380, legend_title_text="")
        st.plotly_chart(chart, width="stretch")
    with branch_col:
        branch_summary = (
            df.groupby("Branch", observed=True)[TARGET]
            .agg(total="count", placed=lambda values: (values == "Placed").sum())
            .reset_index()
        )
        branch_summary["Tỷ lệ Placed (%)"] = (
            branch_summary["placed"] / branch_summary["total"] * 100
        )
        chart = px.bar(
            branch_summary,
            x="Branch",
            y="Tỷ lệ Placed (%)",
            color="Branch",
            text_auto=".1f",
            hover_data=["total", "placed"],
            title="Tỷ lệ Placed theo chuyên ngành",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        chart.update_layout(height=380, showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(chart, width="stretch")

    cgpa_col, skills_col = st.columns(2, gap="large")
    with cgpa_col:
        chart = px.histogram(
            df,
            x="CGPA",
            color=TARGET,
            marginal="box",
            barmode="overlay",
            opacity=0.72,
            color_discrete_map=COLORS,
            title="Phân bố CGPA theo kết quả",
        )
        chart.update_layout(height=400, legend_title_text="")
        st.plotly_chart(chart, width="stretch")
    with skills_col:
        chart = px.scatter(
            df,
            x="Coding_Skills",
            y="CGPA",
            color=TARGET,
            size="Projects",
            hover_data=["Branch", "Internships", "Certifications"],
            color_discrete_map=COLORS,
            title="CGPA và kỹ năng lập trình",
        )
        chart.update_layout(height=400, legend_title_text="")
        st.plotly_chart(chart, width="stretch")

with tab_models:
    st.subheader("So sánh mô hình trên tập kiểm thử")
    st.markdown(
        f'<p class="section-note">Các chỉ số được tính trên {evaluation_description}. '
        + (
            "Mô hình dự đoán được fit trên train.csv; "
            "test.csv không được dùng để fit."
            if test_df is not None
            else "Mô hình dự đoán sau đó được fit trên toàn bộ dataset."
        )
        + "</p>",
        unsafe_allow_html=True,
    )
    metric_labels = {
        "balanced_accuracy": "Balanced accuracy (%)",
        "f1_macro": "Macro F1 (%)",
        "precision_macro": "Macro precision (%)",
        "recall_macro": "Macro recall (%)",
        "roc_auc": "ROC AUC (%)",
    }
    summary_rows = []
    for name in MODEL_NAMES:
        row = {"Mô hình": name}
        for key, label in metric_labels.items():
            row[label] = evaluation_metrics[name][key] * 100
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    chart_data = summary.melt(
        id_vars="Mô hình",
        value_vars=list(metric_labels.values()),
        var_name="Chỉ số",
        value_name="Điểm (%)",
    )
    chart = px.bar(
        chart_data,
        x="Mô hình",
        y="Điểm (%)",
        color="Chỉ số",
        barmode="group",
        title="Chỉ số phân loại trên tập kiểm thử",
        color_discrete_sequence=["#159879", "#4287a0", "#e8a05a", "#916bb5", "#dc7181"],
    )
    chart.update_layout(height=420, yaxis_range=[0, 100], legend_title_text="")
    st.plotly_chart(chart, width="stretch")
    st.dataframe(
        summary.round(1),
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.1f")
            for column in summary.columns
            if column != "Mô hình"
        },
    )
    matrix_col, roc_col = st.columns(2, gap="large")
    selected_predictions = holdout_results[chosen_model]["prediction"]
    selected_probabilities = holdout_results[chosen_model]["probability"]
    with matrix_col:
        matrix = confusion_matrix(
            y_test,
            selected_predictions,
            labels=["Not Placed", "Placed"],
        )
        chart = px.imshow(
            matrix,
            x=["Dự đoán: Not Placed", "Dự đoán: Placed"],
            y=["Thực tế: Not Placed", "Thực tế: Placed"],
            text_auto=True,
            color_continuous_scale=["#edf4f5", "#168b70"],
            title=f"Ma trận nhầm lẫn · {chosen_model}",
        )
        chart.update_layout(height=370, coloraxis_showscale=False)
        st.plotly_chart(chart, width="stretch")
    with roc_col:
        chart = go.Figure()
        for name in MODEL_NAMES:
            probabilities = holdout_results[name]["probability"]
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test == "Placed", probabilities
            )
            chart.add_trace(
                go.Scatter(
                    x=false_positive_rate,
                    y=true_positive_rate,
                    mode="lines",
                    name=f"{name} · AUC {auc(false_positive_rate, true_positive_rate):.2f}",
                )
            )
        chart.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ngẫu nhiên",
                line={"dash": "dash", "color": "#9aaab2"},
            )
        )
        chart.update_layout(
            title="ROC trên tập kiểm thử",
            height=370,
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend_title_text="",
        )
        st.plotly_chart(chart, width="stretch")

    fitted = models[chosen_model]
    classifier = fitted.named_steps["classifier"]
    feature_names = fitted.named_steps["preprocess"].get_feature_names_out()
    if chosen_model == "Logistic Regression":
        raw_importance = np.abs(classifier.coef_[0])
        explanation = (
            "Giá trị tuyệt đối của hệ số sau one-hot encoding/chuẩn hóa; "
            "không biểu thị quan hệ nhân quả."
        )
    else:
        raw_importance = classifier.feature_importances_
        explanation = (
            "Feature importance dựa trên giảm impurity; "
            "không biểu thị quan hệ nhân quả."
        )
    importance = pd.DataFrame(
        {
            "Đặc trưng": [
                name.replace("branch__", "Chuyên ngành: ")
                .replace("numeric__", "")
                .replace("_", " ")
                for name in feature_names
            ],
            "Mức quan trọng": raw_importance,
        }
    ).sort_values("Mức quan trọng", ascending=True)
    chart = px.bar(
        importance,
        x="Mức quan trọng",
        y="Đặc trưng",
        orientation="h",
        color="Mức quan trọng",
        color_continuous_scale=["#cceee3", "#10876b"],
        title=f"Đặc trưng của {chosen_model}",
    )
    chart.update_layout(height=430, coloraxis_showscale=False)
    st.plotly_chart(chart, width="stretch")
    st.caption(explanation)

with tab_data:
    st.subheader("Kiểm tra dữ liệu đã dùng")
    st.caption(
        "Các biến nhân khẩu học/ID không dùng làm đầu vào. "
        "Gender bị loại để không dùng thuộc tính nhạy cảm trong mô hình."
    )
    numeric_correlation = df[NUMERIC_FEATURES].copy()
    numeric_correlation["Placed (0/1)"] = (df[TARGET] == "Placed").astype(int)
    correlation = numeric_correlation.corr()
    chart = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        title="Tương quan số học giữa các biến số",
    )
    chart.update_layout(height=570, coloraxis_colorbar_title="r")
    st.plotly_chart(chart, width="stretch")
    st.dataframe(df, width="stretch", hide_index=True)

st.caption(
    "Lưu ý: nhãn placement không tương đương với việc làm bền vững sau tốt nghiệp. "
    "Muốn kết luận cho đề tài thực tế cần dataset có nguồn gốc, cỡ mẫu và thời điểm "
    "theo dõi việc làm được xác minh."
)
