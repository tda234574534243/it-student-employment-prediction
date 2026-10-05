import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Campus | Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = r"D:\Machine-Course\Placement Dataset\Student Placement Dataset\train.csv"
TARGET = "Placement_Status"
FEATURES = [
    "Degree_Encoded",
    "Branch_Encoded",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Certifications",
    "Backlogs",
]
FEATURE_LABELS = {
    "Degree_Encoded": "Bằng cấp",
    "Branch_Encoded": "Chuyên ngành",
    "CGPA": "Điểm tích lũy",
    "Internships": "Kỳ thực tập",
    "Projects": "Đồ án",
    "Coding_Skills": "Kỹ năng coding",
    "Certifications": "Chứng chỉ",
    "Backlogs": "Môn nợ",
}
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
COLORS = {"Placed": "#19b58a", "Not Placed": "#f07878"}


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #142b3b; --muted: #718391; --line: #e8eef2; --mint: #19b58a; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f8fa; color: var(--ink); }
    [data-testid="stSidebar"] { background: #102c3b; }
    [data-testid="stSidebar"] * { color: #eef7f7; }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #f8fbfc; border-color: #dce8eb;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] * { color: #123746 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
        background: #193e4e !important; color: #eef7f7 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] * {
        color: #eef7f7 !important;
    }
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px; color: white;
        background: linear-gradient(118deg, #102c3b 0%, #165b63 62%, #19a783 100%);
        margin: .4rem 0 1.3rem 0; box-shadow: 0 14px 35px rgba(16,44,59,.14);
    }
    .hero-kicker { color: #a7e8d5; text-transform: uppercase; letter-spacing: .14em;
        font-weight: 700; font-size: .75rem; }
    .hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 3vw, 2.7rem);
        line-height: 1.18; margin: .55rem 0; color: white; }
    .hero p { color: #d7e9e9; margin: 0; max-width: 760px; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line);
        border-radius: 16px; padding: 16px 18px; box-shadow: 0 5px 18px rgba(24,54,70,.04); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stTabs"] button { font-weight: 700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #10896c; }
    div[data-testid="stPlotlyChart"] { background: white; border: 1px solid var(--line);
        border-radius: 16px; padding: 8px; }
    div[data-testid="stButton"] > button {
        border: 0; border-radius: 12px; color: white;
        background: linear-gradient(110deg, #118b70, #19b58a);
        font-weight: 700; padding: .65rem 1rem;
    }
    div[data-testid="stButton"] > button:hover {
        color: white; border: 0; background: linear-gradient(110deg, #0d765f, #139c77);
    }
    .section-note { color: #718391; font-size: .9rem; margin-top: -.55rem; }
    .result-card { background: white; border: 1px solid #e8eef2; border-radius: 18px;
        padding: 1.2rem 1.4rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_raw_data():
    if os.path.exists(DATA_FILE):
        frame = pd.read_csv(DATA_FILE)
        st.sidebar.success("✅ Đã kết nối thành công file train.csv!")
    else:
        st.sidebar.warning(
            f"⚠️ Không tìm thấy file tại '{DATA_FILE}'. Đang dùng dữ liệu mô phỏng."
        )
        np.random.seed(42)
        sample_count = 500
        frame = pd.DataFrame(
            {
                "Degree": np.random.choice(
                    ["B.Tech", "BCA", "MCA", "B.Sc"], sample_count
                ),
                "Branch": np.random.choice(
                    ["CSE", "ECE", "IT", "ME", "Civil"], sample_count
                ),
                "CGPA": np.round(np.random.uniform(5.5, 9.5, sample_count), 2),
                "Internships": np.random.randint(0, 3, sample_count),
                "Projects": np.random.randint(1, 5, sample_count),
                "Coding_Skills": np.random.randint(1, 10, sample_count),
                "Certifications": np.random.randint(0, 4, sample_count),
                "Backlogs": np.random.randint(0, 4, sample_count),
                TARGET: np.random.choice(["Placed", "Not Placed"], sample_count),
            }
        )
    return frame


df_raw = get_raw_data()
df_raw.columns = df_raw.columns.str.strip()
required_columns = {
    "Degree",
    "Branch",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Certifications",
    "Backlogs",
    TARGET,
}
missing_columns = sorted(required_columns.difference(df_raw.columns))
if missing_columns:
    st.error(f"Thiếu cột bắt buộc trong dữ liệu: {', '.join(missing_columns)}")
    st.stop()

df_clean = df_raw.dropna(subset=list(required_columns)).copy()
df_clean["Degree"] = df_clean["Degree"].astype(str).str.strip()
df_clean["Branch"] = df_clean["Branch"].astype(str).str.strip()
df_clean[TARGET] = df_clean[TARGET].astype(str).str.strip()
df_clean = df_clean[df_clean[TARGET].isin(["Placed", "Not Placed"])]

if df_clean.empty or df_clean[TARGET].nunique() != 2:
    st.error("Dữ liệu cần có ít nhất một hồ sơ cho mỗi nhãn Placement_Status.")
    st.stop()

le_degree = LabelEncoder()
le_branch = LabelEncoder()
df_clean["Degree_Encoded"] = le_degree.fit_transform(df_clean["Degree"])
df_clean["Branch_Encoded"] = le_branch.fit_transform(df_clean["Branch"])
df_clean["PlacementStatus_Encoded"] = df_clean[TARGET].map(
    {"Placed": 1, "Not Placed": 0}
)

X = df_clean[FEATURES]
y = df_clean["PlacementStatus_Encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42,
    ),
}

accuracies = {}
test_predictions = {}
test_probabilities = {}
for name, model in models.items():
    train_features = X_train_scaled if name == "Logistic Regression" else X_train
    test_features = X_test_scaled if name == "Logistic Regression" else X_test
    model.fit(train_features, y_train)
    test_predictions[name] = model.predict(test_features)
    test_probabilities[name] = model.predict_proba(test_features)[:, 1]
    accuracies[name] = accuracy_score(y_test, test_predictions[name])

st.sidebar.markdown("## 🎓 Campus")
st.sidebar.caption("Placement analytics · Student outcomes")
st.sidebar.markdown("---")
st.sidebar.header("📊 Hiệu năng mô hình")
for name, accuracy in accuracies.items():
    st.sidebar.metric(name, f"{accuracy * 100:.2f}%")
st.sidebar.markdown("---")
chosen_model = st.sidebar.selectbox("🤖 Mô hình dự đoán:", list(models.keys()))
model_active = models[chosen_model]
st.sidebar.caption(
    "Điểm số được tính trên tập kiểm tra giữ riêng, không phải cam kết kết quả tuyển dụng."
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Student outcomes · Machine learning</div>
      <h1>Biến dữ liệu sinh viên thành<br>những quyết định rõ ràng hơn.</h1>
      <p>Khám phá xu hướng placement, so sánh mô hình và thử phân tích một hồ sơ sinh viên trong cùng một không gian.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

placed_count = int((df_clean[TARGET] == "Placed").sum())
not_placed_count = int((df_clean[TARGET] == "Not Placed").sum())
placement_rate = placed_count / len(df_clean) * 100
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Tổng hồ sơ", f"{len(df_clean):,}", help="Số dòng hợp lệ dùng trong ứng dụng")
kpi2.metric("Đã placement", f"{placed_count:,}")
kpi3.metric("Tỷ lệ placement", f"{placement_rate:.1f}%")
kpi4.metric("Mô hình đang chọn", chosen_model)

tab_predict, tab_overview, tab_models, tab_data = st.tabs(
    [
        "🔮  Dự đoán hồ sơ",
        "🌍  Toàn cảnh dữ liệu",
        "🧠  Phân tích mô hình",
        "🗂️  Khám phá dữ liệu",
    ]
)

with tab_predict:
    st.subheader("Thử một hồ sơ sinh viên")
    st.markdown(
        '<p class="section-note">Điều chỉnh các thông số bên dưới và xem mô hình ước lượng mức độ phù hợp.</p>',
        unsafe_allow_html=True,
    )
    input_col, result_col = st.columns([1, 1.15], gap="large")
    with input_col:
        with st.container(border=True):
            st.markdown("#### 📝 Hồ sơ đầu vào")
            input_degree = st.selectbox("Bằng cấp (Degree):", le_degree.classes_)
            input_branch = st.selectbox("Chuyên ngành (Branch):", le_branch.classes_)
            first_row, second_row = st.columns(2)
            with first_row:
                input_cgpa = st.slider(
                    "CGPA", 4.0, 10.0, 7.2, step=0.01
                )
                input_intern = st.number_input(
                    "Kỳ thực tập", min_value=0, max_value=5, value=1
                )
                input_project = st.slider("Đồ án", 0, 10, 2)
            with second_row:
                input_code = st.slider("Kỹ năng coding", 1, 10, 6)
                input_cert = st.number_input(
                    "Chứng chỉ", min_value=0, max_value=10, value=1
                )
                input_backlog = st.number_input(
                    "Môn nợ", min_value=0, max_value=10, value=0
                )

    user_features = pd.DataFrame(
        [
            [
                le_degree.transform([input_degree])[0],
                le_branch.transform([input_branch])[0],
                input_cgpa,
                input_intern,
                input_project,
                input_code,
                input_cert,
                input_backlog,
            ]
        ],
        columns=FEATURES,
    )

    with result_col:
        with st.container(border=True):
            st.markdown("#### ✨ Kết quả & gợi ý")
            st.caption(f"Mô hình đang xử lý: **{chosen_model}**")
            if st.button(
                "🚀  Phân tích hồ sơ",
                type="primary",
                use_container_width=True,
            ):
                model_input = (
                    scaler.transform(user_features)
                    if chosen_model == "Logistic Regression"
                    else user_features
                )
                prediction = model_active.predict(model_input)[0]
                probability = model_active.predict_proba(model_input)[0][1]
                probability_pct = probability * 100
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability_pct,
                        number={"suffix": "%", "font": {"size": 38}},
                        title={"text": "Xác suất dự đoán"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#19b58a"},
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
                    height=270,
                    margin={"l": 25, "r": 25, "t": 65, "b": 10},
                    paper_bgcolor="white",
                    font={"color": "#142b3b"},
                )
                st.plotly_chart(gauge, width="stretch")
                st.metric(
                    label="Xác suất trúng tuyển (Hiring Probability)",
                    value=f"{probability_pct:.2f}%",
                )
                if prediction == 1:
                    st.success(
                        "🎉 **Kết quả:** Xếp loại **Placed**! Hồ sơ của bạn đáp ứng rất tốt tiêu chí tuyển dụng cơ bản."
                    )
                    st.balloons()
                else:
                    st.error(
                        "⚠️ **Kết quả:** Xếp loại **Not Placed**. Hồ sơ hiện tại đang gặp bất lợi cạnh tranh."
                    )
                    suggestions = []
                    if input_backlog > 0:
                        suggestions.append(
                            "Ưu tiên cải thiện kết quả học tập và giảm số môn nợ."
                        )
                    if input_code < 6:
                        suggestions.append(
                            "Luyện coding đều đặn qua bài tập thuật toán và dự án cá nhân."
                        )
                    if input_intern == 0:
                        suggestions.append(
                            "Tìm cơ hội thực tập để tích lũy kinh nghiệm thực tế."
                        )
                    if not suggestions:
                        suggestions.append(
                            "Tiếp tục nâng cao hồ sơ qua dự án, kỹ năng và kinh nghiệm thực tế."
                        )
                    st.markdown("**Một vài hướng cải thiện**")
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
            else:
                st.info(
                    "Điền hồ sơ bên trái, sau đó nhấn **Phân tích hồ sơ** để xem xác suất dự đoán."
                )
                st.caption(
                    "Đây là kết quả của mô hình học máy trên dữ liệu hiện có, không phải lời hứa về tuyển dụng."
                )

with tab_overview:
    st.subheader("Toàn cảnh hồ sơ sinh viên")
    st.markdown(
        '<p class="section-note">Phân bố kết quả và các mối liên hệ trong tập dữ liệu đã nạp.</p>',
        unsafe_allow_html=True,
    )
    distribution_col, rate_col = st.columns(2, gap="large")
    with distribution_col:
        status_counts = (
            df_clean[TARGET]
            .value_counts()
            .rename_axis("Kết quả")
            .reset_index(name="Số hồ sơ")
        )
        donut = px.pie(
            status_counts,
            names="Kết quả",
            values="Số hồ sơ",
            hole=0.64,
            color="Kết quả",
            color_discrete_map=COLORS,
            title="Tỷ trọng kết quả placement",
        )
        donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker={"line": {"color": "white", "width": 3}},
        )
        donut.update_layout(
            height=390,
            margin={"l": 15, "r": 15, "t": 65, "b": 15},
            legend_title_text="",
        )
        st.plotly_chart(donut, width="stretch")
    with rate_col:
        branch_stats = (
            df_clean.groupby("Branch", observed=True)[TARGET]
            .apply(lambda values: (values == "Placed").mean() * 100)
            .rename("Tỷ lệ placement (%)")
            .reset_index()
            .sort_values("Tỷ lệ placement (%)", ascending=True)
        )
        branch_chart = px.bar(
            branch_stats,
            x="Tỷ lệ placement (%)",
            y="Branch",
            orientation="h",
            text_auto=".1f",
            color="Tỷ lệ placement (%)",
            color_continuous_scale=["#b7e9d9", "#13876d"],
            title="Tỷ lệ placement theo chuyên ngành",
        )
        branch_chart.update_layout(
            height=390,
            margin={"l": 15, "r": 20, "t": 65, "b": 25},
            coloraxis_showscale=False,
            xaxis_title="Tỷ lệ placement (%)",
            yaxis_title="",
        )
        st.plotly_chart(branch_chart, width="stretch")

    cgpa_col, skills_col = st.columns(2, gap="large")
    with cgpa_col:
        cgpa_chart = px.histogram(
            df_clean,
            x="CGPA",
            color=TARGET,
            marginal="box",
            barmode="overlay",
            opacity=0.72,
            nbins=28,
            color_discrete_map=COLORS,
            title="Phân bố CGPA theo kết quả",
        )
        cgpa_chart.update_layout(
            height=410,
            margin={"l": 15, "r": 15, "t": 65, "b": 25},
            xaxis_title="CGPA",
            yaxis_title="Số hồ sơ",
            legend_title_text="",
        )
        st.plotly_chart(cgpa_chart, width="stretch")
    with skills_col:
        scatter_data = df_clean.sample(
            n=min(3500, len(df_clean)), random_state=42
        )
        scatter = px.scatter(
            scatter_data,
            x="CGPA",
            y="Coding_Skills",
            color=TARGET,
            size="Projects",
            hover_data=["Branch", "Internships", "Certifications"],
            opacity=0.62,
            color_discrete_map=COLORS,
            title="CGPA và kỹ năng coding",
        )
        scatter.update_layout(
            height=410,
            margin={"l": 15, "r": 15, "t": 65, "b": 25},
            xaxis_title="CGPA",
            yaxis_title="Kỹ năng coding",
            legend_title_text="",
        )
        st.plotly_chart(scatter, width="stretch")

with tab_models:
    st.subheader("So sánh và hiểu hiệu năng mô hình")
    st.markdown(
        '<p class="section-note">Các chỉ số được tính trên cùng một tập test độc lập.</p>',
        unsafe_allow_html=True,
    )
    accuracy_data = pd.DataFrame(
        {
            "Mô hình": list(models.keys()),
            "Accuracy (%)": [accuracies[name] * 100 for name in models],
            "Precision (%)": [
                precision_score(y_test, test_predictions[name], zero_division=0) * 100
                for name in models
            ],
            "Recall (%)": [
                recall_score(y_test, test_predictions[name], zero_division=0) * 100
                for name in models
            ],
        }
    )
    score_chart = px.bar(
        accuracy_data.melt(
            id_vars="Mô hình", var_name="Chỉ số", value_name="Điểm (%)"
        ),
        x="Mô hình",
        y="Điểm (%)",
        color="Chỉ số",
        barmode="group",
        text_auto=".1f",
        color_discrete_sequence=["#19b58a", "#4389a2", "#efaa62"],
        title="Accuracy, precision và recall",
    )
    score_chart.update_layout(
        height=410,
        margin={"l": 15, "r": 15, "t": 65, "b": 25},
        yaxis_range=[0, 100],
        legend_title_text="",
    )
    st.plotly_chart(score_chart, width="stretch")

    matrix_col, roc_col = st.columns(2, gap="large")
    with matrix_col:
        matrix = confusion_matrix(y_test, test_predictions[chosen_model], labels=[0, 1])
        matrix_chart = px.imshow(
            matrix,
            x=["Dự đoán: Not Placed", "Dự đoán: Placed"],
            y=["Thực tế: Not Placed", "Thực tế: Placed"],
            text_auto=True,
            color_continuous_scale=["#edf4f5", "#168b70"],
            aspect="auto",
            title=f"Ma trận nhầm lẫn · {chosen_model}",
        )
        matrix_chart.update_layout(
            height=390,
            margin={"l": 15, "r": 15, "t": 65, "b": 50},
            coloraxis_showscale=False,
        )
        st.plotly_chart(matrix_chart, width="stretch")
    with roc_col:
        roc_chart = go.Figure()
        for name in models:
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test, test_probabilities[name]
            )
            roc_auc = auc(false_positive_rate, true_positive_rate)
            roc_chart.add_trace(
                go.Scatter(
                    x=false_positive_rate,
                    y=true_positive_rate,
                    mode="lines",
                    name=f"{name} · AUC {roc_auc:.3f}",
                )
            )
        roc_chart.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ngẫu nhiên",
                line={"dash": "dash", "color": "#9aaab2"},
            )
        )
        roc_chart.update_layout(
            title="Đường cong ROC",
            height=390,
            margin={"l": 15, "r": 15, "t": 65, "b": 45},
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend_title_text="",
        )
        st.plotly_chart(roc_chart, width="stretch")

    if chosen_model == "Logistic Regression":
        importance_values = np.abs(model_active.coef_[0])
        importance_note = (
            "Logistic Regression: hệ số tuyệt đối trên dữ liệu đã chuẩn hóa; "
            "không thể diễn giải là quan hệ nhân quả."
        )
    else:
        importance_values = model_active.feature_importances_
        importance_note = (
            "Tree models: độ quan trọng theo mức giảm impurity; "
            "không thể diễn giải là quan hệ nhân quả."
        )
    importance_data = pd.DataFrame(
        {
            "Đặc trưng": [FEATURE_LABELS[name] for name in FEATURES],
            "Mức quan trọng": importance_values,
        }
    ).sort_values("Mức quan trọng", ascending=True)
    importance_chart = px.bar(
        importance_data,
        x="Mức quan trọng",
        y="Đặc trưng",
        orientation="h",
        text_auto=".3f",
        color="Mức quan trọng",
        color_continuous_scale=["#cceee3", "#10876b"],
        title=f"Đặc trưng quan trọng · {chosen_model}",
    )
    importance_chart.update_layout(
        height=410,
        margin={"l": 15, "r": 20, "t": 65, "b": 25},
        coloraxis_showscale=False,
        yaxis_title="",
    )
    st.plotly_chart(importance_chart, width="stretch")
    st.caption(importance_note)
    st.info(
        "Accuracy/precision/recall/AUC mô tả hiệu quả trên tập test của dataset này; "
        "không đảm bảo khả năng dự đoán cho thị trường tuyển dụng thực tế."
    )

with tab_data:
    st.subheader("Khám phá và lọc dữ liệu")
    st.markdown(
        '<p class="section-note">Tương tác với biểu đồ và bảng để kiểm tra các nhóm hồ sơ.</p>',
        unsafe_allow_html=True,
    )
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        degree_filter = st.multiselect(
            "Lọc bằng cấp",
            options=sorted(df_clean["Degree"].unique()),
            default=sorted(df_clean["Degree"].unique()),
        )
    with filter_col2:
        status_filter = st.multiselect(
            "Lọc kết quả",
            options=["Placed", "Not Placed"],
            default=["Placed", "Not Placed"],
        )

    filtered_data = df_clean[
        df_clean["Degree"].isin(degree_filter)
        & df_clean[TARGET].isin(status_filter)
    ]
    st.caption(f"Đang hiển thị {len(filtered_data):,} / {len(df_clean):,} hồ sơ")
    left_chart, right_chart = st.columns(2, gap="large")
    with left_chart:
        degree_counts = (
            filtered_data.groupby(["Degree", TARGET], observed=True)
            .size()
            .reset_index(name="Số hồ sơ")
        )
        degree_chart = px.bar(
            degree_counts,
            x="Degree",
            y="Số hồ sơ",
            color=TARGET,
            barmode="group",
            color_discrete_map=COLORS,
            title="Số hồ sơ theo bằng cấp",
        )
        degree_chart.update_layout(height=380, legend_title_text="")
        st.plotly_chart(degree_chart, width="stretch")
    with right_chart:
        correlation_columns = [
            "CGPA",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Certifications",
            "Backlogs",
            "PlacementStatus_Encoded",
        ]
        correlation = filtered_data[correlation_columns].corr(numeric_only=True)
        correlation.index = [
            "CGPA",
            "Thực tập",
            "Đồ án",
            "Coding",
            "Chứng chỉ",
            "Môn nợ",
            "Placement",
        ]
        correlation.columns = correlation.index
        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Tương quan giữa các thuộc tính số",
        )
        heatmap.update_layout(height=380, coloraxis_colorbar_title="r")
        st.plotly_chart(heatmap, width="stretch")
    display_columns = [
        column
        for column in [
            "Student_ID",
            "Degree",
            "Branch",
            "CGPA",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Certifications",
            "Backlogs",
            TARGET,
        ]
        if column in filtered_data.columns
    ]
    st.dataframe(
        filtered_data[display_columns].head(500),
        width="stretch",
        hide_index=True,
    )
    st.caption("Bảng giới hạn 500 dòng đầu để giữ giao diện phản hồi nhanh.")
