<div align="center">

# 🎓 Placement Pulse

### Dự đoán placement cho sinh viên CNTT mới ra trường

**Một dashboard Streamlit để khám phá dữ liệu, so sánh mô hình ML và thử dự đoán hồ sơ CSE/IT.**

<br>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/Data%20license-CC0-159879?style=for-the-badge)

</div>

---

## ✨ Tổng quan

Placement Pulse giúp bạn:

- Khám phá phân bố placement và đặc trưng hồ sơ trên dashboard tương tác.
- So sánh **Logistic Regression**, **Decision Tree** và **Random Forest**.
- Xem confusion matrix, ROC curve và mức độ quan trọng của đặc trưng.
- Thử dự đoán cho một hồ sơ CSE/IT mới.
- Chọn giữa hai nguồn dữ liệu Kaggle; mỗi nguồn có cách đánh giá riêng.

> [!IMPORTANT]
> Các nguồn placement hiện có là **dữ liệu tổng hợp** theo mô tả của tác giả Kaggle.
> Đây là project học tập/thực nghiệm, không đại diện cho sinh viên hay thị trường
> tuyển dụng thực tế. Điểm mô hình không phải cam kết cơ hội việc làm.

## 🚀 Bắt đầu nhanh

### 1. Chuẩn bị môi trường

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Chạy dashboard

```powershell
streamlit run app.py
```

Streamlit sẽ khởi chạy server cục bộ và mở dashboard trong trình duyệt. Nếu muốn
chạy trong terminal đang mở:

```powershell
python -m streamlit run app.py
```

### 3. Chạy kiểm thử

```powershell
python -m pytest -q
```

## 📚 Nguồn dữ liệu

Chọn nguồn trong thanh bên của ứng dụng:

| Nguồn | Dữ liệu dùng | Đánh giá |
|---|---|---|
| **Kaggle Student Placement** | Chỉ nhánh CSE/IT từ `train.csv` | Huấn luyện trên train; chấm trên `test.csv` riêng |
| **Kaggle Placement Prediction 2026** | CSE/IT từ CSV đã chuẩn hóa | Holdout phân tầng 80/20 trong app |

Kaggle ghi nhận giấy phép **CC0** cho các dataset dưới đây; tác giả mô tả các bản
ghi là dữ liệu tổng hợp. Chi tiết nguồn và giới hạn được ghi tại
[ghi chú Student Placement](docs/kaggle-student-placement.md) và
[ghi chú Placement Prediction 2026](docs/kaggle-placement-prediction-2026.md).

### Chuẩn bị lại dữ liệu 2026

CSV nguồn đã được đặt tại:

```text
data/raw/kaggle_placement_prediction_2026/
```

Tạo lại bản CSE/IT đã chuẩn hóa:

```powershell
python scripts/prepare_external_dataset.py
```

Để tải lại CSV nguồn từ Kaggle:

```powershell
python scripts/download_placement_dataset.py
```

Lệnh tải cần KaggleHub và quyền truy cập internet. Dataset train/test 50K được
đóng gói sẵn trong `data/raw/kaggle_student_placement/`; nếu cần tải lại:

```powershell
python scripts/install_dataset.py
```

## 🧠 Đầu vào mô hình

| Trường | Ý nghĩa |
|---|---|
| `Branch` | CSE hoặc IT |
| `CGPA` | Điểm học tập |
| `Internships` | Số kỳ thực tập |
| `Projects` | Số dự án |
| `Coding_Skills` | Điểm kỹ năng lập trình |
| `Certifications` | Số chứng chỉ |
| `Backlogs` | Số môn nợ |

Không dùng `Student_ID`, giới tính, bằng cấp hay điểm aptitude làm đặc trưng.
Pipeline dùng one-hot encoding cho nhánh và chuẩn hóa các biến số. Mỗi nguồn có
split đánh giá riêng; test set được giữ ngoài fit của mô hình.

## 🗂️ Cấu trúc dự án

```text
.
├── app.py                              # Điểm chạy Streamlit
├── src/
│   └── placement_data.py               # Nạp, lọc và xác thực train/test
├── scripts/
│   ├── download_placement_dataset.py   # Tải nguồn Kaggle 2026
│   ├── install_dataset.py              # Tải train/test Kaggle
│   └── prepare_external_dataset.py     # Chuẩn hóa dữ liệu CSE/IT
├── data/
│   ├── raw/                            # CSV nguyên bản
│   └── processed/                      # CSV app sử dụng
├── docs/                               # Ghi chú nguồn và giới hạn dữ liệu
├── tests/                              # Pytest
├── requirements.txt
└── README.md
```

## 🔗 Tham khảo

- [Kaggle — Student Placement Dataset](https://www.kaggle.com/datasets/sonalshinde123/student-placement-dataset)
- [Kaggle — Student Placement Prediction Dataset 2026](https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026)
- [Streamlit — chạy ứng dụng](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)
- [scikit-learn — model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

<div align="center">

**Học máy tốt bắt đầu từ dữ liệu minh bạch.**  
Built with Python · Streamlit · scikit-learn

</div>
