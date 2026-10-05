# Kaggle: Student Placement Dataset

- Dataset: [Kaggle](https://www.kaggle.com/datasets/sonalshinde123/student-placement-dataset)
- License shown by Kaggle: CC0
- Kaggle lists `train.csv` (45,000 rows) and `test.csv` (5,000 rows).
- The author states that all records are synthetically generated.

The files are stored under `data/raw/kaggle_student_placement/`. The app keeps
only CSE/IT rows, fits models using filtered `train.csv`, and evaluates against
filtered `test.csv`. Student ID overlap is checked. IDs, age, gender, degree,
aptitude, and unrelated fields are not model features.

Synthetic placement labels are not real student/hiring records and cannot
establish real-world employment prospects or model performance.
