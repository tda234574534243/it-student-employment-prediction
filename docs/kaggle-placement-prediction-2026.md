# Kaggle: Student Placement Prediction Dataset 2026

- Dataset: [Kaggle](https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026)
- License shown by Kaggle: CC0 1.0 Universal
- The author describes the 100,000 records as synthetic simulations.

The original CSV is stored at
`data/raw/kaggle_placement_prediction_2026/student_placement_prediction_dataset_2026.csv`.
Run `python scripts/prepare_external_dataset.py` to create
`data/processed/it_placement_prediction_2026.csv`.

Preparation keeps CSE and IT only. The app schema excludes IDs, gender, salary,
and aptitude scores. This source is synthetically generated and must not be
treated as evidence of real-world employment or model performance.
