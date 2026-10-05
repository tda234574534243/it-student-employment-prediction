from pathlib import Path

import pandas as pd


PLACEMENT_FEATURES = [
    "Branch",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Certifications",
    "Backlogs",
]
TARGET = "Placement_Status"
SPLIT_FEATURES = {
    "Branch": "Branch",
    "CGPA": "CGPA",
    "Internships": "Internships",
    "Projects": "Projects",
    "Coding_Skills": "Coding_Skills",
    "Certifications": "Certifications",
    "Backlogs": "Backlogs",
}
REQUIRED_COLUMNS = {
    "Student_ID",
    *SPLIT_FEATURES,
    TARGET,
}


def _normalize_split_file(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required placement split columns in {path}: "
            f"{', '.join(missing_columns)}"
        )

    ids = frame["Student_ID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Student_ID values must be present and unique in {path}.")

    normalized = frame[list(SPLIT_FEATURES) + [TARGET]].copy()
    normalized["Branch"] = normalized["Branch"].astype("string").str.strip()
    normalized[TARGET] = normalized[TARGET].astype("string").str.strip()
    for column in PLACEMENT_FEATURES[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.loc[
        normalized["Branch"].isin({"CSE", "IT"})
        & normalized[TARGET].isin({"Placed", "Not Placed"})
    ].dropna(subset=PLACEMENT_FEATURES + [TARGET])
    ids = ids.loc[normalized.index]
    normalized = normalized.reset_index(drop=True)
    ids = ids.reset_index(drop=True)
    if normalized.empty or normalized[TARGET].nunique() != 2:
        raise ValueError(
            f"The CSE/IT split must contain both placement outcomes: {path}"
        )
    return normalized, ids


def load_kaggle_train_test(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle-provided train/test files without using test rows for fit."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    train, train_ids = _normalize_split_file(train_path)
    test, test_ids = _normalize_split_file(test_path)
    overlapping_ids = set(train_ids).intersection(test_ids)
    if overlapping_ids:
        raise ValueError(
            "Kaggle train/test files share Student_ID values; refusing evaluation "
            "because the split would not be independent."
        )
    return train, test
