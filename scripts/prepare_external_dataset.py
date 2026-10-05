from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle_placement_prediction_2026"
    / "student_placement_prediction_dataset_2026.csv"
)
OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "it_placement_prediction_2026.csv"
)
BRANCHES = {"CSE", "IT"}
NUMERIC_FEATURE_MAPPING = {
    "cgpa": "CGPA",
    "internships_count": "Internships",
    "projects_count": "Projects",
    "coding_skill_score": "Coding_Skills",
    "certifications_count": "Certifications",
    "backlogs": "Backlogs",
}
FEATURE_COLUMNS = [
    "Branch",
    *NUMERIC_FEATURE_MAPPING.values(),
]
TARGET_COLUMN = "Placement_Status"
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def prepare_external_dataset(
    source_path: Path = RAW_DATASET,
    output_path: Path = OUTPUT_DATASET,
) -> pd.DataFrame:
    """Normalize the public synthetic dataset and keep CSE/IT records only."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    frame = pd.read_csv(source_path)
    frame.columns = frame.columns.str.strip()

    required_columns = {
        "branch",
        "placement_status",
        *NUMERIC_FEATURE_MAPPING,
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required Kaggle dataset columns: {', '.join(missing_columns)}"
        )

    normalized = frame[
        ["branch", "placement_status", *NUMERIC_FEATURE_MAPPING]
    ].copy()
    normalized["Branch"] = normalized.pop("branch").astype("string").str.strip()
    normalized[TARGET_COLUMN] = (
        normalized.pop("placement_status").astype("string").str.strip()
    )
    normalized = normalized.rename(columns=NUMERIC_FEATURE_MAPPING)
    normalized = normalized.loc[
        normalized["Branch"].isin(BRANCHES)
        & normalized[TARGET_COLUMN].isin({"Placed", "Not Placed"})
    ].copy()

    for column in NUMERIC_FEATURE_MAPPING.values():
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = (
        normalized.dropna(subset=OUTPUT_COLUMNS)
        .loc[:, OUTPUT_COLUMNS]
        .reset_index(drop=True)
    )
    if normalized.empty or normalized[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "The filtered CSE/IT dataset must contain both placement outcomes."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
    return normalized


if __name__ == "__main__":
    prepared = prepare_external_dataset()
    counts = prepared[TARGET_COLUMN].value_counts().to_dict()
    print(
        f"Created {OUTPUT_DATASET} with {len(prepared)} CSE/IT records; "
        f"outcomes: {counts}"
    )
