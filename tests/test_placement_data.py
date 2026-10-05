from pathlib import Path

import pandas as pd
import pytest

from src.placement_data import PLACEMENT_FEATURES, load_kaggle_train_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "kaggle_student_placement"


def test_kaggle_train_test_are_cse_it_only_and_preserve_separate_test_set():
    train, test = load_kaggle_train_test(
        DATA_DIRECTORY / "train.csv",
        DATA_DIRECTORY / "test.csv",
    )

    assert len(train) == 17_937
    assert len(test) == 2_039
    assert train.columns.tolist() == [*PLACEMENT_FEATURES, "Placement_Status"]
    assert test.columns.tolist() == [*PLACEMENT_FEATURES, "Placement_Status"]
    assert set(train["Branch"]) == {"CSE", "IT"}
    assert set(test["Branch"]) == {"CSE", "IT"}
    assert set(train["Placement_Status"]) == {"Placed", "Not Placed"}
    assert set(test["Placement_Status"]) == {"Placed", "Not Placed"}


def test_kaggle_train_test_reject_overlapping_student_ids(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Student_ID": student_id,
                "Branch": branch,
                "CGPA": cgpa,
                "Internships": internships,
                "Projects": 2,
                "Coding_Skills": 7,
                "Certifications": 1,
                "Backlogs": 0,
                "Placement_Status": label,
                "Aptitude_Test_Score": 80,
            }
            for student_id, branch, cgpa, internships, label in [
                (1, "CSE", 8.0, 1, "Placed"),
                (2, "IT", 6.0, 0, "Not Placed"),
            ]
        ]
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    frame.to_csv(train_path, index=False)
    frame.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="share Student_ID values"):
        load_kaggle_train_test(train_path, test_path)


def test_kaggle_train_test_reports_missing_required_columns(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text("Branch,Placement_Status\nCSE,Placed\n", encoding="utf-8")
    test_path.write_text("Branch,Placement_Status\nIT,Not Placed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required placement split columns"):
        load_kaggle_train_test(train_path, test_path)
