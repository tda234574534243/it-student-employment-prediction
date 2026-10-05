from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_external_dataset import (
    OUTPUT_COLUMNS,
    prepare_external_dataset,
)


def test_prepare_external_dataset_filters_and_maps_cse_it(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "normalized" / "it-placement.csv"
    pd.DataFrame(
        [
            {
                "branch": "CSE",
                "placement_status": "Placed",
                "cgpa": 8.1,
                "internships_count": 2,
                "projects_count": 3,
                "coding_skill_score": 85,
                "certifications_count": 1,
                "backlogs": 0,
                "student_id": "hidden-id",
                "gender": "hidden-demographic",
                "salary_package_lpa": 12.5,
                "aptitude_score": 78,
            },
            {
                "branch": "IT",
                "placement_status": "Not Placed",
                "cgpa": 6.2,
                "internships_count": 0,
                "projects_count": 1,
                "coding_skill_score": 55,
                "certifications_count": 0,
                "backlogs": 2,
                "student_id": "another-id",
                "gender": "hidden-demographic",
                "salary_package_lpa": 0,
                "aptitude_score": 60,
            },
            {
                "branch": "CSE",
                "placement_status": "Placed",
                "cgpa": 8.1,
                "internships_count": 2,
                "projects_count": 3,
                "coding_skill_score": 85,
                "certifications_count": 1,
                "backlogs": 0,
                "student_id": "same-features-different-record",
                "gender": "hidden-demographic",
                "salary_package_lpa": 12.5,
                "aptitude_score": 79,
            },
            {
                "branch": "Mechanical",
                "placement_status": "Placed",
                "cgpa": 8.0,
                "internships_count": 1,
                "projects_count": 2,
                "coding_skill_score": 70,
                "certifications_count": 1,
                "backlogs": 0,
                "student_id": "excluded-id",
                "gender": "hidden-demographic",
                "salary_package_lpa": 8,
                "aptitude_score": 70,
            },
        ]
    ).to_csv(source, index=False)

    prepared = prepare_external_dataset(source, output)

    assert prepared.columns.tolist() == OUTPUT_COLUMNS
    assert prepared["Branch"].tolist() == ["CSE", "IT", "CSE"]
    assert prepared["Placement_Status"].tolist() == [
        "Placed",
        "Not Placed",
        "Placed",
    ]
    assert prepared.iloc[0]["Coding_Skills"] == 85
    assert "Aptitude_Test_Score" not in prepared.columns
    assert prepared.iloc[0].equals(prepared.iloc[2])
    assert output.is_file()
    saved = pd.read_csv(output, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(saved, prepared, check_dtype=False)


def test_prepare_external_dataset_reports_missing_required_columns(tmp_path):
    source = tmp_path / "invalid.csv"
    source.write_text("branch,placement_status\nIT,Placed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required Kaggle dataset columns"):
        prepare_external_dataset(source, tmp_path / "output.csv")


def test_project_external_dataset_is_cse_it_only_and_keeps_both_labels():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "data"
        / "processed"
        / "it_placement_prediction_2026.csv"
    )
    frame = pd.read_csv(source, encoding="utf-8-sig")

    assert len(frame) == 54_817
    assert frame.columns.tolist() == OUTPUT_COLUMNS
    assert set(frame["Branch"]) == {"CSE", "IT"}
    assert set(frame["Placement_Status"]) == {"Placed", "Not Placed"}
    assert not any("aptitude" in column.lower() for column in frame.columns)
