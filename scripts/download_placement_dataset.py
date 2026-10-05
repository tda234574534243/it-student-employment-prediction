from pathlib import Path
import shutil

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATASET = "sehaj1104/student-placement-prediction-dataset-2026"
RAW_DIRECTORY = (
    PROJECT_ROOT / "data" / "raw" / "kaggle_placement_prediction_2026"
)
RAW_FILE = RAW_DIRECTORY / "student_placement_prediction_dataset_2026.csv"


def download_dataset() -> Path:
    downloaded_directory = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    downloaded_file = (
        downloaded_directory / "student_placement_prediction_dataset_2026.csv"
    )
    if not downloaded_file.is_file():
        raise FileNotFoundError(
            f"Kaggle download did not contain the expected file: {downloaded_file}"
        )
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_file, RAW_FILE)
    return RAW_FILE


if __name__ == "__main__":
    print(f"Downloaded dataset to: {download_dataset()}")
