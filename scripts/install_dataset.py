import kagglehub
import shutil
import os
from pathlib import Path

DATASET_HANDLE = "sonalshinde123/student-placement-dataset"
LOCAL_PATH = str(
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "kaggle_student_placement"
)


def copy_dataset_contents(source_path, destination_path):
    os.makedirs(destination_path, exist_ok=True)

    for filename in os.listdir(source_path):
        source_file = os.path.join(source_path, filename)
        dest_file = os.path.join(destination_path, filename)

        if os.path.isdir(source_file):
            shutil.copytree(source_file, dest_file, dirs_exist_ok=True)
        else:
            shutil.copy(source_file, dest_file)


def install_dataset():
    download_path = kagglehub.dataset_download(DATASET_HANDLE)
    nested_dataset_path = os.path.join(download_path, "Student Placement Dataset")
    source_path = (
        nested_dataset_path
        if os.path.isdir(nested_dataset_path)
        else download_path
    )
    copy_dataset_contents(source_path, LOCAL_PATH)
    print("Path to dataset files:", LOCAL_PATH)


if __name__ == "__main__":
    install_dataset()
