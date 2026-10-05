import kagglehub
import shutil
import os

DATASET_HANDLE = "sonalshinde123/student-placement-dataset"
LOCAL_PATH = r"D:\Machine-Course\Placement Dataset"


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
    copy_dataset_contents(download_path, LOCAL_PATH)
    print("Path to dataset files:", LOCAL_PATH)


if __name__ == "__main__":
    install_dataset()
