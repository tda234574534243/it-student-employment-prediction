import kagglehub
import shutil
import os

# 1. Tải về thư mục tạm
path = kagglehub.dataset_download("sonalshinde123/student-placement-dataset")

# 2. Định nghĩa thư mục đích của bạn
localpath = r"D:\Machine-Course\Placement Dataset"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(localpath, exist_ok=True)

# 3. Sao chép tất cả các file và thư mục con sang thư mục đích
for filename in os.listdir(path):
    source_file = os.path.join(path, filename)
    dest_file = os.path.join(localpath, filename)
    
    if os.path.isdir(source_file):
        # Nếu là thư mục con, dùng copytree (dirs_exist_ok giúp ghi đè nếu đã tồn tại)
        shutil.copytree(source_file, dest_file, dirs_exist_ok=True)
    else:
        # Nếu là file đơn lẻ, dùng copy như cũ
        shutil.copy(source_file, dest_file)

print("Path to dataset files:", localpath)
