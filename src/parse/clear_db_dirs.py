import os
import shutil

# 指定要清理的目录
db_directory = 'db'

# 检查db_directory是否存在
if not os.path.exists(db_directory):
    print(f"The directory {db_directory} does not exist.")
else:
    # 遍历db_directory中的所有子目录
    for folder in os.listdir(db_directory):
        folder_path = os.path.join(db_directory, folder)
        
        # 确保是目录而不是文件
        if os.path.isdir(folder_path):
            # 检查index.faiss和index.pkl是否存在于子目录中
            if os.path.isfile(os.path.join(folder_path, 'index.faiss')) and os.path.isfile(os.path.join(folder_path, 'index.pkl')):
                # 删除子目录及其内容
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            else:
                print(f"Folder {folder_path} does not contain both index.faiss and index.pkl.")