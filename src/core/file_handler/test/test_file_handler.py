import os
import sys

# change rootpath to current path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
current_path = os.path.abspath(__file__)

# 获取当前脚本的父目录的父目录
parent_directory = os.path.dirname(os.path.dirname(current_path))

sys.path.append(parent_directory)

from file_handler import FileHandler

# readme.md std-Rust.pdf RAG.png 开题报告规范化要求

file_path = "std-Rust"

# https://doc.rust-lang.org/stable/std/
url_path = "https://doc.rust-lang.org/stable/std/"

# Check if the file exists

if os.path.exists(file_path):
    print(f"文件 {file_path} 存在。")
else:
    print(f"文件 {file_path} 不存在。")

# print(FileHandler.load_docx(file_path))

print(FileHandler.load_url(url_path))
