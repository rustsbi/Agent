from pymilvus import connections, db
import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
conn = connections.connect(host="127.0.0.1", port=19530)
database = db.create_database("my_database")