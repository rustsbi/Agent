import sys
import os
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本的父目录的路径，即`qanything_server`目录
current_dir = os.path.dirname(current_script_path)

# 获取`qanything_server`目录的父目录，即`qanything_kernel`
root_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(root_dir)
root_dir = os.path.dirname(root_dir)
# 将项目根目录添加到sys.path
sys.path.append(root_dir)
from sanic_api_handler import *
from src.core.local_doc_qa import LocalDocQA
from src.utils.log_handler import debug_logger, qa_logger
from sanic.worker.manager import WorkerManager
from sanic import Sanic
from sanic_ext import Extend
import time
import argparse
import webbrowser

WorkerManager.THRESHOLD = 6000

# 接收外部参数mode
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0', help='host')
parser.add_argument('--port', type=int, default=8777, help='port')
parser.add_argument('--workers', type=int, default=1, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()

start_time = time.time()
app = Sanic("RustSBI")
app.config.CORS_ORIGINS = "*"
Extend(app)
# 设置请求体最大为 128MB
app.config.REQUEST_MAX_SIZE = 128 * 1024 * 1024

app.static('/rustsbi/', './dist/', name='rustsbi', index="index.html")


@app.before_server_start
async def init_local_doc_qa(app, loop):
    start = time.time()
    local_doc_qa = LocalDocQA(args.port)
    local_doc_qa.init_cfg(args)
    end = time.time()
    print(f'init local_doc_qa cost {end - start}s', flush=True)
    app.ctx.local_doc_qa = local_doc_qa
    
@app.after_server_start
async def notify_server_started(app, loop):
    print(f"Server Start Cost {time.time() - start_time} seconds", flush=True)

@app.after_server_start
async def start_server_and_open_browser(app, loop):
    try:
        print(f"Opening browser at http://{args.host}:{args.port}/rustsbi/")
        webbrowser.open(f"http://{args.host}:{args.port}/rustsbi/")
    except Exception as e:
        # 记录或处理任何异常
        print(f"Failed to open browser: {e}")

# app.add_route(lambda req: response.redirect('/api/docs'), '/')
# tags=["新建知识库"]
app.add_route(document, "/api/docs", methods=['GET'])
app.add_route(health_check, "/api/health_check", methods=['GET'])  # tags=["健康检查"]
app.add_route(new_knowledge_base, "/api/local_doc_qa/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
app.add_route(upload_files, "/api/local_doc_qa/upload_files", methods=['POST'])  # tags=["上传文件"]
# app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])  # tags=["问答接口"] 
# app.add_route(list_kbs, "/api/local_doc_qa/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
# app.add_route(list_docs, "/api/local_doc_qa/list_files", methods=['POST'])  # tags=["文件列表"]
# app.add_route(get_total_status, "/api/local_doc_qa/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态数据库"]
# app.add_route(clean_files_by_status, "/api/local_doc_qa/clean_files_by_status", methods=['POST'])  # tags=["清理数据库"]
# app.add_route(delete_docs, "/api/local_doc_qa/delete_files", methods=['POST'])  # tags=["删除文件"] 
# app.add_route(delete_knowledge_base, "/api/local_doc_qa/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"] 
# app.add_route(rename_knowledge_base, "/api/local_doc_qa/rename_knowledge_base", methods=['POST'])  # tags=["重命名知识库"]
# app.add_route(get_doc_completed, "/api/local_doc_qa/get_doc_completed", methods=['POST'])  # tags=["获取文档完整解析内容"]
# app.add_route(get_user_id, "/api/local_doc_qa/get_user_id", methods=['POST'])  # tags=["获取用户ID"]
# app.add_route(get_doc, "/api/local_doc_qa/get_doc", methods=['POST'])  # tags=["获取doc详细内容"]
# app.add_route(get_rerank_results, "/api/local_doc_qa/get_rerank_results", methods=['POST'])  # tags=["获取rerank结果"]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=args.port, workers=args.workers, access_log=False)