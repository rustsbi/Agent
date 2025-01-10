import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)

from sanic import Sanic
from sanic.response import json
from src.server.embedding_server.embedding_backend import EmbeddingBackend
from src.configs.configs import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_THREADS
from src.utils.general_utils import get_time_async
import argparse

# 接收外部参数mode
parser = argparse.ArgumentParser()
# mode必须是local或online
# 使用--use_gpu可以让Embedding模型加载到gpu中
parser.add_argument('--use_gpu', action="store_true", help='use gpu or not')
parser.add_argument('--workers', type=int, default=1, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()
print("args:", args)

app = Sanic("embedding_server")


@get_time_async
@app.route("/embedding", methods=["POST"])
async def embedding(request):
    data = request.json
    texts = data.get('texts')
    # print("local embedding texts number:", len(texts), flush=True)

    # onnx_backend: EmbeddingAsyncBackend = request.app.ctx.onnx_backend
    # onnx后端上下文在这里使用
    onnx_backend: EmbeddingBackend = request.app.ctx.onnx_backend
    # result_data = await onnx_backend.embed_documents_async(texts)
    result_data = onnx_backend.predict(texts)
    # print("local embedding result number:", len(result_data), flush=True)
    # print("local embedding result:", result_data, flush=True)

    return json(result_data)


@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    # app.ctx.onnx_backend = EmbeddingAsyncBackend(model_path=LOCAL_EMBED_MODEL_PATH,
    #                                              use_cpu=not args.use_gpu, num_threads=LOCAL_EMBED_THREADS)
    # onnx_backend 是在应用启动时被初始化并存储在上下文中的对象
    # 存储到应用上下文
    app.ctx.onnx_backend = EmbeddingBackend(use_cpu=not args.use_gpu)


if __name__ == "__main__":
    # workers参数指定了服务器启动的工作进程数量。
    app.run(host="0.0.0.0", port=9001, workers=args.workers)