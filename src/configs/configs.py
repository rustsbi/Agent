import os

current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

EMBED_MODEL_PATH="maidalun1020/bce-embedding-base_v1"
LOCAL_EMBED_SERVICE_URL = "localhost:9001"
LOCAL_EMBED_PATH=os.path.join(root_path, 'src/server/embedding_server', 'bce_model')

LOCAL_EMBED_MODEL_PATH=os.path.join(LOCAL_EMBED_PATH, "bce_model.onnx")

LOCAL_EMBED_BATCH=1

LOCAL_EMBED_THREADS=1

LOCAL_RERANK_BATCH = 1
LOCAL_RERANK_MAX_LENGTH=512