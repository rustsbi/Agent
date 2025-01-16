import os

current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
print("UPLOAD_ROOT_PATH:", UPLOAD_ROOT_PATH)

LOCAL_EMBED_SERVICE_URL = "localhost:9001"
EMBED_MODEL_PATH="maidalun1020/bce-embedding-base_v1"
LOCAL_EMBED_PATH=os.path.join(root_path, 'src/server/embedding_server', 'bce_model')
LOCAL_EMBED_MODEL_PATH=os.path.join(LOCAL_EMBED_PATH, "model.onnx")
LOCAL_EMBED_BATCH=1
LOCAL_EMBED_THREADS=1

LOCAL_RERANK_SERVICE_URL = "localhost:8001"
RERANK_MODEL_PATH="maidalun1020/bce-reranker-base_v1"
LOCAL_RERANK_BATCH = 1
LOCAL_RERANK_THREADS = 1
LOCAL_RERANK_MAX_LENGTH=512
LOCAL_RERANK_PATH=os.path.join(root_path, 'src/server/rerank_server', 'bce_model')
LOCAL_RERANK_MODEL_PATH=os.path.join(LOCAL_RERANK_PATH, "model.onnx")

MYSQL_HOST_LOCAL="k8s.richeyjang.com"
MYSQL_PORT_LOCAL="30303"
MYSQL_USER_LOCAL="root"
MYSQL_PASSWORD_LOCAL="123456"
MYSQL_DATABASE_LOCAL="rustsbi_rag"

KB_SUFFIX='_250114'

ES_USER=None
ES_PASSWORD=None
ES_URL="http://localhost:9200"
ES_INDEX_NAME='rustsbi_es_index' + KB_SUFFIX

MILVUS_HOST_LOCAL = "k8s.richeyjang.com"
MILVUS_PORT = 30300
MILVUS_COLLECTION_NAME = 'rustsbi_collection' + KB_SUFFIX

VECTOR_SEARCH_SCORE_THRESHOLD = 0.3
VECTOR_SEARCH_TOP_K = 10
CHUNK_SIZE = 400
DEFAULT_PARENT_CHUNK_SIZE = 800
MAX_CHARS =  1000000
