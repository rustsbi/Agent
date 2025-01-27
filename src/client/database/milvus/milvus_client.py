import os
import sys
import json
import traceback
from pymilvus import connections, FieldSchema, CollectionSchema, DataType,\
      Collection, utility, Partition
from concurrent.futures import ThreadPoolExecutor
from langchain.docstore.document import Document
from typing import List

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(current_script_path) # milvus
root_dir = os.path.dirname(root_dir) # database
root_dir = os.path.dirname(root_dir) # client
root_dir = os.path.dirname(root_dir) # src
root_dir = os.path.dirname(root_dir)
# 将项目根目录添加到sys.path
sys.path.append(root_dir)

from src.utils.log_handler import debug_logger
from src.utils.general_utils import get_time, cur_func_name
from src.configs.configs import MILVUS_HOST_LOCAL, MILVUS_PORT, VECTOR_SEARCH_TOP_K


class MilvusFailed(Exception):
    """异常基类"""
    pass


class MilvusClient:
    def __init__(self):
        self.host = MILVUS_HOST_LOCAL
        self.port = MILVUS_PORT
        self.sess: Collection = None
        self.partitions: List[Partition] = []
        # 可以先不用
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.top_k = VECTOR_SEARCH_TOP_K
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
        self.create_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        # self.create_params = {"metric_type": "L2", "index_type": "GPU_IVF_FLAT", "params": {"nlist": 1024}}  # GPU版本
        try:
            connections.connect(host=self.host, 
                                port=self.port,
                                timeout=3, 
                                timeout_retry=3, 
                                wait_time=1)  # timeout=3 [cannot set]
        except Exception as e:
            debug_logger.error(f'[{cur_func_name()}] [MilvusClient] traceback = {traceback.format_exc()}')

    @get_time 
    def load_collection_(self, user_id):
        if not utility.has_collection(user_id):
            schema = CollectionSchema(self.fields)
            debug_logger.info(f'create collection {user_id}')
            collection = Collection(user_id, schema)
            # 创建索引
            collection.create_index(field_name="embedding", index_params=self.create_params)
        else:
            collection = Collection(user_id)
        collection.load()
        self.sess = collection
        
    def store_doc(self, doc: Document, embedding: List[float]):
        """
        将文档块存储到 Milvus 中。

        Args:
            doc (Document): Langchain 的 Document 对象，包含文档内容及其元数据。
            embedding (List[float]): 文档的向量表示，长度为 768。
        """
        try:
            # 确保 Milvus 集合已加载
            if not self.sess:
                raise MilvusFailed("Milvus collection is not loaded. Call load_collection_() first.")

            # 提取文档元数据
            metadata = doc.metadata
            user_id = metadata.get('user_id')
            kb_id = metadata.get('kb_id')
            file_id = metadata.get('file_id')
            headers = json.dumps(metadata.get('headers', {}))  # 将 headers 转换为 JSON 字符串
            doc_id = metadata.get('doc_id')
            content = doc.page_content

            # 检查字段是否完整
            if not all([user_id, kb_id, file_id, doc_id, content, embedding]):
                raise MilvusFailed("Missing required fields in document metadata or embedding.")

            # 构造插入数据（不需要提供主键值）
            data = [
                [user_id],  # user_id
                [kb_id],    # kb_id
                [file_id],  # file_id
                [headers],  # headers
                [doc_id],   # doc_id
                [content],  # content
                [embedding]  # embedding
            ]

            # 插入数据到 Milvus
            self.sess.insert(data)
            print(f"Document {doc_id} stored successfully in collection {user_id}.")

        except Exception as e:
            print(f'[{cur_func_name()}] [store_doc] Failed to store document: {traceback.format_exc()}')
            raise MilvusFailed(f"Failed to store document: {str(e)}")

    @property
    def fields(self):
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),  # 自增主键
            FieldSchema(name='user_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='kb_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='file_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='headers', dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name='doc_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        return fields

    @property
    def output_fields(self):
        return ['id', 'user_id', 'kb_id', 'file_id', 'headers', 'doc_id', 'content', 'embedding']
    


def main():
    # 初始化 MilvusClient
    client = MilvusClient()

    # 指定用户 ID（即集合名称）
    user_id = "abc1234__5678"  # 测试用户 ID

    # 加载集合
    try:
        client.load_collection_(user_id)
        print(f"Collection {user_id} loaded successfully.")
    except Exception as e:
        print(f"Failed to load collection {user_id}: {traceback.format_exc()}")
        return

    # 检索所有文档
    try:
        # 构造查询表达式（检索所有文档）
        query_expr = ""  # 不设置过滤条件，检索所有文档

        # 执行查询
        results = client.sess.query(
            expr=query_expr,
            output_fields=client.output_fields,  # 指定返回的字段
            limit=1000
        )

        # 打印检索结果
        if not results:
            print(f"No documents found in collection {user_id}.")
            return

        print(f"Found {len(results)} documents in collection {user_id}:")
        for i, result in enumerate(results):
            print(f"\nDocument {i + 1}:")
            print(f"  user_id: {result['user_id']}")
            print(f"  kb_id: {result['kb_id']}")
            print(f"  file_id: {result['file_id']}")
            print(f"  headers: {json.loads(result['headers'])}")  # 将 headers 从 JSON 字符串解析为字典
            print(f"  doc_id: {result['doc_id']}")
            print(f"  content: {result['content']}")
            print(f"  embedding: {result['embedding'][:5]}... (truncated)")  # 只打印前 5 维向量

    except Exception as e:
        print(f"Failed to retrieve documents from collection {user_id}: {traceback.format_exc()}")


if __name__ == "__main__":
    print("start milvus testing")
    main()
