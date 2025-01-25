import json
import traceback
from pymilvus import connections, FieldSchema, CollectionSchema, DataType,\
      Collection, utility, Partition
from concurrent.futures import ThreadPoolExecutor
from src.utils.log_handler import debug_logger
from src.utils.general_utils import get_time, cur_func_name
from src.configs.configs import MILVUS_HOST_LOCAL, MILVUS_PORT, VECTOR_SEARCH_TOP_K
from langchain.docstore.document import Document
from typing import List


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
        self.sess = collection

    @property
    def fields(self):
        fields = [
            FieldSchema(name='user_id', dtype=DataType.VARCHAR, max_length=64, is_primary=True),
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
        return ['user_id', 'kb_id', 'file_id', 'headers', 'doc_id', 'content', 'embedding']
    



