import os
import sys
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
from src.utils.log_handler import debug_logger, insert_logger
from src.configs.configs import ES_USER, ES_PASSWORD, ES_URL, ES_INDEX_NAME
from langchain_elasticsearch import ElasticsearchStore


class ESClient:
    def __init__(self):
        self.es_store = ElasticsearchStore(
            es_url=ES_URL,
            index_name=ES_INDEX_NAME,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            # 检索策略使用BM25
            strategy=ElasticsearchStore.BM25RetrievalStrategy()
        )
        debug_logger.info(f"Init ElasticSearchStore with index_name: {ES_INDEX_NAME}")

    def delete(self, docs_ids):
        try:
            res = self.es_store.delete(docs_ids, timeout=60)
            debug_logger.info(f"Delete ES document with number: {len(docs_ids)}, {docs_ids[0]}, res: {res}")
        except Exception as e:
            debug_logger.error(f"Delete ES document failed with error: {e}")

    def delete_files(self, file_ids, file_chunks):
        docs_ids = []
        for file_id, file_chunk in zip(file_ids, file_chunks):
            # doc_id 是file_id + '_' + i，其中i是range(file_chunk)
            docs_ids.extend([file_id + '_' + str(i) for i in range(file_chunk)])
        if docs_ids:
            self.delete(docs_ids)
