

import os
import sys
from typing import List
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
import time
from src.utils.log_handler import debug_logger
from src.client.database.milvus.milvus_client import MilvusClient
from src.client.database.elasticsearch.es_client import ESClient

class Retriever:
    async def get_retrieved_documents(self, query: str, vector_store: MilvusClient, es_store: ESClient, partition_keys: List[str], time_record: dict,
                                    hybrid_search: bool, top_k: int, expr: str = None):
        milvus_start_time = time.perf_counter()
        # TODO 把milvus搜索转为Document类型 
        query_docs = vector_store.search_docs(query, expr, top_k, partition_keys)
        for doc in query_docs:
            doc.metadata['retrieval_source'] = 'milvus'
        milvus_end_time = time.perf_counter()
        time_record['retriever_search_by_milvus'] = round(milvus_end_time - milvus_start_time, 2)

        if not hybrid_search:
            return query_docs
        try:
            filter = [{"terms": {"metadata.kb_id.keyword": partition_keys}}]
            es_sub_docs = await es_store.asimilarity_search(query, k=top_k, filter=filter)
            print(es_sub_docs)
            for doc in es_sub_docs:
                doc.metadata['retrieval_source'] = 'es'
            time_record['retriever_search_by_es'] = round(time.perf_counter() - milvus_end_time, 2)
            debug_logger.info(f"Got {len(query_docs)} documents from vectorstore and {len(es_sub_docs)} documents from es, total {len(query_docs) + len(es_sub_docs)} merged documents.")
            query_docs.extend(es_sub_docs)
        except Exception as e:
            debug_logger.error(f"Error in get_retrieved_documents on es_search: {e}")
        return query_docs