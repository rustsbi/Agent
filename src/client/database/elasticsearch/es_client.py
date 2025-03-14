import asyncio
import os
import sys

from elasticsearch import Elasticsearch,exceptions
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
from src.utils.log_handler import debug_logger
from src.configs.configs import ES_USER, ES_PASSWORD, ES_URL, ES_INDEX_NAME
from langchain_elasticsearch import ElasticsearchStore



class ESClient:
    def __init__(self):
        try:
            es_client = Elasticsearch(
                hosts=[ES_URL],
                basic_auth=(ES_USER, ES_PASSWORD),
                verify_certs=False,
                ssl_show_warn=False,
                retry_on_timeout=True,
                max_retries=3,
                timeout=30
            )

            # 初始化 ElasticsearchStore
            self.es_store = ElasticsearchStore(
                es_connection=es_client,
                index_name=ES_INDEX_NAME,
                strategy=ElasticsearchStore.BM25RetrievalStrategy()
            )

            debug_logger.info(f"Init ElasticSearchStore with index_name: {ES_INDEX_NAME}")
        except exceptions.ConnectionError as e:
            debug_logger.error(f"ES connection error: {e}")
            raise
        except exceptions.AuthenticationException as e:
            debug_logger.error(f"ES authentication failed: {e}")
            raise
        except Exception as e:
            debug_logger.error(f"Unexpected error initializing ES client: {e}")
            raise

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

# async def main():
#     es_client = ESClient()
#     filter = [{"terms": {"metadata.kb_id.keyword": ["KBbf9488a498cf4407a6abdf477208c3ed"]}}]
#     es_sub_docs = await es_client.es_store.asimilarity_search("路上只我一个人", top_k=10, filter=filter)
#     print(es_sub_docs)

# asyncio.run(main())