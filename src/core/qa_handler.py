
import sys
import os
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(root_dir)

from src.client.embedding.embedding_client import SBIEmbeddings
from src.client.rerank.client import SBIRerank
from src.configs.configs import VECTOR_SEARCH_SCORE_THRESHOLD
from src.utils.log_handler import debug_logger, embed_logger, rerank_logger
from src.client.database.mysql.mysql_client import MysqlClient
from src.client.database.milvus.milvus_client import MilvusClient
from src.client.database.elasticsearch.es_client import ESClient
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class QAHandler:
    def __init__(self, port):
        self.port = port
        self.milvus_cache = None
        self.embeddings: SBIEmbeddings = None
        self.rerank: SBIRerank = None
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        # self.milvus_kb: MilvusClient = None
        # self.retriever: ParentRetriever = None
        self.milvus_summary: MysqlClient = None
        self.es_client: ESClient = None
        self.session = self.create_retry_session(retries=3, backoff_factor=1)
        # self.doc_splitter = CharacterTextSplitter(
        #     chunk_size=LOCAL_EMBED_MAX_LENGTH / 2,
        #     chunk_overlap=0,
        #     length_function=len
        # )
    
    @staticmethod
    def create_retry_session(retries, backoff_factor):
        """
        创建一个带有重试机制的 requests Session
        
        参数：
        retries: int - 重试次数
        backoff_factor: float - 重试间隔因子
        """
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )
        # 创建适配器并设置重试策略
        adapter = HTTPAdapter(max_retries=retry)

        # 将适配器挂载到会话上，分别处理 http 和 https
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def init_cfg(self, args=None):
        self.embeddings = SBIEmbeddings()
        self.rerank = SBIRerank()
        self.milvus_summary = MysqlClient()
        self.milvus_kb = MilvusClient()
        self.es_client = ESClient()
        # self.retriever = ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)

    
    

    
