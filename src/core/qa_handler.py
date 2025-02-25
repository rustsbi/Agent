
import re
import sys
import os
import time
import traceback
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(root_dir)

from src.client.embedding.embedding_client import SBIEmbeddings
from src.client.rerank.client import SBIRerank
from src.configs.configs import VECTOR_SEARCH_SCORE_THRESHOLD
from src.utils.log_handler import debug_logger
from src.client.database.mysql.mysql_client import MysqlClient
from src.client.database.milvus.milvus_client import MilvusClient
from src.client.database.elasticsearch.es_client import ESClient
from src.core.retriever.retriever import Retriever
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from langchain.schema.messages import AIMessage, HumanMessage
from src.client.llm.llm_client import OpenAILLM
from src.core.chains.condense_q_chain import RewriteQuestionChain

from src.utils.general_utils import deduplicate_documents, num_tokens_rerank

class QAHandler:
    def __init__(self, port):
        self.port = port
        self.milvus_cache = None
        self.embeddings: SBIEmbeddings = None
        self.rerank: SBIRerank = None
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.milvus_kb: MilvusClient = None
        self.retriever: Retriever = None
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
        self.mysql_client = MysqlClient()
        self.milvus_client = MilvusClient()
        self.es_client = ESClient()
        self.retriever = Retriever()
        
    async def get_source_documents(self, query, retriever: Retriever, kb_ids, time_record, hybrid_search, top_k):
        source_documents = []
        start_time = time.perf_counter()
        query_docs = await retriever.get_retrieved_documents(query, self.milvus_client, self.es_client, partition_keys=kb_ids, time_record=time_record,
                                                             hybrid_search=hybrid_search, top_k=top_k)
        print(query_docs)
        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")
        # debug_logger.info(f"query_docs num: {len(query_docs)}, query_docs: {query_docs}")
        for idx, doc in enumerate(query_docs):
            if retriever.mysql_client.is_deleted_file(doc.metadata['file_id']):
                debug_logger.warning(f"file_id: {doc.metadata['file_id']} is deleted")
                continue
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 1 - (idx / len(query_docs))  # TODO 这个score怎么获取呢
            source_documents.append(doc)
        debug_logger.info(f"embed scores: {[doc.metadata['score'] for doc in source_documents]}")
        # if cosine_thresh:
        #     source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, top_k, web_chunk_size,
                                         chat_history=None, streaming: bool = True, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)
        if chat_history is None:
            chat_history = []
        retrieval_query = query
        condense_question = query
        if chat_history:
            formatted_chat_history = []
            for msg in chat_history:
                # 对话历史格式化
                formatted_chat_history += [
                    HumanMessage(content=msg[0]),
                    AIMessage(content=msg[1]),
                ]
            debug_logger.info(f"formatted_chat_history: {formatted_chat_history}")

            rewrite_q_chain = RewriteQuestionChain(model_name=model, openai_api_base=api_base, openai_api_key=api_key)
            # 将对话历史和查询输入到对话模版中
            full_prompt = rewrite_q_chain.condense_q_prompt.format(
                chat_history=formatted_chat_history,
                question=query
            )
            # 如果总token数超过限制(4096-256)，从前往后删除历史消息
            while custom_llm.num_tokens_from_messages([full_prompt]) >= 4096 - 256:
                formatted_chat_history = formatted_chat_history[2:]
                full_prompt = rewrite_q_chain.condense_q_prompt.format(
                    chat_history=formatted_chat_history,
                    question=query
                )
            debug_logger.info(
                f"Subtract formatted_chat_history: {len(chat_history) * 2} -> {len(formatted_chat_history)}")
            try:
                t1 = time.perf_counter()
                # 根据模版对查询进行简化
                condense_question = await rewrite_q_chain.condense_q_chain.ainvoke(
                    {
                        "chat_history": formatted_chat_history,
                        "question": query,
                    },
                )
                t2 = time.perf_counter()
                # 时间保留两位小数
                time_record['condense_q_chain'] = round(t2 - t1, 2)
                time_record['rewrite_completion_tokens'] = custom_llm.num_tokens_from_messages([condense_question])
                debug_logger.info(f"condense_q_chain time: {time_record['condense_q_chain']}s")
            except Exception as e:
                debug_logger.error(f"condense_q_chain error: {e}")
                condense_question = query
            debug_logger.info(f"condense_question: {condense_question}")
            time_record['rewrite_prompt_tokens'] = custom_llm.num_tokens_from_messages([full_prompt, condense_question])
        # 如果有kb_ids那么需要对相关知识库进行检索
        if kb_ids:
            source_documents = await self.get_source_documents(retrieval_query, retriever, kb_ids, time_record,
                                                            hybrid_search, top_k)
        else:
            source_documents = []
        # 这里处理网络搜索
        # if need_web_search:
        #     t1 = time.perf_counter()
        #     web_search_results = self.web_page_search(query, top_k=3)
        #     web_splitter = RecursiveCharacterTextSplitter(
        #         separators=SEPARATORS,
        #         chunk_size=web_chunk_size,
        #         chunk_overlap=int(web_chunk_size / 4),
        #         length_function=num_tokens_embed,
        #     )
        #     web_search_results = web_splitter.split_documents(web_search_results)

        #     current_doc_id = 0
        #     current_file_id = web_search_results[0].metadata['file_id']
        #     for doc in web_search_results:
        #         if doc.metadata['file_id'] == current_file_id:
        #             doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
        #             current_doc_id += 1
        #         else:
        #             current_file_id = doc.metadata['file_id']
        #             current_doc_id = 0
        #             doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
        #             current_doc_id += 1
        #         doc_json = doc.to_json()
        #         if doc_json['kwargs'].get('metadata') is None:
        #             doc_json['kwargs']['metadata'] = doc.metadata
        #         self.milvus_summary.add_document(doc_id=doc.metadata['doc_id'], json_data=doc_json)

        #     t2 = time.perf_counter()
        #     time_record['web_search'] = round(t2 - t1, 2)
        
        # 对检索的内容进行rerank
        source_documents = deduplicate_documents(source_documents)
        if rerank and len(source_documents) > 1 and num_tokens_rerank(query) <= 300:
            try:
                t1 = time.perf_counter()
                debug_logger.info(f"use rerank, rerank docs num: {len(source_documents)}")
                source_documents = await self.rerank.arerank_documents(condense_question, source_documents)
                t2 = time.perf_counter()
                time_record['rerank'] = round(t2 - t1, 2)
                # 过滤掉低分的文档
                debug_logger.info(f"rerank step1 num: {len(source_documents)}")
                debug_logger.info(f"rerank step1 scores: {[doc.metadata['score'] for doc in source_documents]}")
                if len(source_documents) > 1:
                    if filtered_documents := [doc for doc in source_documents if doc.metadata['score'] >= 0.28]:
                        source_documents = filtered_documents
                    debug_logger.info(f"rerank step2 num: {len(source_documents)}")
                    saved_docs = [source_documents[0]]
                    for doc in source_documents[1:]:
                        debug_logger.info(f"rerank doc score: {doc.metadata['score']}")
                        relative_difference = (saved_docs[0].metadata['score'] - doc.metadata['score']) / saved_docs[0].metadata['score']
                        if relative_difference > 0.5:
                            break
                        else:
                            saved_docs.append(doc)
                    source_documents = saved_docs
                    debug_logger.info(f"rerank step3 num: {len(source_documents)}")
            except Exception as e:
                time_record['rerank'] = 0.0
                debug_logger.error(f"query {query}: kb_ids: {kb_ids}, rerank error: {traceback.format_exc()}")

        # es检索+milvus检索结果最多可能是2k
        source_documents = source_documents[:top_k]
        print(source_documents)
        # TODO:
        # rerank之后删除headers，只保留文本内容，用于后续处理
        # TODO: 不知道这个在rerank里什么作用
        # for doc in source_documents:
        #     doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)
    
        # 这里是处理FAQ的逻辑，后续在开发
        # high_score_faq_documents = [doc for doc in source_documents if
        #                             doc.metadata['file_name'].endswith('.faq') and doc.metadata['score'] >= 0.9]
        # if high_score_faq_documents:
        #     source_documents = high_score_faq_documents
        # # FAQ完全匹配处理逻辑
        # for doc in source_documents:
        #     if doc.metadata['file_name'].endswith('.faq') and clear_string_is_equal(
        #             doc.metadata['faq_dict']['question'], query):
        #         debug_logger.info(f"match faq question: {query}")
        #         if only_need_search_results:
        #             yield source_documents, None
        #             return
        #         res = doc.metadata['faq_dict']['answer']
        #         async for response, history in self.generate_response(query, res, condense_question, source_documents,
        #                                                               time_record, chat_history, streaming, 'MATCH_FAQ'):
        #             yield response, history
        #         return