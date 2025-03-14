
import json
import re
import sys
import os
import time
import traceback
from typing import List, Tuple
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(root_dir)

from src.client.embedding.embedding_client import SBIEmbeddings
from src.client.rerank.client import SBIRerank
from src.configs.configs import VECTOR_SEARCH_SCORE_THRESHOLD, CUSTOM_PROMPT_TEMPLATE,\
    SYSTEM, PROMPT_TEMPLATE, INSTRUCTIONS, SIMPLE_PROMPT_TEMPLATE
from src.utils.log_handler import debug_logger
from src.client.database.mysql.mysql_client import MysqlClient
from src.client.database.milvus.milvus_client import MilvusClient
from src.client.database.elasticsearch.es_client import ESClient
from src.core.retriever.retriever import Retriever
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema import Document
from src.client.llm.llm_client import OpenAILLM
from src.core.chains.condense_q_chain import RewriteQuestionChain

from src.utils.general_utils import deduplicate_documents, num_tokens, num_tokens_rerank, my_print, replace_image_references

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
        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")
        # debug_logger.info(f"query_docs num: {len(query_docs)}, query_docs: {query_docs}")
        for idx, doc in enumerate(query_docs):
            if self.mysql_client.is_deleted_file(doc.metadata['file_id']):
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
    
    def reprocess_source_documents(self, custom_llm: OpenAILLM, query: str,
                    source_docs: List[Document],
                    history: List[str],
                    prompt_template: str) -> Tuple[List[Document], int, str]:
        # 组装prompt,根据max_token
        query_token_num = int(custom_llm.num_tokens_from_messages([query]) * 4)
        history_token_num = int(custom_llm.num_tokens_from_messages([x for sublist in history for x in sublist]))
        template_token_num = int(custom_llm.num_tokens_from_messages([prompt_template]))
        # 计算引用所消耗的token
        reference_field_token_num = int(custom_llm.num_tokens_from_messages(
            [f"<reference>[{idx + 1}]</reference>" for idx in range(len(source_docs))]))
        # 计算还能容纳多少token的doc，之后往里面填充doc
        limited_token_nums = custom_llm.token_window - custom_llm.max_token - custom_llm.offcut_token - query_token_num - history_token_num - template_token_num - reference_field_token_num

        debug_logger.info(f"=============================================")
        debug_logger.info(f"token_window = {custom_llm.token_window}")
        debug_logger.info(f"max_token = {custom_llm.max_token}")
        debug_logger.info(f"offcut_token = {custom_llm.offcut_token}")
        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"reference_field token nums: {reference_field_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"=============================================")

        tokens_msg = """
        token_window = {custom_llm.token_window}, max_token = {custom_llm.max_token},       
        offcut_token = {custom_llm.offcut_token}, docs_available_token_nums: {limited_token_nums}, 
        template token nums: {template_token_num}, reference_field token nums: {reference_field_token_num}, 
        query token nums: {query_token_num}, history token nums: {history_token_num}
        docs_available_token_nums = token_window - max_token - offcut_token - query_token_num * 4 - history_token_num - template_token_num - reference_field_token_num
        """.format(custom_llm=custom_llm, limited_token_nums=limited_token_nums, template_token_num=template_token_num,
                     reference_field_token_num=reference_field_token_num, query_token_num=query_token_num // 4,
                     history_token_num=history_token_num)

        # if limited_token_nums < 200:
        #     return []

        new_source_docs = []
        total_token_num = 0
        not_repeated_file_ids = []
        # 从前向后填doc，直到长度超出就停止
        for doc in source_docs:
            headers_token_num = 0
            file_id = doc.metadata['file_id']
            if file_id not in not_repeated_file_ids:
                not_repeated_file_ids.append(file_id)
                if 'headers' in doc.metadata:
                    headers = f"headers={doc.metadata['headers']}"
                    headers_token_num = custom_llm.num_tokens_from_messages([headers])
            doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)
            doc_token_num = custom_llm.num_tokens_from_messages([doc_valid_content])
            doc_token_num += headers_token_num
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                break

        debug_logger.info(f"new_source_docs token nums: {custom_llm.num_tokens_from_docs(new_source_docs)}")
        # 返回新的doc列表，给doc剩余的token数量，token计算的信息
        return new_source_docs, limited_token_nums, tokens_msg
    
    @staticmethod
    async def generate_response(query, res, condense_question, source_documents, time_record, chat_history, streaming, prompt):
        """
        生成response并使用yield返回。

        :param query: 用户的原始查询
        :param res: 生成的答案
        :param condense_question: 压缩后的问题
        :param source_documents: 从检索中获取的文档
        :param time_record: 记录时间的字典
        :param chat_history: 聊天历史
        :param streaming: 是否启用流式输出
        :param prompt: 生成response时的prompt类型
        """
        history = chat_history + [[query, res]]

        if streaming:
            res = 'data: ' + json.dumps({'answer': res}, ensure_ascii=False)

        response = {
            "query": query,
            "prompt": prompt,  # 允许自定义 prompt
            "result": res,
            "condense_question": condense_question,
            "retrieval_documents": source_documents,
            "source_documents": source_documents
        }

        if 'llm_completed' not in time_record:
            time_record['llm_completed'] = 0.0
        if 'total_tokens' not in time_record:
            time_record['total_tokens'] = 0
        if 'prompt_tokens' not in time_record:
            time_record['prompt_tokens'] = 0
        if 'completion_tokens' not in time_record:
            time_record['completion_tokens'] = 0

        # 使用yield返回response和history
        yield response, history

        # 如果是流式输出，发送结束标志
        if streaming:
            response['result'] = "data: [DONE]\n\n"
            yield response, history
    # 生成prompt
    def generate_prompt(self, query, source_docs, prompt_template):
        if source_docs:
            context = ''
            not_repeated_file_ids = []
            for doc in source_docs:
                # 生成prompt时去掉图片
                doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)
                file_id = doc.metadata['file_id']
                # 如果这个file_id是第一次遇到
                if file_id not in not_repeated_file_ids:
                    # 说明前面也引用了一个文档，给前面的引用加入闭合符号
                    if len(not_repeated_file_ids) != 0:
                        context += '</reference>\n'
                    # 将file_id加入到已处理列表
                    not_repeated_file_ids.append(file_id)
                    # 如果有headers则加入headers， 没有的话只加入内容
                    if 'headers' in doc.metadata:
                        headers = f"headers={doc.metadata['headers']}"
                        context += f"<reference {headers}>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                    else:
                        context += f"<reference>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                else:
                    # 如果file_id不是第一次处理
                    context += doc_valid_content + '\n'
            context += '</reference>\n'

            # prompt = prompt_template.format(context=context).replace("{{question}}", query)
            prompt = prompt_template.replace("{{context}}", context).replace("{{question}}", query)
        else:
            prompt = prompt_template.replace("{{question}}", query)
        return prompt

    async def prepare_source_documents(self, custom_llm: OpenAILLM, retrieval_documents: List[Document],
                                       limited_token_nums: int, rerank: bool):
        debug_logger.info(f"retrieval_documents len: {len(retrieval_documents)}")
        try:
            new_docs = self.aggregate_documents(retrieval_documents, limited_token_nums, custom_llm, rerank)
            if new_docs:
                source_documents = new_docs
            else:
                # 合并所有候选文档，从前往后，所有file_id相同的文档合并，按照doc_id排序
                merged_documents_file_ids = []
                for doc in retrieval_documents:
                    if doc.metadata['file_id'] not in merged_documents_file_ids:
                        merged_documents_file_ids.append(doc.metadata['file_id'])
                source_documents = []
                for file_id in merged_documents_file_ids:
                    docs = [doc for doc in retrieval_documents if doc.metadata['file_id'] == file_id]
                    docs = sorted(docs, key=lambda x: int(x.metadata['doc_id'].split('_')[-1]))
                    source_documents.extend(docs)

            # source_documents = self.incomplete_table(source_documents, limited_token_nums, custom_llm)
        except Exception as e:
            debug_logger.error(f"aggregate_documents error w/ {e}: {traceback.format_exc()}")
            source_documents = retrieval_documents

        debug_logger.info(f"source_documents len: {len(source_documents)}")
        return source_documents, retrieval_documents

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, top_k, web_chunk_size,
                                         chat_history=None, streaming: bool = True, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        # 创建与大模型交互句柄
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)
        if chat_history is None:
            chat_history = []
        retrieval_query = query
        condense_question = query
        # 如果有对话历史就将对话历史和query结合进行query重写
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
                # 调用大模型对 带有对话历史的查询 进行重写
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
        # 如果有kb_ids那么需要对重写后的查询进行向量检索
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
        # 将检索内容进行去重
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
                    # 如果没有大于等于0.28的分数则保留
                    if filtered_documents := [doc for doc in source_documents if doc.metadata['score'] >= 0.28]:
                        source_documents = filtered_documents
                    debug_logger.info(f"rerank step2 num: {len(source_documents)}")
                    saved_docs = [source_documents[0]]
                    # 根据相对分数来过滤文档块
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
        my_print(source_documents)
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
        # 获取今日日期
        today = time.strftime("%Y-%m-%d", time.localtime())
        # 获取当前时间
        now = time.strftime("%H:%M:%S", time.localtime())

        extra_msg = None
        total_images_number = 0
        retrieval_documents = []
        if source_documents:
            # 如果有自定义Prompt，则使用自定义Prompt
            if custom_prompt:
                # escaped_custom_prompt = custom_prompt.replace('{', '{{').replace('}', '}}')
                # prompt_template = CUSTOM_PROMPT_TEMPLATE.format(custom_prompt=escaped_custom_prompt)
                # 将自定义提示插入到预定义的自定义提示模板中
                prompt_template = CUSTOM_PROMPT_TEMPLATE.replace("{{custom_prompt}}", custom_prompt)
            else:
                # 否则使用系统默认提示
                # 首先在系统提示中替换日期和时间
                # system_prompt = SYSTEM.format(today_date=today, current_time=now)
                system_prompt = SYSTEM.replace("{{today_date}}", today).replace("{{current_time}}", now)
                # prompt_template = PROMPT_TEMPLATE.format(system=system_prompt, instructions=INSTRUCTIONS)
                # 然后构建完整的提示模板
                prompt_template = PROMPT_TEMPLATE.replace("{{system}}", system_prompt).replace("{{instructions}}",
                                                                                               INSTRUCTIONS)

            t1 = time.perf_counter()
            # 计算已使用token， 如果超过限制的token数量则用贪心减少doc数量
            retrieval_documents, limited_token_nums, tokens_msg = self.reprocess_source_documents(custom_llm=custom_llm,
                                                                                                  query=query,
                                                                                                  source_docs=source_documents,
                                                                                                  history=chat_history,
                                                                                                  prompt_template=prompt_template)
            if len(retrieval_documents) < len(source_documents):
                # 重新处理后文档数量减少，说明由于tokens不足而被裁切
                if len(retrieval_documents) == 0:  # 说明被裁切后文档数量为0
                    debug_logger.error(f"limited_token_nums: {limited_token_nums} < {web_chunk_size}!")
                    res = (
                        f"抱歉，由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums} < 文本分片大小: {web_chunk_size})，"
                        f"\n无法保证回答质量，请在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。"
                        f"\n计算方式：{tokens_msg}")
                    # 生成统一的恢复，并返回
                    async for response, history in self.generate_response(query, res, condense_question, source_documents,
                                                                          time_record, chat_history, streaming,
                                                                          'TOKENS_NOT_ENOUGH'):
                        yield response, history
                    return
                extra_msg = (
                    f"\n\nWARNING: 由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums})，"
                    f"\n检索到的部分文档chunk被裁切，原始来源数量：{len(source_documents)}，裁切后数量：{len(retrieval_documents)}，"
                    f"\n可能会影响回答质量，尤其是问题涉及的相关内容较多时。"
                    f"\n可在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。\n")
            source_documents, retrieval_documents = await self.prepare_source_documents(custom_llm,
                                                                                        retrieval_documents,
                                                                                        limited_token_nums,
                                                                                        rerank)
            for doc in source_documents:
                if doc.metadata.get('images', []):
                    total_images_number += len(doc.metadata['images'])
                    doc.page_content = replace_image_references(doc.page_content, doc.metadata['file_id'])
            debug_logger.info(f"total_images_number: {total_images_number}")

            t2 = time.perf_counter()
            time_record['reprocess'] = round(t2 - t1, 2)
        else:
            # 如果不存在引用文档，则选用简单的Prompt
            if custom_prompt:
                # escaped_custom_prompt = custom_prompt.replace('{', '{{').replace('}', '}}')
                # prompt_template = SIMPLE_PROMPT_TEMPLATE.format(today=today, now=now, custom_prompt=escaped_custom_prompt)
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", custom_prompt)
            else:
                simple_custom_prompt = """
                - If you cannot answer based on the given information, you will return the sentence \"抱歉，已知的信息不足，因此无法回答。\". 
                """
                # prompt_template = SIMPLE_PROMPT_TEMPLATE.format(today=today, now=now, custom_prompt=simple_custom_prompt)
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", simple_custom_prompt)
        # 如果只需要搜索到的结果，则直接返回搜索到的文档
        if only_need_search_results:
            yield source_documents, None
            return

        t1 = time.perf_counter()
        has_first_return = False

        acc_resp = ''
        # 在这之前应该对source_docs的file_id进行排序后，生成Prompt
        # 上面的prepare_source_documents做了这件事
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)
        # debug_logger.info(f"prompt: {prompt}")
        # 计算Prompt的token数量
        est_prompt_tokens = num_tokens(prompt) + num_tokens(str(chat_history))
        # 调用大模型结构，生成回答
        async for answer_result in custom_llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
            resp = answer_result.llm_output["answer"]
            if 'answer' in resp:
                acc_resp += json.loads(resp[6:])['answer']
            prompt = answer_result.prompt
            history = answer_result.history
            total_tokens = answer_result.total_tokens
            prompt_tokens = answer_result.prompt_tokens
            completion_tokens = answer_result.completion_tokens
            history[-1][0] = query
            response = {"query": query,
                        "prompt": prompt,
                        "result": resp,
                        "condense_question": condense_question,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents}
            # 记录token和耗时等信息
            time_record['prompt_tokens'] = prompt_tokens if prompt_tokens != 0 else est_prompt_tokens
            time_record['completion_tokens'] = completion_tokens if completion_tokens != 0 else num_tokens(acc_resp)
            time_record['total_tokens'] = total_tokens if total_tokens != 0 else time_record['prompt_tokens'] + \
                                                                                 time_record['completion_tokens']
            # 记录第一次返回的时间
            if has_first_return is False:
                first_return_time = time.perf_counter()
                has_first_return = True
                time_record['llm_first_return'] = round(first_return_time - t1, 2)
            # 处理流式输出
            if resp[6:].startswith("[DONE]"):
                if extra_msg is not None:
                    msg_response = {"query": query,
                                "prompt": prompt,
                                "result": f"data: {json.dumps({'answer': extra_msg}, ensure_ascii=False)}",
                                "condense_question": condense_question,
                                "retrieval_documents": retrieval_documents,
                                "source_documents": source_documents}
                    yield msg_response, history
                last_return_time = time.perf_counter()
                time_record['llm_completed'] = round(last_return_time - t1, 2) - time_record['llm_first_return']
                history[-1][1] = acc_resp
                # 如果有图片，需要处理回答带图的情况
                if total_images_number != 0:  
                    docs_with_images = [doc for doc in source_documents if doc.metadata.get('images', [])]
                    time1 = time.perf_counter()
                    relevant_docs = await self.calculate_relevance_optimized(
                        question=query,
                        llm_answer=acc_resp,
                        reference_docs=docs_with_images,
                        top_k=1
                    )
                    show_images = ["\n### 引用图文如下：\n"]
                    # 输出图片信息
                    for doc in relevant_docs:
                        print(f"文档: {doc['document']}...")  # 只打印前50个字符
                        print(f"最相关段落: {doc['segment']}...")  # 打印最相关段落的前100个字符
                        print(f"与LLM回答的相似度: {doc['similarity_llm']:.4f}")
                        print(f"原始问题相关性分数: {doc['question_score']:.4f}")
                        print(f"综合得分: {doc['combined_score']:.4f}")
                        print()
                        for image in doc['document'].metadata.get('images', []):
                            image_str = replace_image_references(image, doc['document'].metadata['file_id'])
                            debug_logger.info(f"image_str: {image} -> {image_str}")
                            show_images.append(image_str + '\n')
                    debug_logger.info(f"show_images: {show_images}")
                    time_record['obtain_images'] = round(time.perf_counter() - last_return_time, 2)
                    time2 = time.perf_counter()
                    debug_logger.info(f"obtain_images time: {time2 - time1}s")
                    time_record["obtain_images_time"] = round(time2 - time1, 2)
                    if len(show_images) > 1:
                        response['show_images'] = show_images
            # my_print(response)
            # my_print(history)
            yield response, history