import asyncio
import json
import re
import sys
import os

import time
import traceback
import urllib
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

current_dir = os.path.dirname(current_script_path)

root_dir = os.path.dirname(current_dir)

root_dir = os.path.dirname(root_dir)

root_dir = os.path.dirname(root_dir)

# 将项目根目录添加到sys.path
sys.path.append(root_dir)
from src.utils.general_utils import get_time, get_time_async, \
    safe_get, check_user_id_and_user_info, correct_kb_id, \
        check_filename
from src.core.qa_handler import QAHandler
from src.utils.log_handler import debug_logger
from src.utils.general_utils import  fast_estimate_file_char_count
from src.core.file_handler.file_handler import LocalFile, FileHandler
from sanic import request
from sanic.response import text as sanic_text
from sanic.response import json as sanic_json
from sanic.response import ResponseStream
from datetime import datetime, timedelta
from src.configs.configs import DEFAULT_PARENT_CHUNK_SIZE, \
    MAX_CHARS, VECTOR_SEARCH_TOP_K, DEFAULT_API_BASE, DEFAULT_API_KEY,\
          DEFAULT_API_CONTEXT_LENGTH, DEFAULT_MODEL_PATH
import uuid

def format_source_documents(ori_source_documents):
    source_documents = []
    for inum, doc in enumerate(ori_source_documents):
        source_info = {'file_id': doc.metadata.get('file_id', ''),
                       'file_name': doc.metadata.get('file_name', ''),
                       'content': doc.page_content,
                       'retrieval_query': doc.metadata.get('retrieval_query', ''),
                       # 'kernel': doc.metadata['kernel'],
                       'file_url': doc.metadata.get('file_url', ''),
                       'score': str(doc.metadata['score']),
                       'embed_version': doc.metadata.get('embed_version', ''),
                       'nos_keys': doc.metadata.get('nos_keys', ''),
                       'doc_id': doc.metadata.get('doc_id', ''),
                       'retrieval_source': doc.metadata.get('retrieval_source', ''),
                       'headers': doc.metadata.get('headers', {}),
                       'page_id': doc.metadata.get('page_id', 0),
                       }
        source_documents.append(source_info)
    return source_documents

def format_time_record(time_record):
    token_usage = {}
    time_usage = {}
    for k, v in time_record.items():
        if 'tokens' in k:
            token_usage[k] = round(v)
        else:
            time_usage[k] = round(v, 2)
    if 'rewrite_prompt_tokens' in token_usage:
        if 'prompt_tokens' in token_usage:
            token_usage['prompt_tokens'] += token_usage['rewrite_prompt_tokens']
        if 'total_tokens' in token_usage:
            token_usage['total_tokens'] += token_usage['rewrite_prompt_tokens']
    if 'rewrite_completion_tokens' in token_usage:
        if 'completion_tokens' in token_usage:
            token_usage['completion_tokens'] += token_usage['rewrite_completion_tokens']
        if 'total_tokens' in token_usage:
            token_usage['total_tokens'] += token_usage['rewrite_completion_tokens']
    return {"time_usage": time_usage, "token_usage": token_usage}

@get_time_async
async def document(req: request):
    description = """
# SBIRag 介绍
一个用于rustsbi内部的RAG问答系统

**目前已支持格式:**
* PDF
* TXT
* adoc
* ...更多格式，敬请期待
"""
    return sanic_text(description)

@get_time_async
async def health_check(req: request):
    # 实现一个服务健康检查的逻辑，正常就返回200，不正常就返回500
    return sanic_json({"code": 200, "msg": "success"})

@get_time_async
async def new_knowledge_base(req: request):
    qa_handler: QAHandler = req.app.ctx.qa_handler
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    # 检查请求的 user id 和 user info是否合规
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("new_knowledge_base %s", user_id)
    # 获取知识库名称
    kb_name = safe_get(req, 'kb_name')
    debug_logger.info("kb_name: %s", kb_name)
    # 创建知识库id
    kb_id = 'KB' + uuid.uuid4().hex
    # default_kb_id = 'KB' + uuid.uuid4().hex
    # 这里是干啥的我觉得没啥用
    # kb_id = safe_get(req, 'kb_id', default_kb_id)
    # kb_id = correct_kb_id(kb_id)

    if kb_id[:2] != 'KB':
        return sanic_json({"code": 2001, "msg": "fail, kb_id must start with 'KB'"})
    # 判断kb_id是否存在，不存在则返回kb_id，存在则返回空
    not_exist_kb_ids = qa_handler.mysql_client.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        return sanic_json({"code": 2001, "msg": "fail, knowledge Base {} already exist".format(kb_id)})

    # local_doc_qa.create_milvus_collection(user_id, kb_id, kb_name)
    qa_handler.mysql_client.new_milvus_base(kb_id, user_id, kb_name)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create knowledge base {}".format(kb_id),
                       "data": {"kb_id": kb_id, "kb_name": kb_name, "timestamp": timestamp}})

@get_time_async
async def upload_files(req: request):
    # 取qa_handler
    qa_handler: QAHandler = req.app.ctx.qa_handler
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_files %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    # kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id %s", kb_id)
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    debug_logger.info("mode: %s", mode)
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)
    files = req.files.getlist('files')
    debug_logger.info(f"{user_id} upload files number: {len(files)}")
    not_exist_kb_ids = qa_handler.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    
    exist_files = qa_handler.mysql_client.get_files(user_id, kb_id)
    if len(exist_files) + len(files) > 10000:
        return sanic_json({"code": 2002,
                           "msg": f"fail, exist files is {len(exist_files)}, upload files is {len(files)}, total files is {len(exist_files) + len(files)}, max length is 10000."})

    data = []
    local_files = []
    file_names = []
    # 遍历上传文件
    for file in files:
        if isinstance(file, str):
            file_name = os.path.basename(file)
        else:
            debug_logger.info('ori name: %s', file.name)
            # 用于解码URL编码的文件名，主要包含中文和空格
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
        # 删除掉全角字符
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
        debug_logger.info('cleaned name: %s', file_name)

        file_name = check_filename(file_name, max_length=200)
        if file_name is None:
            return sanic_json({"code": 2001, "msg": "fail, file name {} exceeds length limit".format(file_name)})
        file_names.append(file_name)   

    exist_file_names = []
    if mode == 'soft':
        exist_files = qa_handler.mysql_client.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        for exist_file in exist_files:
            file_id, file_name, file_size, status = exist_file
            debug_logger.info(f"{file_name}, {status}, existed files, skip upload")
            # await post_data(user_id, -1, file_id, status, msg='existed files, skip upload')

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    failed_files = []
    record_exist_files = []
    for file, file_name in zip(files, file_names):
        # 对于数据库中同名文件直接跳过，不保存到本地服务器上
        if file_name in exist_file_names:
            record_exist_files.append(file_name)
            continue
        # 将文件保存到本地
        local_file = LocalFile(user_id, kb_id, file, file_name)
        # TODO：现在只能处理txt
        chars = fast_estimate_file_char_count(local_file.file_location)
        debug_logger.info(f"{file_name} char_size: {chars}")
        if chars and chars > MAX_CHARS:
            debug_logger.warning(f"fail, file {file_name} chars is {chars}, max length is {MAX_CHARS}.")
            failed_files.append(file_name)
            continue
        file_id = local_file.file_id
        file_size = len(local_file.file_content)
        file_location = local_file.file_location
        # local_files.append(local_file)
        # 加到mysql数据库中
        msg = qa_handler.mysql_client.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        # 将文件切割向量化并保存到向量数据库中
        kb_name = qa_handler.mysql_client.get_knowledge_base_name([local_file.kb_id])[0][2]
        file_handler = FileHandler(local_file.user_id, kb_name, local_file.kb_id, 
                                         local_file.file_id, local_file.file_location, 
                                         local_file.file_name, chunk_size)
        # txt
        # 将文件转换为Document类型，langchain
        file_handler.split_file_to_docs() # Document类
        # print(file_handler.docs)
        # 将处理好的Document中内容进行切分 切父块800 没重叠  切子块400 重叠部分100
        file_handler.docs, full_docs = FileHandler.split_docs(file_handler.docs)
        parent_chunk_number = len(set(doc.metadata["doc_id"] for doc in file_handler.docs)) # file_handler.docs 列表中每个元素 doc 的不重复的 doc.doc_id 数量
        # 将切分好的Document存入向量数据库中
        qa_handler.milvus_client.load_collection_(user_id)
        for doc in file_handler.docs:
            # 这里应该能用列表直接把所有的给向量化
            textvec = qa_handler.embeddings.embed_query(doc.page_content)
            # print(textvec)
            file_handler.embs.append(textvec)
            qa_handler.milvus_client.store_doc(doc, textvec)
        # 打印切好的子块
        # print(file_handler.docs)
        # 将切分好的子块存入es数据库中
        if qa_handler.es_client is not None:
            try:
                # docs的doc_id是file_id + '_' + i 注意这里的docs_id指的是es数据库中的唯一标识
                # 而不是父块编号
                docs_ids = [doc.metadata['file_id'] + '_' + str(i) for i, doc in enumerate(file_handler.docs)]
                # ids指定文档的唯一标识符
                es_res = await qa_handler.es_client.es_store.aadd_documents(file_handler.docs, ids=docs_ids)
                debug_logger.info(f'es_store insert number: {len(es_res)}, {es_res[0]}')
            except Exception as e:
                debug_logger.error(f"Error in aadd_documents on es_store: {traceback.format_exc()}")
        # 存入完以后更新mysql中file表的chunks_number
        # 将切好的父doc存入mysql数据库中
        qa_handler.mysql_client.store_parent_chunks(full_docs)
        # 更新文件的chunk number
        qa_handler.mysql_client.modify_file_chunks_number(file_id, user_id, kb_id, parent_chunk_number)
        # 返回给前端的数据
        data.append({"file_id": file_id, "file_name": file_name, "status": "green", 
                     "bytes": len(local_file.file_content), "timestamp": timestamp, "estimated_chars": chars})
    # qanything 1.x版本处理方式，2.0以后的版本都是起另外一个服务轮询文件状态，之后添加到向量数据库中
    # 后面做优化在像他们那样做，这样文件上传流程会快不少
    # asyncio.create_task(local_doc_qa.insert_files_to_l(user_id, kb_id, local_files))
    if failed_files:
        msg = f"warning, {failed_files} chars is too much, max characters length is {MAX_CHARS}, skip upload."
    elif record_exist_files:
        msg = f"warning, {record_exist_files} exist in {user_id} and {kb_id}, skip upload."
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


# 获取环境变量GATEWAY_IP 可以在运行之前设置的
GATEWAY_IP = os.getenv("GATEWAY_IP", "localhost")
debug_logger.info(f"GATEWAY_IP: {GATEWAY_IP}")
@get_time_async
async def local_doc_chat(req: request):
    # 具体就是获取一些参数之类的
    preprocess_start = time.perf_counter()
    qa_handler: QAHandler = req.app.ctx.qa_handler
    # 开始处理所需要的参数
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('local_doc_chat %s', user_id)
    debug_logger.info('user_info %s', user_info)
    kb_ids = safe_get(req, 'kb_ids')
    custom_prompt = safe_get(req, 'custom_prompt', None)
    rerank = safe_get(req, 'rerank', default=True)
    only_need_search_results = safe_get(req, 'only_need_search_results', False)
    need_web_search = safe_get(req, 'networking', False)
    api_base = safe_get(req, 'api_base', DEFAULT_API_BASE)
    # 如果api_base中包含0.0.0.0或127.0.0.1或localhost，替换为GATEWAY_IP
    api_base = api_base.replace('0.0.0.0', GATEWAY_IP).replace('127.0.0.1', GATEWAY_IP).replace('localhost',
                                                                                                GATEWAY_IP)
    api_key = safe_get(req, 'api_key', DEFAULT_API_KEY)
    api_context_length = safe_get(req, 'api_context_length', DEFAULT_API_CONTEXT_LENGTH)
    top_p = safe_get(req, 'top_p', 0.99)
    temperature = safe_get(req, 'temperature', 0.5)
    top_k = safe_get(req, 'top_k', VECTOR_SEARCH_TOP_K)

    model = safe_get(req, 'model', DEFAULT_MODEL_PATH)
    max_token = safe_get(req, 'max_token')

    hybrid_search = safe_get(req, 'hybrid_search', False)
    chunk_size = safe_get(req, 'chunk_size', DEFAULT_PARENT_CHUNK_SIZE)

    debug_logger.info('rerank %s', rerank)

    if len(kb_ids) > 20:
        return sanic_json({"code": 2005, "msg": "fail, kb_ids length should less than or equal to 20"})
    
    # kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    question = safe_get(req, 'question')
    streaming = safe_get(req, 'streaming', False)
    history = safe_get(req, 'history', [])

    if top_k > 100:
        return sanic_json({"code": 2003, "msg": "fail, top_k should less than or equal to 100"})

    missing_params = []
    if not api_base:
        missing_params.append('api_base')
    if not api_key:
        missing_params.append('api_key')
    if not api_context_length:
        missing_params.append('api_context_length')
    if not top_p:
        missing_params.append('top_p')
    if not top_k:
        missing_params.append('top_k')
    if top_p == 1.0:
        top_p = 0.99
    if not temperature:
        missing_params.append('temperature')

    if missing_params:
        missing_params_str = " and ".join(missing_params) if len(missing_params) > 1 else missing_params[0]
        return sanic_json({"code": 2003, "msg": f"fail, {missing_params_str} is required"})
    
    if only_need_search_results and streaming:
        return sanic_json(
            {"code": 2006, "msg": "fail, only_need_search_results and streaming can't be True at the same time"})
    request_source = safe_get(req, 'source', 'unknown')

    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)
    debug_logger.info("custom_prompt: %s", custom_prompt)
    debug_logger.info("model: %s", model)
    debug_logger.info("max_token: %s", max_token)
    debug_logger.info("request_source: %s", request_source)
    debug_logger.info("only_need_search_results: %s", only_need_search_results)
    debug_logger.info("need_web_search: %s", need_web_search)
    debug_logger.info("api_base: %s", api_base)
    debug_logger.info("api_key: %s", api_key)
    debug_logger.info("api_context_length: %s", api_context_length)
    debug_logger.info("top_p: %s", top_p)
    debug_logger.info("top_k: %s", top_k)
    debug_logger.info("temperature: %s", temperature)
    debug_logger.info("hybrid_search: %s", hybrid_search)
    debug_logger.info("chunk_size: %s", chunk_size)

    qa_handler.milvus_client.load_collection_(user_id)
    if kb_ids:
        not_exist_kb_ids = qa_handler.mysql_client.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})
    
    # print("DEBUG: ", user_id, kb_ids)
    
    file_infos = []
    for kb_id in kb_ids:
        file_infos.extend(qa_handler.mysql_client.get_files(user_id, kb_id))
        
    # print("DEBUG: ", file_infos)
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        debug_logger.info("valid_files is empty, use only chat mode.")
        kb_ids = []
    
    preprocess_end = time.perf_counter()
    time_record = {}
    time_record['preprocess'] = round(preprocess_end - preprocess_start, 2)
    # 获取查询的时间，更新mysql中知识库的最后查询时间,获取格式为'2021-08-01 00:00:00'的时间戳
    # qa_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # for kb_id in kb_ids:
    #     qa_handler.mysql_client.update_knowledge_base_latest_qa_time(kb_id, qa_timestamp)
    debug_logger.info("streaming: %s", streaming)
    if streaming:
        debug_logger.info("start generate answer")
        async def generate_answer(response):
            debug_logger.info("start generate...")
            async for resp, next_history in qa_handler.get_knowledge_based_answer(model=model,
                                                                                    max_token=max_token,
                                                                                    kb_ids=kb_ids,
                                                                                    query=question,
                                                                                    retriever=qa_handler.retriever,
                                                                                    chat_history=history,
                                                                                    streaming=True,
                                                                                    rerank=rerank,
                                                                                    custom_prompt=custom_prompt,
                                                                                    time_record=time_record,
                                                                                    need_web_search=need_web_search,
                                                                                    hybrid_search=hybrid_search,
                                                                                    web_chunk_size=chunk_size,
                                                                                    temperature=temperature,
                                                                                    api_base=api_base,
                                                                                    api_key=api_key,
                                                                                    api_context_length=api_context_length,
                                                                                    top_p=top_p,
                                                                                    top_k=top_k
                                                                                    ):
                    chunk_data = resp["result"]
                    if not chunk_data:
                        continue
                    chunk_str = chunk_data[6:]
                    if chunk_str.startswith("[DONE]"):
                        retrieval_documents = format_source_documents(resp["retrieval_documents"])
                        source_documents = format_source_documents(resp["source_documents"])
                        result = next_history[-1][1]
                        # result = resp['result']
                        time_record['chat_completed'] = round(time.perf_counter() - preprocess_start, 2)
                        if time_record.get('llm_completed', 0) > 0:
                            time_record['tokens_per_second'] = round(
                                len(result) / time_record['llm_completed'], 2)
                        formatted_time_record = format_time_record(time_record)
                        chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, "model": model,
                                    "product_source": request_source, 'time_record': formatted_time_record,
                                    'history': history,
                                    'condense_question': resp['condense_question'], 'prompt': resp['prompt'],
                                    'result': result, 'retrieval_documents': retrieval_documents,
                                    'source_documents': source_documents}
                        qa_handler.mysql_client.add_qalog(**chat_data)
                        debug_logger.info("chat_data: %s", chat_data)
                        debug_logger.info("response: %s", chat_data['result'])
                        stream_res = {
                            "code": 200,
                            "msg": "success stream chat",
                            "question": question,
                            "response": result,
                            "model": model,
                            "history": next_history,
                            "condense_question": resp['condense_question'],
                            "source_documents": source_documents,
                            "retrieval_documents": retrieval_documents,
                            "time_record": formatted_time_record,
                            "show_images": resp.get('show_images', [])
                        }
                    else:
                        time_record['rollback_length'] = resp.get('rollback_length', 0)
                        if 'first_return' not in time_record:
                            time_record['first_return'] = round(time.perf_counter() - preprocess_start, 2)
                        chunk_js = json.loads(chunk_str)
                        delta_answer = chunk_js["answer"]
                        stream_res = {
                            "code": 200,
                            "msg": "success",
                            "question": "",
                            "response": delta_answer,
                            "history": [],
                            "source_documents": [],
                            "retrieval_documents": [],
                            "time_record": format_time_record(time_record),
                        }
                    await response.write(f"data: {json.dumps(stream_res, ensure_ascii=False)}\n\n")
                    if chunk_str.startswith("[DONE]"):
                        await response.eof()
                    await asyncio.sleep(0.001)

            response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
            return response_stream
    else:
        # 进行检索生成回答
        async for resp, history in qa_handler.get_knowledge_based_answer(model=model,
                                                                           max_token=max_token,
                                                                           kb_ids=kb_ids,
                                                                           query=question,
                                                                           retriever=qa_handler.retriever,
                                                                           chat_history=history, streaming=False,
                                                                           rerank=rerank,
                                                                           custom_prompt=custom_prompt,
                                                                           time_record=time_record,
                                                                           only_need_search_results=only_need_search_results,
                                                                           need_web_search=need_web_search,
                                                                           hybrid_search=hybrid_search,
                                                                           web_chunk_size=chunk_size,
                                                                           temperature=temperature,
                                                                           api_base=api_base,
                                                                           api_key=api_key,
                                                                           api_context_length=api_context_length,
                                                                           top_p=top_p,
                                                                           top_k=top_k
                                                                           ):
            # 如果只需要检索到的文档
            if only_need_search_results:
                return sanic_json(
                    {"code": 200, "question": question, "source_documents": format_source_documents(resp)})
            # 格式化检索到的文档信息
            retrieval_documents = format_source_documents(resp["retrieval_documents"])
            source_documents = format_source_documents(resp["source_documents"])
            formatted_time_record = format_time_record(time_record)
            chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'time_record': formatted_time_record,
                        'history': history, "condense_question": resp['condense_question'], "model": model,
                        "product_source": request_source,
                        'retrieval_documents': retrieval_documents, 'prompt': resp['prompt'], 'result': resp['result'],
                        'source_documents': source_documents}
            # qa_handler.mysql_client.add_qalog(**chat_data)
            debug_logger.info("chat_data: %s", chat_data)
            debug_logger.info("response: %s", chat_data['result'])
            return sanic_json({"code": 200, "msg": "success no stream chat", "question": question,
                            "response": resp["result"], "model": model,
                            "history": history, "condense_question": resp['condense_question'],
                            "source_documents": source_documents, "retrieval_documents": retrieval_documents,
                            "time_record": formatted_time_record})

@get_time_async
async def list_kbs(req: request):
    """
    Placeholder: This function is intended to list all knowledge bases for a given user.
    The actual implementation for querying the database and returning the list
    needs to be added here.
    """
    return sanic_json({"code": 200, "msg": "success", "data": []})