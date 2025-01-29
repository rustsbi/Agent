import re
import sys
import os

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
from datetime import datetime, timedelta
from src.configs.configs import DEFAULT_PARENT_CHUNK_SIZE, MAX_CHARS
import uuid

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
    not_exist_kb_ids = qa_handler.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        return sanic_json({"code": 2001, "msg": "fail, knowledge Base {} already exist".format(kb_id)})

    # local_doc_qa.create_milvus_collection(user_id, kb_id, kb_name)
    qa_handler.milvus_summary.new_milvus_base(kb_id, user_id, kb_name)
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
    not_exist_kb_ids = qa_handler.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    
    exist_files = qa_handler.milvus_summary.get_files(user_id, kb_id)
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
        exist_files = qa_handler.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
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
        msg = qa_handler.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        # 将文件切割向量化并保存到向量数据库中
        kb_name = qa_handler.milvus_summary.get_knowledge_base_name([local_file.kb_id])[0][2]
        file_handler = FileHandler(local_file.user_id, kb_name, local_file.kb_id, 
                                         local_file.file_id, local_file.file_location, 
                                         local_file.file_name, chunk_size)
        # txt
        # 将文件转换为Document类型，langchain
        file_handler.split_file_to_docs() # Document类
        # print(file_handler.docs)
        # 将处理好的Document中内容进行切分 切父块800 没重叠  切子块400 重叠部分100
        file_handler.docs = FileHandler.split_docs(file_handler.docs)
        parent_chunk_number = len(set(doc.metadata["doc_id"] for doc in file_handler.docs)) # file_handler.docs 列表中每个元素 doc 的不重复的 doc.doc_id 数量
        # TODO 将切分好的Document存入向量数据库中
        qa_handler.milvus_kb.load_collection_(user_id)
        for doc in file_handler.docs:
            textvec = qa_handler.embeddings.embed_query(doc.page_content)
            # print(textvec)
            file_handler.embs.append(textvec)
            qa_handler.milvus_kb.store_doc(doc, textvec)
        # 向量数据库存 向量数据库搜索
        print(file_handler.docs)
        # TODO：存入完以后更新mysql中file表的chunks_number，状态之类的后面再说吧，先把关键的增删改查写了
        qa_handler.milvus_summary.modify_file_chunks_number(file_id, user_id, kb_id, parent_chunk_number)
        # 返回给前端的数据
        data.append({"file_id": file_id, "file_name": file_name, "status": "green", 
                     "bytes": len(local_file.file_content), "timestamp": timestamp, "estimated_chars": chars})
    # qanything 1.x版本处理方式，2.0以后的版本都是起另外一个服务轮训文件状态，之后添加到向量数据库中
    # 后面做优化在像他们那样做，这样文件上传流程会快不少
    # asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, local_files))
    if failed_files:
        msg = f"warning, {failed_files} chars is too much, max characters length is {MAX_CHARS}, skip upload."
    elif record_exist_files:
        msg = f"warning, {record_exist_files} exist in {user_id} and {kb_id}, skip upload."
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})