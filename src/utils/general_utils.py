import time
from functools import wraps  # 添加这行导入
from src.utils.log_handler import debug_logger, embed_logger, rerank_logger
from src.configs.configs import KB_SUFFIX
from sanic.request import Request
from sanic.exceptions import BadRequest
import logging
import traceback
import re
import mimetypes
import os
import fitz  # PyMuPDF
import chardet
# 异步执行环境下的耗时统计装饰器

from typing import Union, Tuple, Dict
from sanic.request import File
from src.configs.configs import UPLOAD_ROOT_PATH
import uuid
import os


class LocalFile:
    def __init__(self, user_id, kb_id, file: File, file_name):
        self.user_id = user_id
        self.kb_id = kb_id
        self.file_id = uuid.uuid4().hex
        self.file_name = file_name
        self.file_content = file.body
        upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
        file_dir = os.path.join(upload_path, self.kb_id, self.file_id)
        os.makedirs(file_dir, exist_ok=True)
        self.file_location = os.path.join(file_dir, self.file_name)
        #  如果文件不存在：
        if not os.path.exists(self.file_location):
            with open(self.file_location, 'wb') as f:
                f.write(self.file_content)


def get_time_async(func):
    @wraps(func)
    async def get_time_async_inner(*args, **kwargs):
        s_time = time.perf_counter()
        res = await func(*args, **kwargs)  # 注意这里使用 await 来调用异步函数
        e_time = time.perf_counter()
        if 'embed' in func.__name__:
            embed_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        elif 'rerank' in func.__name__:
            rerank_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        else:
            debug_logger.info('函数 {} 执行耗时: {:.2f} 毫秒'.format(func.__name__, (e_time - s_time) * 1000))
        return res

    return get_time_async_inner

# 同步执行环境下的耗时统计装饰器
def get_time(func):
    def get_time_inner(*arg, **kwargs):
        s_time = time.time()
        res = func(*arg, **kwargs)
        e_time = time.time()
        if 'embed' in func.__name__:
            embed_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        elif 'rerank' in func.__name__:
            rerank_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        else:
            debug_logger.info('函数 {} 执行耗时: {:.2f} 毫秒'.format(func.__name__, (e_time - s_time) * 1000))
        return res

    return get_time_inner

def safe_get(req: Request, attr: str, default=None):
    """
    安全地从请求中获取参数值
    
    参数：
    req: Request - Flask/FastAPI的请求对象
    attr: str - 要获取的参数名
    default: Any - 如果获取失败时返回的默认值
    """
    try:
        # 1. 检查表单数据（multipart/form-data 或 application/x-www-form-urlencoded）
        if attr in req.form:
            # Sanic中form数据是列表形式，取第一个值
            return req.form.getlist(attr)[0]
        # 2. 检查URL查询参数 (?key=value)
        if attr in req.args:
            return req.args[attr]
        # 3. 检查JSON数据体 (application/json)
        if attr in req.json:
            return req.json[attr]
    except BadRequest:
        logging.warning(f"missing {attr} in request")
    except Exception as e:
        logging.warning(f"get {attr} from request failed:")
        logging.warning(traceback.format_exc())
    return default

def validate_user_id(user_id):
    if len(user_id) > 64:
        return False
    # 定义正则表达式模式
    pattern = r'^[A-Za-z][A-Za-z0-9_]*$'
    # 检查是否匹配
    if isinstance(user_id, str) and re.match(pattern, user_id):
        return True
    else:
        return False

def get_invalid_user_id_msg(user_id):
    return "fail, Invalid user_id: {}. user_id 长度必须小于64，且必须只含有字母，数字和下划线且字母开头".format(user_id)

def correct_kb_id(kb_id):
    if not kb_id:
        return kb_id
    # 如果kb_id末尾不是KB_SUFFIX,则加上
    if KB_SUFFIX not in kb_id:
        # 判断有FAQ的时候
        # if kb_id.endswith('_FAQ'):  # KBc86eaa3f278f4ef9908780e8e558c6eb_FAQ
        #     return kb_id.split('_FAQ')[0] + KB_SUFFIX + '_FAQ'
        # else:  # KBc86eaa3f278f4ef9908780e8e558c6eb
        #     return kb_id + KB_SUFFIX
        return kb_id + KB_SUFFIX
    else:
        return kb_id

def check_user_id_and_user_info(user_id, user_info):
    if user_id is None or user_info is None:
        msg = "fail, user_id 或 user_info 为 None"
        return False, msg
    if not validate_user_id(user_id):
        msg = get_invalid_user_id_msg(user_id)
        return False, msg
    if not user_info.isdigit():
        msg = "fail, user_info 必须是纯数字"
        return False, msg
    return True, 'success'

def read_files_with_extensions():
    # 获取当前脚本文件的路径
    current_file = os.path.abspath(__file__)

    # 获取当前脚本文件所在的目录
    current_dir = os.path.dirname(current_file)

    # 获取项目根目录
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    directory = project_dir + '/data'

    extensions = ['.md', '.txt', '.pdf', '.jpg', '.docx', '.xlsx', '.eml', '.csv', 'pptx', 'jpeg', 'png']

    files = []
    for root, dirs, files_list in os.walk(directory):
        for file in files_list:
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    mime_type, _ = mimetypes.guess_type(file_path)
                    if mime_type is None:
                        mime_type = 'application/octet-stream'
                    # 模拟 req.files.getlist('files') 返回的对象
                    file_obj = type('FileStorage', (object,), {
                        'name': file,
                        'type': mime_type,
                        'body': file_content
                    })()
                    files.append(file_obj)
    return files

def check_filename(filename, max_length=200):

    # 计算文件名长度，注意中文字符
    filename_length = len(filename.encode('utf-8'))

    # 如果文件名长度超过最大长度限制
    if filename_length > max_length:
        debug_logger.warning("文件名长度超过最大长度限制，返回None")
        return None

    return filename

def fast_estimate_file_char_count(file_path):
    """
    快速估算文件的字符数，如果超过max_chars则返回False，否则返回True
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension in ['.md', '.txt', '.csv']:
            with open(file_path, 'rb') as file:
                raw = file.read(1024)
                encoding = chardet.detect(raw)['encoding']
            with open(file_path, 'r', encoding=encoding) as file:
                char_count = sum(len(line) for line in file)

        elif file_extension == '.pdf':
            doc = fitz.open(file_path)
            char_count = sum(len(page.get_text()) for page in doc)
            doc.close()

        else:
            # 不支持的文件类型
            return None

        return char_count

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None