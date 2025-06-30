import aiohttp
import asyncio
from typing import Optional, Dict, Any, Union
import logging
from aiohttp import ClientError
from asyncio import TimeoutError
import os
import sys
import json

# --- ANSI 转义码，用于颜色 ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m' # 黄色
    FAIL = '\033[91m'    # 红色
    ENDC = '\033[0m'     # 重置颜色
    BOLD = '\033[1m'     # 粗体
    UNDERLINE = '\033[4m'# 下划线

# --- 配置日志，使用颜色 ---
class ColoredFormatter(logging.Formatter):
    FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
    LOG_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.FAIL,
        logging.CRITICAL: Colors.BOLD + Colors.FAIL
    }

    def format(self, record):
        log_fmt = self.LOG_COLORS.get(record.levelno) + self.FORMAT + Colors.ENDC
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# 清除现有日志处理器并添加我们的彩色处理器
logging.getLogger().handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# --- 路径配置 ---
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(root_dir)

# 从 src.configs.configs 导入 LLM 相关的 API_BASE 和 API_KEY
from src.configs.configs import DEFAULT_API_BASE, DEFAULT_API_KEY

# 定义 Sanic API 服务器的固定地址
SANIC_API_SERVER_URL = "http://127.0.0.1:8777/api"

class AsyncHTTPClient:
    def __init__(self, retries: int = 3, timeout: int = 10): 
        self.retries = retries
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    async def request(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> Optional[Union[Dict, str]]: 
        
        for attempt in range(self.retries):
            try:
                logging.debug(f"尝试 {attempt + 1}: {method} {url}")
                async with self.session.request(
                    method, 
                    url, 
                    timeout=self.timeout,
                    **kwargs
                ) as response:
                    if response.status in (200, 201):
                        content_type = response.headers.get('Content-Type', '')
                        
                        if 'application/json' in content_type:
                            return await response.json()
                        elif 'text/plain' in content_type:
                            return await response.text()
                        else:
                            text = await response.text()
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                logging.warning(f"{Colors.WARNING}响应既不是 JSON 也不是纯文本。返回原始文本。{Colors.ENDC}")
                                return text
                    
                    if response.status >= 500:
                        if attempt < self.retries - 1:
                            logging.warning(f"{Colors.WARNING}服务器错误 {response.status} ({url})。在 {2 ** attempt} 秒后重试...{Colors.ENDC}")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        
                    response.raise_for_status() 
                    
            except (ClientError, TimeoutError) as e:
                if attempt < self.retries - 1:
                    logging.warning(f"{Colors.WARNING}请求失败 (尝试 {attempt + 1}/{self.retries}): {e}。在 {2 ** attempt} 秒后重试...{Colors.ENDC}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logging.error(f"{Colors.FAIL}{self.retries} 次尝试全部失败 ({url}): {e}{Colors.ENDC}")
                    raise 
        
        return None

async def display_api_docs(client: AsyncHTTPClient):
    """获取并显示 API 文档。"""
    url = f"{SANIC_API_SERVER_URL}/docs"
    logging.info(f"{Colors.BLUE}正在从 {url} 获取 API 文档...{Colors.ENDC}")
    try:
        data = await client.request('GET', url)
        print(f"\n{Colors.HEADER}{Colors.BOLD}--- API 文档 ---{Colors.ENDC}")
        if isinstance(data, dict):
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(data)
        print(f"{Colors.HEADER}{Colors.BOLD}------------------{Colors.ENDC}\n")
    except Exception as e:
        logging.error(f"{Colors.FAIL}获取 API 文档失败: {e}{Colors.ENDC}")

async def perform_health_check(client: AsyncHTTPClient):
    """对 API 执行健康检查。"""
    url = f"{SANIC_API_SERVER_URL}/health_check"
    logging.info(f"{Colors.BLUE}正在对 {url} 执行健康检查...{Colors.ENDC}")
    try:
        data = await client.request('GET', url)
        print(f"\n{Colors.HEADER}{Colors.BOLD}--- 健康检查结果 ---{Colors.ENDC}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"{Colors.WARNING}健康检查未收到数据。{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}--------------------{Colors.ENDC}\n")
    except Exception as e:
        logging.error(f"{Colors.FAIL}健康检查失败: {e}{Colors.ENDC}")

async def create_new_knowledge_base(client: AsyncHTTPClient):
    """提示详细信息并创建新的知识库。"""
    logging.info(f"{Colors.BLUE}正在启动新知识库创建流程...{Colors.ENDC}")
    user_id = input(f"{Colors.CYAN}请输入用户ID (例如: user_001, 默认: abc1234): {Colors.ENDC}") or "abc1234"
    user_info = input(f"{Colors.CYAN}请输入用户信息 (可选, 例如: A部门, 默认: 5678): {Colors.ENDC}") or "5678"
    kb_name = input(f"{Colors.CYAN}请输入新知识库名称 (必填): {Colors.ENDC}")
    
    if not kb_name:
        logging.warning(f"{Colors.WARNING}知识库名称不能为空。操作中止。{Colors.ENDC}")
        return

    payload = {
        'user_id': user_id, 
        'user_info': user_info, 
        'kb_name': kb_name
    }
    url = f"{SANIC_API_SERVER_URL}/qa_handler/new_knowledge_base"
    logging.info(f"{Colors.BLUE}正在发送请求以创建知识库 '{kb_name}'...{Colors.ENDC}")
    try:
        data = await client.request(
            'POST',
            url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"\n{Colors.HEADER}{Colors.BOLD}--- 新知识库创建响应 ---{Colors.ENDC}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"{Colors.HEADER}{Colors.BOLD}------------------------{Colors.ENDC}\n")
    except Exception as e:
        logging.error(f"{Colors.FAIL}创建知识库失败: {e}{Colors.ENDC}")

async def upload_files_to_kb(client: AsyncHTTPClient):
    """提示文件路径和知识库详情以上传文档。"""
    logging.info(f"{Colors.BLUE}正在启动文件上传到知识库流程...{Colors.ENDC}")
    file_path = input(f"{Colors.CYAN}请输入要上传文件的绝对路径 (例如: /path/to/my_document.pdf): {Colors.ENDC}")
    
    if not os.path.exists(file_path):
        logging.error(f"{Colors.FAIL}错误: 文件 '{file_path}' 不存在。请提供有效的文件路径。{Colors.ENDC}")
        return
    if not os.path.isfile(file_path):
        logging.error(f"{Colors.FAIL}错误: 路径 '{file_path}' 不是文件。请提供有效的文件路径。{Colors.ENDC}")
        return
    
    user_id = input(f"{Colors.CYAN}请输入用户ID (例如: user_001, 默认: abc1234): {Colors.ENDC}") or "abc1234"
    user_info = input(f"{Colors.CYAN}请输入用户信息 (可选, 默认: 5678): {Colors.ENDC}") or "5678"
    kb_id = input(f"{Colors.CYAN}请输入目标知识库ID (必填, 例如: KBxxxxxxxxxxxxxxxxxxxxxxxxxxxxx): {Colors.ENDC}")
    mode = input(f"{Colors.CYAN}请输入处理模式 (soft/strict, 默认: soft): {Colors.ENDC}") or "soft"

    if not kb_id:
        logging.warning(f"{Colors.WARNING}知识库ID不能为空。中止文件上传。{Colors.ENDC}")
        return

    f = None
    try:
        logging.info(f"{Colors.BLUE}正在上传文件 '{os.path.basename(file_path)}' 到知识库 '{kb_id}'...{Colors.ENDC}")
        upload_client = AsyncHTTPClient(retries=1, timeout=300) 
        async with upload_client:
            form_data = aiohttp.FormData()
            f = open(file_path, 'rb')
            form_data.add_field('files',
                                 f,
                                 filename=os.path.basename(file_path),
                                 content_type='application/octet-stream')
            
            form_data.add_field('user_id', user_id)
            form_data.add_field('user_info', user_info)
            form_data.add_field('kb_id', kb_id)
            form_data.add_field('mode', mode)

            url = f"{SANIC_API_SERVER_URL}/qa_handler/upload_files"
            data = await upload_client.request(
                'POST',
                url,
                data=form_data
            )
            print(f"\n{Colors.HEADER}{Colors.BOLD}--- 文件上传响应 ---{Colors.ENDC}")
            if data:
                print(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"{Colors.HEADER}{Colors.BOLD}--------------------{Colors.ENDC}\n")
    except Exception as e:
        logging.error(f"{Colors.FAIL}文件上传失败: {e}{Colors.ENDC}")
    finally:
        if f:
            f.close()

async def conduct_local_doc_chat(client: AsyncHTTPClient):
    """启动与指定知识库中文档的聊天会话。"""
    logging.info(f"{Colors.BLUE}正在启动本地文档聊天会话...{Colors.ENDC}")
    user_id = input(f"{Colors.CYAN}请输入用户ID (例如: user_001, 默认: abc1234): {Colors.ENDC}") or "abc1234"
    user_info = input(f"{Colors.CYAN}请输入用户信息 (可选, 默认: 5678): {Colors.ENDC}") or "5678"
    kb_ids_str = input(f"{Colors.CYAN}请输入用逗号分隔的知识库ID (必填, 例如: KB1,KB2): {Colors.ENDC}")
    kb_ids = [kb_id.strip() for kb_id in kb_ids_str.split(',') if kb_id.strip()]

    if not kb_ids:
        logging.warning(f"{Colors.WARNING}聊天至少需要一个知识库ID。中止操作。{Colors.ENDC}")
        return

    question = input(f"{Colors.CYAN}请输入您的问题 (必填): {Colors.ENDC}")
    if not question:
        logging.warning(f"{Colors.WARNING}问题不能为空。中止聊天。{Colors.ENDC}")
        return

    streaming_input = input(f"{Colors.CYAN}是否启用流式响应? (yes/no, 默认: no): {Colors.ENDC}").lower()
    streaming = streaming_input == 'yes' 

    history_str = input(f"{Colors.CYAN}请输入聊天历史记录，格式为 JSON 数组 (例如: [[\"你好\", \"您好\"]], 留空表示无): {Colors.ENDC}")
    history = []
    if history_str:
        try:
            history = json.loads(history_str)
            if not isinstance(history, list) or not all(isinstance(item, list) and len(item) == 2 for item in history):
                raise ValueError("历史记录必须是 [问题, 回答] 对的列表。")
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"{Colors.WARNING}聊天历史格式无效: {e}。将忽略历史记录。{Colors.ENDC}")
            history = []
    
    max_token_input = input(f"{Colors.CYAN}响应最大 Token 数 (默认: 3000): {Colors.ENDC}")
    max_token = int(max_token_input) if max_token_input.isdigit() else 3000

    temperature_input = input(f"{Colors.CYAN}温度 (0.0-1.0, 默认: 0.7): {Colors.ENDC}")
    try:
        temperature = float(temperature_input)
    except ValueError:
        temperature = 0.7

    top_p_input = input(f"{Colors.CYAN}Top-p (0.0-1.0, 默认: 0.99): {Colors.ENDC}")
    try:
        top_p = float(top_p_input)
    except ValueError:
        top_p = 0.99
        
    top_k_input = input(f"{Colors.CYAN}检索 Top-k (默认: 5): {Colors.ENDC}")
    top_k = int(top_k_input) if top_k_input.isdigit() else 5

    rerank_input = input(f"{Colors.CYAN}是否启用重排 (rerank)? (yes/no, 默认: yes): {Colors.ENDC}").lower()
    rerank = rerank_input != 'no'

    payload = {
        "user_id": user_id,
        "max_token": max_token,
        "user_info": user_info,
        "kb_ids": kb_ids,
        "question": question,
        "history": history,
        "streaming": streaming, 
        "rerank": rerank,
        "custom_prompt": None,
        "api_base": DEFAULT_API_BASE, # 这里的 DEFAULT_API_BASE 是 LLM 地址
        "api_key": DEFAULT_API_KEY,   # LLM API 密钥
        "api_context_length": 10000, 
        "top_p": top_p,
        "temperature": temperature,
        "top_k": top_k
    }

    url = f"{SANIC_API_SERVER_URL}/local_doc_qa/local_doc_chat" # 完整的 Sanic API URL
    logging.info(f"{Colors.BLUE}正在发送聊天请求到 {url}...{Colors.ENDC}")
    try:
        chat_client = AsyncHTTPClient(retries=3, timeout=300) 
        async with chat_client:
            response_data = await chat_client.request(
                method="POST",
                url=url, 
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            print(f"\n{Colors.HEADER}{Colors.BOLD}--- 聊天响应 ---{Colors.ENDC}")
            if response_data:
                # 提取并打印 "response" 字段中的实际回答
                if isinstance(response_data, dict) and "response" in response_data:
                    raw_response_content = response_data["response"]
                    # 检查是否包含 "data: {"answer": "..."}" 这种格式
                    if isinstance(raw_response_content, str) and raw_response_content.startswith("data: "):
                        try:
                            # 尝试解析内部的JSON
                            inner_json_str = raw_response_content[len("data: "):]
                            inner_data = json.loads(inner_json_str)
                            if "answer" in inner_data:
                                print(f"{Colors.GREEN}{inner_data['answer']}{Colors.ENDC}") # 只打印答案
                            else:
                                print(f"{Colors.WARNING}未找到 'answer' 字段，打印原始 response 内容：{Colors.ENDC}")
                                print(f"{Colors.GREEN}{raw_response_content}{Colors.ENDC}")
                        except json.JSONDecodeError:
                            print(f"{Colors.WARNING}无法解析 response 字段内的 JSON，打印原始 response 内容：{Colors.ENDC}")
                            print(f"{Colors.GREEN}{raw_response_content}{Colors.ENDC}")
                    else:
                        # 如果不是这种特定格式，直接打印 response 字段内容
                        print(f"{Colors.GREEN}{raw_response_content}{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}聊天API未收到有效响应数据。{Colors.ENDC}")
                    print(json.dumps(response_data, indent=2, ensure_ascii=False)) # 打印完整的原始数据以供调试
            else:
                print(f"{Colors.WARNING}聊天API未收到响应。{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}----------------{Colors.ENDC}\n")

    except Exception as e:
        logging.error(f"{Colors.FAIL}聊天请求失败: {e}{Colors.ENDC}")

async def list_knowledge_bases(client: AsyncHTTPClient):
    """提示用户ID和用户信息，并列出该用户的所有知识库。"""
    logging.info(f"{Colors.BLUE}正在启动列出知识库流程...{Colors.ENDC}")
    user_id = input(f"{Colors.CYAN}请输入用户ID (例如: user_001, 默认: abc1234): {Colors.ENDC}") or "abc1234"
    user_info = input(f"{Colors.CYAN}请输入用户信息 (可选, 默认: 5678): {Colors.ENDC}") or "5678"

    payload = {
        "user_id": user_id,
        "user_info": user_info
    }
    url = f"{SANIC_API_SERVER_URL}/qa_handler/list_knowledge_base" 
    logging.info(f"{Colors.BLUE}正在发送请求以列出用户 '{user_id}' 的知识库...{Colors.ENDC}")
    try:
        data = await client.request(
            'POST',
            url, 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"\n{Colors.HEADER}{Colors.BOLD}--- 知识库列表响应 ---{Colors.ENDC}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"{Colors.WARNING}列出知识库API未收到响应。{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}----------------------{Colors.ENDC}\n")
    except Exception as e:
        logging.error(f"{Colors.FAIL}列出知识库失败: {e}{Colors.ENDC}")


async def main():
    """RAG 系统客户端的主交互循环。"""
    while True:
        client = AsyncHTTPClient(retries=3, timeout=10) 
        
        print(f"\n{Colors.BOLD}{Colors.HEADER}--- RAG 系统客户端菜单 ---{Colors.ENDC}")
        print(f"{Colors.GREEN}1. 获取 API 文档{Colors.ENDC}")
        print(f"{Colors.GREEN}2. 执行健康检查{Colors.ENDC}")
        print(f"{Colors.GREEN}3. 创建新知识库{Colors.ENDC}")
        print(f"{Colors.GREEN}4. 上传文件到知识库{Colors.ENDC}")
        print(f"{Colors.GREEN}5. 开始文档聊天{Colors.ENDC}")
        print(f"{Colors.GREEN}6. 列出知识库{Colors.ENDC}") 
        print(f"{Colors.FAIL}7. 退出{Colors.ENDC}") 
        print(f"{Colors.BOLD}{Colors.HEADER}--------------------------{Colors.ENDC}\n")

        choice = input(f"{Colors.CYAN}请输入您的选择 (1-7): {Colors.ENDC}") 

        try:
            if choice == '1':
                async with client:
                    await display_api_docs(client)
            elif choice == '2':
                async with client:
                    await perform_health_check(client)
            elif choice == '3':
                async with client:
                    await create_new_knowledge_base(client)
            elif choice == '4':
                await upload_files_to_kb(client) 
            elif choice == '5':
                await conduct_local_doc_chat(client)
            elif choice == '6': 
                async with client:
                    await list_knowledge_bases(client)
            elif choice == '7': 
                print(f"{Colors.BOLD}{Colors.WARNING}正在退出 RAG 系统客户端。再见！{Colors.ENDC}")
                break
            else:
                print(f"{Colors.FAIL}无效的选择。请输入 1 到 7 之间的数字。{Colors.ENDC}")
        except Exception as e:
            logging.critical(f"{Colors.FAIL}操作期间发生未处理的错误: {Colors.ENDC}")
            logging.critical(f"{Colors.FAIL}错误详情: {e}{Colors.ENDC}")
        finally:
            if client.session and not client.session.closed:
                await client.session.close()

if __name__ == "__main__":
    if sys.platform.lower() == "win32" and os.getenv("TERM") != "xterm":
        try:
            import colorama
            colorama.init() 
        except ImportError:
            logging.warning(f"{Colors.WARNING}请安装 'colorama' 以在 Windows 上获得彩色输出: pip install colorama{Colors.ENDC}")
    
    asyncio.run(main())