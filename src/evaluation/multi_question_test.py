import aiohttp
import asyncio
import json
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Any

usage_guide = '''
参数说明
- `--file`: file文件路径 (默认: ./riscv-privileged.pdf)
- `--questions`: 测试问题JSON文件路径 (默认: ./test_questions.json)
- `--results`: 测试结果保存路径 (默认: ./test_results.json)
- `--api-base`: API服务器地址 (默认: http://127.0.0.1:8777)
- `--api-key`: API密钥 (默认: your_api_key)
- `--user-id`: 用户ID (默认: rx01)
- `--user-info`: 用户信息 (默认: rx for test)
- `--kb-name`: 知识库名称 (默认: TKB01)
- `--kb-id`: 知识库ID，如果提供则跳过创建知识库步骤

测试问题格式
JSON格式，包含问题字符串:
```json
[
  "RISC-V架构的特点是什么？",
  "RISC-V中的特权级别有哪些？",
  "什么是机器模式(Machine Mode)？"
]
```
'''

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(root_dir)

from src.configs.configs import DEFAULT_API_BASE, DEFAULT_API_KEY

class AsyncHTTPClient:
    """异步HTTP客户端，处理重试和超时"""
    def __init__(self, retries: int = 3, timeout: int = 60):
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
    ) -> Optional[dict]:
        """发送HTTP请求，处理重试和错误"""
        for attempt in range(self.retries):
            try:
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
                            # 尝试智能处理
                            text = await response.text()
                            try:
                                # 尝试解析为JSON
                                import json
                                return json.loads(text)
                            except json.JSONDecodeError:
                                # 如果不是JSON，返回原文本
                                return text
                    
                    # 服务器错误时重试
                    if response.status >= 500:
                        if attempt < self.retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                            
                    response.raise_for_status()
                    
            except Exception as e:
                if attempt < self.retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logging.error(f"All attempts failed for {url}: {str(e)}")
                    raise
        
        return None

class SystemTest:
    """系统测试类，处理知识库创建、文件上传和问答测试"""
    def __init__(self, api_base: str = DEFAULT_API_BASE, api_key: str = DEFAULT_API_KEY, 
                 user_id: str = 'rx01', user_info: str = '12345678', 
                 kb_name: str = 'TKB01', kb_id: str = None):
        self.url_base = "http://127.0.0.1:8777"
        self.api_base = DEFAULT_API_BASE
        self.api_key = DEFAULT_API_KEY
        self.user_id = user_id
        self.user_info = user_info
        self.kb_name = kb_name
        self.kb_id = kb_id
        self.test_results = []
        
    async def create_knowledge_base(self) -> str:
        """创建新的知识库并返回kb_id"""
        logger.info(f"Creating knowledge base: {self.kb_name}")
        
        async with AsyncHTTPClient(retries=3, timeout=30) as client:
            try:
                data = await client.request(
                    'POST',
                    f'{self.url_base}/api/qa_handler/new_knowledge_base',
                    json={
                        'user_id': self.user_id, 
                        'user_info': self.user_info, 
                        'kb_name': self.kb_name
                    },
                    headers={'Content-Type': 'application/json'}
                )
                
                if isinstance(data, dict) and data.get('code') == 200:
                    self.kb_id = data.get('data', {}).get('kb_id')
                    logger.info(f"Knowledge base created with ID: {self.kb_id}")
                    return self.kb_id
                else:
                    logger.error(f"Failed to create knowledge base: {data}")
                    raise Exception(f"Failed to create knowledge base: {data}")
                    
            except Exception as e:
                logger.error(f"Error creating knowledge base: {str(e)}")
                raise
    
    async def upload_file(self, file_path: str) -> bool:
        """上传文件到知识库"""
        if not self.kb_id:
            raise ValueError("Knowledge base ID not set. Create knowledge base first.")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Uploading file: {file_path} to knowledge base: {self.kb_id}")
        
        # 不使用 AsyncHTTPClient 类的 request 方法，因为需要为每次重试创建新的 FormData
        session = aiohttp.ClientSession()
        try:
            retries = 3
            for attempt in range(retries):
                try:
                    # 每次重试都创建新的 FormData 对象
                    form_data = aiohttp.FormData()
                    with open(file_path, 'rb') as f:
                        form_data.add_field('files',
                                        f.read(),  # 读取文件内容而不是文件对象
                                        filename=os.path.basename(file_path),
                                        content_type='application/octet-stream')
                    
                    # 添加其他字段
                    form_data.add_field('user_id', self.user_id)
                    form_data.add_field('user_info', self.user_info)
                    form_data.add_field('kb_id', self.kb_id)
                    form_data.add_field('mode', 'soft')

                    # 发送请求
                    async with session.post(
                        f'{self.url_base}/api/qa_handler/upload_files',
                        data=form_data,
                        timeout=300
                    ) as response:
                        if response.status in (200, 201):
                            data = await response.json()
                            logger.info(f"File uploaded successfully: {data}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to upload file (HTTP {response.status}): {error_text}")
                            if attempt < retries - 1:
                                await asyncio.sleep(2 ** attempt)
                            else:
                                return False
                                
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Error uploading file: {str(e)}")
                        return False
                        
        finally:
            await session.close()
        
        return False
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """向知识库提问并获取回答"""
        if not self.kb_id:
            raise ValueError("Knowledge base ID not set. Create knowledge base first.")
        
        logger.info(f"Asking question: {question}")
        
        async with AsyncHTTPClient(retries=3, timeout=60) as client:
            try:
                # 请求参数
                payload = {
                    "user_id": self.user_id,
                    "max_token": 3000,
                    "user_info": self.user_info,
                    "kb_ids": [self.kb_id],
                    "question": question,
                    "history": [],
                    "streaming": False,
                    "rerank": True,
                    "custom_prompt": None,
                    "api_base": self.api_base,
                    "api_key": self.api_key,
                    "api_context_length": 10000,
                    "top_p": 0.99,
                    "temperature": 0.7,
                    "top_k": 5
                }

                # 发送POST请求
                response = await client.request(
                    method="POST",
                    url=f"{self.url_base}/api/local_doc_qa/local_doc_chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info(f"Response received for question: {question}")
                
                return {
                    "question": question,
                    "response": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Error asking question: {str(e)}")
                return {
                    "question": question,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
    
    async def run_test_questions(self, questions_file: str, results_file: str) -> None:
        """从JSON文件读取问题并保存结果"""
        if not os.path.exists(questions_file):
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            if not isinstance(questions, list):
                raise ValueError("Questions file should contain a JSON array of questions")
                
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")
            
            for question in questions:
                if isinstance(question, str):
                    result = await self.ask_question(question)
                    self.test_results.append(result)
                else:
                    logger.warning(f"Skipping non-string question: {question}")
            
            # 保存结果
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error running test questions: {str(e)}")
            raise
    
    async def run_full_test(self, file_path: str, questions_file: str, results_file: str) -> None:
        """运行完整的系统测试流程"""
        try:
            # 1. 创建知识库（如果kb_id未提供）
            if not self.kb_id:
                await self.create_knowledge_base()
            else:
                logger.info(f"Using existing knowledge base with ID: {self.kb_id}")
            
            # 2. 上传文件
            upload_success = await self.upload_file(file_path)
            if not upload_success:
                raise Exception("Failed to upload file")
            
            # 等待一段时间，确保文件处理完成
            logger.info("Waiting for file processing to complete...")
            await asyncio.sleep(10)
            
            # 3. 运行测试问题
            await self.run_test_questions(questions_file, results_file)
            
            logger.info("Full system test completed successfully")
            
        except Exception as e:
            logger.error(f"System test failed: {str(e)}")
            raise

async def main():
    """主函数，处理命令行参数并执行测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run system test for knowledge base QA')
    parser.add_argument('--file', type=str, default='./riscv-privileged.pdf',
                        help='Path to file')
    parser.add_argument('--questions', type=str, default='./test_questions.json',
                        help='Path to questions JSON file')
    parser.add_argument('--results', type=str, default='./test_results.json',
                        help='Path to save results JSON file')
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE,
                        help='API base URL')
    parser.add_argument('--api-key', type=str, default=DEFAULT_API_KEY,
                        help='API key')
    parser.add_argument('--user-id', type=str, default='rx01',
                        help='User ID')
    parser.add_argument('--user-info', type=str, default='12345678',
                        help='User info')
    parser.add_argument('--kb-name', type=str, default='TKB01',
                        help='Knowledge base name')
    parser.add_argument('--kb-id', type=str, default=None,
                        help='Knowledge base ID (if provided, skip creating knowledge base)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        return 1
    
    # 检查问题文件是否存在
    if not os.path.exists(args.questions):
        print(f"Error: Questions file not found at {args.questions}")
        return 1
    
    # 创建并运行系统测试
    test = SystemTest(
        api_base=args.api_base, 
        api_key=args.api_key,
        user_id=args.user_id,
        user_info=args.user_info,
        kb_name=args.kb_name,
        kb_id=args.kb_id
    )
    await test.run_full_test(args.file, args.questions, args.results)
    
    return 0

if __name__ == "__main__":
    try:
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('system_test.log')
            ]
        )
        logger = logging.getLogger('system_test')
        
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)