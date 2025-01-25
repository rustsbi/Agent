import aiohttp
import asyncio
from typing import Optional
import logging
from aiohttp import ClientError
from asyncio import TimeoutError
import os

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
    ) -> Optional[dict]:
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
                            return await response.text()  # 返回文本
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
                    
            except (ClientError, TimeoutError) as e:
                if attempt < self.retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logging.error(f"All attempts failed for {url}: {str(e)}")
                    raise
        
        return None
    
async def test_document():
    async with AsyncHTTPClient(retries=3, timeout=10) as client:
        try:
            data = await client.request(
                'GET',
                'http://127.0.0.1:8777/api/docs'
            )
            print(data)
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")

async def test_health_check():
    async with AsyncHTTPClient(retries=3, timeout=10) as client:
        try:
            data = await client.request(
                'GET',
                'http://127.0.0.1:8777/api/health_check'
            )
            print(data)
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")

async def test_new_knowledge_base():
    async with AsyncHTTPClient(retries=3, timeout=10) as client:
        try:
            data = await client.request(
                'POST',
                'http://127.0.0.1:8777/api/qa_handler/new_knowledge_base',
                json={'user_id': 'abc1234', 'user_info': '5678', 'kb_name': 'zzh', },
                headers={'Content-Type': 'application/json'}
            )
            print(data)
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")

async def test_upload_files(file_path: str):
    async with AsyncHTTPClient(retries=1, timeout=10000) as client:
        try:
            # 准备文件和表单数据
            form_data = aiohttp.FormData()
            f = open(file_path, 'rb')
            # 添加文件
            form_data.add_field('files',
                                f,
                                filename=os.path.basename(file_path),
                                content_type='application/octet-stream')
            
            # 添加其他字段
            form_data.add_field('user_id', 'abc1234')
            form_data.add_field('user_info', '5678')
            form_data.add_field('kb_id', 'KBbf9488a498cf4407a6abdf477208c3ed')
            form_data.add_field('mode', 'soft')

            # 发送请求
            data = await client.request(
                'POST',
                'http://127.0.0.1:8777/api/qa_handler/upload_files',
                data=form_data
            )
            print(data)
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
        finally:
            # 确保关闭所有打开的文件
            f.close()



def run_test():    
    asyncio.run(test_document())
    asyncio.run(test_health_check())
    asyncio.run(test_new_knowledge_base())
    asyncio.run(test_upload_files('./这是一个测试文件.txt'))

asyncio.run(test_upload_files('./这是一个测试文件.txt'))


