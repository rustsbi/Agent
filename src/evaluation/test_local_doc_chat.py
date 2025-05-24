import aiohttp
import asyncio
from typing import Optional
import logging
from aiohttp import ClientError
from asyncio import TimeoutError
import os
import sys

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(root_dir)
from src.configs.configs import DEFAULT_API_BASE, DEFAULT_API_KEY
from src.server.api_server.api_client import AsyncHTTPClient

async def local_doc_chat(question):
    async with AsyncHTTPClient(retries=3, timeout=30) as client:
        try:
            # 模拟请求参数
            payload = {
                "user_id": "abc1234",
                "max_token": 3000,
                "user_info": "5678",
                "kb_ids": ["KB2ed627becda34af0a85cb1d104d90ebb"],  # 替换为实际的知识库ID
                "question": question,
                "history": [],
                "streaming": False,  # 设置为False以获取完整回答
                "rerank": True,
                "custom_prompt": None,
                "api_base": DEFAULT_API_BASE,  # 替换为实际API地址
                "api_key": DEFAULT_API_KEY,  # 替换为实际API密钥
                # "api_base": "https://api.siliconflow.cn/v1",
                # "api_key": "sk-fwkzrhyjznwtubvqhthwklzcekkfemebypavdvtehsprtjni",
                "api_context_length": 10000,
                "top_p": 0.99,
                "temperature": 0.7,
                "top_k": 5
            }

            # 发送POST请求
            response = await client.request(
                method="POST",
                url="http://127.0.0.1:8777/api/local_doc_qa/local_doc_chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # 打印返回结果
            print(response)
            print('\n')
            print(response['question'])
            print(response['response'])

        except Exception as e:
            logging.error(f"Request to local_doc_chat failed: {str(e)}")

question = "用简洁的语言对文段进行总结和概括, 并给出文段的主题, 要求字数在100字以内"
question = "概括一下文段的主要内容"
question = "采莲赋的内容是什么"
question = "'莲南塘秋，莲花过人头；低头弄莲子，莲子清如水。'这首词出自哪里 "
question = "这段文字描述的是哪个季节的场景？"
# question = "文中提到的“妖童媛女”在做什么？"
asyncio.run(local_doc_chat(question))