import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)

from typing import List
from src.utils.log_handler import debug_logger
from src.utils.general_utils import get_time_async, get_time
from src.configs.configs import LOCAL_RERANK_BATCH,LOCAL_RERANK_SERVICE_URL
from langchain.schema import Document
import traceback
import aiohttp
import asyncio
import requests


class SBIRerank:
    def __init__(self):
        """初始化重排序客户端"""
        self.url = f"http://{LOCAL_RERANK_SERVICE_URL}/rerank"
        # 不支持异步的session
        self.session = requests.Session()

    async def _get_rerank_async(self, query: str, passages: List[str]) -> List[float]:
        """异步请求重排序服务"""
        data = {'query': query, 'passages': passages}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=data) as response:
                    return await response.json()
        except Exception as e:
            debug_logger.error(f'async rerank error: {traceback.format_exc()}')
            return [0.0] * len(passages)

    @get_time_async
    async def arerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """Embed search docs using async calls, maintaining the original order."""
        batch_size = LOCAL_RERANK_BATCH  # 增大客户端批处理大小
        all_scores = [0 for _ in range(len(source_documents))]
        passages = [doc.page_content for doc in source_documents]

        tasks = []
        for i in range(0, len(passages), batch_size):
            task = asyncio.create_task(self._get_rerank_async(query, passages[i:i + batch_size]))
            tasks.append((i, task))

        for start_index, task in tasks:
            res = await task
            if res is None:
                return source_documents
            all_scores[start_index:start_index + batch_size] = res
            print(res)

        for idx, score in enumerate(all_scores):
            source_documents[idx].metadata['score'] = round(float(score), 2)
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents



#使用示例
async def main():
    reranker = SBIRerank()
    query = "什么是人工智能"
    documents = [Document(page_content="阿爸巴sss啊啊啊啊s巴爸爸"), 
                    Document(page_content="AI技术在各领域广泛应用"),
                    Document(page_content="机器学习是AI的核心技术。"),
                    Document(page_content="人工智能是计算机科学的一个分支。")]  # 示例文档
    reranked_docs = await reranker.arerank_documents(query, documents)
    return reranked_docs


# 运行异步主函数
if __name__ == "__main__":
    reranked_docs = asyncio.run(main())
    print(reranked_docs)