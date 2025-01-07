
import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
from typing import List
from src.utils.log_handler import debug_logger, embed_logger
from src.utils.general_utils import get_time_async, get_time
from langchain_core.embeddings import Embeddings
from src.configs.configs import LOCAL_EMBED_SERVICE_URL, LOCAL_RERANK_BATCH
import traceback
import aiohttp
import asyncio
import requests

# 清除多余换行以及以![figure]和![equation]起始的行
def _process_query(query):
    return '\n'.join([line for line in query.split('\n') if
                      not line.strip().startswith('![figure]') and
                      not line.strip().startswith('![equation]')])


class SBIEmbeddings(Embeddings):
    # 初始化请求embedding服务的url
    def __init__(self):
        self.model_version = 'local_v20250107'
        self.url = f"http://{LOCAL_EMBED_SERVICE_URL}/embedding"
        self.session = requests.Session()
        super().__init__()
    # 异步向embedding服务请求获取文本的向量
    async def _get_embedding_async(self, session, texts):
        # 去除多余换行和特殊标记
        data = {'texts': [_process_query(text) for text in texts]}
        async with session.post(self.url, json=data) as response:
            return await response.json()

    @get_time_async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # 设置批量大小
        batch_size = LOCAL_RERANK_BATCH 
        # 向上取整
        embed_logger.info(f'embedding texts number: {len(texts) / batch_size}')
        all_embeddings = []
        # 分批请求获取文本向量
        async with aiohttp.ClientSession() as session:
            tasks = [self._get_embedding_async(session, texts[i:i + batch_size])
                     for i in range(0, len(texts), batch_size)]
            # 收集所有任务结果，
            # asyncio.gather 的一个重要特性是：虽然任务是并发执行的，但返回结果时会保持跟任务列表相同的顺序。
            # 即使后面的批次先处理完，最终 results 中的顺序仍然与 tasks 列表的顺序一致。
            results = await asyncio.gather(*tasks)
            # 合并所有任务结果
            for result in results:
                all_embeddings.extend(result)
        debug_logger.info(f'success embedding number: {len(all_embeddings)}')
        # 返回结果
        return all_embeddings
    # 专门用于处理单个查询文本。将单个text转换为列表，因为是单个所以只取第一条embedding向量
    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]
    # 同步方法
    def _get_embedding_sync(self, texts):
        # 为什么同步去除，异步没去除标记啊，我先都给加上
        data = {'texts': [_process_query(text) for text in texts]}
        try:
            response = self.session.post(self.url, json=data)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            debug_logger.error(f'sync embedding error: {traceback.format_exc()}')
            return None

    # @get_time
    # 同步方法，列表请求
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embedding_sync(texts)

    @get_time
    #同步方法，单个请求
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # return self._get_embedding([text])['embeddings'][0]
        return self._get_embedding_sync([text])[0]

    @property
    def embed_version(self):
        return self.model_version

# 使用示例
async def main():
    embedder = SBIEmbeddings()
    texts = ["text1"]  # 示例文本
    embeddings = await embedder.aembed_documents(texts)
    return embeddings

if __name__ == '__main__':
    embeddings = asyncio.run(main())
    for embed in embeddings:
        print(len(embed))