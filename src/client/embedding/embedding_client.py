
import sys
import os
import time
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

async def test_async_methods():
    """测试异步方法"""
    embedder = SBIEmbeddings()
    
    # 测试单个文本的embedding
    debug_logger.info("\n测试异步单个文本embedding:")
    single_text = "人工智能正在改变我们的生活方式。"
    single_embedding = await embedder.aembed_query(single_text)
    debug_logger.info(f"文本: {single_text}")
    debug_logger.info(f"向量维度: {len(single_embedding)}")
    
    # 测试批量文本的embedding
    debug_logger.info("\n测试异步批量文本embedding:")
    texts = [
        "深度学习是人工智能的一个重要分支。",
        "自然语言处理技术正在不断进步。",
        "机器学习算法可以从数据中学习规律。"
    ]
    
    embeddings = await embedder.aembed_documents(texts)
    for text, embedding in zip(texts, embeddings):
        debug_logger.info(f"文本: {text}")
        debug_logger.info(f"向量维度: {len(embedding)}")


def test_sync_methods():
    """测试同步方法"""
    embedder = SBIEmbeddings()
    
    # 测试单个文本的embedding
    debug_logger.info("\n测试同步单个文本embedding:")
    single_text = "这是一个测试文本。"
    single_embedding = embedder.embed_query(single_text)
    debug_logger.info(f"文本: {single_text}")
    debug_logger.info(f"向量维度: {len(single_embedding)}")
    
    # 测试批量文本的embedding
    debug_logger.info("\n测试同步批量文本embedding:")
    texts = [
        "第一个测试文本",
        "第二个测试文本",
        "第三个测试文本"
    ]
    embeddings = embedder.embed_documents(texts)
    for text, embedding in zip(texts, embeddings):
        debug_logger.info(f"文本: {text}")
        debug_logger.info(f"向量维度: {len(embedding)}")


def test_error_handling():
    """测试错误处理"""
    embedder = SBIEmbeddings()
    
    debug_logger.info("\n测试错误处理:")
    # 测试空文本
    try:
        embedding = embedder.embed_query("")
        debug_logger.info("空文本处理成功")
    except Exception as e:
        debug_logger.error(f"空文本处理失败: {str(e)}")
    
    # 测试None值
    try:
        embedding = embedder.embed_documents([None])
        debug_logger.info("None值处理成功")
    except Exception as e:
        debug_logger.error(f"None值处理失败: {str(e)}")


async def performance_test():
    """性能测试"""
    embedder = SBIEmbeddings()
    
    debug_logger.info("\n执行性能测试:")
    # 准备测试数据
    test_sizes = [10, 50, 100]
    
    for size in test_sizes:
        texts = [f"这是第{i}个性能测试文本。" for i in range(size)]
        
        # 测试同步方法性能
        start_time = time.time()
        embeddings = embedder.embed_documents(texts)
        sync_time = time.time() - start_time
        debug_logger.info(f"同步处理 {size} 个文本耗时: {sync_time:.2f}秒")
        
        # 测试异步方法性能
        start_time = time.time()
        embeddings = await embedder.aembed_documents(texts)
        async_time = time.time() - start_time
        debug_logger.info(f"异步处理 {size} 个文本耗时: {async_time:.2f}秒")

def embed_user_input(user_input: str):
    """测试用户输入的文本嵌入"""
    embedder = SBIEmbeddings()
    
    # 对用户输入的文本进行预处理
    processed_input = _process_query(user_input)
    
    debug_logger.info("\n测试用户输入的嵌入:")
    debug_logger.info(f"用户输入: {user_input}")
    debug_logger.info(f"预处理后的输入: {processed_input}")
    
    try:
        # 使用同步方法获取嵌入向量
        embedding = embedder.embed_query(processed_input)
        debug_logger.info(f"嵌入向量维度: {len(embedding)}")
        debug_logger.info(f"嵌入向量: {embedding}")
    except Exception as e:
        debug_logger.error(f"嵌入过程中发生错误: {str(e)}")

    return embedding


async def main():
    """主测试函数"""
    debug_logger.info(f"开始embedding客户端测试...")
    
    try:
        # 测试异步方法
        await test_async_methods()
        
        # 测试同步方法
        test_sync_methods()
        
        # 测试错误处理
        test_error_handling()
        
        # 执行性能测试
        await performance_test()
    except Exception as e:
        debug_logger.error(f"测试过程中发生错误: {str(e)}")

    debug_logger.info("embedding客户端测试完成")


if __name__ == "__main__":
    asyncio.run(main())