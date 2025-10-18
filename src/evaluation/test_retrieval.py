from src.configs.configs import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL_NAME
import asyncio
import json
import logging
import os
import sys
from aiohttp import ClientSession
import csv
from datetime import datetime

# 配置日志


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("retrieval_test.log"),
            logging.StreamHandler()
        ]
    )


# 将项目根目录添加到sys.path
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(current_script_path)))
sys.path.append(root_dir)
timestamp = datetime.now().strftime("%m%d%H%M%S")
INPUT_FILE_NAME = "rust_rag_dataset_deepseek50"


async def test_retrieval(question, kb_ids, top_k=20):
    """
    测试RAG系统的检索能力，获取检索到的文档结果

    参数:
        question: 查询问题
        kb_ids: 知识库ID列表
        top_k: 检索的文档数量

    返回:
        dict: 包含检索结果的字典
    """
    logging.info(f"开始测试检索能力 - 问题: {question}")

    try:
        async with ClientSession() as session:
            # 构建请求参数
            payload = {
                # "user_id": "abc1234",
                "user_id": "Qwen3",
                "user_info": "5678",
                "max_token": 3000,
                "kb_ids" : ["KB2ed627becda34af0a85cb1d104d90ebb"],  # abc
                # "kb_ids" = ["KBae355fdbaade40a4a479c17d752ec0e0"] # Qwen3
                "question": question,
                "history": [],  # 对话历史
                "streaming": False,  # 设置为False以获取完整结果
                "rerank": True,  # 启用重排序
                "custom_prompt": None,
                "model": DEFAULT_MODEL_NAME,
                "api_base": DEFAULT_API_BASE,
                "api_key": DEFAULT_API_KEY,
                "api_context_length": 10000,
                "top_p": 0.99,
                "temperature": 0.7,
                "top_k": top_k,  # 设置返回的文档数量
                "only_need_search_results": True,  # 关键参数：只返回检索结果
                "hybrid_search": False,  # 是否使用混合检索
                "need_web_search": False  # 是否需要网络搜索
            }

            # 发送POST请求
            async with session.post(
                "http://127.0.0.1:8777/api/local_doc_qa/local_doc_chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            ) as response:
                response_data = await response.json()

                if response.status == 200 and response_data.get('code') == 200:
                    logging.info(
                        f"检索成功，获取到 {len(response_data.get('source_documents', []))} 个文档")

                    # 保存结果到文件
                    # timestamp = asyncio.get_event_loop().time()
                    # filename = f"retrieval_result_{int(timestamp)}.json"
                    # with open(filename, 'w', encoding='utf-8') as f:
                    #     json.dump(response_data, f, ensure_ascii=False, indent=2)
                    # logging.info(f"检索结果已保存到: {filename}")

                    return response_data
                else:
                    error_msg = f"检索失败: HTTP {response.status}, 响应: {response_data}"
                    logging.error(error_msg)
                    return {"error": error_msg}

    except Exception as e:
        error_msg = f"请求过程中发生异常: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}


async def batch_test_retrieval(test_cases, kb_ids):
    """
    批量测试多个查询的检索能力
    """
    results = []
    for i, test_case in enumerate(test_cases):
        question = test_case.get('question', '')
        expected = test_case.get('context', '')

        logging.info(f"=== 测试用例 {i+1}/{len(test_cases)} ===")
        print(f"=== 测试用例 {i+1}/{len(test_cases)} ===")
        print(f"问题: {question}")
        print(f"预期上下文: {expected}")
        result = await test_retrieval(question, kb_ids)

        # 添加测试用例信息
        result['test_case_index'] = i+1
        result['question'] = question
        result['expected'] = expected

        results.append(result)

        # 测试用例之间添加间隔
        await asyncio.sleep(0.5)
        if i >= 10:
            break

    # 保存所有结果

    batch_filename = f"{INPUT_FILE_NAME}_result_{timestamp}.json"
    with open(batch_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"批量检索结果已保存到: {batch_filename}")

    return results


async def main():
    setup_logging()

    # 示例知识库ID，需要替换为实际存在的知识库

    # 单个测试用例
    # result = await test_retrieval("什么是RAG系统？", kb_ids)
    # print("\n检索结果:")
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    # 批量测试用例
    # test_cases = [
    #     {"question": "什么是RAG系统？", "expected": "RAG系统相关的文档"},
    #     {"question": "如何提高RAG系统的检索准确率？", "expected": "RAG系统优化相关的文档"},
    #     {"question": "什么是向量数据库？", "expected": "向量数据库相关的文档"}
    # ]
    test_cases = []

    with open(f'{INPUT_FILE_NAME}.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(
                {"question": row['question'],  "context": row['context']})

    await batch_test_retrieval(test_cases, kb_ids)

if __name__ == "__main__":
    asyncio.run(main())
