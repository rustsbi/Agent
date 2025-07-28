#!/usr/bin/env python3
"""
测试query_rewrite在RAG流程中的集成
"""

import sys
import os
import json
import requests

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


def test_query_rewrite_integration():
    """测试query_rewrite集成到RAG流程"""

    # API端点
    url = "http://localhost:8777/api/local_doc_qa/local_doc_chat"

    # 测试数据
    test_cases = [
        {
            "name": "中文查询测试",
            "question": "如何在Rust中实现内存分配？",
            "expected_rewrite": "How to implement memory allocation in Rust?"
        },
        {
            "name": "英文查询测试",
            "question": "How to write a function in Rust?",
            "expected_rewrite": "How to write a function in Rust?"
        },
        {
            "name": "混合查询测试",
            "question": "Rust中的错误处理机制是什么？",
            "expected_rewrite": "What is the error handling mechanism in Rust?"
        }
    ]

    for test_case in test_cases:
        print(f"\n=== {test_case['name']} ===")
        print(f"原始查询: {test_case['question']}")
        print(f"期望重写: {test_case['expected_rewrite']}")

        # 构建请求数据
        payload = {
            "user_id": "test_user",
            "user_info": "1234",
            "kb_ids": [],  # 空知识库，只测试query_rewrite
            "question": test_case['question'],
            "streaming": False,
            "rerank": False,  # 关闭rerank，专注于测试query_rewrite
            "query_rewrite": True,  # 启用query_rewrite
            "custom_prompt": None,
            "model": "/home/model/Qwen2.5-7B-Instruct",
            "max_token": 2048,
            "temperature": 0.5,
            "top_p": 0.99,
            "top_k": 10,
            "api_base": "http://0.0.0.0:2333/v1",
            "api_key": "YOUR_API_KEY",
            "api_context_length": 4096,
            "hybrid_search": False,
            "chunk_size": 800
        }

        try:
            print("发送请求...")
            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print(f"响应状态: {result.get('code')}")
                print(f"响应消息: {result.get('msg')}")

                # 检查是否有condense_question字段（重写后的查询）
                if 'condense_question' in result:
                    actual_rewrite = result['condense_question']
                    print(f"实际重写: {actual_rewrite}")

                    # 检查时间记录中是否有query_rewrite
                    time_record = result.get('time_record', {})
                    if 'time_usage' in time_record:
                        time_usage = time_record['time_usage']
                        if 'query_rewrite' in time_usage:
                            print(
                                f"Query rewrite耗时: {time_usage['query_rewrite']}s")
                        else:
                            print("未找到query_rewrite时间记录")

                    # 简单验证
                    if actual_rewrite != test_case['question']:
                        print("✅ Query rewrite生效")
                    else:
                        print("⚠️ Query rewrite未生效（可能是英文查询无需翻译）")
                else:
                    print("❌ 响应中未找到condense_question字段")

            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")

        except Exception as e:
            print(f"❌ 测试异常: {str(e)}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_query_rewrite_integration()
