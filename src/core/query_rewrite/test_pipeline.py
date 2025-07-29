#!/usr/bin/env python3
"""
测试QueryRewritePipeline的脚本
"""

import sys
import os
from pipeline import QueryRewritePipeline

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_pipeline():
    """测试QueryRewritePipeline"""
    print("=== QueryRewritePipeline 测试 ===\n")

    pipeline = QueryRewritePipeline()

    test_queries = [
        "如何在Rust中实现内存分配？",
        "怎么用Rust写一个加法函数？",
        "How to implement memory allocation in Rust?",
        "What is RAG and how does it work?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"--- 测试 {i} ---")
        print(f"原始问题: {query}")

        try:
            result = pipeline.process(query)
            print(f"语言检测: {result.get('original_lang', 'unknown')}")
            print(f"翻译结果: {result['translated']}")
            print(f"重写结果: {result['rewrites']}")
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")

        print()  # 空行分隔

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_pipeline()
 