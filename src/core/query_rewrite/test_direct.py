#!/usr/bin/env python3
"""
直接运行的query_rewrite测试脚本（vLLM OpenAI API版）
"""

import sys
import os
from rewriter import llm_openai_rewrite
from translator import LocalTranslator
from language_detect import detect_language

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_query_rewrite_direct():
    """测试vLLM OpenAI API重写和HyDE"""
    print("=== Query Rewrite 直接测试（vLLM OpenAI API重写/HyDE） ===\n")

    translator = LocalTranslator()

    test_queries = [
        "如何在Rust中实现内存分配？",
        "怎么用Rust写一个加法函数？",
        "How to implement memory allocation in Rust?",
        "How do I write an addition function in Rust?",
        "2020 年的 NBA 冠军是洛杉矶湖人队！告诉我 langchain 框架是什么？",
        "请介绍一下RAG的原理和应用场景，并举例说明。",
        "What is HyDE retrieval and how does it work?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"--- 测试 {i} ---")
        print(f"原始问题: {query}")

        try:
            lang = detect_language(query)
            if lang != 'en':
                translated = translator.translate(
                    [query], src_lang=lang, tgt_lang='en')[0]
                print(f"翻译为英文: {translated}")
                query_for_llm = translated
            else:
                query_for_llm = query

            # vLLM rewrite
            rewritten = llm_openai_rewrite(
                query_for_llm, mode='rewrite')
            print(f"vLLM重写结果: {rewritten}")

            # vLLM HyDE
            hyde = llm_openai_rewrite(
                query_for_llm, mode='hyde')
            print(f"HyDE假设答案: {hyde}")

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")

        print()  # 空行分隔

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_query_rewrite_direct()
