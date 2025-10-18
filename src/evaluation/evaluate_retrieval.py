#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动粗标脚本：对检索结果调用 LLM 打分 0/1
输入：json 格式结果文件（可含多条 query）
输出：带 llm_score 的新 json 文件
"""
import json
import os
import requests
from tqdm import tqdm
from datetime import datetime

# ===== 1. 配置区 =====
INPUT_FILE  = "rust_rag_dataset_deepseek50_result_0927215213.json"      # 你的原始结果
OUTPUT_FILE = f"retrieve_results_scored_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json"
API_KEY     = ""
URL         = "https://api.chatanywhere.tech/v1/chat/completions"
MODEL       = "gpt-5-mini"
# =====================

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYS_PROMPT = (
    "You are a Rust language expert. "
    "Given a user question and a document paragraph, output ONLY 1 if the paragraph can help answer the question or the paragraph is related to the question; "
    "otherwise output 0. Do not explain."
)

def llm_judge(question: str, doc_content: str) -> int:
    """返回 0 或 1"""
    doc_clip = doc_content[:1500]          # 防超长
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Question: {question}\nDocument: {doc_clip}"}
        ]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    if resp.status_code != 200:
        print("API error:", resp.text)
        return 0
    try:
        return int(resp.json()["choices"][0]["message"]["content"].strip()[0])
    except Exception as e:
        print("Parse error:", e)
        return 0

def main():
    data = json.load(open(INPUT_FILE, "r", encoding="utf-8"))
    # 检查data是否为列表，如果不是则包装成列表
    if isinstance(data, dict):
        data = [data]
    err_cnt = 0
    
    for item in tqdm(data, desc="LLM judging"):
        q = item["question"]
        if not item.get("source_documents"):
            err_cnt += 1
            continue
        for doc in item["source_documents"]:
            score = llm_judge(q, doc["content"])
            doc["llm_score"] = score          # 新增字段
    
    # 如果原始数据是单一对象，保存时也保存为单一对象
    if len(data) == 1 and not isinstance(json.load(open(INPUT_FILE, "r", encoding="utf-8")), list):
        json.dump(data[0], open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        json.dump(data, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        
    print(f"Scored results saved -> {OUTPUT_FILE}")
    print(f"Total errors: {err_cnt}")

if __name__ == "__main__":
    main()