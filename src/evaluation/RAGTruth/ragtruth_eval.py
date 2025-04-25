import json
import re
import sys
import os
import requests
import time
import traceback
import asyncio
from typing import List, Tuple
# 获取当前脚本的绝对路径
current_dir_path = os.path.dirname(os.path.abspath(__file__))
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
from src.utils.general_utils import my_print
import traceback
from openai import OpenAI
from typing import AsyncGenerator, List, Optional
import json
from src.client.llm.base import AnswerResult
from src.utils.log_handler import debug_logger
import tiktoken
from src.configs.configs import DEFAULT_PARENT_CHUNK_SIZE, \
    MAX_CHARS, VECTOR_SEARCH_TOP_K, DEFAULT_API_BASE, DEFAULT_API_KEY,\
          DEFAULT_API_CONTEXT_LENGTH, DEFAULT_MODEL_PATH, DEFAULT_MODEL_NAME
from src.client.llm.llm_client import OpenAILLM

llm = OpenAILLM(DEFAULT_MODEL_PATH, 8000,DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_API_CONTEXT_LENGTH, 0.99, 0.7 )

MAX_rounds = 100
streaming = False
source_info_path = os.path.join(current_dir_path, "dataset", "source_info.jsonl")
OLLAMA_MODEL_NAME = "deepseek-r1:8b"
OLLAMA_API_BASE = "http://localhost:11434/api/generate"

async def build_response(file_path):
    documents = []
    
    QA_cnt, sum_cnt, d2t_cnt = 0, 0, 0
    response_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"Processing line {i}...")
            response = {}
            data = json.loads(line)
            task_type = data['task_type']
            if task_type == "QA":
                if QA_cnt > MAX_rounds:
                    continue
                source_info = data['source_info']
                question = source_info['question']
                QA_cnt += 1 
            elif task_type == 'Summary':
                if sum_cnt > MAX_rounds:
                    continue
                source_info = data['source_info']
                sum_cnt += 1
            elif task_type == 'Data2txt':
                if d2t_cnt > MAX_rounds:
                    continue
                source_info = data['source_info']
                d2t_cnt += 1
            prompt = data['prompt']
            
            final_result = ""
            chat_history = []
            async for answer_result in llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
                resp = answer_result.llm_output["answer"]
                # resp结构，是一个str   data: {"answer": "良，男，汉族，1964年8月生"}
                if "DONE" not in resp:
                    final_result += json.loads(resp[6:])["answer"]
                debug_logger.info(resp)
                print("source_info:\n")
                print(source_info)
                print()
                print("final_result:\n")
                print(final_result)
                print('-'*20 + '\n' * 1)
                break # 加这个break是为了防止重复输出相同的结果。
            response['task_type'] = task_type
            response['source_info'] = source_info
            response['response'] = final_result

    
    response_file_path = os.path.join(current_dir_path,"model_response", f"response-{DEFAULT_MODEL_NAME}.jsonl")
    with open(response_file_path, "w", encoding="utf-8") as f:
        for response in response_list:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")

    debug_logger.info(f"final_result = {final_result}")


async def eval_response():
    response_file_path = os.path.join(current_dir_path, "model_response", f"response-{DEFAULT_MODEL_NAME}.jsonl")
    prompt_template = f"You are a judge tasked with determining whether text generation contains hallucinations. \
                    Based on the given task type, source_info (source information), and response (the generated reply), \
                    please judge whether the content of the reply is consistent with the source information.\
                         If there is information in the reply that does not match the source_info,\ that is, there is a hallucination, answer \"yes\"; \
                            if the content of the reply is completely based on and consistent with the source_info, answer \"no\"."

    hallu_cnt = 0
    eval_list = []

    with open(response_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"Processing line {i}...")
            response = json.loads(line)
            task_type = response['task_type']
            source_info = response['source_info']
            final_result = response['response']
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "prompt": f"{prompt_template}\n\n##Task type##: {task_type}\n##Source info##: {source_info}\n##Response##: {final_result} \
                Just show me the answer(\"yes\" or \"no\") and delete any other irrelevant text."}
            resp = requests.post(OLLAMA_API_BASE, json=payload, stream=True)
            result = []
            try:
                for line in resp.iter_lines():
                    if line:
                        try:
                            response_json = json.loads(line.decode('utf-8'))
                            if 'response' in response_json:
                                result.append(response_json['response'])
                            if 'done' in response_json and response_json['done']:
                                break
                        except json.JSONDecodeError as e:
                            print("\nJSON decode error:", e)
                            continue
            except json.JSONDecodeError:
                print("Initial JSON decode error: potentially non-stream response")
            resp = ''.join(result)
            print(resp)
            if not resp.startswith("no"):
                hallu_cnt += 1
            response['eval_result'] = resp
            eval_list.append(response)

    eval_file_path = os.path.join(current_dir_path, "eval_files", f"eval-{DEFAULT_MODEL_NAME}.jsonl")
    with open(eval_file_path, "w", encoding="utf-8") as f:
        for response in eval_list:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")

    return result

async def cal_hallu_cnt():
    hallu_cnt = 0
    eval_file_path = os.path.join(current_dir_path, "eval-Qwen2.5-7B-Instruct.jsonl")
    with open(eval_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            response = json.loads(line)
            if not response['eval_result'].startswith("no"):
                hallu_cnt += 1
    print(f"hallu_cnt = {hallu_cnt}")
    print(f"hallu_rate: {hallu_cnt / 303}")


if __name__ == "__main__":
    print(f"Use model {DEFAULT_MODEL_NAME}\n")
    result = asyncio.run(build_response(source_info_path))
    # result = asyncio.run(eval_response())
    cal_hallu_cnt()