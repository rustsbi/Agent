import json
import re
import os

current_path = os.path.dirname(os.path.abspath(__file__))
eval_file_path = os.path.join(current_path,"model_response", 'response-internlm3-8b-instruct.jsonl')



def calculate_hallucination(eval_file_path):
    # 定义正则表达式，确保匹配独立的单词
    yes_pattern = re.compile(r'\byes\b', re.IGNORECASE)
    no_pattern = re.compile(r'\bno\b', re.IGNORECASE)

    # 初始化统计变量
    hallucination_count = 0
    no_hallucination_count = 0


    # 加载数据
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            eval_result = data['eval_result'].lower()
            if yes_pattern.search(eval_result):
                hallucination_count += 1
            elif no_pattern.search(eval_result):
                no_hallucination_count += 1
            else:
                print(f"Unrecognized eval_result: {eval_result}")

    # 输出统计结果
    print(f"Hallucination count: {hallucination_count}")
    print(f"No hallucination count: {no_hallucination_count}")

def extract_pure_eval_and_calculate(eval_file_path):
    # 定义正则表达式，用于去除 <think> 和 </think> 包裹的内容
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    yes_pattern = re.compile(r'\byes\b', re.IGNORECASE)
    no_pattern = re.compile(r'\bno\b', re.IGNORECASE)    
    # 初始化存储纯粹答案的列表
    pure_evals = []
    hallucination_count = 0
    no_hallucination_count = 0

    # 打开并逐行读取 JSONL 文件
    with open(eval_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行为 JSON 对象
            item = json.loads(line)
            eval_result = item.get('eval_result', '')
            
            # 使用正则表达式去除 <think> 和 </think> 包裹的内容
            pure_eval = think_pattern.sub('', eval_result).strip()
            # print(pure_eval)
            # print('-'*40)
            # 将纯粹的答案添加到列表中
            pure_evals.append(pure_eval)
            if pure_eval == 'yes':
                hallucination_count += 1
            elif pure_eval == 'no':
                no_hallucination_count += 1
            elif yes_pattern.search(pure_eval):
                hallucination_count += 1
            elif no_pattern.search(pure_eval):
                no_hallucination_count += 1
            else:
                print(f"Unrecognized eval_result: {pure_eval}")
    print(f"Hallucination count: {hallucination_count}")
    print(f"No hallucination count: {no_hallucination_count}")
    print(f"Hallucinatioin rate: {(hallucination_count / (hallucination_count + no_hallucination_count)):.4f}")
    # 返回纯粹的答案列表
    return pure_evals


# calculate_hallucination(eval_file_path)
# print('\n\n\n')
extract_pure_eval_and_calculate(eval_file_path)
