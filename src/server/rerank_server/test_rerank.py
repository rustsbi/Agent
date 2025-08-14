import os
import time
import numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import json

# 配置本地模型路径
BCE_RERANK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'bce_model', 'model.onnx')
BGE_RERANK_MODEL_PATH = os.path.join(os.path.dirname(
    __file__), 'bge-reranker-large', 'onnx', 'model.onnx')
BCE_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'bce_model')
BGE_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), 'bge-reranker-large')


def sigmoid(x):
    """Sigmoid函数，用于将logits转换为概率分数"""
    x = x.astype('float32')
    scores = 1/(1+np.exp(-x))
    scores = np.clip(1.5*(scores-0.5)+0.5, 0, 1)
    return scores


class RerankTester:
    def __init__(self, model_path, tokenizer_path, model_name):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_name = model_name

        # 加载tokenizer
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 加载ONNX模型
        print(f"Loading ONNX model for {model_name}...")
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession(
            model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        # 获取输入输出信息
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [
            output.name for output in self.session.get_outputs()]
        print(f"{model_name} - Input names: {self.input_names}")
        print(f"{model_name} - Output names: {self.output_names}")

        # 设置参数
        self.max_length = 512
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize_query_passage(self, query, passage):
        """将query和passage编码为模型输入格式"""
        # 编码query
        query_inputs = self.tokenizer.encode_plus(
            query, truncation=False, padding=False)

        # 编码passage
        passage_inputs = self.tokenizer.encode_plus(
            passage, truncation=False, padding=False)

        # 计算最大passage长度
        max_passage_length = self.max_length - \
            len(query_inputs['input_ids']) - 2

        # 截断passage
        if len(passage_inputs['input_ids']) > max_passage_length:
            passage_inputs['input_ids'] = passage_inputs['input_ids'][:max_passage_length]
            passage_inputs['attention_mask'] = passage_inputs['attention_mask'][:max_passage_length]
            if 'token_type_ids' in passage_inputs:
                passage_inputs['token_type_ids'] = passage_inputs['token_type_ids'][:max_passage_length]

        # 合并query和passage
        input_ids = query_inputs['input_ids'] + [self.sep_token_id] + \
            passage_inputs['input_ids'] + [self.sep_token_id]
        attention_mask = query_inputs['attention_mask'] + \
            [1] + passage_inputs['attention_mask'] + [1]

        # 添加token_type_ids（如果模型需要）
        token_type_ids = None
        if 'token_type_ids' in query_inputs and 'token_type_ids' in passage_inputs:
            token_type_ids = [0] * len(query_inputs['input_ids']) + \
                [0] + [1] * len(passage_inputs['input_ids']) + [1]

        # 填充到最大长度
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            if token_type_ids:
                token_type_ids += [0] * padding_length

        return {
            'input_ids': np.array([input_ids]),
            'attention_mask': np.array([attention_mask]),
            'token_type_ids': np.array([token_type_ids]) if token_type_ids else None
        }

    def predict_single(self, query, passage):
        """预测单个query-passage对的分数"""
        # 编码输入
        inputs = self.tokenize_query_passage(query, passage)

        # 准备模型输入
        model_inputs = {
            self.input_names[0]: inputs['input_ids'],
            self.input_names[1]: inputs['attention_mask']
        }

        # 如果有token_type_ids且模型支持
        if inputs['token_type_ids'] is not None and len(self.input_names) > 2:
            model_inputs[self.input_names[2]] = inputs['token_type_ids']

        # 执行推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, model_inputs)
        inference_time = time.time() - start_time

        # 应用sigmoid函数
        scores = sigmoid(np.array(outputs[0]))

        return float(scores[0]), inference_time

    def predict_batch(self, query, passages):
        """批量预测query和多个passages的分数"""
        results = []
        total_time = 0

        for i, passage in enumerate(passages):
            score, inference_time = self.predict_single(query, passage)
            results.append({
                'index': i,
                'passage': passage[:100] + '...' if len(passage) > 100 else passage,
                'score': score,
                'inference_time': inference_time
            })
            total_time += inference_time

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)

        return results, total_time


def create_test_cases():
    """创建测试用例"""
    test_cases = [
        {
            'name': '中文问答测试',
            'query': '什么是机器学习？',
            'passages': [
                '机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。',
                '深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的学习过程。',
                '自然语言处理是人工智能的一个重要应用领域，专注于计算机理解和生成人类语言。',
                '计算机视觉是人工智能的一个分支，致力于让计算机能够理解和处理图像和视频。',
                '强化学习是机器学习的一种方法，通过与环境交互来学习最优策略。'
            ]
        },
        {
            'name': '英文问答测试',
            'query': 'What is artificial intelligence?',
            'passages': [
                'Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.',
                'Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
                'Deep learning is a subset of machine learning that uses neural networks to simulate the learning process of the human brain.',
                'Natural language processing is an important application of AI that focuses on enabling computers to understand and generate human language.',
                'Computer vision is a branch of AI that aims to enable computers to understand and process images and videos.'
            ]
        },
        {
            'name': '技术文档测试',
            'query': '如何安装Python包？',
            'passages': [
                '使用pip install命令可以安装Python包，例如：pip install numpy。',
                'conda是另一个包管理工具，可以使用conda install来安装包。',
                'Python是一种高级编程语言，具有简洁的语法和强大的功能。',
                '虚拟环境可以帮助隔离不同项目的依赖，使用venv或conda创建。',
                'requirements.txt文件可以列出项目的所有依赖包。'
            ]
        },
        {
            'name': '长文本测试',
            'query': 'Rust编程语言的特点是什么？',
            'passages': [
                'Rust是一种系统编程语言，专注于安全性、速度和并发性。它通过所有权系统提供内存安全保证，无需垃圾回收器。Rust的零成本抽象特性使得高级编程概念不会带来运行时开销。',
                'Python是一种解释型语言，具有简洁的语法和丰富的库生态系统。它广泛应用于数据科学、机器学习和Web开发等领域。Python的易学易用特性使其成为初学者的理想选择。',
                'Java是一种面向对象的编程语言，具有跨平台特性。它使用虚拟机来运行代码，提供了良好的安全性和稳定性。Java在企业级应用开发中非常流行。',
                'C++是一种多范式编程语言，支持面向对象、泛型和过程式编程。它提供了对硬件的低级控制，同时支持高级抽象。C++在系统编程和游戏开发中广泛使用。',
                'JavaScript是一种动态类型语言，主要用于Web开发。它可以在浏览器中运行，也可以使用Node.js在服务器端运行。JavaScript的灵活性和丰富的生态系统使其成为Web开发的标准语言。'
            ]
        }
    ]
    return test_cases


def run_comparison_test():
    """运行比较测试"""
    print("=== Rerank模型性能比较测试 ===\n")

    # 创建测试器
    bce_tester = RerankTester(BCE_RERANK_MODEL_PATH,
                              BCE_TOKENIZER_PATH, "BCE-Reranker")
    bge_tester = RerankTester(BGE_RERANK_MODEL_PATH,
                              BGE_TOKENIZER_PATH, "BGE-Reranker")

    # 获取测试用例
    test_cases = create_test_cases()

    # 运行测试
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试用例: {test_case['name']}")
        print(f"查询: {test_case['query']}")
        print(f"{'='*60}")

        # BCE测试
        print(f"\n--- BCE-Reranker 结果 ---")
        bce_results, bce_total_time = bce_tester.predict_batch(
            test_case['query'], test_case['passages'])

        # BGE测试
        print(f"\n--- BGE-Reranker 结果 ---")
        bge_results, bge_total_time = bge_tester.predict_batch(
            test_case['query'], test_case['passages'])

        # 显示结果
        print(f"\n--- 详细结果对比 ---")
        print(
            f"{'排名':<4} {'BCE分数':<10} {'BGE分数':<10} {'BCE时间(ms)':<12} {'BGE时间(ms)':<12} {'内容'}")
        print("-" * 80)

        for i in range(len(test_case['passages'])):
            bce_item = next(item for item in bce_results if item['index'] == i)
            bge_item = next(item for item in bge_results if item['index'] == i)

            print(f"{i+1:<4} {bce_item['score']:<10.4f} {bge_item['score']:<10.4f} "
                  f"{bce_item['inference_time']*1000:<12.2f} {bge_item['inference_time']*1000:<12.2f} "
                  f"{bce_item['passage']}")

        # 性能统计
        print(f"\n--- 性能统计 ---")
        print(f"BCE总推理时间: {bce_total_time:.4f}秒")
        print(f"BGE总推理时间: {bge_total_time:.4f}秒")
        print(
            f"BCE平均推理时间: {bce_total_time/len(test_case['passages'])*1000:.2f}毫秒")
        print(
            f"BGE平均推理时间: {bge_total_time/len(test_case['passages'])*1000:.2f}毫秒")

        # 排名一致性分析
        bce_ranks = {item['index']: rank for rank,
                     item in enumerate(bce_results)}
        bge_ranks = {item['index']: rank for rank,
                     item in enumerate(bge_results)}

        print(f"\n--- 排名一致性分析 ---")
        same_rank_count = sum(1 for i in range(
            len(test_case['passages'])) if bce_ranks[i] == bge_ranks[i])
        print(
            f"排名完全一致的passage数量: {same_rank_count}/{len(test_case['passages'])}")
        print(
            f"排名一致性比例: {same_rank_count/len(test_case['passages'])*100:.1f}%")

        # 分数分布分析
        bce_scores = [item['score'] for item in bce_results]
        bge_scores = [item['score'] for item in bge_results]

        print(f"\n--- 分数分布分析 ---")
        print(f"BCE分数范围: {min(bce_scores):.4f} - {max(bce_scores):.4f}")
        print(f"BGE分数范围: {min(bge_scores):.4f} - {max(bge_scores):.4f}")
        print(f"BCE分数标准差: {np.std(bce_scores):.4f}")
        print(f"BGE分数标准差: {np.std(bge_scores):.4f}")


def test_single_model():
    """测试单个模型"""
    print("=== 单个模型测试 ===\n")

    # 测试BGE模型
    bge_tester = RerankTester(BGE_RERANK_MODEL_PATH,
                              BGE_TOKENIZER_PATH, "BGE-Reranker")

    # 简单测试
    query = "什么是人工智能？"
    passages = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习使用神经网络来模拟人脑的学习过程，是机器学习的一个重要分支。"
    ]

    print(f"查询: {query}")
    print(f"Passages数量: {len(passages)}")

    results, total_time = bge_tester.predict_batch(query, passages)

    print(f"\n结果:")
    for i, result in enumerate(results):
        print(
            f"{i+1}. 分数: {result['score']:.4f}, 时间: {result['inference_time']*1000:.2f}ms")
        print(f"   内容: {result['passage']}")

    print(f"\n总推理时间: {total_time:.4f}秒")
    print(f"平均推理时间: {total_time/len(passages)*1000:.2f}毫秒")


if __name__ == "__main__":
    try:
        # 检查模型文件是否存在
        if not os.path.exists(BCE_RERANK_MODEL_PATH):
            print(f"警告: BCE模型文件不存在: {BCE_RERANK_MODEL_PATH}")
            print("将只测试BGE模型...")
            test_single_model()
        elif not os.path.exists(BGE_RERANK_MODEL_PATH):
            print(f"警告: BGE模型文件不存在: {BGE_RERANK_MODEL_PATH}")
            print("将只测试BCE模型...")
            # 可以添加BCE单独测试
        else:
            # 运行完整比较测试
            run_comparison_test()

    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
