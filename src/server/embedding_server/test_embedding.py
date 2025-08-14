import os
import time
from transformers import AutoTokenizer
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import numpy as np

# 配置本地模型路径
BCE_MODEL_PATH = os.path.join(os.path.dirname(
    __file__), 'bce_model', 'model.onnx')
BGE_MODEL_PATH = os.path.join(os.path.dirname(
    __file__), 'bge-base-en-v1.5', 'onnx', 'model.onnx')
BCE_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'bce_model')
BGE_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), 'bge-base-en-v1.5')

TEST_TEXTS = [
    "如何在Rust中实现内存分配？",
    "用Rust写一个加法函数的步骤是什么？",
    "What is the difference between stack and heap in Rust?",
    "Explain memory safety in Rust programming language."
] * 8  # 批量测试


def test_embedding(model_path, tokenizer_path, texts, batch_size=8, repeat=3):
    print(f"\n===== 测试模型: {model_path} =====")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 加载ONNX模型
    sess_options = SessionOptions()
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 0
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(model_path, sess_options=sess_options, providers=[
                               "CPUExecutionProvider"])

    output_name = [o.name for o in session.get_outputs()]

    # 统计推理时间
    total_time = 0
    for r in range(repeat):
        start = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="np")
            outputs = session.run(output_names=output_name, input_feed={
                                  k: v for k, v in inputs.items()})
            # 取[CLS]向量
            embedding = outputs[0][:, 0]
            # L2归一化
            norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
            embeddings_normalized = embedding / norm_arr
        end = time.time()
        total_time += (end - start)
    avg_time = total_time / repeat
    print(f"平均推理耗时: {avg_time:.4f} 秒, 向量维度: {embeddings_normalized.shape}")
    print(f"示例向量: {embeddings_normalized[0][:8]}")


if __name__ == "__main__":
    test_embedding(BCE_MODEL_PATH, BCE_TOKENIZER_PATH, TEST_TEXTS)
    test_embedding(BGE_MODEL_PATH, BGE_TOKENIZER_PATH, TEST_TEXTS)
