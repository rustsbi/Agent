import transformers
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification

from pathlib import Path
import torch

# 1. 加载模型和分词器
model_name = "maidalun1020/bce-reranker-base_v1"  # 或其他模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 创建样本输入
encoded_input = tokenizer("这是一个测试句子", return_tensors="pt")

# 3. 导出为ONNX
output_path = Path("bce_model/model.onnx")

torch.onnx.export(
    model,                       # 要导出的模型
    tuple(encoded_input.values()),  # 模型输入
    output_path,                 # 保存路径
    export_params=True,          # 存储训练好的参数权重
    opset_version=14,           # ONNX 算子集版本
    do_constant_folding=True,   # 是否执行常量折叠优化
    input_names=['input_ids', 'attention_mask'],    # 输入节点的名称
    output_names=['logits'],    # 输出节点的名称
    dynamic_axes={              # 动态尺寸的设置
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
)