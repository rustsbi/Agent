import numpy as np
import time
from typing import List, Union
from numpy import ndarray
import torch
from torch import Tensor
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from src.configs.configs import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_PATH, LOCAL_EMBED_BATCH, LOCAL_RERANK_MAX_LENGTH,EMBED_MODEL_PATH
from src.utils.log_handler import debug_logger
from transformers import AutoTokenizer

class EmbeddingBackend:
    def __init__(self, use_cpu: bool = False):
        # 初始化分词器
        self._tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
        # 设置返回numpy数组形式
        self.return_tensors = "np"
        # 批处理大小
        self.batch_size = LOCAL_EMBED_BATCH
        # 最大文本长度
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        # 进行onnx会话配置
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            # CUDA优先，如果GPU不可用会自动降级到CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 这行代码创建了一个ONNX模型的推理会话，是ONNX Runtime的核心组件。
        # 路径.onnx为后缀的文件，这是转换自其他深度学习框架（如PyTorch、TensorFlow、Transformer）的模型
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options, providers=providers)
        debug_logger.info(f"EmbeddingClient: model_path: {LOCAL_EMBED_MODEL_PATH}")
    # 获取文本嵌入向量
    def get_embedding(self, sentences, max_length):
        #                            输入文本列表  填充到最长序列 截断超长序列      最大序列长度            返回数组类型，numpy数组
        # 文本标记化
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors=self.return_tensors)
        # 将tokenizer输出转换为字典形式
        # 通常包含 'input_ids', 'attention_mask' 等键值对
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        # 记录开始时间
        start_time = time.time()
        # 模型推理
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"onnx infer time: {time.time() - start_time}")
        # outputs_onnx[0]: 获取第一个（也是唯一的）输出
        #  [:,0]使用numpy切片，选择所有样本的[CLS]标记对应的向量
        # BERT类模型中，[CLS]标记被设计用来表示整个序列的语义
        # 这个位置的向量包含了整个句子的上下文信息
        # 是获取句子级别表示的标准做法
        # 假设输入两个句子，模型输出形状为[2, 512, 768]
        # outputs_onnx[0]  # 形状：[2, 512, 768]
        # outputs_onnx[0][:,0]  # 形状：[2, 768]
        embedding = outputs_onnx[0][:,0]
        debug_logger.info(f'embedding shape: {embedding.shape}')
        # 计算l2范数
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        # 对向量进行L2归一化，便于后续计算相似度
        embeddings_normalized = embedding / norm_arr
        # 返回归一化后的向量列表
        return embeddings_normalized.tolist()

    # 使用IO Binding的优势：
    #     内存效率：减少不必要的数据复制
    #     性能提升：更好的内存管理和设备间数据传输
    #     更细粒度的控制：可以精确控制输入输出的设备位置
    def inference(self, inputs):
        outputs_onnx = None
        # 最多尝试2次
        try_num = 2
        while outputs_onnx is None and try_num > 0:
            try:
                io_binding = self._session.io_binding()
                # 绑定输入
                for k, v in inputs.items():
                    # 将输入数据绑定到CPU内存
                    io_binding.bind_cpu_input(k, v)
                # 确保输入数据同步
                io_binding.synchronize_inputs()
                # 绑定输出
                io_binding.bind_output('output')
                # 使用IO binding执行推理
                self._session.run_with_iobinding(io_binding)
                # 确保输出数据同步
                io_binding.synchronize_outputs()
                # 确保输出数据同步
                outputs_onnx = io_binding.copy_outputs_to_cpu()
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()
            except:
                outputs_onnx = None
            try_num -= 1
        # 返回结果
        return outputs_onnx

    def encode(self, sentence: Union[str, List[str]],
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = True,
               batch_size: int = 64,
               max_length: int = 384,
               tokenizer=None,
               return_tokens_num=False,
               return_time_log=False) -> Union[ndarray, Tensor]:

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []

        tokens_num = 0
        using_time_tokenizer = 0
        using_time_model = 0
        # batch数，向上取整
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
        # 开始处理batch
        for batch_id in range(total_batch):
            start_time_tokenizer = time.time()
            # 如果指定了标记化的tokenizer
            if tokenizer is not None:
                inputs = tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            else:
            # 如果没指定就用默认的
                inputs = self._tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            using_time_tokenizer += (time.time() - start_time_tokenizer)
            # 这行代码计算实际的token数量，减去了特殊token（如[CLS]和[SEP]）的数
            if return_tokens_num:
                tokens_num += (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])

            inputs = {k: v for k, v in inputs.items()}

            start_time_model = time.time()
            # 执行推理
            outputs_onnx = self.inference(inputs)
            using_time_model += (time.time() - start_time_model)
            # 和get_embedding函数一样，看上面注释
            embeddings = np.asarray(outputs_onnx[0][:, 0])
            # 如果需要正则化
            if normalize_to_unit:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embedding_list.append(embeddings)
        # # 合并np数组
        # embedding_list = [
        #     np.array([[1, 2, 3],
        #             [4, 5, 6]]),      # 形状(2, 3)的数组
        #     np.array([[7, 8, 9],
        #             [10, 11, 12]])    # 形状(2, 3)的数组
        # ]

        # # axis=0 表示在第一个维度（垂直方向）拼接
        # result = np.concatenate(embedding_list, axis=0)
        # # 结果形状(4, 3):
        # # [[1,  2,  3],
        # #  [4,  5,  6],
        # #  [7,  8,  9],
        # #  [10, 11, 12]]
        embeddings = np.concatenate(embedding_list, axis=0)
        # 当输入是单个句子且不需要保持维度时
        # 去掉第一个维度，从2D变为1D
        # 例如：从形状(1, 768)变为(768,)  
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        # 如果不需要返回numpy数组且当前是numpy数组
        # 将numpy数组转换为PyTorch张量
        if not return_numpy and isinstance(embeddings, ndarray):
            embeddings = torch.from_numpy(embeddings)
        # 同时返回token数量和时间日志
        if return_tokens_num and return_time_log:
            return embeddings, tokens_num, using_time_tokenizer, using_time_model
        # 只返回token数量
        elif return_tokens_num:
            return embeddings, tokens_num
        # 只返回时间日志
        elif return_time_log:
            return embeddings, using_time_tokenizer, using_time_model
        # 只返回嵌入向量
        else:
            return embeddings
    # 对给定queries
    def predict(self, queries, return_tokens_num=False):
        print(queries)
        embeddings = self.encode(
            queries, batch_size=self.batch_size, normalize_to_unit=True, return_numpy=True, max_length=self.max_length,
            tokenizer=self._tokenizer,
            return_tokens_num=return_tokens_num
        )
        print(embeddings.shape)
        return embeddings.tolist()