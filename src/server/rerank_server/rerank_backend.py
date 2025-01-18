from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from src.configs.configs import LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_BATCH, RERANK_MODEL_PATH, LOCAL_RERANK_THREADS,\
    LOCAL_RERANK_MODEL_PATH
from src.utils.log_handler import debug_logger
from src.utils.general_utils import get_time
import concurrent.futures
import onnxruntime
import numpy as np


def sigmoid(x):
    x = x.astype('float32')
    scores = 1/(1+np.exp(-x))
    scores = np.clip(1.5*(scores-0.5)+0.5, 0, 1)
    return scores

class RerankBackend():
    def __init__(self, use_cpu: bool = False):
        self._tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
        self.spe_id = self._tokenizer.sep_token_id
        # 设置重叠长度，80，方便记录上下文
        self.overlap_tokens = 80
        self.batch_size = LOCAL_RERANK_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = None
        self.workers = LOCAL_RERANK_THREADS
        self.use_cpu = use_cpu
        self.return_tensors = "np"
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(LOCAL_RERANK_MODEL_PATH, sess_options, providers=providers)
    # 推理
    def inference(self, batch):
        # 准备输入数据，准备ONNX模型输入
        print("开始推理......")
        inputs = {self.session.get_inputs()[0].name: batch['input_ids'],
                  self.session.get_inputs()[1].name: batch['attention_mask']}
        # 可选的token_type_ids输入
        if 'token_type_ids' in batch:
            inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']

        # 执行推理 输出为logits, None表示获取所有输出
        result = self.session.run(None, inputs)
        # debug_logger.info(f"rerank result: {result}")

        # 应用sigmoid函数
        # sigmoid_scores = 1 / (1 + np.exp(-np.array(result[0])))
        # 为什么是result[0]因为只有一个输出
        # 将logits使用sigmoid函数映射到(0,1)区间内
        # print("logits shape: ", result[0].shape)
        sigmoid_scores = sigmoid(np.array(result[0]))
        # 5. 整理输出格式，转换为一维
        return sigmoid_scores.reshape(-1).tolist() 

    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)

        # 在 chunk1 的末尾添加分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为分隔符添加 attention mask

        # 添加 chunk2 的内容
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])

        # 在整个序列的末尾再添加一个分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为最后的分隔符添加 attention mask

        if 'token_type_ids' in chunk1:
            # 为 chunk2 和两个分隔符添加 token_type_ids
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)

        return chunk1
    # 处理长文本重排序的预处理函数，将query和passages转换为模型可以处理的格式。
    def tokenize_preproc(self,
                         query: str,
                         passages: List[str]):
        # 先对query进行编码
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)
        # 计算passage最大长度，减2是因为添加了两个分隔符
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 2 
        # 例如：
        # self.max_length = 512
        # query长度 = 30
        # 最大passage长度 = 512 - 30 - 2 = 480
        assert max_passage_inputs_length > 10
        # 计算重叠token数
        # 防止重叠太大，最多是passage最大长度的2/7
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            # 对passage进行编码
            passage_inputs = self._tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                         add_special_tokens=False)
            # 编码长度
            passage_inputs_length = len(passage_inputs['input_ids'])
            # 当passage长度小于最大允许长度时
            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue
                # 直接合并query和passage
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                # print("query ids: ",query_inputs)
                # print("passage ids: ",passage_inputs)
                # print(query, " " , passage)
                # print(qp_merge_inputs)
                # 记录原始passage的索引，排序用
                merge_inputs_idxs.append(pid)
            else:
            # 当passage过长时，需要分段处理
                start_id = 0
                while start_id < passage_inputs_length:
                    # 切分passage
                    end_id = start_id + max_passage_inputs_length
                    # 提取子段
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    # 计算下一段的开始位置（考虑重叠）
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id
                    # query和子段合并
                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    # 放入合并后的
                    merge_inputs.append(qp_merge_inputs)
                    # 记录原始索引
                    merge_inputs_idxs.append(pid)
        # 返回合并后的输入，和记录的原始索引位置
        return merge_inputs, merge_inputs_idxs

    @get_time
    def get_rerank(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self._tokenizer.pad(
                    tot_batches[k:k + self.batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                future = executor.submit(self.inference, batch)
                futures.append(future)
            # debug_logger.info(f'rerank number: {len(futures)}')
            for future in futures:
                scores = future.result()
                # todo
                tot_scores.extend(scores)
                # print(len(scores))
                # print(scores[:5])
        # 对于被分段的文档，取分段的最高分数
        # print("passages len: ", len(passages))
        # print("merge inputs idx sort: ", len(merge_inputs_idxs_sort))
        # print("tot_scores: ", len(tot_scores))

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        # print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores