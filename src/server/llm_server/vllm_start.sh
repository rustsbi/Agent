#!/bin/bash

# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
export CUDA_VISIBLE_DEVICES=0,1  # 指定使用 GPU 0 和 GPU 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 优化内存分配


# 模型路径和日志文件路径
# MODEL_NAME="Qwen2.5-14B-Instruct"
MODEL_NAME="Qwen2.5-32B-Instruct-AWQ"
# MODEL_NAME="Qwen2.5-32B-Instruct-GPTQ-Int4"
MODEL_PATH="/home/model/${MODEL_NAME}"

LOG_FILE="vllm_record.log"

# 启动 vLLM 服务
vllm serve \
  ${MODEL_PATH} \
  --served-model-name ${MODEL_PATH} \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --port 2333 \
  --gpu-memory-utilization 0.90 \
  >> ${LOG_FILE} 2>&1 &

echo "vLLM service started on port 2333. Logs are being written to ${LOG_FILE}"