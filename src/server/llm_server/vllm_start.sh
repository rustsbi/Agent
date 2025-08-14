#!/bin/bash

# 解析 --gpu 参数
GPU_ARG=""
if [[ "$1" == "--gpu" && ( "$2" == "0" || "$2" == "1" ) ]]; then
    GPU_ARG="$2"
fi

# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
if [[ "$GPU_ARG" == "0" || "$GPU_ARG" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ARG  # 只用指定单卡
    export MODEL_NAME="Qwen2.5-7B-Instruct"
    export MODEL_PATH="/home/model/Qwen2.5-7B-Instruct"
    TP_SIZE=1
else
    export CUDA_VISIBLE_DEVICES=0,1  # 默认双卡
    export MODEL_NAME="Qwen2.5-32B-Instruct-AWQ"
    export MODEL_PATH="/home/model/${MODEL_NAME}"
    TP_SIZE=2
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 优化内存分配

# 模型路径和日志文件路径
# MODEL_NAME="Qwen2.5-14B-Instruct"
# MODEL_NAME="Qwen2.5-32B-Instruct-GPTQ-Int4"
# MODEL_PATH="/home/model/${MODEL_NAME}"

LOG_FILE="vllm_record.log"
MAX_LOG_SIZE=10485760  # 最大日志文件大小（10MB，单位为字节）

# 检查日志文件大小并清理
check_and_clean_log() {
    if [ -f "${LOG_FILE}" ]; then
        LOG_SIZE=$(stat -c%s "${LOG_FILE}")
        if [ "${LOG_SIZE}" -ge "${MAX_LOG_SIZE}" ]; then
            echo "Log file size exceeded ${MAX_LOG_SIZE} bytes. Cleaning log file..."
            > "${LOG_FILE}"  # 清空日志文件
        fi
    fi
}

# 启动 vLLM 服务
echo "Starting vLLM service with model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES}"

# 使用 nohup 在后台启动 vLLM 服务
nohup vllm serve \
  ${MODEL_PATH} \
  --served-model-name ${MODEL_NAME} \
  --tensor-parallel-size ${TP_SIZE} \
  --dtype float16 \
  --port 2333 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 15888 \
  >> ${LOG_FILE} 2>&1 &

# 获取后台进程的PID
VLLM_PID=$!
echo "vLLM service started with PID: ${VLLM_PID}"
echo "vLLM service started on port 2333. Logs are being written to ${LOG_FILE}"

# 将PID保存到文件中，方便后续管理
echo ${VLLM_PID} > vllm.pid
echo "PID saved to vllm.pid file"

# 每隔一段时间检查日志文件大小并清理
while true; do
    check_and_clean_log
    sleep 3600  # 每隔 1 小时检查一次日志文件大小
done