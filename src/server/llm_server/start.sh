# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
export CUDA_VISIBLE_DEVICES=0,1
nohup lmdeploy serve api_server --log-level=INFO --server-port 2333 --tp=2 --dtype=float16 /home/zzh/Qwen2-VL-7B-Instruct > record.log 2>&1 &