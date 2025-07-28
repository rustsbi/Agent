#!/bin/bash

# vLLM 服务停止脚本

PID_FILE="vllm.pid"
LOG_FILE="vllm_record.log"

echo "Stopping vLLM service..."

# 检查PID文件是否存在
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found vLLM process with PID: $PID"
    
    # 检查进程是否还在运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing vLLM process (PID: $PID)..."
        kill $PID
        
        # 等待进程结束
        sleep 5
        
        # 检查进程是否还在运行，如果还在运行则强制杀死
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 $PID
        fi
        
        echo "vLLM service stopped successfully"
    else
        echo "Process with PID $PID is not running"
    fi
    
    # 删除PID文件
    rm -f "$PID_FILE"
    echo "PID file removed"
else
    echo "PID file not found, trying to find vLLM process by name..."
    
    # 通过进程名查找并杀死vLLM进程
    PIDS=$(pgrep -f "vllm serve")
    if [ -n "$PIDS" ]; then
        echo "Found vLLM processes: $PIDS"
        for pid in $PIDS; do
            echo "Killing process $pid..."
            kill $pid
        done
        echo "vLLM service stopped"
    else
        echo "No vLLM processes found"
    fi
fi

# 检查端口2333是否还在被占用
if lsof -i :2333 > /dev/null 2>&1; then
    echo "Warning: Port 2333 is still in use"
    echo "You may need to manually kill the process using port 2333"
else
    echo "Port 2333 is now free"
fi

echo "vLLM service stop script completed" 