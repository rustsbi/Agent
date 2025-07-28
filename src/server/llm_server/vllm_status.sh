#!/bin/bash

# vLLM 服务状态检查脚本

PID_FILE="vllm.pid"
LOG_FILE="vllm_record.log"

echo "=== vLLM Service Status ==="

# 检查PID文件
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "PID file found: $PID_FILE"
    echo "Stored PID: $PID"
    
    # 检查进程是否运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Process is running (PID: $PID)"
        
        # 显示进程详细信息
        echo "Process details:"
        ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem
        
        # 检查端口占用
        if lsof -i :2333 > /dev/null 2>&1; then
            echo "✓ Port 2333 is in use"
            echo "Port details:"
            lsof -i :2333
        else
            echo "✗ Port 2333 is not in use"
        fi
        
        # 检查GPU使用情况
        echo "GPU usage:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        
    else
        echo "✗ Process with PID $PID is not running"
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
else
    echo "✗ PID file not found: $PID_FILE"
fi

# 检查日志文件
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "=== Log File Info ==="
    echo "Log file: $LOG_FILE"
    echo "Log file size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "Last modified: $(stat -c %y "$LOG_FILE")"
    
    # 显示最后几行日志
    echo ""
    echo "=== Recent Log Entries ==="
    tail -10 "$LOG_FILE"
else
    echo "✗ Log file not found: $LOG_FILE"
fi

# 检查是否有其他vLLM进程
echo ""
echo "=== All vLLM Processes ==="
PIDS=$(pgrep -f "vllm serve")
if [ -n "$PIDS" ]; then
    echo "Found vLLM processes:"
    for pid in $PIDS; do
        ps -p $pid -o pid,ppid,cmd,etime
    done
else
    echo "No vLLM processes found"
fi

echo ""
echo "=== Service Status Summary ==="
if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1 && lsof -i :2333 > /dev/null 2>&1; then
    echo "✓ vLLM service is running and healthy"
    exit 0
else
    echo "✗ vLLM service is not running or not healthy"
    exit 1
fi 