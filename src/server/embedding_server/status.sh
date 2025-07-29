#!/bin/bash

# Embedding 服务状态检查脚本

LOG_FILE="record.log"
PORT=9001

echo "=== Embedding Service Status ==="

# 检查端口占用情况
echo "=== Port Status ==="
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "✓ Port $PORT is in use"
    echo "Port details:"
    lsof -i :$PORT
else
    echo "✗ Port $PORT is not in use"
fi

# 检查embedding相关进程
echo ""
echo "=== Process Status ==="

# 查找包含embedding_server.py的进程
EMBEDDING_PIDS=$(pgrep -f "embedding_server.py" 2>/dev/null)
if [ -n "$EMBEDDING_PIDS" ]; then
    echo "✓ Found embedding server processes:"
    for pid in $EMBEDDING_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime,pcpu,pmem
    done
else
    echo "✗ No embedding server processes found"
fi

# 查找包含python和embedding的进程
PYTHON_EMBEDDING_PIDS=$(pgrep -f "python.*embedding" 2>/dev/null)
if [ -n "$PYTHON_EMBEDDING_PIDS" ]; then
    echo "✓ Found python embedding processes:"
    for pid in $PYTHON_EMBEDDING_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime,pcpu,pmem
    done
else
    echo "✗ No python embedding processes found"
fi

# 查找所有embedding相关进程
ALL_EMBEDDING_PIDS=$(pgrep -f "embedding" 2>/dev/null)
if [ -n "$ALL_EMBEDDING_PIDS" ]; then
    echo "✓ All embedding-related processes:"
    for pid in $ALL_EMBEDDING_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime
    done
else
    echo "✗ No embedding-related processes found"
fi

# 检查日志文件
echo ""
echo "=== Log File Info ==="
if [ -f "$LOG_FILE" ]; then
    echo "✓ Log file found: $LOG_FILE"
    echo "Log file size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "Last modified: $(stat -c %y "$LOG_FILE")"
    
    # 显示最后几行日志
    echo ""
    echo "=== Recent Log Entries ==="
    tail -15 "$LOG_FILE"
else
    echo "✗ Log file not found: $LOG_FILE"
fi

# 检查服务健康状态
echo ""
echo "=== Service Health Check ==="
if lsof -i :$PORT > /dev/null 2>&1 && pgrep -f "embedding_server.py" > /dev/null 2>&1; then
    echo "✓ Embedding service appears to be healthy"
    echo "  - Port $PORT is in use"
    echo "  - Embedding server process is running"
    
    # 尝试简单的HTTP请求测试
    echo ""
    echo "=== HTTP Health Check ==="
    if command -v curl > /dev/null 2>&1; then
        echo "Testing HTTP endpoint..."
        HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/embedding 2>/dev/null || echo "Connection failed")
        if [ "$HTTP_RESPONSE" = "405" ]; then
            echo "✓ HTTP endpoint is responding (405 Method Not Allowed is expected for GET request)"
        elif [ "$HTTP_RESPONSE" = "Connection failed" ]; then
            echo "✗ HTTP endpoint is not responding"
        else
            echo "? HTTP endpoint returned: $HTTP_RESPONSE"
        fi
    else
        echo "curl not available, skipping HTTP health check"
    fi
else
    echo "✗ Embedding service appears to be unhealthy"
    if ! lsof -i :$PORT > /dev/null 2>&1; then
        echo "  - Port $PORT is not in use"
    fi
    if ! pgrep -f "embedding_server.py" > /dev/null 2>&1; then
        echo "  - Embedding server process is not running"
    fi
fi

echo ""
echo "=== Service Status Summary ==="
if lsof -i :$PORT > /dev/null 2>&1 && pgrep -f "embedding_server.py" > /dev/null 2>&1; then
    echo "✓ Embedding service is running and healthy"
    exit 0
else
    echo "✗ Embedding service is not running or not healthy"
    exit 1
fi 