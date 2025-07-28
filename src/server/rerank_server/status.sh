#!/bin/bash
# Rerank 服务状态检查脚本
LOG_FILE="record.log"
PORT=8001
echo "=== Rerank Service Status ==="

echo "=== Port Status ==="
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "✓ Port $PORT is in use"
    echo "Port details:"
    lsof -i :$PORT
else
    echo "✗ Port $PORT is not in use"
fi

echo ""
echo "=== Process Status ==="
RERANK_PIDS=$(pgrep -f "rerank_server.py" 2>/dev/null)
if [ -n "$RERANK_PIDS" ]; then
    echo "✓ Found rerank server processes:"
    for pid in $RERANK_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime,pcpu,pmem
    done
else
    echo "✗ No rerank server processes found"
fi

PYTHON_RERANK_PIDS=$(pgrep -f "python.*rerank" 2>/dev/null)
if [ -n "$PYTHON_RERANK_PIDS" ]; then
    echo "✓ Found python rerank processes:"
    for pid in $PYTHON_RERANK_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime,pcpu,pmem
    done
else
    echo "✗ No python rerank processes found"
fi

ALL_RERANK_PIDS=$(pgrep -f "rerank" 2>/dev/null)
if [ -n "$ALL_RERANK_PIDS" ]; then
    echo "✓ All rerank-related processes:"
    for pid in $ALL_RERANK_PIDS; do
        echo "Process details (PID: $pid):"
        ps -p $pid -o pid,ppid,cmd,etime
    done
else
    echo "✗ No rerank-related processes found"
fi

echo ""
echo "=== Log File Info ==="
if [ -f "$LOG_FILE" ]; then
    echo "✓ Log file found: $LOG_FILE"
    echo "Log file size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "Last modified: $(stat -c %y "$LOG_FILE")"
    echo ""
    echo "=== Recent Log Entries ==="
    tail -15 "$LOG_FILE"
else
    echo "✗ Log file not found: $LOG_FILE"
fi

echo ""
echo "=== Service Health Check ==="
if lsof -i :$PORT > /dev/null 2>&1 && pgrep -f "rerank_server.py" > /dev/null 2>&1; then
    echo "✓ Rerank service appears to be healthy"
    echo "  - Port $PORT is in use"
    echo "  - Rerank server process is running"
    echo ""
    echo "=== HTTP Health Check ==="
    if command -v curl > /dev/null 2>&1; then
        echo "Testing HTTP endpoint..."
        HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/rerank 2>/dev/null || echo "Connection failed")
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
    echo "✗ Rerank service appears to be unhealthy"
    if ! lsof -i :$PORT > /dev/null 2>&1; then
        echo "  - Port $PORT is not in use"
    fi
    if ! pgrep -f "rerank_server.py" > /dev/null 2>&1; then
        echo "  - Rerank server process is not running"
    fi
fi

echo ""
echo "=== Service Status Summary ==="
if lsof -i :$PORT > /dev/null 2>&1 && pgrep -f "rerank_server.py" > /dev/null 2>&1; then
    echo "✓ Rerank service is running and healthy"
    exit 0
else
    echo "✗ Rerank service is not running or not healthy"
    exit 1
fi 