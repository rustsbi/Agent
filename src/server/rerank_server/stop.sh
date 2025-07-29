#!/bin/bash
# Rerank 服务停止脚本
PID_FILE="rerank.pid"
LOG_FILE="record.log"
PORT=8001
echo "Stopping Rerank service..."

# 检查PID文件是否存在
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found rerank process with PID: $PID"
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing rerank process (PID: $PID)..."
        kill $PID
        sleep 5
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 $PID
        fi
        echo "Rerank service stopped successfully"
    else
        echo "Process with PID $PID is not running"
    fi
    rm -f "$PID_FILE"
    echo "PID file removed"
else
    echo "PID file not found, trying to find rerank processes by other methods..."
fi

# 通过端口查找并杀死rerank服务进程
echo "Looking for rerank service processes on port $PORT..."
PORT_PIDS=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    echo "Found processes using port $PORT: $PORT_PIDS"
    for pid in $PORT_PIDS; do
        echo "Killing process $pid..."
        kill $pid
        sleep 3
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Port $PORT processes killed"
else
    echo "No processes found using port $PORT"
fi

# 通过进程名查找并杀死rerank相关进程
echo "Looking for rerank-related processes..."
RERANK_PIDS=$(pgrep -f "rerank_server.py" 2>/dev/null)
if [ -n "$RERANK_PIDS" ]; then
    echo "Found rerank server processes: $RERANK_PIDS"
    for pid in $RERANK_PIDS; do
        echo "Killing rerank server process $pid..."
        kill $pid
        sleep 3
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Rerank server processes killed"
else
    echo "No rerank server processes found"
fi

PYTHON_RERANK_PIDS=$(pgrep -f "python.*rerank" 2>/dev/null)
if [ -n "$PYTHON_RERANK_PIDS" ]; then
    echo "Found python rerank processes: $PYTHON_RERANK_PIDS"
    for pid in $PYTHON_RERANK_PIDS; do
        echo "Killing python rerank process $pid..."
        kill $pid
        sleep 3
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Python rerank processes killed"
else
    echo "No python rerank processes found"
fi

if lsof -i :$PORT > /dev/null 2>&1; then
    echo "Warning: Port $PORT is still in use"
    echo "Remaining processes using port $PORT:"
    lsof -i :$PORT
else
    echo "✓ Port $PORT is now free"
fi

REMAINING_PIDS=$(pgrep -f "rerank" 2>/dev/null)
if [ -n "$REMAINING_PIDS" ]; then
    echo "Warning: Some rerank-related processes are still running:"
    for pid in $REMAINING_PIDS; do
        ps -p $pid -o pid,ppid,cmd,etime
    done
else
    echo "✓ All rerank-related processes have been stopped"
fi

echo "Rerank service stop script completed" 