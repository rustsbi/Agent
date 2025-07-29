#!/bin/bash

# Embedding 服务停止脚本

PID_FILE="embedding.pid"
LOG_FILE="record.log"
PORT=9001

echo "Stopping Embedding service..."

# 检查PID文件是否存在
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found embedding process with PID: $PID"
    
    # 检查进程是否还在运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing embedding process (PID: $PID)..."
        kill $PID
        
        # 等待进程结束
        sleep 5
        
        # 检查进程是否还在运行，如果还在运行则强制杀死
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 $PID
        fi
        
        echo "Embedding service stopped successfully"
    else
        echo "Process with PID $PID is not running"
    fi
    
    # 删除PID文件
    rm -f "$PID_FILE"
    echo "PID file removed"
else
    echo "PID file not found, trying to find embedding processes by other methods..."
fi

# 通过端口查找并杀死embedding服务进程
echo "Looking for embedding service processes on port $PORT..."

# 查找使用端口9001的进程
PORT_PIDS=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    echo "Found processes using port $PORT: $PORT_PIDS"
    for pid in $PORT_PIDS; do
        echo "Killing process $pid..."
        kill $pid
        
        # 等待进程结束
        sleep 3
        
        # 检查进程是否还在运行，如果还在运行则强制杀死
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Port $PORT processes killed"
else
    echo "No processes found using port $PORT"
fi

# 通过进程名查找并杀死embedding相关进程
echo "Looking for embedding-related processes..."

# 查找包含embedding_server.py的进程
EMBEDDING_PIDS=$(pgrep -f "embedding_server.py" 2>/dev/null)
if [ -n "$EMBEDDING_PIDS" ]; then
    echo "Found embedding server processes: $EMBEDDING_PIDS"
    for pid in $EMBEDDING_PIDS; do
        echo "Killing embedding server process $pid..."
        kill $pid
        
        # 等待进程结束
        sleep 3
        
        # 检查进程是否还在运行，如果还在运行则强制杀死
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Embedding server processes killed"
else
    echo "No embedding server processes found"
fi

# 查找包含python和embedding的进程
PYTHON_EMBEDDING_PIDS=$(pgrep -f "python.*embedding" 2>/dev/null)
if [ -n "$PYTHON_EMBEDDING_PIDS" ]; then
    echo "Found python embedding processes: $PYTHON_EMBEDDING_PIDS"
    for pid in $PYTHON_EMBEDDING_PIDS; do
        echo "Killing python embedding process $pid..."
        kill $pid
        
        # 等待进程结束
        sleep 3
        
        # 检查进程是否还在运行，如果还在运行则强制杀死
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        fi
    done
    echo "Python embedding processes killed"
else
    echo "No python embedding processes found"
fi

# 检查端口是否还在被占用
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "Warning: Port $PORT is still in use"
    echo "Remaining processes using port $PORT:"
    lsof -i :$PORT
else
    echo "✓ Port $PORT is now free"
fi

# 检查是否还有embedding相关进程
REMAINING_PIDS=$(pgrep -f "embedding" 2>/dev/null)
if [ -n "$REMAINING_PIDS" ]; then
    echo "Warning: Some embedding-related processes are still running:"
    for pid in $REMAINING_PIDS; do
        ps -p $pid -o pid,ppid,cmd,etime
    done
else
    echo "✓ All embedding-related processes have been stopped"
fi

echo "Embedding service stop script completed" 