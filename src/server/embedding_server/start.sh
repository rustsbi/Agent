#!/bin/bash

# Embedding 服务启动脚本

echo "Starting Embedding service..."

# 检查是否已经有embedding服务在运行
if lsof -i :9001 > /dev/null 2>&1; then
    echo "Warning: Port 9001 is already in use"
    echo "Current processes using port 9001:"
    lsof -i :9001
    echo "Please stop the existing service first using ./stop.sh"
    exit 1
fi

# 检查embedding_server.py是否存在
if [ ! -f "embedding_server.py" ]; then
    echo "Error: embedding_server.py not found in current directory"
    exit 1
fi

# 启动embedding服务
echo "Starting embedding server with nohup..."
nohup python embedding_server.py > record.log 2>&1 &

# 获取后台进程的PID
EMBEDDING_PID=$!
echo "Embedding service started with PID: $EMBEDDING_PID"

# 将PID保存到文件中，方便后续管理
echo $EMBEDDING_PID > embedding.pid
echo "PID saved to embedding.pid file"

# 等待几秒钟让服务启动
echo "Waiting for service to start..."
sleep 10

# 检查服务是否成功启动
if lsof -i :9001 > /dev/null 2>&1; then
    echo "✓ Embedding service started successfully"
    echo "✓ Service is running on port 9001"
    echo "✓ Log file: record.log"
    echo "✓ PID file: embedding.pid"
else
    echo "✗ Failed to start embedding service"
    echo "Check the log file for errors: record.log"
    exit 1
fi

echo "Embedding service startup completed"