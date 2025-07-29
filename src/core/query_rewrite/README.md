# Query Rewrite 集成说明

## 概述

Query Rewrite 模块已经成功集成到 RAG 流程中，提供查询翻译和扩展功能。

## 集成方案

### 1. 配置选项

在 `configs.py` 中添加了以下配置：

```python
# Query Rewrite 配置
QUERY_REWRITE_ENABLED = True  # 是否启用查询重写
QUERY_REWRITE_TARGET_LANG = 'en'  # 目标语言，默认为英文
```

### 2. 集成位置

- **QAHandler**: 在 `get_knowledge_based_answer` 方法的最开始进行 query_rewrite 处理
- **API层面**: 添加了可选的 `query_rewrite` 参数（默认为 `True`）

### 3. 处理流程

1. 接收用户查询
2. 如果启用了 query_rewrite，使用 `QueryRewritePipeline` 处理查询
3. 处理包括：
   - 语言检测
   - 翻译（如需要）
   - 查询扩展
4. 使用处理后的查询进行后续的检索和生成

### 4. 使用方法

#### API 调用

```python
payload = {
    "user_id": "your_user_id",
    "user_info": "your_user_info", 
    "kb_ids": ["your_kb_id"],
    "question": "如何在Rust中实现内存分配？",
    "query_rewrite": True,  # 启用query_rewrite
    # ... 其他参数
}
```

#### 配置控制

- 全局开关：修改 `configs.py` 中的 `QUERY_REWRITE_ENABLED`
- 单次控制：在API请求中设置 `query_rewrite` 参数

### 5. 测试

运行测试脚本验证集成：

```bash
cd /home/kjn/Agent/src/evaluation
python test_query_rewrite_integration.py
```

### 6. 日志输出

集成后会在日志中输出：

- Query rewrite 处理时间
- 原始查询和处理后查询的对比
- 处理过程中的详细信息

### 7. 优势

- **简单集成**: 最小化对现有代码的修改
- **灵活控制**: 可以通过配置和API参数控制
- **向后兼容**: 不影响现有功能
- **性能监控**: 提供详细的时间记录

## 注意事项

1. 确保 `qanything` 环境已激活
2. 首次运行时会下载翻译模型（MarianMT）
3. 翻译模型需要一定的内存和计算资源
4. 建议在生产环境中根据实际需求调整配置 