# 1月工作记录

本月主要工作集中在编写、测试向量数据库的检索逻辑代码

## milvus检索代码

`milvus` 向量数据库提供了两种检索的方法，分别是 query 和 search 。

其中，search 方法主要用于执行近似最近邻搜索（Approximate Nearest Neighbors, ANN），即根据给定的查询向量找到与之最相似的向量。它的核心功能是基于**向量相似性**进行检索。

query 方法用于执行更广泛的基于条件的查询，主要用于基于条件的过滤，根据指定的条件表达式检索数据

在代码编写上选择使用更适合 RAG 系统的 search 方法。

检索逻辑代码放在了 milvus_client.py 下：

```python
    @get_time
    def search_docs(self, query_embedding: List[float] = None, filter_expr: str = None, doc_limit: int = 10):
        """
        从 Milvus 集合中检索文档。

        Args:
            query_embedding (List[float]): 查询向量，用于基于向量相似性检索。
            filter_expr (str): 过滤条件表达式，用于基于字段值的过滤。如"user_id == 'abc1234'"
            limit (int): 返回的文档数量上限，默认为 10。

        Returns:
            List[dict]: 检索到的文档列表，每个文档是一个字典，包含字段值和向量。
        """
        try:
            if not self.sess:
                raise MilvusFailed("Milvus collection is not loaded. Call load_collection_() first.")

            # 构造查询参数
            search_params = {
                "metric_type": self.search_params["metric_type"],
                "params": self.search_params["params"]
            }

            # 构造查询表达式
            expr = ""
            if filter_expr:
                expr = filter_expr

            # 构造检索参数
            search_params.update({
                "data": [query_embedding] if query_embedding else None,
                "anns_field": "embedding", # 指定集合中存储向量的字段名称。Milvus 会在该字段上进行向量相似性检索。
                "param": {"metric_type": "L2", "params": {"nprobe": 128}}, # 检索的精度和性能
                "limit": doc_limit, # 指定返回的最相似文档的数量上限
                "expr": expr,
                "output_fields": self.output_fields
            })

            # 执行检索
            results = self.sess.search(**search_params)

            # 处理检索结果
            retrieved_docs = []
            for hits in results:
                for hit in hits:
                    doc = {
                        # "id": hit.id,
                        # "distance": hit.distance,
                        "user_id": hit.entity.get("user_id"),
                        "kb_id": hit.entity.get("kb_id"),
                        "file_id": hit.entity.get("file_id"),
                        "headers": json.loads(hit.entity.get("headers")),
                        "doc_id": hit.entity.get("doc_id"),
                        "content": hit.entity.get("content"),
                        "embedding": hit.entity.get("embedding")
                    }
                    retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            print(f'[{cur_func_name()}] [search_docs] Failed to search documents: {traceback.format_exc()}')
            raise MilvusFailed(f"Failed to search documents: {str(e)}")
```

## 测试milvus检索逻辑

利用已有的 embedding 文件夹下的 embedding_client.py（原名为 client.py ）中的embedding处理代码，同时编写了 embed_user_input 方便测试。

同时在 milvus_client.py 的 main 函数中调用 search_docs 函数进行测试，测试结果如下。

不设置过滤条件正常检索：

![search_true](/rfcs/assets/search_docs_true.png)

设置过滤条件，检索结果为空：

![search_false](/rfcs/assets/search_docs_false.png)


## 未来工作

后续继续实现 server 与 client 的交互处理，方便更好地测试用户的输入经过 embedding 后到 milvus 中进行检索的过程。

RAG 系统的 UI 界面逐步完善。