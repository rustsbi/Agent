# 7月工作记录

7月的工作主要是针对文档检索工作进行初步的优化，按照项目申请书中定的计划，优化主要分为三个部分：Query 改写与翻译模块，向量模型优化、排序与后处理

## Query 改写与翻译模块

该模块在 /src/core/query_rewrite 文件夹中。

language_detect.py 利用 langdetect 库检测输入文本的语言类型。

translator.py 利用 MarianMTModel 库实现中英互译。

rewriter.py 利用 vLLM 的 API 实现 query 的同义改写和hyde假设文档扩展两种重写模式。

pipeline.py 将上述三个模块串联起来，实现完整的 query 预处理 pipeline。

test_direct.py 可以简单测试整个query_rewrite模块的运行效果。

最后，在 qa_handler中增加了该模块的集成和调用，经过初步测试，可以正常运行。

## 向量模型优化

当前向量化处理操作主要在 embedding_server.py 中实现。

embedding_server.py 通过调用 configs.py 中的向量模型地址，实现分词和推理两步向量化操作。

而 configs.py 中使用的向量模型为网易的 bce 模型，现今用的比较多的是 BAAI 的 bge 模型。bge 模型比 bce 模型包含更多参数，效果更好，推理速读稍慢。

修改 configs.py 中的 EMBED_MODEL_PATH 为 BAAI/bge-large-en-v1.5，并修改 LOCAL_EMBED_MODEL_PATH 为 bge-large-en-v1.5/onnx/model.onnx。

同时，我编写了一个简单的测试程序 test_embedding.py，用于测试两个模型的向量化速度，结果如下：

```sh
(qanything) root@451f9b706ca3:/home/kjn/Agent/src/server/embedding_server# python test_embedding.py 

===== 测试模型: /home/kjn/Agent/src/server/embedding_server/bce_model/model.onnx =====
平均推理耗时: 0.2061 秒, 向量维度: (8, 768)
示例向量: [ 0.04291952 -0.00315673  0.0221201   0.0508904  -0.04219415 -0.0110904
 -0.01523068  0.04258551]

===== 测试模型: /home/kjn/Agent/src/server/embedding_server/bge-large-en-v1.5/onnx/model.onnx =====
平均推理耗时: 0.7434 秒, 向量维度: (8, 1024)
示例向量: [-7.62612140e-03  1.51189035e-02  1.89875085e-02  2.42083520e-03
  1.36781475e-02 -7.23496560e-05  1.61461625e-02  4.52082464e-03]
```

但在实际测试时，使用 BGE 模型进行测试时，报错：

```sh
ONNX 推理失败，outputs_onnx[0] 为 None
```

排查发现，是因为 BGE 模型的输出名称是 'last_hidden_state'，而不是 'output'，在 embedding_backend.py 中修改代码，通过动态获取输出名称，使得代码可以兼容不同模型的输出名称，从而解决该问题。


## 排序与后处理

和向量模型优化类似，排序模型也可以进行优化，当前使用的排序模型也是 BCE，我们可以升级为 BGE，该模型在多项公开榜单上Top1，支持中英双语，效果远超BCE。

rerank 过程也可以考虑使用多模型打分融合来提升鲁棒性，但这样效率会大大降低，暂时不考虑。

可以考虑使用 BAAI/bge-reranker-base、BAAI/bge-reranker-large 或 BAAI/bge-reranker-v2-minicpm-layerwise 三个版本。其中 base 参数小，large 参数最大，layerwise 均衡参数与推理速度。

暂时选择 BAAI/bge-reranker-large 模型，在 src/server/rerank_server 文件夹中，修改 configs.py 中的 RERANK_MODEL_PATH 为 BAAI/bge-reranker-large，并修改 LOCAL_RERANK_MODEL_PATH 为 bge-reranker-large/onnx/model.onnx。

初步测试发现 BGE 模型的推理速度稍慢，但是准确度高。

## configs.py 中的配置更新

```py
EMBED_MODEL_PATH = "/home/model/bge-base-en-v1.5"
LOCAL_EMBED_PATH = "/home/model/bge-base-en-v1.5"
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "onnx", "model.onnx")

RERANK_MODEL_PATH = "BAAI/bge-reranker-large"
LOCAL_RERANK_PATH = "/home/model/bge-reranker-large"
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "onnx", "model.onnx")


```

## 后续工作

1. 完成数据集的收集
2. 完成评估测试流程的构建
