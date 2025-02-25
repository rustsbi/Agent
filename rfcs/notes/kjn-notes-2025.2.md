# 2月工作记录

本月工作集中在跑通 llm-client 和增加对于更多格式的文件解析支持。

## 跑通llm-client

在审查 logs 后，发现 llm-client 跑不通的原因在于两个方面：

1. configs.py 中的 `API-BASE` 和 `API-KEY` 写反了））
2. llm-client 中的 `llm = OpenAILLM(DEFAULT_MODEL_PATH, 8000, DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_API_CONTEXT_LENGTH, 5, 0.5)` 中的 `top-k` 的范围必须在0~1之间，我改成了0.5。

修改以上两个小错误后即可成功跑通 llm-client。

## 增加对于更多格式的文件解析支持

文件解析的处理集中在 `src/core/file_handler/file_handler.py` 里，我们目前使用的是 langchain 框架，langchain-community中的 [document_loaders](https://python.langchain.com/api_reference/community/document_loaders.html) 中包含多个文件解析器，功能十分强大。

我选取了其中的 TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredXMLLoader，PyPDFLoader, UnstructuredImageLoader, UnstructuredHTMLLoader, UnstructuredURLLoader 九个常用文件格式的解析器，并对其中的md、ppt、pdf、image解析器进行了测试，均能成功解析。

这些 Loader 的用法大同小异，都是直接 `loader(file_path)`，然后再 `docs = loader.load()`即可，其中不同的loader可能会提供参数方便设置。例如下面的代码例子：

```python
    def load_md(file_path):
        try:
            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            docs = loader.load()
            insert_logger.info(f"UnstructuredMarkdownLoader success: {file_path}")
            return docs
        except Exception:
            insert_logger.error(f"UnstructuredMarkdownLoader error: {file_path}, {traceback.format_exc()}")
            return []
```


在第一次使用某个 loader 时，可能需要根据报错安装相印的依赖，如在使用 `UnstructuredImageLoader` 时，需要安装 `tesseract-ocr` 等依赖。

在阅读 langchain-community 中的文档时，我还发现它提供一个通用的文档解析器 `UnstructuredFileLoader`，可以解析多种文件格式，可以通过本地分区或远程 API 调用来加载文件，同时提供了更通用的配置选项，例如可以通过 `chunking_strategy` 和 `max_characters` 参数控制分块行为，需要安装 `unstructured` 包。

在适用场景上的区别：
- UnstructuredLoader：
    适用于需要处理多种文件格式的场景，尤其是当需要统一处理不同格式的文件时。
    适合需要灵活配置加载行为的场景，例如通过远程 API 加载文件。
- langchain_community.document_loaders 中的加载器：
    适用于需要对特定文件格式进行精细处理的场景。
    如果项目中主要处理单一文件格式，使用这些加载器可以提高效率。

我们这里暂时先用更精细的 document_loaders 中的加载器。