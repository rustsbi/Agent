# 2024.12

## 开发记录

### 环境配置

由于 Elasticsearch 要求非 root 用户运行，同时为简化开发环境的管理，新建了非 root 用户并进行以下配置：

#### 安装 Miniconda

Link：`https://www.anaconda.com/download/success`

正常下载脚本安装即可。

#### 安装 Elasticsearch

Link：`https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.17.0-linux-x86_64.tar.gz`

解包安装即可。安装之后，通过安装路径下的 `bin/elasticsearch` 可执行文件运行 `Elasticsearch` 程序，于 `0.0.0.0:9200` 监听外部请求。传入的文档切片将被保存在安装目录下的 `data/indices` 目录中。程序监听端口，使用协议等配置位于 `config/elasticsearch.yml` 文件中。

#### 安装 faiss

Link：`https://github.com/facebookresearch/faiss/blob/main/INSTALL.md`

`Faiss` 作为一个 `python` 模块，正常来说可以通过 `pip` 安装。但是实践发现该模块在 conda 环境下的 pip 维护已经停止，需要通过 conda 进行安装。具体安装指导见 Link。

```
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
```

### Files2VectorDB 代码优化

在将原有的 `files2db.py` 代码优化为 `FileToVectorDB` 类时，进行了以下改进和优化：

#### 重构为类结构

- 将原有的函数式代码封装到 FileToVectorDB 类中，提高代码的模块化和复用性。
- 在初始化方法中统一处理数据库根目录和嵌入模型的配置，确保所有实例化对象的一致性。

#### 优化文件解析方法

- 将各类文件的解析函数（如 parse_adoc, parse_markdown, parse_latex, parse_rst, parse_txt）作为类的成员方法，使得代码组织更加清晰。
- 在 `Asciidoc` 和 `reStructuredText` 的解析中，增强了对标题和内容的处理逻辑，确保更准确的内容分块。

#### 改进文件处理逻辑

- 在 `process_file` 方法中，增加了对文件哈希值的计算和向量数据库的检查，避免重复处理已存在的文件。
- 使用 `RecursiveCharacterTextSplitter` 对提取的文本进行合理的分块处理，确保向量数据库的有效性和查询性能。
- 在处理新上传的文件时，先将文件内容写入指定路径，再进行后续的解析和向量存储操作，确保文件的完整性。

### Files2ElasticSearch 代码编写

#### 代码设计

`Files2ElasticSearch` 的目标是将文件解析后切片存储到 `Elasticsearch` 索引中，结构和 `Files2VectorDB` 类似。

- **类初始化：** `FileToElasticSearch` 类在初始化时接收 `Elasticsearch` 的主机地址、端口、索引名称及嵌入模型等参数。
初始化时自动检查并创建所需的索引。

```python
file_to_es = FileToElasticSearch(
    es_host="localhost",
    es_port=9200,
    index_name="documents",
    embedding_model="qwen:7b"
)
```

- **主要方法说明：**
    - `process_file(file_path: str)`： 解析单个文件，按块存储到指定的 `Elasticsearch` 索引中。
    - `process_directory(directory_path: str)`： 批量处理目录中的所有文件。
    - 内部解析方法（如 `parse_adoc`, `parse_markdown`）直接复用 `Files2VectorDB` 中的实现。

#### **遇到的问题及解决方案**

1. **Elasticsearch 配置问题**
    - **问题描述：**  
     初始配置中启用了 HTTPS，但未正确生成自签名证书，导致客户端连接时报错 `[SSL: WRONG_VERSION_NUMBER]` 或 `[SSL: CERTIFICATE_VERIFY_FAILED]`。  
    - **解决方案：**  
        - 通过 `elasticsearch-certutil http` 重新生成自签名证书 `http.p12`。
        - 在客户端代码中禁用证书验证，仅用于测试环境：
            ```python
            from elasticsearch import Elasticsearch
            es_client = Elasticsearch(
                [{"host": "localhost", "port": 9200, "scheme": "https"}],
                verify_certs=False
            )
            ```
        - 确保 Elasticsearch 的 `elasticsearch.yml` 配置文件正确启用了 HTTPS。
            ```yaml
            xpack.security.http.ssl.enabled: true
            xpack.security.http.ssl.keystore.path: certs/http.p12
            ```
    - **临时解决方案：**
        为了避免密钥上传至公网，由于 Elasticsearch 服务器运行在 localhost，因此修改配置为 http 且禁用身份验证机制。如果环境改为正式生产，需要避免使用该解决方案。

## 杂项问题

### 用户不在 sudo group 中

为了更清晰的权限管理，在开发服务器上新建用户，并使用

```bash
usermod -aG sudo
```

添加权限。

然后新登录一个 shell，尝试 `sudo visudo` 命令，出现报错：

```
*** is not in the sudoers file.  This incident will be reported.
```

检查相关权限文件，未找到问题。采用 root 登陆再降级到 rx 用户解决。

### 无法安装 Anaconda

采用 Anaconda 官网链接，使用 wget 下载其安装 shell 脚本，安装过程出现报错：

```
    RuntimeError: Failed to extract
    .conda: 'ascii' codec can't encode character '\xe4' in position 93: ordinal not in range(128)
```

检查路径中是否含有特殊字符或非 ASCII 编码字符，未发现问题；尝试修改安装路径，在 bashrc 中添加编码相关选项，未成功。

解决：改为安装 Miniconda，安装成功。