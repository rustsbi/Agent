# 5月工作记录

5月的工作主要是将Agent之前的服务从外部迁移到本地来，保证rag系统的稳定运行

## mysql服务

```bash
(base) root@451f9b706ca3:/home# service mysql start
 * Starting MySQL database server mysqld          [fail] 
```

启动显示失败的原因是 MySQL 服务可能已经处于运行状态
```bash
mysql -h localhost -P 3306 -u root -p
Enter password:123456
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 8
Server version: 8.0.40 MySQL Community Server - GPL
mysql>
```

输入password后，即可启动mysql服务。

在mysql中可以看到当前的数据库：

```bash
mysql> USE rustsbi_rag;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
mysql> SELECT * FROM KnowledgeBase;
+----+------------------------------------+---------------+---------+---------+----------------+--------------------+
| id | kb_id                              | user_id       | kb_name | deleted | latest_qa_time | latest_insert_time |
+----+------------------------------------+---------------+---------+---------+----------------+--------------------+
|  1 | KBb7eafaabc07743b6a7a0a4119283b1e6 | abc1234__5678 | zzh     |       0 | NULL           | NULL               |
|  2 | KB2ed627becda34af0a85cb1d104d90ebb | abc1234__5678 | zzh     |       0 | NULL           | NULL               |
+----+------------------------------------+---------------+---------+---------+----------------+--------------------+
2 rows in set (0.00 sec)
```

## elastic服务

服务器之前通过python包安装了elasticsearch，通过
```bash
sudo find / -name elasticsearch
```

可以找到elasticsearch的安装位置，以及二进制文件存放的位置。

这里我的路径为`/home/kjn/elastic/elasticsearch-8.17.0/bin/elasticsearch`

由于 Elasticsearch 默认不允许以 root 用户身份运行，因为这可能会带来安全风险。因此，我们需要新建一个非root用户，命名为 es 。

同时记得让es用户拥有执行文件的权限，具体命令如下：

```bash
$ adduser es
$ chown -R es /home/kjn/elastic/elasticsearch-8.17.0/bin/elasticsearch
$ sudo chmod 755 /home/kjn 或者 sudo chown es:es /home/kjn
$ su es
es@451f9b706ca3:~$ /home/kjn/elastic/elasticsearch-8.17.0/bin/elasticsearch
```
即可启动成功。后续貌似会一直在后台挂着，无须重复启动。

## milvus服务

参考[官方仓库](https://github.com/milvus-io/milvus)的信息，milvus主流的安装方式是使用docker，但是由于服务器本身是一个docker，docker-in-docker的操作比较复杂。

尝试寻找源码编译的方法，但是[官方发行包](https://github.com/milvus-io/milvus/releases)中只给出了docker-compose配置文件以及docker安装的源码，貌似没有给出源码编译的安装方式。

于是最终使用pytyon提供的milvus包进行安装，参考[如何在本地部署milvus服务器（无需docker）](https://blog.csdn.net/wjylovewjy/article/details/147279935)，需要使用pip下载pymilvus，milvus，然后在/home下拉取文件并启动milvus服务，具体指令如下：

```bash
pip install pymilvus
pip install milvus
milvus-server --data /home/milvus
```

启动sanic等服务，测试milvus服务，执行Agent/src/evaluation/test_local_doc_chat.py。

期间可能会遇上`pymilvus.exceptions.ConnectionNotExistException`的报错，这是milvus的接口连接不上的问题，在保持milvus启动的条件下，重新启动sanic即可解决。输出如下：

```bash
(qanything) root@451f9b706ca3:/home/kjn/Agent/src/evaluation# python test_local_doc_chat.py 
{'code': 200, 'msg': 'success no stream chat', 'question': '文中提到的“妖童媛女”在做什么？', 'response': 'data: {"answer": "根据参考信息，文中提到的“妖童媛女”在荡舟嬉游。具体来说，他们荡舟、传杯、唱艳歌，且举止轻盈，展现了当时嬉游的光景。这些描述出自《采莲赋》中的描写。"}', 'model': '/home/model/Qwen2.5-32B-Instruct-AWQ', 'history': [['文中提到的“妖童媛女”在做什么？', '根据参考信息，文中提到的“妖童媛女”在荡舟嬉游。具体来说，他们荡舟、传杯、唱艳歌，且举止轻盈，展现了当时嬉游的光景。这些描述出自《采莲赋》中的描写。']], 'condense_question': '文中提到的“妖童媛女”在做什么？', 'source_documents': [{'file_id': '6c11303f7bc14f1dac61756bf24a2f6c', 'file_name': '', 'content': '于是妖童媛女⑿，荡舟心许；鷁首⒀徐回，兼传羽杯⒁；棹⒂将移而藻挂，船欲动而萍开。尔其纤腰束素⒃，迁延顾步⒄；夏始春余，叶嫩花初，恐沾裳而浅笑，畏倾船而敛裾⒅。\n可见当时嬉游的光景了。这真是有趣的事，可惜我们现在早已无福消受了。\n于是又记起，《西洲曲》里的句子：\n采莲南塘秋，莲花过人头；低头弄莲子，莲子清如水。\n今晚若有采莲人，这儿的莲花也算得“过人头”了；只不见一些流水的影子，是不行的。这令我到底惦着江南了。——这样想着，猛一抬头，不觉已是自己的门前；轻轻地推门进去，什么声息也没有，妻已睡熟好久了。\n一九二七年七月，北京清华园。', 'retrieval_query': '文中提到的“妖童媛女”在做什么？', 'file_url': '', 'score': '0.9', 'embed_version': '', 'nos_keys': '', 'doc_id': '6c11303f7bc14f1dac61756bf24a2f6c_1', 'retrieval_source': 'milvus', 'headers': [{'知识库名': 'zzh', '文件名': '这是一个测试文件.txt'}], 'page_id': 0}, {'file_id': '6c11303f7bc14f1dac61756bf24a2f6c', 'file_name': '', 'content': '荷塘的四面，远远近近，高高低低都是树，而杨柳最多。这些树将一片荷塘重重围住；只在小路一旁，漏着几段空隙，像是特为月光留下的。树色一例是阴阴的，乍看像一团烟雾；但杨柳的丰姿⑽，便在烟雾里也辨得出。树梢上隐隐约约的是一带远山，只有些大意罢了。树缝里也漏着一两点路灯光，没精打采的，是渴睡⑾人的眼。这时候最热闹的，要数树上的蝉声与水里的蛙声；但热闹是它们的，我什么也没有。\n忽然想起采莲的事情来了。采莲是江南的旧俗，似乎很早就有，而六朝时为盛；从诗歌里可以约略知道。采莲的是少年的女子，她们是荡着小船，唱着艳歌去的。采莲人不用说很多，还有看采莲的人。那是一个热闹的季节，也是一个风流的季节。梁元帝《采莲赋》里说得好：\n于是妖童媛女⑿，荡舟心许；鷁首⒀徐回，兼传羽杯⒁；棹⒂将移而藻挂，船欲动而萍开。尔其纤腰束素⒃，迁延顾步⒄；夏始春余，叶嫩花初，恐沾裳而浅笑，畏倾船而敛裾⒅。\n可见当时嬉游的光景了。这真是有趣的事，可惜我们现在早已无福消受了。', 'retrieval_query': '文中提到的“妖童媛女”在做什么？', 'file_url': '', 'score': '0.85', 'embed_version': '', 'nos_keys': '', 'doc_id': '6c11303f7bc14f1dac61756bf24a2f6c_1', 'retrieval_source': 'milvus', 'headers': [{'知识库名': 'zzh', '文件名': '这是一个测试文件.txt'}], 'page_id': 0}], 'retrieval_documents': [{'file_id': '6c11303f7bc14f1dac61756bf24a2f6c', 'file_name': '', 'content': '于是妖童媛女⑿，荡舟心许；鷁首⒀徐回，兼传羽杯⒁；棹⒂将移而藻挂，船欲动而萍开。尔其纤腰束素⒃，迁延顾步⒄；夏始春余，叶嫩花初，恐沾裳而浅笑，畏倾船而敛裾⒅。\n可见当时嬉游的光景了。这真是有趣的事，可惜我们现在早已无福消受了。\n于是又记起，《西洲曲》里的句子：\n采莲南塘秋，莲花过人头；低头弄莲子，莲子清如水。\n今晚若有采莲人，这儿的莲花也算得“过人头”了；只不见一些流水的影子，是不行的。这令我到底惦着江南了。——这样想着，猛一抬头，不觉已是自己的门前；轻轻地推门进去，什么声息也没有，妻已睡熟好久了。\n一九二七年七月，北京清华园。', 'retrieval_query': '文中提到的“妖童媛女”在做什么？', 'file_url': '', 'score': '0.9', 'embed_version': '', 'nos_keys': '', 'doc_id': '6c11303f7bc14f1dac61756bf24a2f6c_1', 'retrieval_source': 'milvus', 'headers': [{'知识库名': 'zzh', '文件名': '这是一个测试文件.txt'}], 'page_id': 0}, {'file_id': '6c11303f7bc14f1dac61756bf24a2f6c', 'file_name': '', 'content': '荷塘的四面，远远近近，高高低低都是树，而杨柳最多。这些树将一片荷塘重重围住；只在小路一旁，漏着几段空隙，像是特为月光留下的。树色一例是阴阴的，乍看像一团烟雾；但杨柳的丰姿⑽，便在烟雾里也辨得出。树梢上隐隐约约的是一带远山，只有些大意罢了。树缝里也漏着一两点路灯光，没精打采的，是渴睡⑾人的眼。这时候最热闹的，要数树上的蝉声与水里的蛙声；但热闹是它们的，我什么也没有。\n忽然想起采莲的事情来了。采莲是江南的旧俗，似乎很早就有，而六朝时为盛；从诗歌里可以约略知道。采莲的是少年的女子，她们是荡着小船，唱着艳歌去的。采莲人不用说很多，还有看采莲的人。那是一个热闹的季节，也是一个风流的季节。梁元帝《采莲赋》里说得好：\n于是妖童媛女⑿，荡舟心许；鷁首⒀徐回，兼传羽杯⒁；棹⒂将移而藻挂，船欲动而萍开。尔其纤腰束素⒃，迁延顾步⒄；夏始春余，叶嫩花初，恐沾裳而浅笑，畏倾船而敛裾⒅。\n可见当时嬉游的光景了。这真是有趣的事，可惜我们现在早已无福消受了。', 'retrieval_query': '文中提到的“妖童媛女”在做什么？', 'file_url': '', 'score': '0.85', 'embed_version': '', 'nos_keys': '', 'doc_id': '6c11303f7bc14f1dac61756bf24a2f6c_1', 'retrieval_source': 'milvus', 'headers': [{'知识库名': 'zzh', '文件名': '这是一个测试文件.txt'}], 'page_id': 0}], 'time_record': {'time_usage': {'preprocess': 0.01, 'retriever_search_by_milvus': 0.03, 'retriever_search': 0.03, 'rerank': 0.7, 'reprocess': 0.01, 'llm_first_return': 4.7}, 'token_usage': {'prompt_tokens': 2997, 'completion_tokens': 120, 'total_tokens': 3117}}}


文中提到的“妖童媛女”在做什么？
data: {"answer": "根据参考信息，文中提到的“妖童媛女”在荡舟嬉游。具体来说，他们荡舟、传杯、唱艳歌，且举止轻盈，展现了当时嬉游的光景。这些描述出自《采莲赋》中的描写。"}
```

表明所有的服务已经成功迁移，Agent可以正常运行。

## configs更新

针对迁移的部署的服务，需要相应更新参数配置：
```py
# MYSQL_HOST_LOCAL="k8s.richeyjang.com"
# MYSQL_PORT_LOCAL="30303"
# MYSQL_USER_LOCAL="root"
# MYSQL_PASSWORD_LOCAL="123456"
# MYSQL_DATABASE_LOCAL="rustsbi_rag"
MYSQL_HOST_LOCAL = "localhost"  # 修改为本地地址
MYSQL_PORT_LOCAL = "3306"       # MySQL 默认端口
MYSQL_USER_LOCAL = "root"
MYSQL_PASSWORD_LOCAL = "123456"
MYSQL_DATABASE_LOCAL = "rustsbi_rag"

# ES_USER="elastic"
# ES_PASSWORD="XXIFkcJyTX=O2fgqLr=T"
# ES_URL="https://k8s.richeyjang.com:30301"
# ES_INDEX_NAME='rustsbi_es_index' + KB_SUFFIX
ES_USER = "elastic"
ES_PASSWORD = "XXIFkcJyTX=O2fgqLr=T"
ES_URL = "http://localhost:9200"  # 修改为本地地址和默认端口
ES_INDEX_NAME = 'rustsbi_es_index' + KB_SUFFIX

# MILVUS_HOST_LOCAL = "k8s.richeyjang.com"
# MILVUS_PORT = 30300
# MILVUS_COLLECTION_NAME = 'rustsbi_collection' + KB_SUFFIX
MILVUS_HOST_LOCAL = "localhost"  # 修改为本地地址
MILVUS_PORT = 19530
MILVUS_COLLECTION_NAME = 'rustsbi_collection' + KB_SUFFIX
```

## 其他工作

1. 改进了vllm_start.sh，增加了及时清理log的操作。
2. 在/milvus下编写了测试milvus连接的程序
3. 更新了api_client.py