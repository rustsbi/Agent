import copy
import os
import sys
import time
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
print(root_dir)
from src.utils.general_utils import get_time, num_tokens_embed, \
     clear_string
from typing import List
from src.configs.configs import DEFAULT_CHILD_CHUNK_SIZE, \
      UPLOAD_ROOT_PATH, SEPARATORS, DEFAULT_PARENT_CHUNK_SIZE
from langchain.docstore.document import Document
from src.utils.log_handler import insert_logger
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import threading
import re
import traceback
from sanic.request import File
import chardet

# TODO同名文件直接覆盖
class LocalFile:
    def __init__(self, user_id, kb_id, file: File, file_name):
        self.user_id = user_id
        self.kb_id = kb_id
        self.file_name = file_name
        self.file_id = uuid.uuid4().hex
        self.file_content = file.body
        upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
        file_dir = os.path.join(upload_path, self.kb_id)
        os.makedirs(file_dir, exist_ok=True)
        self.file_location = os.path.join(file_dir, self.file_name)
        with open(self.file_location, 'wb') as f:
            f.write(self.file_content)

class FileHandler:
    def __init__(self, user_id, kb_name,kb_id, file_id, file_location, file_name, chunk_size):
        self.chunk_size = chunk_size
        self.user_id = user_id
        self.kb_name = kb_name
        self.kb_id = kb_id
        self.file_id = file_id
        self.docs: List[Document] = []
        self.embs = []
        self.file_name = file_name
        self.file_location = file_location
        self.file_path = ""
        self.file_path = self.file_location
        self.event = threading.Event()


    @staticmethod
    def get_page_id(doc, pre_page_id):
        # 查找 page_id 标志行
        lines = doc.page_content.split('\n')
        for line in lines:
            if re.match(r'^#+ 当前页数:\d+$', line):
                try:
                    page_id = int(line.split(':')[-1])
                    return page_id
                except ValueError:
                    continue
        return pre_page_id

    @staticmethod
    def load_text(file_path):
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']

        for encoding in encodings:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()
                insert_logger.info(f"TextLoader {encoding} success: {file_path}")
                return docs
            except Exception:
                insert_logger.warning(f"TextLoader {encoding} error: {file_path}, {traceback.format_exc()}")

        insert_logger.error(f"Failed to load file with all attempted encodings: {file_path}")
        return []
    # 将file文件变成Document类型
    @get_time
    def split_file_to_docs(self):
        # 处理txt
        if self.file_path.lower().endswith(".txt"):
            docs = self.load_text(self.file_path)
        # 
        else:
            raise TypeError("文件类型不支持，目前仅支持：[txt]")
        # 注入一些属性后，保存在self.docs中 
        self.inject_metadata(docs)

    def inject_metadata(self, docs: List[Document]):
        # 这里给每个docs片段的metadata里注入file_id
        new_docs = []
        for doc in docs:
            page_content = re.sub(r'\t+', ' ', doc.page_content)  # 将制表符替换为单个空格
            page_content = re.sub(r'\n{3,}', '\n\n', page_content)  # 将三个或更多换行符替换为两个
            page_content = page_content.strip()  # 去除首尾空白字符
            new_doc = Document(page_content=page_content)
            new_doc.metadata["user_id"] = self.user_id
            new_doc.metadata["kb_id"] = self.kb_id
            new_doc.metadata["file_id"] = self.file_id
            new_doc.metadata["file_name"] = self.file_name
            new_doc.metadata["nos_key"] = self.file_location
            new_doc.metadata["title_lst"] = doc.metadata.get("title_lst", [])
            new_doc.metadata["page_id"] = doc.metadata.get("page_id", 0)
            new_doc.metadata["has_table"] = doc.metadata.get("has_table", False)
            new_doc.metadata["images"] = re.findall(r'!\[figure]\(\d+-figure-\d+.jpg.*?\)', page_content)
            metadata_infos = {"知识库名": self.kb_name, '文件名': self.file_name}
            new_doc.metadata['headers'] = metadata_infos

            if 'faq_dict' not in doc.metadata:
                new_doc.metadata['faq_dict'] = {}
            else:
                new_doc.metadata['faq_dict'] = doc.metadata['faq_dict']
            new_docs.append(new_doc)
        if new_docs:
            insert_logger.info('langchain analysis content head: %s', new_docs[0].page_content[:100])
        else:
            insert_logger.info('langchain analysis docs is empty!')

        # 合并短的document
        insert_logger.info(f"before merge doc lens: {len(new_docs)}")
        child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(self.chunk_size / 2))
        merged_docs = []
        for doc in new_docs:
            if not merged_docs:
                merged_docs.append(doc)
            else:
                last_doc = merged_docs[-1]
                if num_tokens_embed(last_doc.page_content) + num_tokens_embed(doc.page_content) <= child_chunk_size or \
                        num_tokens_embed(doc.page_content) < child_chunk_size / 4:
                    tmp_content_slices = doc.page_content.split('\n')
                    tmp_content_slices_clear = [line for line in tmp_content_slices if clear_string(line) not in
                                                [clear_string(t) for t in last_doc.metadata['title_lst']]]
                    tmp_content = '\n'.join(tmp_content_slices_clear)
                    last_doc.page_content += '\n\n' + tmp_content
                    last_doc.metadata['title_lst'] += doc.metadata.get('title_lst', [])
                    last_doc.metadata['has_table'] = last_doc.metadata.get('has_table', False) or doc.metadata.get(
                        'has_table', False)
                    last_doc.metadata['images'] += doc.metadata.get('images', [])
                else:
                    merged_docs.append(doc)
        insert_logger.info(f"after merge doc lens: {len(merged_docs)}")
        self.docs = merged_docs
    
    # TODO：可以异步进行
    @staticmethod
    def split_docs(docs: List[Document], parent_chunk_size=DEFAULT_PARENT_CHUNK_SIZE):
        # parent chunk size 默认是800
        parent_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            length_function=num_tokens_embed)
        # # This text splitter is used to create the child documents
        # # It should create documents smaller than the parent
        # child chunk size 默认是400 其中重叠部分长度为 100
        child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(parent_chunk_size / 2))
        child_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=child_chunk_size,
            chunk_overlap=int(child_chunk_size / 3),
            length_function=num_tokens_embed,
            keep_separator=True)
        # 先处理父文档，父文档没有重叠部分每一个都是单独的
        # documents = self.parent_splitter.split_documents(documents)
        split_documents = []
        need_split_docs = []
        # 高效的方式，如果需要切分的文档多的话
        # 当遇到不需要分割的文档时，才对之前收集的need_split_docs进行分割处理
        # 比如处理10个需要分割的文档：
        #       这种写法只会调用一次split_documents
        for doc in docs:
            if doc.metadata['has_table'] or num_tokens_embed(doc.page_content) <= parent_chunk_size:
                if need_split_docs:
                    split_documents.extend(parent_splitter.split_documents(need_split_docs))
                    need_split_docs = []
                split_documents.append(doc)
            else:
                need_split_docs.append(doc)
        if need_split_docs:
            split_documents.extend(parent_splitter.split_documents(need_split_docs))
        insert_logger.info(f"Inserting {len(split_documents)} parent documents")
        file_id = split_documents[0].metadata['file_id']
        doc_ids = [file_id + '_' + str(i) for i, _ in enumerate(split_documents)]
        # 是否加入全文数据库中
        # if not add_to_docstore:
        #     raise ValueError(
        #         "If ids are not passed in, `add_to_docstore` MUST be True"
        #     )
        documents = []
        full_documents = []
        for i, doc in enumerate(split_documents):
            _id = doc_ids[i]
            sub_docs = child_splitter.split_documents([doc])
            for _doc in sub_docs:
                # 为每个子片段添加主片段id
                _doc.metadata["doc_id"] = _id
                # _doc.page_content = f"[headers]({_doc.metadata['headers']})\n" + _doc.page_content  # 存入page_content，向量检索时会带上headers
            documents.extend(sub_docs)
            # TODO: 先不加下面的，后面要用再说
            # doc.page_content = f"[headers]({doc.metadata['headers']})\n" + doc.page_content  # 存入page_content，等检索后rerank时会带上headers信息
            # 用来存储每个完整的片段
            full_documents.append((_id, doc))
        insert_logger.info(f"Inserting {len(documents)} child documents, metadata: {documents[0].metadata}, page_content: {docs[0].page_content[:100]}...")
        embed_documents = copy.deepcopy(documents)
        # 补充metadata信息
        for doc in embed_documents:
            del doc.metadata['title_lst']
            del doc.metadata['has_table']
            del doc.metadata['images']
            del doc.metadata['file_name']
            del doc.metadata['nos_key']
            del doc.metadata['page_id']
        # full_documents用来记录每一个父片段
        # 返回可以通用
        return embed_documents, full_documents




