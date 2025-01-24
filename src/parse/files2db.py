# edited by kjn 12.23
# 程序功能：解析files文件夹中的文件，同时将文件切片通过faiss构建向量数据库，存入db文件夹中
import hashlib
import os
import subprocess
import time
import threading
from markdown import markdown
from pylatexenc.latex2text import LatexNodes2Text

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS  # 使用 FAISS

from bs4 import BeautifulSoup
from docutils.core import publish_parts

import re

def parse_adoc(file_path):
    """Parse Asciidoc file and split content into chunks based on structure."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

    chunks = []
    current_chunk = []
    current_title = None

    for line in lines:
        # Ignore include directives and blank lines
        if line.strip().startswith("include::") or not line.strip():
            continue

        # Detect titles (e.g., = Title, == Subtitle)
        title_match = re.match(r"^(=+)\s+(.*)", line)
        if title_match:
            # Save the previous chunk if any
            if current_chunk:
                chunks.append({
                    "title": current_title or {"level": 0, "title": "No Title"},
                    "content": "\n".join(current_chunk)
                })
                current_chunk = []

            # Update current title
            level, title = title_match.groups()
            current_title = {"level": len(level), "title": title.strip()}
        else:
            # Append content to the current chunk
            current_chunk.append(line.strip())

    # Save the last chunk
    if current_chunk:
        chunks.append({
            "title": current_title or {"level": 0, "title": "No Title"},
            "content": "\n".join(current_chunk)
        })

    return chunks


def parse_markdown(file_path):
    """Process Markdown file into plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return markdown(content)


def parse_latex(file_path):
    """Process LaTeX file into plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return LatexNodes2Text().latex_to_text(content)


def parse_asciidoc(file_path):
    """Extract content from Asciidoc file and split by structure."""
    try:
        chunks = parse_adoc(file_path)
        print("Parsed chunks:", chunks)
        # Combine chunks for embedding (preserve structure with a fallback for missing titles)
        combined_chunks = [
            f"{chunk['title'].get('title', 'No Title')}\n\n{chunk['content']}"
            for chunk in chunks
        ]
        return combined_chunks
    except Exception as e:
        print(f"Failed to process Asciidoc file: {e}")
        return []

def parse_rst(file_path):
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # document = publish_parts(content, writer_name='html')
        
        # soup = BeautifulSoup(document['html_body'], 'html.parser')
        # text = soup.get_text()
        # return text

        current_lines = []
        current_title = ""

        for content_line in content.split('\n'):
            # 去除回车行
            if content_line.strip() == '':
                continue
            title_match = re.search(r'^={3,}$', content_line)
            subtitle_match = re.search(r'^-{3,}$', content_line)
            if title_match or subtitle_match:
                if title_match:
                    level = 1
                else:
                    level = 2
                if len(current_lines) > 0:
                    new_title = current_lines[-1]
                    current_lines.pop()
                    if len(current_lines) > 0:
                        new_string = "\n".join(current_lines)
                        chunks.append({'title': current_title,'level': level,'content': new_string})
                    current_title = new_title
                    current_lines = []
            else:
                current_lines.append(content_line)

        if len(current_lines) > 0:
            new_string = "\n".join(current_lines)
            chunks.append({'title': current_title,'level': 1,'content': new_string})

        text_data = [f"{item['title']} (Level {item['level']}): {item['content']}" for item in chunks if 'title' in item and 'content' in item and 'level' in item]
        return text_data


    except Exception as e:
        print(f"Error reading .rst file: {e}")
        return ""

def parse_txt(file_path):
    """Read and return the content of a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Failed to process TXT file: {e}")
        return ""

def file2db(uploaded_file, filename):
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)  # 重置文件指针
    print(f"Processing {filename} to db...")
    # file_path = f"files/{file_hash}.{uploaded_file.name.split('.')[-1]}"
    file_path = uploaded_file.name
    db_root = "db"
    db_path = os.path.join(db_root, file_hash)  # 针对每个文件的独立子目录
    # 获取文件扩展名
    file_extension = os.path.splitext(filename)[-1].lower()
    if file_extension not in [ ".pdf", ".md", ".tex", ".adoc", ".txt", ".rst" ]:
        print(f"Unsupported file type: {file_extension}")
        return
    embedding_path = os.path.join(db_path, 'index.pkl')
    os.makedirs(db_path, exist_ok=True)
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    # 检查向量数据库是否存在
    if os.path.exists(db_path) and os.path.exists(embedding_path):
        print(f"Vector database for {file} already exists.")
    else:
        # 处理新上传的文件
        bytes_data = uploaded_file.read()
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        # 根据文件扩展名解析文件内容
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            data = loader.load()
            text_data = "\n".join([doc.page_content for doc in data])
        elif file_extension == ".md":
            text_data = parse_markdown(file_path)
        elif file_extension == ".tex":
            text_data = parse_latex(file_path)
        elif file_extension == ".adoc":
            text_data = parse_asciidoc(file_path)
        elif file_extension == ".rst":
            text_data = parse_rst(file_path)
        elif file_extension == ".txt":
            text_data = parse_txt(file_path)
        else:
            print("Unsupported file format.")
            return 

        # 如果返回值是列表，则拼接为单个字符串
        if isinstance(text_data, list):
            text_data = "\n".join(text_data)

        # 检查是否成功解析文本
        if not text_data.strip():
            print("No text could be extracted from the file. Please check the content.")

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, length_function=len
        )
        all_splits = text_splitter.split_text(text_data)

        # 检查分块是否成功
        if not all_splits:
            print("Failed to split the text into chunks. Please check the file content.")

        # Create and save the vector store
        Vectorstore = FAISS.from_texts(all_splits, embeddings)
        # 保存向量数据库
        Vectorstore.save_local(db_path)
        print(f"Vector database for {filename} has been saved.")

current_dir = os.path.dirname(os.path.abspath(__file__))
files_path = os.path.join(current_dir, "files")

if not os.path.exists("files"):
    os.mkdir("files")

if not os.path.exists("db"):
    os.mkdir("db")

files = os.listdir(files_path)

for file in files:
    file_path = os.path.join(files_path, file)
    print(file_path)
    with open(file_path, 'rb') as uploaded_file:
        file2db(uploaded_file, file)