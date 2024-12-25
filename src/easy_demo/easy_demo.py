import hashlib
import os
import subprocess
import time
import threading
from markdown import markdown
from pylatexenc.latex2text import LatexNodes2Text

import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS  # 使用 FAISS

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


if not os.path.exists("files"):
    os.mkdir("files")

if not os.path.exists("db"):
    os.mkdir("db")

if "template" not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if "prompt" not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history", return_messages=True, input_key="question"
    )

# 生成模型
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="gemma2:27b",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG Chatbot")

# Upload a file
uploaded_file = st.file_uploader("Upload your file (PDF, Markdown, LaTeX, Asciidoc)", type=["pdf", "md", "tex", "adoc"])


def process_markdown(file_path):
    """Process Markdown file into plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return markdown(content)


def process_latex(file_path):
    """Process LaTeX file into plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return LatexNodes2Text().latex_to_text(content)


def process_asciidoc(file_path):
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
        st.error(f"Failed to process Asciidoc file: {e}")
        return []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    # 生成文件的唯一标识符
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)  # 重置文件指针

    # file_path = f"files/{file_hash}.{uploaded_file.name.split('.')[-1]}"
    file_path = f"files/{uploaded_file.name}"
    db_root = "db"
    db_path = os.path.join(db_root, file_hash)  # 针对每个文件的独立子目录
    embedding_path = os.path.join(db_path, 'index.pkl')
    os.makedirs(db_path, exist_ok=True)
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="qwen:7b")

    # 检查向量数据库是否存在
    if os.path.exists(db_path) and os.path.exists(embedding_path):
        st.write("Loading existing vector database...")
        st.session_state.vectorstore = FAISS.load_local(db_path, embeddings,allow_dangerous_deserialization=True)
        st.write("Done!")
    else:
        # 处理新上传的文件
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            with open(file_path, "wb") as f:
                f.write(bytes_data)

            print(uploaded_file.type)

            # 获取文件扩展名
            file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

            # 根据文件扩展名解析文件内容
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                data = loader.load()
                text_data = "\n".join([doc.page_content for doc in data])
            elif file_extension == ".md":
                text_data = process_markdown(file_path)
            elif file_extension == ".tex":
                text_data = process_latex(file_path)
            elif file_extension == ".adoc":
                text_data = process_asciidoc(file_path)
            else:
                st.error("Unsupported file format.")
                st.stop()

            # 如果返回值是列表，则拼接为单个字符串
            if isinstance(text_data, list):
                text_data = "\n".join(text_data)

            # 检查是否成功解析文本
            if not text_data.strip():
                st.error("No text could be extracted from the file. Please check the content.")
                st.stop()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200, length_function=len
            )
            all_splits = text_splitter.split_text(text_data)

            # 检查分块是否成功
            if not all_splits:
                st.error("Failed to split the text into chunks. Please check the file content.")
                st.stop()

            # Create and save the vector store
            st.session_state.vectorstore = FAISS.from_texts(all_splits, embeddings)
            # 保存向量数据库
            st.session_state.vectorstore.save_local(db_path)

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            },
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            
            # 创建一个占位符，用于显示模拟输入效果
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response["result"].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            
            # 清除之前的模拟输入效果，并显示最终的格式化回答
            message_placeholder.empty()
            message_placeholder.markdown(response["result"])

        chatbot_message = {"role": "assistant", "message": response["result"]}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a file.")
