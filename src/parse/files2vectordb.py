import hashlib
import os
import re
from typing import List, Dict, Union

from markdown import markdown
from pylatexenc.latex2text import LatexNodes2Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class FileToVectorDB:
    def __init__(self, db_root: str = "db", embedding_model: str = "qwen:7b"):
        """
        Initialize the FileToVectorDB class.

        :param db_root: Root directory for storing vector databases, default is "db".
        :param embedding_model: Embedding model to use, default is "qwen:7b".
        """
        self.db_root = db_root
        self.embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model)
        os.makedirs(self.db_root, exist_ok=True)

    def parse_adoc(self, file_path: str) -> List[Dict[str, Union[str, Dict]]]:
        """Parse an Asciidoc file and return chunked content."""
        print(f"Parsing Asciidoc file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")

        chunks = []
        current_chunk = []
        current_title = None

        for line in lines:
            # Ignore include directives and empty lines
            if line.strip().startswith("include::") or not line.strip():
                continue

            # Detect titles (e.g., = Title, == Subtitle)
            title_match = re.match(r"^(=+)\s+(.*)", line)
            if title_match:
                # Save the current chunk
                if current_chunk:
                    chunks.append({
                        "title": current_title or {"level": 0, "title": "No Title"},
                        "content": "\n".join(current_chunk)
                    })
                    current_chunk = []

                # Update the current title
                level, title = title_match.groups()
                current_title = {"level": len(level), "title": title.strip()}
            else:
                # Add content to the current chunk
                current_chunk.append(line.strip())

        # Save the last chunk
        if current_chunk:
            chunks.append({
                "title": current_title or {"level": 0, "title": "No Title"},
                "content": "\n".join(current_chunk)
            })
        print(f"Finished parsing Asciidoc file: {file_path}")
        return chunks

    def parse_markdown(self, file_path: str) -> str:
        """Parse a Markdown file and return plain text content."""
        print(f"Parsing Markdown file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        text = markdown(content)
        print(f"Finished parsing Markdown file: {file_path}")
        return text

    def parse_latex(self, file_path: str) -> str:
        """Parse a LaTeX file and return plain text content."""
        print(f"Parsing LaTeX file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        text = LatexNodes2Text().latex_to_text(content)
        print(f"Finished parsing LaTeX file: {file_path}")
        return text

    def parse_asciidoc(self, file_path: str) -> List[str]:
        """Parse an Asciidoc file and return chunked content."""
        print(f"Parsing Asciidoc file: {file_path}")
        try:
            chunks = self.parse_adoc(file_path)
            # Combine chunked content into a list of strings
            combined_chunks = [
                f"{chunk['title'].get('title', 'No Title')}\n\n{chunk['content']}"
                for chunk in chunks
            ]
            print(f"Finished parsing Asciidoc file: {file_path}")
            return combined_chunks
        except Exception as e:
            print(f"Failed to parse Asciidoc file: {e}")
            return []

    def parse_rst(self, file_path: str) -> List[str]:
        """Parse a reStructuredText file and return chunked content."""
        print(f"Parsing reStructuredText file: {file_path}")
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            current_lines = []
            current_title = ""

            for content_line in content.split('\n'):
                # Skip empty lines
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
                            chunks.append({'title': current_title, 'level': level, 'content': new_string})
                        current_title = new_title
                        current_lines = []
                else:
                    current_lines.append(content_line)

            if len(current_lines) > 0:
                new_string = "\n".join(current_lines)
                chunks.append({
                    'title': current_title,
                    'level': 1,
                    'content': new_string
                })

            text_data = [
                f"{item['title']} (Level {item['level']}): {item['content']}"
                for item in chunks
                if 'title' in item and 'content' in item and 'level' in item
            ]
            print(f"Finished parsing reStructuredText file: {file_path}")
            return text_data

        except Exception as e:
            print(f"Error reading .rst file: {e}")
            return []

    def parse_txt(self, file_path: str) -> str:
        """Parse a TXT file and return content."""
        print(f"Parsing TXT file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"Finished parsing TXT file: {file_path}")
            return content
        except Exception as e:
            print(f"Failed to parse TXT file: {e}")
            return ""

    def process_file(self, file_path: str):
        """Process a single file and build the vector database."""
        print(f"Processing file: {file_path}")
        file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()
        db_path = os.path.join(self.db_root, file_hash)
        embedding_path = os.path.join(db_path, 'index.pkl')

        # Check if the vector database already exists
        if os.path.exists(db_path) and os.path.exists(embedding_path):
            print(f"Vector database for {file_path} already exists. Skipping.")
            return

        # Get file extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension not in [".pdf", ".md", ".tex", ".adoc", ".txt", ".rst"]:
            print(f"Unsupported file type: {file_extension}")
            return

        # Parse the file content based on its extension
        if file_extension == ".pdf":
            print(f"Loading PDF file: {file_path}")
            loader = PyPDFLoader(file_path)
            data = loader.load()
            text_data = "\n".join([doc.page_content for doc in data])
            print(f"Finished loading PDF file: {file_path}")
        elif file_extension == ".md":
            text_data = self.parse_markdown(file_path)
        elif file_extension == ".tex":
            text_data = self.parse_latex(file_path)
        elif file_extension == ".adoc":
            text_data = self.parse_asciidoc(file_path)
        elif file_extension == ".rst":
            text_data = self.parse_rst(file_path)
        elif file_extension == ".txt":
            text_data = self.parse_txt(file_path)

        # If the return value is a list, concatenate into a single string
        if isinstance(text_data, list):
            text_data = "\n".join(text_data)

        # Check if text extraction was successful
        if not text_data.strip():
            print("Failed to extract text from the file. Please check the file content.")
            return

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        all_splits = text_splitter.split_text(text_data)
        print(f"Split the text into {len(all_splits)} chunks.")

        # Check if splitting was successful
        if not all_splits:
            print("Text splitting failed. Please check the file content.")
            return

        # Create and save the vector database
        print(f"Creating vector database for: {file_path}")
        vectorstore = FAISS.from_texts(all_splits, self.embeddings)
        vectorstore.save_local(db_path)
        print(f"Vector database saved: {file_path}")

    def process_directory(self, directory_path: str):
        """Process all files in a directory."""
        print(f"Processing directory: {directory_path}")
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                self.process_file(file_path)
        print(f"Finished processing directory: {directory_path}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process files and build a vector database.")
    parser.add_argument("--input", type=str, help="Path to the file or directory to process.")
    parser.add_argument("--db_root", type=str, default="db", help="Root directory for storing vector databases.")
    parser.add_argument("--embedding_model", type=str, default="qwen:7b",
                        help="Embedding model to use for generating vectors.")
    args = parser.parse_args()

    # Initialize FileToVectorDB class
    file_to_db = FileToVectorDB(db_root=args.db_root, embedding_model=args.embedding_model)

    # Process the input path
    if args.input:
        if os.path.isfile(args.input):
            file_to_db.process_file(args.input)
        elif os.path.isdir(args.input):
            file_to_db.process_directory(args.input)
        else:
            print(f"Invalid input path: {args.input}")
    else:
        # If no input argument is provided, process the example folder
        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        if not os.path.exists(example_dir):
            os.makedirs(example_dir)
            print(f"Created example directory: {example_dir}")
        file_to_db.process_directory(example_dir)