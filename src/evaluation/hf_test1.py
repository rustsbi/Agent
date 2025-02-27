import asyncio
from openai import OpenAI
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

sys.path.append(root_dir)

# 加载 RAG 后端接口
from src.utils.log_handler import debug_logger
from src.client.llm.llm_client import OpenAILLM

# 定义 Hugging Face 方法的评估函数
async def evaluate_rag_hf():
    # 1. 加载开源数据集（使用 SQuAD 数据集作为示例）
    print("Loading dataset...")
    dataset = load_dataset("squad", split="validation[:100]")  # 取前 100 条数据
    knowledge_base = [{"text": qa["context"], "source": f"sample-{i}"} for i, qa in enumerate(dataset)]

    # 2. 初始化检索器（FAISS 向量数据库）
    print("Building retriever...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, tokenizer=tokenizer)
    chunks = text_splitter.split_documents(knowledge_base)
    retriever = FAISS.from_documents(chunks)

    # 3. 初始化生成器（基于你提供的后端接口调用 OpenAI 模型）
    llm = OpenAILLM(
        model="gpt-4",
        max_token=1000,
        api_base="https://api.openai.com/v1",
        api_key="your_openai_api_key",
        api_context_length=4096,
        top_p=0.9,
        temperature=0.7
    )

    # 4. 对数据集进行评估
    print("Evaluating RAG system...")
    results = []
    for qa in dataset:
        question = qa["question"]
        true_answer = qa["answers"]["text"][0]

        # 检索阶段
        retrieved_docs = retriever.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # 生成阶段
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        async for response in llm.generatorAnswer(prompt=prompt):
            generated_answer = response.llm_output["answer"].strip()
            break  # 获取第一个生成结果即可

        # 保存结果
        results.append({"question": question, "true_answer": true_answer, "generated_answer": generated_answer})

    # 5. 输出结果
    for result in results[:5]:  # 打印前 5 个结果
        print(f"Question: {result['question']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"True Answer: {result['true_answer']}")
        print("-" * 50)

# 运行评估脚本
if __name__ == "__main__":
    asyncio.run(evaluate_rag_hf())
