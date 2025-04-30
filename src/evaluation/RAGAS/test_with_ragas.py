import aiohttp
import asyncio
import json
import os
import sys
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas import evaluate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_evaluation.log')
    ]
)
logger = logging.getLogger('rag_evaluation')

class AsyncHTTPClient:
    """异步HTTP客户端，处理重试和超时"""
    def __init__(self, retries: int = 3, timeout: int = 60):
        self.retries = retries
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    async def request(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> Optional[dict]:
        """发送HTTP请求，处理重试和错误"""
        for attempt in range(self.retries):
            try:
                async with self.session.request(
                    method, 
                    url, 
                    timeout=self.timeout,
                    **kwargs
                ) as response:
                    if response.status in (200, 201):
                        content_type = response.headers.get('Content-Type', '')
                        
                        if 'application/json' in content_type:
                            return await response.json()
                        else:
                            # 尝试智能处理
                            text = await response.text()
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                return text
                    
                    # 服务器错误时重试
                    if response.status >= 500:
                        if attempt < self.retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                            
                    response.raise_for_status()
                    
            except Exception as e:
                if attempt < self.retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"All attempts failed for {url}: {str(e)}")
                    raise
        
        return None

class RAGEvaluator:
    """使用RAGAS评估RAG系统性能的类"""
    def __init__(self, api_base: str = "http://127.0.0.1:8777", 
                 user_id: str = 'rx01', user_info: str = '12345678', 
                 kb_id: str = None):
        self.url_base = api_base
        self.user_id = user_id
        self.user_info = user_info
        self.kb_id = kb_id
        self.evaluation_results = []
        
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """向RAG系统提问并获取回答"""
        if not self.kb_id:
            raise ValueError("Knowledge base ID not set.")
        
        logger.info(f"Asking question: {question}")
        
        async with AsyncHTTPClient(retries=3, timeout=60) as client:
            try:
                # 请求参数
                payload = {
                    "user_id": self.user_id,
                    "max_token": 3000,
                    "user_info": self.user_info,
                    "kb_ids": [self.kb_id],
                    "question": question,
                    "history": [],
                    "streaming": False,
                    "rerank": True,
                    "custom_prompt": None,
                    "api_base": "your_api_base",
                    "api_key": "your_api_key",
                    "api_context_length": 10000,
                    "top_p": 0.99,
                    "temperature": 0.7,
                    "top_k": 5
                }

                # 发送POST请求
                response = await client.request(
                    method="POST",
                    url=f"{self.url_base}/api/local_doc_qa/local_doc_chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info(f"Response received for question: {question}")
                
                # 提取回答和上下文
                answer = response.get('response', '')
                contexts = response.get('source_documents', [])
                context_text = '\n'.join([doc.get('page_content', '') for doc in contexts]) if contexts else ''
                
                return {
                    "question": question,
                    "answer": answer,
                    "contexts": context_text,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Error asking question: {str(e)}")
                return {
                    "question": question,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
    
    async def evaluate_with_ragas(self, dataset_path: str, ground_truth_included: bool = False) -> pd.DataFrame:
        """使用RAGAS评估RAG系统性能"""
        # 加载数据集
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            raise ValueError("Unsupported dataset format. Use JSON or CSV.")
        
        questions = []
        ground_truths = []
        answers = []
        contexts = []
        
        # 处理每个问题
        for item in data:
            if isinstance(item, dict):
                question = item.get('question', '')
                if not question:
                    continue
                
                # 获取RAG系统的回答
                result = await self.ask_question(question)
                
                if 'error' in result:
                    logger.warning(f"Error for question '{question}': {result['error']}")
                    continue
                
                questions.append(question)
                answers.append(result['answer'])
                contexts.append(result['contexts'])
                
                # 如果数据集包含参考答案
                if ground_truth_included and 'ground_truth' in item:
                    ground_truths.append(item['ground_truth'])
                else:
                    # 如果没有参考答案，RAGAS可以不使用ground_truth进行评估
                    ground_truths.append("")
        
        # 创建评估数据集
        eval_dataset = pd.DataFrame({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths if ground_truth_included else None
        })
        
        # 使用RAGAS评估
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        result = evaluate(
            eval_dataset, 
            metrics=metrics
        )
        
        # 保存评估结果
        self.evaluation_results = result
        
        return result
    
    def save_results(self, output_path: str) -> None:
        """保存评估结果到文件"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save.")
            return
        
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(self.evaluation_results)
        
        # 保存为CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Evaluation results saved to {output_path}")
        
        # 同时保存详细结果为JSON
        with open(output_path.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)

async def main():
    """主函数，处理命令行参数并执行评估"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG system using RAGAS')
    parser.add_argument('--api-base', type=str, default='http://127.0.0.1:8777',
                        help='API base URL')
    parser.add_argument('--user-id', type=str, default='rx01',
                        help='User ID')
    parser.add_argument('--user-info', type=str, default='12345678',
                        help='User info')
    parser.add_argument('--kb-id', type=str, default='TKB01',
                        help='Knowledge base ID')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to evaluation dataset (JSON or CSV)')
    parser.add_argument('--output', type=str, default='./ragas_results.csv',
                        help='Path to save evaluation results')
    parser.add_argument('--ground-truth', action='store_true',
                        help='Dataset includes ground truth answers')
    
    args = parser.parse_args()
    
    # 检查数据集文件是否存在
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at {args.dataset}")
        return 1
    
    # 创建并运行评估
    evaluator = RAGEvaluator(
        api_base=args.api_base,
        user_id=args.user_id,
        user_info=args.user_info,
        kb_id=args.kb_id
    )
    
    results = await evaluator.evaluate_with_ragas(
        args.dataset,
        ground_truth_included=args.ground_truth
    )
    
    # 保存结果
    evaluator.save_results(args.output)
    
    # 打印摘要结果
    print("\nRAGAS Evaluation Summary:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
