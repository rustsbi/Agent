import aiohttp
import asyncio
import json
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Any

usage_guide = '''
参数说明
- `--file`: file文件路径 (默认: ./file/riscv-privileged.pdf)
- `--questions`: 测试问题JSON文件路径 (默认: ./questions/test_questions.json)
- `--results`: 测试结果保存路径 (默认: ./results/test_results.json)
- `--api-base`: API服务器地址 (默认: http://127.0.0.1:8777)
- `--api-key`: API密钥 (默认: your_api_key)
- `--user-id`: 用户ID (默认: rx01)
- `--user-info`: 用户信息 (默认: rx for test)
- `--kb-name`: 知识库名称 (默认: TKB01)
- `--kb-id`: 知识库ID，如果提供则跳过创建知识库步骤

测试问题格式
JSON格式，包含问题字符串:
```json
[
"RISC-V架构的特点是什么？",
"RISC-V中的特权级别有哪些？",
"什么是机器模式(Machine Mode)？"
]
```
'''

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(root_dir)

from src.configs.configs import DEFAULT_API_BASE, DEFAULT_API_KEY

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
                      elif 'text/plain' in content_type:
                          return await response.text()
                      else:
                          # 尝试智能处理
                          text = await response.text()
                          try:
                              # 尝试解析为JSON
                              import json
                              return json.loads(text)
                          except json.JSONDecodeError:
                              # 如果不是JSON，返回原文本
                              return text
                  
                  # 服务器错误时重试
                  if response.status >= 500:
                      if attempt < self.retries - 1:
                          await asyncio.sleep(2 ** attempt)
                          continue
                          
                  response.raise_for_status()
                  
          except Exception as e:
              if attempt < self.retries - 1:
                  logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                  await asyncio.sleep(2 ** attempt)
              else:
                  logging.error(f"All attempts failed for {url}: {str(e)}")
                  raise
      
      return None

class SystemTest:
  """系统测试类，处理知识库创建、文件上传和问答测试"""
  def __init__(self, api_base: str = DEFAULT_API_BASE, api_key: str = DEFAULT_API_KEY, 
               user_id: str = 'rx01', user_info: str = '12345678', 
               kb_name: str = 'TKB01', kb_id: str = None):
      self.url_base = "http://127.0.0.1:8777"
      self.api_base = DEFAULT_API_BASE
      self.api_key = DEFAULT_API_KEY
      self.user_id = user_id
      self.user_info = user_info
      self.kb_name = kb_name
      self.kb_id = kb_id
      self.test_results = []
      
  async def create_knowledge_base(self) -> str:
      """创建新的知识库并返回kb_id"""
      logger.info(f"Creating knowledge base: {self.kb_name}")
      
      async with AsyncHTTPClient(retries=3, timeout=30) as client:
          try:
              data = await client.request(
                  'POST',
                  f'{self.url_base}/api/qa_handler/new_knowledge_base',
                  json={
                      'user_id': self.user_id, 
                      'user_info': self.user_info, 
                      'kb_name': self.kb_name
                  },
                  headers={'Content-Type': 'application/json'}
              )
              
              if isinstance(data, dict) and data.get('code') == 200:
                  self.kb_id = data.get('data', {}).get('kb_id')
                  logger.info(f"Knowledge base created with ID: {self.kb_id}")
                  return self.kb_id
              else:
                  logger.error(f"Failed to create knowledge base: {data}")
                  raise Exception(f"Failed to create knowledge base: {data}")
                  
          except Exception as e:
              logger.error(f"Error creating knowledge base: {str(e)}")
              raise
  
  async def upload_file(self, file_path: str) -> bool:
      """上传文件到知识库"""
      if not self.kb_id:
          raise ValueError("Knowledge base ID not set. Create knowledge base first.")
      
      if not os.path.exists(file_path):
          raise FileNotFoundError(f"File not found: {file_path}")
      
      logger.info(f"Uploading file: {file_path} to knowledge base: {self.kb_id}")
      
      # 不使用 AsyncHTTPClient 类的 request 方法，因为需要为每次重试创建新的 FormData
      session = aiohttp.ClientSession()
      try:
          retries = 3
          for attempt in range(retries):
              try:
                  # 每次重试都创建新的 FormData 对象
                  form_data = aiohttp.FormData()
                  with open(file_path, 'rb') as f:
                      form_data.add_field('files',
                                      f.read(),  # 读取文件内容而不是文件对象
                                      filename=os.path.basename(file_path),
                                      content_type='application/octet-stream')
                  
                  # 添加其他字段
                  form_data.add_field('user_id', self.user_id)
                  form_data.add_field('user_info', self.user_info)
                  form_data.add_field('kb_id', self.kb_id)
                  form_data.add_field('mode', 'soft')

                  # 发送请求
                  async with session.post(
                      f'{self.url_base}/api/qa_handler/upload_files',
                      data=form_data,
                      timeout=300
                  ) as response:
                      if response.status in (200, 201):
                          data = await response.json()
                          logger.info(f"File uploaded successfully: {data}")
                          return True
                      else:
                          error_text = await response.text()
                          logger.error(f"Failed to upload file (HTTP {response.status}): {error_text}")
                          if attempt < retries - 1:
                              await asyncio.sleep(2 ** attempt)
                          else:
                              return False
                              
              except Exception as e:
                  logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                  if attempt < retries - 1:
                      await asyncio.sleep(2 ** attempt)
                  else:
                      logger.error(f"Error uploading file: {str(e)}")
                      return False
                      
      finally:
          await session.close()
      
      return False
  
  def _extract_response_content(self, response_data: Any) -> str:
    """从响应数据中提取实际的回答内容"""
    if isinstance(response_data, dict):
      # 根据已知的响应格式直接提取
      if 'response' in response_data and isinstance(response_data['response'], str):
        response_str = response_data['response']
        
        # 处理 "data: {json}" 格式
        if response_str.startswith('data: '):
          json_str = response_str[6:]  # 去掉 "data: " 前缀
          try:
            import json
            data = json.loads(json_str)
            if isinstance(data, dict) and 'answer' in data:
              return data['answer']
          except json.JSONDecodeError:
            pass
        
        # 如果不是预期格式，直接返回原字符串
        return response_str
      
      # 备用方案：检查是否有嵌套的response字段
      elif 'response' in response_data and isinstance(response_data['response'], dict):
        nested_response = response_data['response']
        if 'response' in nested_response:
          return self._extract_response_content({'response': nested_response['response']})
      
      # 如果都没找到，尝试其他可能的字段
      possible_fields = ['answer', 'content', 'text', 'message', 'data']
      for field in possible_fields:
        if field in response_data:
          content = response_data[field]
          if isinstance(content, str):
            return content
      
      # 最后返回整个响应的字符串表示
      return str(response_data)
    
    elif isinstance(response_data, str):
      return response_data
    else:
      return str(response_data)
  
  def _print_qa_result(self, question: str, response_content: str, error: str = None):
      """格式化打印问答结果"""
      print("\n" + "="*80)
      print(f"问题: {question}")
      print("-"*80)
      
      if error:
          print(f"错误: {error}")
      else:
          print(f"回答: {response_content}")
      
      print("="*80)
  
  async def ask_question(self, question: str) -> Dict[str, Any]:
      """向知识库提问并获取回答"""
      if not self.kb_id:
          raise ValueError("Knowledge base ID not set. Create knowledge base first.")
      
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
                  "api_base": self.api_base,
                  "api_key": self.api_key,
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
              
              # 提取响应内容
              response_content = self._extract_response_content(response)
              
              # 在控制台打印问答结果
              self._print_qa_result(question, response_content)
              
              logger.info(f"Response received for question: {question}")
              
              result = {
                  "question": question,
                  "response": response,
                  "response_content": response_content,  # 添加提取的响应内容
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
              }
              
              return result
              
          except Exception as e:
              error_msg = str(e)
              logger.error(f"Error asking question: {error_msg}")
              
              # 在控制台打印错误信息
              self._print_qa_result(question, "", error_msg)
              
              return {
                  "question": question,
                  "error": error_msg,
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
              }
  
  async def run_test_questions(self, questions_file: str, results_file: str) -> None:
      """从JSON文件读取问题并保存结果"""
      if not os.path.exists(questions_file):
          raise FileNotFoundError(f"Questions file not found: {questions_file}")
      
      try:
          with open(questions_file, 'r', encoding='utf-8') as f:
              questions = json.load(f)
          
          if not isinstance(questions, list):
              raise ValueError("Questions file should contain a JSON array of questions")
              
          logger.info(f"Loaded {len(questions)} questions from {questions_file}")
          print(f"\n开始测试 {len(questions)} 个问题...")
          
          for i, question in enumerate(questions, 1):
              if isinstance(question, str):
                  print(f"\n[{i}/{len(questions)}] 正在处理问题...")
                  result = await self.ask_question(question)
                  self.test_results.append(result)
              else:
                  logger.warning(f"Skipping non-string question: {question}")
          
          # 确保结果目录存在
          os.makedirs(os.path.dirname(results_file), exist_ok=True)
          
          # 保存结果
          with open(results_file, 'w', encoding='utf-8') as f:
              json.dump(self.test_results, f, ensure_ascii=False, indent=2)
              
          logger.info(f"Test results saved to {results_file}")
          print(f"\n测试完成！结果已保存到: {results_file}")
          
          # 打印测试总结
          successful_tests = len([r for r in self.test_results if 'error' not in r])
          failed_tests = len(self.test_results) - successful_tests
          print(f"\n测试总结:")
          print(f"- 总问题数: {len(self.test_results)}")
          print(f"- 成功: {successful_tests}")
          print(f"- 失败: {failed_tests}")
          
      except Exception as e:
          logger.error(f"Error running test questions: {str(e)}")
          raise
  
  async def run_full_test(self, file_path: str, questions_file: str, results_file: str) -> None:
      """运行完整的系统测试流程"""
      try:
          print("开始系统测试...")
          
          # 1. 创建知识库（如果kb_id未提供）
          if not self.kb_id:
              print("正在创建知识库...")
              await self.create_knowledge_base()
          else:
              logger.info(f"Using existing knowledge base with ID: {self.kb_id}")
              print(f"使用现有知识库: {self.kb_id}")
          
          # 2. 上传文件
          print(f"正在上传文件: {file_path}")
          upload_success = await self.upload_file(file_path)
          if not upload_success:
              raise Exception("Failed to upload file")
          
          # 等待一段时间，确保文件处理完成
          print("等待文件处理完成...")
          await asyncio.sleep(10)
          
          # 3. 运行测试问题
          await self.run_test_questions(questions_file, results_file)
          
          logger.info("Full system test completed successfully")
          print("\n系统测试完成！")
          
      except Exception as e:
          logger.error(f"System test failed: {str(e)}")
          print(f"\n系统测试失败: {str(e)}")
          raise

async def main():
  """主函数，处理命令行参数并执行测试"""
  import argparse
  
  parser = argparse.ArgumentParser(description='Run system test for knowledge base QA')
  parser.add_argument('--file', type=str, default='./file/riscv-privileged.pdf',
                      help='Path to file')
  parser.add_argument('--questions', type=str, default='./questions/test_questions.json',
                      help='Path to questions JSON file')
  parser.add_argument('--results', type=str, default='./results/test_results.json',
                      help='Path to save results JSON file')
  parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE,
                      help='API base URL')
  parser.add_argument('--api-key', type=str, default=DEFAULT_API_KEY,
                      help='API key')
  parser.add_argument('--user-id', type=str, default='rx01',
                      help='User ID')
  parser.add_argument('--user-info', type=str, default='12345678',
                      help='User info')
  parser.add_argument('--kb-name', type=str, default='TKB01',
                      help='Knowledge base name')
  parser.add_argument('--kb-id', type=str, default=None,
                      help='Knowledge base ID (if provided, skip creating knowledge base)')
  
  args = parser.parse_args()
  
  # 检查文件是否存在
  if not os.path.exists(args.file):
      print(f"Error: File not found at {args.file}")
      return 1
  
  # 检查问题文件是否存在
  if not os.path.exists(args.questions):
      print(f"Error: Questions file not found at {args.questions}")
      return 1
  
  # 创建并运行系统测试
  test = SystemTest(
      api_base=args.api_base, 
      api_key=args.api_key,
      user_id=args.user_id,
      user_info=args.user_info,
      kb_name=args.kb_name,
      kb_id=args.kb_id
  )
  await test.run_full_test(args.file, args.questions, args.results)
  
  return 0

if __name__ == "__main__":
  try:
      # 配置日志
      logging.basicConfig(
          level=logging.INFO,
          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          handlers=[
              logging.StreamHandler(),
              logging.FileHandler('system_test.log')
          ]
      )
      logger = logging.getLogger('system_test')
      
      exit_code = asyncio.run(main())
      sys.exit(exit_code)
  except KeyboardInterrupt:
      print("Test interrupted by user")
      sys.exit(1)
  except Exception as e:
      print(f"Error: {str(e)}")
      sys.exit(1)