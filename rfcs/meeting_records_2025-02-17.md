## 会议主题： 
问答模块优化讨论与bug调试
## 会议时间： 
2025年2月17日
## 会议时长:  
1h
## 会议方式： 
线上腾讯会议
## 参会人员：
[洛佳](https://github.com/luojia65)
[马兴宇](https://github.com/xingyuma618)
[张子涵](https://github.com/ArchLance)
[任潇](https://github.com/wyywwi)
[邝嘉诺](https://github.com/gitveg)
## 会议内容
讨论当前RAG系统开发检索模块待处理工作与优化点
- 待进行工作： 问答函数调用调试
  
- 优化点1： bug调试
  在审查 logs 后，发现 llm-client 跑不通的原因在于两个方面：
  -  configs.py 中的 `API-BASE` 和 `API-KEY` 写反了））
  -  llm-client 中的 `llm = OpenAILLM(DEFAULT_MODEL_PATH, 8000, DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_API_CONTEXT_LENGTH, 5, 0.5)` 中的 `top-k` 的范围必须在0~1之间，改成了0.5。
- 优化点2：讨论增加测评指标
  - 检索评测指标P,R,F1
  - 排序评测指标，NDCG, ROC, AUC
   
明确下一步工作，开始调用LLM服务进行生成模块开发。