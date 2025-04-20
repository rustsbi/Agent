## 会议主题： 
测试报告总结汇报会议
## 会议时间： 
2025年04月12日
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
1. 张子涵介绍检索评估结果。
   - 存在的问题：
     - 目前是基于开源数据集评估检索指标， 缺少进一步的case分析, 缺少业务数据的检索测试
   - 下一步工作建议：
     - 针对检索出现的bad case进行分析，提出优化改进方案
     - 针对业务数据，手工或者合成少量业务问答数据，测试检索效果
2. 邝嘉诺介绍QA测试报告
   - 存在的问题：
     - lmdeploy 部署模型未跑通
     - 模型输出结果中英文混杂
   - 下一步工作建议
     - 检查lmdeploy部署时模型路径问题、以及发送request请求时，model_path参数与模型部署路径是否完全一致。可进一步参考[lmdeploy开发文档](https://lmdeploy.readthedocs.io/en/latest/)
     - 尝试将prompt中的所有英文数据换为中文的表述方式
3. 任潇介绍QA测试报告
   - 存在的问题
     - 模型输出结果中中英文混杂
     - 模型内容理解能力差，不能很好理解检索出的文档中的关键内容
   - 下一步工作建议
     - 尝试将prompt中的所有英文数据换为中文的表述方式
     - 切换模型Qwen2.5-7b-Instruct为Qwen2.5-14B-Instruct，或者测试QwQ2.5-32B。在部署时，如果显存资源有限部署失败，可以尝试用VLLM部署，设置内存利用率。或者使用量化版本的14B或者32B模型
