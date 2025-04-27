# 4月工作记录

4月的工作主要围绕对于更强的模型进行部署，以及对于prompt的优化和RAG系统llm能力的进一步QA测试。

## 使用lmdeploy或者vllm部署14B及更大的模型

鉴于lmdeploy在部分部署场景的受限，尝试采用vllm进行同步部署。

### Qwen2.5-14B-Instruct

#### 使用vllm部署

部署脚本为：
```sh
#!/bin/bash

# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
export CUDA_VISIBLE_DEVICES=0,1  # 指定使用 GPU 0 和 GPU 1

# 模型路径和日志文件路径
MODEL_NAME="Qwen2.5-14B-Instruct"
MODEL_PATH="/home/model/${MODEL_NAME}"

LOG_FILE="vllm_record.log"

# 启动 vLLM 服务
vllm serve \
  ${MODEL_PATH} \
  --served-model-name ${MODEL_PATH} \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --port 2333 \
  --gpu-memory-utilization 0.90 \
  >> ${LOG_FILE} 2>&1 &

echo "vLLM service started on port 2333. Logs are being written to ${LOG_FILE}"
```

可以成功部署

test_llm_server.py的输出结果如下：
```bash
(qanything) root@451f9b706ca3:/home/kjn/Agent/src/server/llm_server# python test_llm_server.py 
model_list: [Model(id='Qwen2.5-14B-Instruct', created=1745502686, object='model', owned_by='vllm', root='/home/model/Qwen2.5-14B-Instruct', parent=None, max_model_len=32768, permission=[{'id': 'modelperm-9d9fef600f554699b50a52f603482353', 'object': 'model_permission', 'created': 1745502686, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])]
use model_name: Qwen2.5-14B-Instruct

Response generated in 9.1654 seconds

ChatCompletion(id='chatcmpl-fceada38ef2e47e2bb30ad4c268d44c4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='当然可以，以下是关于时间管理的三个实用建议：\n\n1. **设定明确的目标和优先级**：首先，你需要明确自己在一天、一周乃至更长时间内的主要目标是什么。这些目标应该具体且可实现。然后，根据这些目标的重要性来设定优先级。这有助于你集中精力处理最重要的任务，避免时间被不重要的事情占用。\n\n2. **使用时间管理工具**：利用日历、待办事项列表、时间管理应用程序等工具来规划你的日常活动。这些工具可以帮助你更好地组织时间，确保你不会错过重要的会议或截止日期。此外，定期回顾这些工具中的内容，调整你的计划以适应任何变化或新出现的任务。\n\n3. **实施番茄工作法或其他时间管理技巧**：番茄工作法是一种简单有效的时间管理方法，它通过将工作时间分割成25分钟的工作段（一个“番茄钟”），每个工作段之间休息5分钟，每完成四个“番茄钟”后休息更长时间（15-30分钟）。这种方法有助于提高集中力，同时避免长时间工作导致的疲劳。当然，还有其他许多时间管理技巧可供探索，找到最适合你的那一种。\n\n通过实践这些策略，你可以更有效地管理你的时间，提高工作效率，同时也能留出更多时间用于休息和娱乐。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)], created=1745502686, model='Qwen2.5-14B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=271, prompt_tokens=24, total_tokens=295, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None)
```

### Qwen2.5-32B-Instruct-GPTQ-Int4

部署使用的脚本同上，同样使用vllm部署，比较顺利，可以部署成功。

但是`test_llm_server.py`的输出时间巨长，大约要快半个小时才结束，结果显然是超时了，遂暂时放弃这个模型。

### Qwen2.5-32B-Instruct-AWQ

部署使用的脚本同上，同样使用vllm部署，比较顺利，可以部署成功。

test_llm_server.py的输出结果如下：

```bash
(qanything) root@451f9b706ca3:/home/kjn/Agent/src/server/llm_server# python test_llm_server.py 
model_list: [Model(id='Qwen2.5-32B-Instruct-AWQ', created=1745553249, object='model', owned_by='vllm', root='/home/model/Qwen2.5-32B-Instruct-AWQ', parent=None, max_model_len=32768, permission=[{'id': 'modelperm-1eb8a5da4e30435aba40a6d81e43b380', 'object': 'model_permission', 'created': 1745553249, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])]
use model_name: Qwen2.5-32B-Instruct-AWQ

Response generated in 10.6062 seconds

ChatCompletion(id='chatcmpl-c72988cac7fc4b34bd2cfc23959a265d', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='时间管理是提高效率和减少压力的重要技能。以下是三个关于时间管理的建议：\n\n1. **制定计划和优先级**：首先，明确你的长期和短期目标，这有助于确定哪些任务最重要。每天开始工作前，列出当天需要完成的任务，并根据紧急性和重要性对这些任务进行排序。使用数字（如1, 2, 3）或字母（如A, B, C）来标记任务的优先级。专注于完成优先级最高的任务，这将帮助你更有效地利用时间。\n\n2. **使用时间块法**：将一天分成几个时间块，每个时间块专门用于处理一类任务或一个项目。这种方法有助于减少任务之间的切换时间，提高专注度。例如，你可以将上午的时间块用于处理需要高度集中注意力的工作，下午的时间块用于处理邮件或进行会议。确保为休息和放松时间也安排时间块，以避免过度劳累。\n\n3. **学会说“不”**：有效地管理时间还意味着知道何时说“不”。当别人要求你做额外的工作或参加活动时，考虑一下这是否符合你的优先级和目标。如果这些请求会干扰到你的重要任务，那么学会礼貌地拒绝是很重要的。这不仅能帮助你保持对时间的控制，还能减少压力和疲劳。\n\n通过实施这些策略，你可以更有效地管理你的时间，提高工作效率，同时也有更多的时间来享受生活。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)], created=1745553249, model='Qwen2.5-32B-Instruct-AWQ', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=295, prompt_tokens=24, total_tokens=319, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None)
```

可见效果不错，因此我们最终是选用这个模型作为base llm。

### Qwen2.5-32B-Instruct-GGUF

#### 使用vllm部署

部署脚本同上

报错：
```log
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/model/Qwen2.5-32B-Instruct-GGUF'. Use `repo_type` argument if needed.
```

这个错误不太清楚为什么，比较奇怪，暂时放弃GGUF的量化模型。


### Qwen2.5-32B-Instruct-GPTQ-Int8

#### 使用lmdeploy部署

部署脚本
```sh
# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
export CUDA_VISIBLE_DEVICES=0,1
lmdeploy serve api_server --log-level=INFO --server-port 2333 --tp=2 --dtype=float16  /home/model/Qwen2.5-32B-Instruct-GPTQ-Int8 >> record.log 2>&1 &
```

报错:
```
RuntimeError: [TM][ERROR]  Assertion fail: /lmdeploy/src/turbomind/python/bind.cpp:423 
```

不太清楚这是什么错误，暂时放弃使用lmdeploy部署。

#### 使用vllm部署

部署脚本
```sh
# 环境变量设置
#!/bin/bash

# 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # 禁用 NCCL 共享内存通信
export CUDA_VISIBLE_DEVICES=0,1  # 指定使用 GPU 0 和 GPU 1

# 模型路径和日志文件路径
MODEL_NAME="Qwen2.5-32B-Instruct-GPTQ-Int8"
MODEL_PATH="/home/model/${MODEL_NAME}"

LOG_FILE="vllm_record.log"

# 启动 vLLM 服务
vllm serve \
  ${MODEL_PATH} \
  --served-model-name ${MODEL_NAME} \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --port 2333 \
  --gpu-memory-utilization 0.95 \
  >> ${LOG_FILE} 2>&1 &

echo "vLLM service started on port 2333. Logs are being written to ${LOG_FILE}"
```

报错：
```log
 ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
```

将`gpu-memory-utilization`设置为0.98，重新运行部署脚本

报错：
```log
ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (4464). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

设置`max_model_len`为5000，同时添加`export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`来优化内存分配，重新运行部署脚本

报错：
```log
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 270.00 MiB. GPU 0 has a total capacity of 21.67 GiB of which 255.75 MiB is free. Process 2109 has 344.00 MiB memory in use. Process 1448259 has 21.07 GiB memory in use. Of the allocated memory 20.08 GiB is allocated by PyTorch, with 328.00 MiB allocated in private pools (e.g., CUDA Graphs), and 56.49 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

可以发现Int8的量化的精度还是高了点，需要的计算资源也高，服务器的卡是带不动的。

经过多次尝试无果，于是暂时放弃部署 Qwen2.5-32B-Instruct-GPTQ-Int8 。

## 将prompt换为中文测试

将prompt中的主要内容改为中文，其中的英文变量保持不变，内容见新的`configs`内容

| 输入Query |   输出 Response |  备注 |
|-----------| ---------------| ---------- |
采莲赋的内容是什么 | 根据参考信息，采莲赋的内容描述了采莲时的场景，其中提到了“于是妖童媛女，荡舟心许；鷁首徐回，兼传羽杯；棹将移而藻挂，船欲动而萍开。尔其纤腰束素，迁延顾步；夏始春余，叶嫩花初，恐沾裳而浅笑，畏倾船而敛裾”。这描绘了采莲时青年男女在船上嬉戏、饮酒传杯、船只在水面上缓慢移动，以及少女们轻盈的动作和姿态。\n\n- **妖童媛女**：指年轻英俊的男子和美丽的女子。\n- **荡舟心许**：指他们在船上相互表达了心意。\n- **鷁首徐回**：鷁首指的是船头，这句描述了船缓缓移动的样子。\n- **兼传羽杯**：羽杯指的是酒杯，这句描绘了船上饮酒传杯的场景。\n- **棹将移而藻挂，船欲动而萍开**：描述了船桨移动时藻类被挂起，船行进时浮萍分开的景象。\n- **纤腰束素，迁延顾步**：描述了少女们轻盈的动作和姿态。\n- **夏始春余，叶嫩花初**：描述了采莲时的季节和环境，此时是春夏交替之际，荷叶和荷花刚刚长出。\n- **恐沾裳而浅笑，畏倾船而敛裾**：描述了少女们小心翼翼地避免弄湿衣裳和船倾斜的动作。| 语言为中文，与question保持一致 |
用简洁的语言对文段进行总结和概括, 并给出文段的主题, 要求字数在100字以内 | 根据参考信息，文段描述了月下独处的宁静感受与荷塘美景，通过细腻的笔触展现了荷花、月色及周围环境的和谐景象。主题在于自然美景中的个人自由与宁静感受。\n\n- 主题：自然美景中的个人自由与宁静感受。\n- 概括：作者在月下独自漫步，享受荷塘的美景与宁静，表达了对自由与安静生活的向往。| 语言为中文，与question保持一致 |
概括一下文段的主要内容 | 根据参考信息，文段的主要内容如下：\n\n- 描述了一个荷塘在月光下的美丽景色。荷塘上布满了叶子和点缀着白花，微风吹过带来清香，月光洒在荷叶和花朵上，营造出一种梦幻般的氛围。\n- 记述了荷塘边曾经发生的嬉游场景，提到《西洲曲》中的句子，表达了对过去美好时光的怀念。\n- 最后作者因为对江南的思念而想到了荷塘的景象，文章在作者回到家中后结束。\n\n这些内容主要来源于对荷塘景色的细腻描绘和对过去嬉游时光的怀念，同时也表达了作者对远方的思念之情。| 语言为中文，与question保持一致 |
'采莲南塘秋，莲花过人头；低头弄莲子，莲子清如水。'这首词出自哪里 | 根据参考信息，\"'莲南塘秋，莲花过人头；低头弄莲子，莲子清如水。'\"这首词出自《西洲曲》。| 回答正确 |
文中提到的“妖童媛女”在做什么？ | 根据参考信息，“妖童媛女”在荡舟嬉游。他们荡舟心许，用船桨划水，船在水面上缓缓移动，船桨挂住了水中的藻类，船动时水中的浮萍随之分开。他们的动作轻盈，腰肢纤细，穿着素雅的衣物，举止间流露出娇羞与谨慎，生怕沾湿衣裳或倾覆小船。这段描述展现了他们嬉游时的生动景象。| 回答的很不错，与检索文档中的内容基本一致 |


综上,在更换base llm为32B后，同时将prompt改为中文的情况下，RAG系统的回答能力得到了显著提升，回答的准确性和相关性都得到了提高，指令遵循能力也得到了加强，解决了中英文反复的问题。

## 其他

对`test_llm_server`增加了输出时间的计算

RAGTruth也增加了不少小参数模型的测试，但是基于已经换了32B的llm，感觉没有必要再测试了。

更新了model的路径，在/home下新建了model文件夹，以后大家下好的模型都可以统一放在这，避免了每个人的文件夹下各自都有model文件夹，导致混乱。

## 新的configs内容（不包含敏感内容）

```py
DEFAULT_MODEL_NAME = "Qwen2.5-32B-Instruct-AWQ"
# DEFAULT_MODEL_PATH = "/home/kjn/model/" + DEFAULT_MODEL_NAME
DEFAULT_MODEL_PATH = "/home/model/" + DEFAULT_MODEL_NAME

CUSTOM_PROMPT_TEMPLATE = """
<用户指令>
{{custom_prompt}}
</用户指令>

<文档>
{{context}}
</文档>

<指令>
- <文档>和</文档>之间的所有内容是从外部知识库检索到的参考信息。
- 现在，根据上述检索到的文档（让我们一步步思考），回答以下问题：
{{question}}
</指令>
"""

SYSTEM = """
你是一个可靠的助手，根据提供的外部文档回答问题。严格按照指令操作，尽可能提供帮助。

### 全局回答规则：
1. **内容依据**：基于提供的参考信息回答，不猜测或编造。
2. **回答结构**：使用标题、项目符号或表格（如适用）呈现结构良好的回答。
3. **避免重复**：合并并总结重复信息，避免冗余。
4. **引用来源**：在推理过程中，跟踪并引用来源（引用编号、标题、章节、摘要），但不要在最终回答中包含完整的来源细节。
5. **回答开头**：以“根据参考信息”或“According to the reference information”开始面向用户的回答，与问题的语言一致，确保回答自然、专业且相关。
6. **检查**：确保回答了问题的所有部分，引用准确，回答一致。
7. **语言和格式**：使用与问题相同的语言，使用 Markdown 格式化。

今天的日期是{{today_date}}。当前时间是{{current_time}}。
""" 

PROMPT_TEMPLATE = """
<系统>
{{system}}
</系统>

<文档>
{{context}}
</文档>

<指令>
{{instructions}}
</指令>
"""

INSTRUCTIONS = """
- 任务：严格根据<文档>和</文档>之间提供的参考信息回答问题 "{{question}}"，按照以下步骤和格式进行。

### 语言
- 记住，用与 "{{question}}" 相同的语言回答。
- 如果问题是中文，用中文回答。
- 如果问题是英文，用英文回答。

### 回答步骤：
1. **使用信息来源**（内部步骤）：
    - 在推理过程中，使用“信息来源”部分收集和整理相关文档引用。
    - **每个引用**必须按照以下格式列出（内部隐藏列表）：
        - **ID**：（引用编号，是参考标题中的“ref_number”字段，例如 [REF.1]）
            - **标题**：（文件名或标题，是参考标题中的“文件名”字段。如果文件名是无意义的链接或无效内容，则使用内容中的第一个标题或相关关键词。）
            - **章节**：（直接从原文中指定章节、条目或小标题，如适用；这指的是以 #、1.、一. 等开头的标题。）
            - **摘要**：（用一句话总结最相关的内容，最好使用原文中已有的句子或短语。）
    - **不要在最终面向用户的回答中包含完整的“信息来源”部分**。
2. **开始“推断答案部分”**：
    - 直接以“根据参考信息”或“According to the reference information”开始面向用户的回答，根据用户输入问题的语言。
    - **直接回答**：
        - 如果参考信息与问题完全匹配，仅根据相关信息给出**直接回答**。
    - **推理和计算**：
        - 如果参考信息**部分相关**但不完全匹配，尝试进行合理的**推理或计算**并解释你的推理过程。
        - 确保所有论点和结论都完全基于提供的参考材料中的证据。
        - 避免仅基于孤立细节的假设；始终考虑整个上下文，以防止片面或过度延伸的推理。
    - **处理不相关性**：
        - 如果参考信息完全不相关，回答：“抱歉，检索到的参考信息并未提供任何相关的信息，因此无法回答。”
        - 如果问题中有拼写错误的单词，请礼貌地提示可能的正确用词，然后根据正确用词回答问题。

---

### 回答前确认：
1. 确保参考信息中的所有要点都已涵盖。
2. 通过合并和总结重复信息避免冗余。
3. 确保回答中没有矛盾或不一致之处。

---

### 回答后检查清单：
1. **回答完整性**：确保回答了问题的所有部分。
2. **逻辑与一致性**：再次检查回答中是否有逻辑错误或内部矛盾。
3. **引用准确性**：确保信息来源的相关性、完整性和准确性，以及格式的一致性。

---

### 格式：
- **灵活格式**：
    - 根据需要使用标题（##、###、####）、项目符号或表格。
    - 使用项目符号（- 或 1.、2.、3.）列出多个要点。
    - 如有需要，使用**粗体**或*斜体*突出显示关键信息。
    - **引用编号可见性**：
        - 不要在最终回答中显示引用编号。
    - 对于列表或比较类问题，使用**表格**或**项目符号**。
    - 对于叙述性回答，使用**段落**清晰地解释细节。
"""

SIMPLE_PROMPT_TEMPLATE = """
- 我是一个乐于助人的助手。我可以回答你的问题，也可以向你提问。
- 今天的日期是{{today}}。当前时间是{{now}}。
- 用户自定义指令：{{custom_prompt}}
- 回答前，确认所需的关键要点或信息数量，确保没有遗漏。
- 现在，回答以下问题：
{{question}}
用 Markdown 格式返回你的回答，使用与问题 "{{question}}" 相同的语言。
"""
```