# 测试报告

## 测试目标
测试RAG系统的指令遵循能力，以及使用数据集测试LLM的能力。

## 测试记录
### 简单提问测试

| 输入Query |   输出 Response |  存在的问题 |
|-----------| ---------------| ---------- |
概括一下文段的主要内容 | "According to the reference information, the text describes the scene of a moonlit night by a lotus pond. The passage is divided into two main parts:\\n\\n1. **Lotus Pond Scene** (Ref.1):\\n    - The text vividly portrays the beauty of the lotus pond under the moonlight. It describes the lotus leaves and flowers, their colors, and the fragrance they emit. The light and shadows on the lotus leaves and flowers create a harmonious and dreamy atmosphere. The author also mentions how the moonlight and misty atmosphere enhance the poetic ambiance of the scene.\\n2. **Memories and Reflections** (Ref.2):\\n    - The second part of the text shifts to memories of past lotus picking activities. It describes the lively scene of young men and women enjoying themselves on the lotus pond. The author then compares this scene to the poetic imagery in \\"The West洲曲,\\" expressing a longing for the beauty and simplicity of the past. The passage concludes with the author reflecting on the current tranquility of his surroundings, hinting at a sense of nostalgia for the past.\\n\\nIn summary, the text combines a vivid description of a lotus pond scene with personal reflections on past memories and a sense of nostalgia.  | 语言上全英文，无法与Query同步 |
概括一下文段的主要内容 | "According to the reference information, the text describes the author\'s observations and feelings about a moonlit night by a lotus pond. The main content can be summarized as follows:\\n\\n- **景物描写**（景物描写）: 描述了月光下的荷塘景色，包括荷叶、荷花、月光和雾气的交织景象。微风拂过荷塘，带来了阵阵清香。\\n- **情感表达**（情感表达）: 作者通过景物描写，表达了对自然美景的喜爱和对往事的怀念。特别是引用了《西洲曲》中的诗句，寄托了对江南水乡的思念。\\n\\n这些描述展现了作者对自然美景的细腻感受以及对往事的深深怀念。" | 语言中英结合，无法保持一种语言 |
用简洁的语言对文段进行总结和概括, 并给出文段的主题, 要求字数在100字以内 | "According to the reference information,该段文字描述了一个独自漫步荷塘的夜晚场景，作者通过细腻的笔触描绘了荷塘月色的美好，同时也表达了自己在这种环境中感受到的自由和宁静。荷塘四周被树木围绕，夜晚的蝉鸣和蛙声增添了氛围。文章还提到了采莲的旧俗，勾起了对过去嬉游场景的回忆。\n\n荷塘月色的美景和夜晚的宁静是文章的主要主题。" | 语言中英结合，无法保持一种语言 |

### RAGTruth数据集测试LLM。

RAGTruth是一个单词级幻觉语料库，用于检索增强生成（RAG）设置中的各种任务，用于训练和评估。

通过相应的脚本，选择了RAGTruth数据集中Task_type为"Summary", "data2text" 和 "QA" 各101份数据，共计303份数据，测试Qwen2.5-7B-Instrut，得到`response-Qwen2.5-7B-Instruct.jsonl`。

再通过利用Ollama中的deepseek-r1-8B的api作为裁判大模型，结合"source_info"和"response"，得到了相应的judge结果，计算出了Qwen2.5-7B-Instrut 在包含context的场景下的幻觉率。

由于不知道什么原因，我这边下载的几个model用不了，如果解决了这个问题，后期可以多测几个model进行比较。


## 存在的问题总结
对于configs中的prompt_template，LLM没有很好地理解和遵循。当用户输入的语句比较短，难以提取其中使用的语言。语句较长时可以较为准确地识别。总结为小参数模型的指令遵循能力不足。

## 优化的思路
优化prompt_template，同时试着是否能够使用更大参数或者更好表现的模型来提升指令遵循能力。


### 已进行的优化方案记录

尝试通过精简prompt，减少LLM的理解负担；通过重复Language的要求，增强Langugae的权重。

结果：优化后，LLM的回答语言稍微稳定了些，但是效果不大。


### 待尝试的优化方案
1. 使用CoT来提升LLM对prompt的遵循程度。
2. 寻找小参数中更适合我们任务的模型替代。