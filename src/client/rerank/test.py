import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# your query and corresponding passages
query = '什么是人工智能'
passages = ['阿爸巴sss啊啊啊啊s巴爸爸', 'AI技术在各领域广泛应用', 
            '机器学习是AI的核心技术。',
            '人工智能是计算机科学的一个分支。']

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]
# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1')
model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1')

device = 'cuda'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# calculate scores
logits = model(**inputs_on_device, return_dict=True).logits
print(logits.shape)
scores = logits.view(-1,).float()
scores = torch.sigmoid(scores)
print(scores)