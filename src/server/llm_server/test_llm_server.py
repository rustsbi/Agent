from openai import OpenAI
import time  # 导入time模块用于时间计算

client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:2333/v1"
)

model_name = client.models.list().data[0].id
print(f"model_list: {client.models.list().data}")
print(f"use model_name: {model_name}")

# 记录请求发送时间
start_time = time.time()

response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "你是一个时间管理助手"},
    {"role": "user", "content": "对于时间管理给出三个建议"},
  ],
    temperature=0.8,
    top_p=0.8
)

# 记录响应返回时间
end_time = time.time()

# 计算生成response的时间
elapsed_time = end_time - start_time
print(f"\nResponse generated in {elapsed_time:.4f} seconds\n")

print(response)