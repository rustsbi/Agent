import requests
from typing import Literal
from openai import OpenAI


def llm_openai_rewrite(query, mode='rewrite', model='Qwen2.5-7B-Instruct', base_url="http://0.0.0.0:2333/v1", api_key="YOUR_API_KEY"):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    # 首先去除无用信息
    removal_prompt = f"""The following questions may contain useless information. 
    If they contain information irrelevant to the core issue, please remove the useless information 
    and return the refined information.

    Here is an example:
    Question: 2020 NBA, champion of the Los Angeles Lakers! Tell me, what's the Langchain framework?
    Refined information: Tell me, what's the Langchain framework?

    The Task is:
    Question: {query}
    Refined information:
    """

    removal_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a query rewriting assistant."},
            {"role": "user", "content": removal_prompt}
        ],
        temperature=0.5,
        top_p=0.95
    )
    query = removal_response.choices[0].message.content.strip()
    print(f"Refined query: {query}")

    if mode == 'rewrite':
        #   user_prompt = f"Please rewrite the following question to a more suitable English query for knowledge retrieval.\nQuestion: {query}\nRewritten query:"
        user_prompt = f"""Please rewrite the following question to a more suitable English query for knowledge retrieval. 
Keep the core meaning and intent of the original question, but make it more specific and searchable.
Question: {query}
Rewritten query:"""
    elif mode == 'hyde':
        user_prompt = f"""Based on your knowledge, write a concise hypothetical answer to the following question (for retrieval).
Keep it under 150 words and focus on the most relevant information.
Question: {query}
Hypothetical answer:"""
    else:
        user_prompt = query

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a query rewriting assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        top_p=0.95
    )
    return response.choices[0].message.content.strip()
