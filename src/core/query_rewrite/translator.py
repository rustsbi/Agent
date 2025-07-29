# ===== 默认使用 MarianMT 翻译器 =====
from transformers import MarianMTModel, MarianTokenizer
from typing import List


class LocalTranslator:
    def __init__(self):
        self.zh2en_tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-zh-en")
        self.zh2en_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-zh-en")
        self.en2zh_tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-zh")
        self.en2zh_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-zh")

    def zh2en(self, texts: List[str]) -> List[str]:
        inputs = self.zh2en_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True)
        translated = self.zh2en_model.generate(**inputs)
        return [self.zh2en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def en2zh(self, texts: List[str]) -> List[str]:
        inputs = self.en2zh_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True)
        translated = self.en2zh_model.generate(**inputs)
        return [self.en2zh_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        if src_lang == tgt_lang:
            return texts
        if src_lang == 'zh' and tgt_lang == 'en':
            return self.zh2en(texts)
        elif src_lang == 'en' and tgt_lang == 'zh':
            return self.en2zh(texts)
        else:
            raise ValueError(f'不支持的语言对: {src_lang}->{tgt_lang}')


# ===== Seed-X vLLM 方案保留（以后可切换） =====
"""
from vllm import LLM, SamplingParams

class LocalTranslator:
    def __init__(self):
        model_path = "./Seed-X-Instruct-7B"  # 请根据实际路径调整
        self.model = LLM(
            model=model_path,
            max_num_seqs=512,
            tensor_parallel_size=1,  # 单卡
            gpu_memory_utilization=0.95,
            dtype=\"float16\"  # 2080Ti只能用float16
        )

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        if src_lang == tgt_lang:
            return texts
        messages = []
        for text in texts:
            if src_lang == 'en' and tgt_lang == 'zh':
                prompt = f\"Translate the following English sentence into Chinese:\\n{text} <zh>\"
            elif src_lang == 'zh' and tgt_lang == 'en':
                prompt = f\"Translate the following Chinese sentence into English:\\n{text} <en>\"
            else:
                messages.append(text)
                continue
            messages.append(prompt)
        decoding_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=True)
        results = self.model.generate(messages, decoding_params)
        responses = [res.outputs[0].text.strip() for res in results]
        return responses
"""
