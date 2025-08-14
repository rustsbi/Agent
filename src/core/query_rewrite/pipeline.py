from .language_detect import detect_language
from .translator import LocalTranslator
from .rewriter import llm_openai_rewrite
from typing import List, Dict


class QueryRewritePipeline:
    def __init__(self):
        self.translator = LocalTranslator()

    def process(self, query: str, target_lang: str = 'en') -> Dict[str, List[str]]:
        """
        完整的query预处理pipeline：
        1. 语言检测
        2. 必要时翻译
        3. LLM重写优化
        返回：{'original': 原始query, 'translated': 翻译后query, 'rewrites': 重写query列表}
        """
        lang = detect_language(query)
        result = {'original': query, 'translated': None, 'rewrites': []}
        # 翻译（如原文非目标语言）
        if lang != target_lang:
            translated = self.translator.translate(
                [query], src_lang=lang, tgt_lang=target_lang)[0]
            result['translated'] = translated
            # 对翻译后query做扩展
            rewrites = [llm_openai_rewrite(translated, mode='rewrite')]
        else:
            result['translated'] = query
            rewrites = [llm_openai_rewrite(query, mode='rewrite')]
        result['rewrites'] = rewrites
        return result
