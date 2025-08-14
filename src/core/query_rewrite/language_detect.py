from langdetect import detect
from typing import List


def detect_language(text: str) -> str:
    """
    检测输入文本的语言类型。
    返回：'zh'（中文）、'en'（英文）、或其他langdetect支持的语言代码
    """
    try:
        lang = detect(text)
        # 更精确的语言映射
        if lang in ['zh-cn', 'zh-tw', 'zh', 'ko']:
            return 'zh'
        elif lang == 'en':
            return 'en'
        # 对于检测不准确的情况，进行启发式判断
        elif lang in ['vi', 'af', 'ja', 'ko']:  # 可能是误检测
            # 检查是否包含中文字符
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                return 'zh'
            # 检查是否主要是英文字符
            elif all(char.isascii() and char.isalpha() or char.isspace() or char in ',.!?()' for char in text):
                return 'en'
            else:
                return lang
        else:
            return lang
    except Exception as e:
        # 异常情况下进行启发式判断
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        elif all(char.isascii() and char.isalpha() or char.isspace() or char in ',.!?()' for char in text):
            return 'en'
        else:
            return 'unknown'


def batch_detect_language(texts: List[str]) -> List[str]:
    """
    批量检测语言类型
    """
    return [detect_language(t) for t in texts]
