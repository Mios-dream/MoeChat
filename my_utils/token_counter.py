"""
Token 估算工具

使用 tiktoken 库估算文本 token 数，作为 LLM 请求前预算管控的依据。
对所有 deepseek 系列模型，使用 cl100k_base 编码器可获得高精度近似。

失败时回退到字符数近似估算，确保不因 token 估算异常阻塞流程。
"""

import tiktoken

_ENCODING = None


def _get_encoding():
    """懒加载编码器"""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数。

    使用 cl100k_base 编码器（兼容 deepseek 系列模型）。
    失败时回退到 len(text) // 2 近似估算。

    Parameters:
        text: 要估算的文本

    Returns:
        估算的 token 数量
    """
    try:
        enc = _get_encoding()
        return len(enc.encode(text))
    except Exception:
        # 回退：中文约 1 token/1.5 字，英文约 1 token/4 字符，
        # 取 len//2 作为偏安全的估算（略微高估）
        return len(text) // 2
