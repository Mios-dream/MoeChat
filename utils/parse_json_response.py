import json
from typing import Any


def parse_llm_json_response(response: str | None) -> dict[str, Any] | None:
    """
    解析大语言模型返回的JSON格式响应

    Args:
        response (str): 模型返回的原始响应字符串

    Returns:
        Optional[Dict[str, Any]]: 解析后的字典对象，解析失败时返回None
    """
    if not response:
        return None

    try:
        # 尝试直接解析JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试清理响应后再解析
        cleaned_response = _clean_json_response(response)
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            return None


def _clean_json_response(response: str) -> str:
    """
    清理模型返回的JSON响应，移除可能的额外文本

    Args:
        response (str): 原始响应字符串

    Returns:
        str: 清理后的响应字符串
    """
    # 移除首尾空白字符
    response = response.strip()

    # 查找第一个 '{' 和最后一个 '}'
    start_idx = response.find("{")
    end_idx = response.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # 提取JSON部分
        json_part = response[start_idx : end_idx + 1]
        return json_part

    # 如果没有找到完整的JSON结构，返回原响应
    return response
