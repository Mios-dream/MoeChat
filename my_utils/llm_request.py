from typing import AsyncGenerator

from my_utils import config_manager as CConfig
from my_utils import log as Log
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletion,
)
import json
import re


def parse_llm_json_response(content: str):
    """从 LLM 文本响应中提取首个 JSON 对象并反序列化。

    背景：
    - 部分模型可能返回 ```json 包裹文本或夹带说明文本。
    - 这里使用正则抓取最外层 `{...}`，尽量容错。

    异常：
    - 若未找到 JSON 片段，抛出 `ValueError`。
    """
    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            raise ValueError("无法解析 LLM 返回的 JSON")
        return json.loads(json_match.group())
    except Exception as e:
        Log.logger.error(f"解析 LLM JSON 响应失败: {content}")
        raise ValueError("无法解析 LLM 返回的 JSON")


async def llm_request_stream(msg: list[ChatCompletionMessageParam]):
    """
    流式HTTP请求函数，用于与大语言模型进行通信并流式返回输出内容,流式协议为sse

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表

    Yields:
        str: 模型流式输出的内容片段

    Example:
        async for content in llm_request_stream(messages):
            print(content, end='', flush=True)
    """
    client = AsyncOpenAI(
        api_key=CConfig.config["LLM"]["key"], base_url=CConfig.config["LLM"]["api"]
    )
    response = await client.chat.completions.create(
        model=CConfig.config["LLM"]["model"],
        messages=msg,
        stream=True,
        extra_body=CConfig.config["LLM"].get("extra_config", {}),
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def llm_request(
    msg: list[ChatCompletionMessageParam], extra_body: dict | None = None
) -> str | None:
    """
    LLM(大参数模型)快速请求，非流式
    :param data: 消息链
    :return: 请求结果
    """
    try:
        client = AsyncOpenAI(
            api_key=CConfig.config["LLM"]["key"], base_url=CConfig.config["LLM"]["api"]
        )
        response = await client.chat.completions.create(
            model=CConfig.config["LLM"]["model"],
            messages=msg,
            stream=False,
            extra_body=extra_body or CConfig.config["LLM"].get("extra_config", {}),
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        Log.logger.error(f"大模型请求失败: {e}")
        return None


async def chat_llm_request_stream(msg: list[ChatCompletionMessageParam]):
    """
    流式HTTP请求函数，用于与大语言模型进行通信并流式返回输出内容,流式协议为sse

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表

    Yields:
        str: 模型流式输出的内容片段

    Example:
        async for content in llm_request_stream(messages):
            print(content, end='', flush=True)
    """
    client = AsyncOpenAI(
        api_key=CConfig.config["ChatLLM"]["key"],
        base_url=CConfig.config["ChatLLM"]["api"],
    )
    response = await client.chat.completions.create(
        model=CConfig.config["ChatLLM"]["model"],
        messages=msg,
        stream=True,
        extra_body=CConfig.config["ChatLLM"].get("extra_config", {}),
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def llm_request_with_tools(
    msg: list[ChatCompletionMessageParam],
    tools: list[dict] | None = None,
    model_key: str = "ChatLLM",
) -> ChatCompletion:
    """
    带工具调用支持的 LLM 请求（非流式）

    用于支持 OpenAI function calling 协议。调用方需要检查返回的
    response.choices[0].message.tool_calls 来判断是否需要执行工具。

    Parameters:
        msg: 消息链
        tools: OpenAI tools 格式的工具定义列表
        model_key: 使用的模型配置键名（"LLM", "ChatLLM", "SLM"）

    Returns:
        ChatCompletion: 完整的 OpenAI 响应对象，包含可能的 tool_calls
    """
    config = CConfig.config[model_key]
    client = AsyncOpenAI(
        api_key=config["key"],
        base_url=config["api"],
    )

    kwargs = {
        "model": config["model"],
        "messages": msg,
        "stream": False,
        "extra_body": config.get("extra_config", {}),
    }

    # 仅在传入 tools 时添加工具参数
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = await client.chat.completions.create(**kwargs)
    # print(f"LLM 请求完成，response: {response}")
    return response


async def llm_request_with_tools_stream(
    msg: list[ChatCompletionMessageParam],
    tools: list[dict] | None = None,
    model_key: str = "ChatLLM",
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """
    带工具调用支持的 LLM 请求（流式）

    用于在流式响应中判断是否触发 tool_calls。
    调用方需要解析流式增量中的 delta.tool_calls。
    """
    config = CConfig.config[model_key]
    client = AsyncOpenAI(
        api_key=config["key"],
        base_url=config["api"],
    )

    kwargs = {
        "model": config["model"],
        "messages": msg,
        "stream": True,
        "extra_body": config.get("extra_config", {}),
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return await client.chat.completions.create(**kwargs)
