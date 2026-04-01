from my_utils import config as CConfig
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
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
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("无法解析 LLM 返回的 JSON")
    return json.loads(json_match.group())


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


async def llm_request(msg: list[ChatCompletionMessageParam]) -> str | None:
    """
    LLM(大参数模型)快速请求，非流式
    :param data: 消息链
    :return: 请求结果
    """
    client = AsyncOpenAI(
        api_key=CConfig.config["LLM"]["key"], base_url=CConfig.config["LLM"]["api"]
    )
    response = await client.chat.completions.create(
        model=CConfig.config["LLM"]["model"],
        messages=msg,
        stream=False,
        extra_body=CConfig.config["LLM"].get("extra_config", {}),
    )
    content = response.choices[0].message.content
    return content


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


async def slm_request(msg: list[ChatCompletionMessageParam]) -> str | None:
    """
    SLM(小参数模型)快速请求，非流式
    :param data: 消息链
    :return: 请求结果
    """
    client = AsyncOpenAI(
        api_key=CConfig.config["SLM"]["key"], base_url=CConfig.config["SLM"]["api"]
    )
    response = await client.chat.completions.create(
        model=CConfig.config["SLM"]["model"],
        messages=msg,
        stream=False,
    )
    content = response.choices[0].message.content
    return content
