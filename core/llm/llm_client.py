"""
统一的 LLM 客户端

整合提示词管理、响应解析和回调机制，提供统一的 LLM 请求接口。

核心功能：
1. 流式和非流式请求
2. 可插拔的响应解析
3. 灵活的回调机制
4. 请求配置管理
5. 工具调用支持（同步和流式）

设计原则：
- 单一职责：客户端只负责请求和响应处理
- 可组合：支持自定义解析器和回调
- 可扩展：易于添加新的 LLM 提供商
"""

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import AsyncGenerator
import asyncio
import json
import time

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from my_utils import config_manager as CConfig
from my_utils.log import logger as Log
from core.llm.response_parser import ResponseParser, StreamParserProtocol, TextParser
from core.llm.callback_manager import CallbackManager, CallbackEvent


@dataclass
class ToolCallResult:
    """
    工具调用结果

    属性：
    - tool_call_id: 工具调用 ID
    - tool_name: 工具名称
    - arguments: 工具参数
    - result: 执行结果
    - success: 是否执行成功
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str
    success: bool = True


@dataclass
class LLMStreamChunk:
    """
    LLM 响应片段
    属性：
    - parsed_data: 解析器解析后的数据（有 parser 时产出对应数据，无 parser 时 TextParser 输出原始 token）
    - tool_calls: 工具调用列表（流末尾一次性产出，优先于 parser flush，None 表示无工具调用）
    """

    parsed_data: Any | None = None
    tool_calls: list[ChatCompletionMessageFunctionToolCallParam] | None = None


class LLMClient:
    """
    统一的 LLM 客户端

    提供灵活的 LLM 请求接口，支持：
    - 流式和非流式请求
    - 自定义响应解析
    - 请求生命周期回调

    使用示例：
    ```python
    client = LLMClient()

    # 简单请求
    result = await client.request("你好")

    # 流式请求
    async for chunk in client.stream("你好"):
        print(chunk)

    # 使用 PromptManager 和自定义解析器
    pm = PromptManager()
    pm.add_system("你是一个助手")
    pm.add_user("你好")

    parser = JsonLineParser()
    async for chunk in client.stream(pm.messages, parser=parser):
        print(chunk)
    ```
    """

    def __init__(self, model_key: Literal["ChatLLM", "LLM"] = "ChatLLM"):
        """
        初始化 LLM 客户端

        参数：
        - model_key: 默认模型配置键名
        """
        self._model_key = model_key
        self._callbacks = CallbackManager()
        self._clients: dict[str, AsyncOpenAI] = {}

    @property
    def callbacks(self) -> CallbackManager:
        """获取回调管理器"""
        return self._callbacks

    def _get_client(self, model_key: str) -> AsyncOpenAI:
        """
        获取或创建 OpenAI 客户端

        参数：
        - model_key: 模型配置键名

        返回：
        - AsyncOpenAI 客户端实例
        """
        if model_key not in self._clients:
            config = CConfig.config.get(model_key, {})
            self._clients[model_key] = AsyncOpenAI(
                api_key=config.get("key", ""),
                base_url=config.get("api", ""),
            )
        return self._clients[model_key]

    def _get_model_config(self, model_key: str) -> dict:
        """
        获取模型配置

        参数：
        - model_key: 模型配置键名

        返回：
        - 配置字典
        """
        return CConfig.config.get(model_key, {})

    async def request(
        self,
        messages: list[ChatCompletionMessageParam],
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        tools: list[ChatCompletionToolUnionParam] | None = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
    ) -> str | None:
        """
        非流式请求

        参数：
        - messages: 消息列表
        - model_key: 模型配置键名（None 使用默认值）
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）
        - tools: OpenAI 工具定义列表（None 表示不使用工具）
        - tool_choice: 工具选择策略（"auto"/"none"/"required"）

        返回：
        - 响应文本，失败返回 None
          若启用工具且模型返回 tool_calls 则 content 为 None
        """
        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        try:
            # 合并 extra_body
            final_extra = config.get("extra_config", {}).copy()
            if extra_body:
                final_extra.update(extra_body)

            # 构建请求参数
            create_kwargs: dict[str, Any] = {
                "model": config.get("model", ""),
                "messages": messages,
                "stream": False,
                "extra_body": final_extra,
            }
            if tools:
                create_kwargs["tools"] = tools
                create_kwargs["tool_choice"] = tool_choice

            response = await asyncio.wait_for(
                client.chat.completions.create(**create_kwargs),
                timeout=timeout,
            )

            content = response.choices[0].message.content

            # 触发完成回调
            await self._callbacks.emit(CallbackEvent.COMPLETE, content=content)

            return content

        except asyncio.TimeoutError:
            Log.warning(f"[LLM客户端] 请求超时 ({timeout}s)")
            await self._callbacks.emit(CallbackEvent.ERROR, error="请求超时")
            return None

        except Exception as e:
            Log.error(f"[LLM客户端] 请求失败: {e}")
            await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
            return None

    async def stream(
        self,
        messages: list[ChatCompletionMessageParam],
        parser: ResponseParser[Any, Any] | StreamParserProtocol[Any] | None = None,
        model_key: Literal["ChatLLM", "LLM"] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """
        流式请求

        参数：
        - messages: 消息列表
        - parser: 响应解析器（None 使用 TextParser）
                  支持 ResponseParser 或 StreamParserProtocol 协议
        - model_key: 模型配置键名（None 使用默认值）
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）
        - tools: OpenAI 工具定义列表（None 表示不使用工具）
        - tool_choice: 工具选择策略（"auto"/"none"/"required"）

        产出：
        - LLMStreamChunk: 统一结构，通过字段承载不同语义：
          - .parsed_data: parser 解析后的数据块（有 parser 时）
          - .tool_calls: 工具调用列表（优先于 parser flush 产出，None 表示无工具调用）
        """
        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)
        effective_parser = parser or TextParser()

        # 重置解析器状态
        effective_parser.reset()

        # 用于流式模式下累积工具调用
        tool_calls_by_index: dict[int, ChatCompletionMessageFunctionToolCallParam] = {}

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        start_time = time.time()

        print("请求内容", json.dumps(messages, ensure_ascii=False, indent=2))

        try:
            # 合并 extra_body
            final_extra = config.get("extra_config", {}).copy()
            if extra_body:
                final_extra.update(extra_body)

            response = await client.chat.completions.create(
                model=config.get("model", ""),
                messages=messages,
                extra_body=final_extra,
                stream=True,
                tools=tools or [],
                tool_choice=tool_choice,
            )

            async for chunk in response:

                if (
                    chunk is None
                    or chunk.choices is None
                    or len(chunk.choices) == 0
                    or (delta := chunk.choices[0].delta) is None
                ):
                    continue

                # 处理工具调用增量（tool_calls 分步到达）
                if tools and getattr(delta, "tool_calls", None) and delta.tool_calls:
                    for call in delta.tool_calls:
                        idx = call.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": call.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc = tool_calls_by_index[idx]
                        if call.id:
                            tc["id"] = call.id
                        if call.function:
                            if call.function.name:
                                tc["function"]["name"] = call.function.name
                            if call.function.arguments:
                                tc["function"]["arguments"] += call.function.arguments
                    # 正在收集工具调用，不再产出文本 token
                    continue

                # 已有工具调用在收集，跳过后续文本
                if tool_calls_by_index:
                    continue

                # 正常文本 token 处理
                if delta.content:
                    token = delta.content

                    # 触发 token 回调
                    await self._callbacks.emit(CallbackEvent.TOKEN, token=token)

                    # 使用解析器处理 token
                    for parsed in effective_parser.stream_parse(token):
                        # 触发 chunk 回调
                        await self._callbacks.emit(CallbackEvent.CHUNK, chunk=parsed)
                        yield LLMStreamChunk(parsed_data=parsed)

            # 先产出累积的工具调用事件（优先于 parser flush，确保调用方先感知工具调用）
            if tool_calls_by_index:
                sorted_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                    tool_calls_by_index[idx]
                    for idx in sorted(tool_calls_by_index.keys())
                ]
                # 触发工具调用回调
                await self._callbacks.emit(CallbackEvent.TOOL_CALLS, calls=sorted_calls)
                yield LLMStreamChunk(tool_calls=sorted_calls)

            # 处理解析器缓冲区中的剩余数据
            for parsed in effective_parser.flush():
                await self._callbacks.emit(CallbackEvent.CHUNK, chunk=parsed)
                yield LLMStreamChunk(parsed_data=parsed)

            # 触发完成回调
            elapsed = time.time() - start_time
            await self._callbacks.emit(CallbackEvent.COMPLETE, elapsed=elapsed)

        except Exception as e:
            Log.error(f"[LLM客户端] 流式请求失败: {e}")
            await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
