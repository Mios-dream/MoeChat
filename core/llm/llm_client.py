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

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal
import asyncio
import json
import time

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)

from my_utils import config_manager as CConfig
from my_utils.log import logger as Log
from core.llm.prompt_manager import PromptManager
from core.llm.response_parser import ResponseParser, StreamParserProtocol, TextParser
from core.llm.callback_manager import CallbackManager, CallbackEvent


class ToolCallError(Exception):
    """工具调用异常"""

    pass


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
class StreamRequest:
    """
    流式请求配置

    属性：
    - messages: 消息列表
    - parser: 响应解析器
    - model_key: 模型配置键名（"LLM", "ChatLLM", "SLM"）
    - timeout: 超时时间（秒）
    - extra_body: 额外请求参数
    """

    messages: list[dict[str, str]]
    parser: ResponseParser = field(default_factory=TextParser)
    model_key: str = "ChatLLM"
    timeout: float = 30.0
    extra_body: dict[str, Any] = field(default_factory=dict)


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
    ) -> str | None:
        """
        非流式请求

        参数：
        - messages: 消息列表
        - model_key: 模型配置键名（None 使用默认值）
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）

        返回：
        - 响应文本，失败返回 None
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

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.get("model", ""),
                    messages=messages,
                    stream=False,
                    extra_body=final_extra,
                ),
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
        parser: ResponseParser | StreamParserProtocol | None = None,
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> AsyncIterator[Any]:
        """
        流式请求

        参数：
        - messages: 消息列表
        - parser: 响应解析器（None 使用 TextParser）
                  支持 ResponseParser 或 StreamParserProtocol 协议
        - model_key: 模型配置键名（None 使用默认值）
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）

        产出：
        - 解析后的数据块
        """
        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)
        effective_parser = parser or TextParser()

        # 重置解析器状态
        effective_parser.reset()

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
                stream=True,
                extra_body=final_extra,
            )

            async for chunk in response:
                # 检查超时
                if time.time() - start_time > timeout:
                    Log.warning(f"[LLM客户端] 流式请求超时 ({timeout}s)")
                    break

                if chunk is None or chunk.choices is None or len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue

                token = delta.content

                # 触发 token 回调
                await self._callbacks.emit(CallbackEvent.TOKEN, token=token)

                # 使用解析器处理 token
                for parsed in effective_parser.stream_parse(token):
                    # 触发 chunk 回调
                    await self._callbacks.emit(CallbackEvent.CHUNK, chunk=parsed)
                    yield parsed

            # 处理解析器缓冲区中的剩余数据
            for parsed in effective_parser.flush():
                await self._callbacks.emit(CallbackEvent.CHUNK, chunk=parsed)
                yield parsed

            # 触发完成回调
            elapsed = time.time() - start_time
            await self._callbacks.emit(CallbackEvent.COMPLETE, elapsed=elapsed)

        except Exception as e:
            Log.error(f"[LLM客户端] 流式请求失败: {e}")
            await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))

    async def stream_with_prompt(
        self,
        prompt_manager: PromptManager,
        parser: ResponseParser | None = None,
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> AsyncIterator[Any]:
        """
        使用 PromptManager 的流式请求

        参数：
        - prompt_manager: 提示词管理器
        - parser: 响应解析器
        - model_key: 模型配置键名
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）

        产出：
        - 解析后的数据块
        """
        async for chunk in self.stream(
            messages=prompt_manager.messages,
            parser=parser,
            model_key=model_key,
            extra_body=extra_body,
            timeout=timeout,
        ):
            yield chunk

    def create_stream_request(
        self,
        messages: list[ChatCompletionMessageParam],
        parser: ResponseParser | None = None,
        model_key: str | None = None,
        timeout: float = 30.0,
        extra_body: dict[str, Any] | None = None,
    ) -> StreamRequest:
        """
        创建流式请求配置

        参数：
        - messages: 消息列表
        - parser: 响应解析器
        - model_key: 模型配置键名
        - timeout: 超时时间
        - extra_body: 额外请求参数

        返回：
        - StreamRequest 实例
        """
        return StreamRequest(
            messages=messages,
            parser=parser or TextParser(),
            model_key=model_key or self._model_key,
            timeout=timeout,
            extra_body=extra_body or {},
        )

    async def execute_stream_request(
        self, request: StreamRequest
    ) -> AsyncIterator[Any]:
        """
        执行流式请求

        参数：
        - request: StreamRequest 实例

        产出：
        - 解析后的数据块
        """
        async for chunk in self.stream(
            messages=request.messages,
            parser=request.parser,
            model_key=request.model_key,
            extra_body=request.extra_body,
            timeout=request.timeout,
        ):
            yield chunk

    # ========== 工具调用支持 ==========

    def _build_request_kwargs(
        self,
        config: dict,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        构建请求参数

        参数：
        - config: 模型配置
        - messages: 消息列表
        - tools: 工具定义列表
        - extra_body: 额外请求参数
        - stream: 是否流式请求

        返回：
        - 请求参数字典
        """
        kwargs: dict[str, Any] = {
            "model": config.get("model", ""),
            "messages": messages,
            "stream": stream,
            "extra_body": {**config.get("extra_config", {}), **(extra_body or {})},
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return kwargs

    def _normalize_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """
        标准化工具调用格式

        将 OpenAI 返回的 tool_calls 对象转换为可序列化的字典格式。

        参数：
        - tool_calls: OpenAI tool_calls 列表

        返回：
        - 标准化后的工具调用字典列表
        """
        normalized = []
        for tc in tool_calls:
            normalized.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
        return normalized

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
    ) -> list[ToolCallResult]:
        """
        执行工具调用

        参数：
        - tool_calls: 标准化的工具调用列表
        - tool_executor: 工具执行器（需提供 execute 方法）

        返回：
        - 工具执行结果列表
        """
        results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_call_id = tc["id"]
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            try:
                Log.info(f"[LLM客户端] 执行工具: {tool_name}, 参数: {arguments}")
                result = await tool_executor.execute(tool_name, arguments)
                results.append(
                    ToolCallResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result,
                        success=True,
                    )
                )
            except Exception as e:
                Log.error(f"[LLM客户端] 工具执行失败: {tool_name}, 错误: {e}")
                results.append(
                    ToolCallResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=json.dumps({"error": str(e)}, ensure_ascii=False),
                        success=False,
                    )
                )

        return results

    def _build_tool_result_messages(
        self,
        tool_calls: list[dict[str, Any]],
        tool_results: list[ToolCallResult],
    ) -> list[ChatCompletionToolMessageParam]:
        """
        构建工具结果消息

        参数：
        - tool_calls: 工具调用列表
        - tool_results: 工具执行结果列表

        返回：
        - 工具结果消息列表
        """
        messages: list[ChatCompletionToolMessageParam] = []
        for tc, result in zip(tool_calls, tool_results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.result,
                }
            )
        return messages

    async def request_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        tool_executor: Any = None,
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        max_rounds: int = 5,
    ) -> str | None:
        """
        带工具调用的非流式请求（同步工具调用）

        支持 OpenAI function calling 协议，自动执行工具调用循环直到获得最终响应。

        参数：
        - messages: 消息列表
        - tools: 工具定义列表（OpenAI tools 格式）
        - tool_executor: 工具执行器（需提供 execute 方法），默认使用 ToolManager
        - model_key: 模型配置键名
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）
        - max_rounds: 最大工具调用轮次，防止死循环

        返回：
        - 最终响应文本，失败返回 None

        使用示例：
        ```python
        from my_utils.tool_manager import ToolManager

        client = LLMClient()
        tools = ToolManager.get_openai_tools()
        result = await client.request_with_tools(
            messages=[{"role": "user", "content": "你好"}],
            tools=tools,
            tool_executor=ToolManager,
        )
        ```
        """
        # 延迟导入避免循环依赖
        if tool_executor is None:
            from my_utils.tool_manager import ToolManager

            tool_executor = ToolManager

        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        current_messages = list(messages)

        for round_idx in range(max_rounds):
            try:
                kwargs = self._build_request_kwargs(
                    config, current_messages, tools, extra_body, stream=False
                )

                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=timeout,
                )

                message = response.choices[0].message

                # 没有工具调用，返回最终结果
                if not message.tool_calls:
                    content = message.content
                    await self._callbacks.emit(CallbackEvent.COMPLETE, content=content)
                    return content

                # 有工具调用，执行工具
                Log.info(
                    f"[LLM客户端] 工具调用轮次 {round_idx + 1}: "
                    f"{len(message.tool_calls)} 个工具调用"
                )

                # 将助手消息（包含 tool_calls）添加到消息列表
                normalized_calls = self._normalize_tool_calls(message.tool_calls)
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": normalized_calls,
                    }
                )

                # 执行工具调用
                tool_results = await self._execute_tool_calls(
                    normalized_calls, tool_executor
                )

                # 添加工具结果到消息列表
                tool_messages = self._build_tool_result_messages(
                    normalized_calls, tool_results
                )
                current_messages.extend(tool_messages)

            except asyncio.TimeoutError:
                Log.warning(f"[LLM客户端] 请求超时 ({timeout}s)")
                await self._callbacks.emit(CallbackEvent.ERROR, error="请求超时")
                return None

            except Exception as e:
                Log.error(f"[LLM客户端] 请求失败: {e}")
                await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
                return None

        Log.warning(f"[LLM客户端] 工具调用超过最大轮次 {max_rounds}，强制结束")
        return None

    async def stream_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        tool_executor: Any = None,
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        max_rounds: int = 5,
    ) -> AsyncIterator[str]:
        """
        带工具调用的流式请求（同步工具调用）

        流式输出文本内容，当检测到工具调用时执行工具并继续生成。

        参数：
        - messages: 消息列表
        - tools: 工具定义列表（OpenAI tools 格式）
        - tool_executor: 工具执行器（需提供 execute 方法），默认使用 ToolManager
        - model_key: 模型配置键名
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）
        - max_rounds: 最大工具调用轮次，防止死循环

        产出：
        - 文本内容片段

        使用示例：
        ```python
        from my_utils.tool_manager import ToolManager

        client = LLMClient()
        tools = ToolManager.get_openai_tools()
        async for chunk in client.stream_with_tools(
            messages=[{"role": "user", "content": "你好"}],
            tools=tools,
            tool_executor=ToolManager,
        ):
            print(chunk, end="", flush=True)
        ```
        """
        # 延迟导入避免循环依赖
        if tool_executor is None:
            from my_utils.tool_manager import ToolManager

            tool_executor = ToolManager

        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        current_messages = list(messages)
        streamed = False

        # 第一轮：流式请求
        tool_calls_by_index: dict[int, dict] = {}

        try:
            kwargs = self._build_request_kwargs(
                config, current_messages, tools, extra_body, stream=True
            )

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                if chunk is None or chunk.choices is None or len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta

                # 收集工具调用
                if delta and getattr(delta, "tool_calls", None):
                    for call in delta.tool_calls:
                        idx = call.index
                        tool_call = tool_calls_by_index.get(
                            idx,
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if call.id:
                            tool_call["id"] = call.id
                        if call.function and call.function.name:
                            tool_call["function"]["name"] = call.function.name
                        if call.function and call.function.arguments:
                            tool_call["function"][
                                "arguments"
                            ] += call.function.arguments
                        tool_calls_by_index[idx] = tool_call

                # 输出文本内容
                if delta and delta.content and not tool_calls_by_index:
                    streamed = True
                    await self._callbacks.emit(CallbackEvent.TOKEN, token=delta.content)
                    yield delta.content

        except Exception as e:
            Log.error(f"[LLM客户端] 流式请求失败: {e}")
            await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
            return

        # 没有工具调用，直接结束
        if not tool_calls_by_index:
            if not streamed:
                # 回退到普通流式请求
                async for chunk in self.stream(
                    messages=current_messages,
                    model_key=model_key,
                    extra_body=extra_body,
                    timeout=timeout,
                ):
                    yield chunk
            return

        # 有工具调用，进入工具调用循环
        tool_calls = [
            tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index.keys())
        ]

        # 将助手消息（包含 tool_calls）添加到消息列表
        current_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        )

        # 执行工具
        Log.info(f"[LLM客户端] 流式工具调用: {len(tool_calls)} 个工具")
        tool_results = await self._execute_tool_calls(tool_calls, tool_executor)
        tool_messages = self._build_tool_result_messages(tool_calls, tool_results)
        current_messages.extend(tool_messages)

        # 后续轮次：非流式请求（因为需要等待工具执行完成）
        for round_idx in range(1, max_rounds):
            try:
                kwargs = self._build_request_kwargs(
                    config, current_messages, tools, extra_body, stream=False
                )

                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=timeout,
                )

                message = response.choices[0].message

                # 没有工具调用，输出最终结果
                if not message.tool_calls:
                    if message.content:
                        for char in message.content:
                            yield char
                    return

                # 继续工具调用
                Log.info(
                    f"[LLM客户端] 工具调用轮次 {round_idx + 1}: "
                    f"{len(message.tool_calls)} 个工具调用"
                )

                normalized_calls = self._normalize_tool_calls(message.tool_calls)
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": normalized_calls,
                    }
                )

                tool_results = await self._execute_tool_calls(
                    normalized_calls, tool_executor
                )
                tool_messages = self._build_tool_result_messages(
                    normalized_calls, tool_results
                )
                current_messages.extend(tool_messages)

            except asyncio.TimeoutError:
                Log.warning(f"[LLM客户端] 请求超时 ({timeout}s)")
                await self._callbacks.emit(CallbackEvent.ERROR, error="请求超时")
                return

            except Exception as e:
                Log.error(f"[LLM客户端] 请求失败: {e}")
                await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
                return

        Log.warning(f"[LLM客户端] 工具调用超过最大轮次 {max_rounds}，强制结束")

    async def stream_with_tools_async(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        tool_executor: Any = None,
        model_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        max_rounds: int = 5,
    ) -> AsyncIterator[str]:
        """
        带工具调用的流式请求（后台异步工具调用）

        当检测到工具调用时，在后台异步执行工具，同时继续流式输出。
        适用于工具执行耗时较长的场景。

        参数：
        - messages: 消息列表
        - tools: 工具定义列表（OpenAI tools 格式）
        - tool_executor: 工具执行器（需提供 execute 方法），默认使用 ToolManager
        - model_key: 模型配置键名
        - extra_body: 额外请求参数
        - timeout: 超时时间（秒）
        - max_rounds: 最大工具调用轮次，防止死循环

        产出：
        - 文本内容片段

        注意：
        - 此方法会创建后台任务执行工具调用
        - 工具执行完成后，会自动发起新一轮请求
        - 适用于需要同时输出文本和执行工具的场景
        """
        # 延迟导入避免循环依赖
        if tool_executor is None:
            from my_utils.tool_manager import ToolManager

            tool_executor = ToolManager

        effective_model_key = model_key or self._model_key
        config = self._get_model_config(effective_model_key)
        client = self._get_client(effective_model_key)

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        current_messages = list(messages)

        for round_idx in range(max_rounds):
            tool_calls_by_index: dict[int, dict] = {}
            streamed = False

            try:
                kwargs = self._build_request_kwargs(
                    config, current_messages, tools, extra_body, stream=True
                )

                response = await client.chat.completions.create(**kwargs)

                # 后台工具执行任务
                tool_execution_task: asyncio.Task | None = None

                async for chunk in response:
                    if (
                        chunk is None
                        or chunk.choices is None
                        or len(chunk.choices) == 0
                    ):
                        continue

                    delta = chunk.choices[0].delta

                    # 收集工具调用
                    if delta and getattr(delta, "tool_calls", None):
                        for call in delta.tool_calls:
                            idx = call.index
                            tool_call = tool_calls_by_index.get(
                                idx,
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if call.id:
                                tool_call["id"] = call.id
                            if call.function and call.function.name:
                                tool_call["function"]["name"] = call.function.name
                            if call.function and call.function.arguments:
                                tool_call["function"][
                                    "arguments"
                                ] += call.function.arguments
                            tool_calls_by_index[idx] = tool_call

                    # 输出文本内容
                    if delta and delta.content and not tool_calls_by_index:
                        streamed = True
                        await self._callbacks.emit(
                            CallbackEvent.TOKEN, token=delta.content
                        )
                        yield delta.content

                # 没有工具调用，结束
                if not tool_calls_by_index:
                    if not streamed:
                        async for chunk in self.stream(
                            messages=current_messages,
                            model_key=model_key,
                            extra_body=extra_body,
                            timeout=timeout,
                        ):
                            yield chunk
                    return

                # 有工具调用，执行工具
                tool_calls = [
                    tool_calls_by_index[idx]
                    for idx in sorted(tool_calls_by_index.keys())
                ]

                # 将助手消息添加到消息列表
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )

                # 后台执行工具调用
                Log.info(
                    f"[LLM客户端] 后台异步工具调用轮次 {round_idx + 1}: "
                    f"{len(tool_calls)} 个工具"
                )

                tool_results = await self._execute_tool_calls(tool_calls, tool_executor)
                tool_messages = self._build_tool_result_messages(
                    tool_calls, tool_results
                )
                current_messages.extend(tool_messages)

            except Exception as e:
                Log.error(f"[LLM客户端] 请求失败: {e}")
                await self._callbacks.emit(CallbackEvent.ERROR, error=str(e))
                return

        Log.warning(f"[LLM客户端] 工具调用超过最大轮次 {max_rounds}，强制结束")
