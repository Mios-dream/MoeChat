"""
工具系统集成模块

ToolCallIntegration 封装 LLM 工具调用"接收 → 执行 → 聚合 → 返回"的完整流程，
作为 middleware 注入到 Pipeline 和 ChatContext 之间。

使用方式:
    from tool_system.integration import ToolCallIntegration

    integration = ToolCallIntegration(ws_manager)
    integration.set_session_id(session_id)

    # 获取工具定义
    tools = integration.get_tools()

    # 作为 tool_handler 注入 Pipeline
    pipeline = Pipeline(
        ...,
        tools=tools,
        tool_handler=integration.process_tool_calls,
    )

    # 处理客户端工具结果回调
    integration.on_client_result(session_id, payload)
"""

import json
from typing import Any

from tool_system.core.registry import get_registry
from tool_system.core.router import ToolRouter
from tool_system.core.aggregator import ResultAggregator
from tool_system.core.types import ToolCallRequest, ToolCallResult
from tool_system.core.enums import ExecutionDomain, ExecutionMode
from core.scheduler.task import ToolCallEvent, ToolResultEvent, ToolExecutionResult
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
)


class ToolCallIntegration:
    """
    工具调用集成 - 完整的一次工具调用处理

    直接组合 ToolRouter + ResultAggregator，封装
    "接收 LLM tool_calls → 并行执行 → 聚合结果 → 返回 tool 消息"的完整流程。

    使用场景:
    - 在已有的聊天流程中,LLM 决定调用工具时使用
    - 作为 middleware 注入到 Pipeline 和 ChatContext 之间

    Attributes:
        _router: 工具路由器（负责策略路由和执行）
        _aggregator: 结果聚合器（负责转换为 OpenAI tool 消息格式）
        _session_id: 当前会话 ID（工具调用时自动注入）
        _ws_manager: WebSocket 连接管理器引用
    """

    def __init__(self) -> None:
        """
        初始化工具调用集成

        Args:
            ws_manager: WebSocketManager 实例（用于客户端工具通信）
        """
        self._router = ToolRouter()
        self._aggregator = ResultAggregator()
        self._session_id: str = "default"

    def set_session_id(self, session_id: str) -> None:
        """
        设置当前会话 ID

        Args:
            session_id: 会话唯一标识
        """
        self._session_id = session_id

    async def process_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam],
    ) -> ToolExecutionResult:
        """
        处理 LLM 返回的工具调用列表

        一次构建 ToolCallRequest 列表 → 并行执行 → 聚合结果，
        消除重复 registry 查询和重复请求构建。

        Args:
            tool_calls: OpenAI 格式的工具调用列表
                每个元素: {id, type, function: {name, arguments}}

        Returns:
            ToolExecutionResult: 包含标准化事件列表和 LLM 上下文注入消息
        """
        registry = get_registry()

        requests: list[ToolCallRequest] = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            call_id = tc["id"]

            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            meta = registry.get(tool_name)

            if meta is not None:
                request = ToolCallRequest(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    domain=meta.domain,
                    mode=meta.mode,
                    session_id=self._session_id,
                    timeout=meta.timeout,
                    sensitivity=meta.sensitivity,
                )
            else:
                request = ToolCallRequest(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    domain=ExecutionDomain.SERVER,
                    mode=ExecutionMode.SYNC,
                    session_id=self._session_id,
                )

            requests.append(request)

        raw_results = await self._router.route_batch(requests)
        tool_messages = self._aggregator.aggregate(raw_results, requests)

        args_by_call_id = {req.call_id: req.arguments for req in requests}

        tool_call_events = [
            ToolCallEvent(
                call_id=tc["id"],
                tool_name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for tc in tool_calls
        ]

        tool_result_events = [
            ToolResultEvent(
                call_id=r.call_id,
                tool_name=r.tool_name,
                arguments=args_by_call_id.get(r.call_id, {}),
                content=r.content,
                success=r.success,
                error=r.error,
                error_code=r.error_code,
                duration_ms=r.duration_ms,
            )
            for r in raw_results
        ]

        return ToolExecutionResult(
            tool_call_events=tool_call_events,
            tool_result_events=tool_result_events,
            context_messages=tool_messages,
        )

    def get_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """
        获取所有已注册工具的 OpenAI 格式定义

        Returns:
            OpenAI function calling 格式的工具定义列表
        """
        return get_registry().build_openai_tools()

    def cancel_session(self, session_id: str) -> int:
        """
        取消指定会话的所有待处理调用

        客户端断开时调用。

        Args:
            session_id: 会话 ID

        Returns:
            取消的调用数量
        """
        return self._router.cancel_all_for_session(session_id)

    def on_client_result(self, session_id: str, payload: dict[str, Any]) -> bool:
        """
        处理客户端返回的工具执行结果

        将 WebSocket 收到的 tool:result 消息解析后
        通过 PendingCallTable 唤醒等待的 Future。

        Args:
            session_id: 客户端会话 ID
            payload: tool:result 消息内容

        Returns:
            是否成功匹配到等待的调用
        """
        call_id = payload.get("call_id", "")
        result = ToolCallResult(
            call_id=call_id,
            tool_name=payload.get("tool_name", "unknown"),
            content=payload.get("result", ""),
            success=payload.get("success", True),
            error=payload.get("error"),
            error_code=payload.get("error_code"),
            session_id=session_id,
        )
        return self._router.pending_table.resolve(call_id, result)
