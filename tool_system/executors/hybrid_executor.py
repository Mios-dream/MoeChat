"""
混合工具执行器模块

实现混合工具的同步和异步执行策略。

混合工具分两阶段执行:
    Phase 1 (Client):  通过 ClientExecutor 获取客户端数据
    Phase 2 (Server):  通过 ServerExecutor 处理服务端逻辑

同步混合工具 (HybridSyncExecutor):
    1. Client Phase → 等客户端返回数据
    2. Server Phase → 处理数据并返回
    3. 最终结果返回给 LLM

异步混合工具 (HybridAsyncExecutor):
    1. 立即返回占位结果
    2. 后台异步执行 Client Phase → Server Phase
    3. 完成后通过 ResultNotifier 通知
"""

import asyncio
import json
import time
from typing import Any

from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
)
from tool_system.executors.base_executor import BaseExecutor
from tool_system.executors.server_executor import (
    ServerSyncExecutor,
)
from tool_system.executors.client_executor import (
    ClientSyncExecutor,
    PendingCallTable,
)


class HybridSyncExecutor(BaseExecutor):
    """
    混合同步工具执行器

    编排客户端阶段和服务端阶段的同步执行流程:
    1. 客户端阶段: 通过 ClientSyncExecutor 下发指令并等待客户端返回
    2. 服务端阶段: 在服务端处理客户端数据并返回最终结果

    Attributes:
        _client_executor: 客户端同步执行器
        _server_executor: 服务端同步执行器
    """

    def __init__(
        self,
        ws_manager: Any = None,
        pending_table: PendingCallTable | None = None,
    ) -> None:
        """
        初始化混合同步执行器

        Args:
            ws_manager: WebSocketManager 实例
            pending_table: PendingCallTable 实例
        """
        super().__init__()
        self._client_executor = ClientSyncExecutor(
            ws_manager=ws_manager,
            pending_table=pending_table or PendingCallTable(),
        )
        self._server_executor = ServerSyncExecutor()

    def set_ws_manager(self, ws_manager: Any) -> None:
        """
        设置 WebSocket 连接管理器

        Args:
            ws_manager: WebSocketManager 实例
        """
        self._client_executor.set_ws_manager(ws_manager)

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        同步执行混合工具（两阶段编排）

        Phase 1 - 客户端阶段:
            从 Registry 获取混合工具类 → 实例化 → 调用 client_phase() 生成客户端指令
            → 构建临时 ToolCallRequest → ClientSyncExecutor 下发并等待

        Phase 2 - 服务端阶段:
            客户端返回数据 → 调用 server_phase(client_result, **args)
            → 返回最终结果

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 服务端阶段的最终结果
        """
        tool_name = request.tool_name
        start_time = time.perf_counter()

        # 获取混合工具类
        tool_class = self._registry.get_class(tool_name)
        if tool_class is None:
            return self._wrap_error(
                request,
                f"混合工具 '{tool_name}' 未注册",
                error_code="TOOL_NOT_FOUND",
            )

        # 实例化
        try:
            hybrid_tool = tool_class()
        except Exception as e:
            return self._wrap_error(
                request,
                f"混合工具 '{tool_name}' 实例化失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        # 校验参数
        try:
            validated_args = hybrid_tool.validate_arguments(request.arguments)
        except Exception as e:
            return self._wrap_error(
                request,
                f"参数校验失败: {e}",
                error_code="INVALID_ARGUMENTS",
            )

        # ── Phase 1: 客户端阶段 ──
        try:
            client_payload = await hybrid_tool.client_phase(**validated_args)
        except Exception as e:
            return self._wrap_error(
                request,
                f"混合工具客户端阶段准备失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        # 构建客户端请求
        client_request = ToolCallRequest(
            call_id=f"{request.call_id}_phase1",
            tool_name=request.tool_name,
            arguments={
                **validated_args,
                "__hybrid_phase": "client",
                "__client_payload": client_payload,
            },
            domain=request.domain,
            mode=request.mode,
            session_id=request.session_id,
            timeout=request.timeout * 0.6,  # 客户端阶段占 60% 超时
        )

        # 执行客户端阶段
        client_result = await self._client_executor.execute(client_request)
        if not client_result.success:
            client_result.call_id = request.call_id
            return client_result

        # 解析客户端返回的数据
        client_data: dict[str, Any] = {}
        try:
            content = json.loads(client_result.content)
            client_data = content if isinstance(content, dict) else {}
        except (json.JSONDecodeError, TypeError):
            # 如果客户端返回的不是 JSON，将原始字符串作为数据
            client_data = {"raw_result": client_result.content}

        # ── Phase 2: 服务端阶段 ──
        try:
            final_result = await hybrid_tool.server_phase(client_data, **validated_args)
        except Exception as e:
            return self._wrap_error(
                request,
                f"混合工具服务端阶段执行失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolCallResult(
            call_id=request.call_id,
            tool_name=tool_name,
            content=final_result,
            success=True,
            duration_ms=round(duration_ms, 2),
            session_id=request.session_id,
        )


class HybridAsyncExecutor(BaseExecutor):
    """
    混合异步工具执行器

    与 HybridSyncExecutor 的编排逻辑相同，但整个流程在后台执行。
    调用后立即返回占位结果，完成后通过回调通知。

    Attributes:
        _client_executor: 客户端执行器
        _server_executor: 服务端执行器
        _on_complete: 异步完成回调
    """

    def __init__(
        self,
        ws_manager: Any = None,
        pending_table: PendingCallTable | None = None,
    ) -> None:
        """
        初始化混合异步执行器

        Args:
            ws_manager: WebSocketManager 实例
            pending_table: PendingCallTable 实例
        """
        super().__init__()
        self._sync_executor = HybridSyncExecutor(
            ws_manager=ws_manager,
            pending_table=pending_table or PendingCallTable(),
        )
        self._on_complete: Any = None

    def set_ws_manager(self, ws_manager: Any) -> None:
        """设置 WebSocket 连接管理器"""
        self._sync_executor.set_ws_manager(ws_manager)

    def set_complete_callback(self, callback: Any) -> None:
        """
        设置异步完成回调

        Args:
            callback: async def (result: ToolCallResult) -> None
        """
        self._on_complete = callback

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        异步执行混合工具

        立即返回占位结果，后台执行两阶段流程。

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 占位回复结果
        """
        meta = self._registry.get(request.tool_name)
        placeholder_text = (
            meta.placeholder
            if meta and meta.placeholder
            else f"任务 '{request.tool_name}' 正在处理中。"
        )

        # 创建后台执行任务
        async def _background_hybrid() -> None:
            result = await self._sync_executor.execute(request)
            result.is_async_result = True
            result.session_id = request.session_id
            if self._on_complete is not None:
                try:
                    await self._on_complete(result)
                except Exception:
                    pass

        asyncio.create_task(
            _background_hybrid(),
            name=f"hybrid_async_{request.call_id}",
        )

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            content=placeholder_text,
            success=True,
            is_async_result=True,
            session_id=request.session_id,
        )
