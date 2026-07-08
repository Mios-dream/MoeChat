"""
工具路由器模块

ToolRouter 是整个工具系统的核心调度组件。

根据 ToolCallRequest 中的 (domain, mode) 组合，
将工具调用路由到对应的执行器（共 4 种策略）。

策略路由表:
    +------------------+------------------+------------------+
    | Domain \ Mode    |      SYNC        |      ASYNC       |
    +------------------+------------------+------------------+
    |    SERVER        | ServerSyncExec   | ServerAsyncExec  |
    |    CLIENT        | ClientSyncExec   | ClientAsyncExec  |
    +------------------+------------------+------------------+

主要功能:
- 单工具调用: route() → 返回 ToolCallResult
- 批量并行调用: route_batch() → 返回 list[ToolCallResult]
- 执行器管理: 统一管理 4 种执行器的生命周期
"""

from __future__ import annotations

import asyncio
from typing import Any

from tool_system.core.enums import ExecutionDomain, ExecutionMode
from tool_system.core.errors import (
    ToolNotFoundError,
    ToolSystemError,
)
from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
    ToolMeta,
)
from tool_system.core.registry import get_registry
from tool_system.executors.server_executor import (
    ServerSyncExecutor,
    ServerAsyncExecutor,
)
from tool_system.executors.client_executor import (
    ClientSyncExecutor,
    ClientAsyncExecutor,
    PendingCallTable,
)
import json


class ToolRouter:
    """
    工具路由器

    核心职责:
    1. 策略路由: 根据 (domain, mode) 将调用分发到正确执行器
    2. 批量执行: 并行执行多个 tools call（asyncio.gather）
    3. 容错处理: 统一异常捕获，保证所有调用都返回 ToolCallResult

    使用示例:
        router = ToolRouter(ws_manager=ws_manager)
        result = await router.route(request)

        # 批量并行调用
        results = await router.route_batch(requests)
    """

    def __init__(
        self,
        pending_table: PendingCallTable | None = None,
    ) -> None:
        """
        初始化工具路由器

        创建 6 种执行器实例并连接 WebSocket 管理器。

        Args:
            ws_manager: WebSocketManager 实例（用于客户端工具通信）
            pending_table: PendingCallTable 实例（用于管理客户端待处理调用）
        """
        self._registry = get_registry()
        self._pending_table = pending_table or PendingCallTable()

        # ── 服务端执行器 ──
        self._server_sync = ServerSyncExecutor()
        self._server_async = ServerAsyncExecutor()

        # ── 客户端执行器 ──
        self._client_sync = ClientSyncExecutor(
            pending_table=self._pending_table,
        )
        self._client_async = ClientAsyncExecutor(
            pending_table=self._pending_table,
        )

    def set_ws_manager(self, ws_manager: Any) -> None:
        """
        设置/更新 WebSocket 连接管理器

        将所有客户端和混合执行器绑定到新的 WS 管理器。

        Args:
            ws_manager: WebSocketManager 实例
        """
        self._client_sync.set_ws_manager(ws_manager)
        self._client_async.set_ws_manager(ws_manager)

    def set_complete_callback(self, callback: Any) -> None:
        """
        为所有异步执行器设置完成回调

        通常设置为 ResultNotifier.notify。

        Args:
            callback: async def (result: ToolCallResult) -> None
        """
        self._server_async.set_complete_callback(callback)
        self._client_async.set_complete_callback(callback)

    def _get_executor(self, domain: ExecutionDomain, mode: ExecutionMode) -> Any:
        """
        根据 (domain, mode) 获取对应的执行器

        这是策略路由的实现核心。

        Args:
            domain: 执行域
            mode: 执行模式

        Returns:
            对应的执行器实例

        Raises:
            ValueError: 不支持的 (domain, mode) 组合
        """
        strategy_map = {
            # ── 服务端 ──
            (ExecutionDomain.SERVER, ExecutionMode.SYNC): self._server_sync,
            (ExecutionDomain.SERVER, ExecutionMode.ASYNC): self._server_async,
            # ── 客户端 ──
            (ExecutionDomain.CLIENT, ExecutionMode.SYNC): self._client_sync,
            (ExecutionDomain.CLIENT, ExecutionMode.ASYNC): self._client_async,
        }

        executor = strategy_map.get((domain, mode))
        if executor is None:
            raise ValueError(
                f"不支持的路由策略: domain={domain.value}, mode={mode.value}"
            )
        return executor

    async def route(self, request: ToolCallRequest) -> ToolCallResult:
        """
        路由单个工具调用到对应执行器

        内部流程:
        1. 查 ToolRegistry 确认工具已注册
        2. 从 ToolMeta 获取 domain 和 mode
        3. 查策略路由表获取执行器
        4. 调用执行器并返回结果

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 执行结果
        """
        # 获取工具元信息
        meta = self._registry.get(request.tool_name)
        if meta is None:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f'{{"error": "工具 \\"{request.tool_name}\\" 未注册"}}',
                success=False,
                error=f"工具 '{request.tool_name}' 未注册",
                error_code="TOOL_NOT_FOUND",
                session_id=request.session_id,
            )

        # 将元信息中的 domain/mode 设置到请求中
        request.domain = meta.domain
        request.mode = meta.mode
        request.timeout = meta.timeout
        request.max_retries = meta.max_retries
        request.sensitivity = meta.sensitivity

        # 获取执行器
        try:
            executor = self._get_executor(request.domain, request.mode)
        except ValueError as e:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f'{{"error": "{str(e)}"}}',
                success=False,
                error=str(e),
                error_code="TOOL_EXEC_ERROR",
                session_id=request.session_id,
            )

        # 执行
        try:
            return await executor.execute(request)
        except Exception as e:
            import json

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=json.dumps({"error": f"工具执行异常: {e}"}, ensure_ascii=False),
                success=False,
                error=str(e),
                error_code="TOOL_EXEC_ERROR",
                session_id=request.session_id,
            )

    async def route_batch(
        self, requests: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """
        并行路由多个工具调用

        使用 asyncio.gather 并行执行所有工具调用。
        SYNC 工具作为 gather 的一部分等待。
        ASYNC 工具立即返回占位结果，实际执行在后台进行。

        容错原则:
        - 单个工具失败不影响其他工具
        - 所有异常被捕获并包装为 ToolCallResult(success=False)
        - 返回列表与输入列表一一对应

        Args:
            requests: 工具调用请求列表

        Returns:
            工具调用结果列表（与输入顺序一致）
        """
        if not requests:
            return []

        tasks = []
        for req in requests:
            tasks.append(self.route(req))

        # 使用 return_exceptions=True 确保单个失败不影响其他
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ToolCallResult] = []
        for i, item in enumerate(gathered):
            if isinstance(item, ToolCallResult):
                results.append(item)
            elif isinstance(item, BaseException):
                results.append(
                    ToolCallResult(
                        call_id=requests[i].call_id,
                        tool_name=requests[i].tool_name,
                        content=json.dumps({"error": str(item)}, ensure_ascii=False),
                        success=False,
                        error=str(item),
                        error_code="TOOL_EXEC_ERROR",
                        session_id=requests[i].session_id,
                    )
                )
            else:

                results.append(
                    ToolCallResult(
                        call_id=requests[i].call_id,
                        tool_name=requests[i].tool_name,
                        content=json.dumps(
                            {"error": f"未知返回值类型: {type(item).__name__}"},
                            ensure_ascii=False,
                        ),
                        success=False,
                        error=f"未知返回值类型: {type(item).__name__}",
                        error_code="TOOL_EXEC_ERROR",
                        session_id=requests[i].session_id,
                    )
                )
        return results

    @property
    def pending_table(self) -> PendingCallTable:
        """获取待处理调用表（用于客户端工具结果回调）"""
        return self._pending_table

    def cancel_all_for_session(self, session_id: str) -> int:
        """
        取消指定会话的所有待处理客户端工具调用

        当客户端断开连接时调用。

        Args:
            session_id: 会话 ID

        Returns:
            成功取消的调用数量
        """
        return self._pending_table.cancel_all_for_session(
            session_id=session_id,
            error_message=f"客户端 '{session_id}' 已断开",
        )
