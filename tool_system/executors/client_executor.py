"""
客户端工具执行器模块

实现客户端工具的同步和异步执行策略。
客户端工具不在服务端执行，而是通过 WebSocket 下发给客户端。

核心数据结构:
    Pending Call Table: {call_id: asyncio.Future}
    维护所有"已下发但尚未收到客户端响应"的调用。

客户端同步工具 (ClientSyncExecutor):
    1. 创建 asyncio.Future 存入 Pending Call Table
    2. 通过 WebSocketManager 发送 tool:call 消息
    3. await Future（带超时）
    4. 客户端返回结果 → Future.set_result() → 返回

客户端异步工具 (ClientAsyncExecutor):
    1. 通过 WebSocketManager 发送 tool:call 消息
    2. 立即返回占位结果（不等待）
    3. 注册异步回调监听
    4. 客户端结果到达时通过 ResultNotifier 通知
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from tool_system.core.enums import ExecutionMode
from tool_system.core.errors import (
    ToolTimeoutError,
    ClientDisconnectedError,
)
from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
)
from tool_system.executors.base_executor import BaseExecutor


class PendingCallTable:
    """
    待处理调用表

    维护所有已下发但尚未收到客户端响应的工具调用。
    每个调用对应一个 asyncio.Future，客户端返回结果时唤醒。

    线程安全说明:
        此表所有操作必须在同一个 event loop 中执行。
        不支持跨 loop 的并发访问。
    """

    def __init__(self) -> None:
        """初始化待处理调用表"""
        self._pending: dict[str, asyncio.Future[ToolCallResult]] = {}
        """call_id → Future 映射"""

        self._async_listeners: dict[str, Any] = {}
        """
        异步工具回调注册表: call_id → callback
        callback 签名: async def (result: ToolCallResult) -> None
        """

    def add_future(self, call_id: str) -> asyncio.Future[ToolCallResult]:
        """
        创建并注册一个新的 Future

        Args:
            call_id: 工具调用 ID

        Returns:
            新创建的 asyncio.Future
        """
        future: asyncio.Future[ToolCallResult] = (
            asyncio.get_event_loop().create_future()
        )
        self._pending[call_id] = future
        return future

    def add_listener(
        self, call_id: str, callback: Any
    ) -> None:
        """
        注册异步工具的结果监听器

        Args:
            call_id: 工具调用 ID
            callback: async def (result: ToolCallResult) -> None
        """
        self._async_listeners[call_id] = callback

    def resolve(self, call_id: str, result: ToolCallResult) -> bool:
        """
        完成一个待处理的调用

        优先查找 Future（同步调用），其次查找异步监听器。

        Args:
            call_id: 工具调用 ID
            result: 客户端返回的结果

        Returns:
            True: 成功匹配到一个等待者
            False: 没有找到对应的等待者（可能已超时清理）

        Raises:
            RuntimeError: Future 已完成（重复回调）
        """
        future = self._pending.get(call_id)
        if future is not None:
            if future.done():
                return False  # 已完成，跳过
            future.set_result(result)
            return True

        # 尝试异步监听器
        listener = self._async_listeners.pop(call_id, None)
        if listener is not None:
            # 异步调用监听器（不阻塞当前线程，使用 create_task）
            asyncio.create_task(
                listener(result), name=f"async_listener_{call_id}"
            )
            return True

        return False

    def cancel_future(self, call_id: str, error_message: str) -> bool:
        """
        取消一个待处理的 Future

        通常在客户端断开连接时调用。

        Args:
            call_id: 工具调用 ID
            error_message: 取消原因

        Returns:
            True: 成功取消了一个未完成的 Future
        """
        future = self._pending.get(call_id)
        if future is not None and not future.done():
            exception = ClientDisconnectedError(
                session_id="unknown",
                call_id=call_id,
            )
            future.set_exception(exception)
            return True
        return False

    def cancel_all_for_session(
        self, session_id: str, error_message: str
    ) -> int:
        """
        取消指定会话的所有待处理调用

        在客户端断开连接时调用，清理所有该会话的 pending calls。

        Args:
            session_id: 会话 ID
            error_message: 取消原因

        Returns:
            成功取消的调用数量
        """
        cancelled = 0
        # 收集需要取消的 call_id（避免在遍历时修改字典）
        to_cancel = list(self._pending.keys())
        for call_id in to_cancel:
            if self.cancel_future(call_id, error_message):
                cancelled += 1
        return cancelled

    def remove(self, call_id: str) -> None:
        """
        从表中移除指定调用（无论是否完成）

        用于超时清理和正常完成后的清理。

        Args:
            call_id: 工具调用 ID
        """
        self._pending.pop(call_id, None)
        self._async_listeners.pop(call_id, None)

    @property
    def pending_count(self) -> int:
        """当前待处理的调用数量"""
        return len(self._pending)

    def __repr__(self) -> str:
        """人类可读的状态"""
        return (
            f"<PendingCallTable: {self.pending_count} pending, "
            f"{len(self._async_listeners)} async listeners>"
        )


class ClientSyncExecutor(BaseExecutor):
    """
    客户端同步工具执行器

    通过 WebSocket 下发给客户端并等待结果返回。

    依赖:
        WebSocketManager: 用于发送消息到客户端

    执行流程:
        1. 创建 Future 并注册到 Pending Call Table
        2. 通过 WebSocketManager 发送 tool:call 消息
        3. await Future（带超时）
        4. 返回结果

    Attributes:
        _ws_manager: WebSocket 连接管理器（发送消息）
        _pending_table: 待处理调用表
    """

    def __init__(
        self,
        ws_manager: Any = None,
        pending_table: PendingCallTable | None = None,
    ) -> None:
        """
        初始化客户端同步执行器

        Args:
            ws_manager: WebSocketManager 实例（用于与客户端通信）
            pending_table: PendingCallTable 实例（用于管理待处理调用）
        """
        super().__init__()
        self._ws_manager = ws_manager
        self._pending_table = pending_table or PendingCallTable()

    def set_ws_manager(self, ws_manager: Any) -> None:
        """
        设置 WebSocket 连接管理器

        Args:
            ws_manager: WebSocketManager 实例
        """
        self._ws_manager = ws_manager

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        同步执行客户端工具（阻塞等待客户端结果）

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 客户端返回的结果或超时错误
        """
        if self._ws_manager is None:
            return self._wrap_error(
                request,
                "客户端工具执行器未配置 WebSocket 管理器",
                error_code="TOOL_EXEC_ERROR",
            )

        # 检查客户端连接状态
        if not await self._ws_manager.is_connected(request.session_id):
            return self._wrap_error(
                request,
                f"客户端 '{request.session_id}' 未连接",
                error_code="CLIENT_DISCONNECTED",
            )

        # 创建 Future 并注册
        future = self._pending_table.add_future(request.call_id)

        # 通过 WebSocket 下发工具调用
        try:
            message = {
                "type": "tool:call",
                "call_id": request.call_id,
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "timeout_ms": int(request.timeout * 1000),
                "sensitivity": request.sensitivity.value,
            }
            await self._ws_manager.send(request.session_id, message)
        except Exception as e:
            self._pending_table.remove(request.call_id)
            return self._wrap_error(
                request,
                f"下发客户端工具调用失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        # 等待客户端返回结果
        start_time = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                future,
                timeout=request.timeout,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            result.duration_ms = round(duration_ms, 2)
            result.session_id = request.session_id
            return result
        except asyncio.TimeoutError:
            # 超时：移除 Future 并返回错误
            self._pending_table.remove(request.call_id)
            return self._wrap_error(
                request,
                f"客户端工具 '{request.tool_name}' 执行超时（{request.timeout}s）",
                error_code="TOOL_TIMEOUT",
            )
        except ClientDisconnectedError as e:
            self._pending_table.remove(request.call_id)
            return self._wrap_error(
                request,
                str(e),
                error_code="CLIENT_DISCONNECTED",
            )
        except Exception as e:
            self._pending_table.remove(request.call_id)
            return self._wrap_error(
                request,
                f"客户端工具 '{request.tool_name}' 执行异常: {e}",
                error_code="TOOL_EXEC_ERROR",
            )


class ClientAsyncExecutor(BaseExecutor):
    """
    客户端异步工具执行器

    通过 WebSocket 下发工具调用后立即返回占位结果，不等待客户端响应。
    客户端结果到达后通过 ResultNotifier 回调通知。

    适用场景:
    - 后台任务（如下载、截图）
    - 不需要立即反馈的操作
    - 可能耗时较长的客户端操作

    Attributes:
        _ws_manager: WebSocket 连接管理器
        _pending_table: 待处理调用表
        _on_complete: 异步完成回调（ResultNotifier.notify）
    """

    def __init__(
        self,
        ws_manager: Any = None,
        pending_table: PendingCallTable | None = None,
    ) -> None:
        """
        初始化客户端异步执行器

        Args:
            ws_manager: WebSocketManager 实例
            pending_table: PendingCallTable 实例
        """
        super().__init__()
        self._ws_manager = ws_manager
        self._pending_table = pending_table or PendingCallTable()
        self._on_complete: Any = None

    def set_ws_manager(self, ws_manager: Any) -> None:
        """设置 WebSocket 连接管理器"""
        self._ws_manager = ws_manager

    def set_complete_callback(self, callback: Any) -> None:
        """
        设置异步完成回调

        Args:
            callback: async def (result: ToolCallResult) -> None
        """
        self._on_complete = callback

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        异步执行客户端工具（fire-and-forget）

        下发工具调用后立即返回占位结果。

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 占位回复结果
        """
        if self._ws_manager is None:
            return self._wrap_error(
                request,
                "客户端工具执行器未配置 WebSocket 管理器",
                error_code="TOOL_EXEC_ERROR",
            )

        # 注册异步回调
        if self._on_complete is not None:

            async def _on_result(result: ToolCallResult) -> None:
                result.is_async_result = True
                result.session_id = request.session_id
                await self._on_complete(result)

            self._pending_table.add_listener(request.call_id, _on_result)

        # 下发工具调用
        try:
            message = {
                "type": "tool:call",
                "call_id": request.call_id,
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "timeout_ms": int(request.timeout * 1000),
                "sensitivity": request.sensitivity.value,
                "mode": "async",
            }
            await self._ws_manager.send(request.session_id, message)
        except Exception as e:
            self._pending_table.remove(request.call_id)
            return self._wrap_error(
                request,
                f"下发客户端异步工具调用失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        # 生成占位结果
        meta = self._registry.get(request.tool_name)
        placeholder_text = (
            meta.placeholder
            if meta and meta.placeholder
            else f"任务 '{request.tool_name}' 已在后台执行。"
        )

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            content=placeholder_text,
            success=True,
            is_async_result=True,
            session_id=request.session_id,
        )
