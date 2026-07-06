"""
工具执行器抽象基类

定义所有执行器的公共接口和通用逻辑：
- 超时控制
- 异常捕获与包装
- 进度回调
- 参数校验
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from tool_system.core.enums import ExecutionMode
from tool_system.core.errors import (
    ToolExecutionError,
    ToolTimeoutError,
)
from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
    ToolCallProgress,
    ToolMeta,
)
from tool_system.core.registry import get_registry


class BaseExecutor(ABC):
    """
    工具执行器抽象基类

    所有执行器（服务端/客户端/混合）都继承此类。
    提供公共的错误处理、超时控制和进度通知机制。

    子类必须实现:
        _do_sync_execute():  同步执行的内部逻辑
        _do_async_execute(): 异步执行的内部逻辑
    """

    # ── 超时配置 ──
    _default_timeout: float = 30.0
    """默认超时时间（秒）"""

    # ── 进度回调 ──
    _progress_callbacks: list[Any] | None = None
    """进度回调函数列表，可在执行过程中推送进度"""

    def __init__(self) -> None:
        """初始化执行器"""
        self._registry = get_registry()

    def on_progress(self, callback: Any) -> None:
        """
        注册进度回调

        Args:
            callback: async def callback(progress: ToolCallProgress) -> None
        """
        if self._progress_callbacks is None:
            self._progress_callbacks = []
        self._progress_callbacks.append(callback)

    async def _notify_progress(self, progress: ToolCallProgress) -> None:
        """
        通知所有注册的进度回调

        Args:
            progress: 进度信息
        """
        if self._progress_callbacks is None:
            return
        for cb in self._progress_callbacks:
            try:
                await cb(progress)
            except Exception:
                pass

    @abstractmethod
    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        执行工具调用的统一入口

        根据请求的 mode（SYNC/ASYNC）自动选择对应的执行策略。

        Args:
            request: 工具调用请求

        Returns:
            工具调用结果
        """
        ...

    async def _execute_with_timeout(
        self,
        coro: Any,
        request: ToolCallRequest,
    ) -> Any:
        """
        带超时控制的异步执行

        用 asyncio.wait_for 包裹协程，超时时抛出 ToolTimeoutError。

        Args:
            coro: 要执行的协程
            request: 工具调用请求（提供 timeout 配置）

        Returns:
            协程的执行结果

        Raises:
            ToolTimeoutError: 执行超时
        """
        timeout = request.timeout or self._default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(
                tool_name=request.tool_name,
                timeout=timeout,
                call_id=request.call_id,
            )

    def _wrap_success(self, request: ToolCallRequest, content: str) -> ToolCallResult:
        """
        包装成功结果

        Args:
            request: 原始调用请求
            content: 工具返回的 JSON 字符串

        Returns:
            ToolCallResult
        """
        return ToolCallResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            content=content,
            success=True,
            session_id=request.session_id,
        )

    def _wrap_error(
        self,
        request: ToolCallRequest,
        error_message: str,
        error_code: str = "TOOL_EXEC_ERROR",
    ) -> ToolCallResult:
        """
        包装失败结果

        构建标准化的错误结果，content 中包含 JSON 格式的错误信息。

        Args:
            request: 原始调用请求
            error_message: 错误描述
            error_code: 错误码

        Returns:
            ToolCallResult（success=False）
        """
        error_content = json.dumps(
            {
                "success": False,
                "error": error_message,
                "error_code": error_code,
            },
            ensure_ascii=False,
        )
        return ToolCallResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            content=error_content,
            success=False,
            error=error_message,
            error_code=error_code,
            session_id=request.session_id,
        )
