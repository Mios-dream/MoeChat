"""
服务端工具执行器模块

实现服务端工具的同步和异步执行策略：
- ServerSyncExecutor: 直接 await 工具执行，阻塞等待结果
- ServerAsyncExecutor: 创建后台 Task 执行，立即返回占位结果

服务端工具的执行流程（SYNC）:
    1. 从 Registry 获取工具类
    2. 实例化工具对象
    3. 校验参数并填充默认值
    4. 带超时控制执行 tool.execute(**validated_args)
    5. 包装结果返回

服务端工具的执行流程（ASYNC）:
    1. 从 Registry 获取工具类
    2. 返回占位回复（不执行）
    3. 创建 asyncio.Task 在后台执行
    4. Task 完成后通过 ResultNotifier 通知
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tool_system.core.enums import ExecutionMode
from tool_system.core.errors import (
    ToolNotFoundError,
    ToolExecutionError,
    InvalidArgumentsError,
)
from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
    ToolCallProgress,
    ToolMeta,
)
from tool_system.core.registry import get_registry
from tool_system.executors.base_executor import BaseExecutor


class ServerSyncExecutor(BaseExecutor):
    """
    服务端同步工具执行器

    执行策略:
    1. 直接 await 工具执行
    2. 阻塞等待结果返回
    3. 异常捕获并包装为 ToolCallResult

    适用场景:
    - OCR 识别（结果直接影响回复内容）
    - 知识库检索（搜索结果决定 LLM 回答）
    - AI 模型调用（结果需要立即反馈给用户）
    """

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        同步执行服务端工具

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 执行结果
        """
        tool_name = request.tool_name
        start_time = time.perf_counter()

        # 获取工具类
        tool_class = self._registry.get_class(tool_name)
        if tool_class is None:
            return self._wrap_error(
                request,
                f"工具 '{tool_name}' 未注册",
                error_code="TOOL_NOT_FOUND",
            )

        # 获取元信息
        meta = self._registry.get(tool_name)
        if meta is None:
            return self._wrap_error(
                request,
                f"工具 '{tool_name}' 元信息缺失",
                error_code="TOOL_NOT_FOUND",
            )

        # 实例化工具
        try:
            tool_instance = tool_class()
        except Exception as e:
            return self._wrap_error(
                request,
                f"工具 '{tool_name}' 实例化失败: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        # 校验参数
        try:
            validated_args = tool_instance.validate_arguments(request.arguments)
        except InvalidArgumentsError as e:
            return self._wrap_error(
                request,
                str(e),
                error_code="INVALID_ARGUMENTS",
            )

        # 执行工具
        try:
            async with asyncio.timeout(request.timeout):
                result_content = await tool_instance.execute(**validated_args)
        except asyncio.TimeoutError:
            return self._wrap_error(
                request,
                f"工具 '{tool_name}' 执行超时（{request.timeout}s）",
                error_code="TOOL_TIMEOUT",
            )
        except Exception as e:
            return self._wrap_error(
                request,
                f"工具 '{tool_name}' 执行异常: {e}",
                error_code="TOOL_EXEC_ERROR",
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        result = ToolCallResult(
            call_id=request.call_id,
            tool_name=tool_name,
            content=result_content,
            success=True,
            duration_ms=round(duration_ms, 2),
            session_id=request.session_id,
        )
        return result


class ServerAsyncExecutor(BaseExecutor):
    """
    服务端异步工具执行器

    执行策略:
    1. 立即返回占位回复（不阻塞 LLM）
    2. 创建后台 asyncio.Task 执行工具
    3. Task 完成时通过回调通知 ResultNotifier

    回调机制:
    - 工具执行完成后调用 _on_complete 回调
    - 回调将结果传递给 ResultNotifier
    - ResultNotifier 决定主动推送还是被动存储

    Attributes:
        _on_complete: 异步完成回调（通常为 ResultNotifier.notify）
        _default_placeholder: 默认占位文本模板
    """

    def __init__(self) -> None:
        """初始化异步执行器"""
        super().__init__()
        self._on_complete: Any = None
        """异步完成回调: async def (result: ToolCallResult) -> None"""

    def set_complete_callback(self, callback: Any) -> None:
        """
        设置异步完成回调

        Args:
            callback: async def (result: ToolCallResult) -> None
        """
        self._on_complete = callback

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """
        异步执行服务端工具（不阻塞）

        1. 创建后台 Task
        2. 立即返回占位结果
        3. Task 完成后回调 _on_complete

        Args:
            request: 工具调用请求

        Returns:
            ToolCallResult: 占位回复结果（LLM 用它生成过渡文本）
        """
        tool_name = request.tool_name
        meta = self._registry.get(tool_name)

        # 生成占位结果
        placeholder_text = (
            meta.placeholder
            if meta and meta.placeholder
            else f"任务 '{tool_name}' 正在后台执行中。"
        )

        # 创建后台执行任务
        asyncio.create_task(
            self._run_in_background(request),
            name=f"async_tool_{request.call_id}",
        )

        # 立即返回占位
        return ToolCallResult(
            call_id=request.call_id,
            tool_name=tool_name,
            content=placeholder_text,
            success=True,
            is_async_result=True,
            session_id=request.session_id,
        )

    async def _run_in_background(self, request: ToolCallRequest) -> None:
        """
        后台执行工具任务

        执行完成后通过 _on_complete 回调通知结果。
        内部使用 ServerSyncExecutor 执行实际的工具逻辑。

        Args:
            request: 工具调用请求
        """
        # 使用同步执行器实际执行工具
        sync_executor = ServerSyncExecutor()
        result = await sync_executor.execute(request)

        # 标记为异步结果并保留会话信息
        result.is_async_result = True
        result.session_id = request.session_id

        # 通知回调
        if self._on_complete is not None:
            try:
                await self._on_complete(result)
            except Exception:
                # 回调失败不影响主流程
                pass
