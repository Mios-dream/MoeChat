"""
异步结果通知器模块

ResultNotifier 负责处理异步工具执行完成后的结果分发。

核心职责:
1. 接收异步工具的完成结果
2. 根据 ToolMeta.notify_on_complete 决定处理策略:
   - True (主动模式): 将结果注入 LLM 上下文，触发新一轮回复生成
   - False (被动模式): 将结果存入会话状态，等待 LLM 自然引用
3. 防抖聚合: 短时间内多个异步结果合并通知

与外部模块的关系:
    - 被 ServerAsyncExecutor / ClientAsyncExecutor / HybridAsyncExecutor 回调
    - 通过回调函数通知 Chat Orchestrator 注入上下文
"""

from __future__ import annotations

import asyncio
from typing import Any

from tool_system.core.types import (
    ToolCallResult,
)
from tool_system.core.registry import get_registry


class ResultNotifier:
    """
    异步结果通知器

    处理异步工具执行完成后的结果分发。

    Attributes:
        _inject_callback: 上下文注入回调（由 Chat Orchestrator 注册）
        _store_callback: 会话存储回调（由 Chat Orchestrator 注册）
        _pending_notifications: 待聚合的通知队列
        _debounce_window_ms: 防抖窗口（毫秒）
        _debounce_task: 当前运行的防抖定时任务
    """

    def __init__(self) -> None:
        """初始化异步结果通知器"""
        self._inject_callback: Any = None
        """
        上下文注入回调: async def (session_id, messages) -> None
        将结果注入 LLM 上下文并触发新一轮回复。
        """

        self._store_callback: Any = None
        """
        会话存储回调: async def (session_id, result) -> None
        将结果存储到会话，等待 LLM 后续引用。
        """

        self._registry = get_registry()

        # ── 防抖聚合 ──
        self._pending_notifications: dict[str, list[ToolCallResult]] = {}
        """
        按 session_id 分组聚合的待通知结果。
        key: session_id, value: 待通知的结果列表
        """

        self._debounce_window_ms: float = 500.0
        """防抖窗口（毫秒），窗口内的多个结果会被合并为一次通知"""

        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}
        """当前运行的防抖定时任务"""

    def set_inject_callback(self, callback: Any) -> None:
        """
        设置上下文注入回调

        此回调由 Chat Orchestrator 注册，用于将异步工具结果
        注入到 LLM 上下文中并触发新回复。

        Args:
            callback: async def (session_id: str, messages: list[dict]) -> None
        """
        self._inject_callback = callback

    def set_store_callback(self, callback: Any) -> None:
        """
        设置会话存储回调

        此回调由 Chat Orchestrator 注册，用于将结果持久化到
        会话状态中供后续引用。

        Args:
            callback: async def (session_id: str, result: ToolCallResult) -> None
        """
        self._store_callback = callback

    async def notify(self, result: ToolCallResult) -> None:
        """
        接收异步工具的完成结果并分发

        根据工具的 notify_on_complete 属性决定处理策略:
        - True (主动): 通过 inject_callback 触发 LLM 回复
        - False (被动): 通过 store_callback 存入会话

        Args:
            result: 异步工具的完成结果（is_async_result=True）
        """
        session_id = result.session_id
        if not session_id:
            # 没有会话 ID，无法分发
            return

        meta = self._registry.get(result.tool_name)
        notify_on_complete = meta.notify_on_complete if meta else True

        if notify_on_complete:
            # 主动模式: 聚合后注入 LLM 上下文
            await self._schedule_inject(session_id, result)
        else:
            # 被动模式: 存储到会话
            await self._store_result(session_id, result)

    async def _schedule_inject(self, session_id: str, result: ToolCallResult) -> None:
        """
        将结果加入防抖通知队列

        多个短时间内完成的异步工具会被聚合为一次注入，
        避免频繁触发 LLM 生成。

        Args:
            session_id: 会话 ID
            result: 工具调用结果
        """
        # 加入待通知队列
        if session_id not in self._pending_notifications:
            self._pending_notifications[session_id] = []
        self._pending_notifications[session_id].append(result)

        # 取消现有的防抖定时器（重新计时）
        existing_task = self._debounce_tasks.get(session_id)
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()

        # 创建新的防抖定时器
        self._debounce_tasks[session_id] = asyncio.create_task(
            self._debounce_inject(session_id),
            name=f"notify_debounce_{session_id}",
        )

    async def _debounce_inject(self, session_id: str) -> None:
        """
        防抖延迟后执行注入

        等待 debounce_window_ms 后，将该会话所有待通知的结果
        合并为一次注入。

        Args:
            session_id: 会话 ID
        """
        await asyncio.sleep(self._debounce_window_ms / 1000.0)

        # 取出所有待通知的结果
        results = self._pending_notifications.pop(session_id, [])
        if not results:
            return

        # 构建注入消息
        messages = self._build_batch_notification(results)

        # 执行注入
        if self._inject_callback is not None:
            try:
                await self._inject_callback(session_id, messages)
            except Exception:
                # 注入失败不影响主流程
                pass

    def _build_batch_notification(
        self, results: list[ToolCallResult]
    ) -> list[dict[str, str]]:
        """
        将多个异步结果构建为一条聚合系统消息

        Args:
            results: 异步工具结果列表

        Returns:
            包含一条 system 消息的列表
        """
        if len(results) == 1:
            result = results[0]
            content = (
                f"[后台任务完成] {result.tool_name} 已执行完成。"
                f"结果: {result.content}"
            )
        else:
            tool_names = ", ".join(r.tool_name for r in results)
            summaries = []
            for r in results:
                summaries.append(f"- {r.tool_name}: {r.content[:200]}")
            details = "\n".join(summaries)
            content = (
                f"[后台任务完成] 以下 {len(results)} 个任务已完成 ({tool_names})。\n"
                f"{details}"
            )

        return [{"role": "system", "content": content}]

    async def _store_result(self, session_id: str, result: ToolCallResult) -> None:
        """
        将异步工具结果存储到会话中

        被动模式: 结果不会被立即注入，而是在 LLM 后续对话中
        通过状态查询自然引用。

        Args:
            session_id: 会话 ID
            result: 工具调用结果
        """
        if self._store_callback is not None:
            try:
                await self._store_callback(session_id, result)
            except Exception:
                pass

    async def flush(self) -> None:
        """
        立即推送所有待通知的结果

        通常在会话结束或用户主动刷新时调用。
        """
        # 取消所有防抖定时器
        for task in self._debounce_tasks.values():
            if not task.done():
                task.cancel()
        self._debounce_tasks.clear()

        # 立即推送所有待通知的结果
        for session_id, results in list(self._pending_notifications.items()):
            if not results:
                continue

            messages = self._build_batch_notification(results)
            if self._inject_callback is not None:
                try:
                    await self._inject_callback(session_id, messages)
                except Exception:
                    pass

        self._pending_notifications.clear()

    @property
    def pending_count(self) -> int:
        """待通知的结果总数（所有会话）"""
        total = 0
        for results in self._pending_notifications.values():
            total += len(results)
        return total

    def __repr__(self) -> str:
        """人类可读的状态"""
        return f"<ResultNotifier: {self.pending_count} pending notifications>"
