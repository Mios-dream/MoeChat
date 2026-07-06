"""
结果聚合器模块

ResultAggregator 负责将多个 ToolCallResult 转换为 OpenAI 协议的消息格式，
以便注入 LLM 上下文。

主要职责:
1. 将 ToolCallResult 列表转换为 OpenAI tool 角色消息
2. 处理同步/异步结果的差异
3. 构建标准化的错误消息
4. 支持结果过滤和排序
"""

import json
from tool_system.core.types import (
    ToolCallRequest,
    ToolCallResult,
)
from openai.types.chat import (
    ChatCompletionToolMessageParam,
)


class ResultAggregator:
    """
    结果聚合器

    将工具执行结果转换为 OpenAI ChatCompletion 协议中 tool 角色的消息。

    核心方法:
        aggregate(): 批量聚合结果 → list[ChatCompletionToolMessageParam]
        build_error_message(): 构建标准化的错误工具消息
        build_async_notification(): 构建异步完成的系统通知消息

    使用示例:
        aggregator = ResultAggregator()
        messages = aggregator.aggregate(results, requests)
        # messages 可直接追加到 LLM 上下文
    """

    @staticmethod
    def aggregate(
        results: list[ToolCallResult],
        requests: list[ToolCallRequest],
    ) -> list[ChatCompletionToolMessageParam]:
        """
        将工具调用结果聚合为 OpenAI 协议消息

        对于同步工具结果，生成 tool 角色消息。
        对于异步结果（is_async_result=True）和失败的调用同样处理。

        Args:
            results: 工具调用结果列表
            requests: 原始请求列表（用于获取 tool_call_id）

        Returns:
            OpenAI 协议的消息列表:
            [
                {"role": "tool", "tool_call_id": "call_xxx", "content": "..."},
                ...
            ]
        """
        messages: list[ChatCompletionToolMessageParam] = []

        # 构建 call_id → request 的映射（快速查找）
        request_by_call_id: dict[str, ToolCallRequest] = {}
        for req in requests:
            request_by_call_id[req.call_id] = req

        for result in results:
            if result.is_async_result:
                # 异步结果：生成系统通知而非 tool 消息
                # 实际注入逻辑由 ResultNotifier 处理
                continue

            msg: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.content,
            }
            messages.append(msg)

        return messages

    @staticmethod
    def aggregate_single(
        result: ToolCallResult,
        tool_call_id: str | None = None,
    ) -> ChatCompletionToolMessageParam:
        """
        聚合单个工具结果

        Args:
            result: 工具调用结果
            tool_call_id: OpenAI tool_call_id（可选，默认使用 result.call_id）

        Returns:
            OpenAI tool 角色消息
        """
        call_id = tool_call_id or result.call_id
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result.content,
        }

    @staticmethod
    def build_error_message(
        tool_call_id: str,
        error_message: str,
        error_code: str = "TOOL_EXEC_ERROR",
    ) -> ChatCompletionToolMessageParam:
        """
        构建标准化的错误工具消息

        当工具调用失败时，需要将错误信息以 JSON 格式通知 LLM，
        LLM 据此向用户解释错误原因。

        Args:
            tool_call_id: OpenAI tool_call_id
            error_message: 错误描述
            error_code: 结构化错误码

        Returns:
            标准化的 tool 角色错误消息
        """
        content = json.dumps(
            {
                "success": False,
                "error": error_message,
                "error_code": error_code,
            },
            ensure_ascii=False,
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @staticmethod
    def build_async_notification(
        result: ToolCallResult,
    ) -> dict[str, str]:
        """
        构建异步完成通知消息

        用于 ResultNotifier 将异步任务完成结果注入 LLM 上下文。

        Args:
            result: 异步工具的完成结果

        Returns:
            系统通知消息（system 角色）
        """
        return {
            "role": "system",
            "content": (
                f"[系统通知] 后台任务 '{result.tool_name}' 已完成。"
                f"结果: {result.content}"
            ),
        }

    @staticmethod
    def filter_success(results: list[ToolCallResult]) -> list[ToolCallResult]:
        """
        过滤出成功的工具调用结果

        Args:
            results: 工具调用结果列表

        Returns:
            仅包含 success=True 的结果
        """
        return [r for r in results if r.success]

    @staticmethod
    def filter_failed(results: list[ToolCallResult]) -> list[ToolCallResult]:
        """
        过滤出失败的工具调用结果

        Args:
            results: 工具调用结果列表

        Returns:
            仅包含 success=False 的结果
        """
        return [r for r in results if not r.success]

    @staticmethod
    def summary(results: list[ToolCallResult]) -> str:
        """
        生成工具调用摘要（用于日志）

        Args:
            results: 工具调用结果列表

        Returns:
            人类可读的摘要字符串
        """
        total = len(results)
        success_count = len(ResultAggregator.filter_success(results))
        failed_count = total - success_count
        parts = [f"工具调用: {total} 个, 成功 {success_count} 个"]

        if failed_count > 0:
            parts.append(f"失败 {failed_count} 个")
            for r in results:
                if not r.success:
                    parts.append(f"  - {r.tool_name}: {r.error}")

        return "\n".join(parts)
