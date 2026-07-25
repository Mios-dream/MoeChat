"""
历史消息管理器

统一管理聊天历史记录的生命周期：
1. 内存管理：最多 200 条，超限自动压缩
2. Token 缓存：基于 estimate_tokens 带失效标记
3. LLM 查询：按 token 预算 / 消息条数主动压缩返回
"""

import json
from openai.types.chat import ChatCompletionMessageParam
from my_utils.token_counter import estimate_tokens


class HistoryManager:
    """历史消息管理器"""

    # 默认 LLM 上下文窗口（deepseek 系列）
    DEFAULT_CONTEXT_WINDOW: int = 128000
    # 消息硬限制
    HARD_MAX_MESSAGES: int = 200
    # 压缩时保留的最近消息数
    RECENT_KEEP: int = 10

    def __init__(
        self,
        max_messages: int = HARD_MAX_MESSAGES,
        recent_keep: int = RECENT_KEEP,
    ):
        self._messages: list[ChatCompletionMessageParam] = []
        self._max_messages: int = max_messages
        self._recent_keep: int = recent_keep
        self._token_total: int = 0
        self._token_valid: bool = False

    # ============================================================
    # 列表兼容接口（向后兼容，确保所有已有代码正常工作）
    # ============================================================

    def append(self, message: ChatCompletionMessageParam) -> None:
        """追加一条消息，超限时自动压缩"""
        self._messages.append(message)
        self._invalidate_cache()
        if len(self._messages) > self._max_messages:
            self._compress_oldest()

    def extend(self, messages: list[ChatCompletionMessageParam]) -> None:
        """追加多条消息，超限时自动压缩"""
        self._messages.extend(messages)
        self._invalidate_cache()
        if len(self._messages) > self._max_messages:
            self._compress_oldest()

    def pop(self, index: int = -1) -> ChatCompletionMessageParam:
        """弹出指定位置的消息"""
        result = self._messages.pop(index)
        self._invalidate_cache()
        return result

    def clear(self) -> None:
        """清空所有消息"""
        self._messages.clear()
        self._token_total = 0
        self._token_valid = True

    def copy(self) -> list[ChatCompletionMessageParam]:
        """返回消息列表的副本"""
        return self._messages.copy()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __getitem__(self, index):
        return self._messages[index]

    # ============================================================
    # Token 缓存
    # ============================================================

    @property
    def token_count(self) -> int:
        """当前消息列表的 token 总数（带缓存）"""
        if not self._token_valid:
            self._token_total = self._estimate_tokens(self._messages)
            self._token_valid = True
        return self._token_total

    def _estimate_tokens(self, messages: list[ChatCompletionMessageParam]) -> int:
        """估算消息列表的 token 总数"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += estimate_tokens(part["text"])
            total += 4  # role 标记开销
        return total

    def _invalidate_cache(self) -> None:
        """标记 token 缓存失效"""
        self._token_valid = False

    @staticmethod
    def estimate_list_tokens(messages: list[ChatCompletionMessageParam]) -> int:
        """
        估算一组消息的 token 总数（含 role 标记开销）

        供外部在计算固定部分 token 时使用。
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += estimate_tokens(part["text"])
            total += 4
        return total

    # ============================================================
    # LLM 查询
    # ============================================================

    def get_for_llm(
        self,
        reserved_tokens: int = 12000,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_count: int = 50,
    ) -> list[ChatCompletionMessageParam]:
        """
        获取适合传入 LLM 的历史记录

        返回压缩后的聊天历史本身（不含 memory_prompt / dynamic_context）。
        推荐通过 MessageChain 组装完整消息链。

        Parameters:
            reserved_tokens: 固定部分（system + user + 安全缓冲）的 token 开销
            context_window:  LLM 上下文窗口大小
            max_count:        传递给 LLM 的最大消息条数
        """
        return self._get_compressed(reserved_tokens, context_window, max_count)

    def _get_compressed(
        self,
        reserved_tokens: int,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_count: int | None = 50,
    ) -> list[ChatCompletionMessageParam]:
        """内部压缩实现"""
        if not self._messages:
            return []

        history = list(self._messages)
        token_budget = context_window - reserved_tokens

        if max_count and len(history) > max_count:
            history = self._compress_over_count(history, max_count)

        if self._estimate_tokens(history) > token_budget:
            history = self._compress_over_budget(history, token_budget)

        return history

    # ============================================================
    # 压缩
    # ============================================================

    def _compress_over_count(
        self,
        messages: list[ChatCompletionMessageParam],
        max_count: int,
    ) -> list[ChatCompletionMessageParam]:
        """
        按消息条数压缩

        保留最近 self._recent_keep 条消息，将更早的消息
        压缩为单条 system 摘要。
        """
        if len(messages) <= max_count:
            return messages

        keep_count = min(self._recent_keep, max_count - 1)
        compress_msgs = messages[:-keep_count]
        recent_msgs = messages[-keep_count:]

        result = list(recent_msgs)
        if compress_msgs:
            result.insert(0, self._to_summary(compress_msgs))
        return result

    def _compress_over_budget(
        self,
        messages: list[ChatCompletionMessageParam],
        budget: int,
    ) -> list[ChatCompletionMessageParam]:
        """
        按 token 预算渐进压缩

        从保留最多消息开始尝试，逐步减少保留条数，
        直到满足预算。
        """
        max_keep = min(len(messages) - 1, self._recent_keep)
        for keep_count in range(max_keep, 0, -1):
            compress_msgs = messages[:-keep_count]
            recent_msgs = messages[-keep_count:]

            result = list(recent_msgs)
            if compress_msgs:
                result.insert(0, self._to_summary(compress_msgs))

            if self._estimate_tokens(result) <= budget:
                return result

        # 极限：仅保留最近一条消息
        return messages[-1:]

    def _compress_oldest(self) -> None:
        """
        原地压缩最旧消息（达到硬限制时触发）

        保留最近 RECENT_KEEP 条消息完整，将更早消息
        压缩为单条 system 摘要，并清理孤立的 tool 消息。
        """
        keep_count = self._recent_keep
        compress_msgs = self._messages[:-keep_count]
        recent_msgs = self._messages[-keep_count:]

        self._messages = [self._to_summary(compress_msgs)] + recent_msgs

        # 清理孤立 tool 消息（前驱 tool_calls 被压缩后失效）
        while self._messages and self._messages[0].get("role") == "tool":
            self._messages.pop(0)

        self._invalidate_cache()

    # ============================================================
    # 格式化
    # ============================================================

    @staticmethod
    def _to_summary(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletionMessageParam:
        """
        将多条消息压缩为单条 system 摘要

        保留用户和助手消息的纯文本内容，移除 JSON 格式和
        工具调用细节，大幅减少 token 占用。
        """
        parts = ["【以下为压缩的历史对话】"]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                parts.append(f"用户: {content}")
            elif role == "assistant":
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    text = parsed.get("text", "")
                except (json.JSONDecodeError, TypeError):
                    text = content
                parts.append(f"助手: {text}")
            elif role == "system":
                parts.append(f"【设定】{str(content)[:200]}")
        return {"role": "system", "content": "\n".join(parts)}
