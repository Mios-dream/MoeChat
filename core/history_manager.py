"""
历史消息管理器

统一管理聊天历史记录的生命周期，是对话上下文中历史部分的核心管理组件。

核心设计：
1. 内存管理：最多 200 条消息，超限时自动将最旧消息压缩为单条 system 摘要
2. Token 缓存：基于 estimate_tokens 的估算值，带失效标记，避免重复计算
3. 写入时 LLM 压缩：达到阈值时由上层调用 compress_if_needed() 进行 LLM 语义摘要

压缩策略：
- append/extend 路径（同步，200 条硬上限）：_compress_oldest → _to_summary（纯文本拼接，内存安全垫）
- compress_if_needed 路径（异步，50 条上限 + token 预算检查）：_to_summary_llm（LLM 语义摘要，失败时抛出异常）
- get_for_llm 路径（同步，纯读取）：不做 LLM 调用，仅按最大条数截断后返回
"""

import json
from openai.types.chat import ChatCompletionMessageParam
from core.llm.llm_client import LLMClient
from my_utils.token_counter import estimate_tokens
from my_utils.log import logger as Log


class HistoryManager:
    """
    历史消息管理器

    职责：
    - 存储聊天消息列表，提供 list 兼容接口
    - 超限时自动压缩，保持内存使用可控
    - 为 LLM 查询提供压缩后的历史记录

    使用方式：
        history = HistoryManager()
        history.append({"role": "user", "content": "你好"})
        # 自动管理 200 条上限
        # 通过 MessageChain.build() 自动调用 get_for_llm 压缩后传入 LLM
    """

    # 默认 LLM 上下文窗口大小（deepseek 系列模型的上下文长度）
    DEFAULT_CONTEXT_WINDOW: int = 64000

    # 消息硬限制：超过此数量触发 _compress_oldest（纯文本拼接，防 OOM）
    HARD_MAX_MESSAGES: int = 200

    # 写入时 LLM 压缩的触发条数阈值，超过此值 compress_if_needed 会用 LLM 语义摘要
    SOFT_MAX_MESSAGES: int = 50

    # 压缩时保留的最近消息条数，更早的消息被压缩为摘要
    RECENT_KEEP: int = 10

    # LLM 语义摘要的系统提示词，传递给 LLM 指导其生成高质量对话摘要
    SUMMARY_SYSTEM_PROMPT: str = (
        "你是一个对话摘要助手。请将以下对话压缩为一段简洁的文字摘要。\n"
        "要求：\n"
        "1. 保持客观，不添加对话中没有的信息\n"
        "2. 保留重要的情感变化和关键事件\n"
        "3. 用「用户」和「助手」区分角色\n"
        "4. 控制在200字以内\n"
        "5. 直接输出摘要文本，不要任何前缀或解释"
    )

    def __init__(
        self,
        max_messages: int = HARD_MAX_MESSAGES,
        recent_keep: int = RECENT_KEEP,
    ):
        """
        初始化历史消息管理器

        Parameters:
            max_messages: 消息硬限制，超限时触发 _compress_oldest 压缩最旧消息
            recent_keep: 压缩时保留的最近消息条数
        """
        self._messages: list[ChatCompletionMessageParam] = []
        self._max_messages: int = max_messages
        self._recent_keep: int = recent_keep
        self._token_total: int = 0
        # LLM 客户端实例，用于生成语义摘要（由 HistoryManager 内部管理，生命周期与实例一致）
        self._llm_client = LLMClient(model_key="LLM")

    # ============================================================
    # 列表兼容接口
    # 提供与 Python list 一致的接口，确保已有的 history.append() / extend() 等代码无缝迁移
    # ============================================================

    def append(self, message: ChatCompletionMessageParam) -> None:
        """
        追加一条消息到历史列表末尾

        当消息总数超过 max_messages 时，自动将最旧消息压缩为单条 system 摘要。
        此路径使用纯文本拼接（_to_summary），不调用 LLM，保证写入性能。

        Parameters:
            message: OpenAI 格式的聊天消息，格式为 {"role": str, "content": str | list}
        """
        self._messages.append(message)
        self._token_total += self._compute_message_tokens(message)
        if len(self._messages) > self._max_messages:
            self._compress_oldest()

    def extend(self, messages: list[ChatCompletionMessageParam]) -> None:
        """
        追加多条消息到历史列表末尾

        超限时触发压缩，行为同 append。

        Parameters:
            messages: OpenAI 格式的聊天消息列表
        """
        self._messages.extend(messages)
        self._token_total += sum(self._compute_message_tokens(m) for m in messages)
        if len(self._messages) > self._max_messages:
            self._compress_oldest()

    def pop(self, index: int = -1) -> ChatCompletionMessageParam:
        """弹出指定位置的消息，默认弹出末尾"""
        result = self._messages.pop(index)
        self._token_total -= self._compute_message_tokens(result)
        return result

    def clear(self) -> None:
        """清空所有消息和 token 缓存"""
        self._messages.clear()
        self._token_total = 0

    def copy(self) -> list[ChatCompletionMessageParam]:
        """返回消息列表的浅拷贝副本"""
        return self._messages.copy()

    def __len__(self) -> int:
        """返回消息总数"""
        return len(self._messages)

    def __iter__(self):
        """迭代所有消息"""
        return iter(self._messages)

    def __getitem__(self, index):
        """按下标访问消息，支持切片和负数索引"""
        return self._messages[index]

    def get_raw_recent(self, count: int = 10) -> list[ChatCompletionMessageParam]:
        """
        获取最近 count 条原始消息（跳过压缩生成的 system 摘要）

        用于情感分析等需要干净对话历史的场景。_compress_oldest 生成的摘要
        以 system 角色和特定前缀标记，此方法在反向遍历时跳过这些消息，
        确保返回的消息都是原始的 user / assistant 对话。

        Parameters:
            count: 需要返回的原始消息条数，默认 10

        Returns:
            原始消息列表，按时间正序排列；如果原始消息不足，返回全部可用条数
        """
        raw = []
        for msg in reversed(self._messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and "【以下为压缩的历史对话】" in content:
                    continue
            raw.append(msg)
            if len(raw) >= count:
                break
        return list(reversed(raw))

    # ============================================================
    # Token 计数（增量维护）
    # token_count 为 O(1) 属性，_token_total 在 append/extend/pop/compress
    # 时增量更新，无需遍历全量消息，避免高频场景下的重复估算开销。
    # ============================================================

    @property
    def token_count(self) -> int:
        """
        当前消息列表的估算 token 总数（增量维护，O(1) 访问）

        非精确值，基于 cl100k_base tokenizer 估算。用于预算分配参考，
        不用于计费或精确截断。
        """
        return self._token_total

    @staticmethod
    def _compute_message_tokens(message: ChatCompletionMessageParam) -> int:
        """
        计算单条消息的估算 token 数（含 4 token 的 role 标记开销）

        Parameters:
            message: OpenAI 格式的聊天消息

        Returns:
            估算的 token 数
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return estimate_tokens(content) + 4
        elif isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"])
            return total + 4
        return 4

    def _estimate_tokens(self, messages: list[ChatCompletionMessageParam]) -> int:
        """
        估算一组消息的 token 总数（遍历计算，用于非自身消息列表的预算估算）

        Parameters:
            messages: 要估算的消息列表

        Returns:
            估算的 token 总数
        """
        return sum(self._compute_message_tokens(msg) for msg in messages)

    @staticmethod
    def estimate_list_tokens(messages: list[ChatCompletionMessageParam]) -> int:
        """
        静态方法：估算一组消息的 token 总数（含 role 标记开销）

        供 MessageChain 等在构建消息链时计算固定部分（system + 动态上下文 + 用户消息）
        的 token 开销，以便为历史记录分配剩余 token 预算。

        Parameters:
            messages: 要估算的消息列表

        Returns:
            估算的 token 总数
        """
        return sum(HistoryManager._compute_message_tokens(msg) for msg in messages)

    # ============================================================
    # LLM 查询（同步，纯读取）
    #
    # 仅做条数截断，不做 LLM 调用。LLM 语义摘要已在写入时由
    # compress_if_needed() 完成。
    # ============================================================

    def get_for_llm(
        self,
        reserved_tokens: int = 12000,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_count: int = 50,
    ) -> list[ChatCompletionMessageParam]:
        """
        获取适合传入 LLM 的历史记录

        纯读取操作，不做 LLM 调用。当消息条数超过 max_count 时仅截断最旧消息。
        LLM 语义摘要压缩在写入时由 compress_if_needed() 完成。

        Parameters:
            reserved_tokens: 固定部分（system + 用户消息 + 动态上下文 + 安全缓冲）的 token 开销
            context_window:  LLM 上下文窗口大小，默认 64000（deepseek 系列）
            max_count:        传递给 LLM 的最大消息条数，超过此值截断最旧消息

        Returns:
            截断后的消息列表，可直接传入 LLM 的 messages 参数
        """
        if not self._messages:
            return []

        history = list(self._messages)

        # 超过最大条数时，截断中间部分，保留首尾
        if max_count and len(history) > max_count:
            keep_count = min(self._recent_keep, max_count // 2)
            # 保留首条（可能是 system 摘要）+ 最近 keep_count 条
            head = history[:1]
            tail = history[-keep_count:]
            history = head + tail

        return history

    # ============================================================
    # 写入时 LLM 压缩（异步）
    #
    # 由上层（add_msg / add_interaction_msg）在完成一轮对话写入后调用。
    # 使用 LLM 语义摘要将超过阈值的旧消息压缩为一条 system 摘要，
    # 保持历史记录在可控范围内。
    # ============================================================

    async def compress_if_needed(
        self,
        max_count: int = SOFT_MAX_MESSAGES,
    ) -> None:
        """
        写入时 LLM 语义压缩

        当消息条数超过 max_count 时，将最旧消息用 LLM 压缩为语义摘要，
        保留最近 RECENT_KEEP 条完整。此方法由上层在完成每轮对话写入后调用。

        Parameters:
            max_count: 触发压缩的消息条数阈值，默认 SOFT_MAX_MESSAGES（50）

        Raises:
            ValueError: 待压缩的消息中没有有效对话内容
            RuntimeError: LLM 返回空摘要
            以及 LLMClient.request 的各类网络/API 异常
        """
        if len(self._messages) <= max_count:
            return

        keep_count = min(self._recent_keep, max_count - 1)
        compress_msgs = self._messages[:-keep_count]
        recent_msgs = self._messages[-keep_count:]

        Log.info(
            f"[HistoryManager] LLM 语义压缩 {len(compress_msgs)} 条历史为摘要, "
            f"保留 {len(recent_msgs)} 条"
        )

        # 从增量总数中减去被压缩消息的 token
        for msg in compress_msgs:
            self._token_total -= self._compute_message_tokens(msg)

        # 使用 LLM 生成语义摘要
        summary = await self._to_summary_llm(compress_msgs)
        self._messages = [summary] + recent_msgs

        # 加上摘要消息的 token
        self._token_total += self._compute_message_tokens(summary)

        # 清理孤立 tool 消息：压缩后 tool_calls 的前驱消息已丢失，
        # 开头的 tool 消息无法被 LLM 理解，直接移除
        while self._messages and self._messages[0].get("role") == "tool":
            removed = self._messages.pop(0)
            self._token_total -= self._compute_message_tokens(removed)

    # ============================================================
    # 压缩
    # _compress_oldest：纯文本拼接，用于 append 路径的内存水位控制（同步，200 条硬上限）
    # compress_if_needed：LLM 语义摘要，用于写入路径的智能压缩（异步）
    # ============================================================

    def _compress_oldest(self) -> None:
        """
        原地压缩最旧消息（达到 HARD_MAX_MESSAGES 时触发）

        使用纯文本拼接（_to_summary），不调用 LLM，保证写入路径的同步性能。
        保留最近 RECENT_KEEP 条消息完整，将更早消息压缩为单条 system 摘要。

        清理开头可能存在的孤立 tool 消息（其前驱 tool_calls 已被压缩）。
        """
        keep_count = self._recent_keep
        compress_msgs = self._messages[:-keep_count]
        recent_msgs = self._messages[-keep_count:]

        Log.info(
            f"[HistoryManager] 压缩 {len(compress_msgs)} 条历史为摘要, "
            f"保留 {len(recent_msgs)} 条"
        )

        # 从增量总数中减去被压缩消息的 token
        for msg in compress_msgs:
            self._token_total -= self._compute_message_tokens(msg)

        summary = self._to_summary(compress_msgs)
        self._messages = [summary] + recent_msgs

        # 加上摘要消息的 token
        self._token_total += self._compute_message_tokens(summary)

        # 清理孤立 tool 消息：压缩后 tool_calls 的前驱消息已丢失，
        # 开头的 tool 消息无法被 LLM 理解，直接移除
        while self._messages and self._messages[0].get("role") == "tool":
            removed = self._messages.pop(0)
            self._token_total -= self._compute_message_tokens(removed)

    # ============================================================
    # LLM 摘要
    #
    # _to_summary_llm：异步，使用 LLM 生成语义摘要
    # 被 compress_if_needed 调用。
    # ============================================================

    async def _to_summary_llm(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletionMessageParam:
        """
        使用 LLM 将多条消息压缩为单条语义摘要

        将需要压缩的消息列表格式化为"用户:" / "助手:" 对话文本，调用 LLM
        生成摘要，返回 system 角色的摘要消息。如果 LLM 调用失败或返回空值，
        直接抛出异常，不回退到规则拼接——防止语义信息丢失静默发生。

        Parameters:
            messages: 需要压缩的原始消息列表

        Returns:
            system 角色的摘要消息，格式为 {"role": "system", "content": "【以下为压缩的历史对话】..."}

        Raises:
            ValueError: 待压缩的消息中没有有效对话内容
            RuntimeError: LLM 返回空摘要
            以及 LLMClient.request 的各类网络/API 异常
        """
        # 将消息列表格式化为可读的对话文本
        conversation_lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                conversation_lines.append(f"用户: {content}")
            elif role == "assistant":
                # assistant 消息可能是纯文本或 JSON（v3 多任务格式）
                try:
                    parsed = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                    if isinstance(parsed, dict):
                        text = parsed.get("text") or parsed.get("content") or content
                    else:
                        text = str(parsed)
                except (json.JSONDecodeError, TypeError):
                    text = content
                conversation_lines.append(f"助手: {text}")
            elif role == "system":
                conversation_lines.append(f"[系统]: {str(content)[:200]}")

        conversation_text = "\n".join(conversation_lines)
        if not conversation_text.strip():
            raise ValueError("无有效对话内容可供摘要")

        # 调用 LLM 生成摘要
        summary_text = await self._llm_client.request(
            [
                {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text},
            ]
        )
        if not summary_text:
            raise RuntimeError("LLM 摘要返回为空")

        return {
            "role": "system",
            "content": f"【以下为压缩的历史对话】{summary_text}",
        }

    @staticmethod
    def _to_summary(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletionMessageParam:
        """
        纯文本拼接压缩（不使用 LLM）

        将多条消息的文本内容用"用户:" / "助手:" 前缀拼接为一段纯文本。
        用于 append/extend 路径的快速压缩，仅控制内存水位，不直接喂给 LLM。

        Parameters:
            messages: 需要压缩的原始消息列表

        Returns:
            system 角色的摘要消息
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
                    parsed = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                    if isinstance(parsed, dict):
                        text = parsed.get("text") or parsed.get("content") or content
                    else:
                        text = str(parsed)
                except (json.JSONDecodeError, TypeError):
                    text = content
                parts.append(f"助手: {text}")
            elif role == "system":
                parts.append(f"【设定】{str(content)[:200]}")
        return {"role": "system", "content": "\n".join(parts)}
