"""
历史消息管理器（单源私有记录）

统一管理聊天历史记录的生命周期，是对话上下文中历史部分的核心管理组件。

核心设计：
1. 单源存储：``_records: list[ChatRecord]`` 为唯一事实来源，LLM 上下文（get_for_llm）、
   展示输出（copy）、情感引擎原始消息（get_raw_recent）一律调用记录自身的格式化策略
   （to_openai / to_display）多态投影派生，不再维护平行的纯 OpenAI 消息列表；
2. 内存管理：最多 max_messages 条消息，超限时自动将最旧消息压缩为单条 SummaryRecord；
3. Token 缓存：基于 estimate_tokens 的估算值增量维护（O(1) 访问），避免重复计算；
4. 写入时 LLM 压缩：达到阈值时由上层调用 compress_if_needed() 进行 LLM 语义摘要。

投影规则（由各消息类别自决）：
- summary / note / event 等类别 to_display() 返回 None → 展示 API 自动过滤；
- summary / event 等类别 to_openai() 注入 LLM → 上下文自动携带；
- 新增消息类型只需新增一个 ChatRecord 子类，本管理器零改动。
"""

from collections.abc import Iterable
import json

from openai.types.chat import ChatCompletionMessageParam

from core.history.records import (
    ChatRecord,
    SummaryRecord,
)
from core.llm.llm_client import LLMClient
from my_utils.token_counter import estimate_tokens
from my_utils.log import logger as Log


class HistoryManager:
    """
    历史消息管理器

    职责：
    - 存储聊天消息列表（ChatRecord 私有记录），提供 list 兼容接口
    - 超限时自动压缩，保持内存使用可控
    - 为 LLM 查询提供投影后的 OpenAI 消息链
    - 为展示 API 提供投影后的展示字典列表

    使用方式：
        history = HistoryManager()
        history.append(UserRecord(content="你好"))
        # 自动管理条数上限
        # 通过 MessageChain.build() 自动调用 get_for_llm 投影后传入 LLM
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
        self._records: list[ChatRecord] = []
        self._max_messages: int = max_messages
        self._recent_keep: int = recent_keep
        self._token_total: int = 0
        # LLM 客户端实例，用于生成语义摘要（由 HistoryManager 内部管理，生命周期与实例一致）
        self._llm_client = LLMClient(model_key="LLM")

    # ============================================================
    # 列表兼容接口
    # 提供与 Python list 一致的接口，确保已有的 history.append() / extend() 等代码无缝迁移
    # ============================================================

    def append(self, record: ChatRecord) -> None:
        """
        追加一条消息到历史列表末尾

        当消息总数超过 max_messages 时，自动将最旧消息压缩为单条 SummaryRecord。
        此路径使用纯文本拼接（_to_summary），不调用 LLM，保证写入性能。

        Parameters:
            record: 私有聊天记录（ChatRecord 子类实例）
        """
        self._records.append(record)
        self._token_total += record.estimate_tokens()
        if len(self._records) > self._max_messages:
            self._compress_oldest()

    def extend(self, records: Iterable[ChatRecord]) -> None:
        """
        追加多条消息到历史列表末尾

        超限时触发压缩，行为同 append。

        Parameters:
            records: 私有聊天记录的可迭代集合
        """
        for record in records:
            self.append(record)

    def pop(self, index: int = -1) -> ChatRecord:
        """
        弹出指定位置的消息，默认弹出末尾

        单列表弹出，索引与展示记录天然对齐（修复原双层列表错位缺陷）。

        Parameters:
            index: 弹出位置，支持负数索引

        Returns:
            被弹出的记录
        """
        result = self._records.pop(index)
        self._token_total -= result.estimate_tokens()
        return result

    def clear(self) -> None:
        """清空所有消息和 token 缓存"""
        self._records.clear()
        self._token_total = 0

    def copy(self) -> list[dict]:
        """
        返回投影后的展示字典列表（含 kind/timestamp）

        逐条调用 to_display() 派生，返回 None 的类别（summary/note/event/system）自动过滤。

        Returns:
            展示字典列表，供 /api/chat/history 等展示链路使用
        """
        return [
            display
            for record in self._records
            if (display := record.to_display()) is not None
        ]

    def __len__(self) -> int:
        """返回消息总数"""
        return len(self._records)

    def __iter__(self):
        """迭代所有消息记录"""
        return iter(self._records)

    def __getitem__(self, index):
        """按下标访问消息记录，支持切片和负数索引"""
        return self._records[index]

    def get_raw_recent(self, count: int = 10) -> list[ChatCompletionMessageParam]:
        """
        获取最近 count 条原始消息（跳过非原始对话类别）

        用于情感分析等需要干净对话历史的场景。跳过 is_conversation=False 的类别
        （summary/note/event），返回其余记录投影后的 OpenAI 字典。

        Parameters:
            count: 需要返回的原始消息条数，默认 10

        Returns:
            原始消息列表，按时间正序排列；如果原始消息不足，返回全部可用条数
        """
        raw: list[ChatCompletionMessageParam] = []
        for record in reversed(self._records):
            if not record.is_conversation:
                continue
            openai_msg = record.to_openai()
            if openai_msg is None:
                continue
            raw.append(openai_msg)
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
        计算单条 OpenAI 消息的估算 token 数（含 4 token 的 role 标记开销）

        供 estimate_list_tokens 对非自身记录列表做一次性估算。

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

    @staticmethod
    def estimate_list_tokens(messages: list[ChatCompletionMessageParam]) -> int:
        """
        静态方法：估算一组 OpenAI 消息的 token 总数（含 role 标记开销）

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
    # 仅做投影与条数截断，不做 LLM 调用。LLM 语义摘要已在写入时由
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

        纯读取操作，不做 LLM 调用。逐条调用 to_openai() 投影，返回 None 的类别
        （如 note）不进入消息链；超过 max_count 时仅截断最旧消息。
        LLM 语义摘要压缩在写入时由 compress_if_needed() 完成。

        Parameters:
            reserved_tokens: 固定部分（system + 用户消息 + 动态上下文 + 安全缓冲）的 token 开销
            context_window:  LLM 上下文窗口大小，默认 64000（deepseek 系列）
            max_count:        传递给 LLM 的最大消息条数，超过此值截断最旧消息

        Returns:
            截断后的 OpenAI 消息列表，可直接传入 LLM 的 messages 参数
        """
        # 逐条投影，None 类别（如 note）不进入 LLM 消息链
        projected = [
            openai_msg
            for record in self._records
            if (openai_msg := record.to_openai()) is not None
        ]

        if not projected:
            return []

        # 超过最大条数时，截断中间部分，保留首尾
        if max_count and len(projected) > max_count:
            keep_count = min(self._recent_keep, max_count // 2)
            # 保留首条（可能是 system 摘要）+ 最近 keep_count 条
            head = projected[:1]
            tail = projected[-keep_count:]
            projected = head + tail

        return projected

    # ============================================================
    # 写入时 LLM 压缩（异步）
    #
    # 由上层（add_msg）在完成一轮对话写入后调用。
    # 使用 LLM 语义摘要将超过阈值的旧消息压缩为一条 SummaryRecord，
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
        if len(self._records) <= max_count:
            return

        keep_count = min(self._recent_keep, max_count - 1)
        compress_records = self._records[:-keep_count]
        recent_records = self._records[-keep_count:]

        Log.info(
            f"[HistoryManager] LLM 语义压缩 {len(compress_records)} 条历史为摘要, "
            f"保留 {len(recent_records)} 条"
        )

        # 从增量总数中减去被压缩消息的 token
        for record in compress_records:
            self._token_total -= record.estimate_tokens()

        # 使用 LLM 生成语义摘要
        summary = await self._to_summary_llm(compress_records)
        self._records = [summary] + recent_records

        # 加上摘要消息的 token
        self._token_total += summary.estimate_tokens()

        # 清理孤立 tool 消息：压缩后 tool_calls 的前驱消息已丢失，
        # 开头的 tool 消息无法被 LLM 理解，直接移除
        while self._records and self._records[0].kind == "tool_result":
            removed = self._records.pop(0)
            self._token_total -= removed.estimate_tokens()

    # ============================================================
    # 压缩
    # _compress_oldest：纯文本拼接，用于 append 路径的内存水位控制（同步，200 条硬上限）
    # compress_if_needed：LLM 语义摘要，用于写入路径的智能压缩（异步）
    # ============================================================

    def _compress_oldest(self) -> None:
        """
        原地压缩最旧消息（达到 HARD_MAX_MESSAGES 时触发）

        使用纯文本拼接（_to_summary），不调用 LLM，保证写入路径的同步性能。
        保留最近 RECENT_KEEP 条记录完整，将更早记录压缩为单条 SummaryRecord。

        清理开头可能存在的孤立 tool 消息（其前驱 tool_calls 已被压缩）。
        """
        keep_count = self._recent_keep
        compress_records = self._records[:-keep_count]
        recent_records = self._records[-keep_count:]

        Log.info(
            f"[HistoryManager] 压缩 {len(compress_records)} 条历史为摘要, "
            f"保留 {len(recent_records)} 条"
        )

        # 从增量总数中减去被压缩消息的 token
        for record in compress_records:
            self._token_total -= record.estimate_tokens()

        summary = self._to_summary(compress_records)
        self._records = [summary] + recent_records

        # 加上摘要消息的 token
        self._token_total += summary.estimate_tokens()

        # 清理孤立 tool 消息：压缩后 tool_calls 的前驱消息已丢失，
        # 开头的 tool 消息无法被 LLM 理解，直接移除
        while self._records and self._records[0].kind == "tool_result":
            removed = self._records.pop(0)
            self._token_total -= removed.estimate_tokens()

    # ============================================================
    # LLM 摘要
    #
    # _to_summary_llm：异步，使用 LLM 生成语义摘要
    # 被 compress_if_needed 调用。
    # ============================================================

    async def _to_summary_llm(
        self,
        records: list[ChatRecord],
    ) -> SummaryRecord:
        """
        使用 LLM 将多条记录压缩为单条语义摘要

        将需要压缩的记录投影为"用户:" / "助手:" 对话文本，调用 LLM
        生成摘要，返回 SummaryRecord。如果 LLM 调用失败或返回空值，
        直接抛出异常，不回退到规则拼接——防止语义信息丢失静默发生。

        Parameters:
            records: 需要压缩的原始记录列表

        Returns:
            SummaryRecord 摘要记录

        Raises:
            ValueError: 待压缩的消息中没有有效对话内容
            RuntimeError: LLM 返回空摘要
            以及 LLMClient.request 的各类网络/API 异常
        """
        # 将记录投影为可读的对话文本
        conversation_lines = []
        for record in records:
            openai_msg = record.to_openai()
            if not openai_msg:
                continue
            role = openai_msg.get("role", "")
            content = openai_msg.get("content", "")
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

        return SummaryRecord(summary_text=summary_text)

    @staticmethod
    def _to_summary(
        records: list[ChatRecord],
    ) -> SummaryRecord:
        """
        纯文本拼接压缩（不使用 LLM）

        将多条记录的文本内容用"用户:" / "助手:" 前缀拼接为一段纯文本。
        用于 append/extend 路径的快速压缩，仅控制内存水位，不直接喂给 LLM。

        Parameters:
            records: 需要压缩的原始记录列表

        Returns:
            SummaryRecord 摘要记录
        """
        parts = ["【以下为压缩的历史对话】"]
        for record in records:
            openai_msg = record.to_openai()
            if not openai_msg:
                continue
            role = openai_msg.get("role", "")
            content = openai_msg.get("content", "")
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
        return SummaryRecord(summary_text="\n".join(parts))
