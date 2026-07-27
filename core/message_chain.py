"""
消息链构建器

组件式挂载消息模块，自动计算 token 预算、调用 HistoryManager 压缩历史记录。
每个组件通过 add() + 优先级数值确定位置，链负责：
1. 按优先级排序所有槽位
2. 将最高优先级的消息块自动移至末尾（作为用户消息）
3. 计算固定部分的 token 开销，为历史记录分配剩余预算
4. 调用 HistoryManager.get_for_llm() 获取已压缩的历史记录（纯读取，无 LLM 调用）

注意：build() 是同步方法。LLM 语义摘要压缩在写入时由 compress_if_needed() 完成。
"""

from dataclasses import dataclass, field
from openai.types.chat import ChatCompletionMessageParam
from core.history_manager import HistoryManager


@dataclass
class _Slot:
    """
    槽位：消息链中的一个挂载点

    Attributes:
        priority: 优先级，数值越小越靠前。组件只需指定相对优先级（如 0/100/200/300/400），
                  链在 build() 时统一排序。
        messages: 静态消息块。如果此槽位包含最高优先级的 list，build() 自动将其移至末尾
                  作为用户消息处理。
        history_manager: HistoryManager 实例，build() 时调用其 get_for_llm() 获取
                        压缩后的历史记录。多条历史记录可以在不同优先级插入。
        max_count: 传递给 get_for_llm 的最大消息条数限制，默认 50。
    """

    priority: int
    messages: list = field(default_factory=list)
    history_manager: HistoryManager | None = None
    max_count: int = 50


class MessageChain:
    """
    消息链构建器

    通过优先级系统将多个消息模块组装为最终的消息列表，自动处理 token 预算分配
    和历史压缩。典型用法：:

        chain = MessageChain()
        chain.add_system(agent.prompt,           priority=0)    # 角色设定
        chain.add([memory_msg],                  priority=100)  # 记忆系统说明
        chain.add_history(agent.chat_history,    priority=200)  # 对话历史（内部 LLM 压缩）
        chain.add(dynamic_context,               priority=300)  # 知识库检索等
        chain.add(user_message,                  priority=999)  # 用户消息（自动移至末尾）

        messages = chain.build()  # 同步方法
    """

    def __init__(self, context_window: int = 128000):
        """
        Parameters:
            context_window: LLM 上下文窗口大小，用于计算 token 预算。
                            默认 128000（deepseek 系列的标准上下文长度）。
        """
        self._context_window = context_window
        self._slots: list[_Slot] = []

    def add_system(
        self,
        content: str,
        priority: int = 100,
    ) -> "MessageChain":
        """
        挂载一条 system 消息

        Parameters:
            content: 系统提示词文本
            priority: 优先级，数值越小越靠前
        """
        self._slots.append(
            _Slot(priority=priority, messages=[{"role": "system", "content": content}])
        )
        return self

    def add_history(
        self,
        content: HistoryManager,
        priority: int = 100,
        *,
        max_count: int = 50,
    ) -> "MessageChain":
        """
        挂载可压缩的聊天历史

        Parameters:
            content: HistoryManager 实例
            priority: 优先级，数值越小越靠前
            max_count: 压缩时的最大消息条数
        """
        self._slots.append(
            _Slot(
                priority=priority,
                history_manager=content,
                max_count=max_count,
            )
        )
        return self

    def add(
        self,
        content: list[ChatCompletionMessageParam],
        priority: int = 100,
    ) -> "MessageChain":
        """
        挂载一个消息块

        Parameters:
            content: 普通消息块
            priority: 优先级，数值越小越靠前
        """
        self._slots.append(_Slot(priority=priority, messages=list(content)))
        return self

    def build(
        self,
    ) -> list[ChatCompletionMessageParam]:
        """
        按优先级构建完整消息列表

        构建流程：
        1. 按优先级排序所有槽位
        2. 自动识别最高优先级的消息块作为用户消息（移至末尾）
        3. 计算非历史块 + 用户消息的固定 token 开销（+200 安全缓冲 + 4000 生成缓冲）
        4. 遍历 ordered 列表：普通 list 直接追加，HistoryManager 调用 get_for_llm() 读取（无 LLM 调用）
        5. 用户消息追加到末尾

        HistoryManager.get_for_llm() 仅做条数截断，不做 LLM 调用。
        LLM 语义摘要压缩已在写入时由 compress_if_needed() 完成。

        Returns:
            完整的消息列表 list[ChatCompletionMessageParam]
        """
        self._slots.sort(key=lambda s: s.priority)

        ordered: list[tuple[int, list | HistoryManager]] = []
        user_block: tuple[int, list] | None = None

        for slot in self._slots:
            if slot.history_manager:
                ordered.append((slot.priority, slot.history_manager))
            elif slot.messages:
                ordered.append((slot.priority, slot.messages))
                if user_block is None or slot.priority > user_block[0]:
                    user_block = (slot.priority, slot.messages)

        # ---- 从 ordered 中分离出 user_message ----
        user_messages: list[ChatCompletionMessageParam] = []
        if user_block is not None:
            ordered.remove(user_block)
            user_messages = user_block[1]

        # ---- 计算固定 token 开销（非历史块 + user） ----
        fixed: list[ChatCompletionMessageParam] = []
        for _, item in ordered:
            if isinstance(item, list):
                fixed.extend(item)
        fixed.extend(user_messages)

        # +200 安全缓冲 + 4000 LLM 生成缓冲（为模型预留输出 token 空间）
        reserved = HistoryManager.estimate_list_tokens(fixed) + 200 + 4000

        # ---- 按优先级组装最终结果，HistoryManager 替换为压缩版本 ----
        result: list[ChatCompletionMessageParam] = []
        for _, item in ordered:
            if isinstance(item, HistoryManager):
                result.extend(
                    item.get_for_llm(
                        reserved_tokens=reserved,
                        max_count=50,
                    )
                )
            else:
                result.extend(item)
        result.extend(user_messages)

        return result
