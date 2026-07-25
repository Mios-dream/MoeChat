"""
消息链构建器

组件式挂载消息模块，自动计算 token 预算、压缩历史记录。
每个组件通过 add() + 优先级数值确定位置，链负责组装和预算分配。
"""

from dataclasses import dataclass, field
from openai.types.chat import ChatCompletionMessageParam
from core.history_manager import HistoryManager


@dataclass
class _Slot:
    """槽位：一个挂载点"""

    priority: int
    messages: list = field(default_factory=list)
    history_manager: HistoryManager | None = None
    max_count: int = 50


class MessageChain:
    """
    消息链构建器

    用法::

        chain = MessageChain()
        chain.add_system(agent.prompt,       priority=0)
        chain.add([memory_msg],              priority=50)
        chain.add_history(agent.chat_history, priority=100)
        chain.add(dynamic_context,           priority=150)
        chain.add(user_message,              priority=999)

        messages = chain.build()
    """

    def __init__(self, context_window: int = 128000):
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

        - 最高 priority 的 list → user_message（自动移至末尾）
        - HistoryManager → 自动压缩后插入对应优先级位置
        - 其余 list → 按优先级排列

        Returns:
            list[ChatCompletionMessageParam]
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
