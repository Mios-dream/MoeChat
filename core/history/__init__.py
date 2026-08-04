"""
聊天历史包

将聊天记录的「私有记录类别族」与「统一管理器」收束为独立子包，
取代原先分散的 core/chat_record.py 与 core/history_manager.py。

- records: 抽象基类 ChatRecord + 内置消息类别 + 类别注册表/工厂
- manager: 单源私有记录管理器 HistoryManager（LLM/展示投影均由类别自决）
"""

from core.history.records import (
    ChatRecord,
    UserRecord,
    ChatReplyRecord,
    InteractionRecord,
    EventRecord,
    SummaryRecord,
    ToolCallRecord,
    ToolResultRecord,
    NoteRecord,
    SystemRecord,
    KIND_TO_CLASS,
)
from core.history.manager import HistoryManager

__all__ = [
    "ChatRecord",
    "UserRecord",
    "ChatReplyRecord",
    "InteractionRecord",
    "EventRecord",
    "SummaryRecord",
    "ToolCallRecord",
    "ToolResultRecord",
    "NoteRecord",
    "SystemRecord",
    "KIND_TO_CLASS",
    "HistoryManager",
]
