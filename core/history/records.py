"""
聊天记录类别族（抽象基类 + 内置类别）

设计目标：
- ``ChatRecord`` 为抽象基类，仅保留「类别标识 kind + 创建时间 timestamp + 格式化策略」，
  具体字段由每个消息类别自行声明，不做「最大公共字段」的扁平化设计；
- 每个类别自持 ``to_openai()``（自决能否/如何格式化为 OpenAI 消息并注入 LLM）与
  ``to_display()``（自决能否/如何进入展示 API），新增消息类型只需新增一个子类；
- ``is_conversation`` 标记是否属于「原始对话」，供情感分析等场景按类别过滤；

类别总览：
    user          用户主动消息
    chat          用户对话触发的助手回复
    interaction   自动交互的助手回复（前端折叠为"自动回复"徽标）
    event         交互触发事件（仅注入 LLM 的 user 消息，展示折叠）
    summary       压缩摘要（仅注入 LLM 的 system 消息，展示过滤）
    tool_call     工具调用（标准 tool_calls 结构）
    tool_result   工具结果（标准 tool 结构）
    note          内部备注（默认不注入不展示）
    system        角色初始化 system（注入 LLM，展示过滤）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from openai.types.chat import ChatCompletionMessageParam

from my_utils.token_counter import estimate_tokens

# 类别标识 → 记录类 注册表：子类声明 kind 时自动登记（见 __init_subclass__）
KIND_TO_CLASS: dict[str, type["ChatRecord"]] = {}


@dataclass
class ChatRecord(ABC):
    """
    抽象聊天记录

    子类职责：
    1. 声明 ``kind`` 类别标识与自定义字段；
    2. 实现 ``to_openai()`` —— 自决「能否格式化为 OpenAI 消息」及注入形态，
       返回 None 表示不注入 LLM 消息链；
    3. 实现 ``to_display()`` —— 自决「能否进入展示 API」及展示形态，
       返回 None 表示展示过滤；
    4. （可选）覆写 ``estimate_tokens()`` 定制 token 估算；
    5. （可选）将 ``is_conversation`` 置为 False，表示不属于原始对话（不进入情感分析）。
    """

    # 创建时间（ISO，含本地时区偏移），默认取当前时间
    timestamp: str = field(
        default_factory=lambda: datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )
    # 类别标识（子类覆写），声明时自动登记到 KIND_TO_CLASS
    kind: ClassVar[str] = ""
    # 是否属于「原始对话」，供 get_raw_recent 等按类别过滤
    is_conversation: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        """子类创建时自动登记类别注册表（kind 非空才登记）"""
        super().__init_subclass__(**kwargs)
        if cls.kind:
            KIND_TO_CLASS[cls.kind] = cls

    # ============================================================
    # 格式化策略（子类必须实现）
    # ============================================================

    @abstractmethod
    def to_openai(self) -> ChatCompletionMessageParam | None:
        """
        格式化（或决定不可格式化）为 OpenAI 消息；None 表示不注入 LLM。

        注入形态由类别自行决定：如 summary 返回 system、event 返回 user、
        note 默认返回 None（不注入）。
        """
        ...

    @abstractmethod
    def to_display(self) -> dict | None:
        """
        格式化（或决定不可展示）为展示字典；None 表示展示过滤。

        展示字典需包含 kind/timestamp 等展示元数据，由 to_display_dict() 提供公共壳。
        """
        ...

    # ============================================================
    # 公共工具
    # ============================================================

    def estimate_tokens(self) -> int:
        """估算本记录占用的 token（含 4 token 的角色标记开销）"""
        return estimate_tokens(self._text_for_token()) + 4

    def _text_for_token(self) -> str:
        """默认取 content 字段文本估算；无 content 的类别自行覆写"""
        content = getattr(self, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        return ""

    def to_display_dict(self) -> dict:
        """展示字典公共壳：kind + timestamp，供 to_display() 复用"""
        return {"kind": self.kind, "timestamp": self.timestamp}


# ============================================================
# 内置消息类别
# ============================================================


@dataclass
class UserRecord(ChatRecord):
    """用户主动消息"""

    kind: ClassVar[str] = "user"
    content: str | list | None = None

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "user", "content": self.content}

    def to_display(self) -> dict | None:
        result = self.to_display_dict()
        result.update({"content": self.content})
        return result


@dataclass
class ChatReplyRecord(ChatRecord):
    """用户对话触发的助手回复"""

    kind: ClassVar[str] = "chat"
    content: str | list | None = None

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "assistant", "content": self.content}

    def to_display(self) -> dict | None:
        result = self.to_display_dict()
        result.update({"content": self.content})
        return result


@dataclass
class InteractionRecord(ChatRecord):
    """自动交互的助手回复（前端折叠为"自动回复"徽标）"""

    kind: ClassVar[str] = "interaction"
    content: str | list | None = None

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "assistant", "content": self.content}

    def to_display(self) -> dict | None:
        result = self.to_display_dict()
        result.update({"content": self.content})
        return result


@dataclass
class EventRecord(ChatRecord):
    """
    交互触发事件（系统合成的用户上下文）

    与紧邻的 InteractionRecord 成对写入：在 LLM 上下文注入为 user 消息（消除悬空
    assistant），展示侧过滤（折叠为"自动回复"徽标）。
    """

    kind: ClassVar[str] = "event"
    is_conversation: ClassVar[bool] = False
    event_type: str = ""
    scene: str = ""
    # 事件附加上下文（鼠标/应用/电量等状态），暂不参与 LLM 拼装，仅作元数据保留
    context: dict = field(default_factory=dict)

    def _event_text(self) -> str:
        """将事件字段拼装为注入 LLM 的 user 内容（与交互请求链中事件描述格式一致）"""
        return "\n".join(
            [
                f"【事件类型】{self.event_type}",
                f"【场景】{self.scene}",
                f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ]
        )

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "user", "content": self._event_text()}

    def to_display(self) -> dict | None:
        return None

    def _text_for_token(self) -> str:
        return self._event_text()


@dataclass
class SummaryRecord(ChatRecord):
    """
    压缩摘要（仅注入 LLM，展示过滤）

    由 HistoryManager 的压缩路径产出，替换被压缩的旧记录；to_openai() 以 system
    角色注入，to_display() 返回 None 实现展示过滤。
    """

    kind: ClassVar[str] = "summary"
    is_conversation: ClassVar[bool] = False
    summary_text: str = ""

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {
            "role": "system",
            "content": f"【以下为压缩的历史对话】{self.summary_text}",
        }

    def to_display(self) -> dict | None:
        return None

    def _text_for_token(self) -> str:
        return self.summary_text


@dataclass
class ToolCallRecord(ChatRecord):
    """工具调用（标准 tool_calls 结构）"""

    kind: ClassVar[str] = "tool_call"
    tool_calls: list | None = None

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "assistant", "content": None, "tool_calls": self.tool_calls}

    def to_display(self) -> dict | None:
        result = self.to_display_dict()
        result.update({"tool_calls": self.tool_calls})
        return result

    def estimate_tokens(self) -> int:
        """工具调用按函数名与参数文本估算 token"""
        total = 4
        for call in self.tool_calls or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            total += estimate_tokens(str(fn.get("name", "")))
            total += estimate_tokens(str(fn.get("arguments", "")))
        return total


@dataclass
class ToolResultRecord(ChatRecord):
    """工具结果（标准 tool 结构）"""

    kind: ClassVar[str] = "tool_result"
    tool_call_id: str = ""
    content: str | list | None = None

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }

    def to_display(self) -> dict | None:
        result = self.to_display_dict()
        result.update({"tool_call_id": self.tool_call_id, "content": self.content})
        return result


@dataclass
class NoteRecord(ChatRecord):
    """内部备注（默认不注入不展示，按需由子类覆写策略）"""

    kind: ClassVar[str] = "note"
    is_conversation: ClassVar[bool] = False
    content: str = ""

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return None

    def to_display(self) -> dict | None:
        return None

    def _text_for_token(self) -> str:
        return self.content


@dataclass
class SystemRecord(ChatRecord):
    """角色初始化 system（注入 LLM，展示过滤），供 startWith 等上下文连续性使用"""

    kind: ClassVar[str] = "system"
    content: str = ""

    def to_openai(self) -> ChatCompletionMessageParam | None:
        return {"role": "system", "content": self.content}

    def to_display(self) -> dict | None:
        return None

    def _text_for_token(self) -> str:
        return self.content
