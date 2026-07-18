"""
聊天响应数据模型

统一使用与 OpenAI Chat Completion API 兼容的消息格式：
- type: "chat:result" → 聊天结果（文本/音频/动作/工具调用），role 标明角色
- type: "tool" → 工具执行结果
- type: "done" → 流结束
- type: "error" → 错误

设计原则：
- 所有事件内部字段与 OpenAI 消息格式对齐（role/content/tool_calls/tool_call_id）
- 自定义扩展字段命名简洁，便于前端使用和后续拓展
- 前端可用同一套解析逻辑处理实时流事件和历史记录
"""

from typing import Literal, Annotated
from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    """
    OpenAI 格式函数调用信息

    与聊天历史中 tool_calls[].function 结构完全一致。
    """

    name: str
    arguments: str  # JSON 字符串


class ToolCallItem(BaseModel):
    """
    OpenAI 格式单个工具调用项

    与聊天历史中 tool_calls[] 结构完全一致。
    """

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    """
    聊天结果消息

    覆盖三种场景：
     1. 文本输出（可附带音频/动作）：content 非空，extras 可选
     2. 工具调用：tool_calls 非空，无 content
     3. 增量更新：仅携带 extras 字段，前端按 sentence_id 合并

    extras 为统一扩展字段，容纳 audio / motion 等应用层数据。
    """

    type: Literal["chat:result"] = "chat:result"
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallItem] | None = None
    extras: dict | None = None


class ToolMessage(BaseModel):
    """
    工具执行结果消息

    与 OpenAI 的 role: "tool" 消息结构完全一致。
    """

    type: Literal["chat:result"] = "chat:result"
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


class DoneMessage(BaseModel):
    """
    流结束消息
    """

    type: Literal["chat:done"] = "chat:done"
    full_text: str
    done: bool = True


class ErrorMessage(BaseModel):
    """
    错误消息
    """

    type: Literal["error"] = "error"
    error_code: str
    data: str


FullChatResponse = Annotated[
    AssistantMessage | ToolMessage | DoneMessage | ErrorMessage,
    Field(discriminator="type"),
]
