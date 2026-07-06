from typing import Literal, Annotated
from pydantic import BaseModel, Field


class TextResponse(BaseModel):
    """
    聊天响应数据模型
    """

    type: Literal["text"] = "text"
    sentence_id: int  # 句子 ID
    message: str


class MotionResponse(BaseModel):
    """
    动作响应数据模型
    """

    type: Literal["motion"] = "motion"  # 固定为 "motion"
    sentence_id: int  # 句子 ID
    source_text: str  # 原始句子文本
    motions: list[dict]  # 动作数据列表
    duration: int  # 动作持续时间（毫秒）


class AudioResponse(BaseModel):
    """
    音频响应数据模型
    """

    type: Literal["audio"] = "audio"  # 固定为 "audio"
    sentence_id: int  # 句子 ID
    message: str  # 音频消息文本
    source_text: str  # 原始句子文本
    file: str  # 音频文件数据（base64 编码）


class DoneResponse(BaseModel):
    """
    聊天完成后的响应数据模型，表示聊天已结束
    """

    type: Literal["done"] = "done"  # 固定为 "done"
    full_text: str  # 完整文本
    done: bool = True  # 聊天是否完成，固定为 True


# 判别联合类型：根据 `type` 字段自动选择对应的模型
ChatResponse = Annotated[
    TextResponse | MotionResponse | AudioResponse | DoneResponse,
    Field(discriminator="type"),
]


class ToolCallResponse(BaseModel):
    """
    工具调用响应数据模型

    当 LLM 决定调用工具时，产出此响应通知前端工具调用请求的完整信息。
    """

    type: Literal["tool_call"] = "tool_call"
    call_id: str  # 工具调用唯一 ID（OpenAI tool_call_id）
    tool_name: str  # 工具名称
    arguments: str  # 工具参数（JSON 字符串）


class ToolResultResponse(BaseModel):
    """
    工具执行结果响应数据模型

    当工具执行完成后，产出此响应通知前端工具执行结果。
    包含原始调用参数与执行输出，用于前端展示完整的工具调用链路。
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str  # 对应的工具调用 ID
    tool_name: str  # 工具名称
    arguments: dict[str, object]  # 已解析的工具参数
    success: bool  # 执行是否成功
    result: str  # 执行结果（已解析）
    duration_ms: float  # 执行耗时（毫秒）


class ErrorResponse(BaseModel):
    """
    错误响应数据模型
    """

    type: Literal["error"] = "error"  # 固定为 "error"
    error_code: str  # 错误码
    data: str  # 错误数据


FullChatResponse = Annotated[
    ChatResponse | ToolCallResponse | ToolResultResponse | ErrorResponse,
    Field(discriminator="type"),
]
