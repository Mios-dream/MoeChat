"""
聊天请求数据模型

API 层只表达用户意图，不暴露 OpenAI 格式细节。
服务端根据 enable_multimodal 配置自动决定底层请求格式。
"""

from enum import Enum

from pydantic import BaseModel


class FileType(str, Enum):
    """文件类型"""

    IMAGE = "image"
    TEXT = "text"


class FileAttachment(BaseModel):
    """文件附件"""

    data: str
    """Base64 编码的文件数据"""
    name: str = ""
    """文件名"""
    type: FileType = FileType.TEXT
    """文件类型（image / text），前端显式指定"""


class ChatRequest(BaseModel):
    """
    聊天请求

    用户发送一条文本消息，可选附带文件附件（图片或 txt）。
    服务端根据 LLM 配置自动处理（多模态直传/OCR 识别/文本提取）。
    """

    text: str = ""
    """用户文本消息"""
    files: list[FileAttachment] = []
    """文件附件列表（图片自动 OCR 或 image_url，txt 自动读取内容）"""
    generation_motion: bool = False
    """是否生成 Live2D 动作"""
    is_sleep_mode: bool = False
    """是否睡眠模式"""
