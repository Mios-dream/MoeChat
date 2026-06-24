"""
动作生成 V3 数据模型

定义 V3 版本动作生成系统的数据结构。

核心模型：
- MotionChunk: 单个 chunk 数据（文本 + 动作标签）
- MotionResponse: 完整响应（多个 chunk）
- AtomicActionSpec: 动作规格
- MotionCurveData: 参数曲线数据
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AtomicActionSpec:
    """
    动作规格

    属性：
    - act: 动作名称
    - start: 开始时间（秒）
    - dur: 持续时间（秒），None 使用模板默认值
    - scale: 幅度缩放系数
    """
    act: str
    start: float = 0.0
    dur: float | None = None
    scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {"act": self.act, "start": self.start}
        if self.dur is not None:
            result["dur"] = self.dur
        if self.scale != 1.0:
            result["scale"] = self.scale
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicActionSpec":
        """从字典创建"""
        return cls(
            act=data.get("act", data.get("name", "")),
            start=data.get("start", 0.0),
            dur=data.get("dur", data.get("duration")),
            scale=data.get("scale", 1.0),
        )


@dataclass
class MotionCurveData:
    """
    参数曲线数据

    属性：
    - duration: 总时长（秒）
    - curves: 参数曲线 {参数ID: [[时间, 值], ...]}
    """
    duration: float
    curves: dict[str, list[list[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "duration": self.duration,
            "curves": self.curves,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionCurveData":
        """从字典创建"""
        return cls(
            duration=data.get("duration", 0.0),
            curves=data.get("curves", {}),
        )


@dataclass
class MotionChunk:
    """
    单个 chunk 数据

    属性：
    - index: chunk 序号
    - text: 该句回复文本
    - actions: 动作标签列表
    - motion: 动作数据（曲线或单帧）
    - duration: 该 chunk 总时长（秒）
    """
    index: int
    text: str
    actions: list[AtomicActionSpec] = field(default_factory=list)
    motion: MotionCurveData | None = None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "index": self.index,
            "text": self.text,
            "actions": [a.to_dict() for a in self.actions],
            "duration": self.duration,
        }
        if self.motion:
            result["motion"] = self.motion.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionChunk":
        """从字典创建"""
        actions = [
            AtomicActionSpec.from_dict(a) for a in data.get("actions", [])
        ]
        motion = None
        if "motion" in data:
            motion = MotionCurveData.from_dict(data["motion"])
        return cls(
            index=data.get("index", 0),
            text=data.get("text", ""),
            actions=actions,
            motion=motion,
            duration=data.get("duration", 0.0),
        )


@dataclass
class MotionResponse:
    """
    完整响应

    属性：
    - chunks: chunk 列表
    - total_duration: 总时长（秒）
    - session_id: 会话 ID（可选）
    """
    chunks: list[MotionChunk] = field(default_factory=list)
    total_duration: float = 0.0
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "total_duration": self.total_duration,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionResponse":
        """从字典创建"""
        chunks = [
            MotionChunk.from_dict(c) for c in data.get("chunks", [])
        ]
        return cls(
            chunks=chunks,
            total_duration=data.get("total_duration", 0.0),
            session_id=data.get("session_id", ""),
        )


@dataclass
class StreamingChunk:
    """
    流式 chunk 数据（用于增量解析）

    属性：
    - text: 句子文本
    - actions: 动作标签列表（原始字符串）
    - done: 是否完成
    """
    text: str = ""
    actions: list[str] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "t": self.text,
            "a": self.actions,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamingChunk":
        """从字典创建"""
        return cls(
            text=data.get("t", data.get("text", "")),
            actions=data.get("a", data.get("actions", [])),
            done=data.get("done", False),
        )
