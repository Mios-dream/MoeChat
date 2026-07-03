"""
任务定义模块

定义信息调度中心的任务数据结构。

核心概念：
- Task: 任务定义，包含提示词、解析器、字段名
- TaskResult: 任务执行结果

Task 的关键属性：
- name: 任务名称（唯一标识）
- type: 任务类型（用于解析器分发）
- prompt: 提示词片段（注入到系统提示词中）
- parser: 解析器（从 JSON 中提取该任务的数据）
- field_name: JSON 字段名（如 "text", "actions"）
- priority: 优先级（影响提示词顺序）
"""

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
import time


@dataclass
class Task:
    """
    任务定义

    属性：
    - name: 任务名称（唯一标识）
    - type: 任务类型（用于解析器分发）
    - prompt: 提示词片段（注入到系统提示词，说明该任务的要求）
    - parse_fn: 解析函数（从 JSON 字典中提取数据）
    - field_name: JSON 字段名（如 "text", "actions"）
    - priority: 优先级（数字越小优先级越高，影响提示词顺序）
    - example: 输出示例片段（如 '{"text": "你好呀~"}'）
    - rules: 规则列表（如 ["每行必须包含 text 字段"]）

    示例：
    ```python
    task = Task(
        name="text_generation",
        type="text",
        prompt="生成回复文本",
        parse_fn=lambda data: data.get("text", ""),
        field_name="text",
        priority=100,
        example='{"text": "你好呀~"}',
        rules=["每行必须包含 text 字段"]
    )
    ```
    """

    name: str
    type: str
    prompt: str
    parse_fn: Callable[[dict[str, Any]], Any]
    field_name: str = ""
    priority: int = 100
    example: str = ""
    rules: list[str] = field(default_factory=list)

    def __post_init__(self):
        """验证任务定义"""
        if not self.name:
            raise ValueError("任务名称不能为空")
        if not self.type:
            raise ValueError("任务类型不能为空")
        if not self.field_name:
            # 默认使用 type 作为字段名
            self.field_name = self.type

    def parse(self, data: dict[str, Any]) -> Any:
        """
        从 JSON 数据中提取该任务的数据

        参数：
        - data: 解析后的 JSON 字典

        返回：
        - 提取的数据
        """
        return self.parse_fn(data)


@dataclass
class TaskResult:
    """
    任务执行结果

    属性：
    - task_name: 任务名称
    - task_type: 任务类型
    - data: 解析后的数据
    - raw_data: 原始 JSON 数据
    - sentence_id: 句子序号（用于关联同一句的不同任务结果）
    - timestamp: 时间戳

    示例：
    ```python
    result = TaskResult(
        task_name="text_generation",
        task_type="text",
        data="你好呀~",
        raw_data={"text": "你好呀~", "actions": ["smile"]},
        sentence_id=1,
        timestamp=1234567890.0
    )
    ```
    """

    task_name: str
    task_type: str
    data: Any
    raw_data: dict[str, Any] = field(default_factory=dict)
    sentence_id: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "data": self.data,
            "sentence_id": self.sentence_id,
            "timestamp": self.timestamp,
        }
