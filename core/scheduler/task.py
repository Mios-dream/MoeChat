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


# ============================================================
# 工具调用事件类型
# ============================================================


@dataclass
class ToolCallEvent:
    """
    LLM 请求工具调用的标准化事件

    替代直接透传 OpenAI ChatCompletionMessageFunctionToolCallParam 原始类型，
    为上层消费者（ChatContext → SSE 输出）提供类型安全的工具调用信息。

    Attributes:
        call_id: OpenAI tool_call_id，全局唯一
        tool_name: 工具名称
        arguments: LLM 填充的调用参数（JSON 字符串）
    """

    call_id: str
    """OpenAI tool_call_id，格式如 'call_xxxxx'"""

    tool_name: str
    """工具名称，与 ToolMeta.name 对应"""

    arguments: str
    """LLM 填充的调用参数，JSON 字符串格式"""


@dataclass
class ToolResultEvent:
    """
    工具执行结果的标准化事件

    替代直接透传 ToolCallResult 结构化对象，
    为上层消费者提供类型安全、字段精简的执行结果信息。

    Attributes:
        call_id: 对应的请求 call_id
        tool_name: 工具名称
        arguments: 原始调用参数（用于前端展示调用链路）
        content: 返回给 LLM 的执行结果文本
        success: 是否执行成功
        error: 失败时的错误描述
        error_code: 结构化错误码
        duration_ms: 执行耗时（毫秒）
    """

    call_id: str
    """对应的请求 call_id"""

    tool_name: str
    """工具名称"""

    arguments: dict[str, Any]
    """原始调用参数，用于前端展示完整的工具调用链路"""

    content: str
    """返回给 LLM 的执行结果文本"""

    success: bool
    """是否执行成功"""

    error: str | None = None
    """失败时的错误描述"""

    error_code: str | None = None
    """结构化错误码（TOOL_NOT_FOUND, TOOL_TIMEOUT 等）"""

    duration_ms: float = 0.0
    """执行耗时（毫秒）"""


@dataclass
class ToolExecutionResult:
    """
    工具执行统一返回结构

    替代 ToolCallHandler 原先返回的裸 tuple
    (list[ToolMessage], list[Any])，将事件流和上下文注入消息分离。

    Attributes:
        tool_call_events: LLM 请求的工具调用事件列表
        tool_result_events: 工具执行后的结果事件列表
        context_messages: 需要注入 LLM 上下文的 OpenAI tool 角色消息列表
    """

    tool_call_events: list[ToolCallEvent]
    """LLM 请求的每个工具调用的标准化事件"""

    tool_result_events: list[ToolResultEvent]
    """每个工具调用执行后的标准化结果事件"""

    context_messages: list[Any]
    """需要注入 LLM 上下文的 OpenAI ChatCompletionToolMessageParam 列表"""
