"""
内置任务定义

提供开箱即用的任务工厂函数。

内置任务：
- create_text_task(): 文本生成任务
- create_motion_task(): 动作标签任务
- create_field_task(): 自定义字段提取任务

每个任务包含：
1. prompt: 提示词片段（注入到系统提示词）
2. parse_fn: 解析函数（从 JSON 中提取数据）
3. field_name: JSON 字段名
"""

from collections.abc import Awaitable, Callable

from core.scheduler.task import Task, TaskResult

# ============================================================
# 内置任务工厂
# ============================================================


def create_text_task(
    callback: Callable[[TaskResult], Awaitable[None]] | None = None,
    priority: int = 100,
) -> Task:
    """
    创建文本生成任务

    提示词：指导 LLM 生成回复文本
    字段名：text
    解析逻辑：提取 JSON 中的 text 字段

    参数：
    - callback: 完成回调（可选）
    - priority: 优先级（默认 100）

    返回：
    - Task 实例

    示例：
    ```python
    task = create_text_task()
    # task.prompt = "生成回复文本"
    # task.field_name = "text"
    # task.parse({"text": "你好"}) -> "你好"
    ```
    """
    return Task(
        name="text_generation",
        type="text",
        prompt="进行自然、流畅的回答",
        parse_fn=lambda data: data.get("text", ""),
        field_name="text",
        priority=priority,
        example='{"text": "你好呀~"}',
        rules=["每行必须包含 text 字段"],
    )


def create_motion_task(
    callback: Callable[[TaskResult], Awaitable[None]] | None = None,
    priority: int = 200,
) -> Task:
    """
    创建动作标签任务

    提示词：指导 LLM 为每个句子生成动作标签
    字段名：actions
    解析逻辑：提取 JSON 中的 actions 字段

    参数：
    - callback: 完成回调（可选）
    - priority: 优先级（默认 200）

    返回：
    - Task 实例

    示例：
    ```python
    task = create_motion_task()
    # task.prompt = "为每个句子生成动作标签"
    # task.field_name = "actions"
    # task.parse({"actions": ["smile", "nod"]}) -> ["smile", "nod"]
    ```
    """
    return Task(
        name="motion_generation",
        type="motion",
        prompt="为每个句子生成配合内容的动作标签（从下方【可用动作列表】中选择）",
        parse_fn=lambda data: data.get("actions", []),
        field_name="actions",
        priority=priority,
        example='{"text": "你好呀~", "actions": ["smile", "nod"]}',
        rules=["动作标签必须从【可用动作列表】中选择，不要自创动作名"],
    )
