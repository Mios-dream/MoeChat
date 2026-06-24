"""
信息调度中心模块

V4 版本的核心模块，提供动态任务组合和流式处理功能。

核心组件：
- TaskScheduler: 信息调度中心，负责组合提示词和创建管道
- Pipeline: 流式处理管道，负责执行 LLM 调用和解析
- MultiParser: 多任务解析器，负责分发 JSON 字段
- Task/TaskResult: 任务定义和结果

调用流程：
```python
# 1. 创建调度器
scheduler = TaskScheduler()

# 2. 注册任务
scheduler.add_task(create_text_task())      # 文本生成
scheduler.add_task(create_motion_task())    # 动作标签

# 3. 创建管道
pipeline = scheduler.create_pipeline("你好呀~")

# 4. 流式处理结果
async for result in pipeline.execute():
    if result.task_type == "text":
        show_text(result.data)          # "你好呀~"
    elif result.task_type == "motion":
        play_motion(result.data)        # ["smile", "nod"]
```
"""

from core.scheduler.task import Task, TaskResult
from core.scheduler.multi_parser import MultiParser
from core.scheduler.scheduler import TaskScheduler, Pipeline
from core.scheduler.builtin_tasks import (
    create_text_task,
    create_motion_task,
)

__all__ = [
    # 核心组件
    "TaskScheduler",
    "Pipeline",
    "MultiParser",
    # 数据结构
    "Task",
    "TaskResult",
    # 内置任务工厂
    "create_text_task",
    "create_motion_task",
]
