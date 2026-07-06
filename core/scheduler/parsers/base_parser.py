from collections.abc import Generator
from typing import Any
from core.scheduler.task import Task, TaskResult


class BaseParser:
    """
    多任务解析器

    将 LLM 的流式 JSON 输出分发给对应的子解析器。

    使用示例：
    ```python
    parser = MultiParser()
    parser.register_task(create_text_task())
    parser.register_task(create_motion_task())

    # 流式解析
    for token in stream:
        for result in parser.stream_parse(token):
            print(result.task_type, result.data)
    ```
    """

    def register_task(self, task: Task) -> None:
        """
        注册任务

        参数：
        - task: 任务定义
        """
        ...

    def reset(self, keep_counter: bool = False) -> None:
        """
        重置解析器状态

        参数：
        - keep_counter: 是否保留句子计数器（默认 False）
        """
        ...

    def parse(self, data: Any) -> Generator[TaskResult, None, None]:
        """
        解析完整的 JSON 数据
        """
        ...
