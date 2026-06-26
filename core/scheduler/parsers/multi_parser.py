"""
多任务解析器

将 LLM 的流式 JSON 输出分发给对应的子解析器。

核心功能：
1. 流式解析 JSON 行（每行一个 JSON 对象）
2. 根据已注册的任务，从 JSON 中提取对应字段
3. 句子 ID 自动分配（同一行的 text 和 actions 共享同一 ID）

输入格式（LLM 输出）：
```
{"text": "你好呀~", "actions": ["smile", "nod"]}
{"text": "今天天气真好", "actions": ["look_up"]}
```

输出格式（TaskResult 流）：
```
TaskResult(task_type="text", data="你好呀~", sentence_id=1)
TaskResult(task_type="motion", data=["smile", "nod"], sentence_id=1)
TaskResult(task_type="text", data="今天天气真好", sentence_id=2)
TaskResult(task_type="motion", data=["look_up"], sentence_id=2)
```
"""

from collections.abc import Generator
from typing import Any
import json
import time

from my_utils.log import logger as Log
from core.scheduler.task import Task, TaskResult


class MultiParser:
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

    def __init__(self):
        """初始化多任务解析器"""
        # 任务注册表 {task_type: Task}
        self._tasks: dict[str, Task] = {}
        # 行缓冲区（用于处理跨 token 的 JSON 行）
        self._line_buffer: str = ""
        # 句子 ID 计数器
        self._sentence_counter: int = 0
        # 上一次的文本内容（用于检测新句子）
        self._last_text: str = ""

    def register_task(self, task: Task) -> None:
        """
        注册任务

        参数：
        - task: 任务定义
        """
        self._tasks[task.type] = task
        Log.debug(f"[解析器] 注册任务: {task.name} (type={task.type})")

    def reset(self) -> None:
        """重置解析器状态"""
        self._line_buffer = ""
        self._sentence_counter = 0
        self._last_text = ""

    def _get_sentence_id(self, data: dict[str, Any]) -> int:
        """
        获取句子 ID

        如果 JSON 中包含 sentence_id 字段则使用，否则自动分配。
        当检测到新的 text 字段时，自动递增句子 ID。

        参数：
        - data: 解析后的 JSON 数据

        返回：
        - 句子 ID
        """
        # 如果显式指定了 sentence_id，使用它
        if "sentence_id" in data:
            return data["sentence_id"]

        # 检测是否是新的句子
        current_text = data.get("text", "")
        if current_text and current_text != self._last_text:
            self._sentence_counter += 1
            self._last_text = current_text

        return self._sentence_counter

    def _extract_task_data(self, data: dict[str, Any], task: Task) -> TaskResult | None:
        """
        从 JSON 中提取任务数据

        参数：
        - data: 解析后的 JSON 数据
        - task: 任务定义

        返回：
        - TaskResult 实例，如果该任务没有数据则返回 None
        """
        # 检查 JSON 中是否包含该任务的字段
        if task.field_name not in data:
            return None

        # 使用任务的解析函数提取数据
        try:
            parsed_data = task.parse(data)
        except Exception as e:
            Log.warning(f"[解析器] 解析失败 ({task.type}): {e}")
            return None

        # 创建结果
        return TaskResult(
            task_name=task.name,
            task_type=task.type,
            data=parsed_data,
            raw_data=data,
            sentence_id=self._get_sentence_id(data),
            timestamp=time.time(),
        )

    def _process_line(self, line: str) -> Generator[TaskResult, None, None]:
        """
        处理一行 JSON

        参数：
        - line: JSON 字符串

        产出：
        - TaskResult 实例
        """
        line = line.strip()
        if not line:
            return

        # 解析 JSON
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            Log.debug(f"[解析器] 跳过无效行: {line[:50]}...")
            return

        # 确保是字典类型
        if not isinstance(data, dict):
            return

        # 遍历已注册的任务，提取数据
        for task in self._tasks.values():
            result = self._extract_task_data(data, task)
            if result is not None:
                yield result

    def stream_parse(self, token: str) -> Generator[TaskResult, None, None]:
        """
        流式解析

        每接收到一个 token，检查缓冲区是否有完整的 JSON 行。
        如果有，解析并产出结果。

        参数：
        - token: 单个 token

        产出：
        - TaskResult 实例
        """
        self._line_buffer += token

        # 检查是否有完整的行（以换行符分隔）
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            yield from self._process_line(line)

    def flush(self) -> Generator[TaskResult, None, None]:
        """
        刷新缓冲区，处理剩余内容

        在流结束时调用，确保处理所有剩余数据。

        产出：
        - 缓冲区中解析完成的 TaskResult
        """
        if self._line_buffer.strip():
            yield from self._process_line(self._line_buffer)
        self._line_buffer = ""

    @property
    def sentence_count(self) -> int:
        """已处理的句子数量"""
        return self._sentence_counter

    @property
    def registered_task_types(self) -> set[str]:
        """获取已注册的任务类型集合"""
        return set(self._tasks.keys())
