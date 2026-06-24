"""
流式 JSON 行解析器

专门用于解析 LLM 流式输出的 JSON 行格式。

输出格式示例：
```
{"t": "你好呀~", "a": ["smile", "nod"]}
{"t": "今天天气真好呢", "a": ["look_up"]}
{"t": "", "a": [], "done": true}
```

核心功能：
1. 增量解析：逐 token 处理，检测完整 JSON 行
2. 缓冲区管理：处理跨 token 的 JSON 边界
3. 错误容忍：跳过无效的 JSON 行

使用示例：
```python
parser = StreamingJsonLineParser()

# 流式解析
for token in stream:
    for chunk in parser.stream_parse(token):
        print(chunk)  # StreamingChunk 实例

# 处理剩余数据
for chunk in parser.flush():
    print(chunk)
```
"""

from typing import Generator
import json

from my_utils.log import logger as Log
from core.expression_generator.motion_schema import StreamingChunk


class StreamingJsonLineParser:
    """
    流式 JSON 行解析器

    逐 token 处理 LLM 输出，检测完整的 JSON 行并解析为 StreamingChunk。

    特性：
    - 增量解析：不等待完整响应
    - 错误容忍：跳过无效行
    - 缓冲区管理：处理跨 token 边界
    """

    def __init__(self, separator: str = "\n"):
        """
        初始化解析器

        参数：
        - separator: 行分隔符（默认换行符）
        """
        self._separator = separator
        self._buffer = ""
        self._chunk_count = 0

    def reset(self) -> None:
        """重置解析器状态"""
        self._buffer = ""
        self._chunk_count = 0

    def _parse_line(self, line: str) -> StreamingChunk | None:
        """
        解析单行 JSON

        参数：
        - line: JSON 字符串

        返回：
        - StreamingChunk 实例，解析失败返回 None
        """
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
            chunk = StreamingChunk.from_dict(data)
            self._chunk_count += 1
            return chunk
        except json.JSONDecodeError as e:
            Log.debug(f"[流式解析器] JSON 解析失败: {line[:50]}... ({e})")
            return None

    def stream_parse(self, token: str) -> Generator[StreamingChunk, None, None]:
        """
        流式解析

        每接收到一个 token，检查缓冲区是否有完整的 JSON 行。

        参数：
        - token: 单个 token

        产出：
        - StreamingChunk 实例
        """
        self._buffer += token

        # 检查是否有完整的行
        while self._separator in self._buffer:
            line, self._buffer = self._buffer.split(self._separator, 1)
            chunk = self._parse_line(line)
            if chunk is not None:
                yield chunk

    def flush(self) -> Generator[StreamingChunk, None, None]:
        """
        刷新缓冲区，处理剩余内容

        在流结束时调用，确保处理所有剩余数据。

        产出：
        - 缓冲区中解析完成的 StreamingChunk
        """
        if self._buffer.strip():
            chunk = self._parse_line(self._buffer)
            if chunk is not None:
                yield chunk
        self._buffer = ""

    @property
    def chunk_count(self) -> int:
        """已解析的 chunk 数量"""
        return self._chunk_count


class StreamingJsonArrayParser:
    """
    流式 JSON 数组解析器

    用于解析 LLM 输出的 JSON 数组格式。

    输出格式示例：
    ```json
    [
      {"t": "你好呀~", "a": ["smile"]},
      {"t": "今天天气真好呢", "a": ["look_up"]}
    ]
    ```

    特性：
    - 增量解析：逐 token 处理
    - 数组检测：自动检测数组开始和结束
    - 对象累积：处理跨 token 的 JSON 对象
    """

    def __init__(self):
        """初始化解析器"""
        self._buffer = ""
        self._in_array = False
        self._depth = 0  # 大括号深度
        self._object_buffer = ""
        self._chunk_count = 0

    def reset(self) -> None:
        """重置解析器状态"""
        self._buffer = ""
        self._in_array = False
        self._depth = 0
        self._object_buffer = ""
        self._chunk_count = 0

    def _try_parse_object(self) -> StreamingChunk | None:
        """
        尝试解析累积的对象缓冲区

        返回：
        - StreamingChunk 实例，解析失败返回 None
        """
        if not self._object_buffer.strip():
            return None

        try:
            data = json.loads(self._object_buffer)
            if isinstance(data, dict):
                chunk = StreamingChunk.from_dict(data)
                self._chunk_count += 1
                self._object_buffer = ""
                return chunk
        except json.JSONDecodeError:
            pass

        return None

    def stream_parse(self, token: str) -> Generator[StreamingChunk, None, None]:
        """
        流式解析

        参数：
        - token: 单个 token

        产出：
        - StreamingChunk 实例
        """
        self._buffer += token

        for char in token:
            # 检测数组开始
            if char == '[' and not self._in_array:
                self._in_array = True
                continue

            # 检测数组结束
            if char == ']' and self._in_array and self._depth == 0:
                # 尝试解析剩余内容
                chunk = self._try_parse_object()
                if chunk is not None:
                    yield chunk
                self._in_array = False
                continue

            if not self._in_array:
                continue

            # 跟踪大括号深度
            if char == '{':
                if self._depth == 0:
                    self._object_buffer = ""
                self._depth += 1
                self._object_buffer += char
            elif char == '}':
                self._depth -= 1
                self._object_buffer += char
                # 对象完成
                if self._depth == 0:
                    chunk = self._try_parse_object()
                    if chunk is not None:
                        yield chunk
            elif self._depth > 0:
                # 在对象内部，累积字符
                self._object_buffer += char

    def flush(self) -> Generator[StreamingChunk, None, None]:
        """
        刷新缓冲区

        产出：
        - 缓冲区中解析完成的 StreamingChunk
        """
        if self._in_array and self._object_buffer.strip():
            chunk = self._try_parse_object()
            if chunk is not None:
                yield chunk
        self.reset()

    @property
    def chunk_count(self) -> int:
        """已解析的 chunk 数量"""
        return self._chunk_count
