"""
可插拔的响应解析器

支持多种解析策略，可根据需要选择或自定义。

内置解析器：
- TextParser: 纯文本解析（默认）
- JsonParser: JSON 对象解析
- JsonLineParser: JSON 行解析（流式友好）

自定义解析器：
继承 ResponseParser 并实现 parse/stream_parse 方法。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable, TypeVar, Generic
import json
import re

T = TypeVar("T", covariant=True)  # 流式产出元素类型（协变）
R = TypeVar("R")  # parse() 的返回类型


@runtime_checkable
class StreamParserProtocol(Protocol[T]):
    """
    流式解析器协议

    定义流式解析器的接口规范。
    任何实现此协议的类都可以作为流式解析器使用。

    方法：
    - stream_parse: 流式解析（逐 token）
    - flush: 刷新缓冲区
    - reset: 重置状态
    """

    def stream_parse(self, token: str) -> Iterator[T]:
        """流式解析"""
        ...

    def flush(self) -> Iterator[T]:
        """刷新缓冲区"""
        ...

    def reset(self) -> None:
        """重置状态"""
        ...


class ResponseParser(ABC, Generic[T, R]):
    """
    响应解析器基类

    所有自定义解析器都应继承此类。

    方法：
    - parse: 解析完整响应
    - stream_parse: 流式解析（逐 token）
    - reset: 重置解析器状态
    """

    @abstractmethod
    def parse(self, content: str) -> R:
        """
        解析完整响应内容

        参数：
        - content: 完整的响应文本

        返回：
        - 解析后的数据结构
        """
        pass

    @abstractmethod
    def stream_parse(self, token: str) -> Iterator[T]:
        """
        流式解析（逐 token）

        默认实现：累积 token，不产生输出。
        子类可重写此方法实现流式解析。

        参数：
        - token: 单个 token

        产出：
        - 解析完成的数据块
        """
        pass

    def reset(self) -> None:
        """
        重置解析器状态

        用于在新的请求开始时清理内部状态。
        """
        pass

    def flush(self) -> Iterator[T]:
        """
        刷新缓冲区，处理剩余内容

        在流结束时调用，确保处理所有剩余数据。
        默认实现不产生任何输出。

        产出：
        - 缓冲区中解析完成的数据
        """
        return iter(())


class TextParser(ResponseParser[str, str]):
    """
    纯文本解析器

    直接返回文本内容，不做任何解析。
    适用于不需要结构化响应的场景。
    """

    def __init__(self, strip: bool = True):
        """
        初始化文本解析器

        参数：
        - strip: 是否去除首尾空白
        """
        self._strip = strip

    def parse(self, content: str) -> str:
        """返回纯文本"""
        return content.strip() if self._strip else content

    def stream_parse(self, token: str) -> Iterator[str]:
        """直接产出 token"""
        yield token


class JsonParser(ResponseParser[dict | list, dict | list]):
    """
    JSON 对象解析器

    从响应中提取第一个 JSON 对象。
    支持处理 markdown 代码块包裹的 JSON。

    使用示例：
    ```python
    parser = JsonParser()
    result = parser.parse('```json\n{"key": "value"}\n```')
    # result = {"key": "value"}
    ```
    """

    def __init__(self, strict: bool = False):
        """
        初始化 JSON 解析器

        参数：
        - strict: 是否严格模式（不允许非 JSON 内容）
        """
        self._strict = strict
        self._buffer = ""
        self._completed = False

    def reset(self) -> None:
        """重置流式解析状态"""
        self._buffer = ""
        self._completed = False

    def _extract_first_json(self, content: str) -> dict[str, Any] | list[Any] | None:
        """从文本中提取第一个完整的 JSON 对象或数组。"""
        start_candidates = [
            idx for idx in (content.find("{"), content.find("[")) if idx != -1
        ]
        if not start_candidates:
            return None

        candidate = content[min(start_candidates) :]

        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    def parse(self, content: str) -> dict[str, Any] | list[Any]:
        """
        解析 JSON 内容

        参数：
        - content: 响应文本

        返回：
        - 解析后的 JSON 对象或数组

        异常：
        - ValueError: 无法解析 JSON 时抛出
        """
        if not content:
            raise ValueError("响应内容为空")

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 对象
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试提取 JSON 数组
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        if self._strict:
            raise ValueError(f"无法解析 JSON: {content[:100]}...")

        return {}

    def stream_parse(self, token: str) -> Iterator[dict[str, Any] | list[Any]]:
        """
        流式解析 JSON

        累积 token，直到解析出第一个完整的 JSON 对象或数组后立即产出。
        解析完成后会忽略后续 token，直到调用 reset()。

        参数：
        - token: 单个 token

        产出：
        - 第一个完整的 JSON 对象或数组
        """
        if self._completed:
            return iter(())

        self._buffer += token

        parsed = self._extract_first_json(self._buffer)
        if parsed is None:
            return iter(())

        self._completed = True
        self._buffer = ""
        yield parsed

    def flush(self) -> Iterator[dict[str, Any] | list[Any]]:
        """
        刷新缓冲区，尝试输出最后一个完整 JSON。

        若流式过程中已经解析到第一个完整 JSON，则这里不再输出内容。
        """
        if self._completed:
            return iter(())

        parsed = self._extract_first_json(self._buffer)
        if parsed is not None:
            self._completed = True
            self._buffer = ""
            yield parsed

        self._buffer = ""


class JsonLineParser(ResponseParser[dict[str, Any], list[dict[str, Any]]]):
    """
    JSON 行解析器（流式友好）

    每行一个 JSON 对象，支持流式解析。
    适用于 LLM 逐行输出 JSON 的场景。

    输出格式：
    ```
    {"t": "你好", "a": ["smile"]}
    {"t": "吗？", "a": ["nod"]}
    ```

    使用示例：
    ```python
    parser = JsonLineParser()

    # 流式解析
    for token in stream:
        for chunk in parser.stream_parse(token):
            print(chunk)  # 每个 chunk 是一个 JSON 对象

    # 完整解析
    chunks = parser.parse('{"t":"你好"}\n{"t":"吗？"}')
    # chunks = [{"t": "你好"}, {"t": "吗？"}]
    ```
    """

    def __init__(self, separator: str = "\n"):
        """
        初始化 JSON 行解析器

        参数：
        - separator: 行分隔符（默认换行符）
        """
        self._separator = separator
        self._buffer = ""

    def reset(self) -> None:
        """重置缓冲区"""
        self._buffer = ""

    def parse(self, content: str) -> list[dict[str, Any]]:
        """
        解析完整的 JSON 行内容

        参数：
        - content: 包含多行 JSON 的文本

        返回：
        - JSON 对象列表
        """
        results = []
        lines = content.split(self._separator)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return results

    def stream_parse(self, token: str) -> Iterator[dict[str, Any]]:
        """
        流式解析

        每接收到一个 token，检查缓冲区是否有完整的 JSON 行。
        如果有，解析并产出。

        参数：
        - token: 单个 token

        产出：
        - 解析完成的 JSON 对象
        """
        self._buffer += token

        # 检查是否有完整的行
        while self._separator in self._buffer:
            line, self._buffer = self._buffer.split(self._separator, 1)
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 不是有效的 JSON，跳过
                continue

    def flush(self) -> Iterator[dict[str, Any]]:
        """
        刷新缓冲区，处理剩余内容

        在流结束时调用，确保处理所有剩余数据。

        产出：
        - 缓冲区中解析完成的 JSON 对象
        """
        if self._buffer.strip():
            try:
                yield json.loads(self._buffer.strip())
            except json.JSONDecodeError:
                pass
        self._buffer = ""
