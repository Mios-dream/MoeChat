"""
纯文本流解析器

将 LLM 的纯文本流式输出按句子分割，产出 TaskResult。

核心功能：
1. 接收纯文本 token 流
2. 检测句子边界（句尾标点、括号完整性）
3. 按句子产出 TaskResult

特性：
- 支持括号完整性检测
- 支持跨 token 的句子边界检测
- 每个句子有唯一 ID
- 支持 TTS 文本过滤（移除括号内容）

使用示例：
```python
parser = TextStreamParser()

# 流式解析
for token in stream:
    for result in parser.stream_parse(token):
        print(result.task_type, result.data)

# 处理剩余文本
for result in parser.flush():
    print(result.task_type, result.data)
```
"""

from collections.abc import Generator
import time
from core.scheduler.task import TaskResult

# ============================================================
# 常量定义
# ============================================================

# 句尾标点符号模式
SENTENCE_END_PATTERNS = ("。", "！", "？", "……\n", "……", "...\n", "...")

# 最小句子长度（避免太短的句子触发 TTS）
MIN_SENTENCE_LENGTH = 6

# 括号配对映射
BRACKET_PAIRS = {
    "(": ")",
    "（": "）",
    "[": "]",
    "【": "】",
    "{": "}",
}

# 开括号集合
OPENING_BRACKETS = set(BRACKET_PAIRS.keys())


class TextStreamParser:
    """
    纯文本流解析器

    将 LLM 的纯文本流式输出按句子分割，产出 TaskResult。

    特性：
    - 支持括号完整性检测
    - 支持跨 token 的句子边界检测
    - 每个句子有唯一 ID
    - 支持 TTS 文本过滤（移除括号内容）

    使用示例：
    ```python
    parser = TextStreamParser()

    # 流式解析
    for token in stream:
        for result in parser.stream_parse(token):
            print(result.task_type, result.data)

    # 处理剩余文本
    for result in parser.flush():
        print(result.task_type, result.data)
    ```
    """

    def __init__(self, task_type: str = "text", task_name: str = "text_generation"):
        """
        初始化纯文本流解析器

        参数：
        - task_type: 任务类型（用于 TaskResult）
        - task_name: 任务名称（用于 TaskResult）
        """
        self._task_type = task_type
        self._task_name = task_name

        # 普通文本缓存（不包含括号段）
        self._sentence_buffer = ""

        # 括号段缓存：括号内容单独成段，优先保证完整
        self._bracket_buffer = ""
        self._segment_bracket_stack: list[str] = []

        # 句子计数器
        self._sentence_counter = 0

        # TTS 文本过滤用的括号栈
        self._tts_bracket_stack: list[str] = []

    def reset(self) -> None:
        """重置解析器状态"""
        self._sentence_buffer = ""
        self._bracket_buffer = ""
        self._segment_bracket_stack = []
        self._sentence_counter = 0
        self._tts_bracket_stack = []

    @staticmethod
    def _is_sentence_end_at(text: str, idx: int) -> int:
        """
        检测指定位置是否为句尾

        参数：
        - text: 待检测文本
        - idx: 检测位置索引

        返回：
        - 命中长度（0 表示未命中）
        """
        for pattern in sorted(SENTENCE_END_PATTERNS, key=len, reverse=True):
            if text.startswith(pattern, idx):
                return len(pattern)
        return 0

    def _extract_plain_segments(self, force_flush: bool = False) -> list[str]:
        """
        从普通文本缓存中提取可输出片段

        参数：
        - force_flush: 是否强制提取剩余文本

        返回：
        - 可输出的文本片段列表
        """
        ready: list[str] = []

        while True:
            boundary_end = -1
            i = 0
            while i < len(self._sentence_buffer):
                hit_len = self._is_sentence_end_at(self._sentence_buffer, i)
                if hit_len > 0:
                    boundary_end = i + hit_len
                    candidate = self._sentence_buffer[:boundary_end]
                    cleaned = "".join(candidate.split())
                    if len(cleaned) >= MIN_SENTENCE_LENGTH:
                        break
                    i = boundary_end
                    boundary_end = -1
                    continue
                i += 1

            if boundary_end < 0:
                break

            candidate = self._sentence_buffer[:boundary_end]
            ready.append(candidate)
            self._sentence_buffer = self._sentence_buffer[boundary_end:]

        if force_flush and self._sentence_buffer.strip():
            ready.append(self._sentence_buffer)
            self._sentence_buffer = ""

        return ready

    def _filter_tts_text(self, text: str) -> str:
        """
        过滤供 TTS 使用的文本

        移除括号及其内部内容（支持跨句/跨片段）

        参数：
        - text: 原始文本

        返回：
        - 过滤后的文本
        """
        filtered_chars: list[str] = []

        for ch in text:
            if self._tts_bracket_stack:
                if ch in OPENING_BRACKETS:
                    self._tts_bracket_stack.append(BRACKET_PAIRS[ch])
                elif ch == self._tts_bracket_stack[-1]:
                    self._tts_bracket_stack.pop()
                continue

            if ch in OPENING_BRACKETS:
                self._tts_bracket_stack.append(BRACKET_PAIRS[ch])
                continue

            filtered_chars.append(ch)

        return "".join(filtered_chars)

    def _create_result(self, sentence_text: str) -> TaskResult:
        """
        创建句子结果

        参数：
        - sentence_text: 句子文本

        返回：
        - TaskResult 实例
        """
        self._sentence_counter += 1

        # 清理文本
        cleaned = "".join(sentence_text.split())
        # 过滤 TTS 文本
        tts_text = "".join(self._filter_tts_text(sentence_text).split())

        return TaskResult(
            task_name=self._task_name,
            task_type=self._task_type,
            data={
                "text": cleaned,
                "tts_text": tts_text,
            },
            raw_data={"text": cleaned},
            sentence_id=self._sentence_counter,
            timestamp=time.time(),
        )

    def _emit_segment(self, message_chunk: str) -> Generator[TaskResult, None, None]:
        """
        输出一个片段并产出结果

        参数：
        - message_chunk: 待输出的完整单句文本片段

        产出：
        - TaskResult 实例
        """
        if not message_chunk:
            return

        yield self._create_result(message_chunk)

    def stream_parse(self, token: str) -> Generator[TaskResult, None, None]:
        """
        流式解析

        每接收到一个 token，检测完整句子并产出结果。

        参数：
        - token: 单个 token

        产出：
        - TaskResult 实例
        """
        for ch in token:
            if self._segment_bracket_stack:
                self._bracket_buffer += ch
                if ch in OPENING_BRACKETS:
                    self._segment_bracket_stack.append(BRACKET_PAIRS[ch])
                elif ch == self._segment_bracket_stack[-1]:
                    self._segment_bracket_stack.pop()
                    if not self._segment_bracket_stack:
                        yield from self._emit_segment(self._bracket_buffer)
                        self._bracket_buffer = ""
                continue

            if ch in OPENING_BRACKETS:
                for plain_segment in self._extract_plain_segments(force_flush=False):
                    yield from self._emit_segment(plain_segment)
                self._segment_bracket_stack.append(BRACKET_PAIRS[ch])
                self._bracket_buffer = ch
                continue

            self._sentence_buffer += ch
            for plain_segment in self._extract_plain_segments(force_flush=False):
                yield from self._emit_segment(plain_segment)

    def flush(self) -> Generator[TaskResult, None, None]:
        """
        刷新缓冲区，处理剩余内容

        在流结束时调用，确保处理所有剩余数据。

        产出：
        - 缓冲区中解析完成的 TaskResult
        """
        for plain_segment in self._extract_plain_segments(force_flush=False):
            yield from self._emit_segment(plain_segment)

        for plain_segment in self._extract_plain_segments(force_flush=True):
            yield from self._emit_segment(plain_segment)

        if self._bracket_buffer.strip():
            yield from self._emit_segment(self._bracket_buffer)
            self._bracket_buffer = ""
            self._segment_bracket_stack = []

    @property
    def sentence_count(self) -> int:
        """已处理的句子数量"""
        return self._sentence_counter
