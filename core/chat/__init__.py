"""
聊天模块

提供不同版本的聊天实现。

架构说明：
所有版本统一使用 TaskScheduler + Pipeline 模式。
- TaskScheduler: 调度器，负责创建管道
- Pipeline: 流式处理管道，负责执行 LLM 调用
- V3MotionChatContext: 聊天上下文，负责事件处理

使用示例：
```python
from core.chat import V3ChatService

v3_service = V3ChatService()
async for sse_event in v3_service.chat(params):
    send_sse(sse_event)
```
"""

from core.chat.base import (
    TTSData,
    tts_task,
    tts_wrapper,
)

from core.chat.v3_motion import V3ChatService

__all__ = [
    "TTSData",
    "tts_task",
    "tts_wrapper",
    "V3ChatService",
]
