"""
聊天模块

提供不同版本的聊天实现。

版本说明：
- V1: 基础版本（文本 + TTS）
- V2: 带动作版本（使用 V2 生成器）
- V3: 信息调度中心版本（推荐）

架构说明：
所有版本统一使用 TaskScheduler + Pipeline 模式。
- TaskScheduler: 调度器，负责创建管道
- Pipeline: 流式处理管道，负责执行 LLM 调用
- BaseChatContext: 基础聊天上下文，负责事件处理

使用示例：
```python
from core.chat import V1ChatService, V3ChatService

# V1 版本（文本 + TTS）
v1_service = V1ChatService()
async for sse_event in v1_service.chat(params):
    send_sse(sse_event)

# V3 版本（文本 + TTS + 动作 + 工具调用）
v3_service = V3ChatService()
async for sse_event in v3_service.chat(params):
    send_sse(sse_event)
```
"""

from core.chat.base import (
    # 数据类
    TTSData,
    # TTS 任务
    tts_task,
    # 事件封装
    text_wrapper,
    tts_wrapper,
    # 事件聚合
    store_sentence_event,
    drain_ready_sentence_events,
)

from core.chat.v1 import BaseChatContext, V1ChatService
from core.chat.v2_motion import V2ChatService
from core.chat.v3_motion import V3ChatService

__all__ = [
    # 基础组件
    "TTSData",
    "tts_task",
    "text_wrapper",
    "tts_wrapper",
    "store_sentence_event",
    "drain_ready_sentence_events",
    # 上下文
    "BaseChatContext",
    # V1 版本
    "V1ChatService",
    # V2Motion 版本
    "V2ChatService",
    # V3 版本
    "V3ChatService",
]
