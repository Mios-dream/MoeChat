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
from core.chat import llm_chat_with_tts_and_motion_v3

# V3 版本
async for event in llm_chat_with_tts_and_motion_v3(params):
    send_sse(event)
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
    to_sse,
)

from core.chat.v1 import BaseChatContext, llm_chat_with_tts
from core.chat.v2_motion import llm_chat_with_tts_and_motion_v2
from core.chat.v3_motion import V3MotionChatContext, llm_chat_with_tts_and_motion_v3

__all__ = [
    # 基础组件
    "TTSData",
    "tts_task",
    "text_wrapper",
    "tts_wrapper",
    "store_sentence_event",
    "drain_ready_sentence_events",
    "to_sse",
    # 上下文
    "BaseChatContext",
    "V3MotionChatContext",
    # V1 版本
    "llm_chat_with_tts",
    # V2Motion 版本
    "llm_chat_with_tts_and_motion_v2",
    # V3 版本
    "llm_chat_with_tts_and_motion_v3",
]
