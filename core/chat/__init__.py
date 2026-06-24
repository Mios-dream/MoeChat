"""
聊天模块

提供不同版本的聊天实现。

版本说明：
- V1: 基础版本（文本 + TTS）
- V2: 带动作版本（使用 V2 生成器）
- V3: 带动作版本（使用 V3 生成器）
- V4: 信息调度中心版本（推荐）

使用示例：
```python
from core.chat import llm_chat_with_tts_and_motion_v4

# V4 版本
async for event in llm_chat_with_tts_and_motion_v4(params):
    send_sse(event)
```
"""

from core.chat.base import (
    # 数据类
    TTSData,
    # 流式处理器
    StreamProcessor,
    # TTS 任务
    tts_task,
    # 事件封装
    text_wrapper,
    tts_wrapper,
    # 事件聚合
    store_sentence_event,
    drain_ready_sentence_events,
    to_sse,
    # LLM 处理
    start_llm_task,
    # 服务实例
    assistant_service,
)

from core.chat.v1 import llm_chat_with_tts
from core.chat.v2_motion import llm_chat_with_tts_and_motion_v2
from core.chat.v3_motion import llm_chat_with_tts_and_motion_v3
from core.chat.v4_motion import llm_chat_with_tts_and_motion_v4

__all__ = [
    # 基础组件
    "TTSData",
    "StreamProcessor",
    "tts_task",
    "text_wrapper",
    "tts_wrapper",
    "store_sentence_event",
    "drain_ready_sentence_events",
    "to_sse",
    "start_llm_task",
    "assistant_service",
    # V2 版本
    "llm_chat_with_tts",
    # V2Motion 版本
    "llm_chat_with_tts_and_motion_v2",
    # V3 版本
    "llm_chat_with_tts_and_motion_v3",
    # V4 版本
    "llm_chat_with_tts_and_motion_v4",
]
