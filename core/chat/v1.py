"""
V2 版本聊天模块

基础版本：文本 + TTS，不包含动作生成。

核心特性：
1. 流式文本处理，逐句播放
2. TTS 并行合成
3. 事件按句子 ID 聚合输出

架构流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建流处理器，注册回调                                          │
│     processor = StreamProcessor()                                │
│     processor.register_sentence_complete_callback(on_complete)   │
├─────────────────────────────────────────────────────────────────┤
│  2. 启动 LLM 流式任务                                              │
│     llm_task = asyncio.create_task(start_llm_task(msg, proc))   │
├─────────────────────────────────────────────────────────────────┤
│  3. 句子完成时触发 TTS                                              │
│     on_complete → create_text_event + create_audio_event         │
├─────────────────────────────────────────────────────────────────┤
│  4. 事件循环：按序输出就绪事件                                       │
│     while True: drain_ready → yield SSE                          │
└─────────────────────────────────────────────────────────────────┘

SSE 事件格式：
data: {"type": "text", "sentence_id": 1, "message": "你好呀~", ...}
data: {"type": "audio", "sentence_id": 1, "file": "base64...", ...}
"""

import time
import asyncio
from typing import Any

from models.dto.chat_request import chat_data
from my_utils.log import logger
from core.chat.base import (
    StreamProcessor,
    text_wrapper,
    tts_wrapper,
    start_llm_task,
    store_sentence_event,
    drain_ready_sentence_events,
    to_sse,
    build_chat_messages,
)
from services.assistant_service import AssistantService

assistant_service = AssistantService()


class V1ChatContext:
    """
    V1 聊天上下文

    封装聊天过程中的所有状态，避免函数嵌套。

    属性：
    - processor: 流处理器实例
    - sentence_events: 按句子 ID 聚合的事件存储
    - expected_sentence_id: 预期的下一个句子 ID
    - pending_tasks: 异步任务跟踪集合
    - tts_semaphore: TTS 并发控制信号量
    """

    def __init__(self, processor: StreamProcessor):
        """
        初始化 V2 聊天上下文

        参数：
        - processor: 流处理器实例
        """
        self.processor = processor
        self.sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
        self.expected_sentence_id: int = 1
        self.pending_tasks: set[asyncio.Task[Any]] = set()
        self.tts_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

        # 事件类型顺序
        self.event_order: tuple[str, ...] = ("text", "audio")

    def track_task(self, task: asyncio.Task[Any]):
        """
        跟踪异步任务

        参数：
        - task: 要跟踪的异步任务
        """
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)

    def drain_ready_events(self) -> list[dict[str, Any]]:
        """
        排出所有就绪的句子事件

        返回：
        - 就绪的事件载荷列表
        """
        self.expected_sentence_id, ready_payloads = drain_ready_sentence_events(
            sentence_events=self.sentence_events,
            expected_sentence_id=self.expected_sentence_id,
            event_order=self.event_order,
        )
        return ready_payloads


# ============================================================
# 事件处理函数
# ============================================================


async def _create_text_event(ctx: V1ChatContext, sentence_id: int, sentence_text: str):
    """
    创建文本事件

    参数：
    - ctx: V2 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    """
    payload = await text_wrapper(
        sentence_id=sentence_id,
        sentence_text=sentence_text,
    )
    store_sentence_event(ctx.sentence_events, sentence_id, "text", payload)


async def _create_audio_event(
    ctx: V1ChatContext,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
):
    """
    创建音频事件

    参数：
    - ctx: V2 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    - tts_text: 用于 TTS 的文本
    """
    payload = await tts_wrapper(
        tts_semaphore=ctx.tts_semaphore,
        sentence_id=sentence_id,
        sentence_text=sentence_text,
        tts_text=tts_text,
        processor=ctx.processor,
    )
    store_sentence_event(ctx.sentence_events, sentence_id, "audio", payload)


async def _on_sentence_complete(
    ctx: V1ChatContext,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
):
    """
    句子完成时的回调

    参数：
    - ctx: V2 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    - tts_text: 用于 TTS 的文本
    """
    ctx.track_task(
        asyncio.create_task(_create_text_event(ctx, sentence_id, sentence_text))
    )
    ctx.track_task(
        asyncio.create_task(
            _create_audio_event(ctx, sentence_id, sentence_text, tts_text)
        )
    )


# ============================================================
# V2 主函数
# ============================================================


async def llm_chat_with_tts(params: chat_data):
    """
    V1 版本聊天流式输出

    并行文字/语音，极低延迟同步。

    流程概述：
    1. 创建流处理器，注册句子完成回调
    2. 启动 LLM 流式任务
    3. 句子完成时触发 TTS 合成
    4. 事件循环：按序输出就绪事件

    参数：
    - params: 聊天请求参数

    产出：
    - SSE 格式的事件流
    """
    # 第一步：获取助手实例
    agent = assistant_service.get_current_assistant()

    if not agent:
        logger.error("当前没有加载助手")
        return

    # 第二步：构建消息列表（使用统一函数）
    agent, history_messages, user_message = await build_chat_messages(agent, params)

    # 第三步：初始化上下文
    processor = StreamProcessor()
    ctx = V1ChatContext(processor)

    # 注册句子完成回调
    processor.register_sentence_complete_callback(
        lambda sid, text, tts: _on_sentence_complete(ctx, sid, text, tts)
    )

    # 第四步：启动 LLM 任务
    llm_task = asyncio.create_task(start_llm_task(history_messages, processor))

    try:
        # 第五步：事件循环
        while True:
            # 输出就绪的句子事件
            for payload in ctx.drain_ready_events():
                yield to_sse(payload)

            # 检查是否完成
            if llm_task.done() and not ctx.pending_tasks and not ctx.sentence_events:
                break

            await asyncio.sleep(0.01)

        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": max(0, processor.sentence_count - 1),
            "full_text": "".join(processor.full_msg) if processor.full_msg else "",
            "done": True,
        }
        yield to_sse(final_response)

        # 保存到助手上下文（需要先添加用户消息到临时列表）
        user_message = params.msg[-1]["content"]
        agent.msg_data_tmp.append({"role": "user", "content": user_message})
        await agent.add_msg("".join(processor.full_msg))

    except Exception as e:
        # 取消所有任务
        llm_task.cancel()
        for task in list(ctx.pending_tasks):
            task.cancel()

        logger.error(f"处理数据时出错: {e}", exc_info=True)
        yield to_sse(
            {
                "type": "error",
                "timestamp_ms": time.time() * 1000,
                "data": str(e),
                "done": True,
            }
        )
