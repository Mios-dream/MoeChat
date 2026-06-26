"""
V1 版本聊天模块

基础版本：文本 + TTS，不包含动作生成。

核心特性：
1. 流式文本处理，逐句播放
2. TTS 并行合成
3. 事件按句子 ID 聚合输出

架构流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建调度器和纯文本管道                                         │
│     scheduler = TaskScheduler()                                  │
│     pipeline = scheduler.create_text_pipeline(messages)          │
├─────────────────────────────────────────────────────────────────┤
│  2. 流式执行管道，处理文本结果                                      │
│     async for result in pipeline.execute():                      │
│         ctx.handle_text_result(result)                           │
├─────────────────────────────────────────────────────────────────┤
│  3. 事件循环：按序输出就绪事件                                      │
│     for payload in ctx.drain_ready_events():                     │
│         yield to_sse(payload)                                    │
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
    to_sse,
    text_wrapper,
    tts_wrapper,
    store_sentence_event,
    drain_ready_sentence_events,
)
from core.scheduler import TaskResult
from core.scheduler import TaskScheduler
from services.assistant_service import AssistantService

assistant_service = AssistantService()


class BaseChatContext:
    """
    基础聊天上下文

    封装聊天过程中的公共状态和事件处理逻辑。

    属性：
    - sentence_events: 按句子 ID 聚合的事件存储
    - expected_sentence_id: 预期的下一个句子 ID
    - pending_tasks: 异步任务跟踪集合
    - tts_semaphore: TTS 并发控制信号量
    - event_order: 事件类型顺序
    - full_text_list: 收集完整文本
    - user_message: 用户消息
    """

    def __init__(
        self,
        event_order: tuple[str, ...] = ("text", "audio"),
        tts_concurrency: int = 1,
    ):
        """
        初始化基础聊天上下文

        参数：
        - event_order: 事件类型顺序
        - tts_concurrency: TTS 并发数
        """
        self.sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
        self.expected_sentence_id: int = 1
        self.pending_tasks: set[asyncio.Task[Any]] = set()
        self.tts_semaphore: asyncio.Semaphore = asyncio.Semaphore(tts_concurrency)
        self.event_order: tuple[str, ...] = event_order

        # 收集完整文本
        self.full_text_list: list[str] = []
        # 用户消息
        self.user_message: str = ""

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

    async def create_text_event(self, sentence_id: int, sentence_text: str):
        """
        创建文本事件

        参数：
        - sentence_id: 句子 ID
        - sentence_text: 句子文本
        """
        payload = await text_wrapper(
            sentence_id=sentence_id,
            sentence_text=sentence_text,
        )
        store_sentence_event(self.sentence_events, sentence_id, "text", payload)

    async def create_audio_event(
        self, sentence_id: int, sentence_text: str, tts_text: str
    ):
        """
        创建音频事件（使用 tts_wrapper，包含情感检测）

        参数：
        - sentence_id: 句子 ID
        - sentence_text: 句子文本
        - tts_text: 用于 TTS 的文本
        """
        payload = await tts_wrapper(
            tts_semaphore=self.tts_semaphore,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            tts_text=tts_text,
        )
        store_sentence_event(self.sentence_events, sentence_id, "audio", payload)

    async def handle_text_result(self, result: TaskResult):
        """
        处理文本结果（V1 模式：使用 create_text_pipeline）

        参数：
        - result: 文本任务结果，data 格式为 {"text": ..., "tts_text": ...}
        """
        sentence_id = result.sentence_id
        sentence_text = result.data["text"]
        tts_text = result.data["tts_text"]

        # 收集完整文本
        self.full_text_list.append(sentence_text)

        # 创建文本和音频事件
        self.track_task(
            asyncio.create_task(self.create_text_event(sentence_id, sentence_text))
        )
        self.track_task(
            asyncio.create_task(
                self.create_audio_event(sentence_id, sentence_text, tts_text)
            )
        )

    def get_full_text(self) -> str:
        """获取完整文本"""
        return "".join(self.full_text_list)

    async def wait_for_completion(self):
        """等待所有异步任务完成"""
        while self.pending_tasks:
            await asyncio.sleep(0.01)


async def llm_chat_with_tts(params: chat_data):
    """
    V1 版本聊天流式输出

    并行文字/语音，极低延迟同步。

    流程概述：
    1. 构建消息列表
    2. 创建调度器和纯文本管道
    3. 流式执行管道，处理文本结果
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

    # 第三步：初始化上下文（使用 chat_context.BaseChatContext）
    ctx = BaseChatContext()
    ctx.user_message = params.msg[-1]["content"] if params.msg else ""

    # print(ctx.user_message, params.is_sleep_mode)

    # 获取历史消息（包含上下文）
    history_messages = [
        *agent.get_history(),
        {
            "role": "user",
            "content": await agent.get_context(
                msg=ctx.user_message, is_sleep_mode=params.is_sleep_mode
            ),
        },
    ]

    # 第四步：创建调度器和纯文本管道
    scheduler = TaskScheduler()
    pipeline = scheduler.create_text_pipeline(
        system_context=agent.prompt,
        history_messages=history_messages,
        user_message=ctx.user_message,
    )

    try:
        # 第五步：流式执行管道
        async for result in pipeline.execute():
            await ctx.handle_text_result(result)

            # drain_ready_events 内部调用 base.drain_ready_sentence_events
            for payload in ctx.drain_ready_events():
                yield to_sse(payload)

        # 等待所有异步任务完成
        await ctx.wait_for_completion()

        # 输出剩余事件
        for payload in ctx.drain_ready_events():
            yield to_sse(payload)

        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": pipeline.sentence_count,
            "full_text": ctx.get_full_text(),
            "done": True,
        }
        yield to_sse(final_response)

        # 保存到助手上下文
        asyncio.create_task(
            agent.add_msg(
                user_msg=ctx.user_message,
                assistant_msg=ctx.get_full_text(),
            )
        )

    except Exception as e:
        # 取消所有任务
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
