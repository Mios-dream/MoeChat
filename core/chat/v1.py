"""
V1 版本聊天上下文

基础版本：文本 + TTS，不包含动作生成。

架构流程与 V3 一致：handle_result 缓存 → process_all 按序处理 TTS → 队列输出。
"""

import asyncio
from collections.abc import AsyncGenerator
from models.dto.request.chat_request import ChatRequest
from core.chat.multimodal_processor import build_user_message_content
from models.dto.response.ChatResponse import (
    AssistantMessage,
    FullChatResponse,
    ErrorMessage,
    DoneMessage,
)
from my_utils.log import logger
from core.chat.base import tts_wrapper
from core.scheduler import TaskResult
from core.scheduler import TaskScheduler
from core.scheduler.parsers.text_stream_parser import filter_tts_text
from services.assistant_service import AssistantService


class BaseChatContext:
    """
    基础聊天上下文

    所有聊天版本的公共基类，提供：
    - text_cache / _event_buffer: 缓存 + 队列输出（参照 V3 模式）
    - tts_semaphore: TTS 并发控制
    - pending_tasks: 异步任务跟踪（供子类取消用）
    """

    def __init__(self, tts_concurrency: int = 1):
        self.tts_semaphore: asyncio.Semaphore = asyncio.Semaphore(tts_concurrency)
        self.pending_tasks: set[asyncio.Task] = set()
        self.full_text_list: list[str] = []
        self.text_cache: dict[int, str] = {}
        self._event_buffer: list[FullChatResponse] = []

    async def handle_result(self, result: TaskResult):
        """
        缓存文本结果

        参数：
        - result: 任务结果，data 格式为 {"text": ..., "tts_text": ...}
        """
        sentence_id = result.sentence_id
        sentence_text = result.data["text"]
        self.text_cache[sentence_id] = sentence_text
        self.full_text_list.append(sentence_text)

    async def process_all(self):
        """
        按 sentence_id 顺序依次处理每个句子：
        TTS 合成 → 合并为 AssistantMessage 发出
        """
        for sid in sorted(self.text_cache.keys()):
            text = self.text_cache[sid]

            audio_file: str | None = None
            tts_text = filter_tts_text(text)
            if tts_text.strip():
                audio_file = await tts_wrapper(
                    self.tts_semaphore, text, tts_text
                )

            extras = {}
            if audio_file:
                extras["audio"] = audio_file

            self._event_buffer.append(
                AssistantMessage(content=text, extras=extras or None)
            )

    def drain_ready_events(self) -> list[FullChatResponse]:
        """排出所有缓冲事件"""
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return events

    def get_full_text(self) -> str:
        """获取完整文本"""
        return "".join(self.full_text_list)


class V1ChatService:
    """
    V1 聊天服务

    提供 V1 版本的聊天流式输出接口（文本+语音，不含动作）。
    """

    def __init__(self):
        """
        初始化 V1 聊天服务
        """
        self.assistant_service = AssistantService()

    async def chat(self, params: ChatRequest) -> AsyncGenerator[FullChatResponse, None]:
        """
        V1 版本聊天流式输出

        并行文字/语音，极低延迟同步。

        流程概述：
        1. 获取助手实例，构建历史消息
        2. 创建调度器和纯文本管道
        3. 流式执行管道，处理文本结果
        4. 事件循环：按序输出就绪事件

        参数：
        - params: 聊天请求参数
        """
        agent = self.assistant_service.get_current_assistant()

        if not agent:
            logger.error("当前没有加载助手")
            yield ErrorMessage(error_code="NO_ASSISTANT", data="当前没有加载助手")

            return

        ctx = BaseChatContext()

        # 从 ChatRequest 构建用户消息内容
        user_message_raw, user_text = build_user_message_content(params)

        raw_history = agent.get_history()
        history_messages = [
            {
                "role": "system",
                "content": await agent.get_context(
                    msg=user_text, is_sleep_mode=params.is_sleep_mode
                ),
            },
            *raw_history,
        ]

        scheduler = TaskScheduler()
        pipeline = scheduler.create_text_pipeline(
            system_context=agent.prompt,
            history_messages=history_messages,
            user_message=user_message_raw,
        )

        try:
            async for result in pipeline.execute():
                await ctx.handle_result(result)

            await ctx.process_all()
            for payload in ctx.drain_ready_events():
                yield payload

            yield DoneMessage(full_text=ctx.get_full_text())

            agent.chat_history.extend(user_message_raw)
            agent.chat_history.append(
                {"role": "assistant", "content": ctx.get_full_text()}
            )
            asyncio.create_task(
                agent.add_msg(
                    user_msg=user_text,
                    assistant_msg=ctx.get_full_text(),
                )
            )

        except Exception as e:
            for task in list(ctx.pending_tasks):
                task.cancel()

            logger.error(f"处理数据时出错: {e}", exc_info=True)
            yield ErrorMessage(error_code="500", data=f"处理数据时出错: {e}")
