"""
V2+动作版本聊天模块

在 V1 基础上增加动作生成（使用 V2 生成器）。

核心特性：
1. 流式文本处理，逐句播放
2. TTS 并行合成
3. V2 动作生成器生成动作
4. 事件按句子 ID 聚合输出

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
│  3. 句子完成时触发动作生成                                          │
│     handle_text_result → create_motion_event                     │
├─────────────────────────────────────────────────────────────────┤
│  4. 事件循环：按序输出就绪事件                                      │
│     for payload in ctx.drain_ready_events():                     │
│         yield to_sse(payload)                                    │
└─────────────────────────────────────────────────────────────────┘

SSE 事件格式：
data: {"type": "text", "sentence_id": 1, "message": "你好呀~", ...}
data: {"type": "audio", "sentence_id": 1, "file": "base64...", ...}
data: {"type": "motion_frame", "sentence_id": 1, "motions": [...], ...}
"""

import time
import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from models.dto.request.chat_request import ChatRequest
from core.chat.multimodal_processor import build_user_message_content
from models.dto.response.ChatResponse import (
    DoneResponse,
    ErrorResponse,
    FullChatResponse,
    MotionResponse,
)
from my_utils.log import logger
from core.chat.base import store_sentence_event
from core.chat.v1 import BaseChatContext
from core.scheduler import TaskScheduler, TaskResult
from core.expression_generator.expression_generator_service_v2 import (
    ExpressionGeneratorV2,
)
from services.assistant_service import AssistantService


class V2MotionChatContext(BaseChatContext):
    """
    V2Motion 聊天上下文

    继承基础上下文，添加 V2 动作生成逻辑。

    属性：
    - motion_generator: V2 动作生成器实例
    - motion_semaphore: 动作生成并发控制信号量
    """

    def __init__(self, motion_generator: ExpressionGeneratorV2 | None):
        """
        初始化 V2Motion 聊天上下文

        参数：
        - motion_generator: V2 动作生成器实例
        """
        super().__init__(
            event_order=("text", "audio", "motion_frame"),
            tts_concurrency=1,
        )
        self.motion_generator = motion_generator
        self.motion_semaphore: asyncio.Semaphore = asyncio.Semaphore(4)

    async def create_motion_event(self, sentence_id: int, sentence_text: str):
        """
        创建动作事件

        使用 base.motion_wrapper 封装动作生成逻辑。

        参数：
        - sentence_id: 句子 ID
        - sentence_text: 句子文本
        """
        payload = await motion_wrapper(
            motion_semaphore=self.motion_semaphore,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            motion_generator=self.motion_generator,
            user_message=self.user_message,
            assistant_message=self.full_text_list,
        )
        store_sentence_event(self.sentence_events, sentence_id, "motion_frame", payload)

    async def handle_result(self, result: TaskResult):
        """
        处理文本结果，生成文本、音频和动作事件

        调用链：
        1. super().handle_text_result() → text_wrapper + tts_wrapper + store_sentence_event
        2. create_motion_event → motion_wrapper + store_sentence_event

        参数：
        - result: 文本任务结果
        """
        # 调用基类方法处理文本和音频事件
        await super().handle_result(result)

        # 获取句子信息用于动作生成
        sentence_id = result.sentence_id
        sentence_text = result.data["text"]

        # 创建动作事件
        self.track_task(
            asyncio.create_task(self.create_motion_event(sentence_id, sentence_text))
        )


async def motion_wrapper(
    motion_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    motion_generator: Any,
    user_message: str,
    assistant_message: list[str],
    motion_history: dict | None = None,
) -> MotionResponse:
    """
    V2 版本动作规划任务封装

    参数：
    - motion_semaphore: 并发控制信号量
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    - motion_generator: V2 动作生成器实例
    - user_message: 用户消息
    - assistant_message: 助手消息历史
    - motion_history: 历史动作状态字典

    返回：
    - 动作事件字典
    """
    logger.info(
        f"[动作生成] 准备生成动作，句子ID: {sentence_id}, 内容: {sentence_text}"
    )

    motion_event: MotionResponse = MotionResponse(
        sentence_id=sentence_id,
        source_text=sentence_text,
        motions=[],
        duration=0,
    )

    async with motion_semaphore:
        try:
            if motion_generator is None:
                logger.warning(
                    f"[动作生成] 动作生成器不可用，发送空动作事件，sentence_id: {sentence_id}"
                )
                return motion_event

            # 获取前一个动作参数（用于连贯性）
            previous_params = None
            if motion_history and sentence_id > 0:
                if sentence_id - 1 in motion_history:
                    previous_params = motion_history[sentence_id - 1]

            # 构建对话上下文信息
            context = (
                "用户："
                + user_message
                + "\n。助手的上文回复（已经生成完成动作的回复）："
                + "".join(assistant_message[:-1])
                + "\n。当前需要生成动作的回复："
                + sentence_text
            )
            # 清理文本
            sentence_text_clean = sentence_text.replace(" ", "").replace("\n", "")
            duration_time = len(sentence_text_clean) * 100

            # 使用 V2 生成器单次生成动作
            frames = await motion_generator.generate_tts_motion(
                speech_text=sentence_text,
                speech_duration_ms=duration_time,
                previous_params=previous_params,
                context=context,
                timeout_seconds=10.0,
            )

            motion_event.duration = duration_time

            # 转换帧格式为前端可用的格式
            motions = []
            for frame in frames:
                motion_data = {
                    "duration": frame.duration,
                    "parameters": frame.parameters,
                }
                if frame.expression:
                    motion_data["expression"] = frame.expression
                motions.append(motion_data)

            motion_event.motions = motions

            # 记录动作参数到历史（用于连贯性）
            if motion_history is not None and frames:
                last_frame = frames[-1]
                motion_history[sentence_id] = last_frame.parameters.copy()

            logger.info(
                f"[动作生成] 生成 {len(motions)} 个动作帧，句子ID: {sentence_id}"
            )

        except Exception as e:
            logger.warning(f"[动作生成] 第{sentence_id + 1}句动作生成失败: {e}")
        return motion_event


class V2ChatService:
    """
    V2 聊天服务

    提供 V2 版本的聊天流式输出接口（文本+语音+动作）。
    """

    def __init__(self):
        """
        初始化 V2 聊天服务
        """
        self.assistant_service = AssistantService()
        self.motion_generator: ExpressionGeneratorV2 | None = None

    async def _init_motion_generator(
        self, agent_name: str
    ) -> ExpressionGeneratorV2 | None:
        """
        初始化 V2 动作生成器

        参数：
        - agent_name: 助手名称

        返回：
        - 动作生成器实例，失败返回 None
        """
        try:
            generator = ExpressionGeneratorV2()
            await generator.initialize(agent_name)
            return generator
        except Exception as e:
            logger.warning(f"[V2Motion] 获取动作生成器失败: {e}")
            return None

    async def chat(self, params: ChatRequest) -> AsyncGenerator[FullChatResponse]:
        """
        V2 版本聊天流式输出

        并行文字/语音/动作，极低延迟同步。

        流程概述：
        1. 获取助手实例，初始化动作生成器
        2. 创建调度器和纯文本管道
        3. 流式执行管道，处理文本和动作结果
        4. 事件循环：按序输出就绪事件

        参数：
        - params: 聊天请求参数
        """
        agent = self.assistant_service.get_current_assistant()

        if not agent:
            logger.error("[V2Motion] 当前没有加载助手")
            yield ErrorResponse(error_code="NO_ASSISTANT", data="当前没有加载助手")

            return

        self.motion_generator = await self._init_motion_generator(agent.agent_name)

        if self.motion_generator is None:
            logger.warning("[动作规划] 未找到可用 Live2D 参数，将输出空动作事件")

        ctx = V2MotionChatContext(self.motion_generator)

        # 从 ChatRequest 构建用户消息内容
        user_message_raw, user_text = build_user_message_content(params)
        ctx.user_message = user_text

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
            # 流式执行管道
            async for result in pipeline.execute():
                await ctx.handle_result(result)

                for payload in ctx.drain_ready_events():
                    yield payload

            # 等待所有异步任务完成
            await ctx.wait_for_completion()

            # 输出剩余事件
            for payload in ctx.drain_ready_events():
                yield payload

            # 输出完成信号
            yield DoneResponse(full_text=ctx.get_full_text())

            # 追加到上下文（user_message_raw 已是 OpenAI 原生格式）
            agent.chat_history.extend(user_message_raw)
            agent.chat_history.append({"role": "assistant", "content": ctx.get_full_text()})
            # 持久化 + 好感度
            asyncio.create_task(
                agent.add_msg(
                    user_msg=user_text,
                    assistant_msg=ctx.get_full_text(),
                )
            )

        except Exception as e:
            # 取消所有任务
            for task in list(ctx.pending_tasks):
                task.cancel()

            logger.error(f"处理数据时出错: {e}", exc_info=True)
            yield ErrorResponse(error_code="500", data=f"处理数据时出错: {e}")
