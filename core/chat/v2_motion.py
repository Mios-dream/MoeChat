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
from typing import Any
from models.dto.chat_request import chat_data
from my_utils.log import logger
from core.chat.base import store_sentence_event, to_sse
from core.chat.v1 import BaseChatContext
from core.scheduler import TaskScheduler, TaskResult
from core.expression_generator.expression_generator_service_v2 import (
    ExpressionGeneratorV2,
)
from services.assistant_service import AssistantService

assistant_service = AssistantService()


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

    async def handle_text_result(self, result: TaskResult):
        """
        处理文本结果，生成文本、音频和动作事件

        调用链：
        1. super().handle_text_result() → text_wrapper + tts_wrapper + store_sentence_event
        2. create_motion_event → motion_wrapper + store_sentence_event

        参数：
        - result: 文本任务结果
        """
        # 调用基类方法处理文本和音频事件
        await super().handle_text_result(result)

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
) -> dict[str, Any]:
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

    current_time_ms = time.time() * 1000
    motion_event: dict[str, Any] = {
        "type": "motion_frame",
        "sentence_id": sentence_id,
        "source_text": sentence_text,
        "motions": [],
        "duration": 0,
        "timestamp_ms": current_time_ms,
        "done": False,
    }

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

            motion_event["duration"] = duration_time

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

            motion_event["motions"] = motions

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


async def llm_chat_with_tts_and_motion_v2(params: chat_data):
    """
    V2Motion 版本聊天流式输出

    并行文字/语音/动作，极低延迟同步。

    流程概述：
    1. 构建消息列表（包含动作提示词）
    2. 获取 V2 动作生成器
    3. 创建调度器和纯文本管道
    4. 流式执行管道，处理文本和动作结果
    5. 事件循环：按序输出就绪事件

    参数：
    - params: 聊天请求参数

    产出：
    - SSE 格式的事件流
    """
    # ============================================================
    # 第一步：获取助手实例
    # ============================================================
    agent = assistant_service.get_current_assistant()

    if not agent:
        logger.error("[V2Motion] 当前没有加载助手")
        return

    # ============================================================
    # 第三步：获取动作生成器
    # ============================================================
    motion_generator: ExpressionGeneratorV2 | None = None
    try:
        motion_generator = ExpressionGeneratorV2()
        await motion_generator.initialize(agent.agent_name)
    except Exception as e:
        logger.warning(f"[V2Motion] 获取动作生成器失败: {e}")

    if motion_generator is None:
        logger.warning("[动作规划] 未找到可用 Live2D 参数，将输出空动作事件")

    # ============================================================
    # 第四步：初始化上下文
    # ============================================================
    ctx = V2MotionChatContext(motion_generator)
    ctx.user_message = params.msg[-1]["content"] if params.msg else ""

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

    # ============================================================
    # 第五步：创建调度器和纯文本管道
    # ============================================================
    scheduler = TaskScheduler()
    pipeline = scheduler.create_text_pipeline(
        system_context=agent.prompt,
        history_messages=history_messages,
        user_message=ctx.user_message,
    )

    try:
        # ============================================================
        # 第六步：流式执行管道
        # handle_text_result 调用链：
        #   → super().handle_text_result() → base.text_wrapper / base.tts_wrapper
        #   → create_motion_event → base.motion_wrapper
        # ============================================================
        async for result in pipeline.execute():
            await ctx.handle_text_result(result)

            # drain_ready_events → base.drain_ready_sentence_events
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
