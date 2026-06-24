"""
V3 版本聊天模块

使用 V3 生成器的动作生成版本。

核心特性：
1. 流式文本处理，逐句播放
2. TTS 并行合成
3. V3 动作生成器（流式 Batch 架构）
4. 事件按句子 ID 聚合输出

架构流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建流处理器和 V3 动作生成器                                    │
│     processor = StreamProcessor()                                │
│     motion_generator = MotionGeneratorV3()                       │
├─────────────────────────────────────────────────────────────────┤
│  2. 句子完成时触发 TTS + 动作生成                                   │
│     on_complete → text_event + audio_event + motion_event        │
├─────────────────────────────────────────────────────────────────┤
│  3. 事件循环：按序输出就绪事件                                       │
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
from core.chat.base import (
    StreamProcessor,
    text_wrapper,
    tts_wrapper,
    start_llm_task,
    store_sentence_event,
    drain_ready_sentence_events,
    to_sse,
    build_chat_messages,
    get_motion_system_prompt,
)
from core.expression_generator.motion_generator_v3 import (
    MotionGeneratorV3,
    create_v3_generator,
)
from services.assistant_service import AssistantService

assistant_service = AssistantService()

# ============================================================
# V3 上下文
# ============================================================


class V3ChatContext:
    """
    V3 聊天上下文

    封装聊天过程中的所有状态，避免函数嵌套。

    属性：
    - processor: 流处理器实例
    - motion_generator: V3 动作生成器实例
    - sentence_events: 按句子 ID 聚合的事件存储
    - expected_sentence_id: 预期的下一个句子 ID
    - pending_tasks: 异步任务跟踪集合
    - tts_semaphore: TTS 并发控制信号量
    - motion_semaphore: 动作生成并发控制信号量
    """

    def __init__(
        self,
        processor: StreamProcessor,
        motion_generator: MotionGeneratorV3 | None,
    ):
        """
        初始化 V3 聊天上下文

        参数：
        - processor: 流处理器实例
        - motion_generator: V3 动作生成器实例
        """
        self.processor = processor
        self.motion_generator = motion_generator
        self.sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
        self.expected_sentence_id: int = 1
        self.pending_tasks: set[asyncio.Task[Any]] = set()
        self.tts_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)
        self.motion_semaphore: asyncio.Semaphore = asyncio.Semaphore(4)

        # 事件类型顺序
        self.event_order: tuple[str, ...] = ("text", "audio", "motion_frame")

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
# V3 动作生成任务
# ============================================================


async def _v3_motion_task(
    ctx: V3ChatContext,
    sentence_id: int,
    sentence_text: str,
):
    """
    V3 动作生成任务

    使用 V3 生成器的流式生成功能，为每个句子生成动作数据。

    参数：
    - ctx: V3 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    """
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

    async with ctx.motion_semaphore:
        try:
            # 获取前一个动作参数
            previous_params = None
            if sentence_id > 0 and sentence_id - 1 in ctx.processor.motion_history:
                previous_params = ctx.processor.motion_history[sentence_id - 1]

            # 构建对话上下文
            context = (
                "用户："
                + ctx.processor.user_msg
                + "\n。助手的上文回复："
                + "".join(ctx.processor.full_msg[:-1])
                + "\n。当前需要生成动作的回复："
                + sentence_text
            )

            # 估算时长
            sentence_text_clean = sentence_text.replace(" ", "").replace("\n", "")
            duration_time = len(sentence_text_clean) * 100

            # 使用 V3 生成器
            motions = []
            if ctx.motion_generator:
                async for chunk in ctx.motion_generator.stream_generate(
                    text=sentence_text,
                    duration_ms=duration_time,
                    context=context,
                    previous_params=previous_params,
                    is_tts=True,
                    timeout_seconds=10.0,
                ):
                    # 转换为前端格式
                    if chunk.motion:
                        motion_data = {
                            "duration": chunk.motion.duration * 1000,
                            "curves": chunk.motion.curves,
                        }
                        motions.append(motion_data)

            motion_event["duration"] = duration_time
            motion_event["motions"] = motions

            # 记录动作参数到历史
            if motions:
                # 使用最后一个动作的参数
                last_motion = motions[-1]
                if "curves" in last_motion:
                    # 从曲线中提取最后的参数值
                    last_params = {}
                    for param_id, points in last_motion["curves"].items():
                        if points:
                            last_params[param_id] = points[-1][1]
                    ctx.processor.motion_history[sentence_id] = last_params

            logger.info(
                f"[V3动作生成] 生成 {len(motions)} 个动作帧，句子ID: {sentence_id}"
            )

        except Exception as e:
            logger.warning(f"[V3动作生成] 第{sentence_id + 1}句动作生成失败: {e}")

    store_sentence_event(ctx.sentence_events, sentence_id, "motion_frame", motion_event)


# ============================================================
# 事件处理函数
# ============================================================


async def _create_text_event(ctx: V3ChatContext, sentence_id: int, sentence_text: str):
    """
    创建文本事件

    参数：
    - ctx: V3 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    """
    payload = await text_wrapper(
        sentence_id=sentence_id,
        sentence_text=sentence_text,
    )
    store_sentence_event(ctx.sentence_events, sentence_id, "text", payload)


async def _create_audio_event(
    ctx: V3ChatContext,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
):
    """
    创建音频事件

    参数：
    - ctx: V3 聊天上下文
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


async def _create_motion_event(
    ctx: V3ChatContext, sentence_id: int, sentence_text: str
):
    """
    创建动作事件

    参数：
    - ctx: V3 聊天上下文
    - sentence_id: 句子 ID
    - sentence_text: 句子文本
    """
    if ctx.motion_generator:
        await _v3_motion_task(ctx, sentence_id, sentence_text)
    else:
        # 空动作事件
        motion_event = {
            "type": "motion_frame",
            "sentence_id": sentence_id,
            "source_text": sentence_text,
            "motions": [],
            "duration": 0,
            "timestamp_ms": time.time() * 1000,
            "done": False,
        }
        store_sentence_event(
            ctx.sentence_events, sentence_id, "motion_frame", motion_event
        )


async def _on_sentence_complete(
    ctx: V3ChatContext,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
):
    """
    句子完成时的回调

    参数：
    - ctx: V3 聊天上下文
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
    ctx.track_task(
        asyncio.create_task(_create_motion_event(ctx, sentence_id, sentence_text))
    )


# ============================================================
# V3 主函数
# ============================================================


async def llm_chat_with_tts_and_motion_v3(params: chat_data):
    """
    V3 版本聊天流式输出

    使用 V3 生成器的流式 Batch 架构。

    核心改进：
    1. 使用 V3 生成器，单次 LLM 调用同时生成文本和动作
    2. 本地组合引擎，毫秒级延迟
    3. 支持参数曲线输出

    流程概述：
    1. 创建流处理器和 V3 动作生成器
    2. 注册句子完成回调（触发 TTS + 动作生成）
    3. 启动 LLM 流式任务
    4. 事件循环：按序输出就绪事件

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
        logger.error("[V3] 当前没有加载助手")
        return

    # ============================================================
    # 第二步：构建消息列表（使用统一函数，包含动作提示词）
    # ============================================================
    motion_prompt = get_motion_system_prompt()
    agent, history_messages, user_message = await build_chat_messages(
        agent, params, system_prompt_extra=motion_prompt
    )

    # ============================================================
    # 第三步：获取 V3 动作生成器
    # ============================================================
    motion_generator: MotionGeneratorV3 | None = None
    try:
        motion_generator = create_v3_generator()
        await motion_generator.initialize(agent.agent_name)
    except Exception as e:
        logger.warning(f"[V3] 获取动作生成器失败: {e}")

    if motion_generator is None:
        logger.warning("[V3动作规划] 未找到可用动作生成器，将输出空动作事件")

    # ============================================================
    # 第四步：初始化上下文
    # ============================================================
    processor = StreamProcessor()
    processor.user_msg = user_message
    ctx = V3ChatContext(processor, motion_generator)

    # 注册句子完成回调
    processor.register_sentence_complete_callback(
        lambda sid, text, tts: _on_sentence_complete(ctx, sid, text, tts)
    )

    # ============================================================
    # 第五步：启动 LLM 任务
    # ============================================================
    llm_task = asyncio.create_task(start_llm_task(history_messages, processor))

    try:
        # ============================================================
        # 第六步：事件循环
        # ============================================================
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
        agent.msg_data_tmp.append({"role": "user", "content": ctx.processor.user_msg})
        asyncio.create_task(agent.add_msg("".join(processor.full_msg)))

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
