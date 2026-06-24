"""
V4 版本聊天模块

使用信息调度中心（TaskScheduler）实现的聊天版本。

核心特性：
1. 单次 LLM 调用同时生成文本和动作标签
2. 流式解析，逐句播放
3. 可扩展的任务系统

架构流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建调度器，注册任务                                          │
│     scheduler = TaskScheduler()                                  │
│     scheduler.add_task(create_text_task())                       │
│     scheduler.add_task(create_motion_task())                     │
├─────────────────────────────────────────────────────────────────┤
│  2. 调度器自动组合提示词                                           │
│     system_prompt = 角色设定 + 任务说明 + 输出格式                   │
├─────────────────────────────────────────────────────────────────┤
│  3. 创建管道，执行 LLM 流式调用                                    │
│     pipeline = scheduler.create_pipeline(user_message)           │
│     async for result in pipeline.execute():                      │
│         handle(result)                                           │
├─────────────────────────────────────────────────────────────────┤
│  4. 管道内部：LLM 输出 JSON 行 → 解析器分发 → 产出 TaskResult       │
│     LLM: {"text": "你好", "actions": ["smile"]}                  │
│     解析器: text → "你好", actions → ["smile"]                    │
└─────────────────────────────────────────────────────────────────┘

LLM 输出格式（每行一个 JSON）：
{"text": "你好呀~", "actions": ["smile", "nod"]}
{"text": "今天天气真好", "actions": ["look_up"]}

SSE 事件格式：
data: {"type": "text", "sentence_id": 1, "message": "你好呀~", ...}
data: {"type": "audio", "sentence_id": 1, "file": "base64...", ...}
data: {"type": "motion_frame", "sentence_id": 1, "motions": [...], ...}
"""

import time
import asyncio
import re
import base64
from dataclasses import dataclass, field
from typing import Any

from models.dto.chat_request import chat_data
from my_utils.log import logger
from core.scheduler import TaskScheduler, create_text_task, create_motion_task
from core.scheduler.task import TaskResult
from core.chat.base import (
    TTSData,
    tts_task,
    text_wrapper,
    store_sentence_event,
    drain_ready_sentence_events,
    to_sse,
)
from core.expression_generator.motion_combiner import MotionCombiner, ActionSpec

from services.assistant_service import AssistantService

assistant_service = AssistantService()


@dataclass
class V4ChatContext:
    """
    V4 聊天上下文

    封装聊天过程中的所有状态，避免函数嵌套。

    属性：
    - sentence_events: 按句子 ID 聚合的事件存储
    - expected_sentence_id: 预期的下一个句子 ID
    - pending_tasks: 异步任务跟踪集合
    - tts_semaphore: TTS 并发控制信号量
    - text_cache: 文本缓存，关联同一句子的文本和动作
    - motion_cache: 动作缓存，关联同一句子的文本和动作
    """

    # 事件存储，按句子 ID 聚合
    sentence_events: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # 预期的下一个句子 ID
    expected_sentence_id: int = 1
    # 异步任务跟踪集合
    pending_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    # TTS 并发控制信号量，确保一次只处理一个 TTS 请求
    tts_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(1)
    )
    # 缓存，关联同一句子的文本和动作
    text_cache: dict[int, str] = field(default_factory=dict)
    # 缓存，关联同一句子的文本和动作
    motion_cache: dict[int, list[str]] = field(default_factory=dict)
    # 事件类型
    event_order: tuple[str, ...] = ("text", "audio", "motion_frame")

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
# 调度器创建
# ============================================================


def create_scheduler() -> TaskScheduler:
    """
    创建 V4 信息调度器

    注册所有需要的任务：
    - text: 文本生成任务（优先级 100）
    - motion: 动作标签任务（优先级 200）

    返回：
    - 配置好的 TaskScheduler 实例
    """
    scheduler = TaskScheduler()

    # 注册文本任务
    scheduler.add_task(create_text_task(priority=100))

    # 注册动作任务
    scheduler.add_task(create_motion_task(priority=200))

    return scheduler


# ============================================================
# 辅助函数
# ============================================================


def _filter_tts_text(text: str) -> str:
    """
    过滤文本用于 TTS 合成

    处理逻辑：
    1. 移除括号及其内容（如动作描述、心理活动等）
    2. 移除特殊字符（省略号、引号等）

    参数：
    - text: 原始文本

    返回：
    - 过滤后的文本
    """
    # 移除括号内容
    text = re.sub(r"[\(（\[【{].*?[\)）\]】}]", "", text)
    # 移除特殊字符
    text = re.sub(r'[…""' "—\n\r\t\f ]", "", text)
    return text


async def _create_motion_event(
    sentence_id: int,
    text: str,
    actions: list[str],
) -> dict[str, Any]:
    """
    创建动作事件

    使用组合引擎将动作标签转换为参数曲线。

    参数：
    - sentence_id: 句子 ID
    - text: 原始文本
    - actions: 动作标签列表（如 ["smile", "nod"]）

    返回：
    - 动作事件字典
    """

    # 创建组合器
    combiner = MotionCombiner(fps=30.0)

    # 构建动作规格
    action_specs = [ActionSpec(name=name) for name in actions]

    # 根据文本长度估算时长（毫秒）
    text_clean = text.replace(" ", "").replace("\n", "")
    duration_ms = len(text_clean) * 100

    # 组合动作曲线
    motion_curve = combiner.combine_curve(
        action_specs=action_specs,
        text_duration=duration_ms / 1000,
    )

    # 构建动作事件
    return {
        "type": "motion_frame",
        "sentence_id": sentence_id,
        "source_text": text,
        "motions": [
            {
                "duration": motion_curve.duration * 1000,
                "curves": {
                    param: [[t, v] for t, v in points]
                    for param, points in motion_curve.curves.items()
                },
            }
        ],
        "duration": duration_ms,
        "timestamp_ms": time.time() * 1000,
        "done": False,
    }


# ============================================================
# 结果处理函数
# ============================================================


async def _handle_text_result(ctx: V4ChatContext, result: TaskResult):
    """
    处理文本结果

    触发事件：
    1. text 事件（立即）
    2. audio 事件（TTS 合成后）

    参数：
    - ctx: V4 聊天上下文
    - result: 文本任务结果
    """
    sentence_id = result.sentence_id
    text = result.data

    # 缓存文本
    ctx.text_cache[sentence_id] = text

    # 创建文本事件
    async def create_text_event():
        payload = await text_wrapper(
            sentence_id=sentence_id,
            sentence_text=text,
        )
        store_sentence_event(ctx.sentence_events, sentence_id, "text", payload)

    # 创建音频事件
    async def create_audio_event():
        tts_text = _filter_tts_text(text)

        audio_event = {
            "type": "audio",
            "sentence_id": sentence_id,
            "message": tts_text,
            "source_text": text,
            "file": "",
            "timestamp_ms": time.time() * 1000,
            "done": False,
        }

        async with ctx.tts_semaphore:
            try:
                if tts_text:
                    tts_data = TTSData(text=tts_text, ref_audio="", ref_text="")
                    audio_data = await tts_task(tts_data)
                    if audio_data:
                        audio_event["file"] = base64.b64encode(audio_data).decode(
                            "utf-8"
                        )
            except Exception as e:
                logger.error(f"[V4] TTS 合成失败: {e}")

        store_sentence_event(ctx.sentence_events, sentence_id, "audio", audio_event)

    ctx.track_task(asyncio.create_task(create_text_event()))
    ctx.track_task(asyncio.create_task(create_audio_event()))


async def _handle_motion_result(ctx: V4ChatContext, result: TaskResult):
    """
    处理动作结果

    触发事件：
    1. motion_frame 事件（组合引擎处理后）

    参数：
    - ctx: V4 聊天上下文
    - result: 动作任务结果
    """
    sentence_id = result.sentence_id
    actions = result.data

    # 缓存动作
    ctx.motion_cache[sentence_id] = actions

    # 如果已有对应的文本，立即创建动作事件
    if sentence_id in ctx.text_cache:
        motion_event = await _create_motion_event(
            sentence_id=sentence_id,
            text=ctx.text_cache[sentence_id],
            actions=actions,
        )
        store_sentence_event(
            ctx.sentence_events, sentence_id, "motion_frame", motion_event
        )


async def _handle_result(ctx: V4ChatContext, result: TaskResult):
    """
    处理调度器结果

    参数：
    - ctx: V4 聊天上下文
    - result: 调度器产出的任务结果
    """
    if result.task_type == "text":
        await _handle_text_result(ctx, result)
    elif result.task_type == "motion":
        await _handle_motion_result(ctx, result)


# ============================================================
# V4 主函数
# ============================================================


async def llm_chat_with_tts_and_motion_v4(params: chat_data):
    """
    V4 版本聊天流式输出

    使用信息调度中心，单次 LLM 调用同时生成文本和动作标签。

    流程概述：
    1. 创建调度器，注册文本和动作任务
    2. 调度器自动组合提示词
    3. 创建管道，流式调用 LLM
    4. 解析器将 LLM 输出分发给对应任务
    5. 每个任务结果触发对应的事件（文本/音频/动作）

    参数：
    - params: 聊天请求参数

    产出：
    - SSE 格式的事件流
    """
    # 第一步：获取助手实例和消息列表
    agent = assistant_service.get_current_assistant()

    if not agent:
        logger.error("当前没有加载助手")
        return

    # 第二步：创建调度器和管道
    scheduler = create_scheduler()

    # 用户消息
    user_message = params.msg[-1]["content"]
    # 获取历史消息（包含上下文）
    history_messages = [
        *agent.get_history(),
        {
            "role": "user",
            "content": await agent.get_context(
                msg=user_message, is_sleep_mode=params.is_sleep_mode
            ),
        },
    ]

    # 创建管道
    pipeline = scheduler.create_pipeline(
        system_context=agent.prompt,
        history_messages=history_messages,
        user_message=user_message,
    )

    # 第三步：初始化上下文
    chat_context = V4ChatContext()

    # 第四步：执行管道并输出事件

    try:
        # 流式执行管道
        async for result in pipeline.execute():
            # 捕获结果并触发事件
            await _handle_result(chat_context, result)
            # 输出就绪的句子事件
            for payload in chat_context.drain_ready_events():
                yield to_sse(payload)

        # 等待所有异步任务完成（语音合成）
        while chat_context.pending_tasks:
            await asyncio.sleep(0.1)

        # 输出剩余事件
        for payload in chat_context.drain_ready_events():
            yield to_sse(payload)

        # 输出完成信号
        full_text = "".join(
            chat_context.text_cache.get(i, "")
            for i in sorted(chat_context.text_cache.keys())
        )
        yield to_sse(
            {
                "type": "done",
                "timestamp_ms": time.time() * 1000,
                "total_sentences": len(chat_context.text_cache),
                "full_text": full_text,
                "done": True,
            }
        )

        # 保存到助手上下文（需要先添加用户消息到临时列表）
        # agent.msg_data_tmp.append({"role": "user", "content": user_message})
        asyncio.create_task(agent.add_msg(full_text))

    except Exception as e:
        # 取消所有待处理的任务
        for task in list(chat_context.pending_tasks):
            task.cancel()

        logger.error(f"[V4] 处理数据时出错: {e}", exc_info=True)
        yield to_sse(
            {
                "type": "error",
                "timestamp_ms": time.time() * 1000,
                "data": str(e),
                "done": True,
            }
        )
