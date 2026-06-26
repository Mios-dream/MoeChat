"""
V3 版本聊天模块

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

调用链说明：
- BaseChatContext: 基础上下文（v1.py）
  - handle_json_result: 处理文本结果 → text_wrapper + tts_task
- V3MotionChatContext: 继承基础上下文，添加动作处理
  - handle_motion_result: 处理动作结果 → _create_motion_event
  - handle_result: 分发任务结果到对应处理方法

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
from core.scheduler import TaskScheduler, create_text_task, create_motion_task
from core.scheduler.task import TaskResult
from core.chat.base import store_sentence_event, to_sse
from core.chat.v1 import BaseChatContext
from core.expression_generator.motion_combiner import MotionCombiner
from core.expression_generator.action_sequencer import ActionSequencer

from services.assistant_service import AssistantService

assistant_service = AssistantService()


# ============================================================
# 调度器创建
# ============================================================


def create_scheduler() -> TaskScheduler:
    """
    创建 V3 信息调度器

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


def calc_exit_ms(text: str) -> int:
    """
    根据句尾标点计算动作退出恢复时长

    参数：
    - text: 句子文本

    返回：
    - 退出时长（毫秒）
    """
    stripped = text.strip()
    if "…" in stripped[-3:] or "..." in stripped[-3:]:
        return 1000
    elif stripped.endswith("！") or stripped.endswith("？") or stripped.endswith("?"):
        return 700
    elif stripped.endswith("。") or stripped.endswith("."):
        return 500
    elif stripped.endswith("，") or stripped.endswith(","):
        return 350
    else:
        return 400


async def _create_motion_event(
    sentence_id: int,
    text: str,
    actions: list[str],
) -> dict[str, Any]:
    """
    创建动作事件

    动作阶段关键帧由组合引擎生成，hold/exit 阶段由前端根据音频实际播放时长自行处理。

    参数：
    - sentence_id: 句子 ID
    - text: 原始文本
    - actions: LLM 输出的动作标签列表（如 ["smile", "nod"]）

    返回：
    - 动作事件字典，含 action 阶段关键帧 + exit_ms
    """
    combiner = MotionCombiner()

    # 标点感知的时长估算
    text_duration = combiner.estimate_duration(text)

    # 编排动作时序
    sequencer = ActionSequencer()
    action_specs = sequencer.sequence(
        action_names=actions,
        text_duration=text_duration,
    )

    # 退出恢复时长
    exit_ms = calc_exit_ms(text)

    # 组合动作阶段关键帧
    motion_keyframes = combiner.combine_keyframes(
        action_specs=action_specs,
        text_duration=text_duration,
    )

    return {
        "type": "motion_frame",
        "sentence_id": sentence_id,
        "source_text": text,
        "motions": [
            {
                "duration": motion_keyframes.duration,
                "keyframes": {
                    param: [
                        {"time": kp.time, "value": kp.value, "ease": kp.ease}
                        for kp in points
                    ]
                    for param, points in motion_keyframes.keyframes.items()
                },
                "exit_ms": exit_ms,
            }
        ],
        "duration": motion_keyframes.duration,
        "timestamp_ms": time.time() * 1000,
        "done": False,
    }


# ============================================================
# V3 聊天上下文
# ============================================================


class V3MotionChatContext(BaseChatContext):
    """
    V3Motion 聊天上下文

    继承基础上下文，添加 V3 动作处理逻辑。

    调用链：
    - handle_result: 分发任务结果
      - text → handle_json_result（继承自 BaseChatContext）
      - motion → handle_motion_result
    """

    def __init__(self):
        """初始化 V3Motion 聊天上下文"""
        super().__init__(
            event_order=("text", "audio", "motion_frame"),
            tts_concurrency=1,
        )
        # 文本缓存（V3 模式使用）
        self.text_cache: dict[int, str] = {}
        # 动作缓存（V3 模式使用）
        self.motion_cache: dict[int, list[str]] = {}

    async def handle_json_result(self, result: TaskResult):
        """
        处理 JSON 结果

        参数：
        - result: 文本任务结果，data 格式为纯文本字符串
        """
        sentence_id = result.sentence_id
        text = result.data

        # 缓存文本
        self.text_cache[sentence_id] = text
        # 收集完整文本
        self.full_text_list.append(text)

        # 创建文本和音频事件
        self.track_task(asyncio.create_task(self.create_text_event(sentence_id, text)))
        self.track_task(
            asyncio.create_task(self.create_audio_event(sentence_id, text, text))
        )

    async def handle_motion_result(self, result: TaskResult):
        """
        处理动作结果

        触发事件：
        1. motion_frame 事件（组合引擎处理后）

        参数：
        - result: 动作任务结果
        """
        sentence_id = result.sentence_id
        actions = result.data

        # 缓存动作
        self.motion_cache[sentence_id] = actions

        # 如果已有对应的文本，立即创建动作事件
        if sentence_id in self.text_cache:
            motion_event = await _create_motion_event(
                sentence_id=sentence_id,
                text=self.text_cache[sentence_id],
                actions=actions,
            )
            store_sentence_event(
                self.sentence_events, sentence_id, "motion_frame", motion_event
            )

    async def handle_result(self, result: TaskResult):
        """
        处理调度器结果（分发到对应处理方法）

        调用链：
        - text → handle_json_result → text_wrapper + tts_task
        - motion → handle_motion_result → _create_motion_event

        参数：
        - result: 调度器产出的任务结果
        """
        if result.task_type == "text":
            await self.handle_json_result(result)
        elif result.task_type == "motion":
            await self.handle_motion_result(result)


# ============================================================
# V3 主函数
# ============================================================


async def llm_chat_with_tts_and_motion_v3(params: chat_data):
    """
    V3 版本聊天流式输出

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
    start_time = time.time()
    delay_flag = False

    # 第一步：获取助手实例
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
    ctx = V3MotionChatContext()

    # 第四步：执行管道并输出事件

    try:
        # 流式执行管道
        # handle_result 调用链：
        #   text → handle_json_result → text_wrapper + tts_task
        #   motion → handle_motion_result → _create_motion_event
        async for result in pipeline.execute():
            await ctx.handle_result(result)
            for payload in ctx.drain_ready_events():
                if not delay_flag:
                    logger.info(
                        f"[V3] 首条回复已生成，耗时 {(time.time() - start_time):.2f} 秒"
                    )
                    delay_flag = True
                yield to_sse(payload)

        # 等待所有异步任务完成
        await ctx.wait_for_completion()

        # 输出剩余事件
        for payload in ctx.drain_ready_events():
            yield to_sse(payload)

        # 输出完成信号
        full_text = ctx.get_full_text()
        yield to_sse(
            {
                "type": "done",
                "timestamp_ms": time.time() * 1000,
                "total_sentences": len(ctx.text_cache),
                "full_text": full_text,
                "done": True,
            }
        )

        # 保存到助手上下文
        asyncio.create_task(
            agent.add_msg(
                user_msg=user_message,
                assistant_msg=full_text,
            )
        )

    except Exception as e:
        # 取消所有待处理的任务
        for task in list(ctx.pending_tasks):
            task.cancel()

        logger.error(f"[V3] 处理数据时出错: {e}", exc_info=True)
        yield to_sse(
            {
                "type": "error",
                "timestamp_ms": time.time() * 1000,
                "data": str(e),
                "done": True,
            }
        )
