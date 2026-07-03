"""
V3 版本聊天模块

使用信息调度中心（TaskScheduler）实现的聊天版本。

核心特性：
1. 单次 LLM 调用同时生成文本和动作标签
2. 流式解析，逐句播放
3. SQLite 动作数据库 + embedding 语义检索
4. 特殊动作覆盖（面部表情）+ 预录制动作曲线混合

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
│     LLM: {"text": "你好", "actions": ["blush"]}                  │
│     解析器: text → "你好", actions → ["blush"]                    │
├─────────────────────────────────────────────────────────────────┤
│  5. MotionEngineService 处理动作                                  │
│     text → 语义检索 → DB 动作曲线 → 特殊动作覆盖 → 混合 → 关键帧     │
└─────────────────────────────────────────────────────────────────┘

调用链说明：
- BaseChatContext: 基础上下文（v1.py）
  - handle_json_result: 处理文本结果 → text_wrapper + tts_task
- V3MotionChatContext: 继承基础上下文，添加动作处理
  - handle_motion_result: 处理动作结果 → _create_motion_event
  - handle_result: 分发任务结果到对应处理方法
- _create_motion_event: get_motion_engine() → engine.process() → motion_to_keyframes()

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
from core.scheduler import (
    TaskScheduler,
    create_text_task,
    create_motion_task,
    create_bilingual_task,
)
from core.scheduler.task import TaskResult
from core.scheduler.parsers.text_stream_parser import filter_tts_text
from core.chat.base import store_sentence_event, to_sse
from core.chat.v1 import BaseChatContext
from core.expression_generator.motion_engine_v3 import (
    ACTION_DESCRIPTIONS,
    MotionEngineService,
    estimate_text_duration,
)
from core.expression_generator.utils.expression_loader import (
    ExpressionInfo,
    load_expressions,
)
from Config import Config
from services.assistant_service import AssistantService


async def _create_motion_event(
    sentence_id: int,
    text: str,
    actions: list[str],
    engine: MotionEngineService,
    expressions: list[ExpressionInfo] | None = None,
) -> dict[str, Any]:
    """
    创建动作事件

    使用 V3 动作引擎：语义检索预录制动作 → 特殊动作覆盖 → 表情覆盖 → 混合 → 输出逐帧曲线。

    hold/exit 阶段由前端根据音频实际播放时长自行处理。

    参数：
    - sentence_id: 句子 ID
    - text: 原始文本（同时用于语义检索和时长估算）
    - actions: LLM 输出的特殊动作标签列表（如 ["smile", "wink_left"]），
                可同时包含动作名和表情名
    - expressions: 可用表情列表（ExpressionInfo 列表），用于解析表情名称

    返回：
    - 动作事件字典，含逐帧曲线 (curves) + exit_ms
    """

    # 估算文本时长
    text_duration = estimate_text_duration(text)

    # 获取事件循环
    loop = asyncio.get_running_loop()

    # 在线程池中执行动作处理（SentenceTransformer 编码是 CPU 密集型操作）
    motion_data = await loop.run_in_executor(
        None,
        lambda: engine.process(text, actions, text_duration, expressions),
    )

    duration_ms = int((motion_data.duration if motion_data else text_duration) * 1000)

    if motion_data is None:
        # 无匹配结果，返回空动作事件
        return {
            "type": "motion_frame",
            "sentence_id": sentence_id,
            "source_text": text,
            "motions": [],
            "duration": 0,
            "timestamp_ms": time.time() * 1000,
            "done": False,
        }

    return {
        "type": "motion_frame",
        "sentence_id": sentence_id,
        "source_text": text,
        "motions": [
            {
                "duration": duration_ms,
                "curves": motion_data.curves,
                "fps": motion_data.fps,
            }
        ],
        "duration": duration_ms,
        "expression": motion_data.expression,
        "timestamp_ms": time.time() * 1000,
        "done": False,
    }


class V3MotionChatContext(BaseChatContext):
    """
    V3Motion 聊天上下文

    继承基础上下文，添加 V3 动作处理和双语翻译处理逻辑。

    调用链：
    - handle_result: 分发任务结果
      - text → handle_json_result（继承自 BaseChatContext）
      - bilingual → handle_bilingual_result（双语翻译）
      - motion → handle_motion_result
    """

    def __init__(self, tts_lang: str = "zh"):
        """
        初始化 V3Motion 聊天上下文

        参数：
        - tts_lang: GSV 合成目标语言代码（"zh"/"en"/"ja"）
                    当不为 "zh" 时，启用双语翻译模式
        """
        super().__init__(
            event_order=("text", "audio", "motion_frame"),
            tts_concurrency=1,
        )
        # GSV 合成目标语言
        self.tts_lang: str = tts_lang
        # 文本缓存（V3 模式使用）
        self.text_cache: dict[int, str] = {}
        # 动作缓存（V3 模式使用）
        self.motion_cache: dict[int, list[str]] = {}

        self.motion_engine: MotionEngineService = MotionEngineService(
            Config.MOTION_DB_PATH
        )

    async def handle_json_result(self, result: TaskResult):
        """
        处理 JSON 结果

        参数：
        - result: 文本任务结果，data 格式为纯文本字符串
        """
        sentence_id = result.sentence_id
        text = result.data

        # 过滤 TTS 文本：移除括号内的描述内容（如（脸红）（小声）等），避免被错误朗读
        tts_text = filter_tts_text(text)

        # 缓存文本
        self.text_cache[sentence_id] = text
        # 收集完整文本
        self.full_text_list.append(text)

        # 创建文本和音频事件
        self.track_task(asyncio.create_task(self.create_text_event(sentence_id, text)))
        # 只有当 GSV 语言为中文时才创建音频事件，非中文时由双语翻译任务创建音频事件
        if self.tts_lang == "zh":
            self.track_task(
                asyncio.create_task(
                    self.create_audio_event(sentence_id, text, tts_text)
                )
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
                engine=self.motion_engine,
            )
            store_sentence_event(
                self.sentence_events, sentence_id, "motion_frame", motion_event
            )

    async def handle_bilingual_result(self, result: TaskResult):
        """
        处理双语翻译结果

        参数：
        - result: 双语翻译任务结果，data 格式为翻译后的文本字符串
        """
        sentence_id = result.sentence_id
        text = result.data.get("text", "")
        tts_text = result.data.get("tts_text", "")

        self.track_task(
            asyncio.create_task(self.create_audio_event(sentence_id, text, tts_text))
        )

    async def handle_result(self, result: TaskResult):
        """
        处理调度器结果（分发到对应处理方法）

        调用链：
        - text → handle_json_result → text_wrapper + tts_task
        - motion → handle_motion_result → _create_motion_event
        - bilingual → handle_bilingual_result → create_audio_event
        参数：
        - result: 调度器产出的任务结果
        """

        if result.task_type == "text":
            await self.handle_json_result(result)
        elif result.task_type == "bilingual":
            await self.handle_bilingual_result(result)
        elif result.task_type == "motion":
            await self.handle_motion_result(result)


class V3ChatService:
    """
    V3 聊天服务

    提供 V3 版本的聊天流式输出接口。
    """

    def __init__(self):
        self.assistant_service = AssistantService()

    def create_scheduler(self) -> TaskScheduler:
        """
        创建 V3 信息调度器

        注册所有需要的任务：
        - text: 文本生成任务（优先级 100）
        - bilingual: 双语翻译任务（优先级 150，仅在 GSV 语言非中文时注册）
        - motion: 动作标签任务（优先级 200）

        参数：
        - lang: GSV 合成目标语言代码（"zh"/"en"/"ja"）

        返回：
        - 配置好的 TaskScheduler 实例
        """

        scheduler = TaskScheduler()

        current_assistant = self.assistant_service.get_current_assistant()
        if not current_assistant:
            return scheduler

        # 注册文本任务
        scheduler.add_task(create_text_task(priority=100))

        lang = current_assistant.agent_config.gsvSetting.textLang

        # 非中文输出时注册双语翻译任务
        if lang != "zh" and lang in ("en", "ja"):
            scheduler.add_task(create_bilingual_task(target_lang=lang, priority=150))
        # 第二步半：加载助手可用表情，注入调度器提示词
        expressions = load_expressions(current_assistant.agent_name)

        action_prompt = f"""
    可用动作标签：
    {', '.join([f"{action}: {desc}" for action, desc in ACTION_DESCRIPTIONS.items()])}
    可用表情：
    {[expr.name for expr in expressions]}
    """

        # 注册动作任务
        scheduler.add_task(
            create_motion_task(available_actions=action_prompt, priority=200)
        )

        return scheduler

    async def chat(self, params: chat_data):
        """
        V3 版本聊天流式输出

        使用信息调度中心，单次 LLM 调用同时生成文本和动作标签。

        流程概述：
        1. 获取 GSV 合成语言配置
        2. 创建调度器（GSV 非中文时自动注册双语翻译任务），注册文本和动作任务
        3. 调度器自动组合提示词
        4. 创建管道，流式调用 LLM
        5. 解析器将 LLM 输出分发给对应任务
        6. 每个任务结果触发对应的事件（文本/音频/动作/双语翻译）

        参数：
        - params: 聊天请求参数

        产出：
        - SSE 格式的事件流
        """
        start_time = time.time()
        delay_flag = False

        # 第一步：获取助手实例
        agent = self.assistant_service.get_current_assistant()

        if not agent:
            logger.error("当前没有加载助手")
            return

        # 用户消息
        user_message = params.msg[-1]["content"]
        # 获取历史消息（包含上下文）
        history_messages = [
            {
                "role": "system",
                "content": await agent.get_context(
                    msg=user_message, is_sleep_mode=params.is_sleep_mode
                ),
            },
            *agent.get_history(),
        ]
        self.v3_motion_scheduler: TaskScheduler = self.create_scheduler()
        # 创建管道
        pipeline = self.v3_motion_scheduler.create_task_pipeline(
            system_context=agent.prompt,
            history_messages=history_messages,
            user_message=user_message,
        )

        # 第三步：初始化上下文（传入 GSV 语言用于双语翻译控制）
        ctx = V3MotionChatContext(tts_lang=agent.agent_config.gsvSetting.textLang)

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
