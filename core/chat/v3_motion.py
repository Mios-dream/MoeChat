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

from collections.abc import AsyncGenerator
import json
import time
import asyncio
from models.dto.request.chat_request import ChatRequest
from core.chat.multimodal_processor import build_user_message_content
from models.dto.response.ChatResponse import (
    ChatResponse,
    DoneResponse,
    ErrorResponse,
    FullChatResponse,
    MotionResponse,
)
from my_utils.log import logger
from core.scheduler import (
    TaskScheduler,
    create_text_task,
    create_motion_task,
    create_bilingual_task,
)
from core.scheduler.task import TaskResult
from core.scheduler.parsers.text_stream_parser import filter_tts_text
from core.chat.base import store_sentence_event
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
from tool_system.integration import ToolCallIntegration


async def _create_motion_event(
    sentence_id: int,
    text: str,
    motions: list[dict],
    expression: str | None,
    engine: MotionEngineService,
) -> MotionResponse:
    """
    创建动作事件

    使用 V3 动作引擎：语义检索 → anchor 定位 → 跨句子释放 → 覆盖混合 → 输出逐帧曲线。

    参数：
    - sentence_id: 句子 ID
    - text: 原始文本（用于语义检索和 anchor 定位）
    - motions: LLM 输出的动作列表 [{"name": str, "anchor": str, "intensity": float}, ...]
    - expression: 整句表情名（用于面部渲染），None 表示沿用上一句
    - engine: MotionEngineService 实例

    返回：
    - MotionResponse，含逐帧曲线 (curves) + expression
    """

    # 估算文本时长
    text_duration = estimate_text_duration(text)

    # 获取事件循环
    loop = asyncio.get_running_loop()

    # 在线程池中执行动作处理（SentenceTransformer 编码是 CPU 密集型操作）
    motion_data = await loop.run_in_executor(
        None,
        lambda: engine.process(text, motions, expression, text_duration),
    )

    duration_ms = int((motion_data.duration if motion_data else text_duration) * 1000)

    if motion_data is None:
        return MotionResponse(
            sentence_id=sentence_id,
            source_text=text,
            motions=[],
            duration=0,
        )

    motion_dict: dict = {
        "duration": duration_ms,
        "curves": motion_data.curves,
        "fps": motion_data.fps,
    }
    if motion_data.expression:
        motion_dict["expression"] = motion_data.expression

    return MotionResponse(
        sentence_id=sentence_id,
        source_text=text,
        motions=[motion_dict],
        duration=duration_ms,
    )


class V3MotionChatContext(BaseChatContext):
    """
    V3Motion 聊天上下文

    继承基础上下文，处理 text / bilingual / motion 三种任务结果。
    """

    def __init__(
        self, tts_lang: str = "zh", expressions: list[ExpressionInfo] | None = None
    ):
        super().__init__(
            event_order=("text", "audio", "motion_frame"),
            tts_concurrency=1,
        )
        self.tts_lang: str = tts_lang
        self.text_cache: dict[int, str] = {}
        self.motion_cache: dict[int, list[dict]] = {}
        self.expressions: list[ExpressionInfo] = expressions or []
        self.motion_engine: MotionEngineService = MotionEngineService(
            Config.MOTION_DB_PATH
        )
        # 按 sentence_id 收集原始多任务 JSON 行，用于保存到 chat_history
        self._raw_json_lines: dict[int, str] = {}

    async def handle_json_result(self, result: TaskResult):
        """处理文本结果"""
        sentence_id = result.sentence_id
        text = result.data
        tts_text = filter_tts_text(text)

        self.text_cache[sentence_id] = text
        self.full_text_list.append(text)

        await self.create_text_event(sentence_id, text)
        if self.tts_lang == "zh":
            self.track_task(
                asyncio.create_task(
                    self.create_audio_event(sentence_id, text, tts_text)
                )
            )

    async def handle_motion_result(self, result: TaskResult):
        """处理动作结果"""
        sentence_id = result.sentence_id
        motion_data: dict = result.data
        motions: list[dict] = motion_data.get("motions", [])
        expression: str | None = motion_data.get("expression")

        self.motion_cache[sentence_id] = motions

        if sentence_id in self.text_cache:
            motion_event = await _create_motion_event(
                sentence_id=sentence_id,
                text=self.text_cache[sentence_id],
                motions=motions,
                expression=expression,
                engine=self.motion_engine,
            )
            store_sentence_event(
                self.sentence_events, sentence_id, "motion_frame", motion_event
            )

    async def handle_bilingual_result(self, result: TaskResult):
        """处理双语翻译结果"""
        sentence_id = result.sentence_id
        text = result.data.get("text", "")
        tts_text = result.data.get("tts_text", "")
        self.track_task(
            asyncio.create_task(self.create_audio_event(sentence_id, text, tts_text))
        )

    async def handle_result(self, result: TaskResult):
        """分发任务结果：text / bilingual / motion，并捕获原始多任务 JSON"""
        # 每 sentence_id 只捕获一次原始 JSON 行
        if result.raw_data and result.sentence_id not in self._raw_json_lines:
            self._raw_json_lines[result.sentence_id] = json.dumps(
                result.raw_data, ensure_ascii=False
            )
        if result.task_type == "text":
            await self.handle_json_result(result)
        elif result.task_type == "bilingual":
            await self.handle_bilingual_result(result)
        elif result.task_type == "motion":
            await self.handle_motion_result(result)

    def get_raw_output(self) -> str:
        """获取多任务 JSON 格式的完整输出"""
        if not self._raw_json_lines:
            return self.get_full_text()
        sorted_ids = sorted(self._raw_json_lines.keys())
        return "\n".join(self._raw_json_lines[sid] for sid in sorted_ids)


class V3ChatService:
    """
    V3 聊天服务。

    工具调用由 Pipeline 内部闭环 + on_tool_event 回调写历史，
    ChatService 只处理 text / motion / bilingual 结果。
    """

    def __init__(self):
        self.assistant_service = AssistantService()
        self.integration: ToolCallIntegration | None = None

    def set_integration(self, integration: ToolCallIntegration):
        """
        设置工具调用组件实例
        """
        self.integration = integration

    def _build_scheduler(self, agent) -> TaskScheduler:
        scheduler = TaskScheduler()
        scheduler.add_task(create_text_task(priority=100))

        lang = agent.agent_config.gsvSetting.textLang
        if lang != "zh" and lang in ("en", "ja"):
            scheduler.add_task(create_bilingual_task(target_lang=lang, priority=150))

        expressions = load_expressions(agent.agent_name)
        action_prompt = (
            f"可用表情：{[expr.name for expr in expressions]}\n"
            f"可用动作：{', '.join(ACTION_DESCRIPTIONS.keys())}"
        )
        scheduler.add_task(
            create_motion_task(available_actions=action_prompt, priority=200)
        )
        return scheduler

    async def chat(self, params: ChatRequest) -> AsyncGenerator[FullChatResponse]:
        agent = self.assistant_service.get_current_assistant()
        if not agent:
            yield ErrorResponse(error_code="NO_ASSISTANT", data="当前没有加载助手")
            return

        user_message, user_text = build_user_message_content(params)
        ctx = V3MotionChatContext(
            tts_lang=agent.agent_config.gsvSetting.textLang,
            expressions=load_expressions(agent.agent_name),
        )

        pipeline = self._build_scheduler(agent).create_task_pipeline(
            system_context=agent.prompt,
            history_messages=[
                {
                    "role": "system",
                    "content": await agent.get_context(
                        msg=user_text, is_sleep_mode=params.is_sleep_mode
                    ),
                },
                *agent.get_history(),
            ],
            user_message=user_message,
            tools=self.integration.get_tools() if self.integration else None,
            tool_handler=(
                self.integration.process_tool_calls if self.integration else None
            ),
            on_tool_event=lambda msg: agent.chat_history.append(msg),
        )

        agent.chat_history.extend(user_message)

        try:
            async for result in pipeline.execute():
                await ctx.handle_result(result)
                for event in ctx.drain_ready_events():
                    yield event

            await ctx.wait_for_completion()
            for event in ctx.drain_ready_events():
                yield event

            full_text = ctx.get_full_text()
            yield DoneResponse(full_text=full_text)

            # chat_history 保存多任务 JSON 格式，让模型从历史中学习输出格式
            raw_output = ctx.get_raw_output()
            agent.chat_history.append({"role": "assistant", "content": raw_output})
            asyncio.create_task(
                agent.add_msg(user_msg=user_text, assistant_msg=full_text)
            )

        except Exception as e:
            for task in list(ctx.pending_tasks):
                task.cancel()
            logger.error(f"[V3] 处理数据时出错: {e}", exc_info=True)
            yield ErrorResponse(error_code="500", data=f"处理数据时出错: {e}")
