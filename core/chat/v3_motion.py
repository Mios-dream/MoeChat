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
│  5. 句子就绪即发：text + motion 到齐 → 入队 → 后台并行 TTS + 动作检索  │
│     _process_item: asyncio.gather(tts, motion_engine)              │
│     → emit_ready 按入队顺序取出已就绪事件发射                        │
└─────────────────────────────────────────────────────────────────┘

调用链说明：
- BaseChatContext: 基础上下文（v1.py）
  - handle_json_result: 处理文本结果 → text_wrapper + tts_task
- V3MotionChatContext: 继承基础上下文，添加动作处理
  - handle_result: 分发任务结果 → 缓存数据 → 配对入队 → 后台处理
  - _process_item: 后台并行执行 TTS + 动作检索 → 标记就绪
  - emit_ready: 按入队顺序取出已就绪事件
- _create_motion_event: get_motion_engine() → engine.process() → motion_to_keyframes()

SSE 事件格式：
data: {"type": "text", "sentence_id": 1, "message": "你好呀~", ...}
data: {"type": "audio", "sentence_id": 1, "file": "base64...", ...}
data: {"type": "motion_frame", "sentence_id": 1, "motions": [...], ...}
"""

from collections.abc import AsyncGenerator
import json
import asyncio
from models.dto.request.chat_request import ChatRequest
from core.chat.multimodal_processor import build_user_message_content
from models.dto.response.ChatResponse import (
    AssistantMessage,
    DoneMessage,
    ErrorMessage,
    FullChatResponse,
    ToolCallItem,
    ToolCallFunction,
    ToolMessage,
)
from my_utils.log import logger
from services.memory_v2 import MemoryV2
from core.scheduler import (
    TaskScheduler,
    create_text_task,
    create_motion_task,
    create_bilingual_task,
)
from core.scheduler.task import TaskResult, ToolCallEvent, ToolResultEvent
from core.scheduler.parsers.text_stream_parser import filter_tts_text
from core.chat.base import tts_wrapper
from core.chat.v1 import BaseChatContext
from dataclasses import dataclass, field
from typing import Literal


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


@dataclass
class _QueueItem:
    """队列项：句子待处理/工具事件直接就绪"""

    kind: Literal["sentence", "tool"]
    events: list[FullChatResponse] | None = None
    is_ready: bool = False
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    # 句子专用字段
    text: str = ""
    motions: list[dict] | None = None
    expression: str | None = None


async def _create_motion_event(
    text: str,
    motions: list[dict],
    expression: str | None,
    engine: MotionEngineService,
) -> dict | None:
    """
    创建动作事件

    使用 V3 动作引擎：语义检索 → anchor 定位 → 跨句子释放 → 覆盖混合 → 输出逐帧曲线。

    参数：
    - text: 原始文本（用于语义检索和 anchor 定位）
    - motions: LLM 输出的特殊动作列表 [{"name": str, "anchor": str, "intensity": float}, ...]
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
        return None

    motion_dict: dict = {
        "duration": duration_ms,
        "curves": motion_data.curves,
        "fps": motion_data.fps,
        "expression": motion_data.expression,
    }

    return motion_dict


class V3MotionChatContext(BaseChatContext):
    """
    V3Motion 聊天上下文

    输出策略：
    - 句子（text+motion）就绪后立即入队并启动后台 TTS + 动作检索
    - 工具事件直接入队标记就绪
    - emit_ready: 按入队顺序依次取出已就绪事件
    """

    def __init__(
        self, tts_lang: str = "zh", expressions: list[ExpressionInfo] | None = None
    ):
        super().__init__(tts_concurrency=1)
        self.tts_lang: str = tts_lang
        self.text_cache: dict[int, str] = {}
        self.motion_cache: dict[int, list[dict]] = {}
        self.expression_cache: dict[int, str] = {}
        self.expressions: list[ExpressionInfo] = expressions or []
        self.motion_engine: MotionEngineService = MotionEngineService(
            Config.MOTION_DB_PATH
        )
        self._raw_json_lines: dict[int, str] = {}

        # 有序发射队列
        self._queue: list[_QueueItem] = []
        # 缓冲区：等待 text/motion 配对
        self._text_buf: dict[int, str] = {}
        self._motion_buf: dict[int, tuple[list[dict], str | None]] = {}

    async def _build_motion_event(
        self,
        text: str,
        motions: list[dict],
        expression: str | None = None,
    ) -> dict | None:
        """调用动作引擎生成 motion 曲线字典"""
        return await _create_motion_event(
            text=text,
            motions=motions,
            expression=expression,
            engine=self.motion_engine,
        )

    async def handle_json_result(self, result: TaskResult):
        """缓存文本"""
        sentence_id = result.sentence_id
        text = result.data
        self.text_cache[sentence_id] = text
        self.full_text_list.append(text)

    async def handle_motion_result(self, result: TaskResult):
        """缓存动作"""
        sentence_id = result.sentence_id
        motion_data: dict = result.data
        motions: list[dict] = motion_data.get("motions", [])
        expression: str | None = motion_data.get("expression")
        self.motion_cache[sentence_id] = motions
        if expression:
            self.expression_cache[sentence_id] = expression

    async def handle_bilingual_result(self, result: TaskResult):
        """缓存双语文本"""
        sentence_id = result.sentence_id
        text = result.data.get("text", "")
        self.text_cache[sentence_id] = text
        self.full_text_list.append(text)

    async def handle_result(self, result: TaskResult):
        """分发任务结果：缓存数据，配对入队，启动后台处理"""
        if result.raw_data and result.sentence_id not in self._raw_json_lines:
            self._raw_json_lines[result.sentence_id] = json.dumps(
                result.raw_data, ensure_ascii=False
            )

        if result.task_type == "text":
            await self.handle_json_result(result)
            self._text_buf[result.sentence_id] = result.data
        elif result.task_type == "bilingual":
            await self.handle_bilingual_result(result)
            self._text_buf[result.sentence_id] = result.data.get("text", "")
        elif result.task_type == "motion":
            await self.handle_motion_result(result)
            motions = result.data.get("motions", [])
            expression = result.data.get("expression")
            self._motion_buf[result.sentence_id] = (motions, expression)
        elif result.task_type == "tool_call":
            tc: ToolCallEvent = result.data
            # 内部工具（如 remember / recall / update_memory）不转发给客户端，保持沉浸感
            if tc.tool_name in ("remember", "recall", "update_memory"):
                return
            self._queue.append(
                _QueueItem(
                    kind="tool",
                    is_ready=True,
                    events=[
                        AssistantMessage(
                            tool_calls=[
                                ToolCallItem(
                                    id=tc.call_id,
                                    function=ToolCallFunction(
                                        name=tc.tool_name,
                                        arguments=tc.arguments,
                                    ),
                                )
                            ]
                        )
                    ],
                )
            )
            return
        elif result.task_type == "tool_result":
            tr: ToolResultEvent = result.data
            # 内部工具（如 remember / recall / update_memory）的结果也不转发给客户端
            if tr.tool_name in ("remember", "recall", "update_memory"):
                return
            self._queue.append(
                _QueueItem(
                    kind="tool",
                    is_ready=True,
                    events=[ToolMessage(tool_call_id=tr.call_id, content=tr.content)],
                )
            )
            return

        # 尝试将已配对的句子入队
        self._flush_sentences()

    def _flush_sentences(self):
        """将已配对的句子按 sid 顺序入队，并启动后台处理"""
        while self._text_buf:
            sid = min(self._text_buf)
            if sid in self._motion_buf:
                text = self._text_buf.pop(sid)
                motions, expression = self._motion_buf.pop(sid)
            elif len(self._text_buf) > 1:
                # 更高 sid 的 text 已到达 → 本句不会再有 motion
                text = self._text_buf.pop(sid)
                motions, expression = None, None
            else:
                # 唯一待处理句子，等待 motion 到达
                break

            item = _QueueItem(
                kind="sentence",
                text=text,
                motions=motions,
                expression=expression,
            )
            self._queue.append(item)

            # 后台处理：TTS + 动作检索
            asyncio.create_task(self._process_item(item))

    async def _process_item(self, item: _QueueItem):
        """后台处理句子：并行 TTS + 动作检索，完成后标记就绪"""
        try:
            tts_text = filter_tts_text(item.text)

            motion_event: dict | None = None
            audio_file: str | None = None

            if item.motions and self.tts_lang == "zh" and tts_text.strip():
                motion_event, audio_file = await asyncio.gather(
                    self._build_motion_event(item.text, item.motions, item.expression),
                    tts_wrapper(self.tts_semaphore, item.text, tts_text),
                )
            elif item.motions:
                motion_event = await self._build_motion_event(
                    item.text, item.motions, item.expression
                )
            elif self.tts_lang == "zh" and tts_text.strip():
                audio_file = await tts_wrapper(self.tts_semaphore, item.text, tts_text)

            extras = {}
            if motion_event:
                extras["motion"] = motion_event
            if audio_file:
                extras["audio"] = audio_file

            item.events = [AssistantMessage(content=item.text, extras=extras or None)]
            item.is_ready = True
            item._ready_event.set()

        except Exception as e:
            logger.error(f"[V3] 处理句子出错: {e}", exc_info=True)
            item.events = [AssistantMessage(content=item.text)]
            item.is_ready = True
            item._ready_event.set()

    def emit_ready(self) -> list[FullChatResponse]:
        """按入队顺序取出已就绪的事件"""
        events = []
        while self._queue and self._queue[0].is_ready:
            events.extend(self._queue.pop(0).events or [])
        return events

    async def finalize(self):
        """管道结束后：将缓冲区中剩余的句子入队，等待所有后台任务完成"""
        # 清空缓冲区：管道已结束，不再会有新的 TaskResult 到达
        for sid in sorted(self._text_buf):
            text = self._text_buf[sid]
            motions, expression = self._motion_buf.pop(sid, (None, None))
            item = _QueueItem(
                kind="sentence",
                text=text,
                motions=motions,
                expression=expression,
            )
            self._queue.append(item)
            asyncio.create_task(self._process_item(item))
        self._text_buf.clear()

        # 等待所有句子项处理完成
        for item in self._queue:
            if item.kind == "sentence" and not item.is_ready:
                await item._ready_event.wait()

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
        """
        创建调度器，注册任务
        """
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
        """
        聊天编排
        """
        agent = self.assistant_service.get_current_assistant()
        if not agent:
            yield ErrorMessage(error_code="NO_ASSISTANT", data="当前没有加载助手")
            return

        user_message, user_text = build_user_message_content(params)
        ctx = V3MotionChatContext(
            tts_lang=agent.agent_config.gsvSetting.textLang,
            expressions=load_expressions(agent.agent_name),
        )

        # 动态上下文（记忆检索 + 知识库等）放在对话历史之后，避免击穿前缀缓存
        dynamic_context = await agent.get_context(
            msg=user_text, is_sleep_mode=params.is_sleep_mode
        )

        history_messages = []
        # 固定前缀：记忆系统说明（可缓存）
        if agent.enable_long_memory:
            history_messages.append({
                "role": "system",
                "content": MemoryV2.build_system_prompt(
                    agent.char, agent.user
                ),
            })
        # 中间段：对话历史
        history_messages.extend(agent.get_history())
        # 动态后缀：每次变化的上下文
        history_messages.extend(dynamic_context)

        pipeline = self._build_scheduler(agent).create_task_pipeline(
            system_context=agent.prompt,
            history_messages=history_messages,
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
                for event in ctx.emit_ready():
                    yield event

            # 等待所有后台任务完成，按序发射剩余事件
            await ctx.finalize()
            for event in ctx.emit_ready():
                yield event

            full_text = ctx.get_full_text()
            yield DoneMessage(full_text=full_text)

            # chat_history 保存多任务 JSON 格式，让模型从历史中学习输出格式
            raw_output = ctx.get_raw_output()
            agent.chat_history.append({"role": "assistant", "content": raw_output})
            asyncio.create_task(
                agent.add_msg(user_msg=user_text, assistant_msg=ctx.get_full_text())
            )

        except Exception as e:
            for task in list(ctx.pending_tasks):
                task.cancel()
            logger.error(f"[V3] 处理数据时出错: {e}", exc_info=True)
            yield ErrorMessage(error_code="500", data=f"处理数据时出错: {e}")
