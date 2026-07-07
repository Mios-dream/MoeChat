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
import time
import asyncio
from models.dto.request.chat_request import ChatData
from models.dto.response.ChatResponse import (
    ChatResponse,
    DoneResponse,
    ErrorResponse,
    FullChatResponse,
    MotionResponse,
    ToolCallResponse,
    ToolResultResponse,
)
from my_utils.log import logger
from core.scheduler import (
    TaskScheduler,
    create_text_task,
    create_motion_task,
    create_bilingual_task,
)
from core.scheduler.task import TaskResult, ToolCallEvent, ToolResultEvent
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
from openai.types.chat import ChatCompletionMessageParam


async def _create_motion_event(
    sentence_id: int,
    text: str,
    actions: list[str],
    engine: MotionEngineService,
    expressions: list[ExpressionInfo] | None = None,
) -> MotionResponse:
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
        return MotionResponse(
            sentence_id=sentence_id,
            source_text=text,
            motions=[],
            duration=0,
        )

    return MotionResponse(
        sentence_id=sentence_id,
        source_text=text,
        motions=[
            {
                "duration": duration_ms,
                "curves": motion_data.curves,
                "fps": motion_data.fps,
            }
        ],
        duration=duration_ms,
    )


class V3MotionChatContext(BaseChatContext):
    """
    V3Motion 聊天上下文

    继承基础上下文，添加 V3 动作处理和双语翻译处理逻辑。

    调用链：
    - handle_result: 分发任务结果
      - text → handle_json_result（继承自 BaseChatContext）
      - bilingual → handle_bilingual_result（双语翻译）
      - motion → handle_motion_result
      - tool_call → handle_tool_call_result（工具调用事件）
      - tool_result → handle_tool_result_result（工具结果事件）
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
        self.tts_lang: str = tts_lang
        self.text_cache: dict[int, str] = {}
        self.motion_cache: dict[int, list[str]] = {}

        # 全局严格排序输出机制
        # 全局输出序号。每调用一次 +=1
        self._output_seq: int = 1
        # 当前期待排出的序号。只有这个序号的事件就绪了才会释放，释放后 +=1
        self._expected_seq: int = 1
        # 按输出顺序存储所有就绪的事件（句子事件+ 工具事件）
        self._ordered_outputs: dict[int, list[ChatResponse]] = {}
        # 句子 ID → 全局输出序号 映射，因为加入工具调用，句子id不再代表输出顺序，这里将句子id映射到全局输出序号，确保输出顺序正确
        self._sid_to_seq: dict[int, int] = {}

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

        # 为当前句子预分配全局输出序号（若尚未分配），
        # 确保后续 drain_ordered() 按正确顺序释放句子事件包和工具事件
        if sentence_id not in self._sid_to_seq:
            self._sid_to_seq[sentence_id] = self._output_seq
            self._output_seq += 1

        # 过滤 TTS 文本：移除括号内的描述内容（如（脸红）（小声）等），避免被错误朗读
        tts_text = filter_tts_text(text)

        # 缓存文本
        self.text_cache[sentence_id] = text
        # 收集完整文本
        self.full_text_list.append(text)

        # 文本事件：直接 await 同步完成（text_wrapper 内部无异步操作），
        # 确保 drain_ready_events 能立即排出文本，不等待异步的 audio
        await self.create_text_event(sentence_id, text)
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
        - tool_call → handle_tool_call → 即时产出工具调用事件
        - tool_result → handle_tool_result → 即时产出工具结果事件
        参数：
        - result: 调度器产出的任务结果
        """

        if result.task_type == "text":
            await self.handle_json_result(result)
        elif result.task_type == "bilingual":
            await self.handle_bilingual_result(result)
        elif result.task_type == "motion":
            await self.handle_motion_result(result)
        elif result.task_type == "tool_call":
            print(f"[V3] 工具调用事件: {result}")
            self.handle_tool_call_result(result)
        elif result.task_type == "tool_result":
            print(f"[V3] 工具结果事件: {result}")
            self.handle_tool_result_result(result)

    def handle_tool_call_result(self, result: TaskResult):
        """
        处理工具调用事件：分配全局输出序号，直接存入有序队列

        不再使用独立的 tool_events 列表，而是参与全局 strict ordering，
        确保 tool_call 不会被提前排出（必须等前面序号的句子事件包就绪）。
        """
        seq = self._output_seq
        self._output_seq += 1
        # 存储原始 ToolCallEvent / ToolResultEvent 列表，
        # 由上层 chat() 遍历时通过 _tools_response_handler 转换
        self._ordered_outputs[seq] = result.data

    def handle_tool_result_result(self, result: TaskResult):
        """处理工具结果事件：同 handle_tool_call_result，参与全局排序"""
        seq = self._output_seq
        self._output_seq += 1
        self._ordered_outputs[seq] = result.data

    def drain_ordered(self) -> list[ChatResponse | ToolCallEvent | ToolResultEvent]:
        """
        按全局输出序号排序，排出所有连续就绪的事件

        1. 先检查 sentence_events 中是否有完整就绪的句子包
           （text + audio + motion_frame 全部到位），将其移入 _ordered_outputs
        2. 工具事件已在 handle_tool_*_result 时直接写入 _ordered_outputs
        3. 按 _expected_seq 递增顺序取出所有连续就绪的输出

        返回：
        - 有序的混合事件列表（ChatResponse | ToolCallEvent | ToolResultEvent）
        """
        # 步骤 1：将已完整就绪的句子事件包移入全局队列
        while True:
            sid = self.expected_sentence_id
            current = self.sentence_events.get(sid)
            if not current:
                break
            if not all(et in current for et in self.event_order):
                break
            # 句子包已完整：按其预分配的 output_seq 移入全局队列
            seq = self._sid_to_seq.get(sid)
            if seq is None:
                seq = sid  # 兜底：直接使用 sentence_id 作为序号
            self._ordered_outputs[seq] = list(current.values())
            self.sentence_events.pop(sid, None)
            self.expected_sentence_id += 1

        # 步骤 2：按 _expected_seq 递增顺序，取出所有连续就绪的输出
        ready: list[ChatResponse | ToolCallEvent | ToolResultEvent] = []
        while self._expected_seq in self._ordered_outputs:
            ready.extend(self._ordered_outputs.pop(self._expected_seq))
            self._expected_seq += 1
        return ready


class V3ChatService:
    """
    V3 聊天服务

    提供 V3 版本的聊天流式输出接口。
    支持 Function Calling 工具调用。
    """

    def __init__(self):
        """初始化 V3 聊天服务"""
        self.assistant_service = AssistantService()
        self.integration: ToolCallIntegration | None = None

    def set_integration(self, integration: ToolCallIntegration):
        """
        设置工具调用集成

        允许外部注入 ToolCallIntegration 实例，使 V3ChatService 支持工具调用。
        """
        self.integration = integration

    def create_scheduler(self) -> TaskScheduler:
        """
        创建 V3 信息调度器

        注册所有需要的任务：
        - text: 文本生成任务（优先级 100）
        - bilingual: 双语翻译任务（优先级 150，仅在 GSV 语言非中文时注册）
        - motion: 动作标签任务（优先级 200）

        返回：
        - 配置好的 TaskScheduler 实例
        """

        scheduler = TaskScheduler()

        current_assistant = self.assistant_service.get_current_assistant()
        if not current_assistant:
            return scheduler

        scheduler.add_task(create_text_task(priority=100))

        lang = current_assistant.agent_config.gsvSetting.textLang

        if lang != "zh" and lang in ("en", "ja"):
            scheduler.add_task(create_bilingual_task(target_lang=lang, priority=150))

        expressions = load_expressions(current_assistant.agent_name)

        action_prompt = f"""
    可用动作标签：
    {', '.join([f"{action}: {desc}" for action, desc in ACTION_DESCRIPTIONS.items()])}
    可用表情：
    {[expr.name for expr in expressions]}
    """

        scheduler.add_task(
            create_motion_task(available_actions=action_prompt, priority=200)
        )

        return scheduler

    def _tools_response_handler(
        self, tool_event: ToolCallEvent | ToolResultEvent
    ) -> ToolCallResponse | ToolResultResponse | None:
        if isinstance(tool_event, ToolCallEvent):
            return ToolCallResponse(
                call_id=tool_event.call_id,
                tool_name=tool_event.tool_name,
                arguments=tool_event.arguments,
            )
        elif isinstance(tool_event, ToolResultEvent):
            return ToolResultResponse(
                tool_call_id=tool_event.call_id,
                tool_name=tool_event.tool_name,
                arguments=tool_event.arguments,
                success=tool_event.success,
                result=tool_event.content,
                duration_ms=tool_event.duration_ms,
            )

    def _to_response(
        self,
        event: ChatResponse | ToolCallEvent | ToolResultEvent,
    ) -> FullChatResponse | None:
        """将内部事件转换为前端响应，工具事件通过 _tools_response_handler 转换"""
        if isinstance(event, (ToolCallEvent, ToolResultEvent)):
            return self._tools_response_handler(event)
        return event

    async def _stream_pipeline_results(
        self,
        pipeline,
        ctx: V3MotionChatContext,
        start_time: float,
    ) -> AsyncGenerator[FullChatResponse]:
        """
        后台执行管道 + 轮询 drain_ordered，持续产出有序事件

        管道执行放入后台 task，结果通过队列传递。
        主循环用 wait_for(queue.get, timeout=0.3) 轮询，
        在 execute() 被工具执行的长时间 await 阻塞时，
        仍能定期调用 drain_ordered() 释放已就绪的事件。
        """
        queue: asyncio.Queue[TaskResult | None] = asyncio.Queue()

        async def _feed():
            """
            后台执行管道，将管道输出结果放入队列
            """
            async for result in pipeline.execute():
                await queue.put(result)
            # 管道执行完成，放入 None 作为结束标记
            await queue.put(None)

        feed_task = asyncio.create_task(_feed())
        # 延迟标记
        delay_flag = False

        def _collect(events):
            """
            收集 drain_ordered() 的事件，转换为 FullChatResponse 列表
            """
            nonlocal delay_flag
            results: list[FullChatResponse] = []
            for event in events:
                if not delay_flag:
                    logger.info(
                        f"[V3] 首条回复已生成，耗时 {time.time() - start_time:.2f} 秒"
                    )
                    delay_flag = True
                response = self._to_response(event)
                if response is not None:
                    results.append(response)
            return results

        try:
            while True:
                try:
                    result = await asyncio.wait_for(queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    for response in _collect(ctx.drain_ordered()):
                        yield response
                    continue

                if result is None:
                    break

                await ctx.handle_result(result)

                for response in _collect(ctx.drain_ordered()):
                    yield response
        finally:
            if not feed_task.done():
                feed_task.cancel()

    async def chat(self, params: ChatData) -> AsyncGenerator[FullChatResponse]:
        """
        V3 版本聊天流式输出

        使用信息调度中心，单次 LLM 调用同时生成文本和动作标签。

        工具调用: 若 self.integration 已被设置（调用前由外部注入），
        LLM 可在对话中自主调用工具；未设置则不启用工具。

        Args:
            params: 聊天请求参数
        """
        start_time = time.time()

        agent = self.assistant_service.get_current_assistant()
        if not agent:
            logger.error("当前没有加载助手")
            yield ErrorResponse(error_code="NO_ASSISTANT", data="当前没有加载助手")
            return

        user_message = params.msg[-1]["content"]
        history_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": await agent.get_context(
                    msg=user_message, is_sleep_mode=params.is_sleep_mode
                ),
            },
            *agent.get_history(),
        ]
        # history_messages = []

        scheduler = self.create_scheduler()
        pipeline = scheduler.create_task_pipeline(
            system_context=agent.prompt,
            history_messages=history_messages,
            user_message=user_message,
            tools=self.integration.get_tools() if self.integration else None,
            tool_handler=(
                self.integration.process_tool_calls if self.integration else None
            ),
        )

        ctx = V3MotionChatContext(tts_lang=agent.agent_config.gsvSetting.textLang)

        try:
            # 异步迭代管道结果，产出有序事件
            async for response in self._stream_pipeline_results(
                pipeline, ctx, start_time
            ):
                yield response
            # 等待未完成的异步任务完成（如音频生成、动作处理等）
            await ctx.wait_for_completion()
            # 等异步任务完成后，输出剩余就绪事件
            for event in ctx.drain_ordered():
                response = self._to_response(event)
                if response is not None:
                    yield response

            full_text = ctx.get_full_text()
            # 输出最终的 DoneResponse，包含完整文本
            yield DoneResponse(full_text=full_text)
            # 将用户消息和助手完整文本存入数据库，异步执行，不阻塞主流程
            asyncio.create_task(
                agent.add_msg(user_msg=user_message, assistant_msg=full_text)
            )

        except Exception as e:
            for task in list(ctx.pending_tasks):
                task.cancel()
            logger.error(f"[V3] 处理数据时出错: {e}", exc_info=True)
            yield ErrorResponse(error_code="500", data=f"处理数据时出错: {e}")
