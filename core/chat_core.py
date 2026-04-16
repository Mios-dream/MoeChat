from collections.abc import Callable
import json
import os
import time
import asyncio
import base64
import re
from typing import Any

from Config import Config
from models.dto.chat_request import chat_data
from services.tts_service import ttsService
from my_utils import config_manager as CConfig
from my_utils.live2d_parameter_load import load_live2d_parameters
from my_utils.log import logger
from my_utils.llm_request import chat_llm_request_stream
from services.assistant_service import AssistantService
from services.expression_generator_service import ExpressionGenerator
from pydantic import BaseModel

assistant_service = AssistantService()

SENTENCE_END_PATTERNS = ("。", "！", "？", "……\n", "……", "...\n", "...")
MIN_SENTENCE_LENGTH = 6

agent_expression_generators = {}  # 全局缓存助手对应的动作生成器实例，避免重复创建


class TTSData(BaseModel):
    """
    TTS数据类

    Attributes:
        text (str): 待合成的文本
        ref_audio (str): 参考音频路径
        ref_text (str): 参考文本
    """

    text: str
    ref_audio: str
    ref_text: str


class StreamProcessor:
    """
    改进版流式处理器 - 支持优先级、时间戳、句子ID跟踪

    核心改进：
    1. 每个事件包含时间戳，用于前端同步
    2. 每句文本有唯一ID，便于音频对齐
    3. 检测到句尾时立即触发TTS和动作规划（而非等待后续token）
    """

    def __init__(
        self,
        emotion_processed: bool = False,
    ):
        # 全部消息，ai可能回复多条语句
        self.full_msg: list[str] = []

        # 普通文本缓存（不包含括号段）
        self.sentence_buffer = ""

        # 括号段缓存：括号内容单独成段，优先保证完整
        self.bracket_buffer = ""
        self.segment_bracket_stack: list[str] = []

        # 创建标志，用于记录任务是否已完成
        self.llm_done = False
        self.tasks_done = False

        # 标记是否需要处理表情包
        self.emotion_processed = emotion_processed

        # 句子跟踪
        self.sentence_count = 1  # 当前句子计数
        self.sentence_complete_callbacks: list[Callable[[int, str, str], Any]] = (
            []
        )  # 句子完成时的回调函数

        # 时间戳与性能监控
        self.first_token_recorded = False
        # 新增：动作历史状态管理
        self.motion_history: dict[int, str] = {}  # 句子ID -> 动作描述
        # TTS 专用：跨流式片段跟踪括号作用域，保证括号内容可跨句过滤
        self.tts_bracket_stack: list[str] = []

    @staticmethod
    def _is_sentence_end_at(text: str, idx: int) -> int:
        """若 idx 处命中句尾模式，返回命中长度；否则返回 0。"""
        for pattern in sorted(SENTENCE_END_PATTERNS, key=len, reverse=True):
            if text.startswith(pattern, idx):
                return len(pattern)
        return 0

    def _extract_plain_segments(self, force_flush: bool = False) -> list[str]:
        """从普通文本缓存中提取可输出片段：句尾命中且长度达标。"""
        ready: list[str] = []

        while True:
            boundary_end = -1
            i = 0
            while i < len(self.sentence_buffer):
                hit_len = self._is_sentence_end_at(self.sentence_buffer, i)
                if hit_len > 0:
                    boundary_end = i + hit_len
                    candidate = self.sentence_buffer[:boundary_end]
                    cleaned = "".join(candidate.split())
                    if len(cleaned) >= MIN_SENTENCE_LENGTH:
                        break

                    # 该句尾过短，继续查找后续句尾，避免阻断整个缓冲区的切分
                    i = boundary_end
                    boundary_end = -1
                    continue
                i += 1

            if boundary_end < 0:
                break

            candidate = self.sentence_buffer[:boundary_end]
            ready.append(candidate)
            self.sentence_buffer = self.sentence_buffer[boundary_end:]

        if force_flush and self.sentence_buffer.strip():
            ready.append(self.sentence_buffer)
            self.sentence_buffer = ""

        return ready

    @staticmethod
    def _bracket_pairs() -> dict[str, str]:
        return {
            "(": ")",
            "（": "）",
            "[": "]",
            "【": "】",
            "{": "}",
        }

    async def _emit_segment(self, message_chunk: str):
        """统一输出一个片段并触发回调。"""
        if not message_chunk:
            return

        self.full_msg.append(message_chunk)
        cleaned = "".join(message_chunk.split())
        tts_cleaned = "".join(self._filter_tts_text(message_chunk).split())

        await asyncio.gather(
            *[
                callback(self.sentence_count, cleaned, tts_cleaned)
                for callback in self.sentence_complete_callbacks
            ]
        )

        self.sentence_count += 1

    def _filter_tts_text(self, text: str) -> str:
        """过滤供 TTS 使用的文本：移除括号及其内部内容（支持跨句/跨片段）。"""
        bracket_pairs = {
            "(": ")",
            "（": "）",
            "[": "]",
            "【": "】",
            "{": "}",
        }
        opening = set(bracket_pairs.keys())

        filtered_chars: list[str] = []
        for ch in text:
            if self.tts_bracket_stack:
                # 在括号作用域内：支持嵌套与闭合
                if ch in opening:
                    self.tts_bracket_stack.append(bracket_pairs[ch])
                elif ch == self.tts_bracket_stack[-1]:
                    self.tts_bracket_stack.pop()
                continue

            if ch in opening:
                self.tts_bracket_stack.append(bracket_pairs[ch])
                continue

            filtered_chars.append(ch)

        return "".join(filtered_chars)

    def _get_emotion(self, msg: str) -> str | None:
        """查询文字中的情感字段"""
        agent = assistant_service.get_current_assistant()
        if not agent:
            logger.error("[错误] 当前没有加载助手")
            return None
        res = re.findall(r"\[(.*?)\]", msg)
        if len(res) > 0:
            match = res[-1]
            if match and agent.agent_config.gsvSetting.extraRefAudio:
                if match in agent.agent_config.gsvSetting.extraRefAudio:
                    return match

    async def process_text_chunk(self, chunk: str):
        """
        改进版文本块处理：检测完整句子，并推送给前端和任务队列

        改进逻辑：
        1. 接受一个文本块，判断是否构成一个完整句子（根据结尾符号），然后立即推送给前端（SSE）和TTS/动作规划队列
        2. 每个token带有时间戳和sentence_id
        3. 检测到句尾符号时，立即触发回调（TTS+动作规划）
        """

        bracket_pairs = self._bracket_pairs()
        opening_brackets = set(bracket_pairs.keys())

        for ch in chunk:
            if self.segment_bracket_stack:
                # 括号作用域内：无视句尾规则，直到括号完整闭合再输出
                self.bracket_buffer += ch
                if ch in opening_brackets:
                    self.segment_bracket_stack.append(bracket_pairs[ch])
                elif ch == self.segment_bracket_stack[-1]:
                    self.segment_bracket_stack.pop()
                    if not self.segment_bracket_stack:
                        await self._emit_segment(self.bracket_buffer)
                        self.bracket_buffer = ""
                continue

            if ch in opening_brackets:
                # 进入括号前先尝试释放已满足条件的普通句段
                for plain_segment in self._extract_plain_segments(force_flush=False):
                    await self._emit_segment(plain_segment)

                self.segment_bracket_stack.append(bracket_pairs[ch])
                self.bracket_buffer = ch
                continue

            self.sentence_buffer += ch
            for plain_segment in self._extract_plain_segments(force_flush=False):
                await self._emit_segment(plain_segment)

    async def process_remaining_text(self):
        """
        处理缓存的剩余文本（即没有结尾符号的情况）
        """
        # 先释放普通文本中已命中句尾的片段
        for plain_segment in self._extract_plain_segments(force_flush=False):
            await self._emit_segment(plain_segment)

        # 再强制释放普通文本尾部残留
        for plain_segment in self._extract_plain_segments(force_flush=True):
            await self._emit_segment(plain_segment)

        # 最后处理未闭合或尾部括号片段，避免内容丢失
        if self.bracket_buffer.strip():
            await self._emit_segment(self.bracket_buffer)
            self.bracket_buffer = ""
            self.segment_bracket_stack = []

        # 如果LLM已经完成，标记所有任务完成，且当前回调任务也完成也算完成
        if self.llm_done:
            self.tasks_done = True

    def register_sentence_complete_callback(
        self, callback: Callable[[int, str, str], Any]
    ):
        """注册句子完成时的回调函数 (用于触发TTS和动作规划)"""
        self.sentence_complete_callbacks.append(callback)


async def tts_task(tts_data: TTSData) -> bytes | None:
    """
    构建tts任务

    Parameters
        tts_data : list
            包含参考音频、参考文本和合成文本的列表
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[错误] 当前没有加载助手")
        return None

    if len(tts_data.text) == 0:
        return None
    logger.info(f"[tts文本]{tts_data.text}")
    assistant_asset_base_path = f"{Config.BASE_AGENTS_PATH}/{agent.agent_name}/assets"
    is_api = CConfig.config["TTS"]["mode"] == "api"

    data = {
        "text": tts_data.text,
        "text_lang": agent.agent_config.gsvSetting.textLang,
        "ref_audio_path": (
            tts_data.ref_text or agent.agent_config.gsvSetting.refAudioPath
            if is_api
            else os.path.join(
                assistant_asset_base_path,
                "audio",
                (tts_data.ref_text or agent.agent_config.gsvSetting.refAudioPath),
            )
        ),
        "prompt_text": tts_data.ref_text or agent.agent_config.gsvSetting.promptText,
        "prompt_lang": agent.agent_config.gsvSetting.promptLang,
    }
    try:
        start_time = time.time()
        if not is_api:
            print(data)
            byte_data = await ttsService.local_gsv_tts(
                data=data,
            )
        else:
            byte_data = await ttsService.api_gsv_tts(data)
        logger.info(f"合成完成的耗时: {time.time() - start_time}")
        return byte_data
    except Exception as e:
        logger.error(f"[错误] TTS执行失败，错误信息: {e}，文本: {data['text']}")
        return None


# 文本事件封装
async def text_wrapper(sentence_id: int, sentence_text: str) -> dict[str, Any]:
    current_time_ms = time.time() * 1000
    return {
        "type": "text",
        "sentence_id": sentence_id,
        "message": sentence_text,
        "timestamp_ms": current_time_ms,
        "done": False,
    }


# TTS任务封装，使用信号量控制并发，失败时返回空音频事件
async def tts_wrapper(
    tts_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
    processor: StreamProcessor,
) -> dict[str, Any]:
    current_time_ms = time.time() * 1000
    audio_event: dict[str, Any] = {
        "type": "audio",
        "sentence_id": sentence_id,
        "message": tts_text,
        "source_text": sentence_text,
        "file": "",
        "timestamp_ms": current_time_ms,
        "done": False,
    }

    async with tts_semaphore:
        try:
            agent = assistant_service.get_current_assistant()
            if not agent:
                logger.error("[错误] 当前没有加载助手")
                return audio_event
            # 获取情绪和参考音频
            emotion = processor._get_emotion(sentence_text)
            ref_audio = ""
            ref_text = ""
            if emotion:
                agent_config = agent.agent_config.gsvSetting.extraRefAudio
                if emotion in agent_config:
                    ref_audio = agent_config[emotion][0]
                    ref_text = agent_config[emotion][1]
            # 清除文本中的特殊符号
            tts_text = re.sub(r"[…‘’“”'\"—\n\r\t\f ]", "", tts_text)

            if tts_text:
                # 设置TTS数据并执行TTS任务
                tts_data_item = TTSData(
                    text=tts_text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                audio_data = await tts_task(tts_data_item)
                if audio_data:
                    audio_event["file"] = base64.b64encode(audio_data).decode("utf-8")
                else:
                    logger.warning(
                        f"[动作规划] TTS生成空音频，sentence_id: {sentence_id}"
                    )
            else:
                logger.info(
                    f"[动作规划] 句子无可读语音内容，发送空音频事件，sentence_id: {sentence_id}"
                )

        except Exception as e:
            logger.error(f"[动作规划] 启动失败: {e}")
        return audio_event


# 动作规划任务封装，使用信号量控制并发，失败时返回空动作事件
async def motion_wrapper(
    motion_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    motion_generator: ExpressionGenerator | None,
    motion_history: dict | None = None,  # 新增：历史动作状态
) -> dict[str, Any]:
    logger.info(f"准备生成动作，句子ID: {sentence_id}, 内容: {sentence_text}")

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
                    f"[动作规划] 动作生成器不可用，发送空动作事件，sentence_id: {sentence_id}"
                )
                return motion_event

            # 构建上下文信息，包含历史动作状态
            context = ""
            previous_action = ""
            if motion_history and sentence_id > 0:
                prev_actions = []
                for i in range(max(0, sentence_id - 3), sentence_id):  # 查看前3句的动作
                    if i in motion_history:
                        prev_actions.append(f"第{i+1}句动作: {motion_history[i]}")
                if prev_actions:
                    context = "历史动作序列:\n" + "\n".join(prev_actions) + "\n"
                    context += "要求: 新动作要与历史动作保持连贯性，避免突兀的跳跃变化"

                # 获取前一个动作状态
                if sentence_id - 1 in motion_history:
                    previous_action = motion_history[sentence_id - 1]

            sentence_text_clean = sentence_text.replace(" ", "").replace("\n", "")

            duration_time = len(sentence_text_clean) * 100  # 粗略估计每个字100ms

            frame_plans = await motion_generator.generate_motion_plan(
                speech_text=sentence_text,
                context=context,
                speech_duration_ms=duration_time,  # 粗略估计每个字100ms
                timeout_seconds=5,
                previous_action=previous_action,  # 传入前一个动作状态
            )
            motion_event["duration"] = duration_time

            logger.info(f"生成的动作帧: {frame_plans}")

            frame = await motion_generator.generate_tts_motion_frame_with_plan(
                frame_plans
            )
            # 记录动作到历史状态
            if motion_history is not None and frame_plans:
                motion_history[sentence_id] = frame_plans[0].get("action", "自然动作")

            motion_event["motions"] = frame or []
        except Exception as e:
            logger.warning(f"[动作规划] 第{sentence_id + 1}句动作生成失败: {e}")
        return motion_event


def _store_sentence_event(
    sentence_events: dict[int, dict[str, dict[str, Any]]],
    sentence_id: int,
    event_key: str,
    payload: dict[str, Any],
):
    """按句子聚合事件。"""
    if sentence_id not in sentence_events:
        sentence_events[sentence_id] = {}
    sentence_events[sentence_id][event_key] = payload


def _drain_ready_sentence_events(
    sentence_events: dict[int, dict[str, dict[str, Any]]],
    expected_sentence_id: int,
    event_order: tuple[str, ...],
) -> tuple[int, list[dict[str, Any]]]:
    """按 sentence_id 递增释放完整事件集合。"""
    ready_payloads: list[dict[str, Any]] = []

    while True:
        current = sentence_events.get(expected_sentence_id)
        if not current:
            break
        if not all(event_type in current for event_type in event_order):
            break

        for event_type in event_order:
            ready_payloads.append(current[event_type])

        sentence_events.pop(expected_sentence_id, None)
        expected_sentence_id += 1

    return expected_sentence_id, ready_payloads


# 启动LLM任务，处理流式响应，并在完成时处理剩余文本
async def start_llm_task(msg, stream_processor: StreamProcessor):
    """
    将消息发送到大语言模型(LLM)并处理返回的流式响应

    只在流完全结束时处理剩余文本，确保最后一段输出被正确处理

    Args:
        msg: 消息列表
        stream_processor: 流式处理器
    """

    start_time = time.time()
    logger.info("[LLM]：开始处理")

    # 标记第一次打印时间
    first_print_time_flag = True

    # logger.info(f"发送给LLM的消息: {json.dumps(msg, ensure_ascii=False)}")

    try:
        # 正常处理所有流式chunk
        async for chunk in chat_llm_request_stream(msg):
            if first_print_time_flag:
                first_print_time_flag = False
                logger.info(f"[大模型延迟]{time.time() - start_time}")

            # 处理每个文本chunk
            await stream_processor.process_text_chunk(chunk)

    except Exception as e:
        logger.error(f"[LLM] 流处理异常: {e}")
    finally:
        # 无论正常结束还是异常，都在最后处理剩余文本
        await stream_processor.process_remaining_text()
        stream_processor.llm_done = True
        logger.info("[LLM] 流处理完成，处理剩余文本")


async def llm_chat_with_tts(params: chat_data):
    """
    改进版聊天流式输出：并行文字/语音，极低延迟同步

    1. 立即推送每个文字token（TTFT < 500ms）
    2. 检测到句尾时立即启动TTS和动作规划（< 100ms）
    3. 三路事件通过优先级队列并行推送（text>audio>motion）
    4. 所有事件带有时间戳和sentence_id，前端可精确同步
    """
    # 获取当前助手实例
    agent, msg_list_for_llm = await _get_agent_and_msg_list(params)
    if not agent or not msg_list_for_llm:
        logger.error("[错误] 当前没有加载助手 或 没有消息列表")
        return

    # 创建流处理器实例，规划任务
    processor = StreamProcessor()

    sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
    expected_sentence_id = 1
    event_order = ("text", "audio")

    # 任务跟踪集合，确保所有后台任务完成后才结束主循环
    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        # 添加到跟踪集合，任务完成后自动移除，避免存在未完成的任务导致主循环过早结束
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    # 并发控制信号量
    tts_semaphore = asyncio.Semaphore(1)  # 最多1个并发TTS任务，防止过高的资源占用

    # 句子完成时的回调函数：启动TTS和动作规划
    async def on_sentence_complete(sentence_id: int, sentence_text: str, tts_text: str):
        """当检测到句子完成时，立即启动TTS和动作规划"""

        async def create_text_event():
            payload = await text_wrapper(
                sentence_id=sentence_id,
                sentence_text=sentence_text,
            )
            _store_sentence_event(sentence_events, sentence_id, "text", payload)

        async def create_audio_event():
            payload = await tts_wrapper(
                tts_semaphore=tts_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                tts_text=tts_text,
                processor=processor,
            )
            _store_sentence_event(sentence_events, sentence_id, "audio", payload)

        track_task(asyncio.create_task(create_text_event()))
        track_task(asyncio.create_task(create_audio_event()))

    # 注册句子完成回调
    processor.register_sentence_complete_callback(on_sentence_complete)

    # 启动LLM流 和 TTS处理任务
    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        # 持续输出事件，直到 LLM 完成且后台任务全部结束。
        while True:
            expected_sentence_id, ready_payloads = _drain_ready_sentence_events(
                sentence_events=sentence_events,
                expected_sentence_id=expected_sentence_id,
                event_order=event_order,
            )
            for payload in ready_payloads:
                yield _to_sse(payload)

            if llm_task.done() and not pending_tasks and not sentence_events:
                break

            await asyncio.sleep(0.01)
        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": max(0, processor.sentence_count - 1),
            "full_text": ("".join(processor.full_msg) if processor.full_msg else ""),
            "done": True,
        }
        yield _to_sse(final_response)

        # 保存到助手上下文
        await agent.add_msg("".join(processor.full_msg))

    except Exception as e:
        llm_task.cancel()
        for task in list(pending_tasks):
            task.cancel()
        logger.error(f"处理数据时出错: {e}", exc_info=True)
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield _to_sse(error_response)


async def llm_chat_with_tts_and_motion(params: chat_data):
    """
    改进版聊天流式输出：并行文字/语音/动作，极低延迟同步

    1. 立即推送每个文字token（TTFT < 500ms）
    2. 检测到句尾时立即启动TTS和动作规划（< 100ms）
    3. 三路事件通过优先级队列并行推送（text>audio>motion）
    4. 所有事件带有时间戳和sentence_id，前端可精确同步
    """
    # 获取当前助手实例
    agent, msg_list_for_llm = await _get_agent_and_msg_list(params)
    if not agent or not msg_list_for_llm:
        logger.error("[错误] 当前没有加载助手 或 没有消息列表")
        return

    # 创建流处理器实例，规划任务
    processor = StreamProcessor()

    sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
    expected_sentence_id = 1
    event_order = ("text", "audio", "motion_frame")

    # 任务跟踪集合，确保所有后台任务完成后才结束主循环
    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        # 添加到跟踪集合，任务完成后自动移除，避免存在未完成的任务导致主循环过早结束
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    # 为当前助手创建动作生成器
    motion_generator: ExpressionGenerator | None = (
        await _get_agent_expression_generator(agent.agent_name)
    )

    if motion_generator is None:
        logger.warning("[动作规划] 未找到可用 Live2D 参数，将输出空动作事件")

    # 并发控制信号量
    tts_semaphore = asyncio.Semaphore(1)  # 最多1个并发TTS任务，防止过高的资源占用
    motion_semaphore = asyncio.Semaphore(4)  # 最多4个并发动作规划任务

    # 句子完成时的回调函数：启动TTS和动作规划
    async def on_sentence_complete(sentence_id: int, sentence_text: str, tts_text: str):
        """当检测到句子完成时，立即启动TTS和动作规划"""

        async def create_text_event():
            payload = await text_wrapper(
                sentence_id=sentence_id,
                sentence_text=sentence_text,
            )
            _store_sentence_event(sentence_events, sentence_id, "text", payload)

        async def create_audio_event():
            payload = await tts_wrapper(
                tts_semaphore=tts_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                tts_text=tts_text,
                processor=processor,
            )
            _store_sentence_event(sentence_events, sentence_id, "audio", payload)

        async def create_motion_event():
            payload = await motion_wrapper(
                motion_semaphore=motion_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                motion_generator=motion_generator,
                motion_history=processor.motion_history,
            )
            _store_sentence_event(sentence_events, sentence_id, "motion_frame", payload)

        track_task(asyncio.create_task(create_text_event()))
        track_task(asyncio.create_task(create_audio_event()))
        track_task(asyncio.create_task(create_motion_event()))

    # 注册句子完成回调
    processor.register_sentence_complete_callback(on_sentence_complete)

    # 启动LLM流 和 TTS处理任务
    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        # 持续输出事件，直到 LLM 完成且后台任务全部结束。
        while True:
            expected_sentence_id, ready_payloads = _drain_ready_sentence_events(
                sentence_events=sentence_events,
                expected_sentence_id=expected_sentence_id,
                event_order=event_order,
            )
            for payload in ready_payloads:
                yield _to_sse(payload)

            if llm_task.done() and not pending_tasks and not sentence_events:
                break

            await asyncio.sleep(0.01)
        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": max(0, processor.sentence_count - 1),
            "full_text": ("".join(processor.full_msg) if processor.full_msg else ""),
            "done": True,
        }
        yield _to_sse(final_response)

        # 保存到助手上下文
        await agent.add_msg("".join(processor.full_msg))

    except Exception as e:
        llm_task.cancel()
        for task in list(pending_tasks):
            task.cancel()
        logger.error(f"处理数据时出错: {e}", exc_info=True)
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield _to_sse(error_response)


def _to_sse(payload: dict) -> str:
    """将字典统一转换为 SSE 数据块。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _get_agent_and_msg_list(params: chat_data):
    """获取当前助手与输入消息，避免各流程重复代码。"""
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[错误] 当前没有加载助手")
        return None, None
    msg_list_for_llm = await agent.get_msg_data(params.msg[-1]["content"])
    return agent, msg_list_for_llm


async def _get_agent_expression_generator(agent_name: str) -> ExpressionGenerator:
    """获取助手对应的动作生成器实例，避免重复创建"""
    if agent_name not in agent_expression_generators:
        expression_generator = ExpressionGenerator(
            eye_open_binary=True,
            joint_motion_boost=1.25,
            tts_motion_keep_lip_sync=True,
        )
        expression_generator.update_parameters(await load_live2d_parameters(agent_name))
        agent_expression_generators[agent_name] = expression_generator

    return agent_expression_generators[agent_name]
