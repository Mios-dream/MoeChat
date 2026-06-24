"""
聊天基础组件

提供所有聊天版本共享的基础组件：
- TTSData: TTS 数据类
- StreamProcessor: 流式处理器（按句分割，触发回调）
- 事件处理函数（文本、音频、动作事件封装）
- 事件聚合和输出函数

设计原则：
- 基础组件与版本无关，可在 V2/V3/V4 中复用
- 所有事件格式统一，便于前端处理
"""

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
from my_utils.log import logger
from core.llm.llm_client import LLMClient
from my_utils.tool_manager import ToolManager
from services.assistant_service import AssistantService
from pydantic import BaseModel
from core.assistant import Assistant

# ============================================================
# 全局服务实例
# ============================================================

assistant_service = AssistantService()


# ============================================================
# 常量定义
# ============================================================

# 句尾标点符号模式
SENTENCE_END_PATTERNS = ("。", "！", "？", "……\n", "……", "...\n", "...")

# 最小句子长度（避免太短的句子触发 TTS）
MIN_SENTENCE_LENGTH = 6


# ============================================================
# 数据类
# ============================================================


class TTSData(BaseModel):
    """
    TTS 数据类

    属性：
    - text: 待合成的文本
    - ref_audio: 参考音频路径
    - ref_text: 参考文本
    """

    text: str
    ref_audio: str
    ref_text: str


# ============================================================
# 流式处理器
# ============================================================


class StreamProcessor:
    """
    流式处理器

    职责：
    1. 接收 LLM 的 token 流
    2. 检测句子边界（句尾标点、括号完整性）
    3. 按句子触发回调

    特性：
    - 支持括号段完整性检测
    - 支持跨 token 的句子边界检测
    - 每个句子有唯一 ID

    使用示例：
    ```python
    processor = StreamProcessor()

    # 注册句子完成回调
    processor.register_sentence_complete_callback(on_sentence_complete)

    # 处理 token 流
    async for chunk in llm_stream:
        await processor.process_text_chunk(chunk)

    # 处理剩余文本
    await processor.process_remaining_text()
    ```
    """

    def __init__(self, emotion_processed: bool = False):
        """
        初始化流式处理器

        参数：
        - emotion_processed: 是否已处理表情包
        """
        # 用户输入的原始消息
        self.user_msg = ""

        # 全部消息（AI 可能回复多条语句）
        self.full_msg: list[str] = []

        # 普通文本缓存（不包含括号段）
        self.sentence_buffer = ""

        # 括号段缓存：括号内容单独成段，优先保证完整
        self.bracket_buffer = ""
        self.segment_bracket_stack: list[str] = []

        # 状态标志
        self.llm_done = False
        self.tasks_done = False

        # 标记是否需要处理表情包
        self.emotion_processed = emotion_processed

        # 句子跟踪
        self.sentence_count = 1  # 当前句子计数

        # 句子完成时的回调函数列表
        # 签名：callback(sentence_id: int, cleaned_text: str, tts_text: str)
        self.sentence_complete_callbacks: list[Callable[[int, str, str], Any]] = []

        # 动作历史状态管理
        self.motion_history: dict[int, dict[str, float]] = {}  # 句子ID -> 动作参数

        # 跨流式片段跟踪括号作用域
        self.tts_bracket_stack: list[str] = []

    @staticmethod
    def _is_sentence_end_at(text: str, idx: int) -> int:
        """
        检测指定位置是否为句尾

        参数：
        - text: 待检测文本
        - idx: 检测位置索引

        返回：
        - 命中长度（0 表示未命中）
        """
        for pattern in sorted(SENTENCE_END_PATTERNS, key=len, reverse=True):
            if text.startswith(pattern, idx):
                return len(pattern)
        return 0

    def _extract_plain_segments(self, force_flush: bool = False) -> list[str]:
        """
        从普通文本缓存中提取可输出片段

        参数：
        - force_flush: 是否强制提取剩余文本

        返回：
        - 可输出的文本片段列表
        """
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

    async def _emit_segment(self, message_chunk: str):
        """
        输出一个片段并触发回调

        参数：
        - message_chunk: 待输出的完整单句文本片段
        """
        if not message_chunk:
            return

        cleaned = "".join(message_chunk.split())
        self.full_msg.append(cleaned)
        tts_cleaned = "".join(self._filter_tts_text(message_chunk).split())

        await asyncio.gather(
            *[
                callback(self.sentence_count, cleaned, tts_cleaned)
                for callback in self.sentence_complete_callbacks
            ]
        )

        self.sentence_count += 1

    def _filter_tts_text(self, text: str) -> str:
        """
        过滤供 TTS 使用的文本

        移除括号及其内部内容（支持跨句/跨片段）

        参数：
        - text: 原始文本

        返回：
        - 过滤后的文本
        """
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
        """
        查询文字中的情感字段

        参数：
        - msg: 消息文本

        返回：
        - 情感标签，未找到返回 None
        """
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
        处理文本块

        检测完整句子，并推送给前端和任务队列。

        参数：
        - chunk: 文本块
        """
        bracket_pairs = {
            "(": ")",
            "（": "）",
            "[": "]",
            "【": "】",
            "{": "}",
        }
        opening_brackets = set(bracket_pairs.keys())

        for ch in chunk:
            if self.segment_bracket_stack:
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
        处理缓存的剩余文本

        在流结束时调用，确保所有内容都被处理。
        """
        for plain_segment in self._extract_plain_segments(force_flush=False):
            await self._emit_segment(plain_segment)

        for plain_segment in self._extract_plain_segments(force_flush=True):
            await self._emit_segment(plain_segment)

        if self.bracket_buffer.strip():
            await self._emit_segment(self.bracket_buffer)
            self.bracket_buffer = ""
            self.segment_bracket_stack = []

        if self.llm_done:
            self.tasks_done = True

    def register_sentence_complete_callback(
        self, callback: Callable[[int, str, str], Any]
    ):
        """
        注册句子完成时的回调函数

        参数：
        - callback: 回调函数，签名 (sentence_id, cleaned_text, tts_text)
        """
        self.sentence_complete_callbacks.append(callback)


# ============================================================
# TTS 任务
# ============================================================


async def tts_task(tts_data: TTSData) -> bytes | None:
    """
    执行 TTS 合成任务

    参数：
    - tts_data: TTS 数据

    返回：
    - 音频字节数据，失败返回 None
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
            byte_data = await ttsService.local_gsv_tts(data=data)
        else:
            byte_data = await ttsService.api_gsv_tts(data)
        logger.info(f"合成完成的耗时: {time.time() - start_time}")
        return byte_data
    except Exception as e:
        logger.error(f"[错误] TTS执行失败，错误信息: {e}，文本: {data['text']}")
        return None


# ============================================================
# 事件封装函数
# ============================================================


async def text_wrapper(sentence_id: int, sentence_text: str) -> dict[str, Any]:
    """
    封装文本事件

    参数：
    - sentence_id: 句子 ID
    - sentence_text: 句子文本

    返回：
    - 文本事件字典
    """
    return {
        "type": "text",
        "sentence_id": sentence_id,
        "message": sentence_text,
        "timestamp_ms": time.time() * 1000,
        "done": False,
    }


async def tts_wrapper(
    tts_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    tts_text: str,
    processor: StreamProcessor | None = None,
) -> dict[str, Any]:
    """
    封装音频事件

    参数：
    - tts_semaphore: TTS 并发控制信号量
    - sentence_id: 句子 ID
    - sentence_text: 原始句子文本
    - tts_text: 过滤后的 TTS 文本
    - processor: 流式处理器（用于获取情绪）

    返回：
    - 音频事件字典
    """
    audio_event: dict[str, Any] = {
        "type": "audio",
        "sentence_id": sentence_id,
        "message": tts_text,
        "source_text": sentence_text,
        "file": "",
        "timestamp_ms": time.time() * 1000,
        "done": False,
    }

    async with tts_semaphore:
        try:
            agent = assistant_service.get_current_assistant()
            if not agent:
                logger.error("[错误] 当前没有加载助手")
                return audio_event

            emotion = processor._get_emotion(sentence_text) if processor else None
            ref_audio = ""
            ref_text = ""

            if emotion:
                agent_config = agent.agent_config.gsvSetting.extraRefAudio
                if emotion in agent_config:
                    ref_audio = agent_config[emotion][0]
                    ref_text = agent_config[emotion][1]

            tts_text_clean = re.sub(r"[…''" "'\"—\n\r\t\f ]", "", tts_text)

            if tts_text_clean:
                tts_data_item = TTSData(
                    text=tts_text_clean,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                audio_data = await tts_task(tts_data_item)
                if audio_data:
                    audio_event["file"] = base64.b64encode(audio_data).decode("utf-8")
                else:
                    logger.warning(f"[TTS] 生成空音频，sentence_id: {sentence_id}")
            else:
                logger.info(f"[TTS] 无可读语音内容，sentence_id: {sentence_id}")

        except Exception as e:
            logger.error(f"[TTS] 执行失败: {e}")

    return audio_event


# ============================================================
# 事件聚合和输出
# ============================================================


def store_sentence_event(
    sentence_events: dict[int, dict[str, dict[str, Any]]],
    sentence_id: int,
    event_key: str,
    payload: dict[str, Any],
):
    """
    按句子聚合事件

    参数：
    - sentence_events: 事件存储
    - sentence_id: 句子 ID
    - event_key: 事件类型（"text", "audio", "motion_frame"）
    - payload: 事件数据
    """
    if sentence_id not in sentence_events:
        sentence_events[sentence_id] = {}
    sentence_events[sentence_id][event_key] = payload


def drain_ready_sentence_events(
    sentence_events: dict[int, dict[str, dict[str, Any]]],
    expected_sentence_id: int,
    event_order: tuple[str, ...],
) -> tuple[int, list[dict[str, Any]]]:
    """
    按 sentence_id 递增释放完整事件集合

    参数：
    - sentence_events: 事件存储
    - expected_sentence_id: 预期的下一个句子 ID
    - event_order: 事件类型顺序

    返回：
    - (下一个预期句子 ID, 可输出的事件列表)
    """
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


def to_sse(payload: dict) -> str:
    """
    将字典转换为 SSE 格式

    参数：
    - payload: 事件字典

    返回：
    - SSE 格式字符串
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ============================================================
# 统一消息构建
# ============================================================


async def build_chat_messages(
    agent: Assistant,
    params: chat_data,
    system_prompt_extra: str = "",
) -> tuple[Assistant, list[dict[str, str]], str]:
    """
    统一构建聊天消息列表

    为所有聊天版本提供统一的消息构建方式。

    参数：
    - agent: 助手实例
    - params: 聊天请求参数
    - system_prompt_extra: 额外的系统提示词（如动作生成相关）

    返回：
    - (助手实例, 消息列表, 用户消息)
    """
    user_message = params.msg[-1]["content"]

    # 构建系统提示词
    system_prompt = agent.prompt
    if system_prompt_extra:
        system_prompt += "\n\n" + system_prompt_extra

    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        *agent.get_history(),
        {
            "role": "user",
            "content": await agent.get_context(
                msg=user_message, is_sleep_mode=params.is_sleep_mode
            ),
        },
    ]

    return agent, messages, user_message


def get_motion_system_prompt() -> str:
    """
    获取动作生成相关的系统提示词

    返回：
    - 动作生成提示词
    """
    try:
        from core.expression_generator.atomic_actions import get_action_vocab

        action_vocab = get_action_vocab()
        return f"""【动作生成说明】
在回复时，请为每个句子选择合适的动作标签。

【可用动作】
{action_vocab}

【动作选择指南】
- 根据句子的情感和内容选择合适的动作
- 动作可以组合使用（如 smile, nod 表示微笑点头）
- 在句子末尾用括号标注动作，例如：你好呀~（smile, nod）"""
    except ImportError:
        return ""


# ============================================================
# LLM 流处理
# ============================================================

# 全局 LLM 客户端实例（用于流式工具调用）
_llm_client: LLMClient = LLMClient(model_key="ChatLLM")


async def start_llm_task(msg: list, stream_processor: StreamProcessor):
    """
    启动 LLM 流式任务

    参数：
    - msg: 消息列表
    - stream_processor: 流式处理器
    """
    start_time = time.time()
    logger.info("[LLM] 开始处理")

    first_print_time_flag = True

    async def stream_with_timing(chunk: str):
        nonlocal first_print_time_flag
        if first_print_time_flag:
            first_print_time_flag = False
            logger.info(f"[大模型延迟]{time.time() - start_time}")
        await stream_processor.process_text_chunk(chunk)

    try:

        tools = ToolManager.get_openai_tools()

        async for chunk in _llm_client.stream_with_tools(
            messages=msg,
            tools=tools,
            tool_executor=ToolManager,
        ):
            await stream_with_timing(chunk)
    except Exception as e:
        logger.error(f"[LLM] 流处理异常: {e}", exc_info=True)
    finally:
        await stream_processor.process_remaining_text()
        stream_processor.llm_done = True
        logger.info("[LLM] 流处理完成")


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
