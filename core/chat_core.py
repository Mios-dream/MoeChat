from collections.abc import Callable
import json
import os
import time
import asyncio
import base64
import re
import heapq
from typing import Any
from dataclasses import dataclass

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

SENTENCE_END_PATTERNS = ("。", "！", "？", "...\n", "...")

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


@dataclass
class PriorityEvent:
    """优先级事件封装 - 用于SSE事件排序"""

    priority: int  # 0=text_token, 1=sentence_ready, 2=audio, 3=motion, 4=done
    timestamp_ms: float  # 事件时间戳 (ms)
    data: dict  # 事件数据

    def __lt__(self, other: "PriorityEvent") -> bool:
        """用于优先级队列排序"""
        if self.priority == other.priority:
            return self.timestamp_ms < other.timestamp_ms
        return self.priority < other.priority


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

        # 缓存不完整的句子
        self.sentence_buffer = ""

        # 创建标志，用于记录任务是否已完成
        self.llm_done = False
        self.tasks_done = False

        # 标记是否需要处理表情包
        self.emotion_processed = emotion_processed

        # 句子跟踪
        self.sentence_count = 1  # 当前句子计数
        self.sentence_complete_callbacks: list[Callable] = []  # 句子完成时的回调函数

        # 时间戳与性能监控
        self.first_token_recorded = False
        # 新增：动作历史状态管理
        self.motion_history: dict[int, str] = {}  # 句子ID -> 动作描述

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

        # 将新文本添加到缓冲区
        self.sentence_buffer += chunk

        # 检测句尾符号 (句子完成)
        has_sentence_end = any(
            self.sentence_buffer.endswith(p) for p in SENTENCE_END_PATTERNS
        )

        if has_sentence_end:
            # 立即处理完整句子
            message_chunk = self.sentence_buffer

            self.full_msg.append(message_chunk)
            # self.pending_sentences.append(message_chunk)

            # 触发句子完成回调
            for callback in self.sentence_complete_callbacks:
                await callback(self.sentence_count, message_chunk)

            # 递增句子计数器
            self.sentence_count += 1
            self.sentence_buffer = ""

    async def process_remaining_text(self):
        """
        处理缓存的剩余文本（即没有结尾符号的情况）
        """
        if len(self.sentence_buffer) > 0:
            self.full_msg.append(self.sentence_buffer)

            # 并发句子完成回调
            await asyncio.gather(
                *[
                    callback(self.sentence_count, self.sentence_buffer)
                    for callback in self.sentence_complete_callbacks
                ]
            )
            self.sentence_count += 1
            self.sentence_buffer = ""
            # 如果LLM已经完成，标记所有任务完成，且当前回调任务也完成也算完成
            if self.llm_done:
                self.tasks_done = True

    def register_sentence_complete_callback(self, callback: Callable[[int, str], Any]):
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

    msg = tts_data.text
    msg = re.sub(r"\(.*?\)|（.*?）|【.*?】|\[.*?\]|\{.*?\}", "", msg)
    msg = msg.replace(" ", "").replace("\n", "")
    # msg = clear_text(tts_data.text)
    if len(msg) == 0:
        return None
    logger.info(f"[tts文本]{msg}")
    assistant_asset_base_path = f"{Config.BASE_AGENTS_PATH}/{agent.agent_name}/assets"
    data = {
        "text": msg,
        "text_lang": agent.agent_config.gsvSetting.textLang,
        "ref_audio_path": os.path.join(
            assistant_asset_base_path,
            "audio",
            (tts_data.ref_text or agent.agent_config.gsvSetting.refAudioPath),
        ),
        "prompt_text": tts_data.ref_text or agent.agent_config.gsvSetting.promptText,
        "prompt_lang": agent.agent_config.gsvSetting.promptLang,
    }
    tts_mode = CConfig.config.get("TTS", {}).get("mode", "api") or "api"
    try:
        start_time = time.time()
        if tts_mode == "local":
            byte_data = await ttsService.local_gsv_tts(
                data=data,
            )
        else:
            byte_data = await ttsService.api_gsv_tts(data)
        logger.info(f"合成完成的耗时: {time.time() - start_time}")
        return byte_data
    except Exception as e:
        logger.error(f"[错误] TTS执行失败，mode={tts_mode}，错误信息: {e}")
        return None


# 文本事件封装，触发TTS和动作规划任务
async def text_wrapper(sentence_id, sentence_text, priority_queue):
    # 记录文本完成事件（最高优先级）
    current_time_ms = time.time() * 1000
    text_event = PriorityEvent(
        priority=0,
        timestamp_ms=current_time_ms,
        data={
            "type": "text",
            "sentence_id": sentence_id,
            "message": sentence_text,
            "timestamp_ms": current_time_ms,
            "done": False,
        },
    )
    heapq.heappush(priority_queue, text_event)


# TTS任务封装，使用信号量控制并发，完成后推送音频事件到优先级队列
async def tts_wrapper(
    tts_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    priority_queue: list[PriorityEvent],
    processor: StreamProcessor,
):
    async with tts_semaphore:
        try:
            agent = assistant_service.get_current_assistant()
            if not agent:
                logger.error("[错误] 当前没有加载助手")
                return
            # 获取情绪和参考音频
            emotion = processor._get_emotion(sentence_text)
            ref_audio = ""
            ref_text = ""
            if emotion:
                agent_config = agent.agent_config.gsvSetting.extraRefAudio
                if emotion in agent_config:
                    ref_audio = agent_config[emotion][0]
                    ref_text = agent_config[emotion][1]

            # 设置TTS数据并执行TTS任务
            tts_data_item = TTSData(
                text=sentence_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            audio_data = await tts_task(tts_data_item)
            if not audio_data:
                logger.warning(
                    f"[动作规划] TTS生成空音频，跳过 sentence_id: {sentence_id}"
                )
                return
            encode_data = base64.b64encode(audio_data).decode("utf-8")
            audio_event = PriorityEvent(
                priority=1,  # TTS优先级次于文本
                timestamp_ms=time.time() * 1000,
                data={
                    "type": "audio",
                    "sentence_id": sentence_id,
                    "message": sentence_text,
                    "file": encode_data,
                    "timestamp_ms": time.time() * 1000,
                    "done": False,
                },
            )
            heapq.heappush(priority_queue, audio_event)

        except Exception as e:
            logger.error(f"[动作规划] 启动失败: {e}")


# 动作规划任务封装，使用信号量控制并发，完成后推送动作事件到优先级队列
async def motion_wrapper(
    motion_semaphore: asyncio.Semaphore,
    sentence_id: int,
    sentence_text: str,
    priority_queue: list[PriorityEvent],
    motion_generator: ExpressionGenerator,
    motion_history: dict | None = None,  # 新增：历史动作状态
):
    logger.info(f"准备生成动作，句子ID: {sentence_id}, 内容: {sentence_text}")

    async with motion_semaphore:
        try:
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

            sentence_text_clean = re.sub(
                r"\(.*?\)|（.*?）|【.*?】|\[.*?\]|\{.*?\}", "", sentence_text
            )
            sentence_text_clean = sentence_text_clean.replace(" ", "").replace("\n", "")

            frame_plans = await motion_generator.generate_motion_plan(
                speech_text=sentence_text,
                context=context,
                speech_duration_ms=len(sentence_text_clean)
                * 100,  # 粗略估计每个字100ms
                timeout_seconds=5,
                previous_action=previous_action,  # 传入前一个动作状态
            )

            logger.info(f"生成的动作帧: {frame_plans}")

            frame = await motion_generator.generate_tts_motion_frame_with_plan(
                frame_plans
            )
            # 记录动作到历史状态
            if motion_history is not None:
                motion_history[sentence_id] = frame_plans[0].get("action", "自然动作")

            # 推送到优先级队列
            current_time_ms = time.time() * 1000
            motion_event = PriorityEvent(
                priority=2,  # 低优先级
                timestamp_ms=current_time_ms,
                data={
                    "type": "motion_frame",
                    "sentence_id": sentence_id,
                    "source_text": sentence_text,
                    "motions": frame,
                    "timestamp_ms": current_time_ms,
                    "done": False,
                },
            )
            heapq.heappush(priority_queue, motion_event)
        except Exception as e:
            logger.warning(f"[动作规划] 第{sentence_id + 1}句动作生成失败: {e}")


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

    # 优先级事件队列（用于合并所有事件并按优先级输出）
    priority_queue: list[PriorityEvent] = []
    # 任务跟踪集合，确保所有后台任务完成后才结束主循环
    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        # 添加到跟踪集合，任务完成后自动移除，避免存在未完成的任务导致主循环过早结束
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    # 并发控制信号量
    tts_semaphore = asyncio.Semaphore(1)  # 最多1个并发TTS任务，防止过高的资源占用

    # 句子完成时的回调函数：启动TTS和动作规划
    async def on_sentence_complete(sentence_id: int, sentence_text: str):
        """当检测到句子完成时，立即启动TTS和动作规划"""
        track_task(
            asyncio.create_task(
                text_wrapper(
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    priority_queue=priority_queue,
                )
            )
        )
        track_task(
            asyncio.create_task(
                tts_wrapper(
                    tts_semaphore=tts_semaphore,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    priority_queue=priority_queue,
                    processor=processor,
                )
            )
        )

    # 注册句子完成回调
    processor.register_sentence_complete_callback(on_sentence_complete)

    # 启动LLM流 和 TTS处理任务
    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        # 持续输出事件，直到 LLM 完成且后台任务全部结束。
        while True:
            while priority_queue:
                event = heapq.heappop(priority_queue)
                yield _to_sse(event.data)

            if llm_task.done() and not pending_tasks:
                break

            await asyncio.sleep(0.01)
        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": processor.sentence_count,
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

    # 优先级事件队列（用于合并所有事件并按优先级输出）
    priority_queue: list[PriorityEvent] = []
    # 任务跟踪集合，确保所有后台任务完成后才结束主循环
    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        # 添加到跟踪集合，任务完成后自动移除，避免存在未完成的任务导致主循环过早结束
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    # 为当前助手创建动作生成器
    motion_generator = await _get_agent_expression_generator(agent.agent_name)

    if motion_generator is None:
        logger.warning("[动作规划] 未找到可用 Live2D 参数，将仅输出 text/audio")

    # 并发控制信号量
    tts_semaphore = asyncio.Semaphore(1)  # 最多1个并发TTS任务，防止过高的资源占用
    motion_semaphore = asyncio.Semaphore(4)  # 最多4个并发动作规划任务

    # 句子完成时的回调函数：启动TTS和动作规划
    async def on_sentence_complete(sentence_id: int, sentence_text: str):
        """当检测到句子完成时，立即启动TTS和动作规划"""
        track_task(
            asyncio.create_task(
                text_wrapper(
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    priority_queue=priority_queue,
                )
            )
        )
        track_task(
            asyncio.create_task(
                tts_wrapper(
                    tts_semaphore=tts_semaphore,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    priority_queue=priority_queue,
                    processor=processor,
                )
            )
        )
        track_task(
            asyncio.create_task(
                motion_wrapper(
                    motion_semaphore=motion_semaphore,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    priority_queue=priority_queue,
                    motion_generator=motion_generator,
                    motion_history=processor.motion_history,
                )
            )
        )

    # 注册句子完成回调
    processor.register_sentence_complete_callback(on_sentence_complete)

    # 启动LLM流 和 TTS处理任务
    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        # 持续输出事件，直到 LLM 完成且后台任务全部结束。
        while True:
            while priority_queue:
                event = heapq.heappop(priority_queue)
                yield _to_sse(event.data)

            if llm_task.done() and not pending_tasks:
                break

            await asyncio.sleep(0.01)
        # 输出完成信号
        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": processor.sentence_count,
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
