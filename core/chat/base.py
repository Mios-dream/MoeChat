"""
聊天基础组件

提供所有聊天版本共享的基础组件：
- TTSData: TTS 数据类
- 事件处理函数（文本、音频事件封装）
- 事件聚合和输出函数
- 统一消息构建函数

设计原则：
- 基础组件与版本无关，可在 V1/V2/V3 中复用
- 所有事件格式统一，便于前端处理
"""

import json
import os
import time
import asyncio
import base64
import re
from typing import Any

from Config import Config
from services.tts_service import ttsService
from my_utils import config_manager as CConfig
from my_utils.log import logger
from services.assistant_service import AssistantService
from pydantic import BaseModel

assistant_service = AssistantService()


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
) -> dict[str, Any]:
    """
    封装音频事件

    参数：
    - tts_semaphore: TTS 并发控制信号量
    - sentence_id: 句子 ID
    - sentence_text: 原始句子文本
    - tts_text: 过滤后的 TTS 文本

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

            # 查询情感标签
            emotion = _get_emotion(sentence_text)
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


def _get_emotion(msg: str) -> str | None:
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
    return None


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
