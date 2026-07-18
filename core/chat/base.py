"""
聊天基础组件

提供 V3 版本共享的基础组件：
- TTSData: TTS 数据类
- TTS 合成函数（tts_task / tts_wrapper）
- SSE 流转换函数
"""

import os
import time
import asyncio
import base64
import re
from Config import Config
from services.tts_service import ttsService
from my_utils import config_manager as CConfig
from my_utils.log import logger
from services.assistant_service import AssistantService
from pydantic import BaseModel
from collections.abc import AsyncGenerator
from models.dto.response.ChatResponse import FullChatResponse

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


def _get_emotion(msg: str) -> str | None:
    """
    从文本中提取情感标签

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


async def tts_wrapper(
    tts_semaphore: asyncio.Semaphore,
    sentence_text: str,
    tts_text: str,
) -> str | None:
    """
    TTS 合成并返回 base64 音频数据

    参数：
    - tts_semaphore: TTS 并发控制信号量
    - sentence_text: 原始句子文本
    - tts_text: 过滤后的 TTS 文本

    返回：
    - base64 编码的音频数据，合成失败返回 None
    """
    async with tts_semaphore:
        try:
            agent = assistant_service.get_current_assistant()
            if not agent:
                logger.error("[错误] 当前没有加载助手")
                return None

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
                    return base64.b64encode(audio_data).decode("utf-8")
                else:
                    logger.warning(f"[TTS] 生成空音频")
            else:
                logger.info(f"[TTS] 无可读语音内容")

        except Exception as e:
            logger.error(f"[TTS] 执行失败: {e}")

    return None


async def to_sse_stream(
    generator: AsyncGenerator[FullChatResponse, None],
) -> AsyncGenerator[str, None]:
    """
    将模型对象异步生成器统一转换为 SSE 格式流

    参数：
    - generator: 产出 FullChatResponse 模型对象的异步生成器

    返回：
    - 产出 SSE 格式字符串的异步生成器
    """
    async for payload in generator:
        yield "data: " + payload.model_dump_json(ensure_ascii=False, exclude_none=True)
