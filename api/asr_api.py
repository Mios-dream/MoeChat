import json
from fastapi import (
    APIRouter,
)
from models.dto.asr_request import asr_data
import core.chat_core as chat_core

import base64
import soundfile as sf
from fastapi import WebSocket
from io import BytesIO
import numpy as np
from utils.pysilero import VADIterator
from utils.speak_finish import isSpeakFinish, SpeakWithAssistant

# from scipy.signal import resample
from utils.log import logger
from fastapi.responses import JSONResponse

asr_api = APIRouter()


@asr_api.post("/asr")
async def asr_audio(params: asr_data):
    """
    asr 非流式接口
    修复了 base64 解析和返回格式
    """
    try:
        raw_data = params.data
        if "," in raw_data:
            raw_data = raw_data.split(",")[1]
        audio_data = base64.b64decode(raw_data)
        text = chat_core.asr(audio_data)

        if not text:
            logger.warning("[ASR] 识别结果为空")
            return JSONResponse(content={"text": None})

        logger.info(f"[ASR] 识别成功: {text}")
        return JSONResponse(content={"text": text})

    except Exception as e:
        logger.error(f"[ASR] 接口处理出错: {e}", exc_info=True)
        return JSONResponse(content={"text": None}, status_code=500)


@asr_api.websocket("/asr_ws")
async def asr_websocket(c_websocket: WebSocket):
    """
    asr websocket 接口
    仅语音识别
    """

    await c_websocket.accept()

    vad_iterator = VADIterator(speech_pad_ms=120)
    # 当前语音段落
    current_speech = []
    # 语音段落缓存
    current_speech_tmp = []
    status = False

    while True:
        try:
            data = await c_websocket.receive_text()
        except Exception as e:
            logger.info(f"asr客户端下线：{e}")
            return
        try:
            data = json.loads(data)
        except Exception as e:
            logger.warning(f"无法将asr客户端数据解析为json：{e}")
            return

        if data["type"] != "asr":
            return
        try:
            # 改为标准 base64 解码
            audio_data = base64.b64decode(str(data["data"]).encode("utf-8"))
            samples = np.frombuffer(audio_data, dtype=np.int16)
        except Exception as e:
            logger.warning(f"asr客戶端音频数据格式错误：{e}")
            return

        current_speech_tmp.append(samples)
        # if len(current_speech_tmp) < 4:
        #     continue
        # resampled = np.concatenate(current_speech_tmp.copy())
        resampled = np.concatenate(current_speech_tmp)
        resampled = (resampled / 32768.0).astype(np.float32)
        # 语音段落太短则不处理，等待合适的语音段落长度
        if len(resampled) < 240:
            continue

        # 情况语音段落缓存
        current_speech_tmp.clear()

        for speech_dict, speech_samples in vad_iterator(resampled):
            # 语音段落开始
            if "start" in speech_dict:
                current_speech.clear()
                status = True

            if status:
                current_speech.append(speech_samples)
            else:
                continue
            is_last = "end" in speech_dict

            if is_last:
                status = False
                combined = np.concatenate(current_speech)
                audio_bytes = b""
                with BytesIO() as buffer:
                    sf.write(
                        buffer,
                        combined,
                        16000,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    buffer.seek(0)
                    audio_bytes = buffer.read()  # 完整的 WAV bytes
                res_text = chat_core.asr(audio_bytes)

                if res_text:
                    await c_websocket.send_text(res_text)
                current_speech.clear()  # 清空当前段落


@asr_api.websocket("/asr_ws_plus")
async def asr_websocket_plus(c_websocket: WebSocket):
    """
    asr websocket 接口
    包含语音识别,语音完整检测,聊天倾向判断
    """

    await c_websocket.accept()

    vad_iterator = VADIterator(speech_pad_ms=120)
    # 当前语音段落
    current_speech = []
    # 语音段落缓存
    current_speech_tmp = []
    # vad 活动状态
    status = False
    # 缓存对话识别的数据，用于判断是否对话完整
    message_chucks = []

    while True:
        try:
            data = await c_websocket.receive_text()
        except Exception as e:
            logger.info(f"asr客户端下线：{e}")
            return
        try:
            data = json.loads(data)
        except Exception as e:
            logger.warning(f"无法将asr客户端数据解析为json：{e}")
            return

        if data["type"] != "asr":
            return
        try:
            # 改为标准 base64 解码
            audio_data = base64.b64decode(str(data["data"]).encode("utf-8"))
            samples = np.frombuffer(audio_data, dtype=np.int16)
        except Exception as e:
            logger.warning(f"asr客戶端音频数据格式错误：{e}")
            return
        # 对语音段落进行缓存
        current_speech_tmp.append(samples)

        resampled = np.concatenate(current_speech_tmp)
        resampled = (resampled / 32768.0).astype(np.float32)
        # 语音段落太短则不处理，等待合适的语音段落长度
        if len(resampled) < 240:
            continue

        # 情况语音段落缓存
        current_speech_tmp.clear()

        for speech_dict, speech_samples in vad_iterator(resampled):
            # 语音段落开始
            if "start" in speech_dict:
                current_speech.clear()
                status = True

            # 如果处于活动状态，则缓存语音段落
            if status:
                current_speech.append(speech_samples)
            else:
                continue
            is_last = "end" in speech_dict
            if is_last:
                status = False
                combined = np.concatenate(current_speech)
                audio_bytes = b""
                with BytesIO() as buffer:
                    sf.write(
                        buffer,
                        combined,
                        16000,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    buffer.seek(0)
                    audio_bytes = buffer.read()  # 完整的 WAV bytes
                    # 保存为文件
                    with open("test.wav", "wb") as f:
                        f.write(audio_bytes)

                res_text = chat_core.asr(audio_bytes)
                message_chucks.append(res_text)
                finished_text = "".join(message_chucks)
                # 判断是否结束
                if not await isSpeakFinish(finished_text):
                    continue
                speakWithAssistantContent = await SpeakWithAssistant(finished_text, [])
                if not speakWithAssistantContent:
                    continue

                await c_websocket.send_text(
                    json.dumps(
                        {
                            "type": "asr",
                            "data": finished_text,
                            "withAssistant": speakWithAssistantContent.probability
                            >= 0.5,
                        },
                        ensure_ascii=False,
                    ),
                )
                message_chucks.clear()
                current_speech.clear()  # 清空当前段落
