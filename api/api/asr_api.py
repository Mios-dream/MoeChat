import json
from fastapi import (
    APIRouter,
)
from api.models.asr_request import asr_data
import core.chat_core as chat_core

import base64
import soundfile as sf
from fastapi import WebSocket
from io import BytesIO
import numpy as np
from utils.pysilero import VADIterator
from scipy.signal import resample


asr_api = APIRouter()


@asr_api.post("/asr")
async def asr_audio(params: asr_data):
    audio_data = base64.urlsafe_b64decode(params.data.encode("utf-8"))
    text = chat_core.asr(audio_data)
    return text


@asr_api.websocket("/asr_ws")
async def asr_websocket(c_websocket: WebSocket):
    await c_websocket.accept()
    print("WebSocket 连接已建立")

    vad_iterator = VADIterator(speech_pad_ms=300)
    current_speech = []
    current_speech_tmp = []
    status = False

    while True:
        try:
            data = await c_websocket.receive_text()
            data = json.loads(data)

            if data["type"] == "asr":
                # 改为标准 base64 解码
                audio_data = base64.b64decode(str(data["data"]).encode("utf-8"))
                samples = np.frombuffer(audio_data, dtype=np.int16)

                current_speech_tmp.append(samples)

                if len(current_speech_tmp) < 4:
                    continue

                resampled = np.concatenate(current_speech_tmp.copy())
                resampled = (resampled / 32768.0).astype(np.float32)
                current_speech_tmp = []

                for speech_dict, speech_samples in vad_iterator(resampled):

                    if "start" in speech_dict:

                        current_speech = []
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
                                buffer, combined, 16000, format="WAV", subtype="PCM_16"
                            )
                            buffer.seek(0)
                            audio_bytes = buffer.read()
                            res_text = chat_core.asr(audio_bytes)

                            if res_text:
                                await c_websocket.send_text(res_text)
                        current_speech = []

        except Exception as e:
            print(f"错误: {e}")
            break
