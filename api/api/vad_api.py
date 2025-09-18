import base64
import json

import numpy as np
from utils.pysilero.pysilero import VADIterator


# vad接口
from fastapi import APIRouter, WebSocket

vad_api = APIRouter()


@vad_api.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # state = np.zeros((2, 1, 128), dtype=np.float32)
    # sr = np.array(16000, dtype=np.int64)
    # 初始化 VAD 迭代器，指定采样率为 16000Hz
    vad_iterator = VADIterator(speech_pad_ms=90)
    while True:
        try:
            data = await websocket.receive_text()
            data = json.loads(data)
            if data["type"] == "asr":
                audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
                samples = np.frombuffer(audio_data, dtype=np.float32)
                # samples = nr.reduce_noise(y=samples, sr=16000)
                # samples = np.expand_dims(samples, axis=0)
                # ort_inputs = {"input": samples, "state": state, "sr": sr}
                # # 进行 VAD 预测
                # vad_prob = session.run(None, ort_inputs)[0]
                # # 判断是否为语音
                # if vad_prob > 0.7:
                #     print(f"[{time.time()}]说话中...")
                #     await websocket.send_text("说话中...")
                # 将重采样后的数据传递给 VAD 处理
                for speech_dict, speech_samples in vad_iterator(samples):
                    if "start" in speech_dict:
                        # current_speech = []
                        print("开始说话...")
                        await websocket.send_text("开始说话...")
                        pass
                    # if status:
                    #     current_speech.append(speech_samples)
                    # else:
                    #     continue
                    is_last = "end" in speech_dict
                    if is_last:
                        # t = Thread(target=gen_audio, args=(current_speech.copy(), ))
                        # t.daemon = True
                        # t.start()
                        await websocket.send_text("结束说话")
                        print("结束说话")
                        # current_speech = []  # 清空当前段落

        except:
            break
