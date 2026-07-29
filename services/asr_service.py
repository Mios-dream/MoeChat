import os
import threading
import time
import gc
from io import BytesIO

import numpy as np
import sherpa_onnx
import soundfile as sf

from my_utils.log import logger
from Config import Config


class ASRServer:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.recognizer = None
        self.last_used_time = 0.0
        self._model_lock = threading.Lock()

    def load_model(self):
        """
        加载 ASR 模型（线程安全）
        """
        with self._model_lock:
            if self.recognizer is not None:
                return
            provider = "cpu"
            logger.info(f"[ASR] 正在加载 zipformer-ctc 模型（{provider}）...")

            self.recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=os.path.join(Config.ASR_MODEL_DIR, "model.int8.onnx"),
                tokens=os.path.join(Config.ASR_MODEL_DIR, "tokens.txt"),
                num_threads=4,
                sample_rate=16000,
                feature_dim=80,
                provider=provider,
            )
            logger.info(f"[ASR] zipformer-ctc 模型加载完成（{provider}）")

    def ensure_model_loaded(self):
        """
        ASR 前检查模型是否加载
        """
        if self.recognizer is None:
            self.load_model()

        self.last_used_time = time.time()

    def release_model(self):
        """
        主动释放 ASR 模型资源
        """
        with self._model_lock:
            if self.recognizer is None:
                return

            logger.info("[ASR] 释放 ASR 模型资源")
            self.recognizer = None
            gc.collect()

    # ASR 接口
    def asr(self, audio_data: bytes) -> str | None:
        self.ensure_model_loaded()
        if self.recognizer is None:
            return None

        try:
            # 解析 WAV 音频
            with BytesIO(audio_data) as buf:
                data, sr = sf.read(buf)
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # 创建流并喂入完整音频
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, data)
            # 喂入尾部静音确保尾字识别
            tail_paddings = np.zeros(int(0.66 * sr), dtype=np.float32)
            stream.accept_waveform(sr, tail_paddings)
            stream.input_finished()

            # 循环调用 decode_stream 直到 is_ready 返回 False
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            text = self.recognizer.get_result(stream).strip()
            return text if text else None
        except Exception as e:
            logger.error(f"[ASR] 识别出错: {e}", exc_info=True)
            return None
