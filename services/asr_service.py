"""
语音识别（ASR）服务

封装本地 sherpa-onnx 流式 CTC 识别引擎的模型生命周期管理（懒加载、线程安全、
闲置自动卸载）与识别接口。与 OCR 服务保持一致的架构：

- 识别通过 asyncio.to_thread 放入线程池执行（sherpa-onnx 推理释放 GIL，可并行），
  避免阻塞事件循环；
- 用可重入锁保护模型创建/卸载/推理临界区，避免"推理中被闲置监控卸载"的竞态，
  并序列化并发请求对 sherpa-onnx 会话的访问；
- 通过 asyncio 后台协程监控闲置时间，超过阈值自动卸载释放内存；
- 保留 release_model() 供 WS 断开等外部钩子主动释放。
"""

import asyncio
import gc
import os
import threading
import time
from io import BytesIO

import numpy as np
import sherpa_onnx
import soundfile as sf

from my_utils.log import logger
from Config import Config

# ASR 模型闲置超时时间（秒）：3 分钟无使用即卸载释放内存
_ASR_IDLE_TIMEOUT_SECONDS = 180
# 后台闲置监控协程的轮询间隔（秒）
_ASR_MONITOR_INTERVAL_SECONDS = 30


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
        # sherpa-onnx 在线识别器（懒加载）
        self.recognizer = None
        # 最近一次使用时间（time.monotonic() 单调时钟，不受系统改时影响）
        self.last_used_time = 0.0
        # 可重入锁：保护模型创建/卸载，并在整个推理期间持有，防止推理中被卸载
        self._model_lock = threading.RLock()
        # 后台闲置监控协程任务（在事件循环中调度，不占用额外线程）
        self._monitor_task = None
        # 非异步环境下无法启动监控时，仅提示一次
        self._monitor_warned = False

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def load_model(self):
        """加载 ASR 模型（线程安全，加载后刷新闲置时间戳）"""
        with self._model_lock:
            if self.recognizer is None:
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
            # 每次使用都刷新闲置时间戳，供闲置监控协程判断是否卸载
            self.last_used_time = time.monotonic()

    def _unload(self):
        """释放当前 ASR 模型资源（调用方需持有锁，锁为可重入锁）"""
        if self.recognizer is not None:
            logger.info("[ASR] 释放 ASR 模型资源")
            self.recognizer = None
            # 回收 Python 对象与 numpy 缓存，尽快归还内存
            gc.collect()

    async def _idle_monitor(self):
        """后台协程：周期性检查模型闲置时长，超时则卸载并回收内存"""
        while True:
            await asyncio.sleep(_ASR_MONITOR_INTERVAL_SECONDS)
            try:
                if self.recognizer is None:
                    continue
                if time.monotonic() - self.last_used_time < _ASR_IDLE_TIMEOUT_SECONDS:
                    continue
                # 非阻塞抢锁：若正在线程池中推理（锁被持有），说明并不闲置，
                # 跳过本轮，避免阻塞事件循环
                if not self._model_lock.acquire(blocking=False):
                    continue
                try:
                    # 加锁后二次校验，防止与新一轮使用发生竞态
                    if (
                        self.recognizer is not None
                        and time.monotonic() - self.last_used_time
                        >= _ASR_IDLE_TIMEOUT_SECONDS
                    ):
                        self._unload()
                finally:
                    self._model_lock.release()
            except Exception as e:
                logger.error(f"[ASR] 闲置监控异常: {e}")

    def _ensure_idle_monitor(self):
        """确保闲置监控协程已调度（须在事件循环线程中调用）"""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 非异步环境（如同步脚本/单元测试）：跳过后台监控，模型保持常驻
            if not self._monitor_warned:
                logger.warning("[ASR] 无事件循环，ASR 闲置自动卸载不可用")
                self._monitor_warned = True
            return
        self._monitor_task = loop.create_task(self._idle_monitor())

    def release_model(self):
        """主动释放 ASR 模型资源（供 WS 断开等钩子调用）"""
        with self._model_lock:
            self._unload()

    # ------------------------------------------------------------------
    # 识别接口
    # ------------------------------------------------------------------

    async def asr(self, audio_data: bytes) -> str | None:
        """对音频字节数据进行语音识别，返回文本（协程）。

        推理放入线程池（asyncio.to_thread）执行，避免阻塞事件循环。
        """
        # 在事件循环线程调度闲置监控协程（模型加载与推理均在子线程中进行）
        self._ensure_idle_monitor()
        return await asyncio.to_thread(self._predict, audio_data)

    def _predict(self, audio_data: bytes) -> str | None:
        """线程池中执行的实际推理（含 WAV 解析与流式解码）"""
        try:
            # 加锁保护：加载 + 整个推理期间持有锁，确保模型不会被闲置监控
            # 卸载，并序列化并发请求对 sherpa-onnx 会话的访问
            with self._model_lock:
                self.load_model()
                if self.recognizer is None:
                    return None

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
