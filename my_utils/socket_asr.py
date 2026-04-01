import threading
import time
import gc
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from my_utils.log import logger


class ASRServer:
    _instance = None
    _instance_lock = threading.Lock()

    # ===== 可配置参数 =====
    IDLE_TIMEOUT = 300  # 超过 300 秒未使用自动释放（5分钟）
    CHECK_INTERVAL = 30  # 后台检测间隔（秒）

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.asr_model = None
        self.last_used_time = 0.0
        self._model_lock = threading.Lock()

        # 启动后台资源回收线程
        check_thread = threading.Thread(target=self._auto_release_worker, daemon=True)
        check_thread.start()

    # =========================
    # 模型管理
    # =========================
    def load_model(self):
        """
        加载 ASR 模型（线程安全）
        """
        with self._model_lock:
            if self.asr_model is not None:
                return

            model_dir = "./data/models/SenseVoiceSmall"
            logger.info("[ASR] 正在加载 ASR 模型...")

            try:
                self.asr_model = AutoModel(
                    model=model_dir,
                    disable_update=True,
                    device="cuda:0",
                )
                logger.info("[ASR] ASR 模型加载完成（GPU）")
            except Exception as e:
                logger.warning(e)
                logger.info("[ASR] 未检测到本地模型，开始下载...")

                from modelscope import snapshot_download

                model_dir = snapshot_download(
                    model_id="iic/SenseVoiceSmall",
                    local_dir=model_dir,
                    revision="master",
                )

                self.asr_model = AutoModel(
                    model=model_dir,
                    disable_update=True,
                    device="cpu",
                )
                logger.info("[ASR] ASR 模型加载完成（CPU）")

    def ensure_model_loaded(self):
        """
        ASR 前检查模型是否加载
        """
        if self.asr_model is None:
            self.load_model()

        self.last_used_time = time.time()

    def release_model(self):
        """
        主动释放 ASR 模型资源
        """
        with self._model_lock:
            if self.asr_model is None:
                return

            logger.info("[ASR] 长时间未使用，释放 ASR 模型资源")
            self.asr_model = None

            # 释放显存 / 内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _auto_release_worker(self):
        """
        后台检测模型空闲时间，自动释放资源
        """
        while True:
            time.sleep(self.CHECK_INTERVAL)

            if self.asr_model is None:
                continue

            idle_time = time.time() - self.last_used_time
            if idle_time > self.IDLE_TIMEOUT:
                self.release_model()

    # =========================
    # ASR 接口
    # =========================
    def asr(self, audio_data: bytes) -> str | None:
        self.ensure_model_loaded()
        if self.asr_model is None:
            return None
        res = self.asr_model.generate(
            input=audio_data,
            cache={},
            language="zh",
            ban_emo_unk=True,
            use_itn=False,
            disable_pbar=True,
        )

        text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")

        return text if text else None
