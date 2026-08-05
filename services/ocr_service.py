"""
OCR 识别服务

封装本地 PaddleOCR 推理引擎的模型生命周期管理（懒加载、线程安全、闲置自动
卸载）与图片识别接口，供多模态内容处理器及其他需要图片文字识别的模块复用。

并发模型：识别接口通过 asyncio.to_thread 放入线程池执行（onnxruntime 推理
释放 GIL，可并行），因此实例会被多个线程共享，用 _model_lock 保护实例的
创建/卸载/推理临界区，避免"推理中被闲置监控卸载"的竞态。
"""

import asyncio
import gc
import threading
import time
from io import BytesIO
from typing import Any

import numpy
from my_utils.log import logger
from paddleocr import PaddleOCR

# OCR 模型闲置超时时间（秒）：3 分钟无使用即卸载释放内存
_OCR_IDLE_TIMEOUT_SECONDS = 180
# 后台闲置监控协程的轮询间隔（秒）
_OCR_MONITOR_INTERVAL_SECONDS = 30


class OcrService:
    """PaddleOCR 识别服务：懒加载单例 + 线程安全推理 + 闲置自动卸载"""

    def __init__(self) -> None:
        # PaddleOCR 实例（懒加载）
        self._ocr_instance: PaddleOCR | None = None
        # 最近一次使用时间（time.monotonic() 单调时钟，不受系统改时影响）
        self._ocr_last_used: float = 0.0
        # 可重入锁：保护实例创建/卸载，并在整个推理期间持有，防止推理中被卸载
        self._model_lock = threading.RLock()
        # 后台闲置监控协程任务（在事件循环中调度，不占用额外线程）
        self._monitor_task: asyncio.Task | None = None
        # 非异步环境下无法启动监控时，仅提示一次
        self._monitor_warned = False

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> PaddleOCR:
        """懒加载并返回 PaddleOCR 实例（线程安全，刷新闲置时间戳）"""
        with self._model_lock:
            if self._ocr_instance is None:
                self._ocr_instance = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    engine="onnxruntime",
                    device="cpu",
                )
                logger.info(
                    "[OCR服务] PaddleOCR 实例已创建(PP-OCRv5_mobile + onnxruntime + cpu)"
                )
            # 每次使用都刷新闲置时间戳，供闲置监控协程判断是否卸载
            self._ocr_last_used = time.monotonic()
            return self._ocr_instance

    def _unload(self) -> None:
        """关闭并卸载当前 PaddleOCR 实例，释放 onnxruntime 会话占用的内存。

        调用方需持有 _model_lock（或处于已持锁的线程上下文，锁为可重入锁）。
        """
        if self._ocr_instance is not None:
            logger.info("[OCR服务] PaddleOCR 模型已卸载，释放内存")
            try:
                # 显式关闭 paddlex 管线，释放 onnxruntime 会话与底层资源
                self._ocr_instance.close()
            except Exception as e:
                logger.warning(f"[OCR服务] PaddleOCR close 失败: {e}")
            self._ocr_instance = None
            # 回收 Python 对象与 numpy 缓存，尽快归还内存
            gc.collect()

    async def _idle_monitor(self) -> None:
        """后台协程：周期性检查 OCR 实例闲置时长，超时则卸载并回收内存"""
        while True:
            await asyncio.sleep(_OCR_MONITOR_INTERVAL_SECONDS)
            try:
                if self._ocr_instance is None:
                    continue
                if time.monotonic() - self._ocr_last_used < _OCR_IDLE_TIMEOUT_SECONDS:
                    continue
                # 非阻塞抢锁：若 OCR 正在线程池中推理（锁被持有），说明并不闲置，
                # 跳过本轮，避免阻塞事件循环
                if not self._model_lock.acquire(blocking=False):
                    continue
                try:
                    # 加锁后二次校验，防止与新一轮使用发生竞态
                    if (
                        self._ocr_instance is not None
                        and time.monotonic() - self._ocr_last_used
                        >= _OCR_IDLE_TIMEOUT_SECONDS
                    ):
                        self._unload()
                finally:
                    self._model_lock.release()
            except Exception as e:
                logger.error(f"[OCR服务] 闲置监控异常: {e}")

    def _ensure_idle_monitor(self) -> None:
        """确保闲置监控协程已调度（须在事件循环线程中调用）"""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 非异步环境（如同步脚本/单元测试）：跳过后台监控，实例保持常驻
            if not self._monitor_warned:
                logger.warning("[OCR服务] 无事件循环，OCR 闲置自动卸载不可用")
                self._monitor_warned = True
            return
        self._monitor_task = loop.create_task(self._idle_monitor())

    def release_model(self) -> None:
        """主动释放 OCR 模型资源（供测试或关闭钩子调用）"""
        with self._model_lock:
            self._unload()

    # ------------------------------------------------------------------
    # 识别接口
    # ------------------------------------------------------------------

    async def recognize_image(self, image_bytes: bytes) -> str:
        """对图片字节数据进行 OCR 识别，返回拼接后的文本。

        推理放入线程池（asyncio.to_thread）执行，避免阻塞事件循环。
        """
        # 在事件循环线程调度闲置监控协程（实例加载与推理均在子线程中进行）
        self._ensure_idle_monitor()
        return await asyncio.to_thread(self._predict, image_bytes)

    def _predict(self, image_bytes: bytes) -> str:
        """线程池中执行的实际推理（含图像预处理与置信度过滤）"""
        try:
            from PIL import Image

            image = Image.open(BytesIO(image_bytes))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_array = numpy.array(image)
            # 加锁保护：整个推理期间持有锁，确保实例不会被闲置监控卸载，
            # 并序列化并发请求对共享 onnxruntime 会话的访问
            with self._model_lock:
                ocr = self._ensure_loaded()
                result = ocr.predict(image_array)
            texts, _ = self._extract_text_and_score(result)
            combined = "\n".join(texts) if texts else ""
            if combined:
                logger.info(f"[OCR服务] 识别成功: {len(combined)} 字符")
            return combined
        except ImportError:
            logger.warning("paddleocr 未安装，请执行: pip install paddleocr")
            return "[OCR 引擎未安装]"
        except Exception as e:
            logger.error(f"[OCR服务] 识别失败: {e}")
            return "[图片 OCR 失败]"

    @staticmethod
    def _extract_text_and_score(
        result: list[dict[str, Any]],
    ) -> tuple[list[str], float]:
        """从 PaddleOCR 输出中提取文本，过滤低置信度结果"""
        texts: list[str] = []
        scores: list[float] = []
        threshold = 0.9

        for res in result or []:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])

            for idx, text in enumerate(rec_texts):
                cleaned = (text or "").strip()
                if not cleaned:
                    continue

                score = rec_scores[idx] if idx < len(rec_scores) else 0.0
                if isinstance(score, (int, float)):
                    score_value = float(score)
                    if score_value < threshold or len(cleaned) <= 4:
                        continue
                    texts.append(cleaned)
                    scores.append(score_value)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return texts, avg_score


# 模块级单例：供全局复用（与 services/tts_service.py 的 ttsService 保持一致）
ocrService = OcrService()
