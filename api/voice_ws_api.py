"""
统一语音 WebSocket 端点（双模式语音识别）

两种监听模式：

  1. background（后台静默监听，默认）：
     持续 KWS 关键词检测，音频经环形缓冲区暂存。
     命中关键词时按需 ASR，结果返回前端，后端不处理回复。
     适合桌面待机场景，低功耗。

  2. conversation（主动对话）：
     经典 VAD + ASR 全链路 + 意图分析。
     结果返回前端，后端不处理回复。

协议：
  Client → Server:
    {"type": "audio", "data": "<base64_int16>", "sample_rate": 16000}
    {"type": "session_control", "action": "start"|"end", "mode": "background"|"conversation"}

  Server → Client:
    {"type": "vad", "event": "speech_start"|"speech_end", "timestamp": ...}
    {"type": "asr_result", "text": "..."}
    {"type": "error", "message": "..."}
"""

import asyncio
import base64
import json
import random
import time
from dataclasses import dataclass
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import soundfile as sf
from io import BytesIO
from my_utils.log import logger
from my_utils.pysilero import VADIterator
from services.asr_service import ASRServer
from services.wakeword_service import WakeWordService
from services.assistant_service import AssistantService

voice_ws_api = APIRouter()

# ===================== 服务实例 =====================
asr_server = ASRServer()
assistant_service = AssistantService()

# 响应防重入（防止并发 ASR 冲突）
_asr_in_progress: set[int] = set()

# 音频质量预过滤
MIN_SPEECH_DURATION_SAMPLES = 16000 * 0.3
# 音频能量阈值（低于此值认为是静音或噪声）
MIN_SPEECH_ENERGY = 0.015
# 最大语音长度限制（避免过长音频导致 ASR 阻塞）
MAX_SPEECH_DURATION_SAMPLES = 16000 * 30


# ===================== 环形音频缓冲区 =====================


class AudioRingBuffer:
    """环形音频缓冲区，保持最近 N 秒音频。命中关键词时截取送 ASR。

    注意：WakeWordService 内部累积多帧音频后才返回匹配结果，
    导致检测到关键词的时刻远落后于关键词语音的实际发音时刻。
    因此引入标记机制：KWS 命中时标记当前位置，ASR 时从标记点
    前推截取，确保关键词语音被完整包含。
    """

    def __init__(self, max_duration_ms: int = 5000, sample_rate: int = 16000):
        """
        增大缓冲区到 5000ms，给 KWS 检测延迟留出余量，
        确保关键词语音不会被后续音频推出缓冲区。
        """
        self._max_samples = sample_rate * max_duration_ms // 1000
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._marker: int | None = None

    def add(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._buffer = np.concatenate([self._buffer, samples])
        if len(self._buffer) > self._max_samples:
            self._buffer = self._buffer[-self._max_samples :]

    def mark(self) -> None:
        """标记当前写入位置（KWS 命中时调用）"""
        self._marker = len(self._buffer)

    def get_since_mark(
        self,
        before_ms: int = 3500,
        after_ms: int = 500,
    ) -> np.ndarray:
        """从标记点向前取 before_ms + 向后取 after_ms 的音频。

        相比 get_recent() 只取末尾，此方法以标记点为中心截取，
        确保标记时刻前后的音频都被完整保留。
        若从未调用 mark()，则退化为 get_recent(before_ms + after_ms)。
        """
        if self._marker is None:
            return self.get_recent(before_ms + after_ms)

        n_before = int(16000 * before_ms / 1000)
        n_after = int(16000 * after_ms / 1000)

        start = max(0, self._marker - n_before)
        end = min(len(self._buffer), self._marker + n_after)

        return self._buffer[start:end].copy()

    def get_recent(self, duration_ms: int = 2500) -> np.ndarray:
        n = int(16000 * duration_ms / 1000)
        if len(self._buffer) <= n:
            return self._buffer.copy()
        return self._buffer[-n:].copy()

    def clear(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._marker = None


# ===================== 关键词配置 =====================


@dataclass
class KeywordConfig:
    """关键词配置，包含类别、重要度、触发概率等信息"""

    text: str  # 触发词文本
    category: str  # 类别: wake(唤醒) / casual(闲聊) / query(查询) / action(操作)
    importance: str  # 重要度: high / medium / low
    probability: float  # 触发概率 0.0~1.0，随机命中时按概率决定是否触发


_KEYWORD_CONFIGS: list[KeywordConfig] = [
    # 日常闲聊（低重要度，低触发概率）
    KeywordConfig("好无聊", "casual", "low", 0.5),
    KeywordConfig("无聊", "casual", "low", 0.3),
    KeywordConfig("没意思", "casual", "low", 0.3),
    KeywordConfig("好累", "casual", "low", 0.5),
    KeywordConfig("累了", "casual", "low", 0.3),
    KeywordConfig("好烦", "casual", "low", 0.4),
    KeywordConfig("烦死", "casual", "low", 0.4),
    KeywordConfig("好开心", "casual", "low", 0.5),
    KeywordConfig("开心", "casual", "low", 0.2),
    KeywordConfig("难过", "casual", "low", 0.4),
    KeywordConfig("哭了", "casual", "low", 0.2),
    KeywordConfig("饿了", "casual", "low", 0.5),
    KeywordConfig("困了", "casual", "low", 0.4),
]

# 关键词文本 → 配置映射（运行时动态构建，含助手名称/别称）
_keyword_lookup: dict[str, KeywordConfig] | None = None


def _get_keyword_config(keyword: str) -> KeywordConfig | None:
    """根据检测到的关键词文本获取对应的关键词配置"""
    global _keyword_lookup
    if _keyword_lookup is None:
        _rebuild_keyword_lookup()
    return _keyword_lookup.get(keyword) if _keyword_lookup else None


def _rebuild_keyword_lookup() -> None:
    """
    重建关键词文本 → 配置的映射表。
    预置关键词 + 助手名称/别称（唤醒类，高重要度，必触发）。
    """
    lookup: dict[str, KeywordConfig] = {}

    for kw in _KEYWORD_CONFIGS:
        lookup[kw.text] = kw

    agent = assistant_service.get_current_assistant()
    if agent:
        wake_config = KeywordConfig(
            text=agent.char,
            category="wake",
            importance="high",
            probability=1.0,
        )
        lookup[agent.char] = wake_config
        for alias in agent.alias.split(","):
            alias = alias.strip()
            if alias:
                lookup[alias] = KeywordConfig(
                    text=alias,
                    category="wake",
                    importance="high",
                    probability=1.0,
                )

    global _keyword_lookup
    _keyword_lookup = lookup


def _build_keywords() -> list[str]:
    """构建 KWS 关键词文本列表：助手名称/别称 + 预置触发词"""
    _rebuild_keyword_lookup()
    return list(_keyword_lookup.keys()) if _keyword_lookup else []


# ===================== 会话状态封装 =====================


class VoiceSession:
    """语音 WebSocket 会话状态，封装双模式切换与资源管理"""

    def __init__(self) -> None:
        self.active: bool = False
        self.mode: str = "background"
        self.ring_buffer = AudioRingBuffer()
        self.kw_service: WakeWordService | None = None
        self.vad_iterator: VADIterator | None = None
        self.current_speech: list[np.ndarray] = []
        self.current_speech_tmp: list[np.ndarray] = []

    def start(self, mode: str) -> None:
        """启动会话"""
        self.active = True
        self.mode = mode

        if self.mode == "background":
            try:
                print(_build_keywords())
                self.kw_service = WakeWordService(keywords=_build_keywords())
            except Exception as e:
                logger.error(f"[VoiceWS] KWS 初始化失败: {e}")
                self.kw_service = None
            logger.info("[VoiceWS] 背景监听开始")

        elif self.mode == "conversation":
            self.vad_iterator = VADIterator(speech_pad_ms=100)
            self.current_speech.clear()
            self.current_speech_tmp.clear()
            logger.info("[VoiceWS] 对话模式开始")

    def end(self) -> None:
        """结束会话，清理资源"""
        self.active = False
        self.ring_buffer.clear()
        self.current_speech.clear()
        self.current_speech_tmp.clear()
        self.kw_service = None
        self.vad_iterator = None

    def should_process_audio(self) -> bool:
        """判断当前是否需要处理音频帧"""
        return self.active or self.mode == "background"


class MessageRouter:
    """WebSocket 消息路由器，按 type 分发到对应处理器"""

    def __init__(self, websocket: WebSocket, session: VoiceSession) -> None:
        self._ws = websocket
        self._session = session

    async def route(self, raw: str) -> None:
        """解析并分发一条消息"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await _send_error(self._ws, "无效的消息格式")
            return

        handler = {
            "audio": self._handle_audio,
            "session_control": self._handle_session_control,
            "ping": self._handle_ping,
        }.get(msg.get("type", ""))

        if handler is None:
            return

        await handler(msg)

    async def _handle_audio(self, msg: dict) -> None:
        """处理音频数据帧"""
        samples = _decode_audio(msg.get("data", ""))
        if samples is None or samples.size == 0:
            return
        if not self._session.should_process_audio():
            return

        if self._session.mode == "background":
            await _handle_background_audio(
                self._ws,
                samples,
                self._session.ring_buffer,
                self._session.kw_service,
            )
        else:
            await _handle_conversation_audio(
                self._ws,
                samples,
                self._session.vad_iterator,
                self._session.current_speech,
                self._session.current_speech_tmp,
            )

    async def _handle_session_control(self, msg: dict) -> None:
        """处理会话开始/结束控制"""
        action = msg.get("action", "")
        if action == "start":
            self._session.start(msg.get("mode", "background"))
        elif action == "end":
            self._session.end()

    async def _handle_ping(self, msg: dict) -> None:
        """处理心跳 ping"""
        await _send(self._ws, {"type": "pong"})


# ===================== WebSocket 主端点 =====================


@voice_ws_api.websocket("/voice/ws")
async def voice_websocket(websocket: WebSocket) -> None:
    """统一语音 WebSocket 端点（双模式）

    职责：仅负责 WebSocket 生命周期管理与消息分发
    """
    await websocket.accept()
    await _send(websocket, {"type": "ready"})

    session = VoiceSession()
    router = MessageRouter(websocket, session)

    try:
        while True:
            raw = await websocket.receive_text()
            await router.route(raw)
    except WebSocketDisconnect:
        logger.info("[VoiceWS] 客户端断开")
    except Exception as e:
        logger.error(f"[VoiceWS] 异常: {e}", exc_info=True)
    finally:
        _asr_in_progress.discard(id(websocket))
        asr_server.release_model()


# ===================== 背景模式（KWS + 环形缓冲区 + 按需 ASR） =====================


async def _handle_background_audio(
    websocket: WebSocket,
    samples: np.ndarray,
    ring_buffer: AudioRingBuffer,
    kw_service: WakeWordService | None,
) -> None:
    """
    背景模式：缓存音频 → KWS 检测 → 命中关键词时 ASR 并返回结果。
    """
    ring_buffer.add(samples)

    if kw_service is None:
        return

    try:
        keyword = kw_service.detect(samples)
    except Exception as e:
        logger.warning(f"[VoiceWS] KWS 检测异常: {e}")
        return

    if not keyword:
        return

    # 查询关键词配置（类别、重要度、触发概率）
    kw_config = _get_keyword_config(keyword)
    if kw_config is None:
        logger.warning(f"[VoiceWS] 未匹配的关键词配置: {keyword}")
        return

    # 按触发概率过滤（概率 < 1.0 时随机跳过）
    if kw_config.probability < 1.0 and random.random() > kw_config.probability:
        logger.info(
            f"[VoiceWS] 关键词 '{keyword}' 概率未命中 ({kw_config.probability})"
        )
        return

    logger.info(
        f"[VoiceWS] 命中关键词: {keyword} | 类别: {kw_config.category} | 重要度: {kw_config.importance}"
    )

    # 标记当前缓冲区位置，确保 ASR 截取时能定位到关键词语音
    ring_buffer.mark()

    # 触发词 → ASR
    if id(websocket) in _asr_in_progress:
        return
    _asr_in_progress.add(id(websocket))

    asyncio.create_task(_run_background_asr(websocket, ring_buffer, kw_config))


async def _run_background_asr(
    websocket: WebSocket,
    ring_buffer: AudioRingBuffer,
    kw_config: KeywordConfig,
) -> None:
    """
    从环形缓冲区提取音频 → ASR → 返回包含类别标签的 asr_result 给前端。

    使用 get_since_mark() 以 KWS 命中标记点为中心截取音频，
    确保关键词语音被完整包含（解决 KWS 检测延迟导致的截断问题）。
    """
    try:
        audio = ring_buffer.get_since_mark(before_ms=3500, after_ms=500)
        if not _audio_quality_filter(audio):
            return
        if len(audio) > MAX_SPEECH_DURATION_SAMPLES:
            audio = audio[:MAX_SPEECH_DURATION_SAMPLES]

        audio_bytes = b""
        with BytesIO() as buf:
            sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            audio_bytes = buf.read()

        text = await asr_server.asr(audio_bytes)
        if not text:
            return

        logger.info(f"[VoiceWS] 背景 ASR: {text}")

        await _send(
            websocket,
            {
                "type": "asr_result",
                "text": text,
                "category": kw_config.category,
                "importance": kw_config.importance,
            },
        )

    except Exception as e:
        logger.error(f"[VoiceWS] 背景 ASR 失败: {e}", exc_info=True)
    finally:
        _asr_in_progress.discard(id(websocket))


# ===================== 对话模式（VAD + ASR + 意图分析） =====================


async def _handle_conversation_audio(
    websocket: WebSocket,
    samples: np.ndarray,
    vad_iterator: VADIterator | None,
    current_speech: list[np.ndarray],
    current_speech_tmp: list[np.ndarray],
) -> None:
    """对话模式：VAD 分割 → ASR → 返回 asr_result"""
    if vad_iterator is None:
        return

    for speech_dict, speech_samples in vad_iterator(samples):
        if "start" in speech_dict:
            current_speech.clear()
            current_speech_tmp.clear()
            await _send_vad(websocket, "speech_start")

        current_speech_tmp.append(speech_samples)

        if "end" in speech_dict:
            await _send_vad(websocket, "speech_end")
            current_speech.extend(current_speech_tmp)
            current_speech_tmp.clear()

            await _process_conversation_asr(websocket, current_speech)
            current_speech.clear()


async def _process_conversation_asr(
    websocket: WebSocket,
    speech_segments: list[np.ndarray],
) -> None:
    """ASR → 返回 asr_result"""
    if not speech_segments:
        return

    try:
        combined = np.concatenate(speech_segments)
        if not _audio_quality_filter(combined):
            return
        if len(combined) > MAX_SPEECH_DURATION_SAMPLES:
            combined = combined[:MAX_SPEECH_DURATION_SAMPLES]

        audio_bytes = b""
        with BytesIO() as buffer:
            sf.write(buffer, combined, 16000, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            audio_bytes = buffer.read()

        text = await asr_server.asr(audio_bytes)
        if not text:
            return

        await _send(
            websocket,
            {"type": "asr_result", "text": text},
        )
        logger.info(f"[VoiceWS] ASR: {text}")
    except Exception as e:
        logger.error(f"[VoiceWS] ASR 处理失败: {e}", exc_info=True)


# ===================== 工具方法 =====================


def _audio_quality_filter(samples: np.ndarray) -> bool:
    if len(samples) < MIN_SPEECH_DURATION_SAMPLES:
        return False
    if len(samples) > MAX_SPEECH_DURATION_SAMPLES:
        return False
    rms = np.sqrt(np.mean(samples**2))
    if rms < MIN_SPEECH_ENERGY:
        return False
    return True


def _decode_audio(data: str) -> np.ndarray | None:
    try:
        samples_i16 = np.frombuffer(base64.b64decode(data), dtype=np.int16)
        if samples_i16.size == 0:
            return None
        return (samples_i16 / 32768.0).astype(np.float32)
    except Exception:
        return None


async def _send(websocket: WebSocket, msg: dict) -> None:
    try:
        await websocket.send_text(json.dumps(msg, ensure_ascii=False))
    except Exception:
        pass


async def _send_vad(websocket: WebSocket, event: str) -> None:
    await _send(
        websocket,
        {
            "type": "vad",
            "event": event,
            "timestamp": int(time.time() * 1000),
        },
    )


async def _send_error(websocket: WebSocket, message: str) -> None:
    await _send(websocket, {"type": "error", "message": message})
