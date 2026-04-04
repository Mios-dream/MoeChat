"""
语音唤醒API模块
提供语音唤醒功能的RESTful接口和WebSocket实时通信
"""

import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from my_utils.log import logger
from services.assistant_service import AssistantService
from services.wakeword_service import WakeWordService

wakeword_api = APIRouter()
assistant_service = AssistantService()


@wakeword_api.websocket("/wakeword/ws")
async def wakeword_websocket(websocket: WebSocket) -> None:
    """
    语音唤醒WebSocket端点
    实时处理音频流并检测唤醒词

    消息格式:
        客户端发送二进制音频数据 (Int16 PCM格式)
        服务端返回JSON消息:
            {
                "type": "ready",
                "message": "wakeword service ready"
            }
            {
                "type": "wakeword_detected",
                "keyword": " detected_keyword",
                "timestamp_ms": 1234567890
            }
            {
                "type": "error",
                "message": "error message"
            }

    Args:
        websocket: WebSocket连接对象
    """
    await websocket.accept()
    session = None

    try:
        # 创建会话
        try:
            current_assistant = assistant_service.get_current_assistant()
            if current_assistant:
                # 组合别称和角色名称，并清除空字符串
                keywords = [
                    item
                    for item in [
                        current_assistant.char,
                        *current_assistant.alias.split(","),
                    ]
                    if item.strip()
                ]
                # logger.info(f"[WakeWord] Keywords: {keywords}")
                wakewordService = WakeWordService(keywords=keywords)
            else:
                await _send_error(websocket, "No active assistant found")
                await websocket.close()
                return

            session = wakewordService.create_session()
        except Exception as e:
            logger.error(f"[WakeWord] Failed to create session: {e}", exc_info=True)
            await _send_error(websocket, f"Failed to create session: {str(e)}")
            await websocket.close()
            return

        # 发送就绪状态
        await _send_message(
            websocket, {"type": "ready", "message": "WakeWord service is ready"}
        )

        # 处理音频流
        while True:
            try:
                # 接收二进制音频数据
                audio_bytes = await websocket.receive_bytes()
                audio_samples = _parse_audio_bytes(audio_bytes)

                if audio_samples is None:
                    continue

                # 检测唤醒词
                keyword = wakewordService.detect(audio_samples)
                if keyword:
                    await _send_message(
                        websocket,
                        {
                            "type": "wakeword_detected",
                            "keyword": keyword,
                            "timestamp_ms": int(time.time() * 1000),
                        },
                    )

            except WebSocketDisconnect:
                logger.info("[WakeWord] Client disconnected normally")
                break
            except Exception as e:
                logger.error(f"[WakeWord] Error processing audio: {e}")
                await _send_error(websocket, f"Audio processing error: {str(e)}")

    except Exception as e:
        logger.error(f"[WakeWord] WebSocket error: {e}", exc_info=True)
        try:
            await _send_error(websocket, f"Service error: {str(e)}")
        except:
            pass
    finally:
        if session:
            # 清理会话资源
            del session
        try:
            await websocket.close()
        except:
            pass


def _parse_audio_bytes(audio_bytes: bytes) -> np.ndarray | None:
    """
    解析二进制音频数据

    Args:
        audio_bytes: 二进制音频数据 (Int16 PCM格式)

    Returns:
        np.ndarray: 解析后的音频采样数据，失败返回None
    """
    try:
        if not audio_bytes or len(audio_bytes) == 0:
            return None

        # 直接从二进制数据转换为Int16数组
        samples_i16 = np.frombuffer(audio_bytes, dtype=np.int16)

        if samples_i16.size == 0:
            return None

        # 转换为float32并归一化到[-1, 1]范围
        return (samples_i16 / 32768.0).astype(np.float32)

    except Exception as e:
        logger.error(f"[WakeWord] Audio parsing error: {e}")
        return None


async def _send_message(websocket: WebSocket, message: dict) -> None:
    """发送消息到WebSocket客户端"""
    await websocket.send_text(json.dumps(message, ensure_ascii=False))


async def _send_error(websocket: WebSocket, error_message: str) -> None:
    """发送错误消息到WebSocket客户端"""
    await _send_message(websocket, {"type": "error", "message": error_message})
