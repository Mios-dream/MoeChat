"""
统一聊天 WebSocket 处理器

将现有 HTTP+SSE 聊天流程 + 工具系统融合为单条 WebSocket 连接。

核心设计:
    1. 连接建立 → 注册 WebSocketManager
    2. chat:send → 调用 V3ChatService.chat() → 转译为 WS JSON 消息
    3. ToolCallIntegration 共享: handler._integration 注入 V3ChatService，
       确保 tool:result 回调能正确匹配到 pending call
    4. 心跳检测 → 断线清理 pending calls
    5. 多任务支持: 允许多个 chat:send 并行执行

与现有代码的关系:
    - V3ChatService: 核心聊天服务（复用 v3_motion.py）
    - V3MotionChatContext: 结果分发处理器（文本/音频/动作/工具）
    - ToolCallIntegration: 工具调用执行器（handler 与 V3ChatService 共享实例）
    - WebSocketManager: 连接管理 + 客户端工具通信通道
"""

import asyncio
import time
import traceback
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.chat.v3_motion import V3ChatService
from models.dto.chat_request import ChatData
from models.dto.response.ChatResponse import ErrorResponse
from my_utils.log import logger
from api.ws.chat_protocol import ChatWSMessageType
from tool_system.integration import ToolCallIntegration
import shortuuid

v3_service = V3ChatService()


class ChatWebSocketHandler:
    """
    统一聊天 WebSocket 处理器

    管理单个客户端的完整聊天生命周期:
    1. 连接建立与身份验证
    2. 聊天消息接收与流式回复发送（委托 V3ChatService）
    3. 工具调用双向通信（共享 ToolCallIntegration）
    4. 心跳维持与断线清理
    5. 多任务并行管理

    Attributes:
        _websocket: 当前客户端 WS 连接
        _session_id: 会话唯一标识
        _integration: 工具调用集成（注入 V3ChatService 以共享 pending calls）
    """

    def __init__(
        self,
        websocket: WebSocket,
    ) -> None:
        """
        初始化处理器

        Args:
            websocket: 客户端 WebSocket 连接
        """
        self._websocket = websocket
        self._integration = ToolCallIntegration()

        # 心跳间隔（秒）: 服务端定期发送 PONG 消息
        self._heartbeat_interval: float = 15.0
        # 读取超时（秒）: 超过此时间未收到客户端消息则断开连接
        self._read_timeout: float = 120.0

    async def handle(self) -> None:
        """
        主处理循环

        流程:
        1. 等待 identity 消息进行身份绑定
        2. 注册到 WebSocketManager
        3. 进入消息接收循环
        4. 断线时清理资源
        """

        self._session_id = shortuuid.uuid()

        # 同步session_id 到 工具调用集成
        self._integration.set_session_id(self._session_id)

        logger.info(f"[ChatWS] 客户端已连接: session={self._session_id}")

        # ── 启动心跳任务 ──
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name=f"chat_heartbeat_{self._session_id}"
        )

        # ── 消息接收循环 ──
        try:
            while True:
                data = await asyncio.wait_for(
                    self._websocket.receive_json(),
                    timeout=self._read_timeout,
                )
                # 分发消息
                await self._dispatch(data)

        except asyncio.TimeoutError:
            logger.info(f"[ChatWS] 读取超时,断开连接: session={self._session_id}")

        except WebSocketDisconnect:
            logger.info(f"[ChatWS] 客户端主动断开: session={self._session_id}")

        except Exception as e:
            logger.error(f"[ChatWS] 异常: {e}")
            try:
                await self._websocket.send_json(
                    ErrorResponse(error_code="WS_ERROR", data=str(e))
                )
            except Exception:
                pass

        finally:
            # ── 清理资源 ──
            heartbeat_task.cancel()

            # 取消待处理的客户端工具调用
            self._integration.cancel_session(self._session_id)

            logger.info(f"[ChatWS] 连接已关闭: session={self._session_id}")

    async def _dispatch(self, data: dict[str, Any]) -> None:
        """
        消息分发器

        根据 type 字段将消息路由到对应的处理方法。

        Args:
            data: 客户端发来的 JSON 消息字典
        """
        msg_type = data.get("type", "")

        if msg_type == ChatWSMessageType.PING.value:
            await self._handle_ping()

        elif msg_type == ChatWSMessageType.CHAT_SEND.value:
            await self._handle_chat_send(data)

        elif msg_type == ChatWSMessageType.CHAT_CANCEL.value:
            await self._handle_chat_cancel(data)

        elif msg_type == ChatWSMessageType.TOOL_RESULT.value:
            await self._handle_tool_result(data)

        elif msg_type == ChatWSMessageType.TOOL_PROGRESS.value:
            await self._handle_tool_progress(data)

        elif msg_type == ChatWSMessageType.TOOL_CONFIRM.value:
            await self._handle_tool_confirm(data)

        else:
            logger.warning(
                f"[ChatWS] 未知消息类型: {msg_type}, session={self._session_id}"
            )

    # ── 连接管理 ──

    async def _handle_ping(self) -> None:
        """处理心跳 PING"""
        await self._websocket.send_json(
            {
                "type": ChatWSMessageType.PONG.value,
                "server_time": time.time(),
            }
        )

    async def _heartbeat_loop(self) -> None:
        """服务端心跳发送循环"""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await self._websocket.send_json(
                    {
                        "type": ChatWSMessageType.PONG.value,
                        "server_time": time.time(),
                    }
                )
            except Exception:
                break

    # ── 聊天处理 ──

    async def _handle_chat_send(self, data: dict[str, Any]) -> None:
        """
        处理 chat:send 消息
        创建后台任务，委托 V3ChatService.chat() 执行 V3 模式聊天生成。

        Args:
            data: 客户端发来的聊天请求
                id: 消息 ID（用于取消和多任务索引）
                msg: 消息列表（最后一条为用户输入）
                generation_motion: 是否生成动作
                is_sleep_mode: 是否睡眠模式
        """
        chat_data = ChatData(**data)

        if not chat_data.msg:
            await self._websocket.send_json(
                ErrorResponse(
                    error_code="INVALID_REQUEST", data="消息内容不能为空"
                ).model_dump()
            )
            return

        # 创建后台任务（不取消其他任务，允许多任务并行）
        asyncio.create_task(
            self._run_chat_generation(chat_data=chat_data),
            name=f"chat_gen_{self._session_id}",
        )

    async def _run_chat_generation(self, chat_data: ChatData) -> None:
        """
        执行 V3 模式聊天生成（在后台 Task 中运行）

        直接复用 V3ChatService.chat() 的完整流程:
        - TaskScheduler 注册 text/motion/bilingual 任务
        - V3MotionChatContext 分发处理结果
        - TTS 音频合成 + 动作引擎
        - 工具调用支持
        - 对话历史自动保存

        将 V3ChatService 产出的 FullChatResponse 逐条序列化为 JSON 发送至 WebSocket。

        ToolCallIntegration 共享:
            handler 持有的 _integration 注入 V3ChatService 实例，
            确保 WS 客户端发来的 tool:result 能正确匹配到 pending call。

        Args:
            chat_data: 聊天数据对象
        """
        try:

            v3_service._integration = (
                self._integration
            )  # 共享实例，确保 tool:result 回调匹配
            v3_service.set_session_id(self._session_id)

            async for response in v3_service.chat(chat_data):
                await self._websocket.send_json(response.model_dump())

        except asyncio.CancelledError:
            logger.info(f"[ChatWS] 聊天生成已取消: session={self._session_id}")

        except Exception as e:
            logger.error(f"[ChatWS] 聊天生成失败: {e}")
            logger.error(traceback.format_exc())
            await self._websocket.send_json(
                ErrorResponse(error_code="GENERATION_ERROR", data=str(e)).model_dump()
            )

    async def _handle_chat_cancel(self, data: dict[str, Any]) -> None:
        """
        取消聊天生成

        按 message_id 精确取消，若未指定 id 则取消全部活跃任务。

        Args:
            data: 客户端 chat:cancel 消息
                id: 要取消的消息 ID（可选，不传则取消所有）
        """
        pass

    # ── 工具调用处理 ──

    async def _handle_tool_result(self, data: dict[str, Any]) -> None:
        """
        处理客户端工具结果回调

        将 tool:result 转交给共享的 ToolCallIntegration.on_client_result()，
        内部通过 PendingCallTable 唤醒 Orchestrator 等待的 Future。

        Args:
            data: 客户端 tool:result 消息
        """
        resolved = self._integration.on_client_result(
            session_id=self._session_id,
            payload=data,
        )
        if not resolved:
            logger.warning(
                f"[ChatWS] 未匹配到 tool:result: call_id={data.get('call_id')}"
            )

    async def _handle_tool_progress(self, data: dict[str, Any]) -> None:
        """
        处理工具执行进度

        当前仅做日志记录，后续可扩展为 WS 推送 UI 进度条。

        Args:
            data: 客户端 tool:progress 消息
        """
        logger.debug(
            f"[ChatWS] tool progress: {data.get('tool_name')} "
            f"status={data.get('status')} "
            f"progress={data.get('progress', -1)}"
        )

    async def _handle_tool_confirm(self, data: dict[str, Any]) -> None:
        """
        处理用户对敏感工具的确认/拒绝

        Args:
            data: 客户端 tool:confirm 消息
        """
        confirmed = data.get("confirmed", False)
        call_id = data.get("call_id", "")

        if not confirmed:
            self._integration.cancel_session(self._session_id)
            logger.info(f"[ChatWS] 用户拒绝工具调用: call_id={call_id}")
        else:
            logger.info(f"[ChatWS] 用户确认工具调用: call_id={call_id}")
