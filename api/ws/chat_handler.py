"""
统一聊天 WebSocket 处理器

将现有 HTTP+SSE 聊天流程 + 工具系统融合为单条 WebSocket 连接。

核心设计:
    1. 连接建立 → 注册 WebSocketManager
    2. chat:send → 调用 V3ChatService.chat() → 转译为 WS JSON 消息
    3. 工具系统注入: handler 持有 ToolCallIntegration（含 ws_manager），
    4. 心跳检测 → 断线清理 pending calls
    5. 多任务支持: 允许多个 chat:send 并行执行

与现有代码的关系:
    - V3ChatService: 核心聊天服务（chat() 方法纯粹，通过 integration 属性按需配置工具）
    - V3MotionChatContext: 结果分发处理器（文本/音频/动作/工具）
    - ToolCallIntegration: 工具调用执行器（由 handler 持有，注入 V3ChatService）
    - WebSocketManager: 连接管理 + 客户端工具通信通道
"""

import asyncio
import json
import time
import traceback
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.chat.v3_motion import V3ChatService
from core.interaction_core import generate_interaction_message
from models.dto.request.chat_request import ChatRequest
from models.dto.request.interaction_request import InteractionMessageRequest
from models.dto.response.ChatResponse import ErrorResponse
from my_utils.log import logger
from api.ws.chat_protocol import ChatWSMessageType
from tool_system.integration import ToolCallIntegration
from models.dto.response.ToolWsResponse import (
    ToolQueryWsMessage,
    ToolResultWsMessage,
    ToolProgressWsMessage,
    ToolConfirmWsMessage,
    ToolDefinitionsWsMessage,
)
from api.ws.ws_manager import get_ws_manager
from tool_system.core.types import ClientToolDef
import shortuuid

v3_service = V3ChatService()

# ── 模块级 WebSocketManager 单例 ──
ws_manager = get_ws_manager()


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
            ws_manager: WebSocket 连接管理器（模块级单例，用于客户端工具通信）
        """
        self._websocket = websocket
        self._integration = ToolCallIntegration()

        # 将 WS 管理器注入工具系统（使客户端工具和混合工具可工作）
        self._integration.set_ws_manager(ws_manager)

        # 读取超时（秒）: 超过此时间未收到客户端消息则断开连接
        self._read_timeout: float = 120.0

    async def handle(self) -> None:
        """
        主处理循环

        流程:
        1. 注册到 WebSocketManager
        2. 进入消息接收循环（客户端应定期发 ping 保活，超时 120s 断连）
        3. 断线时清理资源
        """

        self._session_id = shortuuid.uuid()

        # 注册连接 → WebSocketManager（客户端工具通信通道）
        await ws_manager.register(self._session_id, self._websocket)

        # 同步session_id 到 工具调用集成
        self._integration.set_session_id(self._session_id)

        logger.info(f"[ChatWS] 客户端已连接: session={self._session_id}")

        # ── 查询客户端工具能力 ──
        await self._query_client_tools()

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

            # 从 WS 管理器注销连接
            await ws_manager.unregister(self._session_id)

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

        elif msg_type == ChatWSMessageType.TOOL_DEFINITIONS.value:
            await self._handle_tool_definitions(data)

        elif msg_type == ChatWSMessageType.INTERACTION_SEND.value:
            await self._handle_interaction_send(data)

        else:
            logger.warning(
                f"[ChatWS] 未知消息类型: {msg_type}, session={self._session_id}"
            )

    # ── 连接管理 ──

    async def _handle_ping(self) -> None:
        """
        处理心跳 PING

        客户端定期发送 ping，服务端回复 pong 并更新心跳时间。
        客户端必须每 _read_timeout 秒内发送一条消息（ping 或任何消息），
        否则服务端判定连接僵死并断开。
        """
        await ws_manager.update_heartbeat(self._session_id)
        await self._websocket.send_json(
            {
                "type": ChatWSMessageType.PONG.value,
                "server_time": time.time(),
            }
        )

    # ── 聊天处理 ──

    async def _handle_chat_send(self, data: dict[str, Any]) -> None:
        """
        处理 chat:send 消息
        创建后台任务，委托 V3ChatService.chat() 执行 V3 模式聊天生成。

        Args:
            data: 客户端发来的聊天请求
                id: 消息 ID
                text: 用户文本消息
                images: Base64 图片列表
                files: 文件附件列表
                generation_motion: 是否生成动作
                is_sleep_mode: 是否睡眠模式
        """
        chat_request = ChatRequest(**data)

        # 创建后台任务（不取消其他任务，允许多任务并行）
        asyncio.create_task(
            self._run_chat_generation(chat_request=chat_request),
            name=f"chat_gen_{self._session_id}",
        )

    async def _run_chat_generation(self, chat_request: ChatRequest) -> None:
        """
        执行 V3 模式聊天生成（在后台 Task 中运行）

        直接复用 V3ChatService.chat() 的完整流程:
        - TaskScheduler 注册 text/motion/bilingual 任务
        - V3MotionChatContext 分发处理结果
        - TTS 音频合成 + 动作引擎
        - 工具调用支持（通过 integration 参数注入）
        - 对话历史自动保存

        将 V3ChatService 产出的 FullChatResponse 逐条序列化为 JSON 发送至 WebSocket。

        ToolCallIntegration 通过 chat() 方法的 integration 参数注入 V3ChatService，
        handler 持有的 _integration 与 V3ChatService 共享同一个实例，
        确保 WS 客户端发来的 tool:result 能正确匹配到 pending call。

        Args:
            chat_data: 聊天数据对象
        """
        try:

            self._integration.set_session_id(self._session_id)
            v3_service.set_integration(self._integration)

            async for response in v3_service.chat(chat_request):
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

    # ── 交互事件处理 ──

    async def _handle_interaction_send(self, data: dict[str, Any]) -> None:
        """
        处理 interaction:send 消息

        创建后台任务，执行交互消息生成管道。
        响应经 WS JSON 发送，格式与 chat:send 完全一致
        （chat:text / chat:audio / chat:motion / chat:done）。

        Args:
            data: 客户端发来的交互事件请求
                event_type: 事件类型（如 sleep.talk）
                scene: 场景描述
                context: 上下文信息
                generation_motion: 是否生成动作
                include_history: 是否包含历史
                history_limit: 历史条数上限
        """
        try:
            params = InteractionMessageRequest(**data)
        except Exception as e:
            await self._websocket.send_json(
                ErrorResponse(
                    error_code="INVALID_REQUEST",
                    data=f"交互请求参数无效: {e}",
                ).model_dump()
            )
            return

        asyncio.create_task(
            self._run_interaction_generation(params=params),
            name=f"interaction_gen_{self._session_id}",
        )

    async def _run_interaction_generation(
        self, params: InteractionMessageRequest
    ) -> None:
        """
        执行交互消息生成（在后台 Task 中运行）

        调用 generate_interaction_message_ws() 获取 FullChatResponse 流，
        逐条序列化为 JSON 发送至 WebSocket。

        Args:
            params: 交互请求参数
        """
        try:
            async for response in generate_interaction_message(params):
                await self._websocket.send_json(response.model_dump())

        except asyncio.CancelledError:
            logger.info(f"[ChatWS] 交互生成已取消: session={self._session_id}")

        except Exception as e:
            logger.error(f"[ChatWS] 交互生成失败: {e}")
            logger.error(traceback.format_exc())
            await self._websocket.send_json(
                ErrorResponse(error_code="INTERACTION_ERROR", data=str(e)).model_dump()
            )

    # ── 工具调用处理 ──

    async def _handle_tool_result(self, data: dict[str, Any]) -> None:
        """
        处理客户端工具结果回调

        将 tool:result 转交给共享的 ToolCallIntegration.on_client_result()，
        内部通过 PendingCallTable 唤醒 Orchestrator 等待的 Future。

        使用 ToolResultWsMessage 进行严格类型验证后再处理。

        Args:
            data: 客户端 tool:result 消息的原始字典
        """
        try:
            msg = ToolResultWsMessage.model_validate(data)
        except Exception as e:
            logger.warning(f"[ChatWS] tool:result 消息格式无效: {e}")
            return

        resolved = self._integration.on_client_result(
            session_id=self._session_id,
            payload=msg.model_dump(),
        )
        if not resolved:
            logger.warning(f"[ChatWS] 未匹配到 tool:result: call_id={msg.call_id}")

    async def _handle_tool_progress(self, data: dict[str, Any]) -> None:
        """
        处理工具执行进度

        使用 ToolProgressWsMessage 进行严格类型验证后记录日志。

        Args:
            data: 客户端 tool:progress 消息的原始字典
        """
        try:
            msg = ToolProgressWsMessage.model_validate(data)
        except Exception as e:
            logger.warning(f"[ChatWS] tool:progress 消息格式无效: {e}")
            return

        logger.debug(
            f"[ChatWS] tool progress: tool={msg.tool_name} "
            f"status={msg.status} progress={msg.progress} "
            f"message={msg.message}"
        )

    async def _handle_tool_confirm(self, data: dict[str, Any]) -> None:
        """
        处理用户对敏感工具的确认/拒绝

        使用 ToolConfirmWsMessage 进行严格类型验证后处理确认结果。

        Args:
            data: 客户端 tool:confirm 消息的原始字典
        """
        try:
            msg = ToolConfirmWsMessage.model_validate(data)
        except Exception as e:
            logger.warning(f"[ChatWS] tool:confirm 消息格式无效: {e}")
            return

        if not msg.confirmed:
            self._integration.cancel_session(self._session_id)
            logger.info(f"[ChatWS] 用户拒绝工具调用: call_id={msg.call_id}")
        else:
            logger.info(f"[ChatWS] 用户确认工具调用: call_id={msg.call_id}")

    # ── 工具定义协商 ──

    async def _query_client_tools(self) -> None:
        """
        向客户端发送工具能力查询

        连接建立后立即发送 tool:query，
        触发客户端上报已注册的客户端工具定义。
        """
        try:
            query_msg = ToolQueryWsMessage()
            await self._websocket.send_json(query_msg.model_dump())
        except Exception as e:
            logger.warning(
                f"[ChatWS] 发送 tool:query 失败: {e}, " f"session={self._session_id}"
            )

    async def _handle_tool_definitions(self, data: dict[str, Any]) -> None:
        """
        处理客户端上报的工具定义

        客户端收到 tool:query 后回复 tool:definitions，
        按组件分组列出所有已注册的客户端工具定义。
        服务端逐条校验后写入该会话的 SessionToolTable。

        Args:
            data: 客户端 tool:definitions 消息的原始字典
        """
        print(f"[ChatWS] 收到 tool:definitions: {json.dumps(data, ensure_ascii=False)}")
        try:
            msg = ToolDefinitionsWsMessage.model_validate(data)
        except Exception as e:
            logger.warning(
                f"[ChatWS] tool:definitions 消息格式无效: {e}, "
                f"session={self._session_id}"
            )
            return

        # ── 提取所有工具定义为 ClientToolDef 列表 ──
        definitions: list[ClientToolDef] = []
        for comp_group in msg.components:
            component = comp_group.component
            version = comp_group.version
            for entry in comp_group.tools:
                client_tool_def = ClientToolDef(
                    name=entry.name,
                    description=entry.description,
                    parameters=entry.parameters,
                    component=component,
                    component_version=version,
                )
                definitions.append(client_tool_def)

        # ── 校验并注册 ──
        count, mismatches = self._integration.register_client_tools(
            session_id=self._session_id,
            definitions=definitions,
        )

        comp_summary = ", ".join(
            f"{g.component}({len(g.tools)})" for g in msg.components
        )
        logger.info(
            f"[ChatWS] tool:definitions 接收完毕: "
            f"session={self._session_id}, "
            f"components=[{comp_summary}], "
            f"上报 {len(definitions)} 个工具, "
            f"校验通过 {count} 个"
        )
