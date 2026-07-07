"""
WebSocket 连接管理器模块

WebSocketManager 是整个 WS 通信层的核心组件，作为模块级单例
所有 ChatWebSocketHandler 实例共享同一管理器，用于：
1. 管理 session_id → WebSocket 连接的映射关系
2. 提供客户端工具调用的消息下发通道
3. 支持连接状态检测与自动清理
4. 支持按 session_id 精确路由消息

角色定位:
    工具系统 ←→ WebSocketManager ←→ WebSocket 连接池 ←→ 客户端

工具系统通过 ws_manager.send(session_id, message) 将工具调用下发到
对应客户端，客户端通过 tool:result 消息回传结果。
"""

import asyncio
import time
from typing import Any

from fastapi import WebSocket
from my_utils.log import logger


class WebSocketManager:
    """
    WebSocket 连接管理器（模块级单例）

    维护所有活跃的 WebSocket 连接，按 session_id 索引。
    为工具系统提供统一的客户端通信接口。

    设计要点:
    - 一个 session_id 对应一个 WebSocket 连接
    - 线程安全：所有操作在同一个 event loop 中执行
    - 自动清理断线连接的残留状态

    Attributes:
        _connections: session_id → WebSocket 连接的映射
        _heartbeat_times: session_id → 最后心跳时间戳的映射
    """

    def __init__(self) -> None:
        """初始化连接管理器"""
        self._connections: dict[str, WebSocket] = {}
        """session_id → 活跃的 WebSocket 连接"""

        self._heartbeat_times: dict[str, float] = {}
        """session_id → 最后一次收到心跳的时间戳"""

        self._lock = asyncio.Lock()
        """保护 _connections 字典的异步锁（防止并发修改）"""

    async def register(self, session_id: str, websocket: WebSocket) -> None:
        """
        注册新的 WebSocket 连接

        ChatWebSocketHandler 在首次接收到客户端消息后调用此方法，
        将 WebSocket 连接与会话 ID 绑定。

        Args:
            session_id: 会话唯一标识（由 handler 生成）
            websocket: 客户端的 WebSocket 连接实例
        """
        async with self._lock:
            # 如果同一 session_id 已有连接（重连场景），先移除旧连接
            if session_id in self._connections:
                old_ws = self._connections.pop(session_id)
                try:
                    await old_ws.close(code=1008, reason="Session replaced")
                except Exception:
                    pass

            self._connections[session_id] = websocket
            self._heartbeat_times[session_id] = time.time()

        logger.debug(
            f"[WSManager] 注册连接: session={session_id}, "
            f"活跃连接数={len(self._connections)}"
        )

    async def unregister(self, session_id: str) -> None:
        """
        注销并清理 WebSocket 连接

        客户端断开连接或会话结束时调用。

        Args:
            session_id: 要注销的会话 ID
        """
        async with self._lock:
            self._connections.pop(session_id, None)
            self._heartbeat_times.pop(session_id, None)

        logger.debug(
            f"[WSManager] 注销连接: session={session_id}, "
            f"剩余连接数={len(self._connections)}"
        )

    async def send(self, session_id: str, message: dict[str, Any]) -> None:
        """
        向指定客户端发送 JSON 消息

        工具系统通过此方法将 tool:call 等消息下发到客户端。

        Args:
            session_id: 目标会话 ID
            message: 要发送的 JSON 消息字典

        Raises:
            KeyError: 指定 session_id 的连接不存在
            ConnectionError: 发送失败（连接已断开）
        """
        async with self._lock:
            websocket = self._connections.get(session_id)
            if websocket is None:
                raise KeyError(f"会话 '{session_id}' 不存在或已断开连接")

        try:
            await websocket.send_json(message)
        except Exception as e:
            # 发送失败意味着连接异常，自动清理
            await self.unregister(session_id)
            raise ConnectionError(f"发送消息到会话 '{session_id}' 失败: {e}") from e

    async def is_connected(self, session_id: str) -> bool:
        """
        检查指定会话的连接是否活跃

        客户端工具执行器在发送 tool:call 前调用此方法
        确认客户端在线。

        Args:
            session_id: 要检查的会话 ID

        Returns:
            连接是否活跃
        """
        async with self._lock:
            return session_id in self._connections

    async def broadcast(self, message: dict[str, Any]) -> None:
        """
        向所有已连接的客户端广播消息

        Args:
            message: 要广播的 JSON 消息
        """
        async with self._lock:
            connections = list(self._connections.items())

        disconnected: list[str] = []
        for session_id, ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(session_id)

        # 清理断线连接
        for sid in disconnected:
            await self.unregister(sid)

    async def update_heartbeat(self, session_id: str) -> None:
        """
        更新指定会话的心跳时间戳

        ChatWebSocketHandler 收到客户端 PING 时调用。

        Args:
            session_id: 会话 ID
        """
        async with self._lock:
            if session_id in self._connections:
                self._heartbeat_times[session_id] = time.time()

    @property
    def connection_count(self) -> int:
        """当前活跃连接数（不可 await，仅用于日志/监控）"""
        return len(self._connections)

    def get_active_sessions(self) -> list[str]:
        """获取所有活跃会话 ID 列表（不可 await）"""
        return list(self._connections.keys())

    async def close_all(self) -> None:
        """
        关闭所有 WebSocket 连接

        服务关闭时调用，优雅断开所有客户端。
        """
        async with self._lock:
            sessions = list(self._connections.items())
            self._connections.clear()
            self._heartbeat_times.clear()

        for session_id, ws in sessions:
            try:
                await ws.close(code=1001, reason="Server shutdown")
            except Exception:
                pass

        logger.info(f"[WSManager] 已关闭所有 {len(sessions)} 个连接")

    def __repr__(self) -> str:
        """人类可读的状态"""
        return f"<WebSocketManager: {self.connection_count} active connections>"


_ws_manager_singleton: WebSocketManager | None = None
"""模块级全局 WebSocketManager 单例"""


def get_ws_manager() -> WebSocketManager:
    """
    获取模块级 WebSocketManager 单例

    懒初始化：首次调用时创建实例。
    所有 ChatWebSocketHandler 和工具执行器共享同一实例。

    Returns:
        全局唯一的 WebSocketManager 实例
    """
    global _ws_manager_singleton
    if _ws_manager_singleton is None:
        _ws_manager_singleton = WebSocketManager()
    return _ws_manager_singleton
