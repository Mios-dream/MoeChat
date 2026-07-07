"""
聊天 WebSocket API 路由

将 WebSocket 连接委托给 ChatWebSocketHandler 处理完整聊天生命周期，
包括身份认证、消息路由、工具调用双向通信。

WebSocketManager 作为模块级单例，所有客户端连接共享
同一管理器实例，用于跨连接的客户端工具下发与回调。
"""

from fastapi import APIRouter, WebSocket
from api.ws.chat_handler import ChatWebSocketHandler

chat_ws_api = APIRouter()


@chat_ws_api.websocket("/chat_ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """
    统一聊天 WebSocket 端点

    连接即用，零配置。服务端内部管理所有状态。

    客户端使用示例:
        const ws = new WebSocket("ws://localhost:8020/api/chat_ws");

        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: "identity",
                user_id: "user_xxx"
            }));
            ws.send(JSON.stringify({
                type: "chat:send",
                id: "msg_1",
                content: "你好!"
            }));
        };

        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === "chat:text") {
                process.stdout.write(msg.content);
            } else if (msg.type === "tool:call") {
                // 执行客户端工具 → 回复 tool:result
                ws.send(JSON.stringify({
                    type: "tool:result",
                    call_id: msg.call_id,
                    success: true,
                    result: "..."
                }));
            }
        };
    """
    await websocket.accept()
    handler = ChatWebSocketHandler(
        websocket=websocket,
    )
    await handler.handle()
