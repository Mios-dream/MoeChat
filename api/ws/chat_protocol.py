"""
统一聊天 WebSocket 协议定义

将 HTTP+SSE 聊天 + 工具调用双向通信统一为单条 WebSocket 连接。

设计原则:
    1. 一条连接承载: 聊天消息发送、流式响应接收、工具调用双向通信
    2. REST API 保留: 配置管理、助手管理、历史记录等无状态操作
    3. 向后兼容: 消息格式可覆盖原 SSE event 类型,前端迁移成本低
    4. 音频 WS 独立: ASR/VAD/WakeWord 处理二进制音频数据,保持独立

消息格式 (JSON):
    {
        "type": "消息类型",
        "id": "可选的消息ID(用于请求-响应匹配)",
        ... 各类型特有字段
    }

连接生命周期:
    connect → identity → chat_loop → disconnect
"""

from enum import Enum
from typing import Any


class ChatWSMessageType(str, Enum):
    """
    统一聊天 WebSocket 消息类型

    覆盖原 SSE 事件 + 工具系统协议的所有通信需求。
    """

    # ═══════════════════════════════════════════════════════════
    # 客户端 → 服务端
    # ═══════════════════════════════════════════════════════════

    PING = "ping"
    """
    应用层心跳请求: {type, timestamp}

    注意: WebSocket 协议(RFC 6455)原生 Ping/Pong 帧(Opcode 0x9/0xA)
    由 uvicorn 在传输层自动处理,应用层不可见。
    此处为补充的应用层心跳,检测业务层连接僵死。
    """

    # ── 对话 ──
    CHAT_SEND = "chat:send"
    """
    发送聊天消息: {
        type, id,
        content: str,                用户输入文本
        generation_motion: bool,     是否生成 Live2D 动作
        is_sleep_mode: bool,         是否睡眠模式
        include_history: bool,       是否包含历史
        history_limit: int,          历史消息条数上限
    }
    """

    CHAT_CANCEL = "chat:cancel"
    """取消当前生成: {type, id(要取消的消息ID)}"""

    # ── 工具系统 ──
    TOOL_RESULT = "tool:result"
    """
    客户端工具执行结果: {
        type, call_id, success: bool,
        result: str, error: str?, error_code: str?, data: {}
    }
    """

    TOOL_PROGRESS = "tool:progress"
    """
    客户端工具执行进度: {
        type, call_id, status: str,
        progress: float, message: str
    }
    """

    TOOL_CONFIRM = "tool:confirm"
    """
    用户对敏感工具的确认: {
        type, call_id, confirmed: bool,
        deny_reason: str?, extra_data: {}
    }
    """

    # ═══════════════════════════════════════════════════════════
    # 服务端 → 客户端
    # ═══════════════════════════════════════════════════════════

    # ── 连接管理 ──
    CONNECTED = "connected"
    """连接确认: {type, server_time}"""

    PONG = "pong"
    """
    心跳响应: {type, timestamp, server_time}

    对应应用层 PING 的响应,不做业务逻辑处理。
    传输层 Ping/Pong 帧由 uvicorn 自动处理,与此独立。
    """

    # ── 对话流式输出 ──
    CHAT_TOKEN = "chat:text"
    """
    文本 token: {
        type, content: str,
        sentence_id: int?,          句子序号(多句场景)
        timestamp_ms: float
    }
    等价于原 SSE: data: {"type": "text", "message": "...", "sentence_id": ...}
    """

    CHAT_AUDIO = "chat:audio"
    """
    TTS 音频: {
        type, sentence_id: int,
        file: str?,                  Base64 音频数据
        format: str,                音频格式(mp3/wav)
        duration_ms: int,           音频时长
        timestamp_ms: float
    }
    等价于原 SSE: data: {"type": "audio", "file": "...", ...}
    """

    CHAT_MOTION = "chat:motion"
    """
    Live2D 动作帧: {
        type, sentence_id: int,
        motions: [{duration, curves, fps}],
        expression: str?,
        duration: int,
        timestamp_ms: float
    }
    等价于原 SSE: data: {"type": "motion_frame", ...}
    """

    CHAT_DONE = "chat:done"
    """
    本轮回复完成: {
        type, full_text: str,
        elapsed_ms: float,
        tool_calls_count: int,      本轮工具调用次数
        timestamp_ms: float
    }
    等价于原 SSE: data: {"type": "done", "full_text": "...", "done": true}
    """

    CHAT_ERROR = "chat:error"
    """
    错误: {
        type, data: str,
        error_code: str?,
        timestamp_ms: float
    }
    等价于原 SSE: data: {"type": "error", "data": "...", "done": true}
    """

    # ── 工具系统 ──
    TOOL_CALL = "tool:call"
    """
    下发工具调用给客户端: {
        type, call_id: str,
        tool_name: str,
        arguments: {},
        timeout_ms: int,
        sensitivity: str,           safe/normal/sensitive/dangerous
        confirm_message: str?,      需要确认时的提示文本
        mode: str                   sync/async
    }
    """

    TOOL_CANCEL = "tool:cancel"
    """取消工具调用: {type, call_id, reason}"""

    TOOL_ASYNC_RESULT = "tool:async_result"
    """异步工具完成通知: {type, call_id, success, result, data}"""

    # ── 系统通知 ──
    SYSTEM_NOTIFY = "system:notify"
    """
    系统通知(如好感度变化、后台任务完成等): {
        type, title: str,
        content: str,
        level: str,                 info/warning/error
        timestamp_ms: float
    }
    """
