"""
WebSocket 工具协议消息模型

严格类型化的 WS 双向通信消息，替代原始的 dict[str, Any] 裸字典。
与 chat_protocol.py 的 ChatWSMessageType 枚举一一对应。

消息方向:
    服务端 → 客户端:
        ToolCallWsMessage      - 下发工具调用
        ToolCancelWsMessage    - 取消工具调用
        ToolAsyncResultWsMessage - 异步工具完成通知

    客户端 → 服务端:
        ToolResultWsMessage    - 客户端工具执行结果
        ToolProgressWsMessage  - 客户端工具执行进度
        ToolConfirmWsMessage   - 用户确认/拒绝敏感工具

使用方式:
    from models.dto.response.ToolWsResponse import ToolCallWsMessage

    msg = ToolCallWsMessage(
        call_id="call_xxx",
        tool_name="set_weather",
        arguments={"city": "北京"},
        timeout_ms=10000,
        sensitivity="normal",
        mode="sync",
    )
    await ws_manager.send(session_id, msg.model_dump())
"""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# 服务端 → 客户端 消息
# ═══════════════════════════════════════════════════════════════


class ToolQueryWsMessage(BaseModel):
    """
    服务端查询客户端工具能力

    客户端连接成功后服务端立即发送此消息，
    客户端需回复 ToolDefinitionsWsMessage 列出所有已注册的客户端工具。

    协议字段:
        type: 固定 "tool:query"
    """

    type: Literal["tool:query"] = "tool:query"
    """消息类型标识，固定值"""


class ToolCallWsMessage(BaseModel):
    """
    下发工具调用给客户端

    服务端通过 WebSocket 将 LLM 发起的工具调用下发给客户端执行。
    客户端收到后应执行对应工具逻辑，完成后回复 ToolResultWsMessage。

    协议字段:
        type: 固定 "tool:call"
        call_id: 全局唯一的工具调用 ID（OpenAI tool_call_id）
        tool_name: 工具名称，客户端据此路由到对应的处理函数
        arguments: LLM 填充的工具参数字典
        timeout_ms: 超时时间（毫秒），客户端应在此时间内完成
        sensitivity: 敏感度等级，决定是否需要用户确认
        mode: 执行模式（sync=同步等待 / async=fire-and-forget）
        confirm_message: 敏感工具的确认提示文本，非敏感工具为 None
    """

    type: Literal["tool:call"] = "tool:call"
    """消息类型标识，固定值"""

    call_id: str
    """OpenAI tool_call_id，格式如 'call_xxxxx'，全局唯一"""

    tool_name: str
    """工具名称，客户端据此查找对应的工具处理函数"""

    arguments: dict[str, Any] = Field(default_factory=dict)
    """LLM 填充的工具参数字典"""

    timeout_ms: int = 30000
    """客户端执行超时时间（毫秒），默认 30s"""

    sensitivity: Literal["safe", "normal", "sensitive", "dangerous"] = "normal"
    """工具敏感度等级

    - safe:      自动执行，无需确认
    - normal:    根据用户设置决定是否确认
    - sensitive: 需要弹出确认对话框
    - dangerous: 需要二次身份验证
    """

    mode: Literal["sync", "async"] = "sync"
    """执行模式

    - sync:  客户端需阻塞等待并即时返回结果
    - async: 客户端可异步执行，稍后通过 tool:result 回传
    """

    confirm_message: str | None = None
    """敏感工具确认提示文本，仅 sensitivity 为 sensitive/dangerous 时有效"""


class ToolCancelWsMessage(BaseModel):
    """
    取消已下发的工具调用

    当用户取消、超时或上下文变化时，服务端通知客户端中止执行。

    协议字段:
        type: 固定 "tool:cancel"
        call_id: 要取消的工具调用 ID
        reason: 取消原因（用于客户端日志/UI 显示）
    """

    type: Literal["tool:cancel"] = "tool:cancel"
    """消息类型标识，固定值"""

    call_id: str
    """要取消的工具调用 ID"""

    reason: str = "服务端已取消该工具调用"
    """取消原因描述"""


class ToolAsyncResultWsMessage(BaseModel):
    """
    异步工具完成后向客户端推送结果

    当客户端工具以 async 模式执行完成后，服务端将 LLM 处理结果
    回推给客户端，客户端可用于 UI 更新。

    协议字段:
        type: 固定 "tool:async_result"
        call_id: 对应的工具调用 ID
        success: 执行是否成功
        result: 工具执行结果（JSON 字符串）
        data: 额外携带的结构化数据
    """

    type: Literal["tool:async_result"] = "tool:async_result"
    """消息类型标识，固定值"""

    call_id: str
    """对应的工具调用 ID"""

    success: bool = True
    """工具执行是否成功"""

    result: str = ""
    """工具执行结果，JSON 格式字符串"""

    data: dict[str, Any] | None = None
    """额外携带的结构化数据，如文件路径、缩略图等"""


# ═══════════════════════════════════════════════════════════════
# 客户端 → 服务端 消息
# ═══════════════════════════════════════════════════════════════


class ToolResultWsMessage(BaseModel):
    """
    客户端工具执行结果回调

    客户端执行完工具后通过此消息将结果回传给服务端。
    服务端通过 PendingCallTable 唤醒等待的 Future。

    协议字段:
        type: 固定 "tool:result"
        call_id: 对应的工具调用 ID（必须与 ToolCallWsMessage 中的一致）
        success: 执行是否成功
        result: 工具执行结果，JSON 格式字符串
        error: 失败时的错误描述
        error_code: 结构化错误码
        data: 额外携带的结构化数据
    """

    type: Literal["tool:result"] = "tool:result"
    """消息类型标识，固定值"""

    call_id: str
    """对应的工具调用 ID，用于匹配 pending call"""

    success: bool = True
    """工具执行是否成功"""

    result: str = ""
    """工具执行结果，JSON 格式字符串，LLM 可直接理解"""

    error: str | None = None
    """失败时的错误描述"""

    error_code: str | None = None
    """结构化错误码，如 TOOL_EXEC_ERROR / TOOL_TIMEOUT"""

    data: dict[str, Any] | None = None
    """额外携带的结构化数据"""


class ToolProgressWsMessage(BaseModel):
    """
    客户端工具执行进度推送

    长时间执行的客户端工具（如文件下载、截图处理）可在执行过程中
    定期推送进度，服务端记录日志或转发为 UI 进度条。

    协议字段:
        type: 固定 "tool:progress"
        call_id: 对应的工具调用 ID
        tool_name: 工具名称
        status: 当前状态标识
        progress: 进度值（0.0 ~ 1.0，-1.0 表示不确定）
        message: 人类可读的进度描述
    """

    type: Literal["tool:progress"] = "tool:progress"
    """消息类型标识，固定值"""

    call_id: str
    """对应的工具调用 ID"""

    tool_name: str = ""
    """工具名称"""

    status: Literal["started", "executing", "finalizing"] = "executing"
    """当前状态标识

    - started:    工具已开始执行
    - executing:  执行中
    - finalizing: 收尾处理中
    """

    progress: float = -1.0
    """进度值，范围 0.0 ~ 1.0

    -1.0 表示无法确定进度（如网络下载无法预知总大小）
    """

    message: str = ""
    """人类可读的进度描述，如「正在下载模型文件...」"""


class ToolConfirmWsMessage(BaseModel):
    """
    用户对敏感工具调用的确认/拒绝响应

    敏感工具（sensitive）或危险工具（dangerous）被下发到客户端后，
    客户端弹出确认对话框，用户操作后通过此消息回复。

    协议字段:
        type: 固定 "tool:confirm"
        call_id: 对应的工具调用 ID
        confirmed: 用户是否确认执行
        deny_reason: 拒绝原因
        extra_data: 额外信息（如二次验证码）
    """

    type: Literal["tool:confirm"] = "tool:confirm"
    """消息类型标识，固定值"""

    call_id: str
    """对应的工具调用 ID"""

    confirmed: bool
    """用户是否确认执行

    - True:  确认执行，服务端继续工具调用流程
    - False: 取消执行，服务端中断工具调用
    """

    deny_reason: str = ""
    """用户拒绝执行的原因，如「用户取消了操作」"""

    extra_data: dict[str, Any] = Field(default_factory=dict)
    """额外信息，如二次验证码等"""


# ═══════════════════════════════════════════════════════════════
# 客户端 → 服务端：工具定义上报
# ═══════════════════════════════════════════════════════════════


class ClientToolDefEntry(BaseModel):
    """
    客户端单个工具定义条目（扁平格式，兼容客户端直接发送）

    客户端发送格式示例:
        {
            "name": "set_weather_location",
            "description": "设置天气查询的目标城市",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "..."}},
                "required": ["city"]
            }
        }

    协议字段:
        name: 工具名称，必须与 ToolRegistry 注册名一致
        description: 工具描述
        parameters: JSON Schema 参数定义
    """

    name: str
    """工具名称"""

    description: str = ""
    """工具描述"""

    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema 参数定义"""


class ComponentToolGroup(BaseModel):
    """
    客户端组件维度工具组

    一个客户端组件（如 weather 天气组件）可能注册多个工具。
    按组件分组上报，服务端校验后写入 SessionToolTable 的组件映射。

    协议字段:
        component: 组件标识（如 'weather' / 'todo' / 'file'）
        version: 组件版本号
        tools: 该组件注册的工具定义列表
    """

    component: str
    """组件标识，如 'weather' / 'todo'"""

    version: str = "1.0.0"
    """组件版本号"""

    tools: list[ClientToolDefEntry] = Field(default_factory=list)
    """该组件注册的工具定义列表"""


class ToolDefinitionsWsMessage(BaseModel):
    """
    客户端上报工具定义（按组件分组）

    客户端收到服务端 tool:query 后回复此消息，
    列出所有已注册的客户端工具定义。
    服务端逐条校验 name + parameters schema，
    校验通过的写入该会话的 SessionToolTable。

    协议字段:
        type: 固定 "tool:definitions"
        components: 按组件分组的工具定义列表
    """

    type: Literal["tool:definitions"] = "tool:definitions"
    """消息类型标识，固定值"""

    components: list[ComponentToolGroup] = Field(default_factory=list)
    """按组件分组的工具定义列表"""


# ═══════════════════════════════════════════════════════════════
# 类型导出汇总
# ═══════════════════════════════════════════════════════════════

# 服务端 → 客户端
ServerToClientToolMessage = (
    ToolQueryWsMessage | ToolCallWsMessage | ToolCancelWsMessage | ToolAsyncResultWsMessage
)

# 客户端 → 服务端
ClientToServerToolMessage = (
    ToolResultWsMessage
    | ToolProgressWsMessage
    | ToolConfirmWsMessage
    | ToolDefinitionsWsMessage
)

# 全部 WS 工具消息
WsToolMessage = ServerToClientToolMessage | ClientToServerToolMessage

__all__ = [
    # 服务端 → 客户端
    "ToolQueryWsMessage",
    "ToolCallWsMessage",
    "ToolCancelWsMessage",
    "ToolAsyncResultWsMessage",
    # 客户端 → 服务端
    "ToolResultWsMessage",
    "ToolProgressWsMessage",
    "ToolConfirmWsMessage",
    # 客户端 → 服务端：工具定义上报
    "ClientToolDefEntry",
    "ComponentToolGroup",
    "ToolDefinitionsWsMessage",
    # 类型别名
    "ServerToClientToolMessage",
    "ClientToServerToolMessage",
    "WsToolMessage",
]
