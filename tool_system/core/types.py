"""
工具系统核心数据类型定义模块

定义工具系统所有核心数据结构：
- ToolMeta: 工具元信息（声明式注册时定义）
- ToolCallRequest: 工具调用请求（由 LLM tool_call 解析后派发）
- ToolCallResult: 工具调用结果（执行器返回的统一结果）
- ToolCallProgress: 工具执行进度（长时间操作推送）
- ToolConfirmResponse: 用户确认响应（敏感工具确认流程）
"""

from dataclasses import dataclass, field
from typing import Any

from tool_system.core.enums import (
    ExecutionDomain,
    ExecutionMode,
    ToolSensitivity,
)


@dataclass(slots=True)
class ToolMeta:
    """
    工具元信息 - 描述一个工具的完整静态属性

    每个工具通过 @register_tool 装饰器声明这些元信息，
    ToolRegistry 以 name 为主键管理所有工具的元信息。

    核心属性：
    - name: 全局唯一工具标识符，LLM 通过此名称调用工具
    - description: 工具描述，指导 LLM 何时/如何使用
    - domain + mode: 决定路由策略（6 种组合）
    - parameters: OpenAI function calling 兼容的 JSON Schema

    Attributes:
        name: 工具名称，全局唯一，LLM 调用时的标识符
        description: 工具功能描述，告诉 LLM 何时以及如何使用此工具
        domain: 执行域，指定工具代码的物理执行位置
        mode: 执行模式，指定 LLM 如何等待工具结果
        parameters: JSON Schema 格式的参数定义
        timeout: 超时时间（秒）
        max_retries: 失败最大重试次数
        sensitivity: 敏感度等级，决定是否需要用户确认
        tags: 工具标签列表，用于分组和筛选
        placeholder: 异步工具的占位回复模板
        notify_on_complete: 异步工具完成后是否主动通知
        version: 工具版本号
        author: 工具作者
        deprecation_message: 非空表示工具已废弃，内容为迁移指引
    """

    # ── 基础标识 ──
    name: str
    """工具名称，LLM 调用时使用的标识符，全局唯一，如 'desktop_ocr'"""

    description: str
    """工具功能描述，告诉 LLM 在何时以及如何使用此工具"""

    # ── 执行属性 ──
    domain: ExecutionDomain
    """工具执行域：SERVER(服务端) / CLIENT(客户端) / HYBRID(混合)"""

    mode: ExecutionMode = ExecutionMode.SYNC
    """工具执行模式：SYNC(同步阻塞) / ASYNC(异步不阻塞)"""

    # ── 参数定义 ──
    parameters: dict[str, Any] = field(default_factory=dict)
    """
    OpenAI function calling 兼容的 JSON Schema 参数定义。

    示例:
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    """

    # ── 运行时约束 ──
    timeout: float = 30.0
    """工具执行超时时间（秒），超时后返回 ToolTimeoutError"""

    max_retries: int = 0
    """工具执行失败后的最大重试次数，0 表示不重试"""

    sensitivity: ToolSensitivity = ToolSensitivity.NORMAL
    """工具敏感度等级，决定是否需要用户确认"""

    # ── 组织与元数据 ──
    tags: list[str] = field(default_factory=list)
    """
    工具标签列表，用于分组和构建工具集。

    建议标签分类:
        "vision"   - 视觉/图像相关
        "audio"    - 音频相关
        "system"   - 系统操作
        "network"  - 网络请求
        "ui"       - UI 交互
        "file"     - 文件操作
        "ai"       - AI 服务调用
        "settings" - 配置修改
    """

    version: str = "1.0.0"
    """工具版本号，使用语义化版本格式"""

    author: str = ""
    """工具作者/维护者"""

    # ── 异步工具专属 ──
    placeholder: str = ""
    """
    异步工具的占位回复模板。

    当 mode=ASYNC 时，框架将此文本作为过渡信息返回给 LLM，
    引导 LLM 生成自然的处理中过渡回复。

    示例: "正在为您处理截图请求，请稍候..."

    若为空字符串，框架使用默认占位文本。
    """

    notify_on_complete: bool = True
    """
    异步工具完成后是否主动触发新一轮 LLM 回复。

    True:  工具完成后 ResultNotifier 将结果注入 LLM 上下文并触发回复生成
    False: 工具完成后结果存入会话状态，等待 LLM 后续对话中自然引用
    """

    # ── 废弃管理 ──
    deprecation_message: str = ""
    """
    工具废弃提示。非空字符串表示该工具已被废弃。
    内容应包含迁移指引，告诉使用者替代工具的名称或使用方式。

    示例: "该工具已废弃，请使用 'screenshot_v2' 替代"
    """

    # ── 组件归属 ──
    component: str = ""
    """
    工具归属的客户端组件标识，如 'weather' / 'todo' / 'file'。
    
    仅用于 CLIENT 域工具。客户端通过 tool:definitions 上报工具时
    按组件维度组织，服务端校验后写入 SessionToolTable 的组件映射。
    
    为空字符串表示未指定组件（视为全局通用工具）。
    """

    # ── 内部使用 ──
    tool_class: type | None = field(default=None, repr=False, compare=False)
    """工具对应的类引用，由 @register_tool 自动填充，不需要手动设置"""


@dataclass(slots=True)
class ToolCallRequest:
    """
    工具调用请求 - LLM 决定调用工具后由框架生成的内部请求对象

    从 OpenAI tool_call 消息解析而来，附加了执行域、执行模式等路由信息。
    ToolRouter 根据此对象将调用分发到对应的执行器。

    Attributes:
        call_id: OpenAI tool_call_id，全局唯一
        tool_name: 工具名称，与 ToolMeta.name 对应
        arguments: LLM 填充的调用参数
        domain: 工具执行域，从 ToolMeta 继承
        mode: 工具执行模式，从 ToolMeta 继承
        session_id: 会话 ID，用于 WebSocket 路由回调
        timeout: 超时时间（秒），从 ToolMeta 继承
        max_retries: 最大重试次数，从 ToolMeta 继承
        sensitivity: 敏感度等级，从 ToolMeta 继承
        context: 额外上下文信息（user_id, chat_id 等）
    """

    call_id: str
    """OpenAI 协议中的 tool_call_id，格式如 'call_xxxxx'"""

    tool_name: str
    """工具名称，与 ToolMeta.name 严格一致"""

    arguments: dict[str, Any]
    """LLM 填充的调用参数，已解析为字典"""

    domain: ExecutionDomain
    """工具执行域"""

    mode: ExecutionMode
    """工具执行模式"""

    session_id: str
    """会话 ID，客户端工具通过此 ID 回传结果"""

    timeout: float = 30.0
    """执行超时时间（秒）"""

    max_retries: int = 0
    """失败最大重试次数"""

    sensitivity: ToolSensitivity = ToolSensitivity.NORMAL
    """敏感度等级"""

    context: dict[str, Any] = field(default_factory=dict)
    """
    额外上下文信息，至少包含:
        "user_id":   str  当前用户标识
        "chat_id":   str  当前对话 ID（可选）
        "client_id": str  客户端连接标识（可选）
    """


@dataclass(slots=True)
class ToolCallResult:
    """
    工具调用结果 - 工具执行后返回的统一结果对象

    执行器将执行结果包装为此对象，ResultAggregator 将其转换为
    OpenAI tool 角色消息返回给 LLM。

    对于失败的情况，content 字段包含结构化的错误 JSON，
    LLM 可据此向用户解释错误原因。

    Attributes:
        call_id: 对应的请求 call_id
        tool_name: 工具名称
        content: 返回给 LLM 的文本（JSON 格式字符串）
        success: 是否执行成功
        error: 失败时的错误描述
        error_code: 结构化错误码
        duration_ms: 执行耗时（毫秒）
        retry_count: 实际重试次数
        client_payload: 需要推送客户端的额外数据
        is_async_result: 是否为异步工具回调结果
        session_id: 所属会话 ID（异步结果回调时使用）
    """

    call_id: str
    """对应的请求 call_id"""

    tool_name: str
    """工具名称"""

    content: str
    """返回给 LLM 的文本内容，JSON 格式字符串"""

    success: bool = True
    """是否执行成功"""

    error: str | None = None
    """失败时的错误描述，供日志和 LLM 使用"""

    error_code: str | None = None
    """
    结构化错误码，用于程序化处理。

    取值见 ToolErrorCode（TOOL_NOT_FOUND, TOOL_TIMEOUT 等）
    """

    # ── 可观测性 ──
    duration_ms: float = 0.0
    """工具执行总耗时（毫秒），用于性能监控"""

    retry_count: int = 0
    """实际执行的重试次数"""

    # ── 客户端推送 ──
    client_payload: dict[str, Any] | None = None
    """
    需要单独推送到客户端的额外数据载荷。

    与 content 不同，此字段不传给 LLM，而是通过 WebSocket
    独立推送给客户端渲染。用于进度更新、UI 状态变更等场景。
    """

    # ── 异步结果标记 ──
    is_async_result: bool = False
    """标记此结果为异步工具回调结果，Orchestrator 需特殊处理（注入上下文而非 tool 消息）"""

    session_id: str = ""
    """异步结果回调时使用的会话 ID"""


@dataclass(slots=True)
class ToolCallProgress:
    """
    工具执行进度 - 长时间操作的进度通知

    通过 WebSocket 推送给客户端，用于在 UI 上展示进度条或状态更新。

    Attributes:
        call_id: 对应的工具调用 ID
        tool_name: 工具名称
        status: 当前状态标识
        progress: 进度值（0.0 ~ 1.0）
        message: 人类可读的进度描述
        data: 额外进度数据
    """

    call_id: str
    """对应的工具调用 ID"""

    tool_name: str
    """工具名称"""

    status: str
    """
    当前状态标识:
        "started"     - 已开始执行
        "executing"   - 执行中
        "finalizing"  - 收尾处理中
    """

    progress: float = -1.0
    """
    进度值，范围 0.0 ~ 1.0。
    -1.0 表示无法确定进度（如网络下载无法预知总大小）
    """

    message: str = ""
    """人类可读的进度描述，如 '正在下载模型文件...'"""

    data: dict[str, Any] = field(default_factory=dict)
    """额外的进度数据，客户端用于自定义渲染"""


@dataclass(slots=True)
class ToolConfirmResponse:
    """
    用户确认响应 - 敏感工具确认流程中客户端返回的响应

    服务端收到此响应后决定是否继续执行敏感工具。

    Attributes:
        call_id: 对应的工具调用 ID
        confirmed: 用户是否确认执行
        extra_data: 额外信息（二次验证数据等）
        deny_reason: 拒绝原因（用户取消时）
    """

    call_id: str
    """对应的工具调用 ID"""

    confirmed: bool
    """用户是否确认执行（True=确认, False=取消）"""

    extra_data: dict[str, Any] = field(default_factory=dict)
    """额外信息，如二次验证码等"""

    deny_reason: str = ""
    """用户拒绝执行的原因（如 '用户取消了操作'）"""


@dataclass(slots=True)
class ClientToolDef:
    """
    客户端上报的单个工具定义

    客户端通过 tool:definitions 消息上报正在使用的工具定义，
    每条定义必须包含完整的 OpenAI function calling schema 和所属组件。

    Attributes:
        name: 工具名称，必须与 ToolRegistry 中的注册名一致
        description: 工具描述
        parameters: JSON Schema 参数定义
        component: 所属组件标识（如 'weather' / 'todo'）
        component_version: 组件版本号
    """

    name: str
    """工具名称，必须与服务端 ToolRegistry 中注册的名称一致"""

    description: str
    """工具功能描述"""

    parameters: dict[str, Any] = field(default_factory=dict)
    """OpenAI function calling 兼容的 JSON Schema 参数定义"""

    component: str = ""
    """所属客户端组件标识，如 'weather' / 'todo' / 'file'"""

    component_version: str = "1.0.0"
    """客户端组件版本号"""


class SessionToolTable:
    """
    会话级客户端工具表

    管理单个会话中校验通过的客户端工具集。
    客户端通过 WebSocket 连接后上报支持的客户端工具定义，
    服务端按 name + parameters schema 逐条校验，
    只有完全匹配的才会写入本表。

    内部数据结构:
        _tools: dict[name, ToolMeta] - 校验通过的工具元信息
        _component_map: dict[component, set[name]] - 组件 → 工具名称集合
        _definition_map: dict[name, ClientToolDef] - 客户端原始定义

    生命周期:
        - 客户端连接 → tool:query 触发上报
        - 收到 tool:definitions → 逐条校验 → 写入本表
        - 客户端断开 → clear()
    """

    def __init__(self) -> None:
        """初始化空的会话工具表"""
        self._tools: dict[str, ToolMeta] = {}
        """name → ToolMeta（校验通过的客户端工具元信息）"""

        self._component_map: dict[str, set[str]] = {}
        """component → {tool_name, ...}（组件维度索引）"""

        self._definition_map: dict[str, ClientToolDef] = {}
        """name → ClientToolDef（客户端原始定义，用于校验和调试）"""

        self._mismatch_log: list[str] = []
        """校验不通过的记录列表"""

    def register_from_client(
        self,
        definitions: list[ClientToolDef],
        registry: Any,
    ) -> int:
        """
        从客户端上报的工具定义批量校验并注册

        逐条比对服务端 ToolRegistry 中已注册的 CLIENT 域工具：
        1. name 必须匹配已注册工具
        2. parameters.required 字段名集合必须一致
        3. parameters.properties 的 key 不能少于服务端定义

        校验通过的写入内部索引。

        Args:
            definitions: 客户端上报的 ClientToolDef 列表
            registry: ToolRegistry 全局单例引用

        Returns:
            校验通过并成功注册的工具数量
        """
        self.clear()
        count = 0

        for client_tool_def in definitions:
            tool_name = client_tool_def.name
            component = client_tool_def.component or ""

            meta = registry.get(tool_name)
            if meta is None:
                self._mismatch_log.append(f"[SKIP] '{tool_name}': 服务端未注册此工具")
                continue

            if meta.domain.name != "CLIENT":
                self._mismatch_log.append(
                    f"[SKIP] '{tool_name}': 服务端注册域为 {meta.domain.value}，非 CLIENT"
                )
                continue

            if not self._validate_parameters(meta, client_tool_def):
                continue

            self._tools[tool_name] = meta
            self._definition_map[tool_name] = client_tool_def

            if component:
                if component not in self._component_map:
                    self._component_map[component] = set()
                self._component_map[component].add(tool_name)

            count += 1

        return count

    def _validate_parameters(
        self,
        server_meta: ToolMeta,
        client_def: ClientToolDef,
    ) -> bool:
        """
        校验客户端工具参数 schema 是否与服务端定义一致

        校验项：
        1. required 字段名集合必须一致
        2. properties key 集必须一致（不允许缺失，允许客户端多）

        Args:
            server_meta: 服务端注册的 ToolMeta
            client_def: 客户端上报的 ClientToolDef

        Returns:
            校验通过 True / 不通过 False
        """
        tool_name = client_def.name
        server_params = server_meta.parameters or {}
        client_params = client_def.parameters or {}

        server_props = server_params.get("properties", {})
        client_props = client_params.get("properties", {})

        server_required = set(server_params.get("required", []))
        client_required = set(client_params.get("required", []))

        if server_required != client_required:
            missing_in_client = server_required - client_required
            extra_in_client = client_required - server_required
            parts = []
            if missing_in_client:
                parts.append(f"缺少 required: {list(missing_in_client)}")
            if extra_in_client:
                parts.append(f"多余 required: {list(extra_in_client)}")
            self._mismatch_log.append(f"[MISMATCH] '{tool_name}': {', '.join(parts)}")
            return False

        server_keys = set(server_props.keys())
        client_keys = set(client_props.keys())
        missing_keys = server_keys - client_keys
        if missing_keys:
            self._mismatch_log.append(
                f"[MISMATCH] '{tool_name}': " f"缺少 properties: {list(missing_keys)}"
            )
            return False

        return True

    def get_tools(self) -> list[ToolMeta]:
        """
        获取该会话所有校验通过的客户端工具元信息

        Returns:
            ToolMeta 列表
        """
        return list(self._tools.values())

    def get_component_tools(self, component: str) -> list[ToolMeta]:
        """
        按组件获取工具

        Args:
            component: 组件标识

        Returns:
            该组件下的 ToolMeta 列表
        """
        names = self._component_map.get(component, set())
        return [self._tools[n] for n in names if n in self._tools]

    def get_component_names(self) -> list[str]:
        """
        获取所有已注册的组件名称

        Returns:
            组件名称列表
        """
        return list(self._component_map.keys())

    def get_mismatch_log(self) -> list[str]:
        """
        获取校验不通过记录

        Returns:
            日志行列表
        """
        return list(self._mismatch_log)

    def clear(self) -> None:
        """清空所有会话工具数据（客户端断开时调用）"""
        self._tools.clear()
        self._component_map.clear()
        self._definition_map.clear()
        self._mismatch_log.clear()

    def __contains__(self, name: str) -> bool:
        """检查工具是否在会话表中（支持 'name' in table 语法）"""
        return name in self._tools

    def __len__(self) -> int:
        """已注册的客户端工具数量"""
        return len(self._tools)

    def __repr__(self) -> str:
        """可读的会话工具表状态"""
        comps = ", ".join(f"{c}:{len(v)}" for c, v in self._component_map.items())
        return f"<SessionToolTable: {len(self._tools)} tools " f"[{comps}]>"
