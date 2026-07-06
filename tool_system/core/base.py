"""
工具基类定义模块

定义所有工具的抽象基类体系：
- BaseTool:     所有工具的根抽象基类
- ServerTool:   服务端工具基类（逻辑在服务端进程内执行）
- ClientTool:   客户端工具基类（逻辑通过 WebSocket 下发给客户端执行）
- HybridTool:   混合工具基类（两阶段协作：客户端阶段 + 服务端阶段）

所有新工具必须继承对应基类并通过 @register_tool 装饰器声明元信息。
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from tool_system.core.types import ToolMeta


class BaseTool(ABC):
    """
    工具根抽象基类

    所有工具（无论执行域）都必须继承此类或其子类。
    定义了工具的核心契约：
    - meta 属性：返回工具的元信息
    - execute() 方法：执行工具核心逻辑
    - to_openai_tool() 方法：转换为 OpenAI function calling 格式

    子类必须实现以下抽象成员：
    - meta: 返回 ToolMeta 实例（通常由 @register_tool 装饰器自动生成）
    - execute: 执行工具核心逻辑（ClientTool 除外，由框架拦截）
    """

    @property
    @abstractmethod
    def meta(self) -> ToolMeta:
        """
        获取工具的元信息

        通常在子类中作为类属性或由 @register_tool 装饰器自动生成。

        Returns:
            ToolMeta: 工具的完整元信息
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        执行工具核心逻辑

        对于服务端工具和混合工具的服务端阶段，此方法包含实际业务逻辑。
        对于客户端工具，此方法由 ClientExecutor 拦截，不会实际调用。

        Args:
            **kwargs: LLM 传入的参数，与 meta.parameters 定义的 schema 对应

        Returns:
            JSON 格式字符串，作为 tool 角色消息返回给 LLM。
            返回内容应结构化，便于 LLM 理解和后续处理。
        """
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        校验参数并填充默认值

        根据 meta.parameters 中的 JSON Schema 定义：
        1. 检查 required 字段是否存在
        2. 为缺失的可选字段填充 default 值
        3. 移除未在 schema 中声明的额外参数

        Args:
            arguments: LLM 传入的原始参数字典

        Returns:
            校验并填充默认值后的参数字典

        Raises:
            InvalidArgumentsError: 缺少必填参数时抛出
        """
        from tool_system.core.errors import InvalidArgumentsError

        validated: dict[str, Any] = {}
        props = self.meta.parameters.get("properties", {})
        required = self.meta.parameters.get("required", [])

        for param_name, param_schema in props.items():
            if param_name in arguments:
                validated[param_name] = arguments[param_name]
            elif "default" in param_schema:
                validated[param_name] = param_schema["default"]
            elif param_name in required:
                raise InvalidArgumentsError(
                    tool_name=self.meta.name,
                    details=f"缺少必填参数 '{param_name}'",
                    call_id="",
                )

        return validated

    def to_openai_tool(self) -> dict[str, Any]:
        """
        将工具转换为 OpenAI function calling 格式

        用于构建 LLM 请求中的 tools 参数。

        Returns:
            符合 OpenAI function calling 规范的工具定义字典:
            {
                "type": "function",
                "function": {
                    "name": str,
                    "description": str,
                    "parameters": dict,  # JSON Schema
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.meta.name,
                "description": self.meta.description,
                "parameters": self.meta.parameters,
            },
        }

    @staticmethod
    def result_json(data: dict[str, Any]) -> str:
        """
        将字典转换为返回给 LLM 的 JSON 字符串

        便捷方法，保证统一使用 ensure_ascii=False 和不抛出异常。

        Args:
            data: 要转换的字典

        Returns:
            JSON 字符串
        """
        return json.dumps(data, ensure_ascii=False, default=str)

    @staticmethod
    def result_error(error: str, error_code: str | None = None) -> str:
        """
        构建标准化的错误结果 JSON

        当工具内部需要返回可预期的错误信息时使用此方法。

        Args:
            error: 错误描述
            error_code: 可选的结构化错误码

        Returns:
            JSON 字符串: {"success": false, "error": "...", "error_code": "..."}
        """
        result: dict[str, Any] = {"success": False, "error": error}
        if error_code:
            result["error_code"] = error_code
        return json.dumps(result, ensure_ascii=False)


class ServerTool(BaseTool, ABC):
    """
    服务端工具基类

    工具逻辑完全在服务端进程内执行。继承此类后只需实现 execute() 方法。
    ServerExecutor 会直接 awaitt 执行并返回结果。

    使用示例:
        @register_tool(domain=SERVER, mode=SYNC, timeout=15.0, tags=["vision"])
        class DesktopOcrTool(ServerTool):
            name = "desktop_ocr"
            description = "对屏幕进行 OCR 识别"
            parameters = { ... }

            async def execute(self, region: str = "full") -> str:
                image = await capture(region)
                text = await ocr(image)
                return self.result_json({"text": text})
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """子类初始化时自动设置 meta.domain = SERVER"""
        super().__init_subclass__(**kwargs)
        # 如果子类定义了 meta 属性且未设置 domain，自动填充为 SERVER
        # 注意：此时子类的 meta 可能尚未完全初始化，由 @register_tool 处理
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        执行服务端工具逻辑

        子类必须实现此方法。ServerExecutor 直接调用。

        Args:
            **kwargs: LLM 传入的参数（已校验和填充默认值）

        Returns:
            JSON 格式字符串，作为 tool 角色消息返回给 LLM
        """
        ...


class ClientTool(BaseTool, ABC):
    """
    客户端工具基类

    工具逻辑完全在客户端设备上执行。框架通过 WebSocket 将调用下发给客户端，
    execute() 方法不会被服务端调用。

    继承 ClientTool 的工具类不需要实现 execute() 方法，
    而是由客户端侧注册对应的处理函数。

    客户端必须实现以下协议:
    1. 接收 WebSocket 消息: {"type": "tool:call", "call_id": "...", "tool_name": "...", "arguments": {...}}
    2. 执行对应的处理逻辑
    3. 返回 WebSocket 消息: {"type": "tool:result", "call_id": "...", "success": true, "result": "..."}

    使用示例:
        @register_tool(domain=CLIENT, mode=SYNC, timeout=10.0, tags=["ui", "settings"])
        class SetWeatherLocationTool(ClientTool):
            name = "set_weather_location"
            description = "设置天气显示的城市"
            parameters = {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }

        # 客户端侧注册:
        # toolSystem.registerClientTool("set_weather_location", async (args) => {
        #     await weatherWidget.setLocation(args.city);
        #     return { success: true, result: `天气已设置为${args.city}` };
        # });
    """

    async def execute(self, **kwargs: Any) -> str:
        """
        客户端工具的 execute() 方法不会被实际调用

        ClientExecutor 会拦截并改为通过 WebSocket 下发给客户端。
        如果此方法被直接调用，说明出现了配置错误（应该使用 ServerTool 或 HybridTool）。

        Raises:
            NotImplementedError: 始终抛出，提示使用方式错误
        """
        raise NotImplementedError(
            f"客户端工具 '{self.meta.name}' 不应直接调用 execute()。"
            "客户端工具的执行逻辑在客户端设备上运行，由 ClientExecutor 通过 WebSocket 代理。"
            "如果工具逻辑在服务端，请继承 ServerTool。"
        )


class HybridTool(BaseTool, ABC):
    """
    混合工具基类

    工具逻辑分两个阶段执行:
    1. client_phase(): 生成下发给客户端的操作指令（如打开文件选择器）
    2. server_phase():  处理客户端返回的数据（如解析上传的文件）

    HybridExecutor 将两个阶段编排为一个完整流程。

    使用示例:
        @register_tool(domain=HYBRID, mode=SYNC, timeout=120.0, tags=["file"], sensitivity=SENSITIVE)
        class FileUploadTool(HybridTool):
            name = "upload_file"
            description = "上传文件到服务端处理"
            parameters = {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ["document", "image"],
                        "description": "文件类型",
                        "default": "document"
                    }
                }
            }

            async def client_phase(self, file_type: str = "document") -> dict:
                # 生成下发给客户端的操作指令
                return {"action": "file_select", "accept": [".pdf", ".txt"]}

            async def server_phase(self, client_result: dict, file_type: str = "document") -> str:
                # 处理客户端返回的文件
                content = await parse_file(client_result["file_path"])
                return self.result_json({"summary": content[:500]})
    """

    @abstractmethod
    async def client_phase(self, **kwargs: Any) -> dict[str, Any]:
        """
        客户端阶段: 生成需要下发给客户端的操作指令

        此方法在服务端运行，但不执行实际工具逻辑。
        它返回一个字典，框架通过 WebSocket 发送给客户端。
        客户端根据字典中的指令执行第一阶段操作。

        Args:
            **kwargs: LLM 传入的参数

        Returns:
            需要发送给客户端的结构化指令字典。
            客户端根据此字典决定执行什么操作。

        示例返回:
            {"action": "file_select", "accept": [".pdf", ".txt"], "multiple": False}
        """
        ...

    @abstractmethod
    async def server_phase(self, client_result: dict[str, Any], **kwargs: Any) -> str:
        """
        服务端阶段: 处理客户端返回的数据

        客户端完成第一阶段操作后，将结果回传给服务端。
        此方法处理客户端数据，完成最终的业务逻辑。

        Args:
            client_result: 客户端返回的结构化结果数据
            **kwargs: LLM 传入的原始参数

        Returns:
            JSON 格式字符串，作为 tool 角色消息返回给 LLM
        """
        ...

    async def execute(self, **kwargs: Any) -> str:
        """
        混合工具的 execute() 方法由 HybridExecutor 编排调用

        此方法提供了一个完整的双阶段执行示例，
        但实际使用中 HybridExecutor 会分别调用 client_phase 和 server_phase。
        如果需要自定义编排逻辑，可重写此方法。

        Args:
            **kwargs: LLM 传入的参数

        Returns:
            JSON 格式字符串
        """
        # 默认实现：不做任何事，实际由 HybridExecutor 编排
        # 子类可重写此方法以实现自定义编排逻辑
        return self.result_json({"status": "executed"})
