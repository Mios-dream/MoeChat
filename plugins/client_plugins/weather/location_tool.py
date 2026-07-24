"""
天气位置设置工具

客户端同步工具示例：设置天气组件显示的城市和区域。

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, settings, weather
Sensitivity: NORMAL

使用场景:
    - "帮我把天气调到北京"
    - "设置天气位置为上海浦东"
    - "切换天气显示城市"

客户端实现指南:
    客户端需要监听 WebSocket 消息 type='tool:call'，
    当 tool_name='set_weather_location' 时执行以下逻辑:

    1. 解析 arguments.city 和 arguments.district
    2. 调用天气组件 API 设置显示位置
    3. 返回 tool:result 消息:
       {
         "type": "tool:result",
         "call_id": "...",
         "success": true,
         "result": "天气位置已设置为北京市海淀区"
       }

    示例 JavaScript 代码:
    ```javascript
    toolSystem.registerClientTool("set_weather_location", async (args) => {
        try {
            await weatherWidget.setLocation(args.city, args.district || '');
            return {
                success: true,
                result: `天气位置已设置为${args.city}${args.district ? args.district : ''}`
            };
        } catch (e) {
            return {
                success: false,
                error: `设置天气位置失败: ${e.message}`,
                error_code: "TOOL_EXEC_ERROR"
            };
        }
    });
    ```
"""

from tool_system.core.base import ClientTool
from tool_system.core.enums import (
    ExecutionDomain,
    ExecutionMode,
    ToolSensitivity,
)
from tool_system.core.registry import register_tool
from tool_system.core.types import ToolMeta


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.NORMAL,
    tags=["ui", "settings", "weather"],
    version="1.0.0",
)
class SetWeatherLocationTool(ClientTool):
    """
    天气位置设置工具

    设置天气组件显示的城市和区域。
    工具逻辑在客户端设备上执行，服务端通过 WebSocket 下发调用指令。

    客户端必须实现:
    1. 接收 tool:call 消息并解析 tool_name 和 arguments
    2. 调用天气组件 API 设置位置
    3. 返回 tool:result 消息（包含成功/失败状态）
    """

    name: str = "set_weather_location"
    """工具名称：LLM 通过此名称调用"""

    description: str = (
        "设置天气查询的目标城市，修改后小组件会自动刷新显示该城市的实时天气。"
    )
    """工具描述：告诉 LLM 何时使用此工具"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "城市名称，如 '北京'、'上海'、'深圳'。"
                    "使用中文完整名称。"
                ),
            },
        },
        "required": ["city"],
    }
    """
    JSON Schema 参数定义

    city: 必填，城市名称
    """

    @property
    def meta(self) -> ToolMeta:
        """获取工具元信息（由 @register_tool 装饰器自动注入 _tool_meta）"""
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        """
        客户端工具的 execute() 不会被服务端调用

        此方法仅为文档说明用途，实际执行流程:
        1. ClientSyncExecutor 通过 WebSocket 下发 tool:call
        2. 客户端执行 weatherWidget.setLocation(city, district)
        3. 客户端返回 tool:result

        Returns:
            占位 JSON（实际不会被使用）

        Raises:
            NotImplementedError: 始终
        """
        # ClientTool 的 execute() 由 ClientExecutor 拦截，
        # 不会在此处执行。保留此方法是为了满足 ABC 的抽象方法要求。
        raise NotImplementedError(
            "客户端工具 'set_weather_location' 不应直接调用 execute()。"
            "工具逻辑在客户端设备上运行，由 ClientExecutor 通过 WebSocket 代理。"
        )
