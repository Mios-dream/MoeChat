"""
获取天气数据工具

客户端同步工具：获取天气组件当前显示的实时天气数据。

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, weather
Sensitivity: SAFE
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
    sensitivity=ToolSensitivity.SAFE,
    tags=["ui", "weather"],
    version="1.0.0",
)
class GetWeatherTool(ClientTool):
    """
    获取天气数据工具

    获取天气组件当前显示的实时天气数据，包括位置、天气状况和温度。
    工具逻辑在客户端设备上执行。
    """

    name: str = "get_weather"
    """工具名称"""

    description: str = (
        "获取小组件当前显示的天气数据，包括位置、天气状况和温度。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {},
    }
    """无参数的 JSON Schema"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'get_weather' 不应直接调用 execute()。"
        )
