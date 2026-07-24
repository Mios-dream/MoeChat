"""
时钟格式设置工具

客户端同步工具：切换时钟组件的时间显示格式。

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, clock, settings
Sensitivity: NORMAL
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
    tags=["ui", "clock", "settings"],
    version="1.0.0",
)
class SetClockFormatTool(ClientTool):
    """
    时钟格式设置工具

    切换时钟组件的 12 小时制 / 24 小时制显示。
    """

    name: str = "set_clock_format"
    """工具名称"""

    description: str = (
        "切换时钟的 12 小时制 / 24 小时制显示。"
        "true 为 24 小时制，false 为 12 小时制。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "is_24h": {
                "type": "boolean",
                "description": "是否使用 24 小时制，true=24H，false=12H",
            },
        },
        "required": ["is_24h"],
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'set_clock_format' 不应直接调用 execute()。"
        )
