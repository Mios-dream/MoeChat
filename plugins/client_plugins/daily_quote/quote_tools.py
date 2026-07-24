"""
每日一言工具集

客户端同步工具：管理每日一言（Daily Quote）小组件。

包含工具:
- get_quote:    获取当前显示的名言
- refresh_quote: 刷新随机切换名言

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, daily-quote
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
    tags=["ui", "daily-quote"],
    version="1.0.0",
)
class GetQuoteTool(ClientTool):
    """
    获取名言工具

    获取每日一言小组件当前显示的经典名言。
    """

    name: str = "get_quote"
    """工具名称"""

    description: str = (
        "获取每日一句小组件当前显示的经典名言或警句，包括文本、作者和出处。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {},
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'get_quote' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.NORMAL,
    tags=["ui", "daily-quote"],
    version="1.0.0",
)
class RefreshQuoteTool(ClientTool):
    """
    刷新名言工具

    刷新每日一言，随机切换到另一条名言警句。
    """

    name: str = "refresh_quote"
    """工具名称"""

    description: str = (
        "刷新每日一句，随机切换到另一条名言警句。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {},
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'refresh_quote' 不应直接调用 execute()。"
        )
