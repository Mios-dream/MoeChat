"""
便签工具集

客户端同步工具：管理澪的便签（Note）。

包含工具:
- set_note:   设置便签标题和/或正文
- get_note:   获取当前便签内容
- clear_note: 清空便签

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, note
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
    tags=["ui", "note"],
    version="1.0.0",
)
class SetNoteTool(ClientTool):
    """
    设置便签工具

    设置便签的标题和/或正文内容。可以只更新标题、只更新内容，或两者都更新。
    """

    name: str = "set_note"
    """工具名称"""

    description: str = (
        "设置便签的标题和/或正文内容。可以只更新标题、只更新内容，或两者都更新。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "便签标题（可选），不传则保持原标题不变",
            },
            "content": {
                "type": "string",
                "description": "便签正文内容，不传则保持原内容不变",
            },
        },
    }
    """参数定义（全部可选）"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'set_note' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.SAFE,
    tags=["ui", "note"],
    version="1.0.0",
)
class GetNoteTool(ClientTool):
    """
    获取便签内容工具

    获取当前便签的完整内容。
    """

    name: str = "get_note"
    """工具名称"""

    description: str = (
        "获取当前便签的完整内容，包括标题和正文。"
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
            "客户端工具 'get_note' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.SENSITIVE,
    tags=["ui", "note"],
    version="1.0.0",
)
class ClearNoteTool(ClientTool):
    """
    清空便签工具

    清空便签的所有内容。
    """

    name: str = "clear_note"
    """工具名称"""

    description: str = (
        "清空便签的所有内容（标题和正文）。"
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
            "客户端工具 'clear_note' 不应直接调用 execute()。"
        )
