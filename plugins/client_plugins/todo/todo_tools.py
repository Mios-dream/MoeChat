"""
待办事项工具集

客户端同步工具：管理澪的任务板（Todo 看板）。

包含工具:
- add_todo:              添加待办事项
- get_todos:             获取所有待办事项
- toggle_todo:           切换完成状态
- delete_todo:           删除指定项
- clear_completed_todos: 清除已完成项

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       ui, todo
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
    tags=["ui", "todo"],
    version="1.0.0",
)
class AddTodoTool(ClientTool):
    """
    添加待办事项工具

    在任务板中添加一条新的待办事项。
    """

    name: str = "add_todo"
    """工具名称"""

    description: str = (
        "在澪的任务板中添加一条新的待办事项，例如: 记得喝水、完成周报。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "待办事项的文本内容，简洁明了即可",
            },
        },
        "required": ["text"],
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'add_todo' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.SAFE,
    tags=["ui", "todo"],
    version="1.0.0",
)
class GetTodosTool(ClientTool):
    """
    获取待办事项列表工具

    获取任务板中所有待办事项。
    """

    name: str = "get_todos"
    """工具名称"""

    description: str = (
        "获取澪的任务板中所有待办事项列表，包含每条任务的ID、文本和完成状态。"
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
            "客户端工具 'get_todos' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.NORMAL,
    tags=["ui", "todo"],
    version="1.0.0",
)
class ToggleTodoTool(ClientTool):
    """
    切换待办事项完成状态工具

    将指定待办事项标记为已完成或未完成。
    """

    name: str = "toggle_todo"
    """工具名称"""

    description: str = (
        "切换某条待办事项的完成状态（已完成 <-> 未完成）。需要传入任务ID。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "待办事项的唯一标识ID，可通过 get_todos 获取",
            },
        },
        "required": ["id"],
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'toggle_todo' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.SENSITIVE,
    tags=["ui", "todo"],
    version="1.0.0",
)
class DeleteTodoTool(ClientTool):
    """
    删除待办事项工具

    删除指定 ID 的待办事项。
    """

    name: str = "delete_todo"
    """工具名称"""

    description: str = (
        "删除指定ID的待办事项。需要传入任务ID。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "待办事项的唯一标识ID，可通过 get_todos 获取",
            },
        },
        "required": ["id"],
    }
    """参数定义"""

    @property
    def meta(self) -> ToolMeta:
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: object) -> str:
        raise NotImplementedError(
            "客户端工具 'delete_todo' 不应直接调用 execute()。"
        )


@register_tool(
    domain=ExecutionDomain.CLIENT,
    mode=ExecutionMode.SYNC,
    timeout=10.0,
    sensitivity=ToolSensitivity.SENSITIVE,
    tags=["ui", "todo"],
    version="1.0.0",
)
class ClearCompletedTodosTool(ClientTool):
    """
    清除已完成待办事项工具

    一键清除所有已完成的待办事项。
    """

    name: str = "clear_completed_todos"
    """工具名称"""

    description: str = (
        "一键清除所有已完成的待办事项，清扫任务板。"
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
            "客户端工具 'clear_completed_todos' 不应直接调用 execute()。"
        )
