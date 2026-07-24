"""
文件上传工具

客户端工具 + 服务端后处理示例：让用户选择文件并上传到服务端进行分析处理。

ExecDomain: CLIENT
ExecMode:   SYNC
Tags:       file, ui
Sensitivity: SENSITIVE

使用场景:
    - "帮我分析一下这个 PDF 文件"
    - "上传一个文本文件让助手看看"
    - "读取这个文档的内容"

执行流程:
    1. client_instruction(): 生成客户端指令 {action: "file_select", accept: [...]}
       → ClientSyncExecutor 合并到 WS arguments 中下发给客户端
    2. 客户端执行文件选择逻辑 → 返回 tool:result（含 file_path 等信息）
    3. server_postprocess(): 服务端接收客户端数据 → 读取文件 → 解析 → 返回 LLM

客户端实现指南:
    客户端需要处理 _client_instruction 字段中的指令:
    1. 接收 tool:call 消息
    2. 从 arguments._client_instruction 中获指令 {action: "file_select", ...}
    3. 执行文件选择操作
    4. 返回 tool:result 消息，包含选择结果
"""

from typing import Any
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
    timeout=120.0,
    sensitivity=ToolSensitivity.SENSITIVE,
    tags=["file", "ui"],
    version="1.0.0",
)
class FileUploadTool(ClientTool):
    """
    文件上传与解析工具

    执行流程:
    1. client_instruction(): 将 LLM 参数翻译为客户端文件选择器指令
    2. server_postprocess(): 客户端选择文件后，服务端解析最终返回给 LLM

    敏感度等级为 SENSITIVE，每次调用需要用户确认。
    """

    name: str = "upload_file"
    """工具名称：LLM 通过此名称调用"""

    description: str = (
        "让用户选择本地文件并上传到服务端进行内容分析和处理。"
        "当用户需要上传文档、图片、文本文件让助手分析时使用此工具。"
        "支持 document（文档类）和 image（图片类）两种文件类型。"
        "注意: 此操作需要用户确认后才能执行。"
    )
    """工具描述"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "file_type": {
                "type": "string",
                "enum": ["document", "image", "any"],
                "description": (
                    "期望的文件类型："
                    "'document' - 文档类 (.txt, .pdf, .docx, .md)；"
                    "'image' - 图片类 (.png, .jpg, .jpeg)；"
                    "'any' - 不限类型"
                ),
                "default": "any",
            },
            "purpose": {
                "type": "string",
                "description": (
                    "上传目的简述，用于在确认对话框中向用户说明上传原因。"
                    "如 '分析文档内容'、'提取关键信息' 等。"
                ),
                "default": "分析文件内容",
            },
        },
    }
    """
    JSON Schema 参数定义

    file_type: 期望的文件类型（document / image / any）
    purpose:   上传目的简述（用于确认对话框）
    """

    @property
    def meta(self) -> ToolMeta:
        """获取工具元信息（由 @register_tool 装饰器自动注入）"""
        return self._tool_meta  # type: ignore[attr-defined]

    async def client_instruction(self, **kwargs: Any) -> dict:
        """
        生成客户端文件选择器指令

        ClientSyncExecutor 在通过 WebSocket 下发调用前调用此方法，
        返回值合并到 arguments._client_instruction 字段发送给客户端。

        Args:
            file_type: 文件类型筛选（'document' / 'image' / 'any'）
            purpose: 上传目的说明

        Returns:
            发送给客户端的结构化指令字典:
            {
                "action": "file_select",
                "accept": [".txt", ".pdf", ...],
                "multiple": false,
                "purpose": "分析文件内容",
                "title": "请选择要上传的文件"
            }
        """
        file_type = str(kwargs.get("file_type", "any"))
        purpose = str(kwargs.get("purpose", "分析文件内容"))

        # ── 文件类型 → 接受的文件后缀映射 ──
        accept_map = {
            "document": [".txt", ".pdf", ".docx", ".doc", ".md", ".rtf"],
            "image": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"],
            "any": [
                ".txt",
                ".pdf",
                ".docx",
                ".doc",
                ".md",
                ".rtf",
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".gif",
                ".webp",
            ],
        }

        accepted_types = accept_map.get(file_type, accept_map["any"])

        # ── 构建客户端指令 ──
        return {
            "action": "file_select",
            "accept": accepted_types,
            "multiple": False,
            "purpose": purpose,
            "title": f"请选择要上传的文件（类型: {file_type}）",
            "file_type_filter": file_type,
        }

    async def server_postprocess(
        self,
        client_result: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """
        服务端后处理：解析客户端选择的文件

        客户端返回选择结果后，ClientSyncExecutor 调用此方法进行服务端处理。
        处理结果作为最终 tool 消息返回给 LLM。

        client_result 字典结构（从客户端 tool:result 的 content 解析得到）:
            {
                "file_path": str,      客户端本地文件路径
                "file_name": str,      文件名
                "file_size": int,      文件大小（字节）
                "file_type_hint": str, 文件类型提示
            }

        Args:
            client_result: 客户端返回的结构化数据
            file_type: 原始参数 - 文件类型
            purpose: 原始参数 - 上传目的

        Returns:
            JSON 格式字符串，包含文件解析结果的摘要
        """
        file_type = str(kwargs.get("file_type", "any"))
        purpose = str(kwargs.get("purpose", "分析文件内容"))
        file_path = client_result.get("file_path", "")
        file_name = client_result.get("file_name", "unknown")
        file_size = client_result.get("file_size", 0)

        if not file_path:
            return self.result_error("客户端未返回有效的文件路径", "INVALID_ARGUMENTS")

        # ── 解析文件内容 ──
        try:
            content_summary = await self._parse_file(file_path, file_name)
        except Exception as e:
            return self.result_error(f"文件解析失败: {e}", "TOOL_EXEC_ERROR")

        # ── 统计信息 ──
        char_count = len(content_summary)

        # ── 返回结构化的解析结果 ──
        return self.result_json(
            {
                "success": True,
                "file_name": file_name,
                "file_size": file_size,
                "file_size_human": self._format_size(file_size),
                "file_type": file_type,
                "char_count": char_count,
                "summary": (
                    content_summary[:1000] if char_count > 1000 else content_summary
                ),
                "truncated": char_count > 1000,
                "purpose": purpose,
            }
        )

    async def _parse_file(self, file_path: str, file_name: str) -> str:
        """
        根据文件类型解析文件内容

        当前为示例实现，返回模拟解析结果。
        实际部署时替换为真实的文件解析逻辑。

        Args:
            file_path: 文件路径
            file_name: 文件名

        Returns:
            解析出的文本内容
        """
        return (
            f"[文件内容解析示例]\n"
            f"文件: {file_name}\n"
            f"路径: {file_path}\n"
            f"---\n"
            f"这是一个示例文件的内容。实际部署时，这里会显示真实文件的解析结果。\n"
            f"支持的文件类型: TXT, PDF, DOCX, MD, RTF 等。\n"
            f"---\n"
            f"提示: 请部署实际的文件解析器以获取真实内容。"
        )

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        将字节数格式化为人类可读的大小

        Args:
            size_bytes: 文件大小（字节）

        Returns:
            格式化后的大小字符串，如 '1.5 MB'
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
