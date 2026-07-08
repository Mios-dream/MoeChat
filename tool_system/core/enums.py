"""
工具系统枚举定义模块

定义工具系统的所有枚举类型，包括：
- ExecutionDomain: 工具执行域（服务端/客户端）
- ExecutionMode: 工具执行模式（同步/异步）
- ToolSensitivity: 工具敏感度等级
"""

from __future__ import annotations

from enum import Enum


class ExecutionDomain(str, Enum):
    """
    工具执行域 - 定义工具代码的物理执行位置

    Attributes:
        SERVER: 工具逻辑在服务端进程内执行
        CLIENT: 工具逻辑在客户端设备上执行（支持可选的 server_postprocess 服务端后处理）
    """

    SERVER = "server"
    """服务端执行：工具代码在 Python 服务端进程中运行"""

    CLIENT = "client"
    """客户端执行：工具代码通过 WebSocket 下发给客户端执行，支持可选 server_postprocess 回执处理"""


class ExecutionMode(str, Enum):
    """
    工具执行模式 - 定义 LLM 如何等待工具执行结果

    Attributes:
        SYNC: 同步模式，LLM 阻塞等待工具完成后继续生成
        ASYNC: 异步模式，框架返回占位回复后 LLM 立即继续生成，
               工具执行结果通过事件系统异步回调注入
    """

    SYNC = "sync"
    """
    同步执行：
    - LLM 生成 tool_call → 阻塞等待 → 收到 result → 继续生成回复
    - 适用于结果必须直接影响回复内容的场景（OCR、知识检索等）
    """

    ASYNC = "async"
    """
    异步执行：
    - LLM 生成 tool_call → 立即获得占位回复 → LLM 继续生成回复
    - 工具在后台执行，完成后通过 ResultNotifier 注入上下文
    - 适用于耗时长或不阻塞对话的场景（下载、上传等）
    """


class ToolSensitivity(str, Enum):
    """
    工具敏感度等级 - 定义工具是否需要用户确认

    Attributes:
        SAFE: 安全工具，自动执行无需确认
        NORMAL: 普通工具，根据用户设置决定是否确认
        SENSITIVE: 敏感工具，每次调用需用户确认
        DANGEROUS: 危险工具，需用户确认 + 二次验证
    """

    SAFE = "safe"
    """安全：自动执行，无需任何确认（如 OCR 识别、知识搜索）"""

    NORMAL = "normal"
    """普通：根据用户全局隐私设置决定是否需要确认（如天气设置、播放音乐）"""

    SENSITIVE = "sensitive"
    """敏感：每次调用前弹出确认对话框让用户确认（如文件上传、截屏分享）"""

    DANGEROUS = "dangerous"
    """危险：需要用户确认 + 二次身份验证（如删除文件、执行系统命令）"""
