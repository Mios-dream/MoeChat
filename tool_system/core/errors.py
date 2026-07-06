"""
工具系统异常体系定义模块

定义工具系统所有异常类，采用层次化设计：
- ToolSystemError: 基础异常
  ├── ToolNotFoundError: 工具未注册
  ├── ToolTimeoutError: 工具执行超时
  ├── ToolExecutionError: 工具执行内部异常
  ├── ClientDisconnectedError: 客户端断开连接
  ├── InvalidArgumentsError: 参数校验失败
  ├── SensitivityBlockedError: 用户取消敏感操作
  └── ToolRateLimitedError: 工具调用频率限制

所有异常携带 error_code 和 call_id，便于日志追踪和客户端处理。
"""

from __future__ import annotations


class ToolSystemError(Exception):
    """
    工具系统基础异常

    所有工具系统异常的基类，统一携带 error_code 和 call_id。
    上层代码可通过 error_code 判断异常类型并进行相应处理。

    Attributes:
        error_code: 结构化错误码，如 'TOOL_NOT_FOUND'
        call_id: 关联的工具调用 ID，用于日志追踪
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        call_id: str | None = None,
    ) -> None:
        """
        初始化基础异常

        Args:
            message: 人类可读的错误描述
            error_code: 结构化错误码
            call_id: 关联的工具调用 ID（可选）
        """
        self.error_code: str = error_code
        """结构化错误码，程序可通过此字段判断异常类型"""
        self.call_id: str | None = call_id
        """关联的工具调用 ID，用于全链路追踪"""
        super().__init__(message)


class ToolNotFoundError(ToolSystemError):
    """
    工具未注册异常

    当 LLM 尝试调用一个未在 ToolRegistry 中注册的工具时抛出。
    可能原因：工具名称拼写错误、工具未加载、插件目录配置错误。
    """

    def __init__(self, tool_name: str, call_id: str) -> None:
        """
        Args:
            tool_name: 未找到的工具名称
            call_id: 工具调用 ID
        """
        super().__init__(
            f"工具 '{tool_name}' 未在注册中心找到，请检查工具是否正确注册",
            error_code="TOOL_NOT_FOUND",
            call_id=call_id,
        )


class ToolTimeoutError(ToolSystemError):
    """
    工具执行超时异常

    工具执行时间超过 ToolMeta.timeout 设定值时抛出。
    """

    def __init__(self, tool_name: str, timeout: float, call_id: str) -> None:
        """
        Args:
            tool_name: 工具名称
            timeout: 超时时间（秒）
            call_id: 工具调用 ID
        """
        super().__init__(
            f"工具 '{tool_name}' 执行超时（{timeout}s 限制）",
            error_code="TOOL_TIMEOUT",
            call_id=call_id,
        )


class ToolExecutionError(ToolSystemError):
    """
    工具执行内部异常

    工具 execute() 方法中抛出的未捕获异常的封装。
    保留原始异常信息用于调试。
    """

    def __init__(
        self,
        tool_name: str,
        original_error: str,
        call_id: str,
    ) -> None:
        """
        Args:
            tool_name: 工具名称
            original_error: 原始异常信息的字符串表示
            call_id: 工具调用 ID
        """
        super().__init__(
            f"工具 '{tool_name}' 执行失败: {original_error}",
            error_code="TOOL_EXEC_ERROR",
            call_id=call_id,
        )


class ClientDisconnectedError(ToolSystemError):
    """
    客户端断开连接异常

    当正在等待客户端工具返回结果时客户端 WebSocket 连接断开。
    此时 Pending Call Table 中对应的 Future 将被取消。
    """

    def __init__(self, session_id: str, call_id: str) -> None:
        """
        Args:
            session_id: 断开的会话 ID
            call_id: 工具调用 ID
        """
        super().__init__(
            f"客户端 '{session_id}' 已断开连接，无法执行客户端工具",
            error_code="CLIENT_DISCONNECTED",
            call_id=call_id,
        )


class InvalidArgumentsError(ToolSystemError):
    """
    参数校验失败异常

    当 LLM 传入的参数不符合 ToolMeta.parameters 定义的 JSON Schema 时抛出。
    此异常通常触发 LLM 重试，让模型修正参数。
    """

    def __init__(self, tool_name: str, details: str, call_id: str) -> None:
        """
        Args:
            tool_name: 工具名称
            details: 参数校验失败的详细描述
            call_id: 工具调用 ID
        """
        super().__init__(
            f"工具 '{tool_name}' 参数无效: {details}",
            error_code="INVALID_ARGUMENTS",
            call_id=call_id,
        )


class SensitivityBlockedError(ToolSystemError):
    """
    用户取消敏感操作异常

    当工具标记为 SENSITIVE 或 DANGEROUS 等级，
    用户在确认对话框中拒绝执行时抛出。
    """

    def __init__(self, tool_name: str, call_id: str) -> None:
        """
        Args:
            tool_name: 工具名称
            call_id: 工具调用 ID
        """
        super().__init__(
            f"工具 '{tool_name}' 执行被用户取消",
            error_code="SENSITIVITY_BLOCKED",
            call_id=call_id,
        )


class ToolRateLimitedError(ToolSystemError):
    """
    工具调用频率限制异常

    当同一工具在短时间内被调用次数超过限制时抛出。
    防止 LLM 进入工具调用死循环或滥用工具。
    """

    def __init__(
        self,
        tool_name: str,
        call_id: str,
        retry_after: float = 5.0,
    ) -> None:
        """
        Args:
            tool_name: 工具名称
            call_id: 工具调用 ID
            retry_after: 建议冷却时间（秒）
        """
        super().__init__(
            f"工具 '{tool_name}' 调用过于频繁，请 {retry_after}s 后重试",
            error_code="RATE_LIMITED",
            call_id=call_id,
        )
