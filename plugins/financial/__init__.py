"""
Financial Plugin for MoeChat
财务记录插件包
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "智能财务记录插件，支持自然语言记账和多轮对话"

# 导入主要组件
from .plugin import FinancialPlugin
from .api_client import FinancialAPIClient
from .state_manager import SessionStateManager

# 便捷函数
from .plugin import (
    get_plugin,
    initialize_plugin,
    process_message,
    financial_plugin_hook,
)

# 包级别的公开接口
__all__ = [
    "FinancialPlugin",
    "FinancialAPIClient",
    "SessionStateManager",
    "get_plugin",
    "initialize_plugin",
    "process_message",
    "financial_plugin_hook",
]
