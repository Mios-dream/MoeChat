"""
LLM 请求系统模块

提供统一的 LLM 请求管理、提示词组合、响应解析和回调机制。

核心组件：
- PromptManager: 提示词管理器，支持模板组合
- ResponseParser: 可插拔的响应解析器
- CallbackManager: 回调管理器
- LLMClient: 统一的 LLM 客户端

使用示例：
```python
from core.llm import LLMClient, PromptManager, JsonLineParser

# 创建客户端
client = LLMClient()

# 组合提示词
prompt = PromptManager()
prompt.add_system("你是一个 Live2D 动作控制器")
prompt.add_user("请为以下文本生成动作：你好")

# 流式请求并解析
async for chunk in client.stream(prompt.messages, parser=JsonLineParser()):
    print(chunk)
```
"""

from core.llm.prompt_manager import PromptManager, PromptTemplate
from core.llm.response_parser import (
    ResponseParser,
    JsonLineParser,
    JsonParser,
    TextParser,
)
from core.llm.callback_manager import CallbackManager, CallbackEvent
from core.llm.llm_client import (
    LLMClient,
    LLMStreamChunk,
    ToolCallResult,
)

__all__ = [
    # 提示词管理
    "PromptManager",
    "PromptTemplate",
    # 响应解析
    "ResponseParser",
    "JsonLineParser",
    "JsonParser",
    "TextParser",
    # 回调管理
    "CallbackManager",
    "CallbackEvent",
    # LLM 客户端
    "LLMClient",
    "LLMStreamChunk",
    "ToolCallResult",
]
