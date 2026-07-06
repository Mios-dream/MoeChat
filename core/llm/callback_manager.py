"""
回调管理器

支持在 LLM 请求的不同阶段注册回调函数。

回调事件：
- on_start: 请求开始前
- on_token: 每收到一个 token
- on_chunk: 解析出一个完整数据块
- on_tool_calls: 流式响应中累积到工具调用
- on_complete: 请求完成
- on_error: 请求出错

使用示例：
```python
cm = CallbackManager()

@cm.on("on_token")
def handle_token(token: str):
    print(f"收到 token: {token}")

@cm.on("on_chunk")
def handle_chunk(chunk: dict):
    print(f"收到 chunk: {chunk}")

# 触发回调
cm.emit("on_token", "你好")
```
"""

from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, Callable
import asyncio

from my_utils.log import logger as Log


class CallbackEvent(str, Enum):
    """
    回调事件类型

    枚举值：
    - START: 请求开始前
    - TOKEN: 每收到一个 token
    - CHUNK: 解析出一个完整数据块
    - TOOL_CALLS: 流式响应中累积到工具调用
    - COMPLETE: 请求完成
    - ERROR: 请求出错
    """

    START = "on_start"
    TOKEN = "on_token"
    CHUNK = "on_chunk"
    TOOL_CALLS = "on_tool_calls"
    COMPLETE = "on_complete"
    ERROR = "on_error"


@dataclass
class Callback:
    """
    回调函数包装

    属性：
    - func: 回调函数
    - event: 事件类型
    - priority: 优先级（数字越小优先级越高）
    - is_async: 是否为异步函数
    """

    func: Callable
    event: CallbackEvent
    priority: int = 100
    is_async: bool = False


class CallbackManager:
    """
    回调管理器

    支持注册、移除和触发回调函数。

    特性：
    - 支持同步和异步回调
    - 支持优先级排序
    - 支持装饰器语法注册
    - 错误隔离（单个回调失败不影响其他回调）

    使用示例：
    ```python
    cm = CallbackManager()

    # 装饰器方式注册
    @cm.on(CallbackEvent.TOKEN)
    def on_token(token: str):
        print(token)

    # 函数方式注册
    cm.register(CallbackEvent.COMPLETE, lambda result: print("完成"))

    # 带优先级注册
    cm.register(CallbackEvent.TOKEN, high_priority_handler, priority=10)

    # 触发回调
    await cm.emit(CallbackEvent.TOKEN, "你好")
    ```
    """

    def __init__(self):
        """初始化回调管理器"""
        # 事件 -> 回调列表的映射
        self._callbacks: dict[CallbackEvent, list[Callback]] = {
            event: [] for event in CallbackEvent
        }

    def on(
        self,
        event: CallbackEvent,
        priority: int = 100,
    ) -> Callable:
        """
        装饰器：注册回调函数

        参数：
        - event: 事件类型
        - priority: 优先级（数字越小优先级越高）

        返回：
        - 装饰器函数

        使用示例：
        ```python
        @cm.on(CallbackEvent.TOKEN)
        def handle_token(token: str):
            print(token)
        ```
        """

        def decorator(func: Callable) -> Callable:
            self.register(event, func, priority)
            return func

        return decorator

    def register(
        self,
        event: CallbackEvent,
        func: Callable,
        priority: int = 100,
    ) -> None:
        """
        注册回调函数

        参数：
        - event: 事件类型
        - func: 回调函数
        - priority: 优先级（数字越小优先级越高）
        """
        is_async = inspect.iscoroutinefunction(func)
        callback = Callback(
            func=func,
            event=event,
            priority=priority,
            is_async=is_async,
        )

        self._callbacks[event].append(callback)
        # 按优先级排序
        self._callbacks[event].sort(key=lambda c: c.priority)

    def unregister(self, event: CallbackEvent, func: Callable) -> bool:
        """
        移除回调函数

        参数：
        - event: 事件类型
        - func: 要移除的回调函数

        返回：
        - 是否成功移除
        """
        callbacks = self._callbacks[event]
        for i, cb in enumerate(callbacks):
            if cb.func is func:
                callbacks.pop(i)
                return True
        return False

    def clear(self, event: CallbackEvent | None = None) -> None:
        """
        清空回调

        参数：
        - event: 指定事件类型，None 表示清空所有
        """
        if event:
            self._callbacks[event].clear()
        else:
            for event_type in CallbackEvent:
                self._callbacks[event_type].clear()

    async def emit(self, event: CallbackEvent, *args: Any, **kwargs: Any) -> None:
        """
        触发回调

        按优先级顺序执行所有注册的回调函数。
        单个回调失败不会影响其他回调的执行。

        参数：
        - event: 事件类型
        - *args: 位置参数
        - **kwargs: 关键字参数
        """
        callbacks = self._callbacks.get(event, [])

        for callback in callbacks:
            try:
                if callback.is_async:
                    await callback.func(*args, **kwargs)
                else:
                    callback.func(*args, **kwargs)
            except Exception as e:
                Log.error(
                    f"[回调管理器] 回调执行失败 "
                    f"(event={event.value}, func={callback.func.__name__}): {e}"
                )

    def get_callback_count(self, event: CallbackEvent) -> int:
        """
        获取指定事件的回调数量

        参数：
        - event: 事件类型

        返回：
        - 回调数量
        """
        return len(self._callbacks.get(event, []))
