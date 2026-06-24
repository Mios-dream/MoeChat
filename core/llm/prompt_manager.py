"""
提示词管理器

支持提示词模板的组合、复用和动态构建。

核心功能：
1. 模板定义与注册
2. 动态变量替换
3. 多段提示词组合
4. 上下文消息管理

设计原则：
- 模块化：每个提示词片段独立定义
- 可组合：支持自由组合不同的提示词片段
- 可复用：模板支持参数化
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptTemplate:
    """
    提示词模板

    用于定义可复用的提示词片段，支持变量替换。

    属性：
    - name: 模板名称（唯一标识）
    - template: 模板内容，使用 {variable} 作为占位符
    - description: 模板描述（可选）
    - required_vars: 必需的变量列表
    """

    name: str
    template: str
    description: str = ""
    required_vars: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """
        渲染模板，替换变量

        参数：
        - **kwargs: 模板变量

        返回：
        - 渲染后的字符串

        异常：
        - ValueError: 缺少必需变量时抛出
        """
        # 检查必需变量
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"模板 '{self.name}' 缺少必需变量: {missing}")

        # 替换变量
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result


class PromptManager:
    """
    提示词管理器

    支持动态组合多段提示词，构建完整的消息列表。

    使用示例：
    ```python
    pm = PromptManager()

    # 注册模板
    pm.register_template(PromptTemplate(
        name="role",
        template="你是一个 Live2D 虚拟形象的动作控制器。",
        description="角色定义"
    ))

    # 组合提示词
    pm.add_system("你是一个助手")
    pm.add_user("你好")
    pm.add_template("role")

    # 获取消息列表
    messages = pm.messages
    ```
    """

    def __init__(self):
        """初始化提示词管理器"""
        # 消息列表：[{"role": "system/user/assistant", "content": "..."}]
        self._messages: list[dict[str, str]] = []
        # 已注册的模板
        self._templates: dict[str, PromptTemplate] = {}
        # 片段缓存（用于组合）
        self._system_parts: list[str] = []
        self._context_parts: list[str] = []

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        获取完整的消息列表

        返回：
        - OpenAI 格式的消息列表
        """
        return self._messages.copy()

    @property
    def system_prompt(self) -> str:
        """
        获取系统提示词（所有 system 消息的组合）

        返回：
        - 组合后的系统提示词
        """
        system_messages = [
            msg["content"] for msg in self._messages if msg["role"] == "system"
        ]
        return "\n\n".join(system_messages)

    def register_template(self, template: PromptTemplate) -> None:
        """
        注册提示词模板

        参数：
        - template: 提示词模板实例

        异常：
        - ValueError: 模板名称已存在时抛出
        """
        if template.name in self._templates:
            raise ValueError(f"模板 '{template.name}' 已存在")
        self._templates[template.name] = template

    def get_template(self, name: str) -> PromptTemplate | None:
        """
        获取已注册的模板

        参数：
        - name: 模板名称

        返回：
        - 模板实例，不存在返回 None
        """
        return self._templates.get(name)

    def clear(self) -> None:
        """清空所有消息和上下文"""
        self._messages.clear()
        self._system_parts.clear()
        self._context_parts.clear()

    def add_system(self, content: str, append: bool = False) -> "PromptManager":
        """
        添加系统消息

        参数：
        - content: 系统提示词内容
        - append: 是否追加到现有系统消息（用换行分隔）

        返回：
        - self，支持链式调用
        """
        if append and self._system_parts:
            self._system_parts.append(content)
            # 更新最后一条系统消息
            for i in range(len(self._messages) - 1, -1, -1):
                if self._messages[i]["role"] == "system":
                    self._messages[i]["content"] = "\n\n".join(self._system_parts)
                    break
        else:
            self._system_parts = [content]
            self._messages.append({"role": "system", "content": content})
        return self

    def add_user(self, content: str) -> "PromptManager":
        """
        添加用户消息

        参数：
        - content: 用户消息内容

        返回：
        - self，支持链式调用
        """
        self._messages.append({"role": "user", "content": content})
        return self

    def add_assistant(self, content: str) -> "PromptManager":
        """
        添加助手消息（用于 few-shot 示例）

        参数：
        - content: 助手消息内容

        返回：
        - self，支持链式调用
        """
        self._messages.append({"role": "assistant", "content": content})
        return self

    def add_template(self, name: str, **kwargs: Any) -> "PromptManager":
        """
        添加已注册模板作为系统消息

        参数：
        - name: 模板名称
        - **kwargs: 模板变量

        返回：
        - self，支持链式调用

        异常：
        - ValueError: 模板不存在时抛出
        """
        template = self._templates.get(name)
        if not template:
            raise ValueError(f"模板 '{name}' 不存在")

        rendered = template.render(**kwargs)
        return self.add_system(rendered, append=True)

    def add_context(self, content: str) -> "PromptManager":
        """
        添加上下文信息（作为用户消息）

        用于添加对话场景、前一个动作参数等变化的上下文信息。

        参数：
        - content: 上下文内容

        返回：
        - self，支持链式调用
        """
        self._context_parts.append(content)
        return self

    def build_context_message(self) -> "PromptManager":
        """
        将累积的上下文信息构建为用户消息

        返回：
        - self，支持链式调用
        """
        if self._context_parts:
            context = "\n\n".join(self._context_parts)
            self._messages.append({"role": "user", "content": context})
            self._context_parts.clear()
        return self

    def add_few_shot(self, examples: list[dict[str, str]]) -> "PromptManager":
        """
        添加 few-shot 示例

        参数：
        - examples: 示例列表，每项包含 role 和 content

        返回：
        - self，支持链式调用

        示例：
        ```python
        pm.add_few_shot([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": '{"t": "你好~", "a": ["smile"]}'},
        ])
        ```
        """
        for example in examples:
            role = example.get("role", "user")
            content = example.get("content", "")
            self._messages.append({"role": role, "content": content})
        return self

    def clone(self) -> "PromptManager":
        """
        克隆当前管理器（深拷贝）

        返回：
        - 新的 PromptManager 实例
        """
        new_manager = PromptManager()
        new_manager._messages = self._messages.copy()
        new_manager._templates = self._templates.copy()
        new_manager._system_parts = self._system_parts.copy()
        new_manager._context_parts = self._context_parts.copy()
        return new_manager
