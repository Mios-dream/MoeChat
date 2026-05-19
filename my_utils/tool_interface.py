"""
工具接口定义模块
定义所有技能/工具的抽象基类和数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """
    工具参数定义

    Attributes:
        name: 参数名称
        type: 参数类型 ("string", "integer", "boolean", "array", "object")
        description: 参数描述，告诉 LLM 此参数的含义
        required: 是否为必填参数
        enum: 可选值列表（仅适用于 string 类型）
        default: 默认值
    """

    name: str
    type: str
    description: str
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


class BaseTool(ABC):
    """
    工具基类 - 所有技能必须继承此类

    子类需要实现以下属性和方法：
    - name: 工具名称（LLM 调用时的标识符）
    - description: 工具描述（告诉 LLM 何时/如何使用）
    - parameters: JSON Schema 格式的参数定义
    - execute(): 执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，LLM 调用时使用的标识符，如 'desktop_ocr'"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，告诉 LLM 何时以及如何使用此工具"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        JSON Schema 格式的参数定义

        示例:
        {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "截屏区域",
                    "default": "full"
                }
            },
            "required": []
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        执行工具逻辑

        Parameters:
            **kwargs: 由 LLM 传入的参数，与 parameters 定义对应

        Returns:
            str: 执行结果的 JSON 字符串，将作为 tool 角色消息返回给 LLM
        """
        ...

    def to_openai_tool(self) -> dict:
        """
        转换为 OpenAI tools 格式

        Returns:
            dict: 符合 OpenAI function calling 规范的工具定义
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: dict) -> dict:
        """
        校验并填充参数默认值

        Parameters:
            arguments: LLM 传入的原始参数

        Returns:
            dict: 校验后的参数（已填充默认值）
        """
        validated = {}
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        for param_name, param_schema in props.items():
            if param_name in arguments:
                validated[param_name] = arguments[param_name]
            elif "default" in param_schema:
                validated[param_name] = param_schema["default"]
            elif param_name in required:
                raise ValueError(f"缺少必填参数: {param_name}")

        return validated
