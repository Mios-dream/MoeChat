"""
工具管理器模块
负责工具的注册、发现、调度和执行
"""

import importlib
import json
from pathlib import Path
from my_utils import log as Log
from my_utils.tool_interface import BaseTool


class ToolManager:
    """
    工具注册与调度中心（单例模式）

    职责：
    - 管理所有已注册的工具实例
    - 提供 OpenAI tools 格式的工具定义
    - 执行工具调用并处理异常
    - 自动扫描 plugins/ 目录发现新技能
    """

    _tools: dict[str, BaseTool] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """
        注册工具实例

        Parameters:
            tool: 继承 BaseTool 的工具实例
        """
        if tool.name in cls._tools:
            Log.logger.warning(f"工具 '{tool.name}' 已存在，将被覆盖")
        cls._tools[tool.name] = tool
        Log.logger.info(f"已注册工具: {tool.name}")

    @classmethod
    def unregister(cls, tool_name: str) -> bool:
        """
        注销工具

        Parameters:
            tool_name: 工具名称

        Returns:
            bool: 是否成功注销
        """
        if tool_name in cls._tools:
            del cls._tools[tool_name]
            Log.logger.info(f"已注销工具: {tool_name}")
            return True
        return False

    @classmethod
    def get_openai_tools(cls) -> list[dict]:
        """
        获取所有已注册工具的 OpenAI 格式定义

        Returns:
            list[dict]: OpenAI tools 格式的工具定义列表
        """
        return [tool.to_openai_tool() for tool in cls._tools.values()]

    @classmethod
    def get_tool(cls, tool_name: str) -> BaseTool | None:
        """
        获取指定名称的工具实例

        Parameters:
            tool_name: 工具名称

        Returns:
            BaseTool | None: 工具实例，未找到返回 None
        """
        return cls._tools.get(tool_name)

    @classmethod
    def list_tools(cls) -> list[str]:
        """
        获取所有已注册工具的名称列表

        Returns:
            list[str]: 工具名称列表
        """
        return list(cls._tools.keys())

    @classmethod
    async def execute(cls, tool_name: str, arguments: dict) -> str:
        """
        执行指定工具

        Parameters:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            str: 工具执行结果的 JSON 字符串
        """
        tool = cls._tools.get(tool_name)
        if not tool:
            Log.logger.error(f"尝试调用未注册的工具: {tool_name}")
            return json.dumps(
                {"error": f"工具 '{tool_name}' 未注册"}, ensure_ascii=False
            )

        try:
            # 参数校验与默认值填充
            validated_args = tool.validate_arguments(arguments)
            Log.logger.info(f"执行工具: {tool_name}, 参数: {validated_args}")

            result = await tool.execute(**validated_args)

            Log.logger.info(f"工具 {tool_name} 执行完成")
            return result
        except Exception as e:
            Log.logger.error(f"工具 {tool_name} 执行失败: {e}", exc_info=True)
            return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)

    @classmethod
    def load_plugins(cls, plugins_dir: str = "plugins") -> int:
        """
        自动扫描 plugins/ 目录并加载技能

        扫描规则：
        - 查找 plugins_dir 下的子目录
        - 每个子目录必须包含 __init__.py
        - 在 __init__.py 中查找继承 BaseTool 的类并自动注册

        Parameters:
            plugins_dir: 插件目录路径

        Returns:
            int: 成功加载的插件数量
        """
        loaded_count = 0
        plugins_path = Path(plugins_dir)

        if not plugins_path.exists():
            Log.logger.warning(f"插件目录不存在: {plugins_dir}")
            return 0

        for plugin_dir in plugins_path.iterdir():
            if not plugin_dir.is_dir():
                continue
            if plugin_dir.name.startswith("_"):
                continue

            init_file = plugin_dir / "__init__.py"
            if not init_file.exists():
                continue

            try:
                # 动态导入插件模块
                module_name = f"plugins.{plugin_dir.name}"
                module = importlib.import_module(module_name)

                # 查找模块中继承 BaseTool 的类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseTool)
                        and attr is not BaseTool
                    ):
                        # 实例化并注册
                        tool_instance = attr()
                        cls.register(tool_instance)
                        loaded_count += 1

                # 如果模块定义了 register_tools 函数，调用它
                if hasattr(module, "register_tools"):
                    before_count = len(cls._tools)
                    module.register_tools(cls)
                    # 统计 register_tools 实际新增的工具数量
                    loaded_count += max(0, len(cls._tools) - before_count)
                    Log.logger.info(f"插件 {plugin_dir.name} 通过 register_tools 加载")

            except Exception as e:
                Log.logger.error(f"加载插件 {plugin_dir.name} 失败: {e}", exc_info=True)

        Log.logger.info(f"插件加载完成，共加载 {loaded_count} 个工具")
        return loaded_count

    @classmethod
    def clear(cls) -> None:
        """清空所有已注册的工具（主要用于测试）"""
        cls._tools.clear()
        Log.logger.info("已清空所有工具注册")

    @classmethod
    def is_initialized(cls) -> bool:
        """检查管理器是否已初始化"""
        return cls._initialized

    @classmethod
    def initialize(cls, plugins_dir: str = "plugins") -> None:
        """
        初始化工具管理器

        Parameters:
            plugins_dir: 插件目录路径
        """
        if cls._initialized:
            Log.logger.warning("ToolManager 已经初始化，跳过重复初始化")
            return

        cls.load_plugins(plugins_dir)
        cls._initialized = True
        Log.logger.info(f"ToolManager 初始化完成，已注册工具: {cls.list_tools()}")
