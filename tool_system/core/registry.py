"""
工具注册中心模块

ToolRegistry 是整个工具系统的元信息管理中心。负责：
1. 工具的注册与注销（支持手动注册和自动扫描发现）
2. 按 name / domain / tags 多级索引查询
3. 构建 OpenAI function calling 格式的工具集
4. 管理 register_tool 装饰器的全局注册表

设计为单例模式，确保全局唯一的工具注册视图。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable


from tool_system.core.enums import (
    ExecutionDomain,
    ExecutionMode,
    ToolSensitivity,
)
from tool_system.core.types import ToolMeta
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
)

# ═══════════════════════════════════════════════════════════════
# 装饰器内部状态（模块级变量，由 @register_tool 和 ToolRegistry 共用）
# ═══════════════════════════════════════════════════════════════

_registry: ToolRegistry | None = None
"""全局唯一的 ToolRegistry 实例，在首次调用 get_registry() 时懒初始化"""

# 待注册的工具类队列：用于在类定义时（ToolRegistry 可能尚未初始化）暂存注册信息
_pending_classes: list[tuple[type, ToolMeta]] = []


def get_registry() -> ToolRegistry:
    """
    获取全局 ToolRegistry 单例

    懒初始化：首次调用时创建实例，并将 _pending_classes 中的待注册类批量注册。

    Returns:
        全局唯一的 ToolRegistry 实例
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        # 批量注册在 Registry 初始化前由 @register_tool 收集的待注册类
        for tool_class, meta in _pending_classes:
            _registry._register_class(tool_class, meta)  # type: ignore[attr-defined]
        _pending_classes.clear()
    return _registry


# ═══════════════════════════════════════════════════════════════
# register_tool 装饰器
# ═══════════════════════════════════════════════════════════════


def register_tool(
    *,
    domain: ExecutionDomain,
    mode: ExecutionMode = ExecutionMode.SYNC,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    timeout: float = 30.0,
    max_retries: int = 0,
    sensitivity: ToolSensitivity = ToolSensitivity.NORMAL,
    tags: list[str] | None = None,
    placeholder: str = "",
    notify_on_complete: bool = True,
    version: str = "1.0.0",
    author: str = "",
    deprecation_message: str = "",
) -> Callable[[type], type]:
    """
    工具注册装饰器

    声明式定义工具元信息，将普通类标记为可被 LLM 调用的工具。
    被装饰的类必须继承 BaseTool / ServerTool / ClientTool / HybridTool 之一。

    装饰器内部自动完成:
    1. 从类的 name/description/parameters 属性合并元信息
    2. 将 meta 注入为被装饰类的类属性
    3. 向全局 ToolRegistry 注册

    Args:
        domain: 工具执行域（必填）
        mode: 工具执行模式，默认 SYNC
        name: 工具名称，默认从类属性 name 读取
        description: 工具描述，默认从类属性 description 读取
        parameters: JSON Schema 参数定义，默认从类属性 parameters 读取
        timeout: 超时时间（秒），默认 30.0
        max_retries: 最大重试次数，默认 0
        sensitivity: 敏感度等级，默认 NORMAL
        tags: 工具标签列表
        placeholder: 异步工具占位回复模板
        notify_on_complete: 异步工具完成后是否主动通知
        version: 工具版本号
        author: 工具作者
        deprecation_message: 废弃提示

    Returns:
        类装饰器函数

    Usage:
        @register_tool(
            domain=ExecutionDomain.SERVER,
            mode=ExecutionMode.SYNC,
            timeout=15.0,
            tags=["vision"],
        )
        class DesktopOcrTool(ServerTool):
            name = "desktop_ocr"
            description = "对当前屏幕进行 OCR 文字识别"
            parameters = {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "截屏区域"}
                }
            }

            async def execute(self, region: str = "full") -> str:
                ...
    """

    def decorator(cls: type) -> type:
        """
        类装饰器: 注入 ToolMeta 并注册到全局 Registry

        Args:
            cls: 被装饰的工具类

        Returns:
            装饰后的工具类（同一类，已附加 meta 属性）
        """
        # 收集元信息：装饰器参数 > 类属性默认值
        effective_name = name or getattr(cls, "name", cls.__name__)
        effective_description = description or getattr(cls, "description", "")
        effective_parameters = parameters or getattr(cls, "parameters", {})
        effective_tags = list(tags) if tags else []

        # 构建 ToolMeta 实例
        meta = ToolMeta(
            name=effective_name,
            description=effective_description,
            domain=domain,
            mode=mode,
            parameters=effective_parameters,
            timeout=timeout,
            max_retries=max_retries,
            sensitivity=sensitivity,
            tags=effective_tags,
            placeholder=placeholder,
            notify_on_complete=notify_on_complete,
            version=version,
            author=author,
            deprecation_message=deprecation_message,
            tool_class=cls,
        )

        # 注入 meta 为类属性（替代子类手动定义 meta property）
        # 工具实例通过 self.meta 访问此属性
        cls._tool_meta = meta  # type: ignore[attr-defined]

        # 向全局注册表注册
        registry = get_registry()
        registry._register_class(cls, meta)  # type: ignore[attr-defined]

        return cls

    return decorator


# ═══════════════════════════════════════════════════════════════
# ToolRegistry 注册中心
# ═══════════════════════════════════════════════════════════════


class ToolRegistry:
    """
    工具注册中心（单例）

    管理所有已注册工具的元信息，提供多维度索引查询。
    所有工具通过 @register_tool 装饰器自动注册到此中心。

    内部索引结构:
        _tools:       dict[name, ToolMeta]                主索引（按名称）
        _classes:     dict[name, type]                    类引用（按名称）
        _domain_index: dict[ExecutionDomain, set[name]]    域索引（按执行域）
        _tag_index:    dict[str, set[name]]                标签索引（按标签）

    使用方式:
        registry = get_registry()  # 获取全局单例
        tool = registry.get("desktop_ocr")  # 按名称查询
        tools = registry.get_by_domain(ExecutionDomain.SERVER)  # 按域查询
        tool_defs = registry.build_openai_tools(domains=[SERVER])  # 构建 LLM 工具集
    """

    def __init__(self) -> None:
        """初始化空的注册中心"""
        self._tools: dict[str, ToolMeta] = {}
        """主索引: 工具名称 → ToolMeta"""

        self._classes: dict[str, type] = {}
        """类引用索引: 工具名称 → 工具类"""

        self._domain_index: dict[ExecutionDomain, set[str]] = {
            ExecutionDomain.SERVER: set(),
            ExecutionDomain.CLIENT: set(),
            ExecutionDomain.HYBRID: set(),
        }
        """域索引: ExecutionDomain → 该域下的工具名称集合"""

        self._tag_index: dict[str, set[str]] = {}
        """标签索引: tag → 拥有该标签的工具名称集合"""

    def _register_class(self, cls: type, meta: ToolMeta) -> None:
        """
        内部注册方法: 将工具类和其元信息注册到所有索引中

        由 @register_tool 装饰器在类定义时自动调用。

        Args:
            cls: 工具类
            meta: 工具的 ToolMeta 实例
        """
        name = meta.name

        # 检查名称唯一性
        if name in self._tools:
            # 如果同一个类重复注册（热加载场景），视为更新
            existing = self._tools[name]
            if existing.tool_class is not None and existing.tool_class is not cls:
                existing_cls_name = (
                    existing.tool_class.__name__ if existing.tool_class else "unknown"
                )
                raise ValueError(
                    f"工具名称冲突: '{name}' 已被 " f"'{existing_cls_name}' 注册"
                )

        # 注册到主索引
        self._tools[name] = meta
        self._classes[name] = cls

        # 注册到域索引
        self._domain_index[meta.domain].add(name)

        # 注册到标签索引
        for tag in meta.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(name)

    def register(self, tool_instance: Any) -> None:
        """
        手动注册工具实例（用于兼容旧代码或动态创建工具的场景）

        从工具实例中提取 meta 并注册。

        Args:
            tool_instance: 已实例化的工具对象（必须具有 meta 属性）
        """
        meta = getattr(tool_instance, "meta", None)
        if not isinstance(meta, ToolMeta):
            raise TypeError(
                f"工具实例必须具有 ToolMeta 类型的 meta 属性，"
                f"得到: {type(meta).__name__}"
            )
        self._register_class(type(tool_instance), meta)

    def unregister(self, name: str) -> bool:
        """
        注销指定名称的工具

        从所有索引中移除该工具的记录。

        Args:
            name: 工具名称

        Returns:
            是否成功注销（工具不存在返回 False）
        """
        meta = self._tools.get(name)
        if meta is None:
            return False

        # 从主索引移除
        del self._tools[name]
        self._classes.pop(name, None)

        # 从域索引移除
        if meta.domain in self._domain_index:
            self._domain_index[meta.domain].discard(name)

        # 从标签索引移除
        for tag in meta.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(name)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

        return True

    def get(self, name: str) -> ToolMeta | None:
        """
        按名称获取工具元信息

        Args:
            name: 工具名称

        Returns:
            ToolMeta 实例，工具不存在返回 None
        """
        return self._tools.get(name)

    def get_class(self, name: str) -> type | None:
        """
        按名称获取工具类引用

        Args:
            name: 工具名称

        Returns:
            工具类，工具不存在返回 None
        """
        return self._classes.get(name)

    def get_by_domain(self, domain: ExecutionDomain) -> list[ToolMeta]:
        """
        获取指定执行域下的所有工具

        Args:
            domain: 执行域（SERVER / CLIENT / HYBRID）

        Returns:
            该域下所有工具的 ToolMeta 列表
        """
        names = self._domain_index.get(domain, set())
        return [self._tools[n] for n in names if n in self._tools]

    def get_by_tag(self, tag: str) -> list[ToolMeta]:
        """
        获取拥有指定标签的所有工具

        Args:
            tag: 标签名称

        Returns:
            拥有该标签的所有工具的 ToolMeta 列表
        """
        names = self._tag_index.get(tag, set())
        return [self._tools[n] for n in names if n in self._tools]

    def list_tools(
        self,
        domain: ExecutionDomain | None = None,
        tag: str | None = None,
    ) -> list[ToolMeta]:
        """
        按条件查询工具列表

        Args:
            domain: 按执行域筛选（None 表示不筛选）
            tag: 按标签筛选（None 表示不筛选）

        Returns:
            满足所有筛选条件的工具列表
        """
        result = list(self._tools.values())

        if domain is not None:
            result = [m for m in result if m.domain == domain]

        if tag is not None:
            result = [m for m in result if tag in m.tags]

        return result

    def list_names(
        self,
        domain: ExecutionDomain | None = None,
        tag: str | None = None,
    ) -> list[str]:
        """
        按条件查询工具名称列表

        Args:
            domain: 按执行域筛选
            tag: 按标签筛选

        Returns:
            满足筛选条件的工具名称列表
        """
        return [m.name for m in self.list_tools(domain=domain, tag=tag)]

    def build_openai_tools(
        self,
        domains: list[ExecutionDomain] | None = None,
        tags: list[str] | None = None,
        exclude_deprecated: bool = True,
    ) -> list[ChatCompletionFunctionToolParam]:
        """
        构建 OpenAI function calling 格式的工具定义列表

        用于构建 LLM 请求中的 tools 参数。

        Args:
            domains: 限定执行域列表（None 表示包含所有域）
            tags: 限定标签列表（None 表示包含所有标签）
            exclude_deprecated: 是否排除已废弃的工具

        Returns:
            OpenAI tools 格式的工具定义列表:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "tool description",
                        "parameters": { ... }
                    }
                },
                ...
            ]
        """
        tools = self.list_tools()

        # 按域筛选
        if domains is not None:
            domain_set = set(domains)
            tools = [m for m in tools if m.domain in domain_set]

        # 按标签筛选
        if tags is not None:
            tag_set = set(tags)
            tools = [m for m in tools if tag_set & set(m.tags)]

        # 排除废弃工具
        if exclude_deprecated:
            tools = [m for m in tools if not m.deprecation_message]

        result: list[ChatCompletionFunctionToolParam] = []
        for meta in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": meta.name,
                        "description": meta.description,
                        "parameters": meta.parameters,
                    },
                }
            )
        return result

    def discover(self, *scan_paths: str) -> int:
        """
        自动扫描目录并加载工具插件

        扫描逻辑:
        1. 遍历 scan_paths 下所有 .py 文件
        2. 动态导入模块
        3. 导入过程触发模块内 @register_tool 装饰器
        4. 返回新发现的工具数量

        注意: 此方法导入的模块在工具系统生命周期内不会卸载。
        如需热加载，使用 reload_tool() 方法。

        Args:
            *scan_paths: 要扫描的目录路径（可多个）

        Returns:
            新注册的工具数量
        """
        count_before = len(self._tools)

        for scan_path in scan_paths:
            path = Path(scan_path)
            if not path.exists():
                continue
            if not path.is_dir():
                continue

            # 将扫描路径的父目录加入 sys.path（确保 import 能找到）
            # 使用 parent 而非 path 自身：module_name 是按 path.parent 作为根计算的，
            # 例如模块名为 server_plugins.vision.ocr_tool 时，需 plugins/ 在 sys.path 中
            path_str = str(path.parent.absolute())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

            # 遍历模块
            for py_file in path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                # 构建模块名
                rel_path = py_file.relative_to(path.parent)
                module_name = (
                    str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
                )

                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    # 单个模块导入失败不影响其他模块
                    import logging

                    logging.getLogger("ToolRegistry").warning(
                        f"加载工具模块 '{module_name}' 失败: {e}"
                    )

        return len(self._tools) - count_before

    def get_domain(self, name: str) -> ExecutionDomain | None:
        """
        获取指定工具的 ExecutionDomain

        Args:
            name: 工具名称

        Returns:
            执行域枚举值，工具不存在返回 None
        """
        meta = self._tools.get(name)
        return meta.domain if meta else None

    def get_mode(self, name: str) -> ExecutionMode | None:
        """
        获取指定工具的 ExecutionMode

        Args:
            name: 工具名称

        Returns:
            执行模式枚举值，工具不存在返回 None
        """
        meta = self._tools.get(name)
        return meta.mode if meta else None

    def clear(self) -> None:
        """
        清空所有已注册的工具

        主要用于测试环境的清理。
        """
        self._tools.clear()
        self._classes.clear()
        self._domain_index = {
            ExecutionDomain.SERVER: set(),
            ExecutionDomain.CLIENT: set(),
            ExecutionDomain.HYBRID: set(),
        }
        self._tag_index.clear()

    def __len__(self) -> int:
        """已注册的工具数量"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """检查工具是否已注册（支持 'name' in registry 语法）"""
        return name in self._tools

    def __repr__(self) -> str:
        """可读的注册中心状态"""
        return (
            f"<ToolRegistry: {len(self._tools)} tools "
            f"(server={len(self._domain_index[ExecutionDomain.SERVER])}, "
            f"client={len(self._domain_index[ExecutionDomain.CLIENT])}, "
            f"hybrid={len(self._domain_index[ExecutionDomain.HYBRID])})>"
        )
