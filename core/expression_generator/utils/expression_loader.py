"""
Live2D 表情加载器

模块职责：
1. 扫描模型目录，直接查找 .exp3.json 表情文件

设计原则：
- 表情文件通过目录递归扫描获取
"""

from pathlib import Path
from dataclasses import dataclass, field
from core.expression_generator.utils.path import resolve_assets_root
from my_utils.log import logger as Log
import json


@dataclass
class ExpressionInfo:
    """
    表情信息数据类

    存储单个表情的完整信息，包括名称、文件路径、参数和描述。

    属性：
    - name: 表情名称（如 "happy", "开心"）
    - file_path: .exp3.json 文件的绝对路径
    - parameters: 表情包含的参数 {ParamId: value}
    - description: LLM 生成的中文描述（如 "开心微笑，眼睛弯起"）
    """

    name: str
    file_path: str
    parameters: dict[str, float] = field(default_factory=dict)
    description: str = ""


def _scan_expression_files_from_directory(assets_root: Path) -> list[dict]:
    """
    直接扫描目录查找所有 .exp3.json 表情文件

    扫描策略：
    - 递归搜索 assets_root 下所有 *.exp3.json 文件
    - 从文件名提取表情名称（去掉 .exp3.json 后缀）
    - 返回相对于 assets_root 的文件路径

    参数：
    - assets_root: Live2D 资源根目录

    返回：
    - 表情文件信息列表 [{"name": "happy", "file": "expressions/happy.exp3.json"}, ...]
    - 未找到文件返回空列表
    """
    expression_files = list(assets_root.rglob("*.exp3.json"))

    if not expression_files:
        Log.info("[表情加载] 未找到任何 .exp3.json 表情文件")
        return []

    expressions = []
    for file_path in expression_files:
        # 从文件名提取表情名称
        name = file_path.stem.replace(".exp3", "")
        if not name:
            name = file_path.stem

        # 获取相对于 assets_root 的路径
        try:
            rel_path = file_path.relative_to(assets_root)
        except ValueError:
            rel_path = file_path

        expressions.append(
            {
                "name": name,
                "file": str(rel_path).replace("\\", "/"),
            }
        )

    Log.info(f"[表情加载] 扫描到 {len(expressions)} 个表情文件")
    return expressions


def _parse_expression_file(file_path: Path) -> dict[str, float]:
    """
    解析 .exp3.json 文件，提取表情参数

    exp3.json 文件格式：
    {
      "Type": "Additive",
      "FadeInTime": 0.5,
      "FadeOutTime": 0.5,
      "Parameters": [
        {"Id": "ParamEyeLSmile", "Value": 1.0},
        {"Id": "ParamMouthForm", "Value": 0.5}
      ]
    }

    参数：
    - file_path: .exp3.json 文件路径

    返回：
    - 参数字典 {ParamId: value}
    - 解析失败返回空字典
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            expr_data = json.load(f)
    except Exception as e:
        Log.warning(f"[表情加载] 读取表情文件失败 {file_path}: {e}")
        return {}

    # 提取参数列表
    parameters_raw = expr_data.get("Parameters", [])
    if not parameters_raw:
        return {}

    # 构建参数字典
    parameters = {}
    for param in parameters_raw:
        param_id = param.get("Id")
        value = param.get("Value")
        if param_id is not None and value is not None:
            try:
                parameters[param_id] = float(value)
            except (TypeError, ValueError):
                continue

    return parameters


def load_expressions(
    assistant_name: str,
) -> list[ExpressionInfo]:
    """
    加载角色的全部表情信息

    执行流程：
    1. 定位 Live2D 资源目录
    2. 扫描目录查找所有 .exp3.json 表情文件
    3. 解析每个 .exp3.json 文件获取参数
    4. 尝试从缓存加载表情描述
    5. 缓存未命中时调用 LLM 生成描述
    6. 保存缓存并返回完整表情信息

    参数：
    - assistant_name: 角色名称
    - use_cache: 是否使用缓存（默认 True）

    返回：
    - ExpressionInfo 列表，包含所有表情的完整信息
    - 加载失败返回空列表
    """
    # Step 1: 定位 Live2D 资源目录
    assets_root = resolve_assets_root(assistant_name)
    if not assets_root:
        return []

    # Step 2: 扫描目录查找表情文件
    expression_files = _scan_expression_files_from_directory(assets_root)
    if not expression_files:
        return []

    # Step 3: 解析每个表情文件的参数
    expressions_with_params: list[tuple[str, dict[str, float], str]] = []
    for expr_info in expression_files:
        name = expr_info["name"]
        file_rel = expr_info["file"]
        file_abs = assets_root / file_rel

        if not file_abs.exists():
            Log.warning(f"[表情加载] 表情文件不存在: {file_abs}")
            continue

        params = _parse_expression_file(file_abs)
        if params:
            expressions_with_params.append((name, params, str(file_abs)))

    if not expressions_with_params:
        Log.warning("[表情加载] 未解析到有效表情参数")
        return []

    Log.info(f"[表情加载] 成功解析 {len(expressions_with_params)} 个表情文件")

    # Step 4: 尝试从缓存加载表情描述
    cached_descriptions: dict[str, str] | None = None

    # Step 5: 缓存未命中的表情需要调用 LLM 生成描述
    need_llm = []
    descriptions = cached_descriptions or {}

    for name, params, _ in expressions_with_params:
        if name not in descriptions:
            need_llm.append((name, params))
    # 构建最终的表情信息列表
    result = []
    for name, params, file_path in expressions_with_params:
        expr_info = ExpressionInfo(
            name=name,
            file_path=file_path,
            parameters=params,
            description=descriptions.get(name, name),  # 无描述时使用名称
        )
        result.append(expr_info)

    Log.info(f"[表情加载] 完成: 共 {len(result)} 个表情")
    return result
