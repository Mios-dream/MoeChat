"""
Live2D 表情加载器

模块职责：
1. 扫描模型目录，直接查找 .exp3.json 表情文件
2. 解析每个 .exp3.json 文件获取表情参数
3. 调用 LLM 为表情生成中文描述
4. 缓存表情描述信息，避免重复调用 LLM

设计原则：
- 表情文件通过目录递归扫描获取
- 表情描述通过 LLM 推断生成
- 缓存机制减少 LLM 调用开销
"""

from pathlib import Path
from dataclasses import dataclass, field
from my_utils.log import logger as Log
from core.llm.llm_client import LLMClient
from core.llm.response_parser import parse_llm_json_response
import json
import time

# 模块级 LLM 客户端实例
_llm_client = LLMClient(model_key="LLM")


# ============================================================
# 数据结构定义
# ============================================================

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


# ============================================================
# 缓存相关常量
# ============================================================

# 缓存文件名
CACHE_FILE_NAME = "expression_descriptions_cache.json"


def _get_cache_path(model_dir: Path) -> Path:
    """
    获取表情描述缓存文件路径

    参数：
    - model_dir: 模型目录路径

    返回：
    - 缓存文件的完整路径
    """
    return model_dir / CACHE_FILE_NAME


def _load_cache(model_dir: Path) -> dict[str, str] | None:
    """
    加载表情描述缓存

    参数：
    - model_dir: 模型目录路径

    返回：
    - 缓存成功返回 {表情名: 描述} 字典
    - 缓存不存在或读取失败返回 None
    """
    cache_path = _get_cache_path(model_dir)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # 验证缓存格式
        if not isinstance(cache_data, dict):
            return None

        descriptions = cache_data.get("descriptions", {})
        if not isinstance(descriptions, dict):
            return None

        Log.info(f"[表情加载] 命中缓存，共 {len(descriptions)} 个表情描述")
        return descriptions

    except Exception as e:
        Log.warning(f"[表情加载] 读取缓存失败: {e}")
        return None


def _save_cache(model_dir: Path, descriptions: dict[str, str]) -> None:
    """
    保存表情描述缓存

    参数：
    - model_dir: 模型目录路径
    - descriptions: {表情名: 描述} 字典
    """
    cache_path = _get_cache_path(model_dir)
    try:
        cache_data = {
            "version": 1,
            "generated_at": int(time.time()),
            "description_count": len(descriptions),
            "descriptions": descriptions,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        Log.info(f"[表情加载] 缓存已保存: {len(descriptions)} 个描述")
    except Exception as e:
        Log.warning(f"[表情加载] 保存缓存失败: {e}")


# ============================================================
# 表情文件扫描
# ============================================================

def _resolve_assets_root(assistant_name: str) -> Path | None:
    """
    解析当前角色的 Live2D 资源根目录

    路径策略：
    - 返回 data/agents/{assistant_name}/assets/live2d 目录

    参数：
    - assistant_name: 角色名称

    返回：
    - 目录存在时返回目录路径
    - 目录不存在返回 None
    """
    assets_root = Path("data") / "agents" / assistant_name / "assets" / "live2d"
    if not assets_root.exists():
        Log.warning(f"[表情加载] 模型目录不存在: {assets_root}")
        return None

    return assets_root


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

        expressions.append({
            "name": name,
            "file": str(rel_path).replace("\\", "/"),
        })

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


# ============================================================
# LLM 表情描述生成
# ============================================================

def _build_llm_prompt(expressions: list[tuple[str, dict[str, float]]]) -> str:
    """
    构建 LLM 表情描述生成的提示词

    参数：
    - expressions: [(表情名, {参数ID: 值}), ...] 列表

    返回：
    - 格式化的提示词字符串
    """
    # 参数含义说明
    param_meanings = """
参数含义参考：
- ParamEyeLSmile/ParamEyeRSmile: 眼睛微笑程度
- ParamEyeLOpen/ParamEyeROpen: 眼睛开度（1=全开，0=全闭）
- ParamEyeBallX/ParamEyeBallY: 眼球看向方向
- ParamBrowLY/ParamBrowRY: 眉毛高度（正=上扬，负=下压）
- ParamBrowLForm/ParamBrowRForm: 眉毛形态
- ParamMouthForm: 嘴型（正=微笑，负=悲伤）
- ParamMouthOpenY: 嘴巴张开程度
- ParamCheek: 脸颊红晕程度
- ParamAngleX/Y/Z: 头部角度
"""

    # 构建表情参数列表
    expr_lines = []
    for name, params in expressions:
        # 将参数格式化为简洁形式
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        expr_lines.append(f"- {name}: [{param_str}]")

    expr_list_str = "\n".join(expr_lines)

    return f"""你是一个 Live2D 表情分析专家。根据表情包含的参数组合，推断表情的含义并用简短中文描述。

{param_meanings}

描述要求：
1. 用简短中文描述（5-10个字）
2. 描述表情的视觉效果和情感含义
3. 例如：
   - eye_smile + mouth_form(正) → "微笑"
   - eye_open(0) + cheek(1) → "害羞闭眼"
   - mouth_open(1) → "惊讶张嘴"
   - brow(负) + mouth_form(负) → "生气"

请为以下表情生成描述：
{expr_list_str}

返回 JSON 格式：
{{
  "descriptions": {{
    "表情名": "描述",
    ...
  }}
}}
"""


async def _generate_expression_descriptions(
    expressions: list[tuple[str, dict[str, float]]],
) -> dict[str, str]:
    """
    批量调用 LLM 为表情生成中文描述

    参数：
    - expressions: [(表情名, {参数ID: 值}), ...] 列表

    返回：
    - {表情名: 描述} 字典
    - 调用失败返回空字典
    """
    if not expressions:
        return {}

    prompt = _build_llm_prompt(expressions)

    try:
        Log.info(f"[表情加载] 调用 LLM 生成 {len(expressions)} 个表情描述...")
        content = await _llm_client.request([{"role": "user", "content": prompt}])

        if not content:
            Log.warning("[表情加载] LLM 返回内容为空")
            return {}

        result = parse_llm_json_response(content)
        descriptions = result.get("descriptions", {})

        if not isinstance(descriptions, dict):
            Log.warning("[表情加载] LLM 返回格式错误")
            return {}

        Log.info(f"[表情加载] 成功生成 {len(descriptions)} 个表情描述")
        return descriptions

    except Exception as e:
        Log.error(f"[表情加载] LLM 调用失败: {e}")
        return {}


# ============================================================
# 主入口函数
# ============================================================

async def load_expressions(
    assistant_name: str,
    use_cache: bool = True,
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
    assets_root = _resolve_assets_root(assistant_name)
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
    if use_cache:
        cached_descriptions = _load_cache(assets_root)

    # Step 5: 缓存未命中的表情需要调用 LLM 生成描述
    need_llm = []
    descriptions = cached_descriptions or {}

    for name, params, _ in expressions_with_params:
        if name not in descriptions:
            need_llm.append((name, params))

    # 调用 LLM 生成缺失的描述
    if need_llm:
        new_descriptions = await _generate_expression_descriptions(need_llm)
        descriptions.update(new_descriptions)

        # Step 6: 保存更新后的缓存
        if use_cache and new_descriptions:
            _save_cache(assets_root, descriptions)

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


def build_expression_descriptions(expressions: list[ExpressionInfo]) -> str:
    """
    构建表情描述文本，供 LLM 提示词使用

    输出格式示例：
    - "微笑": 开心微笑，眼睛弯起 [els:1, mf:0.5]
    - "害羞": 害羞闭眼，脸颊泛红 [elo:0, chk:1]

    参数：
    - expressions: ExpressionInfo 列表

    返回：
    - 格式化的表情描述字符串
    """
    if not expressions:
        return "（无可用表情）"

    lines = []
    for expr in expressions:
        # 将参数转换为别名格式（简化显示）
        param_summary = []
        for param_id, value in expr.parameters.items():
            # 只显示关键参数（简化）
            if value != 0:
                param_summary.append(f"{param_id}={value}")

        # 限制显示的参数数量
        if len(param_summary) > 3:
            param_summary = param_summary[:3]
            param_summary.append("...")

        params_str = ", ".join(param_summary)
        lines.append(f'- "{expr.name}": {expr.description} [{params_str}]')

    return "\n".join(lines)
