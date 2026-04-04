from my_utils.llm_request import llm_request, parse_llm_json_response
from my_utils.log import logger
from pathlib import Path
import json
import time

# 模块职责：
# 1) 定位当前角色 Live2D 模型文件
# 2) 读取并整合参数来源（cdi3/vtube）
# 3) 通过 LLM + 启发式筛选动作规划参数
# 4) 通过缓存避免重复筛选，并在 moc3 更新时自动失效

# 基本动作参数 ID，必须保留的参数
ESSENTIAL_MOTION_PARAMETER_IDS = {
    "ParamEyeLOpen",
    "ParamEyeROpen",
    "ParamEyeBallX",
    "ParamEyeBallY",
    "ParamBrowLY",
    "ParamBrowRY",
    "ParamCheek",
    "ParamAngleX",
    "ParamAngleY",
    "ParamAngleZ",
    "ParamBodyAngleX",
    "ParamBodyAngleY",
    "ParamBodyAngleZ",
}


def _build_default_parameter(pid: str, name: str | None = None) -> dict:
    """
    根据参数 ID 推断一个可用的默认范围。

    设计目标：
    - 即使模型缺少完整参数描述，也能给动作规划提供边界值。
    - 不追求绝对精确，重点是避免下游出现空值或越界风险。

    参数：
    - pid: Live2D 参数 ID
    - name: 参数显示名称（可选）

    返回：
    - 包含 name/min/max/default 的参数字典
    """
    param_id = (pid or "").lower()

    if "eye" in param_id and "open" in param_id:
        min_v, max_v, default_v = 0.0, 1.0, 1.0
    elif "mouth" in param_id and "open" in param_id:
        min_v, max_v, default_v = 0.0, 1.0, 0.0
    elif "eyeball" in param_id or "brow" in param_id or "cheek" in param_id:
        min_v, max_v, default_v = -1.0, 1.0, 0.0
    elif "angle" in param_id or "body" in param_id or "head" in param_id:
        min_v, max_v, default_v = -30.0, 30.0, 0.0
    else:
        min_v, max_v, default_v = -30.0, 30.0, 0.0

    return {
        "name": name or pid,
        "min": min_v,
        "max": max_v,
        "default": default_v,
    }


def _resolve_model_json_path(assistant_name: str) -> Path | None:
    """
    解析当前角色的 model3.json 路径。

    路径策略：
    在 assets 目录递归搜索 *.model3.json

    返回：
    - 命中时返回文件路径
    - 未命中返回 None
    """
    assets_root = Path("data") / "agents" / assistant_name / "assets"
    if not assets_root.exists():
        return None

    model_json_path: Path | None = None

    candidates = list(assets_root.glob("*.model3.json"))
    if candidates:
        model_json_path = candidates[0]

    return model_json_path


def _get_motion_parameter_cache_path(assistant_name: str) -> Path:
    """返回角色动作参数缓存路径。"""
    return Path("data") / "agents" / assistant_name / "motion_parameters_cache.json"


def _get_file_mtime(path: Path | None) -> float | None:
    """
    安全读取文件修改时间。

    返回 None 表示不可用（不存在、权限问题或 IO 异常）。
    """
    if not path:
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _load_live2d_parameters_raw(model_json_path: Path) -> dict[str, dict]:
    """
    读取模型参数全集（不做筛选）。

    数据来源：
    - cdi3.json: 参数名称等显示信息
    - vtube.json: 参数范围（min/max）

    返回：
    - key 为参数 ID，value 为参数描述
    - 若读取失败，返回空字典
    - 例子：
    {
        "ParamEyeLOpen": {"name": "左眼开度", "min": 0.0, "max": 1.0, "default": 1.0},
        "ParamAngleX": {"name": "头部左右角度", "min": -30.0, "max": 30.0, "default": 0.0},
        ...
    }
    """
    parameters: dict[str, dict] = {}

    try:
        with open(model_json_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:
        logger.warning(f"[动作规划] 读取 model3.json 失败: {e}")
        return {}
    # 解析 model3 同目录参数描述文件，尽可能获得完整参数信息。
    model_dir = model_json_path.parent
    display_rel = (
        (model_data.get("FileReferences") or {}).get("DisplayInfo") or ""
    ).strip()
    # 优先使用 model3 的 DisplayInfo 指示；无效时再自动搜索 cdi3。
    display_path = model_dir / display_rel if display_rel else None
    if (not display_path) or (not display_path.exists()):
        cdi_candidates = list(model_dir.glob("*.cdi3.json"))
        display_path = cdi_candidates[0] if cdi_candidates else None
    # cdi3 主要提供参数名称，便于后续给 LLM 更清晰的上下文。
    if display_path and display_path.exists():
        try:
            with open(display_path, "r", encoding="utf-8") as f:
                display_data = json.load(f)

            for item in display_data.get("Parameters", []):
                pid = item.get("Id")
                if not pid:
                    continue
                param = _build_default_parameter(pid, item.get("Name", pid))
                parameters[pid] = param
        except Exception as e:
            logger.warning(f"[动作规划] 读取 cdi3.json 失败: {e}")

    vtube_path = model_json_path.with_suffix("")
    vtube_path = vtube_path.with_suffix(".vtube.json")
    # vtube 常用于补充输出范围，优先用同名推导，不存在则按目录搜索。
    if not vtube_path.exists():
        vtube_candidates = list(model_dir.glob("*.vtube.json"))
        vtube_path = vtube_candidates[0] if vtube_candidates else None

    if vtube_path and vtube_path.exists():
        try:
            with open(vtube_path, "r", encoding="utf-8") as f:
                vtube_data = json.load(f)

            for item in vtube_data.get("ParameterSettings", []):
                pid = item.get("OutputLive2D")
                if not pid:
                    continue
                out_lower = item.get("OutputRangeLower", -30)
                out_upper = item.get("OutputRangeUpper", 30)
                try:
                    min_v = float(min(out_lower, out_upper))
                except (TypeError, ValueError):
                    min_v = -30.0
                try:
                    max_v = float(max(out_lower, out_upper))
                except (TypeError, ValueError):
                    max_v = 30.0
                default_v = (min_v + max_v) / 2.0

                if pid not in parameters:
                    parameters[pid] = _build_default_parameter(
                        pid, item.get("Name", pid)
                    )

                parameters[pid]["min"] = min_v
                parameters[pid]["max"] = max_v
                parameters[pid]["default"] = default_v
        except Exception as e:
            logger.warning(f"[动作规划] 读取 vtube.json 失败: {e}")
    # 当外部描述文件缺失时，至少保留基础参数，保证下游可运行。
    if not parameters:
        for pid in ESSENTIAL_MOTION_PARAMETER_IDS:
            parameters[pid] = _build_default_parameter(pid)

    return parameters


def _load_motion_parameter_cache(
    assistant_name: str,
    model_json_path: Path,
) -> dict[str, dict] | None:
    """
    读取参数缓存并判断是否命中。

    返回：
    - 命中返回参数字典
    - 未命中或异常返回 None
    """
    cache_path = _get_motion_parameter_cache_path(assistant_name)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"[动作规划] 读取参数缓存失败: {e}")
        return None

    # 检查 model3 是否变更（路径和修改时间），确保缓存有效性。
    model_mtime = _get_file_mtime(model_json_path)
    if payload.get("model_json_path") != str(model_json_path):
        return None
    if model_mtime is not None and payload.get("model_mtime") != model_mtime:
        return None

    cached_parameters = payload.get("parameters", None)

    logger.info(f"[动作规划] 参数缓存命中: {cache_path}")
    return cached_parameters


def _save_motion_parameter_cache(
    assistant_name: str,
    model_json_path: Path,
    parameters: dict[str, dict],
) -> None:
    """
    保存筛选后的参数缓存。

    缓存中同时记录：
    - model3 路径与 mtime
    - 生成时间、参数数量及参数内容
    """
    cache_path = _get_motion_parameter_cache_path(assistant_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    model_mtime = _get_file_mtime(model_json_path)

    payload = {
        "version": 1,
        "assistant_name": assistant_name,
        "model_json_path": str(model_json_path),
        "model_mtime": model_mtime,
        "generated_at": int(time.time()),
        "parameter_count": len(parameters),
        "parameters": parameters,
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[动作规划] 保存参数缓存失败: {e}")


async def _filter_parameters_with_llm(
    all_parameters: dict[str, dict],
) -> set[str] | None:
    """
    使用 LLM 对参数做二次筛选。

    输入为全量参数，输出为“建议保留”的参数 ID 集合。
    函数内部会始终保留基础参数，并在异常时自动降级为本地启发式策略。
    """
    # 没有参数信息时无法筛选，直接返回空集
    if not all_parameters:
        return set()
    # 基础参数
    essential_ids = {
        pid for pid in ESSENTIAL_MOTION_PARAMETER_IDS if pid in all_parameters
    }
    # 传给 LLM 的候选描述尽量包含名称和范围，提升筛选稳定性。
    candidate_lines = []

    for pid in all_parameters.keys():
        info = all_parameters.get(pid, {})
        candidate_lines.append(
            f"- {pid} | name={info.get('name', pid)} | min={info.get('min', -30)} | max={info.get('max', 30)}"
        )

    prompt = f"""你是一位Live2D动作规划专家。请从以下参数列表中筛选出对角色动作规划最有用的参数。
所有参数：
{chr(10).join(candidate_lines)}
要求：
1. 必须保留基础表情和动作参数（眼睛、眉毛、嘴巴、脸颊、头部角度、身体角度等）
2. 筛选出对丰富表情和动作变化最重要的参数
3. 移除过于细微或很少使用的参数（例如部件物理,头发细节）
4. 返回JSON格式，包含keep_ids参数列表
请返回：
```json
{{
  "keep_ids": ["ParamEyeLOpen", "ParamEyeROpen", ...]
}}
```"""

    try:
        response = await llm_request(
            [
                {"role": "system", "content": prompt},
            ]
        )
        if not response:
            logger.error("[动作规划] LLM 参数筛选返回空响应")
            return None
        # 解析 LLM 响应
        data = parse_llm_json_response(response)
        # 提取建议保留的参数 ID 列表
        keep_ids_raw = data.get("keep_ids", [])

        keep_ids = {str(pid) for pid in keep_ids_raw if pid in all_parameters}
        # 与基础参数做并集，防止 LLM 漏选核心参数。
        return essential_ids | keep_ids
    except Exception as e:
        logger.error(f"[动作规划] LLM 参数筛选失败{e}")
        # 异常时走启发式方案，优先保障可用性。
        return None


async def load_live2d_parameters(assistant_name: str) -> dict[str, dict]:
    """
    对外主入口：加载并筛选动作参数。

    执行顺序：
    1) 定位 model3
    2) 尝试命中缓存
    3) 读取全量参数
    4) 调用 LLM 筛选
    5) 保存缓存并返回
    """
    model_json_path = _resolve_model_json_path(assistant_name)
    if not model_json_path or not model_json_path.exists():
        return {}

    cached = _load_motion_parameter_cache(assistant_name, model_json_path)
    if cached:
        return cached

    all_parameters = _load_live2d_parameters_raw(model_json_path)
    if not all_parameters:
        return {}

    selected_ids = await _filter_parameters_with_llm(all_parameters)
    # 双保险：即使前面筛选过程有偏差，也强制补齐基础参数。
    if selected_ids is None:
        return {}
    filtered_parameters = {
        pid: all_parameters.get(pid, _build_default_parameter(pid))
        for pid in selected_ids
    }

    _save_motion_parameter_cache(assistant_name, model_json_path, filtered_parameters)
    logger.info(
        f"[动作规划] 参数筛选完成: 全量 {len(all_parameters)} -> 保留 {len(filtered_parameters)}"
    )
    return filtered_parameters
