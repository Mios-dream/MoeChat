from utils.llm_request import llm_request, parse_llm_json_response
from utils.log import logger
from pathlib import Path
import json
import time

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
    """根据参数 ID 提供兜底范围，避免动作生成阶段缺少边界信息。"""
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
    """解析当前助手 Live2D model3.json 的绝对路径。"""
    assets_root = Path("data") / "agents" / assistant_name / "assets"
    if not assets_root.exists():
        return None

    assets_json_path = assets_root / "assets.json"
    model_json_path: Path | None = None

    if assets_json_path.exists():
        try:
            with open(assets_json_path, "r", encoding="utf-8") as f:
                assets_data = json.load(f)
            raw_path = (
                (assets_data.get("live2d") or {}).get("modelJsonPath")
                if isinstance(assets_data, dict)
                else None
            )
            if raw_path:
                normalized = str(raw_path).replace("\\", "/")
                if normalized.startswith("assistants/"):
                    marker = "/assets/"
                    if marker in normalized:
                        suffix = normalized.split(marker, 1)[1]
                        candidate = assets_root / suffix
                        if candidate.exists():
                            model_json_path = candidate
                else:
                    candidate = Path(normalized)
                    if candidate.exists():
                        model_json_path = candidate
                    else:
                        candidate_in_assets = assets_root / normalized
                        if candidate_in_assets.exists():
                            model_json_path = candidate_in_assets
        except Exception:
            logger.warning("[动作规划] 读取 assets.json 失败，尝试自动搜索 model3.json")

    if not model_json_path:
        candidates = list(assets_root.rglob("*.model3.json"))
        if candidates:
            model_json_path = candidates[0]

    return model_json_path


def _get_motion_parameter_cache_path(assistant_name: str) -> Path:
    """返回动作参数缓存文件路径。"""
    return Path("data") / "agents" / assistant_name / "motion_parameters_cache.json"


def _load_live2d_parameters_raw(model_json_path: Path) -> dict[str, dict]:
    """加载 Live2D 全量参数（未筛选）。"""
    parameters: dict[str, dict] = {}

    try:
        with open(model_json_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:
        logger.warning(f"[动作规划] 读取 model3.json 失败: {e}")
        return {}
    # 解析 model3.json 同目录下的 cdi3.json 和 vtube.json，获取参数的 min/max 信息，提升动作规划质量
    model_dir = model_json_path.parent
    display_rel = (
        (model_data.get("FileReferences") or {}).get("DisplayInfo") or ""
    ).strip()
    # 优先使用 assets.json 中的路径指示，如果缺失或无效，再自动搜索同目录下的 cdi3.json 文件
    display_path = model_dir / display_rel if display_rel else None
    if (not display_path) or (not display_path.exists()):
        cdi_candidates = list(model_dir.glob("*.cdi3.json"))
        display_path = cdi_candidates[0] if cdi_candidates else None
    # cdi3.json 中通常包含参数的友好名称，vtube.json 中通常包含参数的有效范围，结合两者可以显著提升动作规划的效果
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
    # 如果 assets.json 中没有明确指示 vtube.json 的位置，且同目录下存在 vtube.json 文件，则使用它来补充参数的 min/max 信息
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
    # 如果 cdi3.json 和 vtube.json 都没有提供参数信息，则使用默认的参数构建逻辑
    if not parameters:
        for pid in ESSENTIAL_MOTION_PARAMETER_IDS:
            parameters[pid] = _build_default_parameter(pid)

    return parameters


def _load_motion_parameter_cache(
    assistant_name: str,
    model_json_path: Path,
) -> dict[str, dict] | None:
    """读取参数缓存，命中时直接返回。"""
    cache_path = _get_motion_parameter_cache_path(assistant_name)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"[动作规划] 读取参数缓存失败: {e}")
        return None

    if not isinstance(payload, dict):
        return None

    try:
        model_mtime = model_json_path.stat().st_mtime
    except Exception:
        model_mtime = None

    if payload.get("model_json_path") != str(model_json_path):
        return None
    if model_mtime is not None and payload.get("model_mtime") != model_mtime:
        return None

    cached_parameters = payload.get("parameters")
    if not isinstance(cached_parameters, dict):
        return None

    logger.info(f"[动作规划] 参数缓存命中: {cache_path}")
    return cached_parameters


def _save_motion_parameter_cache(
    assistant_name: str,
    model_json_path: Path,
    parameters: dict[str, dict],
) -> None:
    """保存已筛选的动作参数，避免重复调用 LLM。"""
    cache_path = _get_motion_parameter_cache_path(assistant_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model_mtime = model_json_path.stat().st_mtime
    except Exception:
        model_mtime = None

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


def _fallback_select_parameter_ids(all_parameters: dict[str, dict]) -> set[str]:
    """LLM 异常时的本地启发式筛选。"""
    keep_ids: set[str] = set()
    keywords = (
        "eye",
        "brow",
        "mouth",
        "cheek",
        "angle",
        "body",
        "head",
        "neck",
        "breath",
        "arm",
        "hand",
    )
    for pid in all_parameters.keys():
        lower_id = pid.lower()
        if any(k in lower_id for k in keywords):
            keep_ids.add(pid)
    return keep_ids


async def _filter_parameters_with_llm(
    all_parameters: dict[str, dict],
) -> set[str]:
    """
    让 LLM 判断动作规划需要保留的参数 ID。
    return: 需要保留的参数 ID 集合
    """
    # 没有参数信息时无法筛选，直接返回空集
    if not all_parameters:
        return set()
    # 基础参数是动作规划的核心，必须保留；其他参数交由 LLM 判断
    essential_ids = {
        pid for pid in ESSENTIAL_MOTION_PARAMETER_IDS if pid in all_parameters
    }
    # 构建额外参数候选列表，排除基础参数
    extra_candidates = [
        pid for pid in all_parameters.keys() if pid not in essential_ids
    ]
    # 没有额外参数时直接返回基础参数，避免调用 LLM
    if not extra_candidates:
        return essential_ids
    # 构建参数描述列表，用于 LLM 筛选
    candidate_lines = []
    for pid in extra_candidates:
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
            logger.warning("[动作规划] LLM 参数筛选返回空响应，降级启发式筛选")
            return essential_ids | _fallback_select_parameter_ids(all_parameters)
        data = parse_llm_json_response(response)
        keep_ids_raw = data.get("keep_ids") if isinstance(data, dict) else None
        keep_ids = {
            str(pid)
            for pid in (keep_ids_raw or [])
            if isinstance(pid, str) and pid in all_parameters
        }
        # 返回基础参数和 LLM 筛选的参数的并集，确保动作规划的核心功能，同时尽可能丰富表情和动作变化
        return essential_ids | keep_ids
    except Exception as e:
        logger.warning(f"[动作规划] LLM 参数筛选失败，降级启发式筛选: {e}")
        # LLM 异常时返回基础参数和启发式筛选的参数的并集，确保动作规划的核心功能，同时尽可能保留有用参数
        return essential_ids | _fallback_select_parameter_ids(all_parameters)


async def load_live2d_parameters(assistant_name: str) -> dict[str, dict]:
    """加载并筛选 Live2D 参数，优先使用缓存，缺失时走 LLM 筛选。"""
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
    selected_ids |= ESSENTIAL_MOTION_PARAMETER_IDS

    filtered_parameters = {
        pid: all_parameters.get(pid, _build_default_parameter(pid))
        for pid in selected_ids
    }

    _save_motion_parameter_cache(assistant_name, model_json_path, filtered_parameters)
    logger.info(
        f"[动作规划] 参数筛选完成: 全量 {len(all_parameters)} -> 保留 {len(filtered_parameters)}"
    )
    return filtered_parameters
