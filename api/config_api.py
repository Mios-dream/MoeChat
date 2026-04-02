from my_utils import config as CConfig
from fastapi import (
    APIRouter,
    HTTPException,
)
from pydantic import BaseModel, ConfigDict, Field
from typing import Any


config_api = APIRouter()


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any] = Field(..., description="待更新的配置项")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_value_rule(path: str, value: Any):
    enum_rules = {
        "TTS.mode": {"api", "local"},
    }
    url_rules = {
        "LLM.api",
        "ChatLLM.api",
        "SLM.api",
        "TTS.gptsovits.api",
    }

    if path in enum_rules and value not in enum_rules[path]:
        allowed = ", ".join(sorted(enum_rules[path]))
        raise HTTPException(status_code=400, detail=f"{path} 仅支持: {allowed}")

    if path in url_rules and not (
        isinstance(value, str)
        and (value.startswith("http://") or value.startswith("https://"))
    ):
        raise HTTPException(
            status_code=400, detail=f"{path} 必须是有效的 http/https URL"
        )

    if path == "SV.thr" and value is not None:
        if not _is_number(value):
            raise HTTPException(status_code=400, detail=f"{path} 必须在 [0, 1] 区间")
        numeric_value = float(value)
        if not (0 <= numeric_value <= 1):
            raise HTTPException(status_code=400, detail=f"{path} 必须在 [0, 1] 区间")

    if path == "SLM.extra_config.temperature":
        if not _is_number(value):
            raise HTTPException(
                status_code=400,
                detail=f"{path} 建议在 [0, 2] 区间",
            )
        numeric_value = float(value)
        if not (0 <= numeric_value <= 2):
            raise HTTPException(
                status_code=400,
                detail=f"{path} 建议在 [0, 2] 区间",
            )


def _validate_config_patch(patch: Any, current: Any, path: str = ""):
    if isinstance(patch, dict):
        if not isinstance(current, dict):
            raise HTTPException(
                status_code=400, detail=f"{path or 'root'} 类型应为对象"
            )

        # extra_config 允许透传任意模型参数
        if path.endswith("extra_config"):
            for k, v in patch.items():
                next_path = f"{path}.{k}" if path else k
                _check_value_rule(next_path, v)
            return

        for key, value in patch.items():
            if key not in current:
                raise HTTPException(
                    status_code=400,
                    detail=f"未知配置项: {path + '.' if path else ''}{key}",
                )
            next_path = f"{path}.{key}" if path else key
            _validate_config_patch(value, current[key], next_path)
        return

    if isinstance(current, bool):
        if not isinstance(patch, bool):
            raise HTTPException(status_code=400, detail=f"{path} 类型应为 bool")
    elif _is_number(current):
        if not _is_number(patch):
            raise HTTPException(status_code=400, detail=f"{path} 类型应为 number")
    elif isinstance(current, str):
        if not isinstance(patch, str):
            raise HTTPException(status_code=400, detail=f"{path} 类型应为 string")
    elif isinstance(current, list):
        if not isinstance(patch, list):
            raise HTTPException(status_code=400, detail=f"{path} 类型应为数组")
    elif current is None:
        # 原值为 None 时不做类型限制，仅做规则校验
        pass

    _check_value_rule(path, patch)


# 客户端获取配置信息
@config_api.get("/get_config")
async def get_config():
    return CConfig.config


# 更新配置文件
@config_api.post("/update_config")
async def update_config(data: ConfigUpdateRequest):
    if not data.data:
        raise HTTPException(status_code=400, detail="更新内容不能为空")

    _validate_config_patch(data.data, CConfig.config)
    CConfig.update_config(data.data)
    return {"message": "配置更新成功", "config": CConfig.config}
