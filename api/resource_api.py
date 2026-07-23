"""
资源库 API

提供预设资源库的查询接口，让前端可以浏览可用的通用资源。
预留未来扩展其他资源类型的查询（表情、动作等）。
"""

from fastapi import APIRouter, HTTPException
from services.resource_service import resource_service
from my_utils.log import logger

resource_api = APIRouter()


@resource_api.get("/resources/presets")
async def list_presets():
    """
    获取所有可用的预设资源列表

    Returns:
        包含预设资源信息的列表，每个元素含 name, type, description, files, config
    """
    try:
        presets = resource_service.list_presets()
        return {
            "msg": "获取预设资源列表成功",
            "data": presets,
            "count": len(presets),
        }
    except Exception as e:
        logger.error(f"获取预设资源列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预设资源列表失败: {e}")


@resource_api.get("/resources/presets/{name}")
async def get_preset(name: str):
    """
    获取指定预设资源的详细信息

    Args:
        name: 预设资源名称

    Returns:
        预设资源详细信息
    """
    try:
        preset = resource_service.get_preset(name)
        if not preset:
            raise HTTPException(status_code=404, detail=f"预设资源 '{name}' 不存在")
        return {
            "msg": f"获取预设资源 '{name}' 成功",
            "data": preset,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取预设资源失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预设资源失败: {e}")
