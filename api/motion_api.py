import asyncio
import json
import re
from fastapi import APIRouter
from pydantic import BaseModel

from core.expression_generator.motion_engine_v3 import (
    MotionEngineService,
    estimate_text_duration,
)
from Config import Config


class msg_data(BaseModel):
    msg: str


motion_api = APIRouter()


# 全局引擎实例（模块加载时创建，避免每次请求重建）
_motion_engine = MotionEngineService(Config.MOTION_DB_PATH)


async def _generate_motion(text: str) -> dict | None:
    """
    使用 MotionEngineService 生成动作数据（在线程池中运行 CPU 密集型处理）
    返回与 V3 模块兼容的字典：{"duration": ms, "curves": ..., "fps": ..., "expression": ...}
    """
    text_duration = estimate_text_duration(text)
    loop = asyncio.get_running_loop()

    motion_data = await loop.run_in_executor(
        None, lambda: _motion_engine.process(text, [], None, text_duration)
    )

    if motion_data is None:
        return None

    duration_ms = int((motion_data.duration if motion_data else text_duration) * 1000)

    motion_dict: dict = {
        "duration": duration_ms,
        "curves": motion_data.curves,
        "fps": motion_data.fps,
        "expression": motion_data.expression,
    }

    return motion_dict


@motion_api.post("/generate_motion")
async def generate_motion(params: msg_data):
    # 移除括号中的内容，行为与语音接口一致
    msg = re.sub(r"\(.*?\)|（.*?）|【.*?】|\[.*?\]|\{.*?\}", "", params.msg)

    motion = await _generate_motion(msg)

    if motion is None:
        return {"message": params.msg, "motion": None}

    return {"message": params.msg, "motion": motion}
