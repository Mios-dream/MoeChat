"""
聊天 API 模块

提供聊天相关的 HTTP 接口。

支持的版本：
- v1: 基础版本（文本 + TTS）
- v2: 带动作版本（使用 V2 生成器）
- v3: 带动作版本（使用 V3 生成器）
- v4: 信息调度中心版本（推荐）

通过配置文件选择版本：
```yaml
MotionGenerator:
  version: v4  # v1 | v2 | v3 | v4
```
"""

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query
from models.dto.chat_request import chat_data
from my_utils import config_manager as CConfig

# 导入基础组件
from core.chat.base import assistant_service

# 导入各版本的聊天函数
from core.chat import (
    llm_chat_with_tts,  # V1 版本
    llm_chat_with_tts_and_motion_v2,  # V2Motion 版本
    llm_chat_with_tts_and_motion_v3,  # V3 版本
    llm_chat_with_tts_and_motion_v4,  # V4 版本
)

chat_api = APIRouter()


def _get_motion_version() -> str:
    """
    获取配置中的动作生成器版本

    返回：
    - "v2", "v3", "v4"
    """
    motion_config = CConfig.config.get("MotionGenerator", {})
    return motion_config.get("version", "v2")


@chat_api.get("/chat")
async def tts_api_get(
    text: str = Query(..., description="用户输入的文本"),
    generation_motion: bool = False,
    is_sleep_mode: bool = Query(False, description="是否处于睡眠模式"),
):
    """
    GET 方式聊天接口

    参数：
    - text: 用户输入文本
    - generation_motion: 是否生成动作
    - is_sleep_mode: 是否睡眠模式
    """
    if not text:
        raise ValueError("消息内容不能为空")

    message_list = [{"role": "user", "content": text}]
    params = chat_data(msg=message_list, is_sleep_mode=is_sleep_mode)

    if generation_motion:
        # 根据配置选择版本
        motion_version = _get_motion_version()

        if motion_version == "v4":
            # V4 版本：信息调度中心
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v4(params),
                media_type="text/event-stream",
            )
        elif motion_version == "v3":
            # V3 版本：流式 Batch 架构
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v3(params),
                media_type="text/event-stream",
            )
        else:
            # V2 版本：基础版本
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v2(params),
                media_type="text/event-stream",
            )
    else:
        # 不生成动作
        return StreamingResponse(
            llm_chat_with_tts(params),
            media_type="text/event-stream",
        )


@chat_api.post("/chat")
async def tts_api(params: chat_data):
    """
    POST 方式聊天接口

    参数：
    - params: 聊天请求参数
    """
    if params.msg is None:
        raise ValueError("消息内容不能为空")

    if params.generation_motion:
        # 根据配置选择版本
        motion_version = _get_motion_version()

        if motion_version == "v4":
            # V4 版本：信息调度中心
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v4(params),
                media_type="text/event-stream",
            )
        elif motion_version == "v3":
            # V3 版本：流式 Batch 架构
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v3(params),
                media_type="text/event-stream",
            )
        else:
            # V2 版本：基础版本
            return StreamingResponse(
                llm_chat_with_tts_and_motion_v2(params),
                media_type="text/event-stream",
            )
    else:
        # 不生成动作
        return StreamingResponse(
            llm_chat_with_tts(params),
            media_type="text/event-stream",
        )


@chat_api.get("/chat/history")
async def get_chat_history(
    only_assistant: bool = Query(False, description="是否只返回助手消息"),
    limit: int = Query(10, ge=1, le=50, description="最多返回的消息条数"),
):
    """
    获取聊天历史

    参数：
    - only_assistant: 是否只返回助手消息
    - limit: 最多返回的消息条数
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")

    history_list = agent.memoryEngine.get_recent_chat_turns(
        limit=limit, only_assistant=only_assistant
    )

    return {
        "msg": "Get chat history success",
        "assistant": agent.agent_name,
        "onlyAssistant": only_assistant,
        "source": "sqlite",
        "count": len(history_list),
        "data": history_list,
    }


@chat_api.get("/chat/diary")
async def get_chat_diary(
    limit: int = Query(20, ge=1, le=100, description="单次返回的日记条数"),
    offset: int = Query(0, ge=0, description="分页偏移量"),
    start_day: str | None = Query(None, description="起始日期，格式 YYYY-MM-DD"),
    end_day: str | None = Query(None, description="结束日期，格式 YYYY-MM-DD"),
):
    """
    获取日记记录

    参数：
    - limit: 单次返回的日记条数
    - offset: 分页偏移量
    - start_day: 起始日期
    - end_day: 结束日期
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")

    try:
        records, total = agent.memoryEngine.get_diary_records(
            limit=limit,
            offset=offset,
            start_day=start_day,
            end_day=end_day,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日记记录失败: {str(e)}")

    return {
        "msg": "Get diary records success",
        "assistant": agent.agent_name,
        "limit": limit,
        "offset": offset,
        "startDay": start_day,
        "endDay": end_day,
        "count": len(records),
        "total": total,
        "data": records,
    }
