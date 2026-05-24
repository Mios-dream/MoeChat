from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
from models.dto.chat_request import chat_data
import core.chat_core as chat_core
from fastapi import Query

chat_api = APIRouter()


@chat_api.get("/chat")
async def tts_api_get(
    text: str = Query(..., description="用户输入的文本"),
    generation_motion: bool = False,
    is_sleep_mode: bool = Query(False, description="是否处于睡眠模式"),
):
    if not text:
        raise ValueError("消息内容不能为空")
    message_list = [{"role": "user", "content": text}]
    if generation_motion:
        return StreamingResponse(
            chat_core.llm_chat_with_tts_and_motion(
                chat_data(msg=message_list, is_sleep_mode=is_sleep_mode)
            ),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            chat_core.llm_chat_with_tts(
                chat_data(msg=message_list, is_sleep_mode=is_sleep_mode)
            ),
            media_type="text/event-stream",
        )


@chat_api.post("/chat")
async def tts_api(params: chat_data):
    if params.msg is None:
        raise ValueError("消息内容不能为空")
    if params.generation_motion:
        return StreamingResponse(
            chat_core.llm_chat_with_tts_and_motion(params),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            chat_core.llm_chat_with_tts(params), media_type="text/event-stream"
        )


# 客户端获取聊天记录
@chat_api.get("/chat/history")
async def get_chat_history(
    only_assistant: bool = Query(False, description="是否只返回助手消息"),
    limit: int = Query(10, ge=1, le=50, description="最多返回的消息条数"),
):
    agent = chat_core.assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")

    try:
        history_list = agent.memoryEngine.get_recent_chat_turns(
            limit=limit, only_assistant=only_assistant
        )
    except Exception:
        history_list = agent.msg_data.copy()

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
    agent = chat_core.assistant_service.get_current_assistant()
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
