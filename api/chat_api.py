from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
from models.dto.chat_request import chat_data
import core.chat_core as chat_core
from fastapi import Query
import os
import yaml

chat_api = APIRouter()


@chat_api.get("/chat")
async def tts_api_get(
    text: str = Query(..., description="用户输入的文本"),
    generation_motion: bool = False,
):
    if not text:
        raise ValueError("消息内容不能为空")
    message_list = [{"role": "user", "content": text}]
    if generation_motion:
        return StreamingResponse(
            chat_core.llm_chat_with_tts_and_motion(chat_data(msg=message_list)),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            chat_core.llm_chat_with_tts(chat_data(msg=message_list)),
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

    history_path = f"./data/agents/{agent.agent_name}/history.yaml"
    history_list = []

    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or []
                if isinstance(loaded, list):
                    history_list = loaded
        else:
            history_list = agent.msg_data.copy()
    except Exception:
        history_list = agent.msg_data.copy()

    if only_assistant:
        history_list = [
            item
            for item in history_list
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
    else:
        history_list = [
            item
            for item in history_list
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
        ]

    history_list = history_list[-limit:]

    return {
        "msg": "Get chat history success",
        "assistant": agent.agent_name,
        "onlyAssistant": only_assistant,
        "count": len(history_list),
        "data": history_list,
    }
