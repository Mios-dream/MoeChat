import time
from fastapi.responses import StreamingResponse
from fastapi import (
    APIRouter,
)
from utils.agent import Agent
from api.models.tts_request import tts_data
import core.chat_core as chat_core


chat_api = APIRouter()


@chat_api.post("/chat")
async def tts(params: tts_data):
    return StreamingResponse(
        chat_core.text_llm_tts(params), media_type="text/event-stream"
    )


@chat_api.post("/chat_v2")
async def tts_api(params: tts_data):
    return StreamingResponse(
        chat_core.text_llm_tts_v2(params), media_type="text/event-stream"
    )


# 客户端获取聊天记录
@chat_api.post("/get_context")
async def get_context():
    agent = Agent()
    return agent.msg_data
