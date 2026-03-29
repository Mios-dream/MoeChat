from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from models.dto.chat_request import chat_data
import core.chat_core as chat_core
from fastapi import Query

chat_api = APIRouter()


@chat_api.get("/chat")
async def tts_api_get_v1(
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
async def tts_api_v3(params: chat_data):
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


# # 客户端获取聊天记录
# @chat_api.post("/get_context")
# async def get_context():
#     agent = Agent()
#     return agent.msg_data
