from fastapi import APIRouter

from fastapi.responses import StreamingResponse

from models.dto.request.interaction_request import InteractionMessageRequest
from core.interaction_core import generate_interaction_message

interaction_api = APIRouter()


@interaction_api.post("/interaction/message")
async def interaction_message(params: InteractionMessageRequest):
    """交互消息生成接口。接收前端事件上下文，生成符合角色口吻的回复。

    响应格式：SSE 流 (text/event-stream)，与 /api/chat 完全一致。
    """
    return StreamingResponse(
        generate_interaction_message(params),
        media_type="text/event-stream",
    )
