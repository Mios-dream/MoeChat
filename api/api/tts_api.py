import base64
import json
from fastapi.responses import StreamingResponse
from fastapi import (
    APIRouter,
)
from core.chat_core import tts_task, TTSData
from utils.split_text import remove_parentheses_content_and_split

# 聊天接口
from pydantic import BaseModel


class msg_data(BaseModel):
    msg: str


tts_api = APIRouter()


async def start_tts_task(msg: list[str]):
    for i in msg:
        item: TTSData = TTSData(text=i, ref_audio="", ref_text="")
        audio_data = await tts_task(item)

        if audio_data is None:
            continue
        encode_data = base64.b64encode(audio_data).decode("utf-8")
        data = {"text": i, "audio": encode_data}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@tts_api.post("/gptsovits")
async def tts(params: msg_data):

    msg = remove_parentheses_content_and_split(params.msg, is_remove_incomplete=False)

    return StreamingResponse(start_tts_task(msg), media_type="text/event-stream")
