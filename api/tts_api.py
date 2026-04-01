import base64
import json
from fastapi import (
    APIRouter,
)
from core.chat_core import tts_task, TTSData
from pydantic import BaseModel
from my_utils.split_text import remove_parentheses_content_and_split


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
        data = {"message": i, "file": encode_data}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@tts_api.post("/gptsovits")
async def tts(params: msg_data):

    msg = remove_parentheses_content_and_split(params.msg, is_remove_incomplete=False)

    audio_data = await tts_task(TTSData(text=params.msg, ref_audio="", ref_text=""))

    if audio_data is None:
        return {"message": msg, "file": None}
    encode_data = base64.b64encode(audio_data).decode("utf-8")
    return {"message": params.msg, "file": encode_data}
