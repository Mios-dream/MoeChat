# 聊天接口
from pydantic import BaseModel


class tts_data(BaseModel):
    msg: list
