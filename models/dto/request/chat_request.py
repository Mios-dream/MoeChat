# 聊天接口
from pydantic import BaseModel


class ChatData(BaseModel):
    msg: list[dict[str, str]]
    generation_motion: bool = False
    is_sleep_mode: bool = False  # 睡眠模式标识
