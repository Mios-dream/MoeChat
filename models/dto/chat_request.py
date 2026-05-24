# 聊天接口
from pydantic import BaseModel


class chat_data(BaseModel):
    msg: list
    generation_motion: bool = False
    is_sleep_mode: bool = False  # 睡眠模式标识
