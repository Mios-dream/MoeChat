# asr接口
from pydantic import BaseModel


class asr_data(BaseModel):
    data: str
