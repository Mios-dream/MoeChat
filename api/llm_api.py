from fastapi import APIRouter
from utils.llm_request import Message, llm_request
from pydantic import BaseModel
from typing import cast


class llm_data(BaseModel):
    msg: list[dict]


llm_api = APIRouter()


@llm_api.post("/llm_chat")
async def llm(params: llm_data):
    return {"content": await llm_request(cast(list[Message], params.msg))}
