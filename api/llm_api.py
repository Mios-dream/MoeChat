from fastapi import APIRouter
from my_utils.llm_request import llm_request
from pydantic import BaseModel
from typing import Any, cast
from openai.types.chat import ChatCompletionMessageParam


class llm_data(BaseModel):
    msg: list[dict[str, Any]]


llm_api = APIRouter()


@llm_api.post("/llm_chat")
async def llm(params: llm_data):
    return {
        "content": await llm_request(cast(list[ChatCompletionMessageParam], params.msg))
    }
