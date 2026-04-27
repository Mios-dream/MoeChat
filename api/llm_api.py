from fastapi import APIRouter, HTTPException
from my_utils.llm_request import llm_request
from pydantic import BaseModel
from typing import Any, cast
from openai.types.chat import ChatCompletionMessageParam


class llm_data(BaseModel):
    msg: list[dict[str, Any]]


llm_api = APIRouter()


@llm_api.post("/llm_chat")
async def llm(params: llm_data):
    result = await llm_request(cast(list[ChatCompletionMessageParam], params.msg))
    if result:
        return {"content": result}
    else:
        raise HTTPException(status_code=500, detail="大模型服务异常")
