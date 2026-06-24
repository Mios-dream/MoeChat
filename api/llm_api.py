from fastapi import APIRouter, HTTPException
from core.llm.llm_client import LLMClient
from pydantic import BaseModel
from typing import Any, cast
from openai.types.chat import ChatCompletionMessageParam


class llm_data(BaseModel):
    msg: list[dict[str, Any]]


llm_api = APIRouter()

# 全局 LLM 客户端实例
_llm_client = LLMClient(model_key="LLM")


@llm_api.post("/llm_chat")
async def llm(params: llm_data):
    result = await _llm_client.request(cast(list[ChatCompletionMessageParam], params.msg))
    if result:
        return {"content": result}
    else:
        raise HTTPException(status_code=500, detail="大模型服务异常")
