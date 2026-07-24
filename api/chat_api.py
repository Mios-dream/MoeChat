"""
聊天 API 模块

提供聊天相关的 HTTP 接口。
```
"""

import json

from fastapi import APIRouter, HTTPException, Query
from openai.types.chat import ChatCompletionMessageParam

# 导入基础组件
from core.chat.base import assistant_service

chat_api = APIRouter()


@chat_api.get("/chat/history")
async def get_chat_history():
    """
    获取聊天历史（内存中的完整消息记录，包含工具调用信息）

    参数：
    - only_assistant: 是否只返回助手消息
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")

    history_list = agent.get_history()
    history_list = _simplify_history(history_list)

    return {
        "msg": "Get chat history success",
        "assistant": agent.agent_name,
        "onlyAssistant": history_list,
        "source": "memory",
        "count": len(history_list),
        "data": history_list,
    }


def _simplify_history(
    history: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """
    简化聊天历史中的助手消息。

    V3 版本将多任务 JSON 存入 content（如 ``{"text": "你好", "actions": {...}}``），
    前端直接展示需要自行解析 JSON，本函数预处理提取纯文本，方便前端直接使用。

    处理规则：
    - 对 assistant 消息，尝试解析 content 为 JSON（支持逐行解析），
      成功则用 text 字段替换原 content；
    - 解析失败或非 assistant 消息则原样保留。
    """
    simplified: list[ChatCompletionMessageParam] = []
    for msg in history:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content: str = msg["content"]  # type: ignore[assignment]
            texts: list[str] = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        texts.append(data["text"])
                    else:
                        # 有可解析 JSON 但无 text 字段，不做处理
                        texts = []
                        break
                except json.JSONDecodeError:
                    # 非 JSON 行（纯文本），不做处理
                    texts = []
                    break
            if texts:
                plain_text = " ".join(texts)
                msg = {**msg, "content": plain_text}
        simplified.append(msg)
    return simplified


@chat_api.get("/chat/diary")
async def get_chat_diary(
    limit: int = Query(20, ge=1, le=100, description="单次返回的日记条数"),
    offset: int = Query(0, ge=0, description="分页偏移量"),
    start_day: str | None = Query(None, description="起始日期，格式 YYYY-MM-DD"),
    end_day: str | None = Query(None, description="结束日期，格式 YYYY-MM-DD"),
):
    """
    获取日记记录（来自记忆系统 v2）

    参数：
    - limit: 单次返回的日记条数
    - offset: 分页偏移量
    - start_day: 起始日期
    - end_day: 结束日期
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")

    try:
        records, total = agent.memoryEngine.get_diary_records(
            limit=limit,
            offset=offset,
            start_day=start_day,
            end_day=end_day,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日记记录失败: {str(e)}")

    return {
        "msg": "Get diary records success",
        "assistant": agent.agent_name,
        "limit": limit,
        "offset": offset,
        "startDay": start_day,
        "endDay": end_day,
        "count": len(records),
        "total": total,
        "data": records,
    }
