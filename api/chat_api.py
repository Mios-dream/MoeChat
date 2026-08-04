"""
聊天 API 模块

提供聊天相关的 HTTP 接口。
```
"""

import json

from fastapi import APIRouter, HTTPException, Query

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

    history_list = _simplify_history(agent.chat_history.copy())

    return {
        "msg": "Get chat history success",
        "assistant": agent.agent_name,
        "onlyAssistant": history_list,
        "source": "memory",
        "count": len(history_list),
        "data": history_list,
    }


def _simplify_history(
    history: list[dict],
) -> list[dict]:
    """
    将聊天记录转换为前端可直接展示的 OpenAI 消息链。

    数据源为 HistoryManager.copy() 的投影展示记录（含 kind/timestamp），
    LLM 上下文（get_for_llm 投影）不经过此接口。

    处理规则：
    - 跳过压缩摘要等非展示类别（kind 为 summary/note/event 的记录；copy() 已过滤，此处防御性兜底）；
    - 对 V3 多任务 JSON 格式的 assistant 类记录（kind 为 chat/interaction，content 为逐行
      JSON，形如 ``{"text": "句子", "actions": [...]}``），逐行解析并提取 text，
      展开为逐句的独立展示项，去掉多任务结构；
    - 其余记录（user / tool_call / tool_result / 纯文本 assistant）原样返回。

    展示记录中仍以多任务 JSON 整体存储（供模型学习输出格式），
    仅在对外输出时解析拆分，保证与前端按句展示的格式一致。
    """
    simplified: list[dict] = []
    for msg in history:
        # 过滤压缩摘要/事件/备注等非展示类别（copy() 已过滤，此处防御性兜底）
        if msg.get("kind") in ("summary", "note", "event"):
            continue

        if msg.get("kind") in ("chat", "interaction") and isinstance(msg.get("content"), str):
            texts: list[str] = []
            is_multi_task = True
            for line in msg["content"].splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # 非 JSON 行（纯文本消息）
                    is_multi_task = False
                    break
                if isinstance(data, dict) and "text" in data:
                    text = data["text"].strip()
                    if text:
                        texts.append(text)
                else:
                    # 可解析 JSON 但无 text 字段，按普通消息处理
                    is_multi_task = False
                    break
            if is_multi_task and texts:
                # 多任务 JSON：逐句展开为独立展示项（保留类别/时间标记，供前端回合分组）
                for text in texts:
                    entry: dict = {"kind": msg.get("kind"), "content": text}
                    if msg.get("timestamp"):
                        entry["timestamp"] = msg["timestamp"]
                    simplified.append(entry)
                continue

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
