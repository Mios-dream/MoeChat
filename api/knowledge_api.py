import asyncio

from fastapi import APIRouter, HTTPException

from services.assistant_service import AssistantService


knowledge_api = APIRouter()
assistant_service = AssistantService()


def _get_current_database_engine():
    """
    获取当前助手的知识库引擎。
    Returns:
        DataBase: 知识库引擎实例
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise HTTPException(status_code=400, detail="当前没有加载助手")
    if not hasattr(agent, "databaseEngine") or agent.databaseEngine is None:
        raise HTTPException(status_code=500, detail="知识库引擎未初始化")
    return agent.databaseEngine


@knowledge_api.get("/knowledge/status")
async def knowledge_status():
    """
    获取知识库状态。
    """
    engine = _get_current_database_engine()
    result = await asyncio.to_thread(engine.get_status)
    return {"msg": "ok", "data": result}


@knowledge_api.post("/knowledge/rebuild")
async def knowledge_rebuild():
    """
    手动触发知识库重建。
    """
    engine = _get_current_database_engine()
    result = await asyncio.to_thread(engine.rebuild, False)
    return {"msg": "ok", "data": result}


@knowledge_api.post("/knowledge/health-check")
async def knowledge_health_check(auto_fix: bool = True):
    """
    执行知识库健康检查。
    Parameters:
        auto_fix (bool): 是否自动修复
    """
    engine = _get_current_database_engine()
    result = await asyncio.to_thread(engine.run_health_check, auto_fix)
    return {"msg": "ok", "data": result}
