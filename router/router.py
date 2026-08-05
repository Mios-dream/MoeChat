from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.api_router import api_router
from exceptions.error_handlers import setup_exception_handlers
from my_utils.log import logger
from my_utils.memory_cleanup import DEFAULT_CLEANUP_INTERVAL_SECONDS, start_periodic_cleanup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时创建后台周期内存清理任务，退出时取消"""
    # 后台周期内存清理（默认 20 分钟一次，随 uvicorn 事件循环运行）
    cleanup_task = start_periodic_cleanup(DEFAULT_CLEANUP_INTERVAL_SECONDS)
    logger.info("已启动后台周期内存清理任务（20 分钟）")
    yield
    cleanup_task.cancel()
    logger.info("后台周期内存清理任务已停止")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载各种路由
app.include_router(api_router)

# 设置全局异常处理
setup_exception_handlers(app)
