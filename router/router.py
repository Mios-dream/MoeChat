from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.api_router import api_router
from exceptions.error_handlers import setup_exception_handlers

app = FastAPI()
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
