import asyncio
from init_server import init
from web.src.router.router import app
import uvicorn


def start_server():
    # 等待初始化完成
    asyncio.run(init())
    # 启动web服务，应该在最后
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    start_server()
