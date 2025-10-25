import asyncio
from init_server import init
from web.src.router.router import app
import uvicorn
from threading import Thread
from api.api.socket_api import start_socket_server


def start_server():
    # 等待初始化完成
    asyncio.run(init())
    # 启动socket服务
    Thread(target=start_socket_server, args=("0.0.0.0", 8002), daemon=True).start()
    # 启动web服务，应该在最后
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    start_server()
