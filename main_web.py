import asyncio
from init_server import init
import uvicorn


def start_server():
    """
    启动主服务器
    """
    # 等待初始化完成
    asyncio.get_event_loop().run_until_complete(init())
    # 路由导入链会加载 embedding 模型，因此必须在首次启动的模型下载完成后导入。
    from router.router import app

    # 启动web服务，应该在最后
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    start_server()
