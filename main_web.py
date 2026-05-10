from Config import Config
import asyncio
import sys
from init_server import init
from router.router import app
import uvicorn


def start_server():
    """
    启动主服务器
    """
    # 等待初始化完成
    asyncio.get_event_loop().run_until_complete(init())
    # 启动web服务，应该在最后
    uvicorn.run(app, host="0.0.0.0", port=8001)


def check_update_only():
    """
    仅执行更新检查，不启动服务器
    """
    import update

    update.check_and_update()


if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--check-update", "-u"):
            check_update_only()
            sys.exit(0)
        elif arg == "--update-only":
            check_update_only()
            sys.exit(0)

    start_server()
