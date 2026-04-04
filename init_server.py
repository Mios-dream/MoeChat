import sys
import traceback
from my_utils.log import logger
import os
from my_utils.logo import print_moechat_logo
from my_utils.version import get_project_version
from services.assistant_service import AssistantService


assistant_service = AssistantService()


async def init():
    """
    系统初始化
    所有的初始化操作都将在这里处理
    """
    try:
        print_moechat_logo()
        logger.info(f"当前版本为: {get_project_version()}")

        await create_data_folder()
        await initialize_assistant()

    except Exception as e:
        _exc_type, _exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        logger.error(f"初始化失败")
        logger.error(
            f"文件路径: {tb[-1].filename} \n行号：{tb[-1].lineno} \n错误源码:{traceback.format_exc()}\n错误信息为: {e}",
        )
        exit()


async def create_data_folder():
    """
    创建数据文件夹
    确保数据文件夹和子文件夹存在
    """
    # 创建数据文件夹
    os.path.exists("data") or os.mkdir("data")  # type: ignore
    os.path.exists("data/agents") or os.mkdir("data/agents")  # type: ignore


async def initialize_assistant():
    """
    初始化助手服务
    加载助手信息并初始化默认助手
    """
    logger.info("开始初始化助手服务...")

    # 加载所有助手信息
    assistants = assistant_service.load_assistant_info()
    logger.info(f"已加载 {len(assistants)} 个助手")

    # 初始化默认助手
    agent = await assistant_service.initialize_default_assistant()
    if agent:
        logger.info(f"成功初始化默认助手: {agent.agent_name}")
    else:
        logger.warning("未找到可用的助手，请创建一个助手后再使用")
