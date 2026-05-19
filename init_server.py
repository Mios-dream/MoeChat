import traceback
import sys

from my_utils.check_and_download_default_assistant import (
    check_and_download_default_assistant,
)
from my_utils.log import logger
import os
from my_utils.logo import print_moechat_logo
from my_utils.version import get_project_version
from my_utils.tool_manager import ToolManager
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
        await check_and_download_default_assistant()
        await initialize_assistant()
        await initialize_tools()

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
        exit(1)


async def initialize_tools():
    """
    初始化工具/技能系统
    扫描 plugins/ 目录并加载所有技能
    """
    from my_utils import config_manager as CConfig

    # 检查工具系统是否启用
    tools_config = CConfig.config.get("Tools", {})
    if not tools_config.get("enabled", True):
        logger.info("工具系统已禁用，跳过初始化")
        return

    logger.info("开始初始化工具/技能系统...")

    # 自动发现并加载插件
    if tools_config.get("auto_discover", True):
        loaded_count = ToolManager.load_plugins("plugins")
        logger.info(f"工具系统初始化完成，共加载 {loaded_count} 个工具")
    else:
        logger.info("自动发现已禁用，跳过插件扫描")
