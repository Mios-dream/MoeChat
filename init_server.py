import asyncio
import traceback
import sys
from my_utils.log import logger
import os
from my_utils.logo import print_moechat_logo
from my_utils.version import get_project_version
from my_utils.memory_cleanup import cleanup
from tool_system.core.registry import get_registry as get_tool_registry
from download import check_and_download_models


async def init():
    """
    系统初始化
    所有的初始化操作都将在这里处理
    """
    try:
        print_moechat_logo()
        logger.info(f"当前版本为: {get_project_version()}")

        await create_data_folder()
        # 下载工作可能耗时较长，放在线程中执行，避免阻塞初始化事件循环。
        await asyncio.to_thread(check_and_download_models)
        await initialize_assistant()
        await initialize_tools()

        cleanup()

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
    # AssistantService 的导入链会加载 embedding 模型。必须在模型下载完成后再导入，
    # 否则首次启动会将尚不存在的本地目录误判为 Hugging Face repo id。
    from services.assistant_service import AssistantService

    assistant_service = AssistantService()

    # 加载所有助手信息
    assistants = assistant_service.load_assistant_info()
    logger.info(f"已加载 {len(assistants)} 个助手")

    # 初始化默认助手
    agent = await assistant_service.initialize_default_assistant()
    if agent:
        logger.info(f"成功初始化默认助手: {agent.agent_name}")
    else:
        logger.warning("未找到可用的助手，请创建一个助手后再使用")


async def initialize_tools():
    """
    初始化工具/技能系统
    扫描 plugins/ 目录并加载所有技能
    旧系统路径: plugins/ → ToolManager.load_plugins()
    新系统路径: tool_system/server_plugins/ 等 → ToolRegistry.discover()
    """
    from my_utils import config_manager as CConfig

    # 检查工具系统是否启用
    tools_config = CConfig.config.get("Tools", {})
    if not tools_config.get("enabled", True):
        logger.info("工具系统已禁用，跳过初始化")
        return

    logger.info("开始初始化工具/技能系统...")

    total_tools = 0

    # ── 新系统: 自动扫描 tool_system 插件目录 ──
    new_registry = get_tool_registry()
    scan_paths = [
        "plugins/server_plugins",
        "plugins/client_plugins",
    ]
    for scan_path in scan_paths:
        try:
            new_count = new_registry.discover(scan_path)
            if new_count > 0:
                logger.info(f"新工具系统: {scan_path} 加载了 {new_count} 个工具")
                total_tools += new_count
        except Exception as e:
            logger.warning(f"新工具系统扫描 {scan_path} 失败: {e}")

    logger.info(f"工具系统初始化完成，共加载 {total_tools} 个工具")
    logger.info(f"新工具系统状态: {new_registry}")
