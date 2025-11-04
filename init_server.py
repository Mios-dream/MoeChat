import sys
import traceback
from utils import config as CConfig
from utils.socket_asr import ASRServer
from utils.log import logger
import httpx
import os
from utils.logo import print_moechat_logo


async def init():
    """
    系统初始化
    所以的初始化操作就将在这里处理
    """
    try:
        print_moechat_logo()

        await create_data_folder()
        # await init_gptsovits()
        await load_asr_model()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        logger.error(f"初始化失败")
        logger.error(
            f"文件路径: {tb[-1].filename} \n行号：{tb[-1].lineno} \n错误源码:{traceback.format_exc()}\n错误信息为: {e}",
        )
        exit()


async def create_data_folder():
    # 创建数据文件夹
    os.path.exists("data") or os.mkdir("data")  # type: ignore
    os.path.exists("data/agents") or os.mkdir("data/agents")  # type: ignore
    os.path.exists("data/agents/Chat酱") or os.mkdir("data/agents/Chat酱")  # type: ignore


async def load_asr_model():
    """
    加载asr模型
    """
    ASRServer().load_model()


async def init_gptsovits():
    """
    对gptsovits进行初始化
    """
    t2s_weights = CConfig.config["GSV"]["GPT_weight"]
    vits_weights = CConfig.config["GSV"]["SoVITS_weight"]
    if t2s_weights:
        logger.info(f"设置GPT_weights...")
        params = {"weights_path": t2s_weights}
        try:
            httpx.get(
                str(CConfig.config["GSV"]["api"]).replace("/tts", "/set_gpt_weights"),
                params=params,
            )
        except:
            raise TimeoutError(f"设置GPT_weights失败")
    if vits_weights:
        logger.info(f"设置SoVITS...")
        params = {"weights_path": vits_weights}
        try:
            httpx.get(
                str(CConfig.config["GSV"]["api"]).replace(
                    "/tts", "/set_sovits_weights"
                ),
                params=params,
            )
        except:
            raise TimeoutError(f"设置SoVITS失败")
