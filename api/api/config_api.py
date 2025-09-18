from utils import config as CConfig
from fastapi import (
    APIRouter,
)


config_api = APIRouter()


# 客户端获取配置信息
@config_api.post("/get_config")
async def get_config():
    return CConfig.config


# 更新配置文件
# @config_api.post("/update_config")
# async def update_config(data: dict):
#     global agent

#     CConfig.update_config(data)
#     if CConfig.config["Agent"]["is_up"]:
#         agent.update_config()
