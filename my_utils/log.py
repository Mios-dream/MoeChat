import logging
import colorlog

# 创建格式化器
formatter = colorlog.ColoredFormatter(
    "[%(asctime)s]%(log_color)s%(levelname)s %(filename)s:%(lineno)d: %(message)s",
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 创建handler
handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

# 获取logger并配置
logger = colorlog.getLogger("colored_logger")
logger.handlers.clear()  # 清除已存在的handlers
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # 阻止传播到父logger
