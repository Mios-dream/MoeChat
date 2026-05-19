import logging


class ColoredFormatter(logging.Formatter):
    """使用ANSI转义码实现彩色日志输出"""

    COLORS = {
        "DEBUG": "\033[34m",  # 蓝色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[1;31m",  # 粗体红
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


# 创建格式化器
formatter = ColoredFormatter(
    "[%(asctime)s]%(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 创建handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# 获取logger并配置
logger = logging.getLogger("colored_logger")
logger.handlers.clear()  # 清除已存在的handlers
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # 阻止传播到父logger
