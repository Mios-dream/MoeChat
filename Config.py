import os


class Config:
    """
    配置类，用于存储全局配置，不需要修改这个
    """

    # 基础数据路径
    BASE_DATA_PATH = "data"

    # 助手目录基础路径
    BASE_AGENTS_PATH = "data/agents"

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # nltk数据路径
    NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "models", "nltk_data")
    # FFMPEG路径
    FFMPEG_BIN = os.path.join(PROJECT_ROOT, "data", "models", "ffmpeg", "bin")

    GSV_MODELS_PATH = os.path.join(PROJECT_ROOT, "data", "models", "gsv")


os.environ["NLTK_DATA"] = Config.NLTK_DATA_DIR
os.environ["PATH"] = Config.FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
