import os


class Config:
    """
    配置类，用于存储全局配置，不需要修改这个
    """

    # 基础数据路径
    BASE_DATA_PATH = "data"

    # 助手目录基础路径
    BASE_AGENTS_PATH = "data/agents"
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # embedding模型路径
    EMBEDDING_MODEL_PATH = os.path.join(
        PROJECT_ROOT, "data", "models", "nlp_gte_sentence-embedding_chinese-base"
    )
    # nltk数据路径
    NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "models", "nltk_data")
    # GSV模型路径,设置为None以自动下载
    GSV_MODELS_PATH = os.path.join(PROJECT_ROOT, "data", "models", "gsv")
    # ASR模型路径
    ASR_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models", "SenseVoiceSmall")
    # 唤醒词模型基础路径
    WAKEWORD_MODEL_DIR = os.path.join(
        PROJECT_ROOT, "data", "models", "sherpa-onnx-kws-zipformer-zh-en-3M"
    )
    WAKEWORD_MODELS = {
        "tokens": os.path.join(WAKEWORD_MODEL_DIR, "tokens.txt"),
        "encoder": os.path.join(
            WAKEWORD_MODEL_DIR, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
        ),
        "decoder": os.path.join(
            WAKEWORD_MODEL_DIR, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
        ),
        "joiner": os.path.join(
            WAKEWORD_MODEL_DIR, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
        ),
        "keywords_file": os.path.join(WAKEWORD_MODEL_DIR, "keywords.txt"),
        "lexicon": os.path.join(WAKEWORD_MODEL_DIR, "en.phone"),
        "tokens_type": "phone+ppinyin",
    }


# 设置环境变量,使用项目目录下的模型数据,而不是下载模型到默认位置，取消以下载到默认位置
os.environ["NLTK_DATA"] = Config.NLTK_DATA_DIR
