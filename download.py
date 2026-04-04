# 下载模型到指定文件夹
from modelscope import snapshot_download
from Config import Config


def download_embedding_model():
    """
    下载句向量模型到本地指定路径。
    """
    snapshot_download(
        model_id="iic/nlp_gte_sentence-embedding_chinese-base",
        local_dir=Config.PROJECT_ROOT + "data" + "models",
    )


def download_asr_model():
    """
    下载语音识别模型到本地指定路径。
    """
    snapshot_download(
        model_id="iic/SenseVoiceSmall",
        local_dir=Config.PROJECT_ROOT + "data" + "models",
    )


def download_kws_model():
    """
    下载关键词检测模型。需要手动下载并填写到Config.py中。
    见https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
    """
    pass


def main():
    download_embedding_model()
    download_asr_model()
    download_kws_model()
