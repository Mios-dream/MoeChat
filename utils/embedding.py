from transformers import AutoTokenizer, AutoModel
import torch
from modelscope import snapshot_download
import numpy as np
from utils.log import logger as Log

import os


# 禁用ModelScope的自动依赖安装
os.environ["MODELSCOPE_DISABLE_AUTO_INSTALL"] = "true"


# 加载embedding模型
def load_model():
    model_path = "./data/models/nlp_gte_sentence-embedding_chinese-base"
    if not os.path.exists(model_path):
        Log.warning(f"embedding模型未安装，开始安装embedding模型...")
        model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
        snapshot_download(model_id=model_id, local_dir=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return {
        'tokenizer': tokenizer,
        'model': model
    }


embedding_model = load_model()


def t2vect(text: list[str]) -> np.ndarray:
    model_data = embedding_model
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    # 对输入文本进行编码
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=100)
    # 获取 embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
    return embeddings.numpy()

