from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from Config import Config


# 加载embedding模型
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL_PATH)
    model = AutoModel.from_pretrained(Config.EMBEDDING_MODEL_PATH)
    return {"tokenizer": tokenizer, "model": model}


embedding_model = load_model()


def t2vect(text: list[str]) -> np.ndarray:
    model_data = embedding_model
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    # 对输入文本进行编码
    inputs = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=100
    )
    # 获取 embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
    return embeddings.numpy()
