from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from Config import Config


def _get_device() -> torch.device:
    """自动选择可用设备，优先 CUDA"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL_PATH)
    model = AutoModel.from_pretrained(Config.EMBEDDING_MODEL_PATH)

    # CPU 下 PyTorch 2.0+ SDP 仅支持 math 内核
    if device.type == "cpu":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    model = model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


embedding_model = load_model()


def t2vect(text: list[str]) -> np.ndarray:
    model_data = embedding_model
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    device = model_data["device"]

    inputs = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=100
    )
    # 将输入移到模型所在设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()
