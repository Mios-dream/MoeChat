"""
多模态内容处理器

将 ChatRequest（text + files）转换为 LLM 管道所需的 user_message 格式。
根据 enable_multimodal 配置和文件类型自动选择处理方式：

- 图片文件（.png/.jpg/…）：多模态 → image_url content part，否则 → PaddleOCR
- 文本文件（.txt/…）：读取文本内容

所有类型：text 和文件内容各自作为独立 text part，前端通过 content parts list 区分
"""

import base64
from io import BytesIO
from typing import Any
import numpy
from models.dto.request.chat_request import ChatRequest, FileType
from my_utils import config_manager as CConfig
from my_utils.log import logger
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from paddleocr import PaddleOCR

# PaddleOCR 全局单例
_ocr_instance: PaddleOCR | None = None


def _get_ocr_instance() -> PaddleOCR:
    """获取或创建 PaddleOCR 实例（懒加载单例）"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ch",
            engine="onnxruntime",
            device="cpu",
        )
        logger.info("[多模态处理器] PaddleOCR 实例已创建(onnxruntime+cpu)")
    return _ocr_instance


def _extract_text_and_score(
    result: list[dict[str, Any]],
) -> tuple[list[str], float]:
    """从 PaddleOCR 输出中提取文本，过滤低置信度结果"""
    texts: list[str] = []
    scores: list[float] = []
    threshold = 0.9

    for res in result or []:
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])

        for idx, text in enumerate(rec_texts):
            cleaned = (text or "").strip()
            if not cleaned:
                continue

            score = rec_scores[idx] if idx < len(rec_scores) else 0.0
            if isinstance(score, (int, float)):
                score_value = float(score)
                if score_value < threshold or len(cleaned) <= 4:
                    continue
                texts.append(cleaned)
                scores.append(score_value)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return texts, avg_score


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """对图片字节数据进行 OCR 识别（使用 PaddleOCR）"""
    try:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image_array = numpy.array(image)
        ocr = _get_ocr_instance()
        result = ocr.predict(image_array)
        texts, _ = _extract_text_and_score(result)
        combined = "\n".join(texts) if texts else ""
        if combined:
            logger.info(f"[多模态处理器] PaddleOCR 识别成功: {len(combined)} 字符")
        return combined
    except ImportError:
        logger.warning("paddleocr 未安装，请执行: pip install paddleocr")
        return "[OCR 引擎未安装]"
    except Exception as e:
        logger.error(f"[多模态处理器] PaddleOCR 识别失败: {e}")
        return "[图片 OCR 失败]"


def _decode_base64_to_bytes(data: str) -> bytes:
    """将 Base64 字符串解码为字节（自动跳过 data URI 前缀）"""
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _read_txt_content(data: str) -> str:
    """解码 Base64 txt 文件"""
    try:
        raw = _decode_base64_to_bytes(data)
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            raw = _decode_base64_to_bytes(data)
            return raw.decode("gbk").strip()
        except Exception:
            return "[txt 编码不支持]"
    except Exception:
        return "[txt 读取失败]"


def build_user_message_content(
    request: ChatRequest,
    model_key: str = "ChatLLM",
) -> tuple[list[ChatCompletionMessageParam], str]:
    """
    从 ChatRequest 构建用户消息内容。

    返回 (user_message_content, user_text)：
    - user_message_content: list[ChatCompletionMessageParam]，OpenAI 原生格式
      始终为 content parts list，方便前端区分用户文本和附件内容
    - user_text: 用户原始输入的纯文本（不含附件内容），供存档和检索
    """

    config = CConfig.config.get(model_key, {})
    enable_multimodal = bool(config.get("enable_multimodal", False))

    text_parts: list[ChatCompletionContentPartTextParam] = []
    ocr_texts: list[str] = []
    image_parts: list[ChatCompletionContentPartImageParam] = []
    ocr_count = 0

    user_message_content: list[ChatCompletionMessageParam] = []
    user_text = request.text or ""

    if request.text:
        text_parts.append({"type": "text", "text": request.text})

    # 统一处理所有附件：按文件类型自动分流
    for file in request.files:
        if file.type == FileType.IMAGE:
            if enable_multimodal:
                # 多模态模式：直接透传 image_url（前端应已包含 data: URI 前缀）
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": file.data},
                    }
                )
                logger.info(
                    f"[多模态处理器] 多模态模式: image_url({file.name or '未命名'})"
                )
            else:
                # 非多模态模式：OCR 识别
                ocr_count += 1
                try:
                    image_bytes = _decode_base64_to_bytes(file.data)
                    ocr_text = _ocr_image_bytes(image_bytes)
                except Exception as e:
                    logger.error(f"[多模态处理器] 图片解码失败({file.name}): {e}")
                    ocr_text = ""
                if ocr_text:
                    name_tag = f"({file.name}) " if file.name else ""
                    ocr_texts.append(
                        f"[图片 {name_tag}#{ocr_count} 中的文字]: {ocr_text}"
                    )
                logger.info(
                    f"[多模态处理器] OCR 模式: 识别图片 #{ocr_count}({file.name})"
                )
        else:
            # 非图片文件：当作文本读取
            text = _read_txt_content(file.data)
            if text:
                label = f"[文件 {file.name or '未命名'} 内容]"
                ocr_texts.append(f"{label}:\n{text}")

    # 组装 user_message_content（始终为 content parts list）
    content_parts: list[ChatCompletionContentPartParam] = []
    if enable_multimodal and image_parts:
        content_parts.extend(text_parts)
        content_parts.extend(image_parts)
    else:
        content_parts.extend(text_parts)
        for ocr_text in ocr_texts:
            content_parts.append({"type": "text", "text": ocr_text})

    user_message_content.append(
        {
            "role": "user",
            "content": content_parts or [{"type": "text", "text": ""}],
        }
    )

    return user_message_content, user_text
