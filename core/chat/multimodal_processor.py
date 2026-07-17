"""
多模态内容处理器

将 ChatRequest（text + images + files）转换为 LLM 管道所需的 user_message 格式。
根据 enable_multimodal 配置决定处理方式：

- 启用多模态：images 转为 image_url content part，text 原样保留
- 未启用多模态：images OCR 识别，files 读取文本，与原 text 合并

返回 (user_message_content, user_text, attachment_messages) 三元组：
- user_message_content: list[ChatCompletionMessageParam]，供管道使用（含附件信息）
- user_text: 用户原始输入的纯文本，不含 OCR/文件内容，供存档和检索
- attachment_messages: list[dict]，附件解析结果的 system 消息，供追加到 chat_history
"""

import base64
from io import BytesIO
from models.dto.request.chat_request import ChatRequest
from my_utils import config_manager as CConfig
from my_utils.log import logger
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """对图片字节数据进行 OCR 识别"""
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(image_bytes))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        result = text.strip()
        if result:
            logger.info(f"[多模态处理器] OCR 识别成功: {len(result)} 字符")
        return result
    except ImportError:
        logger.warning("pytesseract 未安装，请执行: pip install pytesseract")
        return "[OCR 引擎未安装]"
    except FileNotFoundError:
        logger.warning(
            "Tesseract-OCR 未部署，请从 https://github.com/UB-Mannheim/tesseract/wiki 下载"
        )
        return "[OCR 引擎未部署]"
    except Exception as e:
        logger.error(f"[多模态处理器] OCR 失败: {e}")
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
      多模态：{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
      非多模态：{"role": "user", "content": "文本"}
    - user_text: 用户原始输入的纯文本（不含 OCR/文件内容），供存档和检索
    """

    config = CConfig.config.get(model_key, {})
    enable_multimodal = bool(config.get("enable_multimodal", False))

    text_parts: list[ChatCompletionContentPartTextParam] = []
    ocr_texts: list[str] = []
    image_parts: list[ChatCompletionContentPartImageParam] = []
    ocr_count = 0

    user_message_content: list[ChatCompletionMessageParam] = []
    # 用户原始输入，不含附件解析内容
    user_text = request.text or ""

    # 如果用户有输入文本，先加入
    if request.text:
        text_parts.append({"type": "text", "text": request.text})

    # 处理图片
    for img_data in request.images:
        if not img_data:
            continue
        # 补全 data URI 前缀
        if not img_data.startswith("data:"):
            img_data = f"data:image/png;base64,{img_data}"

        if enable_multimodal:
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": img_data},
                }
            )
            logger.info("[多模态处理器] 多模态模式: 添加 image_url part")
        else:
            ocr_count += 1
            try:
                image_bytes = _decode_base64_to_bytes(img_data)
                ocr_text = _ocr_image_bytes(image_bytes)
            except Exception as e:
                logger.error(f"[多模态处理器] 图片解码失败: {e}")
                ocr_text = ""
            if ocr_text:
                ocr_texts.append(f"[图片 {ocr_count} 中的文字]: {ocr_text}")
            logger.info(f"[多模态处理器] OCR 模式: 识别图片 #{ocr_count}")

    # 处理文件（仅 txt）
    for file in request.files:
        text = _read_txt_content(file.data)
        if text:
            label = f"[文件 {file.name or '未命名'} 内容]"
            ocr_texts.append(f"{label}:\n{text}")

    # 组装 user_message_content（OpenAI 原生格式）
    if enable_multimodal and image_parts:
        content_parts: list[ChatCompletionMessageParam] = []
        if text_parts:
            content_parts.extend(text_parts)
        content_parts.extend(image_parts)
        user_message_content.append({"role": "user", "content": content_parts})
    else:
        # 非多模态：合并文本 + OCR + 文件内容
        combined_parts = []
        if request.text:
            combined_parts.append(request.text)
        if ocr_texts:
            combined_parts.extend(ocr_texts)
        combined_text = "\n".join(combined_parts) or ""
        user_message_content.append({"role": "user", "content": combined_text})

    return user_message_content, user_text
