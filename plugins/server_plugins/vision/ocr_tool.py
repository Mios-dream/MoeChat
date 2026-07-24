"""
桌面 OCR 识别工具

服务端同步工具示例：对用户屏幕指定区域进行 OCR 文字识别。

ExecDomain: SERVER
ExecMode:   SYNC
Tags:       vision, ai
Sensitivity: SAFE

使用场景:
    - "帮我看下屏幕上有什么"
    - "识别一下当前窗口的文字内容"
    - "给这个截图做文字识别"

注意事项:
    1. 此工具依赖 OCR 引擎（如 PaddleOCR / Tesseract），
       需预先安装并配置。
    2. 截图功能需要服务端有桌面访问权限。
    3. 返回的文本会完整传给 LLM，大段文字注意 token 限制。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tool_system.core.base import ServerTool
from tool_system.core.enums import (
    ExecutionDomain,
    ExecutionMode,
    ToolSensitivity,
)
from tool_system.core.registry import register_tool
from tool_system.core.types import ToolMeta


@register_tool(
    domain=ExecutionDomain.SERVER,
    mode=ExecutionMode.SYNC,
    timeout=120.0,
    sensitivity=ToolSensitivity.SAFE,
    tags=["vision", "ai"],
    version="1.0.0",
)
class DesktopOcrTool(ServerTool):
    """
    桌面 OCR 识别工具

    对用户当前屏幕或指定窗口进行光学字符识别（OCR），
    返回识别到的文字内容。LLM 可根据返回的文字内容进行
    分析和回答用户问题。

    当前版本为示例实现，使用模拟数据。
    实际部署时需集成真实的 OCR 引擎。
    """

    """工具名称：LLM 通过 'desktop_ocr' 调用此工具"""
    name: str = "desktop_ocr"

    """工具描述：告诉 LLM 在何时以及如何使用此工具"""
    description: str = (
        "对用户当前屏幕进行 OCR（光学字符识别）文字识别。"
        "当用户询问屏幕上有什么内容、需要识别屏幕中的文字、"
        "或询问当前活动窗口的内容时使用此工具。"
        "支持全屏识别和仅限当前活动窗口识别两种模式。"
    )

    """
    JSON Schema 参数定义（OpenAI function calling 兼容格式）

    region:  截屏区域选择
    language: OCR 识别语言
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "enum": ["full", "active_window"],
                "description": (
                    "识别区域："
                    "'full' - 识别整个屏幕内容；"
                    "'active_window' - 仅识别当前活动窗口的内容"
                ),
                "default": "full",
            },
            "language": {
                "type": "string",
                "enum": ["auto", "zh", "en", "ja"],
                "description": (
                    "识别语言："
                    "'auto' - 自动检测语言；"
                    "'zh' - 简体中文；"
                    "'en' - 英文；"
                    "'ja' - 日文"
                ),
                "default": "auto",
            },
        },
    }

    @property
    def meta(self) -> ToolMeta:
        """获取工具元信息（由 @register_tool 装饰器自动注入 _tool_meta）"""
        return self._tool_meta  # type: ignore[attr-defined]

    async def execute(self, **kwargs: Any) -> str:
        """
        执行 OCR 识别

        流程:
        1. 根据 region 参数截取对应区域的屏幕图像
        2. 调用 OCR 引擎进行文字识别
        3. 返回结构化的识别结果 JSON

        参数通过 **kwargs 从 LLM JSON 参数中提取，
        execute() 被调用前参数已通过 validate_arguments() 校验并填充默认值。

        支持的 kwargs:
            region: 识别区域（'full' 或 'active_window'）
            language: 识别语言（'auto' / 'zh' / 'en' / 'ja'）

        Returns:
            JSON 格式字符串
        """
        region = str(kwargs.get("region", "full"))
        language = str(kwargs.get("language", "auto"))

        # ── 截取屏幕 ──
        screenshot_text = await self._capture_screen(region)

        # ── OCR 识别 ──
        recognized_text = await self._ocr_recognize(screenshot_text, language)

        # ── 返回结构化的结果 ──
        return self.result_json(
            {
                "success": True,
                "text": recognized_text,
                "region": region,
                "language": language,
                "word_count": len(recognized_text),
            }
        )

    async def _capture_screen(self, region: str) -> str:
        """
        截取屏幕指定区域的文字内容

        当前为示例实现，返回模拟截图结果。
        实际部署时替换为 PIL.ImageGrab 或系统截图 API。

        Args:
            region: 识别区域标识

        Returns:
            截取到的文字内容（实际应为图像数据）
        """
        # ── 示例: 模拟截图返回 ──
        region_labels = {
            "full": "全屏",
            "active_window": "当前活动窗口",
        }
        label = region_labels.get(region, "未知区域")

        # 实际部署代码:
        # import io
        # from PIL import ImageGrab
        # img = ImageGrab.grab()  # 全屏截图
        # buffer = io.BytesIO()
        # img.save(buffer, format="PNG")
        # return buffer.getvalue()

        return f"[{label}截图: 示例文字内容 - 这是 OCR 识别出的模拟文本内容]"

    async def _ocr_recognize(self, image_text: str, language: str) -> str:
        """
        对图像内容进行 OCR 文字识别

        当前为示例实现，返回模拟识别结果。
        实际部署时替换为 PaddleOCR / Tesseract 等 OCR 引擎。

        Args:
            image_text: 图像内容的文字表示（实际应为二进制图像数据）
            language: 识别语言

        Returns:
            识别出的文字内容
        """
        # ── 示例: 模拟 OCR 返回 ──
        # 实际部署代码:
        # from paddleocr import PaddleOCR
        # import numpy as np
        # ocr_engine = PaddleOCR(lang=language, use_angle_cls=True)
        # results = ocr_engine.ocr(np.array(image), cls=True)
        # lines = [line[1][0] for line in results[0]]
        # return "\n".join(lines)
        await asyncio.sleep(5)  # 模拟 OCR 处理时间
        raise NotImplementedError("请部署实际 OCR 引擎以获取真实识别结果。")

        return (
            f"[OCR 测试结果] 检测到 {language} 语言文字内容。\n"
            f"示例识别文本行 1: 这是桌面上的一个文档窗口\n"
            f"示例识别文本行 2: 内容包含各种文字信息\n"
            f"---\n"
            f"提示: 请部署实际 OCR 引擎以获取真实识别结果。"
        )
