"""
资源库服务

提供默认资源库的管理能力，按 resource.json 描述文件扫描预设资源。
预留未来资源类型的扩展点（表情、动作等）。

目录结构：
data/resources/
    <预设名称>/
        resource.json    # 资源描述文件（JSON 格式）
        assets/          # 资源文件目录
            models/      # GPT/SoVITS 模型文件
            audio/       # 参考音频文件

resource.json 格式：
    {
        "name": "default",              # 预设名称
        "type": "gsv",                  # 资源类型，当前支持 "gsv"
        "description": "默认中文语音模型",  # 描述
        "files": {                       # 资源文件映射，路径相对于 assets/
            "gptModel": "models/gpt.ckpt",
            "sovitsModel": "models/sovits.pth",
            "refAudio": "audio/ref_audio.wav"
        },
        "config": {                      # 配置参数
            "textLang": "zh",
            "promptText": "你好，我是你的助手。",
            "promptLang": "zh"
        }
    }
"""

import json
import os
from Config import Config
import my_utils.log as Log


class ResourceService:
    """
    资源库服务，单例模式
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._presets_cache: dict[str, dict] | None = None

    def _scan_presets(self) -> dict[str, dict]:
        """
        扫描所有预设资源库目录，通过 resource.json 发现预设。
        返回 {预设名称: 预设信息} 字典。
        """
        presets_path = Config.RESOURCES_PATH
        presets: dict[str, dict] = {}

        if not os.path.isdir(presets_path):
            return presets

        for name in os.listdir(presets_path):
            preset_dir = os.path.join(presets_path, name)
            if not os.path.isdir(preset_dir):
                continue

            resource_file = os.path.join(preset_dir, "resource.json")
            if not os.path.isfile(resource_file):
                continue

            try:
                with open(resource_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                presets[name] = {
                    "name": data.get("name", name),
                    "type": data.get("type", "gsv"),
                    "description": data.get("description", ""),
                    "files": data.get("files", {}),
                    "config": data.get("config", {}),
                }
            except Exception as e:
                Log.logger.warning(f"解析预设 [{name}] resource.json 失败: {e}")

        return presets

    def list_presets(self) -> list[dict]:
        """
        列出所有可用预设资源

        返回：
            预设信息列表，每个元素含 name, type, description, files, config
        """
        if self._presets_cache is None:
            self._presets_cache = self._scan_presets()
        return list(self._presets_cache.values())

    def get_preset(self, name: str) -> dict | None:
        """
        获取指定预设资源信息

        参数：
        - name: 预设名称

        返回：
            预设信息字典，不存在返回 None
        """
        if self._presets_cache is None:
            self._presets_cache = self._scan_presets()
        return self._presets_cache.get(name)

    def get_preset_assets_path(self, name: str) -> str:
        """
        获取预设的资源文件目录路径

        参数：
        - name: 预设名称

        返回：
            assets 目录的绝对路径
        """
        return os.path.join(Config.RESOURCES_PATH, name, "assets")

    def refresh_cache(self):
        """
        刷新预设缓存，供外部在预设变更后调用
        """
        self._presets_cache = None


resource_service = ResourceService()
