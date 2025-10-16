# config.py

import yaml
from pathlib import Path


def _load_config():
    """
    加载 YAML 配置文件并返回其内容。
    使用 pathlib 来确保路径的健壮性。
    """
    # 定位配置文件
    config_path = Path(__file__).parent / "weather_config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            raise ValueError("配置文件为空或格式不正确。")
        return config_data
    except FileNotFoundError:
        # 无配置文件报错
        raise SystemExit(f"错误：配置文件未找到，路径: {config_path}")
    except yaml.YAMLError as e:
        # YAML格式错误报错
        raise SystemExit(f"错误：配置文件格式错误。\n{e}")


# 将配置项暴露为模块级别的常量

# 执行加载
_config = _load_config()

# API 相关配置
API = _config.get("api", {})
IP_SERVICE_URL = API.get("ip_service_url")
GEO_SERVICE_URL = API.get("geo_service_url")
TIMEOUT = API.get("timeout", 2.0)  # 默认值超时的值

# 数据源相关配置
SOURCES = _config.get("sources", {})

# 验证关键配置是否存在
if not all([IP_SERVICE_URL, GEO_SERVICE_URL, SOURCES]):
    raise SystemExit("错误：配置文件中缺少必要的 'api' 或 'sources' 配置项。")
