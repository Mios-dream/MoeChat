import tomllib
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_version():
    """
    获取项目的版本号，默认从 pyproject.toml 中读取。
    使用 lru_cache 缓存结果
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    try:
        with open(pyproject_path, "rb") as f:  # 注意这里是二进制模式
            config = tomllib.load(f)
        _cache_version = config.get("project", {}).get("version", "unknown")
        return _cache_version
    except Exception:
        return "unknown"


if __name__ == "__main__":
    print(get_project_version())
