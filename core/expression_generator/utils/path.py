from pathlib import Path
from my_utils.log import logger as Log


def resolve_assets_root(assistant_name: str) -> Path | None:
    """
    解析当前角色的 Live2D 资源根目录

    路径策略：
    - 返回 data/agents/{assistant_name}/assets/live2d 目录

    参数：
    - assistant_name: 角色名称

    返回：
    - 目录存在时返回目录路径
    - 目录不存在返回 None
    """
    assets_root = Path("data") / "agents" / assistant_name / "assets" / "live2d"
    if not assets_root.exists():
        Log.warning(f"[表情加载] 模型目录不存在: {assets_root}")
        return None

    return assets_root
