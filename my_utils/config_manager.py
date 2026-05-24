import os
import shutil
import yaml
from my_utils.log import logger


def _init_config():
    """
    初始化配置文件：如果 config.yaml 不存在，则从 config.example.yaml 模板创建。
    """
    if not os.path.exists("config.yaml"):
        if os.path.exists("config.example.yaml"):
            shutil.copy2("config.example.yaml", "config.yaml")
            logger.warning(
                "未检测到 config.yaml，已从模板 config.example.yaml 创建，请修改配置后重新启动。"
            )
        else:
            raise FileNotFoundError(
                "config.example.yaml 模板文件不存在，无法创建 config.yaml。"
            )


_init_config()

# 读取配置文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def recursive_update(parent, key_or_index, original, new):
    """
    递归更新原配置数据（通过父对象引用正确修改值）
    :param parent: 父对象（dict或list）
    :param key_or_index: 键名（字典）或索引（列表）
    :param original: 原配置中的值
    :param new: 客户端JSON中的新值
    """
    # 1. 处理字典/映射类型
    if isinstance(original, dict) and isinstance(new, dict):
        for k, v in new.items():
            if k in original:
                # 递归更新原配置中已存在的键
                recursive_update(original, k, original[k], v)
            else:
                # 新增原配置中不存在的键
                original[k] = v
        return

    # 2. 处理列表/序列类型
    if isinstance(original, list) and isinstance(new, list):
        # 按索引匹配更新（假设客户端列表与原配置列表顺序、结构一致）
        min_len = min(len(original), len(new))
        for i in range(min_len):
            recursive_update(original, i, original[i], new[i])
        # 可选：若客户端列表更长，追加剩余元素
        # for i in range(min_len, len(new)):
        #     original.append(new[i])
        return

    # 3. 基本类型（字符串、数字、布尔值等）：通过父对象更新值
    parent[key_or_index] = new


# 完整更新流程
def update_config(client_json):
    global config
    # 处理根节点（根节点的父对象设为None，用特殊方式处理）
    if isinstance(config, dict) and isinstance(client_json, dict):
        for key, value in client_json.items():
            if key in config:
                recursive_update(config, key, config[key], value)
            else:
                config[key] = value

    # 写回文件
    with open("./config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
