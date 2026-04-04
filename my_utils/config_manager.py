from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


# 读取配置文件
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f)


def recursive_update(parent, key_or_index, original, new):
    """
    递归更新原配置数据（通过父对象引用正确修改值）
    :param parent: 父对象（CommentedMap或CommentedSeq）
    :param key_or_index: 键名（字典）或索引（列表）
    :param original: 原配置中的值
    :param new: 客户端JSON中的新值
    """
    # 1. 处理字典/映射类型
    if isinstance(original, CommentedMap) and isinstance(new, dict):
        for k, v in new.items():
            if k in original:
                # 递归更新原配置中已存在的键
                recursive_update(original, k, original[k], v)
            else:
                # 新增原配置中不存在的键
                original[k] = v
        return

    # 2. 处理列表/序列类型
    if isinstance(original, CommentedSeq) and isinstance(new, list):
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
    if isinstance(config, CommentedMap) and isinstance(client_json, dict):
        for key, value in client_json.items():
            if key in config:
                recursive_update(config, key, config[key], value)
            else:
                config[key] = value

    # 写回文件
    with open("./config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
