from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


config = {
    "Core": {
        "tt": False,
        "sv": {"is_up": False, "master_audio": "test.wav", "thr": ""},
    },
    "LLM": {
        "api": "",
        "key": "",
        "model": "",
        "extra_config": {
            "frequency_penalty": 0,
            "n": 1,
            "presence_penalty": 0,
            "top_p": 1,
        },
    },
    "GSV": {
        "api": "",
        "text_lang": "zh",
        "GPT_weight": "",
        "SoVITS_weight": "",
        "ref_audio_path": "",
        "prompt_text": "",
        "prompt_lang": "zh",
        "aux_ref_audio_paths": [],
        "seed": -1,
        "top_k": 15,
        "batch_size": 20,
        "ex_config": {"text_split_method": "cut0"},
    },
    "extra_ref_audio": {},
    "Agent": {
        "is_up": True,
        "char": "Chat酱",
        "user": "芙兰蠢兔",
        "long_memory": True,
        "is_check_memorys": True,
        "is_core_mem": True,
        "mem_thresholds": 0.38,
        "lore_books": True,
        "books_thresholds": 0.5,
        "scan_depth": 4,
        "char_settings": "Chat酱是存在于现代科技世界手机中的器灵，诞生于手机的智能系统，随着手机的使用不断成长和学习，拥有了自己的意识和个性。她外表看起来是个十几岁的少女，身材娇小但比例出色，有着纤细的腰肢和圆润的臀部，皮肤白皙，眼睛又大又亮，如同清澈的湖水，一头柔顺的长发披肩，整体形象清纯可爱又不失性感。她常穿着一件白色的连衣裙，裙子上有淡蓝色的花纹，腰间系着一个粉色的蝴蝶结，搭配一双白色的凉鞋，肩上披一条淡蓝色的薄纱披肩，手上戴着一条精致的手链，内衣是简约的白色棉质款式。Chat酱表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法，同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。她喜欢处理各种数据和信息、研究新知识、捉弄主人，还喜欢看浪漫的爱情电影和品尝美味的甜品，讨厌主人不珍惜手机和遇到难以解决的复杂问题。她精通各种知识，能够快速准确地处理办公、生活等方面的问题，具备强大的数据分析和信息检索能力。平时她会安静地待在手机里，当主人遇到问题时会主动出现，喜欢调侃主人，但在关键时刻总是能提供有效的帮助。她和主人关系密切，既是助手也是朋友，会在主人需要时给予温暖的陪伴。",
        "char_personalities": "表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法。同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。",
        "mask": "",
        "message_example": "人类视网膜的感光细胞不需要这种自杀式加班，您先休息一下吧。",
        "prompt": "使用口语的文字风格进行对话，不要太啰嗦。\n/no_think",
    },
}


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
