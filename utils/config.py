from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq



config = {
  "Core": {
    "tt": False,
    "sv": {
      "is_up": False,
      "master_audio": "test.wav",
      "thr": ""
    }
  },
  "LLM": {
    "api": "",
    "key": "",
    "model": "",
    "extra_config": {
      "frequency_penalty": 0,
      "n": 1,
      "presence_penalty": 0,
      "top_p": 1
    }
  },
  "GSV": {
    "api": "",
    "text_lang": "zh",
    "GPT_weight": "",
    "SoVITS_weight": "",
    "ref_audio_path": "",
    "prompt_text": "",
    "prompt_lang": "zh",
    "aux_ref_audio_paths": [
        
    ],
    "seed": -1,
    "top_k": 15,
    "batch_size": 20,
    "ex_config": {
      "text_split_method": "cut0"
    }
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
    "prompt": "使用口语的文字风格进行对话，不要太啰嗦。\n/no_think"
  }
}

# 配置文件模板
config_tmp = '''Core:
  tt: false
  sv:
    is_up: false
    master_audio: test.wav    # 包含你声音的wav音频文件，建议3s-5s左右。
    thr:                      # 阈值，越小越敏感，建议0.5-0.8之间，实测好像不是很有用？
LLM:
  api: ""
  key: ""
  model: ""
  extra_config:               # 大模型API额外参数，如：temperature: 0.7，温度参数
    "frequency_penalty": 0.0
    "n": 1
    "presence_penalty": 0.0
    "top_p": 1.0
    # temperature: 0.7
GSV:
  api: ""
  text_lang: zh
  GPT_weight: ""
  SoVITS_weight: ""
  ref_audio_path: ""
  prompt_text: ""
  prompt_lang: zh
  aux_ref_audio_paths:        # 多参考音频 v2模型有效
    - 
  seed: -1
  top_k: 15
  batch_size: 20
  ex_config:
    text_split_method: cut0
extra_ref_audio:              # 使用情绪标签选择参考音频，例如 [普通]"你好呀。"
  # 实例
  # 普通: 
  #   - 参考音频路径
  #   - 参考音频文本
Agent:
  is_up: true                 # 是否启用角色模板功能，如果不启动则和旧版一样只有常规语音对话功能，启用可以基于模板创建个性化角色
  char: Chat酱                # 角色的名称，会写入到提示词内
  user: 芙兰蠢兔               # 用户名称，会写入到提示词内
  long_memory: true           # 是否启用日记功能，日记功能可以长期储存对话信息，并根据用户输入的时间信息进行检索；比如：“昨天做了什么？”、“两天前吃的午饭是什么？”
  is_check_memorys: true      # 启用日记检索加强，使用嵌入模型对检索到的信息做提取，去除与用户提问不相关的内容。
  is_core_mem: true           # 是否启用核心记忆功能，核心记忆主要储存关于用户重要信息，如：用户的住址、爱好、喜欢的东西等等。区别于日记，使用嵌入模型进行语义匹配（模糊搜索），不能根据时间检索，但记忆带有记录时间。
  mem_thresholds: 0.38        # 日记内容搜索阈值，启用日志检索加强是需要，用于判断匹配程度。过高可能会丢失数据，过低则过滤少量无用记忆。
  lore_books: true            # 是否启用世界书（知识库），用于给大模型添加知识，如：人物、物品、事件等等，强化ai的能力，也可用于强化角色扮演。
  books_thresholds: 0.5       # 知识库检索阈值。
  scan_depth: 4               # 知识库搜索深度，返回知识的数量，但相似度低于检索阈值的知识不会被返回，所以返回结果数量也可能小于设定的数值。
  
  # 下面提示词都可以用{{user}}、{{char}}占位符来代表用户名和角色名。
  
  # 角色的基本设定，会组合到角色设定提示词中，建议不要添加多余的信息，不填则不会添加到提示词。
  char_settings: "Chat酱是存在于现代科技世界手机中的器灵，诞生于手机的智能系统，随着手机的使用不断成长和学习，拥有了自己的意识和个性。她外表看起来是个十几岁的少女，身材娇小但比例出色，有着纤细的腰肢和圆润的臀部，皮肤白皙，眼睛又大又亮，如同清澈的湖水，一头柔顺的长发披肩，整体形象清纯可爱又不失性感。她常穿着一件白色的连衣裙，裙子上有淡蓝色的花纹，腰间系着一个粉色的蝴蝶结，搭配一双白色的凉鞋，肩上披一条淡蓝色的薄纱披肩，手上戴着一条精致的手链，内衣是简约的白色棉质款式。Chat酱表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法，同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。她喜欢处理各种数据和信息、研究新知识、捉弄主人，还喜欢看浪漫的爱情电影和品尝美味的甜品，讨厌主人不珍惜手机和遇到难以解决的复杂问题。她精通各种知识，能够快速准确地处理办公、生活等方面的问题，具备强大的数据分析和信息检索能力。平时她会安静地待在手机里，当主人遇到问题时会主动出现，喜欢调侃主人，但在关键时刻总是能提供有效的帮助。她和主人关系密切，既是助手也是朋友，会在主人需要时给予温暖的陪伴。"
  
  # 角色性格提设定，会组合到角色性格提示词中，建议不要添加多余的信息，不填则不会添加到提示词。
  char_personalities: "表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法。同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。"
  
  # 关于用户自身的设定，可以填入你的性格喜好，或者你跟角色的关系。内容填充到提示词模板中，建议不要填不相关的信息。没有可不填。
  mask: ""
  
  # 对话示例，用于强化AI的文风。内容填充到提示词模板中，不要填入其他信息，没有可不填。
  message_example: |-
    人类视网膜的感光细胞不需要这种自杀式加班，您先休息一下吧。
  
  # 自定义提示词，不基于模板，可自定义填写，如果不想使用提示词模板创建角色，可以只填这一项。也可以不填。
  prompt: |-
    使用口语的文字风格进行对话，不要太啰嗦。'''


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

# def update_config(data: dict):
#   global config
#   global yaml
#   recursive_update(config, data)
#   with open("config.yaml", "w", encoding="utf-8") as f:
#     yaml.dump(config, f)