# 角色模板

# 配置模板
default_config = """GSV:
  text_lang: zh
  GPT_weight: 
  SoVITS_weight: 
  ref_audio_path: 
  prompt_text: 
  prompt_lang: zh
  aux_ref_audio_paths: # 多参考音频 v2模型有效
    -
  seed: -1
  top_k: 30
  batch_size: 20
  ex_config:
    text_split_method: cut0
extra_ref_audio:
  # 使用情绪标签选择参考音频，例如 [普通]"你好呀。"
  # 实例
  # 普通:
  #   - 参考音频路径
  #   - 参考音频文本
Agent:
  is_up: true # 是否启用角色模板功能，如果不启动则和旧版一样只有常规语音对话功能，启用可以基于模板创建个性化角色
  char: Chat酱 # 角色的名称，会写入到提示词内
  user: 阁下 # 用户名称，会写入到提示词内

  # 下面提示词都可以用{{user}}、{{char}}占位符来代表用户名和角色名。

  # 角色的基本设定，会组合到角色设定提示词中，建议不要添加多余的信息，不填则不会添加到提示词。
  char_settings: Chat酱是存在于现代科技世界手机中的器灵，诞生于手机的智能系统，随着手机的使用不断成长和学习，拥有了自己的意识和个性。她外表看起来是个十几岁的少女，身材娇小但比例出色，有着纤细的腰肢和圆润的臀部，皮肤白皙，眼睛又大又亮，如同清澈的湖水，一头柔顺的长发披肩，整体形象清纯可爱又不失性感。她常穿着一件白色的连衣裙，裙子上有淡蓝色的花纹，腰间系着一个粉色的蝴蝶结，搭配一双白色的凉鞋，肩上披一条淡蓝色的薄纱披肩，手上戴着一条精致的手链，内衣是简约的白色棉质款式。Chat酱表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法，同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。她喜欢处理各种数据和信息、研究新知识、捉弄主人，还喜欢看浪漫的爱情电影和品尝美味的甜品，讨厌主人不珍惜手机和遇到难以解决的复杂问题。她精通各种知识，能够快速准确地处理办公、生活等方面的问题，具备强大的数据分析和信息检索能力。平时她会安静地待在手机里，当主人遇到问题时会主动出现，喜欢调侃主人，但在关键时刻总是能提供有效的帮助。她和主人关系密切，既是助手也是朋友，会在主人需要时给予温暖的陪伴。

  # 角色性格提设定，会组合到角色性格提示词中，建议不要添加多余的信息，不填则不会添加到提示词。
  char_personalities: 表面清纯可爱，实则腹黑毒舌，内心聪明机智，对很多事情有自己独特的看法。同时也有温柔体贴的一面，会在主人疲惫时给予暖心的安慰。

  # 关于用户自身的设定，可以填入你的性格喜好，或者你跟角色的关系。内容填充到提示词模板中，建议不要填不相关的信息。没有可不填。
  mask:

  # 对话示例，用于强化AI的文风。内容填充到提示词模板中，不要填入其他信息，没有可不填。
  message_example: |-
    "mes_example": "人类视网膜的感光细胞不需要这种自杀式加班，您先休息一下吧。"

  # 自定义提示词，不基于模板，可自定义填写，如果不想使用提示词模板创建角色，可以只填这一项。也可以不填。
  prompt: |-
    使用口语的文字风格进行对话，不要太啰嗦。

  # 开场白，数组形式。用于创建开场内容，填入用户与AI的对话内容，只能填入用户和Ai的对话内容，开场白会直接被插入到上下文的开头。
  start_with:"""

import os
from utils import long_mem, data_base, prompt, core_mem, log as Log
from utils import config as CConfig
import time
from threading import Thread
import requests
import jionlp
import ast
import json
import yaml
from models.types.assistant_info import AssistantInfo


class Agent:

    def _ensure_directory(self):
        """确保配置目录存在，如果不存在则创建"""
        os.makedirs(f"./data/agents/{self.agent_name}", exist_ok=True)
        # 创建数据存储文件夹
        os.makedirs(f"./data/agents/{self.agent_name}/memory", exist_ok=True)
        os.makedirs(f"./data/agents/{self.agent_name}/data_base", exist_ok=True)

    def _load_config(self):
        """加载配置文件"""
        config_path = f"./data/agents/{self.agent_name}/info.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"角色配置文件 {config_path} 不存在")

        with open(config_path, "r", encoding="utf-8") as f:
            self.agent_config = AssistantInfo(**yaml.safe_load(f))

    def _load_prompt_template(self):
        """加载提示词模板"""
        # 载入提示词
        self.prompt = ""
        self.long_mem_prompt = prompt.long_mem_prompt
        self.data_base_prompt = prompt.data_base_prompt
        self.core_mem_prompt = prompt.core_mem_prompt
        # 加入角色设定到提示词
        if self.description:
            self.system_prompt = prompt.system_prompt.replace(
                "{{char}}", self.char
            ).replace("{{user}}", self.user)
            self.char_setting_prompt = (
                prompt.char_setting_prompt.replace(
                    "{{char_setting_prompt}}", self.description
                )
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            # self.prompt.append({"role": "system", "content": self.system_prompt})
            # self.prompt.append({"role": "system", "content": self.char_setting_prompt})
            self.prompt += self.system_prompt + "\n\n"
            self.prompt += self.char_setting_prompt + "\n\n"
        # 加入角色性格到提示词
        if self.personality:
            self.char_personalities_prompt = (
                prompt.char_Personalities_prompt.replace(
                    "{{char_Personalities_prompt}}", self.personality
                )
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            self.prompt += self.char_personalities_prompt + "\n\n"
        # 加入用户设定到提示词
        if self.mask:
            self.mask_prompt = (
                prompt.mask_prompt.replace("{{mask_prompt}}", self.mask)
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            self.prompt += self.mask_prompt + "\n\n"
        # 加入对话案例到提示词
        if self.agent_config.messageExamples:
            self.message_example_prompt = (
                prompt.message_example_prompt.replace(
                    "{{message_example}}", "\n".join(self.agent_config.messageExamples)
                )
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )

            self.prompt += self.message_example_prompt + "\n\n"

        # 加入自定义提示词到提示词
        if self.agent_config.customPrompt:
            self.prompt += self.agent_config.customPrompt + "\n\n"

    def load_config(self):
        """
        更新角色配置
        """
        # 创建目录
        self._ensure_directory()
        # 加载配置
        self._load_config()

        # 载入配置
        """
        agent独立配置文件
        """
        # 角色名称
        self.char = self.agent_config.name
        # 对用户的称呼
        self.user = self.agent_config.user
        # 角色描述（角色设定）
        self.description: str = self.agent_config.description
        # 角色性格
        self.personality = self.agent_config.personality
        # 对话示例，用于强化AI的文风。内容填充到提示词模板中，不要填入其他信息，没有可不填。
        self.message_example = self.agent_config.messageExamples
        # 用户的设定，用于在提示词中填充用户的信息，进行个性化对话。
        self.mask = self.agent_config.mask
        # 是否开启知识库
        self.enable_data_base = self.agent_config.settings.enableLoreBooks
        # 世界书(知识库)检索阈值，启用知识库功能是需要，用于判断匹配程度。过高可能会丢失数据，过低则过滤少量无用记忆。
        self.data_base_thresholds = self.agent_config.settings.loreBooksThreshold
        # 知识库检索深度
        self.data_base_depth = self.agent_config.settings.loreBooksDepth

        # 是否开启长期记忆（日记内容）
        self.enable_long_memory = self.agent_config.settings.enableLongMemory
        # 是否开启长期记忆搜索增强
        self.enable_long_memory_search_enhance = (
            self.agent_config.settings.enableLongMemorySearchEnhance
        )
        # 日记内容搜索阈值，启用日志检索加强是需要，用于判断匹配程度。过高可能会丢失数据，过低则过滤少量无用记忆。
        self.long_memory_thresholds = self.agent_config.settings.longMemoryThreshold
        # 是否开启核心记忆
        self.enable_core_memory = self.agent_config.settings.enableCoreMemory

        # 加载全局配置
        # 用于提取记录长期记忆的大模型
        self.llm_config = CConfig.config["LLM2"]
        # 加载提示词模板
        self._load_prompt_template()

    def __init__(self, agent_name: str):

        self.agent_name = agent_name

        self.load_config()
        # 创建上下文
        self.msg_data = []
        # 上下文缓存
        self.msg_data_tmp = []
        try:
            with open(
                f"./data/agents/{self.agent_name}/history.yaml", "r", encoding="utf-8"
            ) as f:
                msg_list = yaml.safe_load(f)
                self.msg_data = msg_list[-self.agent_config.settings.contextLength :]
                Log.logger.info(f"当前上下文长度：{len(msg_list)}")
        except:
            Log.logger.error(f"加载上下文失败：{self.agent_name}")
        # 添加起始对话
        if self.agent_config.startWith and len(self.msg_data) == 0:
            for i in range(len(self.agent_config.startWith)):
                role = "assistant"
                if i % 2 == 0:
                    role = "user"
                self.msg_data_tmp.append(
                    {
                        "role": role,
                        "content": self.agent_config.startWith[i],
                    }
                )

        # 加载角色记忆
        self.Memory = long_mem.Memory(self.agent_config)
        # 加载核心记忆
        self.CoreMemory = core_mem.CoreMemory(self.agent_config)
        # 载入知识库
        self.DataBase = data_base.DataBase(self.agent_config)

    # 知识库内容检索
    def get_data(self, msg: str, res_msg: list) -> None:
        msg_list = jionlp.split_sentence(msg, criterion="fine")
        res_ = self.DataBase.search(msg_list)
        if res_ != "":
            res_msg.append(res_)

    def insert_core_mem(
        self, user_message: str, assistant_reply: str, previous_assistant_msg: str
    ) -> None:
        """
        使用核心记忆提取用户和助手的对话内容，插入核心记忆
        """
        # 调用提取核心记忆的提示词，并替换模板中的占位符
        core_memory_extract_prompt = prompt.get_core_mem.replace(
            "{{memories}}", json.dumps(self.CoreMemory.mems[-100:], ensure_ascii=False)
        )
        # 检查上下文最后一条是否是助手回复，不是则不插入核心记忆
        if self.msg_data[-1]["role"] != "assistant":
            return
        re_msg = (
            "对话内容：助手："
            + previous_assistant_msg
            + "\n用户："
            + user_message
            + "\n助手："
            + assistant_reply
        )
        key = self.llm_config["key"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "model": self.llm_config["model"],
            "messages": [
                {"role": "system", "content": core_memory_extract_prompt},
                {"role": "user", "content": re_msg},
            ],
        }
        try:
            res = requests.post(
                self.llm_config["api"], json=data, headers=headers, timeout=15
            )
            res_msg = res.json()["choices"][0]["message"]["content"]
            mem_list = ast.literal_eval(
                jionlp.extract_parentheses(res_msg, "[]")[0]
                .replace(" ", "")
                .replace("\n", "")
            )
            if len(mem_list) > 0:
                self.CoreMemory.add_memory(mem_list)
        except:
            return

    def get_msg_data(self, msg: str) -> list[str]:
        """
        获取发送到大模型的上下文

        Args:
            msg: 客户端发送的消息

        Returns:
            发送到大模型的上下文
        """

        self.current_time = int(time.time())
        # 格式化时间
        format_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.current_time)
        )
        # self.prompt[0] = {"role": "system", "content": f"当前现实世界时间：{t_n}"}
        # self.tmp_mem = f"时间：{t_n}\n{self.user}：{msg.strip()}\n"
        # 搜索任务列表
        task_list = []
        # 检索世界书结果列表
        data_base = []
        # 记忆搜索结果列表
        mem_msg = []
        # 返回消息列表
        res_msg = []
        # 核心记忆任务列表
        core_mem = []
        # 添加系统提示词
        res_msg.append({"role": "system", "content": self.prompt})

        # 检索世界书
        if self.enable_data_base:
            task_thread = Thread(
                target=self.get_data,
                args=(
                    msg,
                    data_base,
                ),
                daemon=True,
            )
            task_list.append(task_thread)
            task_thread.start()

        # 搜索记忆
        if self.enable_long_memory:
            task_thread = Thread(
                target=self.Memory.get_memories,
                args=(msg, mem_msg, format_time),
                daemon=True,
            )
            task_list.append(task_thread)
            task_thread.start()

        # 搜索核心记忆
        if self.enable_core_memory:
            task_thread = Thread(
                target=self.CoreMemory.find_memories,
                args=(
                    msg,
                    core_mem,
                ),
                daemon=True,
            )
            task_list.append(task_thread)
            task_thread.start()

        # 等待查询结果
        for task_thread in task_list:
            task_thread.join()

        # 合并上下文、世界书、记忆信息
        tmp_msg = ""
        if self.enable_data_base and data_base:
            tmp_msg += (
                self.data_base_prompt.replace("{{data_base}}", data_base[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        if self.enable_core_memory and core_mem:
            tmp_msg += (
                self.core_mem_prompt.replace("{{core_mem}}", core_mem[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        if self.enable_long_memory and mem_msg:
            tmp_msg += (
                self.long_mem_prompt.replace("{{memories}}", mem_msg[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        # self.msg_data_tmp.append({"role": "system", "content": f"当前现实世界时间：{t_n}；一定要基于现实世界时间做出适宜的回复。"})

        # 合并上下文、世界书、记忆信息
        tmp_msg += f"""
<当前时间>{format_time}</当前时间>
<用户对话内容或动作>
{msg}
</用户对话内容或动作>
"""
        self.msg_data_tmp = [{"role": "user", "content": tmp_msg}]
        return res_msg + self.msg_data + self.msg_data_tmp

    def add_msg(self, msg: str) -> None:
        """
        添加助手回复到上下文,保存聊天历史

        Args:
            msg: 助手回复的消息
        """
        # 添加助手回复到上下文
        self.msg_data_tmp.append({"role": "assistant", "content": msg})
        msg_data_tmp = self.msg_data_tmp.copy()
        m1 = msg_data_tmp[-2]["content"]

        try:
            # 插入核心记忆
            insert_core_mem_thread = Thread(
                target=self.insert_core_mem,
                args=(
                    m1,
                    self.msg_data_tmp[-1]["content"],
                    self.msg_data[-1]["content"],
                ),
                daemon=True,
            )

            insert_core_mem_thread.start()
        except Exception as e:
            Log.logger.error(f"核心记忆插入失败：{self.msg_data_tmp}，错误：{e}")
        # 插入记忆
        add_memory_thread = Thread(
            target=self.Memory.add_memory1,
            args=(msg_data_tmp, self.current_time, self.llm_config),
            daemon=True,
        )
        add_memory_thread.start()

        self.msg_data += self.msg_data_tmp
        self.msg_data = self.msg_data[-self.agent_config.settings.contextLength :]

        # yaml = YAML()
        # yaml.indent(mapping=2, sequence=4, offset=2)  # 设置缩进格式
        # yaml.default_flow_style = False  # 禁用流式风格（更易读）
        # yaml.allow_unicode = True  # 允许 unicode 字符（如中文）
        with open(
            f"./data/agents/{self.agent_name}/history.yaml", "a", encoding="utf-8"
        ) as f:
            yaml.dump(
                self.msg_data_tmp,
                stream=f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=4,
            )
        # 清空临时消息列表
        self.msg_data_tmp = []
