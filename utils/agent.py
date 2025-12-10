import asyncio
import os
import re
from Config import Config
from utils import long_mem, data_base, prompt, core_mem, log as Log
from utils import config as CConfig
import time
import jionlp
import ast
import json
import yaml
from models.types.assistant_info import AssistantInfo
from core.emotion.emotion_engine import EmotionEngine
from concurrent.futures import ThreadPoolExecutor
from utils.llm_request import Message, llm_request


class Agent:
    # 情绪系统实例
    emotionEngine: EmotionEngine
    # 角色记忆实例
    memoryEngine: long_mem.Memory
    # 核心记忆
    coreMemoryEngine: core_mem.CoreMemory
    # 数据知识库实例
    databaseEngine: data_base.DataBase

    # 好感度等级配置
    LOVE_LEVELS = {
        0: {
            "name": "疏远",
            "description": "与用户保持距离，关系比较陌生",
            "suggestion": "回复应该保持礼貌但疏远，避免过于亲密的表达",
            "value": -50,
        },
        1: {
            "name": "中性",
            "description": "对用户的态度保持中立，关系普通",
            "suggestion": "回复应该礼貌友好，但不过分亲昵",
            "value": -20,
        },
        2: {
            "name": "友好",
            "description": "与用户关系良好，有一定的亲近感",
            "suggestion": "回复应该热情友好，可以使用一些亲昵的表达",
            "value": 1000,
        },
        3: {
            "name": "亲密",
            "description": "与用户关系非常亲密，如同好友或家人",
            "suggestion": "回复应该非常亲昵，使用温暖贴心的语气和词汇",
            "value": 2000,
        },
        4: {
            "name": "挚爱",
            "description": "与用户关系非常亲密，如同好友或家人或恋人",
            "suggestion": "回复应该非常亲昵，使用温暖贴心的语气和词汇",
            "value": 5000,
        },
    }

    def __init__(self, agent_name: str):
        # 助手名称
        self.agent_name = agent_name
        # 聊天记录
        self.msg_data = []
        # 当前正在处理的消息记录
        self.msg_data_tmp = []
        # 线程池执行器，用于处理同步的 CPU 密集任务
        self._executor = ThreadPoolExecutor(max_workers=4)

        self.load_config()
        self._init_history()

    def _init_history(self):
        """初始化历史记录"""
        history_path = f"./data/agents/{self.agent_name}/history.yaml"
        try:
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    msg_list = yaml.safe_load(f) or []
                    self.msg_data = msg_list[
                        -self.agent_config.settings.contextLength :
                    ]
                    Log.logger.info(f"当前上下文长度：{len(msg_list)}")
        except Exception as e:
            Log.logger.error(f"加载上下文失败：{self.agent_name}, 错误: {e}")

        # 添加起始对话
        if self.agent_config.startWith and not self.msg_data:
            for i, content in enumerate(self.agent_config.startWith):
                role = "user" if i % 2 == 0 else "assistant"
                self.msg_data_tmp.append({"role": role, "content": content})

    def _load_config(self):
        """加载配置文件"""
        config_path = f"./data/agents/{self.agent_name}/info.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"角色配置文件 {config_path} 不存在")

        with open(config_path, "r", encoding="utf-8") as f:
            self.agent_config = AssistantInfo.from_dict(yaml.safe_load(f))

    def _get_love_prompt(self):
        """
        获取好感度相关的提示词
        """
        love_level = self._get_love_level(self.agent_config.love)
        level_info = self.LOVE_LEVELS.get(love_level, self.LOVE_LEVELS[0])

        return prompt.love_level_prompt.format(
            char=self.char,
            user=self.user,
            love_level=level_info["name"],
            love_description=level_info["description"],
            interaction_suggestion=level_info["suggestion"],
        )

    def _load_prompt_template(self):
        """加载提示词模板"""
        # 载入提示词
        self.prompt = ""
        self.long_mem_prompt = prompt.long_mem_prompt
        self.data_base_prompt = prompt.data_base_prompt
        self.core_mem_prompt = prompt.core_mem_prompt
        # 加入角色设定到提示词
        if self.description:
            # 格式化系统提示词
            self.system_prompt = prompt.system_prompt.format(
                char=self.char, user=self.user
            )
            # 格式化角色设定提示词
            self.char_setting_prompt = prompt.char_setting_prompt.format(
                char_setting_prompt=self.description, char=self.char, user=self.user
            )
            self.prompt += self.system_prompt + "\n\n"
            self.prompt += self.char_setting_prompt + "\n\n"
        # 加入角色性格到提示词
        if self.personality:
            self.char_personalities_prompt = prompt.char_Personalities_prompt.format(
                char_Personalities_prompt=self.personality,
                char=self.char,
                user=self.user,
            )
            self.prompt += self.char_personalities_prompt + "\n\n"
        # 加入用户设定到提示词
        if self.mask:
            self.mask_prompt = prompt.mask_prompt.format(
                mask_prompt=self.mask, char=self.char, user=self.user
            )
            self.prompt += self.mask_prompt + "\n\n"
        # 加入对话案例到提示词
        if self.agent_config.messageExamples:
            self.message_example_prompt = prompt.message_example_prompt.format(
                message_example="\n".join(self.agent_config.messageExamples),
                char=self.char,
                user=self.user,
            )
            self.prompt += self.message_example_prompt + "\n\n"

        # 加入自定义提示词到提示词
        if self.agent_config.customPrompt:
            self.prompt += self.agent_config.customPrompt + "\n\n"

    async def _calculate_love_change(self, user_message, assistant_reply) -> int:
        """
        使用 LLM 判定用户消息与助手回复的情感、亲密度、互动质量
        Parameters:
            user_message: 用户输入的消息
            assistant_reply: 助手回复的消息
        Returns:
            好感度变化值
        """

        # 构建分析提示词
        emotion_analysis_prompt = prompt.analysis_prompt.format(
            user_message=user_message,
            assistant_reply=assistant_reply,
        )

        try:
            content = await llm_request(
                [
                    {"role": "system", "content": emotion_analysis_prompt},
                ]
            )
        except Exception as e:
            Log.logger.error("LLM 好感度判断失败:", e)
            return 0

        # 解析 JSON

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return 0

        result = json.loads(match.group(0))

        # ====== 根据 LLM 输出计算得分 ======
        change = 0

        # 用户情绪
        emotion_score = {
            "positive": 2,
            "neutral": 0,
            "negative": -3,
        }
        change += emotion_score.get(result.get("user_emotion"), 0)

        # 亲密度
        intimacy_score = {
            "high": 3,
            "medium": 1,
            "low": 0,
        }
        change += intimacy_score.get(result.get("intimacy"), 0)

        # 用户是否关心助手
        if result.get("care_for_assistant"):
            change += 1

        # 用户态度
        attitude_score = {
            "supportive": 2,
            "neutral": 0,
            "hostile": -4,
        }
        change += attitude_score.get(result.get("user_attitude_toward_assistant"), 0)

        # 助手回复质量
        reply_score = {
            "high": 1.5,
            "medium": 0,
            "low": -1.5,
        }
        change += reply_score.get(result.get("reply_quality"), 0)

        # 总倾向
        overall_score = {
            "strong_positive": 3,
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "strong_negative": -3,
        }
        change += overall_score.get(result.get("overall_love_tendency"), 0)
        Log.logger.info(f"LLM 好感度判断结果: {change}")
        # 单次变化限制
        change = max(min(change, 5), -5)
        return int(change)

    async def _async_search_knowledge(self, msg: str) -> tuple[str, float]:
        """
        异步知识库检索任务
        Parameters:
            msg: 用户输入的消息
        Returns:
            知识库检索结果
        """
        start_time = time.time()
        if not self.enable_data_base:
            return "", 0.0
        # jionlp 分词是 CPU 密集型，放入线程池
        msg_list = await self._run_sync_task(jionlp.split_sentence, msg, "fine")
        result = await self._run_sync_task(self.databaseEngine.search, msg_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    async def _async_search_memory(self, msg: str, time_str: str) -> tuple[str, float]:
        """
        异步包装记忆检索
        Parameters:
            msg: 用户输入的消息
            time_str: 时间字符串
        Returns:
            记忆检索结果
        """
        start_time = time.time()
        if not self.enable_long_memory:
            return "", 0.0

        # 假设 get_memories 内部是同步的，需要修改 utils 让其支持返回数据而不是 append 到 list
        # 这里为了兼容旧代码逻辑，我们包装一下
        def wrapper():
            temp_list = []
            self.memoryEngine.get_memories(msg, temp_list, time_str)
            return temp_list[0] if temp_list else ""

        result = await self._run_sync_task(wrapper)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    async def _async_search_core_mem(self, msg: str) -> tuple[str, float]:
        """
        异步包装核心记忆检索任务
        Parameters:
            msg: 用户输入的消息
        Returns:
            核心记忆检索结果
        """
        start_time = time.time()
        if not self.enable_core_memory:
            return "", 0.0

        def wrapper():
            temp_list = []
            self.coreMemoryEngine.find_memories(msg, temp_list)
            return temp_list[0] if temp_list else ""

        result = await self._run_sync_task(wrapper)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    async def _async_process_emotion(self, msg: str) -> tuple[str, float]:
        """
        处理情绪任务
        Parameters:
            msg: 用户输入的消息
        Returns:
            情绪处理结果
        """
        start_time = time.time()
        if not self.enable_emotion_engine:
            return "", 0.0
        result = await self.emotionEngine.process_emotion(msg)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    async def _task_add_long_memory(self, turn_data):
        """
        后台任务：添加长期记忆
        Parameters:
            turn_data: 对话数据
        """

        def wrapper():
            self.memoryEngine.add_memory(turn_data, self.current_time, self.llm_config)

        await self._run_sync_task(wrapper)

    async def _task_save_history(self, turn_data):
        """
        后台任务：保存文件
        Parameters:
            turn_data: 对话数据
        """

        def save():
            with open(
                f"./data/agents/{self.agent_name}/history.yaml", "a", encoding="utf-8"
            ) as f:
                yaml.dump(
                    turn_data, stream=f, allow_unicode=True, indent=2, sort_keys=False
                )

        await self._run_sync_task(save)

    def save_agent_config(self):
        """
        保存角色配置到配置文件
        """
        config_path = os.path.join(
            Config.BASE_AGENTS_PATH, self.agent_name, "info.yaml"
        )

        try:

            # 保存配置
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.agent_config.model_dump(),
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            Log.logger.error(f"保存好感度失败: {e}")

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
        # 是否开启情绪系统
        self.enable_emotion_engine = self.agent_config.settings.enableEmotionSystem

        # 加载全局配置
        # 用于提取记录长期记忆的大模型
        self.llm_config = CConfig.config["LLM2"]
        # 加载提示词模板
        self._load_prompt_template()

        # 加载角色记忆
        self.memoryEngine = long_mem.Memory(self.agent_config)
        # 加载核心记忆
        self.coreMemoryEngine = core_mem.CoreMemory(self.agent_config)
        # 载入知识库
        self.databaseEngine = data_base.DataBase(self.agent_config)
        # 加载情绪系统
        self.emotionEngine = EmotionEngine(
            agent_config=self.agent_config, llm_config=self.llm_config
        )

    async def insert_core_mem(
        self, user_message: str, assistant_reply: str, previous_assistant_msg: str
    ) -> None:
        """
        使用核心记忆提取用户和助手的对话内容，插入核心记忆任务
        Parameters:
            user_message: 用户输入的消息
            assistant_reply: 助手回复的消息
            previous_assistant_msg: 上一条助手回复的消息
        """
        # 调用提取核心记忆的提示词，并替换模板中的占位符
        core_memory_extract_prompt = prompt.get_core_mem.replace(
            "{{memories}}",
            json.dumps(self.coreMemoryEngine.mems[-100:], ensure_ascii=False),
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

        try:

            res_msg = await llm_request(
                [
                    {"role": "system", "content": core_memory_extract_prompt},
                    {"role": "user", "content": re_msg},
                ]
            )

            mem_list = ast.literal_eval(
                jionlp.extract_parentheses(res_msg, "[]")[0]
                .replace(" ", "")
                .replace("\n", "")
            )
            if len(mem_list) > 0:
                self.coreMemoryEngine.add_memory(mem_list)
        except Exception as e:
            Log.logger.error(f"核心记忆提取出错: {e}")
            return

    async def update_love_level(self, user_message, assistant_reply):
        """
        异步更新好感度任务
        Parameters:
            user_message: 用户输入的消息
            assistant_reply: 助手回复的消息
        """
        change = await self._calculate_love_change(user_message, assistant_reply)
        self.agent_config.love = max(self.agent_config.love + change, -50)

        # 异步保存配置
        def save_config():
            self.save_agent_config()

        await self._run_sync_task(save_config)

        Log.logger.info(
            f"助手 {self.agent_name} 好感度更新: 变化 {change}, 当前 {self.agent_config.love}"
        )

    async def get_msg_data(self, msg: str) -> list[Message]:
        """
        获取发送到大模型的上下文

        Parameters:
            msg: 客户端发送的消息

        Returns:
            发送到大模型的上下文
        """

        self.current_time = int(time.time())
        # 格式化时间
        format_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.current_time)
        )
        # 使用 asyncio.gather 同时启动所有任务
        tasks = [
            self._async_search_knowledge(msg),
            self._async_search_memory(msg, format_time),
            self._async_search_core_mem(msg),
            self._async_process_emotion(msg),
        ]
        results = await asyncio.gather(*tasks)
        # 解包结果和耗时
        db_info, db_time = results[0]
        mem_info, mem_time = results[1]
        core_info, core_time = results[2]
        emotion_info, emotion_time = results[3]

        # 打印或记录耗时
        print(f"Knowledge search time: {db_time:.4f}s")
        print(f"Memory search time: {mem_time:.4f}s")
        print(f"Core memory search time: {core_time:.4f}s")
        print(f"Emotion processing time: {emotion_time:.4f}s")

        # 返回消息列表
        res_msg = []
        # 添加系统提示词
        res_msg.append({"role": "system", "content": self.prompt})

        context_extras = []
        # 添加知识库信息
        if db_info:
            context_extras.append(
                self.data_base_prompt.format(
                    data_base=db_info, user=self.user, char=self.char
                )
            )
        # 添加核心记忆信息
        if core_info:
            context_extras.append(
                self.core_mem_prompt.format(
                    core_mem=core_info, user=self.user, char=self.char
                )
            )
        # 添加长期记忆信息
        if mem_info:
            context_extras.append(
                self.long_mem_prompt.format(
                    memories=mem_info, user=self.user, char=self.char
                )
            )
        # 添加好感度提示词
        context_extras.append(self._get_love_prompt())
        # 添加情绪信息
        if emotion_info:
            context_extras.append(emotion_info)

        # 合并上下文、世界书、记忆信息, 并添加情绪指令
        final_content = "\n".join(context_extras)
        final_content += f"\n<当前时间>{format_time}</当前时间>\n<用户对话内容或动作>\n{msg}\n</用户对话内容或动作>"

        self.msg_data_tmp = [{"role": "user", "content": final_content}]

        # 系统 Prompt + 历史记录 + 当前构建的 Context
        system_msg: list[Message] = [{"role": "system", "content": self.prompt}]

        return system_msg + self.msg_data + self.msg_data_tmp  # type: ignore

    async def add_msg(self, assistant_msg: str) -> None:
        """
        添加助手回复到上下文,保存聊天历史,更新长期记忆,更新好感度

        Parameters:
            assistant_msg: 助手回复的消息
        """
        # 添加助手回复到上下文
        self.msg_data_tmp.append({"role": "assistant", "content": assistant_msg})

        # 获取用于分析的内容
        user_msg_content = self.msg_data_tmp[-2]["content"]  # 刚刚的用户输入
        previous_assistant_msg = self.msg_data[-1]["content"] if self.msg_data else ""

        current_turn = self.msg_data_tmp.copy()
        self.msg_data += current_turn
        self.msg_data = self.msg_data[-self.agent_config.settings.contextLength :]
        # 清空临时消息列表
        self.msg_data_tmp = []

        await asyncio.gather(
            self.insert_core_mem(
                user_msg_content, assistant_msg, previous_assistant_msg
            ),
            self.update_love_level(user_msg_content, assistant_msg),
            self._task_save_history(current_turn),
            self._task_add_long_memory(current_turn),
        )

    async def _run_sync_task(self, func, *args):
        """
        工具方法
        在线程池中运行同步阻塞函数（如 jionlp 处理或旧的数据库搜索）
        Parameters:
            func: 要运行的同步函数
            *args: 函数的参数
        Returns:
            函数的返回值
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    def _get_love_level(self, love_value: int) -> int:
        """
        根据好感度数值获取好感度等级
        好感度分为5个等级：0(疏远)、1(中性)、2(友好)、3(亲密)、4(挚爱)
        Parameters:
            love_value (int): 好感度数值

        Returns:
            int: 好感度等级
        """
        for level, config in self.LOVE_LEVELS.items().__reversed__():
            if love_value >= config["value"]:
                return level
        return 0

    def _ensure_directory(self):
        """确保配置目录存在，如果不存在则创建"""
        os.makedirs(f"./data/agents/{self.agent_name}", exist_ok=True)
        # 创建数据存储文件夹
        os.makedirs(f"./data/agents/{self.agent_name}/memory", exist_ok=True)
        os.makedirs(f"./data/agents/{self.agent_name}/data_base", exist_ok=True)
