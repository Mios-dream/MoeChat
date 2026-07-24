import asyncio
import os
import re
from Config import Config
from my_utils import prompt, log as Log
from my_utils import config_manager as CConfig
import time
import jionlp
import json
import yaml
from models.types.assistant_info import AssistantInfo
from models.types.user_state import UserStateInfo
from core.emotion.emotion_engine import EmotionEngine
from concurrent.futures import ThreadPoolExecutor
from core.llm.llm_client import LLMClient
from services import data_base
from services.memory_v2 import MemoryV2
from tool_system.tools.memory_tool import RememberTool, RecallTool, UpdateMemoryTool
from openai.types.chat import ChatCompletionMessageParam


class Assistant:
    # 事件驱动情绪引擎
    emotionEngine: EmotionEngine
    # 统一记忆引擎（v2，替代 core_mem + long_mem）
    memoryEngine: MemoryV2
    # 数据知识库实例
    databaseEngine: data_base.DataBase

    # 好感度阶段配置（基于 affinity 值，阈值依次为各阶段最低值）
    # 描述使用叙事体，让 LLM 自然感受关系状态而非机械执行指令
    AFFINITY_STAGES: list[tuple[str, int, str]] = [
        ("疏远", -50, "你对{user}还很陌生，心里保持着明显的距离感，不太愿意主动靠近"),
        ("陌生", -30, "你和{user}才刚认识不久，还在慢慢熟悉彼此的存在，相处时有些拘谨"),
        ("认识", 0, "你和{user}逐渐熟悉起来了，相处还算愉快，开始期待和他的交流"),
        ("亲近", 25, "你和{user}的关系越来越亲近了，和他在一起时你会感到轻松自在"),
        ("亲密", 50, "你已经很信任{user}了，愿意在他面前展现真实的自己，不设防备"),
        ("挚爱", 80, "{user}是你心中最重要的人，和他在一起的每一刻你都很珍惜"),
    ]
    TRUST_THRESHOLDS: list[tuple[int, str]] = [
        (30, "你发自内心地信任{user}，愿意依赖他，在他面前你感到安全"),
        (0, "你还在观察{user}，不确定他是否值得你完全托付信任"),
        (-30, "{user}的一些言行让你感到不安，你开始对他有所防备"),
        (-50, "你不敢轻易相信{user}了，他曾经让你感到失望或受伤"),
    ]

    def __init__(self, agent_name: str):
        # 助手名称
        self.agent_name = agent_name
        # 聊天记录
        self.chat_history: list[ChatCompletionMessageParam] = []
        # 线程池执行器，用于处理同步的 CPU 密集任务
        self._executor = ThreadPoolExecutor(max_workers=4)
        # 用户私有状态（好感度、首次相遇时间等）
        self.user_state: UserStateInfo = UserStateInfo()
        # LLM 客户端实例
        self._llm_client = LLMClient(model_key="LLM")

        self.load_config()

        # 添加起始对话
        if self.agent_config.startWith:
            for content in self.agent_config.startWith:
                self.chat_history.append({"role": "system", "content": content})

    def _load_config(self):
        """加载配置文件"""
        config_path = f"./data/agents/{self.agent_name}/info.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"角色配置文件 {config_path} 不存在")

        with open(config_path, "r", encoding="utf-8") as f:
            self.agent_config = AssistantInfo.from_dict(yaml.safe_load(f))

        # 加载用户私有状态
        user_state_path = f"./data/agents/{self.agent_name}/user_state.yaml"
        if os.path.exists(user_state_path):
            with open(user_state_path, "r", encoding="utf-8") as f:
                self.user_state = UserStateInfo.from_dict(yaml.safe_load(f) or {})
        else:
            self.user_state = UserStateInfo()

    def _get_affection_prompt(self) -> str:
        """
        生成好感度叙事提示词
        用叙事体描述关系状态，让 LLM 自然感受而非机械执行
        """
        stage_name, stage_desc = self._get_affinity_stage(self.user_state.affinity)
        trust_desc = self._get_trust_level(self.user_state.trust)
        days_known = max(1, (int(time.time()) - self.user_state.firstMeetTime) // 86400)

        affection_text = (
            f"你和{self.user}目前的关系是「{stage_name}」。\n"
            f"{stage_desc.format(user=self.user)}\n"
            f"{trust_desc.format(user=self.user)}\n"
            f"你们已经相识 {days_known} 天，共同经历了 {self.user_state.bond} 次有意义的互动——"
            f"这些经历构成了你对他的真实感受。\n"
            f"请自然地体会这些感受，像真实的自己一样去回应，无需刻意强调。"
        )
        return prompt.affection_prompt.format(affection_text=affection_text)

    def _load_prompt_template(self):
        """加载提示词模板"""
        self.prompt = ""
        self.memory_prompt = prompt.memory_prompt
        self.data_base_prompt = prompt.data_base_prompt
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
        if self.agent_config.extraDescription:
            self.extra_description_prompt = prompt.extra_description_prompt.format(
                extra_description=self.agent_config.extraDescription,
                char=self.char,
                user=self.user,
            )
            self.prompt += self.extra_description_prompt + "\n\n"
        # 加入自定义提示词到提示词
        if self.agent_config.customPrompt:
            self.prompt += self.agent_config.customPrompt + "\n\n"
        # 对话案例置于尾部——所有角色核心设定（description / personality / customPrompt）
        # 完成后才追加风格参考，避免喧宾夺主
        if self.agent_config.messageExamples:
            self.message_example_prompt = prompt.message_example_prompt.format(
                message_example="\n".join(self.agent_config.messageExamples),
                char=self.char,
                user=self.user,
            )
            self.prompt += self.message_example_prompt + "\n\n"

    def _get_interaction_summary(self, window: int = 20) -> str:
        """
        统计最近互动的构成，纯描述不做价值判断

        用户可能喜欢主动对话，也可能只是享受后台陪伴——两者都正常。
        统计只反映互动模式，不作为评价用户好坏的依据。
        """
        recent = self.chat_history[-window:]
        if not recent:
            return ""

        user_turns = 0
        assistant_initiated = 0
        responded = 0

        i = 0
        while i < len(recent):
            if recent[i]["role"] == "user":
                user_turns += 1
                i += 1
            elif recent[i]["role"] == "assistant":
                # 无前驱 user → 助手主动发起的互动
                if i == 0 or recent[i - 1]["role"] != "user":
                    assistant_initiated += 1
                    # 下一条是否为用户续接？
                    if i + 1 < len(recent) and recent[i + 1]["role"] == "user":
                        responded += 1
                        i += 2
                        continue
                i += 1
            else:
                i += 1

        parts = []
        if user_turns:
            parts.append(f"用户主动对话 {user_turns} 次")
        if assistant_initiated:
            parts.append(f"助手主动互动 {assistant_initiated} 次")
        return "；".join(parts) if parts else ""

    async def _calculate_affection_change(self) -> dict:
        """
        统一 LLM 分析：好感度 + 信任度 + 羁绊 + 情绪冲击

        分析仅基于实际对话内容的质量，而非互动频率或响应率。
        用户选择不回应助手主动互动是完全正常的行为，
        好感度不因此惩罚用户——它只反映说了什么，而非说了多少次。

        构建消息链以利用 LLM KV 缓存：
          1. system（静态角色信息 → 缓存命中）
          2. 最近对话历史（上下文）
          3. user（当前状态 + 互动统计 + 分析请求）
        """
        current_emotions = self.emotionEngine.get_mood_prompt()
        days_known = max(1, (int(time.time()) - self.user_state.firstMeetTime) // 86400)
        default_result = {
            "affinity_change": 0,
            "trust_change": 0,
            "bond_increment": 0,
            "emotional_impact": {"emotion": "neutral", "intensity": 0.0, "reason": ""},
        }

        # 1. system：静态角色信息（可缓存）
        system_content = prompt.analysis_system_prompt.format(
            char=self.char,
            user=self.user,
            personality=self.personality or "",
            description=self.description or "",
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
        ]

        # 2. 最近对话上下文（原始历史，自动交互与用户对话共存）
        #    上限 10 条防止单方消息过度主导，配合下方互动统计供 LLM 权衡判断
        recent = self.chat_history[-10:]
        messages.extend(recent)

        # 3. 分析请求（动态状态 + 互动概况 + 输出格式）
        request_text = (
            f"当前关系状态：\n"
            f"- 好感度：{self.user_state.affinity}\n"
            f"- 信任度：{self.user_state.trust}\n"
            f"- 羁绊值：{self.user_state.bond}\n"
            f"- 相识天数：{days_known}\n"
            f"- 当前情绪状态：{current_emotions or '平静'}\n"
            f"- 互动概况：{self._get_interaction_summary()}\n\n"
            f"请根据以上完整对话，分析最新一轮对话对{self.char}和{self.user}之间关系的影响。\n\n"
            f"1. affinity_change: 好感度变化（整数 -3 到 +3）\n"
            f"2. trust_change: 信任度变化（整数 -3 到 +3，降快升慢）\n"
            f"3. bond_increment: 羁绊增量（整数 0 到 2）\n"
            f'4. emotional_impact: {{"emotion":"...", "intensity":0.0~1.0, "reason":"用一句话描述触发情绪的具体情景"}}\n\n'
            f'{{"affinity_change":0,"trust_change":0,"bond_increment":0,'
            f'"emotional_impact":{{"emotion":"neutral","intensity":0.0,"reason":""}}}}'
        )
        messages.append({"role": "user", "content": request_text})

        # 4. 请求 LLM
        try:
            content = await self._llm_client.request(messages)
        except Exception as e:
            Log.logger.error("LLM 好感度判断失败:", e)
            return default_result

        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            return default_result

        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default_result

        affinity_change = max(-3, min(3, result.get("affinity_change", 0)))
        trust_change = max(-3, min(3, result.get("trust_change", 0)))
        bond_increment = max(0, min(2, result.get("bond_increment", 0)))
        emotional_impact = result.get(
            "emotional_impact", {"emotion": "neutral", "intensity": 0.0, "reason": ""}
        )

        Log.logger.info(
            f"好感度分析: affinity={affinity_change}, trust={trust_change}, bond={bond_increment}, emotion={emotional_impact}"
        )
        return {
            "affinity_change": affinity_change,
            "trust_change": trust_change,
            "bond_increment": bond_increment,
            "emotional_impact": emotional_impact,
        }

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

    async def _async_search_memory(self, msg: str) -> tuple[str, float]:
        """
        异步包装记忆检索（统一 v2 引擎，替代旧 long_mem + core_mem）

        Parameters:
            msg: 用户输入的消息

        Returns:
            (格式化记忆文本, 耗时)
        """
        start_time = time.time()
        if not self.enable_long_memory:
            return "", 0.0

        result = await self._run_sync_task(self.memoryEngine.get_context, msg)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    def save_agent_config(self):
        """
        保存用户私有状态到 user_state.yaml
        """
        config_path = os.path.join(
            Config.BASE_AGENTS_PATH, self.agent_name, "user_state.yaml"
        )

        try:
            self.user_state.updatedAt = int(time.time())

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.user_state.model_dump(),
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            Log.logger.error(f"保存用户状态失败: {e}")

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
        # 别称
        self.alias = self.agent_config.alias
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
        # 是否开启记忆系统（v2 统一引擎，替代旧 long_mem + core_mem）
        self.enable_long_memory = self.agent_config.settings.enableLongMemory

        # 加载全局配置
        # 用于提取记录长期记忆的大模型
        self.llm_config = CConfig.config["LLM"]
        # 加载提示词模板
        self._load_prompt_template()

        # 加载统一记忆引擎 v2（替代旧 long_mem + core_mem）
        self.memoryEngine = MemoryV2(
            self.agent_config, firstMeetTime=self.user_state.firstMeetTime
        )
        # 注入记忆引擎到记忆工具，使 LLM 可通过工具自主记录、回忆和更新记忆
        RememberTool.set_engine(self.memoryEngine)
        RecallTool.set_engine(self.memoryEngine)
        UpdateMemoryTool.set_engine(self.memoryEngine)
        # 载入知识库
        self.databaseEngine = data_base.DataBase(self.agent_config)
        # 加载事件驱动情绪引擎
        self.emotionEngine = EmotionEngine(agent_name=self.agent_name)

    async def update_affection(self, user_message, assistant_reply):
        """
        异步更新好感度+信任度+羁绊+情绪事件
        Parameters:
            user_message: 用户输入的消息
            assistant_reply: 助手回复的消息
        """
        result = await self._calculate_affection_change()

        # 更新好感度（-50~100）
        self.user_state.affinity = max(
            -50, min(100, self.user_state.affinity + result["affinity_change"])
        )
        # 更新信任度（-50~100）
        self.user_state.trust = max(
            -50, min(100, self.user_state.trust + result["trust_change"])
        )
        # 更新羁绊（只增不减）
        self.user_state.bond += result["bond_increment"]

        # 记录情绪事件
        self.emotionEngine.process_event(result["emotional_impact"])

        # 异步保存配置
        def save_config():
            self.save_agent_config()

        await self._run_sync_task(save_config)

        Log.logger.info(
            f"助手 {self.agent_name} 状态更新: affinity={self.user_state.affinity}, trust={self.user_state.trust}, bond={self.user_state.bond}"
        )

    async def get_context(
        self, msg: str, is_sleep_mode: bool = False
    ) -> list[ChatCompletionMessageParam]:
        """
        获取动态上下文消息列表（知识库 + 记忆 + 好感度）

        注意：本方法只返回每次轮询变化的动态内容，记忆系统说明等静态指令
        由调用方在固定前缀位置构建，以最大化推理缓存命中率。

        Parameters:
            msg: 客户端发送的消息
            is_sleep_mode: 是否处于睡眠模式

        Returns:
            动态上下文的 system 消息列表（通常 0~1 条，放在对话历史之后）
        """
        tasks = [
            self._async_search_knowledge(msg),
            self._async_search_memory(msg),
        ]
        results = await asyncio.gather(*tasks)
        db_info, _ = results[0]
        mem_info, _ = results[1]

        messages: list[ChatCompletionMessageParam] = []
        if db_info:
            messages.append(
                {
                    "role": "system",
                    "content": self.data_base_prompt.format(
                        data_base=db_info, user=self.user, char=self.char
                    ),
                }
            )
        if mem_info:
            messages.append(
                {
                    "role": "system",
                    "content": self.memory_prompt.format(
                        memories=mem_info, user=self.user, char=self.char
                    ),
                }
            )
        # 好感度与情绪各自独立成段，避免被其他内容淹没
        messages.append(
            {
                "role": "system",
                "content": self._get_affection_prompt(),
            }
        )
        mood_prompt = self.emotionEngine.get_mood_prompt()
        if mood_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": mood_prompt,
                }
            )
        if is_sleep_mode:
            messages.append(
                {
                    "role": "system",
                    "content": prompt.sleep_mode_prompt.format(char=self.char),
                }
            )
        return messages

    def get_history(self) -> list[ChatCompletionMessageParam]:
        """
        获取当前上下文的历史记录

        Returns:
            当前上下文的历史记录
        """
        return self.chat_history.copy()

    def update_history(self, chat_turns: list[ChatCompletionMessageParam]):
        """
        追加或更新聊天历史
        """
        self.chat_history.extend(chat_turns)

    def clear_history(self) -> None:
        """
        清除内存中的聊天历史
        """
        self.chat_history = []

    async def add_msg(self, user_msg: str, assistant_msg: str) -> None:
        """
        添加对话回合后的后续处理。

        chat_history 由调用方管理（调用前已追加完整序列），
        本方法负责：
        1. 好感度更新
        2. 原始对话存储（供日记生成使用）
        3. 跨天日记生成检查

        Parameters:
            user_msg: 用户输入的消息
            assistant_msg: 助手回复的消息
        """
        # 安全截断：移除孤立 tool 消息（前驱 tool_calls 被切掉时）
        max_len = self.agent_config.settings.contextLength
        if len(self.chat_history) > max_len:
            truncated = self.chat_history[-max_len:]
            while truncated and truncated[0].get("role") == "tool":
                truncated = truncated[1:]
            self.chat_history = truncated

        # 存储原始对话轮次供日记使用
        now_ts = int(time.time())
        if self.enable_long_memory:
            self.memoryEngine.add_chat_turn("user", user_msg, now_ts)
            self.memoryEngine.add_chat_turn("assistant", assistant_msg, now_ts)
            # 跨天日记生成检查（后台任务）
            asyncio.create_task(self.memoryEngine.check_and_generate_diary(now_ts))

        await self.update_affection(user_msg, assistant_msg)

    async def add_interaction_msg(
        self, msg: str, plain_text: str | None = None
    ) -> None:
        """
        保存交互事件消息到上下文
        Parameters:
            msg: 助手回复消息
            plain_text: 纯文本版本，暂未使用（保留接口兼容）
        """
        self.chat_history.append({"role": "assistant", "content": msg})

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

    def _get_affinity_stage(self, affinity: int) -> tuple[str, str]:
        """
        根据好感度数值获取阶段名和描述
        Parameters:
            affinity (int): 好感度数值

        Returns:
            (阶段名, 一句话描述)
        """
        for name, threshold, desc in reversed(self.AFFINITY_STAGES):
            if affinity >= threshold:
                return name, desc
        return self.AFFINITY_STAGES[0][0], self.AFFINITY_STAGES[0][2]

    def _get_trust_level(self, trust: int) -> str:
        """根据信任度返回一句话状态"""
        for threshold, desc in reversed(self.TRUST_THRESHOLDS):
            if trust >= threshold:
                return desc
        return self.TRUST_THRESHOLDS[-1][1]

    def _ensure_directory(self):
        """确保配置目录存在，如果不存在则创建"""
        os.makedirs(f"./data/agents/{self.agent_name}", exist_ok=True)
        # 创建数据存储文件夹
        os.makedirs(f"./data/agents/{self.agent_name}/memory", exist_ok=True)
        os.makedirs(f"./data/agents/{self.agent_name}/data_base", exist_ok=True)
