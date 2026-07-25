import asyncio
from datetime import datetime
import os
from my_utils import prompt
from my_utils import config_manager as CConfig
import time
import jionlp
import yaml
from models.types.assistant_info import AssistantInfo
from core.emotion.emotion_engine import EmotionEngine
from concurrent.futures import ThreadPoolExecutor
from core.llm.llm_client import LLMClient
from services import data_base
from services.memory_v2 import MemoryV2
from tool_system.tools.memory_tool import RememberTool, RecallTool, UpdateMemoryTool
from openai.types.chat import ChatCompletionMessageParam
from core.history_manager import HistoryManager
from core.message_chain import MessageChain


class Assistant:
    # 事件驱动情绪引擎
    emotionEngine: EmotionEngine
    # 统一记忆引擎（v2，替代 core_mem + long_mem）
    memoryEngine: MemoryV2
    # 数据知识库实例
    databaseEngine: data_base.DataBase

    def __init__(self, agent_name: str):
        # 助手名称
        self.agent_name = agent_name
        # 聊天记录（使用 HistoryManager 统一管理）
        self.chat_history: HistoryManager = HistoryManager()
        # 线程池执行器，用于处理同步的 CPU 密集任务
        self._executor = ThreadPoolExecutor(max_workers=4)
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

        # 加载事件驱动情绪引擎（含关系状态）
        self.emotionEngine = EmotionEngine(agent_name=self.agent_name)
        self.emotionEngine.configure(
            agent_config=self.agent_config,
            chat_history=self.chat_history,
        )

        # 加载统一记忆引擎 v2（替代旧 long_mem + core_mem）
        self.memoryEngine = MemoryV2(
            self.agent_config,
            firstMeetTime=self.emotionEngine.user_state.firstMeetTime,
        )
        # 注入记忆引擎到记忆工具，使 LLM 可通过工具自主记录、回忆和更新记忆
        RememberTool.set_engine(self.memoryEngine)
        RecallTool.set_engine(self.memoryEngine)
        UpdateMemoryTool.set_engine(self.memoryEngine)
        # 载入知识库
        self.databaseEngine = data_base.DataBase(self.agent_config)

    async def get_dynamic_context(
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
        if db_info or mem_info:
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
        # 好感度与情绪各独立成段，避免被其他内容淹没
        mood_prompt = self.emotionEngine.get_mood_prompt()
        messages.append(
            {
                "role": "system",
                "content": "\n".join(
                    [self.emotionEngine.get_affection_prompt(), mood_prompt]
                ),
            }
        )

        messages.append(
            {
                "role": "system",
                "content": (
                    (f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    + (
                        prompt.sleep_mode_prompt.format(char=self.char)
                        if is_sleep_mode
                        else ""
                    )
                ),
            }
        )

        return messages

    async def build_chat_chain(
        self,
        user_text: str,
        user_message: list[ChatCompletionMessageParam],
        is_sleep_mode=False,
    ) -> MessageChain:
        """
        构建聊天消息链，返回 (system_context, history_messages, user_message)

        优先级划分（每段间隔 100，中间留空便于插入）：
            0:   角色设定 (agent.prompt)
            100:  记忆系统说明
            200:  聊天历史（自动压缩）
            300:  动态上下文
            400:  用户消息
        """
        chain = MessageChain()

        chain.add_system(self.prompt, priority=0)
        chain.add(
            [
                {
                    "role": "system",
                    "content": MemoryV2.build_system_prompt(self.char, self.user),
                }
            ],
            priority=100,
        )
        chain.add_history(self.chat_history, priority=200)
        # 动态上下文（记忆检索 + 知识库等）放在对话历史之后，避免击穿前缀缓存
        dynamic_context = await self.get_dynamic_context(
            msg=user_text, is_sleep_mode=is_sleep_mode
        )
        chain.add(dynamic_context, priority=300)
        chain.add(user_message, priority=400)
        return chain

    async def build_interaction_chain(
        self,
        event_message: list[ChatCompletionMessageParam],
        is_sleep_mode: bool = False,
    ) -> MessageChain:
        """
        构建交互消息链

        优先级划分（每段间隔 100，中间留空便于插入）：
            0:   交互系统提示词
            50:   任务系统提示词（由调度器添加）
            100:  记忆系统说明
            200:  聊天历史（自动压缩）
            300:  额外上下文（好感度 + 情绪）
            400:  用户消息
        """
        # 构建交互系统提示词
        system_prompt = prompt.interaction_event_prompt.format(
            char=self.char,
            user=self.user,
            description=self.description,
            char_personality=self.agent_config.personality,
            message_example=self.message_example,
            extra_description=self.agent_config.customPrompt or "",
        )

        # 睡眠模式下追加疲倦语调提示
        if is_sleep_mode:
            sleep_prompt = prompt.sleep_mode_prompt.format(char=self.char)
            system_prompt += "\n\n" + sleep_prompt

        # 额外上下文（好感度 + 情绪）
        extra_context: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.emotionEngine.get_affection_prompt()
                + self.emotionEngine.get_mood_prompt(),
            },
        ]

        chain = MessageChain()
        chain.add_system(system_prompt, priority=0)
        chain.add(
            [
                {
                    "role": "system",
                    "content": MemoryV2.build_system_prompt(self.char, self.user),
                }
            ],
            priority=100,
        )
        chain.add_history(self.chat_history, priority=200)
        chain.add(extra_context, priority=300)
        chain.add(event_message, priority=400)
        return chain

    async def add_msg(self, user_msg: str, assistant_msg: str) -> None:
        """
        添加对话回合后的后续处理。

        HistoryManager 已自动管理 200 条上限和压缩，
        本方法仅负责非消息存储的后续逻辑：
        1. 好感度更新
        2. 原始对话存储（供日记生成使用）
        3. 跨天日记生成检查

        Parameters:
            user_msg: 用户输入的消息
            assistant_msg: 助手回复的消息
        """
        # 存储原始对话轮次供日记使用
        now_ts = int(time.time())
        if self.enable_long_memory:
            self.memoryEngine.add_chat_turn("user", user_msg, now_ts)
            self.memoryEngine.add_chat_turn("assistant", assistant_msg, now_ts)
            # 跨天日记生成检查（取消旧任务防堆积，异常兜底）
            self.memoryEngine.schedule_diary_check(now_ts)

        await self.emotionEngine.update_affection()

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

    def _ensure_directory(self):
        """确保配置目录存在，如果不存在则创建"""
        os.makedirs(f"./data/agents/{self.agent_name}", exist_ok=True)
        # 创建数据存储文件夹
        os.makedirs(f"./data/agents/{self.agent_name}/memory", exist_ok=True)
        os.makedirs(f"./data/agents/{self.agent_name}/data_base", exist_ok=True)
