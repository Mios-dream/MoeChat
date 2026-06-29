"""
V4 信息调度中心

核心设计思路：
1. 任务（Task）定义了"做什么"和"怎么解析"
2. 调度器（TaskScheduler）负责组合提示词和创建管道
3. 管道（Pipeline）负责执行 LLM 调用和流式解析

调用流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建调度器，注册任务                                          │
│     scheduler = TaskScheduler()                                  │
│     scheduler.add_task(create_text_task())                       │
│     scheduler.add_task(create_motion_task())                     │
├─────────────────────────────────────────────────────────────────┤
│  2. 调度器自动组合提示词                                           │
│     system_prompt = 角色设定 + 任务说明 + 输出格式                   │
├─────────────────────────────────────────────────────────────────┤
│  3. 创建管道，执行 LLM 流式调用                                    │
│     pipeline = scheduler.create_pipeline(user_message)           │
│     async for result in pipeline.execute():                      │
│         handle(result)                                           │
├─────────────────────────────────────────────────────────────────┤
│  4. 管道内部：LLM 输出 JSON 行 → 解析器分发 → 产出 TaskResult       │
│     LLM: {"text": "你好", "actions": ["smile"]}                  │
│     解析器: text → "你好", actions → ["smile"]                    │
└─────────────────────────────────────────────────────────────────┘
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from my_utils.log import logger as Log
from core.llm import LLMClient
from core.llm.prompt_manager import PromptManager, PromptTemplate
from core.scheduler.task import Task, TaskResult
from core.scheduler.parsers.multi_parser import MultiParser
from core.scheduler.parsers.text_stream_parser import TextStreamParser
from typing import Any, Literal
from core.llm.response_parser import ResponseParser, JsonLineParser, TextParser

# ============================================================
# 提示词片段定义
# ============================================================


class PromptSection(Enum):
    """
    提示词片段位置

    定义提示词片段在系统提示词中的插入位置。
    数字越小，位置越靠前。
    """

    # 角色设定（最前面）
    ROLE = 100
    # 任务指令
    TASK_INSTRUCTION = 200
    # 任务详细说明（如动作词汇表）
    TASK_DETAIL = 300
    # 输出格式
    OUTPUT_FORMAT = 400
    # 输出示例
    OUTPUT_EXAMPLE = 500
    # 规则说明
    RULES = 600
    # 其他（最后面）
    OTHER = 900


@dataclass
class PromptFragment:
    """
    提示词片段

    属性：
    - section: 插入位置（PromptSection 常量）
    - priority: 同一位置内的优先级（数字越小越靠前）
    - content: 提示词内容
    - source: 来源标识（用于调试）
    """

    section: int
    priority: int
    content: str
    source: str = ""


# ============================================================
# 提示词模板定义
# ============================================================

# 任务指令模板
TASK_INSTRUCTION_TEMPLATE = PromptTemplate(
    name="task_instruction",
    template="""【需要执行的任务】：

{task_instructions}""",
    description="任务指令区域",
    required_vars=["task_instructions"],
)

# 输出格式模板
OUTPUT_FORMAT_TEMPLATE = PromptTemplate(
    name="output_format",
    template="""【输出格式】
每行输出一个 JSON 对象，包含以下字段：
{output_format}""",
    description="输出格式说明",
    required_vars=["output_format"],
)


# ============================================================
# 调度器核心
# ============================================================


class TaskScheduler:
    """
    信息调度中心

    职责：
    1. 管理任务注册表
    2. 管理提示词片段（支持优先级和插入位置）
    3. 组合系统提示词（使用 PromptManager）
    4. 创建处理管道

    使用示例：
    ```python
    # 创建调度器
    scheduler = TaskScheduler()

    # 注册任务
    scheduler.add_task(create_text_task())
    scheduler.add_task(create_motion_task())

    # 添加自定义提示词片段
    scheduler.add_prompt_fragment(PromptFragment(
        section=PromptSection.TASK_DETAIL,
        priority=10,
        content="自定义说明...",
        source="custom"
    ))

    # 创建管道
    pipeline = scheduler.create_pipeline("你好呀~")

    # 流式处理结果
    async for result in pipeline.execute():
        if result.task_type == "text":
            show_text(result.data)
        elif result.task_type == "motion":
            play_motion(result.data)
    ```
    """

    def __init__(self):
        """初始化调度器"""
        # 任务注册表 {task_type: Task}
        self._tasks: dict[str, Task] = {}
        # 提示词片段列表
        self._fragments: list[PromptFragment] = []
        # 提示词管理器
        self._prompt_manager = PromptManager()
        Log.info("[调度器] 初始化完成")

    @property
    def tasks(self) -> dict[str, Task]:
        """获取已注册的任务"""
        return self._tasks.copy()

    def add_task(self, task: Task) -> "TaskScheduler":
        """
        注册任务

        参数：
        - task: 任务定义

        返回：
        - self，支持链式调用
        """
        self._tasks[task.type] = task
        Log.info(f"[调度器] 注册任务: {task.name} (type={task.type})")
        return self

    def remove_task(self, task_type: str) -> bool:
        """
        移除任务

        参数：
        - task_type: 任务类型

        返回：
        - 是否成功移除
        """
        if task_type in self._tasks:
            del self._tasks[task_type]
            Log.info(f"[调度器] 移除任务: {task_type}")
            return True
        return False

    def add_prompt_fragment(self, fragment: PromptFragment) -> "TaskScheduler":
        """
        添加提示词片段

        参数：
        - fragment: 提示词片段

        返回：
        - self，支持链式调用
        """
        self._fragments.append(fragment)
        Log.debug(
            f"[调度器] 添加提示词片段: section={fragment.section}, source={fragment.source}"
        )
        return self

    def _build_task_instructions(self) -> str:
        """
        构建任务指令部分

        将所有任务的提示词组合成编号列表。

        返回：
        - 任务指令字符串
        """
        instructions = []
        # 按优先级排序
        sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.priority)

        for i, task in enumerate(sorted_tasks, 1):
            if task.prompt:
                instructions.append(f"{i}. {task.prompt}")

        return "\n".join(instructions) if instructions else "生成回复文本"

    def _build_output_format(self) -> str:
        """
        构建输出格式说明

        根据注册的任务，生成对应的 JSON 字段说明。

        返回：
        - 输出格式说明字符串
        """
        formats = []

        # 文本字段（始终包含）
        if "text" in self._tasks:
            formats.append('- "text": 回复文本（必须）')

        # 动作字段
        if "motion" in self._tasks:
            formats.append('- "actions": 动作标签列表（可选，如 ["smile", "nod"]）')

        # 其他自定义字段
        for task_type, task in self._tasks.items():
            if task_type not in ("text", "motion") and task.field_name:
                formats.append(
                    f'- "{task.field_name}": {task.prompt or task.field_name}'
                )

        return "\n".join(formats) if formats else '- "text": 回复文本'

    def _build_task_details(self) -> str:
        """
        构建任务详细说明

        为特定任务提供详细的上下文信息，如动作词汇表、选择指南等。
        这些信息帮助 LLM 更好地理解任务要求。

        返回：
        - 任务详细说明字符串
        """
        details = []

        # 动作任务详细说明
        if "motion" in self._tasks:
            try:
                from core.expression_generator.motion_engine_v3 import (
                    get_action_vocab_v3,
                )

                action_vocab = get_action_vocab_v3()
                details.append(action_vocab)
            except ImportError:
                Log.warning("[调度器] 无法导入动作词汇表，跳过动作详细说明")

        return "\n".join(details) if details else ""

    def _collect_fragments(self) -> list[PromptFragment]:
        """
        收集所有提示词片段（包括任务自带的和手动添加的）

        返回：
        - 按 section 和 priority 排序的片段列表
        """
        fragments = []

        # 1. 从任务中收集提示词片段
        sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.priority)
        for i, task in enumerate(sorted_tasks, 1):
            if task.prompt:
                fragments.append(
                    PromptFragment(
                        section=PromptSection.TASK_INSTRUCTION.value,
                        priority=task.priority,
                        content=f"{i}. {task.prompt}",
                        source=f"task:{task.name}",
                    )
                )

        # 2. 添加任务详细说明（如动作词汇表）
        task_details = self._build_task_details()
        if task_details:
            fragments.append(
                PromptFragment(
                    section=PromptSection.TASK_DETAIL.value,
                    priority=100,
                    content=task_details,
                    source="task_details",
                )
            )

        # 3. 添加输出格式说明
        output_format = self._build_output_format()
        fragments.append(
            PromptFragment(
                section=PromptSection.OUTPUT_FORMAT.value,
                priority=100,
                content=OUTPUT_FORMAT_TEMPLATE.render(output_format=output_format),
                source="output_format",
            )
        )

        # 4. 动态组合输出示例（从任务中收集）
        examples = []
        for task in sorted_tasks:
            if task.example:
                examples.append(task.example)
        if examples:
            examples_text = "\n".join(examples)
            fragments.append(
                PromptFragment(
                    section=PromptSection.OUTPUT_EXAMPLE.value,
                    priority=100,
                    content=f"【输出示例】\n{examples_text}",
                    source="output_example",
                )
            )

        # 5. 动态组合规则说明（从任务中收集 + 通用规则）
        rules = [
            "每行必须是完整的、合法的 JSON 对象",
            "每行对应一句话，所有字段放在同一个 JSON 对象中",
            "不要输出 JSON 以外的任何内容（不要输出 markdown 代码块、解释说明等）",
            "注意：对话历史中你的回复格式可能与当前要求的 JSON 格式不同，你必须忽略历史格式，严格按照上述规则以 JSON 格式输出",
        ]
        for task in sorted_tasks:
            if task.rules:
                rules.extend(task.rules)
        rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
        fragments.append(
            PromptFragment(
                section=PromptSection.RULES.value,
                priority=100,
                content=f"【严格规则】\n{rules_text}",
                source="rules",
            )
        )

        # 6. 添加手动注册的片段
        fragments.extend(self._fragments)

        # 按 section 和 priority 排序
        fragments.sort(key=lambda f: (f.section, f.priority))

        return fragments

    def _build_task_system_prompt(self) -> str:
        """
        使用 PromptManager 构建任务系统提示词

        返回：
        - 完整的系统提示词
        """
        # 清空提示词管理器
        self._prompt_manager.clear()

        # 收集所有片段
        fragments = self._collect_fragments()

        # 按 section 分组组合
        current_section = None
        section_parts = []

        for fragment in fragments:
            if fragment.section != current_section:
                # 新的 section，添加之前的内容
                if section_parts:
                    self._prompt_manager.add_system(
                        "\n\n".join(section_parts), append=True
                    )
                current_section = fragment.section
                section_parts = [fragment.content]
            else:
                section_parts.append(fragment.content)

        # 添加最后一个 section
        if section_parts:
            self._prompt_manager.add_system("\n\n".join(section_parts), append=True)

        return self._prompt_manager.system_prompt

    def _normalize_history_for_json(
        self,
        history_messages: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """
        将历史消息中 assistant 角色的纯文本内容转换为 JSON 格式

        根据当前注册的任务动态构建完整的 JSON 字段结构：
        - text 字段填充原始内容
        - 其他字段（如 actions）使用空值填充
        确保模型看到的所有历史 assistant 回复都是完整的 JSON 格式，
        消除纯文本历史对模型 JSON 输出格式的 few-shot 干扰。

        参数：
        - history_messages: 原始历史消息列表

        返回：
        - 格式化后的历史消息列表（assistant 纯文本 → 完整 JSON）
        """
        if not history_messages:
            return []

        # 根据注册的任务构建完整 JSON 模板的默认值
        field_defaults: dict[str, object] = {}
        for task in self._tasks.values():
            if task.field_name == "text":
                field_defaults[task.field_name] = ""  # placeholder，稍后替换
            elif task.field_name == "actions":
                field_defaults[task.field_name] = []
            else:
                field_defaults[task.field_name] = ""

        # 检查是否有 text 之外的字段需要包含
        has_extra_fields = len(field_defaults) > 1

        formatted = []
        for msg in history_messages:
            if msg.get("role") == "assistant":
                content = msg["content"]
                stripped = content.strip()
                # 已经是 JSON 格式则跳过
                if stripped.startswith("{") and stripped.endswith("}"):
                    formatted.append(msg)
                else:
                    if has_extra_fields:
                        # 构建包含所有注册字段的完整 JSON
                        json_obj = dict(field_defaults)
                        json_obj["text"] = content
                        wrapped = json.dumps(json_obj, ensure_ascii=False)
                    else:
                        # 只有 text 任务，简化输出
                        wrapped = json.dumps({"text": content}, ensure_ascii=False)
                    formatted.append({"role": "assistant", "content": wrapped})
            else:
                formatted.append(msg)
        return formatted

    def create_task_pipeline(
        self,
        user_message: str | None = None,
        system_context: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        max_retries: int = 2,
        retry_delay: float = 0,
    ) -> "Pipeline":
        """
        创建多行json返回处理管道
        用于让 LLM 输出多行 JSON，每行对应一句话，包含文本和动作等字段，进行多任务输出。

        参数：
        - user_message: 用户提问消息
        - system_context: 系统上下文（如角色设定，追加到系统提示词）
        - history_messages: 历史消息列表（可选）
        - max_retries: 最大重试次数（默认 2 次）
        - retry_delay: 重试间隔（秒）

        返回：
        - Pipeline 实例
        """
        # 构建系统提示词
        task_system_prompt = self._build_task_system_prompt()

        # 构建消息列表
        messages = [
            # 角色的系统提示词
            {"role": "system", "content": system_context},
            # 任务系统提示词
            {"role": "system", "content": task_system_prompt},
        ]

        # 添加历史消息（将 assistant 纯文本转为 JSON 格式，消除 few-shot 干扰）
        if history_messages:
            messages.extend(self._normalize_history_for_json(history_messages))

        # 格式化时间
        format_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        if user_message:
            # 添加用户消息
            messages.append(
                {
                    "role": "user",
                    "content": f"当前时间:{format_time}\n用户对话内容或动作:\n{user_message}",
                }
            )

        # 创建解析器
        parser = MultiParser()
        for task in self._tasks.values():
            parser.register_task(task)

        # 创建管道
        return Pipeline(
            messages=messages,
            llm_parser=JsonLineParser(),
            task_parser=parser,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def create_text_pipeline(
        self,
        user_message: str | None = None,
        system_context: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
    ) -> "Pipeline":
        """
        创建纯文本处理管道,只能处理文本，不能进行多任务
        用于普通聊天和交互场景，LLM 输出纯文本而非 JSON。

        参数：
        - user_message: 用户提问消息
        - system_context: 系统上下文（如角色设定，追加到系统提示词）
        - history_messages: 历史消息列表（可选）

        返回：
        - Pipeline 实例（使用 TextStreamParser）
        """
        parser = TextStreamParser()

        # 构建消息列表
        messages = [
            # 角色的系统提示词
            {"role": "system", "content": system_context},
        ]

        # 添加历史消息
        if history_messages:
            messages.extend(history_messages)

        # 格式化时间
        format_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        if user_message:
            # 添加用户消息
            messages.append(
                {
                    "role": "user",
                    "content": f"当前时间:{format_time}\n用户对话内容或动作:\n{user_message}",
                }
            )

        return Pipeline(
            messages=messages,
            llm_parser=TextParser(),
            task_parser=parser,
        )


# ============================================================
# 处理管道
# ============================================================


class Pipeline:
    """
    流式处理管道

    职责：
    1. 调用 LLM 获取流式响应
    2. 将响应传递给解析器
    3. 产出解析后的结果
    4. 检测任务完成情况，必要时自动重试

    支持的解析器：
    - MultiParser: JSON 格式解析（用于 V4 带动作帧的聊天）
    - TextStreamParser: 纯文本格式解析（用于普通聊天和交互）

    使用示例：
    ```python
    # JSON 模式（V4）
    parser = MultiParser()
    parser.register_task(create_text_task())
    pipeline = scheduler.create_pipeline("你好呀~")

    # 纯文本模式
    parser = TextStreamParser()
    pipeline = Pipeline(messages=messages, parser=parser)

    async for result in pipeline.execute():
        print(result.task_type, result.data)
    ```
    """

    def __init__(
        self,
        messages: list[dict[str, str]],
        llm_parser: ResponseParser,
        task_parser: MultiParser | TextStreamParser,
        model_key: Literal["ChatLLM", "LLM"] = "ChatLLM",
        max_retries: int = 2,
        retry_delay: float = 0,
    ):
        """
        初始化管道

        参数：
        - messages: 消息列表（包含系统提示词和用户消息）
        - llm_parser: LLM 解析器（JsonLineParser）
        - task_parser: 任务解析器（MultiParser 或 TextStreamParser）
        - model_key: 模型配置键名
        - max_retries: 最大重试次数（默认 2 次）
        - retry_delay: 重试间隔（秒）
        """
        # 消息列表（包含系统提示词和用户消息）
        self.messages = messages

        self.llm_parser = llm_parser
        # 任务解析器可以是 MultiParser 或 TextStreamParser
        self.task_parser = task_parser
        # 模型配置键名
        self.model_key = model_key
        # 最大重试次数
        self.llm_client = LLMClient(model_key=model_key)
        # 重试配置
        self.max_retries = max_retries
        # 重试间隔
        self.retry_delay = retry_delay

    def _build_retry_messages(self) -> list[dict[str, str]]:
        """
        构建重试消息列表

        在原有消息基础上，添加强化提示词，强调必须输出缺失的任务字段。

        参数：
        - missing_tasks: 缺失的任务类型集合

        返回：
        - 重试消息列表
        """
        messages = self.messages.copy()

        retry_hint = f"\n【重要提醒】你的上一次回复没有包含全部所需字段，或者输出的不是要求的json格式，请确保本次回复必须符合格式要求。每行 JSON 对象都必须包含必要字段。"

        # 在最后一条用户消息后追加提醒
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"] + retry_hint,
            }
        else:
            messages.append({"role": "user", "content": retry_hint})

        return messages

    async def execute(self) -> AsyncGenerator[TaskResult, Any]:
        """
        执行管道

        流程：
        1. 将调度器解析器直接传递给 LLM 客户端流式请求
        2. LLM 客户端内部完成 reset / stream_parse / flush 全流程
        3. 管道只负责结果计数、完成检测和重试逻辑

        产出：
        - TaskResult 实例
        """

        start_time = time.time()
        # 记录产出结果数量
        result_count = 0
        # 记录尝试次数
        attempt = 0
        # 记录已完成的任务类型
        completed_tasks: set[str] = set()

        while attempt <= self.max_retries:
            if attempt > 0:
                Log.info(f"[管道] 第 {attempt} 次重试...")
                await asyncio.sleep(self.retry_delay)
                completed_tasks = set()

            # 将解析器直接传入 LLM 客户端，由客户端统一管理解析生命周期
            async for result in self.llm_client.stream(
                messages=self.messages,  # type: ignore
                parser=self.llm_parser,
            ):
                for task_result in self.task_parser.parse(
                    result
                ):  # 将 LLM 输出传给任务解析器,解析为多个任务
                    result_count += 1
                    completed_tasks.add(task_result.task_type)
                    yield task_result

            # 检查是否有任务结果产出
            if completed_tasks:
                break

            Log.info("[管道] 未检测到任何任务结果，准备重试")
            self.messages = self._build_retry_messages()
            attempt += 1

        if not completed_tasks:
            Log.warning(f"[管道] 重试 {attempt} 次后仍未产出任何任务结果")

        elapsed = time.time() - start_time
        Log.info(
            f"[管道] 执行完成: {result_count} 个结果, "
            f"{self.task_parser.sentence_count} 个句子, "
            f"尝试次数: {attempt + 1}, "
            f"耗时 {elapsed:.2f}s"
        )

    @property
    def sentence_count(self) -> int:
        """已处理的句子数量"""
        return self.task_parser.sentence_count
