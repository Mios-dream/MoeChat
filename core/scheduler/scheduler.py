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
from collections.abc import AsyncGenerator, Awaitable, Callable
import time
import asyncio
from typing import Any, Literal
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)

from my_utils.log import logger as Log
from core.llm import LLMClient
from core.llm.response_parser import JsonLineParser, TextParser
from core.scheduler.task import (
    Task,
    TaskResult,
    ToolCallEvent,
    ToolExecutionResult,
)
from core.scheduler.parsers.multi_parser import MultiParser
from core.scheduler.parsers.text_stream_parser import TextStreamParser

ToolCallHandler = Callable[
    [list[ChatCompletionMessageFunctionToolCallParam]],
    Awaitable[ToolExecutionResult],
]

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
        self._tasks: dict[str, Task] = {}

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

    def _build_task_system_prompt(self) -> str:
        """
        使用 PromptManager 构建任务系统提示词

        返回：
        - 完整的系统提示词
        """

        fragments: list[str] = ["你需要完成以下任务，并严格输出要求的格式。"]

        # 工具调用规则（独立章节，在任务列表之前确保 LLM 优先理解）

        # 1. 从任务中收集提示词片段
        sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.priority)

        for index, task in enumerate(sorted_tasks):
            fragments.append(
                f"""【任务{index + 1}: {task.name}】\n{task.prompt}。\n输出字段: {task.field_name}。例如: {task.example}\n【任务规则】\n{"。".join(task.rules)}"""
            )
        fragments.append("""【工具调用规则 - 最高优先级】
如果你判断当前对话需要调用外部工具（如搜索、查询、截图、OCR等）：
1. 可以先用一行 JSON 输出一句简短提示文本，
   如 {"text": "让我帮你搜索一下..."}
2. 然后使用函数调用（function calling）机制实际调用工具
3. 工具结果返回后，继续以 JSON 格式逐句输出包含结果的完整回复
4. 绝对不要在 JSON 文本中表演或模拟工具执行过程，应实际调用工具后根据真实结果回复
不需要调用工具时，直接以 JSON 格式输出回复即可，不受此规则影响。""")
        # 5. 动态组合规则说明（从任务中收集 + 通用规则）
        rules = [
            "每行必须是完整的、合法的 JSON 对象",
            "每行对应一句话，所有字段放在同一个 JSON 对象中",
            "不要输出 JSON 以外的任何内容（不要输出 markdown 代码块、解释说明等）。函数调用（function calling/tool calls）不受此条限制——工具调用和 JSON 文本输出是两个并行通道",
            "注意：对话历史中你的回复格式可能与当前要求的 JSON 格式不同，你必须忽略历史格式，严格按照上述规则以 JSON 格式输出",
        ]

        rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
        fragments.append(f"【严格规则】\n{rules_text}")

        return "\n".join(fragments)

    def _normalize_history_for_json(
        self,
        history_messages: list[ChatCompletionMessageParam] | None,
    ) -> list[ChatCompletionMessageParam]:
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
        # 数组类型字段（如 actions）用 []，字符串类型字段用 ""
        field_defaults: dict[str, object] = {}
        for task in self._tasks.values():
            if task.field_name == "actions":
                field_defaults[task.field_name] = []
            else:
                field_defaults[task.field_name] = ""

        formatted = []
        for msg in history_messages:
            if msg.get("role") == "assistant":
                content: str = msg["content"]  # type: ignore
                stripped = content.strip()
                # 已经是 JSON 格式则跳过
                if stripped.startswith("{") and stripped.endswith("}"):
                    formatted.append(msg)
                else:
                    # 构建包含所有注册字段的完整 JSON
                    json_obj = dict(field_defaults)
                    json_obj["text"] = content
                    wrapped = json.dumps(json_obj, ensure_ascii=False)
                    formatted.append({"role": "assistant", "content": wrapped})
            else:
                formatted.append(msg)
        return formatted

    def create_task_pipeline(
        self,
        user_message: str,
        system_context: str,
        history_messages: list[ChatCompletionMessageParam] | None = None,
        max_retries: int = 2,
        retry_delay: float = 0,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        tool_handler: ToolCallHandler | None = None,
        max_tool_rounds: int = 10,
    ) -> "Pipeline":
        """
        创建多行json返回处理管道
        用于让 LLM 输出多行 JSON，每行对应一句话，包含文本和动作等字段，进行多任务输出。

        支持 Function Calling 工具调用：传入 tools 和 tool_handler 后，
        管道会在 LLM 决定调用工具时自动执行工具并将结果注入上下文，继续生成回复。

        参数：
        - user_message: 用户提问消息
        - system_context: 系统上下文（如角色设定，追加到系统提示词）
        - history_messages: 历史消息列表（可选）
        - max_retries: 最大重试次数（默认 2 次）
        - retry_delay: 重试间隔（秒）
        - tools: OpenAI 工具定义列表（None 表示不使用工具）
        - tool_handler: 工具调用处理器，用于执行工具并返回结果
        - max_tool_rounds: 最大工具调用轮次（默认 10 轮）

        返回：
        - Pipeline 实例
        """
        task_system_prompt = self._build_task_system_prompt()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
            {"role": "system", "content": task_system_prompt},
        ]

        if history_messages:
            messages.extend(self._normalize_history_for_json(history_messages))

        format_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        if user_message:
            messages.append(
                {
                    "role": "user",
                    "content": f"当前时间:{format_time}\n用户对话内容或动作:\n{user_message}",
                }
            )

        parser = MultiParser()
        for task in self._tasks.values():
            parser.register_task(task)

        return Pipeline(
            messages=messages,
            llm_parser=JsonLineParser(),
            task_parser=parser,
            max_retries=max_retries,
            retry_delay=retry_delay,
            tools=tools,
            tool_handler=tool_handler,
            max_tool_rounds=max_tool_rounds,
        )

    def create_text_pipeline(
        self,
        user_message: str,
        system_context: str,
        history_messages: list[ChatCompletionMessageParam] | None = None,
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
        messages: list[ChatCompletionMessageParam] = [
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


class Pipeline:
    """
    流式处理管道

    职责：
    1. 调用 LLM 获取流式响应
    2. 将响应传递给解析器
    3. 产出解析后的结果
    4. 检测任务完成情况，必要时自动重试
    5. 支持 Function Calling 工具调用循环

    支持的解析器：
    - MultiParser: JSON 格式解析（用于 V4 带动作帧的聊天）
    - TextStreamParser: 纯文本格式解析（用于普通聊天和交互）

    使用示例：
    ```python
    # JSON 模式（V4，带工具调用）
    parser = MultiParser()
    parser.register_task(create_text_task())
    pipeline = Pipeline(
        messages=messages,
        llm_parser=JsonLineParser(),
        task_parser=parser,
        tools=tool_defs,
        tool_handler=integration.process_tool_calls,
    )
    async for result in pipeline.execute():
        print(result.task_type, result.data)

    # 纯文本模式
    parser = TextStreamParser()
    pipeline = Pipeline(messages=messages, llm_parser=TextParser(), task_parser=parser)

    async for result in pipeline.execute():
        print(result.task_type, result.data)
    ```
    """

    def __init__(
        self,
        messages: list[ChatCompletionMessageParam],
        llm_parser: JsonLineParser | TextParser,
        task_parser: MultiParser | TextStreamParser,
        model_key: Literal["ChatLLM", "LLM"] = "ChatLLM",
        max_retries: int = 2,
        retry_delay: float = 0,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        tool_handler: ToolCallHandler | None = None,
        max_tool_rounds: int = 10,
    ):
        """
        初始化管道

        参数：
        - messages: 消息列表（包含系统提示词和用户消息）
        - llm_parser: LLM 解析器（JsonLineParser 或 TextParser）
        - task_parser: 任务解析器（MultiParser 或 TextStreamParser）
        - model_key: 模型配置键名
        - max_retries: 最大重试次数（默认 2 次）
        - retry_delay: 重试间隔（秒）
        - tools: OpenAI 工具定义列表（None 表示不使用工具）
        - tool_handler: 工具调用处理器，签名为 async (tool_calls) -> (tool_messages, raw_results)
        - max_tool_rounds: 最大工具调用轮次（默认 10 轮，防止死循环）
        """
        self.messages: list[ChatCompletionMessageParam] = messages
        self.llm_parser = llm_parser
        self.task_parser = task_parser
        self.model_key = model_key
        self.llm_client = LLMClient(model_key=model_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tools = tools
        self.tool_handler = tool_handler
        self.max_tool_rounds = max_tool_rounds

    def _build_retry_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        构建重试消息列表

        在原有消息基础上，添加强化提示词，强调必须输出缺失的任务字段。

        参数：
        - messages: 当前消息列表

        返回：
        - 重试消息列表
        """
        messages_copy = list(messages)

        retry_hint = "\n【重要提醒】你的上一次回复没有包含全部所需字段，或者输出的不是要求的json格式，请确保本次回复必须符合格式要求。每行 JSON 对象都必须包含必要字段。"

        if messages_copy and messages_copy[-1]["role"] == "user":
            messages_copy[-1] = {
                "role": "user",
                "content": f"{messages_copy[-1]['content']}{retry_hint}",
            }
        else:
            messages_copy.append({"role": "user", "content": retry_hint})

        return messages_copy

    async def _execute_stream_round(
        self,
        current_messages: list[ChatCompletionMessageParam],
        effective_tools: list[ChatCompletionFunctionToolParam] | None,
        state: dict,
    ) -> AsyncGenerator[TaskResult]:
        """
        执行单轮带重试的流式 LLM 调用

        负责一完整的工具轮次内的流式输出 + 重试逻辑：
        1. 发起 LLM 流式调用
        2. 逐 chunk 解析：工具调用请求存入 state["tool_calls"]，
           任务解析结果存入 state["completed_tasks"] 并 yield 产出
        3. 若本轮无产出，重试（重置解析器 + 构建重试消息）
        4. 全部重试耗尽后通过 state 的空白标记通知调用方

        参数：
        - current_messages: 当前消息列表
        - effective_tools: 当前轮次有效的工具定义列表
        - state: 可变字典，用于向外传递中间状态（出参），键名：
          - "completed_tools" (set[str]): 本轮完成的任务类型集合
          - "tool_calls" (list | None): 累积的工具调用请求

        产出：
        - TaskResult 实例（解析后的任务结果）
        """
        state["completed_tasks"] = set()
        state["tool_calls"] = None

        for attempt in range(self.max_retries + 1):
            stream_messages = list(current_messages)
            # 如果是重试，则清空状态，在用户消息中追加强化提示词
            if attempt > 0:
                Log.info(f"[管道] 第 {attempt} 次重试...")
                await asyncio.sleep(self.retry_delay)
                state["completed_tasks"].clear()
                state["tool_calls"] = None
                self.llm_parser.reset()
                self.task_parser.reset()
                stream_messages = self._build_retry_messages(current_messages)

            async for chunk in self.llm_client.stream(
                messages=stream_messages,
                parser=self.llm_parser,
                tools=effective_tools,
            ):
                if chunk.tool_calls is not None:
                    state["tool_calls"] = chunk.tool_calls
                elif chunk.parsed_data is not None:
                    for task_result in self.task_parser.parse(chunk.parsed_data):
                        state["completed_tasks"].add(task_result.task_type)
                        yield task_result

            if state["completed_tasks"] or state["tool_calls"]:
                return

            Log.info("[管道] 未检测到任何任务结果，准备重试")

        Log.warning(f"[管道] 重试 {self.max_retries} 次后仍未产出任何任务结果")

    def _build_tool_call_result(
        self,
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam],
    ) -> TaskResult:
        """
        从原始 LLM tool_calls 构建 tool_call TaskResult

        不等待工具执行，让上层 chat() 能立即排出已就绪的句子事件。
        tool_result 在工具执行完成后单独产出。

        参数：
        - tool_calls: LLM 返回的工具调用请求列表

        返回：
        - tool_call TaskResult
        """
        events: list[ToolCallEvent] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            events.append(
                ToolCallEvent(
                    call_id=tc.get("id", ""),
                    tool_name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                )
            )
        return TaskResult(
            task_name="tool_call",
            task_type="tool_call",
            data=events,
        )

    async def execute(self) -> AsyncGenerator[TaskResult]:
        """
        执行管道（顶层编排）

        流程：
        1. 工具调用轮次循环：每轮先请求 LLM（带 tools），根据响应决定是工具调用还是正常输出
        2. 每轮内部包含重试逻辑（JSON 格式验证失败时重试）
        3. 工具调用后自动注入结果并继续

        事件类型：
        - text / motion / bilingual / ...: 通过 task_parser 解析的多任务结果
        - tool_call: 工具调用请求（data 为 tool_calls 列表）
        - tool_result: 工具调用执行结果（data 为 raw_results 列表）

        产出：
        - TaskResult 实例
        """
        start_time = time.time()
        result_count = 0
        current_messages: list[ChatCompletionMessageParam] = list(self.messages)
        effective_tools = self.tools
        total_tool_rounds = 0
        # 记录每轮的工具调用和任务完成情况，供外部判断是否继续下一轮
        round_state: dict = {}

        for tool_round in range(self.max_tool_rounds + 2):
            async for task_result in self._execute_stream_round(
                current_messages, effective_tools, round_state
            ):
                result_count += 1
                yield task_result
            # 如果 LLM 请求了工具调用
            if round_state["tool_calls"] and self.tool_handler:
                total_tool_rounds = tool_round + 1

                # 阶段 1：立即产出 tool_call 事件（不等工具执行），
                # 让上层 chat() 有机会调用 drain_ordered() 释放已就绪的句子事件
                yield self._build_tool_call_result(round_state["tool_calls"])

                # 阶段 2：执行工具（可能耗时很长），产出 tool_result 事件
                exec_result = await self.tool_handler(round_state["tool_calls"])

                if exec_result.tool_result_events:
                    yield TaskResult(
                        task_name="tool_result",
                        task_type="tool_result",
                        data=exec_result.tool_result_events,
                    )

                # 注入工具消息到上下文，准备下一轮 LLM 调用
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": round_state["tool_calls"],
                    }
                )
                current_messages.extend(exec_result.context_messages)

                self.llm_parser.reset()
                self.task_parser.reset(keep_counter=True)

                continue
            # 如果本轮没有工具调用请求，且已经产出任务结果，则结束循环
            if round_state["completed_tasks"] and not round_state["tool_calls"]:
                break

        elapsed = time.time() - start_time
        Log.info(
            f"[管道] 执行完成: {result_count} 个结果, "
            f"{self.task_parser.sentence_count} 个句子, "
            f"工具轮次: {total_tool_rounds}, "
            f"耗时 {elapsed:.2f}s"
        )

    @property
    def sentence_count(self) -> int:
        """已处理的句子数量"""
        return self.task_parser.sentence_count
