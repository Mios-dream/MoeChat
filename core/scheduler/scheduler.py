"""
V4 信息调度中心

核心设计思路：
1. 任务（Task）定义了"做什么"和"怎么解析"
2. 调度器（TaskScheduler）负责组合提示词和创建管道
3. 管道（Pipeline）负责执行 LLM 调用和流式解析
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
import time
import asyncio
from typing import Literal
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


class TaskScheduler:
    """
    信息调度中心

    职责：
    1. 管理任务注册表
    2. 管理提示词片段（支持优先级和插入位置）
    3. 组合系统提示词（使用 PromptManager）
    4. 创建处理管道
    """

    def __init__(self):
        """初始化调度器"""
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> "TaskScheduler":
        """注册任务"""
        self._tasks[task.type] = task
        Log.info(f"[调度器] 注册任务: {task.name} (type={task.type})")
        return self

    def remove_task(self, task_type: str) -> bool:
        """移除任务"""
        if task_type in self._tasks:
            del self._tasks[task_type]
            Log.info(f"[调度器] 移除任务: {task_type}")
            return True
        return False

    def _build_task_system_prompt(self) -> str:
        """使用 PromptManager 构建任务系统提示词"""
        fragments: list[str] = ["你需要完成以下任务，并严格输出要求的格式。"]

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
        rules = [
            "每行必须是完整的、合法的 JSON 对象",
            "每行对应一句话，所有字段放在同一个 JSON 对象中",
            "不要输出 JSON 以外的任何内容（不要输出 markdown 代码块、解释说明等）。函数调用（function calling/tool calls）不受此条限制——工具调用和 JSON 文本输出是两个并行通道",
            "注意：对话历史中你的回复格式可能与当前要求的 JSON 格式不同，你必须忽略历史格式，严格按照上述规则以 JSON 格式输出",
        ]

        rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
        fragments.append(f"【严格规则】\n{rules_text}")

        return "\n".join(fragments)

    def create_task_pipeline(
        self,
        user_message: list[ChatCompletionMessageParam],
        system_context: str,
        history_messages: list[ChatCompletionMessageParam] | None = None,
        max_retries: int = 2,
        retry_delay: float = 0,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        tool_handler: ToolCallHandler | None = None,
        max_tool_rounds: int = 10,
        on_tool_event: Callable[[ChatCompletionMessageParam], None] | None = None,
    ) -> "Pipeline":
        """
        创建多行json返回处理管道

        支持 Function Calling 工具调用。
        user_message 支持字符串或多模态内容部分列表。
        on_tool_event: 工具调用/结果实时回调（用于追加到 chat_history）
        """
        task_system_prompt = self._build_task_system_prompt()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
            {"role": "system", "content": task_system_prompt},
        ]

        if history_messages:
            messages.extend(history_messages)

        messages.extend(user_message)

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
            on_tool_event=on_tool_event,
        )

    def create_text_pipeline(
        self,
        user_message: list[ChatCompletionMessageParam],
        system_context: str,
        history_messages: list[ChatCompletionMessageParam] | None = None,
    ) -> "Pipeline":
        """
        创建纯文本处理管道

        user_message 支持字符串或多模态内容部分列表。
        """
        parser = TextStreamParser()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
        ]

        if history_messages:
            messages.extend(history_messages)

        # format_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        messages.extend(user_message)

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
    4. 工具调用内部闭环，不对外暴露工具 TaskResult
    """

    ToolEventCallback = Callable[[ChatCompletionMessageParam], None]

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
        on_tool_event: ToolEventCallback | None = None,
    ):
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
        self.on_tool_event = on_tool_event

    def _build_retry_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """构建重试消息列表"""
        messages_copy = list(messages)

        retry_hint = "\n【重要提醒】你的上一次回复没有包含全部所需字段，或者输出的不是要求的json格式，请确保本次回复必须符合格式要求。每行 JSON 对象都必须包含必要字段。"

        if messages_copy and messages_copy[-1]["role"] == "user":
            last = messages_copy[-1]
            content = last.get("content", "")
            if isinstance(content, list):
                # 多模态内容：追加一个文本部分
                parts = list(content) + [{"type": "text", "text": retry_hint}]
                messages_copy[-1] = {"role": "user", "content": parts}
            else:
                messages_copy[-1] = {
                    "role": "user",
                    "content": f"{content}{retry_hint}",
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
        """执行单轮带重试的流式 LLM 调用"""
        state["completed_tasks"] = set()
        state["tool_calls"] = None

        for attempt in range(self.max_retries + 1):
            stream_messages = list(current_messages)
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

    async def execute(self) -> AsyncGenerator[TaskResult]:
        """执行管道（顶层编排），工具调用内部闭环"""
        start_time = time.time()
        result_count = 0
        current_messages: list[ChatCompletionMessageParam] = list(self.messages)
        effective_tools = self.tools
        total_tool_rounds = 0
        round_state: dict = {}

        for tool_round in range(self.max_tool_rounds + 2):
            async for task_result in self._execute_stream_round(
                current_messages, effective_tools, round_state
            ):
                result_count += 1
                yield task_result

            if round_state["tool_calls"] and self.tool_handler:
                total_tool_rounds = tool_round + 1

                tool_calls = round_state["tool_calls"]

                # 回调：通知外部工具调用事件
                if self.on_tool_event:
                    self.on_tool_event(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                    )

                exec_result = await self.tool_handler(tool_calls)

                # 回调：通知外部工具结果事件
                if self.on_tool_event and exec_result.tool_result_events:
                    for event in exec_result.tool_result_events:
                        self.on_tool_event(
                            {
                                "role": "tool",
                                "tool_call_id": event.call_id,
                                "content": event.content,
                            }
                        )

                # 注入工具消息到上下文（内部闭环）
                current_messages.append(
                    {"role": "assistant", "content": "", "tool_calls": tool_calls}
                )
                current_messages.extend(exec_result.context_messages)

                self.llm_parser.reset()
                self.task_parser.reset(keep_counter=True)

                continue
            if round_state["completed_tasks"] and not round_state["tool_calls"]:
                break

        elapsed = time.time() - start_time
        Log.info(
            f"[管道] 执行完成: {result_count} 个结果, "
            f"{self.task_parser.sentence_count} 个句子, "
            f"工具轮次: {total_tool_rounds}, "
            f"耗时 {elapsed:.2f}s"
        )
