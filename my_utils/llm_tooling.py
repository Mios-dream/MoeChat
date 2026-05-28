"""
LLM 工具调用流程封装。

将 LLM 请求与工具管理解耦，ToolManager 只负责工具注册与执行。
"""

import json
from collections.abc import AsyncIterator

from my_utils import log as Log
from my_utils.llm_request import (
    chat_llm_request_stream,
    llm_request_with_tools,
    llm_request_with_tools_stream,
)
from my_utils.tool_manager import ToolManager


async def _execute_tool_calls(tool_calls: list[dict], messages: list[dict]) -> None:
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        Log.logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")
        result = await ToolManager.execute(tool_name, tool_args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            }
        )


async def stream_chat_with_tools(
    messages: list[dict],
    max_rounds: int = 5,
) -> AsyncIterator[str]:
    """
    流式请求并探测是否触发工具调用。

    - 若未触发工具调用，则直接流式输出文本。
    - 若触发工具调用，则执行工具后流式输出最终文本。
    Parameters:
        messages: 包含对话历史和当前用户消息的消息列表
        max_rounds: 最大工具调用轮次，防止死循环
    """
    tools = ToolManager.get_openai_tools()
    # 如果没有工具，直接使用普通流式请求
    streamed = False
    # 记录工具调用的中间结果
    tool_calls_by_index: dict[int, dict] = {}

    response = await llm_request_with_tools_stream(messages, tools)  # type: ignore

    async for chunk in response:
        if chunk is None or chunk.choices is None or len(chunk.choices) == 0:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta and getattr(delta, "tool_calls", None):
            for call in delta.tool_calls:  # type: ignore
                idx = call.index
                tool_call = tool_calls_by_index.get(
                    idx,
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call.id:
                    tool_call["id"] = call.id
                if call.function and call.function.name:
                    tool_call["function"]["name"] = call.function.name
                if call.function and call.function.arguments:
                    tool_call["function"]["arguments"] += call.function.arguments
                tool_calls_by_index[idx] = tool_call

        if delta and delta.content and not tool_calls_by_index:
            streamed = True
            yield delta.content

    if not tool_calls_by_index:
        if not streamed:
            async for chunk in chat_llm_request_stream(messages):  # type: ignore
                yield chunk
        return

    tool_calls = [
        tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index.keys())
    ]
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    )
    await _execute_tool_calls(tool_calls, messages)

    last_content = None
    for round_idx in range(1, max_rounds):
        response = await llm_request_with_tools(messages, tools)  # type: ignore
        if response is None:
            continue
        message = response.choices[0].message
        last_content = message.content

        if not message.tool_calls:
            if message.content:
                for char in message.content:
                    yield char
            return

        Log.logger.info(
            f"工具调用轮次 {round_idx + 1}: " f"{len(message.tool_calls)} 个工具调用"
        )

        messages.append(message.model_dump())

        normalized_calls = []
        for next_tool_call in message.tool_calls:
            normalized_calls.append(
                {
                    "id": next_tool_call.id,
                    "type": "function",
                    "function": {
                        "name": next_tool_call.function.name,  # type: ignore
                        "arguments": next_tool_call.function.arguments,  # type: ignore
                    },
                }
            )

        await _execute_tool_calls(normalized_calls, messages)

    Log.logger.warning(f"工具调用超过最大轮次 {max_rounds}，强制结束")
    if last_content:
        for char in last_content:
            yield char
