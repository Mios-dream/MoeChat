from datetime import datetime
import time
import asyncio
from typing import Any
import json
from models.dto.interaction_request import InteractionMessageRequest
from my_utils import prompt as prompt_templates
from my_utils.log import logger
from services.assistant_service import AssistantService

from core.chat.base import (
    StreamProcessor,
    text_wrapper,
    tts_wrapper,
    to_sse,
    store_sentence_event,
    drain_ready_sentence_events,
    start_llm_task,
)
from core.chat.v4_motion import (
    V4ChatContext,
    create_scheduler,
    _handle_result,
)

assistant_service = AssistantService()


async def _build_interaction_message_list(
    params: InteractionMessageRequest,
) -> list[dict[str, str]]:
    """构建交互事件的 LLM 消息列表。

    缓存优化策略：
    1. system prompt 完全静态（同一 agent ），放在最前，作为缓存主力
    2. 对话历史放在中间，半静态
    3. 事件描述放在末尾，仅包含语义信息，剥离时间戳和 null 字段
    4. 随机注入风格提示以增加回复多样性
    5. 睡眠模式下追加疲倦语调提示
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise RuntimeError("当前没有加载助手")

    char = agent.agent_config.name
    user = agent.agent_config.user
    description = agent.agent_config.description or ""
    char_personality = agent.agent_config.personality or ""
    examples = agent.agent_config.messageExamples or []
    message_example = "\n".join(examples[:3]) if examples else "无"
    extra_setting = agent.agent_config.customPrompt or ""

    system_prompt = prompt_templates.interaction_event_prompt.format(
        char=char,
        user=user,
        description=description,
        char_personality=char_personality,
        message_example=message_example,
        extra_setting=extra_setting,
    )

    # 睡眠模式下追加疲倦语调提示
    is_sleep_mode = params.context.isSleepMode if params.context else False
    if is_sleep_mode:
        sleep_prompt = prompt_templates.sleep_mode_prompt.format(char=char)
        system_prompt += "\n\n" + sleep_prompt

    # 构建角色设定、性格特质、说话风格等核心信息，具有较强的缓存价值。
    msg_list: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # 添加对话历史
    if params.include_history:
        try:
            recent_turns = agent.memoryEngine.get_recent_chat_turns(
                params.history_limit
            )
            for turn in recent_turns:
                msg_list.append(
                    {
                        "role": turn["role"],
                        "content": turn["content"],
                    }
                )
        except Exception as e:
            logger.warning(f"[交互] 获取对话历史失败: {e}")

    user_message_lines = [
        f"【事件类型】{params.event_type}",
        f"【场景】{params.scene}",
        f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]

    # 梦话事件使用专用场景描述
    if params.event_type == "sleep.talk":
        dream_prompt = prompt_templates.dream_talk_prompt.format(char=char)
        user_message_lines = [
            f"【事件类型】{params.event_type}",
            f"【场景】{dream_prompt}",
            f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]

    user_message = "\n".join(user_message_lines)
    msg_list.append({"role": "user", "content": user_message})

    return msg_list


async def generate_interaction_message(params: InteractionMessageRequest):
    """交互消息生成（文本 + 语音，无动作帧）。"""
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[交互] 当前没有加载助手")
        return

    try:
        msg_list_for_llm = await _build_interaction_message_list(params)
    except Exception as e:
        logger.error(f"[交互] 构建消息列表失败: {e}")
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield to_sse(error_response)
        return

    processor = StreamProcessor()

    sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
    expected_sentence_id = 1
    event_order = ("text", "audio")

    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    tts_semaphore = asyncio.Semaphore(1)

    async def on_sentence_complete(sentence_id: int, sentence_text: str, tts_text: str):
        async def create_text_event():
            payload = await text_wrapper(
                sentence_id=sentence_id,
                sentence_text=sentence_text,
            )
            store_sentence_event(sentence_events, sentence_id, "text", payload)

        async def create_audio_event():
            payload = await tts_wrapper(
                tts_semaphore=tts_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                tts_text=tts_text,
                processor=processor,
            )
            store_sentence_event(sentence_events, sentence_id, "audio", payload)

        track_task(asyncio.create_task(create_text_event()))
        track_task(asyncio.create_task(create_audio_event()))

    processor.register_sentence_complete_callback(on_sentence_complete)

    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        while True:
            expected_sentence_id, ready_payloads = drain_ready_sentence_events(
                sentence_events=sentence_events,
                expected_sentence_id=expected_sentence_id,
                event_order=event_order,
            )
            for payload in ready_payloads:
                yield to_sse(payload)

            if llm_task.done() and not pending_tasks and not sentence_events:
                break

            await asyncio.sleep(0.01)

        final_response = {
            "type": "done",
            "timestamp_ms": time.time() * 1000,
            "total_sentences": max(0, processor.sentence_count - 1),
            "full_text": ("".join(processor.full_msg) if processor.full_msg else ""),
            "done": True,
        }
        yield to_sse(final_response)

        await agent.add_interaction_msg("".join(processor.full_msg))

    except Exception as e:
        llm_task.cancel()
        for task in list(pending_tasks):
            task.cancel()
        logger.error(f"[交互] 处理数据时出错: {e}", exc_info=True)
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield to_sse(error_response)


async def generate_interaction_message_with_motion(params: InteractionMessageRequest):
    """交互消息生成（文本 + 语音 + 动作帧）- 使用V4方案。"""
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[交互] 当前没有加载助手")
        return

    try:
        msg_list_for_llm = await _build_interaction_message_list(params)
    except Exception as e:
        logger.error(f"[交互] 构建消息列表失败: {e}")
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield to_sse(error_response)
        return

    # 创建调度器
    scheduler = create_scheduler()

    # 创建管道
    pipeline = scheduler.create_pipeline(
        system_context=agent.prompt,
        history_messages=msg_list_for_llm[:-1],  # 除最后一条用户消息外的历史
        user_message=msg_list_for_llm[-1]["content"] if msg_list_for_llm else "",
    )

    # 初始化V4上下文
    chat_context = V4ChatContext()

    try:
        # 流式执行管道
        async for result in pipeline.execute():
            # 捕获结果并触发事件
            await _handle_result(chat_context, result)
            # 输出就绪的句子事件
            for payload in chat_context.drain_ready_events():
                yield to_sse(payload)

        # 等待所有异步任务完成（语音合成）
        while chat_context.pending_tasks:
            await asyncio.sleep(0.1)

        # 输出剩余事件
        for payload in chat_context.drain_ready_events():
            yield to_sse(payload)

        # 输出完成信号
        full_text = "".join(
            chat_context.text_cache.get(i, "")
            for i in sorted(chat_context.text_cache.keys())
        )
        yield to_sse(
            {
                "type": "done",
                "timestamp_ms": time.time() * 1000,
                "total_sentences": len(chat_context.text_cache),
                "full_text": full_text,
                "done": True,
            }
        )

        # 保存到助手上下文
        await agent.add_interaction_msg(full_text)

    except Exception as e:
        # 取消所有待处理的任务
        for task in list(chat_context.pending_tasks):
            task.cancel()

        logger.error(f"[交互] 处理数据时出错: {e}", exc_info=True)
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield to_sse(error_response)
