from datetime import datetime
import time
import asyncio
from typing import Any
import json
from models.dto.interaction_request import InteractionMessageRequest
from my_utils import prompt as prompt_templates
from my_utils.log import logger
from services.assistant_service import AssistantService

from core.chat_core import (
    StreamProcessor,
    text_wrapper,
    tts_wrapper,
    motion_wrapper,
    _to_sse,
    _store_sentence_event,
    _drain_ready_sentence_events,
    start_llm_task,
    _get_agent_expression_generator,
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
        yield _to_sse(error_response)
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
            _store_sentence_event(sentence_events, sentence_id, "text", payload)

        async def create_audio_event():
            payload = await tts_wrapper(
                tts_semaphore=tts_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                tts_text=tts_text,
                processor=processor,
            )
            _store_sentence_event(sentence_events, sentence_id, "audio", payload)

        track_task(asyncio.create_task(create_text_event()))
        track_task(asyncio.create_task(create_audio_event()))

    processor.register_sentence_complete_callback(on_sentence_complete)

    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        while True:
            expected_sentence_id, ready_payloads = _drain_ready_sentence_events(
                sentence_events=sentence_events,
                expected_sentence_id=expected_sentence_id,
                event_order=event_order,
            )
            for payload in ready_payloads:
                yield _to_sse(payload)

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
        yield _to_sse(final_response)

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
        yield _to_sse(error_response)


async def generate_interaction_message_with_motion(params: InteractionMessageRequest):
    """交互消息生成（文本 + 语音 + 动作帧）。"""
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[交互] 当前没有加载助手")
        return

    try:
        msg_list_for_llm = await _build_interaction_message_list(params)
        print(json.dumps(msg_list_for_llm, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"[交互] 构建消息列表失败: {e}")
        error_response = {
            "type": "error",
            "timestamp_ms": time.time() * 1000,
            "data": str(e),
            "done": True,
        }
        yield _to_sse(error_response)
        return

    processor = StreamProcessor()

    sentence_events: dict[int, dict[str, dict[str, Any]]] = {}
    expected_sentence_id = 1
    event_order = ("text", "audio", "motion_frame")

    pending_tasks: set[asyncio.Task[Any]] = set()

    def track_task(task: asyncio.Task[Any]):
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    motion_generator = await _get_agent_expression_generator(agent.agent_name)
    if motion_generator is None:
        logger.warning("[交互] 未找到可用 Live2D 参数，将输出空动作事件")

    tts_semaphore = asyncio.Semaphore(1)
    motion_semaphore = asyncio.Semaphore(4)

    async def on_sentence_complete(sentence_id: int, sentence_text: str, tts_text: str):
        async def create_text_event():
            payload = await text_wrapper(
                sentence_id=sentence_id,
                sentence_text=sentence_text,
            )
            _store_sentence_event(sentence_events, sentence_id, "text", payload)

        async def create_audio_event():
            payload = await tts_wrapper(
                tts_semaphore=tts_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                tts_text=tts_text,
                processor=processor,
            )
            _store_sentence_event(sentence_events, sentence_id, "audio", payload)

        async def create_motion_event():
            payload = await motion_wrapper(
                motion_semaphore=motion_semaphore,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                motion_generator=motion_generator,
                motion_history=processor.motion_history,
            )
            _store_sentence_event(sentence_events, sentence_id, "motion_frame", payload)

        track_task(asyncio.create_task(create_text_event()))
        track_task(asyncio.create_task(create_audio_event()))
        track_task(asyncio.create_task(create_motion_event()))

    processor.register_sentence_complete_callback(on_sentence_complete)

    llm_task = asyncio.create_task(start_llm_task(msg_list_for_llm, processor))

    try:
        while True:
            expected_sentence_id, ready_payloads = _drain_ready_sentence_events(
                sentence_events=sentence_events,
                expected_sentence_id=expected_sentence_id,
                event_order=event_order,
            )
            for payload in ready_payloads:
                yield _to_sse(payload)

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
        yield _to_sse(final_response)

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
        yield _to_sse(error_response)
