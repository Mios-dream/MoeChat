from datetime import datetime
from collections.abc import AsyncGenerator

from core.scheduler.builtin_tasks import (
    create_motion_task,
    create_text_task,
    create_bilingual_task,
)
from models.dto.request.interaction_request import InteractionMessageRequest
from models.dto.response.ChatResponse import (
    DoneMessage,
    ErrorMessage,
    FullChatResponse,
)
from my_utils import prompt as prompt_templates
from my_utils.log import logger
from services.assistant_service import AssistantService
from core.chat.v3_motion import V3MotionChatContext
from core.expression_generator.motion_engine_v3 import ACTION_DESCRIPTIONS
from core.expression_generator.utils.expression_loader import load_expressions
from core.scheduler import TaskScheduler
from openai.types.chat import ChatCompletionMessageParam

assistant_service = AssistantService()


def _build_interaction_system_prompt(params: InteractionMessageRequest) -> str:
    """构建交互事件的系统提示词。"""
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
    return system_prompt


def _build_interaction_history(
    params: InteractionMessageRequest,
) -> list[ChatCompletionMessageParam]:
    """构建交互事件的对话历史消息列表（不含当前事件消息）

    使用当前内存中的聊天历史（agent.chat_history），
    因为其中已包含多任务 JSON 格式的回复，能让模型学习输出格式。
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise RuntimeError("当前没有加载助手")
    if not params.include_history:
        return []
    limit = params.history_limit
    history = agent.get_history()
    if limit > 0 and len(history) > limit:
        history = history[-limit:]
    return history


def _build_event_user_message(
    params: InteractionMessageRequest, agent
) -> ChatCompletionMessageParam:
    """构建当前事件的用户消息"""
    if params.event_type == "sleep.talk":
        scene = prompt_templates.dream_talk_prompt.format(char=agent.char)
    else:
        scene = params.scene

    user_message_lines = [
        f"【事件类型】{params.event_type}",
        f"【场景】{scene}",
        f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    return {"role": "user", "content": "\n".join(user_message_lines)}


def _create_interaction_scheduler(
    tts_lang: str = "zh",
    available_actions: str | None = None,
) -> TaskScheduler:
    """
    创建交互事件调度器

    参数：
    - tts_lang: GSV 合成目标语言代码（"zh"/"en"/"ja"），非中文时注册双语翻译任务
    - available_actions: 可用动作和表情列表描述，传入 create_motion_task
    """
    scheduler = TaskScheduler()

    scheduler.add_task(create_text_task(priority=100))

    if tts_lang != "zh" and tts_lang in ("en", "ja"):
        scheduler.add_task(create_bilingual_task(target_lang=tts_lang, priority=150))

    scheduler.add_task(
        create_motion_task(available_actions=available_actions, priority=200)
    )

    return scheduler


async def generate_interaction_message(
    params: InteractionMessageRequest,
) -> AsyncGenerator[FullChatResponse]:
    """
    WebSocket 版交互消息生成管道

    与 generate_interaction_message 逻辑一致，但直接产出 FullChatResponse
    Pydantic 模型（TextResponse / AudioResponse / MotionResponse / DoneResponse），
    供 ChatWebSocketHandler 直接序列化为 WebSocket JSON 发送。

    响应格式与 V3ChatService.chat() 完全一致，客户端可按相同方式消费。

    参数：
    - params: 交互请求参数

    产出：
    - FullChatResponse 模型的异步生成器
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[交互WS] 当前没有加载助手")
        yield ErrorMessage(error_code="NO_ASSISTANT", data="当前没有加载助手")
        return

    try:
        msg_list_for_llm = _build_interaction_history(params)
    except Exception as e:
        logger.error(f"[交互WS] 构建历史消息列表失败: {e}")
        yield ErrorMessage(
            error_code="INTERACTION_BUILD_ERROR",
            data=f"构建历史消息列表失败: {e}",
        )
        return

    tts_lang = agent.agent_config.gsvSetting.textLang

    # 构建可用动作和表情列表，传入动作任务
    expressions = load_expressions(agent.agent_name)
    action_prompt = (
        f"可用动作标签：\n"
        f"{', '.join([f'{action}: {desc}' for action, desc in ACTION_DESCRIPTIONS.items()])}\n"
        f"可用表情：\n"
        f"{[expr.name for expr in expressions]}"
    )

    scheduler = _create_interaction_scheduler(
        tts_lang=tts_lang,
        available_actions=action_prompt,
    )

    pipeline = scheduler.create_task_pipeline(
        user_message=[_build_event_user_message(params, agent)],
        system_context=_build_interaction_system_prompt(params),
        history_messages=msg_list_for_llm,
    )

    chat_context = V3MotionChatContext(tts_lang=tts_lang)

    try:
        async for result in pipeline.execute():
            await chat_context.handle_result(result)

            for payload in chat_context.emit_ready():
                yield payload

        await chat_context.finalize()

        for payload in chat_context.emit_ready():
            yield payload

        full_text = chat_context.get_full_text()
        yield DoneMessage(full_text=full_text)

        # chat_history 保存多任务 JSON 格式，长期记忆保存纯文本
        await agent.add_interaction_msg(
            chat_context.get_raw_output(), plain_text=full_text
        )

    except Exception as e:
        for task in list(chat_context.pending_tasks):
            task.cancel()

        logger.error(f"[交互WS] 处理数据时出错: {e}", exc_info=True)
        yield ErrorMessage(
            error_code="INTERACTION_ERROR",
            data=f"处理数据时出错: {e}",
        )
