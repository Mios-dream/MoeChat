from datetime import datetime
import time
from core.scheduler.builtin_tasks import (
    create_motion_task,
    create_text_task,
    create_bilingual_task,
)
from models.dto.request.interaction_request import InteractionMessageRequest
from my_utils import prompt as prompt_templates
from my_utils.log import logger
from services.assistant_service import AssistantService
from core.chat.base import to_sse
from core.chat.v1 import BaseChatContext
from core.chat.v3_motion import V3MotionChatContext
from core.scheduler import TaskScheduler

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


def _build_interaction_message_list(
    params: InteractionMessageRequest,
) -> list[dict[str, str]]:
    """构建交互事件的 LLM 消息列表。

    1. 包含历史消息
    2. 随机注入风格提示以增加回复多样性
    3. 睡眠模式下追加疲倦语调提示
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        raise RuntimeError("当前没有加载助手")
    msg_list: list[dict[str, str]] = []
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
        dream_prompt = prompt_templates.dream_talk_prompt.format(char=agent.char)
        user_message_lines = [
            f"【事件类型】{params.event_type}",
            f"【场景】{dream_prompt}",
            f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]

    user_message = "\n".join(user_message_lines)
    msg_list.append({"role": "user", "content": user_message})
    return msg_list


def _create_interaction_scheduler(
    with_motion: bool = False,
    tts_lang: str = "zh",
) -> TaskScheduler:
    """
    创建交互事件调度器

    参数：
    - with_motion: 是否包含动作任务
    - tts_lang: GSV 合成目标语言代码（"zh"/"en"/"ja"），非中文时注册双语翻译任务

    返回：
    - 配置好的 TaskScheduler 实例
    """
    scheduler = TaskScheduler()

    # 注册文本任务
    scheduler.add_task(create_text_task(priority=100))

    # 非中文输出时注册双语翻译任务
    if tts_lang != "zh" and tts_lang in ("en", "ja"):
        scheduler.add_task(create_bilingual_task(target_lang=tts_lang, priority=150))

    # 根据需要注册动作任务
    if with_motion:
        scheduler.add_task(create_motion_task(priority=200))

    return scheduler


async def generate_interaction_message(
    params: InteractionMessageRequest,
):
    """
    执行交互事件管道（统一实现）

    参数：
    - params: 交互请求参数
    - with_motion: 是否包含动作帧

    产出：
    - SSE 格式的事件流
    """
    agent = assistant_service.get_current_assistant()
    if not agent:
        logger.error("[交互] 当前没有加载助手")
        return

    try:
        msg_list_for_llm = _build_interaction_message_list(params)
    except Exception as e:
        logger.error(f"[交互] 构建消息列表失败: {e}")
        yield to_sse(
            {
                "type": "error",
                "timestamp_ms": time.time() * 1000,
                "data": str(e),
                "done": True,
            }
        )
        return

    # 获取 GSV 合成语言配置
    tts_lang = agent.agent_config.gsvSetting.textLang

    # 创建调度器（根据是否需要动作和语言配置）
    scheduler = _create_interaction_scheduler(
        with_motion=params.generation_motion,
        tts_lang=tts_lang,
    )
    # 创建管道
    # 注意：系统提示词通过 system_context 传入，历史消息和用户消息通过 history_messages 传入
    pipeline = scheduler.create_task_pipeline(
        system_context=_build_interaction_system_prompt(params),
        history_messages=msg_list_for_llm,
    )

    # 初始化上下文（根据是否需要动作选择不同的 Context）
    if params.generation_motion:
        chat_context = V3MotionChatContext(tts_lang=tts_lang)
    else:
        chat_context = BaseChatContext(event_order=("text", "audio"))

    try:
        # 流式执行管道
        async for result in pipeline.execute():
            # 处理结果
            await chat_context.handle_result(result)

            # 输出就绪的句子事件
            for payload in chat_context.drain_ready_events():
                yield to_sse(payload)

        # 等待所有异步任务完成（语音合成）
        await chat_context.wait_for_completion()

        # 输出剩余事件
        for payload in chat_context.drain_ready_events():
            yield to_sse(payload)

        # 输出完成信号
        full_text = chat_context.get_full_text()
        yield to_sse(
            {
                "type": "done",
                "timestamp_ms": time.time() * 1000,
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
        yield to_sse(
            {
                "type": "error",
                "timestamp_ms": time.time() * 1000,
                "data": str(e),
                "done": True,
            }
        )
