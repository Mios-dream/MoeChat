from datetime import datetime
from collections.abc import AsyncGenerator

from models.dto.request.interaction_request import InteractionMessageRequest
from models.dto.response.ChatResponse import (
    DoneMessage,
    ErrorMessage,
    FullChatResponse,
)
from my_utils import prompt as prompt_templates
from my_utils.log import logger
from services.assistant_service import AssistantService
from core.chat.v3_motion import V3ChatService, V3MotionChatContext
from core.expression_generator.utils.expression_loader import load_expressions

assistant_service = AssistantService()


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

    event_message_content = "\n".join(
        [
            f"【事件类型】{params.event_type}",
            f"【场景】{params.scene}",
            f"【当前时间】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
    )

    try:
        chain = await agent.build_interaction_chain(
            event_message=[{"role": "user", "content": event_message_content}],
            is_sleep_mode=params.context.isSleepMode,
        )
    except Exception as e:
        logger.error(f"[交互WS] 构建交互消息链失败: {e}")
        yield ErrorMessage(
            error_code="INTERACTION_BUILD_ERROR",
            data=f"构建交互消息链失败: {e}",
        )
        return

    tts_lang = agent.agent_config.gsvSetting.textLang
    chat_context = V3MotionChatContext(tts_lang=tts_lang)

    expressions = load_expressions(agent.agent_name)
    pipeline = V3ChatService._build_scheduler(
        tts_lang, expressions
    ).create_task_pipeline(chain=chain)

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
        # 触发事件与回复成对写入（EventRecord + InteractionRecord），保证消息链可区分
        await agent.add_interaction_msg(
            chat_context.get_raw_output(), plain_text=full_text, event=params
        )

    except Exception as e:
        for task in list(chat_context.pending_tasks):
            task.cancel()

        logger.error(f"[交互WS] 处理数据时出错: {e}", exc_info=True)
        yield ErrorMessage(
            error_code="INTERACTION_ERROR",
            data=f"处理数据时出错: {e}",
        )
