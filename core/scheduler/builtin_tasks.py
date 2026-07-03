"""
内置任务定义

提供开箱即用的任务工厂函数。

内置任务：
- create_text_task(): 文本生成任务
- create_motion_task(): 动作标签任务
- create_bilingual_task(): 双语翻译任务（GSV 非中文时启用）
- create_field_task(): 自定义字段提取任务

每个任务包含：
1. prompt: 提示词片段（注入到系统提示词）
2. parse_fn: 解析函数（从 JSON 中提取数据）
3. field_name: JSON 字段名
"""

from collections.abc import Awaitable, Callable

from core.scheduler.task import Task, TaskResult

# GSV 目标语言映射（textLang → 中文名称）
GSV_LANG_NAMES = {
    "en": "英语",
    "ja": "日语",
}


# 内置任务工厂
def create_text_task(
    callback: Callable[[TaskResult], Awaitable[None]] | None = None,
    priority: int = 100,
) -> Task:
    """
    创建文本生成任务

    提示词：指导 LLM 生成回复文本
    字段名：text
    解析逻辑：提取 JSON 中的 text 字段

    参数：
    - callback: 完成回调（可选）
    - priority: 优先级（默认 100）

    返回：
    - Task 实例

    示例：
    ```python
    task = create_text_task()
    # task.prompt = "生成回复文本"
    # task.field_name = "text"
    # task.parse({"text": "你好"}) -> "你好"
    ```
    """
    return Task(
        name="text_generation",
        type="text",
        prompt="助手的回复文本内容，保持语气、情感和表达风格",
        parse_fn=lambda data: data.get("text", ""),
        field_name="text",
        priority=priority,
        example='{"text": "你好呀~"}',
        rules=["每行必须包含 text 字段"],
    )


def create_motion_task(
    callback: Callable[[TaskResult], Awaitable[None]] | None = None,
    available_actions: str | None = None,
    priority: int = 200,
) -> Task:
    """
    创建动作标签任务

    提示词：指导 LLM 为每个句子生成动作标签
    字段名：actions
    解析逻辑：提取 JSON 中的 actions 字段

    参数：
    - callback: 完成回调（可选）
    - priority: 优先级（默认 200）

    返回：
    - Task 实例

    示例：
    ```python
    task = create_motion_task()
    # task.prompt = "为每个句子生成动作标签"
    # task.field_name = "actions"
    # task.parse({"actions": ["smile", "nod"]}) -> ["smile", "nod"]
    ```
    """
    return Task(
        name="motion_generation",
        type="motion",
        prompt=f"助手应做出的动作或表情（从下方【可用动作列表】中选择）。【可用动作列表】\n{available_actions}",
        parse_fn=lambda data: data.get("actions", []),
        field_name="actions",
        priority=priority,
        example='{"text": "你好呀~", "actions": ["blush"]}',
        rules=[
            "动作标签和表情必须从【可用动作列表】中选择，不要自创动作名",
            "每句话建议 0-2 个动作，仅在必要时使用非必需",
            "动作标签和表情都应放在 actions 数组中",
        ],
    )


def create_bilingual_task(
    target_lang: str = "en",
    callback: Callable[[TaskResult], Awaitable[None]] | None = None,
    priority: int = 150,
) -> Task:
    """
    创建双语翻译任务

    当 GSV 合成语言不是中文（textLang != "zh"）时自动启用。
    LLM 在输出中文回复的同时，额外将文本翻译为目标语言，
    存入 tts_text 字段供 GSV 语音合成使用。

    面向用户展示的依然是中文 text 字段，
    tts_text 仅用于驱动 TTS 引擎以目标语言朗读。

    参数：
    - target_lang: GSV 目标语言代码（如 "en"、"ja"）
    - callback: 完成回调（可选）
    - priority: 优先级（默认 150，介于 text 和 motion 之间）

    返回：
    - Task 实例

    示例：
    ```python
    task = create_bilingual_task(target_lang="en")
    # task.field_name = "tts_text"
    # task.parse({"text": "你好", "tts_text": "Hello"}) -> "Hello"
    ```
    """
    lang_name = GSV_LANG_NAMES.get(target_lang, target_lang)
    examples = {
        "en": "Hello~",
        "ja": "こんにちは〜",
    }

    return Task(
        name="bilingual_translation",
        type="bilingual",
        prompt=(
            f"将每句中文回复同步输出为{lang_name}，存入 tts_text 字段（用于语音合成）。"
            f"保持原文语气、情感和表达风格。"
            f"tts_text 中不需要包含括号及其内容（如（微笑）（小声）等动作描述）。"
        ),
        parse_fn=lambda data: {
            "text": data.get("text", ""),
            "tts_text": data.get("tts_text", ""),
        },
        field_name="tts_text",
        priority=priority,
        example=f'{{"text": "你好呀~", "tts_text": "{examples.get(target_lang, "Hello~")}"}}',
        rules=[
            "tts_text 必须是目标语言的翻译，用于 TTS 语音合成",
            "保持原文的语气、情感和表达风格",
            "tts_text 中不应包含括号标注（动作描述、表情说明等）",
            "text 字段保持中文不变，tts_text 仅影响语音输出",
        ],
    )
