"""
动作时序编排器

将 LLM 输出的动作标签列表自动编排为带时序偏移的动作规格列表。
LLM 只需选择动作名称，时序、强度、持续时间由编排器智能分配。

编排规则:
1. emotion 层: 情感基调类动作 (smile/blush/sad 等)
   start=0, duration 覆盖整句, scale 默认 0.8
2. accent 层: 短时强调动作 (nod/wink/shrug 等)
   按 0.4s 间隔自动错开排列，避免堆叠，呼吸窗口 0.2s
3. gaze 层: 视线/微表情动作 (look_*/tilt_* 等)
   start=0, duration 按模板默认值

输入示例:
    ["smile", "nod", "wink_left"]  文字时长 3s

输出示例:
    [
        ActionSpec(name="smile",  stage="emotion", start=0.0, duration=3.0,  scale=0.8),
        ActionSpec(name="nod",    stage="accent",  start=0.8, duration=1.2,  scale=1.0),
        ActionSpec(name="wink_left",stage="accent",start=1.6, duration=0.8,  scale=1.0),
    ]
"""

from core.expression_generator.motion_combiner import ActionSpec
from core.expression_generator.atomic_actions import get_action
from my_utils.log import logger as Log

BREATHING_WINDOW = 0.2
ACCENT_INTERVAL = 0.4
EMOTION_DEFAULT_SCALE = 0.8


class ActionSequencer:
    """
    动作时序编排器

    使用示例:
        sequencer = ActionSequencer()
        specs = sequencer.sequence(
            action_names=["smile", "nod", "wink_left"],
            text_duration=3.0,
        )
        for spec in specs:
            print(f"{spec.name}: start={spec.start}, dur={spec.duration}")
    """

    def sequence(
        self,
        action_names: list[str],
        text_duration: float,
    ) -> list[ActionSpec]:
        """
        编排动作时序

        参数:
        - action_names: 动作标签列表 (如 ["smile", "nod"])
        - text_duration: 文本估算时长（秒）

        返回:
        - 带时序偏移的 ActionSpec 列表
        """
        if not action_names or not text_duration:
            return []

        specs: list[ActionSpec] = []

        # 收集 accent 类动作，用于后续交错排列
        accent_actions: list[str] = []

        for name in action_names:
            action = get_action(name)
            if not action:
                Log.warning(f"[编排器] 未知动作: {name}，跳过")
                continue

            stage = action.stage

            if stage == "emotion":
                emotion_duration = min(action.duration, text_duration)
                specs.append(
                    ActionSpec(
                        name=name,
                        stage=stage,
                        start=0.0,
                        duration=emotion_duration,
                        scale=EMOTION_DEFAULT_SCALE,
                    )
                )

            elif stage == "gaze":
                specs.append(
                    ActionSpec(
                        name=name,
                        stage=stage,
                        start=0.0,
                        duration=None,
                        scale=1.0,
                    )
                )

            elif stage == "accent":
                accent_actions.append(name)

        # accent 类动作交错排列
        if accent_actions:
            accent_specs = self._stagger_accents(accent_actions, text_duration)
            specs.extend(accent_specs)

        return specs

    def _stagger_accents(
        self,
        action_names: list[str],
        text_duration: float,
    ) -> list[ActionSpec]:
        """
        交错排列 accent 类动作

        参数:
        - action_names: accent 类动作名称列表
        - text_duration: 文本估算时长（秒）

        返回:
        - 按间隔排布的 ActionSpec 列表
        """
        specs: list[ActionSpec] = []
        accent_count = len(action_names)

        if accent_count == 0:
            return specs

        # 可用时长：去掉呼吸窗口后的剩余时间
        available_duration = text_duration - BREATHING_WINDOW
        if available_duration <= 0:
            available_duration = text_duration

        # 理想间隔 = 可用时长 / accent 数量
        ideal_interval = available_duration / accent_count
        # 实际间隔：取理想间隔和标准间隔的最小值
        actual_interval = min(ideal_interval, ACCENT_INTERVAL)

        for i, name in enumerate(action_names):
            action = get_action(name)
            if not action:
                continue

            # 起始偏移 = 呼吸窗口 + 间隔 × 序号
            start_offset = BREATHING_WINDOW + actual_interval * i

            # 动作持续时长：不超过可用时长和下一个动作的间隔
            default_duration = action.duration
            if i < accent_count - 1:
                available_for_this = actual_interval
            else:
                # 最后一个动作，用剩余时长
                available_for_this = max(
                    text_duration - start_offset, default_duration
                )

            action_duration = min(default_duration, available_for_this)
            action_duration = max(action_duration, 0.3)

            specs.append(
                ActionSpec(
                    name=name,
                    stage="accent",
                    start=round(start_offset, 2),
                    duration=round(action_duration, 2),
                    scale=1.0,
                )
            )

        return specs
