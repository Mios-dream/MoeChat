"""
动作组合引擎

将原子动作标签组合为关键帧数据，支持单帧和关键帧两种输出格式。

核心功能：
1. 动作标签解析
2. 时间线构建
3. 关键帧合并
4. 关键帧数据生成

算法流程：
1. 接收动作标签列表 + 文字时长
2. 按文字时长等比缩放所有动作的时间戳
3. 收集每个参数的所有关键帧到统一时间线
4. 按时间排序，去重
5. 动作结束后保持 3 秒，再用 1 秒回到默认值

时间线结构：
```
|<--- 动作阶段（按文字时长缩放） --->|<-- 3s 保持 -->|<-- 1s 退出 -->|
0s                                text_dur       +3s            +4s
```
"""

from dataclasses import dataclass, field
from typing import Any

from my_utils.log import logger as Log
from core.expression_generator.atomic_actions import (
    AtomicAction,
    get_action,
    get_action_names,
    check_mutex_conflict,
    resolve_mutex_conflict,
)

# ============================================================
# 参数默认值
# ============================================================

# 参数默认值（与 Live2D 通用参数对应）
PARAM_DEFAULTS: dict[str, float] = {
    "ParamAngleX": 0.0,
    "ParamAngleY": 0.0,
    "ParamAngleZ": 0.0,
    "ParamEyeLOpen": 1.0,
    "ParamEyeROpen": 1.0,
    "ParamEyeLSmile": 0.0,
    "ParamEyeRSmile": 0.0,
    "ParamEyeBallX": 0.0,
    "ParamEyeBallY": 0.0,
    "ParamBrowLForm": 0.0,
    "ParamBrowRForm": 0.0,
    "ParamBrowLAngle": 0.0,
    "ParamBrowRAngle": 0.0,
    "ParamBrowLX": 0.0,
    "ParamBrowLY": 0.0,
    "ParamBrowRX": 0.0,
    "ParamBrowRY": 0.0,
    "ParamMouthForm": 0.0,
    "ParamMouthOpenY": 0.0,
    "ParamCheek": 0.0,
    "ParamBodyAngleX": 0.0,
    "ParamBodyAngleY": 0.0,
    "ParamBodyAngleZ": 0.0,
}


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ActionSpec:
    """
    动作规格

    属性：
    - name: 动作名称
    - stage: 编排层级（emotion / accent / gaze）
    - start: 开始时间（秒），相对于文本起始
    - duration: 持续时间（秒），None 使用模板默认值
    - scale: 幅度缩放系数（0.0~2.0）
    """

    name: str
    stage: str = "emotion"
    start: float = 0.0
    duration: float | None = None
    scale: float = 1.0


@dataclass
class Keyframe:
    """
    关键帧

    属性：
    - time: 时间点（秒）
    - value: 参数值
    """

    time: float
    value: float


@dataclass
class MotionFrame:
    """
    单帧动作数据

    属性：
    - duration: 持续时间（毫秒）
    - parameters: 参数字典 {参数ID: 值}
    """

    duration: float
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class KeyframePoint:
    """
    关键帧点

    属性：
    - time: 时间点（毫秒）
    - value: 参数值
    - ease: 缓动类型
        - "in": 动作开始（从当前位置过渡到目标值）
        - "hold": 动作保持（维持当前值）
        - "out": 动作结束（从当前值过渡回默认值）
    """

    time: int
    value: float
    ease: str = "in"


@dataclass
class MotionKeyframes:
    """
    关键帧动作数据

    属性：
    - duration: 总时长（毫秒）
    - keyframes: 关键帧数据 {参数ID: [KeyframePoint, ...]}
    """

    duration: int
    keyframes: dict[str, list[KeyframePoint]] = field(default_factory=dict)


# ============================================================
# 组合引擎
# ============================================================


# ============================================================
# MotionCombiner 类
# ============================================================


class MotionCombiner:
    """
    动作组合引擎

    将原子动作标签组合为关键帧数据。

    使用示例：
    ```python
    combiner = MotionCombiner()

    action_specs = [
        ActionSpec(name="smile", start=0.0, duration=1.5),
        ActionSpec(name="nod", start=0.5, duration=1.0),
    ]

    result = combiner.combine_keyframes(action_specs, text_duration=3.0)
    print(result.duration)  # 7000 (毫秒)
    print(result.keyframes)  # {"ParamMouthForm": [KeyframePoint(...), ...]}
    ```
    """

    # 默认动作时长（秒）
    DEFAULT_ACTION_DURATION = 1.0

    def estimate_duration(self, text: str) -> float:
        """
        根据文本估算朗读时长（标点感知）

        参数：
        - text: 输入文本

        返回：
        - 估算的时长（秒）
        """
        # 基础朗读速度：中文约 0.35 秒/字
        cn_count = sum(
            1
            for c in text
            if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
        )
        other_count = len(text) - cn_count

        duration = cn_count * 0.35 + other_count * 0.15

        # 标点停顿
        duration += text.count("。") * 0.5
        duration += text.count("！") * 0.5
        duration += text.count("？") * 0.5
        duration += text.count("，") * 0.25
        duration += text.count("、") * 0.2
        duration += text.count("…") * 1.0
        duration += text.count("...") * 1.0

        return max(1.5, min(duration, 30.0))

    def _scale_action_duration(
        self,
        action: AtomicAction,
        specified_duration: float | None,
        text_duration: float,
    ) -> float:
        """
        计算动作的实际时长

        参数：
        - action: 原子动作定义
        - specified_duration: 用户指定的时长（秒）
        - text_duration: 文本时长（秒）

        返回：
        - 实际动作时长（秒）
        """
        if specified_duration is not None:
            return specified_duration

        # 使用模板默认时长，但不超过文本时长
        return min(action.duration, text_duration)

    def _ease_in_out(self, t: float) -> float:
        """
        平滑插值函数（ease-in-out）

        参数：
        - t: 归一化时间 (0.0 - 1.0)

        返回：
        - 平滑后的值 (0.0 - 1.0)
        """
        return t * t * (3.0 - 2.0 * t)

    def _interpolate_keyframes(
        self,
        keyframes: list[tuple[float, float]],
        current_time: float,
    ) -> float:
        """
        在关键帧之间插值

        参数：
        - keyframes: 关键帧列表 [(时间, 值), ...]
        - current_time: 当前时间

        返回：
        - 插值后的参数值
        """
        if not keyframes:
            return 0.0

        # 只有一个关键帧
        if len(keyframes) == 1:
            return keyframes[0][1]

        # 找到当前时间所在的关键帧区间
        for i in range(len(keyframes) - 1):
            t0, v0 = keyframes[i]
            t1, v1 = keyframes[i + 1]

            if t0 <= current_time <= t1:
                # 计算归一化时间
                if t1 == t0:
                    frac = 0.0
                else:
                    frac = (current_time - t0) / (t1 - t0)

                # 应用 ease-in-out 插值
                smooth_frac = self._ease_in_out(frac)
                return v0 + (v1 - v0) * smooth_frac

        # 超出范围，返回最后一个关键帧的值
        return keyframes[-1][1]

    def _build_timeline(
        self,
        action_specs: list[ActionSpec],
        text_duration: float,
    ) -> dict[str, list[Keyframe]]:
        """
        构建参数时间线

        将所有动作的关键帧合并到统一的时间线中。

        参数：
        - action_specs: 动作规格列表
        - text_duration: 文本时长（秒）

        返回：
        - {参数ID: [Keyframe, ...]} 字典
        """
        # 时间线：{参数ID: [(时间, 值), ...]}
        timeline: dict[str, list[tuple[float, float]]] = {}

        for spec in action_specs:
            # 获取动作定义
            action = get_action(spec.name)
            if not action:
                Log.warning(f"[组合引擎] 未知动作: {spec.name}")
                continue

            # 计算动作实际时长
            action_duration = self._scale_action_duration(
                action, spec.duration, text_duration
            )

            # 计算时间缩放因子
            scale_factor = (
                action_duration / action.duration if action.duration > 0 else 1.0
            )

            # 添加关键帧到时间线
            for param_id, keyframes in action.keyframes.items():
                if param_id not in timeline:
                    timeline[param_id] = []

                for kf_time, kf_value in keyframes:
                    # 缩放时间并加上偏移
                    actual_time = spec.start + kf_time * scale_factor
                    # 应用幅度缩放
                    actual_value = kf_value * spec.scale
                    timeline[param_id].append((actual_time, actual_value))

        # 按时间排序并去重
        result: dict[str, list[Keyframe]] = {}
        for param_id, points in timeline.items():
            points.sort(key=lambda x: x[0])
            # 去重（保留同一时间的最后一个值）
            seen_times = set()
            unique_points = []
            for t, v in reversed(points):
                if t not in seen_times:
                    seen_times.add(t)
                    unique_points.append(Keyframe(time=t, value=v))
            unique_points.reverse()
            result[param_id] = unique_points

        return result

    def combine_keyframes(
        self,
        action_specs: list[ActionSpec] | list[dict[str, Any]],
        text_duration: float | None = None,
    ) -> MotionKeyframes:
        """
        组合动作为动作阶段关键帧数据

        只输出动作阶段（0 到 text_duration）的关键帧，
        hold 和 exit 阶段由前端根据音频时长自行处理。

        会自动检测并解决互斥动作冲突。

        参数：
        - action_specs: 动作规格列表
        - text_duration: 文本估算时长（秒）

        返回：
        - MotionKeyframes 实例（duration=动作阶段时长）
        """
        # 转换 ActionSpec
        specs = []
        for spec in action_specs:
            if isinstance(spec, dict):
                specs.append(
                    ActionSpec(
                        name=spec.get("act", spec.get("name", "")),
                        stage=spec.get("stage", "emotion"),
                        start=spec.get("start", 0.0),
                        duration=spec.get("dur", spec.get("duration")),
                        scale=spec.get("scale", 1.0),
                    )
                )
            else:
                specs.append(spec)

        # 检测并解决互斥冲突
        action_names = [s.name for s in specs]
        conflicts = check_mutex_conflict(action_names)
        if conflicts:
            Log.warning(f"[组合引擎] 检测到互斥冲突: {conflicts}")
            resolved_names = resolve_mutex_conflict(action_names)
            specs = [s for s in specs if s.name in resolved_names]
            Log.info(f"[组合引擎] 冲突解决后: {[s.name for s in specs]}")

        # 估算文本时长
        if text_duration is None:
            text_duration = 5.0

        # 构建动作阶段时间线（0 到 text_duration）
        timeline = self._build_timeline(specs, text_duration)

        if not timeline:
            Log.warning("[组合引擎] 未生成有效时间线")
            return MotionKeyframes(duration=int(text_duration * 1000))

        # 转换为关键帧输出格式
        keyframes_output: dict[str, list[KeyframePoint]] = {}

        for param_id, kf_list in timeline.items():
            points: list[KeyframePoint] = []
            for i, kf in enumerate(kf_list):
                time_ms = int(round(kf.time * 1000))
                value = round(kf.value, 4)

                # 合并连续相同值的关键帧
                if points and points[-1].value == value:
                    continue

                # 避免重复时间点
                if points and points[-1].time == time_ms:
                    continue

                ease = "in" if i == 0 else "hold"

                points.append(KeyframePoint(time=time_ms, value=value, ease=ease))

            if points:
                keyframes_output[param_id] = points

        return MotionKeyframes(
            duration=int(round(text_duration * 1000)),
            keyframes=keyframes_output,
        )
