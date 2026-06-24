"""
动作组合引擎

将原子动作标签组合为参数曲线，支持单帧和曲线两种输出格式。

核心功能：
1. 动作标签解析
2. 时间线构建
3. 关键帧合并
4. 参数曲线生成

算法流程：
1. 接收动作标签列表 + 文字时长
2. 按文字时长等比缩放所有动作的时间戳
3. 收集每个参数的所有关键帧到统一时间线
4. 按时间排序，去重
5. 使用 ease-in-out 插值生成平滑曲线
6. 动作结束后保持 3 秒，再用 1 秒回到默认值

时间线结构：
```
|<--- 动作阶段（按文字时长缩放） --->|<-- 3s 保持 -->|<-- 1s 退出 -->|
0s                                text_dur       +3s            +4s
```
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import math

from my_utils.log import logger as Log
from core.expression_generator.atomic_actions import (
    AtomicAction,
    get_action,
    get_action_names,
)


# ============================================================
# 参数默认值
# ============================================================

# 参数默认值（与 Live2D 通用参数对应）
PARAM_DEFAULTS: Dict[str, float] = {
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
    - start: 开始时间（秒），相对于 chunk 开始
    - duration: 持续时间（秒），None 使用模板默认值
    - scale: 幅度缩放系数（0.0-2.0）
    """
    name: str
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
class MotionCurve:
    """
    参数曲线

    属性：
    - duration: 总时长（秒）
    - curves: 参数曲线 {参数ID: [(时间, 值), ...]}
    """
    duration: float
    curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


@dataclass
class MotionFrame:
    """
    单帧动作数据

    属性：
    - duration: 持续时间（毫秒）
    - parameters: 参数字典 {参数ID: 值}
    """
    duration: float
    parameters: Dict[str, float] = field(default_factory=dict)


# ============================================================
# 组合引擎
# ============================================================

class MotionCombiner:
    """
    动作组合引擎

    将原子动作标签组合为参数曲线或单帧数据。

    使用示例：
    ```python
    combiner = MotionCombiner(fps=30.0)

    # 输入动作规格
    action_specs = [
        ActionSpec(name="smile", start=0.0, duration=1.5),
        ActionSpec(name="nod", start=0.5, duration=1.0),
    ]

    # 输出参数曲线
    result = combiner.combine(action_specs, text_duration=3.0)
    print(result.duration)  # 7.0 (3s动作 + 3s保持 + 1s退出)
    print(result.curves)    # {"ParamMouthForm": [(0.0, 0.0), ...], ...}
    ```
    """

    # 保持阶段时长（秒）
    HOLD_DURATION = 3.0
    # 退出阶段时长（秒）
    EXIT_DURATION = 1.0
    # 默认动作时长（秒）
    DEFAULT_ACTION_DURATION = 1.0

    def __init__(self, fps: float = 30.0):
        """
        初始化组合引擎

        参数：
        - fps: 输出曲线的帧率（仅对曲线输出有效）
        """
        self._fps = fps

    def estimate_duration(self, text: str) -> float:
        """
        根据文本估算时长

        参数：
        - text: 输入文本

        返回：
        - 估算的时长（秒）
        """
        # 统计中文字符数
        cn_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        # 统计其他字符数
        other_count = len(text) - cn_count

        # 估算时长：中文约 0.5 秒/字，英文约 0.15 秒/字符
        duration = cn_count * 0.5 + other_count * 0.15

        # 限制范围：2-20 秒
        return max(2.0, min(duration + 1.0, 20.0))

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
        keyframes: List[Tuple[float, float]],
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
        action_specs: List[ActionSpec],
        text_duration: float,
    ) -> Dict[str, List[Keyframe]]:
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
        timeline: Dict[str, List[Tuple[float, float]]] = {}

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
            scale_factor = action_duration / action.duration if action.duration > 0 else 1.0

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
        result: Dict[str, List[Keyframe]] = {}
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

    def _add_hold_and_exit(
        self,
        timeline: Dict[str, List[Keyframe]],
        text_duration: float,
    ) -> Dict[str, List[Keyframe]]:
        """
        添加保持阶段和退出阶段

        参数：
        - timeline: 原始时间线
        - text_duration: 文本时长

        返回：
        - 添加保持和退出后的时间线
        """
        hold_start = text_duration
        exit_start = hold_start + self.HOLD_DURATION
        total_duration = exit_start + self.EXIT_DURATION

        result: Dict[str, List[Keyframe]] = {}

        for param_id, keyframes in timeline.items():
            new_keyframes = list(keyframes)

            # 获取最后的值
            last_value = keyframes[-1].value if keyframes else PARAM_DEFAULTS.get(param_id, 0.0)

            # 添加保持阶段关键帧
            new_keyframes.append(Keyframe(time=hold_start, value=last_value))
            new_keyframes.append(Keyframe(time=exit_start, value=last_value))

            # 添加退出阶段关键帧（回到默认值）
            default_value = PARAM_DEFAULTS.get(param_id, 0.0)
            new_keyframes.append(Keyframe(time=total_duration, value=default_value))

            # 按时间排序
            new_keyframes.sort(key=lambda kf: kf.time)
            result[param_id] = new_keyframes

        return result

    def _sample_curve(
        self,
        timeline: Dict[str, List[Keyframe]],
        total_duration: float,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        对时间线进行采样，生成参数曲线

        参数：
        - timeline: 时间线
        - total_duration: 总时长

        返回：
        - {参数ID: [(时间, 值), ...]} 字典
        """
        result: Dict[str, List[Tuple[float, float]]] = {}
        frame_interval = 1.0 / self._fps

        for param_id, keyframes in timeline.items():
            curve_points = []
            current_time = 0.0

            while current_time <= total_duration:
                # 在关键帧之间插值
                value = self._interpolate_keyframes(
                    [(kf.time, kf.value) for kf in keyframes],
                    current_time,
                )
                curve_points.append((round(current_time, 4), round(value, 4)))
                current_time += frame_interval

            # 确保包含最后一帧
            if curve_points and curve_points[-1][0] < total_duration:
                value = self._interpolate_keyframes(
                    [(kf.time, kf.value) for kf in keyframes],
                    total_duration,
                )
                curve_points.append((round(total_duration, 4), round(value, 4)))

            result[param_id] = curve_points

        return result

    def combine(
        self,
        action_specs: List[ActionSpec] | List[Dict[str, Any]],
        text_duration: float | None = None,
        output_format: str = "curve",
    ) -> MotionCurve | MotionFrame:
        """
        组合动作为参数曲线或单帧

        参数：
        - action_specs: 动作规格列表
        - text_duration: 文本时长（秒），None 时自动估算
        - output_format: 输出格式 ("curve" | "frame")

        返回：
        - MotionCurve（曲线格式）或 MotionFrame（单帧格式）
        """
        # 转换 ActionSpec
        specs = []
        for spec in action_specs:
            if isinstance(spec, dict):
                specs.append(ActionSpec(
                    name=spec.get("act", spec.get("name", "")),
                    start=spec.get("start", 0.0),
                    duration=spec.get("dur", spec.get("duration")),
                    scale=spec.get("scale", 1.0),
                ))
            else:
                specs.append(spec)

        # 估算文本时长
        if text_duration is None:
            text_duration = 5.0  # 默认 5 秒

        # 构建时间线
        timeline = self._build_timeline(specs, text_duration)

        if not timeline:
            Log.warning("[组合引擎] 未生成有效时间线")
            if output_format == "curve":
                return MotionCurve(duration=text_duration + 4.0)
            else:
                return MotionFrame(duration=text_duration * 1000)

        # 添加保持和退出阶段
        full_timeline = self._add_hold_and_exit(timeline, text_duration)
        total_duration = text_duration + self.HOLD_DURATION + self.EXIT_DURATION

        if output_format == "curve":
            # 输出参数曲线
            curves = self._sample_curve(full_timeline, total_duration)
            return MotionCurve(duration=total_duration, curves=curves)
        else:
            # 输出单帧（取第一个时间点的值）
            parameters = {}
            for param_id, keyframes in full_timeline.items():
                if keyframes:
                    parameters[param_id] = keyframes[0].value
            return MotionFrame(duration=text_duration * 1000, parameters=parameters)

    def combine_curve(
        self,
        action_specs: List[ActionSpec] | List[Dict[str, Any]],
        text_duration: float | None = None,
    ) -> MotionCurve:
        """
        组合动作为参数曲线（便捷方法）

        参数：
        - action_specs: 动作规格列表
        - text_duration: 文本时长（秒）

        返回：
        - MotionCurve 实例
        """
        result = self.combine(action_specs, text_duration, output_format="curve")
        return result  # type: ignore

    def combine_frame(
        self,
        action_specs: List[ActionSpec] | List[Dict[str, Any]],
        text_duration: float | None = None,
    ) -> MotionFrame:
        """
        组合动作为单帧数据（便捷方法）

        参数：
        - action_specs: 动作规格列表
        - text_duration: 文本时长（秒）

        返回：
        - MotionFrame 实例
        """
        result = self.combine(action_specs, text_duration, output_format="frame")
        return result  # type: ignore


# ============================================================
# 公开接口
# ============================================================

def create_combiner(fps: float = 30.0) -> MotionCombiner:
    """
    创建组合引擎实例

    参数：
    - fps: 输出曲线的帧率

    返回：
    - MotionCombiner 实例
    """
    return MotionCombiner(fps=fps)
