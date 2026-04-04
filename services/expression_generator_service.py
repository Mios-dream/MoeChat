from pathlib import Path
from my_utils.log import logger as Log
from my_utils import config_manager as CConfig
from my_utils.llm_request import llm_request, parse_llm_json_response
import asyncio
import json
import time


class ExpressionGenerator:
    """通过远程 LLM API 生成 Live2D 表情/动作参数。

    该类的核心职责：
    1. 管理当前模型可用参数及其范围信息。
    2. 组织不同任务的提示词（单次表情、动作规划、TTS 连续动作帧）。
    3. 调用 LLM 并解析 JSON 响应。
    4. 对 LLM 输出参数做二次约束（过滤、放大、眼睛二值化、范围裁剪）。

    设计原则：
    - LLM 负责“语义决策”（该做什么表情/动作）。
    - 本地逻辑负责“数值安全”（参数必须可用、可控、在合法范围内）。
    """

    MOUTH_PARAM_HINTS = ("parammouth", "mouth")
    JOINT_PARAM_HINTS = (
        "anglex",
        "angley",
        "anglez",
        "bodyangle",
        "body",
        "head",
        "neck",
        "shoulder",
        "arm",
        "hand",
        "wrist",
        "elbow",
        "forearm",
        "spine",
        "torso",
        "hip",
        "leg",
        "knee",
        "foot",
    )

    def __init__(
        self,
        eye_open_binary: bool = False,
        joint_motion_boost: float = 1.25,
        tts_motion_keep_lip_sync: bool = True,
    ):
        """初始化生成器与运行时调优选项。

        参数：
        - eye_open_binary: 是否将眼睛开闭参数强制二值化（只取最小或最大）。
        - joint_motion_boost: 关节类参数放大倍率，最小限制为 1.0。
        - tts_motion_keep_lip_sync: TTS 连续动作时是否排除嘴部参数，避免破坏口型同步。
        """
        self.config = CConfig.config
        # 当前模型可用参数信息，结构为 {param_id: {name, min, max, default}}
        self.available_parameters: dict[str, dict] = {}
        # 是否启用眼睛开闭参数二值化策略
        self.eye_open_binary = eye_open_binary
        # 关节类参数的放大倍率，默认1.25倍，最小限制为1.0，避免缩小动作幅度
        self.joint_motion_boost = max(1.0, float(joint_motion_boost))
        # TTS 连续动作时是否保持嘴部参数同步，默认为 True，即排除嘴部参数，由口型系统独立控制
        self.tts_motion_keep_lip_sync = tts_motion_keep_lip_sync
        # 模型专属 prompt
        self.custom_prompt: str = ""

        # 新增：缓存机制（LRU风格）
        self._motion_plan_cache: dict[str, list] = {}  # text -> frames
        self._expression_cache: dict[str, dict] = {}  # action_desc -> parameters
        self._cache_max_size = 100  # 最多缓存100个条目

    def update_parameters(self, parameters: dict[str, dict]) -> None:
        """更新当前模型可用参数清单。

        参数结构通常包含：
        - `name`: 参数显示名
        - `min` / `max`: 取值范围
        - `default`: 默认值

        注意：
        - 该方法不会做深层校验，实际数值校验在 `_clamp_parameters` 中完成。
        """
        self.available_parameters = parameters
        print(f"🎭 参数已更新: {len(parameters)} 个参数")

    @classmethod
    def _is_mouth_param(cls, param_id: str) -> bool:
        """判断参数是否属于嘴部相关（用于口型保护与过滤）。"""
        pid = (param_id or "").lower()
        return any(hint in pid for hint in cls.MOUTH_PARAM_HINTS)

    @staticmethod
    def _is_eye_open_param(param_id: str) -> bool:
        """判断参数是否为眼睛开闭类参数。

        规则：
        - 参数名同时包含 `eye` 与 `open`（忽略大小写与下划线）。
        """
        pid = (param_id or "").lower().replace("_", "")
        return "eye" in pid and "open" in pid

    @classmethod
    def _is_joint_motion_param(cls, param_id: str) -> bool:
        """判断参数是否属于可放大的关节/姿态动作参数。

        过滤策略：
        - 明确排除嘴部参数（防止影响口型）。
        - 明确排除眼睛开闭参数（避免与二值化策略冲突）。
        """
        pid = (param_id or "").lower()
        if cls._is_mouth_param(param_id):
            return False
        if cls._is_eye_open_param(param_id):
            return False
        return any(hint in pid for hint in cls.JOINT_PARAM_HINTS)

    def _apply_eye_open_binary(self, value: float, min_v: float, max_v: float) -> float:
        """按配置将眼睛开闭值二值化。

        当 `eye_open_binary=True` 时：
        - 以区间中点为阈值，大于等于中点取最大值，否则取最小值。
        """
        if not self.eye_open_binary:
            return value
        mid = (min_v + max_v) / 2.0
        return max_v if value >= mid else min_v

    def _apply_joint_boost(
        self, param_id: str, value: float, default_v: float
    ) -> float:
        """对关节类参数执行幅度放大。

        计算方式：
        - 以默认值为中心进行线性放大：
          `default + (value - default) * boost`
        - 非关节参数保持原值。
        """
        if not self._is_joint_motion_param(param_id):
            return value
        return default_v + (value - default_v) * self.joint_motion_boost

    def _clamp_parameters(
        self,
        parameters: dict[str, float],
        exclude_mouth: bool = False,
    ) -> dict[str, float]:
        """校验并裁剪 LLM 输出参数，返回安全可用的参数字典。

        处理流程：
        1. 丢弃模型未声明的参数（防止幻觉参数）。
        2. 可选排除嘴部参数（用于 TTS 保持口型同步）。
        3. 读取并兜底 `min/max/default`。
        4. 将值转为 float，无法转换则跳过。
        5. 应用关节放大与眼睛二值化策略。
        6. 最终按 `[min, max]` 裁剪。
        """
        validated = {}
        for param_id, value in (parameters or {}).items():
            if param_id not in self.available_parameters:
                continue
            if exclude_mouth and self._is_mouth_param(param_id):
                continue

            info = self.available_parameters[param_id]
            try:
                min_v = float(info.get("min", -30))
            except (TypeError, ValueError):
                min_v = -30.0
            try:
                max_v = float(info.get("max", 30))
            except (TypeError, ValueError):
                max_v = 30.0
            if min_v > max_v:
                min_v, max_v = max_v, min_v
            try:
                default_v = float(info.get("default", 0))
            except (TypeError, ValueError):
                default_v = 0.0
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue

            # 先应用策略，再做最终区间裁剪。
            num = self._apply_joint_boost(param_id, num, default_v)
            if self._is_eye_open_param(param_id):
                num = self._apply_eye_open_binary(num, min_v, max_v)

            validated[param_id] = max(min_v, min(max_v, num))
        return validated

    def _build_parameter_descriptions(self, exclude_mouth: bool = False) -> str:
        """将可用参数整理为提示词文本片段。

        用途：
        - 给 LLM 明确告知每个参数的 ID、名称及取值范围。
        - 可选排除嘴部参数，便于生成非口型动作。
        """
        rows = []
        for pid, info in self.available_parameters.items():
            if exclude_mouth and self._is_mouth_param(pid):
                continue
            rows.append(
                f"  - {pid}: {info.get('name', pid)}, 范围[{info.get('min', -30)}, {info.get('max', 30)}]"
            )
        return "\n".join(rows)

    def _has_joint_params(self) -> bool:
        """判断当前模型是否包含可识别的关节动作参数。"""
        return any(
            self._is_joint_motion_param(pid) for pid in self.available_parameters.keys()
        )

    def _generate_system_prompt(self) -> str:
        """生成单次表情参数推理的系统提示词。

        关键点：
        - 注入模型参数能力清单。
        - 强制输出固定 JSON 结构。
        - 声明关键参数必须每次都返回。
        - 根据开关动态调整眼睛与关节参数规则。
        - 末尾追加模型专属 prompt（若存在）。
        """
        if not self.available_parameters:
            return "模型参数尚未加载，请稍后再试。"

        param_descriptions = self._build_parameter_descriptions()
        eye_rule = (
            "眼睛开闭类参数必须只输出最大值或最小值。"
            if self.eye_open_binary
            else "眼睛开闭类参数可以输出区间内连续值。"
        )
        joint_rule = (
            "头部/身体/手臂等关节参数可适度放大变化，让动作更明显。"
            if self._has_joint_params()
            else "优先输出与当前模型相关的参数。"
        )
        base_prompt = f"""你是一个 Live2D 虚拟形象的表情控制器。根据场景、对话或情感描述，生成表情参数。
当前模型可用参数：
{param_descriptions}
返回 JSON 格式：
{{
  "parameters": {{
    "参数ID": 数值
  }}
}}

【必须输出的参数】每次生成必须包含以下所有参数的数值：
ParamEyeLOpen, ParamEyeROpen, ParamEyeBallX, ParamEyeBallY,
ParamBrowLY, ParamBrowRY, ParamCheek, ParamAngleX, ParamAngleY, ParamAngleZ,
ParamBodyAngleX, ParamBodyAngleY, ParamBodyAngleZ

要求：
1. 参数组合要自然且可感知，使用参数要足够多样化，避免单一参数主导的表情
2. 眼睛、眉毛、嘴巴、头部角度可组合表达，动作要明显，面部的参数一定要非常丰富，尤其是眼睛眉毛眼球等最细节的面部表情
3. {eye_rule}
4. {joint_rule}
5. 仅返回最必要的参数，减少动作延迟。
6. 除非明确提及，否则不要修改发型等内容。
"""

        # 如果有模型专属 prompt，附加到末尾
        if self.custom_prompt:
            base_prompt += f"\n\n【模型专属规则】\n{self.custom_prompt}"

        return base_prompt

    def _generate_motion_plan_prompt(self) -> str:
        """生成“动作规划”阶段的系统提示词。

        该阶段不直接产出数值参数，而是产出逐帧动作语义描述，
        供后续 `generate_tts_motion_frame_with_plan` 再转换为参数值。
        """
        if not self.available_parameters:
            return "模型参数尚未加载，请稍后再试。"

        # 构建参数能力描述，让 LLM 了解模型可执行的动作维度。
        param_capabilities = []
        for pid, info in self.available_parameters.items():
            name = info.get("name", pid)
            param_capabilities.append(f"  - {name} ({pid})")

        capabilities_text = "\n".join(param_capabilities)

        base_prompt = f"""你是一个 Live2D 动作规划器。你需要为语音播报阶段规划整体的动作序列。
当前模型支持的动作能力：
{capabilities_text}

你的任务是根据语音内容、总时长ms和模型的动作能力，规划每一个序列应该执行的动作。

返回 JSON 格式：
{{
  "motions": [
    {{
      "duration": 1000,
      "action": "微笑并轻轻侧头看向右边"
    }},
    {{
      "duration": 1500,
      "action": "挥手同时身体前倾眼睛眨动"
    }},
    {{
      "duration": 1500,
      "action": "歪头卖萌脸颊泛红"
    }}
  ]
}}

核心规则：
1. 动作分配：根据语音内容的情感密度分配动作时长。例如：
    - "你好"（短促） -> 动作持续 1000ms
    - "今天天气真好啊~"（舒缓） -> 动作持续 2000ms
2. 时间约束：规划的所有动作帧的 duration 总和必须严格等于总时长ms。单个动作帧时间最短不低于1000ms,如果指定的总时长不足以支持至少一个动作帧，则规划一个持续整个时长的动作。
3. 节奏感：避免每个动作时长相同（如全是 1000ms），要有快慢变化。
4. action 描述要充分利用模型的动作能力，包含多个维度的动作组合（10-20个字）：
    - 根据上述参数列表，自由组合各种动作（角度、眼睛、眉毛、嘴巴、脸颊、身体等）
    - 不要局限于固定的动作模式，要根据模型实际支持的参数来设计动作
    - 每一个序列都要有明显的变化，充分展现模型的表现力
5. 动作描述要具体且可执行，例如：
    - "微笑并轻轻点头眼睛半闭看向左边" - 明确指出微笑、点头、眼睛开闭、眼睛看的方向
    - "害羞地低头身体右倾脸颊泛红，左眼睛闭上" - 明确指出害羞表情、头部角度、身体倾斜、脸颊效果
    - "挥手同时侧头微笑眉毛上扬" - 明确指出手部动作、头部角度、表情、眉毛状态
6. 优先使用模型特色动作（如果语音内容适合）
7. 动作连贯性要求：
    - 如果提供了前一个动作状态，新动作要与之保持自然的过渡和连贯性
    - 避免与前一个动作产生突兀的方向或状态变化（如突然从左转向右）
    - 动作变化要有合理的节奏和过渡，可以包含轻微的缓冲动作
    - 必须包含眼睛看的方向，眼睛看的参数可以适当夸张，但要考虑连贯性
8. 只返回 JSON，不要额外解释

注意：充分利用模型的所有参数能力，不要只使用基础的几个参数。每一个序列都应该是独特且富有表现力的，同时保持与前一个动作的自然连贯。
"""

        # 如果有模型专属 prompt，附加到末尾
        if self.custom_prompt:
            base_prompt += f"\n\n【模型专属动作提示】\n{self.custom_prompt}"

        return base_prompt

    async def _call_llm(
        self, request_body: list, log_prefix: str = "🎭 [表情生成]"
    ) -> dict:
        """统一封装 LLM HTTP 调用与响应解析。"""

        start_time = time.time()
        content = await llm_request(request_body)
        elapsed = (time.time() - start_time) * 1000
        print(f"{log_prefix} 完成 ⏱️ {elapsed:.0f}ms")
        if not content:
            print("LLM 返回内容为空")
            return {}
        return parse_llm_json_response(content)

    async def generate(self, input_text: str, context: str = "") -> dict:
        """生成单次表情参数。

        输入：
        - input_text: 用户当前文本（情绪/动作描述）。
        - context: 可选场景背景，用于增强一致性。

        输出：
        - 包含 expression / parameters / duration 的结果字典。

        流程：
        - 组装 system + user 消息。
        - 合并配置中的额外采样参数。
        - 调用 LLM。
        - 对返回参数进行安全裁剪后输出。
        """
        if not self.available_parameters:
            raise ValueError("模型参数尚未加载")

        user_message = (
            f"场景背景：{context}\n\n当前输入：{input_text}" if context else input_text
        )

        Log.info(f"🎭 [表情生成] 调用 API ({self.config['LLM']['model']})...")
        result = await self._call_llm(
            [
                {"role": "system", "content": self._generate_system_prompt()},
                {"role": "user", "content": user_message},
            ]
        )
        result["parameters"] = self._clamp_parameters(result.get("parameters", {}))
        return result

    async def generate_motion_plan(
        self,
        speech_text: str,
        speech_duration_ms: float,
        previous_action: str = "",  # 前一个动作状态
        context: str = "",
        timeout_seconds: float = 1.0,  # 超时参数
    ) -> list[dict]:
        """根据语音内容生成整段动作规划（语义层）。

        输入：
        - speech_text: TTS 即将播报的完整文本。
        - speech_duration_ms: 语音播报时长（毫秒）。
        - previous_action: 前一个动作状态，用于上下文关联。
        - context: 可选场景背景。
        - timeout_seconds: LLM调用超时时间 (默认1秒)。


        输出：
        - `motions` 列表，每项含 `duration` 与 `action` 描述。
        - 如果超时或失败，返回默认的 "自然动作" 列表。
        """

        if not self.available_parameters:
            raise ValueError("模型参数尚未加载")

        # 查询缓存
        cache_key = f"{speech_text}_{speech_duration_ms}_{context}"
        if cache_key in self._motion_plan_cache:
            Log.info(f"📋 [动作规划] 缓存命中: {speech_text[:20]}...")
            return self._motion_plan_cache[cache_key]

        user_message = f"语音内容：{speech_text}\n" f"语音时长：{speech_duration_ms}\n"

        # 添加上下文信息
        if context:
            user_message = f"{context}\n\n{user_message}"

        # 添加前一个动作状态信息
        if previous_action:
            user_message += f"前一个动作状态: {previous_action}\n"

        user_message += f"请为这段语音规划 {speech_duration_ms}ms 动作序列。"
        Log.info(
            f"📋 [动作规划] 调用 API ({self.config['LLM']['model']}) 规划 {speech_duration_ms}ms..."
        )

        try:
            # 使用asyncio.wait_for添加超时保护
            result = await asyncio.wait_for(
                self._call_llm(
                    [
                        {
                            "role": "system",
                            "content": self._generate_motion_plan_prompt(),
                        },
                        {"role": "user", "content": user_message},
                    ],
                    log_prefix="📋 [动作规划]",
                ),
                timeout=timeout_seconds,
            )
            # print(result)
            frames = result.get("motions", [])
            Log.info(f"📋 [动作规划] 共规划 {len(frames)} 个动作序列")

            # 缓存结果
            if len(self._motion_plan_cache) >= self._cache_max_size:
                # 清理最旧的条目
                oldest_key = next(iter(self._motion_plan_cache))
                del self._motion_plan_cache[oldest_key]
            self._motion_plan_cache[cache_key] = frames

            return frames

        except asyncio.TimeoutError:
            Log.warning(f"⚠️  [动作规划] 超时（{timeout_seconds}s），降级为自然动作")
            # 返回降级的默认动作列表
            default_frames = [{"duration": speech_duration_ms, "action": "自然动作"}]
            return default_frames

        except Exception as e:
            Log.error(f"❌ [动作规划] 错误: {e}")
            # 返回降级的默认动作列表
            default_frames = [{"duration": speech_duration_ms, "action": "自然动作"}]
            return default_frames

    async def generate_tts_motion_frame_with_plan(
        self, frame_plans: list[dict], context: str = ""
    ) -> list[dict]:
        """将动作规划中的单帧语义描述转换为具体参数值。

        设计说明：
        - 该方法复用单次表情生成提示词，保持参数风格一致。
        - 输入来自 `generate_motion_plan` 的某一序列的 action。
        - 输出参数会再次经过 `_clamp_parameters`，确保合法。
          由口型系统独立控制嘴部开合。
        - 使用并发处理同时生成多个动作帧，提高性能
        """
        if not self.available_parameters:
            raise ValueError("模型参数尚未加载")

        async def _generate_single_frame(frame_plan: dict) -> dict:
            """生成单个动作帧的参数"""
            if ("duration" not in frame_plan) or ("action" not in frame_plan):
                raise ValueError("每个动作帧必须包含 'duration' 和 'action' 字段")

            action = frame_plan.get("action", "自然动作")

            # 直接使用动作描述作为输入，复用单个表情生成的系统提示词。
            user_message = action
            if context:
                user_message = f"场景背景：{context}\n\n当前输入：{action}"

            result = await self._call_llm(
                [
                    {
                        "role": "system",
                        "content": self._generate_system_prompt(),  # 复用单个表情的系统提示词
                    },
                    {"role": "user", "content": user_message},
                ]
            )

            # 根据配置决定是否过滤嘴部参数。
            result["parameters"] = self._clamp_parameters(
                result.get("parameters", {}),
                exclude_mouth=self.tts_motion_keep_lip_sync,
            )

            # 输出生成的参数内容，便于排查动作幅度与参数覆盖情况。
            params = result.get("parameters", {})

            # 返回完整的动作帧数据
            return {
                "duration": frame_plan["duration"],
                "action": action,
                "parameters": params,
            }

        # 并发处理所有动作帧
        tasks = [_generate_single_frame(frame_plan) for frame_plan in frame_plans]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                Log.error(f"❌ [动作帧生成] 第{i+1}帧生成失败: {result}")
                # 使用默认动作作为降级方案
                processed_results.append(
                    {
                        "duration": frame_plans[i]["duration"],
                        "action": frame_plans[i].get("action", "自然动作"),
                        "parameters": {},
                    }
                )
            else:
                processed_results.append(result)

        return processed_results
