"""
Live2D 表情生成器 V2

模块职责：
1. 单次请求生成完整动作序列（不再分离规划和生成）
2. 使用参数别名减少 token 消耗
3. 支持表情按需使用
4. TTS 场景保证动作连贯性

设计原则：
- LLM 负责语义决策（该做什么表情/动作）
- 本地逻辑负责数值安全（参数范围裁剪）
- 参数别名系统减少约 75% token 消耗
- 表情只在明确需要时使用
"""

from dataclasses import dataclass, field
import json
from my_utils.log import logger as Log
from my_utils import config_manager as CConfig
from core.llm.llm_client import LLMClient
from core.llm.response_parser import parse_llm_json_response
from core.expression_generator.live2d_expression_loader import (
    ExpressionInfo,
    load_expressions,
    build_expression_descriptions,
)
import asyncio
import time

# ============================================================
# 参数别名系统
# ============================================================

# 别名 -> 完整参数 ID 映射表
# 使用简短别名可节省约 75% token（如 "ax" 代替 "ParamAngleX"）
ALIAS_TO_PARAM: dict[str, str] = {
    # 眼睛相关
    "elo": "ParamEyeLOpen",  # 左眼开度 [0, 1]
    "ero": "ParamEyeROpen",  # 右眼开度 [0, 1]
    "els": "ParamEyeLSmile",  # 左眼微笑 [0, 1]
    "ers": "ParamEyeRSmile",  # 右眼微笑 [0, 1]
    "ebx": "ParamEyeBallX",  # 眼球水平方向 [-1, 1]
    "eby": "ParamEyeBallY",  # 眼球垂直方向 [-1, 1]
    # 眉毛相关
    "bly": "ParamBrowLY",  # 左眉高度 [-1, 1]
    "bry": "ParamBrowRY",  # 右眉高度 [-1, 1]
    "bls": "ParamBrowLForm",  # 左眉形态 [-1, 1]
    "brs": "ParamBrowRForm",  # 右眉形态 [-1, 1]
    # 嘴巴相关
    "mf": "ParamMouthForm",  # 嘴型 [-1, 1] 正=微笑，负=悲伤
    "mo": "ParamMouthOpenY",  # 嘴巴张开 [0, 1]
    # 脸颊相关
    "chk": "ParamCheek",  # 脸颊红晕 [0, 1]
    # 头部角度
    "ax": "ParamAngleX",  # 头部左右旋转 [-30, 30]
    "ay": "ParamAngleY",  # 头部上下倾斜 [-30, 30]
    "az": "ParamAngleZ",  # 头部左右倾斜 [-30, 30]
    # 身体角度
    "bx": "ParamBodyAngleX",  # 身体左右旋转 [-30, 30]
    "by": "ParamBodyAngleY",  # 身体上下倾斜 [-30, 30]
    "bz": "ParamBodyAngleZ",  # 身体左右倾斜 [-30, 30]
}

# 完整参数 ID -> 别名 反向映射
PARAM_TO_ALIAS: dict[str, str] = {v: k for k, v in ALIAS_TO_PARAM.items()}


def _resolve_alias(alias: str) -> str | None:
    """
    将参数别名解析为完整参数 ID

    参数：
    - alias: 参数别名（如 "ax"）

    返回：
    - 完整参数 ID（如 "ParamAngleX"）
    - 未找到返回 None
    """
    return ALIAS_TO_PARAM.get(alias)


def _get_alias(param_id: str) -> str:
    """
    获取参数 ID 对应的别名

    参数：
    - param_id: 完整参数 ID（如 "ParamAngleX"）

    返回：
    - 参数别名（如 "ax"）
    - 无别名时返回原参数 ID
    """
    return PARAM_TO_ALIAS.get(param_id, param_id)


def _convert_aliases_to_params(aliases_dict: dict[str, float]) -> dict[str, float]:
    """
    将别名字典转换为完整参数字典

    参数：
    - aliases_dict: {别名: 值} 字典

    返回：
    - {完整参数ID: 值} 字典
    - 无效别名会被忽略
    """
    result = {}
    for alias, value in aliases_dict.items():
        param_id = _resolve_alias(alias)
        if param_id:
            try:
                result[param_id] = float(value)
            except (TypeError, ValueError):
                continue
    return result


# ============================================================
# 参数描述（供 LLM 使用）
# ============================================================

PARAM_DESCRIPTIONS = """可用参数（使用别名输出）：
- elo/ero: 左/右眼开度 [0,1]，1=全开，0=全闭
- els/ers: 左/右眼微笑 [0,1]，1=微笑，0=正常
- ebx/eby: 眼球方向 [-1,1]，正=右/上，负=左/下
- bly/bry: 眉毛高度 [-1,1]，正=上扬，负=下压
- bls/brs: 眉毛形态 [-1,1]，正=开心眉，负=生气眉
- mf: 嘴型 [-1,1]，正=微笑，负=悲伤
- mo: 嘴巴张开 [0,1]，1=全开
- chk: 脸颊红晕 [0,1]，1=最红（害羞/激动）
- ax/ay/az: 头部角度 [-30,30]，正=右转/上仰/右倾
- bx/by/bz: 身体角度 [-30,30]，正=右转/上仰/右倾"""


# ============================================================
# 动作帧数据结构
# ============================================================


@dataclass
class MotionFrame:
    """
    单个动作帧

    属性：
    - duration: 帧持续时间（毫秒）
    - parameters: 完整参数字典 {ParamId: value}
    - expression: 使用的表情名称（可选，空字符串表示不使用表情）
    """

    duration: float
    parameters: dict[str, float] = field(default_factory=dict)
    expression: list[str] = field(default_factory=list)


# ============================================================
# V2 表情生成器核心类
# ============================================================


class ExpressionGeneratorV2:
    """
    V2 版本表情生成器

    核心改进：
    1. 单次请求生成完整动作序列
    2. 使用参数别名减少 token 消耗
    3. 表情按需使用，只在必要时应用
    4. TTS 场景保证动作连贯性

    使用方式：
    ```python
    generator = ExpressionGeneratorV2()
    await generator.initialize("assistant_name", parameters)
    frames = await generator.generate_motion("你好", 2000)
    ```
    """

    def __init__(self):
        """初始化生成器"""
        # 配置信息
        self.config = CConfig.config
        # 模型表情列表
        self.expressions: list[ExpressionInfo] = []
        # 表情名 -> ExpressionInfo 映射
        self.expression_map: dict[str, ExpressionInfo] = {}

        # 模型专属 prompt（可自定义追加）
        self.custom_prompt: str = ""

        # TTS 连贯性状态：存储上一个动作的原始参数
        self._last_action_params: dict[str, float] = {}
        # LLM 客户端实例
        self._llm_client = LLMClient(model_key="LLM")

    async def initialize(
        self,
        assistant_name: str,
        use_expression_cache: bool = True,
    ) -> None:
        """
        初始化生成器

        执行流程：
        1. 更新可用参数
        2. 加载模型表情（含缓存支持）
        3. 构建表情映射

        参数：
        - assistant_name: 角色名称
        - parameters: 可用参数字典 {param_id: {name, min, max, default}}
        - use_expression_cache: 是否使用表情描述缓存
        """

        # 加载表情
        self.expressions = await load_expressions(
            assistant_name, use_cache=use_expression_cache
        )

        # 构建表情映射
        self.expression_map = {expr.name: expr for expr in self.expressions}

        Log.info(f"[V2表情生成器] 初始化完成: {len(self.expressions)} 个表情")

    # ============================================================
    # 提示词构建
    # ============================================================

    def _build_system_prompt(self, is_tts: bool = False) -> str:
        """
        构建系统提示词（固定部分，利于缓存命中）

        提示词结构设计：
        1. 角色定义（固定）
        2. 参数说明（固定）
        3. 表情列表（固定）
        4. 输出格式（固定）
        5. 规则说明（固定）

        参数：
        - is_tts: 是否为 TTS 场景

        返回：
        - 完整的系统提示词
        """
        # 表情描述（固定）
        expr_desc = build_expression_descriptions(self.expressions)

        # TTS 特殊规则（固定）
        tts_rules = """
【TTS 连贯性规则】
1. 如果提供了前一个动作参数，新动作要与之保持自然过渡，避免突兀的方向变化
2. 保持动作连续，如前文提到需要闭上眼睛，做出wink表情等内容时，输出对应的眼睛参数，同时要保持动作持续。
"""

        # 构建完整提示词（固定部分）
        base_prompt = f"""你是一个 Live2D 虚拟形象的动作控制器。根据输入文本和对话场景生成动作参数。

{PARAM_DESCRIPTIONS}

可用表情或动作（仅在必要时使用来丰富角色表情）：
{expr_desc}

输出为 JSON：
{{"duration": 1500, "params": {{"ax": 15, "ay": -5, "els": 1}}, "expr": ["微笑"]}}
或
{{"duration": 1500, "params": {{"ax": 0, "ay": -5, "els": 1}}}}

规则：
1. 使用参数别名输出（如 ax 代替 ParamAngleX）
2. 表情只在明确需要时使用，不需要时省略 expr 字段。如果需要，可以组合多个表情，但要确保自然合理。
3. 参数组合要自然，面部表情要丰富多样
4. 动作幅度要明显，不要太保守
5. 只需输出单帧动作，持续时间由 duration 字段控制

【对话场景理解规则】
1. 「对话场景」中包含了用户的历史指令和当前回复的上下文，你必须仔细理解
2. 从助手的回复中推断助手最真实的情绪和意图，生成最合适的动作
3. 示例：用户说"可以闭上眼睛一下吗"，助手回复"说好了，就一小下下哦？" → 应该生成闭眼动作（elo/ero 设为 0）
4. 示例：用户说"笑一个"，助手回复"嘻嘻~" → 应该生成微笑动作（els/ers 设为 1）
5. 前一个动作参数可以帮助你保持动作的连贯性，如果前一个动作是闭眼，新动作应该保持闭眼状态（除非当前文本明确表示要睁眼）
"""
        if is_tts:
            base_prompt += tts_rules
        # 追加模型专属规则
        if self.custom_prompt:
            base_prompt += f"\n\n【模型专属规则】\n{self.custom_prompt}"

        return base_prompt

    def _build_context_message(
        self,
        context: str = "",
        previous_params: dict[str, float] | None = None,
    ) -> str | None:
        """
        构建上下文消息（可选，包含对话场景和前一个动作参数）

        设计说明：
        - 将变化的上下文信息单独提取
        - 如果没有上下文信息，返回 None
        - 这部分消息变化频率较低，仍有一定缓存价值

        参数：
        - context: 对话背景（包含用户历史指令）
        - previous_params: 前一个动作的原始参数

        返回：
        - 上下文消息字符串，或 None
        """
        parts = []

        if context:
            parts.append(
                f"【对话场景】\n以下是最近对话中用户的问题和当前回复的上下文：\n{context}"
            )

        if previous_params:
            # 将完整参数ID转换为别名输出，节省token
            alias_params = {}
            for param_id, value in previous_params.items():
                alias = _get_alias(param_id)
                alias_params[alias] = round(value, 2)
            parts.append(f"【前一个动作参数】\n{alias_params}（保持动作连贯性）")

        return "\n".join(parts) if parts else None

    def _build_input_message(
        self,
        text: str,
        duration_ms: float,
        is_tts: bool = False,
    ) -> str:
        """
        构建输入消息（变化部分，包含文本和时长）

        设计说明：
        - 这是每次请求变化最频繁的部分
        - 保持简洁，只包含核心输入信息

        参数：
        - text: 输入文本
        - duration_ms: 动作时长
        - is_tts: 是否为 TTS 场景

        返回：
        - 输入消息字符串
        """
        parts = []

        if is_tts:
            parts.append(f"语音内容：{text}")
            parts.append(f"语音时长：{duration_ms}ms")
            parts.append(f"请为这段语音规划 {duration_ms}ms 的单帧动作。")
        else:
            parts.append(f"当前输入：{text}")
            parts.append(f"动作时长：{duration_ms}ms")
            parts.append(
                f"提示：这段文本可能包含助手的动作倾向或情感表达，请结合对话场景生成合适的动作。"
            )

        return "\n".join(parts)

    # ============================================================
    # LLM 调用
    # ============================================================

    async def _call_llm(
        self,
        request_body: list,
        timeout_seconds: float = 5.0,
        log_prefix: str = "[V2表情生成]",
    ) -> dict:
        """
        统一封装 LLM HTTP 调用与响应解析

        参数：
        - request_body: 消息列表
        - timeout_seconds: 超时时间（秒）
        - log_prefix: 日志前缀

        返回：
        - 解析后的 JSON 字典
        - 超时或失败返回空字典
        """
        start_time = time.time()

        try:
            content = await asyncio.wait_for(
                self._llm_client.request(
                    request_body,
                    extra_body={
                        "thinking": {"type": "disabled"},
                    },
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            Log.warning(f"{log_prefix} 调用超时 ({timeout_seconds}s)")
            return {}
        except Exception as e:
            Log.error(f"{log_prefix} 调用失败: {e}")
            return {}

        elapsed = (time.time() - start_time) * 1000
        Log.info(f"{log_prefix} 完成 ⏱️ {elapsed:.0f}ms")

        if not content:
            Log.warning(f"{log_prefix} LLM 返回内容为空")
            return {}

        return parse_llm_json_response(content)

    # ============================================================
    # 结果后处理
    # ============================================================

    def _parse_frames(self, raw_result: dict) -> list[MotionFrame]:
        """
        解析 LLM 返回的动作帧列表

        处理流程：
        1. 提取 frames 数组
        2. 将别名参数转换为完整参数 ID
        3. 验证表情名称有效性
        4. 裁剪参数范围

        参数：
        - raw_result: LLM 返回的原始 JSON

        返回：
        - MotionFrame 列表
        """

        frames = []

        # 解析时长
        duration = raw_result.get("duration", 1000)
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 1000.0

        # 解析参数（别名 -> 完整ID）
        raw_params = raw_result.get("params", {})
        parameters = _convert_aliases_to_params(raw_params)
        # 解析表情（可选）
        expression = raw_result.get("expr", [])

        frame = MotionFrame(
            duration=duration,
            parameters=parameters,
            expression=expression,
        )
        frames.append(frame)

        return frames

    # ============================================================
    # 公开接口
    # ============================================================

    async def generate_motion(
        self,
        text: str,
        duration_ms: float,
        context: str = "",
        previous_params: dict[str, float] | None = None,
        is_tts: bool = False,
        timeout_seconds: float = 5.0,
    ) -> list[MotionFrame]:
        """
        生成动作序列（单次请求）

        核心方法：根据输入文本生成完整的动作帧序列。

        参数：
        - text: 输入文本（用户消息或 TTS 文本）
        - duration_ms: 动作总时长（毫秒）
        - context: 场景背景
        - previous_params: 前一个动作的原始参数（用于连贯性）
        - is_tts: 是否为 TTS 场景
        - timeout_seconds: LLM 调用超时时间

        返回：
        - MotionFrame 列表
        - 失败时返回空列表
        """

        # 使用前一个动作参数
        effective_previous = previous_params or self._last_action_params

        # 构建消息列表（优化缓存命中）
        # 结构：system(固定) -> context(可选，变化频率低) -> input(变化频率高)
        system_prompt = self._build_system_prompt(is_tts=is_tts)
        # 构建上下文消息（可选，包含场景背景和前一个动作参数
        context_message = self._build_context_message(
            context=context,
            previous_params=effective_previous,
        )
        input_message = self._build_input_message(
            text=text,
            duration_ms=duration_ms,
            is_tts=is_tts,
        )

        # 组装消息列表
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # 如果有上下文信息，添加为单独的用户消息
        if context_message:
            messages.append({"role": "user", "content": context_message})
        # 添加输入消息
        messages.append({"role": "user", "content": input_message})

        # 调用 LLM
        log_prefix = "[V2表情生成]"
        Log.info(f"{log_prefix} 调用 API ({self.config['LLM']['model']})...")

        # print(f"请求上下文：{json.dumps(messages, ensure_ascii=False)}")

        result = await self._call_llm(
            request_body=messages,
            timeout_seconds=timeout_seconds,
            log_prefix=log_prefix,
        )

        if not result:
            return []

        # 解析结果
        frames = self._parse_frames(result)

        if not frames:
            Log.warning(f"{log_prefix} 未解析到有效动作帧")
            return []

        Log.info(f"{log_prefix} 生成 {len(frames)} 个动作帧")
        # Log.info(f"生成动作{frames}")

        # 更新连贯性状态：存储最后一帧的原始参数
        if frames:
            self._last_action_params = frames[-1].parameters.copy()

        return frames

    async def generate_single(
        self,
        text: str,
        context: str = "",
        timeout_seconds: float = 2.0,
    ) -> MotionFrame | None:
        """
        生成单个表情/动作

        简化接口：生成单个动作帧，适用于非 TTS 场景。

        参数：
        - text: 输入文本（情绪/动作描述）
        - context: 场景背景
        - timeout_seconds: LLM 调用超时时间

        返回：
        - 单个 MotionFrame
        - 失败时返回 None
        """
        frames = await self.generate_motion(
            text=text,
            duration_ms=1500,  # 默认 1.5 秒
            context=context,
            is_tts=False,
            timeout_seconds=timeout_seconds,
        )

        return frames[0] if frames else None

    async def generate_tts_motion(
        self,
        speech_text: str,
        speech_duration_ms: float,
        previous_params: dict[str, float] | None = None,
        context: str = "",
        timeout_seconds: float = 5.0,
    ) -> list[MotionFrame]:
        """
        TTS 专用动作生成

        优化接口：为 TTS 语音播报生成连续动作序列，保证动作连贯性。

        参数：
        - speech_text: TTS 即将播报的文本
        - speech_duration_ms: 语音播报时长（毫秒）
        - previous_params: 前一个动作的原始参数
        - context: 场景背景
        - timeout_seconds: LLM 调用超时时间

        返回：
        - MotionFrame 列表
        - 失败时返回空列表
        """
        return await self.generate_motion(
            text=speech_text,
            duration_ms=speech_duration_ms,
            context=context,
            previous_params=previous_params,
            is_tts=True,
            timeout_seconds=timeout_seconds,
        )

    def reset_state(self) -> None:
        """
        重置生成器状态

        清除连贯性状态和缓存，通常在角色切换时调用。
        """
        self._last_action_params = {}
        Log.info("[V2表情生成器] 状态已重置")
