"""
Live2D 动作生成器 V3

基于流式 Batch 架构的动作生成系统。

核心改进：
1. 单次 LLM 调用同时生成文本和动作标签
2. 流式解析，逐句播放
3. 本地组合引擎，毫秒级延迟
4. 可扩展的提示词和解析器系统

架构流程：
```
用户消息 → LLM（流式输出JSON行）→ 增量解析器 → 每检测到完整行 → 立即输出chunk
                                                         ↓
                                              并发：TTS + 组合引擎（毫秒级）
```

使用示例：
```python
generator = MotionGeneratorV3()
await generator.initialize("assistant_name")

# 流式生成
async for chunk in generator.stream_generate("你好呀"):
    print(chunk.text)      # 句子文本
    print(chunk.motion)    # 参数曲线数据
```
"""

from dataclasses import dataclass, field
import time
from collections.abc import AsyncIterator
from my_utils.log import logger as Log
from core.llm import (
    LLMClient,
    PromptManager,
    PromptTemplate,
    CallbackManager,
    CallbackEvent,
)
from core.expression_generator.atomic_actions import (
    get_action,
    get_action_vocab,
    get_action_names,
)
from core.expression_generator.motion_combiner import (
    MotionCombiner,
    MotionCurve,
    MotionFrame,
    ActionSpec,
)
from core.expression_generator.motion_schema import (
    StreamingChunk,
    MotionChunk,
    MotionCurveData,
    AtomicActionSpec,
)
from core.expression_generator.streaming_json_parser import StreamingJsonLineParser

# ============================================================
# 参数别名系统（复用 V2）
# ============================================================

# 别名 -> 完整参数 ID 映射表
ALIAS_TO_PARAM: dict[str, str] = {
    "elo": "ParamEyeLOpen",
    "ero": "ParamEyeROpen",
    "els": "ParamEyeLSmile",
    "ers": "ParamEyeRSmile",
    "ebx": "ParamEyeBallX",
    "eby": "ParamEyeBallY",
    "bly": "ParamBrowLY",
    "bry": "ParamBrowRY",
    "bls": "ParamBrowLForm",
    "brs": "ParamBrowRForm",
    "mf": "ParamMouthForm",
    "mo": "ParamMouthOpenY",
    "chk": "ParamCheek",
    "ax": "ParamAngleX",
    "ay": "ParamAngleY",
    "az": "ParamAngleZ",
    "bx": "ParamBodyAngleX",
    "by": "ParamBodyAngleY",
    "bz": "ParamBodyAngleZ",
}

# 完整参数 ID -> 别名 反向映射
PARAM_TO_ALIAS: dict[str, str] = {v: k for k, v in ALIAS_TO_PARAM.items()}


def _get_alias(param_id: str) -> str:
    """获取参数 ID 对应的别名"""
    return PARAM_TO_ALIAS.get(param_id, param_id)


# ============================================================
# 提示词模板
# ============================================================

# 系统提示词模板
SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    name="v3_system",
    template="""你是一个 Live2D 虚拟形象的动作控制器。根据输入文本生成动作标签。

【任务说明】
你需要将回复文本按句子拆分为多个 chunk，每个 chunk 包含：
- t: 句子文本
- a: 动作标签列表（从可用动作中选择）

【可用动作】
{action_vocab}

【输出格式】
每行输出一个 JSON 对象，格式如下：
{{"t": "句子文本", "a": ["动作1", "动作2"]}}

示例：
{{"t": "你好呀~", "a": ["smile", "nod"]}}
{{"t": "今天天气真好呢", "a": ["look_up"]}}
{{"t": "", "a": [], "done": true}}

【规则】
1. 每个 chunk 对应一句话或一个语义段落
2. 动作应自然配合说话内容和情绪
3. 动作可以重叠（同时进行）
4. 最后一行输出 done: true 表示完成
5. 不要输出任何其他内容，只输出 JSON 行""",
    description="V3 系统提示词",
    required_vars=["action_vocab"],
)

# 上下文消息模板
CONTEXT_TEMPLATE = PromptTemplate(
    name="v3_context",
    template="""【对话场景】
{context}

【前一个动作参数】
{previous_params}（保持动作连贯性）""",
    description="V3 上下文模板",
)


# ============================================================
# V3 动作生成器
# ============================================================


@dataclass
class V3MotionFrame:
    """
    V3 动作帧数据

    属性：
    - duration: 持续时间（毫秒）
    - parameters: 参数字典
    - expression: 表情列表（可选）
    """

    duration: float
    parameters: dict[str, float] = field(default_factory=dict)
    expression: list[str] = field(default_factory=list)


class MotionGeneratorV3:
    """
    V3 版本动作生成器

    基于流式 Batch 架构，单次 LLM 调用同时生成文本和动作标签。

    特性：
    - 流式输出：逐句播放
    - 本地组合：毫秒级延迟
    - 可扩展：自定义提示词和解析器

    使用示例：
    ```python
    generator = MotionGeneratorV3()
    await generator.initialize("assistant_name")

    # 流式生成
    async for chunk in generator.stream_generate("你好呀"):
        print(chunk.text)      # 句子文本
        print(chunk.motion)    # 参数曲线数据
    ```
    """

    def __init__(self, fps: float = 30.0):
        """
        初始化 V3 生成器

        参数：
        - fps: 输出曲线的帧率
        """
        self._fps = fps
        self._combiner = MotionCombiner(fps=fps)
        self._llm_client = LLMClient()
        self._callbacks = CallbackManager()

        # 连贯性状态
        self._last_action_params: dict[str, float] = {}

    @property
    def callbacks(self) -> CallbackManager:
        """获取回调管理器"""
        return self._callbacks

    async def initialize(self, assistant_name: str) -> None:
        """
        初始化生成器

        参数：
        - assistant_name: 角色名称
        """
        Log.info(f"[V3生成器] 初始化: {assistant_name}")
        # 注册系统提示词模板
        # 注意：这里只是示例，实际模板可能需要根据模型调整

    def _build_system_prompt(self, is_tts: bool = False) -> str:
        """
        构建系统提示词

        参数：
        - is_tts: 是否为 TTS 场景

        返回：
        - 系统提示词
        """
        # 获取动作词汇表
        action_vocab = get_action_vocab()

        # 渲染模板
        return SYSTEM_PROMPT_TEMPLATE.render(action_vocab=action_vocab)

    def _build_context_message(
        self,
        context: str = "",
        previous_params: dict[str, float] | None = None,
    ) -> str | None:
        """
        构建上下文消息

        参数：
        - context: 对话场景
        - previous_params: 前一个动作参数

        返回：
        - 上下文消息，无上下文返回 None
        """
        if not context and not previous_params:
            return None

        # 转换参数为别名格式
        alias_params = {}
        if previous_params:
            for param_id, value in previous_params.items():
                alias = _get_alias(param_id)
                alias_params[alias] = round(value, 2)

        return CONTEXT_TEMPLATE.render(
            context=context or "无",
            previous_params=str(alias_params) if alias_params else "无",
        )

    def _build_input_message(
        self,
        text: str,
        duration_ms: float,
        is_tts: bool = False,
    ) -> str:
        """
        构建输入消息

        参数：
        - text: 输入文本
        - duration_ms: 动作时长（毫秒）
        - is_tts: 是否为 TTS 场景

        返回：
        - 输入消息
        """
        if is_tts:
            return f"语音内容：{text}\n语音时长：{duration_ms}ms\n请为这段语音生成动作标签。"
        else:
            return f"当前输入：{text}\n动作时长：{duration_ms}ms\n请为这段文本生成动作标签。"

    def _create_parser(self) -> StreamingJsonLineParser:
        """
        创建流式解析器

        返回：
        - StreamingJsonLineParser 实例
        """
        return StreamingJsonLineParser()

    async def stream_generate(
        self,
        text: str,
        duration_ms: float | None = None,
        context: str = "",
        previous_params: dict[str, float] | None = None,
        is_tts: bool = False,
        timeout_seconds: float = 30.0,
    ) -> AsyncIterator[MotionChunk]:
        """
        流式生成动作

        核心方法：根据输入文本流式生成动作 chunk。

        参数：
        - text: 输入文本
        - duration_ms: 动作总时长（毫秒），None 时自动估算
        - context: 对话场景
        - previous_params: 前一个动作参数
        - is_tts: 是否为 TTS 场景
        - timeout_seconds: 超时时间

        产出：
        - MotionChunk 实例
        """
        # 估算时长
        if duration_ms is None:
            duration_ms = self._combiner.estimate_duration(text) * 1000

        # 使用前一个动作参数
        effective_previous = previous_params or self._last_action_params

        # 构建消息
        system_prompt = self._build_system_prompt(is_tts=is_tts)
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
        messages = [{"role": "system", "content": system_prompt}]
        if context_message:
            messages.append({"role": "user", "content": context_message})
        messages.append({"role": "user", "content": input_message})

        # 创建解析器
        parser = self._create_parser()

        # 触发开始回调
        await self._callbacks.emit(CallbackEvent.START, messages=messages)

        Log.info(f"[V3生成器] 开始生成: {text[:20]}...")

        chunk_index = 0
        start_time = time.time()

        # 流式请求
        async for streaming_chunk in self._llm_client.stream(
            messages=messages,
            parser=parser,
            timeout=timeout_seconds,
        ):
            # 检查超时
            if time.time() - start_time > timeout_seconds:
                Log.warning(f"[V3生成器] 超时 ({timeout_seconds}s)")
                break

            # 转换为 MotionChunk
            motion_chunk = await self._process_streaming_chunk(
                streaming_chunk=streaming_chunk,
                index=chunk_index,
                is_tts=is_tts,
            )

            if motion_chunk:
                # 更新连贯性状态
                if motion_chunk.actions:
                    # 使用最后一个动作的参数
                    last_action_name = motion_chunk.actions[-1].act
                    last_action = get_action(last_action_name)
                    if last_action:
                        self._last_action_params = {}
                        for param_id, keyframes in last_action.keyframes.items():
                            if keyframes:
                                self._last_action_params[param_id] = keyframes[-1][1]

                # 触发 chunk 回调
                await self._callbacks.emit(CallbackEvent.CHUNK, chunk=motion_chunk)

                yield motion_chunk
                chunk_index += 1

        # 处理解析器缓冲区中的剩余数据
        for streaming_chunk in parser.flush():
            motion_chunk = await self._process_streaming_chunk(
                streaming_chunk=streaming_chunk,
                index=chunk_index,
                is_tts=is_tts,
            )
            if motion_chunk:
                await self._callbacks.emit(CallbackEvent.CHUNK, chunk=motion_chunk)
                yield motion_chunk
                chunk_index += 1

        # 触发完成回调
        elapsed = time.time() - start_time
        await self._callbacks.emit(
            CallbackEvent.COMPLETE, elapsed=elapsed, chunk_count=chunk_index
        )

        Log.info(f"[V3生成器] 完成: {chunk_index} 个 chunk, 耗时 {elapsed:.2f}s")

    async def _process_streaming_chunk(
        self,
        streaming_chunk: StreamingChunk,
        index: int,
        is_tts: bool = False,
    ) -> MotionChunk | None:
        """
        处理流式 chunk

        参数：
        - streaming_chunk: 流式 chunk 数据
        - index: chunk 序号
        - is_tts: 是否为 TTS 场景

        返回：
        - MotionChunk 实例，无效 chunk 返回 None
        """
        # 跳过空 chunk 和完成标记
        if streaming_chunk.done or (
            not streaming_chunk.text and not streaming_chunk.actions
        ):
            return None

        # 解析动作规格
        action_specs = []
        for action_name in streaming_chunk.actions:
            action = get_action(action_name)
            if action:
                action_specs.append(ActionSpec(name=action_name))
            else:
                Log.warning(f"[V3生成器] 未知动作: {action_name}")

        # 估算 chunk 时长
        chunk_duration = self._combiner.estimate_duration(streaming_chunk.text)

        # 使用组合引擎生成曲线
        motion_curve = self._combiner.combine_curve(
            action_specs=action_specs,
            text_duration=chunk_duration,
        )

        # 转换为 MotionCurveData
        curve_data = MotionCurveData(
            duration=motion_curve.duration,
            curves={
                param: [[t, v] for t, v in points]
                for param, points in motion_curve.curves.items()
            },
        )

        # 构建 MotionChunk
        return MotionChunk(
            index=index,
            text=streaming_chunk.text,
            actions=[AtomicActionSpec(act=name) for name in streaming_chunk.actions],
            motion=curve_data,
            duration=chunk_duration,
        )

    def reset_state(self) -> None:
        """重置生成器状态"""
        self._last_action_params = {}
        Log.info("[V3生成器] 状态已重置")


# ============================================================
# 公开接口
# ============================================================


def create_v3_generator(fps: float = 30.0) -> MotionGeneratorV3:
    """
    创建 V3 生成器实例

    参数：
    - fps: 输出曲线的帧率

    返回：
    - MotionGeneratorV3 实例
    """
    return MotionGeneratorV3(fps=fps)
