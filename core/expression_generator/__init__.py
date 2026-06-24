"""
Live2D 表情/动作生成器模块

提供多个版本的动作生成系统：
- V1: 规划+生成分离（两阶段 LLM 调用）
- V2: 单次请求生成完整动作序列
- V3: 流式 Batch 架构（推荐）

核心组件：
- atomic_actions: 原子动作模板库（39个动作）
- motion_combiner: 动作组合引擎
- motion_generator_v3: V3 版本生成器
- streaming_json_parser: 流式 JSON 解析器
- motion_schema: 数据模型

使用示例：
```python
from core.expression_generator import (
    MotionGeneratorV3,
    get_action,
    get_action_vocab,
    MotionCombiner,
)

# 创建 V3 生成器
generator = MotionGeneratorV3()
await generator.initialize("assistant_name")

# 流式生成
async for chunk in generator.stream_generate("你好呀"):
    print(chunk.text)      # 句子文本
    print(chunk.motion)    # 参数曲线数据
```
"""

from core.expression_generator.atomic_actions import (
    AtomicAction,
    get_action,
    get_action_vocab,
    get_action_names,
    get_all_actions,
    get_actions_by_category,
    ALL_ACTIONS,
)
from core.expression_generator.motion_combiner import (
    MotionCombiner,
    MotionCurve,
    MotionFrame,
    ActionSpec,
    create_combiner,
)
from core.expression_generator.motion_generator_v3 import (
    MotionGeneratorV3,
    V3MotionFrame,
    create_v3_generator,
)
from core.expression_generator.motion_schema import (
    MotionChunk,
    MotionResponse,
    MotionCurveData,
    AtomicActionSpec,
    StreamingChunk,
)
from core.expression_generator.streaming_json_parser import (
    StreamingJsonLineParser,
    StreamingJsonArrayParser,
)

__all__ = [
    # 原子动作
    "AtomicAction",
    "get_action",
    "get_action_vocab",
    "get_action_names",
    "get_all_actions",
    "get_actions_by_category",
    "ALL_ACTIONS",
    # 组合引擎
    "MotionCombiner",
    "MotionCurve",
    "MotionFrame",
    "ActionSpec",
    "create_combiner",
    # V3 生成器
    "MotionGeneratorV3",
    "V3MotionFrame",
    "create_v3_generator",
    # 数据模型
    "MotionChunk",
    "MotionResponse",
    "MotionCurveData",
    "AtomicActionSpec",
    "StreamingChunk",
    # 解析器
    "StreamingJsonLineParser",
    "StreamingJsonArrayParser",
]
