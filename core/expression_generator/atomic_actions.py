"""
原子动作模板库

定义 Live2D 角色的原子动作模板，用于标准化动作生成。

核心设计：
1. 原子动作：最小的、不可再分的动作单元
2. 关键帧：每个动作包含多个参数的关键帧数据
3. 分类管理：按功能分组（眼部、头部、嘴巴等）

动作数量：39个
- 眼部（8个）：eye_close, eye_open, wink_left, wink_right, eye_squint, eye_wide, peek, eye_smile
- 视线（6个）：look_left, look_right, look_up, look_down, look_away, look_at
- 头部（6个）：nod, shake_head, tilt_left, tilt_right, head_down, head_up
- 嘴巴（6个）：smile, big_smile, frown, pout, mouth_open, talk
- 身体（5个）：lean_forward, lean_back, shrug, body_lean_left, body_lean_right
- 情绪（8个）：blush, angry, surprise, sad, shy, excited, thinking, embarrassed

使用示例：
```python
from core.expression_generator.atomic_actions import get_action, get_action_vocab

# 获取动作定义
action = get_action("smile")
print(action.duration)  # 1.5
print(action.keyframes)  # {"ParamMouthForm": [(0.0, 0.8), ...]}

# 获取动作词汇表（供 LLM 提示词使用）
vocab = get_action_vocab()
```
"""

from dataclasses import dataclass, field


@dataclass
class AtomicAction:
    """
    原子动作定义

    属性：
    - name: 动作标识（唯一）
    - category: 分类（eye/head/mouth/body/emotion/look）
    - description: 动作描述（中文，供 LLM 理解）
    - duration: 基础时长（秒）
    - keyframes: 关键帧数据 {参数ID: [(时间偏移, 目标值), ...]}
    - tags: 标签列表（用于分类和搜索）
    - stage: 时序编排层级（emotion/accent/gaze），决定自动编排时的时序行为
        - emotion: 贯穿全句的情感基调，句首触发，持续整句
        - accent:  配合语气的短时强调动作，自动错开排列
        - gaze:    视线方向/微表情，中等持留
    """

    name: str
    category: str
    description: str
    duration: float
    keyframes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    stage: str = "accent"


# ============================================================
# 原子动作定义（39个）
# ============================================================

# ---------- 眼部动作（8个）----------

EYE_CLOSE = AtomicAction(
    name="eye_close",
    category="eye",
    description="闭眼",
    duration=2.0,
    stage="gaze",
    keyframes={
        "ParamEyeLOpen": [(0.0, 1.0), (0.3, 0.0), (2.0, 0.0)],
        "ParamEyeROpen": [(0.0, 1.0), (0.3, 0.0), (2.0, 0.0)],
    },
    tags=["眼", "闭眼", "睡觉", "休息"],
)

EYE_OPEN = AtomicAction(
    name="eye_open",
    category="eye",
    description="睁眼",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeLOpen": [(0.0, 0.0), (0.3, 1.0), (1.0, 1.0)],
        "ParamEyeROpen": [(0.0, 0.0), (0.3, 1.0), (1.0, 1.0)],
    },
    tags=["眼", "睁眼", "醒来"],
)

WINK_LEFT = AtomicAction(
    name="wink_left",
    category="eye",
    description="左眼眨一下（wink）",
    duration=0.8,
    stage="accent",
    keyframes={
        "ParamEyeLOpen": [(0.0, 1.0), (0.15, 0.0), (0.3, 1.0), (0.8, 1.0)],
    },
    tags=["眼", " wink", "眨眼", "可爱"],
)

WINK_RIGHT = AtomicAction(
    name="wink_right",
    category="eye",
    description="右眼眨一下（wink）",
    duration=0.8,
    stage="accent",
    keyframes={
        "ParamEyeROpen": [(0.0, 1.0), (0.15, 0.0), (0.3, 1.0), (0.8, 1.0)],
    },
    tags=["眼", "wink", "眨眼", "可爱"],
)

EYE_SQUINT = AtomicAction(
    name="eye_squint",
    category="eye",
    description="眯眼（怀疑、思考）",
    duration=1.5,
    stage="gaze",
    keyframes={
        "ParamEyeLOpen": [(0.0, 1.0), (0.3, 0.4), (1.5, 0.4)],
        "ParamEyeROpen": [(0.0, 1.0), (0.3, 0.4), (1.5, 0.4)],
    },
    tags=["眼", "眯眼", "怀疑", "思考"],
)

EYE_WIDE = AtomicAction(
    name="eye_wide",
    category="eye",
    description="睁大眼睛（惊讶）",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeLOpen": [(0.0, 1.0), (0.2, 1.3), (1.0, 1.0)],
        "ParamEyeROpen": [(0.0, 1.0), (0.2, 1.3), (1.0, 1.0)],
    },
    tags=["眼", "睁大", "惊讶", "震惊"],
)

PEEK = AtomicAction(
    name="peek",
    category="eye",
    description="偷看（小心翼翼地看）",
    duration=0.6,
    stage="gaze",
    keyframes={
        "ParamEyeLOpen": [(0.0, 0.0), (0.1, 0.5), (0.4, 0.5), (0.6, 0.0)],
        "ParamEyeROpen": [(0.0, 0.0), (0.1, 0.5), (0.4, 0.5), (0.6, 0.0)],
        "ParamEyeBallY": [(0.0, 0.0), (0.1, 0.3), (0.4, 0.3), (0.6, 0.0)],
    },
    tags=["眼", "偷看", "害羞", "小心"],
)

EYE_SMILE = AtomicAction(
    name="eye_smile",
    category="eye",
    description="眼睛微笑（眯眼笑）",
    duration=1.5,
    stage="gaze",
    keyframes={
        "ParamEyeLSmile": [(0.0, 0.0), (0.3, 1.0), (1.5, 1.0)],
        "ParamEyeRSmile": [(0.0, 0.0), (0.3, 1.0), (1.5, 1.0)],
    },
    tags=["眼", "微笑", "开心", "笑"],
)

# ---------- 视线动作（6个）----------

LOOK_LEFT = AtomicAction(
    name="look_left",
    category="look",
    description="向左看",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeBallX": [(0.0, 0.0), (0.3, -1.0), (1.0, -1.0)],
    },
    tags=["视线", "左"],
)

LOOK_RIGHT = AtomicAction(
    name="look_right",
    category="look",
    description="向右看",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeBallX": [(0.0, 0.0), (0.3, 1.0), (1.0, 1.0)],
    },
    tags=["视线", "右"],
)

LOOK_UP = AtomicAction(
    name="look_up",
    category="look",
    description="向上看",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeBallY": [(0.0, 0.0), (0.3, 1.0), (1.0, 1.0)],
    },
    tags=["视线", "上", "思考"],
)

LOOK_DOWN = AtomicAction(
    name="look_down",
    category="look",
    description="向下看",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeBallY": [(0.0, 0.0), (0.3, -1.0), (1.0, -1.0)],
    },
    tags=["视线", "下", "害羞", "悲伤"],
)

LOOK_AWAY = AtomicAction(
    name="look_away",
    category="look",
    description="看向别处（不自然、害羞）",
    duration=1.5,
    stage="gaze",
    keyframes={
        "ParamEyeBallX": [(0.0, 0.0), (0.3, 0.8), (0.5, 0.8), (1.0, 0.0)],
        "ParamAngleY": [(0.0, 0.0), (0.3, -5.0), (0.5, -5.0), (1.0, 0.0)],
    },
    tags=["视线", "别处", "害羞", "不自然"],
)

LOOK_AT = AtomicAction(
    name="look_at",
    category="look",
    description="看向对方（注视）",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamEyeBallX": [(0.0, 0.0), (0.3, 0.0)],
        "ParamEyeBallY": [(0.0, 0.0), (0.3, 0.0)],
    },
    tags=["视线", "注视", "认真"],
)

# ---------- 头部动作（6个）----------

NOD = AtomicAction(
    name="nod",
    category="head",
    description="点头（同意、理解）",
    duration=1.2,
    stage="accent",
    keyframes={
        "ParamAngleY": [(0.0, 0.0), (0.2, -10.0), (0.5, -10.0), (0.8, 0.0), (1.2, 0.0)],
    },
    tags=["头", "点头", "同意", "理解"],
)

SHAKE_HEAD = AtomicAction(
    name="shake_head",
    category="head",
    description="摇头（否定、不同意）",
    duration=1.5,
    stage="accent",
    keyframes={
        "ParamAngleX": [(0.0, 0.0), (0.2, -10.0), (0.5, 10.0), (0.8, -5.0), (1.2, 0.0)],
    },
    tags=["头", "摇头", "否定", "不"],
)

TILT_LEFT = AtomicAction(
    name="tilt_left",
    category="head",
    description="头向左倾斜（疑惑、可爱）",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamAngleZ": [(0.0, 0.0), (0.3, -10.0), (1.0, -10.0)],
    },
    tags=["头", "倾斜", "疑惑", "可爱"],
)

TILT_RIGHT = AtomicAction(
    name="tilt_right",
    category="head",
    description="头向右倾斜（疑惑、可爱）",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamAngleZ": [(0.0, 0.0), (0.3, 10.0), (1.0, 10.0)],
    },
    tags=["头", "倾斜", "疑惑", "可爱"],
)

HEAD_DOWN = AtomicAction(
    name="head_down",
    category="head",
    description="低头（害羞、悲伤、思考）",
    duration=1.5,
    stage="gaze",
    keyframes={
        "ParamAngleY": [(0.0, 0.0), (0.4, -15.0), (1.5, -15.0)],
    },
    tags=["头", "低头", "害羞", "悲伤"],
)

HEAD_UP = AtomicAction(
    name="head_up",
    category="head",
    description="抬头（自信、惊讶）",
    duration=1.0,
    stage="gaze",
    keyframes={
        "ParamAngleY": [(0.0, 0.0), (0.3, 10.0), (1.0, 10.0)],
    },
    tags=["头", "抬头", "自信", "惊讶"],
)

# ---------- 嘴巴动作（6个）----------

SMILE = AtomicAction(
    name="smile",
    category="mouth",
    description="微笑",
    duration=1.5,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.3, 0.8), (1.5, 0.8)],
    },
    tags=["嘴", "微笑", "开心", "笑"],
)

BIG_SMILE = AtomicAction(
    name="big_smile",
    category="mouth",
    description="大笑（开心）",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.3, 1.0), (2.0, 1.0)],
        "ParamMouthOpenY": [(0.0, 0.0), (0.3, 0.5), (2.0, 0.5)],
        "ParamEyeLSmile": [(0.0, 0.0), (0.3, 1.0), (2.0, 1.0)],
        "ParamEyeRSmile": [(0.0, 0.0), (0.3, 1.0), (2.0, 1.0)],
    },
    tags=["嘴", "大笑", "开心", "高兴"],
)

FROWN = AtomicAction(
    name="frown",
    category="mouth",
    description="皱眉（不满、悲伤）",
    duration=1.5,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.3, -0.6), (1.5, -0.6)],
        "ParamBrowLForm": [(0.0, 0.0), (0.3, -0.5), (1.5, -0.5)],
        "ParamBrowRForm": [(0.0, 0.0), (0.3, -0.5), (1.5, -0.5)],
    },
    tags=["嘴", "皱眉", "不满", "悲伤"],
)

POUT = AtomicAction(
    name="pout",
    category="mouth",
    description="撅嘴（撒娇、不满）",
    duration=1.5,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.3, -0.3), (1.5, -0.3)],
        "ParamMouthOpenY": [(0.0, 0.0), (0.3, 0.2), (1.5, 0.2)],
    },
    tags=["嘴", "撅嘴", "撒娇", "不满"],
)

MOUTH_OPEN = AtomicAction(
    name="mouth_open",
    category="mouth",
    description="张嘴（惊讶、说话）",
    duration=1.0,
    stage="accent",
    keyframes={
        "ParamMouthOpenY": [(0.0, 0.0), (0.2, 0.8), (1.0, 0.8)],
    },
    tags=["嘴", "张嘴", "惊讶"],
)

TALK = AtomicAction(
    name="talk",
    category="mouth",
    description="说话（嘴巴张合）",
    duration=2.0,
    stage="accent",
    keyframes={
        "ParamMouthOpenY": [
            (0.0, 0.0),
            (0.1, 0.5),
            (0.2, 0.2),
            (0.3, 0.6),
            (0.4, 0.3),
            (0.5, 0.0),
        ],
    },
    tags=["嘴", "说话", "聊天"],
)

# ---------- 身体动作（5个）----------

LEAN_FORWARD = AtomicAction(
    name="lean_forward",
    category="body",
    description="身体前倾（好奇、亲近）",
    duration=1.5,
    stage="accent",
    keyframes={
        "ParamBodyAngleY": [(0.0, 0.0), (0.4, -10.0), (1.5, -10.0)],
    },
    tags=["身体", "前倾", "好奇", "亲近"],
)

LEAN_BACK = AtomicAction(
    name="lean_back",
    category="body",
    description="身体后仰（惊讶、躲避）",
    duration=1.0,
    stage="accent",
    keyframes={
        "ParamBodyAngleY": [(0.0, 0.0), (0.3, 10.0), (1.0, 10.0)],
    },
    tags=["身体", "后仰", "惊讶", "躲避"],
)

SHRUG = AtomicAction(
    name="shrug",
    category="body",
    description="耸肩（无奈、不知道）",
    duration=1.5,
    stage="accent",
    keyframes={
        "ParamBodyAngleY": [(0.0, 0.0), (0.3, 5.0), (0.6, 5.0), (1.0, 0.0)],
    },
    tags=["身体", "耸肩", "无奈", "不知道"],
)

BODY_LEAN_LEFT = AtomicAction(
    name="body_lean_left",
    category="body",
    description="身体向左倾斜",
    duration=1.0,
    stage="accent",
    keyframes={
        "ParamBodyAngleZ": [(0.0, 0.0), (0.3, -8.0), (1.0, -8.0)],
    },
    tags=["身体", "左倾"],
)

BODY_LEAN_RIGHT = AtomicAction(
    name="body_lean_right",
    category="body",
    description="身体向右倾斜",
    duration=1.0,
    stage="accent",
    keyframes={
        "ParamBodyAngleZ": [(0.0, 0.0), (0.3, 8.0), (1.0, 8.0)],
    },
    tags=["身体", "右倾"],
)

# ---------- 情绪动作（8个）----------

BLUSH = AtomicAction(
    name="blush",
    category="emotion",
    description="脸红（害羞、尴尬）",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamCheek": [(0.0, 0.0), (0.5, 0.8), (2.0, 0.8)],
    },
    tags=["情绪", "脸红", "害羞", "尴尬"],
)

ANGRY = AtomicAction(
    name="angry",
    category="emotion",
    description="生气",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamBrowLForm": [(0.0, 0.0), (0.3, -0.8), (2.0, -0.8)],
        "ParamBrowRForm": [(0.0, 0.0), (0.3, -0.8), (2.0, -0.8)],
        "ParamMouthForm": [(0.0, 0.0), (0.3, -0.6), (2.0, -0.6)],
    },
    tags=["情绪", "生气", "愤怒", "不满"],
)

SURPRISE = AtomicAction(
    name="surprise",
    category="emotion",
    description="惊讶",
    duration=1.5,
    stage="emotion",
    keyframes={
        "ParamEyeLOpen": [(0.0, 1.0), (0.2, 1.3), (1.5, 1.0)],
        "ParamEyeROpen": [(0.0, 1.0), (0.2, 1.3), (1.5, 1.0)],
        "ParamMouthOpenY": [(0.0, 0.0), (0.2, 0.7), (1.0, 0.3)],
        "ParamBrowLY": [(0.0, 0.0), (0.2, 0.8), (1.5, 0.3)],
        "ParamBrowRY": [(0.0, 0.0), (0.2, 0.8), (1.5, 0.3)],
    },
    tags=["情绪", "惊讶", "震惊"],
)

SAD = AtomicAction(
    name="sad",
    category="emotion",
    description="悲伤",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.4, -0.7), (2.0, -0.7)],
        "ParamEyeBallY": [(0.0, 0.0), (0.4, -0.5), (2.0, -0.5)],
        "ParamAngleY": [(0.0, 0.0), (0.4, -8.0), (2.0, -8.0)],
    },
    tags=["情绪", "悲伤", "难过", "伤心"],
)

SHY = AtomicAction(
    name="shy",
    category="emotion",
    description="害羞",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamCheek": [(0.0, 0.0), (0.4, 0.6), (2.0, 0.6)],
        "ParamEyeBallX": [(0.0, 0.0), (0.3, 0.5), (2.0, 0.5)],
        "ParamAngleZ": [(0.0, 0.0), (0.3, -5.0), (2.0, -5.0)],
    },
    tags=["情绪", "害羞", "不好意思"],
)

EXCITED = AtomicAction(
    name="excited",
    category="emotion",
    description="兴奋、激动",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamMouthForm": [(0.0, 0.0), (0.3, 0.9), (2.0, 0.9)],
        "ParamEyeLSmile": [(0.0, 0.0), (0.3, 0.8), (2.0, 0.8)],
        "ParamEyeRSmile": [(0.0, 0.0), (0.3, 0.8), (2.0, 0.8)],
        "ParamAngleY": [
            (0.0, 0.0),
            (0.2, 5.0),
            (0.5, 5.0),
            (0.8, 3.0),
            (1.2, 5.0),
            (2.0, 3.0),
        ],
    },
    tags=["情绪", "兴奋", "激动", "开心"],
)

THINKING = AtomicAction(
    name="thinking",
    category="emotion",
    description="思考",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamEyeBallX": [(0.0, 0.0), (0.3, 0.6), (1.5, 0.6), (2.0, 0.0)],
        "ParamEyeBallY": [(0.0, 0.0), (0.3, 0.4), (1.5, 0.4), (2.0, 0.0)],
        "ParamAngleZ": [(0.0, 0.0), (0.3, 5.0), (1.5, 5.0), (2.0, 0.0)],
    },
    tags=["情绪", "思考", "想"],
)

EMBARRASSED = AtomicAction(
    name="embarrassed",
    category="emotion",
    description="尴尬",
    duration=2.0,
    stage="emotion",
    keyframes={
        "ParamCheek": [(0.0, 0.0), (0.4, 0.5), (2.0, 0.5)],
        "ParamMouthForm": [(0.0, 0.0), (0.3, -0.3), (1.0, 0.2), (2.0, -0.2)],
        "ParamEyeBallX": [(0.0, 0.0), (0.3, -0.4), (1.5, -0.4), (2.0, 0.0)],
    },
    tags=["情绪", "尴尬", "不好意思"],
)


# ============================================================
# 动作注册表
# ============================================================

# 所有原子动作的字典 {name: AtomicAction}
ALL_ACTIONS: dict[str, AtomicAction] = {
    # 眼部
    "eye_close": EYE_CLOSE,
    "eye_open": EYE_OPEN,
    "wink_left": WINK_LEFT,
    "wink_right": WINK_RIGHT,
    "eye_squint": EYE_SQUINT,
    "eye_wide": EYE_WIDE,
    "peek": PEEK,
    "eye_smile": EYE_SMILE,
    # 视线
    "look_left": LOOK_LEFT,
    "look_right": LOOK_RIGHT,
    "look_up": LOOK_UP,
    "look_down": LOOK_DOWN,
    "look_away": LOOK_AWAY,
    "look_at": LOOK_AT,
    # 头部
    "nod": NOD,
    "shake_head": SHAKE_HEAD,
    "tilt_left": TILT_LEFT,
    "tilt_right": TILT_RIGHT,
    "head_down": HEAD_DOWN,
    "head_up": HEAD_UP,
    # 嘴巴
    "smile": SMILE,
    "big_smile": BIG_SMILE,
    "frown": FROWN,
    "pout": POUT,
    "mouth_open": MOUTH_OPEN,
    "talk": TALK,
    # 身体
    "lean_forward": LEAN_FORWARD,
    "lean_back": LEAN_BACK,
    "shrug": SHRUG,
    "body_lean_left": BODY_LEAN_LEFT,
    "body_lean_right": BODY_LEAN_RIGHT,
    # 情绪
    "blush": BLUSH,
    "angry": ANGRY,
    "surprise": SURPRISE,
    "sad": SAD,
    "shy": SHY,
    "excited": EXCITED,
    "thinking": THINKING,
    "embarrassed": EMBARRASSED,
}

# 按分类分组的动作字典
ACTIONS_BY_CATEGORY: dict[str, list[AtomicAction]] = {}
for action in ALL_ACTIONS.values():
    if action.category not in ACTIONS_BY_CATEGORY:
        ACTIONS_BY_CATEGORY[action.category] = []
    ACTIONS_BY_CATEGORY[action.category].append(action)


# ============================================================
# 互斥组定义
# ============================================================

# 互斥组：同一组内的动作不能同时触发
MUTEX_GROUPS: list[list[str]] = [
    # 眼睛开合组
    ["eye_close", "eye_open", "eye_squint", "eye_wide", "peek"],
    # 水平视线组
    ["look_left", "look_right"],
    # 垂直视线组
    ["look_up", "look_down"],
    # 头部点头组
    ["nod", "head_down", "head_up"],
    # 头部倾斜组
    ["tilt_left", "tilt_right"],
    # 嘴型正向组（微笑）
    ["smile", "big_smile"],
    # 嘴型负向组（皱眉/撅嘴）
    ["frown", "pout"],
    # 身体前倾组
    ["lean_forward", "lean_back"],
    # 身体侧倾组
    ["body_lean_left", "body_lean_right"],
]

# 动作到互斥组的映射
ACTION_MUTEX_MAP: dict[str, list[int]] = {}
for i, group in enumerate(MUTEX_GROUPS):
    for action_name in group:
        if action_name not in ACTION_MUTEX_MAP:
            ACTION_MUTEX_MAP[action_name] = []
        ACTION_MUTEX_MAP[action_name].append(i)


def check_mutex_conflict(action_names: list[str]) -> list[list[str]]:
    """
    检测动作列表中的互斥冲突

    参数：
    - action_names: 动作名称列表

    返回：
    - 冲突组列表，每组包含冲突的动作名称
    """
    conflicts: list[list[str]] = []
    seen_groups: set[int] = set()

    for name in action_names:
        if name in ACTION_MUTEX_MAP:
            for group_idx in ACTION_MUTEX_MAP[name]:
                if group_idx in seen_groups:
                    # 找到冲突，提取该组中在action_names里的动作
                    group = MUTEX_GROUPS[group_idx]
                    conflicting = [a for a in action_names if a in group]
                    if len(conflicting) > 1:
                        conflicts.append(conflicting)
                else:
                    seen_groups.add(group_idx)

    # 去重
    unique_conflicts = []
    seen_sets: set[frozenset[str]] = set()
    for conflict in conflicts:
        conflict_set = frozenset(conflict)
        if conflict_set not in seen_sets:
            seen_sets.add(conflict_set)
            unique_conflicts.append(list(conflict))

    return unique_conflicts


def resolve_mutex_conflict(action_names: list[str]) -> list[str]:
    """
    解决互斥冲突，保留优先级最高的动作

    优先级规则：
    1. 复合动作（emotion类）优先级最高
    2. 单一动作按字母顺序

    参数：
    - action_names: 动作名称列表

    返回：
    - 解决冲突后的动作名称列表
    """
    if not action_names:
        return []

    # 按互斥组分组
    grouped: dict[int, list[str]] = {}
    ungrouped: list[str] = []

    for name in action_names:
        if name in ACTION_MUTEX_MAP:
            for group_idx in ACTION_MUTEX_MAP[name]:
                if group_idx not in grouped:
                    grouped[group_idx] = []
                grouped[group_idx].append(name)
        else:
            ungrouped.append(name)

    # 从每组中选择一个动作
    selected: list[str] = list(ungrouped)

    for group_idx, actions in grouped.items():
        # 去重
        unique_actions = list(set(actions))
        if len(unique_actions) <= 1:
            selected.extend(unique_actions)
            continue

        # 选择优先级最高的
        # 优先级：emotion > 其他，同级按字母顺序
        def priority_key(name: str) -> tuple[int, str]:
            action = get_action(name)
            if action and action.category == "emotion":
                return (0, name)
            return (1, name)

        unique_actions.sort(key=priority_key)
        best = unique_actions[0]
        selected.append(best)

    return list(set(selected))


# ============================================================
# 公开接口
# ============================================================


def get_action(name: str) -> AtomicAction | None:
    """
    获取原子动作定义

    参数：
    - name: 动作名称

    返回：
    - AtomicAction 实例，不存在返回 None
    """
    return ALL_ACTIONS.get(name)


def get_actions_by_category(category: str) -> list[AtomicAction]:
    """
    获取指定分类的所有动作

    参数：
    - category: 分类名称（eye/look/head/mouth/body/emotion）

    返回：
    - 动作列表
    """
    return ACTIONS_BY_CATEGORY.get(category, [])


def get_all_actions() -> dict[str, AtomicAction]:
    """
    获取所有原子动作

    返回：
    - {动作名: AtomicAction} 字典
    """
    return ALL_ACTIONS.copy()


def get_action_vocab() -> str:
    """
    获取动作词汇表（供 LLM 提示词使用）

    返回：
    - 格式化的动作描述字符串

    示例输出：
    ```
    【眼部动作】
    - eye_close: 闭眼
    - eye_open: 睁眼
    ...
    ```
    """
    lines = []
    category_names = {
        "eye": "眼部动作",
        "look": "视线动作",
        "head": "头部动作",
        "mouth": "嘴巴动作",
        "body": "身体动作",
        "emotion": "情绪动作",
    }

    for category, name in category_names.items():
        actions = get_actions_by_category(category)
        if not actions:
            continue

        lines.append(f"【{name}】")
        for action in actions:
            lines.append(f"- {action.name}: {action.description}")
        lines.append("")

    return "\n".join(lines)


def get_action_names() -> list[str]:
    """
    获取所有动作名称列表

    返回：
    - 动作名称列表
    """
    return list(ALL_ACTIONS.keys())
