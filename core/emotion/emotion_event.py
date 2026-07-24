import time
from dataclasses import dataclass, field, asdict
from typing import List

# 衰减基准：强度 1.0 的情绪持续 4 小时（秒）
_DECAY_BASE = 4 * 3600


@dataclass
class EmotionEvent:
    """情感事件——记录一次有因可循的情绪冲击"""

    emotion: str  # "joy" | "sadness" | "anger" | "fear" | "gratitude" | "hurt" | "neutral"
    intensity: float  # 0.0~1.0
    reason: str  # 触发原因简述，如"用户刚才说话很凶"
    created_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: int = field(default=0)

    def __post_init__(self):
        if self.expires_at == 0:
            self.expires_at = self.created_at + int(self.intensity * _DECAY_BASE)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


EMOTION_MAP_ZH = {
    "joy": "开心",
    "sadness": "难过",
    "anger": "生气",
    "fear": "害怕",
    "gratitude": "感激",
    "hurt": "委屈",
    "neutral": "平静",
}


def serialize_events(events: List[EmotionEvent]) -> list[dict]:
    return [asdict(e) for e in events]


def deserialize_events(data: list[dict]) -> List[EmotionEvent]:
    return [EmotionEvent(**d) for d in data]
