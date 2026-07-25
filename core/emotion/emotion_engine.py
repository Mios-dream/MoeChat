import os
import json
import time
from typing import List

from Config import Config
from core.emotion.emotion_event import (
    EmotionEvent,
    serialize_events,
    deserialize_events,
    EMOTION_MAP_ZH,
)


class EmotionEngine:
    """
    事件驱动情绪引擎
    记录每次有因可循的情绪冲击，持久化到 emotion_events.json，
    独立于聊天上下文，清空历史不影响情绪状态。
    """

    def __init__(self, agent_name: str):
        self._events_file = os.path.join(
            Config.BASE_AGENTS_PATH, agent_name, "emotion_events.json"
        )
        self._events: List[EmotionEvent] = []
        self._load()

    # ── 公开接口 ──────────────────────────────────────────

    def process_event(self, impact: dict) -> None:
        """
        记录一次情绪冲击

        Parameters:
            impact: {
                "emotion": "joy" | "sadness" | "anger" | "fear" | "gratitude" | "hurt" | "neutral",
                "intensity": 0.0~1.0,
                "reason": "触发原因简述"
            }
        """
        if impact.get("emotion", "neutral") == "neutral":
            return
        event = EmotionEvent(
            emotion=impact["emotion"],
            intensity=impact["intensity"],
            reason=impact.get("reason", ""),
        )
        self._events.append(event)
        self._cleanup()
        self._save()

    def get_mood_prompt(self) -> str:
        """获取当前情绪状态的提示词片段"""
        self._cleanup()
        if not self._events:
            return ""
        active = sorted(self._events, key=lambda e: e.intensity, reverse=True)
        parts = []
        for e in active[:3]:
            zh = EMOTION_MAP_ZH.get(e.emotion, e.emotion)
            parts.append(f"{zh}——{e.reason}")
        return "你此刻的心情：\n" + "\n".join(parts)

    # ── 内部 ──────────────────────────────────────────────

    def _cleanup(self):
        self._events = [e for e in self._events if not e.is_expired]

    def _save(self):
        try:
            with open(self._events_file, "w", encoding="utf-8") as f:
                json.dump(
                    serialize_events(self._events), f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            print(f"[情绪引擎] 保存情绪事件失败: {e}")

    def _load(self):
        if not os.path.exists(self._events_file):
            return
        try:
            with open(self._events_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._events = deserialize_events(data)
            self._cleanup()
        except Exception as e:
            print(f"[情绪引擎] 加载情绪事件失败: {e}")
