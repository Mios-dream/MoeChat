import json
import os
import re
import time
from typing import Optional, List

import yaml
from openai.types.chat import ChatCompletionMessageParam

from Config import Config
from core.emotion.emotion_event import (
    EmotionEvent,
    serialize_events,
    deserialize_events,
    EMOTION_MAP_ZH,
)
from core.llm.llm_client import LLMClient
from core.history_manager import HistoryManager
from models.types.assistant_info import AssistantInfo
from models.types.user_state import UserStateInfo
from my_utils import prompt
from my_utils import log as Log


class EmotionEngine:
    """
    综合情感与关系引擎

    职责：
    1. 事件驱动的短期情绪状态（喜怒哀乐等即时情绪冲击）
    2. 长期关系状态管理（好感度、信任度、羁绊值）
    3. 基于 LLM 的对话情感分析，自动更新关系数值
    4. 生成供 LLM 使用的关系叙事提示词与情绪提示词
    """

    # 好感度阶段配置（基于 affinity 值，阈值依次为各阶段最低值）
    # 描述使用叙事体，让 LLM 自然感受关系状态而非机械执行指令
    AFFINITY_STAGES: list[tuple[str, int, str]] = [
        ("疏远", -50, "你对{user}还很陌生，心里保持着明显的距离感，不太愿意主动靠近"),
        ("陌生", -30, "你和{user}才刚认识不久，还在慢慢熟悉彼此的存在，相处时有些拘谨"),
        ("认识", 0, "你和{user}逐渐熟悉起来了，相处还算愉快，开始期待和他的交流"),
        ("亲近", 25, "你和{user}的关系越来越亲近了，和他在一起时你会感到轻松自在"),
        ("亲密", 50, "你已经很信任{user}了，愿意在他面前展现真实的自己，不设防备"),
        ("挚爱", 80, "{user}是你心中最重要的人，和他在一起的每一刻你都很珍惜"),
    ]
    TRUST_THRESHOLDS: list[tuple[int, str]] = [
        (30, "你发自内心地信任{user}，愿意依赖他，在他面前你感到安全"),
        (0, "你还在观察{user}，不确定他是否值得你完全托付信任"),
        (-30, "{user}的一些言行让你感到不安，你开始对他有所防备"),
        (-50, "你不敢轻易相信{user}了，他曾经让你感到失望或受伤"),
    ]

    def __init__(self, agent_name: str):
        """
        初始化情感与关系引擎

        Parameters:
            agent_name: 助手名称，用于确定持久化路径
        """
        self._agent_name = agent_name

        # ── 短期情绪状态 ──
        self._events_file = os.path.join(
            Config.BASE_AGENTS_PATH, agent_name, "emotion_events.json"
        )
        self._events: list[EmotionEvent] = []

        # ── 长期关系状态 ──
        self.user_state: UserStateInfo = UserStateInfo()

        # ── 依赖（通过 configure() 注入） ──
        self._agent_config: AssistantInfo | None = None
        self._chat_history: HistoryManager | None = None
        self._llm_client: LLMClient | None = None

        self._load_user_state()
        self._load()

    def configure(
        self,
        agent_config: AssistantInfo,
        chat_history: HistoryManager,
    ):
        """
        注入外部依赖

        在引擎初始化后调用，将 Assistant 中的配置和核心依赖注入引擎。

        Parameters:
            agent_config: 助手完整配置（含角色名、性格、描述等所有信息）
            chat_history: 聊天历史管理器引用
        """
        self._agent_config = agent_config
        self._chat_history = chat_history
        self._llm_client = LLMClient(model_key="LLM")

    # ==================== 公开接口：情绪事件 ====================

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

    # ==================== 公开接口：关系状态 ====================

    def get_affection_prompt(self) -> str:
        """
        生成好感度叙事提示词

        用叙事体描述关系状态，让 LLM 自然感受而非机械执行指令
        """
        if self._agent_config is None:
            return "<关系状态></关系状态>"

        user_name = self._agent_config.user or "用户"
        stage_name, stage_desc = self._get_affinity_stage(self.user_state.affinity)
        trust_desc = self._get_trust_level(self.user_state.trust)
        days_known = max(1, (int(time.time()) - self.user_state.firstMeetTime) // 86400)

        affection_text = (
            f"你和{user_name}目前的关系是「{stage_name}」。\n"
            f"{stage_desc.format(user=user_name)}\n"
            f"{trust_desc.format(user=user_name)}\n"
            f"你们已经相识 {days_known} 天，共同经历了 {self.user_state.bond} 次有意义的互动——"
            f"这些经历构成了你对他的真实感受。\n"
            f"请自然地体会这些感受，像真实的自己一样去回应，无需刻意强调。"
        )
        return f"<关系状态>{affection_text}</关系状态>"

    async def update_affection(self):
        """
        异步更新好感度 + 信任度 + 羁绊 + 情绪事件

        基于最近对话进行 LLM 分析，更新关系数值并记录情绪冲击。
        """
        result = await self._calculate_affection_change()

        # 更新好感度（-50~100）
        self.user_state.affinity = max(
            -50, min(100, self.user_state.affinity + result["affinity_change"])
        )
        # 更新信任度（-50~100）
        self.user_state.trust = max(
            -50, min(100, self.user_state.trust + result["trust_change"])
        )
        # 更新羁绊（只增不减）
        self.user_state.bond += result["bond_increment"]

        # 记录情绪事件
        self.process_event(result["emotional_impact"])

        self._save_user_state()

        Log.logger.info(
            f"助手 {self._agent_name} 状态更新: "
            f"affinity={self.user_state.affinity}, "
            f"trust={self.user_state.trust}, "
            f"bond={self.user_state.bond}"
        )

    def get_relationship_summary(self) -> dict:
        """
        获取当前关系状态摘要

        Returns:
            包含 affinity、trust、bond、firstMeetTime 的字典
        """
        return {
            "affinity": self.user_state.affinity,
            "trust": self.user_state.trust,
            "bond": self.user_state.bond,
            "firstMeetTime": self.user_state.firstMeetTime,
            "days_known": max(
                1, (int(time.time()) - self.user_state.firstMeetTime) // 86400
            ),
        }

    # ==================== 内部：关系分析 ====================

    def _get_interaction_summary(self, window: int = 20) -> str:
        """
        统计最近互动的构成，纯描述不做价值判断

        用户可能喜欢主动对话，也可能只是享受后台陪伴——两者都正常。
        统计只反映互动模式，不作为评价用户好坏的依据。
        """
        if self._chat_history is None:
            return ""
        recent = self._chat_history.get_raw_recent(window)
        if not recent:
            return ""

        user_turns = 0
        assistant_initiated = 0
        responded = 0

        i = 0
        while i < len(recent):
            if recent[i]["role"] == "user":
                user_turns += 1
                i += 1
            elif recent[i]["role"] == "assistant":
                if i == 0 or recent[i - 1]["role"] != "user":
                    assistant_initiated += 1
                    if i + 1 < len(recent) and recent[i + 1]["role"] == "user":
                        responded += 1
                        i += 2
                        continue
                i += 1
            else:
                i += 1

        parts = []
        if user_turns:
            parts.append(f"用户主动对话 {user_turns} 次")
        if assistant_initiated:
            parts.append(f"助手主动互动 {assistant_initiated} 次")
        return "；".join(parts) if parts else ""

    async def _calculate_affection_change(self) -> dict:
        """
        统一 LLM 分析：好感度 + 信任度 + 羁绊 + 情绪冲击

        分析仅基于实际对话内容的质量，而非互动频率或响应率。
        用户选择不回应助手主动互动是完全正常的行为，
        好感度不因此惩罚用户——它只反映说了什么，而非说了多少次。

        构建消息链以利用 LLM KV 缓存：
          1. system（静态角色信息 → 缓存命中）
          2. 最近对话历史（上下文）
          3. user（当前状态 + 互动统计 + 分析请求）
        """
        if (
            self._llm_client is None
            or self._chat_history is None
            or self._agent_config is None
        ):
            return {
                "affinity_change": 0,
                "trust_change": 0,
                "bond_increment": 0,
                "emotional_impact": {
                    "emotion": "neutral",
                    "intensity": 0.0,
                    "reason": "",
                },
            }

        cfg = self._agent_config
        current_emotions = self.get_mood_prompt()
        days_known = max(1, (int(time.time()) - self.user_state.firstMeetTime) // 86400)
        default_result = {
            "affinity_change": 0,
            "trust_change": 0,
            "bond_increment": 0,
            "emotional_impact": {
                "emotion": "neutral",
                "intensity": 0.0,
                "reason": "",
            },
        }

        # 合并所有角色描述字段，确保 LLM 获得完整的角色认知
        personality = cfg.personality or ""
        description = "\n".join(
            filter(
                None,
                [
                    cfg.description,
                    cfg.extraDescription,
                    cfg.customPrompt,
                ],
            )
        )

        # 1. system：静态角色信息（可缓存）
        system_content = prompt.analysis_system_prompt.format(
            char=cfg.name,
            user=cfg.user,
            personality=personality,
            description=description,
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
        ]

        # 2. 最近对话上下文（跳过压缩摘要，避免失真）
        recent = self._chat_history.get_raw_recent(10)
        messages.extend(recent)

        # 3. 分析请求
        request_text = (
            f"当前关系状态：\n"
            f"- 好感度：{self.user_state.affinity}\n"
            f"- 信任度：{self.user_state.trust}\n"
            f"- 羁绊值：{self.user_state.bond}\n"
            f"- 相识天数：{days_known}\n"
            f"- 当前情绪状态：{current_emotions or '平静'}\n"
            f"- 互动概况：{self._get_interaction_summary()}\n\n"
            f"请根据以上完整对话，分析最新一轮对话对"
            f"{cfg.name}和{cfg.user}之间关系的影响。\n\n"
            f"1. affinity_change: 好感度变化（整数 -3 到 +3）\n"
            f"2. trust_change: 信任度变化（整数 -3 到 +3，降快升慢）\n"
            f"3. bond_increment: 羁绊增量（整数 0 到 2）\n"
            f'4. emotional_impact: {{"emotion":"...", "intensity":0.0~1.0, '
            f'"reason":"用一句话描述触发情绪的具体情景"}}\n\n'
            f'{{"affinity_change":0,"trust_change":0,"bond_increment":0,'
            f'"emotional_impact":{{"emotion":"neutral","intensity":0.0,"reason":""}}}}'
        )
        messages.append({"role": "user", "content": request_text})

        # 4. 请求 LLM
        try:
            content = await self._llm_client.request(messages)
        except Exception as e:
            Log.logger.error("LLM 好感度判断失败:", e)
            return default_result

        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            return default_result

        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default_result

        affinity_change = max(-3, min(3, result.get("affinity_change", 0)))
        trust_change = max(-3, min(3, result.get("trust_change", 0)))
        bond_increment = max(0, min(2, result.get("bond_increment", 0)))
        emotional_impact = result.get(
            "emotional_impact",
            {"emotion": "neutral", "intensity": 0.0, "reason": ""},
        )

        Log.logger.info(
            f"好感度分析: affinity={affinity_change}, trust={trust_change}, "
            f"bond={bond_increment}, emotion={emotional_impact}"
        )
        return {
            "affinity_change": affinity_change,
            "trust_change": trust_change,
            "bond_increment": bond_increment,
            "emotional_impact": emotional_impact,
        }

    # ==================== 内部：阈值工具 ====================

    def _get_affinity_stage(self, affinity: int) -> tuple[str, str]:
        """
        根据好感度数值获取阶段名和描述

        Parameters:
            affinity: 好感度数值

        Returns:
            (阶段名, 一句话描述)
        """
        for name, threshold, desc in reversed(self.AFFINITY_STAGES):
            if affinity >= threshold:
                return name, desc
        return self.AFFINITY_STAGES[0][0], self.AFFINITY_STAGES[0][2]

    def _get_trust_level(self, trust: int) -> str:
        """根据信任度返回一句话状态"""
        for threshold, desc in reversed(self.TRUST_THRESHOLDS):
            if trust >= threshold:
                return desc
        return self.TRUST_THRESHOLDS[-1][1]

    # ==================== 内部：持久化 ====================

    def _save_user_state(self):
        """保存用户私有状态到 user_state.yaml"""
        config_path = os.path.join(
            Config.BASE_AGENTS_PATH, self._agent_name, "user_state.yaml"
        )
        try:
            self.user_state.updatedAt = int(time.time())
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.user_state.model_dump(),
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            Log.logger.error(f"保存用户状态失败: {e}")

    def _load_user_state(self):
        """加载用户私有状态"""
        config_path = os.path.join(
            Config.BASE_AGENTS_PATH, self._agent_name, "user_state.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.user_state = UserStateInfo.from_dict(yaml.safe_load(f) or {})
        else:
            self.user_state = UserStateInfo()

    # ==================== 内部：情绪事件持久化 ====================

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
