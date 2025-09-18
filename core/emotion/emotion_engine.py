# emotion/emotion_engine.py

import math
import datetime
from enum import Enum
import json
import httpx
import re
import os  # <--- 1. 新增导入 os 模块，用于检查文件是否存在

# 下级目录导入计算函数
from emotion.f_valence_map import f_valence_map
from emotion.compute_acceptance_ratio import compute_acceptance_ratio
from emotion.compute_arousal_permission_factor import compute_arousal_permission_factor
from emotion.create_mood_instruction import create_mood_instruction

class EmotionState(Enum):
    NORMAL = "正常"
    MELTDOWN = "爆发中"
    RECOVERING = "冷却恢复"

class EmotionEngine:
    def __init__(self, agent_config, llm_config):
        print("情绪引擎已启动...")
        self.STATE_FILE = "emotion_state.json"  # 定义状态文件的路径
        self.FRUSTRATION_THRESHOLD = agent_config.get("FRUSTRATION_THRESHOLD", 10.0)
        self.FRUSTRATION_DECAY_RATE = agent_config.get("FRUSTRATION_DECAY_RATE", 0.95)
        self.MAX_MOOD_AMPLIFICATION_BONUS = agent_config.get("MAX_MOOD_AMPLIFICATION_BONUS", 0.75)
        self.MELTDOWN_DURATION_MINUTES = agent_config.get("MELTDOWN_DURATION_MINUTES", 90.0)
        self.RECOVERY_DURATION_MINUTES = agent_config.get("RECOVERY_DURATION_MINUTES", 10.0)
        self.emotion_profile_matrix = agent_config.get("emotion_profile_matrix", [])
        self.llm_api_for_sentiment = llm_config.get("api")
        self.llm_key_for_sentiment = llm_config.get("key")
        self.llm_model_for_sentiment = llm_config.get("model")
        self.TIME_SCALING_FACTOR = agent_config.get("TIME_SCALING_FACTOR", 5.0)

        # 设置默认情绪状态
        self.valence = 0.0
        self.arousal = 0.0
        self.character_state = EmotionState.NORMAL
        self.latent_emotions = {"frustration": 0.0}
        self.meltdown_start_time = None

        # 覆盖默认值
        self._load_state()

    # ave_state 用于保存, _load_state 用于加载
    def _save_state(self):
        """将当前情绪状态保存到文件"""
        state_to_save = {
            "valence": self.valence,
            "arousal": self.arousal,
            "character_state": self.character_state.value,  # Enum 类型保存其字符串值
            "latent_emotions": self.latent_emotions,
            
            "meltdown_start_time": self.meltdown_start_time.isoformat() if self.meltdown_start_time else None,
        }
        try:
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[情绪引擎] 错误: 保存状态失败 - {e}")

    def _load_state(self):
        """从文件加载情绪状态"""
        if not os.path.exists(self.STATE_FILE):
            print("[情绪引擎] 状态文件不存在，使用默认值初始化。")
            return

        try:
            with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)

            self.valence = loaded_state.get("valence", self.valence)
            self.arousal = loaded_state.get("arousal", self.arousal)
            # 从字符串值恢复
            self.character_state = EmotionState(loaded_state.get("character_state", self.character_state.value))
            self.latent_emotions = loaded_state.get("latent_emotions", self.latent_emotions)

            # 恢复 datetime 对象
            meltdown_time_str = loaded_state.get("meltdown_start_time")
            self.meltdown_start_time = datetime.datetime.fromisoformat(meltdown_time_str) if meltdown_time_str else None

            print(f"[情绪引擎] 成功从文件加载过往情绪状态 (V: {self.valence:.2f}, A: {self.arousal:.2f})。")
        except Exception as e:
            print(f"[情绪引擎] 警告: 加载状态失败，将使用默认值。错误: {e}")
            # 加载失败，则保持 __init__ 中设置的默认值


    def _update_latent_emotions(self, current_frustration: float, sentiment: str, impact_strength: float, current_valence: float) -> float:
        beta = self.FRUSTRATION_DECAY_RATE
        gamma = 1.0
        eta = 0.5
        new_frustration = beta * current_frustration
        if sentiment == "negative":
            # 调用从外部导入的函数
            v_abs = f_valence_map(current_valence)
            mood_bonus = self.MAX_MOOD_AMPLIFICATION_BONUS * (math.exp(v_abs) - 1) / (math.e - 1)
            amplified_impact = impact_strength * (1 + mood_bonus)
            new_frustration += gamma * amplified_impact
        # 调用从外部导入的函数
        new_frustration += eta * f_valence_map(current_valence)
        return new_frustration

    def _compute_valence_pull(self, valence: float, arousal_impact: float) -> float:
        if arousal_impact > 2.5:
            if valence > 0.8:
                return 0.05
            return 0.0
        for lower_bound, upper_bound, pull_strength in self.emotion_profile_matrix:
            if lower_bound == -1.0 and valence <= upper_bound:
                return pull_strength
            if lower_bound < valence <= upper_bound:
                return pull_strength
            if upper_bound == 1.0 and valence > lower_bound:
                return pull_strength
        return 0.0

    # 核心流程函数
    async def _update_emotion_state(self, text: str) -> tuple:

        sentiment_system_prompt = (
            "You are a sophisticated social and emotional analysis expert. Your task is to analyze the LATEST user message. "
            "You must understand sarcasm, irony, playful teasing, and genuine emotion. Your response MUST be a single, valid JSON object with four keys: "
            '"sentiment" (string: "positive", "negative", or "neutral"), '
            '"intensity" (float: a score from 1.0 to 5.0), '
            '"intention" (string: a label like "genuine_praise", "neutral_statement", "harsh_insult"), '
            'and "arousal_impact" (float: a score from -5.0 for calming to +5.0 for exciting).'
        )
        messages_for_sentiment = [{"role": "system", "content": sentiment_system_prompt}, {"role": "user", "content": text}]
        headers = {"Authorization": f"Bearer {self.llm_key_for_sentiment}", "Content-Type": "application/json"}
        payload = {"model": self.llm_model_for_sentiment, "messages": messages_for_sentiment, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.llm_api_for_sentiment, json=payload, headers=headers)
            
            if response.status_code == 200:
                try:
                    llm_content_str = response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    print("[情绪引擎] 错误: LLM返回的JSON结构不完整。")
                    return self.valence, self.arousal, "neutral", 0.0

                json_match = re.search(r'\{.*\}', llm_content_str, re.DOTALL)
                
                if not json_match:
                    print("[情绪引擎] 警告: 在LLM的返回中未找到有效的JSON结构。")
                    return self.valence, self.arousal, "neutral", 0.0

                cleaned_json_str = json_match.group(0)

                try:
                    analysis = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    print("[情绪引擎] 警告: 清洗后的字符串依然不是有效的JSON，本轮情绪无变化。")
                    return self.valence, self.arousal, "neutral", 0.0

                sentiment = analysis.get("sentiment", "neutral")
                intensity = float(analysis.get("intensity", 0.0))
                arousal_impact = float(analysis.get("arousal_impact", 0.0))
                
                if sentiment == "neutral":
                    new_valence = self.valence
                    impact_strength = 0.0
                else:
                    impact_strength = (intensity / 8.1) ** 1.1
                    potential_delta = impact_strength if sentiment == "positive" else -impact_strength

                    acceptance_ratio = compute_acceptance_ratio(self.valence, impact_strength)
                    final_delta = potential_delta * acceptance_ratio
                    new_valence = self.valence + final_delta
                
                base_delta_arousal = arousal_impact / 10.0

                permission_factor = compute_arousal_permission_factor(self.arousal)
                damped_delta_arousal = base_delta_arousal * permission_factor
                valence_pull = self._compute_valence_pull(new_valence, arousal_impact)
                new_arousal = self.arousal + damped_delta_arousal + valence_pull
                
                final_valence = max(-1.0, min(1.0, new_valence))
                final_arousal = max(0.0, min(1.0, new_arousal))

                return final_valence, final_arousal, sentiment, impact_strength
            else:
                print(f"[情绪系统] API请求失败，状态码: {response.status_code}")
                return self.valence, self.arousal, "neutral", 0.0
        except Exception as e:
            print(f"[情绪系统] 情绪状态更新过程中发生错误: {e}")
            return self.valence, self.arousal, "neutral", 0.0

    async def process_emotion(self, text: str) -> str:

        # 状态一：熔断期
        if self.character_state == EmotionState.MELTDOWN:
            elapsed_time = (datetime.datetime.now() - self.meltdown_start_time).total_seconds() / 60.0

            if self.valence >= -0.3 or elapsed_time >= self.MELTDOWN_DURATION_MINUTES:
                print(f"[情绪引擎] 爆发期结束。切换到恢复期。")
                self.character_state = EmotionState.RECOVERING
                self.meltdown_start_time = datetime.datetime.now()
            else:
                x = elapsed_time * self.TIME_SCALING_FACTOR 
                decay_value = 1000 / (x**2 + 1000)
                self.arousal = decay_value
                self.valence = -decay_value

        # 状态二：恢复期
        elif self.character_state == EmotionState.RECOVERING:
            initial_valence = -0.3
            initial_arousal = 0.1
            elapsed_time = (datetime.datetime.now() - self.meltdown_start_time).total_seconds() / 60.0
            progress = min(elapsed_time / self.RECOVERY_DURATION_MINUTES, 1.0)

            if progress >= 1.0:
                self.character_state = EmotionState.NORMAL
                self.valence = 0.0
                self.arousal = 0.0
            else:
                self.valence = initial_valence * (1 - progress)
                self.arousal = initial_arousal * (1 - progress)

        # 状态三：正常状态
        else: # EmotionState.NORMAL
            new_valence, new_arousal, sentiment, impact_strength = await self._update_emotion_state(text)
            
            self.latent_emotions["frustration"] = self._update_latent_emotions(
                self.latent_emotions["frustration"], sentiment, impact_strength, self.valence
            )
            
            self.valence = new_valence
            self.arousal = new_arousal

            if self.latent_emotions["frustration"] > self.FRUSTRATION_THRESHOLD:
                print(f"[情绪引擎] 烦躁值超出阈值，触发情绪熔断！")
                self.character_state = EmotionState.MELTDOWN
                self.meltdown_start_time = datetime.datetime.now()
                self.valence = -1.0
                self.arousal = 1.0
                self.latent_emotions["frustration"] = 0.0

        print(f"[情绪引擎] 状态: {self.character_state.value} | V: {self.valence:.2f}, A: {self.arousal:.2f} | Frustration: {self.latent_emotions['frustration']:.2f}")


        self._save_state()


        return create_mood_instruction(self.valence, self.arousal)