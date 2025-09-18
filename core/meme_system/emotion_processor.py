"""
emotion_processor.py - 情感处理器
职责：
1. 分析LLM文本，使用权重计分系统计算各情感得分
2. 选择最佳情感文件夹
3. 从文件夹中随机选择表情包
4. 控制发送概率
5. 格式化为SSE响应数据
"""

import json
import os
import random
import re
from typing import Dict, List, Tuple, Optional
from .keyword_loader import KeywordLoader


class EmotionProcessor:
    def __init__(self, config_path: str = "config.json"):
        # 先设置默认值
        self.config = {}
        self.scoring_weights = {}
        self.probability_config = {}
        self.context_keywords = {}
        self.keyword_loader = None
        
        # 加载配置文件
        if not self.load_config(config_path):
            print("[情感处理器] 配置加载失败，使用默认配置")
            self._set_default_config()
        
        # 初始化词库加载器
        keywords_dir = self.config.get("paths", {}).get("keywords_dir", "./")
        self.keyword_loader = KeywordLoader(keywords_dir)
    
    def load_config(self, config_path: str) -> bool:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(config_path):
                print(f"[情感处理器] 配置文件 {config_path} 不存在")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 提取各部分配置
            self.scoring_weights = self.config.get("scoring_weights", {})
            self.probability_config = self.config.get("probability_config", {})
            self.context_keywords = self.config.get("context_keywords", {})
            
            print("[情感处理器] 配置文件加载成功")
            print(f"  - 计分权重: {len(self.scoring_weights)} 项")
            print(f"  - 概率配置: {len(self.probability_config)} 项") 
            print(f"  - 语境关键词: {len(self.context_keywords)} 类")
            return True
            
        except Exception as e:
            print(f"[情感处理器] 配置文件加载失败：{e}")
            return False
    
    def _set_default_config(self):
        """设置默认配置（当配置文件加载失败时使用）"""
        self.config = {
            "paths": {
                "memes_base_dir": "memes/",
                "keywords_dir": "./",
                "expression_url_prefix": "/expression/memes/"
            }
        }
        
        self.scoring_weights = {
            "基础词汇": 10, "语气词": 8, "感叹词": 7, "网络用语": 6,
            "强化词_高": 1.5, "强化词_中": 1.3, "弱化词": 0.8,
            "感叹句": 3, "疑问句": 2, "重复标点": 2,
            "最低阈值": 3, "default兜底": 5
        }
        
        self.probability_config = {
            "main_meme_probability": 1.0,
            "default_meme_probability": 1.0
        }
        
        self.context_keywords = {
            "强化词_高": ["太", "超", "特别", "非常", "极其"],
            "强化词_中": ["真的", "确实", "的确", "实在"],
            "弱化词": ["有点", "稍微", "还算", "不太", "不是很"],
            "积极极性": ["好", "棒", "赞", "优秀", "完美", "成功"],
            "消极极性": ["坏", "糟", "差", "失败", "问题", "错误"]
        }
    
    def calculate_emotion_scores(self, llm_text: str) -> Dict[str, float]:
        """
        计算各情感类别的得分
        
        Args:
            llm_text: LLM回复文本
            
        Returns:
            Dict[str, float]: 各情感类别及其得分
        """
        print(f"[情感计分] 开始分析文本：{llm_text}")
        
        # 确保词库已加载
        if not self.keyword_loader.is_loaded:
            self.keyword_loader.load_all_keywords()
        
        # 获取所有情感类别
        all_emotions = self.keyword_loader.get_all_emotions()
        emotion_scores = {emotion: 0.0 for emotion in all_emotions}
        
        # ===== 第一阶段：直接文本匹配计分 =====
        print("[情感计分] 第一阶段：直接文本匹配")
        for emotion in all_emotions:
            emotion_data = self.keyword_loader.get_emotion_keywords(emotion)
            if not emotion_data:
                continue
            
            emotion_score = 0.0
            matched_words = []
            
            # 遍历该情感的各个词汇分类
            for category, keywords in emotion_data.items():
                if not isinstance(keywords, list):
                    continue
                
                # 检查每个关键词是否在文本中
                for keyword in keywords:
                    if keyword in llm_text:
                        # 根据词汇分类给不同权重
                        if category in self.scoring_weights:
                            score = self.scoring_weights[category]
                        else:
                            score = self.scoring_weights.get("基础词汇", 5)  # 默认分数
                        
                        emotion_score += score
                        matched_words.append(f"{keyword}({category}:+{score})")
            
            emotion_scores[emotion] = emotion_score
            if matched_words:
                print(f"  {emotion}: {emotion_score}分 - 匹配词汇: {matched_words}")
        
        # ===== 第二阶段：语境强化分析 =====
        print("[情感计分] 第二阶段：语境强化分析")
        context_multiplier = 1.0
        context_bonus = 0
        
        # 检查强化/弱化词
        for word_type, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in llm_text:
                    if word_type in ["强化词_高", "强化词_中"]:
                        multiplier = self.scoring_weights[word_type]
                        context_multiplier = max(context_multiplier, multiplier)
                        print(f"  发现强化词 '{keyword}': 倍数调整为 {multiplier}")
                    elif word_type == "弱化词":
                        context_multiplier = min(context_multiplier, self.scoring_weights[word_type])
                        print(f"  发现弱化词 '{keyword}': 倍数调整为 {self.scoring_weights[word_type]}")
        
        # 检查句式特征
        if llm_text.endswith('!') or '!' in llm_text:
            context_bonus += self.scoring_weights["感叹句"]
            print(f"  发现感叹句: +{self.scoring_weights['感叹句']}分")
        
        if llm_text.endswith('?') or '?' in llm_text:
            context_bonus += self.scoring_weights["疑问句"] 
            print(f"  发现疑问句: +{self.scoring_weights['疑问句']}分")
        
        # 检查重复标点
        if re.search(r'[!?。]{2,}|\.{3,}', llm_text):
            context_bonus += self.scoring_weights["重复标点"]
            print(f"  发现重复标点: +{self.scoring_weights['重复标点']}分")
        
        # ===== 第三阶段：应用语境调整 =====
        print("[情感计分] 第三阶段：应用语境调整")
        final_scores = {}
        for emotion, base_score in emotion_scores.items():
            if base_score > 0:
                # 对有基础分数的情感应用语境调整
                adjusted_score = base_score * context_multiplier + context_bonus
                final_scores[emotion] = adjusted_score
                if adjusted_score != base_score:
                    print(f"  {emotion}: {base_score} → {adjusted_score} (×{context_multiplier} +{context_bonus})")
            else:
                final_scores[emotion] = base_score
        
        # ===== 第四阶段：特殊处理default =====
        if "default" in final_scores:
            max_score = max(final_scores.values())
            if max_score < self.scoring_weights["最低阈值"]:
                final_scores["default"] = self.scoring_weights["default兜底"]
                print(f"[情感计分] 所有得分都很低，default获得兜底分数: {self.scoring_weights['default兜底']}")
        
        # 显示最终得分
        print("[情感计分] 最终得分排序:")
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_scores[:5]:  # 只显示前5名
            if score > 0:
                print(f"  {emotion}: {score}分")
        
        return final_scores
    
    def select_best_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[Optional[str], float]:
        """
        选择得分最高的情感
        
        Args:
            emotion_scores: 各情感得分字典
            
        Returns:
            Tuple[Optional[str], float]: (最佳情感名称, 得分)
        """
        if not emotion_scores:
            return None, 0.0
        
        # 找到最高分
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        
        print(f"[情感选择] 最佳情感: {best_emotion} (得分: {best_score})")
        return best_emotion, best_score
    
    def select_meme_file(self, emotion_folder: str) -> Optional[str]:
        """
        从指定情感文件夹中随机选择一个表情包文件
        
        Args:
            emotion_folder: 情感文件夹名称
            
        Returns:
            Optional[str]: 表情包文件名，如果失败返回None
        """
        # 从配置中获取memes基础目录
        memes_base_dir = self.config.get("paths", {}).get("memes_base_dir", "memes/")
        folder_path = os.path.join(memes_base_dir, emotion_folder)
        
        if not os.path.exists(folder_path):
            print(f"[表情包选择] 错误：文件夹 {folder_path} 不存在")
            return None
        
        # 从配置中获取支持的图片格式
        image_extensions = self.config.get("system_settings", {}).get(
            "supported_image_formats", 
            ['.png', '.jpg', '.jpeg', '.gif', '.webp']
        )
        
        meme_files = []
        
        try:
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    meme_files.append(file)
            
            if not meme_files:
                print(f"[表情包选择] 警告：文件夹 {emotion_folder} 中没有图片文件")
                return None
            
            # 随机选择一个文件
            selected_file = random.choice(meme_files)
            print(f"[表情包选择] 从 {emotion_folder} 中选择: {selected_file}")
            return selected_file
            
        except Exception as e:
            print(f"[表情包选择] 扫描文件夹失败：{e}")
            return None
    
    def check_send_probability(self, emotion_name: str, emotion_score: float) -> bool:
        """
        检查是否应该发送表情包（概率控制）
        
        Args:
            emotion_name: 情感名称
            emotion_score: 情感得分
            
        Returns:
            bool: 是否发送表情包
        """
        # 判断是明确匹配还是default兜底
        if emotion_name == "default" or emotion_score < self.scoring_weights["最低阈值"]:
            # default情况，使用default概率
            probability = self.probability_config["default_meme_probability"]
            prob_type = "default"
        else:
            # 明确匹配情况，使用主概率
            probability = self.probability_config["main_meme_probability"] 
            prob_type = "main"
        
        # 掷骰子
        random_value = random.random()
        will_send = random_value <= probability
        
        print(f"[概率控制] {prob_type}概率检查: {probability} vs {random_value:.3f} → {'发送' if will_send else '不发送'}")
        return will_send
    
    def format_sse_response(self, emotion_folder: str, meme_filename: str) -> str:
        """
        格式化为SSE响应数据
        
        Args:
            emotion_folder: 情感文件夹名称
            meme_filename: 表情包文件名
            
        Returns:
            str: SSE格式的响应数据
        """
        # 从配置中获取URL前缀
        url_prefix = self.config.get("paths", {}).get("expression_url_prefix", "/expression/memes/")
        
        response_data = {
            "file": None,
            "message": f"{{img}}{url_prefix}{emotion_folder}/{meme_filename}",
            "done": False
        }
        return f"data: {json.dumps(response_data)}\n\n"
    
    def process_emotion(self, llm_response_text: str) -> Optional[str]:
        """
        主处理函数：分析LLM回复文本的情感并返回SSE响应
        
        Args:
            llm_response_text: LLM的回复文本
            
        Returns:
            Optional[str]: SSE格式响应数据，如果不发送则返回None
        """
        print(f"\n[情感处理器] 开始处理LLM回复: '{llm_response_text}'")
        
        try:
            # 第一步：计算各情感得分
            emotion_scores = self.calculate_emotion_scores(llm_response_text)
            
            # 第二步：选择最佳情感
            best_emotion, best_score = self.select_best_emotion(emotion_scores)
            
            if not best_emotion:
                print("[情感处理器] 没有找到合适的情感，不发送表情包")
                return None
            
            # 第三步：检查发送概率
            if not self.check_send_probability(best_emotion, best_score):
                print("[情感处理器] 概率检查未通过，不发送表情包")
                return None
            
            # 第四步：选择表情包文件
            meme_filename = self.select_meme_file(best_emotion)
            if not meme_filename:
                print("[情感处理器] 未找到表情包文件，不发送")
                return None
            
            # 第五步：格式化响应
            sse_response = self.format_sse_response(best_emotion, meme_filename)
            print(f"[情感处理器] 成功生成响应: {best_emotion}/{meme_filename}")
            return sse_response
            
        except Exception as e:
            print(f"[情感处理器] 处理失败：{e}")
            return None


if __name__ == "__main__":
    # 测试情感处理器
    processor = EmotionProcessor()
    
    # 测试用例
    test_texts = [
        "哈哈，太好了！我很开心！",
        "唉，真是糟糕...",
        "这个问题很奇怪，我不太懂？？？",
        "还行吧，没什么特别的"
    ]
    
    for text in test_texts:
        print("=" * 50)
        result = processor.process_emotion(text)
        if result:
            print(f"响应结果: {result.strip()}")
        else:
            print("不发送表情包")