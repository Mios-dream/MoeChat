"""
keyword_loader.py - 词库加载器
职责：加载JSON词库文件，提供词库查询接口
"""

import json
import os
from typing import Dict, List, Optional


class KeywordLoader:
    def __init__(self, keywords_dir: str = "keywords/"):
        self.keywords_dir = keywords_dir
        self.all_keywords = {}
        self.is_loaded = False
        self.keyword_files = [
            "emotions.json",
            "behavior_emotions.json", 
            "functional_types.json"
        ]
    
    def load_all_keywords(self) -> bool:
        """加载所有词库文件"""
        try:
            print("[词库加载器] 开始加载词库...")
            self.all_keywords = {}
            
            for file_name in self.keyword_files:
                file_path = os.path.join(self.keywords_dir, file_name)
                
                if not os.path.exists(file_path):
                    print(f"[词库加载器] 警告：{file_path} 不存在")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    self.all_keywords.update(file_data)
                    print(f"[词库加载器] 已加载 {file_name}")
            
            self.is_loaded = True
            print(f"[词库加载器] 加载完成，共 {len(self.all_keywords)} 个情感类别")
            return True
            
        except Exception as e:
            print(f"[词库加载器] 加载失败：{e}")
            return False
    
    def get_all_keywords_for_emotion(self, emotion_name: str) -> List[str]:
        """获取指定情感的所有关键词（扁平化列表）"""
        if not self.is_loaded:
            self.load_all_keywords()
        
        emotion_data = self.all_keywords.get(emotion_name)
        if not emotion_data:
            return []
        
        all_words = []
        for category, words in emotion_data.items():
            if isinstance(words, list):
                all_words.extend(words)
        
        return all_words
    
    def get_all_emotions(self) -> List[str]:
        """获取所有情感类别名称"""
        if not self.is_loaded:
            self.load_all_keywords()
        
        return list(self.all_keywords.keys())
    
    def get_emotion_keywords(self, emotion_name: str) -> Optional[Dict[str, List[str]]]:
        """
        获取指定情感的所有关键词分类字典
        
        Args:
            emotion_name: 情感名称（如"Happy", "Angry"等）
            
        Returns:
            Dict[str, List[str]]: 该情感的词汇分类字典，如果不存在返回None
        """
        if not self.is_loaded:
            print("[词库加载器] 警告：词库未加载，尝试自动加载...")
            if not self.load_all_keywords():
                return None
        
        return self.all_keywords.get(emotion_name)
    
    def reload_keywords(self) -> bool:
        """重新加载词库（热更新）"""
        print("[词库加载器] 重新加载词库...")
        return self.load_all_keywords()
    
    def get_statistics(self) -> Dict[str, any]:
        """获取词库统计信息"""
        if not self.is_loaded:
            return {"error": "词库未加载"}
        
        stats = {
            "total_emotions": len(self.all_keywords),
            "emotions": {},
            "total_keywords": 0
        }
        
        for emotion_name, emotion_data in self.all_keywords.items():
            if isinstance(emotion_data, dict):
                emotion_keyword_count = sum(len(words) for words in emotion_data.values() if isinstance(words, list))
                stats["emotions"][emotion_name] = emotion_keyword_count
                stats["total_keywords"] += emotion_keyword_count
        
        return stats


if __name__ == "__main__":
    # 测试
    loader = KeywordLoader()
    if loader.load_all_keywords():
        print("所有情感类别:", loader.get_all_emotions())
        
        # 测试统计功能
        stats = loader.get_statistics()
        print(f"\n词库统计: 共{stats['total_emotions']}个情感类别，{stats['total_keywords']}个关键词")
        
        # 测试热更新
        print("\n测试热更新...")
        loader.reload_keywords()
        
        happy_words = loader.get_all_keywords_for_emotion("Happy")
        print("Happy关键词:", happy_words[:5] if happy_words else "无")