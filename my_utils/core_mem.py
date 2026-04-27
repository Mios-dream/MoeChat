"""
核心记忆模块

基于SQLite存储 + FAISS向量检索
- 快速查询：FAISS向量相似度搜索
- 持久化索引：启动时直接加载，无需重建向量
- 新增时去重：自动检查相似记忆，避免重复添加
"""

import sqlite3
import faiss
import os
import time
import numpy as np
from typing import Any
from models.types.assistant_info import AssistantInfo
from my_utils import embedding
from my_utils import log as Log

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CoreMemory:
    """
    核心记忆管理器

    Attributes:
        agent_id: 角色ID
        thresholds: 相似度阈值，默认0.7
    """

    def __init__(self, agent_config: AssistantInfo):
        self.agent_id = agent_config.name
        self.user = agent_config.user
        # 检索阈值（余弦相似度）
        self.thresholds = 0.7
        # 去重阈值应更严格，避免把新记忆误判为已有
        self.dedupe_threshold = 0.6

        self.data_dir = f"./data/agents/{self.agent_id}"
        self.db_path = f"{self.data_dir}/core_mem.db"
        self.index_path = f"{self.data_dir}/core_mem.index"
        # 核心记忆时间列表，保持与FAISS索引顺序一致
        self.times = []
        # 核心记忆文本列表，保持与FAISS索引顺序一致
        self.mems = []
        self.index: Any = None

        self._init_db()
        self._load_data()
        self._init_index()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """文本编码并做L2归一化，配合IndexFlatIP实现余弦相似度检索。"""
        vectors = embedding.t2vect(texts)
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        faiss.normalize_L2(vectors)
        return vectors

    def _init_db(self):
        """初始化SQLite数据库"""
        os.makedirs(self.data_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                text TEXT NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

    def _load_data(self):
        """从SQLite加载记忆数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT time, text FROM memories ORDER BY time")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            t, text = row
            self.times.append(t)
            self.mems.append(text)

        if not self.mems:
            self._add_initial_memory()

    def _add_initial_memory(self):
        """添加初始记忆"""
        t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        text = "第一次相遇"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (time, text) VALUES (?, ?)",
            (t_n, text),
        )
        conn.commit()
        conn.close()

        self.times.append(t_n)
        self.mems.append(text)

    def _init_index(self):
        """初始化FAISS索引"""
        if not self.mems:
            vect = self._encode_texts(["placeholder"])
            self.index = faiss.IndexFlatIP(vect.shape[1])
            return

        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    index_bytes = f.read()
                self.index = faiss.deserialize_index(
                    np.frombuffer(index_bytes, dtype="uint8")
                )
            except Exception as e:
                Log.logger.warning(f"加载FAISS索引失败: {e}, 将重建索引")
                self._rebuild_index()
        else:
            self._build_index()

    def _build_index(self):
        """构建FAISS索引"""
        try:
            vectors = self._encode_texts(self.mems)
            dim = len(vectors[0])
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)
            self._save_index()
        except Exception as e:
            Log.logger.warning(f"构建FAISS索引失败: {e}")

    def _rebuild_index(self):
        """重建FAISS索引"""
        vectors = self._encode_texts(self.mems)
        dim = len(vectors[0])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self._save_index()

    def _save_index(self):
        """持久化FAISS索引"""
        try:
            index_data = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                f.write(index_data.tobytes())
        except Exception as e:
            print(self.index)
            Log.logger.warning(f"保存FAISS索引失败: {e}")

    def _find_similar(self, text: str, threshold: float | None = None) -> str | None:
        """查找相似记忆（用于去重检查）"""
        if threshold is None:
            threshold = self.dedupe_threshold
        if not self.mems:
            return None

        try:
            vect = self._encode_texts([text])
            D, I = self.index.search(vect, 1)
            top_score = float(D[0][0])
            top_text = self.mems[I[0][0]]
            Log.logger.info(
                f"[核心记忆去重]score={top_score:.4f} query={text} top={top_text}"
            )
            if D[0][0] >= threshold:
                return self.mems[I[0][0]]
        except Exception:
            pass
        return None

    def find_memories(self, msg: str) -> None | str:
        """查找核心记忆"""
        if not self.mems:
            Log.logger.info("核心记忆为空，无法检索")
            return

        try:
            D, I = self.index.search(self._encode_texts([msg]), 5)
            result = ""
            for i in range(len(D[0])):
                if D[0][i] >= self.thresholds:
                    idx = I[0][i]
                    result += f"记忆获取时间：{self.times[idx]}\n{self.mems[idx]}\n"
            if result:
                return result
        except Exception as e:
            Log.logger.warning(f"检索核心记忆失败: {e}")

    def add_memory(self, msg: list):
        """添加新记忆（带去重检查）"""
        new_memories = []

        for m in msg:
            similar = self._find_similar(m)

            if similar:
                Log.logger.info(f"[去重]记忆已存在: {m}")
            else:
                new_memories.append(m)

        if not new_memories:
            Log.logger.info(f"[核心记忆]所有记忆都已存在，跳过添加")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for m in new_memories:
            t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            cursor.execute(
                "INSERT INTO memories (time, text) VALUES (?, ?)",
                (t_n, m),
            )
            self.times.append(t_n)
            self.mems.append(m)

        conn.commit()
        conn.close()

        try:
            vector = self._encode_texts(new_memories)
            self.index.add(vector)
            self._save_index()
        except Exception as e:
            Log.logger.warning(f"更新FAISS索引失败: {e}")

        Log.logger.info(f"[核心记忆]添加了 {len(new_memories)} 条新记忆")

    def get_all_memories(self) -> list:
        """获取所有核心记忆"""
        return self.mems.copy()

    def get_memory_count(self) -> int:
        """获取记忆数量"""
        return len(self.mems)
