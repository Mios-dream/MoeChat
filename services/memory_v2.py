"""
记忆系统 v2

核心设计：
1. 助手自主决策：通过 remember tool 由助手自己判断什么重要
2. 四象限分类：about_user / about_us / about_world / about_self
3. 双层级存储：core（永久）和 normal（随时间衰减）
4. 统一向量检索：替代原来 core_mem + long_mem 两套系统
5. 记忆衰减：normal 类型记忆随时间和访问频率渐进遗忘

日记系统（独立于 memories 表）：
- chat_turns 表：原始对话记录，跨天时触发日记生成
- diary_days 表：结构化日记存储（day / event_summary / content）
- 日记不混入 memories 表，通过 get_context() 同时检索两者
- 日记内容加入独立 FAISS 索引，支持语义检索
- API 查询直接走 diary_days 表，支持日期过滤和分页

增强特性：
1. 并发安全：threading.RLock 保护所有共享内存 + FAISS 索引
2. 访问计数缓存：避免 search() 中每条记忆都查一次 DB
3. 日记语义检索：日记内容独立 FAISS 索引，替代硬编码最近 N 天
4. 记忆关联网络：memory_links 表 + 语义关联自动发现 + 检索提升
5. recall 工具支持：供助手主动查询自身记忆（增强人格化）
6. 更新记忆工具：update_memory 工具，支持覆盖/修正已有信息

架构替代关系：
- core_mem.py -> 合并入此模块
- long_mem.py -> 合并入此模块
"""

import sqlite3
import faiss
import os
import time
import math
import threading
import numpy as np
from datetime import datetime
from typing import Any
from models.types.assistant_info import AssistantInfo
from my_utils import embedding
from my_utils import log as Log
from core.llm.llm_client import LLMClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MemoryV2:
    """
    统一记忆管理器 v2

    四象限记忆 + 双层级（core/normal）+ FAISS 语义检索 + 时间衰减。
    每条记忆由助手自主通过 remember tool 写入，无需外部 LLM 提取。

    新增日记功能：
    - chat_turns 表记录原始对话，跨天时触发日记生成
    - 日记以 about_us 类别存入 memories，带 [日记] 前缀
    - 日记同样可通过语义检索注入 prompt

    Attributes:
        categories: 记忆四象限分类
        DEFAULT_DECAY_RATE: normal 类型记忆的默认衰减速率
        ARCHIVE_THRESHOLD: 衰减后低于此阈值的记忆自动归档
    """

    CATEGORIES = ["about_user", "about_us", "about_world", "about_self"]
    """记忆四象限分类"""

    CATEGORY_DESCRIPTIONS = {
        "about_user": "关于{user}的个人信息、偏好、经历和习惯",
        "about_us": "关于你和{user}的互动、关系发展和共同经历",
        "about_world": "在对话中了解到的外部知识和信息",
        "about_self": "你对自己的认知、价值观和情感的反思与成长",
    }
    """各分类的中文描述，用于系统 prompt"""

    DEFAULT_DECAY_RATE = 0.001
    """normal 类型记忆的默认每日衰减速率"""

    ARCHIVE_THRESHOLD = 0.1
    """衰减后有效重要度低于此阈值的记忆会被归档"""

    MIN_TURNS_FOR_DIARY = 6
    """当日对话轮次达到此数量才触发日记生成"""

    # 记忆工具系统提示词模板
    MEMORY_INSTRUCTION_PROMPT = """
【记忆系统说明】
你拥有一个名为 remember 的工具，可以记录重要信息。如果有多个重要信息，则可以多次调用进行记录。

记忆分为四个类别：
1. about_user - 关于{user}的信息：偏好、经历、习惯、性格特征等
2. about_us - 关于你和{user}的关系：共同经历、互动感受、关系变化等
3. about_world - 从对话中学到的外部知识
4. about_self - 你对自己的认知和感受的变化

使用准则：
- 只有真正重要的事情才值得记录，不要事无巨细
- 用你自己的视角来描述记忆内容
- 极其重要的信息（影响性格和关系的核心记忆）设为 core 类型和 0.9+ 重要度
- 普通重要的设为 normal 类型，会随时间逐渐淡忘
"""

    # 主动回忆指令（recall 工具说明）
    RECALL_INSTRUCTION_PROMPT = """
【主动回忆说明】
当你对某件事印象模糊、或需要确认之前的对话细节时，
可以使用 recall 工具主动查询自己的记忆。
这能帮助你更好地记住和{user}之间的互动。

使用准则：
- 在需要确认信息时使用，不要频繁调用
- 查询描述越具体，返回结果越准确
"""

    @staticmethod
    def build_system_prompt(char: str, user: str) -> str:
        """构建记忆系统说明（独立 system 角色），包含 remember + recall 工具指令"""
        return (
            MemoryV2.MEMORY_INSTRUCTION_PROMPT.format(char=char, user=user)
            + "\n\n"
            + MemoryV2.RECALL_INSTRUCTION_PROMPT.format(char=char, user=user)
        )

    LINK_SIMILARITY_MIN = 0.40
    """自动关联的记忆相似度下限"""

    LINK_SIMILARITY_MAX = 0.74
    """自动关联的记忆相似度上限（超过此值为重复而非关联）"""

    LINK_BOOST_FACTOR = 0.15
    """关联记忆在检索时获得的分数提升系数"""

    # 日记检索时的时间衰减因子（天），30 天前的日记语义分打 5 折
    DIARY_TIME_DECAY_DAYS = 30.0

    def __init__(self, agent_config: AssistantInfo, firstMeetTime: int = 0):
        self.agent_config = agent_config
        self.agent_id = agent_config.name
        self.user = agent_config.user
        self.char = agent_config.name
        self.firstMeetTime = firstMeetTime

        self.data_dir = f"./data/agents/{self.agent_id}"
        self.db_path = f"{self.data_dir}/memory_v2.db"
        self.index_path = f"{self.data_dir}/memory_v2.index"
        self.diary_index_path = f"{self.data_dir}/diary_v2.index"

        # 并发安全锁（保护所有共享内存 + FAISS 索引）
        self._lock = threading.RLock()

        # 内存缓存（与 FAISS 索引顺序保持严格一致）
        self.ids: list[int] = []
        self.contents: list[str] = []
        self.categories: list[str] = []
        self.importances: list[float] = []
        self.memory_types: list[str] = []
        self.archived: list[bool] = []
        self.created_times: list[str] = []
        self.decay_rates: list[float] = []
        self.access_counts: list[int] = []

        self.index: Any = None

        # 日记 FAISS 索引（独立于 memories，支持语义检索）
        self.diary_index: Any = None
        self.diary_day_list: list[str] = []
        self.diary_content_list: list[str] = []

        # 日记 LLM 客户端（lazy 初始化）
        self._diary_llm: Any = None

        # 角色画像直接从 agent_config 读取

        self._init_db()
        self._load_data()
        self._init_index()
        self._init_diary_index()

    # ============================================================
    # 向量编码
    # ============================================================

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """文本编码并做 L2 归一化，配合 IndexFlatIP 实现余弦相似度检索"""
        vectors = embedding.t2vect(texts)
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        faiss.normalize_L2(vectors)
        return vectors

    # ============================================================
    # 数据库初始化与加载
    # ============================================================

    def _init_db(self):
        """初始化 SQLite 数据库"""
        os.makedirs(self.data_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                memory_type TEXT NOT NULL DEFAULT 'normal',
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL NOT NULL DEFAULT 0.001,
                archived INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_archived
            ON memories(archived)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category)
        """)

        # 日记用对话原始记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_sec INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_turns_ts
            ON chat_turns(timestamp_sec)
        """)

        # 日记存储表（独立于 memories，按日期结构化存储）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diary_days (
                day TEXT PRIMARY KEY,
                event_summary TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # 记忆关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id_a INTEGER NOT NULL,
                memory_id_b INTEGER NOT NULL,
                similarity REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                UNIQUE(memory_id_a, memory_id_b),
                FOREIGN KEY (memory_id_a) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id_b) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_links_a
            ON memory_links(memory_id_a)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_links_b
            ON memory_links(memory_id_b)
        """)

        conn.commit()
        conn.close()

    def _load_data(self):
        """从 SQLite 加载所有记忆到内存缓存（线程安全）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, category, content, importance, memory_type, "
                "created_at, decay_rate, archived, access_count "
                "FROM memories ORDER BY id"
            )
            rows = cursor.fetchall()
            conn.close()

            self.ids.clear()
            self.categories.clear()
            self.contents.clear()
            self.importances.clear()
            self.memory_types.clear()
            self.created_times.clear()
            self.decay_rates.clear()
            self.archived.clear()
            self.access_counts.clear()

            for row in rows:
                self.ids.append(row[0])
                self.categories.append(row[1])
                self.contents.append(row[2])
                self.importances.append(row[3])
                self.memory_types.append(row[4])
                self.created_times.append(row[5])
                self.decay_rates.append(row[6])
                self.archived.append(bool(row[7]))
                self.access_counts.append(row[8] if row[8] is not None else 0)

    def _load_diary_index_data(self):
        """从 SQLite 加载日记数据到内存缓存（线程安全，_lock 由调用者持有）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT day, content FROM diary_days ORDER BY day"
        )
        rows = cursor.fetchall()
        conn.close()

        self.diary_day_list.clear()
        self.diary_content_list.clear()
        for day, content in rows:
            self.diary_day_list.append(day)
            self.diary_content_list.append(content)

    # ============================================================
    # FAISS 索引管理（Memories）
    # ============================================================

    def _init_index(self):
        """初始化或加载 FAISS 索引"""
        with self._lock:
            if not self.contents:
                vect = self._encode_texts(["占位"])
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
                    Log.logger.warning(f"加载 FAISS 索引失败: {e}，将重建")
                    self._rebuild_index()
            else:
                self._build_index()

    def _build_index(self):
        """根据当前内存缓存构建 FAISS 索引（调用者持有 _lock）"""
        if not self.contents:
            return
        try:
            vectors = self._encode_texts(self.contents)
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)
            self._save_index()
        except Exception as e:
            Log.logger.warning(f"构建 FAISS 索引失败: {e}")

    def _rebuild_index(self):
        """重建 FAISS 索引（调用者持有 _lock）"""
        if not self.contents:
            return
        vectors = self._encode_texts(self.contents)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self._save_index()

    def _save_index(self):
        """持久化 FAISS 索引到磁盘（调用者持有 _lock）"""
        try:
            index_data = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                f.write(index_data.tobytes())
        except Exception as e:
            Log.logger.warning(f"保存 FAISS 索引失败: {e}")

    # ============================================================
    # FAISS 索引管理（Diary）
    # ============================================================

    def _init_diary_index(self):
        """初始化或加载日记 FAISS 索引"""
        with self._lock:
            self._load_diary_index_data()
            if not self.diary_content_list:
                dim = self._encode_texts(["占位"]).shape[1]
                self.diary_index = faiss.IndexFlatIP(dim)
                return

            if os.path.exists(self.diary_index_path):
                try:
                    with open(self.diary_index_path, "rb") as f:
                        index_bytes = f.read()
                    self.diary_index = faiss.deserialize_index(
                        np.frombuffer(index_bytes, dtype="uint8")
                    )
                    # 验证索引与缓存一致
                    if self.diary_index.ntotal != len(self.diary_content_list):
                        Log.logger.warning(
                            "[日记] 索引大小不匹配，将重建"
                        )
                        self._build_diary_index()
                except Exception as e:
                    Log.logger.warning(f"加载日记 FAISS 索引失败: {e}，将重建")
                    self._build_diary_index()
            else:
                self._build_diary_index()

    def _build_diary_index(self):
        """根据当前日记缓存构建 FAISS 索引（调用者持有 _lock）"""
        if not self.diary_content_list:
            dim = self._encode_texts(["占位"]).shape[1]
            self.diary_index = faiss.IndexFlatIP(dim)
            return
        try:
            vectors = self._encode_texts(self.diary_content_list)
            dim = vectors.shape[1]
            self.diary_index = faiss.IndexFlatIP(dim)
            self.diary_index.add(vectors)
            self._save_diary_index()
        except Exception as e:
            Log.logger.warning(f"构建日记 FAISS 索引失败: {e}")

    def _rebuild_diary_index(self):
        """重建日记 FAISS 索引（调用者持有 _lock）"""
        self._load_diary_index_data()
        self._build_diary_index()

    def _save_diary_index(self):
        """持久化日记 FAISS 索引到磁盘（调用者持有 _lock）"""
        try:
            index_data = faiss.serialize_index(self.diary_index)
            with open(self.diary_index_path, "wb") as f:
                f.write(index_data.tobytes())
        except Exception as e:
            Log.logger.warning(f"保存日记 FAISS 索引失败: {e}")

    def _add_to_diary_index(self, day: str, content: str):
        """向日记索引添加一条条目（调用者持有 _lock）"""
        self.diary_day_list.append(day)
        self.diary_content_list.append(content)
        try:
            vector = self._encode_texts([content])
            if self.diary_index is not None and self.diary_index.ntotal > 0:
                self.diary_index.add(vector)
            else:
                self._rebuild_diary_index()
            self._save_diary_index()
        except Exception as e:
            Log.logger.warning(f"[日记] 更新 FAISS 索引失败: {e}")

    def _remove_from_diary_index(self, day: str):
        """从日记索引移除指定日期的条目（重建方式，调用者持有 _lock）"""
        self._rebuild_diary_index()

    # ============================================================
    # 核心操作：添加记忆
    # ============================================================

    def _check_duplicate(self, content: str) -> bool:
        """
        检查内容是否与已有记忆高度重复

        Returns:
            True 表示已存在相似记忆（去重阈值为 0.75）
        """
        with self._lock:
            if not self.contents:
                return False
            try:
                vect = self._encode_texts([content])
                D, _ = self.index.search(vect, 1)
                return float(D[0][0]) >= 0.75
            except Exception:
                return False

    def add_memory(
        self,
        content: str,
        category: str,
        importance: float = 0.5,
        memory_type: str = "normal",
    ) -> int | None:
        """
        添加一条记忆

        Args:
            content: 记忆内容（由助手自主决定的内容）
            category: 分类（about_user / about_us / about_world / about_self）
            importance: 记忆重要度 0.0~1.0（由助手自行评估）
            memory_type: 记忆类型（core=永久 / normal=衰减）

        Returns:
            新记忆的 ID，重复或失败时返回 None
        """
        if category not in self.CATEGORIES:
            Log.logger.warning(f"[记忆 v2] 未知分类: {category}，使用 about_user")
            category = "about_user"

        if memory_type not in ("core", "normal"):
            Log.logger.warning(f"[记忆 v2] 未知类型: {memory_type}，使用 normal")
            memory_type = "normal"

        importance = max(0.0, min(1.0, importance))

        if self._check_duplicate(content):
            Log.logger.info(f"[记忆 v2] 去重跳过: {content[:50]}...")
            return None

        decay_rate = self.DEFAULT_DECAY_RATE if memory_type == "normal" else 0.0
        now_str = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (category, content, importance, memory_type, "
            "created_at, decay_rate) VALUES (?, ?, ?, ?, ?, ?)",
            (category, content, importance, memory_type, now_str, decay_rate),
        )
        conn.commit()
        mem_id = cursor.lastrowid
        conn.close()

        if mem_id is None:
            Log.logger.warning("[记忆 v2] 插入记忆失败：未获取到 ID")
            return None

        with self._lock:
            self.ids.append(mem_id)
            self.categories.append(category)
            self.contents.append(content)
            self.importances.append(importance)
            self.memory_types.append(memory_type)
            self.created_times.append(now_str)
            self.decay_rates.append(decay_rate)
            self.archived.append(False)
            self.access_counts.append(0)

            try:
                vector = self._encode_texts([content])
                if self.index is not None and self.index.ntotal > 0:
                    self.index.add(vector)
                else:
                    self._rebuild_index()
                self._save_index()
            except Exception as e:
                Log.logger.warning(f"[记忆 v2] 更新 FAISS 索引失败: {e}")

        # 自动发现并记录关联（不阻塞主流程）
        try:
            self._auto_link_memory(mem_id, content)
        except Exception as e:
            Log.logger.warning(f"[记忆 v2] 自动关联失败: {e}")

        Log.logger.info(
            f"[记忆 v2] 新增记忆 id={mem_id} "
            f"category={category} type={memory_type} "
            f"importance={importance:.2f}"
        )
        return mem_id

    def update_memory(
        self,
        mem_id: int,
        content: str,
        category: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
    ) -> bool:
        """
        更新/覆写一条已有记忆

        Args:
            mem_id: 要更新的记忆 ID
            content: 新的记忆内容
            category: 新分类（None 表示不改变）
            importance: 新重要度（None 表示不改变）
            memory_type: 新类型（None 表示不改变）

        Returns:
            是否更新成功
        """
        # 查找当前内存索引
        idx = None
        with self._lock:
            for i, mid in enumerate(self.ids):
                if mid == mem_id:
                    idx = i
                    break

        if idx is None:
            Log.logger.warning(f"[记忆 v2] 更新失败：id={mem_id} 不存在")
            return False

        new_category = category if category is not None else self.categories[idx]
        new_importance = importance if importance is not None else self.importances[idx]
        new_memory_type = memory_type if memory_type is not None else self.memory_types[idx]

        if new_category not in self.CATEGORIES:
            new_category = "about_user"
        if new_memory_type not in ("core", "normal"):
            new_memory_type = "normal"
        new_importance = max(0.0, min(1.0, new_importance))

        now_str = datetime.now().isoformat()
        new_decay_rate = self.DEFAULT_DECAY_RATE if new_memory_type == "normal" else 0.0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE memories SET category=?, content=?, importance=?, "
            "memory_type=?, decay_rate=?, last_accessed=? WHERE id=?",
            (new_category, content, new_importance, new_memory_type,
             new_decay_rate, now_str, mem_id),
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if updated:
            with self._lock:
                self.contents[idx] = content
                self.categories[idx] = new_category
                self.importances[idx] = new_importance
                self.memory_types[idx] = new_memory_type
                self.decay_rates[idx] = new_decay_rate
                self._rebuild_index()

            Log.logger.info(f"[记忆 v2] 更新记忆 id={mem_id}")
            return True

        return False

    # ============================================================
    # 记忆关联网络
    # ============================================================

    def _auto_link_memory(self, new_mem_id: int, new_content: str):
        """
        为新记忆自动发现并创建关联

        与新记忆语义相似度在 [LINK_SIMILARITY_MIN, LINK_SIMILARITY_MAX]
        范围内的已有记忆自动建立关联。
        """
        with self._lock:
            if not self.contents or len(self.contents) <= 1:
                return

            try:
                vect = self._encode_texts([new_content])
                # 搜索所有已有记忆
                search_k = min(20, self.index.ntotal)
                D, I = self.index.search(vect, search_k)
            except Exception:
                return

            linked_pairs = []
            for i in range(len(D[0])):
                idx = I[0][i]
                if idx >= len(self.ids):
                    continue
                existing_id = self.ids[idx]
                if existing_id == new_mem_id:
                    continue

                sim = float(D[0][i])
                if self.LINK_SIMILARITY_MIN <= sim <= self.LINK_SIMILARITY_MAX:
                    linked_pairs.append((existing_id, sim))

            if not linked_pairs:
                return

            now_str = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for existing_id, sim in linked_pairs:
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO memory_links "
                        "(memory_id_a, memory_id_b, similarity, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (min(new_mem_id, existing_id),
                         max(new_mem_id, existing_id),
                         round(sim, 4), now_str),
                    )
                except Exception:
                    continue
            conn.commit()
            conn.close()

            Log.logger.info(
                f"[记忆 v2] 自动关联: id={new_mem_id} 关联了 {len(linked_pairs)} 条记忆"
            )

    def _get_linked_memory_ids(self, mem_id: int) -> list[int]:
        """获取指定记忆的所有关联记忆 ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT memory_id_a, memory_id_b, similarity FROM memory_links "
            "WHERE memory_id_a = ? OR memory_id_b = ?",
            (mem_id, mem_id),
        )
        rows = cursor.fetchall()
        conn.close()

        linked = []
        for a, b, sim in rows:
            linked.append(b if a == mem_id else a)
        return linked

    def add_memory_link(self, mem_id_a: int, mem_id_b: int, similarity: float = 0.5) -> bool:
        """手动添加两条记忆之间的关联"""
        if mem_id_a == mem_id_b:
            return False
        now_str = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO memory_links "
                "(memory_id_a, memory_id_b, similarity, created_at) "
                "VALUES (?, ?, ?, ?)",
                (min(mem_id_a, mem_id_b), max(mem_id_a, mem_id_b),
                 round(similarity, 4), now_str),
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False
        finally:
            conn.close()

    def remove_memory_link(self, mem_id_a: int, mem_id_b: int) -> bool:
        """删除两条记忆之间的关联"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM memory_links "
            "WHERE (memory_id_a = ? AND memory_id_b = ?) "
            "OR (memory_id_a = ? AND memory_id_b = ?)",
            (mem_id_a, mem_id_b, mem_id_b, mem_id_a),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    # ============================================================
    # 核心操作：检索记忆
    # ============================================================

    def _calc_effective_importance(
        self,
        importance: float,
        created_at: str,
        memory_type: str,
        decay_rate: float | None,
        access_count: int,
    ) -> float:
        """
        计算当前有效重要度

        core 类型永不衰减；normal 类型按指数衰减，访问次数提供小幅增益。
        """
        if memory_type == "core":
            return importance

        dr = decay_rate if decay_rate is not None else 0.0
        if dr <= 0:
            return importance

        try:
            created = datetime.fromisoformat(created_at).timestamp()
        except Exception:
            return importance

        now = time.time()
        days_elapsed = max(0, (now - created) / 86400.0)

        time_factor = math.exp(-dr * days_elapsed)
        access_boost = min(access_count * 0.03, 0.2)

        return importance * time_factor + access_boost

    def _calc_diary_time_score(self, day: str) -> float:
        """
        计算日记的时间衰减系数

        越新的日记分越高，DIARY_TIME_DECAY_DAYS 天后衰减到约 0.5。
        """
        try:
            diary_ts = time.mktime(
                time.strptime(f"{day} 00:00:00", "%Y-%m-%d %H:%M:%S")
            )
            days_ago = max(0, (time.time() - diary_ts) / 86400.0)
            return math.exp(-days_ago / self.DIARY_TIME_DECAY_DAYS)
        except Exception:
            return 0.5

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_archived: bool = False,
        include_diaries: bool = True,
    ) -> list[dict[str, Any]]:
        """
        语义检索记忆（支持关联提升 + 日记语义检索）

        Args:
            query: 查询文本
            top_k: 返回的最大条数
            include_archived: 是否包含已归档记忆
            include_diaries: 是否同时检索日记

        Returns:
            按有效相关度降序排列的记忆列表：
            [{id, content, category, importance, memory_type, score, source}, ...]
        """
        results: list[dict[str, Any]] = []

        with self._lock:
            # ---- 1. 检索 memories ----
            if self.contents and self.index is not None and self.index.ntotal > 0:
                try:
                    query_vec = self._encode_texts([query])
                    search_k = min(top_k * 3, self.index.ntotal)
                    D, I = self.index.search(query_vec, search_k)
                except Exception as e:
                    Log.logger.warning(f"[记忆 v2] 检索失败: {e}")
                    return []

                # 收集每个记忆的有效重要度（使用缓存的 access_count）
                importance_cache: dict[int, float] = {}
                for idx, mem_id in enumerate(self.ids):
                    ac = self.access_counts[idx] if idx < len(self.access_counts) else 0
                    imp = self._calc_effective_importance(
                        self.importances[idx],
                        self.created_times[idx],
                        self.memory_types[idx],
                        self.decay_rates[idx],
                        ac,
                    )
                    importance_cache[mem_id] = imp

                memory_results: list[dict[str, Any]] = []
                for i in range(len(D[0])):
                    idx = I[0][i]
                    if idx >= len(self.ids):
                        continue
                    if self.archived[idx] and not include_archived:
                        continue

                    similarity = float(D[0][i])
                    mem_id = self.ids[idx]
                    effective_imp = importance_cache.get(mem_id, 0.0)
                    final_score = similarity * effective_imp

                    memory_results.append({
                        "id": mem_id,
                        "content": self.contents[idx],
                        "category": self.categories[idx],
                        "importance": self.importances[idx],
                        "effective_importance": round(effective_imp, 4),
                        "memory_type": self.memory_types[idx],
                        "score": round(final_score, 4),
                        "source": "memory",
                    })

                # ---- 关联记忆提升 ----
                # 对 top-K 结果中的每条记忆，查找其关联记忆并给予分数加成
                top_memory_ids = {r["id"] for r in memory_results[:max(3, top_k)]}
                for r in memory_results:
                    if r["id"] in top_memory_ids:
                        linked_ids = self._get_linked_memory_ids(r["id"])
                        for linked_id in linked_ids:
                            for other in memory_results:
                                if other["id"] == linked_id and other["id"] not in top_memory_ids:
                                    other["score"] = round(
                                        other["score"] * (1.0 + self.LINK_BOOST_FACTOR), 4
                                    )
                                    break

                results.extend(memory_results)

            # ---- 2. 检索日记 ----
            if include_diaries and self.diary_index is not None and self.diary_index.ntotal > 0:
                try:
                    query_vec = self._encode_texts([query])
                    search_k = min(top_k * 2, self.diary_index.ntotal)
                    D_d, I_d = self.diary_index.search(query_vec, search_k)
                except Exception as e:
                    Log.logger.warning(f"[记忆 v2] 日记检索失败: {e}")
                    D_d, I_d = None, None

                if D_d is not None and I_d is not None:
                    for i in range(len(D_d[0])):
                        idx = I_d[0][i]
                        if idx >= len(self.diary_day_list):
                            continue

                        similarity = float(D_d[0][i])
                        day = self.diary_day_list[idx]
                        content = self.diary_content_list[idx]
                        time_score = self._calc_diary_time_score(day)
                        final_score = similarity * 0.7 + time_score * 0.3

                        results.append({
                            "id": f"diary_{day}",
                            "content": content,
                            "category": "diary",
                            "importance": round(time_score, 4),
                            "effective_importance": round(time_score, 4),
                            "memory_type": "diary",
                            "score": round(final_score, 4),
                            "source": "diary",
                            "day": day,
                        })

        # ---- 3. 排序与截断 ----
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        # ---- 4. 更新访问统计 ----
        for r in top_results:
            if r.get("source") == "memory":
                self._update_access(r["id"])

        return top_results

    def _update_access(self, mem_id: int):
        """更新记忆的最后访问时间和访问计数（同时更新内存缓存）"""
        now_str = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE memories SET last_accessed = ?, "
            "access_count = access_count + 1 WHERE id = ?",
            (now_str, mem_id),
        )
        conn.commit()
        conn.close()

        # 同步更新内存缓存
        with self._lock:
            for idx, mid in enumerate(self.ids):
                if mid == mem_id:
                    self.access_counts[idx] = self.access_counts[idx] + 1
                    break

    # ============================================================
    # 获取上下文（注入到 LLM prompt 的格式化文本）
    # ============================================================

    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        根据用户消息检索相关记忆和日记，返回格式化文本供注入 system prompt

        Returns:
            格式化的文本，无匹配时返回空字符串
        """
        lines: list[str] = []

        # 统一检索 memories + diaries（语义检索）
        results = self.search(query, top_k=top_k, include_diaries=True)

        for r in results:
            if r.get("source") == "diary":
                day = r.get("day", "")
                lines.append(f"【日记 · {day}】{r['content']}")
            else:
                prefix = "【重要】" if r["memory_type"] == "core" else "【记忆】"
                cat_label = {
                    "about_user": "关于用户",
                    "about_us": "关于彼此",
                    "about_world": "关于世界",
                    "about_self": "关于自身",
                }.get(r["category"], r["category"])
                lines.append(f"{prefix}[{cat_label}] {r['content']}")

        return "\n".join(lines)

    def search_raw(
        self,
        query: str,
        top_k: int = 5,
        include_diaries: bool = True,
    ) -> list[dict[str, Any]]:
        """
        供 recall 工具调用的原始检索接口，返回完整数据结构

        Returns:
            含 source/day 等字段的完整结果列表
        """
        return self.search(
            query, top_k=top_k, include_archived=False, include_diaries=include_diaries
        )

    # ============================================================
    # 记忆衰减与归档
    # ============================================================

    def decay_and_archive(self):
        """
        对所有 normal 类型记忆执行衰减检查，低于阈值的标记为归档

        应由外部调度器定期调用（如每日一次或启动时）。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, importance, created_at, decay_rate, access_count, archived "
            "FROM memories WHERE memory_type = 'normal' AND archived = 0"
        )
        rows = cursor.fetchall()

        archived_count = 0
        for row in rows:
            mem_id, importance, created_at, decay_rate, access_count, archived = row
            if archived:
                continue

            effective = self._calc_effective_importance(
                importance, created_at, "normal", decay_rate, access_count
            )
            if effective < self.ARCHIVE_THRESHOLD:
                cursor.execute(
                    "UPDATE memories SET archived = 1 WHERE id = ?", (mem_id,)
                )
                archived_count += 1
                Log.logger.info(
                    f"[记忆 v2] 归档记忆 id={mem_id} "
                    f"effective={effective:.4f} < threshold={self.ARCHIVE_THRESHOLD}"
                )

        if archived_count > 0:
            conn.commit()

        conn.close()

        if archived_count > 0:
            with self._lock:
                self._load_data()
                self._rebuild_index()
            Log.logger.info(f"[记忆 v2] 归档完成: {archived_count} 条")

        return archived_count

    # ============================================================
    # 管理与统计
    # ============================================================

    def get_statistics(self) -> dict[str, Any]:
        """获取记忆统计信息"""
        with self._lock:
            total = len(self.contents)
            core_count = self.memory_types.count("core")
            normal_count = self.memory_types.count("normal")
            archived_count = sum(self.archived)
            active_count = total - archived_count

            category_counts: dict[str, int] = {}
            for cat in self.categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # 统计关联数量
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memory_links")
            link_count = int(cursor.fetchone()[0])
            conn.close()

            return {
                "total": total,
                "active": active_count,
                "archived": archived_count,
                "core": core_count,
                "normal": normal_count,
                "categories": category_counts,
                "links": link_count,
            }

    def get_all_memories(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """获取全部记忆（用于管理界面）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if include_archived:
            cursor.execute(
                "SELECT id, category, content, importance, memory_type, "
                "created_at, last_accessed, access_count, archived "
                "FROM memories ORDER BY id DESC"
            )
        else:
            cursor.execute(
                "SELECT id, category, content, importance, memory_type, "
                "created_at, last_accessed, access_count, archived "
                "FROM memories WHERE archived = 0 ORDER BY id DESC"
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "category": r[1],
                "content": r[2],
                "importance": r[3],
                "memory_type": r[4],
                "created_at": r[5],
                "last_accessed": r[6],
                "access_count": r[7],
                "archived": bool(r[8]),
            }
            for r in rows
        ]

    def delete_memory(self, mem_id: int) -> bool:
        """删除指定记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        deleted = cursor.rowcount > 0

        # 同时清理关联
        if deleted:
            cursor.execute(
                "DELETE FROM memory_links WHERE memory_id_a = ? OR memory_id_b = ?",
                (mem_id, mem_id),
            )

        conn.commit()
        conn.close()

        if deleted:
            with self._lock:
                self._load_data()
                self._rebuild_index()
            Log.logger.info(f"[记忆 v2] 删除记忆 id={mem_id}")

        return deleted

    def get_links_for_memory(self, mem_id: int) -> list[dict[str, Any]]:
        """获取指定记忆的所有关联信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ml.memory_id_a, ml.memory_id_b, ml.similarity, ml.created_at, "
            "m.category, m.content, m.importance "
            "FROM memory_links ml "
            "JOIN memories m ON m.id = CASE WHEN ml.memory_id_a = ? "
            "THEN ml.memory_id_b ELSE ml.memory_id_a END "
            "WHERE ml.memory_id_a = ? OR ml.memory_id_b = ?",
            (mem_id, mem_id, mem_id),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "linked_id": r[1] if r[0] == mem_id else r[0],
                "similarity": r[2],
                "created_at": r[3],
                "category": r[4],
                "content": r[5],
                "importance": r[6],
            }
            for r in rows
        ]

    # ============================================================
    # 日记系统：对话原始记录
    # ============================================================

    def add_chat_turn(self, role: str, content: str, timestamp_sec: int):
        """
        存储对话原始轮次，供日记生成使用

        Args:
            role: user / assistant
            content: 消息文本
            timestamp_sec: Unix 时间戳（秒）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_turns (timestamp_sec, role, content) VALUES (?, ?, ?)",
            (timestamp_sec, role, content),
        )
        conn.commit()
        conn.close()

    def _get_turns_for_time_range(
        self, start_ts: int, end_ts: int
    ) -> list[dict[str, Any]]:
        """获取指定时间范围内的对话轮次（按时间升序）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp_sec, role, content FROM chat_turns "
            "WHERE timestamp_sec >= ? AND timestamp_sec < ? "
            "ORDER BY timestamp_sec ASC, id ASC",
            (start_ts, end_ts),
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"timestamp_sec": r[0], "role": r[1], "content": r[2]} for r in rows]

    def _get_last_diary_day(self) -> str | None:
        """获取最后生成日记的日期，None 表示从未生成过"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(day) FROM diary_days")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None

    def _format_turns_for_prompt(self, turns: list[dict[str, Any]]) -> str:
        """将对话轮次列表格式化为 LLM 可读的文本"""
        lines = []
        for t in turns:
            speaker = self.user if t["role"] == "user" else self.char
            t_str = time.strftime("%H:%M:%S", time.localtime(t["timestamp_sec"]))
            lines.append(f"[{t_str}] {speaker}: {t['content']}")
        return "\n".join(lines)

    def _collect_important_context(self, top_k: int = 10) -> str:
        """
        收集已有的重要记忆，供日记生成时参考

        包括 core 类型记忆和高 importance 的记忆（如生日、节日等）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, content, importance, memory_type FROM memories "
            "WHERE archived = 0 AND (memory_type = 'core' OR importance >= 0.6) "
            "ORDER BY importance DESC LIMIT ?",
            (top_k,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return ""

        lines = ["一些你可能记得的相关信息（写日记时可以参考）："]
        for category, content, importance, mem_type in rows:
            cat_label = {
                "about_user": "关于用户",
                "about_us": "关于彼此",
                "about_world": "关于世界",
                "about_self": "关于自身",
            }.get(category, category)
            prefix = "【重要】" if mem_type == "core" else "【信息】"
            lines.append(f"  {prefix}[{cat_label}] {content}")

        return "\n".join(lines)

    def _ensure_diary_llm(self):
        """确保日记 LLM 客户端已初始化（lazy）"""
        if self._diary_llm is None:
            self._diary_llm = LLMClient(model_key="LLM")

    async def generate_diary_for_day(
        self, day_str: str, turns: list[dict[str, Any]]
    ) -> bool:
        """
        为指定日期生成日记并存入 diary_days 表 + 日记 FAISS 索引

        Args:
            day_str: 日期字符串 YYYY-MM-DD
            turns: 该日的对话轮次列表

        Returns:
            是否成功生成了日记
        """
        if len(turns) < self.MIN_TURNS_FOR_DIARY:
            Log.logger.info(
                f"[日记] 跳过 {day_str}：仅 {len(turns)} 轮对话，"
                f"不足阈值 {self.MIN_TURNS_FOR_DIARY}"
            )
            return False

        self._ensure_diary_llm()

        conversation_text = self._format_turns_for_prompt(turns)
        important_context = self._collect_important_context()

        # 计算认识天数
        days_known = 0
        if self.firstMeetTime > 0:
            day_start = int(
                time.mktime(time.strptime(f"{day_str} 00:00:00", "%Y-%m-%d %H:%M:%S"))
            )
            days_known = max(0, int((day_start - self.firstMeetTime) // 86400))

        # 第一步：提取摘要
        summary_prompt = (
            f"请总结以下对话中的关键事件和情感要点，用于生成日记：\n"
            f"{conversation_text}\n"
            f"请提取：\n"
            f"1. 主要事件（发生了什么）\n"
            f"2. 情感要点（{self.char}的感受）\n"
            f"3. 特别的细节（值得记录的小事）\n"
            f"以简洁的列表形式输出。"
            f"\n如果今天没有什么值得记录的事情，请回复：无"
        )

        try:
            facts = (
                await self._diary_llm.request(
                    [{"role": "system", "content": summary_prompt}]
                )
                or ""
            )
        except Exception:
            Log.logger.error(f"[日记] 摘要生成失败: {day_str}", exc_info=True)
            return False

        if not facts or facts.strip() == "无":
            Log.logger.info(f"[日记] 跳过 {day_str}：无值得记录的内容")
            return False

        # 第二步：生成角色口吻日记正文
        context_block = ""
        if important_context:
            context_block = f"\n{important_context}\n"

        char_profile_parts = []
        if self.agent_config.description:
            char_profile_parts.append(f"角色设定：{self.agent_config.description}")
        if self.agent_config.personality:
            char_profile_parts.append(f"角色性格：{self.agent_config.personality}")
        char_profile = "\n".join(char_profile_parts)

        diary_system_prompt = (
            f'现在请你以你扮演的角色"{self.char}"的视角，'
            f'以第一人称的口吻，用"{self.char}"的语气和思维，'
            f'把今天的对话内容写成一篇"日记"记录。\n'
            f"要求：\n"
            f"1. 用自然、贴近你性格的语言，不要像AI总结报告\n"
            f"2. 记录对话中让你印象深刻的事情、感受和情绪\n"
            f"3. 允许适度加入内心独白\n"
            f"4. 不要逐字复述对话，要像真实日记那样有个人感受\n"
            f'5. 描述"{self.user}"时，不要编造他没说过的事情\n'
            f"6. 直接输出日记正文，不需要以日记，日期等内容开头，禁止输出其他任何无关内容\n"
            f"7. 字数在100字到500字之间"
        )
        if char_profile:
            diary_system_prompt = f"{char_profile}\n\n{diary_system_prompt}"

        day_info = f"今天是{day_str}"
        if days_known > 0:
            day_info += f"，是和{self.user}认识的第{days_known}天"

        diary_user_prompt = (
            f"{day_info}。\n"
            f"以下是今日互动摘要：\n{facts}\n"
            f"{context_block}"
            f"请根据以上信息写一篇日记，注意保持角色口吻和情感表达。"
        )

        try:
            diary_text = (
                await self._diary_llm.request(
                    [
                        {"role": "system", "content": diary_system_prompt},
                        {"role": "user", "content": diary_user_prompt},
                    ]
                )
                or ""
            )
        except Exception:
            Log.logger.error(f"[日记] 日记正文生成失败: {day_str}", exc_info=True)
            return False

        diary_text = diary_text.strip()
        if not diary_text:
            Log.logger.info(f"[日记] 跳过 {day_str}：日记正文为空")
            return False

        # 第三步：存入 diary_days 独立表
        now_iso = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO diary_days (day, event_summary, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (day_str, facts, diary_text, now_iso),
        )
        conn.commit()
        conn.close()

        # 第四步：加入日记 FAISS 索引（支持语义检索）
        with self._lock:
            self._add_to_diary_index(day_str, diary_text)

        Log.logger.info(f"[日记] 已生成 {day_str}")
        return True

    async def check_and_generate_diary(self, now_ts: int):
        """
        检查是否需要生成日记——跨天时触发未归档日期的日记生成

        应在每次 add_chat_turn 后调用
        """
        t = time.localtime(now_ts)
        today_start = int(
            time.mktime(
                time.strptime(
                    f"{t.tm_year}-{t.tm_mon}-{t.tm_mday} 00:00:00",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
        )
        last_diary_day = self._get_last_diary_day()

        # 解码最后日记日期的时间戳起点
        last_diary_ts = 0
        if last_diary_day:
            try:
                last_diary_ts = int(
                    time.mktime(
                        time.strptime(f"{last_diary_day} 00:00:00", "%Y-%m-%d %H:%M:%S")
                    )
                )
            except Exception:
                last_diary_ts = 0

        # 收集从上次日记之后到今天之前的所有未归档日期
        if today_start <= last_diary_ts:
            return  # 没有需要归档的日期

        # 按天分组
        turns = self._get_turns_for_time_range(last_diary_ts, today_start)
        if not turns:
            return

        day_groups: dict[str, list[dict[str, Any]]] = {}
        for t_turn in turns:
            d = time.strftime("%Y-%m-%d", time.localtime(t_turn["timestamp_sec"]))
            day_groups.setdefault(d, []).append(t_turn)

        for day_str, day_turns in day_groups.items():
            await self.generate_diary_for_day(day_str, day_turns)

    def get_diary_records(
        self,
        limit: int = 20,
        offset: int = 0,
        start_day: str | None = None,
        end_day: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        分页获取日记记录（从 diary_days 表直接查询）

        Args:
            limit: 每页条数
            offset: 偏移量
            start_day: 起始日期 YYYY-MM-DD
            end_day: 结束日期 YYYY-MM-DD

        Returns:
            (records, total)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_parts: list[str] = []
        params: list[str] = []

        if start_day:
            where_parts.append("day >= ?")
            params.append(start_day)
        if end_day:
            where_parts.append("day <= ?")
            params.append(end_day)

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)

        # 总数
        cursor.execute(f"SELECT COUNT(*) FROM diary_days {where_sql}", tuple(params))
        total = int(cursor.fetchone()[0])

        # 分页查询
        cursor.execute(
            f"SELECT day, event_summary, content, created_at "
            f"FROM diary_days {where_sql} ORDER BY day DESC "
            f"LIMIT ? OFFSET ?",
            tuple(params + [str(limit), str(offset)]),
        )
        rows = cursor.fetchall()
        conn.close()

        records: list[dict[str, Any]] = []
        for r in rows:
            day, event_summary, content, created_at = r
            ts = 0
            try:
                dt = datetime.fromisoformat(created_at)
                ts = int(dt.timestamp())
            except Exception:
                pass

            records.append(
                {
                    "day": day,
                    "summary": content,
                    "facts": event_summary,
                    "dayLastTimestampSec": ts,
                    "dayLastTimestamp": created_at,
                }
            )

        return records, total
