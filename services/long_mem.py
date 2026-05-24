"""
长期记忆模块（SQLite版）

设计目标：
1. 聊天历史与长期日记统一存储在同一个 SQLite 文件中，减少系统复杂度。
2. 日记按“天”归档，仅在跨天时生成，避免每轮都调用 LLM。
3. 检索时同时返回：历史日记 + 今日未归档对话，保证信息不遗漏。
4. 兼容现有调用方式：保留 get_memories(...) 与 add_memory(...)
"""

import os
import sqlite3
import time

import jionlp as jio
import numpy as np

from models.types.assistant_info import AssistantInfo
from my_utils import embedding
from my_utils import log as Log
from my_utils.llm_request import llm_request


class Memory:
    """
    日记化长期记忆管理器

    核心思路：
    - chat_turns 表保存完整历史消息，作为可追溯的事实主源。
    - diary_days 表保存每日总结（summary + facts + vector），作为长期压缩层。
    - 查询时优先检索 diary_days，再补充尚未归档的 chat_turns。
    """

    def __init__(self, agent_config: AssistantInfo, firstMeetTime: int = 0):
        self.agent_id = agent_config.name
        self.char = agent_config.name
        self.user = agent_config.user
        self.firstMeetTime = firstMeetTime
        self.thresholds = agent_config.settings.longMemoryThreshold
        self.enable_search_enhance = agent_config.settings.enableLongMemorySearchEnhance
        # 每日对话记录阈值：仅当日记录数 > 5 时才生成日记。
        self.min_daily_records_for_diary = 6

        self.data_dir = f"./data/agents/{self.agent_id}/memory"
        self.db_path = os.path.join(self.data_dir, "memory.db")
        os.makedirs(self.data_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """创建数据库连接。每次操作使用独立连接，减少并发线程共享问题。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        """
        初始化最小表结构：仅两张核心表。
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_sec INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                vec_blob BLOB
            )
            """)



        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diary_days (
                day TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                facts TEXT NOT NULL,
                vec_blob BLOB,
                day_last_timestamp_sec INTEGER NOT NULL
            )
            """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_turns_timestamp_sec ON chat_turns(timestamp_sec)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_diary_days_day ON diary_days(day)"
        )

        conn.commit()
        conn.close()

    def _day_str(self, ts: int) -> str:
        """将时间戳转换为日期字符串，格式为 YYYY-MM-DD。"""
        return time.strftime("%Y-%m-%d", time.localtime(ts))

    def _vector_to_blob(self, vector: np.ndarray) -> bytes:
        """将向量转换为二进制 blob 以存储在 SQLite 中。"""
        v = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tobytes()

    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        """将 SQLite 中的二进制 blob 转换回向量。"""
        return np.frombuffer(blob, dtype=np.float32)

    def append_turn(self, turn_data: list[dict], turn_ts: int):
        """
        写入一轮对话到 chat_turns。

        Parameters:
            turn_data: 对话数据，格式为 [{"role": "user"/"assistant", "content": "..."}]
            turn_ts: 该轮对话的时间戳（秒级）
        """
        rows = []
        text_list = []
        for item in turn_data:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                rows.append((turn_ts, role, content))
                text_list.append(f"{role}: {content}")

        if not rows:
            return

        # 批量计算向量并转换为 blob 存储
        vectors = embedding.t2vect(text_list)
        vec_blobs = [self._vector_to_blob(v) for v in vectors]

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO chat_turns (timestamp_sec, role, content, vec_blob) VALUES (?, ?, ?, ?)",
            [(ts, role, content, vec_blob) for (ts, role, content), vec_blob in zip(rows, vec_blobs)],
        )
        conn.commit()
        conn.close()

    def get_recent_chat_turns(
        self, limit: int, only_assistant: bool = False
    ) -> list[dict]:
        """
        获取最近历史消息，用于上下文恢复与 /chat/history 接口。
        """
        if limit <= 0:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        if only_assistant:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_turns
                WHERE role = 'assistant'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_turns
                WHERE role IN ('user', 'assistant')
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()
        # 查询使用倒序取最近 N 条，这里反转回正常时间顺序。
        rows.reverse()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def _get_last_diary_ts(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(day_last_timestamp_sec) FROM diary_days")
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row and row[0] is not None else 0

    async def finalize_previous_days(self, now_ts: int):
        """
        跨天归档：将“尚未归档且不属于今天”的对话按天生成日记并写入 diary_days。

        说明：
        - 仅跨天时触发，不在每轮对话中生成日记。
        - 同一天只会写入一次（day 为主键，使用 INSERT OR REPLACE）。
        """
        t = time.localtime(now_ts)

        today_start = int(
            time.mktime(
                time.strptime(
                    f"{t.tm_year}-{t.tm_mon}-{t.tm_mday} 00:00:00", "%Y-%m-%d %H:%M:%S"
                )
            )
        )
        last_diary_ts = self._get_last_diary_ts()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_sec, role, content
            FROM chat_turns
            WHERE timestamp_sec > ? AND timestamp_sec < ?
            ORDER BY timestamp_sec ASC, id ASC
            """,
            (last_diary_ts, today_start),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return

        day_groups: dict[str, list[tuple[int, str, str]]] = {}
        for ts, role, content in rows:
            d = self._day_str(int(ts))
            day_groups.setdefault(d, []).append((int(ts), role, content))

        for day, day_rows in day_groups.items():
            if len(day_rows) < self.min_daily_records_for_diary:
                Log.logger.info(
                    f"[长期记忆] 跳过日记归档: {day}, 当日记录数={len(day_rows)}，阈值>{self.min_daily_records_for_diary - 1}"
                )
                continue

            # 第一阶段：先提取“对话摘要/事实要点”，作为日记生成输入。
            facts = await self._build_diary_summary_text(day_rows)
            # 第二阶段：根据摘要生成日记正文。
            summary = await self._build_diary_text(facts)
            if not summary:
                Log.logger.warning(f"[长期记忆] 日记生成失败，跳过归档: {day}")
                continue

            vector = embedding.t2vect([f"{summary}\n{facts}"])[0]
            vec_blob = self._vector_to_blob(vector)
            day_last_ts = day_rows[-1][0]

            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO diary_days (day, summary, facts, vec_blob, day_last_timestamp_sec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (day, summary, facts, vec_blob, day_last_ts),
            )
            conn.commit()
            conn.close()

            Log.logger.info(f"[长期记忆] 已归档日记: {day}")

    async def _build_diary_summary_text(
        self, day_rows: list[tuple[int, str, str]]
    ) -> str:
        """
        基于当天对话生成结构化摘要。

        输出内容作为 diary_days.facts 存储，并作为下一阶段日记生成输入。
        """

        lines = []
        for ts, role, content in day_rows:
            speaker = self.user if role == "user" else self.char
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"[{t_str}] {speaker}: {content}")
        conversation_text = "\n".join(lines)

        prompt = f"""
        请总结以下对话中的关键事件和情感要点，用于生成日记：
        {conversation_text}
        请提取：
        1. 主要事件（发生了什么）
        2. 情感要点（"{self.char}"的感受）
        3. 特别的细节（值得记录的小事）
        4. 生成的短句要适合用于向量检索
        以简洁的列表形式输出。
        """
        try:
            summary = (
                await llm_request(
                    [{"role": "system", "content": prompt}],
                )
                or ""
            )
        except Exception:
            Log.logger.error("生成日记摘要失败", exc_info=True)
            summary = ""

        if not summary:
            summary = "- 日常交流\n- 有对话发生"

        return summary.strip()

    async def _build_diary_text(self, summary: str) -> str:
        """
        基于“当日摘要”生成角色口吻日记正文。

        Parameters:
            summary: 由 _build_diary_summary_text 生成的当日互动摘要

        Returns:
            diary_text: 角色口吻日记（可读）
        """

        diary_system_prompt = f"""
        现在请你以你扮演的角色"{self.char}"的视角，以第一人称的口吻，用你所扮演的角色"{self.char}"的的语气和思维，把刚才你和"{self.user}"之间的全部对话内容，写成一篇"日记"记录。
        要求：
        1.用自然、贴近你性格的语言，不要像AI总结报告。
        2.记录对话中让你印象深刻的事情、感受和情绪。
        3.允许适度加入内心独白。
        4.不要逐字复述对话，要真实日记那样有个人感受和小情绪。
        5.日记在描述"{self.user}"时，不要增加"{self.user}"在对话中没说过的事情。
        7.日记内容不要太长，字数100字到500字之间)
        """
        diary_user_prompt = f"今天是你和{self.user}认识的的第{self.firstMeetTime//(60*60*24)}天，以下是今日互动摘要:{summary}。\n请写日记："

        diary_text = ""
        try:
            diary_text = (
                await llm_request(
                    [
                        {"role": "system", "content": diary_system_prompt},
                        {"role": "user", "content": diary_user_prompt},
                    ]
                )
                or ""
            )
        except Exception:
            Log.logger.error("记录日记失败", exc_info=True)

        if not diary_text:
            return ""
        return diary_text.strip()

    def _extract_time_range(self, msg: str) -> tuple[int, int] | None:
        """
        提取查询中的时间范围。
        解析失败返回 None，让检索走兜底语义路径。
        """
        # 从文本中提取所有时间实体
        res = jio.ner.extract_time(f"{msg}", with_parsing=False)

        spans: list[tuple[int, int]] = []
        if res:
            for t in res:
                try:
                    # 使用当前时间作为基准时间，避免把首个实体文本误当作 time_base。
                    res_t = jio.parse_time(t["text"], time_base=time.time())
                    st1 = int(
                        time.mktime(
                            time.strptime(res_t["time"][0], "%Y-%m-%d %H:%M:%S")
                        )
                    )
                    st2 = int(
                        time.mktime(
                            time.strptime(res_t["time"][1], "%Y-%m-%d %H:%M:%S")
                        )
                    )
                    spans.append((st1, st2))
                except Exception as e:
                    Log.logger.error(f"解析时间范围失败: {e}")
                    continue

        if not spans:
            return None

        low = min(s[0] for s in spans)
        high = max(s[1] for s in spans)
        return low, high

    def _search_diary_rows(
        self, msg: str, time_range: tuple[int, int] | None, top_k: int = 3
    ) -> list[tuple[str, str, str, float]]:
        """
        检索日记记录，返回 [(day, summary, facts, score)]。
        day: 日期字符串，summary: 日记文本，facts: 事实要点，score: 相关度分数。

        说明：
        - 时间命中时：先按日期范围过滤，再做向量评分。
        - 时间未命中时：对全部日记做向量评分。
        - 若关闭语义增强，则按照日期倒序返回。
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if time_range:
            low_day = self._day_str(time_range[0])
            high_day = self._day_str(time_range[1])
            cursor.execute(
                """
                SELECT day, summary, facts, vec_blob
                FROM diary_days
                WHERE day >= ? AND day <= ?
                ORDER BY day DESC
                """,
                (low_day, high_day),
            )
        else:
            cursor.execute("""
                SELECT day, summary, facts, vec_blob
                FROM diary_days
                ORDER BY day DESC
                """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []
        # 如果没有语义增强，直接返回最近的 top_k 条日记，相关度分数默认为 1.0。
        if not self.enable_search_enhance:
            return [(r[0], r[1], r[2], 1.0) for r in rows[:top_k]]
        # 使用向量搜索对日记进行相关度评分，返回超过阈值的 top_k 条。
        q_v = embedding.t2vect([msg])[0].astype(np.float32)
        q_norm = np.linalg.norm(q_v)
        if q_norm > 0:
            q_v = q_v / q_norm
        # 遍历日记记录，计算与查询的余弦相似度，筛选出超过阈值的记录。
        scored: list[tuple[str, str, str, float]] = []
        for day, summary, facts, vec_blob in rows:
            if not vec_blob:
                continue
            d_v = self._blob_to_vector(vec_blob)
            score = float(np.dot(d_v, q_v))
            if score >= self.thresholds:
                scored.append((day, summary, facts, score))
        # 如果没有任何记录超过阈值，保底返回最近的 1 条日记，相关度分数为 0.0，避免完全失忆。
        if not scored:
            # 没有超过阈值时保底返回最近的1条，避免完全失忆。
            day, summary, facts, _ = rows[0]
            return [(day, summary, facts, 0.0)]

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]

    def _search_pending_turns(
        self, msg: str, time_range: tuple[int, int] | None, top_k: int = 6
    ) -> list[tuple[int, str, str, float]]:
        """
        检索尚未归档的历史消息。

        判定规则：ts > max(diary_days.last_ts)
        """
        last_diary_ts = self._get_last_diary_ts()

        conn = self._get_connection()
        cursor = conn.cursor()
        if time_range:
            low, high = time_range
            low = max(low, last_diary_ts + 1)
            cursor.execute(
                """
                SELECT timestamp_sec, role, content, vec_blob
                FROM chat_turns
                WHERE timestamp_sec >= ? AND timestamp_sec <= ?
                ORDER BY id ASC
                """,
                (low, high),
            )
        else:
            cursor.execute(
                """
                SELECT timestamp_sec, role, content, vec_blob
                FROM chat_turns
                WHERE timestamp_sec > ?
                ORDER BY id ASC
                """,
                (last_diary_ts,),
            )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        if not self.enable_search_enhance:
            return [(int(r[0]), r[1], r[2], 1.0) for r in rows[-top_k:]]

        q_v = embedding.t2vect([msg])[0].astype(np.float32)
        q_norm = np.linalg.norm(q_v)
        if q_norm > 0:
            q_v = q_v / q_norm

        scored_rows: list[tuple[int, str, str, float]] = []
        for ts, role, content, vec_blob in rows:
            if not vec_blob:
                continue
            d_v = self._blob_to_vector(vec_blob)
            score = float(np.dot(d_v, q_v))
            if score >= self.thresholds:
                scored_rows.append((int(ts), role, content, score))

        if not scored_rows:
            # 保底返回最近几条当日未归档消息。
            tail = rows[-top_k:]
            return [(int(r[0]), r[1], r[2], 0.0) for r in tail]

        scored_rows.sort(key=lambda x: x[3], reverse=True)
        return scored_rows[:top_k]

    def get_diary_records(
        self,
        limit: int = 20,
        offset: int = 0,
        start_day: str | None = None,
        end_day: str | None = None,
    ) -> tuple[list[dict], int]:
        """
        分页获取日记记录

        Parameters:
            limit: 单次返回条数
            offset: 偏移量
            start_day: 起始日期（YYYY-MM-DD），可选
            end_day: 结束日期（YYYY-MM-DD），可选

        Returns:
            records: 日记记录列表
            total: 满足条件的总记录数
        """
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        where_clauses = []
        params: list = []

        if start_day:
            where_clauses.append("day >= ?")
            params.append(start_day)
        if end_day:
            where_clauses.append("day <= ?")
            params.append(end_day)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM diary_days {where_sql}", tuple(params))
        count_row = cursor.fetchone()
        total = int(count_row[0]) if count_row and count_row[0] is not None else 0

        query_params = params + [limit, offset]
        cursor.execute(
            f"""
            SELECT day, summary, facts, day_last_timestamp_sec
            FROM diary_days
            {where_sql}
            ORDER BY day DESC
            LIMIT ? OFFSET ?
            """,
            tuple(query_params),
        )
        rows = cursor.fetchall()
        conn.close()

        records = []
        for day, summary, facts, last_ts in rows:
            ts = int(last_ts)
            records.append(
                {
                    "day": day,
                    "summary": summary,
                    "facts": facts,
                    "dayLastTimestampSec": ts,
                    "dayLastTimestamp": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(ts)
                    ),
                }
            )

        return records, total

    def get_memories(self, msg: str) -> str:
        """
        对外统一检索接口，返回可直接注入 prompt 的文本。

        输出结构：
        - 历史日记摘要（长期归档）
        - 未归档对话片段（当日或尚未跨天归档）
        """
        start_time = time.time()
        # 根据语义提取时间范围，若解析失败则返回 None，由后续检索走兜底语义路径。
        time_range = self._extract_time_range(msg)
        print(f"[提取时间范围耗时]：{time.time() - start_time}")
        start_time = time.time()
        # 先检索日记表，获取相关日记摘要；再检索 chat_turns，获取相关的未归档对话片段。
        diary_hits = self._search_diary_rows(msg, time_range)
        print(f"[检索日记耗时]：{time.time() - start_time}")
        start_time = time.time()
        # 检索尚未归档的对话消息，也就是今天内的消息
        pending_hits = self._search_pending_turns(msg, time_range)
        print(f"[检索未归档对话耗时]：{time.time() - start_time}")

        blocks = []

        if diary_hits:
            lines = ["[历史日记]"]
            for day, summary, facts, score in diary_hits:
                lines.append(f"日期: {day} (相关度: {score:.3f})")
                lines.append(summary)
                lines.append("事实要点:")
                lines.append(facts)
                lines.append("")
            blocks.append("\n".join(lines).strip())

        if pending_hits:
            lines = ["[未归档对话]"]
            for ts, role, content, score in pending_hits:
                t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                speaker = self.user if role == "user" else self.char
                lines.append(f"{t_str} {speaker}: {content} (相关度: {score:.3f})")
            blocks.append("\n".join(lines).strip())

        return "\n\n".join(blocks)

    async def add_memory(self, data: list, t_n: int) -> None:
        """
        兼容旧调用方式：
        - 先写入当前轮对话
        - 再尝试执行跨天归档
        """
        self.append_turn(data, t_n)
        await self.finalize_previous_days(t_n)
