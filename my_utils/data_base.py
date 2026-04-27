import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

from models.types.assistant_info import AssistantInfo
from my_utils import config_manager as CConfig
from my_utils import embedding
from my_utils import log as Log
from my_utils.llm_request import llm_request, parse_llm_json_response


RAW_SUFFIXES = {".txt", ".md"}
# 知识库抽取提示词模板
INVALID_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _now_date() -> str:
    """
    获取当前日期字符串。
    Returns:
        str: YYYY-MM-DD
    """
    return time.strftime("%Y-%m-%d", time.localtime())


def _now_datetime() -> str:
    """
    获取当前时间字符串。
    Returns:
        str: YYYY-MM-DD HH:MM:SS
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _sum_md5(file_path: Path) -> str:
    """
    计算文件 MD5 值。
    Parameters:
        file_path (Path): 文件路径
    Returns:
        str: 文件 MD5
    """
    md5_obj = hashlib.md5()
    with file_path.open("rb") as file:
        while True:
            data = file.read(4096)
            if not data:
                break
            md5_obj.update(data)
    return md5_obj.hexdigest()


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    统一向量格式并进行 L2 归一化。
    Parameters:
        vectors (np.ndarray): 输入向量
    Returns:
        np.ndarray: 归一化向量
    """
    arr = np.ascontiguousarray(vectors.astype(np.float32))
    faiss.normalize_L2(arr)
    return arr


def _clean_text(text: str) -> str:
    """
    清洗输入文本。
    Parameters:
        text (str): 原始文本
    Returns:
        str: 清洗后文本
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _sanitize_page_name(name: str) -> str:
    """
    将标题转换为安全文件名。
    Parameters:
        name (str): 页面标题
    Returns:
        str: 安全文件名（无扩展名）
    """
    cleaned = INVALID_FILE_CHARS.sub("_", name).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "未命名"


def _split_semantic_blocks(
    text: str, max_chars: int = 900, min_chars: int = 120
) -> list[str]:
    """
    以段落优先策略进行语义分块。
    Parameters:
        text (str): 输入文本
        max_chars (int): 单块最大字符数
        min_chars (int): 单块最小字符数
    Returns:
        list[str]: 分块结果
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    blocks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            blocks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if buffer and len(buffer) >= min_chars:
                flush_buffer()
            sentences = re.split(r"(?<=[。！？!?\.])", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        fragment = sentence[i : i + max_chars].strip()
                        if fragment:
                            blocks.append(fragment)
                    continue
                if not buffer:
                    buffer = sentence
                    continue
                if len(buffer) + len(sentence) + 1 <= max_chars:
                    buffer = f"{buffer} {sentence}".strip()
                else:
                    flush_buffer()
                    buffer = sentence
            continue

        if not buffer:
            buffer = para
            continue

        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            flush_buffer()
            buffer = para

    flush_buffer()

    merged: list[str] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        if (
            len(merged[-1]) < min_chars
            and len(merged[-1]) + len(block) + 2 <= max_chars
        ):
            merged[-1] = f"{merged[-1]}\n\n{block}".strip()
        else:
            merged.append(block)

    return merged


def _sync_run_coroutine(coro: Any, timeout: int = 120) -> Any:
    """
    在同步上下文安全执行协程。
    Parameters:
        coro (Coroutine): 协程对象
        timeout (int): 超时时间（秒）
    Returns:
        Any: 协程返回值
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container: dict[str, Any] = {}
    error_container: dict[str, Exception] = {}

    def runner() -> None:
        try:
            result_container["result"] = asyncio.run(coro)
        except Exception as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        Log.logger.warning("LLM 抽取超时，回退到规则抽取。")
        return None

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


class DataBase:
    """
    新知识库引擎（LLM + Wiki + FAISS）。

    对外兼容点：
    - 保留 DataBase.search(text_list) -> str 语义，供 Agent 继续调用。
    """

    def __init__(self, agent_config: AssistantInfo):
        """
        初始化知识库引擎。
        Parameters:
            agent_config (AssistantInfo): 助手配置
        """
        settings = agent_config.settings
        # 检索阈值和返回数量
        self.thresholds = float(settings.loreBooksThreshold)
        self.top_k = int(settings.loreBooksDepth)
        # 定义目录结构和数据库路径
        kb_config = CConfig.config.get("KnowledgeBase", {})
        self.base_dir = Path("./data/worldbook")
        self.raw_dir = self.base_dir / "raw"
        self.embeddings_dir = self.base_dir / "embeddings"
        self.wiki_dir = self.base_dir / "wiki"
        self.wiki_entity_dir = self.wiki_dir / "entity"
        self.wiki_concept_dir = self.wiki_dir / "concept"
        self.wiki_source_dir = self.wiki_dir / "source"

        self.index_path = self.embeddings_dir / "vectors.faiss"
        self.sqlite_path = self.embeddings_dir / "metadata.db"
        # 健康检查间隔和是否启用 LLM 抽取
        self.health_check_interval_sec = int(
            kb_config.get("health_check_interval_sec", 3600)
        )
        # 启用 LLM 抽取会增加新文件的处理时间，但能获得更丰富的结构化信息，提升检索质量。可根据实际情况调整。
        self.enable_llm_extract = bool(kb_config.get("enable_llm_extract", True))
        # FAISS 索引
        self.index: faiss.IndexIDMap2 | None = None
        # 向量维度
        self.dimension: int | None = None
        # 数据总量（块数量）
        self.data_count = 0
        # 上次健康检查时间戳和上次重建摘要
        self._last_health_check_ts = 0
        # 上次重建摘要用于快速判断重建结果是否有实质性变化，避免频繁无效重建
        self._last_rebuild_summary: dict[str, Any] = {}
        # 初始化目录和数据库连接
        self._ensure_directories()
        # search/rebuild 会在 run_in_executor 线程中执行，需允许跨线程访问同一连接。
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

        self.rebuild(startup=True)

    def _ensure_directories(self) -> None:
        """
        确保目录结构存在。
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_entity_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_concept_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_source_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        """
        初始化元数据库。
        """
        # raw_files 记录原始文件的相对路径、MD5、对应页面和更新时间
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_files (
                raw_path TEXT PRIMARY KEY,
                file_md5 TEXT NOT NULL,
                source_page_id TEXT NOT NULL,
                updated TEXT NOT NULL
            )
            """
        )
        # wiki_pages 记录页面的基本信息和内容
        # page_id 格式为 "{page_type}/{sanitized_title}"
        # page_type 是 entity/concept/source 之一
        # file_path 是实际 Markdown 文件的绝对路径
        # content 是页面内容的冗余存储，主要用于快速访问和更新，避免频繁读写文件
        # tags_json 和 sources_json 存储标签和来源的 JSON 数组文本
        # raw_path 记录对应的 raw 文件路径，便于追溯和管理
        # created 和 updated 记录页面的创建和更新时间，格式为 "YYYY-MM-DD HH:MM:SS"
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_pages (
                page_id TEXT PRIMARY KEY,
                page_type TEXT NOT NULL,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                raw_path TEXT
            )
            """
        )
        # page_refs 记录页面之间的引用关系，用于构建关联网络和辅助检索
        # ref_type 可以是 "entity"、"concept" 或 "source"，表示引用的页面类型
        # updated 记录引用关系的更新时间，格式为 "YYYY-MM-DD HH:MM:SS"
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS page_refs (
                from_page_id TEXT NOT NULL,
                to_page_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                updated TEXT NOT NULL,
                PRIMARY KEY(from_page_id, to_page_id, ref_type)
            )
            """
        )
        # chunks 记录页面分块后的文本内容和对应的 FAISS ID，用于向量检索
        # chunk_id 是自增主键，page_id 是所属页面 ID，chunk_text 是分块文本内容，faiss_id 是对应的向量 ID，updated 记录更新时间
        # 当页面更新时，相关的 chunk 记录会被删除，新的 chunk 会被插入，并且对应的向量会被更新到 FAISS 索引中
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_id INTEGER UNIQUE,
                updated TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES wiki_pages(page_id) ON DELETE CASCADE
            )
            """
        )
        # facts 记录从文本中抽取的结构化事实，便于构建知识图谱和辅助推理
        # fact_id 是自增主键，subject 是事实主体，predicate 是关系，object 是客体，source_page_id 是来源页面 ID，raw_path 是对应的 raw 文件路径，updated 记录更新时间
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source_page_id TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                updated TEXT NOT NULL
            )
            """
        )

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wiki_pages_type ON wiki_pages(page_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wiki_pages_raw ON wiki_pages(raw_path)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_refs_to ON page_refs(to_page_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject_pred ON facts(subject, predicate)"
        )
        self._conn.commit()

    def _scan_raw_files(self) -> dict[str, str]:
        """
        扫描 raw 目录并返回文件 MD5 映射。
        Returns:
            dict[str, str]: {raw_relative_path: file_md5}
        """
        results: dict[str, str] = {}
        for file_path in self.raw_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in RAW_SUFFIXES:
                continue
            rel_path = file_path.relative_to(self.raw_dir).as_posix()
            try:
                results[rel_path] = _sum_md5(file_path)
            except Exception as exc:
                Log.logger.warning(f"扫描 raw 文件失败，已跳过: {file_path}, {exc}")
        return results

    def _get_known_raw_files(self) -> dict[str, str]:
        """
        读取元数据库中的 raw 文件记录。
        Returns:
            dict[str, str]: {raw_relative_path: file_md5}
        """
        cursor = self._conn.execute("SELECT raw_path, file_md5 FROM raw_files")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def _make_page_id(self, page_type: str, title: str) -> str:
        """
        生成页面唯一 ID。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
        Returns:
            str: 页面 ID
        """
        safe = _sanitize_page_name(title)
        return f"{page_type}/{safe}"

    def _page_file_path(self, page_type: str, title: str) -> Path:
        """
        计算页面文件路径。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
        Returns:
            Path: 页面文件绝对路径
        """
        safe = _sanitize_page_name(title)
        if page_type == "entity":
            return self.wiki_entity_dir / f"{safe}.md"
        if page_type == "concept":
            return self.wiki_concept_dir / f"{safe}.md"
        return self.wiki_source_dir / f"{safe}.md"

    def _get_page(self, page_id: str) -> sqlite3.Row | None:
        """
        获取页面记录。
        Parameters:
            page_id (str): 页面 ID
        Returns:
            sqlite3.Row | None: 页面记录
        """
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE page_id = ? LIMIT 1", (page_id,)
        )
        row = cursor.fetchone()
        self._conn.row_factory = None
        return row

    def _render_frontmatter(
        self,
        page_type: str,
        tags: list[str],
        sources: list[str],
        created: str,
        updated: str,
    ) -> str:
        """
        渲染 Markdown frontmatter。
        Parameters:
            page_type (str): 页面类型
            tags (list[str]): 标签
            sources (list[str]): 来源
            created (str): 创建日期
            updated (str): 更新日期
        Returns:
            str: frontmatter 文本
        """
        data = {
            "type": page_type,
            "tags": tags,
            "sources": sources,
            "created": created,
            "updated": updated,
        }
        yaml_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()
        return f"---\n{yaml_text}\n---\n"

    def _write_page_file(
        self,
        page_type: str,
        title: str,
        body: str,
        tags: list[str],
        sources: list[str],
        created: str,
        updated: str,
    ) -> tuple[Path, str]:
        """
        将页面落盘为 Markdown。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
            body (str): 页面正文
            tags (list[str]): 标签
            sources (list[str]): 来源
            created (str): 创建日期
            updated (str): 更新日期
        Returns:
            tuple[Path, str]: 文件路径与完整内容
        """
        file_path = self._page_file_path(page_type, title)
        frontmatter = self._render_frontmatter(
            page_type=page_type,
            tags=tags,
            sources=sources,
            created=created,
            updated=updated,
        )
        content = f"{frontmatter}\n# {title}\n\n{body.strip()}\n"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path, content

    def _upsert_page(
        self,
        page_type: str,
        title: str,
        body: str,
        tags: list[str],
        sources: list[str],
        raw_path: str | None = None,
    ) -> str:
        """
        写入或更新页面及元数据。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
            body (str): 页面正文
            tags (list[str]): 标签
            sources (list[str]): 来源
            raw_path (str | None): 对应 raw 文件
        Returns:
            str: 页面 ID
        """
        page_id = self._make_page_id(page_type, title)
        now_date = _now_date()
        now_dt = _now_datetime()

        existing = self._get_page(page_id)
        created = existing["created"] if existing else now_date

        file_path, content = self._write_page_file(
            page_type=page_type,
            title=title,
            body=body,
            tags=tags,
            sources=sources,
            created=created,
            updated=now_date,
        )

        if existing and existing["file_path"] != str(file_path):
            old_path = Path(existing["file_path"])
            if old_path.exists():
                old_path.unlink(missing_ok=True)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO wiki_pages (
                page_id, page_type, title, file_path, content,
                tags_json, sources_json, created, updated, raw_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                page_type,
                title,
                str(file_path),
                content,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(sources, ensure_ascii=False),
                created,
                now_dt,
                raw_path,
            ),
        )
        self._conn.commit()
        return page_id

    def _delete_page(self, page_id: str) -> None:
        """
        删除页面及其相关元数据。
        Parameters:
            page_id (str): 页面 ID
        """
        row = self._get_page(page_id)
        if row:
            file_path = Path(row["file_path"])
            if file_path.exists():
                file_path.unlink(missing_ok=True)

        self._conn.execute("DELETE FROM page_refs WHERE from_page_id = ?", (page_id,))
        self._conn.execute("DELETE FROM page_refs WHERE to_page_id = ?", (page_id,))
        self._conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        self._conn.execute("DELETE FROM wiki_pages WHERE page_id = ?", (page_id,))
        self._conn.commit()

    def _ensure_stub_page(self, page_type: str, title: str, source_hint: str) -> str:
        """
        确保实体/概念页面存在，缺失则创建占位页。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
            source_hint (str): 来源提示
        Returns:
            str: 页面 ID
        """
        page_id = self._make_page_id(page_type, title)
        if self._get_page(page_id):
            return page_id

        body = """## 简介\n\n该页面由知识库自动创建，后续会持续补充。\n\n## 关联来源\n\n- 待补充\n"""
        return self._upsert_page(
            page_type=page_type,
            title=title,
            body=body,
            tags=["auto-generated", page_type],
            sources=[source_hint],
            raw_path=None,
        )

    def _replace_refs(self, from_page_id: str, refs: list[tuple[str, str]]) -> None:
        """
        替换页面出链关系。
        Parameters:
            from_page_id (str): 来源页面 ID
            refs (list[tuple[str, str]]): [(to_page_id, ref_type)]
        """
        now_dt = _now_datetime()
        self._conn.execute(
            "DELETE FROM page_refs WHERE from_page_id = ?", (from_page_id,)
        )
        for to_page_id, ref_type in refs:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO page_refs (from_page_id, to_page_id, ref_type, updated)
                VALUES (?, ?, ?, ?)
                """,
                (from_page_id, to_page_id, ref_type, now_dt),
            )
        self._conn.commit()

    def _extract_title_from_text(self, text: str, fallback: str) -> str:
        """
        从 Markdown 文本提取标题。
        Parameters:
            text (str): 文本内容
            fallback (str): 默认标题
        Returns:
            str: 标题
        """
        heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        if heading:
            return heading.group(1).strip()
        return fallback

    def _fallback_extract(self, raw_rel_path: str, cleaned_text: str) -> dict[str, Any]:
        """
        规则回退抽取。
        Parameters:
            raw_rel_path (str): raw 相对路径
            cleaned_text (str): 清洗后的文本
        Returns:
            dict[str, Any]: 结构化结果
        """
        stem = Path(raw_rel_path).stem
        title = self._extract_title_from_text(cleaned_text, stem)
        blocks = _split_semantic_blocks(cleaned_text, max_chars=500, min_chars=80)
        summary = "\n\n".join(blocks[:2]) if blocks else cleaned_text[:300]
        summary = summary.strip() or f"来源 {raw_rel_path} 的摘要待补充。"

        return {
            "title": title,
            "summary": summary,
            "tags": ["source", "auto"],
            "entities": [],
            "concepts": [],
            "relations": [],
        }

    def _extract_structured_info(
        self, raw_rel_path: str, cleaned_text: str
    ) -> dict[str, Any]:
        """
        调用 LLM 阅读理解并提取结构化信息。
        Parameters:
            raw_rel_path (str): raw 相对路径
            cleaned_text (str): 清洗后的文本
        Returns:
            dict[str, Any]: 抽取结果
        """
        if not self.enable_llm_extract:
            return self._fallback_extract(raw_rel_path, cleaned_text)

        stem = Path(raw_rel_path).stem
        sample_text = cleaned_text
        if len(sample_text) > 9000:
            sample_text = f"{sample_text[:5500]}\n\n...\n\n{sample_text[-2500:]}"

        prompt = f"""
你是知识工程助手。请阅读下面的原始资料，并输出严格 JSON（不要输出多余文字）：

{{
  "title": "来源标题",
  "summary": "200-500字中文摘要",
  "tags": ["标签1", "标签2"],
  "entities": ["实体名1", "实体名2"],
  "concepts": ["概念1", "概念2"],
  "relations": [
    {{"subject": "主体", "predicate": "关系", "object": "客体"}}
  ]
}}

要求：
1. entities 和 concepts 去重后返回。
2. relations 只保留可被文本支持的关系。
3. 若信息不足可返回空数组，但字段必须存在。
4. title 尽量简洁，且适合写入 Markdown 文件名。

来源文件：{raw_rel_path}
默认标题：{stem}
资料内容：
{sample_text}
        """.strip()

        try:
            content = _sync_run_coroutine(
                llm_request([{"role": "user", "content": prompt}]), timeout=120
            )
            if not content:
                return self._fallback_extract(raw_rel_path, cleaned_text)
            parsed = parse_llm_json_response(content)
        except Exception as exc:
            Log.logger.warning(f"LLM 抽取失败，使用回退逻辑: {exc}")
            return self._fallback_extract(raw_rel_path, cleaned_text)

        title = str(parsed.get("title") or stem).strip()
        if not title:
            title = stem

        summary = str(parsed.get("summary") or "").strip()
        if not summary:
            summary = self._fallback_extract(raw_rel_path, cleaned_text)["summary"]

        tags = [
            str(item).strip() for item in parsed.get("tags", []) if str(item).strip()
        ]
        entities = [
            str(item).strip()
            for item in parsed.get("entities", [])
            if str(item).strip()
        ]
        concepts = [
            str(item).strip()
            for item in parsed.get("concepts", [])
            if str(item).strip()
        ]

        normalized_relations: list[dict[str, str]] = []
        for item in parsed.get("relations", []):
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            normalized_relations.append(
                {"subject": subject, "predicate": predicate, "object": obj}
            )

        def uniq(values: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for value in values:
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append(value)
            return result

        return {
            "title": title,
            "summary": summary,
            "tags": uniq(tags) or ["source"],
            "entities": uniq(entities),
            "concepts": uniq(concepts),
            "relations": normalized_relations,
        }

    def _build_source_body(
        self,
        summary: str,
        entities: list[str],
        concepts: list[str],
        relations: list[dict[str, str]],
        raw_preview: str,
    ) -> str:
        """
        构建 source 页面正文。
        Parameters:
            summary (str): 摘要
            entities (list[str]): 实体列表
            concepts (list[str]): 概念列表
            relations (list[dict[str, str]]): 关系列表
            raw_preview (str): 原文预览
        Returns:
            str: 页面正文
        """
        lines: list[str] = ["## 摘要", "", summary.strip(), ""]

        lines.append("## 实体")
        lines.append("")
        if entities:
            for entity in entities:
                safe = _sanitize_page_name(entity)
                lines.append(f"- [{entity}](../entity/{safe}.md)")
        else:
            lines.append("- 无")
        lines.append("")

        lines.append("## 概念")
        lines.append("")
        if concepts:
            for concept in concepts:
                safe = _sanitize_page_name(concept)
                lines.append(f"- [{concept}](../concept/{safe}.md)")
        else:
            lines.append("- 无")
        lines.append("")

        lines.append("## 关系")
        lines.append("")
        if relations:
            for relation in relations:
                subj = relation["subject"]
                pred = relation["predicate"]
                obj = relation["object"]
                subj_safe = _sanitize_page_name(subj)
                obj_safe = _sanitize_page_name(obj)
                lines.append(
                    f"- [{subj}](../entity/{subj_safe}.md) {pred} [{obj}](../entity/{obj_safe}.md)"
                )
        else:
            lines.append("- 无")
        lines.append("")

        lines.append("## 原文片段")
        lines.append("")
        lines.append(raw_preview.strip() if raw_preview.strip() else "（空）")

        return "\n".join(lines)

    def _upsert_entity_or_concept_page(
        self, page_type: str, title: str, source_rel_path: str
    ) -> str:
        """
        新建或更新实体/概念页。
        Parameters:
            page_type (str): 页面类型
            title (str): 页面标题
            source_rel_path (str): 来源路径
        Returns:
            str: 页面 ID
        """
        page_id = self._make_page_id(page_type, title)
        existing = self._get_page(page_id)
        if existing:
            try:
                existing_sources = json.loads(existing["sources_json"])
            except Exception:
                existing_sources = []
            merged_sources = sorted(set(existing_sources + [source_rel_path]))

            content = existing["content"]
            body = self._strip_frontmatter(content)
            body = self._strip_title_heading(body, title)
            self._upsert_page(
                page_type=page_type,
                title=title,
                body=body,
                tags=[page_type, "knowledge"],
                sources=merged_sources,
                raw_path=None,
            )
            return page_id

        body = """## 定义\n\n该页面由知识库自动创建，后续会持续补充定义与细节。\n\n## 关联来源\n\n- 待补充\n"""
        return self._upsert_page(
            page_type=page_type,
            title=title,
            body=body,
            tags=[page_type, "knowledge", "auto-generated"],
            sources=[source_rel_path],
            raw_path=None,
        )

    def _strip_frontmatter(self, content: str) -> str:
        """
        去除 Markdown frontmatter。
        Parameters:
            content (str): 原始内容
        Returns:
            str: 去除 frontmatter 后内容
        """
        if not content.startswith("---\n"):
            return content
        parts = content.split("\n---\n", maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()
        return content

    def _strip_title_heading(self, body: str, title: str) -> str:
        """
        去除正文开头重复的一级标题。
        Parameters:
            body (str): 页面正文
            title (str): 页面标题
        Returns:
            str: 去标题后的正文
        """
        heading = f"# {title}".strip()
        normalized = body.strip()
        if normalized.startswith(heading):
            normalized = normalized[len(heading) :].strip()
        return normalized

    def _delete_raw_related_data(self, raw_rel_path: str) -> None:
        """
        删除某个 raw 文件关联的数据。
        Parameters:
            raw_rel_path (str): raw 相对路径
        """
        cursor = self._conn.execute(
            "SELECT source_page_id FROM raw_files WHERE raw_path = ? LIMIT 1",
            (raw_rel_path,),
        )
        row = cursor.fetchone()
        if row and row[0]:
            self._delete_page(row[0])

        self._conn.execute("DELETE FROM facts WHERE raw_path = ?", (raw_rel_path,))
        self._conn.execute("DELETE FROM raw_files WHERE raw_path = ?", (raw_rel_path,))
        self._conn.commit()

    def _ingest_raw_file(self, raw_rel_path: str, file_md5: str) -> dict[str, Any]:
        """
        处理单个 raw 文件，生成 Wiki 页面与元数据。
        Parameters:
            raw_rel_path (str): raw 相对路径
            file_md5 (str): 文件 md5
        Returns:
            dict[str, Any]: ingest 摘要
        """
        raw_abs_path = self.raw_dir / raw_rel_path
        if not raw_abs_path.exists():
            return {"raw": raw_rel_path, "status": "missing"}

        raw_text = raw_abs_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = _clean_text(raw_text)

        self._delete_raw_related_data(raw_rel_path)

        if not cleaned:
            return {"raw": raw_rel_path, "status": "empty"}

        structured = self._extract_structured_info(raw_rel_path, cleaned)
        source_title = _sanitize_page_name(
            structured.get("title") or Path(raw_rel_path).stem
        )

        entities: list[str] = [
            _sanitize_page_name(item)
            for item in structured.get("entities", [])
            if str(item).strip()
        ]
        concepts: list[str] = [
            _sanitize_page_name(item)
            for item in structured.get("concepts", [])
            if str(item).strip()
        ]
        relations: list[dict[str, str]] = structured.get("relations", [])

        entity_page_ids: list[str] = []
        concept_page_ids: list[str] = []

        for entity in entities:
            entity_page_ids.append(
                self._upsert_entity_or_concept_page("entity", entity, raw_rel_path)
            )
        for concept in concepts:
            concept_page_ids.append(
                self._upsert_entity_or_concept_page("concept", concept, raw_rel_path)
            )

        raw_blocks = _split_semantic_blocks(cleaned, max_chars=600, min_chars=100)
        raw_preview = "\n\n".join(raw_blocks[:3])
        source_body = self._build_source_body(
            summary=str(structured.get("summary") or "").strip(),
            entities=entities,
            concepts=concepts,
            relations=relations,
            raw_preview=raw_preview,
        )

        source_page_id = self._upsert_page(
            page_type="source",
            title=source_title,
            body=source_body,
            tags=list(set(structured.get("tags", []) + ["source"])),
            sources=[raw_rel_path],
            raw_path=raw_rel_path,
        )

        refs: list[tuple[str, str]] = []
        refs.extend((page_id, "mention_entity") for page_id in entity_page_ids)
        refs.extend((page_id, "mention_concept") for page_id in concept_page_ids)
        self._replace_refs(source_page_id, refs)

        now_dt = _now_datetime()
        for relation in relations:
            subject = _sanitize_page_name(str(relation.get("subject") or "").strip())
            predicate = str(relation.get("predicate") or "").strip()
            obj = _sanitize_page_name(str(relation.get("object") or "").strip())
            if not (subject and predicate and obj):
                continue
            self._ensure_stub_page("entity", subject, raw_rel_path)
            self._ensure_stub_page("entity", obj, raw_rel_path)
            self._conn.execute(
                """
                INSERT INTO facts (subject, predicate, object, source_page_id, raw_path, updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (subject, predicate, obj, source_page_id, raw_rel_path, now_dt),
            )
        self._conn.commit()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO raw_files (raw_path, file_md5, source_page_id, updated)
            VALUES (?, ?, ?, ?)
            """,
            (raw_rel_path, file_md5, source_page_id, now_dt),
        )
        self._conn.commit()

        return {
            "raw": raw_rel_path,
            "status": "ingested",
            "source_page_id": source_page_id,
            "entities": len(entity_page_ids),
            "concepts": len(concept_page_ids),
            "relations": len(relations),
        }

    def _list_all_pages(self) -> list[tuple[str, str, str]]:
        """
        列出全部页面。
        Returns:
            list[tuple[str, str, str]]: [(page_id, title, content)]
        """
        cursor = self._conn.execute(
            "SELECT page_id, title, content FROM wiki_pages ORDER BY page_type, title"
        )
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def _rebuild_embeddings_index(self) -> None:
        """
        基于 Wiki 页面重建向量索引。
        """
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

        pages = self._list_all_pages()
        all_chunk_ids: list[int] = []
        all_chunk_texts: list[str] = []

        now_dt = _now_datetime()
        for page_id, _title, content in pages:
            body = self._strip_frontmatter(content)
            chunks = _split_semantic_blocks(body, max_chars=650, min_chars=90)
            if not chunks:
                continue

            for chunk_text in chunks:
                cursor = self._conn.execute(
                    """
                    INSERT INTO chunks (page_id, chunk_text, faiss_id, updated)
                    VALUES (?, ?, NULL, ?)
                    """,
                    (page_id, chunk_text, now_dt),
                )
                row_id = cursor.lastrowid
                if row_id is None:
                    continue
                all_chunk_ids.append(int(row_id))
                all_chunk_texts.append(chunk_text)

        self._conn.commit()

        if not all_chunk_ids:
            self.index = None
            self.dimension = None
            self.data_count = 0
            if self.index_path.exists():
                self.index_path.unlink(missing_ok=True)
            return

        vectors = _normalize_vectors(embedding.t2vect(all_chunk_texts))
        dimension = int(vectors.shape[1])

        base_index = faiss.IndexFlatIP(dimension)
        index_obj = faiss.IndexIDMap2(base_index)
        ids = np.asarray(all_chunk_ids, dtype=np.int64)

        try:
            index_obj.add_with_ids(vectors, ids)  # type: ignore[call-arg]
        except TypeError:
            index_obj.add_with_ids(
                int(vectors.shape[0]),
                faiss.swig_ptr(vectors),
                faiss.swig_ptr(ids),
            )

        for chunk_id in all_chunk_ids:
            self._conn.execute(
                "UPDATE chunks SET faiss_id = ? WHERE chunk_id = ?",
                (chunk_id, chunk_id),
            )
        self._conn.commit()

        self.index = index_obj
        self.dimension = dimension
        self.data_count = len(all_chunk_ids)
        faiss.write_index(self.index, str(self.index_path))

    def _refresh_wiki_index_page(self) -> None:
        """
        重建 wiki/index.md 页面。
        """
        cursor = self._conn.execute(
            "SELECT page_type, title, file_path FROM wiki_pages ORDER BY page_type, title"
        )
        rows = cursor.fetchall()

        grouped: dict[str, list[tuple[str, str]]] = {
            "source": [],
            "entity": [],
            "concept": [],
        }
        for page_type, title, file_path in rows:
            grouped.setdefault(page_type, []).append((title, file_path))

        lines: list[str] = [
            "# 知识库索引",
            "",
            f"更新时间: {_now_datetime()}",
            "",
            "## source",
            "",
        ]
        if grouped.get("source"):
            for title, file_path in grouped["source"]:
                rel = Path(file_path).relative_to(self.wiki_dir).as_posix()
                lines.append(f"- [{title}]({rel})")
        else:
            lines.append("- 无")

        lines.extend(["", "## entity", ""])
        if grouped.get("entity"):
            for title, file_path in grouped["entity"]:
                rel = Path(file_path).relative_to(self.wiki_dir).as_posix()
                lines.append(f"- [{title}]({rel})")
        else:
            lines.append("- 无")

        lines.extend(["", "## concept", ""])
        if grouped.get("concept"):
            for title, file_path in grouped["concept"]:
                rel = Path(file_path).relative_to(self.wiki_dir).as_posix()
                lines.append(f"- [{title}]({rel})")
        else:
            lines.append("- 无")

        (self.wiki_dir / "index.md").write_text(
            "\n".join(lines).strip() + "\n", encoding="utf-8"
        )

    def _ensure_orphan_summary_page(self, orphan_page_ids: list[str]) -> int:
        """
        将孤儿页面汇总到专用页面，避免无入链状态。
        Parameters:
            orphan_page_ids (list[str]): 孤儿页面 ID 列表
        Returns:
            int: 新增关联数
        """
        if not orphan_page_ids:
            return 0

        title = "孤儿页面汇总"
        summary_page_id = self._upsert_page(
            page_type="concept",
            title=title,
            body="## 说明\n\n该页由系统自动维护，用于链接暂时没有入链的页面。\n",
            tags=["health-check", "summary"],
            sources=["system"],
            raw_path=None,
        )

        now_dt = _now_datetime()
        count = 0
        for page_id in orphan_page_ids:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO page_refs (from_page_id, to_page_id, ref_type, updated)
                VALUES (?, ?, ?, ?)
                """,
                (summary_page_id, page_id, "orphan-link", now_dt),
            )
            count += 1
        self._conn.commit()
        return count

    def _check_and_fix_contradictions(
        self, auto_fix: bool
    ) -> tuple[list[dict[str, Any]], int]:
        """
        检查并修复事实冲突。
        Parameters:
            auto_fix (bool): 是否自动修复
        Returns:
            tuple[list[dict[str, Any]], int]: 冲突列表与修复数量
        """
        cursor = self._conn.execute(
            """
            SELECT subject, predicate, COUNT(DISTINCT object) AS obj_count
            FROM facts
            GROUP BY subject, predicate
            HAVING obj_count > 1
            """
        )
        groups = cursor.fetchall()

        conflicts: list[dict[str, Any]] = []
        fixed = 0

        for subject, predicate, _obj_count in groups:
            c2 = self._conn.execute(
                """
                SELECT fact_id, object, updated
                FROM facts
                WHERE subject = ? AND predicate = ?
                ORDER BY updated DESC
                """,
                (subject, predicate),
            )
            rows = c2.fetchall()
            objects = [row[1] for row in rows]
            conflicts.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "objects": sorted(set(objects)),
                }
            )

            if auto_fix and rows:
                latest_object = rows[0][1]
                self._conn.execute(
                    """
                    DELETE FROM facts
                    WHERE subject = ? AND predicate = ? AND object != ?
                    """,
                    (subject, predicate, latest_object),
                )
                fixed += 1

        if auto_fix and fixed > 0:
            self._conn.commit()

        return conflicts, fixed

    def run_health_check(self, auto_fix: bool = True) -> dict[str, Any]:
        """
        执行健康检查并按策略修复。
        Parameters:
            auto_fix (bool): 是否自动修复
        Returns:
            dict[str, Any]: 健康检查报告
        """
        report: dict[str, Any] = {
            "checked_at": _now_datetime(),
            "auto_fix": auto_fix,
            "contradictions": [],
            "orphan_pages": [],
            "missing_targets": [],
            "outdated_sources": [],
            "index_sync_ok": True,
            "summary_page_candidates": [],
            "fixed": {
                "contradictions": 0,
                "orphan_links": 0,
                "missing_pages": 0,
                "outdated_reingest": 0,
                "index_rebuilt": False,
            },
        }

        contradictions, contradiction_fixed = self._check_and_fix_contradictions(
            auto_fix=auto_fix
        )
        report["contradictions"] = contradictions
        report["fixed"]["contradictions"] = contradiction_fixed

        cursor = self._conn.execute(
            """
            SELECT p.page_id
            FROM wiki_pages p
            LEFT JOIN page_refs r ON p.page_id = r.to_page_id
            WHERE p.page_type IN ('entity', 'concept')
            GROUP BY p.page_id
            HAVING COUNT(r.from_page_id) = 0
            """
        )
        orphan_ids = [row[0] for row in cursor.fetchall()]
        report["orphan_pages"] = orphan_ids

        if auto_fix and orphan_ids:
            linked = self._ensure_orphan_summary_page(orphan_ids)
            report["fixed"]["orphan_links"] = linked

        cursor = self._conn.execute(
            """
            SELECT DISTINCT r.to_page_id
            FROM page_refs r
            LEFT JOIN wiki_pages p ON r.to_page_id = p.page_id
            WHERE p.page_id IS NULL
            """
        )
        missing_targets = [row[0] for row in cursor.fetchall()]
        report["missing_targets"] = missing_targets

        if auto_fix and missing_targets:
            fixed_missing = 0
            for page_id in missing_targets:
                if "/" not in page_id:
                    continue
                page_type, title = page_id.split("/", maxsplit=1)
                if page_type not in {"entity", "concept", "source"}:
                    continue
                self._ensure_stub_page(page_type, title, "system")
                fixed_missing += 1
            report["fixed"]["missing_pages"] = fixed_missing

        cursor = self._conn.execute(
            """
            SELECT rf.raw_path, rf.updated, p.updated, rf.file_md5
            FROM raw_files rf
            JOIN wiki_pages p ON rf.source_page_id = p.page_id
            """
        )
        outdated = []
        current_raw = self._scan_raw_files()
        for raw_path, _raw_updated, _page_updated, stored_md5 in cursor.fetchall():
            current_md5 = current_raw.get(raw_path)
            if current_md5 and current_md5 != stored_md5:
                outdated.append(raw_path)
        report["outdated_sources"] = outdated

        if auto_fix and outdated:
            fixed_count = 0
            for raw_path in outdated:
                current_md5 = current_raw.get(raw_path)
                if not current_md5:
                    continue
                self._ingest_raw_file(raw_path, current_md5)
                fixed_count += 1
            report["fixed"]["outdated_reingest"] = fixed_count

        cursor = self._conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = int(cursor.fetchone()[0])

        index_count = int(self.index.ntotal) if self.index is not None else 0
        index_ok = chunk_count == index_count
        report["index_sync_ok"] = index_ok

        if auto_fix and not index_ok:
            self._rebuild_embeddings_index()
            report["fixed"]["index_rebuilt"] = True

        cursor = self._conn.execute(
            """
            SELECT to_page_id, COUNT(*) AS inbound_count
            FROM page_refs
            GROUP BY to_page_id
            HAVING inbound_count >= 5
            ORDER BY inbound_count DESC
            """
        )
        candidates = [row[0] for row in cursor.fetchall()]
        report["summary_page_candidates"] = candidates

        self._refresh_wiki_index_page()
        self._last_health_check_ts = int(time.time())

        return report

    def _maybe_periodic_health_check(self) -> None:
        """
        按周期触发健康检查。
        """
        if self.health_check_interval_sec <= 0:
            return
        now_ts = int(time.time())
        if now_ts - self._last_health_check_ts < self.health_check_interval_sec:
            return

        try:
            self.run_health_check(auto_fix=True)
        except Exception as exc:
            Log.logger.warning(f"周期健康检查失败: {exc}")

    def rebuild(self, startup: bool = False) -> dict[str, Any]:
        """
        重建知识库（扫描 raw、更新 wiki、重建索引）。
        Parameters:
            startup (bool): 是否为启动触发
        Returns:
            dict[str, Any]: 重建摘要
        """
        current_raw = self._scan_raw_files()
        known_raw = self._get_known_raw_files()

        deleted = [raw for raw in known_raw.keys() if raw not in current_raw]
        changed = [
            raw
            for raw, md5 in current_raw.items()
            if raw not in known_raw or known_raw.get(raw) != md5
        ]

        deleted_count = 0
        for raw_rel_path in deleted:
            self._delete_raw_related_data(raw_rel_path)
            deleted_count += 1

        ingest_details: list[dict[str, Any]] = []
        for raw_rel_path in changed:
            try:
                detail = self._ingest_raw_file(raw_rel_path, current_raw[raw_rel_path])
                ingest_details.append(detail)
            except Exception as exc:
                Log.logger.error(f"ingest 失败: {raw_rel_path}, {exc}")

        self._rebuild_embeddings_index()
        health_report = self.run_health_check(auto_fix=True)

        summary = {
            "startup": startup,
            "raw_total": len(current_raw),
            "changed": len(changed),
            "deleted": deleted_count,
            "ingest_details": ingest_details,
            "wiki_pages": self._count_wiki_pages(),
            "vector_chunks": self.data_count,
            "health": health_report,
            "updated_at": _now_datetime(),
        }
        self._last_rebuild_summary = summary
        return summary

    def _count_wiki_pages(self) -> dict[str, int]:
        """
        统计各类页面数量。
        Returns:
            dict[str, int]: 统计结果
        """
        cursor = self._conn.execute(
            "SELECT page_type, COUNT(*) FROM wiki_pages GROUP BY page_type"
        )
        data = {"source": 0, "entity": 0, "concept": 0}
        for page_type, count in cursor.fetchall():
            data[page_type] = int(count)
        data["total"] = sum(data.values())
        return data

    def get_status(self) -> dict[str, Any]:
        """
        获取知识库状态。
        Returns:
            dict[str, Any]: 状态摘要
        """
        cursor = self._conn.execute("SELECT COUNT(*) FROM raw_files")
        raw_count = int(cursor.fetchone()[0])

        cursor = self._conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = int(cursor.fetchone()[0])

        index_count = int(self.index.ntotal) if self.index is not None else 0

        return {
            "base_dir": str(self.base_dir),
            "raw_dir": str(self.raw_dir),
            "wiki_dir": str(self.wiki_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "raw_count": raw_count,
            "wiki_pages": self._count_wiki_pages(),
            "chunk_count": chunk_count,
            "index_count": index_count,
            "threshold": self.thresholds,
            "top_k": self.top_k,
            "last_rebuild": self._last_rebuild_summary,
            "last_health_check_ts": self._last_health_check_ts,
        }

    def _fetch_chunk_payload(self, chunk_id: int) -> dict[str, str] | None:
        """
        根据 chunk_id 获取文本和页面信息。
        Parameters:
            chunk_id (int): chunk 主键
        Returns:
            dict[str, str] | None: 查询结果
        """
        cursor = self._conn.execute(
            """
            SELECT c.chunk_text, p.title, p.page_type, p.file_path
            FROM chunks c
            JOIN wiki_pages p ON c.page_id = p.page_id
            WHERE c.chunk_id = ?
            LIMIT 1
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "chunk_text": row[0],
            "title": row[1],
            "page_type": row[2],
            "file_path": row[3],
        }

    def search(self, text: list[str]) -> str:
        """
        在知识库中执行语义搜索。
        Parameters:
            text (list[str]): 查询文本列表
        Returns:
            str: 可注入 Prompt 的检索结果
        """
        self._maybe_periodic_health_check()

        if self.index is None or self.data_count == 0:
            return ""

        queries = [item.strip() for item in text if item and item.strip()]
        if not queries:
            return ""

        query_vectors = _normalize_vectors(embedding.t2vect(queries))
        query_count = int(query_vectors.shape[0])
        try:
            distances, ids = self.index.search(query_vectors, self.top_k)  # type: ignore[call-arg]
        except TypeError:
            distances = np.empty((query_count, self.top_k), dtype=np.float32)
            ids = np.empty((query_count, self.top_k), dtype=np.int64)
            self.index.search(
                query_count,
                faiss.swig_ptr(query_vectors),
                self.top_k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(ids),
            )

        seen_chunk_ids: set[int] = set()
        output_blocks: list[str] = []

        for row_index in range(query_count):
            for col_index in range(self.top_k):
                score = float(distances[row_index][col_index])
                chunk_id = int(ids[row_index][col_index])
                if chunk_id < 0:
                    continue
                if score < self.thresholds:
                    continue
                if chunk_id in seen_chunk_ids:
                    continue

                payload = self._fetch_chunk_payload(chunk_id)
                if not payload:
                    continue

                seen_chunk_ids.add(chunk_id)
                block = (
                    f"[{payload['page_type']}] {payload['title']} (score={score:.3f})\n"
                    f"{payload['chunk_text']}"
                )
                output_blocks.append(block)

        return "\n\n".join(output_blocks) + ("\n\n" if output_blocks else "")
