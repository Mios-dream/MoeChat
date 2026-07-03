"""
Live2D 动作引擎 V3 — 服务端适配器

核心架构：
    SQLite 动作数据库 → embedding 语义检索 → 特殊动作覆盖 → 逐帧曲线输出

本模块是 motion_engine.py 的服务端适配版本，移除了 PySide6/MotionPlayer 依赖，
专注于动作数据生成，直接输出逐帧曲线 (curves: {param_id: [frame_values]}) 供 SSE 传输。

类：
    MotionData          — 封装动作数据（参数帧数组）
    MotionMeta          — 动作元数据 dataclass（不含曲线）
    MotionDatabase      — SQLite 数据库，含预计算 embedding + 预解析动作曲线
    SemanticMatcher     — embedding 语义检索（基于预计算向量）
    ActionOverlay       — 特殊动作 → 参数帧数组生成
    MotionEngineService — 服务端统一入口，串联检索/覆盖/混合全流程

使用示例：
    from core.expression_generator.motion_engine_v3 import get_motion_engine

    engine = get_motion_engine()
    motion_data = engine.process(
        text="你好呀~",
        actions=["smile", "wink_left"],
        max_duration=3.0,
    )
    # motion_data.curves: {"ParamAngleX": [-2.0, -1.9, ...], ...}
"""

import sqlite3
from dataclasses import dataclass
from core.expression_generator.utils.expression_loader import ExpressionInfo
from my_utils import embedding
from my_utils.log import logger as Log
import numpy as np

# 特殊动作定义: {param_id: [(time_offset_seconds, target_value), ...]}
SIMPLE_ACTIONS: dict[str, dict[str, list[tuple[float, float]]]] = {
    "close_eyes": {
        "ParamEyeLOpen": [(1, 0.0)],
        "ParamEyeROpen": [(1, 0.0)],
    },
    "wink_left": {
        "ParamEyeLOpen": [(1, 0.0), (1.3, 1.0)],
    },
    "wink_right": {
        "ParamEyeROpen": [(1, 0.0), (1.3, 1.0)],
    },
    "blush": {
        "ParamCheek": [(0.0, 0.8)],
        "ParamEyeLSmile": [(0.0, 0.3)],
        "ParamEyeRSmile": [(0.0, 0.3)],
    },
    "surprise": {
        "ParamEyeLOpen": [(0.0, 1.0)],
        "ParamEyeROpen": [(0.0, 1.0)],
        "ParamBrowLY": [(0.0, 0.5)],
        "ParamBrowRY": [(0.0, 0.5)],
    },
    "pout": {
        "ParamMouthForm": [(0.0, -0.4)],
        "ParamMouthOpenY": [(0.0, 0.2)],
    },
}

ACTION_DESCRIPTIONS = {
    "close_eyes": "闭眼（害羞、享受）",
    "wink_left": "左眼 wink（俏皮、可爱）",
    "wink_right": "右眼 wink（俏皮、可爱）",
    "blush": "脸红（害羞、尴尬）",
    "surprise": "惊讶（睁大眼睛）",
    "pout": "嘟嘴（撒娇、不满）",
}

# 参数默认值（用于动作播放结束后恢复 + 嘴部参数默认填充）
PARAM_DEFAULTS: dict[str, float] = {
    "ParamAngleX": 0.0,
    "ParamAngleY": 0.0,
    "ParamAngleZ": 0.0,
    "ParamEyeLOpen": 1.0,
    "ParamEyeROpen": 1.0,
    "ParamEyeLSmile": 0.0,
    "ParamEyeRSmile": 0.0,
    "ParamEyeBallX": 0.0,
    "ParamEyeBallY": 0.0,
    "ParamBrowLAngle": 0.0,
    "ParamBrowRAngle": 0.0,
    "ParamBrowLY": 0.0,
    "ParamBrowRY": 0.0,
    "ParamBrowLForm": 0.0,
    "ParamBrowRForm": 0.0,
    "ParamMouthForm": 0.0,
    "ParamMouthOpenY": 0.0,
    "ParamCheek": 0.0,
    "ParamTear": 0.0,
    "ParamBodyAngleX": 0.0,
    "ParamBodyAngleY": 0.0,
    "ParamBodyAngleZ": 0.0,
    "ParamBreath": 0.0,
}


@dataclass
class MotionData:
    """
    解析后的一帧级动作数据

    Attributes:
        curves: param_id → [frame0_value, frame1_value, ...]
        duration: 动作时长（秒）
        fps: 帧率
        expression: 表情名称列表（不覆盖动作曲线，由前端单独处理）
    """

    curves: dict[str, list[float]]
    duration: float
    fps: float = 60.0
    expression: list[str] | None = None

    @property
    def frame_count(self) -> int:
        """总帧数"""
        return int(self.duration * self.fps) + 1


@dataclass
class MotionMeta:
    """
    动作元数据（不含曲线数据）

    Attributes:
        motion_id: 数据库中的 motion 主键
        duration: 动作时长（秒）
        fps: 帧率
        frame_count: 总帧数
        param_count: 参数个数
    """

    motion_id: int
    duration: float
    fps: float
    frame_count: int
    param_count: int


class MotionDatabase:
    """
    基于 SQLite 的预构建动作数据库

    启动时加载 embedding 矩阵到内存（~9MB），动作曲线按需从 BLOB 反序列化。

    Attributes:
        _conn: SQLite 只读连接
        _embeddings: (N, D) float32 — 所有条目的预计算 embedding
        _texts: list[str] — 对话文本列表
        _motion_ids: (N,) int32 — 每条对话关联的 motion_id
        _param_ids: list[str] — 参数 ID 顺序（对应 curves_blob 的行）
        _motion_meta: dict[int, MotionMeta] — motion_id → 元数据
    """

    def __init__(self, db_path: str | None = None):
        """
        初始化数据库，从 SQLite 加载 embedding 和元数据

        Args:
            db_path: motion.db 路径，默认 data/motion.db（相对于项目根目录）
        """

        self._conn: sqlite3.Connection = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._texts: list[str] = []
        self._motion_ids: np.ndarray = np.empty((0,), dtype=np.int32)
        self._param_ids: list[str] = []
        self._motion_meta: dict[int, MotionMeta] = {}
        self._embedding_model: str = "unknown"

        self._load()

    def _load(self) -> None:
        """从 SQLite 加载全部 embedding 和元数据到内存"""
        rows = self._conn.execute(
            "SELECT param_id FROM param_order ORDER BY idx"
        ).fetchall()
        self._param_ids = [r["param_id"] for r in rows]

        rows = self._conn.execute(
            "SELECT id, duration, fps, frame_count, param_count FROM motions"
        ).fetchall()
        for r in rows:
            self._motion_meta[r["id"]] = MotionMeta(
                motion_id=r["id"],
                duration=r["duration"],
                fps=r["fps"],
                frame_count=r["frame_count"],
                param_count=r["param_count"],
            )

        rows = self._conn.execute(
            "SELECT id, text, embedding_blob, motion_id FROM entries ORDER BY id"
        ).fetchall()
        if not rows:
            Log.info("[MotionDatabase] 数据库为空，请先运行 build_motion_db.py")
            return

        n = len(rows)
        first_blob = rows[0]["embedding_blob"]
        dim = len(first_blob) // 4

        embeddings = np.empty((n, dim), dtype=np.float32)
        texts: list[str] = []
        motion_ids = np.empty(n, dtype=np.int32)

        for i, r in enumerate(rows):
            texts.append(r["text"])
            embeddings[i] = np.frombuffer(r["embedding_blob"], dtype=np.float32)
            motion_ids[i] = r["motion_id"]

        self._embeddings = embeddings
        self._texts = texts
        self._motion_ids = motion_ids

        row = self._conn.execute(
            "SELECT value FROM _meta WHERE key='embedding_model'"
        ).fetchone()
        if row:
            self._embedding_model = row["value"]

        Log.info(
            f"[MotionDatabase] 加载 {n} 条条目, "
            f"{len(self._motion_meta)} 个动作, "
            f"embedding 维度 {dim}"
        )

    def get_motion_curves(self, motion_id: int) -> dict[str, list[float]]:
        """
        按需从 BLOB 反序列化动作参数曲线

        Args:
            motion_id: motions 表主键

        Returns:
            dict[str, list[float]]: param_id → [frame0, frame1, ...]
        """
        row = self._conn.execute(
            "SELECT curves_blob, param_count, frame_count FROM motions WHERE id=?",
            (motion_id,),
        ).fetchone()
        if row is None:
            return {}

        arr = np.frombuffer(row["curves_blob"], dtype=np.float32).reshape(
            row["param_count"], row["frame_count"]
        )
        curves: dict[str, list[float]] = {}
        for i in range(row["param_count"]):
            curves[self._param_ids[i]] = arr[i].tolist()
        return curves

    def get_motion_meta(self, motion_id: int) -> MotionMeta | None:
        """
        获取动作元数据

        Args:
            motion_id: motions 表主键

        Returns:
            MotionMeta | None
        """
        return self._motion_meta.get(motion_id)

    def close(self) -> None:
        """关闭数据库连接"""
        self._conn.close()

    @property
    def num_entries(self) -> int:
        """对话条目总数"""
        return len(self._texts)

    @property
    def num_motions(self) -> int:
        """动作总数"""
        return len(self._motion_meta)

    @property
    def embedding_dim(self) -> int:
        """embedding 维度"""
        return self._embeddings.shape[1] if self._embeddings.size > 0 else 0

    @property
    def embedding_model(self) -> str:
        """构建时使用的 embedding 模型名"""
        return self._embedding_model


class SemanticMatcher:
    """
    语义检索器 — embedding 模式

    使用预计算 embedding 矩阵做余弦相似度检索。
    首次使用时加载 sentence-transformers 模型（仅用于查询编码）。

    Attributes:
        _model: SentenceTransformer 实例
        _embeddings: (N, D) 预计算文本 embedding 矩阵（引用自 MotionDatabase）
        _texts: 对话文本列表
        _motion_ids: (N,) 对应 motion_id
        _motion_durations: (N,) 对应动作时长（用于按时长优选）
    """

    def __init__(self, database: MotionDatabase):
        """
        初始化检索器

        加载 embedding 模型并从 MotionDatabase 获取预计算数据。

        Args:
            database: 已加载的 MotionDatabase 实例
        """
        self._texts: list[str] = database._texts
        self._motion_ids: np.ndarray = database._motion_ids
        self._embeddings: np.ndarray = database._embeddings

        durations: list[float] = []
        for mid in self._motion_ids:
            meta = database.get_motion_meta(int(mid))
            durations.append(meta.duration if meta else 2.0)
        self._motion_durations = np.array(durations, dtype=np.float32)

    def search(self, query: str, k: int = 3) -> list[tuple[str, int, float]]:
        """
        语义检索 top-k 最相似对话

        Args:
            query: 查询文本
            k: 返回结果数

        Returns:
            list[tuple[str, int, float]]: [(文本, motion_id, 相似度), ...]
        """

        query_embedding: np.ndarray = embedding.t2vect([query])
        similarities: np.ndarray = self._embeddings @ query_embedding.T
        top_indices = np.argsort(similarities[:, 0])[::-1][:k]

        results: list[tuple[str, int, float]] = []
        for idx in top_indices:
            score = float(similarities[idx, 0])
            text = self._texts[idx]
            motion_id = int(self._motion_ids[idx])
            results.append((text, motion_id, score))
        return results

    def search_with_duration(
        self, query: str, max_duration: float, k: int = 5
    ) -> tuple[str, int, float] | None:
        """
        语义检索 + 按时长接近度优选

        在 top-k 候选中选择时长最接近 max_duration 的动作，
        避免动作与语音时长不匹配导致的异常截断。

        Args:
            query: 查询文本
            max_duration: 目标时长（秒），通常为语音预估时长
            k: 候选数

        Returns:
            (文本, motion_id, 分数) | None: 最优匹配，无结果返回 None
        """
        candidates = self.search(query, k)
        if not candidates:
            return None

        if max_duration <= 0:
            return candidates[0]

        best: tuple[str, int, float] | None = None
        best_gap = float("inf")
        for item in candidates:
            idx = self._texts.index(item[0]) if item[0] in self._texts else -1
            if idx < 0:
                continue
            dur = float(self._motion_durations[idx])
            gap = abs(dur - max_duration)
            if gap < best_gap:
                best_gap = gap
                best = item

        return best if best is not None else candidates[0]


# ============================================================================
# ActionOverlay — 特殊动作覆盖
# ============================================================================


class ActionOverlay:
    """
    特殊动作 → 参数帧数组生成器

    根据 SIMPLE_ACTIONS 定义，将多个特殊动作（如 smile、wink）
    转换为 A 组表情参数的帧级值数组。
    """

    @staticmethod
    def _generate_param_curve(
        keyframes: list[tuple[float, float]],
        duration: float,
        fps: float = 60.0,
        default_value: float = 0.0,
    ) -> list[float]:
        """
        从关键帧生成帧级值数组

        keyframes 是相对于动作起始时间的偏移序列。
        在第一个关键帧之前使用 default_value（避免外推到负值）。
        最后一个关键帧的值在整个剩余时长保持。

        Args:
            keyframes: [(时间偏移, 值), ...]
            duration: 动作总时长
            fps: 帧率
            default_value: 第一个关键帧之前的默认值

        Returns:
            list[float]: 每帧参数值
        """
        num_frames = int(duration * fps) + 1
        dt = 1.0 / fps
        values: list[float] = []

        if not keyframes:
            return [default_value] * num_frames

        kf_idx = 0

        for f in range(num_frames):
            t = float(f) * dt

            # 在第一个关键帧之前，使用默认值避免外推到异常范围
            if t < keyframes[0][0]:
                values.append(default_value)
                continue

            while kf_idx < len(keyframes) - 1 and keyframes[kf_idx + 1][0] <= t:
                kf_idx += 1
            if kf_idx >= len(keyframes) - 1:
                values.append(keyframes[-1][1])
            else:
                t1, v1 = keyframes[kf_idx]
                t2, v2 = keyframes[kf_idx + 1]
                if t2 > t1:
                    alpha = (t - t1) / (t2 - t1)
                    values.append(v1 + (v2 - v1) * alpha)
                else:
                    values.append(v1)

        return values

    @classmethod
    def generate_all(
        cls,
        action_names: list[str],
        duration: float,
        fps: float = 60.0,
    ) -> dict[str, list[float]]:
        """
        为多个特殊动作生成合并后的参数帧数组

        当多个动作作用于同一参数时，后续动作覆盖先前的。

        Args:
            action_names: 特殊动作名列表（如 ["smile", "wink_left"]）
            duration: 动作总时长
            fps: 帧率

        Returns:
            dict[str, list[float]]: {param_id: [frame_values]}
        """
        overlay: dict[str, list[float]] = {}

        for action_name in action_names:
            if action_name not in SIMPLE_ACTIONS:
                continue
            action_def = SIMPLE_ACTIONS[action_name]
            for param_id, keyframes in action_def.items():
                curve = cls._generate_param_curve(
                    keyframes,
                    duration,
                    fps,
                    default_value=PARAM_DEFAULTS.get(param_id, 0.0),
                )
                overlay[param_id] = curve

        return overlay


def estimate_text_duration(text: str) -> float:
    """
    根据文本估算朗读时长（标点感知）

    Args:
        text: 输入文本

    Returns:
        估算的时长（秒）
    """
    cn_count = sum(
        1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
    )
    other_count = len(text) - cn_count

    duration = cn_count * 0.35 + other_count * 0.15

    duration += text.count("。") * 0.5
    duration += text.count("！") * 0.5
    duration += text.count("？") * 0.5
    duration += text.count("，") * 0.25
    duration += text.count("、") * 0.2
    duration += text.count("…") * 1.0
    duration += text.count("...") * 1.0

    return max(1.5, min(duration, 30.0))


# ============================================================================
# MotionEngineService — 服务端统一入口
# ============================================================================


class MotionEngineService:
    """
    服务端动作引擎入口

    串联完整管线：
        文本 + 动作标签 → 语义检索 → 从 DB 加载预解析动作 →
        特殊动作覆盖 → 混合 → 产出 MotionData

    与 MotionEngine 的区别：
    - 不包含 MotionPlayer（无需实时播放）
    - 不依赖 PySide6
    - 输出为 MotionData（逐帧曲线），由调用方决定如何消费

    Attributes:
        database: MotionDatabase 实例（含预计算 embedding）
        matcher: SemanticMatcher 实例（延迟初始化）
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        初始化引擎

        Args:
            db_path: motion.db 路径，默认 data/motion.db
        """
        self.database = MotionDatabase(db_path)
        self.matcher: SemanticMatcher | None = None
        self.matcher = SemanticMatcher(self.database)

    def process(
        self,
        text: str,
        actions: list[str],
        max_duration: float | None = None,
        expressions: list[ExpressionInfo] | None = None,
    ) -> MotionData | None:
        """
        处理语义输入：检索 + 覆盖 + 混合

        Args:
            text: 语义描述文本（用于检索匹配的预录制动作）
            actions: 特殊动作名列表（如 ["smile", "wink_left"]）
            max_duration: 动作最大时长（秒），用于匹配语音时长。
            expressions: 可用表情列表（ExpressionInfo 列表），用于解析表情名称

        Returns:
            MotionData | None: 混合后的动作数据，无匹配结果返回 None
        """
        if self.matcher is None or self.database.num_entries == 0:
            return None

        # 1. 从数据库进行，语义检索，然后按时长优选
        if max_duration and max_duration > 0:
            result = self.matcher.search_with_duration(text, max_duration, k=5)
        else:
            results = self.matcher.search(text, k=1)
            result = results[0] if results else None

        if result is None:
            return None

        matched_text, motion_id, score = result
        Log.info(f"[MotionEngineService] 检索匹配 [{score:.3f}]: {matched_text}")

        # 2. 从 DB 加载预解析的动作曲线
        meta = self.database.get_motion_meta(motion_id)
        if meta is None:
            Log.warning(f"[MotionEngineService] motion_id={motion_id} 元数据缺失")
            return None

        curves = self.database.get_motion_curves(motion_id)
        if not curves:
            Log.warning(f"[MotionEngineService] motion_id={motion_id} 曲线数据缺失")
            return None

        base_motion = MotionData(
            curves=curves,
            duration=meta.duration,
            fps=meta.fps,
        )

        # 3. 分离特殊动作和表情：表情不覆盖动作曲线，由前端单独处理
        # 简单动作
        simple_actions: list[str] = []
        # 表情动作
        expression_names: list[str] = []

        expression_map = [expr.name for expr in (expressions if expressions else {})]

        for action_name in actions:
            if action_name in SIMPLE_ACTIONS:
                simple_actions.append(action_name)

            if action_name in expression_map:
                expression_names.append(action_name)

        # 4. 生成特殊动作覆盖（仅对 SIMPLE_ACTIONS 中的动作）
        overlays = ActionOverlay.generate_all(
            simple_actions, base_motion.duration, base_motion.fps
        )

        # 5. 混合: overlay 覆盖
        mixed_curves: dict[str, list[float]] = {}
        for param_id, values in base_motion.curves.items():
            if param_id in overlays:
                mixed_curves[param_id] = overlays[param_id]
            else:
                mixed_curves[param_id] = values

        # 添加 overlay 中存在但 base_motion 中不存在的参数
        for param_id, overlay_curve in overlays.items():
            if param_id not in mixed_curves:
                mixed_curves[param_id] = overlay_curve

        # 5. 时长截断
        effective_duration = base_motion.duration

        if max_duration and 0 < max_duration < base_motion.duration:
            effective_duration = max_duration
            max_frame = int(effective_duration * base_motion.fps) + 1
            for param_id in list(mixed_curves.keys()):
                vals = mixed_curves[param_id]
                if len(vals) > max_frame:
                    mixed_curves[param_id] = vals[:max_frame]
            Log.info(
                f"[MotionEngineService] 截断动作: {base_motion.duration:.2f}s → "
                f"{effective_duration:.2f}s"
            )
            return MotionData(
                curves=mixed_curves,
                duration=effective_duration,
                fps=base_motion.fps,
                expression=expression_names,
            )
        else:
            return MotionData(
                curves=mixed_curves,
                duration=base_motion.duration,
                fps=base_motion.fps,
                expression=expression_names,
            )
