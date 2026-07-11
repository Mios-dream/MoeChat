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
        motions=[{"name": "blush", "anchor": "你好呀"}],
        expression="happy",
        max_duration=3.0,
    )
    # motion_data.curves: {"ParamAngleX": [-2.0, -1.9, ...], ...}
    # motion_data.expression: ["happy"]
"""

import sqlite3
from dataclasses import dataclass
from my_utils import embedding
from my_utils.log import logger as Log
import numpy as np

# 特殊动作定义: {param_id: [(relative_time, target_value), ...]}
# relative_time 范围 [0.0, 1.0]，相对于动作段时长
SIMPLE_ACTIONS: dict[str, dict[str, list[tuple[float, float]]]] = {
    "close_eyes": {
        "ParamEyeLOpen": [(0.0, 0.0)],
        "ParamEyeROpen": [(0.0, 0.0)],
    },
    "wink_left": {
        "ParamEyeLOpen": [(0.0, 0.0), (1.0, 1.0)],
    },
    "wink_right": {
        "ParamEyeROpen": [(0.0, 0.0), (1.0, 1.0)],
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

# 动作元数据
# type: sustained（持续整段）/ punctual（短促单次）
# natural_duration: 动作自然时长（秒），仅 punctual 类型需要
ACTION_METADATA: dict[str, dict] = {
    "close_eyes": {"type": "punctual", "natural_duration": 0.4},
    "wink_left": {"type": "punctual", "natural_duration": 0.15},
    "wink_right": {"type": "punctual", "natural_duration": 0.15},
    "blush": {"type": "sustained"},
    "surprise": {"type": "punctual", "natural_duration": 0.6},
    "pout": {"type": "sustained"},
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
# 缓动函数
# ============================================================================


def _smoothstep(t: float) -> float:
    """
    三次 smoothstep 缓动函数

    比线性插值更自然的加速/减速曲线。

    Args:
        t: 输入 [0.0, 1.0]

    Returns:
        float: 缓动后的值 [0.0, 1.0]
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# ============================================================================
# ActionOverlay — 动作覆盖引擎
# ============================================================================


class ActionOverlay:
    """
    特殊动作 → 在基底动作曲线上做插值覆盖

    核心设计：
    1. 使用 anchor（文本子串）定位动作位置，而非均匀分摊
    2. 支持 intensity 缩放参数值，强度可控
    3. 使用 smoothstep 过渡替代线性 ease，消除边界跳跃
    4. 短促动作（punctual）单实例播放，不重复
    5. 持续动作（sustained）从 anchor 位置持续至句尾
    """

    EASE_DURATION: float = 0.12  # 过渡时长（秒）

    @staticmethod
    def _resolve_position(text: str, anchor: str) -> float:
        """
        将 anchor 文本子串解析为句中相对位置 [0.0, 1.0]

        Args:
            text: 完整句子文本
            anchor: 标记动作位置的文本子串

        Returns:
            float: 相对位置 [0.0, 1.0]，找不到 anchor 返回 0.0
        """
        idx = text.find(anchor)
        if idx < 0 or not text:
            return 0.0
        return idx / len(text)

    @staticmethod
    def _generate_segment(
        keyframes: list[tuple[float, float]],
        seg_duration: float,
        fps: float = 60.0,
        default_value: float = 0.0,
    ) -> list[float]:
        """
        从关键帧生成动作段的帧值数组

        Args:
            keyframes: [(相对时间 0.0~1.0, 值), ...]
            seg_duration: 动作段时长（秒）
            fps: 帧率
            default_value: 段起始前的默认值

        Returns:
            list[float]: 动作段每帧参数值
        """
        num_frames = int(seg_duration * fps) + 1
        dt = 1.0 / fps
        values: list[float] = []

        if not keyframes:
            return [default_value] * num_frames

        kf_idx = 0

        for f in range(num_frames):
            t = float(f) * dt / seg_duration if seg_duration > 0 else 1.0

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
    def _blend_segment(
        cls,
        base_curve: list[float],
        seg_values: list[float],
        start_frame: int,
        fps: float,
    ) -> list[float]:
        """
        将动作段 blend 到基底曲线上，带 smoothstep 过渡

        过渡区内做平滑插值：base * (1 - alpha) + seg * alpha

        Args:
            base_curve: 完整基底曲线
            seg_values: 动作段帧值
            start_frame: 动作段起始帧索引
            fps: 帧率

        Returns:
            list[float]: blend 后的完整曲线
        """
        ease_frames = max(2, int(cls.EASE_DURATION * fps))
        result = list(base_curve)
        seg_len = len(seg_values)

        for i in range(seg_len):
            idx = start_frame + i
            if idx >= len(result):
                break

            if i < ease_frames:
                raw_alpha = i / ease_frames
                alpha = _smoothstep(raw_alpha)
            elif seg_len - i <= ease_frames:
                raw_alpha = (seg_len - i) / ease_frames
                alpha = _smoothstep(raw_alpha)
            else:
                alpha = 1.0

            result[idx] = base_curve[idx] * (1.0 - alpha) + seg_values[i] * alpha

        return result

    @classmethod
    def generate_all(
        cls,
        motion_items: list[dict],
        base_curves: dict[str, list[float]],
        total_duration: float,
        fps: float = 60.0,
    ) -> dict[str, list[float]]:
        """
        在基底曲线上叠加特殊动作覆盖，返回 blend 后的完整曲线

        对每个 motion_item 按 position 单实例定位（不自动重复），
        同一参数的多动作按顺序叠加。

        Args:
            motion_items: [{"name": str, "position": float, "intensity": float}, ...]
            base_curves: 基底曲线 {param_id: [frame_values]}
            total_duration: 总时长（秒）
            fps: 帧率

        Returns:
            dict[str, list[float]]: blend 后的完整曲线
        """
        num_frames = int(total_duration * fps) + 1
        result: dict[str, list[float]] = {
            pid: list(curve) for pid, curve in base_curves.items()
        }

        for item in motion_items:
            name: str = item["name"]
            position: float = item.get("position", 0.0)
            intensity: float = item.get("intensity", 1.0)

            if name not in SIMPLE_ACTIONS:
                continue

            action_def = SIMPLE_ACTIONS[name]
            metadata = ACTION_METADATA.get(name, {})
            action_type = metadata.get("type", "punctual")

            # 计算动作段起止时间
            if action_type == "sustained":
                seg_start = position * total_duration
                seg_end = total_duration
            else:
                natural_dur = metadata.get("natural_duration", 0.3)
                seg_start = position * total_duration
                seg_end = min(seg_start + natural_dur, total_duration)

            seg_duration = seg_end - seg_start
            if seg_duration <= 0:
                continue

            start_frame = int(seg_start * fps)

            for param_id, keyframes in action_def.items():
                default_val = PARAM_DEFAULTS.get(param_id, 0.0)

                if param_id not in result:
                    result[param_id] = [default_val] * num_frames

                # intensity 缩放关键帧值
                scaled_keyframes = [(t, v * intensity) for t, v in keyframes]

                seg_values = cls._generate_segment(
                    scaled_keyframes, seg_duration, fps, default_val
                )
                result[param_id] = cls._blend_segment(
                    result[param_id], seg_values, start_frame, fps
                )

        return result


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
        文本 + motions + expression → 语义检索 → 从 DB 加载预解析动作 →
        anchor 解析 → 跨句子持续动作释放 → 特殊动作覆盖 → 产出 MotionData

    跨句子状态：
        内部维护 _active_sustained 和 _active_expression，
        每句处理后更新，下一句处理时自动对未延续的持续动作做渐出。

    Attributes:
        database: MotionDatabase 实例（含预计算 embedding）
        matcher: SemanticMatcher 实例
    """

    # 跨句子持续动作渐出时长（秒）
    SUSTAINED_RELEASE_DURATION: float = 0.3

    def __init__(self, db_path: str | None = None) -> None:
        """
        初始化引擎

        Args:
            db_path: motion.db 路径，默认 data/motion.db
        """
        self.database = MotionDatabase(db_path)
        self.matcher: SemanticMatcher | None = None
        if self.database.num_entries > 0:
            self.matcher = SemanticMatcher(self.database)

        # 跨句子持续动作状态：{动作名: {参数名: 目标值}}
        self._active_sustained: dict[str, dict[str, list[tuple[float, float]]]] = {}
        # 跨句子表情状态
        self._active_expression: str | None = None

    def _apply_sustained_release(
        self,
        curves: dict[str, list[float]],
        fps: float,
    ) -> dict[str, list[float]]:
        """
        对上一句活跃但本句未延续的持续动作做渐出

        在句首 release_duration 秒内，从持续动作的目标值平滑过渡到新基底曲线值。

        Args:
            curves: 当前句的基底曲线
            fps: 帧率

        Returns:
            dict[str, list[float]]: 渐出处理后的曲线
        """
        if not self._active_sustained:
            return curves

        release_frames = int(self.SUSTAINED_RELEASE_DURATION * fps)
        result: dict[str, list[float]] = {
            pid: list(vals) for pid, vals in curves.items()
        }

        for action_name, params in self._active_sustained.items():
            for param_id, keyframes in params.items():
                if param_id not in result:
                    continue
                # 获取持续动作的目标值（最后一个关键帧的值）
                target_val = keyframes[-1][1] if keyframes else 0.0
                vals = result[param_id]
                max_frames = min(release_frames, len(vals))
                for i in range(max_frames):
                    t = i / release_frames
                    alpha = _smoothstep(t)
                    vals[i] = target_val + (curves[param_id][i] - target_val) * alpha

        return result

    def process(
        self,
        text: str,
        motions: list[dict],
        expression: str | None = None,
        max_duration: float | None = None,
    ) -> MotionData | None:
        """
        处理语义输入：检索 + anchor 解析 + 跨句子释放 + 覆盖混合

        motions 中的动作通过 anchor（文本子串）定位到句中位置，
        expression 跨越句子保持，无需在每个句子重复指定。

        Args:
            text: 对话文本（用于语义检索和 anchor 定位）
            motions: 动作列表 [{"name": str, "anchor": str, "intensity": float}, ...]
            expression: 整句表情名（None 表示沿用上一句）
            max_duration: 动作最大时长（秒），用于匹配语音时长

        Returns:
            MotionData | None: 混合后的动作数据
        """
        if self.matcher is None or self.database.num_entries == 0:
            return None

        # 1. 语义检索，按时长优选
        if max_duration and max_duration > 0:
            result = self.matcher.search_with_duration(text, max_duration, k=5)
        else:
            results = self.matcher.search(text, k=1)
            result = results[0] if results else None

        if result is None:
            return None

        matched_text, motion_id, score = result
        Log.info(f"[MotionEngineService] 检索匹配 [{score:.3f}]: {matched_text}")

        # 2. 从 DB 加载预解析动作曲线
        meta = self.database.get_motion_meta(motion_id)
        if meta is None:
            return None

        curves = self.database.get_motion_curves(motion_id)
        if not curves:
            return None

        # 3. 解析 motions：anchor → position
        resolved_motions: list[dict] = []
        for m in motions:
            name: str = m["name"]
            anchor: str = m.get("anchor", "")
            intensity: float = m.get("intensity", 1.0)
            if anchor:
                position = ActionOverlay._resolve_position(text, anchor)
            else:
                position = 0.0
            resolved_motions.append(
                {
                    "name": name,
                    "position": position,
                    "intensity": intensity,
                }
            )

        # 4. 跨句子持续动作释放：上一句的持续动作渐出
        curves = self._apply_sustained_release(curves, meta.fps)

        # 5. 生成并混合特殊动作覆盖
        mixed_curves = ActionOverlay.generate_all(
            resolved_motions, curves, meta.duration, meta.fps
        )

        # 6. 更新跨句子持续动作状态
        new_sustained: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for m in resolved_motions:
            name = m["name"]
            if name in SIMPLE_ACTIONS:
                action_type = ACTION_METADATA.get(name, {}).get("type", "punctual")
                if action_type == "sustained":
                    new_sustained[name] = dict(SIMPLE_ACTIONS[name])
        self._active_sustained = new_sustained

        # 7. 更新跨句子表情状态
        if expression is not None:
            self._active_expression = expression

        # 8. 构造输出 expression 列表
        expression_output: list[str] | None = None
        if self._active_expression is not None:
            expression_output = [self._active_expression]

        # 9. 时长截断
        effective_duration = meta.duration
        if max_duration and 0 < max_duration < meta.duration:
            effective_duration = max_duration
            max_frame = int(effective_duration * meta.fps) + 1
            for param_id in list(mixed_curves.keys()):
                vals = mixed_curves[param_id]
                if len(vals) > max_frame:
                    mixed_curves[param_id] = vals[:max_frame]
            Log.info(
                f"[MotionEngineService] 截断动作: {meta.duration:.2f}s → "
                f"{effective_duration:.2f}s"
            )

        return MotionData(
            curves=mixed_curves,
            duration=effective_duration,
            fps=meta.fps,
            expression=expression_output,
        )
