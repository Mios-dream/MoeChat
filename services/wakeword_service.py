import numpy as np
from Config import Config
from my_utils import config_manager as CConfig
import sherpa_onnx
import re
from my_utils.log import logger as Log


class WakeWordService:

    def __init__(self, keywords: list[str]):
        self.keywords = self._build_keywords(keywords)
        Log.info(f"[WakeWord] 唤醒词: {self.keywords}")
        # 初始化唤醒词检测器
        self._spotter: sherpa_onnx.KeywordSpotter | None = None
        # 唤醒词模型采样率
        self._sample_rate = 16000
        # 初始化会话
        self.session = None

        self.ensure_loaded()

    def _build_keywords(self, keywords: list[str]) -> list[str]:
        """
        构建关键词词典
        对无法完整解析的关键词执行整词跳过，避免音素与别名错位
        """
        tokens_path = Config.WAKEWORD_MODELS["tokens"]
        tokens_type = Config.WAKEWORD_MODELS["tokens_type"]
        lexicon_path = Config.WAKEWORD_MODELS["lexicon"]

        lexicon_words: set[str] = set()
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                word = raw.split()[0]
                lexicon_words.add(word.upper())

        output: list[str] = []

        for line in keywords:
            original = line
            line = re.sub(r"([\u4e00-\u9fff])([a-zA-Z])", r"\1 \2", line)
            line = re.sub(r"([a-zA-Z])([\u4e00-\u9fff])", r"\1 \2", line)
            if "@" not in line:
                line = line.upper()
                normalized = line.replace(" ", "_")
                line = f"{line} @{normalized}"

            extra = []
            text = []
            for tok in line.split():
                if tok[0] in {":", "#", "@"}:
                    extra.append(tok)
                else:
                    text.append(tok)

            text_str = " ".join(text)
            if not text_str:
                Log.warning(f"[WakeWord] 跳过空关键词: {original}")
                continue

            # 先校验非 CJK 的英文词是否在 lexicon 中，缺失则整词跳过
            english_words = [
                w
                for w in text_str.split()
                if re.search(r"[A-Za-z]", w) and not re.search(r"[\u4e00-\u9fff]", w)
            ]
            missing_words = [w for w in english_words if w.upper() not in lexicon_words]
            if missing_words:
                Log.warning(
                    f"[WakeWord] 跳过关键词 '{original}'，词典缺失单词: {', '.join(missing_words)}"
                )
                continue

            encoded = sherpa_onnx.text2token(
                [text_str],
                tokens=tokens_path,
                tokens_type=tokens_type,
                lexicon=lexicon_path,
            )
            if not encoded or not encoded[0]:
                Log.warning(f"[WakeWord] 跳过关键词 '{original}'，音素解析失败")
                continue

            output.append(" ".join(str(x) for x in encoded[0] + extra))
        return output

    def ensure_loaded(self):
        """
        确保唤醒词模型已加载，支持运行时配置覆盖
        """
        if self._spotter is not None:
            return

        # 验证并获取完整路径
        tokens_path = Config.WAKEWORD_MODELS["tokens"]
        encoder_path = Config.WAKEWORD_MODELS["encoder"]
        decoder_path = Config.WAKEWORD_MODELS["decoder"]
        joiner_path = Config.WAKEWORD_MODELS["joiner"]

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=Config.WAKEWORD_MODELS["keywords_file"],
            sample_rate=int(self._sample_rate),
            keywords_score=float(CConfig.config.get("keywords_score", 1.0)),
            keywords_threshold=float(CConfig.config.get("keywords_threshold", 0.25)),
            provider=str(CConfig.config.get("provider", "cpu")),
        )

    def create_session(self):
        """
        创建唤醒词会话
        """
        self.ensure_loaded()
        return self._spotter.create_stream("/".join(self.keywords))  # type: ignore[union-attr]

    def detect(self, samples: np.ndarray) -> str | None:
        """
        检测唤醒词
        """
        if samples.size == 0:
            return None
        if self.session is None:
            self.session = self.create_session()
        # sherpa-onnx expects mono float32 in [-1, 1]
        chunk = samples.astype(np.float32, copy=False)
        self.session.accept_waveform(self._sample_rate, chunk)

        detected = None
        while self._spotter.is_ready(self.session):  # type: ignore[union-attr]
            self._spotter.decode_stream(self.session)  # type: ignore[union-attr]
            result = self._spotter.get_result(self.session)  # type: ignore[union-attr]
            if result:
                detected = str(result).strip()
                self._spotter.reset_stream(self.session)  # type: ignore[union-attr]
                break

        return detected
