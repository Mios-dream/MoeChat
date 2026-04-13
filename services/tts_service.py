import httpx
from gsv_tts import TTS
import soundfile as sf
import io
from Config import Config
from my_utils.log import logger
from my_utils import config_manager as CConfig
import my_utils.log as Log


class TTSService:
    """
    TTS服务类，提供GPT-SoVITS模型切换和语音合成接口
    """

    # 初始化TTS引擎
    _local_tts_engine: TTS | None = None

    def __init__(self):
        if CConfig.config.get("TTS", {}).get("mode", "api") == "local":
            self.get_tts_engine()

    def get_tts_engine(self) -> TTS:
        """
        加载GPT-SoVITS底膜
        """
        if self._local_tts_engine:
            return self._local_tts_engine

        self._local_tts_engine = TTS(
            models_dir=Config.GSV_MODELS_PATH,
            use_bert=CConfig.config.get("TTS", {}).get("use_bert", True),
            use_flash_attn=CConfig.config.get("TTS", {}).get("use_flash_attn", False),
        )
        return self._local_tts_engine

    async def switch_tts_models(
        self,
        gpt_model_path: str,
        sovits_model_path: str,
        spk_audio_path: str,
        ref_audio_path: str,
        prompt_text: str,
        language: str,
    ):
        """
        切换GPT-SoVITS模型
        """
        if CConfig.config.get("TTS", {}).get("mode", "api") == "api":
            # API模式下，调用接口设置模型路径
            if not sovits_model_path or not gpt_model_path:
                Log.logger.warning(
                    "设置gptsovits模型出错，SoVITS模型路径或GPT_weights路径为空"
                )
                return
            try:
                params = {"weights_path": sovits_model_path}
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        str(CConfig.config["TTS"]["gptsovits"]["api"]).replace(
                            "/tts", "/set_sovits_weights"
                        ),
                        params=params,
                    )
                if response.status_code != 200:
                    Log.logger.error(f"设置SoVITS模型失败: {response.text}")
                    raise Exception(f"设置SoVITS模型失败: {response.text}")
            except Exception as e:
                Log.logger.error(f"设置SoVITS模型失败: {e}")
                raise Exception(f"设置SoVITS模型失败: {e}")

            try:
                params = {"weights_path": gpt_model_path}
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        str(CConfig.config["TTS"]["gptsovits"]["api"]).replace(
                            "/tts", "/set_gpt_weights"
                        ),
                        params=params,
                    )
                if response.status_code != 200:
                    Log.logger.error(f"设置GPT_weights失败: {response.text}")
                    raise Exception(f"设置GPT_weights失败: {response.text}")
            except Exception as e:
                Log.logger.error(f"设置GPT_weights失败: {e}")
                raise Exception(f"设置GPT_weights失败: {e}")
        else:
            # 本地模式下，直接加载模型
            if not gpt_model_path or not sovits_model_path:
                logger.warning(
                    "GPT模型路径或SOVITS模型路径为空，无法切换GPT-SoVITS模型"
                )
                return
            if not spk_audio_path and not ref_audio_path:
                logger.warning(
                    "SPK音频路径和参考音频路径都为空，无法切换GPT-SoVITS模型"
                )
                return
            if language not in ["zh", "en", "ja"]:
                logger.warning("不支持的语言类型，无法切换GPT-SoVITS模型")
                return
            # 拼接资源路径
            local_tts_engine = self.get_tts_engine()

            local_tts_engine.load_gpt_model(gpt_model_path)
            local_tts_engine.load_sovits_model(sovits_model_path)
            local_tts_engine.init_language_module(language)
            local_tts_engine.cache_prompt_audio(ref_audio_path, prompt_text)
            local_tts_engine.cache_spk_audio(spk_audio_path)

    async def local_gsv_tts(
        self,
        data: dict,
    ) -> bytes | None:
        """
        在异步环境中调用的本地GPT-SoVITS推理函数，内部使用线程池执行同步推理。
        """
        if self._local_tts_engine is None:
            logger.warning("本地GPT-SoVITS模型未加载，无法进行语音合成")
            return None
        clip = await self._local_tts_engine.infer_async(
            text=data["text"],
            spk_audio_path=data["ref_audio_path"],
            prompt_audio_path=data["ref_audio_path"],
            prompt_audio_text=data["prompt_text"],
        )

        # 将 AudioClip 对象转换为字节
        with io.BytesIO() as buffer:
            sf.write(
                buffer,
                clip.audio_data,
                clip.samplerate,
                format="WAV",
                subtype="PCM_16",
            )
            buffer.seek(0)
            return buffer.read()

    async def api_gsv_tts(self, data: dict) -> bytes | None:
        """
        调用gptsovits进行语音合成

        Parameters:
            data (dict): 符合gptsovits的语音合成参数
        """
        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(
                    CConfig.config["TTS"]["gptsovits"]["api"], json=data, timeout=60
                )
                if res.status_code == 200:
                    return res.content
                else:
                    logger.error(f"[错误]tts语音合成失败！！！")
                    logger.error(data)
                    logger.error(res.text)
                    return None
            except httpx.TimeoutException as e:
                logger.error(f"[错误]tts语音合成超时！！！")
                return None
            except Exception as e:
                logger.error(f"[错误]tts语音合成失败！！！ 错误信息: {e}")
                logger.error(data)
                return None


ttsService = TTSService()
