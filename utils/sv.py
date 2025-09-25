# 声纹识别模块

from modelscope.pipelines import pipeline
import soundfile as sf
import numpy as np
from scipy.signal import resample
import io


class SV:
    def __init__(self, config: dict):
        self.thr = ""
        with open(config["master_audio"], "rb") as f:
            audio_bytes = f.read()
        self.master_audio = self.resample_wav_bytes(audio_bytes)
        if "thr" in config:
            if config["thr"]:
                self.thr = str(config["thr"])
        try:
            self.sv_pipeline = pipeline(
                task="speaker-verification",
                model="./data/models/speech_res2net_sv_zh-cn_3dspeaker_16k",
                model_revision="master",
            )
        except:
            print("[错误]未安装声纹模型，开始自动安装声纹模型。")
            from modelscope import snapshot_download

            snapshot_download(
                model_id="iic/speech_res2net_sv_zh-cn_3dspeaker_16k",
                local_dir="./data/models/speech_res2net_sv_zh-cn_3dspeaker_16k",
                revision="master",
            )
            self.sv_pipeline = pipeline(
                task="speaker-verification",
                model="./data/models/speech_res2net_sv_zh-cn_3dspeaker_16k",
                model_revision="master",
            )

    def resample_wav_bytes(self, wav_bytes, target_sr=16000):
        # 使用BytesIO将字节转为文件类对象
        with io.BytesIO(wav_bytes) as wav_file:
            # 读取音频数据
            data, original_sr = sf.read(wav_file, dtype="float32")

            # 立体声转单声道（取均值）
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 计算重采样比例
            resample_ratio = target_sr / original_sr

            # 使用scipy的signal.resample进行重采样
            target_samples = int(len(data) * resample_ratio)
            resampled_data = resample(data, target_samples)

            # 归一化并转为16bit PCM格式
            resampled_data = np.clip(resampled_data, -1.0, 1.0)
            resampled_data = (resampled_data * 32767).astype(np.int16)

            return resampled_data

    def check_speaker(self, speaker_audio: bytes) -> bool:
        # with open("ttmp.wav", "wb") as f:
        #     f.write(speaker_audio)
        with io.BytesIO(speaker_audio) as f:
            speaker_audio_1, _ = sf.read(f)
        res = {}
        if self.thr:
            res = self.sv_pipeline([speaker_audio_1, self.master_audio], self.thr)
        else:
            res = self.sv_pipeline([speaker_audio_1, self.master_audio])
        print(f"[声纹识别结果]结果相似度{res['score']}, 目标相似度{self.thr}")
        if res["text"] == "yes":
            return True
        else:
            return False
