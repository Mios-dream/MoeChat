import socket
import threading
import struct
import json
from utils.pysilero import VADIterator
import numpy as np
import base64
from scipy.signal import resample
from io import BytesIO
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from utils.log import logger


class ASRServer:
    _instance = None
    _lock = threading.Lock()
    asr_model: AutoModel

    # 单例模式
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定模式
                if cls._instance is None:
                    cls._instance = super(ASRServer, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """
        加载asr模型
        """
        model_dir = "./data/models/SenseVoiceSmall"
        try:
            self.asr_model = AutoModel(
                model=model_dir,
                disable_update=True,
                device="cuda:0",
            )
        except Exception as e:
            logger.info(e)
            logger.info("[提示]未安装ASR模型，开始自动安装ASR模型。")
            from modelscope import snapshot_download

            model_dir = snapshot_download(
                model_id="iic/SenseVoiceSmall",
                local_dir=model_dir,
                revision="master",
            )
            model_dir = model_dir
            self.asr_model = AutoModel(
                model=model_dir,
                disable_update=True,
                # device="cuda:0",
                device="cpu",
            )

    def asr(self, audio_data: bytes) -> str | None:
        audio_buffer = BytesIO(audio_data)
        res = self.asr_model.generate(
            input=audio_buffer,
            cache={},
            language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            ban_emo_unk=True,
            use_itn=False,
            disable_pbar=True,
            # batch_size=200,
        )
        text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")

        if text:
            return text
        return None

    # def handle_client(self, client_socket: socket.socket):
    #     """处理客户端连接，确保完整接收消息"""
    #     vad_iterator = VADIterator(speech_pad_ms=300)
    #     current_speech = []
    #     current_speech_tmp = []
    #     status = False
    #     # 发送欢迎消息
    #     while True:
    #         # 接收完整消息
    #         data = self.rec(client_socket)

    #         if data is None:
    #             client_socket.close()
    #             Log.logger.info(f"客户端断开：{client_socket}")
    #             return

    #         # print(f"[客户端 {client_address}] {message}")
    #         data = json.loads(data)
    #         if data["type"] == "asr":
    #             audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
    #             samples = np.frombuffer(audio_data, dtype=np.int16)
    #             current_speech_tmp.append(samples)
    #             if len(current_speech_tmp) < 4:
    #                 continue
    #             resampled = np.concatenate(current_speech_tmp.copy())
    #             resampled = (resampled / 32768.0).astype(np.float32)
    #             current_speech_tmp = []

    #             for speech_dict, speech_samples in vad_iterator(resampled):
    #                 if "start" in speech_dict:
    #                     current_speech = []
    #                     status = True
    #                     # print("开始说话")
    #                     pass
    #                 if status:
    #                     current_speech.append(speech_samples)
    #                 else:
    #                     continue
    #                 is_last = "end" in speech_dict
    #                 if is_last:
    #                     # print("结束说话")
    #                     status = False
    #                     combined = np.concatenate(current_speech)
    #                     audio_bytes = b""
    #                     with BytesIO() as buffer:
    #                         sf.write(
    #                             buffer,
    #                             combined,
    #                             16000,
    #                             format="WAV",
    #                             subtype="PCM_16",
    #                         )
    #                         buffer.seek(0)
    #                         audio_bytes = buffer.read()  # 完整的 WAV bytes
    #                         res_text = self.asr(audio_bytes)
    #                         if res_text:
    #                             # await c_websocket.send_text(res_text)
    #                             try:
    #                                 self.send(client_socket, res_text)
    #                             except:
    #                                 client_socket.close()
    #                                 return
    #                     current_speech.clear()  # 清空当前段落

    # def send(self, sock, data):
    #     """发送消息：先发送长度（4字节前缀），再发送数据"""
    #     # 计算数据长度（字节数）
    #     data_bytes = data.encode("utf-8")
    #     length = len(data_bytes)

    #     # 发送长度（使用4字节无符号整数，网络字节序）
    #     sock.sendall(struct.pack(">I", length))
    #     # 发送实际数据
    #     sock.sendall(data_bytes)

    # def rec(self, sock):
    #     """接收消息：先读取长度前缀，再读取对应长度的数据"""
    #     # 先读取4字节的长度前缀
    #     length_bytes = sock.recv(4)
    #     if not length_bytes:
    #         return None  # 连接关闭

    #     # 解析长度（网络字节序转主机字节序）
    #     length = struct.unpack(">I", length_bytes)[0]

    #     # 循环读取直到获取完整数据
    #     data_bytes = b""
    #     while len(data_bytes) < length:
    #         # 每次最多读取剩余长度的数据
    #         remaining = length - len(data_bytes)
    #         chunk = sock.recv(min(remaining, 4096))  # 缓冲区设为4096字节
    #         if not chunk:
    #             return None  # 连接中断
    #         data_bytes += chunk

    #     return data_bytes.decode("utf-8")

    # def start_server(self, host: str, port: int):
    #     """启动服务器 - 多线程版本"""
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     server_socket.bind((host, port))
    #     server_socket.listen(5)
    #     Log.logger.info(f"socket_asr服务启动，监听 {host}:{port}...")

    #     try:
    #         while True:
    #             client_socket, addr = server_socket.accept()
    #             Log.logger.info(f"新连接：{addr}")
    #             # 为每个客户端创建新线程
    #             client_thread = threading.Thread(
    #                 target=self.handle_client, args=(client_socket,)
    #             )
    #             client_thread.daemon = True
    #             client_thread.start()
    #     except KeyboardInterrupt:
    #         Log.logger.info("服务器正在关闭...")
    #     finally:
    #         server_socket.close()
