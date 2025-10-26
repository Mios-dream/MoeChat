import socket
import threading
from utils.pysilero import VADIterator
from utils.log import logger
from io import BytesIO
import core.chat_core as chat_core
import soundfile as sf
import numpy as np


def handle_client(client_socket: socket.socket):
    """处理客户端连接，确保完整接收消息"""
    vad_iterator = VADIterator(speech_pad_ms=120)
    current_speech = []
    current_speech_tmp = []
    status = False
    audio_len = 0
    # 发送欢迎消息
    while True:
        # 接收完整消息
        try:
            # data = rec(client_socket)
            data = client_socket.recv(1024)
        except:
            client_socket.close()
            logger.info(f"客户端断开：{client_socket}")
            return
            
        if data is None:
            client_socket.close()
            logger.info(f"客户端断开：{client_socket}")
            return
        
        # print(f"[客户端 {client_address}] {message}")
        # data = json.loads(data)
        # if data["type"] == "asr":
        # audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
        samples = np.frombuffer(data, dtype=np.int16)
        current_speech_tmp.append(samples)
        audio_len += len(samples)
        if audio_len < 240:
            continue
        else:
            audio_len = 0
        resampled = np.concatenate(current_speech_tmp.copy())
        resampled = (resampled / 32768.0).astype(np.float32)
        current_speech_tmp = []
        
        for speech_dict, speech_samples in vad_iterator(resampled):
            if "start" in speech_dict:
                current_speech = []
                status = True
                # print("开始说话")
                pass
            if status:
                current_speech.append(speech_samples)
            else:
                continue
            is_last = "end" in speech_dict
            if is_last:
                # print("结束说话")
                status = False
                combined = np.concatenate(current_speech)
                audio_bytes = b""
                with BytesIO() as buffer:
                    sf.write(
                        buffer,
                        combined,
                        16000,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    buffer.seek(0)
                    audio_bytes = buffer.read()  # 完整的 WAV bytes
                    res_text = chat_core.asr(audio_bytes)
                    if res_text:
                        # await c_websocket.send_text(res_text)
                        try:
                            # send(client_socket, res_text)
                            client_socket.send(res_text.encode("utf-8"))
                        except:
                            client_socket.close()
                            return
                current_speech = []  # 清空当前段落
            # if not message:
            #     break
                
            # print(f"客户端: {message}")
            
            # # 如果客户端发送'quit'，则断开连接
            # if message.lower() == 'quit':
            #     send(client_socket, "已收到退出请求，再见！")
            #     break
                
            # # 回复客户端
            # response = f"已收到消息，长度为 {len(message)} 字节"
            # send(client_socket, response)
            
    # except Exception as e:
    #     print(f"处理客户端错误: {e}")
    # finally:
    #     client_socket.close()
    #     print("客户端连接已关闭")

def handle_client_2(client_socket: socket.socket):
    try:
        client_socket.send("ok".encode("utf-8"))
    except:
        return
    while True:
        try:
            data = client_socket.recv(1024).decode("utf-8")
            if data == "ok":
                client_socket.send("ok".encode("utf-8"))
        except:
            return

# def send(sock, data):
#     """发送消息：先发送长度（4字节前缀），再发送数据"""
#     # 计算数据长度（字节数）
#     data_bytes = data.encode('utf-8')
#     length = len(data_bytes)
    
#     # 发送长度（使用4字节无符号整数，网络字节序）
#     sock.sendall(struct.pack('>I', length))
#     # 发送实际数据
#     sock.sendall(data_bytes)

# def rec(sock):
#     """接收消息：先读取长度前缀，再读取对应长度的数据"""
#     # 先读取4字节的长度前缀
#     length_bytes = sock.recv(4)
#     if not length_bytes:
#         return None  # 连接关闭
    
#     # 解析长度（网络字节序转主机字节序）
#     length = struct.unpack('>I', length_bytes)[0]
    
#     # 循环读取直到获取完整数据
#     data_bytes = b''
#     while len(data_bytes) < length:
#         # 每次最多读取剩余长度的数据
#         remaining = length - len(data_bytes)
#         chunk = sock.recv(min(remaining, 4096))  # 缓冲区设为4096字节
#         if not chunk:
#             return None  # 连接中断
#         data_bytes += chunk
    
#     return data_bytes.decode('utf-8')

def start_socket_server(host: str, port: int):
    """启动服务器"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(100)
    logger.info(f"socket_asr服务启动，监听 {host}:{port}...")
    
    try:
        while True:
            client_socket, addr = server_socket.accept()
            client_socket.settimeout(60)
            logger.info(f"新连接：{addr}")
            try:
                data = client_socket.recv(1024)
                
            except:
                client_socket.close()
                continue
            try:
                data = data.decode("utf-8")
                if data == "ok":
                    threading.Thread(target=handle_client_2, args=(client_socket, ), daemon=True).start()
            except:
                # 启动线程处理客户端
                threading.Thread(target=handle_client, args=(client_socket, ), daemon=True).start()
    except KeyboardInterrupt:
        logger.info("服务器正在关闭...")
    finally:
        server_socket.close()