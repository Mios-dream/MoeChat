# Versdion 0.99

# 2025.05.13
# 修复一些bug
# 新增功能：声纹识别、更具情绪标签选择指定参考音频

# 2025.06.11
# 增加角色模板功能，可以使用内置提示词模板创建角色
# 增加日记系统（长期记忆），使ai可以记住所有的聊天内容，
#   并且可以使用像”昨天聊了什么“、”上周去了哪里“和”今天中午吃了什么“这样的语句进行基于时间范围的精确查询，不会像传统向量数据库那样因为时间维度而丢失记忆
# 增加核心记忆功能，使ai可以记住关于用户的重要回忆、信息和个人喜好
# 上述功均需要启用角色模板功能
# 脱离原有的GPT-SoVITS代码，改为API接口调用

# 2025.07.17
# 修复若干bug
# 新增websocket在线vad+asr接口
# 新增chat_v2接口
# 新增获取聊天记录接口
# 新增获取服务器配置接口
# 新增更新服务器配置接口
# 新增运行时重载配置功能


# 读取配置文件
# config_data = {}
# with open("config.yaml", "r", encoding="utf-8") as file:
#     config_data = yaml.safe_load(file)

# 设置配置文件
# api_type = str(config_data["LLM"]["type"])
# llm_api = config_data["LLM"]["api"]
# llm_key = config_data["LLM"]["key"]
# model1 = config_data["LLM"]["model"]
# extra_config = config_data["LLM"]["extra_config"]
# is_ttt =  config_data["Core"]["tt"]
# emotion_list = {}
# if "extra_ref_audio" in CConfig.config:
#     emotion_list = CConfig.config["extra_ref_audio"]
# batch_size = 1
# top_k = 15
# try:
#     batch_size = int(CConfig.config["GSV"]["batch_size"])
# except:
#     batch_size = 1
# try:
#     top_k = int(CConfig.config["GSV"]["top_k"])
# except:
#     top_k = 15

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {CConfig.config["LLM"]["key"]}"
# }

# req_data = {
#     "model": CConfig.config["LLM"]["model"] ,
#     "stream": True
# }
# if CConfig.config["LLM"]["extra_config"]:
#     for c in extra_config:
#         req_data[c] = extra_config[c]

import os
import sys

# import io
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/vad_utils" % (now_dir))
from utilss import config as CConfig
from utilss import socket_asr as Socket_asr
from utilss import log as Log
import requests
import json
import time
import asyncio
from threading import Thread, Event


# 创建数据文件夹
os.path.exists("data") or os.mkdir("data")

import base64
import soundfile as sf
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utilss.sv import SV
import numpy as np
from utilss.agent import Agent

# import torch
import re

# import unicodedata
import jionlp
from pysilero import VADIterator
from scipy.signal import resample

# import scipy.io.wavfile as wavfile


# t2s_weights = config_data["GSV"]["GPT_weight"]
# vits_weights =  config_data["GSV"]["SoVITS_weight"]
# is_agent = config_data["Agent"]["is_up"]
# if is_agent:
#     a_config = {
#         "char": config_data["Agent"]["char"],
#         "user": config_data["Agent"]["user"],
#         "is_data_base": config_data["Agent"]["lore_books"],
#         "data_base_thresholds": config_data["Agent"]["books_thresholds"],
#         "data_base_depth": config_data["Agent"]["scan_depth"],
#         "is_long_mem": config_data["Agent"]["long_memory"],
#         "is_check_memorys": config_data["Agent"]["is_check_memorys"],
#         "mem_thresholds": config_data["Agent"]["mem_thresholds"],
#         "char_settings": config_data["Agent"]["char_settings"],
#         "char_personalities": config_data ["Agent"]["char_personalities"],
#         "mask": config_data["Agent"]["mask"],
#         "message_example":  config_data["Agent"]["message_example"],
#         "prompt": config_data["Agent"]["prompt"],
#         "is_core_mem": config_data["Agent"]["is_core_mem"],
#         "llm": {"api": llm_api, "key": llm_key, "model": model1}
#     }
#     agent = Agent(a_config)
if CConfig.config["Agent"]["is_up"]:
    agent = Agent()

try:
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        device="cuda:0",
    )
except:
    Log.logger.warning("[提示]未安装ASR模型，开始自动安装ASR模型。")
    from modelscope import snapshot_download

    model_dir = snapshot_download(
        model_id="iic/SenseVoiceSmall",
        local_dir="./utilss/models/SenseVoiceSmall",
        revision="master",
    )
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        # device="cuda:0",
        device="cpu",
    )

# 载入声纹识别模型
sv_pipeline = ""
if CConfig.config["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(CConfig.config["Core"]["sv"])
    is_sv = True
else:
    is_sv = False


# 提交到大模型
def to_llm(
    msg: list, res_msg_list: list, full_msg: list, tmp_msg_list: list, event: Event
):
    """
    将消息发送到大语言模型(LLM)并处理返回的流式响应

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表
        res_msg_list (list): 存储处理后的消息片段的列表，用于TTS合成
        full_msg (list): 存储完整回复消息的列表
        tmp_msg_list (list): 存储临时消息内容的列表
        event (Event): 用于控制线程中断的事件对象，当客户端打断时设置该事件

    功能说明:
        1. 构造请求头和请求数据
        2. 发送POST请求到LLM API
        3. 流式处理返回的数据
        4. 解析情绪标签并设置相应的参考音频
        5. 按标点符号分割文本，分批加入TTS队列
        6. 处理完整响应并更新智能体上下文(如果启用)
    """

    # 获取多线程锁
    # if CConfig.config["Agent"]["is_up"]:
    #     agent.lock.acquire()
    def get_emotion(msg: str):
        res = re.findall(r"\[(.*?)\]", msg)
        if len(res) > 0:
            match = res[-1]
            if match and CConfig.config["extra_ref_audio"]:
                if match in CConfig.config["extra_ref_audio"]:
                    return match

    # def add_msg(msg: str):
    key = CConfig.config["LLM"]["key"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    data = {"model": CConfig.config["LLM"]["model"], "stream": True}
    if CConfig.config["LLM"]["extra_config"]:
        data.update(CConfig.config["LLM"]["extra_config"])
    data["messages"] = msg

    t_t = time.time()
    try:
        response = requests.post(
            url=CConfig.config["LLM"]["api"], json=data, headers=headers, stream=True
        )
        # print(response.status_code)
    except:
        Log.logger.error("无法链接到LLM服务器")
        return
        # return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})
    if response.status_code != 200:
        res_msg_list.append("DONE_DONE")
        Log.logger.error("无法链接到LLM服务器")
        return
        # return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})

    # 信息处理
    # biao_dian_2 = ["…", "~", "～", "。", "？", "！", "?", "!"]
    biao_dian_3 = ["…", "~", "～", "。", "？", "！", "?", "!", ",", "，"]
    biao_dian_4 = ["…", "~", "～", ",", "，"]

    res_msg = ""
    tmp_msg = ""
    # 记录首次输出模型延迟的flag
    j = True
    # 记录首句输出延迟的flag
    j2 = True
    ref_audio = ""
    ref_text = ""
    # biao_tmp = biao_dian_3
    # if response.status_code == 200:
    for line in response.iter_lines():
        if event.is_set():
            response.close()
            Log.logger.info(f"客户端打断回答")
            break
        if line:
            try:
                if j:
                    Log.logger.info(f"\n[大模型延迟]{time.time() - t_t}")
                    t_t = time.time()
                    j = False
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str:
                        msg_t = json.loads(data_str)["choices"][0]["delta"]["content"]
                        tmp_msg_list[0] += msg_t
                        res_msg += msg_t
                        tmp_msg += msg_t
                res_msg = res_msg.replace("...", "…")
                tmp_msg = tmp_msg.replace("...", "…")
            except Exception as e:
                continue
                # err = line.decode("utf-8")
                # print(f"[错误]：{err}")
                # print(e)
                # break
            # if not tmp_msg:
            #     continue
            ress = ""
            stat = 0
            for ii in range(len(tmp_msg)):
                if tmp_msg[ii] in ["(", "（", "["]:
                    stat += 1
                    continue
                if tmp_msg[ii] in [")", "）", "]"]:
                    stat -= 1
                    continue
                if stat != 0:
                    continue
                if tmp_msg[ii] not in biao_dian_3:
                    continue
                if (
                    (tmp_msg[ii] in biao_dian_4)
                    and j2 == False
                    and len(re.sub(r"[$(（[].*?[]）)]", "", tmp_msg[: ii + 1])) <= 10
                ):
                    continue

                # 提取文本中的情绪标签，并设置参考音频
                emotion = get_emotion(tmp_msg)
                if emotion:
                    if emotion in CConfig.config["extra_ref_audio"]:
                        ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                        ref_text = CConfig.config["extra_ref_audio"][emotion][1]
                ress = tmp_msg[: ii + 1]
                ress = jionlp.remove_html_tag(ress)
                ttt = ress
                if j2:
                    Log.logger.info(f"\n开始合成首句语音：{time.time() - t_t}")
                    for i in range(len(ress)):
                        if ress[i] == "\n" or ress[i] == " ":
                            try:
                                ttt = ress[i + 1 :]
                            except:
                                ttt = ""
                # if not j2:
                #     if len(re.sub(r'[$(（[].*?[]）)]', '', ttt)) < 4:
                #         continue
                if ttt:
                    res_msg_list.append([ref_audio, ref_text, ttt])
                # print(f"[合成文本]{ress}")
                if j2:
                    j2 = False
                try:
                    tmp_msg = tmp_msg[ii + 1 :]
                except:
                    tmp_msg = ""
                break

    if not event.is_set():
        if len(tmp_msg) > 0:
            emotion = get_emotion(tmp_msg)
            if emotion:
                if emotion in CConfig.config["extra_ref_audio"]:
                    ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                    ref_text = CConfig.config["extra_ref_audio"][emotion][1]
            res_msg_list.append([ref_audio, ref_text, tmp_msg])

        # 返回完整上下文
        res_msg = jionlp.remove_html_tag(res_msg)
        if len(res_msg) == 0:
            full_msg.append(res_msg)
            res_msg_list.append("DONE_DONE")
            return
        ttt = ""
        for i in range(len(res_msg)):
            if res_msg[i] != "\n" and res_msg[i] != " ":
                ttt = res_msg[i:]
                break

        full_msg.append(ttt)
        Log.logging.info(f"完整回复：{full_msg[0]}")

    # print(res_msg_list)
    res_msg_list.append("DONE_DONE")
    if (
        CConfig.config["Agent"]["is_up"] and len(full_msg[0]) != 0
    ):  # 刷新智能体上下文内容
        agent.add_msg(re.sub(r"<.*?>", "", full_msg[0]).strip())

    # 释放多线程锁
    # if CConfig.config["Agent"]["is_up"]:
    #     agent.lock.release()


def tts(data: dict):
    """
    语音合成

    Parameters
        data : dict
    """
    res = requests.post(CConfig.config["GSV"]["api"], json=data, timeout=10)
    if res.status_code == 200:
        return res.content
    else:
        Log.logger.warning(f"语音合成失败：{data}")
        # print(f"[错误]tts语音合成失败！！！")
        # print(data)
        return None


def clear_text(msg: str):
    """
    清洗文本内容，去除无效字符和多余空白

    该函数用于清理输入文本，去除括号内容、空格、换行符等无效字符，
    并确保文本以非标点符号开头，为后续的语音合成做准备。

    Args:
        msg (str): 需要清洗的原始文本字符串

    Returns:
        str: 清洗后的文本字符串

    处理步骤:
        1. 使用正则表达式去除所有括号及其中的内容（包括中文和英文括号）
        2. 移除所有空格和换行符
        3. 找到第一个非标点符号字符，返回从该字符开始的子字符串
        4. 如果文本全为标点符号，则返回空字符串

    Example:
        >>> clear_text(" (旁白) 你好，世界！ ")
        '你好，世界！'
        >>> clear_text("   \n  ")
        ''
    """
    msg = re.sub(r"[$(（[].*?[]）)]", "", msg)
    msg = msg.replace(" ", "").replace("\n", "")
    tmp_msg = ""
    punctuation_table = ["…", "~", "～", "。", "？", "！", "?", "!", ",", "，"]
    for i in range(len(msg)):
        if msg[i] not in punctuation_table:
            tmp_msg = msg[i:]
            break
    # msg = jionlp.remove_exception_char(msg)
    return tmp_msg


# TTS并写入队
def to_tts(tts_data: list):
    """
    TTS并写入队列

    Parameters
        tts_data : list
            包含参考音频、参考文本和合成文本的列表
    """
    # def is_punctuation(char):
    #     return unicodedata.category(char).startswith('P')
    msg = clear_text(tts_data[2])
    # print(f"[实际输入文本]{tts_data[2]}[tts文本]{msg}")
    if len(msg) == 0:
        return
    ref_audio = tts_data[0]
    ref_text = tts_data[1]
    data = {
        "text": msg,
        "text_lang": CConfig.config["GSV"]["text_lang"],
        "ref_audio_path": CConfig.config["GSV"]["ref_audio_path"],
        "prompt_text": CConfig.config["GSV"]["prompt_text"],
        "prompt_lang": CConfig.config["GSV"]["prompt_lang"],
        "seed": CConfig.config["GSV"]["seed"],
        "top_k": CConfig.config["GSV"]["top_k"],
        "batch_size": CConfig.config["GSV"]["batch_size"],
    }
    if CConfig.config["GSV"]["ex_config"]:
        for key in CConfig.config["GSV"]["ex_config"]:
            data[key] = CConfig.config["GSV"]["ex_config"][key]
    if ref_audio:
        data["ref_audio_path"] = ref_audio
        data["prompt_text"] = ref_text
    try:
        byte_data = tts(data)
        # audio_b64 = base64.urlsafe_b64encode(byte_data).decode("utf-8")
        # return audio_b64
        return byte_data
    except:
        return


def ttts(res_list: list, audio_list: list, event: Event):
    """
    合并多个语音并返回

    Parameters
        res_list : list
            合成文本列表
        audio_list : list
            合成音频列表
        event : Event
            事件对象，用于控制合成进程

    Returns
        list: 合成音频列表，包含合成音频和DONE_DONE标识符
    """
    i = 0
    while True:
        if event.is_set():
            break
        if i < len(res_list):
            if res_list[i] == "DONE_DONE":
                audio_list.append("DONE_DONE")
                break
            # t_t = time.time()
            audio_list.append(to_tts(res_list[i]))
            # if i == 0:
            #     print(f"\n[首句语音耗时]{time.time() - t_t}")
            i += 1
        time.sleep(0.05)


# asr功能
def asr(audio_data: bytes):
    """
    语音识别

    Parameters
        audio_data : bytes
            语音数据

    Returns
        str: 识别结果文本
    """
    global asr_model
    global is_sv
    global sv_pipeline

    tt = time.time()
    if is_sv:
        if not sv_pipeline.check_speaker(audio_data):
            return None
    # with open(f"./tmp/{tt}.wav", "wb") as file:
    #     file.write(audio_data)
    audio_data = BytesIO(audio_data)
    res = asr_model.generate(
        input=audio_data,
        # input=f"{model.model_path}/example/zh.mp3",
        cache={},
        language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        ban_emo_unk=True,
        use_itn=False,
        disable_pbar=True,
        # batch_size=200,
    )
    # print(f"{model.model_path}/example/zh.mp3",)
    text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")
    # text = res[0]["text"]
    # print()
    # print(f"[{time.time() - tt}]{text}\n\n")
    if text:
        return text
    return None


# -----------------------------------API接口部分----------------------------------------------------------

app = FastAPI()


class tts_data(BaseModel):
    msg: list


# 聊天接口
async def text_llm_tts(params: tts_data, start_time):
    # print(params)
    res_list = []
    audio_list = []
    full_msg = []
    tmp_msg = [""]
    if CConfig.config["Agent"]["is_up"]:
        global agent
        t = time.time()
        msg_list = agent.get_msg_data(params.msg[-1]["content"])
        Log.logger.info(f"获取上下文耗时：{time.time() - t}")
    else:
        msg_list = params.msg
    llm_stop = Event()
    llm_t = Thread(
        target=to_llm,
        args=(
            msg_list,
            res_list,
            full_msg,
            tmp_msg,
            llm_stop,
        ),
    )
    llm_t.daemon = True
    llm_t.start()
    tts_stop = Event()
    tts_t = Thread(
        target=ttts,
        args=(
            res_list,
            audio_list,
            tts_stop,
        ),
    )
    tts_t.daemon = True
    tts_t.start()

    i = 0
    stat = True
    while True:
        if i < len(audio_list):
            if audio_list[i] == None:
                i += 1
                continue
            if audio_list[i] == "DONE_DONE":
                data = {"file": None, "message": full_msg[0], "done": True}
                # if CConfig.config["Agent"]["is_up"]:    # 刷新智能体上下文内容
                #     agent.add_msg(re.sub(r'<.*?>', '', full_msg[0]).strip())
                yield f"data: {json.dumps(data)}\n\n"
                return
            audio_b64 = base64.urlsafe_b64encode(audio_list[i]).decode("utf-8")
            data = {"file": audio_b64, "message": res_list[i][2], "done": False}
            # audio = str(audio_list[i])
            # yield str(data)
            if stat:
                Log.logger.info(f"\n服务端首句处理耗时：{time.time() - start_time}\n")
                stat = False
            yield f"data: {json.dumps(data)}\n\n"
            i += 1
        await asyncio.sleep(0.05)


@app.post("/api/chat")
async def tts_api_api(params: tts_data):
    return StreamingResponse(
        text_llm_tts(params, time.time()), media_type="text/event-stream"
    )


async def text_llm_tts2(params: tts_data, start_time):
    # print(params)
    res_list = []  # 储存需要tts的文本
    audio_list = []  # 储存合成好的音频
    full_msg = []  # 储存大模型的完整上下文
    tmp_list = [""]  # 储存需要返回客户端的文本

    if CConfig.config["Agent"]["is_up"]:
        global agent
        t = time.time()
        msg_list = agent.get_msg_data(params.msg[-1]["content"])
        Log.logger.info(f"获取上下文耗时：{time.time() - t}")
    else:
        msg_list = params.msg
    llm_stop = Event()
    llm_t = Thread(
        target=to_llm,
        args=(
            msg_list,
            res_list,
            full_msg,
            tmp_list,
            llm_stop,
        ),
    )
    llm_t.daemon = True
    llm_t.start()
    tts_stop = Event()
    tts_t = Thread(
        target=ttts,
        args=(
            res_list,
            audio_list,
            tts_stop,
        ),
    )
    tts_t.daemon = True
    tts_t.start()

    audio_index = 0  # 标记当前音频索引
    msg_index = 0  # 标记当前文本索引
    stat = True

    while True:
        await asyncio.sleep(0.05)
        if audio_index < len(audio_list):
            if audio_list[audio_index]:
                if audio_list[audio_index] == "DONE_DONE":
                    message = full_msg[0]
                    data = {"type": "text", "data": message, "done": True}
                    try:
                        yield f"data: {json.dumps(data)}\n\n"
                    except:
                        break
                    break
                try:
                    # message = audio_list[audio_index]
                    audio_b64 = base64.urlsafe_b64encode(
                        audio_list[audio_index]
                    ).decode("utf-8")
                    data = {"type": "audio", "data": audio_b64, "done": False}
                    yield f"data: {json.dumps(data)}\n\n"
                except:
                    break
            audio_index += 1
        ll = len(tmp_list[0])
        if msg_index < ll:
            text = tmp_list[0][msg_index:]
            try:
                data = json.dumps({"type": "text", "data": text, "done": False})
                yield f"data: {data}\n\n"
            except:
                break
            msg_index = ll
        # if i < len(audio_list):
        #     if audio_list[i] == None:
        #         continue
        #     if audio_list[i] == "DONE_DONE":
        #         data = {"file": None, "message": full_msg[0], "done": True}
        #         # if CConfig.config["Agent"]["is_up"]:    # 刷新智能体上下文内容
        #         #     agent.add_msg(re.sub(r'<.*?>', '', full_msg[0]).strip())
        #         yield f"data: {json.dumps(data)}\n\n"
        #     audio_b64 = base64.urlsafe_b64encode(audio_list[i]).decode("utf-8")
        #     data = {"file": audio_b64, "message": res_list[i][2], "done": False}
        #     # audio = str(audio_list[i])
        #     # yield str(data)
        #     if stat:
        #         print(f"\n[服务端首句处理耗时]{time.time() - start_time}\n")
        #         stat = False
        #     yield f"data: {json.dumps(data)}\n\n"
        #     i += 1

    # llm_stop.set()
    # tts_stop.set()
    llm_t.join()
    tts_t.join()
    # print("[提示]完成...")


@app.post("/api/chat_v2")
async def tts_api(params: tts_data):
    return StreamingResponse(
        text_llm_tts2(params, time.time()), media_type="text/event-stream"
    )


# asr接口
class asr_data(BaseModel):
    data: str


@app.post("/api/asr")
async def asr_api(params: asr_data):
    audio_data = base64.urlsafe_b64decode(params.data.encode("utf-8"))
    text = asr(audio_data)
    return text


# vad接口
@app.websocket("/api/asr_ws")
async def websocket_endpoint(c_websocket: WebSocket):
    await c_websocket.accept()
    # state = np.zeros((2, 1, 128), dtype=np.float32)
    # sr = np.array(16000, dtype=np.int64)
    # 初始化 VAD 迭代器，指定采样率为 16000Hz
    vad_iterator = VADIterator(speech_pad_ms=300)
    current_speech = []
    current_speech_tmp = []
    status = False
    while True:
        try:
            data = await c_websocket.receive_text()
            data = json.loads(data)
            if data["type"] == "asr":
                audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
                samples = np.frombuffer(audio_data, dtype=np.int16)
                current_speech_tmp.append(samples)
                if len(current_speech_tmp) < 4:
                    continue
                resampled = np.concatenate(current_speech_tmp.copy())
                resampled = (resampled / 32768.0).astype(np.float32)
                current_speech_tmp = []

                for speech_dict, speech_samples in vad_iterator(resampled):
                    if "start" in speech_dict:
                        current_speech = []
                        status = True
                        pass
                    if status:
                        current_speech.append(speech_samples)
                    else:
                        continue
                    is_last = "end" in speech_dict
                    if is_last:
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
                            res_text = asr(audio_bytes)
                            if res_text:
                                await c_websocket.send_text(res_text)
                        current_speech = []  # 清空当前段落

        except:
            break


# 客户端获取聊天记录
@app.post("/api/get_context")
async def get_context():
    global agent
    return agent.msg_data


# 客户端获取配置信息
@app.post("/api/get_config")
async def get_config():
    return CConfig.config


# 更新配置文件
@app.post("/api/update_config")
async def update_config(data: dict):
    global agent

    CConfig.update_config(data)
    if CConfig.config["Agent"]["is_up"]:
        agent.update_config()


# -----------------------------------API接口部分----------------------------------------------------------


if __name__ == "__main__":
    # global config_data
    t2s_weights = CConfig.config["GSV"]["GPT_weight"]
    vits_weights = CConfig.config["GSV"]["SoVITS_weight"]
    if t2s_weights:
        Log.logger.info(f"设置GPT_weights...")
        params = {"weights_path": t2s_weights}
        try:
            requests.get(
                str(CConfig.config["GSV"]["api"]).replace("/tts", "/set_gpt_weights"),
                params=params,
            )
        except:
            Log.logger.warning(f"设置GPT_weights失败")
    if vits_weights:
        Log.logger.info(f"设置SoVITS...")
        params = {"weights_path": vits_weights}
        try:
            requests.get(
                str(CConfig.config["GSV"]["api"]).replace(
                    "/tts", "/set_sovits_weights"
                ),
                params=params,
            )
        except:
            Log.logger.warning(f"设置SoVITS失败")
    Thread(
        target=Socket_asr.start_server,
        args=(
            "0.0.0.0",
            8002,
            asr_model,
        ),
        daemon=True,
    ).start()
    # s_asr = Socket_asr.ImprovedFullDuplexServer(host="0.0.0.0", port=8002, asr_model=asr_model)
    # Thread(target=s_asr.start_server, args=(), daemon=True).start()
    # socket_asr_t = Thread(target=)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)
