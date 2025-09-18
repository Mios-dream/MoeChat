from api.models.tts_request import tts_data
from utils import config as CConfig
import json
import time
import asyncio
from threading import Event, Thread
import base64
from fastapi.responses import JSONResponse
from utils.sv import SV
from utils.agent import Agent
from plugins.financial.plugin import financial_plugin_hook
import re
from utils.socket_asr import ASRServer
import jionlp
from utils.log import logger
import httpx

if CConfig.config["Agent"]["is_up"]:
    agent = Agent()


# 载入声纹识别模型
sv_pipeline: SV
if CConfig.config["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(CConfig.config["Core"]["sv"])
    is_sv = True
else:
    is_sv = False


# 提交到大模型
async def to_llm(msg: list, res_msg_queue: asyncio.Queue, full_msg: list):
    """
    将消息发送到大语言模型(LLM)并处理返回的流式响应

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表
        res_msg_queue (asyncio.Queue): 存储处理后的消息片段的队列，用于TTS合成
        full_msg (list): 存储完整回复消息的列表

    功能说明:
        1. 构造请求头和请求数据
        2. 发送POST请求到LLM API
        3. 流式处理返回的数据
        4. 解析情绪标签并设置相应的参考音频
        5. 按标点符号分割文本，分批加入TTS队列
        6. 处理完整响应并更新智能体上下文(如果启用)
    """

    def get_emotion(msg: str) -> str | None:
        """
        查询文字中的情感字段，检测是否有对应的音频文件，存在则返回对应的情感字段
        """
        res = re.findall(r"\[(.*?)\]", msg)
        if len(res) > 0:
            match = res[-1]
            if match and CConfig.config["extra_ref_audio"]:
                if match in CConfig.config["extra_ref_audio"]:
                    return match

    # 大模型api key
    key = CConfig.config["LLM"]["key"]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    data = {"model": CConfig.config["LLM"]["model"], "stream": True}
    # 此处需要优化
    if CConfig.config["LLM"]["extra_config"]:
        data.update(CConfig.config["LLM"]["extra_config"])

    data["messages"] = msg

    # 统计大模型延迟
    start_time = time.time()
    logger.info("[LLM]：开始处理")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            async with client.stream(
                "POST", url=CConfig.config["LLM"]["api"], json=data, headers=headers
            ) as response:
                logger.info("[LLM]：创建流")
                # 信息处理
                # biao_dian_2 = ["…", "~", "～", "。", "？", "！", "?", "!"]
                punctuation_form_3 = [
                    "…",
                    "~",
                    "～",
                    "。",
                    "？",
                    "！",
                    "?",
                    "!",
                    ",",
                    "，",
                ]
                punctuation_form_4 = ["…", "~", "～", ",", "，"]

                res_msg = ""
                tmp_msg = ""
                first_print_time_flag = True
                j2 = True
                ref_audio = ""
                ref_text = ""
                async for line in response.aiter_lines():
                    if line:
                        try:
                            if first_print_time_flag:
                                logger.info(f"\n[大模型延迟]{time.time() - start_time}")
                                first_print_time_flag = False
                            decoded_line = line
                            if decoded_line.startswith("data:"):
                                data_str = decoded_line[5:].strip()
                                if not data_str or data_str == "[DONE]":
                                    continue
                                msg_t = json.loads(data_str)["choices"][0]["delta"][
                                    "content"
                                ]
                                if msg_t:
                                    res_msg += msg_t
                                    tmp_msg += msg_t
                            res_msg = res_msg.replace("...", "…")
                            tmp_msg = tmp_msg.replace("...", "…")
                        except Exception as e:
                            logger.error(f"[错误]：{e}", exc_info=True)
                            continue
                        ress = ""
                        stat = 0
                        for ii in range(len(tmp_msg)):
                            if tmp_msg[ii] in ["(", "（", "[", "{"]:
                                stat += 1
                                continue
                            if tmp_msg[ii] in [")", "）", "]", "}"]:
                                stat -= 1
                                continue
                            if stat != 0:
                                continue
                            if tmp_msg[ii] not in punctuation_form_3:
                                continue
                            if (
                                (tmp_msg[ii] in punctuation_form_4)
                                and j2 == False
                                and len(
                                    re.sub(r"[$(（[].*?[]）)]", "", tmp_msg[: ii + 1])
                                )
                                <= 10
                            ):
                                continue

                            # 提取文本中的情绪标签，并设置参考音频
                            emotion = get_emotion(tmp_msg)
                            if emotion:
                                if emotion in CConfig.config["extra_ref_audio"]:
                                    ref_audio = CConfig.config["extra_ref_audio"][
                                        emotion
                                    ][0]
                                    ref_text = CConfig.config["extra_ref_audio"][
                                        emotion
                                    ][1]
                            ress = tmp_msg[: ii + 1]
                            ress = jionlp.remove_html_tag(ress)
                            ttt = ress
                            if j2:
                                for i in range(len(ress)):
                                    if ress[i] == "\n" or ress[i] == " ":
                                        try:
                                            ttt = ress[i + 1 :]
                                        except:
                                            ttt = ""
                            if ttt:
                                await res_msg_queue.put([ref_audio, ref_text, ttt])
                            # print(f"[合成文本]{ress}")
                            if j2:
                                j2 = False
                            try:
                                tmp_msg = tmp_msg[ii + 1 :]
                            except:
                                tmp_msg = ""
                            break

                if len(tmp_msg) > 0:
                    emotion = get_emotion(tmp_msg)
                    if emotion:
                        if emotion in CConfig.config["extra_ref_audio"]:
                            ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                            ref_text = CConfig.config["extra_ref_audio"][emotion][1]
                    await res_msg_queue.put([ref_audio, ref_text, tmp_msg])

                # 返回完整上下文
                res_msg = jionlp.remove_html_tag(res_msg)
                if len(res_msg) == 0:
                    full_msg.append(res_msg)
                    await res_msg_queue.put("DONE_DONE")
                    return
                ttt = ""
                for i in range(len(res_msg)):
                    if res_msg[i] != "\n" and res_msg[i] != " ":
                        ttt = res_msg[i:]
                        break

                full_msg.append(ttt)
                logger.info(full_msg)
                # print(res_msg_list)
                await res_msg_queue.put("DONE_DONE")
    except:
        logger.error("无法链接到LLM服务器")
        return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})


async def tts(data: dict):
    """
    语音合成

    Parameters:
        data (dict): 符合gptsovits的语音合成参数
    """
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(CConfig.config["GSV"]["api"], json=data, timeout=10)
            if res.status_code == 200:
                return res.content
            else:
                logger.error(f"[错误]tts语音合成失败！！！")
                logger.error(data)
                return None
        except Exception as e:
            logger.error(f"[错误]tts语音合成失败！！！ 错误信息: {e}")
            logger.error(data)
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
    # 新增：移除所有image和meme标签
    msg = re.sub(r"\{(image|meme|pics):.*?\}", "", msg)
    msg = re.sub(r"[$(（[].*?[]）)]", "", msg)
    msg = msg.replace(" ", "").replace("\n", "")
    tmp_msg = ""
    punctuation_form = ["…", "~", "～", "。", "？", "！", "?", "!", ",", "，"]
    for i in range(len(msg)):
        if msg[i] not in punctuation_form:
            tmp_msg = msg[i:]
            break
    # msg = jionlp.remove_exception_char(msg)
    return tmp_msg


async def tts_task(tts_data: list) -> bytes | None:
    """
    构建tts任务

    Parameters
        tts_data : list
            包含参考音频、参考文本和合成文本的列表
    """

    msg = clear_text(tts_data[2])
    if len(msg) == 0:
        return None
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
        byte_data = await tts(data)
        # audio_b64 = base64.urlsafe_b64encode(byte_data).decode("utf-8")
        return byte_data
    except:
        return None


async def start_tts(res_queue: asyncio.Queue, audio_queue: asyncio.Queue):
    """
    合并多个语音并返回

    Parameters
        res_queue : asyncio.Queue
            合成文本队列
        audio_queue : asyncio.Queue
            合成音频队列
    """
    logger.info("开始合成语言...")

    while True:
        try:
            # 从文本队列获取待合成的文本
            item = await res_queue.get()

            if item == "DONE_DONE":
                await audio_queue.put("DONE_DONE")
                print("完成...")
                break

            # 合成音频
            logger.info("添加语音到合成队列")
            audio_data = await tts_task(item)
            if audio_data is None:
                logger.error("TTS处理出错")
                continue
            encode_data = base64.urlsafe_b64encode(audio_data).decode("utf-8")
            await audio_queue.put(encode_data)

        except Exception as e:
            logger.error(f"TTS处理出错: {e}")
            break


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
    global is_sv
    global sv_pipeline

    if is_sv:
        if not sv_pipeline.check_speaker(audio_data):
            return None

    asrServer = ASRServer()
    return asrServer.asr(audio_data)


# 新增函数处理prompt
def _create_llm_prompt_for_financial_task(
    plugin_result: dict, original_msg_history: list
) -> list:
    """
    根据插件返回结果，创建用于LLM润色的新Prompt。
    这是一个辅助函数，专门为财务任务生成给LLM的指令。
    """

    llm_context = plugin_result.get("llm_context", {})
    suggestion = llm_context.get("suggestion_for_llm", "")

    if plugin_result["status"] == "success":
        system_prompt = (
            "一个财务插件刚刚成功处理了一笔用户的交易。"
            "你的任务是：基于插件提供的'任务总结'和'详细数据'，生成一句自然、友好、符合你人设的确认消息给用户。"
            "请不要重复数据，而是用口语化的方式进行确认和反馈。"
            f"任务总结: {suggestion}\n"
            f"详细数据: {json.dumps(llm_context.get('transaction_info', {}), ensure_ascii=False)}"
        )
    elif plugin_result["status"] == "incomplete":
        system_prompt = (
            "一个财务插件发现用户的记账信息不完整，需要向用户提问以补全信息。"
            "你的任务是：基于插件提供的'提问建议'，生成一句自然、友好、符合你人设的问句来引导用户。"
            f"提问建议: {suggestion}\n"
            f"已提取的信息: {json.dumps(llm_context.get('extracted_info', {}), ensure_ascii=False)}"
        )
    else:
        #
        system_prompt = "请基于以下信息和用户对话。"

    # 将这个特殊的系统提示和用户的对话历史结合起来，构成完整的上下文
    # 我们将系统提示放在历史消息的最前面，以指导LLM的后续行为
    final_prompt_list = [
        {"role": "system", "content": system_prompt}
    ] + original_msg_history

    return final_prompt_list


async def text_llm_tts(params: tts_data):
    start_time = time.time()
    # 初始化将要传递给LLM的最终消息列表
    msg_list_for_llm = []

    # 应该放入插件加载器
    # 检查MoeChat总配置文件中的插件开关
    # is_balancer_enabled = (
    #     CConfig.config.get("Plugins", {}).get("Balancer", {}).get("enabled", False)
    # )

    # if is_balancer_enabled:
    #     # 插件已启用，进入插件处理流程
    #     session_id = (
    #         "user_main_session"  # 关键点：实际应用中这里应该是动态的、每个用户唯一的ID
    #     )
    #     user_message = params.msg[-1]["content"]

    #     print(f"[插件钩子] 正在处理消息: '{user_message}'")
    #     plugin_result = financial_plugin_hook(user_message, session_id)
    #     print(
    #         f"[插件钩子] 返回结果: \n{json.dumps(plugin_result, indent=2, ensure_ascii=False)}"
    #     )

    #     if plugin_result["financial_detected"]:
    #         # 只要检测到财务意图（无论成功与否），就调用辅助函数为LLM创建新的、带有指导的Prompt
    #         print("[决策] 检测到财务意图，正在为LLM创建润色任务...")
    #         msg_list_for_llm = _create_llm_prompt_for_financial_task(
    #             plugin_result, params.msg
    #         )

    # 如果经过插件处理后，msg_list_for_llm 列表仍然为空
    # (这说明插件被禁用，或者插件判断当前消息与财务无关)
    # 则执行MoeChat原来的常规聊天逻辑来填充它。

    if CConfig.config["Agent"]["is_up"]:
        global agent
        t = time.time()
        msg_list_for_llm = agent.get_msg_data(params.msg[-1]["content"])
        print(f"[提示]获取上下文耗时：{time.time() - t}")
    else:
        msg_list_for_llm = params.msg

    # ================== 2. 统一的LLM和TTS处理阶段 ==================
    # 无论消息来自插件包装还是常规聊天，最终都汇入到这里，使用同一套处理流水线

    full_msg = []
    # 使用队列协调LLM和TTS
    res_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()

    async def llm_wrapper():
        await to_llm(msg_list_for_llm, res_queue, full_msg)

    async def tts_wrapper():
        await start_tts(res_queue, audio_queue)

    # 启动LLM和TTS任务
    print("[核心流程] 已将最终Prompt送入LLM和TTS处理流水线。")
    llm_task = asyncio.create_task(llm_wrapper())
    tts_task = asyncio.create_task(tts_wrapper())

    stat = True
    emotion_processed = False  # 标记是否已处理表情包

    while True:
        try:
            # 从音频队列获取结果，设置超时避免无限等待
            audio_item = await asyncio.wait_for(audio_queue.get(), timeout=1.0)

            if audio_item == "DONE_DONE":
                # === 新增：在对话结束前处理表情包 ===
                if not emotion_processed and len(full_msg) > 0 and full_msg[0]:
                    try:
                        # 导入表情包系统（延迟导入避免循环依赖）
                        from meme_system import get_emotion_service

                        # 获取表情包服务实例
                        emotion_service = get_emotion_service()
                        if not emotion_service.is_healthy():
                            print("[表情包系统] 初始化表情包服务...")
                            emotion_service.initialize()

                        # 处理LLM回复，获取表情包响应
                        meme_sse_response = emotion_service.process_llm_response(
                            full_msg[0]
                        )

                        if meme_sse_response:
                            print("[表情包系统] 发送表情包到前端")
                            yield meme_sse_response
                        else:
                            print("[表情包系统] 本次不发送表情包")

                    except ImportError:
                        print("[表情包系统] 表情包模块未安装，跳过表情包处理")
                    except Exception as e:
                        print(f"[表情包系统] 处理表情包时发生错误：{e}")

                    emotion_processed = True
                # =======================================

                # 发送结束信号
                data = {"file": None, "message": full_msg[0], "done": True}
                yield f"data: {json.dumps(data,ensure_ascii=False)}\n\n"
                break  # 结束循环

            elif audio_item is not None:
                # 发送音频数据
                # 注意：这里需要获取对应的文本信息，可能需要修改队列结构
                data = {"file": audio_item, "message": "对应文本", "done": False}
                if stat:
                    logger.info(f"\n[服务端首句处理耗时]{time.time() - start_time}\n")
                    stat = False
                yield f"data: {json.dumps(data,ensure_ascii=False)}\n\n"

        except asyncio.TimeoutError:
            # 检查任务是否已完成
            if llm_task.done() and tts_task.done():
                break
            continue
        except Exception as e:
            logger.error(f"处理音频数据时出错: {e}")
            break

    # 等待所有任务完成
    await asyncio.gather(llm_task, tts_task, return_exceptions=True)


async def text_llm_tts_v2(params: tts_data):
    # print(params)
    res_list = []  # 储存需要tts的文本
    audio_list = []  # 储存合成好的音频
    full_msg = []  # 储存大模型的完整上下文
    tmp_list = [""]  # 储存需要返回客户端的文本

    if CConfig.config["Agent"]["is_up"]:
        global agent
        t = time.time()
        msg_list = agent.get_msg_data(params.msg[-1]["content"])
        logger.info(f"获取上下文耗时：{time.time() - t}")
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
        target=start_tts,
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
