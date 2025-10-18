from api.models.tts_request import tts_data
from utils import config as CConfig
import json
import time
import asyncio
import base64
from fastapi.responses import JSONResponse
from utils.sv import SV
from utils.agent import Agent
from plugins.financial.plugin import financial_plugin_hook
import re
from utils.socket_asr import ASRServer
from utils.log import logger
import httpx
from utils.llm_request import llm_request
from utils.split_text import remove_parentheses_content_and_split_v2

if CConfig.config["Agent"]["is_up"]:
    agent = Agent()


# 载入声纹识别模型
sv_pipeline: SV
if CConfig.config["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(CConfig.config["Core"]["sv"])
    is_sv = True
else:
    is_sv = False


class TTSData:
    """
    TTS数据类

    Attributes:
        text (str): 待合成的文本
        ref_audio (str): 参考音频路径
        ref_text (str): 参考文本
    """

    text: str
    ref_audio: str
    ref_text: str

    def __init__(self, text: str, ref_audio: str, ref_text: str):
        self.text = text
        self.ref_audio = ref_audio
        self.ref_text = ref_text


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
                logger.error(res)
                logger.error(res.text)
                return None
        except Exception as e:
            logger.error(f"[错误]tts语音合成失败！！！ 错误信息: {e}")
            logger.error(data)
            return None


def _clear_text(msg: str):
    msg = re.sub(r"[$(（[].*?[]）)]", "", msg)
    msg = msg.replace(" ", "").replace("\n", "")
    tmp_msg = ""
    table = ["…", "~", "～", "。", "？", "！", "?", "!", ",", "，"]
    for i in range(len(msg)):
        if msg[i] not in table:
            tmp_msg = msg[i:]
            break
    return tmp_msg


async def tts_task(tts_data: TTSData) -> bytes | None:
    """
    构建tts任务

    Parameters
        tts_data : list
            包含参考音频、参考文本和合成文本的列表
    """
    msg = tts_data.text.replace(" ", "").replace("\n", "")
    # msg = clear_text(tts_data.text)
    if len(msg) == 0:
        return None
    ref_audio = tts_data.ref_audio
    ref_text = tts_data.ref_text
    print(f"[tts文本]{msg}")
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
            item: TTSData = await res_queue.get()

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


async def to_llm(
    msg: list,
    res_msg_queue: asyncio.Queue[TTSData | str],
    full_msg: list[str],
    text_queue: asyncio.Queue,
):
    """
    将消息发送到大语言模型(LLM)并处理返回的流式响应

    Args:
        msg: 消息列表
        res_msg_queue: TTS处理队列
        full_msg: 完整消息存储
        text_queue: 文本流式输出队列（新增）
    """

    def get_emotion(msg: str) -> str | None:
        """查询文字中的情感字段"""
        res = re.findall(r"\[(.*?)\]", msg)
        if len(res) > 0:
            match = res[-1]
            if match and CConfig.config["extra_ref_audio"]:
                if match in CConfig.config["extra_ref_audio"]:
                    return match

    start_time = time.time()
    logger.info("[LLM]：开始处理")

    try:
        res_msg = ""  # 完整的原始回复
        tmp_msg = ""  # 切句后剩余文本
        # 标记是否首句
        is_first_msg = True
        # 标记第一次打印时间
        first_print_time_flag = True
        # 参考音频
        ref_audio = ""
        # 参考文本
        ref_text = ""
        # 消息索引,用于判断是否已经处理过
        message_index = 0

        async for line in llm_request(msg):
            try:
                if first_print_time_flag:
                    logger.info(f"\n[大模型延迟]{time.time() - start_time}")
                    first_print_time_flag = False

                # 累积文本
                res_msg += line
                tmp_msg += line

                # res_msg = res_msg.replace("（", "(").replace("）", ")")

                # 立即将新文本发送到文本队列（流式输出）
                await text_queue.put(("text", line))

                # split_texts = remove_parentheses_content_and_split(res_msg)
                message_chuck, tmp_msg = remove_parentheses_content_and_split_v2(res_msg, is_first_msg)

                if len(message_chuck) == 0:
                    continue

                # if len(split_texts) <= message_index:
                #     continue
                # # 获取最新拆分文本
                # message_chuck = split_texts[-1]
                # # 更新消息索引
                # message_index += 1
                # print(split_texts)

                full_msg.append(message_chuck)

                # 检查情绪标签
                emotion = get_emotion(message_chuck)

                if emotion and emotion in CConfig.config.get("extra_ref_audio", {}):
                    ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                    ref_text = CConfig.config["extra_ref_audio"][emotion][1]
                # 发送到tts队列，进行语音合成
                await res_msg_queue.put(
                    TTSData(
                        text=message_chuck,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                    )
                )

            except Exception as e:
                logger.error(f"[错误]：{e}", exc_info=True)
                continue
        
        if len(tmp_msg) > 0:
            full_msg.append(tmp_msg)

            # 检查情绪标签
            emotion = get_emotion(tmp_msg)

            if emotion and emotion in CConfig.config.get("extra_ref_audio", {}):
                ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                ref_text = CConfig.config["extra_ref_audio"][emotion][1]
            # 发送到tts队列，进行语音合成
            await res_msg_queue.put(
                TTSData(
                    text=tmp_msg,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            )

        logger.info(f"完整回复: {full_msg}")

        # 发送完成信号
        await res_msg_queue.put("DONE_DONE")
        await text_queue.put(("done", None))


    except Exception as e:
        logger.error(f"无法链接到LLM服务器: {e}", exc_info=True)
        await text_queue.put(("error", str(e)))
        return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})


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
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break  # 结束循环

            elif audio_item is not None:
                # 发送音频数据
                # 注意：这里需要获取对应的文本信息，可能需要修改队列结构
                data = {"file": audio_item, "message": "对应文本", "done": False}
                if stat:
                    logger.info(f"\n[服务端首句处理耗时]{time.time() - start_time}\n")
                    stat = False
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

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
    """
    主处理函数：同时处理LLM流式文本输出和TTS音频合成
    """
    global agent

    start_time = time.time()

    # 初始化消息列表
    msg_list_for_llm = []

    # 处理Agent上下文
    if CConfig.config["Agent"]["is_up"]:
        t = time.time()
        msg_list_for_llm = agent.get_msg_data(params.msg[-1]["content"])
        print(f"[提示]获取上下文耗时：{time.time() - t}")
    else:
        msg_list_for_llm = params.msg

    # 全部消息，ai可能回复多条语句
    full_msg: list[str] = []

    # 创建三个队列
    res_queue: asyncio.Queue[TTSData | str] = asyncio.Queue()  # TTS文本队列
    audio_queue = asyncio.Queue()  # TTS音频队列
    text_queue = asyncio.Queue()  # 流式文本输出队列

    async def llm_wrapper():
        await to_llm(msg_list_for_llm, res_queue, full_msg, text_queue)

    async def tts_wrapper():
        await start_tts(res_queue, audio_queue)

    # 启动异步任务
    llm_task = asyncio.create_task(llm_wrapper())
    tts_task = asyncio.create_task(tts_wrapper())

    stat = True
    emotion_processed = False
    llm_done = False
    tts_done = False

    while True:
        try:
            # 使用 asyncio.wait 同时监听多个队列
            text_task = asyncio.create_task(text_queue.get())
            audio_task = asyncio.create_task(audio_queue.get())

            done, pending = await asyncio.wait(
                [text_task, audio_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=0.1,
            )

            # 取消未完成的任务
            for task in pending:
                task.cancel()

            # 处理完成的任务
            for task in done:
                try:
                    if task == text_task:
                        # 处理文本流式输出
                        msg_type, content = await task

                        if msg_type == "text" and content:
                            # 发送文本片段
                            data = {"type": "text", "data": content, "done": False}
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                        elif msg_type == "done":
                            llm_done = True

                        elif msg_type == "error":
                            # 发送错误信息
                            data = {"type": "error", "data": content, "done": True}
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            break

                    elif task == audio_task:
                        # 处理音频输出
                        audio_item = await task

                        if audio_item == "DONE_DONE":
                            tts_done = True

                            # 处理表情包系统
                            if (
                                not emotion_processed
                                and len(full_msg) > 0
                                and full_msg[0]
                            ):
                                try:
                                    from meme_system import get_emotion_service

                                    emotion_service = get_emotion_service()

                                    if not emotion_service.is_healthy():
                                        print("[表情包系统] 初始化表情包服务...")
                                        emotion_service.initialize()

                                    meme_sse_response = (
                                        emotion_service.process_llm_response(
                                            full_msg[0]
                                        )
                                    )
                                    if meme_sse_response:
                                        print("[表情包系统] 发送表情包到前端")
                                        yield meme_sse_response

                                except ImportError:
                                    print("[表情包系统] 表情包模块未安装")
                                except Exception as e:
                                    print(f"[表情包系统] 处理表情包时发生错误：{e}")

                                emotion_processed = True

                        elif audio_item is not None:
                            # 发送音频数据
                            data = {"type": "audio", "data": audio_item, "done": False}
                            if stat:
                                logger.info(
                                    f"\n[服务端首句处理耗时]{time.time() - start_time}\n"
                                )
                                stat = False
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                except asyncio.CancelledError:
                    pass

            # 检查是否所有任务都完成
            if llm_done and tts_done:
                # 发送最终的完成消息
                data = {
                    "type": "complete",
                    "data": "".join(full_msg) if full_msg else "",
                    "done": True,
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                # agent写入上下文、日记
                agent.add_msg("".join(full_msg))
                break

        except asyncio.TimeoutError:
            # 检查任务是否完成
            if llm_task.done() and tts_task.done():
                if not (llm_done and tts_done):
                    # 确保发送完成信号
                    data = {
                        "type": "complete",
                        "data": "".join(full_msg) if full_msg else "",
                        "done": True,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break
            continue

        except Exception as e:
            logger.error(f"处理数据时出错: {e}", exc_info=True)
            data = {"type": "error", "data": str(e), "done": True}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            break

    # 等待所有任务完成
    await asyncio.gather(llm_task, tts_task, return_exceptions=True)
