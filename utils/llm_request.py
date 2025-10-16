import json
import re
import aiohttp
import httpx
from utils import config as CConfig


# 大模型配置
llm_key = CConfig.config["LLM"]["key"]
llm_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {llm_key}"}
llm_data = {"model": CConfig.config["LLM"]["model"], "stream": True}
# 添加额外配置（如果存在）
if CConfig.config["LLM"]["extra_config"]:
    llm_data.update(CConfig.config["LLM"]["extra_config"])

# 小模型配置
slm_key = CConfig.config["SLM"]["key"]
slm_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {slm_key}"}
slm_data = {"model": CConfig.config["SLM"]["model"], "stream": False}
# 添加额外配置（如果存在）
if CConfig.config["SLM"]["extra_config"]:
    slm_data.update(CConfig.config["SLM"]["extra_config"])


async def llm_request(msg: list):
    """
    流式HTTP请求函数，用于与大语言模型进行通信并流式返回输出内容,流式协议为sse

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表

    Yields:
        str: 模型流式输出的内容片段

    Example:
        async for content in llm_request(messages):
            print(content, end='', flush=True)
    """
    # 构造请求数据
    llm_data["messages"] = msg

    # 发起流式请求
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST",
                url=CConfig.config["LLM"]["api"],
                json=llm_data,
                headers=llm_headers,
            ) as response:
                # 检查响应状态
                if response.status_code != 200:
                    raise Exception(f"LLM API请求失败，状态码: {response.status_code}")

                # 流式处理响应
                async for line in response.aiter_lines():
                    if line:
                        # 处理SSE格式的数据行
                        if line.startswith("data:"):
                            data_str = line[5:].strip()

                            # 检查是否为结束标记
                            if data_str == "[DONE]":
                                break

                            try:
                                # 解析JSON数据
                                json_data = json.loads(data_str)

                                # 提取内容
                                content = json_data["choices"][0]["delta"].get(
                                    "content", ""
                                )

                                # 如果有内容则返回
                                if content:
                                    yield content

                            except json.JSONDecodeError:
                                # 忽略无法解析的行
                                continue

    except httpx.TimeoutException:
        raise Exception("LLM API请求超时")
    except httpx.RequestError as e:
        raise Exception(f"LLM API请求错误: {str(e)}")
    except Exception as e:
        raise Exception(f"LLM处理过程中发生错误: {str(e)}")


async def slm_request(messages: list) -> str | None:
    """
    SLM(小参数模型)快速请求，非流式
    :param data: 消息链
    :return: 请求结果
    """

    slm_data["messages"] = messages

    # 异步发送post请求
    # 创建一个aiohttp的session
    async with aiohttp.ClientSession() as session:
        # 发送post请求
        async with session.post(
            f"",
            headers=slm_headers,
            data=json.dumps(slm_data),
        ) as response:
            if response.status != 200:
                raise Exception(f"请求失败，状态码: {response.status}")
            response_json = await response.json()

            content = response_json["choices"][0]["message"]["content"]
            # reasoning_content = response_json["choices"][0]["message"]["reasoning"]

            return content
            # 非思考输出兼容
            # content = response_json["choices"][0]["message"]["content"]
            # content = re.split(r"</think>", content)
            # return content[1]

    return None
