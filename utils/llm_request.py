import json
import httpx
from utils import config as CConfig


# 获取配置信息
key = CConfig.config["LLM"]["key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
data = {"model": CConfig.config["LLM"]["model"], "stream": True}

# 添加额外配置（如果存在）
if CConfig.config["LLM"]["extra_config"]:
    data.update(CConfig.config["LLM"]["extra_config"])


async def llm_request(msg: list):
    """
    流式HTTP请求函数，用于与大语言模型进行通信并流式返回输出内容

    Args:
        msg (list): 包含对话历史和当前用户消息的消息列表

    Yields:
        str: 模型流式输出的内容片段

    Example:
        async for content in llm_request(messages):
            print(content, end='', flush=True)
    """
    # 构造请求数据
    data["messages"] = msg

    # 发起流式请求
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST", url=CConfig.config["LLM"]["api"], json=data, headers=headers
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
