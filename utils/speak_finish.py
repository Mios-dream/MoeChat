from utils.llm_request import slm_request
from utils.parse_json_response import parse_llm_json_response


async def isSpeakFinish(message: str) -> bool:
    prompt = f"""
你是一个 AI 助手的回复决策模型，能帮判断用户的输入信息是否完整。
--- 最新消息 ---
{message}

--- 判断标准 ---
1.  **状态判断**：给出一个确定的布尔值。
    -   如果是无意义的对话，或闲聊请返回False
    -   如果包含完整的句子结构请返回True
    
--- 回复格式 ---
你的回答必须是一个严格的 JSON 对象，格式如下：
{{"state": true/false}}

根据以上标准，你的决策是？
    """
    # prepare the model input
    messages = [
        {"role": "system", "content": prompt},
    ]

    content = await slm_request(messages)
    print("是否已经说完")
    print("content:", content)
    return bool(content)


class SpeakWithAssistantContent:
    probability: float
    message: str | None

    def __init__(self, probability: float, message: str | None):
        self.probability = probability
        self.message = message


async def SpeakWithAssistant(
    message: str, message_queue: list
) -> SpeakWithAssistantContent | None:
    prompt = f"""
    你是一个 AI 助手的回复决策模型。你的任务是分析以下聊天记录和最新消息，判断 AI 助手进行回复的“必要性”和“趣味性”，同时对用户输入的消息进行改写，因为用户为语音输入，可能存在输出错字。你的名字叫“澪”，也可叫“小澪”

--- 最近的聊天记录 ---
{message_queue}

--- 最新消息 ---
{message}

--- 回复标准 ---
1.  **必要性判断**：给出一个 0.0 到 1.0 之间的概率分数。
    -   当最新消息明显是对话题的延续，或者直接提到了你（澪），或者向你提问时，概率应为 1.0。
    -   话题轻松有趣、适合闲聊，概率可以高一些 (0.6-0.8)。
    -   话题与 AI、代码等相关，概率较高 (0.5-0.7)。
    -   如果大家在讨论一个你完全不了解的私人话题，或者最新消息明显是用户间的对话，请保持安静，概率为 0.0。
    -   无意义的闲聊或表情符号，概率应接近 0.0。
2.  **改写标准**：
    -   改写应该贴合用户表达的原意，避免引入新的内容。
    -   对于输入的错误，请尝试修复。
    -   对于比较混乱的内容，请尝试提取重要信息，重新组织。
    -   对于识别错误的助手名字，请改为正确的名字。(例如 “小林” 改为 “小澪”，林在读音上和澪相似，极大概率为识别错误)

    
--- 回复格式 ---
你的回答必须是一个严格的 JSON 对象，格式如下：
{{"probability": <概率值>,"message": <改写后的消息>}}

根据以上标准，你的决策是？
    """
    messages = [
        {"role": "system", "content": prompt},
    ]

    content = parse_llm_json_response(await slm_request(messages))

    if not content:
        return None

    print("助手对话的概率：", content.get("probability"))
    print("助手对话的改写：", content.get("message"))

    return SpeakWithAssistantContent(
        probability=content.get("probability", 0), message=content.get("message", None)
    )
