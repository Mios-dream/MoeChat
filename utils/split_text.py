import jionlp
import re


def remove_parentheses_content_and_split(text) -> list[str]:
    """
    根据标点符号进行消息的分句，且去除括号内的内容,且保证内容完整
    不考虑括号嵌套情况

    Args:
        text (str): 输入的字符串

    Returns:
        str: 移除括号内容后的字符串

    处理规则：
    - 移除完整的括号对及其内容：(内容)
    - 移除不完整的左括号及其后面所有内容：(后面的内容
    - 移除不完整的右括号及其前面内容到字符串开头或上一个左括号
    """
    if not isinstance(text, str):
        return text

    # 先移除完整的括号对
    text = re.sub(r"\([^)]*\)", "", text)

    # 移除剩余的左括号及其后面的所有内容
    text = re.sub(r"\(.*$", "", text)

    # 移除剩余的右括号及其前面的内容（到字符串开头）
    text = re.sub(r"^[^(]*?\)", "", text)

    # 切分文本
    temp = jionlp.split_sentence(text)

    # 去除空字串
    result = [i for i in temp if i.strip() != "" and is_sentence_complete_simple(i)]

    return result


def is_sentence_complete_simple(sentence):
    """
    简单判断句子是否以标点符号结尾
    """
    punctuation_form = (
        "…",
        "~",
        "～",
        "。",
        "？",
        "！",
        "?",
        "!",
    )
    if not sentence or not sentence.strip():
        return False

    sentence = sentence.strip()
    # 检查是否以句号、问号、感叹号结尾
    return sentence.endswith(punctuation_form)


if __name__ == "__main__":
    print(remove_parentheses_content_and_split("(这是个括号)你好~我喜欢你！阁下。"))
